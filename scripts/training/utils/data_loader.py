"""
SOLARIS-X Memory-Optimized Data Pipeline - CORRECTED VERSION
NASA-Level Space Weather Data Processing for CPU Training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
import psutil
import gc
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class SolarisDataPipeline:
    """Memory-optimized data pipeline for SOLARIS-X training"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = None
        self.feature_columns = None
        self.target_columns = None
        
    def monitor_memory(self):
        """Monitor current memory usage"""
        memory = psutil.virtual_memory()
        print(f"💾 RAM: {memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB "
              f"({memory.percent:.1f}% used, {memory.available/1024**3:.1f}GB available)")
        
        if memory.percent > 85:
            print("⚠️ WARNING: High memory usage detected - running garbage collection")
            gc.collect()
    
    def load_feature_data(self) -> pd.DataFrame:
        """Load engineered features with memory optimization"""
        print("📁 Loading SOLARIS-X feature dataset...")
        self.monitor_memory()
        
        # Load data with memory optimization
        data_path = self.config.DATA_PATH / self.config.FEATURE_FILE
        
        try:
            # Load with chunked reading for large datasets
            df = pd.read_parquet(data_path)
            print(f"✅ Loaded {len(df):,} records with {len(df.columns)} features")
            
            # Convert datetime if needed
            if self.config.DATETIME_COLUMN in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df[self.config.DATETIME_COLUMN]):
                    df[self.config.DATETIME_COLUMN] = pd.to_datetime(df[self.config.DATETIME_COLUMN])
            
            # Memory optimization
            df = self.optimize_memory_usage(df)
            self.monitor_memory()
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataframe memory usage"""
        print("🔧 Optimizing memory usage...")
        
        start_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.columns:
            if col in [self.config.DATETIME_COLUMN, 'Solar_Cycle']:
                continue
                
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        
        end_memory = df.memory_usage(deep=True).sum() / 1024**2
        reduction = (start_memory - end_memory) / start_memory * 100
        
        print(f"✅ Memory optimized: {start_memory:.2f}MB → {end_memory:.2f}MB "
              f"({reduction:.1f}% reduction)")
        
        return df
    
    def create_temporal_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create temporal train/validation/test splits"""
        print("📅 Creating temporal data splits...")
        
        # Sort by datetime
        df = df.sort_values(self.config.DATETIME_COLUMN).reset_index(drop=True)
        
        # Create splits based on dates
        train_mask = df[self.config.DATETIME_COLUMN] <= pd.to_datetime(self.config.TRAIN_END_DATE)
        val_mask = ((df[self.config.DATETIME_COLUMN] > pd.to_datetime(self.config.TRAIN_END_DATE)) & 
                   (df[self.config.DATETIME_COLUMN] <= pd.to_datetime(self.config.VAL_END_DATE)))
        test_mask = df[self.config.DATETIME_COLUMN] >= pd.to_datetime(self.config.TEST_START_DATE)
        
        splits = {
            'train': df[train_mask].copy().reset_index(drop=True),
            'validation': df[val_mask].copy().reset_index(drop=True), 
            'test': df[test_mask].copy().reset_index(drop=True)
        }
        
        print("📊 Data split summary:")
        for split_name, split_df in splits.items():
            date_range = f"{split_df[self.config.DATETIME_COLUMN].min().date()} to {split_df[self.config.DATETIME_COLUMN].max().date()}"
            storm_rate = split_df[self.config.TARGET_COLUMN].mean() * 100
            print(f"  {split_name:>10}: {len(split_df):>7,} samples | {date_range} | Storm rate: {storm_rate:.1f}%")
        
        return splits
    
    def prepare_features_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Separate features and targets - NO LEAKAGE VERSION"""
    
        # Identify feature and target columns
        exclude_cols = [self.config.DATETIME_COLUMN, 'Solar_Cycle'] + self.config.LEAKAGE_FEATURES
        target_cols = [col for col in df.columns if 'Storm' in col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    
        # Remove any remaining leakage features
        feature_cols = [col for col in feature_cols if not any(leak in col for leak in self.config.LEAKAGE_FEATURES)]
    
        # Store for later use
        self.feature_columns = feature_cols
        self.target_columns = target_cols
    
        print(f"🔍 Features: {len(feature_cols)} columns")
        print(f"🎯 Targets: {len(target_cols)} columns")
        print(f"🚫 Excluded leakage features: {len(self.config.LEAKAGE_FEATURES)}")
    
        # Handle missing values BEFORE splitting
        X = df[feature_cols].copy()
        y = df[target_cols].copy()
        datetime_col = df[self.config.DATETIME_COLUMN].copy()
    
        # Remove rows with missing targets
        valid_mask = ~y[self.config.TARGET_COLUMN].isna()
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)
        datetime_col = datetime_col[valid_mask].reset_index(drop=True)

        # Fill remaining missing values in features
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
        print(f"✅ Clean dataset: {len(X):,} samples ready for training")
        print(f"📊 Feature categories preserved:")
    
        # Categorize remaining features
        physics_features = [col for col in feature_cols if any(x in col for x in ['IMF_', 'Newell_', 'Epsilon_', 'Merging_', 'Alfven_', 'Beta_'])]
        temporal_features = [col for col in feature_cols if any(x in col for x in ['_lag_', '_mean_', '_std_', 'Sin', 'Cos', 'Phase'])]
        basic_features = [col for col in feature_cols if col not in physics_features + temporal_features]
    
        print(f"  🧲 Physics features: {len(physics_features)}")
        print(f"  ⏰ Temporal features: {len(temporal_features)}")  
        print(f"  📊 Basic features: {len(basic_features)}")
    
        return X, y, datetime_col

    
    def scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame = None, 
                      X_test: pd.DataFrame = None) -> Tuple:
        """Scale features using robust scaling"""
        print("⚖️ Scaling features...")
        
        # Use RobustScaler for better handling of outliers in space weather data
        self.scaler = RobustScaler()
        
        # Fit on training data only
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        results = [X_train_scaled]
        
        if X_val is not None:
            X_val_scaled = pd.DataFrame(
                self.scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )
            results.append(X_val_scaled)
        
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            results.append(X_test_scaled)
        
        print("✅ Feature scaling completed")
        return tuple(results)
    
    def compute_class_weights(self, y: pd.Series) -> Dict:
        """Compute class weights for imbalanced data"""
        print("⚖️ Computing class weights for imbalanced data...")
        
        weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        
        class_weights = dict(zip(np.unique(y), weights))
        
        storm_rate = y.mean() * 100
        print(f"📊 Storm rate: {storm_rate:.1f}% | Class weights: {class_weights}")
        
        return class_weights
    
    def prepare_training_data(self) -> Dict[str, Any]:
        """Complete data preparation pipeline - FIXED VERSION"""
        print("🚀 SOLARIS-X Data Pipeline - INITIATED")
        print("=" * 60)
        
        # Load data
        df = self.load_feature_data()
        
        # Create temporal splits  
        splits = self.create_temporal_splits(df)
        
        # Prepare features and targets for each split
        prepared_splits = {}
        
        for split_name, split_df in splits.items():
            X, y, datetime_col = self.prepare_features_targets(split_df)
            prepared_splits[f'X_{split_name}'] = X
            prepared_splits[f'y_{split_name}'] = y[self.config.TARGET_COLUMN]
            prepared_splits[f'datetime_{split_name}'] = datetime_col
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            prepared_splits['X_train'],
            prepared_splits['X_validation'], 
            prepared_splits['X_test']
        )
        
        # Update with scaled features
        prepared_splits['X_train_scaled'] = X_train_scaled
        prepared_splits['X_validation_scaled'] = X_val_scaled
        prepared_splits['X_test_scaled'] = X_test_scaled
        
        # Compute class weights
        class_weights = self.compute_class_weights(prepared_splits['y_train'])
        prepared_splits['class_weights'] = class_weights
        
        # Add metadata
        prepared_splits['feature_columns'] = self.feature_columns
        prepared_splits['scaler'] = self.scaler
        
        print("=" * 60)
        print("✅ SOLARIS-X Data Pipeline - COMPLETE")
        self.monitor_memory()
        
        return prepared_splits
