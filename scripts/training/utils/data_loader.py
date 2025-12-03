def safe_sequence_split(X, y, datetime_col, split_date, sequence_length):
    """
    Create non-overlapping train/val sequences for time series.
    X, y, datetime_col: full dataset (chronologically sorted)
    split_date: the datetime at which to split (e.g., '2017-01-01')
    sequence_length: number of timesteps in each sequence
    Returns: X_train_seq, y_train_seq, X_val_seq, y_val_seq
    """
    import pandas as pd
    # Reset all indices to ensure alignment
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    datetime_col = pd.Series(datetime_col).reset_index(drop=True)
    # Ensure datetime_col is sorted
    datetime_col = pd.Series(datetime_col).sort_values().reset_index(drop=True)
    X = X.loc[datetime_col.index].reset_index(drop=True)
    y = y.loc[datetime_col.index].reset_index(drop=True)

    split_idx = datetime_col.searchsorted(pd.to_datetime(split_date))

    # Training: sequences ending strictly before split_idx
    train_indices = [i for i in range(sequence_length, split_idx)]
    # Validation: sequences starting at split_idx (first val seq uses only val data)
    val_indices = [i for i in range(split_idx + sequence_length, len(X))]

    def build_sequences(indices):
        X_seq = np.array([X.iloc[i-sequence_length:i].values for i in indices])
        y_seq = np.array([y.iloc[i] for i in indices])
        return X_seq, y_seq

    X_train_seq, y_train_seq = build_sequences(train_indices)
    X_val_seq, y_val_seq = build_sequences(val_indices)
    return X_train_seq, y_train_seq, X_val_seq, y_val_seq
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
        """Load CORRECTED engineered features with proper storm definition"""
        print("📁 Loading CORRECTED SOLARIS-X feature dataset...")
        self.monitor_memory()
        
        # FIXED: Simple absolute path to corrected features
        data_path = Path('data/processed/features/solaris_x_features_corrected.parquet')
        
        try:
            # Load corrected data
            df = pd.read_parquet(data_path)
            print(f"✅ Loaded CORRECTED data: {len(df):,} records with {len(df.columns)} features")
            
            # Verify corrected storm rates
            if 'Storm' in df.columns:
                storm_rate = df['Storm'].mean() * 100
                print(f"🌪️  CORRECTED Storm rate (Kp≥5): {storm_rate:.1f}%")
                
                if storm_rate > 20:
                    print("🚨 WARNING: Storm rate still too high! Check storm definition.")
                else:
                    print("✅ Storm rate looks realistic")
            
            # Memory optimization
            df = self.optimize_memory_usage(df)
            self.monitor_memory()
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading corrected data: {e}")
            print(f"💡 Looking for file at: {data_path.absolute()}")
            print("💡 Make sure you're in the SOLARIS-X root directory!")
            raise
    
    def optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataframe memory usage"""
        print("🔧 Optimizing memory usage...")
        
        start_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df.index):
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
        """Create temporal train/validation/test splits using datetime index"""
        print("📅 Creating temporal data splits...")
        
        # Use datetime index for splitting
        train_mask = df.index.year <= 2016
        val_mask = (df.index.year >= 2017) & (df.index.year <= 2020)
        test_mask = df.index.year >= 2021
        
        splits = {
            'train': df[train_mask].copy(),
            'validation': df[val_mask].copy(), 
            'test': df[test_mask].copy()
        }
        
        print("📊 CORRECTED Data split summary:")
        for split_name, split_df in splits.items():
            date_range = f"{split_df.index.min().date()} to {split_df.index.max().date()}"
            storm_rate = split_df['Storm'].mean() * 100
            print(f"  {split_name:>10}: {len(split_df):>7,} samples | {date_range} | Storm rate: {storm_rate:.1f}%")
        
        return splits
    
    def prepare_features_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Separate features and targets with corrected storm definition - FIXED DATETIME HANDLING"""
        
        # CORRECTED: Use proper storm column
        target_col = 'Storm'
        exclude_cols = ['Storm', 'Moderate_Storm', 'Kp_actual', 'Kp_index']
        
        # Get feature columns (exclude storm-related columns)
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"🔍 CORRECTED Features: {len(feature_cols)} columns")
        print(f"🎯 Target: {target_col} (corrected storm definition)")
        
        # Prepare data
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # FIXED: Convert datetime index to Series for compatibility
        datetime_series = pd.Series(df.index, index=df.index)
        
        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Verify target distribution
        storm_rate = y.mean() * 100
        print(f"✅ Clean dataset: {len(X):,} samples, storm rate: {storm_rate:.1f}%")
        
        return X, y, datetime_series
    
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
        print(f"📊 CORRECTED Storm rate: {storm_rate:.1f}% | Class weights: {class_weights}")
        
        return class_weights
    
    def prepare_training_data(self) -> Dict[str, Any]:
        """Complete CORRECTED data preparation pipeline"""
        print("🚀 SOLARIS-X CORRECTED Data Pipeline - INITIATED")
        print("=" * 60)
        
        # Load corrected data
        df = self.load_feature_data()
        
        # Create temporal splits  
        splits = self.create_temporal_splits(df)
        
        # Prepare features and targets for each split
        prepared_splits = {}
        
        for split_name, split_df in splits.items():
            X, y, datetime_series = self.prepare_features_targets(split_df)
            prepared_splits[f'X_{split_name}'] = X
            prepared_splits[f'y_{split_name}'] = y
            prepared_splits[f'datetime_{split_name}'] = datetime_series
        
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
        prepared_splits['feature_columns'] = list(X_train_scaled.columns)
        prepared_splits['scaler'] = self.scaler
        
        print("=" * 60)
        print("✅ SOLARIS-X CORRECTED Data Pipeline - COMPLETE")
        self.monitor_memory()
        
        return prepared_splits
