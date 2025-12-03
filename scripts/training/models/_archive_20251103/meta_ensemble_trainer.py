"""
SOLARIS-X Meta-Ensemble Trainer - FINAL WORKING VERSION
Combines LightGBM + Enhanced BiGRU via Meta-Learning
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import joblib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.utils.base_trainer import BaseModelTrainer

class MetaEnsembleTrainer(BaseModelTrainer):
    """Meta-learning ensemble combining multiple models"""
    
    def __init__(self, config):
        super().__init__(config, "Meta_Ensemble")
        self.base_models = {}
        self.meta_learner = None
        self.model = None  # This will be the meta_learner
        
    def load_trained_models(self):
        """Load pre-trained base models"""
        print("📁 Loading pre-trained base models...")
        
        # Load LightGBM
        lgb_path = self.config.MODEL_PATH / "trained" / "LightGBM" / "model.pkl"
        if lgb_path.exists():
            self.base_models['lightgbm'] = joblib.load(lgb_path)
            print("✅ LightGBM model loaded")
        else:
            print("❌ LightGBM model not found - train it first")
            
        # Load Enhanced BiGRU - Skip if failed training
        bigru_path = self.config.MODEL_PATH / "checkpoints" / "enhanced_bigru_best.h5"
        try:
            if bigru_path.exists():
                import tensorflow as tf
                # Load without custom objects to avoid errors
                self.base_models['enhanced_bigru'] = tf.keras.models.load_model(
                    bigru_path, 
                    compile=False
                )
                print("✅ Enhanced BiGRU model loaded")
            else:
                print("⚠️ Enhanced BiGRU model not found - using LightGBM only")
        except Exception as e:
            print(f"⚠️ Enhanced BiGRU loading failed: {e} - using LightGBM only")
    
    def generate_meta_features(self, X_data, datetime_data, split_name: str):
        """Generate meta-features from base model predictions - NaN-PROOF VERSION"""
        print(f"🔮 Generating meta-features for {split_name}...")
        
        meta_features = []
        
        # LightGBM predictions (direct on scaled features)
        if 'lightgbm' in self.base_models:
            lgb_proba = self.base_models['lightgbm'].predict(X_data)
            
            # Handle any potential NaNs in LightGBM predictions
            lgb_proba = np.nan_to_num(lgb_proba, nan=0.5, posinf=1.0, neginf=0.0)
            meta_features.append(lgb_proba)
            print(f"  ✅ LightGBM meta-features: shape {lgb_proba.shape}")
            
            # Add LightGBM confidence (distance from 0.5)
            lgb_confidence = np.abs(lgb_proba - 0.5)
            meta_features.append(lgb_confidence)
            print(f"  ✅ LightGBM confidence: shape {lgb_confidence.shape}")
        
        # Enhanced BiGRU predictions (if available and working)
        if 'enhanced_bigru' in self.base_models:
            try:
                # Create simplified sequences for BiGRU
                sequence_length = 72
                
                # Use rolling window approach for sequences
                sequences_X = []
                valid_indices = []
                
                for i in range(sequence_length, len(X_data)):
                    sequences_X.append(X_data.iloc[i-sequence_length:i].values)
                    valid_indices.append(i)
                
                if sequences_X:
                    X_seq = np.array(sequences_X, dtype=np.float32)
                    bigru_proba = self.base_models['enhanced_bigru'].predict(X_seq, verbose=0).flatten()
                    
                    # Handle NaNs in BiGRU predictions
                    bigru_proba = np.nan_to_num(bigru_proba, nan=0.5, posinf=1.0, neginf=0.0)
                    
                    # Create full-length predictions
                    full_bigru_proba = np.full(len(X_data), 0.5)  # Default neutral
                    full_bigru_proba[valid_indices] = bigru_proba
                    
                    # Fill initial values with mean
                    if len(bigru_proba) > 0:
                        initial_value = np.nanmean(bigru_proba[:min(10, len(bigru_proba))])
                        if np.isnan(initial_value):
                            initial_value = 0.5
                        full_bigru_proba[:sequence_length] = initial_value
                    
                    meta_features.append(full_bigru_proba)
                    print(f"  ✅ Enhanced BiGRU meta-features: shape {full_bigru_proba.shape}")
            except Exception as e:
                print(f"  ⚠️ BiGRU prediction failed: {e} - skipping BiGRU features")
        
        # Statistical features with NaN handling (FIXED)
        try:
            # Use first few columns for rolling statistics (most stable features)
            feature_cols = X_data.columns[:2]  # Use first 2 most stable features
            
            for col in feature_cols:
                # Rolling mean with NaN handling
                rolling_mean = X_data[col].rolling(window=24, min_periods=1).mean()
                rolling_mean = rolling_mean.fillna(rolling_mean.median())
                rolling_mean = np.nan_to_num(rolling_mean, nan=0.0)
                
                meta_features.append(rolling_mean.values)  # FIXED: .values for pandas Series
                print(f"  ✅ Rolling mean {col}: added")
            
            print(f"  ✅ Statistical meta-features added (NaN-safe)")
            
        except Exception as e:
            print(f"  ⚠️ Statistical features failed: {e} - using fallback")
            # Fallback: simple feature averages
            fallback_feature = np.mean(X_data.iloc[:, :5].values, axis=1)  # FIXED: .values
            fallback_feature = np.nan_to_num(fallback_feature, nan=0.0)
            meta_features.append(fallback_feature)
            print(f"  ✅ Fallback meta-features added")
        
        # Combine meta-features with comprehensive NaN cleaning
        if meta_features:
            meta_features_array = np.column_stack(meta_features)
            
            # CRITICAL: Final NaN cleaning pass
            print(f"🔍 Pre-cleaning meta-features shape: {meta_features_array.shape}")
            print(f"🔍 NaNs detected: {np.isnan(meta_features_array).sum()}")
            
            # Replace all NaNs with appropriate defaults
            meta_features_array = np.nan_to_num(
                meta_features_array, 
                nan=0.5,     # Default neutral prediction
                posinf=1.0,  # Cap positive infinity
                neginf=0.0   # Cap negative infinity
            )
            
            # Final validation
            nan_count_after = np.isnan(meta_features_array).sum()
            inf_count = np.isinf(meta_features_array).sum()
            
            print(f"🎯 Final meta-features shape: {meta_features_array.shape}")
            print(f"✅ NaNs after cleaning: {nan_count_after}")
            print(f"✅ Infinities after cleaning: {inf_count}")
            
            if nan_count_after == 0 and inf_count == 0:
                print(f"🏆 Meta-features are clean and ready!")
            
            return meta_features_array
        else:
            raise ValueError("No meta-features could be generated")
    
    def train_meta_learner(self, data: dict) -> 'MetaEnsembleTrainer':
        """Train meta-learner on base model predictions"""
        print(f"\n🎯 Training {self.model_name}...")
        print("=" * 60)
        
        # Load base models
        self.load_trained_models()
        
        # Generate meta-features for training
        X_train_meta = self.generate_meta_features(
            data['X_train_scaled'], 
            data['datetime_train'], 
            'train'
        )
        
        X_val_meta = self.generate_meta_features(
            data['X_validation_scaled'], 
            data['datetime_validation'], 
            'validation'
        )
        
        # Train meta-learner (Logistic Regression optimized for recall)
        self.meta_learner = LogisticRegression(
            class_weight='balanced',
            C=1.0,  # Less regularization for better fit
            max_iter=1000,
            random_state=self.config.RANDOM_STATE
        )
        
        print("🚀 Training meta-learner...")
        self.meta_learner.fit(X_train_meta, data['y_train'])
        
        # CRITICAL: Set model attribute for BaseTrainer compatibility
        self.model = self.meta_learner
        
        # Validation performance
        val_pred = self.meta_learner.predict(X_val_meta)
        val_proba = self.meta_learner.predict_proba(X_val_meta)[:, 1]
        
        val_metrics = self.calculate_metrics(data['y_validation'], val_pred, val_proba)
        self.metrics['validation'] = val_metrics
        
        # Store meta-features for later use
        self.train_meta_features = X_train_meta
        self.val_meta_features = X_val_meta
        
        print("✅ Meta-learner training completed!")
        print(f"\n📊 Meta-Ensemble Validation Results:")
        print("-" * 45)
        for metric, value in val_metrics.items():
            print(f"  {metric:>18}: {value:.4f}")
        
        return self
    
    def evaluate_model(self, data: dict, split_name: str = 'validation') -> dict:
        """Custom evaluation using meta-features - OVERRIDDEN METHOD"""
        print(f"\n📊 {self.model_name} - Evaluating on {split_name.upper()}...")
        
        # Generate meta-features for the evaluation split
        X_raw = data[f'X_{split_name}_scaled']
        y_true = data[f'y_{split_name}']
        datetime_data = data[f'datetime_{split_name}']
        
        # Generate meta-features (same as training)
        X_meta = self.generate_meta_features(X_raw, datetime_data, split_name)
        
        # Make predictions using meta-features
        y_pred = self.meta_learner.predict(X_meta)
        y_proba = self.meta_learner.predict_proba(X_meta)[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_proba)
        
        # Generate plots
        self.plot_confusion_matrix(y_true, y_pred, split_name)
        if y_proba is not None:
            self.plot_roc_curve(y_true, y_proba, split_name)
        
        # Store results
        self.metrics[split_name] = metrics
        
        print(f"📊 {self.model_name} - {split_name.upper()} RESULTS:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"  {metric:>18}: {value:.4f}")
        
        return metrics
    
    def predict(self, X_data):
        """Make ensemble predictions - REQUIRED METHOD"""
        if self.meta_learner is None:
            raise ValueError("Meta-learner not trained. Call train_meta_learner() first.")
        
        # Generate meta-features for prediction
        datetime_dummy = pd.Series(range(len(X_data)))  # Dummy datetime for compatibility
        meta_features = self.generate_meta_features(X_data, datetime_dummy, 'inference')
        
        # Meta-learner prediction
        predictions = self.meta_learner.predict(meta_features)
        return predictions
    
    def predict_proba(self, X_data):
        """Make ensemble probability predictions"""
        if self.meta_learner is None:
            raise ValueError("Meta-learner not trained. Call train_meta_learner() first.")
        
        # Generate meta-features for prediction
        datetime_dummy = pd.Series(range(len(X_data)))  # Dummy datetime for compatibility
        meta_features = self.generate_meta_features(X_data, datetime_dummy, 'inference')
        
        # Meta-learner prediction
        probabilities = self.meta_learner.predict_proba(meta_features)
        return probabilities
