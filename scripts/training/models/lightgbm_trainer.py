"""
SOLARIS-X LightGBM Trainer
Advanced Gradient Boosting for Space Weather Prediction
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.utils.base_trainer import BaseModelTrainer

class LightGBMTrainer(BaseModelTrainer):
    """LightGBM trainer with advanced features"""
    
    def __init__(self, config):
        super().__init__(config, "LightGBM")
        self.feature_importance = None
        self.lgb_train = None
        self.lgb_valid = None
        
    def train_model(self, data: Dict[str, Any]) -> 'LightGBMTrainer':
        """Train LightGBM model with CPU optimization"""
        
        print(f"\n🌲 Training {self.model_name} Model...")
        print("=" * 50)
        
        # Prepare data
        X_train = data['X_train_scaled']
        y_train = data['y_train']
        X_val = data['X_validation_scaled']
        y_val = data['y_validation']
        
        # Create LightGBM datasets
        self.lgb_train = lgb.Dataset(X_train, label=y_train)
        self.lgb_valid = lgb.Dataset(X_val, label=y_val, reference=self.lgb_train)
        
        # Model parameters optimized for CPU and imbalanced data
        params = {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': self.config.N_JOBS,
            'class_weight': 'balanced',
            'random_state': self.config.RANDOM_STATE
        }
        
        # Training callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50)
        ]
        
        # Train model
        print("🚀 Starting LightGBM training...")
        self.model = lgb.train(
            params,
            self.lgb_train,
            valid_sets=[self.lgb_train, self.lgb_valid],
            valid_names=['train', 'validation'],
            num_boost_round=1000,
            callbacks=callbacks
        )
        
        # Extract feature importance
        self.feature_importance = pd.DataFrame({
            'feature': data['feature_columns'],
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        print(f"✅ LightGBM training completed!")
        print(f"📊 Best validation AUC: {self.model.best_score['validation']['auc']:.4f}")
        
        return self
    
    def plot_feature_importance(self, top_n: int = 20):
        """Plot top feature importances"""
        
        if self.feature_importance is None:
            print("❌ No feature importance available. Train model first.")
            return
        
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance (Gain)')
        plt.title(f'{self.model_name} - Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Save plot
        plot_path = self.config.RESULTS_PATH / "plots" / f"{self.model_name}_feature_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Feature importance plot saved to: {plot_path}")
        
        # Print top features
        print(f"\n🏆 TOP {min(top_n, 10)} MOST IMPORTANT FEATURES:")
        print("-" * 50)
        for i, (_, row) in enumerate(top_features.head(10).iterrows(), 1):
            print(f"  {i:>2}. {row['feature']:<30} {row['importance']:>10.1f}")
    
    def predict_with_uncertainty(self, X: pd.DataFrame, n_samples: int = 100) -> tuple:
        """Make predictions with uncertainty estimation"""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Base prediction
        y_pred_proba = self.model.predict(X, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Simple uncertainty estimate based on prediction confidence
        uncertainty = np.abs(y_pred_proba - 0.5)  # Distance from decision boundary
        
        return y_pred, y_pred_proba, uncertainty
