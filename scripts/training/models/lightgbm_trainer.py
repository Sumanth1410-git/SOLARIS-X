"""
SOLARIS-X LightGBM Trainer - PRODUCTION VERSION
Advanced Gradient Boosting for Space Weather Prediction
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import json
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.utils.base_trainer import BaseModelTrainer

class LightGBMTrainer(BaseModelTrainer):
    """LightGBM trainer - Production ready"""
    
    def __init__(self, config):
        super().__init__(config, "LightGBM")
        self.feature_importance = None
        self.lgb_train = None
        self.lgb_valid = None
        self.feature_names = None
    
    def train_model(self, data: Dict[str, Any]) -> 'LightGBMTrainer':
        """Train LightGBM model with CPU optimization"""
        print(f"\n🌲 Training {self.model_name} Model...")
        print("=" * 50)
        
        # Prepare data
        X_train = data['X_train_scaled']
        y_train = data['y_train']
        X_val = data['X_validation_scaled']
        y_val = data['y_validation']
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Create LightGBM datasets
        self.lgb_train = lgb.Dataset(X_train, label=y_train)
        self.lgb_valid = lgb.Dataset(X_val, label=y_val, reference=self.lgb_train)
        
        # Model parameters
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
            'feature': self.feature_names,
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
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Feature importance plot saved to: {plot_path}")
        
        # Print top features
        print(f"\n🏆 TOP {min(top_n, 10)} MOST IMPORTANT FEATURES:")
        print("-" * 50)
        for i, (_, row) in enumerate(top_features.head(10).iterrows(), 1):
            print(f"  {i:>2}. {row['feature']:<30} {row['importance']:>10.1f}")
        
        return top_features
    
    def save_model(self, data: Dict[str, Any]):
        """Save model with metadata - PRODUCTION VERSION"""
        if self.model is None:
            print("❌ No model to save. Train first.")
            return
        
        # Create save directory
        save_dir = Path("models/production")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save LightGBM model
        model_path = save_dir / f"lightgbm_storm_predictor_{timestamp}.txt"
        self.model.save_model(str(model_path))
        print(f"✅ Model saved: {model_path}")
        
        # 2. Save feature metadata
        metadata = {
            "model_type": "LightGBM",
            "timestamp": timestamp,
            "features": self.feature_names,
            "n_features": len(self.feature_names),
            "best_iteration": self.model.best_iteration,
            "best_validation_auc": self.model.best_score['validation']['auc']
        }
        
        metadata_path = save_dir / f"lightgbm_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Metadata saved: {metadata_path}")
        
        # 3. Save feature importance
        importance_path = save_dir / f"lightgbm_importance_{timestamp}.csv"
        self.feature_importance.to_csv(importance_path, index=False)
        print(f"✅ Feature importance saved: {importance_path}")
        
        print(f"\n🎯 Production model package saved to: {save_dir}/")
