"""
SOLARIS-X Ensemble LightGBM Trainer - PRODUCTION VERSION (FINAL WORKING)
Weighted ensemble of specialized LightGBM models for improved recall & precision
Uses PROVEN parameters from your working baseline model
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import joblib
import json
from datetime import datetime
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class EnsembleLightGBM:
    """
    Weighted ensemble of three specialized LightGBM models:
    - Model 1: Precision-focused (conservative, fewer false alarms)
    - Model 2: Recall-focused (aggressive, catches more storms)
    - Model 3: Balanced (original performance)
    """
    
    def __init__(self, config):
        self.config = config
        self.model_precision = None
        self.model_recall = None
        self.model_balanced = None
        
        # Default weights (will be optimized later)
        self.weights = {
            'precision': 0.2,
            'recall': 0.5,
            'balanced': 0.3
        }
        
        self.feature_names = None
        self.training_history = []
    
    def train_all_models(self, data: Dict[str, Any]) -> 'EnsembleLightGBM':
        """Train all three specialized models"""
        print("\n" + "="*70)
        print("🎯 ENSEMBLE TRAINING: 3 Specialized LightGBM Models")
        print("="*70)
        
        X_train = data['X_train_scaled']
        y_train = data['y_train']
        X_val = data['X_validation_scaled']
        y_val = data['y_validation']
        
        self.feature_names = list(X_train.columns)
        
        # ====================================================================
        # MODEL 1: PRECISION-FOCUSED (Conservative - Fewer False Alarms)
        # ====================================================================
        print("\n🎯 [1/3] Training PRECISION-FOCUSED Model...")
        print("    Goal: Minimize false alarms (conservative predictions)")
        
        # 🔥 Using PROVEN parameters from your working baseline + conservative tweaks
        params_precision = {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 30,  # 🔥 Conservative: higher = fewer splits
            'class_weight': 'balanced',  # 🔥 PROVEN to work in your baseline
            'n_jobs': self.config.N_JOBS,
            'random_state': self.config.RANDOM_STATE,
            'verbose': 1
        }
        
        lgb_train_precision = lgb.Dataset(X_train, label=y_train)
        lgb_val_precision = lgb.Dataset(X_val, label=y_val, reference=lgb_train_precision)
        
        self.model_precision = lgb.train(
            params_precision,
            lgb_train_precision,
            valid_sets=[lgb_train_precision, lgb_val_precision],
            valid_names=['train', 'validation'],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=50)
            ]
        )
        
        val_metrics_precision = self._evaluate_model(
            self.model_precision, X_val, y_val, "Precision Model"
        )
        self.training_history.append({
            'model': 'precision',
            'specialization': 'conservative',
            'min_child_samples': 30,
            **val_metrics_precision
        })
        
        print(f"✅ Precision Model trained. Best iteration: {self.model_precision.best_iteration}")
        
        # ====================================================================
        # MODEL 2: RECALL-FOCUSED (Aggressive - Catch More Storms)
        # ====================================================================
        print("\n🎯 [2/3] Training RECALL-FOCUSED Model...")
        print("    Goal: Maximize storm detection (catch more storms)")
        
        params_recall = {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 10,  # 🔥 Aggressive: lower = more splits
            'class_weight': 'balanced',
            'n_jobs': self.config.N_JOBS,
            'random_state': self.config.RANDOM_STATE,
            'verbose': 1
        }
        
        lgb_train_recall = lgb.Dataset(X_train, label=y_train)
        lgb_val_recall = lgb.Dataset(X_val, label=y_val, reference=lgb_train_recall)
        
        self.model_recall = lgb.train(
            params_recall,
            lgb_train_recall,
            valid_sets=[lgb_train_recall, lgb_val_recall],
            valid_names=['train', 'validation'],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=50)
            ]
        )
        
        val_metrics_recall = self._evaluate_model(
            self.model_recall, X_val, y_val, "Recall Model"
        )
        self.training_history.append({
            'model': 'recall',
            'specialization': 'aggressive',
            'min_child_samples': 10,
            **val_metrics_recall
        })
        
        print(f"✅ Recall Model trained. Best iteration: {self.model_recall.best_iteration}")
        
        # ====================================================================
        # MODEL 3: BALANCED (Your Working Baseline Configuration)
        # ====================================================================
        print("\n🎯 [3/3] Training BALANCED Model...")
        print("    Goal: Balanced performance (baseline)")
        
        params_balanced = {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'class_weight': 'balanced',  # 🔥 Exact same as your working baseline
            'n_jobs': self.config.N_JOBS,
            'random_state': self.config.RANDOM_STATE,
            'verbose': 1
        }
        
        lgb_train_balanced = lgb.Dataset(X_train, label=y_train)
        lgb_val_balanced = lgb.Dataset(X_val, label=y_val, reference=lgb_train_balanced)
        
        self.model_balanced = lgb.train(
            params_balanced,
            lgb_train_balanced,
            valid_sets=[lgb_train_balanced, lgb_val_balanced],
            valid_names=['train', 'validation'],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=50)
            ]
        )
        
        val_metrics_balanced = self._evaluate_model(
            self.model_balanced, X_val, y_val, "Balanced Model"
        )
        self.training_history.append({
            'model': 'balanced',
            'specialization': 'balanced',
            'min_child_samples': 20,
            **val_metrics_balanced
        })
        
        print(f"✅ Balanced Model trained. Best iteration: {self.model_balanced.best_iteration}")
        
        print("\n" + "="*70)
        print("✅ ALL THREE MODELS TRAINED SUCCESSFULLY!")
        print("="*70)
        
        return self
    
    def _evaluate_model(self, model, X, y_true, model_name: str) -> Dict:
        """Evaluate a single model"""
        y_proba = model.predict(X)
        y_pred = (y_proba >= 0.5).astype(int)
        
        metrics = {
            'auc': roc_auc_score(y_true, y_proba),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        print(f"\n    📊 {model_name} Validation Metrics:")
        print(f"       AUC:       {metrics['auc']:.4f}")
        print(f"       Precision: {metrics['precision']:.4f}")
        print(f"       Recall:    {metrics['recall']:.4f}")
        print(f"       F1-Score:  {metrics['f1']:.4f}")
        
        return metrics
    
    def set_weights(self, w_precision: float, w_recall: float, w_balanced: float):
        """Update ensemble weights (normalized to sum to 1.0)"""
        total = w_precision + w_recall + w_balanced
        self.weights = {
            'precision': w_precision / total,
            'recall': w_recall / total,
            'balanced': w_balanced / total
        }
    
    def predict_proba(self, X) -> np.ndarray:
        """Get ensemble probability predictions"""
        proba_precision = self.model_precision.predict(X)
        proba_recall = self.model_recall.predict(X)
        proba_balanced = self.model_balanced.predict(X)
        
        # Weighted combination
        ensemble_proba = (
            self.weights['precision'] * proba_precision +
            self.weights['recall'] * proba_recall +
            self.weights['balanced'] * proba_balanced
        )
        
        return ensemble_proba
    
    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Get binary predictions"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def evaluate(self, X, y_true, threshold: float = 0.5) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """Evaluate ensemble performance"""
        y_proba = self.predict_proba(X)
        y_pred = (y_proba >= threshold).astype(int)
        
        metrics = {
            'auc': roc_auc_score(y_true, y_proba),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        return metrics, y_pred, y_proba
    
    def save_ensemble(self, save_dir: Path, timestamp: str = None):
        """Save all ensemble models and configuration"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving Ensemble Models...")
        
        # Save individual models
        self.model_precision.save_model(str(save_dir / f"model_precision_{timestamp}.txt"))
        self.model_recall.save_model(str(save_dir / f"model_recall_{timestamp}.txt"))
        self.model_balanced.save_model(str(save_dir / f"model_balanced_{timestamp}.txt"))
        
        print(f"✅ Individual models saved")
        
        # Save ensemble configuration
        ensemble_config = {
            'timestamp': timestamp,
            'weights': self.weights,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'training_history': self.training_history,
            'model_files': {
                'precision': f"model_precision_{timestamp}.txt",
                'recall': f"model_recall_{timestamp}.txt",
                'balanced': f"model_balanced_{timestamp}.txt"
            }
        }
        
        config_path = save_dir / f"ensemble_config_{timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2)
        
        print(f"✅ Configuration saved: {config_path}")
        
        # Save complete ensemble object
        ensemble_path = save_dir / f"ensemble_lightgbm_{timestamp}.pkl"
        joblib.dump(self, ensemble_path)
        
        print(f"✅ Complete ensemble saved: {ensemble_path}")
        print(f"\n📁 All artifacts saved to: {save_dir}/")
        
        return timestamp
