"""
SOLARIS-X Base Model Trainer - FIXED VERSION
NASA-Level Space Weather Prediction Training Framework
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
import time
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class BaseModelTrainer:
    """Base class for all SOLARIS-X model trainers"""
    
    def __init__(self, config, model_name: str):
        self.config = config
        self.model_name = model_name
        self.model = None
        self.training_history = {}
        self.metrics = {}
        
    def calculate_metrics(self, y_true, y_pred, y_proba=None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        # Ensure y_pred is binary (0/1)
        if not np.array_equal(y_pred, y_pred.astype(bool).astype(int)):
            # Convert continuous predictions to binary
            y_pred = (y_pred > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_proba is not None:
            metrics.update({
                'roc_auc': roc_auc_score(y_true, y_proba),
                'average_precision': average_precision_score(y_true, y_proba),
                'brier_score': brier_score_loss(y_true, y_proba)
            })
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, split_name: str):
        """Plot and save confusion matrix"""
        
        # Ensure binary predictions
        y_pred_binary = (y_pred > 0.5).astype(int) if y_pred.dtype == float else y_pred
        
        cm = confusion_matrix(y_true, y_pred_binary)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Storm', 'Storm'],
                   yticklabels=['No Storm', 'Storm'])
        
        plt.title(f'{self.model_name} - Confusion Matrix ({split_name})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save plot
        plot_path = self.config.RESULTS_PATH / "plots" / f"{self.model_name}_{split_name}_confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return cm
    
    def plot_roc_curve(self, y_true, y_proba, split_name: str):
        """Plot ROC curve"""
        from sklearn.metrics import roc_curve
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.model_name} - ROC Curve ({split_name})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = self.config.RESULTS_PATH / "plots" / f"{self.model_name}_{split_name}_roc_curve.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_model(self, data: Dict, split_name: str = 'validation') -> Dict[str, Any]:
        """Comprehensive model evaluation - FIXED VERSION"""
        
        # Get appropriate data split
        X = data[f'X_{split_name}_scaled']
        y = data[f'y_{split_name}']
        
        # Make predictions - HANDLE DIFFERENT MODEL TYPES
        if hasattr(self.model, 'predict'):
            y_pred_raw = self.model.predict(X)
        else:
            raise AttributeError(f"Model {self.model_name} doesn't have predict method")
        
        # Handle LightGBM probability outputs
        if hasattr(self.model, 'predict_proba'):
            # Standard sklearn-style models
            y_proba = self.model.predict_proba(X)[:, 1]
            y_pred = (y_proba > 0.5).astype(int)
        elif str(type(self.model)).find('lightgbm') >= 0:
            # LightGBM returns probabilities directly from predict()
            y_proba = y_pred_raw
            y_pred = (y_proba > 0.5).astype(int)
        elif hasattr(self.model, 'decision_function'):
            # SVM-style models
            decision = self.model.decision_function(X)
            y_pred = (decision > 0).astype(int)
            # Convert decision to probabilities (sigmoid approximation)
            y_proba = 1 / (1 + np.exp(-decision))
        else:
            # Binary predictions only
            y_pred = (y_pred_raw > 0.5).astype(int) if y_pred_raw.dtype == float else y_pred_raw
            y_proba = None
        
        # Calculate metrics
        metrics = self.calculate_metrics(y, y_pred, y_proba)
        
        # Generate plots
        self.plot_confusion_matrix(y, y_pred, split_name)
        if y_proba is not None:
            self.plot_roc_curve(y, y_proba, split_name)
        
        # Store results
        self.metrics[split_name] = metrics
        
        print(f"\n📊 {self.model_name} - {split_name.upper()} RESULTS:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"  {metric:>18}: {value:.4f}")
        
        return metrics
    
    def save_model(self, data: Dict):
        """Save trained model and metadata"""
        
        # Create model directory
        model_dir = self.config.MODEL_PATH / "trained" / self.model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pkl"
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = model_dir / "scaler.pkl" 
        joblib.dump(data['scaler'], scaler_path)
        
        # Save feature columns
        features_path = model_dir / "features.json"
        with open(features_path, 'w') as f:
            json.dump(data['feature_columns'], f)
        
        # Save metrics
        metrics_path = model_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save training history if available
        if self.training_history:
            history_path = model_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
        
        print(f"Model saved to: {model_dir}")
        
        return model_dir
    
    def load_model(self, model_dir: Path):
        """Load saved model and metadata"""
        
        # Load model
        model_path = model_dir / "model.pkl"
        self.model = joblib.load(model_path)
        
        # Load metrics if available
        metrics_path = model_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
        
        print(f"Model loaded from: {model_dir}")
        
        return self