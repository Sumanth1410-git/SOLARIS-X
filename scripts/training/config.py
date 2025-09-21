"""
SOLARIS-X Training Configuration - FIXED VERSION
NASA-Level Space Weather Prediction System
CPU-Optimized for 16GB RAM Environment - NO DATA LEAKAGE
"""

import os
from pathlib import Path
import numpy as np

class SolarisConfig:
    """Master configuration for SOLARIS-X training pipeline"""
    
    # Project Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_PATH = PROJECT_ROOT / "data/processed/features"
    MODEL_PATH = PROJECT_ROOT / "models"
    RESULTS_PATH = PROJECT_ROOT / "results"
    
    # Data Configuration
    FEATURE_FILE = "solaris_x_features.parquet"
    TARGET_COLUMN = "Storm_Binary"  # Primary target for binary classification
    DATETIME_COLUMN = "Datetime"
    
    # CRITICAL: Features to EXCLUDE to prevent data leakage
    LEAKAGE_FEATURES = [
        'Dst_index',           # Direct target source - MUST EXCLUDE
        'Storm_Binary',        # Target itself
        'Storm_Intensity',     # Based on Dst
        'Storm_Onset',         # Based on Dst derivatives  
        'Storm_6h_Ahead',      # Future information
        'Dst_Rate'             # Dst derivative (leakage)
    ]
    
    # CPU Optimization Settings
    N_JOBS = min(12, os.cpu_count())  # Use available cores efficiently
    RANDOM_STATE = 42
    
    # Memory Management (16GB RAM Optimization)
    MAX_MEMORY_GB = 12  # Reserve 4GB for system
    BATCH_SIZE = 1000   # Conservative batch size for CPU training
    
    # Data Splits (Temporal)
    TRAIN_END_DATE = "2016-12-31"    # 1996-2016: Training (20 years)
    VAL_END_DATE = "2020-12-31"      # 2017-2020: Validation (4 years)
    TEST_START_DATE = "2021-01-01"   # 2021-2025: Test (5 years)
    
    # Feature Engineering
    SEQUENCE_LENGTH = 72  # 72-hour lookback window
    PREDICTION_HORIZON = 6  # 6-hour ahead prediction
    
    # Model-Specific Configurations - ANTI-OVERFITTING
    MODELS = {
        'lightgbm': {
            'n_estimators': 200,      # Reduced from 1000
            'max_depth': 6,           # Reduced from 8  
            'num_leaves': 31,         # Reduced from 127
            'learning_rate': 0.05,    # Reduced from 0.1
            'feature_fraction': 0.7,  # Feature subsampling
            'bagging_fraction': 0.8,  # Row subsampling
            'bagging_freq': 5,
            'min_child_samples': 100, # Prevent small leaf nodes
            'reg_alpha': 0.1,         # L1 regularization
            'reg_lambda': 0.1,        # L2 regularization
            'n_jobs': N_JOBS,
            'random_state': RANDOM_STATE,
            'verbose': -1
        },
        'xgboost': {
            'n_estimators': 200,      # Reduced
            'max_depth': 5,           # Reduced
            'learning_rate': 0.05,    # Reduced
            'subsample': 0.8,         # Row subsampling
            'colsample_bytree': 0.7,  # Feature subsampling
            'min_child_weight': 10,   # Prevent overfitting
            'reg_alpha': 0.1,         # L1 regularization
            'reg_lambda': 0.1,        # L2 regularization
            'n_jobs': N_JOBS,
            'random_state': RANDOM_STATE,
            'tree_method': 'hist'     # CPU optimized
        },
        'neural_network': {
            'sequence_length': SEQUENCE_LENGTH,
            'lstm_units': 32,         # Reduced from 64
            'dropout_rate': 0.3,      # Increased dropout
            'batch_size': 64,         # Increased batch size
            'epochs': 30,             # Reduced epochs
            'patience': 5,            # Earlier stopping
            'learning_rate': 0.001
        }
    }
    
    # Cross-Validation Settings
    CV_FOLDS = 5
    CV_SCORING = ['roc_auc', 'precision', 'recall', 'f1']
    
    # Hyperparameter Optimization
    OPTUNA_TRIALS = 50            # Reduced for faster execution
    OPTUNA_TIMEOUT = 1800         # 30 minutes max per model
    
    # Ensemble Configuration
    ENSEMBLE_METHODS = ['voting', 'stacking']
    META_MODEL = 'lightgbm'
    
    # Evaluation Metrics
    METRICS = [
        'accuracy', 'precision', 'recall', 'f1_score', 
        'roc_auc', 'average_precision', 'brier_score'
    ]
    
    # Storm Classification Thresholds (for reference only)
    STORM_THRESHOLDS = {
        'weak': -30,      # Dst < -30 nT
        'moderate': -50,  # Dst < -50 nT  
        'strong': -100,   # Dst < -100 nT
        'severe': -200    # Dst < -200 nT
    }

# Global configuration instance
config = SolarisConfig()
