import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import numpy as np
from scripts.training.config import config
from scripts.training.utils.data_loader import SolarisDataPipeline
from scripts.training.models.enhanced_bigru_trainer import EnhancedBiGRUTrainer
from scripts.training.models.lightgbm_trainer import LightGBMTrainer
from scripts.training.models.xgboost_trainer import XGBoostTrainer
from sklearn.metrics import roc_auc_score

# Prepare data
pipeline = SolarisDataPipeline(config)
data = pipeline.prepare_training_data()

# Train models
bigru = EnhancedBiGRUTrainer(config).train_enhanced_model(data)
lgbm = LightGBMTrainer(config).train_model(data)
xgb = XGBoostTrainer(config).train_model(data)

# Get validation predictions
# (Assume you have a function to create validation sequences for BiGRU)
# Replace X_val_seq, y_val_seq with your actual sequence creation logic
X_val_seq, y_val_seq = bigru.create_sequences_safe(
    data['X_validation_scaled'], data['y_validation'], data['datetime_validation'],
    sequence_length=config.MODELS['neural_network']['sequence_length']
)
y_pred_bigru = bigru.model.predict(X_val_seq).flatten()
y_pred_lgbm = lgbm.model.predict_proba(data['X_validation_scaled'])[:, 1]
y_pred_xgb = xgb.model.predict_proba(data['X_validation_scaled'])[:, 1]

# Simple average ensemble
ensemble_pred = (y_pred_bigru + y_pred_lgbm + y_pred_xgb) / 3
auc_ensemble = roc_auc_score(y_val_seq, ensemble_pred)
print(f'Ensemble Validation AUC: {auc_ensemble:.4f}')

# For stacking, use sklearn's StackingClassifier (see previous message for example)
