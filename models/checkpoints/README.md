# SOLARIS-X Model Checkpoints

**Contents:** Trained model checkpoints and weights

## Model Files:
- `enhanced_bigru_best.h5` - TensorFlow BiGRU weights (~3MB)
- `lightgbm_*.txt` - LightGBM model exports (~5MB total)
- `feature_scaler.pkl` - Data preprocessing scaler (~50KB)

## Trained Models Directory:
- `../trained/LightGBM/` - Production LightGBM ensemble
- `../trained/Meta_Ensemble/` - Meta-learning ensemble system

## Performance:
- **Meta-Ensemble**: AUC 0.9646, F1 0.5664, Recall 0.6784
- **LightGBM**: AUC 0.9671, F1 0.5647, Recall 0.4904

**Note:** Large model files excluded from Git due to size constraints.
