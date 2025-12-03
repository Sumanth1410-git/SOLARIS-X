import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import optuna # type: ignore
from scripts.training.config import config
from scripts.training.models.enhanced_bigru_trainer import EnhancedBiGRUTrainer
from scripts.training.utils.data_loader import SolarisDataPipeline

def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 3e-3)
    dropout_rate = trial.suggest_categorical('dropout_rate', [0.2, 0.3, 0.4, 0.5])
    gru_units = trial.suggest_categorical('gru_units', [32, 64, 128])
    l2_reg = trial.suggest_loguniform('l2_regularization', 5e-4, 3e-3)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    sequence_length = trial.suggest_categorical('sequence_length', [24, 36, 48])

    # Update config for this trial
    config.MODELS['neural_network'].update({
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
        'lstm_units': gru_units,
        'l2_regularization': l2_reg,
        'batch_size': batch_size,
        'sequence_length': sequence_length,
        'epochs': 20,
        'patience': 4
    })

    # Prepare data
    pipeline = SolarisDataPipeline(config)
    data = pipeline.prepare_training_data()
    trainer = EnhancedBiGRUTrainer(config)
    trainer.train_enhanced_model(data)
    val_auc = max(trainer.history.history['val_auc'])
    return val_auc

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print('Best trial:', study.best_trial.params)

    # --- Auto-update config.py with best parameters ---
    best_params = study.best_trial.params
    import re
    config_path = os.path.join(os.path.dirname(__file__), 'config.py')
    with open(config_path, 'r', encoding='utf-8') as f:
        config_code = f.read()
    # Update neural_network section in config.MODELS
    def update_param(code, param, value):
        pattern = rf"('{param}'\s*:\s*)([^,\n]+)"
        replacement = rf"\1{repr(value)}"
        return re.sub(pattern, replacement, code)
    for k, v in best_params.items():
        config_code = update_param(config_code, k, v)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_code)
    print(f"✅ config.py updated with best Optuna parameters: {best_params}")
