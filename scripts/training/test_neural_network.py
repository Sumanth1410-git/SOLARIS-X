"""Test SPEED-OPTIMIZED Bidirectional GRU training pipeline"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.config import config
from scripts.training.utils.data_loader import SolarisDataPipeline
from scripts.training.models.enhanced_bigru_trainer import EnhancedBiGRUTrainer
import time

def main():
    print("🧠 Testing ENHANCED Bidirectional GRU Training Pipeline...")
    print("🛡️ Focus: Maximum Generalization (Learning vs. Memorizing)")
    print("=" * 80)
    
    start_time = time.time()
    
    # 1. Load and Prepare Data
    pipeline = SolarisDataPipeline(config)
    full_data = pipeline.prepare_training_data()
    
    # Initialize the trainer to access its utility functions
    trainer = EnhancedBiGRUTrainer(config)
    
    # 2. Apply Anti-Overfitting Data Strategy
    print("\n🛡️ Applying Anti-Overfitting Data Strategy...")
    # Sample data with a large, stable validation set
    X_train_sampled, y_train_sampled, dt_train_sampled = trainer.sample_data_natural(
        full_data['X_train_scaled'], full_data['y_train'], full_data['datetime_train'],
        sample_ratio=0.5, split_name="train"
    )
    X_val_sampled, y_val_sampled, dt_val_sampled = trainer.sample_data_natural(
        full_data['X_validation_scaled'], full_data['y_validation'], full_data['datetime_validation'],
        sample_ratio=0.5, split_name="validation"
    )
    
    # Create sequences with a large stride to reduce correlation
    X_train_seq, y_train_seq = trainer.create_sequences_safe(
        X_train_sampled, y_train_sampled, dt_train_sampled, sequence_length=36
    )
    X_val_seq, y_val_seq = trainer.create_sequences_safe(
        X_val_sampled, y_val_sampled, dt_val_sampled, sequence_length=36
    )
    
    # Prepare the final data dictionary for the trainer
    training_data = {
        **full_data, # Include original data for evaluation and context
        'X_train_seq': X_train_seq,
        'y_train_seq': y_train_seq,
        'X_val_seq': X_val_seq,
        'y_val_seq': y_val_seq
    }
    
    # 3. Train the Model
    # The trainer will now use the pre-sequenced data
    trainer.train_enhanced_model(training_data)
    
    training_time = time.time() - start_time
    print(f"\n⏱️ TRAINING COMPLETED IN: {training_time/60:.1f} minutes")
    
    # 4. Analyze and Save Results
    trainer.plot_training_history()
    
    # --- Safe test split evaluation ---
    from scripts.training.utils.data_loader import safe_sequence_split
    print("\n🛡️ Safe test split evaluation...")
    import pandas as pd
    X_val_test = pd.concat([full_data['X_validation_scaled'], full_data['X_test_scaled']])
    y_val_test = pd.concat([full_data['y_validation'], full_data['y_test']])
    dt_val_test = pd.concat([full_data['datetime_validation'], full_data['datetime_test']])
    seq_len = config.MODELS['neural_network']['sequence_length']
    X_val_seq, y_val_seq, X_test_seq, y_test_seq = safe_sequence_split(
        X_val_test, y_val_test, dt_val_test, config.TEST_START_DATE, seq_len
    )
    # Evaluate on safe test set
    print("\n📊 Evaluating on SAFE TEST SET...")
    test_results = trainer.model.evaluate(X_test_seq, y_test_seq, verbose=1)
    print("Test set results:", test_results)

    trainer.save_model(full_data)

    total_time = time.time() - start_time

    print("\n🎯 ENHANCED BIDIRECTIONAL GRU PIPELINE COMPLETE!")
    print(f"✅ Total execution time: {total_time/60:.1f} minutes")
    
if __name__ == "__main__":
    main()
