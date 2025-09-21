"""Test SOLARIS-X with NO data leakage"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.config import config
from scripts.training.utils.data_loader import SolarisDataPipeline
from scripts.training.models.lightgbm_trainer import LightGBMTrainer

def main():
    print("🧪 Testing NO-LEAKAGE LightGBM Training...")
    
    # Load data with leakage prevention
    pipeline = SolarisDataPipeline(config)
    data = pipeline.prepare_training_data()
    
    # Initialize and train LightGBM with anti-overfitting settings
    trainer = LightGBMTrainer(config)
    trainer.train_model(data)
    
    # Evaluate on validation AND test sets
    val_metrics = trainer.evaluate_model(data, 'validation')
    test_metrics = trainer.evaluate_model(data, 'test')
    
    # Plot feature importance
    trainer.plot_feature_importance(top_n=15)
    
    # Compare metrics
    print(f"\n🎯 ANTI-OVERFITTING RESULTS SUMMARY:")
    print("=" * 60)
    print(f"{'Metric':<20} {'Validation':<12} {'Test':<12} {'Delta':<12}")
    print("-" * 60)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        val_score = val_metrics.get(metric, 0)
        test_score = test_metrics.get(metric, 0)
        delta = abs(val_score - test_score)
        print(f"{metric:<20} {val_score:<12.4f} {test_score:<12.4f} {delta:<12.4f}")
    
    # Save model
    trainer.save_model(data)
    
    print("\n✅ NO-LEAKAGE MODEL TEST COMPLETE!")
    
if __name__ == "__main__":
    main()
