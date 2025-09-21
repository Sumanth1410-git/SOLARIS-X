"""Test LightGBM training pipeline"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.config import config
from scripts.training.utils.data_loader import SolarisDataPipeline
from scripts.training.models.lightgbm_trainer import LightGBMTrainer

def main():
    print("🧪 Testing LightGBM Training Pipeline...")
    
    # Load data
    pipeline = SolarisDataPipeline(config)
    data = pipeline.prepare_training_data()
    
    # Initialize and train LightGBM
    trainer = LightGBMTrainer(config)
    trainer.train_model(data)
    
    # Evaluate on validation set
    trainer.evaluate_model(data, 'validation')
    
    # Plot feature importance
    trainer.plot_feature_importance(top_n=15)
    
    # Save model
    trainer.save_model(data)
    
    print("\n🎯 LightGBM TEST COMPLETE!")
    
if __name__ == "__main__":
    main()
