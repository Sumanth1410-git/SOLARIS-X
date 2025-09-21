"""Test Bidirectional GRU training pipeline"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.config import config
from scripts.training.utils.data_loader import SolarisDataPipeline
from scripts.training.models.neural_network_trainer import NeuralNetworkTrainer

def main():
    print("🧪 Testing Bidirectional GRU Training Pipeline...")
    
    # Load data
    pipeline = SolarisDataPipeline(config)
    data = pipeline.prepare_training_data()
    
    # Initialize and train BiGRU
    trainer = NeuralNetworkTrainer(config)
    trainer.train_model(data)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate on validation and test sequences
    trainer.evaluate_sequences(data, 'validation')
    trainer.evaluate_sequences(data, 'test')
    
    # Save model
    trainer.save_model(data)
    
    print("\n🎯 BIDIRECTIONAL GRU TEST COMPLETE!")
    
if __name__ == "__main__":
    main()
