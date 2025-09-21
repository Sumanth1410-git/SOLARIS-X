"""Test script for SOLARIS-X data pipeline"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import modules
from scripts.training.config import config
from scripts.training.utils.data_loader import SolarisDataPipeline

def main():
    """Test data pipeline functionality"""
    print("🧪 Testing SOLARIS-X Data Pipeline...")
    
    # Initialize pipeline
    pipeline = SolarisDataPipeline(config)
    
    # Run complete data preparation
    data = pipeline.prepare_training_data()
    
    # Validation summary
    print(f"\n🎯 PIPELINE TEST RESULTS:")
    print(f"✅ Training samples: {len(data['X_train']):,}")
    print(f"✅ Validation samples: {len(data['X_validation']):,}")
    print(f"✅ Test samples: {len(data['X_test']):,}")
    print(f"✅ Features: {len(data['feature_columns'])}")
    print(f"✅ Storm rate (train): {data['y_train'].mean()*100:.1f}%")
    
if __name__ == "__main__":
    main()
