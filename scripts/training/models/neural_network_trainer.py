"""
SOLARIS-X Bidirectional GRU Trainer
Advanced Neural Network for Space Weather Time Series Prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Bidirectional, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.utils.base_trainer import BaseModelTrainer

class NeuralNetworkTrainer(BaseModelTrainer):
    """Bidirectional GRU trainer for temporal sequence modeling"""
    
    def __init__(self, config):
        super().__init__(config, "BiGRU_Neural_Network")
        self.sequence_length = config.MODELS['neural_network']['sequence_length']
        self.history = None
        
    def create_sequences(self, X: pd.DataFrame, y: pd.Series, datetime_col: pd.Series):
        """Create time series sequences for GRU training"""
        print(f"🔄 Creating {self.sequence_length}-hour sequences...")
        
        # Sort by datetime
        sort_idx = datetime_col.argsort()
        X_sorted = X.iloc[sort_idx].reset_index(drop=True)
        y_sorted = y.iloc[sort_idx].reset_index(drop=True)
        datetime_sorted = datetime_col.iloc[sort_idx].reset_index(drop=True)
        
        sequences_X, sequences_y = [], []
        
        # Create sequences with sliding window
        for i in range(self.sequence_length, len(X_sorted)):
            # Check if sequence is continuous (no large time gaps)
            time_diff = (datetime_sorted.iloc[i] - datetime_sorted.iloc[i-self.sequence_length]).total_seconds() / 3600
            
            if time_diff <= self.sequence_length * 1.5:  # Allow some missing hours
                sequences_X.append(X_sorted.iloc[i-self.sequence_length:i].values)
                sequences_y.append(y_sorted.iloc[i])
        
        X_seq = np.array(sequences_X, dtype=np.float32)
        y_seq = np.array(sequences_y, dtype=np.int8)
        
        print(f"✅ Created {len(X_seq):,} sequences of shape {X_seq.shape}")
        print(f"📊 Sequence storm rate: {y_seq.mean()*100:.1f}%")
        
        return X_seq, y_seq
    
    def build_model(self, input_shape: tuple, class_weights: dict) -> tf.keras.Model:
        """Build Bidirectional GRU model optimized for CPU"""
        print("🧠 Building Bidirectional GRU architecture...")
        
        model = Sequential([
            # First Bidirectional GRU layer
            Bidirectional(
                GRU(
                    self.config.MODELS['neural_network']['lstm_units'],
                    return_sequences=True,
                    dropout=0.1,
                    recurrent_dropout=0.1
                ),
                input_shape=input_shape,
                name='bigru_layer_1'
            ),
            
            # Batch normalization for stability
            BatchNormalization(),
            
            # Second Bidirectional GRU layer
            Bidirectional(
                GRU(
                    self.config.MODELS['neural_network']['lstm_units'] // 2,
                    return_sequences=False,
                    dropout=0.1,
                    recurrent_dropout=0.1
                ),
                name='bigru_layer_2'
            ),
            
            # Batch normalization
            BatchNormalization(),
            
            # Dense layers with dropout
            Dense(64, activation='relu', name='dense_1'),
            Dropout(self.config.MODELS['neural_network']['dropout_rate']),
            
            Dense(32, activation='relu', name='dense_2'),
            Dropout(self.config.MODELS['neural_network']['dropout_rate']),
            
            # Output layer
            Dense(1, activation='sigmoid', name='output')
        ])
        
        # Calculate class weight ratio for loss function
        pos_weight = class_weights[1] / class_weights[0] if class_weights else 1.0
        
        # Compile with focal loss for imbalanced data
        model.compile(
            optimizer=Adam(learning_rate=self.config.MODELS['neural_network']['learning_rate']),
            loss='binary_crossentropy',  # Could upgrade to focal loss
            metrics=['accuracy', 'precision', 'recall', tf.keras.metrics.AUC(name='auc')]
        )
        
        print(f"🏗️ Model architecture created:")
        model.summary()
        
        return model
    
    def train_model(self, data: dict) -> 'NeuralNetworkTrainer':
        """Train Bidirectional GRU model"""
        print(f"\n🧠 Training {self.model_name} Model...")
        print("=" * 60)
        
        # Prepare sequence data
        print("📊 Preparing training sequences...")
        X_train_seq, y_train_seq = self.create_sequences(
            data['X_train_scaled'], 
            data['y_train'], 
            data['datetime_train']
        )
        
        print("📊 Preparing validation sequences...")
        X_val_seq, y_val_seq = self.create_sequences(
            data['X_validation_scaled'], 
            data['y_validation'], 
            data['datetime_validation']
        )
        
        # Build model
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        self.model = self.build_model(input_shape, data['class_weights'])
        
        # Callbacks for training
        callbacks = [
            EarlyStopping(
                monitor='val_auc',
                patience=self.config.MODELS['neural_network']['patience'],
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(self.config.MODEL_PATH / "checkpoints" / "bigru_best.h5"),
                monitor='val_auc',
                save_best_only=True,
                verbose=1,
                mode='max'
            )
        ]
        
        # Train model
        print("🚀 Starting BiGRU training...")
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=self.config.MODELS['neural_network']['epochs'],
            batch_size=self.config.MODELS['neural_network']['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ BiGRU training completed!")
        return self
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("❌ No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot loss
        axes[0,0].plot(self.history.history['loss'], label='Train Loss')
        axes[0,0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0,0].set_title('Model Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Plot AUC
        axes[0,1].plot(self.history.history['auc'], label='Train AUC')
        axes[0,1].plot(self.history.history['val_auc'], label='Val AUC')
        axes[0,1].set_title('Model AUC')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('AUC')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Plot accuracy
        axes[1,0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[1,0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[1,0].set_title('Model Accuracy')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Plot precision
        axes[1,1].plot(self.history.history['precision'], label='Train Precision')
        axes[1,1].plot(self.history.history['val_precision'], label='Val Precision')
        axes[1,1].set_title('Model Precision')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Precision')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plot_path = self.config.RESULTS_PATH / "plots" / f"{self.model_name}_training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training history plot saved to: {plot_path}")
    
    def evaluate_sequences(self, data: dict, split_name: str = 'validation') -> dict:
        """Evaluate model on sequence data"""
        print(f"\n📊 Evaluating {self.model_name} on {split_name} sequences...")
        
        # Create sequences for evaluation
        X_seq, y_seq = self.create_sequences(
            data[f'X_{split_name}_scaled'], 
            data[f'y_{split_name}'], 
            data[f'datetime_{split_name}']
        )
        
        # Make predictions
        y_pred_proba = self.model.predict(X_seq, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_seq, y_pred, y_pred_proba)
        
        # Store results
        self.metrics[split_name] = metrics
        
        print(f"📊 {self.model_name} - {split_name.upper()} SEQUENCE RESULTS:")
        print("-" * 55)
        for metric, value in metrics.items():
            print(f"  {metric:>18}: {value:.4f}")
        
        return metrics
