from scripts.training.utils.data_loader import safe_sequence_split
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
        
    def create_sequences(self, X: pd.DataFrame, y: pd.Series, datetime_col: pd.Series, stride: int = 6):
        """Create time series sequences for GRU training"""
        print(f"🔄 Creating {self.sequence_length}-hour sequences with stride {stride}...")
        
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        sequences_X, sequences_y = [], []
        
        # Create sequences with sliding window
        for i in range(self.sequence_length, len(X), stride):
            seq = X.iloc[i-self.sequence_length:i].values
            if np.isnan(seq).any():
                continue
            sequences_X.append(seq)
            sequences_y.append(y.iloc[i])
        
        X_seq = np.array(sequences_X, dtype=np.float32)
        y_seq = np.array(sequences_y, dtype=np.int8)
        
        print(f"✅ Created {len(X_seq):,} sequences of shape {X_seq.shape}")
        print(f"📊 Sequence storm rate: {y_seq.mean()*100:.1f}%")
        
        return X_seq, y_seq
    
    def build_model(self, input_shape: tuple, class_weights: dict) -> tf.keras.Model:
        """Build Bidirectional GRU model optimized for CPU"""
        print("🧠 Building a more regularized Bidirectional GRU architecture...")
        
        l2_reg = tf.keras.regularizers.l2(0.001)
        
        model = Sequential([
            # A single, more regularized BiGRU layer is often better
            Bidirectional(
                GRU(
                    self.config.MODELS['neural_network']['lstm_units'],
                    return_sequences=False, # Only need output from the last time step
                    dropout=self.config.MODELS['neural_network']['dropout_rate'],
                    recurrent_dropout=0.2, # Stronger recurrent dropout
                    kernel_regularizer=l2_reg
                ),
                input_shape=input_shape,
                name='bigru_layer'
            ),
            
            # Batch normalization for stability
            BatchNormalization(),
            
            # Dense layers with dropout
            Dense(32, activation='relu', name='dense_1', kernel_regularizer=l2_reg),
            Dropout(self.config.MODELS['neural_network']['dropout_rate']),
            
            # Output layer
            Dense(1, activation='sigmoid', name='output')
        ])
        
        # Use label smoothing to prevent overconfidence
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01)
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.MODELS['neural_network']['learning_rate']),
            loss=loss,
            metrics=['accuracy', 
                     tf.keras.metrics.Precision(name='precision'), 
                     tf.keras.metrics.Recall(name='recall'), 
                     tf.keras.metrics.AUC(name='auc')]
        )
        
        print(f"🏗️ Model architecture created:")
        model.summary()
        
        return model
    
    def train_model(self, data: dict) -> 'NeuralNetworkTrainer':
        """Train Bidirectional GRU model with strict non-overlapping sequence split"""
        print(f"\n60 Training {self.model_name} Model...")
        print("=" * 60)

        # Use safe_sequence_split for strict temporal separation
        print("4ca Creating non-overlapping train/val sequences (safe split)...")
        split_date = '2017-01-01'  # Should match your config temporal split
        seq_len = self.sequence_length
        X_full = pd.concat([data['X_train_scaled'], data['X_validation_scaled']])
        y_full = pd.concat([data['y_train'], data['y_validation']])
        dt_full = pd.concat([data['datetime_train'], data['datetime_validation']])
        X_train_seq, y_train_seq, X_val_seq, y_val_seq = safe_sequence_split(
            X_full, y_full, dt_full, split_date, seq_len
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
        print("680 Starting BiGRU training...")
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=self.config.MODELS['neural_network']['epochs'],
            batch_size=self.config.MODELS['neural_network']['batch_size'],
            callbacks=callbacks,
            class_weight=data.get('class_weights'),
            verbose=1
        )

        print("3c5 BiGRU training completed!")
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
