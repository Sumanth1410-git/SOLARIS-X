"""
SOLARIS-X Enhanced BiGRU Trainer - ANTI-OVERFITTING VERSION
Designed to prevent overfitting through proper regularization and validation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from scripts.training.utils.data_loader import safe_sequence_split


class OverfitMonitor(Callback):
    """Monitor for train-val gap and stop if overfitting detected"""
    
    def __init__(self, gap_threshold=0.025, patience=3, start_epoch=5):
        super().__init__()
        self.gap_threshold = gap_threshold
        self.patience = patience
        self.wait = 0
        self.start_epoch = start_epoch
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        
        train_auc = logs.get('auc', 0)
        val_auc = logs.get('val_auc', 0)
        # gap = train - val (positive when train better than val)
        gap = train_auc - val_auc

        # Don't monitor for the first few epochs to allow model to stabilize
        if epoch < self.start_epoch:
            return

        # Only consider meaningful AUC values
        if train_auc is None or val_auc is None:
            return

        if gap > self.gap_threshold:
            self.wait += 1
            print(f"\n⚠️ Train-Val AUC gap: {gap:.4f} (threshold: {self.gap_threshold}) - {self.wait}/{self.patience}")
        else:
            # decay counter when gap reduces
            self.wait = 0 # Reset on good epoch

        if self.wait >= self.patience:
            print(f"\n🛑 STOPPING: Train-Val AUC gap exceeded threshold for {self.patience} consecutive epochs.")
            self.model.stop_training = True


class DataAugmentation(Callback):
    """Apply Gaussian noise to training data at the start of each epoch"""
    def __init__(self, x_train, noise_level=0.01):
        super().__init__()
        self.x_train_original = np.copy(x_train)
        self.x_train_noisy = x_train  # This is a reference to the array in fit()
        self.noise_level = noise_level

    def on_epoch_begin(self, epoch, logs=None):
        noise = np.random.normal(loc=0.0, scale=self.noise_level, size=self.x_train_original.shape)
        # Add noise to a fresh copy of the original data
        noisy_data = self.x_train_original + noise
        np.copyto(self.x_train_noisy, noisy_data) # Update the training data in-place
        print(f"\n✨ Epoch {epoch + 1}: Applied Gaussian noise (level={self.noise_level}) to training data.")


class EnhancedBiGRUTrainer:
    """Enhanced BiGRU with STRICT ANTI-OVERFITTING measures"""
    
    def __init__(self, config):
        self.config = config
        self.model_name = "BiGRU_Anti_Overfit"
        self.model = None
        self.history = None
    
    def sample_data_natural(self, X: pd.DataFrame, y: pd.Series, datetime_col: pd.Series, 
                           sample_ratio: float, split_name: str) -> tuple:
        """Sample data with NATURAL distribution - NO artificial balancing"""
        print(f"📊 {split_name.upper()}: Sampling {sample_ratio*100:.0f}% (natural distribution)")
        
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        datetime_col = datetime_col.reset_index(drop=True)
        
        # Pure random sampling - maintains natural class distribution
        n_samples = int(len(X) * sample_ratio)
        indices = np.random.choice(len(X), n_samples, replace=False)
        indices = np.sort(indices)
        
        X_sampled = X.iloc[indices].copy()
        y_sampled = y.iloc[indices].copy()
        datetime_sampled = datetime_col.iloc[indices].copy()
        
        storm_rate = y_sampled.mean() * 100
        print(f"✅ {split_name}: {len(X_sampled):,} rows | Natural storm rate: {storm_rate:.2f}%")
        
        return X_sampled, y_sampled, datetime_sampled
    
    def create_sequences_safe(self, X: pd.DataFrame, y: pd.Series, datetime_col: pd.Series, 
                              sequence_length: int = 36) -> tuple:
        """Create sequences with LARGE STRIDE to prevent near-duplicates"""
        print(f"🔄 Creating {sequence_length}-hour sequences (ANTI-OVERFIT MODE)...")
        
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        X_array = X.values.astype(np.float32)
        y_array = y.values.astype(np.int32)
        
        sequences = []
        targets = []

        # stride controls overlap; moderate stride to keep enough training samples
        stride = 3  # reduce from 6 to 3 to increase training data while keeping separation

        for i in range(sequence_length, len(X_array), stride):
            # ensure no NaNs in sequence
            seq = X_array[i-sequence_length:i]
            if np.isnan(seq).any():
                continue
            sequences.append(seq)
            targets.append(y_array[i])
        
        if len(sequences) == 0:
            print("⚠️ No sequences created - data too small")
            for i in range(sequence_length, min(len(X_array), sequence_length + 50)):
                sequences.append(X_array[i-sequence_length:i])
                targets.append(y_array[i])
        
        X_seq = np.array(sequences, dtype=np.float32)
        y_seq = np.array(targets, dtype=np.int32)
        
        storm_rate = y_seq.mean() * 100
        print(f"✅ Created {len(X_seq):,} sequences | Natural storm rate: {storm_rate:.2f}%")
        print(f"   Stride: {stride} hours (reduces temporal correlation)")
        
        return X_seq, y_seq
    
    def build_anti_overfit_model(self, input_shape: tuple) -> tf.keras.Model:
        """Build SMALLER model with STRONG regularization to prevent overfitting"""
        print("🧠 Building ANTI-OVERFITTING BiGRU...")
        print(f"   Input shape: {input_shape}")
        
        # Balanced architecture with reasonable regularization to allow learning
        model = Sequential([
            # Single BiGRU with reduced capacity
            Bidirectional(
                GRU(
                    64,  # Further reduced capacity for stability
                    return_sequences=False,
                    dropout=0.35,  # More aggressive dropout
                    recurrent_dropout=0.25,  # More aggressive recurrent dropout
                    kernel_regularizer=tf.keras.regularizers.l2(0.003)  # Even stronger L2
                ),
                input_shape=input_shape,
                name='anti_overfit_bigru'
            ),
            BatchNormalization(momentum=0.9),  # Stabilize training with adjusted momentum
            
            # Smaller dense layers with strong dropout
            Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.003)),
            Dropout(0.5), # Increased dropout to 50%
            
            Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.003)),
            Dropout(0.4), # Increased dropout
            
            # Output
            Dense(1, activation='sigmoid', name='storm_prediction')
        ])
        
        # Small label smoothing to avoid overconfidence
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.025)

        # Lower initial learning rate to stabilize training
        optimizer = Adam(learning_rate=0.0007)

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        print("🏗️ ANTI-OVERFITTING BiGRU Architecture:")
        model.summary()
        
        total_params = model.count_params()
        print(f"📊 Model size: {total_params:,} parameters")
        print(f"🛡️ Regularization: L2=0.003, Dropout=0.35-0.5, Label Smoothing=0.025")
        
        return model

    def train_enhanced_model(self, data: dict) -> 'EnhancedBiGRUTrainer':
        """Train with ANTI-OVERFITTING strategy using strict non-overlapping sequence split"""
        print(f"\n6e1 Training {self.model_name} - ANTI-OVERFITTING MODE...")
        print("=" * 80)

        # Use safe_sequence_split for strict temporal separation
        print("4ca Creating non-overlapping train/val sequences (safe split)...")
        split_date = '2017-01-01'  # Should match your config temporal split
        seq_len = 36
        X_full = pd.concat([data['X_train_scaled'], data['X_validation_scaled']])
        y_full = pd.concat([data['y_train'], data['y_validation']])
        dt_full = pd.concat([data['datetime_train'], data['datetime_validation']])
        X_train_seq, y_train_seq, X_val_seq, y_val_seq = safe_sequence_split(
            X_full, y_full, dt_full, split_date, seq_len
        )

        # Shuffle training data to avoid order effects
        shuffle_idx = np.random.permutation(len(X_train_seq))
        X_train_seq = X_train_seq[shuffle_idx]
        y_train_seq = y_train_seq[shuffle_idx]
        
        # Validate dataset sizes
        print(f"\n📊 ANTI-OVERFITTING Training Configuration:")
        print(f"   Training sequences: {len(X_train_seq):,}")
        print(f"   Validation sequences: {len(X_val_seq):,}")
        print(f"   Val/Train ratio: {len(X_val_seq)/len(X_train_seq):.3f}")
        
        if len(X_val_seq) < 10000:
            print(f"⚠️ WARNING: Validation set ({len(X_val_seq)}) < 10,000 - risk of memorization")
        
        # Build anti-overfitting model
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        self.model = self.build_anti_overfit_model(input_shape)
        
        # Calculate parameters per validation sample
        params_per_val = self.model.count_params() / len(X_val_seq)
        print(f"   Parameters per validation sample: {params_per_val:.2f}")
        if params_per_val > 10:
            print(f"⚠️ WARNING: High param/val ratio - risk of overfitting")
        
        # STRICT anti-overfitting callbacks
        callbacks = [
            # Data augmentation callback
            DataAugmentation(X_train_seq, noise_level=0.01),
            # Early stopping on validation AUC to encourage learning until val metric plateaus
            EarlyStopping(
                monitor='val_auc',
                patience=10, # Slightly more patience to find the true peak
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            # Learning rate scheduling
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3, # More aggressive reduction
                patience=2, # Reduce LR more quickly if val_loss stagnates
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(self.config.MODEL_PATH / "checkpoints" / "enhanced_bigru_best.h5"),
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            # Monitor train-val gap (fixed logic)
            OverfitMonitor(gap_threshold=0.025, patience=3, start_epoch=5)
        ]
        
        # Cap extreme class weights
        model_class_weights = data.get('class_weights', {0: 1.0, 1: 1.0})
        if model_class_weights[1] > 10:
            model_class_weights[1] = 10.0
            print(f"   Capped storm class weight to: {model_class_weights[1]}")
        
        print(f"\n🛡️ ANTI-OVERFITTING Training Settings:")
        print(f"   Learning rate: 0.0007 (initial)")
        print(f"   Batch size: 128")
        print(f"   Early stopping: val_auc, patience=10")
        print(f"   LR reduction: factor=0.3, patience=2")
        print(f"   Gap monitor: stops if train-val gap > 0.025 for 3 epochs (starts at epoch 5)")
        print(f"   Data augmentation: Gaussian noise (level=0.01) added each epoch")
        print(f"   Natural distribution: maintained throughout")
        
        # Train with anti-overfitting measures
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=60,
            batch_size=128,
            callbacks=callbacks,
            class_weight=model_class_weights,
            shuffle=True,
            verbose=1
        )
        
        print("✅ ANTI-OVERFITTING training completed!")
        
        # Analyze results
        if len(self.history.history['val_auc']) > 0:
            best_epoch = np.argmin(self.history.history['val_loss'])
            best_val_loss = min(self.history.history['val_loss'])
            best_train_loss = self.history.history['loss'][best_epoch]
            best_val_auc = self.history.history['val_auc'][best_epoch]
            best_train_auc = self.history.history['auc'][best_epoch]
            
            print(f"\n📊 ANTI-OVERFITTING RESULTS:")
            print(f"   Best epoch: {best_epoch + 1}")
            print(f"   Train loss: {best_train_loss:.4f} | Val loss: {best_val_loss:.4f}")
            print(f"   Train AUC: {best_train_auc:.4f} | Val AUC: {best_val_auc:.4f}")
            print(f"   AUC gap: {abs(best_train_auc - best_val_auc):.4f}")
            
            # Health check
            if best_val_loss > best_train_loss * 1.2:
                print("⚠️ Validation loss significantly higher - some overfitting may remain")
            elif abs(best_train_auc - best_val_auc) < 0.03:
                print("✅ HEALTHY: Small AUC gap indicates good generalization")
            
        return self
    
    def plot_training_history(self):
        """Plot training history with overfitting analysis"""
        if self.history is None:
            print("⚠️ No training history available")
            return
            
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss - CRITICAL for overfitting detection
            axes[0, 0].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
            axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
            axes[0, 0].set_title('Loss (Anti-Overfit)', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Check for overfitting in loss
            final_train = self.history.history['loss'][-1]
            final_val = self.history.history['val_loss'][-1]
            if final_val < final_train:
                axes[0, 0].text(0.5, 0.95, '✅ Healthy', transform=axes[0, 0].transAxes,
                              ha='center', va='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
            
            # AUC
            axes[0, 1].plot(self.history.history['auc'], label='Train AUC', linewidth=2)
            axes[0, 1].plot(self.history.history['val_auc'], label='Val AUC', linewidth=2)
            axes[0, 1].set_title('AUC (Anti-Overfit)', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('AUC')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Precision
            axes[1, 0].plot(self.history.history['precision'], label='Train Precision', linewidth=2)
            axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision', linewidth=2)
            axes[1, 0].set_title('Precision (Anti-Overfit)', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Recall
            axes[1, 1].plot(self.history.history['recall'], label='Train Recall', linewidth=2)
            axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall', linewidth=2)
            axes[1, 1].set_title('Recall (Anti-Overfit)', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.config.RESULTS_PATH / "plots" / f"{self.model_name}_training_history.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Training history plot saved: {plot_path}")
        except Exception as e:
            print(f"⚠️ Could not plot: {e}")
    
    def evaluate_sequences(self, data: dict, split_name: str):
        """Evaluate on sequences"""
        if self.model is None:
            print("⚠️ Model not trained")
            return
            
        try:
            n_eval = min(15000, len(data[f'X_{split_name}_scaled']))
            eval_indices = np.random.choice(len(data[f'X_{split_name}_scaled']), n_eval, replace=False)
            
            X_eval = data[f'X_{split_name}_scaled'].iloc[eval_indices].reset_index(drop=True)
            y_eval = data[f'y_{split_name}'].iloc[eval_indices].reset_index(drop=True)
            dt_eval = data[f'datetime_{split_name}'].iloc[eval_indices].reset_index(drop=True)
            
            X_seq, y_seq = self.create_sequences_safe(X_eval, y_eval, dt_eval, sequence_length=36)
            
            print(f"\n📊 Evaluating on {split_name.upper()}...")
            results = self.model.evaluate(X_seq, y_seq, verbose=0)
            
            print(f"📊 {split_name.upper()} EVALUATION:")
            for name, value in zip(self.model.metrics_names, results):
                print(f"   {name}: {value:.4f}")
                
        except Exception as e:
            print(f"⚠️ Evaluation error: {e}")
    
    def save_model(self, data: dict):
        """Save model"""
        if self.model is None:
            print("⚠️ No model to save")
            return
            
        try:
            # This is the final model saved after training completes
            model_path = self.config.MODEL_PATH / "trained" / self.model_name / "model.h5"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(model_path)
            print(f"✅ Final model saved: {model_path}")
        except Exception as e:
            print(f"⚠️ Save error: {e}")
