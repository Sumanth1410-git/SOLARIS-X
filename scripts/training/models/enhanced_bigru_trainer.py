"""
SOLARIS-X Enhanced BiGRU Trainer
Optimized for Rare Event Detection with Focal Loss & Recall Enhancement
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, GRU, Dense, Dropout, BatchNormalization, Attention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.models.neural_network_trainer import NeuralNetworkTrainer

class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss for addressing class imbalance in rare storm events"""
    
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Clip predictions to prevent log(0)
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_loss = -alpha_t * tf.pow((1 - p_t), self.gamma) * tf.math.log(p_t)
        
        return tf.reduce_mean(focal_loss)

class RecallOptimizedMetrics(tf.keras.metrics.Metric):
    """Custom metric optimized for storm recall"""
    
    def __init__(self, name='f1_recall_weighted', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.predicted_positives = self.add_weight(name='pp', initializer='zeros')
        self.actual_positives = self.add_weight(name='ap', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        
        tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
        pp = tf.reduce_sum(tf.cast(y_pred, tf.float32))
        ap = tf.reduce_sum(tf.cast(y_true, tf.float32))
        
        self.true_positives.assign_add(tp)
        self.predicted_positives.assign_add(pp)
        self.actual_positives.assign_add(ap)
    
    def result(self):
        precision = self.true_positives / (self.predicted_positives + K.epsilon())
        recall = self.true_positives / (self.actual_positives + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        
        # Weighted F1 favoring recall for storm detection
        recall_weighted_f1 = 0.3 * precision + 0.7 * recall
        return recall_weighted_f1
    
    def reset_state(self):
        self.true_positives.assign(0)
        self.predicted_positives.assign(0)
        self.actual_positives.assign(0)

class EnhancedBiGRUTrainer(NeuralNetworkTrainer):
    """Enhanced BiGRU with focal loss and recall optimization"""
    
    def __init__(self, config):
        super().__init__(config)
        self.model_name = "Enhanced_BiGRU"
    
    def build_enhanced_model(self, input_shape: tuple, class_weights: dict) -> tf.keras.Model:
        """Build enhanced BiGRU with MEMORY OPTIMIZATION"""
        print("🧠 Building Memory-Optimized Enhanced BiGRU...")
    
        model = Sequential([
            # REDUCED: Smaller GRU units for memory efficiency
            Bidirectional(
                GRU(
                    24,  # Reduced from 48
                    return_sequences=True,
                    dropout=0.15,
                    recurrent_dropout=0.15
                ),
                input_shape=input_shape,
                name='enhanced_bigru_1'
            ),
        
            BatchNormalization(),

            Bidirectional(
                GRU(
                    16,  # Reduced from 32
                    return_sequences=False,
                    dropout=0.15,
                    recurrent_dropout=0.15
                ),
                name='enhanced_bigru_2'
            ),
        
            BatchNormalization(),

            # REDUCED: Smaller dense layers
            Dense(48, activation='relu'),  # Reduced from 96
            Dropout(0.3),
        
            Dense(24, activation='relu'),  # Reduced from 48
            Dropout(0.2),
        
            Dense(1, activation='sigmoid', name='storm_prediction')
        ])
    
        # Compile with focal loss
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=FocalLoss(alpha=0.75, gamma=2.0),
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
    
        print("🏗️ Memory-Optimized Enhanced BiGRU:")
        model.summary()

        return model

    def train_enhanced_model(self, data: dict) -> 'EnhancedBiGRUTrainer':
        """Train enhanced BiGRU model with recall optimization"""
        print(f"\n🚀 Training {self.model_name} with Focal Loss...")
        print("=" * 70)
    
        # Prepare sequence data
        X_train_seq, y_train_seq = self.create_sequences(
            data['X_train_scaled'], 
            data['y_train'], 
            data['datetime_train']
        )
    
        X_val_seq, y_val_seq = self.create_sequences(
            data['X_validation_scaled'], 
            data['y_validation'], 
            data['datetime_validation']
        )
    
        # Build enhanced model
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        self.model = self.build_enhanced_model(input_shape, data['class_weights'])

        # Enhanced callbacks for recall optimization
        callbacks = [
            EarlyStopping(
                monitor='val_recall',  # Monitor recall instead of AUC
                patience=8,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.6,
                patience=4,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(self.config.MODEL_PATH / "checkpoints" / "enhanced_bigru_best.h5"),
                monitor='val_recall_weighted_f1',
                save_best_only=True,
                verbose=1,
                mode='max'
            )
        ]
    
        # FIXED: Proper class weights handling
        model_class_weights = data.get('class_weights', {0: 1.0, 1: 1.0})
        sample_weights = np.where(
            y_train_seq == 1, 
            model_class_weights[1], 
            model_class_weights[0]
        )
    
        print("🔥 Starting Enhanced BiGRU training with Focal Loss...")
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=40,  # More epochs for convergence
            batch_size=48,  # Smaller batch for better gradient updates
            callbacks=callbacks,
            sample_weight=sample_weights,
            verbose=1
        )
    
        print("✅ Enhanced BiGRU training completed!")
        return self
