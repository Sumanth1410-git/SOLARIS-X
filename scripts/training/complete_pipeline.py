"""
SOLARIS-X Complete Training Pipeline - PRODUCTION VERSION
Trains and evaluates LightGBM model on all splits
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.config import config
from scripts.training.utils.data_loader import SolarisDataPipeline
from scripts.training.models.lightgbm_trainer import LightGBMTrainer
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, 
                              roc_auc_score, average_precision_score,
                              roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate():
    """Complete pipeline: train, validate, test"""
    
    print("🚀 SOLARIS-X Complete Pipeline - Production Version")
    print("=" * 70)
    
    # Load data using existing pipeline
    print("\n📊 Loading data...")
    pipeline = SolarisDataPipeline(config)
    data = pipeline.prepare_training_data()
    
    # Initialize and train
    print("\n🌲 Training LightGBM...")
    trainer = LightGBMTrainer(config)
    trainer.train_model(data)
    
    # Plot feature importance
    print("\n📊 Analyzing feature importance...")
    trainer.plot_feature_importance(top_n=20)
    
    # Save model
    print("\n💾 Saving production model...")
    trainer.save_model(data)
    
    # VALIDATION EVALUATION
    print("\n" + "=" * 70)
    print("📊 VALIDATION SET EVALUATION")
    print("=" * 70)
    
    X_val = data['X_validation_scaled']
    y_val = data['y_validation']
    
    y_val_pred_proba = trainer.model.predict(X_val)
    y_val_pred = (y_val_pred_proba > 0.5).astype(int)
    
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    print(f"   Validation AUC: {val_auc:.4f}")
    
    # TEST SET EVALUATION
    print("\n" + "=" * 70)
    print("📊 TEST SET EVALUATION (UNSEEN DATA)")
    print("=" * 70)
    
    X_test = data['X_test_scaled']
    y_test = data['y_test']
    
    print(f"\n📈 Test Set Info:")
    print(f"   Samples: {len(X_test):,}")
    print(f"   Storm rate: {y_test.mean()*100:.2f}%")
    print(f"   Storms: {y_test.sum():,}")
    
    # Predict
    print("\n🔮 Generating predictions...")
    y_pred_proba = trainer.model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    test_auc = roc_auc_score(y_test, y_pred_proba)
    test_ap = average_precision_score(y_test, y_pred_proba)
    
    print(f"\n🎯 TEST SET METRICS:")
    print(f"   Test AUC: {test_auc:.4f}")
    print(f"   Test AP:  {test_ap:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['No Storm', 'Storm'],
                                digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔍 Confusion Matrix:")
    print(f"   True Negatives:  {cm[0,0]:>6,}")
    print(f"   False Positives: {cm[0,1]:>6,}")
    print(f"   False Negatives: {cm[1,0]:>6,}")
    print(f"   True Positives:  {cm[1,1]:>6,}")
    
    # False alarm rate
    far = cm[0,1] / (cm[0,0] + cm[0,1])
    pod = cm[1,1] / (cm[1,0] + cm[1,1])
    print(f"\n📊 Operational Metrics:")
    print(f"   False Alarm Rate: {far*100:.2f}%")
    print(f"   Probability of Detection: {pod*100:.2f}%")
    
    # Plot test ROC curve
    print(f"\n📈 Generating test ROC curve...")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'Test AUC = {test_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('Test Set ROC Curve - Production LightGBM Model', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    roc_path = config.RESULTS_PATH / "plots" / "test_roc_curve.png"
    roc_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Test ROC saved: {roc_path}")
    
    # Plot confusion matrix
    print(f"\n📊 Generating confusion matrix plot...")
    plt.figure(figsize=(8, 6))
    
    # Use seaborn for better visualization (fallback to manual if not available)
    try:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Storm', 'Storm'],
                   yticklabels=['No Storm', 'Storm'],
                   cbar_kws={'label': 'Count'},
                   annot_kws={'size': 14, 'weight': 'bold'})
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    except:
        # Manual plotting if seaborn not available
        im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                plt.text(j, i, f'{cm[i, j]:,}',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14, fontweight='bold')
        
        plt.xticks([0, 1], ['No Storm', 'Storm'])
        plt.yticks([0, 1], ['No Storm', 'Storm'])
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.title('Test Set Confusion Matrix - Production LightGBM Model', 
             fontsize=14, fontweight='bold', pad=20)
    
    # Add metrics text box
    metrics_text = f'Accuracy: {(cm[0,0]+cm[1,1])/cm.sum()*100:.2f}%\n'
    metrics_text += f'Precision: {cm[1,1]/(cm[1,1]+cm[0,1])*100:.2f}%\n'
    metrics_text += f'Recall: {cm[1,1]/(cm[1,1]+cm[1,0])*100:.2f}%\n'
    metrics_text += f'FAR: {far*100:.2f}%'
    
    plt.text(1.15, 0.5, metrics_text, 
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    cm_path = config.RESULTS_PATH / "plots" / "test_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved: {cm_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 PRODUCTION MODEL SUMMARY")
    print("=" * 70)
    print(f"\n✅ Training complete!")
    print(f"📊 Validation AUC: {val_auc:.4f}")
    print(f"📊 Test AUC:       {test_auc:.4f}")
    
    auc_drop = (val_auc - test_auc) * 100
    print(f"📉 AUC Drop:       {auc_drop:.2f}%")
    
    if abs(val_auc - test_auc) < 0.02:
        print(f"✅ EXCELLENT: Val and test AUC within 2% - model generalizes well!")
    elif abs(val_auc - test_auc) < 0.05:
        print(f"✅ GOOD: Val and test AUC within 5% - acceptable generalization")
    else:
        print(f"⚠️  WARNING: Large val-test gap - check for overfitting")
    
    print(f"\n🚀 Production model ready for deployment!")
    print(f"\n📁 Artifacts saved:")
    print(f"   • ROC curve: {roc_path}")
    print(f"   • Confusion matrix: {cm_path}")
    
    return trainer, test_auc, test_ap

if __name__ == "__main__":
    trainer, test_auc, test_ap = train_and_evaluate()
