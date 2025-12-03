"""
SOLARIS-X Ensemble Pipeline - PRODUCTION VERSION
Trains 3 specialized LightGBM models and optimizes ensemble weights
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.config import config
from scripts.training.utils.data_loader import SolarisDataPipeline
from scripts.training.models.ensemble_trainer import EnsembleLightGBM
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

def train_ensemble():
    """Complete ensemble pipeline: train, optimize, evaluate"""
    
    print("\n" + "="*70)
    print("🚀 SOLARIS-X ENSEMBLE PIPELINE - Production Version")
    print("="*70)
    print("\nTraining 3 specialized LightGBM models:")
    print("  1. Precision-focused (Conservative - fewer false alarms)")
    print("  2. Recall-focused (Aggressive - catches more storms)")
    print("  3. Balanced (Baseline performance)")
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("\n" + "="*70)
    print("📊 STEP 1: Loading Data")
    print("="*70)
    
    pipeline = SolarisDataPipeline(config)
    data = pipeline.prepare_training_data()
    
    print(f"✅ Data loaded successfully")
    print(f"   Training samples:   {len(data['X_train_scaled']):,}")
    print(f"   Validation samples: {len(data['X_validation_scaled']):,}")
    print(f"   Test samples:       {len(data['X_test_scaled']):,}")
    
    # ========================================================================
    # STEP 2: Train All Three Models
    # ========================================================================
    print("\n" + "="*70)
    print("🌲 STEP 2: Training Ensemble Models")
    print("="*70)
    
    ensemble = EnsembleLightGBM(config)
    ensemble.train_all_models(data)
    
    # ========================================================================
    # STEP 3: Evaluate Individual Models on Validation Set
    # ========================================================================
    print("\n" + "="*70)
    print("📊 STEP 3: Individual Model Performance (Validation Set)")
    print("="*70)
    
    X_val = data['X_validation_scaled']
    y_val = data['y_validation']
    
    # Create comparison table
    comparison_data = []
    for history in ensemble.training_history:
        comparison_data.append({
            'Model': history['model'].capitalize(),
            'Type': history['specialization'].capitalize(),
            'Min Samples': history['min_child_samples'],
            'AUC': f"{history['auc']:.4f}",
            'Precision': f"{history['precision']:.4f}",
            'Recall': f"{history['recall']:.4f}",
            'F1-Score': f"{history['f1']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # ========================================================================
    # STEP 4: Optimize Ensemble Weights with Optuna
    # ========================================================================
    print("\n" + "="*70)
    print("🔍 STEP 4: Optimizing Ensemble Weights")
    print("="*70)
    print("Target: Maximize F1-Score (balance precision & recall)")
    
    def objective(trial):
        """Optuna objective: Maximize F1-score"""
        w_precision = trial.suggest_float('w_precision', 0.0, 1.0)
        w_recall = trial.suggest_float('w_recall', 0.0, 1.0)
        w_balanced = trial.suggest_float('w_balanced', 0.0, 1.0)
        
        ensemble.set_weights(w_precision, w_recall, w_balanced)
        metrics, _, _ = ensemble.evaluate(X_val, y_val)
        
        return metrics['f1']
    
    # Run optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', study_name='ensemble_weights')
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    
    best_weights = study.best_params
    print(f"\n✅ Optimization Complete!")
    print(f"   Best F1-Score: {study.best_value:.4f}")
    print(f"\n   Optimal Weights:")
    print(f"      Precision Model: {best_weights['w_precision']:.3f}")
    print(f"      Recall Model:    {best_weights['w_recall']:.3f}")
    print(f"      Balanced Model:  {best_weights['w_balanced']:.3f}")
    
    # Apply optimal weights
    ensemble.set_weights(
        best_weights['w_precision'],
        best_weights['w_recall'],
        best_weights['w_balanced']
    )
    
    # ========================================================================
    # STEP 5: Evaluate Ensemble on Test Set
    # ========================================================================
    print("\n" + "="*70)
    print("📊 STEP 5: ENSEMBLE EVALUATION (TEST SET - UNSEEN DATA)")
    print("="*70)
    
    X_test = data['X_test_scaled']
    y_test = data['y_test']
    
    print(f"\n📈 Test Set Info:")
    print(f"   Samples: {len(X_test):,}")
    print(f"   Storm rate: {y_test.mean()*100:.2f}%")
    print(f"   Storms: {y_test.sum():,}")
    
    # Get ensemble predictions
    print("\n🔮 Generating ensemble predictions...")
    metrics_ensemble, y_pred_ensemble, y_proba_ensemble = ensemble.evaluate(X_test, y_test)
    
    print(f"\n🎯 ENSEMBLE TEST METRICS:")
    print(f"   Test AUC:       {metrics_ensemble['auc']:.4f} ({metrics_ensemble['auc']*100:.2f}%)")
    print(f"   Test Precision: {metrics_ensemble['precision']:.4f} ({metrics_ensemble['precision']*100:.2f}%)")
    print(f"   Test Recall:    {metrics_ensemble['recall']:.4f} ({metrics_ensemble['recall']*100:.2f}%)")
    print(f"   Test F1-Score:  {metrics_ensemble['f1']:.4f} ({metrics_ensemble['f1']*100:.2f}%)")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred_ensemble, 
                                target_names=['No Storm', 'Storm'],
                                digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_ensemble)
    print(f"\n🔍 Confusion Matrix:")
    print(f"   True Negatives:  {cm[0,0]:>6,}")
    print(f"   False Positives: {cm[0,1]:>6,}")
    print(f"   False Negatives: {cm[1,0]:>6,}")
    print(f"   True Positives:  {cm[1,1]:>6,}")
    
    far = cm[0,1] / (cm[0,0] + cm[0,1])
    pod = cm[1,1] / (cm[1,0] + cm[1,1])
    print(f"\n📊 Operational Metrics:")
    print(f"   False Alarm Rate: {far*100:.2f}%")
    print(f"   Probability of Detection: {pod*100:.2f}%")
    
    # ========================================================================
    # STEP 6: Compare with Baseline (Single Model)
    # ========================================================================
    print("\n" + "="*70)
    print("📊 STEP 6: ENSEMBLE vs BASELINE COMPARISON")
    print("="*70)
    
    # Get baseline (balanced model) predictions
    y_proba_baseline = ensemble.model_balanced.predict(X_test)
    y_pred_baseline = (y_proba_baseline >= 0.5).astype(int)
    
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    
    baseline_metrics = {
        'auc': roc_auc_score(y_test, y_proba_baseline),
        'precision': precision_score(y_test, y_pred_baseline, zero_division=0),
        'recall': recall_score(y_test, y_pred_baseline, zero_division=0),
        'f1': f1_score(y_test, y_pred_baseline, zero_division=0)
    }
    
    # Comparison table
    comparison_final = pd.DataFrame({
        'Model': ['Baseline (Single LightGBM)', '🎯 ENSEMBLE (Optimized)'],
        'AUC': [f"{baseline_metrics['auc']:.4f}", f"{metrics_ensemble['auc']:.4f}"],
        'Precision': [f"{baseline_metrics['precision']:.4f}", f"{metrics_ensemble['precision']:.4f}"],
        'Recall': [f"{baseline_metrics['recall']:.4f}", f"{metrics_ensemble['recall']:.4f}"],
        'F1-Score': [f"{baseline_metrics['f1']:.4f}", f"{metrics_ensemble['f1']:.4f}"]
    })
    
    print("\n" + comparison_final.to_string(index=False))
    
    # Calculate improvements
    print(f"\n📈 ENSEMBLE IMPROVEMENTS OVER BASELINE:")
    print(f"   AUC:       {(metrics_ensemble['auc'] - baseline_metrics['auc'])*100:+.2f}%")
    print(f"   Precision: {(metrics_ensemble['precision'] - baseline_metrics['precision'])*100:+.2f}%")
    print(f"   Recall:    {(metrics_ensemble['recall'] - baseline_metrics['recall'])*100:+.2f}%")
    print(f"   F1-Score:  {(metrics_ensemble['f1'] - baseline_metrics['f1'])*100:+.2f}%")
    
    # ========================================================================
    # STEP 7: Generate Visualizations
    # ========================================================================
    print("\n" + "="*70)
    print("📈 STEP 7: Generating Visualizations")
    print("="*70)
    
    plot_dir = config.RESULTS_PATH / "plots" / "ensemble"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. ROC Curve Comparison
    print("\n📊 Generating ROC curve comparison...")
    fpr_ensemble, tpr_ensemble, _ = roc_curve(y_test, y_proba_ensemble)
    fpr_baseline, tpr_baseline, _ = roc_curve(y_test, y_proba_baseline)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_ensemble, tpr_ensemble, linewidth=2.5, 
             label=f'Ensemble AUC = {metrics_ensemble["auc"]:.4f}', color='darkgreen')
    plt.plot(fpr_baseline, tpr_baseline, linewidth=2, 
             label=f'Baseline AUC = {baseline_metrics["auc"]:.4f}', color='blue', linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('Test Set ROC Curve - Ensemble vs Baseline', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    roc_path = plot_dir / "ensemble_roc_curve.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curve saved: {roc_path}")
    
    # 2. Confusion Matrix
    print("📊 Generating confusion matrix...")
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['No Storm', 'Storm'],
               yticklabels=['No Storm', 'Storm'],
               cbar_kws={'label': 'Count'},
               annot_kws={'size': 14, 'weight': 'bold'})
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title('Ensemble Confusion Matrix - Test Set', fontsize=14, fontweight='bold', pad=20)
    
    # Add metrics text box
    metrics_text = f'Accuracy: {(cm[0,0]+cm[1,1])/cm.sum()*100:.2f}%\n'
    metrics_text += f'Precision: {metrics_ensemble["precision"]*100:.2f}%\n'
    metrics_text += f'Recall: {metrics_ensemble["recall"]*100:.2f}%\n'
    metrics_text += f'F1-Score: {metrics_ensemble["f1"]*100:.2f}%'
    
    plt.text(1.15, 0.5, metrics_text, 
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    cm_path = plot_dir / "ensemble_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved: {cm_path}")
    
    # ========================================================================
    # STEP 8: Save Ensemble
    # ========================================================================
    print("\n" + "="*70)
    print("💾 STEP 8: Saving Ensemble for Production")
    print("="*70)
    
    save_dir = config.MODEL_PATH / "ensemble"
    timestamp = ensemble.save_ensemble(save_dir)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("🎯 ENSEMBLE PIPELINE COMPLETE - SUMMARY")
    print("="*70)
    
    print(f"\n✅ Training successful!")
    print(f"\n📊 Final Test Metrics:")
    print(f"   AUC:       {metrics_ensemble['auc']:.4f} ({metrics_ensemble['auc']*100:.2f}%)")
    print(f"   Precision: {metrics_ensemble['precision']:.4f} ({metrics_ensemble['precision']*100:.2f}%)")
    print(f"   Recall:    {metrics_ensemble['recall']:.4f} ({metrics_ensemble['recall']*100:.2f}%)")
    print(f"   F1-Score:  {metrics_ensemble['f1']:.4f} ({metrics_ensemble['f1']*100:.2f}%)")
    
    print(f"\n🚀 Production ensemble ready for deployment!")
    print(f"\n📁 Artifacts saved:")
    print(f"   • Models: {save_dir}/")
    print(f"   • Plots:  {plot_dir}/")
    
    return ensemble, metrics_ensemble, baseline_metrics


if __name__ == "__main__":
    ensemble, ensemble_metrics, baseline_metrics = train_ensemble()
