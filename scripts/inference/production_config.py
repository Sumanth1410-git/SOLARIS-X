"""
SOLARIS-X Production Configuration
Optimized threshold: 0.309059 (70% recall, 62% precision)
Decision: Safety-critical system - prioritize storm detection over false alarms
"""

from pathlib import Path

class ProductionConfig:
    """Production deployment settings for SOLARIS-X"""
    
    # ========================================================================
    # MODEL CONFIGURATION
    # ========================================================================
    
    # Path to trained ensemble model
    ENSEMBLE_MODEL_PATH = Path("models/ensemble/ensemble_lightgbm_20251203_114330.pkl")
    
    # ========================================================================
    # OPTIMIZED THRESHOLDS (from threshold optimization analysis)
    # ========================================================================
    
    # Primary threshold (RECOMMENDED for production)
    # Optimized for maximum recall (catch more storms)
    PRIMARY_THRESHOLD = 0.309059
    PRIMARY_METRICS = {
        'recall': 0.7001,      # 70% of storms detected
        'precision': 0.6180,   # 62% precision (38% false alarm rate)
        'f1_score': 0.6565,
        'description': 'Safety-critical: Prioritize storm detection'
    }
    
    # Secondary threshold (Conservative option)
    # Higher precision, lower recall
    SECONDARY_THRESHOLD = 0.455839
    SECONDARY_METRICS = {
        'recall': 0.6323,      # 63% of storms detected
        'precision': 0.7006,   # 70% precision
        'f1_score': 0.6647,
        'description': 'Conservative: Fewer false alarms'
    }
    
    # Default threshold (Balanced)
    DEFAULT_THRESHOLD = 0.500000
    DEFAULT_METRICS = {
        'recall': 0.6007,
        'precision': 0.7270,
        'f1_score': 0.6579,
        'description': 'Balanced approach'
    }
    
    # Active threshold selection
    USE_THRESHOLD = "PRIMARY"  # Options: "PRIMARY", "SECONDARY", "DEFAULT"
    
    # ========================================================================
    # ALERT LEVEL THRESHOLDS
    # ========================================================================
    
    # Map probabilities to operational alert levels
    ALERT_THRESHOLDS = {
        'CRITICAL': 0.80,   # Probability ≥80% = Major storm likely
        'HIGH': 0.60,       # Probability 60-80% = Storm probable
        'MODERATE': 0.40,   # Probability 40-60% = Storm possible
        'LOW': 0.31,        # Probability 31-40% = Storm watch
        'WATCH': 0.20       # Probability 20-31% = Monitor conditions
    }
    
    # ========================================================================
    # FEATURE REQUIREMENTS
    # ========================================================================
    
    # Required input features (must match training data exactly)
    REQUIRED_FEATURES = [
        'Bz_mean',      # Average IMF Bz (nT)
        'Bz_min',       # Minimum IMF Bz (nT)
        'Bz_std',       # Standard deviation of Bz
        'V_mean',       # Average solar wind speed (km/s)
        'V_max',        # Maximum solar wind speed (km/s)
        'V_std',        # Standard deviation of speed
        'Bt_mean',      # Average total IMF magnitude (nT)
        'Bt_max',       # Maximum total IMF magnitude (nT)
        'N_mean',       # Average proton density (cm^-3)
        'N_std',        # Standard deviation of density
        'E_field',      # Induced electric field (mV/m)
        'P_dynamic',    # Dynamic pressure (nPa)
        'Beta'          # Plasma beta
    ]
    
    # Feature units (for documentation)
    FEATURE_UNITS = {
        'Bz_mean': 'nT',
        'Bz_min': 'nT',
        'Bz_std': 'nT',
        'V_mean': 'km/s',
        'V_max': 'km/s',
        'V_std': 'km/s',
        'Bt_mean': 'nT',
        'Bt_max': 'nT',
        'N_mean': 'cm^-3',
        'N_std': 'cm^-3',
        'E_field': 'mV/m',
        'P_dynamic': 'nPa',
        'Beta': 'dimensionless'
    }
    
    # ========================================================================
    # MODEL METADATA
    # ========================================================================
    
    MODEL_INFO = {
        'version': 'v1.0.0',
        'model_type': 'Ensemble LightGBM (3 specialized models)',
        'training_period': '1996-05-01 to 2016-12-31',
        'validation_period': '2017-01-01 to 2020-12-31',
        'test_period': '2021-01-01 to 2025-09-03',
        'test_auc': 0.9802,
        'optimization_date': '2025-12-03',
        'threshold_selection_method': 'Cost-benefit analysis + Industry benchmarks'
    }
    
    # Justification for threshold selection
    DEPLOYMENT_RATIONALE = """
    THRESHOLD SELECTION RATIONALE:
    
    Selected: PRIMARY threshold = 0.309059
    
    Decision Factors:
    1. Space weather forecasting is a SAFETY-CRITICAL application
    2. Cost of missing a storm ($10M-$2B) >> Cost of false alarm ($50K-$200K)
    3. Industry benchmark (NOAA SWPC): ~75% recall, ~60% precision
    4. Operational priority: Better to warn unnecessarily than miss disaster
    
    Expected Performance:
    - Recall: 70.01% (catch 7 out of 10 geomagnetic storms)
    - Precision: 61.80% (38% false alarm rate - acceptable trade-off)
    - F1-Score: 65.65%
    - AUC: 98.02% (excellent discrimination capability)
    
    Stakeholder Impact:
    - Satellite operators: Advanced warning for protective measures
    - Power grid utilities: Time to activate storm protocols
    - Airlines: Ability to reroute polar flights
    - Space agencies: Astronaut safety alerts
    
    Risk Assessment:
    - False alarms: Manageable operational costs, no safety risk
    - Missed storms: Potential infrastructure damage, safety hazards
    - Trade-off: Heavily favors false alarms over missed detections
    """
    
    # ========================================================================
    # OPERATIONAL SETTINGS
    # ========================================================================
    
    # Prediction window
    PREDICTION_HORIZON = '1 hour'  # Model predicts Kp in next hour
    
    # Confidence thresholds
    HIGH_CONFIDENCE_MARGIN = 0.20  # |probability - threshold| > 0.20
    MODERATE_CONFIDENCE_MARGIN = 0.10
    
    # Output formatting
    DECIMAL_PLACES = 4


# Create singleton instance
config = ProductionConfig()


# Helper function to get active threshold
def get_active_threshold():
    """Get the currently active threshold value"""
    threshold_map = {
        'PRIMARY': config.PRIMARY_THRESHOLD,
        'SECONDARY': config.SECONDARY_THRESHOLD,
        'DEFAULT': config.DEFAULT_THRESHOLD
    }
    return threshold_map[config.USE_THRESHOLD]


# Helper function to get active metrics
def get_active_metrics():
    """Get expected metrics for active threshold"""
    metrics_map = {
        'PRIMARY': config.PRIMARY_METRICS,
        'SECONDARY': config.SECONDARY_METRICS,
        'DEFAULT': config.DEFAULT_METRICS
    }
    return metrics_map[config.USE_THRESHOLD]


if __name__ == "__main__":
    # Test configuration
    print("="*70)
    print("🚀 SOLARIS-X PRODUCTION CONFIGURATION")
    print("="*70)
    print(f"\n📦 Model: {config.ENSEMBLE_MODEL_PATH.name}")
    print(f"📅 Trained: {config.MODEL_INFO['training_period']}")
    print(f"🎯 Version: {config.MODEL_INFO['version']}")
    
    print(f"\n⚙️  Active Threshold: {config.USE_THRESHOLD}")
    print(f"   Value: {get_active_threshold():.6f}")
    
    metrics = get_active_metrics()
    print(f"\n📊 Expected Performance:")
    print(f"   Recall:    {metrics['recall']*100:.2f}%")
    print(f"   Precision: {metrics['precision']*100:.2f}%")
    print(f"   F1-Score:  {metrics['f1_score']*100:.2f}%")
    print(f"   Strategy:  {metrics['description']}")
    
    print(f"\n🚨 Alert Levels:")
    for level, threshold in config.ALERT_THRESHOLDS.items():
        print(f"   {level:>10}: Probability ≥ {threshold:.2f}")
    
    print(f"\n✅ Required Features: {len(config.REQUIRED_FEATURES)}")
    for i, feature in enumerate(config.REQUIRED_FEATURES, 1):
        unit = config.FEATURE_UNITS[feature]
        print(f"   {i:2}. {feature:15} ({unit})")
    
    print("\n" + "="*70)
