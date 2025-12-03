"""
SOLARIS-X Production Inference System
Real-time geomagnetic storm prediction with optimized threshold
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from scripts.inference.production_config import config, get_active_threshold, get_active_metrics


class SolarStormPredictor:
    """Production-ready geomagnetic storm predictor"""
    
    def __init__(self, verbose=True):
        """
        Initialize predictor with trained ensemble model
        
        Parameters:
        -----------
        verbose : bool
            If True, print initialization messages
        """
        self.verbose = verbose
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🌩️  SOLARIS-X Storm Predictor {config.MODEL_INFO['version']}")
            print(f"{'='*70}")
        
        # Load ensemble model
        self._load_model()
        
        # Get active threshold
        self.threshold = get_active_threshold()
        self.metrics = get_active_metrics()
        
        if self.verbose:
            print(f"\n✅ Initialization complete!")
            print(f"   Active threshold: {self.threshold:.6f} ({config.USE_THRESHOLD})")
            print(f"   Expected recall:  {self.metrics['recall']*100:.2f}%")
            print(f"   Expected precision: {self.metrics['precision']*100:.2f}%")
    
    def _load_model(self):
        """Load trained ensemble model from disk"""
        model_path = config.ENSEMBLE_MODEL_PATH
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Please ensure ensemble model is trained and saved."
            )
        
        if self.verbose:
            print(f"\n📦 Loading model...")
            print(f"   Path: {model_path}")
        
        self.ensemble = joblib.load(model_path)
        
        if self.verbose:
            print(f"   ✅ Model loaded: {model_path.name}")
            print(f"   Ensemble weights:")
            print(f"      Precision Model: {self.ensemble.weights['precision']:.3f}")
            print(f"      Recall Model:    {self.ensemble.weights['recall']:.3f}")
            print(f"      Balanced Model:  {self.ensemble.weights['balanced']:.3f}")
    
    def _validate_features(self, X: pd.DataFrame):
        """
        Validate input features
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        
        Raises:
        -------
        ValueError if required features are missing
        """
        missing_features = set(config.REQUIRED_FEATURES) - set(X.columns)
        
        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}\n"
                f"Required features: {config.REQUIRED_FEATURES}"
            )
    
    def _get_alert_level(self, probability: float) -> str:
        """
        Convert probability to operational alert level
        
        Parameters:
        -----------
        probability : float
            Storm probability (0-1)
        
        Returns:
        --------
        alert_level : str
            Alert level (CRITICAL, HIGH, MODERATE, LOW, WATCH, NONE)
        """
        if probability >= config.ALERT_THRESHOLDS['CRITICAL']:
            return 'CRITICAL'
        elif probability >= config.ALERT_THRESHOLDS['HIGH']:
            return 'HIGH'
        elif probability >= config.ALERT_THRESHOLDS['MODERATE']:
            return 'MODERATE'
        elif probability >= config.ALERT_THRESHOLDS['LOW']:
            return 'LOW'
        elif probability >= config.ALERT_THRESHOLDS['WATCH']:
            return 'WATCH'
        else:
            return 'NONE'
    
    def _get_confidence(self, probability: float) -> str:
        """
        Determine prediction confidence based on distance from threshold
        
        Parameters:
        -----------
        probability : float
            Storm probability
        
        Returns:
        --------
        confidence : str
            HIGH, MODERATE, or LOW
        """
        distance = abs(probability - self.threshold)
        
        if distance > config.HIGH_CONFIDENCE_MARGIN:
            return 'HIGH'
        elif distance > config.MODERATE_CONFIDENCE_MARGIN:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get storm probability predictions
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features (must contain all REQUIRED_FEATURES)
        
        Returns:
        --------
        probabilities : np.ndarray
            Storm probabilities for each sample
        """
        # Validate features
        self._validate_features(X)
        
        # Ensure correct feature order
        X_ordered = X[config.REQUIRED_FEATURES]
        
        # Get predictions from ensemble
        probabilities = self.ensemble.predict_proba(X_ordered)
        
        return probabilities
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get binary storm predictions
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        
        Returns:
        --------
        predictions : np.ndarray
            Binary predictions (0=No Storm, 1=Storm)
        """
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= self.threshold).astype(int)
        return predictions
    
    def predict_single(self, features: dict) -> dict:
        """
        Predict storm for a single observation
        
        Parameters:
        -----------
        features : dict
            Dictionary with feature names and values
            Example: {'Bz_mean': -5.2, 'V_mean': 450, ...}
        
        Returns:
        --------
        result : dict
            Comprehensive prediction results including:
            - storm_probability
            - alert_level
            - prediction (STORM/NO STORM)
            - confidence
            - threshold_used
            - timestamp
        """
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Get prediction
        probability = self.predict_proba(X)[0]
        
        # Determine alert level and confidence
        alert_level = self._get_alert_level(probability)
        confidence = self._get_confidence(probability)
        prediction = 'STORM' if probability >= self.threshold else 'NO STORM'
        
        # Format result
        result = {
            'timestamp': datetime.now().isoformat(),
            'storm_probability': round(float(probability), config.DECIMAL_PLACES),
            'alert_level': alert_level,
            'prediction': prediction,
            'confidence': confidence,
            'threshold_used': self.threshold,
            'model_version': config.MODEL_INFO['version']
        }
        
        return result
    
    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Batch prediction with detailed output
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features (multiple observations)
        
        Returns:
        --------
        results : pd.DataFrame
            DataFrame with predictions, probabilities, and alert levels
        """
        # Get probabilities
        probabilities = self.predict_proba(X)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'storm_probability': probabilities,
            'alert_level': [self._get_alert_level(p) for p in probabilities],
            'prediction': ['STORM' if p >= self.threshold else 'NO STORM' 
                          for p in probabilities],
            'confidence': [self._get_confidence(p) for p in probabilities]
        })
        
        return results


# =============================================================================
# DEMO / TESTING
# =============================================================================

def demo_predictions():
    """Demonstrate predictor with sample data"""
    
    print("\n" + "="*70)
    print("🌩️  SOLARIS-X GEOMAGNETIC STORM PREDICTOR - DEMO")
    print("="*70)
    
    # Initialize predictor
    predictor = SolarStormPredictor(verbose=True)
    
    # =========================================================================
    # Example 1: Single Prediction
    # =========================================================================
    print("\n" + "="*70)
    print("📊 EXAMPLE 1: Single Storm Prediction")
    print("="*70)
    
    # Strong storm conditions (Bz negative, high speed)
    sample_features = {
        'Bz_mean': -8.5,
        'Bz_min': -15.2,
        'Bz_std': 3.4,
        'V_mean': 520.0,
        'V_max': 650.0,
        'V_std': 45.0,
        'Bt_mean': 12.0,
        'Bt_max': 18.5,
        'N_mean': 8.5,
        'N_std': 2.1,
        'E_field': 4.42,
        'P_dynamic': 4.5,
        'Beta': 0.8
    }
    
    result = predictor.predict_single(sample_features)
    
    print(f"\n🎯 Prediction Results:")
    print(f"   Storm Probability: {result['storm_probability']*100:>6.2f}%")
    print(f"   Alert Level:       {result['alert_level']:>8}")
    print(f"   Prediction:        {result['prediction']:>8}")
    print(f"   Confidence:        {result['confidence']:>8}")
    print(f"   Threshold Used:    {result['threshold_used']:.6f}")
    print(f"   Model Version:     {result['model_version']}")
    
    # =========================================================================
    # Example 2: Multiple Scenarios
    # =========================================================================
    print("\n" + "="*70)
    print("📊 EXAMPLE 2: Multiple Space Weather Scenarios")
    print("="*70)
    
    test_scenarios = [
        {
            'name': 'Extreme Storm (CME)',
            'features': {
                'Bz_mean': -25, 'Bz_min': -35, 'Bz_std': 5,
                'V_mean': 750, 'V_max': 900, 'V_std': 80,
                'Bt_mean': 28, 'Bt_max': 40, 
                'N_mean': 15, 'N_std': 4,
                'E_field': 10.5, 'P_dynamic': 12, 'Beta': 0.5
            }
        },
        {
            'name': 'Strong Storm',
            'features': {
                'Bz_mean': -12, 'Bz_min': -18, 'Bz_std': 3,
                'V_mean': 550, 'V_max': 650, 'V_std': 50,
                'Bt_mean': 15, 'Bt_max': 22,
                'N_mean': 10, 'N_std': 2.5,
                'E_field': 5.2, 'P_dynamic': 6, 'Beta': 0.7
            }
        },
        {
            'name': 'Moderate Conditions',
            'features': {
                'Bz_mean': -6, 'Bz_min': -10, 'Bz_std': 2,
                'V_mean': 450, 'V_max': 520, 'V_std': 35,
                'Bt_mean': 8, 'Bt_max': 12,
                'N_mean': 6, 'N_std': 1.5,
                'E_field': 2.1, 'P_dynamic': 3, 'Beta': 1.0
            }
        },
        {
            'name': 'Quiet Conditions',
            'features': {
                'Bz_mean': 2, 'Bz_min': -2, 'Bz_std': 1,
                'V_mean': 350, 'V_max': 400, 'V_std': 20,
                'Bt_mean': 5, 'Bt_max': 7,
                'N_mean': 4, 'N_std': 0.8,
                'E_field': 0.5, 'P_dynamic': 1.5, 'Beta': 1.5
            }
        }
    ]
    
    print(f"\n{'Scenario':<25} {'Probability':>12} {'Alert':>10} {'Prediction':>12} {'Confidence':>12}")
    print("-" * 70)
    
    for scenario in test_scenarios:
        result = predictor.predict_single(scenario['features'])
        print(f"{scenario['name']:<25} {result['storm_probability']*100:>11.2f}% "
              f"{result['alert_level']:>10} {result['prediction']:>12} {result['confidence']:>12}")
    
    # =========================================================================
    # Example 3: Batch Prediction
    # =========================================================================
    print("\n" + "="*70)
    print("📊 EXAMPLE 3: Batch Prediction (Multiple Timestamps)")
    print("="*70)
    
    # Create batch data
    batch_data = pd.DataFrame([scenario['features'] for scenario in test_scenarios])
    
    # Get predictions
    batch_results = predictor.predict_batch(batch_data)
    
    # Add scenario names
    batch_results.insert(0, 'scenario', [s['name'] for s in test_scenarios])
    
    print("\nBatch Results:")
    print(batch_results.to_string(index=False))
    
    print("\n" + "="*70)
    print("✅ Demo Complete - Inference System Ready for Production!")
    print("="*70)


if __name__ == "__main__":
    # Run demo
    demo_predictions()
