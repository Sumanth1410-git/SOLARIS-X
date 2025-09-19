import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SpaceWeatherFeatureEngineer:
    def __init__(self):
        self.cleaned_path = Path("data/processed/cleaned")
        self.features_path = Path("data/processed/features")
        self.features_path.mkdir(exist_ok=True)
        
    def load_cleaned_data(self):
        """Load all cleaned cycle data and combine"""
        print("📁 Loading cleaned datasets...")
        
        cycles = ['cycle23', 'cycle24', 'cycle25']
        dfs = []
        
        for cycle in cycles:
            file_path = self.cleaned_path / f"{cycle}_cleaned.parquet"
            df = pd.read_parquet(file_path)
            df['Solar_Cycle'] = cycle
            dfs.append(df)
            print(f"  ✅ {cycle}: {len(df):,} records")
        
        # Combine all cycles
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values('Datetime').reset_index(drop=True)
        
        print(f"📊 Total combined records: {len(combined_df):,}")
        print(f"📅 Full date range: {combined_df['Datetime'].min()} to {combined_df['Datetime'].max()}")
        
        return combined_df
    
    def create_magnetic_reconnection_features(self, df):
        """Create features based on magnetic reconnection physics"""
        print("\n🧲 Creating Magnetic Reconnection Features...")
        
        # IMF Clock Angle (critical for magnetospheric coupling)
        df['IMF_Clock_Angle'] = np.arctan2(df['IMF_By'], df['IMF_Bz']) * 180 / np.pi
        
        # IMF Cone Angle (solar wind flow alignment)
        df['IMF_Cone_Angle'] = np.arccos(
            np.abs(df['IMF_Bx']) / df['IMF_Magnitude'].clip(lower=0.1)
        ) * 180 / np.pi
        
        # Southward IMF component (key for energy coupling)
        df['IMF_Bz_South'] = np.minimum(df['IMF_Bz'], 0)  # Only negative values
        
        # Northward IMF component
        df['IMF_Bz_North'] = np.maximum(df['IMF_Bz'], 0)  # Only positive values
        
        # IMF Magnitude in GSM Y-Z plane (perpendicular component)
        df['IMF_Bt_YZ'] = np.sqrt(df['IMF_By']**2 + df['IMF_Bz']**2)
        
        # Magnetic field stability (measure of coherence)
        df['IMF_Stability'] = df['IMF_Magnitude'] / (df['IMF_Bt_YZ'] + 0.1)
        
        print(f"  ✅ Created 6 magnetic reconnection features")
        return df
    
    def create_coupling_functions(self, df):
        """Create solar wind-magnetosphere coupling functions"""
        print("\n⚡ Creating Coupling Function Features...")
        
        # Newell Coupling Function (most accurate empirical coupling)
        # CF = v^(4/3) * Bt^(2/3) * sin^8(theta/2)
        theta_rad = df['IMF_Clock_Angle'] * np.pi / 180
        sin_half_theta = np.sin(theta_rad / 2)
        
        df['Newell_Coupling'] = (
            df['Flow_Speed'].clip(lower=200)**(4/3) * 
            df['IMF_Bt_YZ'].clip(lower=0.1)**(2/3) * 
            sin_half_theta**8
        )
        
        # Epsilon Parameter (energy coupling rate)
        # epsilon = v * Bt^2 * sin^4(theta/2) / mu_0
        mu_0 = 4 * np.pi * 1e-7  # Permeability of free space
        df['Epsilon_Parameter'] = (
            df['Flow_Speed'] * 1000 *  # Convert km/s to m/s
            (df['IMF_Bt_YZ'] * 1e-9)**2 *  # Convert nT to T
            sin_half_theta**4
        ) / mu_0 / 1e12  # Scale to reasonable units
        
        # Merging Electric Field (reconnection efficiency)
        # E_m = v * Bt * sin^2(theta/2)
        df['Merging_Electric_Field'] = (
            df['Flow_Speed'] * df['IMF_Bt_YZ'] * sin_half_theta**2 / 1000
        )
        
        # Viscous Interaction (velocity-dependent coupling)
        df['Viscous_Function'] = df['Flow_Speed']**2 * df['Proton_Density']
        
        print(f"  ✅ Created 4 coupling function features")
        return df
    
    def create_solar_wind_dynamics(self, df):
        """Create solar wind dynamic features"""
        print("\n🌊 Creating Solar Wind Dynamic Features...")
        
        # Dynamic pressure variations
        df['Pressure_Gradient'] = df['Flow_Pressure'].diff()
        
        # Mach numbers (flow regime indicators)
        # Alfven Mach Number = v / v_A, where v_A = B/sqrt(mu_0 * rho)
        proton_mass = 1.67e-27  # kg
        rho = df['Proton_Density'] * 1e6 * proton_mass  # Convert to kg/m^3
        
        v_alfven = (df['IMF_Magnitude'] * 1e-9) / np.sqrt(4*np.pi*1e-7 * rho)
        df['Alfven_Mach'] = (df['Flow_Speed'] * 1000) / v_alfven.clip(lower=1e3)
        
        # Beta plasma parameter (thermal vs magnetic pressure)
        k_b = 1.38e-23  # Boltzmann constant
        thermal_pressure = df['Proton_Density'] * 1e6 * k_b * df['Proton_Temp']
        magnetic_pressure = (df['IMF_Magnitude'] * 1e-9)**2 / (2 * 4*np.pi*1e-7)
        df['Plasma_Beta'] = thermal_pressure / magnetic_pressure.clip(lower=1e-15)
        
        # Flow angle variations (Parker spiral deviations)
        df['Flow_Deflection'] = np.arctan2(df['IMF_By'], df['IMF_Bx']) * 180 / np.pi
        
        # Density-speed correlation (solar wind stream structure)
        df['Density_Speed_Product'] = df['Proton_Density'] * df['Flow_Speed']
        
        print(f"  ✅ Created 6 solar wind dynamic features")
        return df
    
    def create_temporal_features(self, df):
        """Create multi-scale temporal features"""
        print("\n⏰ Creating Temporal Features...")
        
        # Time-based cyclical features
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Day_Sin'] = np.sin(2 * np.pi * df['Day'] / 365.25)
        df['Day_Cos'] = np.cos(2 * np.pi * df['Day'] / 365.25)
        
        # Solar cycle phase (11-year cycle)
        reference_year = 1996
        cycle_phase = ((df['Year'] - reference_year) % 11) / 11
        df['Solar_Cycle_Phase'] = np.sin(2 * np.pi * cycle_phase)
        
        print(f"  ✅ Created 5 temporal features")
        return df
    
    def create_lag_features(self, df):
        """Create lag features for temporal dependencies"""
        print("\n🔄 Creating Lag Features...")
        
        # Key parameters for lag analysis
        lag_params = ['IMF_Bz', 'IMF_Magnitude', 'Flow_Speed', 'Flow_Pressure', 'Newell_Coupling']
        lag_hours = [1, 3, 6, 12, 24]
        
        lag_count = 0
        for param in lag_params:
            if param in df.columns:
                for lag in lag_hours:
                    col_name = f'{param}_lag_{lag}h'
                    df[col_name] = df[param].shift(lag)
                    lag_count += 1
        
        print(f"  ✅ Created {lag_count} lag features")
        return df
    
    def create_rolling_statistics(self, df):
        """Create rolling window statistics"""
        print("\n📊 Creating Rolling Statistics...")
        
        # Parameters for rolling statistics
        rolling_params = ['IMF_Bz', 'Flow_Speed', 'Dst_index', 'Newell_Coupling']
        windows = [3, 6, 12, 24]  # Hours
        
        stats_count = 0
        for param in rolling_params:
            if param in df.columns:
                for window in windows:
                    # Rolling mean
                    df[f'{param}_mean_{window}h'] = df[param].rolling(window, min_periods=1).mean()
                    # Rolling standard deviation
                    df[f'{param}_std_{window}h'] = df[param].rolling(window, min_periods=1).std()
                    stats_count += 2
        
        print(f"  ✅ Created {stats_count} rolling statistics")
        return df
    
    def create_storm_targets(self, df):
        """Create storm classification targets"""
        print("\n🎯 Creating Storm Target Variables...")
        
        # Binary storm classification (Dst < -50 nT)
        df['Storm_Binary'] = (df['Dst_index'] < -50).astype(int)
        
        # Multi-class storm intensity
        conditions = [
            df['Dst_index'] >= -30,      # Quiet
            (df['Dst_index'] < -30) & (df['Dst_index'] >= -50),  # Weak
            (df['Dst_index'] < -50) & (df['Dst_index'] >= -100), # Moderate
            (df['Dst_index'] < -100) & (df['Dst_index'] >= -200), # Strong
            df['Dst_index'] < -200       # Severe
        ]
        choices = [0, 1, 2, 3, 4]  # Quiet, Weak, Moderate, Strong, Severe
        
        df['Storm_Intensity'] = np.select(conditions, choices, default=0)
        
        # Storm onset detection (rapid Dst decrease)
        df['Dst_Rate'] = df['Dst_index'].diff()
        df['Storm_Onset'] = (df['Dst_Rate'] < -30).astype(int)  # >30 nT/hr decrease
        
        # Future storm prediction target (6-hour ahead)
        df['Storm_6h_Ahead'] = df['Storm_Binary'].shift(-6)
        
        print(f"  ✅ Created 4 storm target variables")
        return df
    
    def finalize_features(self, df):
        """Final feature processing and validation"""
        print("\n🔧 Finalizing Feature Set...")
        
        # Drop intermediate calculation columns
        cols_to_drop = ['Year', 'Day', 'Hour']
        existing_drops = [col for col in cols_to_drop if col in df.columns]
        if existing_drops:
            df = df.drop(columns=existing_drops)
        
        # Handle infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Feature count and summary
        feature_cols = [col for col in df.columns if col not in ['Datetime', 'Solar_Cycle']]
        target_cols = [col for col in feature_cols if 'Storm' in col]
        predictor_cols = [col for col in feature_cols if col not in target_cols]
        
        print(f"  📊 Total features created: {len(feature_cols)}")
        print(f"  🎯 Target variables: {len(target_cols)}")
        print(f"  🔮 Predictor features: {len(predictor_cols)}")
        print(f"  📅 Records with features: {len(df):,}")
        
        # Data quality check
        missing_per_col = df.isnull().sum()
        high_missing = missing_per_col[missing_per_col > len(df) * 0.1]
        
        if len(high_missing) > 0:
            print(f"  ⚠️ Columns with >10% missing: {len(high_missing)}")
            for col, missing in high_missing.items():
                print(f"    {col}: {missing:,} missing ({missing/len(df)*100:.1f}%)")
        else:
            print(f"  ✅ No columns with excessive missing data")
        
        return df
    
    def save_engineered_features(self, df):
        """Save final feature set"""
        print(f"\n💾 Saving Engineered Features...")
        
        # Save complete feature set
        output_path = self.features_path / "solaris_x_features.parquet"
        df.to_parquet(output_path, compression='snappy', index=False)
        
        file_size = output_path.stat().st_size / (1024*1024)
        print(f"  📦 Saved to: {output_path}")
        print(f"  📊 File size: {file_size:.1f} MB")
        
        # Create feature list documentation
        feature_list = []
        for col in df.columns:
            if col not in ['Datetime', 'Solar_Cycle']:
                feature_list.append(col)
        
        doc_path = self.features_path / "feature_list.txt"
        with open(doc_path, 'w') as f:
            f.write("SOLARIS-X Feature Engineering - Complete Feature List\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total Features: {len(feature_list)}\n")
            f.write(f"Dataset Records: {len(df):,}\n")
            f.write(f"Date Range: {df['Datetime'].min()} to {df['Datetime'].max()}\n\n")
            f.write("Feature Categories:\n")
            f.write("-"*30 + "\n")
            
            categories = {
                'Original Parameters': [col for col in feature_list if not any(x in col for x in ['_lag_', '_mean_', '_std_', 'Storm', 'Clock', 'Coupling', 'Epsilon', 'Mach', 'Beta'])],
                'Magnetic Features': [col for col in feature_list if any(x in col for x in ['Clock', 'Cone', 'IMF_', 'Magnetic'])],
                'Coupling Functions': [col for col in feature_list if any(x in col for x in ['Coupling', 'Epsilon', 'Merging', 'Viscous'])],
                'Solar Wind Dynamics': [col for col in feature_list if any(x in col for x in ['Mach', 'Beta', 'Deflection', 'Gradient'])],
                'Temporal Features': [col for col in feature_list if any(x in col for x in ['Sin', 'Cos', 'Phase', 'Hour', 'Day'])],
                'Lag Features': [col for col in feature_list if '_lag_' in col],
                'Rolling Statistics': [col for col in feature_list if any(x in col for x in ['_mean_', '_std_'])],
                'Storm Targets': [col for col in feature_list if 'Storm' in col]
            }
            
            for category, features in categories.items():
                if features:
                    f.write(f"\n{category} ({len(features)} features):\n")
                    for feature in sorted(features):
                        f.write(f"  - {feature}\n")
        
        print(f"  📋 Feature documentation: {doc_path}")
        return output_path

def main():
    """Execute complete feature engineering pipeline"""
    print("🚀 SOLARIS-X ADVANCED FEATURE ENGINEERING PROTOCOL")
    print("=" * 80)
    
    engineer = SpaceWeatherFeatureEngineer()
    
    try:
        # Step 1: Load cleaned data
        df = engineer.load_cleaned_data()
        
        # Step 2: Create physics-based features
        df = engineer.create_magnetic_reconnection_features(df)
        df = engineer.create_coupling_functions(df)
        df = engineer.create_solar_wind_dynamics(df)
        
        # Step 3: Create temporal features
        df = engineer.create_temporal_features(df)
        df = engineer.create_lag_features(df)
        df = engineer.create_rolling_statistics(df)
        
        # Step 4: Create target variables
        df = engineer.create_storm_targets(df)
        
        # Step 5: Finalize and save
        df = engineer.finalize_features(df)
        output_path = engineer.save_engineered_features(df)
        
        print(f"\n{'='*80}")
        print("🎯 FEATURE ENGINEERING MISSION COMPLETE")
        print("=" * 80)
        print(f"✅ Successfully created comprehensive feature set")
        print(f"📊 Total engineered features: {len([c for c in df.columns if c not in ['Datetime', 'Solar_Cycle']])}")
        print(f"🎯 Storm prediction targets: 4 variables")
        print(f"💾 Output file: {output_path}")
        print(f"\n🚀 READY FOR MODEL TRAINING PHASE")
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
