import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SpaceWeatherCleaner:
    def __init__(self):
        self.data_path = Path("data/processed/parquet")
        self.cleaned_path = Path("data/processed/cleaned")
        self.cleaned_path.mkdir(exist_ok=True)
        
    def load_cycle_data(self, cycle_name):
        """Load parquet data for specific cycle"""
        file_path = self.data_path / f"{cycle_name}.parquet"
        df = pd.read_parquet(file_path)
        print(f"📁 Loaded {cycle_name}: {df.shape[0]:,} records")
        return df
    
    def detect_anomalies(self, df, cycle_name):
        """Detect and flag anomalous values using physics constraints"""
        print(f"\n🔍 ANOMALY DETECTION - {cycle_name.upper()}")
        
        anomaly_flags = pd.DataFrame(index=df.index)
        
        # Physics-based constraints for space weather data
        constraints = {
            'IMF_Magnitude': (0, 100),      # nT - typical range 0-50nT
            'IMF_Bz': (-100, 100),          # nT - can be extreme during storms
            'IMF_By': (-100, 100),          # nT
            'IMF_Bx': (-100, 100),          # nT
            'Flow_Speed': (200, 1200),      # km/s - solar wind speed range
            'Proton_Density': (0.1, 200),  # n/cc - typical range 1-50
            'Proton_Temp': (1000, 2000000), # K - can be very high
            'Flow_Pressure': (0, 50),       # nPa - dynamic pressure
            'Dst_index': (-600, 100),       # nT - storm range
            'AE_index': (0, 4000),          # nT - auroral activity
            'Kp_index': (0, 90)             # Kp*10 format
        }
        
        total_anomalies = 0
        
        for param, (min_val, max_val) in constraints.items():
            if param in df.columns:
                anomalies = (df[param] < min_val) | (df[param] > max_val)
                anomaly_count = anomalies.sum()
                anomaly_flags[f'{param}_anomaly'] = anomalies
                
                if anomaly_count > 0:
                    print(f"  ⚠️ {param}: {anomaly_count:,} anomalies "
                          f"({anomaly_count/len(df)*100:.2f}%)")
                    total_anomalies += anomaly_count
        
        print(f"  🎯 Total anomalies: {total_anomalies:,} values")
        return anomaly_flags
    
    def handle_missing_data(self, df, cycle_name):
        """Intelligent missing data handling for space weather parameters"""
        print(f"\n🔧 MISSING DATA HANDLING - {cycle_name.upper()}")
        
        # Parameters that can be interpolated vs. must be removed
        interpolatable = ['IMF_Magnitude', 'IMF_Bz', 'IMF_By', 'IMF_Bx', 
                         'Flow_Speed', 'Proton_Density', 'Proton_Temp']
        
        indices_params = ['Dst_index', 'AE_index', 'Kp_index']
        
        # Count missing data before cleaning
        missing_before = df.isnull().sum()
        total_missing = missing_before.sum()
        
        print(f"  📊 Missing data before cleaning: {total_missing:,} values")
        
        # Interpolate solar wind parameters (max gap: 6 hours)
        for param in interpolatable:
            if param in df.columns:
                missing_count = df[param].isnull().sum()
                if missing_count > 0:
                    # Linear interpolation for gaps ≤ 6 hours
                    df[param] = df[param].interpolate(
                        method='linear', 
                        limit=6,  # Max 6 consecutive NaNs
                        limit_area='inside'
                    )
                    
                    # Forward fill for remaining short gaps
                    df[param] = df[param].fillna(method='ffill', limit=2)
                    
                    remaining = df[param].isnull().sum()
                    print(f"    {param}: {missing_count:,} → {remaining:,} "
                          f"({((missing_count-remaining)/missing_count*100):.1f}% filled)")
        
        # Handle geomagnetic indices differently (use forward fill only)
        for param in indices_params:
            if param in df.columns and df[param].isnull().sum() > 0:
                missing_count = df[param].isnull().sum()
                df[param] = df[param].fillna(method='ffill', limit=1)
                remaining = df[param].isnull().sum()
                print(f"    {param}: {missing_count:,} → {remaining:,}")
        
        # Final missing data summary
        missing_after = df.isnull().sum()
        total_after = missing_after.sum()
        improvement = ((total_missing - total_after) / total_missing * 100)
        
        print(f"  ✅ Missing data after cleaning: {total_after:,} values "
              f"({improvement:.1f}% improvement)")
        
        return df
    
    def remove_extreme_outliers(self, df, anomaly_flags, cycle_name):
        """Remove data points with multiple anomaly flags"""
        print(f"\n🚮 OUTLIER REMOVAL - {cycle_name.upper()}")
        
        # Count anomaly flags per row
        anomaly_counts = anomaly_flags.sum(axis=1)
        
        # Remove rows with 3+ simultaneous anomalies (likely sensor failures)
        extreme_outliers = anomaly_counts >= 3
        outlier_count = extreme_outliers.sum()
        
        if outlier_count > 0:
            print(f"  🗑️ Removing {outlier_count:,} extreme outliers "
                  f"({outlier_count/len(df)*100:.3f}%)")
            df_clean = df[~extreme_outliers].copy()
        else:
            df_clean = df.copy()
            print(f"  ✅ No extreme outliers detected")
        
        return df_clean
    
    def validate_temporal_consistency(self, df, cycle_name):
        """Check for temporal gaps and duplicates"""
        print(f"\n⏰ TEMPORAL VALIDATION - {cycle_name.upper()}")
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['Datetime']).sum()
        if duplicates > 0:
            print(f"  🔄 Removing {duplicates} duplicate timestamps")
            df = df.drop_duplicates(subset=['Datetime']).copy()
        
        # Check for large time gaps (>6 hours)
        df = df.sort_values('Datetime')
        time_diffs = df['Datetime'].diff()
        large_gaps = (time_diffs > pd.Timedelta(hours=6)).sum()
        
        if large_gaps > 0:
            print(f"  ⏳ Found {large_gaps} gaps >6 hours")
            # Report largest gaps
            max_gap = time_diffs.max()
            print(f"    📅 Largest gap: {max_gap}")
        else:
            print(f"  ✅ No significant temporal gaps")
        
        return df
    
    def generate_data_quality_report(self, df_original, df_cleaned, cycle_name):
        """Generate comprehensive data quality report"""
        print(f"\n📋 DATA QUALITY REPORT - {cycle_name.upper()}")
        
        print(f"  📊 Original records: {len(df_original):,}")
        print(f"  📊 Cleaned records: {len(df_cleaned):,}")
        print(f"  📊 Records retained: {len(df_cleaned)/len(df_original)*100:.2f}%")
        
        # Missing data comparison
        missing_orig = df_original.isnull().sum().sum()
        missing_clean = df_cleaned.isnull().sum().sum()
        
        print(f"  🔍 Missing values: {missing_orig:,} → {missing_clean:,}")
        print(f"  📅 Date range: {df_cleaned['Datetime'].min()} to {df_cleaned['Datetime'].max()}")
        
        # Parameter-wise completeness
        completeness = (1 - df_cleaned.isnull().sum() / len(df_cleaned)) * 100
        print(f"  📈 Data completeness by parameter:")
        for param, comp in completeness.items():
            if param != 'Datetime':
                print(f"    {param}: {comp:.1f}%")
    
    def process_cycle(self, cycle_name):
        """Complete cleaning pipeline for one solar cycle"""
        print(f"\n{'='*60}")
        print(f"🚀 PROCESSING SOLAR {cycle_name.upper()}")
        print(f"{'='*60}")
        
        # Load data
        df = self.load_cycle_data(cycle_name)
        df_original = df.copy()
        
        # Step 1: Anomaly detection
        anomaly_flags = self.detect_anomalies(df, cycle_name)
        
        # Step 2: Missing data handling
        df = self.handle_missing_data(df, cycle_name)
        
        # Step 3: Remove extreme outliers
        df = self.remove_extreme_outliers(df, anomaly_flags, cycle_name)
        
        # Step 4: Temporal validation
        df = self.validate_temporal_consistency(df, cycle_name)
        
        # Step 5: Quality report
        self.generate_data_quality_report(df_original, df, cycle_name)
        
        # Save cleaned data
        output_path = self.cleaned_path / f"{cycle_name}_cleaned.parquet"
        df.to_parquet(output_path, compression='snappy', index=False)
        
        file_size = output_path.stat().st_size / (1024*1024)
        print(f"  💾 Saved to: {output_path}")
        print(f"  📦 File size: {file_size:.1f} MB")
        
        return df

def main():
    """Execute complete data cleaning pipeline"""
    print("🛰️ SOLARIS-X DATA CLEANING PROTOCOL INITIATED")
    print("=" * 80)
    
    cleaner = SpaceWeatherCleaner()
    
    cycles = ['cycle23', 'cycle24', 'cycle25']
    cleaned_cycles = {}
    
    for cycle in cycles:
        try:
            cleaned_df = cleaner.process_cycle(cycle)
            cleaned_cycles[cycle] = cleaned_df
        except Exception as e:
            print(f"❌ Error processing {cycle}: {e}")
    
    # Final mission summary
    print(f"\n{'='*80}")
    print("🎯 MISSION SUMMARY - DATA CLEANING COMPLETE")
    print("=" * 80)
    
    total_records = sum(len(df) for df in cleaned_cycles.values())
    total_size = sum((cleaner.cleaned_path / f"{cycle}_cleaned.parquet").stat().st_size 
                    for cycle in cleaned_cycles.keys()) / (1024*1024)
    
    print(f"✅ Cycles processed: {len(cleaned_cycles)}/3")
    print(f"📊 Total cleaned records: {total_records:,}")
    print(f"💾 Total cleaned data size: {total_size:.1f} MB")
    print(f"📁 Output location: {cleaner.cleaned_path}")
    
    print("\n🚀 READY FOR FEATURE ENGINEERING PHASE")

if __name__ == "__main__":
    main()
