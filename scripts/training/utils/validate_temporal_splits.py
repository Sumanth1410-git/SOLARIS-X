import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

class TemporalSplitValidator:
    def __init__(self, data_folder):
        """
        Initialize with your data folder
        data_folder: path to your data/processed/cleaned/ folder
        """
        self.data_folder = data_folder
        self.df = None
        
    def load_and_combine_cycles(self):
        """Load and combine all cycle files with proper datetime indexing"""
        print("🔍 Loading cycle datasets for validation...")
        
        cycle_files = [
            'cycle23_cleaned.parquet',
            'cycle24_cleaned.parquet', 
            'cycle25_cleaned.parquet'
        ]
        
        dataframes = []
        
        for cycle_file in cycle_files:
            file_path = os.path.join(self.data_folder, cycle_file)
            if os.path.exists(file_path):
                print(f"📊 Loading {cycle_file}...")
                cycle_df = pd.read_parquet(file_path)
                
                # CRITICAL FIX: Set Datetime column as index
                if 'Datetime' in cycle_df.columns:
                    cycle_df = cycle_df.set_index('Datetime')
                    print(f"   ✅ Set Datetime as index")
                else:
                    print(f"   ⚠️  No Datetime column found!")
                    
                print(f"   Shape: {cycle_df.shape}")
                print(f"   Date range: {cycle_df.index.min()} to {cycle_df.index.max()}")
                dataframes.append(cycle_df)
            else:
                print(f"⚠️  {cycle_file} not found at {file_path}")
        
        if not dataframes:
            raise FileNotFoundError("No cycle files found! Check your data paths.")
            
        # Combine all cycles
        self.df = pd.concat(dataframes, axis=0)
        self.df = self.df.sort_index()  # Sort by datetime index
        
        print(f"\n📈 COMBINED DATASET:")
        print(f"   Total shape: {self.df.shape}")
        print(f"   Complete date range: {self.df.index.min()} to {self.df.index.max()}")
        print(f"   Years covered: {self.df.index.year.min()} to {self.df.index.year.max()}")
        print(f"   Available columns: {list(self.df.columns)}")
        
        return self.df
    
    def verify_splits(self):
        """Verify current train/val/test splits"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_combine_cycles() first.")
            
        # Define temporal splits (based on your presentation)
        train_mask = self.df.index.year <= 2016
        val_mask = (self.df.index.year >= 2017) & (self.df.index.year <= 2020)  
        test_mask = self.df.index.year >= 2021
        
        train_data = self.df[train_mask]
        val_data = self.df[val_mask]
        test_data = self.df[test_mask]
        
        print("\n🔍 CURRENT SPLIT ANALYSIS:")
        print(f"Train: {len(train_data):,} samples ({train_data.index.year.min()}-{train_data.index.year.max()})")
        print(f"       Date range: {train_data.index.min()} to {train_data.index.max()}")
        
        print(f"Val:   {len(val_data):,} samples ({val_data.index.year.min()}-{val_data.index.year.max()})")
        print(f"       Date range: {val_data.index.min()} to {val_data.index.max()}")
        
        print(f"Test:  {len(test_data):,} samples ({test_data.index.year.min()}-{test_data.index.year.max()})")
        print(f"       Date range: {test_data.index.min()} to {test_data.index.max()}")
        
        # Check for overlaps (CRITICAL)
        train_max = train_data.index.max()
        val_min = val_data.index.min()
        val_max = val_data.index.max()
        test_min = test_data.index.min()
        
        print("\n⚠️  OVERLAP CHECK:")
        overlap_issues = []
        
        if train_max >= val_min:
            print(f"🚨 CRITICAL: Train-Val overlap detected!")
            print(f"   Train ends:  {train_max}")
            print(f"   Val starts:  {val_min}")
            overlap_issues.append("Train-Val")
        else:
            gap_hours = (val_min - train_max).total_seconds() / 3600
            print(f"✅ No Train-Val overlap (Gap: {gap_hours:.1f} hours)")
            
        if val_max >= test_min:
            print(f"🚨 CRITICAL: Val-Test overlap detected!")
            print(f"   Val ends:    {val_max}")
            print(f"   Test starts: {test_min}")
            overlap_issues.append("Val-Test")
        else:
            gap_hours = (test_min - val_max).total_seconds() / 3600
            print(f"✅ No Val-Test overlap (Gap: {gap_hours:.1f} hours)")
            
        return train_data, val_data, test_data, overlap_issues
    
    def check_storm_distribution(self, train_data, val_data, test_data):
        """Check storm distribution across splits using Kp_index"""
        print(f"\n🌪️  STORM DISTRIBUTION ANALYSIS (using Kp_index):")
        
        storm_stats = []
        
        for name, data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
            # Check Kp_index distribution
            kp_values = data['Kp_index'].value_counts().sort_index()
            print(f"\n{name} - Kp_index distribution:")
            for kp_val in sorted(data['Kp_index'].unique()):
                count = (data['Kp_index'] == kp_val).sum()
                pct = count / len(data) * 100
                print(f"   Kp={kp_val}: {count:,} samples ({pct:.1f}%)")
            
            # Storm definition: Kp >= 5 (standard threshold)
            storm_count_5 = (data['Kp_index'] >= 5).sum()
            total_count = len(data)
            storm_rate_5 = storm_count_5 / total_count * 100
            
            # Moderate storm: Kp >= 4
            storm_count_4 = (data['Kp_index'] >= 4).sum()
            storm_rate_4 = storm_count_4 / total_count * 100
            
            print(f"\n{name} STORM SUMMARY:")
            print(f"   Kp>=4: {storm_count_4:,} storms ({storm_rate_4:.1f}%)")
            print(f"   Kp>=5: {storm_count_5:,} storms ({storm_rate_5:.1f}%)")
            
            storm_stats.append({
                'split': name,
                'total_samples': total_count,
                'storms_kp4': storm_count_4,
                'storms_kp5': storm_count_5,
                'rate_kp4': storm_rate_4,
                'rate_kp5': storm_rate_5
            })
        
        return storm_stats
    
    def check_feature_consistency(self):
        """Check if features are consistent across all data"""
        print(f"\n🔧 FEATURE CONSISTENCY CHECK:")
        print(f"Total features: {len(self.df.columns)}")
        print(f"Feature columns: {list(self.df.columns)}")
        
        # Check for missing values
        missing_summary = self.df.isnull().sum()
        problematic_features = missing_summary[missing_summary > 0]
        
        if len(problematic_features) > 0:
            print(f"\n⚠️ Features with missing values:")
            for feature, missing_count in problematic_features.items():
                missing_pct = (missing_count / len(self.df)) * 100
                print(f"  {feature}: {missing_count:,} missing ({missing_pct:.1f}%)")
        else:
            print("✅ No missing values detected")
            
        # Check data types
        print(f"\n📊 Data types:")
        for col, dtype in self.df.dtypes.items():
            print(f"  {col}: {dtype}")

# Run validation with corrected datetime indexing
if __name__ == "__main__":
    # Path to your cleaned data folder
    data_folder = "../../../data/processed/cleaned"
    
    try:
        # Initialize validator
        validator = TemporalSplitValidator(data_folder)
        
        # Step 1: Load and combine all cycles with proper datetime index
        combined_df = validator.load_and_combine_cycles()
        
        # Step 2: Verify temporal splits
        train_data, val_data, test_data, overlap_issues = validator.verify_splits()
        
        # Step 3: Check storm distribution
        storm_stats = validator.check_storm_distribution(train_data, val_data, test_data)
        
        # Step 4: Check feature consistency
        validator.check_feature_consistency()
        
        # Step 5: Final assessment
        print(f"\n🏆 FINAL ASSESSMENT:")
        if overlap_issues:
            print(f"🚨 CRITICAL ISSUES FOUND: {overlap_issues}")
            print("   ⚠️  You MUST fix data leakage before proceeding!")
        else:
            print("✅ Temporal splits are clean - no data leakage detected")
            
        print(f"📊 Ready to proceed with model training")
        
        # Summary statistics
        print(f"\n📈 SUMMARY STATISTICS:")
        for stat in storm_stats:
            print(f"{stat['split']:5}: {stat['total_samples']:,} samples, "
                  f"{stat['storms_kp5']:,} storms (Kp≥5), "
                  f"Rate: {stat['rate_kp5']:.1f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"💡 Check that your data files exist in the correct location:")
        print(f"   Expected: {os.path.abspath(data_folder)}")
