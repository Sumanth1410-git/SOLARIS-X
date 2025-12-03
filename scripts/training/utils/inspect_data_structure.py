import pandas as pd
import numpy as np
import os

def inspect_data_files():
    """Inspect the actual structure of your cycle files"""
    data_folder = "../../../data/processed/cleaned"
    
    cycle_files = [
        'cycle23_cleaned.parquet',
        'cycle24_cleaned.parquet', 
        'cycle25_cleaned.parquet'
    ]
    
    print("🔍 INSPECTING DATA STRUCTURE...")
    
    for cycle_file in cycle_files:
        file_path = os.path.join(data_folder, cycle_file)
        if os.path.exists(file_path):
            print(f"\n📊 ANALYZING {cycle_file}:")
            
            # Load and inspect
            df = pd.read_parquet(file_path)
            
            print(f"   Shape: {df.shape}")
            print(f"   Index: {df.index}")
            print(f"   Index type: {type(df.index)}")
            print(f"   Columns: {list(df.columns)}")
            
            # Look for datetime columns
            datetime_candidates = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['time', 'date', 'timestamp']):
                    datetime_candidates.append(col)
                    print(f"   📅 Found datetime column: {col}")
                    print(f"      Sample values: {df[col].head(3).tolist()}")
                    print(f"      Data type: {df[col].dtype}")
            
            if not datetime_candidates:
                print("   ⚠️  NO datetime columns found in column names!")
                print("   🔍 Checking first few columns for datetime-like data:")
                
                # Check first 5 columns for datetime-like patterns
                for col in df.columns[:5]:
                    sample_vals = df[col].head(3).tolist()
                    print(f"      {col}: {sample_vals} (type: {df[col].dtype})")
            
            # Look for storm/target columns
            storm_candidates = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['storm', 'kp', 'target', 'label']):
                    storm_candidates.append(col)
                    
            print(f"   🌪️  Storm candidates: {storm_candidates}")
            
            break  # Just inspect first file for now

if __name__ == "__main__":
    inspect_data_files()
