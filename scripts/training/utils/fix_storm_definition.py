import pandas as pd
import numpy as np
import os

def fix_storm_definition_and_save():
    """Fix storm definition and create corrected feature files"""
    
    print("🔧 FIXING STORM DEFINITION...")
    
    data_folder = "../../../data/processed/cleaned"
    output_folder = "../../../data/processed/features"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    cycle_files = [
        'cycle23_cleaned.parquet',
        'cycle24_cleaned.parquet', 
        'cycle25_cleaned.parquet'
    ]
    
    all_dataframes = []
    
    for cycle_file in cycle_files:
        file_path = os.path.join(data_folder, cycle_file)
        if os.path.exists(file_path):
            print(f"📊 Processing {cycle_file}...")
            
            # Load data
            df = pd.read_parquet(file_path)
            df = df.set_index('Datetime')
            
            # CRITICAL FIX: Convert Kp from scaled values to actual values
            df['Kp_actual'] = df['Kp_index'] / 10.0  # Convert Kp*10 back to actual Kp
            
            # Create proper storm definition (Kp >= 5.0)
            df['Storm'] = (df['Kp_actual'] >= 5.0).astype(int)
            
            # Also create moderate storm (Kp >= 4.0) 
            df['Moderate_Storm'] = (df['Kp_actual'] >= 4.0).astype(int)
            
            print(f"   Original Kp range: {df['Kp_index'].min():.1f} to {df['Kp_index'].max():.1f}")
            print(f"   Actual Kp range: {df['Kp_actual'].min():.1f} to {df['Kp_actual'].max():.1f}")
            print(f"   Storm rate (Kp≥5): {df['Storm'].mean()*100:.1f}%")
            print(f"   Moderate storm rate (Kp≥4): {df['Moderate_Storm'].mean()*100:.1f}%")
            
            all_dataframes.append(df)
    
    # Combine all cycles
    combined_df = pd.concat(all_dataframes, axis=0)
    combined_df = combined_df.sort_index()
    
    print(f"\n📈 COMBINED CORRECTED DATASET:")
    print(f"   Total shape: {combined_df.shape}")
    print(f"   Storm rate (Kp≥5): {combined_df['Storm'].mean()*100:.1f}%")
    print(f"   Moderate storm rate (Kp≥4): {combined_df['Moderate_Storm'].mean()*100:.1f}%")
    
    # Save corrected feature file
    output_file = os.path.join(output_folder, 'solaris_x_features_corrected.parquet')
    combined_df.to_parquet(output_file)
    print(f"✅ Saved corrected features to: {output_file}")
    
    # Create train/val/test splits with corrected storm definition
    train_mask = combined_df.index.year <= 2016
    val_mask = (combined_df.index.year >= 2017) & (combined_df.index.year <= 2020)  
    test_mask = combined_df.index.year >= 2021
    
    train_data = combined_df[train_mask]
    val_data = combined_df[val_mask]
    test_data = combined_df[test_mask]
    
    print(f"\n🌪️  CORRECTED STORM DISTRIBUTION:")
    for name, data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        storm_count = data['Storm'].sum()
        total_count = len(data)
        storm_rate = storm_count / total_count * 100
        
        moderate_count = data['Moderate_Storm'].sum()
        moderate_rate = moderate_count / total_count * 100
        
        print(f"{name:5}: {storm_count:,}/{total_count:,} storms (Kp≥5) = {storm_rate:.1f}%")
        print(f"       {moderate_count:,}/{total_count:,} moderate (Kp≥4) = {moderate_rate:.1f}%")
    
    return output_file, combined_df

if __name__ == "__main__":
    output_file, corrected_df = fix_storm_definition_and_save()
    
    print(f"\n✅ CORRECTION COMPLETE!")
    print(f"📁 Corrected file: {output_file}")
    print(f"🎯 Ready to retrain models with correct storm definition")
