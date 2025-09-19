import pandas as pd
import numpy as np
import os
from pathlib import Path
import gc
import psutil

def monitor_memory():
    """Monitor current RAM usage"""
    memory = psutil.virtual_memory()
    print(f"RAM Usage: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")

def process_omni_file(file_path, output_path, cycle_name):
    """Convert OMNI ASCII to optimized Parquet with chunked processing"""
    
    print(f"\n🛰️ Processing {cycle_name}...")
    monitor_memory()
    
    # Define column names based on OMNI format
    columns = [
        'Year', 'Day', 'Hour', 'IMF_Magnitude', 'IMF_Bz', 'IMF_By', 'IMF_Bx',
        'Proton_Temp', 'Proton_Density', 'Flow_Speed', 'Flow_Pressure', 
        'Kp_index', 'Dst_index', 'AE_index'
    ]
    
    # Process in chunks to prevent memory overflow
    chunk_list = []
    chunk_size = 10000  # 10K rows per chunk
    
    try:
        for chunk in pd.read_csv(
            file_path, 
            delim_whitespace=True, 
            chunksize=chunk_size,
            names=columns,
            dtype=np.float32,  # Use float32 to halve memory usage
            na_values=[999.9, 9999.9, 99999.9]  # OMNI missing value codes
        ):
            
            # Data cleaning and optimization
            chunk = chunk.dropna(subset=['Year', 'Day', 'Hour'])  # Remove invalid timestamps
            
            # Convert to appropriate data types
            chunk['Year'] = chunk['Year'].astype('int16')
            chunk['Day'] = chunk['Day'].astype('int16') 
            chunk['Hour'] = chunk['Hour'].astype('int8')
            
            # Create datetime column
            chunk['Datetime'] = pd.to_datetime(
                chunk['Year'].astype(str) + ' ' + 
                chunk['Day'].astype(str), 
                format='%Y %j'
            ) + pd.to_timedelta(chunk['Hour'], unit='h')
            
            chunk_list.append(chunk)
            
            # Monitor memory every 50k rows
            if len(chunk_list) % 5 == 0:
                monitor_memory()
    
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False
    
    # Combine all chunks
    print(f"🔄 Combining {len(chunk_list)} chunks...")
    df = pd.concat(chunk_list, ignore_index=True)
    
    # Final data validation
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📅 Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
    print(f"🔍 Missing data: {df.isnull().sum().sum()} values")
    
    # Save to Parquet with compression
    df.to_parquet(
        output_path, 
        compression='snappy',  # Fast compression with good ratio
        index=False
    )
    
    # Clear memory
    del df, chunk_list
    gc.collect()
    
    print(f"✅ {cycle_name} converted successfully!")
    monitor_memory()
    return True

def main():
    """Main preprocessing pipeline"""
    print("🚀 SOLARIS-X Data Preprocessing Pipeline Initiated")
    
    # File paths
    base_path = Path("data/raw/omni")
    output_path = Path("data/processed/parquet")
    output_path.mkdir(exist_ok=True)
    
    # File mapping
    files_to_process = [
        ("omni_cycle23_1996_2008.txt", "cycle23.parquet", "Solar Cycle 23"),
        ("omni_cycle24_2009_2019.txt", "cycle24.parquet", "Solar Cycle 24"), 
        ("omni_cycle25_2020_2025.txt", "cycle25.parquet", "Solar Cycle 25")
    ]
    
    successful_conversions = 0
    
    for input_file, output_file, cycle_name in files_to_process:
        input_path = base_path / input_file
        output_full_path = output_path / output_file
        
        if input_path.exists():
            success = process_omni_file(input_path, output_full_path, cycle_name)
            if success:
                successful_conversions += 1
        else:
            print(f"❌ File not found: {input_path}")
    
    print(f"\n🎯 MISSION SUMMARY:")
    print(f"✅ Successfully processed: {successful_conversions}/3 cycles")
    print(f"📁 Output location: {output_path}")
    
    # Display final file sizes
    total_parquet_size = 0
    for _, output_file, _ in files_to_process:
        parquet_path = output_path / output_file
        if parquet_path.exists():
            size_mb = parquet_path.stat().st_size / (1024*1024)
            total_parquet_size += size_mb
            print(f"📦 {output_file}: {size_mb:.1f} MB")
    
    print(f"🚀 Total Parquet size: {total_parquet_size:.1f} MB (Est. 60% compression)")
    
if __name__ == "__main__":
    main()
