#!/usr/bin/env python3
"""
Merge multiple basin statistics Excel files into a single file
"""
import os
import pandas as pd
import glob
from pathlib import Path
from collections import defaultdict


def merge_excel_files(input_dir, output_file):
    """
    Merge all Excel files in the input directory into a single Excel file.
    
    Args:
        input_dir: Directory containing Excel files to merge
        output_file: Output Excel file path
    """
    excel_files = sorted(glob.glob(os.path.join(input_dir, 'basin_statistics_analysis_*.xlsx')))
    
    if not excel_files:
        print(f"No Excel files found in {input_dir}")
        return
    
    print(f"Found {len(excel_files)} Excel files to merge")
    
    # Expected sheet names
    sheet_names = ['Instance Summary', 'Run Summary', 'Trial Summary', 
                   'Basin Frequencies (Instance)', 'Basin Connectivity']
    
    # Collect data from all files
    all_instance_summary = []
    all_run_summary = []
    all_trial_summary = []
    all_basin_frequencies = []
    all_basin_connectivity = []
    
    for excel_file in excel_files:
        print(f"Reading: {os.path.basename(excel_file)}")
        try:
            xlsx = pd.ExcelFile(excel_file)
            
            # Read each sheet
            for sheet_name in sheet_names:
                if sheet_name in xlsx.sheet_names:
                    df = pd.read_excel(xlsx, sheet_name=sheet_name)
                    
                    if sheet_name == 'Instance Summary':
                        all_instance_summary.append(df)
                    elif sheet_name == 'Run Summary':
                        all_run_summary.append(df)
                    elif sheet_name == 'Trial Summary':
                        all_trial_summary.append(df)
                    elif sheet_name == 'Basin Frequencies (Instance)':
                        all_basin_frequencies.append(df)
                    elif sheet_name == 'Basin Connectivity':
                        all_basin_connectivity.append(df)
        except Exception as e:
            print(f"Error reading {excel_file}: {e}")
            continue
    
    # Merge dataframes
    print("\nMerging data...")
    
    # Instance Summary: aggregate (should be same instance, but may have different num_runs)
    if all_instance_summary:
        instance_df = pd.concat(all_instance_summary, ignore_index=True)
        # Aggregate by instance
        instance_merged = instance_df.groupby('instance').agg({
            'num_runs': 'sum',
            'num_unique_basins': lambda x: len(set(x)),  # Count unique basins across all runs
            'total_hits': 'sum',
            'total_global_iters': 'sum',
            'freq_min': 'min',
            'freq_max': 'max',
            'freq_mean': lambda x: (instance_df['total_hits'] * instance_df['freq_mean']).sum() / instance_df['total_hits'].sum() if instance_df['total_hits'].sum() > 0 else 0,
            'freq_std': 'mean',  # Approximate
            'freq_median': 'median',
        }).reset_index()
        print(f"  Instance Summary: {len(instance_merged)} rows")
    else:
        instance_merged = pd.DataFrame()
    
    # Run Summary: concatenate all
    if all_run_summary:
        run_merged = pd.concat(all_run_summary, ignore_index=True)
        print(f"  Run Summary: {len(run_merged)} rows")
    else:
        run_merged = pd.DataFrame()
    
    # Trial Summary: concatenate all
    if all_trial_summary:
        trial_merged = pd.concat(all_trial_summary, ignore_index=True)
        print(f"  Trial Summary: {len(trial_merged)} rows")
    else:
        trial_merged = pd.DataFrame()
    
    # Basin Frequencies: merge and aggregate
    if all_basin_frequencies:
        basin_df = pd.concat(all_basin_frequencies, ignore_index=True)
        # Group by instance and basin_hash, sum frequencies
        basin_merged = basin_df.groupby(['instance', 'basin_edges_hash']).agg({
            'frequency': 'sum',
            'num_global_iters': 'sum',  # Sum num_global_iters across all runs
            'num_trials': 'sum',
            'cost': 'first',  # Should be same for same basin
            'gap_to_hgs': 'first',  # Should be same for same basin
        }).reset_index()
        
        # Recalculate frequency_percent based on total hits
        total_hits = instance_merged['total_hits'].sum() if not instance_merged.empty else basin_merged['frequency'].sum()
        basin_merged['frequency_percent'] = (basin_merged['frequency'] / total_hits * 100).apply(lambda x: f"{x:.4f}%")
        
        # Sort by frequency descending
        basin_merged = basin_merged.sort_values('frequency', ascending=False).reset_index(drop=True)
        print(f"  Basin Frequencies: {len(basin_merged)} unique basins")
    else:
        basin_merged = pd.DataFrame()
    
    # Basin Connectivity: merge and aggregate
    if all_basin_connectivity:
        connectivity_df = pd.concat(all_basin_connectivity, ignore_index=True)
        
        # Convert percentage strings to floats for aggregation
        def parse_prob(x):
            if pd.isna(x):
                return 0.0
            if isinstance(x, str):
                return float(x.replace('%', ''))
            return float(x)
        
        connectivity_df['co_occurrence_probability_float'] = connectivity_df['co_occurrence_probability'].apply(parse_prob)
        connectivity_df['jaccard_similarity_float'] = pd.to_numeric(connectivity_df['jaccard_similarity'], errors='coerce').fillna(0.0)
        
        # Group by basin pairs, aggregate connectivity metrics
        connectivity_merged = connectivity_df.groupby(['instance', 'basin1_hash', 'basin2_hash']).agg({
            'basin1_frequency': 'first',
            'basin1_cost': 'first',
            'basin1_gap_to_hgs': 'first',
            'basin2_frequency': 'first',
            'basin2_cost': 'first',
            'basin2_gap_to_hgs': 'first',
            'co_occurrence_count': 'sum',
            'co_occurrence_probability_float': 'max',  # Take max probability across runs
            'jaccard_similarity_float': 'max',  # Take max jaccard across runs
        }).reset_index()
        
        # Filter: keep pairs where prob >= 0.01 or jaccard >= 0.01
        connectivity_merged = connectivity_merged[
            (connectivity_merged['co_occurrence_probability_float'] >= 0.01) | 
            (connectivity_merged['jaccard_similarity_float'] >= 0.01)
        ]
        
        # Sort by co_occurrence_probability descending
        connectivity_merged = connectivity_merged.sort_values('co_occurrence_probability_float', ascending=False).reset_index(drop=True)
        
        # Limit to top 10000 pairs
        if len(connectivity_merged) > 10000:
            connectivity_merged = connectivity_merged.head(10000)
        
        # Convert probability back to percentage string
        connectivity_merged['co_occurrence_probability'] = connectivity_merged['co_occurrence_probability_float'].apply(lambda x: f"{x:.4f}%")
        connectivity_merged['jaccard_similarity'] = connectivity_merged['jaccard_similarity_float'].apply(lambda x: f"{x:.4f}")
        
        # Drop temporary columns before saving
        connectivity_merged = connectivity_merged.drop(columns=['co_occurrence_probability_float', 'jaccard_similarity_float'], errors='ignore')
        
        print(f"  Basin Connectivity: {len(connectivity_merged)} pairs")
    else:
        connectivity_merged = pd.DataFrame()
    
    # Write merged data to Excel
    print(f"\nWriting merged data to: {output_file}")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        if not instance_merged.empty:
            instance_merged.to_excel(writer, sheet_name='Instance Summary', index=False)
        if not run_merged.empty:
            run_merged.to_excel(writer, sheet_name='Run Summary', index=False)
        if not trial_merged.empty:
            trial_merged.to_excel(writer, sheet_name='Trial Summary', index=False)
        if not basin_merged.empty:
            basin_merged.to_excel(writer, sheet_name='Basin Frequencies (Instance)', index=False)
        if not connectivity_merged.empty:
            connectivity_merged.to_excel(writer, sheet_name='Basin Connectivity', index=False)
    
    print("Merge completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge basin statistics Excel files')
    parser.add_argument('--input_dir', type=str, 
                       default='/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#0/basin_analysis_stats',
                       help='Input directory containing Excel files')
    parser.add_argument('--output', type=str,
                       default=None,
                       help='Output Excel file path (default: merged_basin_statistics.xlsx in input_dir)')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = os.path.join(args.input_dir, 'merged_basin_statistics.xlsx')
    
    merge_excel_files(args.input_dir, args.output)
