#!/usr/bin/env python3
"""
Visualize basin co-occurrence matrix for each instance
- Basin x Basin matrix
- Color: positive weights (red) for pairs where both basins >= 10 counts
- Color: negative weights (black) for pairs where at least one basin < 10 counts
- White for no co-occurrence
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd


def parse_instance_range(instance_str):
    """
    Parse instance range string into list of indices
    """
    indices = set()
    parts = instance_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            start, end = int(start.strip()), int(end.strip())
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part.strip()))
    return sorted(list(indices))


def read_basin_pairs(file_path):
    """
    Read basin pairs from JSONL file
    """
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
                pairs.append(pair)
            except Exception as e:
                print(f"Error parsing line: {e}")
                continue
    return pairs


def build_cooccurrence_matrix_from_training_data(training_data_file, min_count=10, max_runs=10):
    """
    Build basin x basin co-occurrence matrix from training data
    - For each record, check all basin pairs
    - If both basins have count >= min_count: add weight (positive)
    - If at least one basin has count < min_count: add -1 (negative)
    - Final value = sum of positive weights - sum of negative counts
    """
    positive_matrix = defaultdict(float)  # (basin1, basin2) -> positive weight sum
    negative_count = defaultdict(int)  # (basin1, basin2) -> negative count
    
    print("  Reading training data (first {} runs only)...".format(max_runs))
    seen_runs = set()
    records_processed = 0
    
    with open(training_data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                rec = json.loads(line)
                run_id = rec.get('run_id')
                
                # Only process first max_runs runs
                if run_id not in seen_runs:
                    if len(seen_runs) >= max_runs:
                        break
                    seen_runs.add(run_id)
                
                if run_id not in seen_runs:
                    continue
                
                basin_dist = rec.get('basin_distribution', {})
                
                if not basin_dist:
                    continue
                
                # Convert probabilities to counts
                basin_counts = {}
                for basin_hash, prob in basin_dist.items():
                    count = int(round(prob * 100))
                    basin_counts[basin_hash] = count
                
                # Generate all pairs in this record
                basin_list = list(basin_counts.keys())
                n = len(basin_list)
                
                if n < 2:
                    continue
                
                K = n * (n - 1) // 2
                
                # Pre-calculate which basins meet the threshold
                valid_basins = {b for b, count in basin_counts.items() if count >= min_count}
                
                # Generate pairs more efficiently
                for i in range(len(basin_list)):
                    b1 = basin_list[i]
                    count1 = basin_counts[b1]
                    p1 = basin_dist[b1]
                    is_valid1 = b1 in valid_basins
                    
                    for j in range(i + 1, len(basin_list)):
                        b2 = basin_list[j]
                        count2 = basin_counts[b2]
                        p2 = basin_dist[b2]
                        is_valid2 = b2 in valid_basins
                        
                        # Create unordered pair key
                        if b1 < b2:
                            pair_key = (b1, b2)
                        else:
                            pair_key = (b2, b1)
                        
                        # Check if both basins have count >= min_count
                        if is_valid1 and is_valid2:
                            # Positive contribution: add weight
                            # weight = sqrt(p1 * p2) / K
                            weight = (p1 * p2) ** 0.5 / K
                            positive_matrix[pair_key] += weight
                        else:
                            # Negative contribution: -1
                            negative_count[pair_key] += 1
                
                records_processed += 1
                if records_processed % 5000 == 0:
                    print(f"    Processed {records_processed} records...")
            
            except Exception as e:
                print(f"    Error parsing line {line_num + 1}: {e}")
                continue
    
    print(f"  Processed {records_processed} records from {len(seen_runs)} runs")
    
    # Combine: positive weights - negative counts
    # Final value = positive_sum - negative_count
    cooccurrence_matrix = {}
    
    # First, add all positive contributions
    for pair_key, pos_weight in positive_matrix.items():
        cooccurrence_matrix[pair_key] = pos_weight
    
    # Then, subtract negative contributions
    for pair_key, neg_count in negative_count.items():
        if pair_key in cooccurrence_matrix:
            cooccurrence_matrix[pair_key] -= neg_count
        else:
            cooccurrence_matrix[pair_key] = -neg_count
    
    return cooccurrence_matrix


def visualize_cooccurrence_matrix(instance_index, cooccurrence_matrix, output_dir):
    """
    Visualize basin x basin co-occurrence matrix as heatmap
    """
    if not cooccurrence_matrix:
        print(f"Instance {instance_index}: No co-occurrence data, skipping visualization")
        return
    
    # Get all unique basins
    all_basins = set()
    for (b1, b2) in cooccurrence_matrix.keys():
        all_basins.add(b1)
        all_basins.add(b2)
    
    all_basins = sorted(list(all_basins))
    n_basins = len(all_basins)
    
    if n_basins == 0:
        print(f"Instance {instance_index}: No basins found, skipping")
        return
    
    print(f"Instance {instance_index}: Creating {n_basins}x{n_basins} matrix")
    
    # Create matrix
    matrix = np.zeros((n_basins, n_basins))
    
    for (b1, b2), value in cooccurrence_matrix.items():
        i = all_basins.index(b1)
        j = all_basins.index(b2)
        # Matrix is symmetric
        matrix[i, j] = value
        matrix[j, i] = value
    
    # If matrix is too large, sample or truncate
    max_basins = 200  # Limit for visualization
    if n_basins > max_basins:
        print(f"  Matrix too large ({n_basins}x{n_basins}), keeping top {max_basins} basins")
        # Prioritize basins with positive contributions, then by activity
        # Calculate basin activity (sum of absolute values in row)
        basin_activity = np.sum(np.abs(matrix), axis=1)
        # Find basins with positive contributions
        has_positive = np.any(matrix > 0, axis=1)
        positive_indices = np.where(has_positive)[0]
        
        # Combine: prioritize positive basins, then by activity
        if len(positive_indices) > 0:
            # Keep all positive basins, then fill rest with high activity
            remaining_slots = max_basins - len(positive_indices)
            if remaining_slots > 0:
                # Get indices not in positive_indices
                other_indices = np.setdiff1d(np.arange(n_basins), positive_indices)
                if len(other_indices) > 0:
                    other_activities = basin_activity[other_indices]
                    top_other_indices = other_indices[np.argsort(other_activities)[-remaining_slots:]]
                    selected_indices = np.concatenate([positive_indices, top_other_indices])
                else:
                    selected_indices = positive_indices
            else:
                selected_indices = positive_indices[:max_basins]
        else:
            # No positive basins, just use activity
            selected_indices = np.argsort(basin_activity)[-max_basins:]
        
        selected_indices = sorted(selected_indices)
        matrix = matrix[np.ix_(selected_indices, selected_indices)]
        all_basins = [all_basins[i] for i in selected_indices]
        n_basins = len(all_basins)
        print(f"  Reduced to {n_basins}x{n_basins} (including {len(positive_indices)} basins with positive contributions)")
    
    # Create DataFrame
    basin_labels = [b[:8] + '...' if len(b) > 8 else b for b in all_basins]
    df = pd.DataFrame(matrix, index=basin_labels, columns=basin_labels)
    
    # Create custom colormap
    # Positive values -> Red (darker red for larger values)
    # Negative values -> Black (lighter black/gray for less negative)
    # Zero -> White
    
    # Find value ranges
    max_val = np.max(matrix[matrix > 0]) if np.any(matrix > 0) else 0
    min_val = np.min(matrix[matrix < 0]) if np.any(matrix < 0) else 0
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, n_basins * 0.3), max(10, n_basins * 0.3)))
    
    # Create custom colormap
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    
    # Colors: negative (black/gray) -> white -> positive (red)
    colors = ['#000000', '#333333', '#666666', '#999999', '#FFFFFF', '#FFCCCC', '#FF9999', '#FF6666', '#FF0000', '#CC0000']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # Normalize: center at 0, with symmetric range
    if max_val > 0 and min_val < 0:
        vmax = max(abs(max_val), abs(min_val))
        vmin = -vmax
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    elif max_val > 0:
        # Only positive values
        vmax = max_val
        vmin = 0
        # Use simple normalization, but still center at 0 for symmetry
        norm = TwoSlopeNorm(vmin=-vmax*0.1, vcenter=0, vmax=vmax)
    elif min_val < 0:
        # Only negative values
        vmin = min_val
        vmax = abs(min_val)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        # All zeros
        vmin = -1
        vmax = 1
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    # Plot heatmap
    sns.heatmap(df, cmap=cmap, norm=norm, center=0,
                square=True, cbar_kws={'label': 'Co-occurrence Score'},
                xticklabels=False if n_basins > 50 else True,
                yticklabels=False if n_basins > 50 else True,
                ax=ax)
    
    ax.set_title(f'Basin Co-occurrence Matrix (Instance {instance_index})\n'
                 f'Red: positive weights (both >= 10), Black: negative (at least one < 10), White: no co-occurrence\n'
                 f'Matrix size: {n_basins}x{n_basins}',
                 fontsize=12)
    ax.set_xlabel('Basin Hash', fontsize=10)
    ax.set_ylabel('Basin Hash', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'basin_cooccurrence_matrix.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Co-occurrence matrix saved to: {output_path}")
    plt.close()
    
    # Print statistics
    positive_count = np.sum(matrix > 0)
    negative_count = np.sum(matrix < 0)
    zero_count = np.sum(matrix == 0)
    print(f"  Statistics: {positive_count} positive, {negative_count} negative, {zero_count} zero")


def build_cooccurrence_matrix_from_training_data_fast(training_data_file, min_count=10, max_runs=10, max_records=None, max_basins_per_record=None):
    """
    Build co-occurrence matrix from training data (optimized version)
    Only process first max_runs, limit to max_records, and max_basins_per_record for performance
    """
    positive_matrix = defaultdict(float)
    negative_count = defaultdict(int)
    
    if max_records is None:
        max_records_str = "all"
    else:
        max_records_str = str(max_records)
    if max_basins_per_record is None:
        max_basins_str = "all"
    else:
        max_basins_str = str(max_basins_per_record)
    
    print("  Reading training data (first {} runs, max {} records, max {} basins/record)...".format(
        max_runs, max_records_str, max_basins_str))
    seen_runs = set()
    records_processed = 0
    
    with open(training_data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if max_records is not None and records_processed >= max_records:
                break
                
            line = line.strip()
            if not line:
                continue
            
            try:
                rec = json.loads(line)
                run_id = rec.get('run_id')
                
                # Only process first max_runs runs
                if run_id not in seen_runs:
                    if len(seen_runs) >= max_runs:
                        break
                    seen_runs.add(run_id)
                
                if run_id not in seen_runs:
                    continue
                
                basin_dist = rec.get('basin_distribution', {})
                if not basin_dist:
                    continue
                
                # Convert probabilities to counts
                all_basin_counts = {h: int(round(p * 100)) for h, p in basin_dist.items()}
                
                # Limit number of basins per record for performance (if specified)
                if max_basins_per_record is not None and len(all_basin_counts) > max_basins_per_record:
                    # Sort by count and take top N
                    sorted_basins = sorted(all_basin_counts.items(), key=lambda x: x[1], reverse=True)
                    all_basin_counts = dict(sorted_basins[:max_basins_per_record])
                    # Update basin_dist accordingly
                    basin_dist = {h: basin_dist[h] for h in all_basin_counts.keys()}
                
                # Separate basins by threshold
                valid_basins = {h: c for h, c in all_basin_counts.items() if c >= min_count}
                invalid_basins = {h: c for h, c in all_basin_counts.items() if c < min_count}
                
                valid_basin_list = list(valid_basins.keys())
                all_basin_list = list(all_basin_counts.keys())
                n_valid = len(valid_basin_list)
                
                # Process all pairs in this record
                # For each pair, check if both basins meet threshold in THIS record
                for i in range(len(all_basin_list)):
                    b1 = all_basin_list[i]
                    count1 = all_basin_counts[b1]
                    p1 = basin_dist[b1]
                    
                    for j in range(i + 1, len(all_basin_list)):
                        b2 = all_basin_list[j]
                        count2 = all_basin_counts[b2]
                        p2 = basin_dist[b2]
                        
                        if b1 < b2:
                            pair_key = (b1, b2)
                        else:
                            pair_key = (b2, b1)
                        
                        # Check if both basins meet threshold in this record
                        if count1 >= min_count and count2 >= min_count:
                            # Positive contribution: calculate weight
                            # Need to know K for this record (number of valid basin pairs)
                            n_valid = len(valid_basin_list)
                            if n_valid >= 2:
                                K = n_valid * (n_valid - 1) // 2
                                weight = (p1 * p2) ** 0.5 / K
                                positive_matrix[pair_key] += weight
                        else:
                            # Negative contribution: at least one basin < min_count
                            negative_count[pair_key] += 1
                
                records_processed += 1
                if records_processed % 500 == 0:
                    print(f"    Processed {records_processed} records...")
            
            except Exception as e:
                print(f"    Error parsing line {line_num + 1}: {e}")
                continue
    
    print(f"  Processed {records_processed} records from {len(seen_runs)} runs")
    
    # Combine: positive contributions first, then (optionally) negative-only pairs
    # Logic (per user):
    # - If a pair has at least one \"positive\" record (both basins >= min_count), its final value
    #   is strictly the sum of those positive weights ( > 0 ), regardless of any \"bad\" records.
    # - Only pairs that never have a positive record can become negative, by accumulating -1.
    cooccurrence_matrix = {}
    
    # First, add all positive contributions (these pairs are always >= 0)
    for pair_key, pos_weight in positive_matrix.items():
        if pos_weight > 0:
            cooccurrence_matrix[pair_key] = pos_weight
    
    # Then, for pairs that never had positive contributions, use only negative counts
    for pair_key, neg_count in negative_count.items():
        if pair_key not in cooccurrence_matrix:
            cooccurrence_matrix[pair_key] = -neg_count
    
    return cooccurrence_matrix


def process_instance(instance_index, min_count=10, max_runs=10):
    """
    Process a single instance
    """
    training_data_file = f"/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#{instance_index}/training_data.jsonl"
    output_dir = f"/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#{instance_index}"
    
    if not os.path.exists(training_data_file):
        print(f"Instance {instance_index}: Training data file not found, skipping...")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing Instance {instance_index}")
    print(f"{'='*60}")
    
    print("Building co-occurrence matrix from training data...")
    # Limit to 1000 records and 50 basins per record for performance
    cooccurrence_matrix = build_cooccurrence_matrix_from_training_data_fast(
        training_data_file, min_count=min_count, max_runs=max_runs, max_records=1000, max_basins_per_record=50)
    print(f"Matrix contains {len(cooccurrence_matrix)} basin pairs")
    
    # Print statistics
    positive_pairs = sum(1 for v in cooccurrence_matrix.values() if v > 0)
    negative_pairs = sum(1 for v in cooccurrence_matrix.values() if v < 0)
    zero_pairs = sum(1 for v in cooccurrence_matrix.values() if v == 0)
    print(f"  Positive pairs: {positive_pairs}, Negative pairs: {negative_pairs}, Zero pairs: {zero_pairs}")
    
    print("Visualizing co-occurrence matrix...")
    visualize_cooccurrence_matrix(instance_index, cooccurrence_matrix, output_dir)
    
    print(f"Instance {instance_index} done!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize basin co-occurrence matrix')
    parser.add_argument('--instances', type=str, default='0-10',
                       help='Instance indices to process (e.g., "0-10", "0,1,2"). Default: 0-10')
    parser.add_argument('--min_count', type=int, default=10,
                       help='Minimum basin count threshold (default: 10)')
    parser.add_argument('--max_runs', type=int, default=10,
                       help='Maximum number of runs to process per instance (default: 10)')
    
    args = parser.parse_args()
    
    # Parse instance range
    instance_indices = parse_instance_range(args.instances)
    
    print(f"Processing {len(instance_indices)} instances: {instance_indices}")
    print(f"Min count threshold: {args.min_count}")
    print(f"Max runs per instance: {args.max_runs}")
    
    for instance_index in instance_indices:
        try:
            process_instance(instance_index, min_count=args.min_count, max_runs=args.max_runs)
        except Exception as e:
            print(f"Error processing instance {instance_index}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("All instances processed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
