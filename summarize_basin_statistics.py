#!/usr/bin/env python3
"""
Summarize basin statistics for all instances:
- Total number of records
- Number of records with 2 basins, 3 basins, etc.
"""

import json
import os
import math
from collections import defaultdict
import pandas as pd


def calculate_n_from_k(k):
    """
    Calculate number of basins (n) from number of combinations K = C(n,2) = n*(n-1)/2
    K = n*(n-1)/2
    n^2 - n - 2K = 0
    n = (1 + sqrt(1 + 8K)) / 2
    """
    if k == 0:
        return 0
    n = (1 + math.sqrt(1 + 8 * k)) / 2
    return int(round(n))


def analyze_instance(instance_index):
    """
    Analyze a single instance
    """
    pairs_file = f"/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#{instance_index}/basin_pairs.jsonl"
    
    if not os.path.exists(pairs_file):
        return None
    
    # Read basin pairs
    pairs = []
    with open(pairs_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
                pairs.append(pair)
            except Exception as e:
                print(f"Error parsing line in instance {instance_index}: {e}")
                continue
    
    if len(pairs) == 0:
        return None
    
    # Group pairs by their source record (using meta_info)
    # Each unique combination of (run_id, trial_id, global_iter, local_iter) represents one original record
    records = defaultdict(lambda: {'K': None, 'pairs': []})
    
    for pair in pairs:
        meta = pair.get('meta_info', {})
        record_key = (
            meta.get('run_id', ''),
            meta.get('trial_id', ''),
            meta.get('global_iter', ''),
            meta.get('local_iter', '')
        )
        records[record_key]['K'] = pair.get('K')
        records[record_key]['pairs'].append(pair)
    
    # Calculate basin counts for each record
    basin_counts = defaultdict(int)  # basin_count -> number of records
    
    for record_key, record_data in records.items():
        K = record_data['K']
        if K is not None:
            n = calculate_n_from_k(K)
            basin_counts[n] += 1
    
    # Calculate total records
    total_records = len(records)
    
    # Calculate total pairs
    total_pairs = len(pairs)
    
    return {
        'instance_index': instance_index,
        'total_records': total_records,
        'total_pairs': total_pairs,
        'basin_counts': dict(basin_counts)
    }


def parse_instance_range(instance_str):
    """
    Parse instance range string into list of indices
    Examples:
        "0-10" -> [0, 1, 2, ..., 10]
        "0,1,2" -> [0, 1, 2]
        "0-5,8,10" -> [0, 1, 2, 3, 4, 5, 8, 10]
    """
    indices = set()
    parts = instance_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range
            start, end = part.split('-')
            start, end = int(start.strip()), int(end.strip())
            indices.update(range(start, end + 1))
        else:
            # Single number
            indices.add(int(part.strip()))
    return sorted(list(indices))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Summarize basin statistics for specified instances')
    parser.add_argument('--instances', type=str, default='0-10',
                       help='Instance indices to process (e.g., "0-10", "0,1,2", "0-5,8,10"). Default: 0-10')
    parser.add_argument('--output', type=str, default='basin_statistics_summary.xlsx',
                       help='Output Excel file path (default: basin_statistics_summary.xlsx)')
    
    args = parser.parse_args()
    
    # Parse instance range
    instance_indices = parse_instance_range(args.instances)
    
    print(f"Analyzing {len(instance_indices)} instances: {instance_indices}")
    
    results = []
    all_basin_counts = set()
    
    for instance_index in instance_indices:
        print(f"Processing instance {instance_index}...")
        result = analyze_instance(instance_index)
        if result:
            results.append(result)
            all_basin_counts.update(result['basin_counts'].keys())
    
    if not results:
        print("No data found!")
        return
    
    # Sort basin counts
    all_basin_counts = sorted(all_basin_counts)
    
    # Create DataFrame
    rows = []
    for result in results:
        row = {
            'instance_index': result['instance_index'],
            'total_records': result['total_records'],
            'total_pairs': result['total_pairs']
        }
        
        # Add counts for each basin number
        for n in all_basin_counts:
            row[f'{n}_basins'] = result['basin_counts'].get(n, 0)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Add summary row
    summary_row = {
        'instance_index': 'TOTAL',
        'total_records': df['total_records'].sum(),
        'total_pairs': df['total_pairs'].sum()
    }
    for n in all_basin_counts:
        summary_row[f'{n}_basins'] = df[f'{n}_basins'].sum()
    
    df_summary = pd.DataFrame([summary_row])
    
    # Append to existing Excel if present: merge by instance_index (new run overwrites rows for same instances)
    output_path = os.path.abspath(args.output)
    print(f"\nWriting to Excel: {output_path}")
    
    if os.path.exists(output_path):
        try:
            existing = pd.read_excel(output_path, sheet_name='By Instance', engine='openpyxl')
            # Keep rows whose instance_index is not in this run's indices (or is 'TOTAL' which we will replace)
            current_indices = set(df['instance_index'].tolist())
            existing = existing[~existing['instance_index'].isin(current_indices)]
            if 'instance_index' in existing.columns and (existing['instance_index'] == 'TOTAL').any():
                existing = existing[existing['instance_index'] != 'TOTAL']
            # Align columns: union of columns, fill missing with 0
            all_cols = sorted(set(existing.columns) | set(df.columns), key=lambda c: (c != 'instance_index', str(c)))
            for c in all_cols:
                if c not in existing.columns:
                    existing[c] = 0
                if c not in df.columns:
                    df[c] = 0
            df = pd.concat([existing[all_cols], df[all_cols]], ignore_index=True)
            # Sort by instance_index (numeric first, then TOTAL if present)
            total_mask = df['instance_index'].astype(str) == 'TOTAL'
            df = pd.concat([df[~total_mask].sort_values('instance_index'), df[total_mask]], ignore_index=True)
            # Recompute summary from merged By Instance (exclude TOTAL if any)
            by_inst = df[df['instance_index'].astype(str) != 'TOTAL']
            summary_row = {
                'instance_index': 'TOTAL',
                'total_records': int(by_inst['total_records'].sum()),
                'total_pairs': int(by_inst['total_pairs'].sum()),
            }
            for c in all_cols:
                if c not in ('instance_index', 'total_records', 'total_pairs') and c in by_inst.columns:
                    summary_row[c] = int(by_inst[c].sum())
            df_summary = pd.DataFrame([summary_row])
            df_summary = df_summary[[c for c in all_cols if c in df_summary.columns]]
            print(f"  Appended/updated instances {instance_indices}; merged with existing file.")
        except Exception as e:
            print(f"  Warning: could not merge with existing Excel ({e}), overwriting.")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='By Instance', index=False)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"Excel file saved to: {output_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total instances: {len(results)}")
    print(f"  Total records: {summary_row['total_records']}")
    print(f"  Total pairs: {summary_row['total_pairs']}")
    print(f"\nBasin count distribution:")
    for n in all_basin_counts:
        count = summary_row[f'{n}_basins']
        if count > 0:
            print(f"  {n} basins: {count} records ({count/summary_row['total_records']*100:.2f}%)")


if __name__ == "__main__":
    main()
