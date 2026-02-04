#!/usr/bin/env python3
"""
Generate basin pair data from training data:
- For each record, create all unordered pairs of basins (C(n,2))
- Each pair includes anchor basin (higher frequency) and neighbor basin (lower frequency)
- Calculate weight: sqrt(p1*p2)/K where K is the number of combinations
- Generate two JSONL files: basin pairs and basin info mapping
"""

import json
import os
import math
from itertools import combinations
from collections import defaultdict


def read_training_data(file_path, max_runs=10):
    """
    Read training_data.jsonl file, keep only data from the first max_runs runs
    """
    data = []
    seen_runs = set()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                rec = json.loads(line)
                run_id = rec.get('run_id')
                
                # Keep only the first max_runs different runs
                if run_id not in seen_runs:
                    if len(seen_runs) >= max_runs:
                        break
                    seen_runs.add(run_id)
                
                # If this run_id is in the first max_runs, keep it
                if run_id in seen_runs:
                    data.append(rec)
            except Exception as e:
                print(f"Error parsing line: {e}")
                continue
    
    return data, seen_runs


def process_single_record(rec, min_count=10):
    """
    Process a single record:
    1. Convert probabilities in basin_distribution to counts (probability * 100)
    2. Filter out basins with count < min_count
    3. Sort by count (descending)
    Returns cleaned basin information
    """
    basin_dist = rec.get('basin_distribution', {})
    basin_features = rec.get('basin_features', {})
    
    if not basin_dist:
        return None
    
    # Convert probabilities to counts (assuming 100 total runs)
    basin_counts = {}
    for basin_hash, prob in basin_dist.items():
        count = int(round(prob * 100))  # Convert probability to count
        if count >= min_count:
            basin_counts[basin_hash] = {
                'count': count,
                'probability': prob,
                'features': basin_features.get(basin_hash, {})
            }
    
    if not basin_counts:
        return None
    
    # Sort by count (descending)
    sorted_basins = sorted(basin_counts.items(), key=lambda x: x[1]['count'], reverse=True)
    
    return sorted_basins


def generate_basin_pairs(data, min_count=10):
    """
    Generate basin pairs for all records
    Returns:
    - basin_pairs: list of pair records
    - basin_info_map: dict mapping basin_hash to {cost, solution_flat}
    """
    basin_pairs = []
    basin_info_map = {}
    
    for rec in data:
        processed_basins = process_single_record(rec, min_count=min_count)
        
        if not processed_basins or len(processed_basins) < 2:
            continue  # Skip records with less than 2 basins
        
        # Get meta information
        meta_info = {
            'run_id': rec.get('run_id', ''),
            'trial_id': rec.get('trial_id', ''),
            'global_iter': rec.get('global_iter', ''),
            'local_iter': rec.get('local_iter', '')
        }
        
        # Calculate K: number of combinations C(n,2) = n*(n-1)/2
        n = len(processed_basins)
        K = n * (n - 1) // 2
        
        # Generate all unordered pairs
        for i in range(len(processed_basins)):
            for j in range(i + 1, len(processed_basins)):
                basin1_hash, basin1_data = processed_basins[i]
                basin2_hash, basin2_data = processed_basins[j]
                
                # Anchor basin (higher frequency), neighbor basin (lower frequency)
                # Since sorted by count descending, basin1 has higher frequency
                anchor_hash = basin1_hash
                anchor_p = basin1_data['probability']
                neighbor_hash = basin2_hash
                neighbor_p = basin2_data['probability']
                
                # Calculate weight: sqrt(p1*p2)/K
                weight = math.sqrt(anchor_p * neighbor_p) / K
                
                # Create pair record
                pair_record = {
                    'anchor_basin': {
                        'hash': anchor_hash,
                        'probability': anchor_p
                    },
                    'neighbor_basin': {
                        'hash': neighbor_hash,
                        'probability': neighbor_p
                    },
                    'K': K,
                    'weight': weight,
                    'meta_info': meta_info
                }
                
                basin_pairs.append(pair_record)
                
                # Store basin info (only store once per basin)
                for basin_hash, basin_data in [(anchor_hash, basin1_data), (neighbor_hash, basin2_data)]:
                    if basin_hash not in basin_info_map:
                        features = basin_data['features']
                        basin_info_map[basin_hash] = {
                            'hash': basin_hash,
                            'cost': features.get('mean_cost', None),
                            'solution_flat': features.get('solution_flat', None)
                        }
    
    return basin_pairs, basin_info_map


def write_jsonl(data, output_path):
    """
    Write data to JSONL file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def process_instance(instance_index, max_runs=10, min_count=10):
    """
    Process a single instance
    """
    file_path = f"/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#{instance_index}/training_data.jsonl"
    output_dir = f"/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#{instance_index}"
    
    if not os.path.exists(file_path):
        print(f"Instance {instance_index}: File not found: {file_path}, skipping...")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing Instance {instance_index}")
    print(f"{'='*60}")
    print(f"Reading file: {file_path}")
    print(f"Keeping only data from the first {max_runs} runs")
    print(f"Filtering basins with count < {min_count}")
    
    # Read data
    data, seen_runs = read_training_data(file_path, max_runs=max_runs)
    print(f"Read {len(data)} records from {len(seen_runs)} runs")
    print(f"Runs: {sorted(seen_runs)}")
    
    # Generate basin pairs
    print(f"\nGenerating basin pairs...")
    basin_pairs, basin_info_map = generate_basin_pairs(data, min_count=min_count)
    
    print(f"Generated {len(basin_pairs)} basin pairs")
    print(f"Unique basins: {len(basin_info_map)}")
    
    # Write basin pairs JSONL
    pairs_output_path = os.path.join(output_dir, "basin_pairs.jsonl")
    print(f"\nWriting basin pairs to: {pairs_output_path}")
    write_jsonl(basin_pairs, pairs_output_path)
    
    # Write basin info JSONL
    basin_info_list = list(basin_info_map.values())
    basin_info_output_path = os.path.join(output_dir, "basin_info.jsonl")
    print(f"Writing basin info to: {basin_info_output_path}")
    write_jsonl(basin_info_list, basin_info_output_path)
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Total basin pairs: {len(basin_pairs)}")
    print(f"  Unique basins: {len(basin_info_map)}")
    
    # Calculate weight statistics
    if basin_pairs:
        weights = [p['weight'] for p in basin_pairs]
        print(f"  Weight statistics:")
        print(f"    Min: {min(weights):.6f}")
        print(f"    Max: {max(weights):.6f}")
        print(f"    Mean: {sum(weights)/len(weights):.6f}")
    
    print(f"Instance {instance_index} done!")


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
    
    parser = argparse.ArgumentParser(description='Generate basin pairs for specified instances')
    parser.add_argument('--instances', type=str, default='0-10',
                       help='Instance indices to process (e.g., "0-10", "0,1,2", "0-5,8,10"). Default: 0-10')
    parser.add_argument('--max_runs', type=int, default=10,
                       help='Maximum number of runs to process per instance (default: 10)')
    parser.add_argument('--min_count', type=int, default=10,
                       help='Minimum basin count threshold (default: 10)')
    
    args = parser.parse_args()
    
    # Parse instance range
    instance_indices = parse_instance_range(args.instances)
    
    print(f"Processing {len(instance_indices)} instances: {instance_indices}")
    print(f"Max runs: {args.max_runs}, Min count: {args.min_count}")
    
    for instance_index in instance_indices:
        try:
            process_instance(instance_index, max_runs=args.max_runs, min_count=args.min_count)
        except Exception as e:
            print(f"Error processing instance {instance_index}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("All instances processed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
