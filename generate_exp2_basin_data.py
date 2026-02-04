#!/usr/bin/env python3
"""
Generate exp2 basin data from remove_and_insert_training_data.ALL_r30.jsonl:
1. Generate basin_pairs_exp2.jsonl
2. Generate basin_info_exp2.jsonl  
3. Generate distant_basins_exp2.jsonl

This script replicates the workflow of generate_basin_pairs.py and find_distant_basins.py
but for exp2 data with different file naming.
"""

import json
import os
import math
import random
import argparse
from itertools import combinations
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
import heapq

import visualize_basin_cooccurrence_matrix as vcm


def read_exp2_training_data(file_path, max_runs=None):
    """Read exp2 training data file, keep only data from the first max_runs runs (if specified)."""
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
                
                if max_runs is not None:
                    if run_id not in seen_runs:
                        if len(seen_runs) >= max_runs:
                            break
                        seen_runs.add(run_id)
                    
                    if run_id not in seen_runs:
                        continue
                else:
                    # Process all runs
                    if run_id not in seen_runs:
                        seen_runs.add(run_id)
                
                data.append(rec)
            except Exception as e:
                print(f"Error parsing line: {e}")
                continue
    
    return data, seen_runs


def process_single_record_exp2(rec, min_count=10):
    """
    Process a single record from exp2 data.
    Extract basin_distribution and solution_flat from final_solutions.
    """
    basin_dist = rec.get('basin_distribution', {})
    final_solutions = rec.get('final_solutions', [])
    
    if not basin_dist:
        return None
    
    # Build a lookup dict from final_solutions list
    # final_solutions is a list of dicts, each with 'edges_hash' as key
    solution_lookup = {}
    for sol in final_solutions:
        if isinstance(sol, dict):
            basin_hash = sol.get('edges_hash')
            if basin_hash:
                solution_lookup[basin_hash] = sol
    
    basin_counts = {}
    for basin_hash, prob in basin_dist.items():
        count = int(round(prob * 100))  # Convert probability to count
        if count >= min_count:
            solution_info = solution_lookup.get(basin_hash, {})
            solution_flat = solution_info.get('solution_flat', [])
            mean_cost = solution_info.get('cost', None)
            
            basin_counts[basin_hash] = {
                'count': count,
                'probability': prob,
                'solution_flat': solution_flat,
                'mean_cost': mean_cost
            }
    
    if not basin_counts:
        return None
    
    # Sort by count (descending)
    sorted_basins = sorted(basin_counts.items(), key=lambda x: x[1]['count'], reverse=True)
    
    return sorted_basins


def generate_basin_pairs_exp2(data, min_count=10):
    """Generate basin pairs for exp2 data."""
    basin_pairs = []
    basin_info_map = {}
    
    for rec in data:
        processed_basins = process_single_record_exp2(rec, min_count=min_count)
        
        if not processed_basins or len(processed_basins) < 2:
            continue
        
        meta_info = {
            'run_id': rec.get('run_id', ''),
            'trial_id': rec.get('trial_id', ''),
            'global_iter': rec.get('global_iter', ''),
            'local_iter': rec.get('local_iter', '')
        }
        
        n = len(processed_basins)
        K = n * (n - 1) // 2
        
        for i in range(len(processed_basins)):
            for j in range(i + 1, len(processed_basins)):
                basin1_hash, basin1_data = processed_basins[i]
                basin2_hash, basin2_data = processed_basins[j]
                
                anchor_hash = basin1_hash
                anchor_p = basin1_data['probability']
                neighbor_hash = basin2_hash
                neighbor_p = basin2_data['probability']
                
                weight = math.sqrt(anchor_p * neighbor_p) / K
                
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
                
                for basin_hash, basin_data in [(anchor_hash, basin1_data), (neighbor_hash, basin2_data)]:
                    if basin_hash not in basin_info_map:
                        basin_info_map[basin_hash] = {
                            'hash': basin_hash,
                            'cost': basin_data.get('mean_cost', None),
                            'solution_flat': basin_data.get('solution_flat', [])
                        }
    
    return basin_pairs, basin_info_map


def extract_adjacent_pairs_from_solution_flat(solution_flat: List[int]) -> Set[Tuple[int, int]]:
    """Extract adjacent pairs from solution_flat."""
    if not solution_flat or len(solution_flat) < 2:
        return set()
    
    pairs = set()
    for i in range(len(solution_flat)):
        u = solution_flat[i]
        v = solution_flat[(i + 1) % len(solution_flat)]
        if u < v:
            pairs.add((u, v))
        else:
            pairs.add((v, u))
    
    return pairs


def calculate_broken_pairs_distance_fast(anchor_pairs: Set[Tuple[int, int]], 
                                         zero_pairs_set: Set[Tuple[int, int]]) -> int:
    """Calculate broken pairs distance: len(anchor_pairs - zero_pairs_set)."""
    return len(anchor_pairs - zero_pairs_set)


def build_zero_pairs_index(cooccurrence_matrix: Dict[Tuple[str, str], float],
                           all_basins: Set[str]) -> Dict[str, Set[str]]:
    """Build index mapping each basin to its non-zero co-occurrence basins."""
    non_zero_index = defaultdict(set)
    
    for (b1, b2), val in cooccurrence_matrix.items():
        if val != 0 and b1 in all_basins and b2 in all_basins:
            non_zero_index[b1].add(b2)
            non_zero_index[b2].add(b1)
    
    return non_zero_index


def find_zero_pairs_for_anchor_fast(anchor_hash: str,
                                    all_basins: Set[str],
                                    non_zero_index: Dict[str, Set[str]]) -> Set[str]:
    """Find all basins that have zero co-occurrence with the anchor basin."""
    non_zero_basins = non_zero_index.get(anchor_hash, set())
    zero_pairs = all_basins - {anchor_hash} - non_zero_basins
    return zero_pairs


def compute_global_basin_counts_from_training_data_exp2(training_data_file: str, 
                                                         max_runs: Optional[int] = None) -> Dict[str, int]:
    """Compute total count for each basin across all training data."""
    if not os.path.exists(training_data_file):
        return {}
    
    global_counts = defaultdict(int)
    seen_runs = set()
    
    with open(training_data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                run_id = rec.get("run_id")
                
                if max_runs is not None:
                    if run_id not in seen_runs:
                        if len(seen_runs) >= max_runs:
                            break
                        seen_runs.add(run_id)
                    if run_id not in seen_runs:
                        continue
                else:
                    if run_id not in seen_runs:
                        seen_runs.add(run_id)
                
                num_runs = rec.get("num_runs", 100)
                basin_dist = rec.get("basin_distribution", {})
                for basin_hash, prob in basin_dist.items():
                    count = int(round(prob * num_runs))
                    global_counts[basin_hash] += count
            except Exception:
                continue
    
    return dict(global_counts)


def find_distant_basins_exp2(training_data_file: str,
                             basin_info: Dict[str, dict],
                             anchor_basins: Set[str],
                             min_count: int = 10,
                             max_runs: Optional[int] = None,
                             top_k: int = 255,
                             max_zero_pairs_to_process: int = None,
                             min_basin_total_count: int = None,
                             select_by_distance: bool = False) -> List[dict]:
    """Find distant basins for exp2 data."""
    print("  Building co-occurrence matrix...")
    # build_cooccurrence_matrix_from_training_data_fast doesn't support None, so use large number
    effective_max_runs = max_runs if max_runs is not None else 999999
    cooccurrence_matrix = vcm.build_cooccurrence_matrix_from_training_data_fast(
        training_data_file,
        min_count=min_count,
        max_runs=effective_max_runs,
        max_records=None,
        max_basins_per_record=None,
    )
    
    all_basins_in_matrix = set()
    for (b1, b2) in cooccurrence_matrix.keys():
        all_basins_in_matrix.add(b1)
        all_basins_in_matrix.add(b2)
    
    if min_basin_total_count is not None:
        print(f"  Computing global basin counts (filtering: count >= {min_basin_total_count})...")
        global_basin_counts = compute_global_basin_counts_from_training_data_exp2(training_data_file, max_runs=max_runs)
        valid_basins = {h for h, count in global_basin_counts.items() if count >= min_basin_total_count}
        print(f"    Found {len(global_basin_counts)} unique basins in training data")
        print(f"    After filtering (count >= {min_basin_total_count}): {len(valid_basins)} basins")
        all_basins = valid_basins
        print(f"  Using {len(all_basins)} basins from valid_basins (min_basin_total_count filter)")
    else:
        all_basins = all_basins_in_matrix | set(basin_info.keys())
        print(f"  Total basins: {len(all_basins)} (in matrix: {len(all_basins_in_matrix)}, in info: {len(basin_info)})")
    
    print("  Building zero pairs index...")
    non_zero_index = build_zero_pairs_index(cooccurrence_matrix, all_basins)
    
    basin_pairs_cache = {}
    for basin_hash, info in basin_info.items():
        solution_flat = info.get("solution_flat", [])
        if solution_flat:
            basin_pairs_cache[basin_hash] = extract_adjacent_pairs_from_solution_flat(solution_flat)
    print(f"    Cached pairs for {len(basin_pairs_cache)} basins from basin_info")
    
    results = []
    processed = 0
    
    for anchor_hash in anchor_basins:
        if anchor_hash not in basin_info:
            continue
        
        anchor_solution = basin_info[anchor_hash].get("solution_flat", [])
        if not anchor_solution:
            continue
        
        anchor_pairs = basin_pairs_cache.get(anchor_hash)
        if anchor_pairs is None:
            anchor_pairs = extract_adjacent_pairs_from_solution_flat(anchor_solution)
            basin_pairs_cache[anchor_hash] = anchor_pairs
        
        zero_pairs = find_zero_pairs_for_anchor_fast(anchor_hash, all_basins, non_zero_index)
        
        if not zero_pairs:
            continue
        
        if max_zero_pairs_to_process is not None and len(zero_pairs) > max_zero_pairs_to_process:
            zero_pairs_list = list(zero_pairs)
            random.shuffle(zero_pairs_list)
            zero_pairs = set(zero_pairs_list[:max_zero_pairs_to_process])
            print(f"  Anchor {anchor_hash[:20]}...: Randomly sampling {max_zero_pairs_to_process} zero pairs from {len(zero_pairs_list)} total")
        
        if select_by_distance:
            print(f"  Anchor {anchor_hash[:20]}...: Computing distances for {len(zero_pairs)} zero pairs to select top {top_k} by distance...")
            zero_pairs_with_distance = []
            
            for zero_basin_hash in zero_pairs:
                zero_solution = basin_info.get(zero_basin_hash, {}).get("solution_flat", [])
                if not zero_solution:
                    continue
                
                zero_pairs_set = basin_pairs_cache.get(zero_basin_hash)
                if zero_pairs_set is None:
                    zero_pairs_set = extract_adjacent_pairs_from_solution_flat(zero_solution)
                    basin_pairs_cache[zero_basin_hash] = zero_pairs_set
                
                broken_distance = calculate_broken_pairs_distance_fast(anchor_pairs, zero_pairs_set)
                zero_pairs_with_distance.append((broken_distance, zero_basin_hash))
            
            zero_pairs_with_distance.sort(key=lambda x: x[0])
            selected_basins = zero_pairs_with_distance[:top_k]
            
            top_distant_basins = [
                {
                    "basin_hash": h,
                    "broken_pairs_distance": broken_distance
                }
                for broken_distance, h in selected_basins
                if h in basin_info and basin_info[h].get("solution_flat")
            ]
            print(f"  Anchor {anchor_hash[:20]}...: Selected {len(top_distant_basins)} distant basins by distance (from {len(zero_pairs_with_distance)} zero pairs)")
        else:
            print(f"  Anchor {anchor_hash[:20]}...: Randomly selecting {top_k} from {len(zero_pairs)} zero pairs...")
            zero_pairs_list = list(zero_pairs)
            random.shuffle(zero_pairs_list)
            selected_basins = zero_pairs_list[:top_k]
            
            top_distant_basins = []
            for h in selected_basins:
                zero_solution = basin_info.get(h, {}).get("solution_flat", [])
                if not zero_solution:
                    continue
                
                zero_pairs_set = basin_pairs_cache.get(h)
                if zero_pairs_set is None:
                    zero_pairs_set = extract_adjacent_pairs_from_solution_flat(zero_solution)
                    basin_pairs_cache[h] = zero_pairs_set
                
                broken_distance = calculate_broken_pairs_distance_fast(anchor_pairs, zero_pairs_set)
                top_distant_basins.append({
                    "basin_hash": h,
                    "broken_pairs_distance": broken_distance
                })
            print(f"  Anchor {anchor_hash[:20]}...: Randomly selected {len(top_distant_basins)} distant basins")
        
        if top_distant_basins:
            results.append({
                "anchor_basin_hash": anchor_hash,
                "distant_basins": top_distant_basins,
            })
        
        processed += 1
        if processed % 10 == 0:
            print(f"    Processed {processed}/{len(anchor_basins)} anchor basins...")
    
    print(f"  Completed: {len(results)} anchor basins processed")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate exp2 basin data (basin_pairs, basin_info, distant_basins) from remove_and_insert_training_data"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input training data file path (e.g., remove_and_insert_training_data.ALL_r30.jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#0",
        help="Output directory for exp2 files (default: /home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#0)",
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=None,
        help="Maximum number of runs to process (default: None = process all runs)",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=10,
        help="Minimum basin count threshold for basin pairs (default: 10)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=255,
        help="Number of distant basins to select per anchor (default: 255)",
    )
    parser.add_argument(
        "--min_basin_total_count",
        type=int,
        default=None,
        help="Only consider basins with total count >= N across all training data (default: None = include all)",
    )
    parser.add_argument(
        "--select_by_distance",
        action='store_true',
        help="If set, select zero pairs by distance (nearest first); otherwise random sample (default: random)",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"{'=' * 60}")
    print("Generating exp2 basin data")
    print(f"{'=' * 60}")
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Parameters: max_runs={args.max_runs if args.max_runs else 'all'}, min_count={args.min_count}, top_k={args.top_k}")
    if args.min_basin_total_count:
        print(f"  min_basin_total_count={args.min_basin_total_count}")
    print(f"  select_by_distance={args.select_by_distance}")
    print()
    
    # Step 1: Generate basin_pairs_exp2.jsonl and basin_info_exp2.jsonl
    print("Step 1: Generating basin_pairs_exp2.jsonl and basin_info_exp2.jsonl...")
    data, seen_runs = read_exp2_training_data(args.input, max_runs=args.max_runs)
    print(f"  Read {len(data)} records from {len(seen_runs)} runs")
    
    basin_pairs, basin_info_map = generate_basin_pairs_exp2(data, min_count=args.min_count)
    print(f"  Generated {len(basin_pairs)} basin pairs")
    print(f"  Unique basins: {len(basin_info_map)}")
    
    pairs_output = os.path.join(args.output_dir, "basin_pairs_exp2.jsonl")
    info_output = os.path.join(args.output_dir, "basin_info_exp2.jsonl")
    
    with open(pairs_output, 'w', encoding='utf-8') as f:
        for record in basin_pairs:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"  Written basin_pairs_exp2.jsonl: {pairs_output}")
    
    basin_info_list = list(basin_info_map.values())
    with open(info_output, 'w', encoding='utf-8') as f:
        for record in basin_info_list:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"  Written basin_info_exp2.jsonl: {info_output}")
    
    # Step 2: Generate distant_basins_exp2.jsonl
    print("\nStep 2: Generating distant_basins_exp2.jsonl...")
    anchor_basins = set()
    for pair in basin_pairs:
        anchor = pair.get("anchor_basin", {}).get("hash")
        if anchor:
            anchor_basins.add(anchor)
    print(f"  Found {len(anchor_basins)} unique anchor basins")
    
    distant_results = find_distant_basins_exp2(
        args.input,
        basin_info_map,
        anchor_basins,
        min_count=args.min_count,
        max_runs=args.max_runs,
        top_k=args.top_k,
        max_zero_pairs_to_process=None,
        min_basin_total_count=args.min_basin_total_count,
        select_by_distance=args.select_by_distance
    )
    
    distant_output = os.path.join(args.output_dir, "distant_basins_exp2.jsonl")
    with open(distant_output, 'w', encoding='utf-8') as f:
        for record in distant_results:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"  Written distant_basins_exp2.jsonl: {distant_output}")
    print(f"  Generated {len(distant_results)} anchor-distant basin records")
    
    # Step 3: Update basin_info_exp2.jsonl with distant basins
    print("\nStep 3: Updating basin_info_exp2.jsonl with distant basins...")
    distant_basins_set = set()
    for rec in distant_results:
        for db in rec.get("distant_basins", []):
            bh = db.get("basin_hash")
            if bh:
                distant_basins_set.add(bh)
    
    # Load missing basins from training data
    missing_basins = distant_basins_set - set(basin_info_map.keys())
    if missing_basins:
        print(f"  Loading {len(missing_basins)} missing basins from training data...")
        seen_runs2 = set()
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    run_id = rec.get("run_id")
                    if args.max_runs is not None:
                        if run_id not in seen_runs2:
                            if len(seen_runs2) >= args.max_runs:
                                break
                            seen_runs2.add(run_id)
                        if run_id not in seen_runs2:
                            continue
                    else:
                        if run_id not in seen_runs2:
                            seen_runs2.add(run_id)
                    
                    final_solutions = rec.get("final_solutions", [])
                    # Build lookup dict from final_solutions list
                    solution_lookup = {}
                    for sol in final_solutions:
                        if isinstance(sol, dict):
                            basin_hash = sol.get('edges_hash')
                            if basin_hash:
                                solution_lookup[basin_hash] = sol
                    
                    for basin_hash in list(missing_basins):
                        if basin_hash in solution_lookup:
                            solution_info = solution_lookup[basin_hash]
                            basin_info_map[basin_hash] = {
                                "hash": basin_hash,
                                "cost": solution_info.get("cost", None),
                                "solution_flat": solution_info.get("solution_flat", [])
                            }
                            missing_basins.remove(basin_hash)
                            if not missing_basins:
                                break
                except Exception:
                    continue
        
        # Append new basins to basin_info_exp2.jsonl
        if missing_basins:
            print(f"  Warning: {len(missing_basins)} basins still missing")
        newly_added = distant_basins_set & set(basin_info_map.keys()) - set(basin_info_map.keys() - distant_basins_set)
        if newly_added:
            with open(info_output, "a", encoding="utf-8") as f:
                for basin_hash in newly_added:
                    if basin_hash in basin_info_map:
                        f.write(json.dumps(basin_info_map[basin_hash], ensure_ascii=False) + "\n")
            print(f"  Updated basin_info_exp2.jsonl with {len(newly_added)} new basins")
    
    print(f"\n{'=' * 60}")
    print("Exp2 basin data generation completed!")
    print(f"{'=' * 60}")
    print(f"Output files:")
    print(f"  - {pairs_output}")
    print(f"  - {info_output}")
    print(f"  - {distant_output}")


if __name__ == "__main__":
    main()
