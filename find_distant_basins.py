#!/usr/bin/env python3
"""
Find distant basins for each anchor basin:
- Read anchor basins from basin_pairs.jsonl
- For each anchor basin, find its zero pairs (basins with no co-occurrence)
- Calculate broken pairs distance between anchor and each zero pair basin
- Select top 255 basins with smallest broken pairs distance
- Ensure all selected basins exist in basin_info.jsonl
"""

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
import argparse
import heapq

import visualize_basin_cooccurrence_matrix as vcm


def parse_instance_range(instance_str: str):
    """Parse instance range string into list of indices."""
    indices = set()
    parts = instance_str.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-')
            start, end = int(start.strip()), int(end.strip())
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part.strip()))
    return sorted(indices)


def load_basin_info(instance_index: int) -> Dict[str, dict]:
    """Load basin info from basin_info.jsonl, return dict mapping hash to {cost, solution_flat}."""
    info_file = f"/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#{instance_index}/basin_info.jsonl"
    basin_info = {}
    if not os.path.exists(info_file):
        print(f"Warning: basin_info.jsonl not found for instance {instance_index}")
        return basin_info
    
    with open(info_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                h = rec.get("hash")
                if h:
                    basin_info[h] = {
                        "cost": rec.get("cost"),
                        "solution_flat": rec.get("solution_flat", [])
                    }
            except Exception as e:
                print(f"Error parsing basin_info line: {e}")
                continue
    return basin_info


def load_solution_from_training_data(instance_index: int, basin_hash: str, max_runs: int = 10) -> List[int]:
    """
    Load solution_flat for a single basin from training_data.jsonl.
    This is used for basins that are not in basin_info.jsonl but appear in co-occurrence matrix.
    Uses a simple linear search through the file (could be optimized with indexing, but kept simple for now).
    """
    training_data_file = (
        f"/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#{instance_index}/training_data.jsonl"
    )
    if not os.path.exists(training_data_file):
        return []
    
    seen_runs = set()
    
    with open(training_data_file, "r", encoding="utf-8") as f:
        for line in f:
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
                
                basin_features = rec.get("basin_features", {})
                if basin_hash in basin_features:
                    solution_flat = basin_features[basin_hash].get("solution_flat", [])
                    if solution_flat:
                        return solution_flat
            except Exception:
                continue
    
    return []


def load_anchor_basins(instance_index: int) -> Set[str]:
    """Load all anchor basins from basin_pairs.jsonl."""
    pairs_file = f"/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#{instance_index}/basin_pairs.jsonl"
    anchor_basins = set()
    
    if not os.path.exists(pairs_file):
        print(f"Warning: basin_pairs.jsonl not found for instance {instance_index}")
        return anchor_basins
    
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                anchor = rec.get("anchor_basin", {}).get("hash")
                if anchor:
                    anchor_basins.add(anchor)
            except Exception as e:
                print(f"Error parsing basin_pairs line: {e}")
                continue
    
    return anchor_basins


def extract_adjacent_pairs_from_solution_flat(solution_flat: List[int]) -> Set[Tuple[int, int]]:
    """
    Extract all adjacent node pairs from solution_flat.
    CVRP is undirected, so normalize pairs to (min, max) form.
    Optimized version using set comprehension.
    
    Args:
        solution_flat: List of node IDs in the solution
        
    Returns:
        Set of normalized adjacent pairs (min, max)
    """
    if not solution_flat or len(solution_flat) < 2:
        return set()
    
    # Use set comprehension for better performance
    return {(min(solution_flat[i], solution_flat[i + 1]), 
             max(solution_flat[i], solution_flat[i + 1]))
            for i in range(len(solution_flat) - 1)}


def calculate_broken_pairs_distance_fast(original_pairs: Set[Tuple[int, int]], 
                                         perturbed_pairs: Set[Tuple[int, int]]) -> int:
    """
    Calculate Broken Pairs Distance (BPD) - number of broken pairs.
    Optimized version that takes pre-computed pairs sets.
    
    Args:
        original_pairs: Set of adjacent pairs from original solution
        perturbed_pairs: Set of adjacent pairs from perturbed solution
        
    Returns: broken_pairs_count
    """
    # Broken pairs: pairs that are adjacent in original but not in perturbed
    return len(original_pairs - perturbed_pairs)


def build_zero_pairs_index(cooccurrence_matrix: Dict[Tuple[str, str], float],
                           all_basins: Set[str]) -> Dict[str, Set[str]]:
    """
    Build an index mapping each basin to its non-zero co-occurrence basins.
    Only includes basins that are in all_basins.
    
    Returns:
        Dict mapping basin_hash -> set of basins with non-zero co-occurrence (subset of all_basins)
    """
    non_zero_index = defaultdict(set)
    
    for (b1, b2), val in cooccurrence_matrix.items():
        if val != 0 and b1 in all_basins and b2 in all_basins:
            non_zero_index[b1].add(b2)
            non_zero_index[b2].add(b1)
    
    return non_zero_index


def find_zero_pairs_for_anchor_fast(anchor_hash: str,
                                    all_basins: Set[str],
                                    non_zero_index: Dict[str, Set[str]]) -> Set[str]:
    """
    Find all basins that have zero co-occurrence with the anchor basin.
    Optimized version using pre-built index.
    """
    # Basins with non-zero co-occurrence with anchor
    non_zero_basins = non_zero_index.get(anchor_hash, set())
    
    # Zero pairs = all basins except anchor and non-zero basins
    zero_pairs = all_basins - {anchor_hash} - non_zero_basins
    
    return zero_pairs


def collect_basins_from_pairs_file(instance_index: int) -> Set[str]:
    """Collect all basin hashes from basin_pairs.jsonl (both anchor and neighbor)."""
    pairs_file = f"/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#{instance_index}/basin_pairs.jsonl"
    basins = set()
    
    if not os.path.exists(pairs_file):
        return basins
    
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                anchor = rec.get("anchor_basin", {}).get("hash")
                neighbor = rec.get("neighbor_basin", {}).get("hash")
                if anchor:
                    basins.add(anchor)
                if neighbor:
                    basins.add(neighbor)
            except Exception:
                continue
    
    return basins


def collect_basins_from_distant_file(instance_index: int, distant_file: str) -> Set[str]:
    """Collect all basin hashes from distant_basins.jsonl (anchor and all distant basins)."""
    basins = set()
    
    if not os.path.exists(distant_file):
        return basins
    
    with open(distant_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                anchor = rec.get("anchor_basin_hash")
                if anchor:
                    basins.add(anchor)
                distant_basins = rec.get("distant_basins", [])
                for db in distant_basins:
                    if isinstance(db, dict):
                        basin_hash = db.get("basin_hash")
                    else:
                        basin_hash = db  # Backward compatibility
                    if basin_hash:
                        basins.add(basin_hash)
            except Exception:
                continue
    
    return basins


def update_basin_info_from_training_data(instance_index: int, 
                                         missing_basins: Set[str],
                                         basin_info: Dict[str, dict]) -> Dict[str, dict]:
    """
    Try to load missing basin info from training_data.jsonl.
    This is a fallback if basin_info.jsonl doesn't have all basins.
    """
    training_data_file = (
        f"/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#{instance_index}/training_data.jsonl"
    )
    if not os.path.exists(training_data_file):
        return basin_info
    
    print(f"  Loading missing basin info from training_data.jsonl...")
    loaded_count = 0
    
    with open(training_data_file, "r", encoding="utf-8") as f:
        for line in f:
            if not missing_basins:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                basin_features = rec.get("basin_features", {})
                for basin_hash, features in basin_features.items():
                    if basin_hash in missing_basins and basin_hash not in basin_info:
                        basin_info[basin_hash] = {
                            "cost": features.get("mean_cost"),
                            "solution_flat": features.get("solution_flat", [])
                        }
                        missing_basins.remove(basin_hash)
                        loaded_count += 1
            except Exception:
                continue
    
    print(f"    Loaded {loaded_count} missing basins from training_data.jsonl")
    return basin_info


def ensure_basin_info_complete(instance_index: int, distant_file: str):
    """
    Ensure all basins in basin_pairs.jsonl and distant_basins.jsonl exist in basin_info.jsonl.
    If missing, try to load from training_data.jsonl and update basin_info.jsonl.
    """
    info_file = f"/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#{instance_index}/basin_info.jsonl"
    
    # Collect all required basins
    required_basins = set()
    required_basins.update(collect_basins_from_pairs_file(instance_index))
    required_basins.update(collect_basins_from_distant_file(instance_index, distant_file))
    
    if not required_basins:
        return
    
    # Load existing basin_info
    basin_info = load_basin_info(instance_index)
    existing_basins = set(basin_info.keys())
    missing_basins = required_basins - existing_basins
    
    if not missing_basins:
        print(f"  All required basins ({len(required_basins)}) exist in basin_info.jsonl")
        return
    
    print(f"  Missing {len(missing_basins)} basins in basin_info.jsonl, trying to load from training_data...")
    
    # Save original missing_basins before it gets modified
    original_missing_basins = missing_basins.copy()
    
    # Try to load missing basins from training_data.jsonl
    basin_info = update_basin_info_from_training_data(instance_index, missing_basins, basin_info)
    
    # Calculate newly loaded basins: basins that are now in basin_info but weren't before
    newly_loaded_basins = set(basin_info.keys()) - existing_basins
    
    # Update basin_info.jsonl with newly loaded basins
    still_missing = required_basins - set(basin_info.keys())
    if still_missing:
        print(f"  Warning: {len(still_missing)} basins still missing (no solution_flat available)")
    
    # Append new basins to basin_info.jsonl
    if newly_loaded_basins:
        with open(info_file, "a", encoding="utf-8") as f:
            for basin_hash in newly_loaded_basins:
                if basin_hash in basin_info:
                    rec = {
                        "hash": basin_hash,
                        "cost": basin_info[basin_hash].get("cost"),
                        "solution_flat": basin_info[basin_hash].get("solution_flat", [])
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  Updated basin_info.jsonl with {len(newly_loaded_basins)} new basins")


def compute_global_basin_counts_from_training_data(instance_index: int, max_runs: int = 10) -> Dict[str, int]:
    """
    Compute total count for each basin across all training data (first max_runs runs).
    Count is accumulated from basin_distribution (probability * num_runs) across all records.
    
    Returns:
        Dict mapping basin_hash -> total_count
    """
    training_data_file = (
        f"/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#{instance_index}/training_data.jsonl"
    )
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
                if run_id not in seen_runs:
                    if len(seen_runs) >= max_runs:
                        break
                    seen_runs.add(run_id)
                if run_id not in seen_runs:
                    continue
                
                # Get num_runs for this record (default to 100 if not present for backward compatibility)
                num_runs = rec.get("num_runs", 100)
                
                # Get basin_distribution (probability -> count)
                basin_dist = rec.get("basin_distribution", {})
                for basin_hash, prob in basin_dist.items():
                    # Convert probability to count: count = probability * num_runs
                    count = int(round(prob * num_runs))
                    global_counts[basin_hash] += count
            except Exception:
                continue
    
    return dict(global_counts)


def find_distant_basins_for_instance(instance_index: int,
                                      min_count: int = 10,
                                      max_runs: int = 10,
                                      max_records: int = None,
                                      max_basins_per_record: int = None,
                                      top_k: int = 255,
                                      max_zero_pairs_to_process: int = None,
                                      min_basin_total_count: int = None,
                                      select_by_distance: bool = False,
                                      solution_lookup: Dict[str, List[int]] = None) -> Tuple[List[dict], str]:
    """
    Find distant basins for each anchor basin in an instance.

    max_zero_pairs_to_process: if None, process all zero pairs; if int, randomly
        sample that many when zero pairs exceed it (for performance).

    Returns list of records, each containing:
    - instance_index
    - anchor_basin_hash
    - distant_basin_hashes: list of top_k basin hashes with smallest broken pairs distance
    """
    print(f"\n{'=' * 60}")
    print(f"Processing Instance {instance_index}")
    print(f"{'=' * 60}")
    
    # Load basin info
    print("  Loading basin_info.jsonl...")
    basin_info = load_basin_info(instance_index)
    print(f"    Loaded {len(basin_info)} basins from basin_info.jsonl")
    
    # Load anchor basins
    print("  Loading anchor basins from basin_pairs.jsonl...")
    anchor_basins = load_anchor_basins(instance_index)
    print(f"    Found {len(anchor_basins)} unique anchor basins")
    
    if not anchor_basins:
        print(f"  No anchor basins found, skipping instance {instance_index}")
        instance_dir = f"/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#{instance_index}"
        output_file = os.path.join(instance_dir, "distant_basins.jsonl")
        return [], output_file
    
    # Build co-occurrence matrix to identify zero pairs
    training_data_file = (
        f"/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#{instance_index}/training_data.jsonl"
    )
    if not os.path.exists(training_data_file):
        print(f"  training_data.jsonl not found, skipping instance {instance_index}")
        instance_dir = f"/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#{instance_index}"
        output_file = os.path.join(instance_dir, "distant_basins.jsonl")
        return [], output_file
    
    print("  Building co-occurrence matrix...")
    cooccurrence_matrix = vcm.build_cooccurrence_matrix_from_training_data_fast(
        training_data_file,
        min_count=min_count,
        max_runs=max_runs,
        max_records=max_records,
        max_basins_per_record=max_basins_per_record,
    )
    
    # Get all basins in the co-occurrence matrix
    all_basins_in_matrix = set()
    for (b1, b2) in cooccurrence_matrix.keys():
        all_basins_in_matrix.add(b1)
        all_basins_in_matrix.add(b2)
    
    # Filter basins by total count if min_basin_total_count is specified
    valid_basins = None
    if min_basin_total_count is not None:
        print(f"  Computing global basin counts (filtering: count >= {min_basin_total_count})...")
        global_basin_counts = compute_global_basin_counts_from_training_data(instance_index, max_runs=max_runs)
        valid_basins = {h for h, count in global_basin_counts.items() if count >= min_basin_total_count}
        print(f"    Found {len(global_basin_counts)} unique basins in training data")
        print(f"    After filtering (count >= {min_basin_total_count}): {len(valid_basins)} basins")
        # If min_basin_total_count is set, use valid_basins as all_basins (includes all basins from training data)
        # Otherwise, use basins from co-occurrence matrix and basin_info
        all_basins = valid_basins
        print(f"  Using {len(all_basins)} basins from valid_basins (min_basin_total_count filter)")
    else:
        # Also include all basins from basin_info that might not be in the matrix
        all_basins = all_basins_in_matrix | set(basin_info.keys())
        print(f"  Total basins: {len(all_basins)} (in matrix: {len(all_basins_in_matrix)}, in info: {len(basin_info)})")
    
    # Build zero pairs index for fast lookup
    print("  Building zero pairs index...")
    non_zero_index = build_zero_pairs_index(cooccurrence_matrix, all_basins)
    
    # Pre-compute pairs sets for all basins in basin_info (cache for performance)
    print("  Pre-computing adjacent pairs for all basins in basin_info...")
    basin_pairs_cache = {}
    for basin_hash, info in basin_info.items():
        solution_flat = info.get("solution_flat", [])
        if solution_flat:
            basin_pairs_cache[basin_hash] = extract_adjacent_pairs_from_solution_flat(solution_flat)
    print(f"    Cached pairs for {len(basin_pairs_cache)} basins from basin_info")
    
    # Pre-load solution_flat for all basins in training_data (first max_runs runs) for O(1) lookup.
    # This avoids repeated full-file scans when resolving zero-pair solutions.
    solution_lookup = solution_lookup if solution_lookup is not None else {}
    if not solution_lookup:
        print("  Pre-loading solution_flat from training_data.jsonl (first max_runs runs)...")
        seen_runs = set()
        with open(training_data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    run_id = rec.get("run_id")
                    if run_id not in seen_runs:
                        if len(seen_runs) >= max_runs:
                            break
                        seen_runs.add(run_id)
                    if run_id not in seen_runs:
                        continue
                    for bh, feats in rec.get("basin_features", {}).items():
                        sol = feats.get("solution_flat", [])
                        if sol and bh not in solution_lookup:
                            solution_lookup[bh] = sol
                except Exception:
                    continue
        print(f"    Pre-loaded {len(solution_lookup)} solution_flat entries from training_data")
    
    # Process each anchor basin
    results = []
    processed = 0
    
    for anchor_hash in anchor_basins:
        if anchor_hash not in basin_info:
            print(f"  Warning: anchor basin {anchor_hash[:20]}... not found in basin_info.jsonl, skipping")
            continue
        
        anchor_solution = basin_info[anchor_hash].get("solution_flat", [])
        if not anchor_solution:
            print(f"  Warning: anchor basin {anchor_hash[:20]}... has no solution_flat, skipping")
            continue
        
        # Get cached anchor pairs
        anchor_pairs = basin_pairs_cache.get(anchor_hash)
        if anchor_pairs is None:
            anchor_pairs = extract_adjacent_pairs_from_solution_flat(anchor_solution)
            basin_pairs_cache[anchor_hash] = anchor_pairs
        
        # Find zero pairs for this anchor
        zero_pairs = find_zero_pairs_for_anchor_fast(anchor_hash, all_basins, non_zero_index)
        
        if not zero_pairs:
            continue
        
        # Sample zero pairs if too many
        if max_zero_pairs_to_process is not None and len(zero_pairs) > max_zero_pairs_to_process:
            zero_pairs_list = list(zero_pairs)
            random.shuffle(zero_pairs_list)
            zero_pairs = set(zero_pairs_list[:max_zero_pairs_to_process])
            print(f"  Anchor {anchor_hash[:20]}...: Randomly sampling {max_zero_pairs_to_process} zero pairs from {len(zero_pairs_list)} total")
        
        # Select top_k distant basins based on select_by_distance flag
        if select_by_distance:
            # Select by distance: compute distance for all sampled zero pairs, then select top_k smallest
            print(f"  Anchor {anchor_hash[:20]}...: Computing distances for {len(zero_pairs)} zero pairs to select top {top_k} by distance...")
            zero_pairs_with_distance = []
            
            for zero_basin_hash in zero_pairs:
                # Get solution_flat
                zero_solution = None
                if zero_basin_hash in basin_info:
                    zero_solution = basin_info[zero_basin_hash].get("solution_flat", [])
                if not zero_solution:
                    zero_solution = solution_lookup.get(zero_basin_hash)
                    if zero_solution:
                        basin_info[zero_basin_hash] = {
                            "cost": None,
                            "solution_flat": zero_solution
                        }
                
                if not zero_solution:
                    continue
                
                # Get cached zero basin pairs
                zero_pairs_set = basin_pairs_cache.get(zero_basin_hash)
                if zero_pairs_set is None:
                    zero_pairs_set = extract_adjacent_pairs_from_solution_flat(zero_solution)
                    basin_pairs_cache[zero_basin_hash] = zero_pairs_set
                
                # Calculate broken pairs distance
                broken_distance = calculate_broken_pairs_distance_fast(anchor_pairs, zero_pairs_set)
                zero_pairs_with_distance.append((broken_distance, zero_basin_hash))
            
            # Sort by distance (ascending) and select top_k
            zero_pairs_with_distance.sort(key=lambda x: x[0])
            selected_basins = zero_pairs_with_distance[:top_k]
            
            # Build result list
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
            # Random selection: randomly select top_k without computing distance
            print(f"  Anchor {anchor_hash[:20]}...: Randomly selecting {top_k} from {len(zero_pairs)} zero pairs...")
            zero_pairs_list = list(zero_pairs)
            random.shuffle(zero_pairs_list)
            selected_basins = zero_pairs_list[:top_k]
            
            # Build result list
            top_distant_basins = []
            for h in selected_basins:
                zero_solution = basin_info.get(h, {}).get("solution_flat") or solution_lookup.get(h)
                if not zero_solution:
                    continue
                
                if h not in basin_info:
                    basin_info[h] = {"cost": None, "solution_flat": zero_solution}
                
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
    
    # Determine output file path (in instance directory)
    instance_dir = f"/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#{instance_index}"
    output_file = os.path.join(instance_dir, "distant_basins.jsonl")
    
    return results, output_file


def main():
    parser = argparse.ArgumentParser(
        description="Find distant basins (zero pairs with smallest broken pairs distance) for each anchor basin."
    )
    parser.add_argument(
        "--instances",
        type=str,
        default="0",
        help='Instance indices to process (e.g., "0-10", "0,1,2"). Default: 0',
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=10,
        help="Minimum basin count threshold for co-occurrence (default: 10)",
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=10,
        help="Maximum number of runs to process per instance (default: 10)",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Maximum number of records per instance to process (default: None = process all)",
    )
    parser.add_argument(
        "--max_basins_per_record",
        type=int,
        default=None,
        help="Maximum number of basins per record to process (default: None = no limit)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=255,
        help="Number of distant basins to select per anchor (default: 255)",
    )
    parser.add_argument(
        "--max_zero_pairs_to_process",
        type=int,
        default=None,
        metavar="N",
        help="Max zero pairs to process per anchor; if not set, process all (default: None)",
    )
    parser.add_argument(
        "--min_basin_total_count",
        type=int,
        default=None,
        metavar="N",
        help="Only consider basins with total count >= N across all training data (default: None = include all)",
    )
    parser.add_argument(
        "--select_by_distance",
        action='store_true',
        help="If set, select zero pairs by distance (nearest first); otherwise random sample (default: random)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file path (default: None = save to instance directory as distant_basins.jsonl)",
    )
    
    args = parser.parse_args()
    
    instance_indices = parse_instance_range(args.instances)
    print(f"Processing instances: {instance_indices}")
    
    all_results = []
    output_files = []
    
    for idx in instance_indices:
        results, output_file = find_distant_basins_for_instance(
            idx,
            min_count=args.min_count,
            max_runs=args.max_runs,
            max_records=args.max_records,
            max_basins_per_record=args.max_basins_per_record,
            top_k=args.top_k,
            max_zero_pairs_to_process=args.max_zero_pairs_to_process,
            min_basin_total_count=args.min_basin_total_count,
            select_by_distance=args.select_by_distance,
        )
        
        if not results:
            print(f"  No results for instance {idx}, skipping")
            continue
        
        # Use custom output path if provided, otherwise use instance directory
        if args.output:
            output_file = args.output
        
        # Write results to JSONL (per instance)
        output_path = os.path.abspath(output_file)
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if dirname is not empty
            os.makedirs(output_dir, exist_ok=True)
        print(f"\n  Writing results to: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        
        print(f"  Wrote {len(results)} records for instance {idx}")
        
        # Ensure all basins exist in basin_info.jsonl
        print(f"  Ensuring basin_info.jsonl completeness for instance {idx}...")
        ensure_basin_info_complete(idx, output_path)
        
        all_results.extend(results)
        output_files.append(output_path)
    
    if not all_results:
        print("No data collected, nothing to write.")
        return
    
    print(f"\nDone. Processed {len(instance_indices)} instance(s), wrote {len(all_results)} total records.")
    print(f"Output files: {output_files}")


if __name__ == "__main__":
    main()
