#!/usr/bin/env python3
"""
Reset basin_info.jsonl to only include basins from basin_pairs.jsonl,
then regenerate distant_basins.jsonl and update basin_info.jsonl.
"""

import json
import os
import sys
import argparse
from typing import Set, Dict


def collect_basins_from_pairs_file(instance_index: int) -> Set[str]:
    """Collect all basin hashes from basin_pairs.jsonl (both anchor and neighbor)."""
    pairs_file = f"/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#{instance_index}/basin_pairs.jsonl"
    basins = set()
    
    if not os.path.exists(pairs_file):
        print(f"Warning: basin_pairs.jsonl not found for instance {instance_index}")
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


def load_basin_info(instance_index: int) -> Dict[str, dict]:
    """Load basin info from basin_info.jsonl."""
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
                    basin_info[h] = rec
            except Exception as e:
                print(f"Error parsing basin_info line: {e}")
                continue
    return basin_info


def save_basin_info(instance_index: int, basin_info: Dict[str, dict], valid_basins: Set[str]):
    """Save basin_info.jsonl with only valid basins."""
    info_file = f"/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#{instance_index}/basin_info.jsonl"
    
    # Filter to only include valid basins
    filtered_info = {h: info for h, info in basin_info.items() if h in valid_basins}
    
    # Write to file
    with open(info_file, "w", encoding="utf-8") as f:
        for basin_hash in sorted(filtered_info.keys()):
            f.write(json.dumps(filtered_info[basin_hash], ensure_ascii=False) + "\n")
    
    print(f"  Saved {len(filtered_info)} basins to basin_info.jsonl (from {len(basin_info)} original)")


def reset_basin_info_for_instance(instance_index: int):
    """Reset basin_info.jsonl to only include basins from basin_pairs.jsonl."""
    print(f"\n{'=' * 60}")
    print(f"Processing instance {instance_index}")
    print(f"{'=' * 60}")
    
    # Collect all basins from basin_pairs.jsonl
    print("  Collecting basins from basin_pairs.jsonl...")
    valid_basins = collect_basins_from_pairs_file(instance_index)
    print(f"    Found {len(valid_basins)} unique basins in basin_pairs.jsonl")
    
    if not valid_basins:
        print(f"  No basins found in basin_pairs.jsonl, skipping instance {instance_index}")
        return
    
    # Load current basin_info.jsonl
    print("  Loading current basin_info.jsonl...")
    basin_info = load_basin_info(instance_index)
    print(f"    Loaded {len(basin_info)} basins from basin_info.jsonl")
    
    # Count how many will be kept
    kept_basins = valid_basins & set(basin_info.keys())
    removed_basins = set(basin_info.keys()) - valid_basins
    
    print(f"    Will keep: {len(kept_basins)} basins")
    print(f"    Will remove: {len(removed_basins)} basins")
    
    # Save filtered basin_info.jsonl
    print("  Resetting basin_info.jsonl...")
    save_basin_info(instance_index, basin_info, valid_basins)
    
    print(f"  Completed reset for instance {instance_index}")


def main():
    parser = argparse.ArgumentParser(
        description="Reset basin_info.jsonl to only include basins from basin_pairs.jsonl"
    )
    parser.add_argument(
        "--instances",
        type=str,
        required=True,
        help="Instance indices (e.g., '0', '0-5', '0,1,2')",
    )
    
    args = parser.parse_args()
    
    # Parse instance range
    indices = set()
    parts = args.instances.split(',')
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
    instance_indices = sorted(indices)
    
    print(f"Resetting basin_info.jsonl for instances: {instance_indices}")
    
    for idx in instance_indices:
        reset_basin_info_for_instance(idx)
    
    print(f"\n{'=' * 60}")
    print("Reset completed!")
    print("Next step: Run find_distant_basins.py to regenerate distant_basins.jsonl")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
