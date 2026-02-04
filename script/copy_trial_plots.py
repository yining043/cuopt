#!/usr/bin/env python3
"""
Extract run_id from trials.jsonl for all instances,
and copy corresponding images from /home/jieyi/plot/plot/basin_{run_id}/ to
/home/jieyi/cuopt/basin_datasets0/cvrp100_uniform.pkl#{idx}/trials_plot/
"""

import json
import os
import shutil
from pathlib import Path

def extract_run_ids_from_trials_jsonl(jsonl_path):
    """Extract all unique run_id from trials.jsonl file"""
    run_ids = set()
    if not os.path.exists(jsonl_path):
        print(f"Warning: File does not exist: {jsonl_path}")
        return run_ids
    
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'run_id' in data:
                        run_ids.add(data['run_id'])
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON line: {line[:50]}... Error: {e}")
    except Exception as e:
        print(f"Error: Failed to read file {jsonl_path}: {e}")
    
    return run_ids

def copy_plot_file(run_id, source_base, target_dir):
    """Copy plot image file"""
    source_file = os.path.join(source_base, f"basin_{run_id}", 
                               "callback_cost_curve_by_trial_trial_colored_cf.png")
    target_file = os.path.join(target_dir, 
                               f"callback_cost_curve_by_trial_trial_colored_cf_{run_id}.png")
    
    if not os.path.exists(source_file):
        print(f"  Warning: Source file does not exist: {source_file}")
        return False
    
    try:
        shutil.copy2(source_file, target_file)
        print(f"  ✓ Copied: {run_id}")
        return True
    except Exception as e:
        print(f"  ✗ Copy failed {run_id}: {e}")
        return False

def process_instance(instance_idx):
    """Process a single instance"""
    print(f"\n{'='*60}")
    print(f"Processing instance #{instance_idx}")
    print(f"{'='*60}")
    
    # Build paths
    instance_dir = f"/home/jieyi/cuopt/basin_datasets0/cvrp100_uniform.pkl#{instance_idx}"
    trials_jsonl = os.path.join(instance_dir, "trials.jsonl")
    target_dir = os.path.join(instance_dir, "trials_plot")
    source_base = "/home/jieyi/plot/plot"
    
    # Check if instance directory exists
    if not os.path.exists(instance_dir):
        print(f"Skipping: Instance directory does not exist: {instance_dir}")
        return
    
    # Extract run_id
    print(f"Extracting run_id from {trials_jsonl}...")
    run_ids = extract_run_ids_from_trials_jsonl(trials_jsonl)
    
    if not run_ids:
        print(f"  No run_id found")
        return
    
    print(f"  Found {len(run_ids)} unique run_id(s)")
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    print(f"Target directory: {target_dir}")
    
    # Copy files
    success_count = 0
    for run_id in sorted(run_ids):
        if copy_plot_file(run_id, source_base, target_dir):
            success_count += 1
    
    print(f"\nCompleted: Successfully copied {success_count}/{len(run_ids)} files")

def main():
    """Main function: Process all instances"""
    print("Starting to process all instances...")
    print(f"Source directory: /home/jieyi/plot/plot/basin_*/")
    print(f"Target directory: /home/jieyi/cuopt/basin_datasets0/cvrp100_uniform.pkl#*/trials_plot/")
    
    # Process all instances (0-79)
    total_instances = 80
    for idx in range(total_instances):
        process_instance(idx)
    
    print(f"\n{'='*60}")
    print("All instances processed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
