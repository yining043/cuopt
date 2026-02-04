#!/usr/bin/env python3
"""
Generate run-level visualizations for completed runs that are missing visualizations.

Usage:
    python3 generate_missing_visualizations.py [--instance_index N] [--run_id RUN_ID]
"""
import sys
import os
import json
import argparse
from collections import defaultdict

# Add path before importing
sys.path.insert(0, '/home/jieyi/cuopt')

# Import functions that don't require cudf
try:
    from analyze_basin import get_all_run_trial_pairs, get_basin_paths
    from utils import get_basin_paths
    from analyze_intermediate_basins import generate_run_visualizations_from_jsonl
    # Import create_vrp_instance_from_pkl from test_basin_pybind
    from test_basin_pybind import create_vrp_instance_from_pkl
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please make sure you're in the correct conda environment")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def load_trial_data_from_jsonl(training_file, run_id):
    """Load all trial data for a run_id from training_data_pybind.jsonl in one pass.
    Returns: dict[trial_id] -> bool (whether trial has valid data)
    """
    trial_has_data = {}
    if not os.path.exists(training_file):
        return trial_has_data
    
    with open(training_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if data.get('run_id') == run_id:
                        trial_id = data.get('trial_id')
                        if trial_id is not None:
                            # Check if record has required fields for visualization
                            if data.get('initial_solution') and data.get('basin_features'):
                                trial_has_data[trial_id] = True
                except:
                    continue
    return trial_has_data

def is_run_complete(instance_path, instance_index, run_id, basin_base_dir, trial_data_cache=None):
    """Check if all trials in a run have data in training_data_pybind.jsonl.
    
    Args:
        trial_data_cache: Optional dict from load_trial_data_from_jsonl to avoid re-reading file
    """
    basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
    all_pairs = get_all_run_trial_pairs(basin_paths['trajectory_path'])
    run_trials = [(r, t) for r, t in all_pairs if r == run_id]
    
    if len(run_trials) == 0:
        return False, []
    
    # Get training file path
    output_dir = os.path.join("basin_datasets0_analyze", basin_paths['instance_id'])
    pybind_dir = os.path.join(output_dir, 'pybind_basin_analysis')
    training_file = os.path.join(pybind_dir, 'training_data_pybind.jsonl')
    
    # Load trial data (use cache if provided)
    if trial_data_cache is None:
        trial_has_data = load_trial_data_from_jsonl(training_file, run_id)
    else:
        trial_has_data = trial_data_cache
    
    # Check if all trials have data
    missing_trials = []
    for r, t in run_trials:
        if t not in trial_has_data:
            missing_trials.append(t)
    
    return len(missing_trials) == 0, missing_trials

def has_visualizations(instance_path, instance_index, run_id, basin_base_dir):
    """Check if run-level visualizations already exist."""
    basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
    instance_id = basin_paths['instance_id']
    viz_dir = os.path.join("basin_datasets0_analyze", instance_id, f"{run_id}_trial_plot")
    
    solution_comparison = os.path.join(viz_dir, f'run_{run_id}_solution_comparison.png')
    cost_curve = os.path.join(viz_dir, f'run_{run_id}_cost_curve_by_trial_pybind.png')
    
    return os.path.exists(solution_comparison) and os.path.exists(cost_curve)

def main():
    parser = argparse.ArgumentParser(description='Generate missing run-level visualizations')
    parser.add_argument('--instance_index', type=int, default=0, help='Instance index')
    parser.add_argument('--run_id', type=str, default=None, help='Specific run_id to process (optional)')
    args = parser.parse_args()
    
    instance_path = "/home/jieyi/cvrp100_uniform.pkl"
    instance_index = args.instance_index
    basin_base_dir = "basin_datasets0"
    hgs_solution_path = "/home/jieyi/hgs_cvrp100_uniform.pkl"
    num_vehicles = 30
    
    print("=" * 80)
    print("Generating Missing Run-Level Visualizations")
    print("=" * 80)
    
    # Get paths
    basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
    output_dir = os.path.join("basin_datasets0_analyze", basin_paths['instance_id'])
    pybind_dir = os.path.join(output_dir, 'pybind_basin_analysis')
    training_file = os.path.join(pybind_dir, 'training_data_pybind.jsonl')
    
    # Get num_orders
    vrp_instance = create_vrp_instance_from_pkl(instance_path, instance_index, num_vehicles)
    num_orders = vrp_instance['num_orders']
    del vrp_instance
    
    # Pre-load all trial data from JSONL for all runs (one-time read for efficiency)
    # Only check runs that actually exist in training_data_pybind.jsonl
    print("\nLoading trial data from training_data_pybind.jsonl...")
    print(f"  Reading {training_file}...")
    
    # Load data for all runs in one pass, and collect run_ids that have data
    all_runs_trial_data = defaultdict(dict)  # run_id -> {trial_id: True}
    all_run_ids_set = set()  # Only runs that have data in JSONL
    
    if os.path.exists(training_file):
        with open(training_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        run_id = data.get('run_id')
                        trial_id = data.get('trial_id')
                        if run_id and trial_id is not None:
                            # Check if record has required fields for visualization
                            if data.get('initial_solution') and data.get('basin_features'):
                                all_runs_trial_data[run_id][trial_id] = True
                                all_run_ids_set.add(run_id)
                    except:
                        continue
        print(f"  Loaded data for {len(all_runs_trial_data)} runs")
    else:
        print(f"  Training file not found: {training_file}")
        return
    
    # Only process runs that exist in training_data_pybind.jsonl
    all_run_ids = sorted(all_run_ids_set)
    
    # Filter by run_id if specified
    if args.run_id:
        if args.run_id not in all_run_ids:
            print(f"Error: run_id {args.run_id} not found in training_data_pybind.jsonl")
            return
        all_run_ids = [args.run_id]
    
    print(f"\nFound {len(all_run_ids)} runs in training_data_pybind.jsonl")
    
    # Process runs one by one: check and generate immediately if ready
    print("\nChecking runs and generating visualizations...")
    print("=" * 80)
    
    generated_count = 0
    skipped_count = 0
    
    # Get trajectory data to check expected trials per run
    trajectory_path = basin_paths['trajectory_path']
    all_pairs = get_all_run_trial_pairs(trajectory_path) if os.path.exists(trajectory_path) else []
    
    for run_id in all_run_ids:
        # Get expected trials for this run from trajectory
        run_trials = [(r, t) for r, t in all_pairs if r == run_id]
        expected_trial_count = len(run_trials)
        
        # Check if run is complete using cached data
        trial_data_cache = all_runs_trial_data.get(run_id, {})
        actual_trial_count = len(trial_data_cache)
        
        # A run is complete if all expected trials have data, or if we have data for all trials in JSONL
        # (if trajectory doesn't exist, just check if we have any data)
        if expected_trial_count > 0:
            missing_trials = [t for r, t in run_trials if t not in trial_data_cache]
            is_complete = len(missing_trials) == 0
        else:
            # No trajectory data, consider complete if we have any data
            is_complete = actual_trial_count > 0
            missing_trials = []
        
        if is_complete:
            # Check if visualizations already exist
            if has_visualizations(instance_path, instance_index, run_id, basin_base_dir):
                print(f"  Run {run_id}: completed and has visualizations (skipping)")
                skipped_count += 1
            else:
                # Generate immediately
                print(f"\n  Run {run_id}: completed but missing visualizations (generating now...)")
                try:
                    generate_run_visualizations_from_jsonl(
                        instance_path, instance_index, run_id,
                        basin_base_dir, hgs_solution_path, num_orders, num_vehicles
                    )
                    print(f"  ✓ Successfully generated visualizations for run_id={run_id}")
                    generated_count += 1
                except Exception as e:
                    print(f"  ✗ Error generating visualizations for run_id={run_id}: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print(f"  Run {run_id}: incomplete ({len(missing_trials)}/{len(run_trials)} trials missing data in JSONL)")
    
    print("\n" + "=" * 80)
    print("Done!")
    print(f"  Generated visualizations for {generated_count} runs")
    print(f"  Skipped {skipped_count} runs (already have visualizations)")
    print("=" * 80)

if __name__ == "__main__":
    main()
