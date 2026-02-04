#!/usr/bin/env python3
"""
Analyze basin structure by running local search from each solution in a trial.
For each solution, run local search 100 times and check which basins (local optima) are reached.
Uses edge set hash to identify basins.
"""
import json
import os
import sys
import random
import gc
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import networkx as nx

# Try to import torch for CUDA cache clearing (optional)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def ensure_cuda_device():
    if HAS_TORCH and torch.cuda.is_available():
        # When CUDA_VISIBLE_DEVICES is set, PyTorch remaps devices
        # Always use device 0 (the first visible device)
        torch.cuda.set_device(0)


def aggressive_gc_cleanup():
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_gpu_memory():
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                           '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, check=True)
    used, total = result.stdout.strip().split('\n')[0].split(',')
    used, total = int(used.strip()), int(total.strip())
    percentage = (used / total) * 100
    print(f"  GPU Memory: {used}/{total} MB ({percentage:.1f}%)")
    return used


from test_basin_pybind import (
    create_vrp_instance_from_pkl, validate_solution_feasibility,
    run_local_search, create_random_initial_solution
)
from test_load_data import load_hgs_solution_from_pkl, calculate_gap
from utils import (
    load_solution_from_trajectory, get_basin_paths,
    edges_hash, routes_to_edges, routes_to_solution_flat
)


def run_local_search_silent(cuopt_env, initial_routes, vrp_instance, max_iterations=100):
    """Run local search from initial solution, return final cost, edges hash, edges, and routes."""
    cuopt_env.initialize_search(initial_routes)
    weights = [10000., 10000., 100., 1000., 1000., 1000., 10000., 10000., 10000.]
    cuopt_env.set_weights(weights)
    cuopt_env.set_selection_weights(weights)
    cuopt_env.acquire_resource()
    cuopt_env.reset_move_candidates()
    cuopt_env.set_routes_to_search()
    cuopt_env.sync_streams()
    
    for outer_iter in range(max_iterations):
        cuopt_env.extract_nodes_to_search()
        while True:
            if not cuopt_env.sample_nodes_to_search(full_set=False):
                break # node pool exhausted
            fast_operators = ['vrp', 'sliding', 'two_opt']
            random.shuffle(fast_operators)
            for op in fast_operators:
                if op == 'vrp':
                    cuopt_env.perform_vrp_search()
                elif op == 'sliding':
                    cuopt_env.run_sliding_search()
                elif op == 'two_opt':
                    cuopt_env.run_two_opt_search()
            cuopt_env.restore_found_nodes()
        if not cuopt_env.run_cycle_finder():
            break # no improvement found
        # Periodic aggressive cleanup during long-running search (every 50 iterations)
        if (outer_iter + 1) % 50 == 0:
            gc.collect()
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Sync streams before getting results to ensure all CUDA operations are complete
    cuopt_env.sync_streams()
    final_cost = cuopt_env.get_cost()
    final_routes = cuopt_env.get_solution_routes().copy()  # Explicit copy to ensure CPU data
    final_edges = routes_to_edges(final_routes)
    edges_hash_val = edges_hash(final_edges)
    cuopt_env.set_routes_to_search()
    cuopt_env.release_resource()
    cuopt_env.sync_streams()  # Final sync after release
    
    return final_cost, edges_hash_val, final_edges, final_routes


def get_all_run_trial_pairs(trajectory_path):
    """Get all unique (run_id, trial_id) pairs from trajectory.jsonl. Skips malformed lines."""
    run_trial_pairs = set()
    with open(trajectory_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # Skip malformed lines
            run_id = data.get('run_id')
            trial_id = data.get('trial_id')
            if run_id is not None and trial_id is not None:
                run_trial_pairs.add((run_id, trial_id))
    return sorted(run_trial_pairs)


def is_trial_already_analyzed(instance_path, instance_index, run_id, trial_id, basin_base_dir="basin_datasets0"):
    basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
    basin_dir = os.path.join("basin_datasets0_analyze", basin_paths['instance_id'])
    viz_file = os.path.join(basin_dir, f'comprehensive_analysis_run_{run_id}_trial_{trial_id}.png')
    
    if not os.path.exists(viz_file):
        return False
    
    trajectory_path = basin_paths['trajectory_path']
    training_file = os.path.join(basin_dir, 'training_data.jsonl')
    
    trial_solutions = set()
    with open(trajectory_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get('run_id') == run_id and data.get('trial_id') == trial_id:
                trial_solutions.add((data.get('global_iter'), data.get('local_iter')))
    
    if not trial_solutions:
        return False
    
    analyzed_solutions = set()
    with open(training_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (data.get('run_id') == run_id and 
                data.get('trial_id') == trial_id and
                'initial_solution' in data):
                analyzed_solutions.add((data.get('global_iter'), data.get('local_iter')))
    
    return trial_solutions.issubset(analyzed_solutions)


def get_completed_solutions(instance_path, instance_index, run_id, trial_id, basin_base_dir="basin_datasets0"):
    basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
    basin_dir = os.path.join("basin_datasets0_analyze", basin_paths['instance_id'])
    training_file = os.path.join(basin_dir, 'training_data.jsonl')
    
    completed = set()
    try:
        with open(training_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (data.get('run_id') == run_id and 
                    data.get('trial_id') == trial_id and
                    'initial_solution' in data):
                    completed.add((data.get('global_iter'), data.get('local_iter')))
    except FileNotFoundError:
        pass
    return completed


def cleanup_trial_data(instance_path, instance_index, run_id, trial_id, basin_base_dir="basin_datasets0"):
    basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
    basin_dir = os.path.join("basin_datasets0_analyze", basin_paths['instance_id'])
    training_file = os.path.join(basin_dir, 'training_data.jsonl')
    
    lines_to_keep = []
    try:
        with open(training_file, 'r') as f:
            for line in f:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                data = json.loads(line_stripped)
                if not (data.get('run_id') == run_id and data.get('trial_id') == trial_id):
                    lines_to_keep.append(line)
    except FileNotFoundError:
        pass
    
    if lines_to_keep:
        with open(training_file, 'w') as f:
            f.writelines(lines_to_keep)
    
    excel_file = os.path.join(basin_dir, f'basin_analysis_run_{run_id}_trial_{trial_id}.xlsx')
    viz_file = os.path.join(basin_dir, f'comprehensive_analysis_run_{run_id}_trial_{trial_id}.png')
    try:
        os.remove(excel_file)
    except FileNotFoundError:
        pass
    try:
        os.remove(viz_file)
    except FileNotFoundError:
        pass

def analyze_trial_basins(instance_path, instance_index, run_id, trial_id, num_runs=100, basin_base_dir="basin_datasets0", hgs_solution_path=None):
    """Analyze basins for a specific trial."""
    
    # Ensure CUDA device is properly set (important for batch processing)
    ensure_cuda_device()
    # Pre-trial cleanup to ensure clean state
    aggressive_gc_cleanup()
    
    # Load HGS solution for gap comparison
    hgs_cost = None
    if hgs_solution_path:
        hgs_solution = load_hgs_solution_from_pkl(hgs_solution_path, instance_index)
        hgs_cost = hgs_solution["hgs_cost"]
        print(f"HGS solution cost: {hgs_cost:.2f}")

    # Get trajectory path
    basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
    trajectory_path = basin_paths['trajectory_path']
    basin_dir = os.path.join("basin_datasets0_analyze", basin_paths['instance_id'])
    
    # Load all solutions for this trial
    trial_solutions = []
    with open(trajectory_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get('run_id') == run_id and data.get('trial_id') == trial_id:
                trial_solutions.append(data)
    if not trial_solutions:
        print(f"No solutions found for run_id={run_id}, trial_id={trial_id}")
        return []
    # Check if trial is already fully completed
    completed_solutions = get_completed_solutions(instance_path, instance_index, run_id, trial_id, basin_base_dir)
    if len(completed_solutions) == len(trial_solutions):
        print(f"All solutions for run_id={run_id}, trial_id={trial_id} have already been analyzed.")
        return []
    # If partially completed, we'll re-run the entire trial for simplicity and data integrity
    # This avoids data precision loss and merging complexity
    if len(completed_solutions) > 0:
        print(f"Found {len(completed_solutions)}/{len(trial_solutions)} solutions already completed.")
        print(f"Re-running entire trial to ensure data integrity (avoiding precision loss from reconstruction).")
        print(f"This will re-process {len(completed_solutions)} completed solutions.")
        # Clean up old data for this trial before re-running
        cleanup_trial_data(instance_path, instance_index, run_id, trial_id, basin_base_dir)
    # Process all solutions (re-run entire trial)
    solutions_to_process = trial_solutions
    # Get final solution edges_hash from trajectory (for comparison)
    final_solution = None
    for sol in trial_solutions:
        if sol.get('is_final_of_trial'):
            final_solution = sol
            break
    final_edges_hash = final_solution.get('edges_hash') if final_solution else None
    print(f"Final solution edges_hash: {final_edges_hash}")
    
    results = []
    for sol_idx, sol_data in enumerate(solutions_to_process):
        cost = sol_data.get('cost', 0)
        gap_str = f", gap_to_hgs={calculate_gap(cost, hgs_cost):.2f}%" if hgs_cost else ""
        print(f"\nSolution {sol_idx + 1}/{len(solutions_to_process)} (total {len(trial_solutions)}): "
              f"global_iter={sol_data.get('global_iter')}, local_iter={sol_data.get('local_iter')}, "
              f"cost={cost:.2f}{gap_str}, edges_hash={sol_data.get('edges_hash', 'N/A')[:8]}")

        # Get num_locations from a temporary instance (will be deleted immediately)
        # Ensure CUDA device is set before creating instance
        ensure_cuda_device()
        temp_instance = create_vrp_instance_from_pkl(instance_path, instance_index, num_vehicles=30)
        num_locations = temp_instance['num_locations']
        del temp_instance
        aggressive_gc_cleanup()
        
        solution_data = load_solution_from_trajectory(trajectory_path, trial_id, 
            sol_data.get('global_iter'), sol_data.get('local_iter'), num_locations)
        initial_routes = solution_data['routes']
        is_cycle_finder = solution_data.get('is_cycle_finder', False)
        
        basin_counts = defaultdict(int)
        basin_data = {}  # edges_hash -> {edges, routes, solution_flat, costs}
        final_costs = []
        matches_final = 0
        
        for run in range(num_runs):
            run_vrp_instance = None
            cuopt_env = None
            
            try:
                # Ensure CUDA device is set before creating instance
                ensure_cuda_device()
                
                # Create new instance for each run (cuopt_env state changes after search)
                # Note: This allocates CUDA memory, must be properly released in finally block
                run_vrp_instance = create_vrp_instance_from_pkl(instance_path, instance_index, num_vehicles=30)
                cuopt_env = run_vrp_instance['cuopt_env']
                
                # Clear large arrays from instance dict to reduce memory footprint
                # Keep only essential references
                if 'cost_matrix' in run_vrp_instance:
                    del run_vrp_instance['cost_matrix']
                if 'node_coords' in run_vrp_instance:
                    del run_vrp_instance['node_coords']
                
                final_cost, edges_hash_val, final_edges, final_routes = run_local_search_silent(
                    cuopt_env, initial_routes, run_vrp_instance, max_iterations=100000)
                
                if edges_hash_val not in basin_data:
                    # Only store essential data, avoid storing routes (may contain CUDA references)
                    basin_data[edges_hash_val] = {
                        'edges': sorted(final_edges),
                        'solution_flat': routes_to_solution_flat(final_routes, num_locations),
                        'costs': []
                    }
                basin_data[edges_hash_val]['costs'].append(final_cost)
                
                # Clear references immediately to help GC
                del final_routes
                del final_edges
                
                basin_counts[edges_hash_val] += 1
                final_costs.append(final_cost)
                if final_edges_hash and edges_hash_val == final_edges_hash:
                    matches_final += 1
            finally:
                # Explicit cleanup after each run to free CUDA memory
                # Note: release_resource() is already called in run_local_search_silent,
                # but we still need to clean up the object references
                if cuopt_env is not None:
                    # Double-check: release again if not already released (safe to call multiple times)
                    try:
                        cuopt_env.set_routes_to_search()  # Reset state before release
                        cuopt_env.release_resource()
                        cuopt_env.sync_streams()
                    except:
                        pass  # Already released in run_local_search_silent
                    finally:
                        del cuopt_env
                    cuopt_env = None
                
                if run_vrp_instance is not None:
                    # Clear any remaining references in the instance dict
                    if 'cuopt_env' in run_vrp_instance:
                        del run_vrp_instance['cuopt_env']
                    del run_vrp_instance
                    run_vrp_instance = None
                
            # Periodic progress report every 20 runs
            if (run + 1) % 20 == 0:
                print(f"  Run {run + 1}/{num_runs}: {len(basin_counts)} unique basins, "
                      f"{matches_final} matches final solution")
        
        mean_final_cost = sum(final_costs) / len(final_costs)
        match_ratio = matches_final / num_runs
        initial_edges = routes_to_edges(initial_routes)
        initial_solution_flat = routes_to_solution_flat(initial_routes, num_locations)
        
        results.append({
            'solution_idx': sol_idx,
            'global_iter': sol_data.get('global_iter'),
            'local_iter': sol_data.get('local_iter'),
            'initial_cost': sol_data.get('cost'),
            'initial_edges_hash': sol_data.get('edges_hash', ''),
            'initial_edges': sorted(initial_edges),
            'initial_solution_flat': initial_solution_flat,
            'is_cycle_finder': is_cycle_finder,
            'basin_counts': dict(basin_counts),
            'basin_data': basin_data,
            'num_unique_basins': len(basin_counts),
            'final_costs': final_costs,
            'mean_final_cost': mean_final_cost,
            'matches_final': matches_final,
            'match_final_ratio': match_ratio,
            'num_runs': num_runs,
        })
        
        gap_str = f", gap_to_hgs={calculate_gap(mean_final_cost, hgs_cost):.2f}%" if hgs_cost else ""
        print(f"  Result: {len(basin_counts)} unique basins, "
              f"mean final cost: {mean_final_cost:.2f}{gap_str}, "
              f"matches final: {matches_final}/{num_runs} ({match_ratio*100:.1f}%)")
        
        # Save results immediately after each solution
        os.makedirs(basin_dir, exist_ok=True)
        
        save_results_to_excel(results, run_id, trial_id, basin_dir, final_edges_hash, hgs_cost, None, None)
        
        # Save training data format (aggregated, one solution per line, includes all basin info)
        # Append to instance-level file (one file per instance, all trials)
        # Note: save_training_data_format will automatically skip duplicates based on (run_id, trial_id, global_iter, local_iter)
        save_training_data_format([results[-1]], run_id, trial_id, basin_dir, final_edges_hash, hgs_cost, 
                                 clear_file=False)
        
        # Cleanup to free memory (data already saved)
        # Clear basin_data routes (already set to None, but ensure cleanup)
        for basin_info in basin_data.values():
            if 'routes' in basin_info:
                basin_info['routes'] = None
            # Clear solution_flat if it's large (it's already saved to disk)
            if 'solution_flat' in basin_info and len(basin_info.get('solution_flat', [])) > 1000:
                basin_info['solution_flat'] = None
        
        # Clear all local variables
        del initial_routes, initial_edges, initial_solution_flat
        del basin_counts, basin_data, final_costs
        
        # Additional cleanup: clear results list periodically to prevent accumulation
        # Only keep last few results in memory (they're already saved to disk)
        if len(results) > 5:
            # Keep only metadata, clear large data structures
            for old_result in results[:-5]:
                if 'basin_data' in old_result:
                    for basin_info in old_result['basin_data'].values():
                        basin_info['routes'] = None
                        if 'solution_flat' in basin_info:
                            basin_info['solution_flat'] = None
    
    print(f"\n{'='*80}")
    print(f"Summary for run_id={run_id}, trial_id={trial_id}")
    print(f"{'='*80}")
    for r in results:
        print(f"\nSolution {r['solution_idx']} (iter {r['global_iter']}/{r['local_iter']}):")
        print(f"  Initial cost: {r['initial_cost']:.2f}, Unique basins: {r['num_unique_basins']}")
        print(f"  Mean final cost: {r['mean_final_cost']:.2f}, Matches final: {r['matches_final']}/{num_runs} ({r['match_final_ratio']*100:.1f}%)")
        print(f"  Top 5 basin frequencies: {sorted(r['basin_counts'].values(), reverse=True)[:5]}")
    
    # Create comprehensive visualization (save to instance folder) - only at the end
    basin_dir = os.path.join("basin_datasets0_analyze", basin_paths['instance_id'])
    create_comprehensive_visualization(results, final_edges_hash, run_id, trial_id, basin_dir, hgs_cost)
    
    # Basin entropy analysis
    entropy_results = analyze_basin_entropy(results, basin_dir, run_id, trial_id, hgs_cost)
    
    # Critical gap analysis (if HGS cost available)
    critical_gap_results = None
    if hgs_cost:
        critical_gap_results = find_critical_gap_for_basin_stability(results, basin_dir, run_id, trial_id, hgs_cost)
    
    # Update Excel with entropy and critical gap results
    save_results_to_excel(results, run_id, trial_id, basin_dir, final_edges_hash, hgs_cost, 
                         entropy_results, critical_gap_results)
    
    return results


def save_basin_analysis_results_incremental(results, run_id, trial_id, basin_dir, final_edges_hash, hgs_cost, clear_file=False):
    """Save basin analysis results to jsonl file incrementally (append mode)."""
    os.makedirs(basin_dir, exist_ok=True)
    output_file = os.path.join(basin_dir, f'basin_analysis_run_{run_id}_trial_{trial_id}.jsonl')
    
    # If clearing file or file doesn't exist, start fresh
    # Otherwise, read existing basins to maintain consistent basin_id mapping
    existing_basin_id_map = {}  # edges_hash -> basin_id
    max_basin_id = 0
    if not clear_file and os.path.exists(output_file):
        # Read existing file to get all basin IDs
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if 'basin_edges_hash' in data and 'basin_id' in data:
                    edges_hash = data['basin_edges_hash']
                    basin_id = data['basin_id']
                    existing_basin_id_map[edges_hash] = basin_id
                    max_basin_id = max(max_basin_id, basin_id)
    
    # Collect all unique basins from current results
    all_basins = set()
    for result in results:
        all_basins.update(result['basin_counts'].keys())
    
    # Assign IDs: existing basins keep their IDs, new basins get new IDs
    basin_id_map = existing_basin_id_map.copy()  # Start with existing mappings
    new_basins_only = all_basins - set(existing_basin_id_map.keys())
    next_id = max_basin_id + 1
    for edges_hash in sorted(new_basins_only):
        basin_id_map[edges_hash] = next_id
        next_id += 1
    
    # Append mode if not clearing
    mode = 'w' if clear_file else 'a'
    with open(output_file, mode) as f:
        for result in results:
            # Save initial solution info
            initial_record = {
                'run_id': run_id,
                'trial_id': trial_id,
                'global_iter': result['global_iter'],
                'local_iter': result['local_iter'],
                'cost': result['initial_cost'],
                'edges_hash': result['initial_edges_hash'],
                'edges': [[u, v] for u, v in result['initial_edges']],
                'solution_flat': result['initial_solution_flat'],
                'is_initial_solution': True,
                'is_cycle_finder': result.get('is_cycle_finder', False),
            }
            if hgs_cost:
                initial_record['gap_to_hgs'] = calculate_gap(result['initial_cost'], hgs_cost)
            f.write(json.dumps(initial_record) + '\n')
            
            # For each unique basin found, create a record
            basin_data = result['basin_data']
            for edges_hash, count in result['basin_counts'].items():
                basin_info = basin_data[edges_hash]
                costs = basin_info['costs']
                mean_cost = sum(costs) / len(costs) if costs else result['mean_final_cost']
                
                record = {
                    'run_id': run_id,
                    'trial_id': trial_id,
                    'basin_id': basin_id_map[edges_hash],
                    'source_global_iter': result['global_iter'],
                    'source_local_iter': result['local_iter'],
                    'source_cost': result['initial_cost'],
                    'source_edges_hash': result['initial_edges_hash'],
                    'basin_edges_hash': edges_hash,
                    'basin_frequency': count,
                    'basin_probability': count / result['num_runs'],
                    'basin_edges': [[u, v] for u, v in basin_info['edges']],
                    'basin_solution_flat': basin_info['solution_flat'],
                    'basin_mean_cost': mean_cost,
                    'num_runs': result['num_runs'],
                    'is_final_basin': (edges_hash == final_edges_hash) if final_edges_hash else False,
                }
                if hgs_cost:
                    record['gap_to_hgs'] = calculate_gap(mean_cost, hgs_cost)
                f.write(json.dumps(record) + '\n')
    
    print(f"  Basin analysis results saved to: {output_file}")


def save_training_data_format(results, run_id, trial_id, basin_dir, final_edges_hash, hgs_cost, clear_file=False):
    os.makedirs(basin_dir, exist_ok=True)
    output_file = os.path.join(basin_dir, 'training_data.jsonl')
    
    existing_keys = set()
    if not clear_file:
        try:
            with open(output_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    existing_keys.add((data.get('run_id'), data.get('trial_id'), 
                                     data.get('global_iter'), data.get('local_iter')))
        except FileNotFoundError:
            pass
    
    new_records = []
    with open(output_file, 'a' if not clear_file else 'w') as f:
        for result in results:
            key = (run_id, trial_id, result['global_iter'], result['local_iter'])
            if key in existing_keys:
                continue
            
            # Build basin distribution and features
            basin_distribution = {}
            basin_features = {}
            
            for edges_hash, count in result['basin_counts'].items():
                basin_info = result['basin_data'][edges_hash]
                basin_id = edges_hash
                probability = count / result['num_runs']
                
                basin_distribution[basin_id] = probability
                
                costs = basin_info['costs']
                mean_cost = sum(costs) / len(costs) if costs else result['mean_final_cost']
                
                basin_features[basin_id] = {
                    'edges': [[u, v] for u, v in basin_info['edges']],
                    'solution_flat': basin_info['solution_flat'],
                    'mean_cost': mean_cost,
                    'frequency': count,
                    'probability': probability,
                    'is_final_basin': (edges_hash == final_edges_hash) if final_edges_hash else False,
                }
                if hgs_cost:
                    basin_features[basin_id]['gap_to_hgs'] = calculate_gap(mean_cost, hgs_cost)
            
            # Create training record
            training_record = {
                'run_id': run_id,
                'trial_id': trial_id,
                'global_iter': result['global_iter'],
                'local_iter': result['local_iter'],
                'initial_solution': {
                    'edges': [[u, v] for u, v in result['initial_edges']],
                    'solution_flat': result['initial_solution_flat'],
                    'cost': result['initial_cost'],
                    'edges_hash': result['initial_edges_hash'],
                    'is_cycle_finder': result.get('is_cycle_finder', False),
                },
                'basin_distribution': basin_distribution,
                'basin_features': basin_features,
                'num_unique_basins': result['num_unique_basins'],
                'num_runs': result['num_runs'],
            }
            if hgs_cost:
                training_record['initial_solution']['gap_to_hgs'] = calculate_gap(result['initial_cost'], hgs_cost)
            
            f.write(json.dumps(training_record) + '\n')
            new_records.append(key)
    
    if new_records:
        print(f"  Training data format saved: {len(new_records)} new records added to {output_file}")
    else:
        print(f"  Training data: no new records (all duplicates skipped)")


def save_basin_analysis_results(results, run_id, trial_id, basin_dir, final_edges_hash, hgs_cost):
    """Save basin analysis results to jsonl file (similar format to trajectory.jsonl)."""
    # Use incremental function with clear_file=True to overwrite
    save_basin_analysis_results_incremental(results, run_id, trial_id, basin_dir, final_edges_hash, hgs_cost, clear_file=True)


def create_comprehensive_visualization(results, final_edges_hash, run_id, trial_id, basin_dir, hgs_cost=None):
    """Create comprehensive visualization with all important plots in one 4x5 figure (20 plots)."""
    os.makedirs(basin_dir, exist_ok=True)
    
    if not results:
        return
    
    results_sorted = sorted(results, key=lambda x: (x['global_iter'], x['local_iter']))
    local_iters = [r['local_iter'] for r in results_sorted]
    is_cycle_finder = [r.get('is_cycle_finder', False) for r in results_sorted]
    regular_mask = [not cf for cf in is_cycle_finder]
    cf_mask = is_cycle_finder
    
    num_basins = [r['num_unique_basins'] for r in results_sorted]
    match_ratios = [r['match_final_ratio'] * 100 for r in results_sorted]
    initial_costs = [r['initial_cost'] for r in results_sorted]
    mean_final_costs = [r['mean_final_cost'] for r in results_sorted]
    
    all_basin_counts = defaultdict(int)
    all_basins = {}
    basin_attraction = defaultdict(int)
    for result in results:
        for edges_hash, count in result['basin_counts'].items():
            all_basin_counts[edges_hash] += count
            basin_attraction[edges_hash] += count
            if edges_hash not in all_basins:
                basin_info = result['basin_data'][edges_hash]
                all_basins[edges_hash] = {'edges': set(basin_info['edges']),
                    'mean_cost': np.mean(basin_info['costs']) if basin_info['costs'] else result['mean_final_cost']}
    sorted_basins = sorted(all_basin_counts.items(), key=lambda x: x[1], reverse=True)
    
    fig = plt.figure(figsize=(20, 16))
    
    # Row 1: Trends vs local_iter
    ax1 = plt.subplot(4, 5, 1)
    ax1.plot(local_iters, num_basins, 'o-', linewidth=2, markersize=5, color='steelblue', zorder=2)
    if any(cf_mask):
        for i, is_cf in enumerate(cf_mask):
            if is_cf:
                ax1.plot(local_iters[i], num_basins[i], 'r*', markersize=10, zorder=3, 
                        label='Cycle Finder' if i == next((j for j, cf in enumerate(cf_mask) if cf), 0) else '')
    ax1.set_xlabel('Local Iter')
    ax1.set_ylabel('Unique Basins')
    ax1.set_title('Unique Basins vs Iter')
    ax1.grid(True, alpha=0.3)
    if any(cf_mask):
        ax1.legend(fontsize=7)
    
    ax2 = plt.subplot(4, 5, 2)
    ax2.plot(local_iters, match_ratios, 'o-', linewidth=2, markersize=5, color='green', zorder=2)
    if any(cf_mask):
        for i, is_cf in enumerate(cf_mask):
            if is_cf:
                ax2.plot(local_iters[i], match_ratios[i], 'r*', markersize=10, zorder=3)
    ax2.set_xlabel('Local Iter')
    ax2.set_ylabel('Match %')
    ax2.set_title('Match Final Basin %')
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(4, 5, 3)
    ax3.plot(local_iters, initial_costs, 'o-', label='Initial', linewidth=2, markersize=5, zorder=2)
    ax3.plot(local_iters, mean_final_costs, 's-', label='Final', linewidth=2, markersize=5, zorder=2)
    if any(cf_mask):
        for i, is_cf in enumerate(cf_mask):
            if is_cf:
                ax3.plot(local_iters[i], initial_costs[i], 'r*', markersize=10, zorder=3,
                        label='Cycle Finder' if i == next((j for j, cf in enumerate(cf_mask) if cf), 0) else '')
                ax3.plot(local_iters[i], mean_final_costs[i], 'r*', markersize=10, zorder=3)
    ax3.set_xlabel('Local Iter')
    ax3.set_ylabel('Cost')
    ax3.set_title('Cost: Initial vs Final')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(4, 5, 4)
    if hgs_cost:
        initial_gaps = [calculate_gap(r['initial_cost'], hgs_cost) for r in results_sorted]
        final_gaps = [calculate_gap(r['mean_final_cost'], hgs_cost) for r in results_sorted]
        ax4.plot(local_iters, initial_gaps, 'o-', label='Initial', linewidth=2, markersize=5, zorder=2)
        ax4.plot(local_iters, final_gaps, 's-', label='Final', linewidth=2, markersize=5, zorder=2)
        if any(cf_mask):
            for i, is_cf in enumerate(cf_mask):
                if is_cf:
                    ax4.plot(local_iters[i], initial_gaps[i], 'r*', markersize=10, zorder=3)
                    ax4.plot(local_iters[i], final_gaps[i], 'r*', markersize=10, zorder=3)
        ax4.set_xlabel('Local Iter')
        ax4.set_ylabel('Gap to HGS (%)')
        ax4.set_title('Gap to HGS')
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'HGS not available', transform=ax4.transAxes, ha='center', va='center', fontsize=10)
        ax4.set_title('Gap to HGS')
    
    # 5. Basin Frequency Heatmap (solutions x top basins)
    ax5_heatmap = plt.subplot(4, 5, 5)
    top_10_basins = sorted_basins[:10]
    top_10_hashes = [h for h, _ in top_10_basins]
    heatmap_data = []
    for result in results_sorted:
        row = [result['basin_counts'].get(h, 0) for h in top_10_hashes]
        heatmap_data.append(row)
    if heatmap_data:
        im = ax5_heatmap.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax5_heatmap.set_xlabel('Top 10 Basins')
        ax5_heatmap.set_ylabel('Local Iter')
        ax5_heatmap.set_title('Basin Frequency Heatmap')
        ax5_heatmap.set_xticks(range(len(top_10_hashes)))
        ax5_heatmap.set_xticklabels([h[:6] for h in top_10_hashes], rotation=45, ha='right', fontsize=6)
        ax5_heatmap.set_yticks(range(len(results_sorted)))
        ax5_heatmap.set_yticklabels(local_iters, fontsize=6)
        plt.colorbar(im, ax=ax5_heatmap, fraction=0.046)
    
    # Row 2: Initial cost relationships
    ax6 = plt.subplot(4, 5, 6)
    if len(initial_costs) > 1 and np.std(initial_costs) > 1e-10 and np.std(num_basins) > 1e-10:
        r, p = stats.pearsonr(initial_costs, num_basins)
        ax6.scatter(initial_costs, num_basins, s=50, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
        if any(cf_mask):
            for i, is_cf in enumerate(cf_mask):
                if is_cf:
                    ax6.plot(initial_costs[i], num_basins[i], 'r*', markersize=10, zorder=3,
                            label='Cycle Finder' if i == next((j for j, cf in enumerate(cf_mask) if cf), 0) else '')
        if np.std(num_basins) > 1e-10:
            slope, intercept, r_value, _, _ = stats.linregress(initial_costs, num_basins)
            x_line = np.linspace(min(initial_costs), max(initial_costs), 100)
            ax6.plot(x_line, slope * x_line + intercept, '--', color='red', linewidth=1.5, alpha=0.8, label=f'R²={r_value**2:.3f}')
        ax6.text(0.05, 0.95, f'r={r:.3f}\np={p:.3f}', transform=ax6.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=8)
        ax6.legend(fontsize=7)
    ax6.set_xlabel('Initial Cost')
    ax6.set_ylabel('Unique Basins')
    ax6.set_title('Initial Cost vs Basins')
    ax6.grid(True, alpha=0.3)
    
    ax7 = plt.subplot(4, 5, 7)
    if len(initial_costs) > 1 and np.std(initial_costs) > 1e-10 and np.std(match_ratios) > 1e-10:
        r, p = stats.pearsonr(initial_costs, match_ratios)
        ax7.scatter(initial_costs, match_ratios, s=50, alpha=0.6, color='green', edgecolors='black', linewidth=0.5)
        if any(cf_mask):
            for i, is_cf in enumerate(cf_mask):
                if is_cf:
                    ax7.plot(initial_costs[i], match_ratios[i], 'r*', markersize=10, zorder=3)
        if np.std(match_ratios) > 1e-10:
            slope, intercept, r_value, _, _ = stats.linregress(initial_costs, match_ratios)
            x_line = np.linspace(min(initial_costs), max(initial_costs), 100)
            ax7.plot(x_line, slope * x_line + intercept, '--', color='red', linewidth=1.5, alpha=0.8, label=f'R²={r_value**2:.3f}')
        ax7.text(0.05, 0.95, f'r={r:.3f}\np={p:.3f}', transform=ax7.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=8)
        ax7.legend(fontsize=7)
    ax7.set_xlabel('Initial Cost')
    ax7.set_ylabel('Match Ratio (%)')
    ax7.set_title('Initial Cost vs Match')
    ax7.grid(True, alpha=0.3)
    
    ax8 = plt.subplot(4, 5, 8)
    if len(initial_costs) > 1 and np.std(initial_costs) > 1e-10 and np.std(mean_final_costs) > 1e-10:
        r, p = stats.pearsonr(initial_costs, mean_final_costs)
        ax8.scatter(initial_costs, mean_final_costs, s=50, alpha=0.6, color='red', edgecolors='black', linewidth=0.5)
        if any(cf_mask):
            for i, is_cf in enumerate(cf_mask):
                if is_cf:
                    ax8.plot(initial_costs[i], mean_final_costs[i], 'r*', markersize=10, zorder=3)
        if np.std(mean_final_costs) > 1e-10:
            slope, intercept, r_value, _, _ = stats.linregress(initial_costs, mean_final_costs)
            x_line = np.linspace(min(initial_costs), max(initial_costs), 100)
            ax8.plot(x_line, slope * x_line + intercept, '--', color='red', linewidth=1.5, alpha=0.8, label=f'R²={r_value**2:.3f}')
        ax8.text(0.05, 0.95, f'r={r:.3f}\np={p:.3f}', transform=ax8.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=8)
        ax8.legend(fontsize=7)
    ax8.set_xlabel('Initial Cost')
    ax8.set_ylabel('Final Cost')
    ax8.set_title('Initial vs Final Cost')
    ax8.grid(True, alpha=0.3)
    
    # 9. Basin Diversity vs Local Iteration (Entropy折线图)
    ax9 = plt.subplot(4, 5, 9)
    entropies = []
    for result in results_sorted:
        counts = list(result['basin_counts'].values())
        if counts:
            total = sum(counts)
            probs = [c / total for c in counts]
            entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
            entropies.append(entropy)
        else:
            entropies.append(0)
    ax9.plot(local_iters, entropies, 'o-', linewidth=2, markersize=5, color='orange')
    if any(cf_mask):
        for i, is_cf in enumerate(cf_mask):
            if is_cf:
                ax9.plot(local_iters[i], entropies[i], 'r*', markersize=10, zorder=3)
    ax9.set_xlabel('Local Iter')
    ax9.set_ylabel('Basin Diversity (Entropy)')
    ax9.set_title('Basin Diversity vs Iter')
    ax9.grid(True, alpha=0.3)
    
    # 10. Basin Co-occurrence Graph
    ax10 = plt.subplot(4, 5, 10)
    basin_cooccurrence = defaultdict(float)
    basin_total_visits = defaultdict(int)
    
    for result in results:
        num_runs = result['num_runs']
        basin_freqs = {basin_hash: count / num_runs 
                      for basin_hash, count in result['basin_counts'].items()}
        
        for basin_hash, count in result['basin_counts'].items():
            basin_total_visits[basin_hash] += count
        
        basin_list = list(basin_freqs.keys())
        for i, basin1 in enumerate(basin_list):
            for basin2 in basin_list[i+1:]:
                freq1 = basin_freqs[basin1]
                freq2 = basin_freqs[basin2]
                cooccurrence_strength = np.sqrt(freq1 * freq2)
                basin_cooccurrence[(basin1, basin2)] += cooccurrence_strength
    
    G = nx.Graph()
    top_basins = sorted(basin_total_visits.items(), key=lambda x: x[1], reverse=True)[:15]
    basin_list = [h for h, _ in top_basins]
    
    for basin_hash in basin_list:
        G.add_node(basin_hash[:6])
    
    # Only process co-occurrence if there are multiple basins and co-occurrence data
    if basin_cooccurrence and len(basin_list) > 1:
        max_cooccurrence = max(basin_cooccurrence.values())
        for (basin1, basin2), strength in basin_cooccurrence.items():
            if basin1 in basin_list and basin2 in basin_list:
                normalized_strength = strength / max_cooccurrence
                if normalized_strength > 0.05:
                    G.add_edge(basin1[:6], basin2[:6], weight=normalized_strength)
    
    if len(G.nodes()) > 0:
        pos = nx.spring_layout(G, k=1.5, iterations=50)
        node_sizes = [basin_total_visits[h] * 2 for h in basin_list]
        
        if G.edges():
            edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]
            edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
        else:
            edge_widths = []
            edge_colors = []
        
        nx.draw_networkx_nodes(G, pos, ax=ax10, node_size=node_sizes, 
                              node_color='steelblue', alpha=0.7)
        if G.edges():
            nx.draw_networkx_edges(G, pos, ax=ax10, width=edge_widths, 
                                  alpha=0.5, edge_color=edge_colors, edge_cmap=plt.cm.Reds)
            edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' 
                          for u, v in G.edges() if G[u][v]['weight'] > 0.1}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax10, font_size=5, alpha=0.8)
        nx.draw_networkx_labels(G, pos, ax=ax10, font_size=6)
    else:
        ax10.text(0.5, 0.5, 'No basin co-occurrence data', 
                 ha='center', va='center', transform=ax10.transAxes, fontsize=10)
    
    ax10.set_title('Basin Co-occurrence\n(Edge = co-occurrence strength)')
    ax10.axis('off')
    
    # 11. Basin Cost Distribution
    ax11 = plt.subplot(4, 5, 11)
    basin_costs = [all_basins[h]['mean_cost'] for h, _ in sorted_basins]
    ax11.hist(basin_costs, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    ax11.set_xlabel('Basin Cost')
    ax11.set_ylabel('Count')
    ax11.set_title('Basin Cost Distribution')
    ax11.grid(True, alpha=0.3, axis='y')
    if hgs_cost:
        ax11.axvline(x=hgs_cost, color='red', linestyle='--', linewidth=1.5, label=f'HGS: {hgs_cost:.1f}')
        ax11.legend(fontsize=7)
    
    # 12. Basin Similarity (Top 20)
    ax12 = plt.subplot(4, 5, 12)
    if len(all_basins) >= 2:
        basin_list = list(all_basins.keys())[:20]
        n_basins = len(basin_list)
        similarity_matrix = np.zeros((n_basins, n_basins))
        for i, basin1 in enumerate(basin_list):
            for j, basin2 in enumerate(basin_list):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    edges1 = all_basins[basin1]['edges']
                    edges2 = all_basins[basin2]['edges']
                    similarity = len(edges1 & edges2) / len(edges1 | edges2) if len(edges1 | edges2) > 0 else 0.0
                    similarity_matrix[i, j] = similarity
        im = ax12.imshow(similarity_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax12.set_title('Basin Similarity (Top 20)')
        plt.colorbar(im, ax=ax12, fraction=0.046)
    else:
        ax12.text(0.5, 0.5, 'Not enough basins', transform=ax12.transAxes, ha='center', va='center', fontsize=10)
        ax12.set_title('Basin Similarity')
    
    # 13. Basin Clustering
    ax13 = plt.subplot(4, 5, 13)
    if len(all_basins) >= 2:
        basin_list = list(all_basins.keys())
        n_basins = min(len(basin_list), 30)
        basin_list = basin_list[:n_basins]
        distance_matrix = np.zeros((n_basins, n_basins))
        for i, basin1 in enumerate(basin_list):
            for j, basin2 in enumerate(basin_list):
                if i != j:
                    edges1 = all_basins[basin1]['edges']
                    edges2 = all_basins[basin2]['edges']
                    similarity = len(edges1 & edges2) / len(edges1 | edges2) if len(edges1 | edges2) > 0 else 0.0
                    distance_matrix[i, j] = 1.0 - similarity
        condensed_distances = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_distances, method='ward')
        dendrogram(linkage_matrix, ax=ax13, labels=[h[:6] for h in basin_list], 
                   leaf_font_size=6, no_labels=(n_basins > 15))
        ax13.set_title('Basin Clustering')
        ax13.set_xlabel('Basin')
        ax13.set_ylabel('Distance')
    else:
        ax13.text(0.5, 0.5, 'Not enough basins', transform=ax13.transAxes, ha='center', va='center', fontsize=10)
        ax13.set_title('Basin Clustering')
    
    # 14. Top 20 Basins (Across All Solutions)
    ax14 = plt.subplot(4, 5, 14)
    top_basins = sorted_basins[:20]
    basin_hashes = [h[:8] for h, _ in top_basins]
    basin_counts = [c for _, c in top_basins]
    ax14.barh(range(len(basin_hashes)), basin_counts, color='steelblue')
    ax14.set_yticks(range(len(basin_hashes)))
    ax14.set_yticklabels(basin_hashes, fontsize=6)
    ax14.set_xlabel('Total Frequency')
    ax14.set_ylabel('Basin Hash (first 8 chars)')
    ax14.set_title('Top 20 Basins (All Solutions)')
    ax14.invert_yaxis()
    ax14.grid(True, alpha=0.3, axis='x')
    
    # Row 3: Additional
    # 15. Basin Frequency Distribution vs Local Iteration (boxplot)
    ax15 = plt.subplot(4, 5, 15)
    basin_counts_per_solution = [list(r['basin_counts'].values()) for r in results_sorted]
    positions = local_iters
    if basin_counts_per_solution and any(len(counts) > 0 for counts in basin_counts_per_solution):
        bp = ax15.boxplot(basin_counts_per_solution, positions=positions, widths=0.6, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
    ax15.set_xlabel('Local Iter')
    ax15.set_ylabel('Basin Frequency')
    ax15.set_title('Basin Frequency Distribution')
    ax15.grid(True, alpha=0.3, axis='y')
    
    # 16. Average Basin Frequency vs Local Iteration
    ax16 = plt.subplot(4, 5, 16)
    avg_basin_freqs = []
    for result in results_sorted:
        if result['basin_counts']:
            avg_freq = np.mean(list(result['basin_counts'].values()))
            avg_basin_freqs.append(avg_freq)
        else:
            avg_basin_freqs.append(0)
    ax16.plot(local_iters, avg_basin_freqs, 'o-', linewidth=2, markersize=5, color='purple')
    if any(cf_mask):
        for i, is_cf in enumerate(cf_mask):
            if is_cf:
                ax16.plot(local_iters[i], avg_basin_freqs[i], 'r*', markersize=10, zorder=3)
    ax16.set_xlabel('Local Iter')
    ax16.set_ylabel('Average Basin Frequency')
    ax16.set_title('Avg Basin Frequency vs Iter')
    ax16.grid(True, alpha=0.3)
    
    # 17. Entropy vs Gap/Cost (scatter plot, colored by basin count)
    ax17 = plt.subplot(4, 5, 17)
    entropy_data = []
    for result in results_sorted:
        counts = list(result['basin_counts'].values())
        if counts:
            total = sum(counts)
            probs = [c / total for c in counts]
            entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
        else:
            entropy = 0
        entropy_data.append({
            'cost': result['initial_cost'],
            'gap': calculate_gap(result['initial_cost'], hgs_cost) if hgs_cost else None,
            'entropy': entropy,
            'num_basins': result['num_unique_basins'],
        })
    
    if entropy_data:
        if hgs_cost and all(d['gap'] is not None for d in entropy_data):
            x_values = np.array([d['gap'] for d in entropy_data])
            x_label = 'Gap to HGS (%)'
        else:
            x_values = np.array([d['cost'] for d in entropy_data])
            x_label = 'Initial Solution Cost'
        
        entropies_arr = np.array([d['entropy'] for d in entropy_data])
        num_basins_arr = np.array([d['num_basins'] for d in entropy_data])
        
        scatter = ax17.scatter(x_values, entropies_arr, s=50, alpha=0.6, c=num_basins_arr, 
                               cmap='viridis', edgecolors='black', linewidth=0.5)
        if len(x_values) > 1 and np.std(x_values) > 1e-10 and np.std(entropies_arr) > 1e-10:
            r, p = stats.pearsonr(x_values, entropies_arr)
            slope, intercept, r_value, _, _ = stats.linregress(x_values, entropies_arr)
            x_line = np.linspace(x_values.min(), x_values.max(), 100)
            ax17.plot(x_line, slope * x_line + intercept, '--', color='red', linewidth=1.5, alpha=0.8, 
                      label=f'R²={r_value**2:.3f}')
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            ax17.text(0.05, 0.95, f'r={r:.3f}, p={p:.4f}{sig}', transform=ax17.transAxes, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                     fontsize=7)
            ax17.legend(fontsize=6)
        ax17.set_xlabel(x_label)
        ax17.set_ylabel('Basin Entropy (bits)')
        ax17.set_title('Entropy vs Gap/Cost\n(colored by # basins)')
        ax17.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax17, fraction=0.046, label='# Basins')
    
    # 18. Basin Transition Graph
    ax18 = plt.subplot(4, 5, 18)
    try:
        basin_solutions = defaultdict(set)
        all_basins_set = set()
        for result in results:
            solution_key = (result['global_iter'], result['local_iter'])
            for edges_hash in result['basin_counts'].keys():
                basin_solutions[edges_hash].add(solution_key)
                all_basins_set.add(edges_hash)
        
        if len(all_basins_set) >= 2:
            G = nx.Graph()
            basin_list = list(all_basins_set)[:15]
            for basin_hash in basin_list:
                G.add_node(basin_hash[:6])
            
            for i, basin1 in enumerate(basin_list):
                for basin2 in basin_list[i+1:]:
                    solutions1 = basin_solutions[basin1]
                    solutions2 = basin_solutions[basin2]
                    similarity = len(solutions1 & solutions2) / len(solutions1 | solutions2) if len(solutions1 | solutions2) > 0 else 0.0
                    if similarity > 0.1:
                        G.add_edge(basin1[:6], basin2[:6], weight=similarity)
            
            if len(G.edges) > 0:
                pos = nx.spring_layout(G, k=1, iterations=30)
                node_sizes = [len(basin_solutions[h]) * 50 for h in basin_list]
                edge_widths = [G[u][v]['weight'] * 3 for u, v in G.edges()]
                nx.draw_networkx_nodes(G, pos, ax=ax18, node_size=node_sizes, node_color='steelblue', alpha=0.7)
                nx.draw_networkx_edges(G, pos, ax=ax18, width=edge_widths, alpha=0.4, edge_color='gray')
                nx.draw_networkx_labels(G, pos, ax=ax18, font_size=6)
                # Add edge weight labels
                edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in G.edges()}
                nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax18, font_size=5, alpha=0.8)
                ax18.set_title('Basin Transition Graph\n(Edge labels = similarity)')
            else:
                ax18.text(0.5, 0.5, 'No transitions', transform=ax18.transAxes, ha='center', va='center', fontsize=10)
                ax18.set_title('Basin Transition Graph')
        else:
            ax18.text(0.5, 0.5, 'Not enough basins', transform=ax18.transAxes, ha='center', va='center', fontsize=10)
            ax18.set_title('Basin Transition Graph')
    except:
        ax18.text(0.5, 0.5, 'Graph unavailable', transform=ax18.transAxes, ha='center', va='center', fontsize=10)
        ax18.set_title('Basin Transition Graph')
    ax18.axis('off')
    
    # 19. Gap vs Dominant Basin Frequency
    ax19 = plt.subplot(4, 5, 19)
    if hgs_cost:
        gaps, dominant_freqs = [], []
        for result in results:
            if not result['basin_counts']:
                continue
            frequencies = np.array(list(result['basin_counts'].values())) / result['num_runs']
            gaps.append(calculate_gap(result['initial_cost'], hgs_cost))
            dominant_freqs.append(np.max(frequencies))
        
        if len(gaps) >= 3:
            gaps_arr = np.array(gaps)
            dominant_freqs_arr = np.array(dominant_freqs)
            sort_idx = np.argsort(gaps_arr)
            gaps_arr = gaps_arr[sort_idx]
            dominant_freqs_arr = dominant_freqs_arr[sort_idx]
            
            # Find critical gap G:
            # largest gap such that for all solutions with gap <= G,
            # dominant basin frequency >= stability_threshold.
            stability_threshold = 0.8
            critical_gap = None
            if len(gaps_arr) > 0:
                meets_threshold = dominant_freqs_arr >= stability_threshold
                # prefix_all_good[i] == True  <=> all gaps <= gaps_arr[i] satisfy the threshold
                prefix_all_good = np.logical_and.accumulate(meets_threshold)
                valid_indices = np.where(prefix_all_good)[0]
                if len(valid_indices) > 0:
                    critical_gap = gaps_arr[valid_indices[-1]]
            
            ax19.scatter(gaps_arr, dominant_freqs_arr * 100, s=50, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
            if critical_gap is not None:
                ax19.axvline(x=critical_gap, color='red', linestyle='--', linewidth=2, label=f'Critical = {critical_gap:.2f}%')
                stable_mask = gaps_arr <= critical_gap
                ax19.scatter(gaps_arr[stable_mask], dominant_freqs_arr[stable_mask] * 100, s=100, alpha=0.8, color='green', marker='*', zorder=5)
            ax19.axhline(y=stability_threshold * 100, color='orange', linestyle=':', linewidth=1.5)
            if critical_gap is not None:
                ax19.legend(fontsize=7)
            ax19.set_xlabel('Gap to HGS (%)')
            ax19.set_ylabel('Dominant Basin Frequency (%)')
            ax19.set_title('Gap vs Dominant Basin Freq')
            ax19.grid(True, alpha=0.3)
        else:
            ax19.text(0.5, 0.5, 'Not enough data', transform=ax19.transAxes, ha='center', va='center', fontsize=10)
            ax19.set_title('Gap vs Dominant Basin Freq')
    else:
        ax19.text(0.5, 0.5, 'HGS not available', transform=ax19.transAxes, ha='center', va='center', fontsize=10)
        ax19.set_title('Gap vs Dominant Basin Freq')
    
    # 20. Cumulative Mean Dominant Frequency
    ax20 = plt.subplot(4, 5, 20)
    if hgs_cost:
        gaps, dominant_freqs = [], []
        for result in results:
            if not result['basin_counts']:
                continue
            frequencies = np.array(list(result['basin_counts'].values())) / result['num_runs']
            gaps.append(calculate_gap(result['initial_cost'], hgs_cost))
            dominant_freqs.append(np.max(frequencies))
        
        if len(gaps) >= 3:
            gaps_arr = np.array(gaps)
            dominant_freqs_arr = np.array(dominant_freqs)
            sort_idx = np.argsort(gaps_arr)
            gaps_arr = gaps_arr[sort_idx]
            dominant_freqs_arr = dominant_freqs_arr[sort_idx]
            
            stability_threshold = 0.8
            critical_gap = None
            if len(gaps_arr) > 0:
                meets_threshold = dominant_freqs_arr >= stability_threshold
                prefix_all_good = np.logical_and.accumulate(meets_threshold)
                valid_indices = np.where(prefix_all_good)[0]
                if len(valid_indices) > 0:
                    critical_gap = gaps_arr[valid_indices[-1]]
            
            gap_thresholds = np.linspace(gaps_arr.min(), gaps_arr.max(), 50)
            mean_freqs = [np.mean(dominant_freqs_arr[gaps_arr <= t]) for t in gap_thresholds]
            ax20.plot(gap_thresholds, np.array(mean_freqs) * 100, 'o-', linewidth=2, markersize=4)
            if critical_gap is not None:
                ax20.axvline(x=critical_gap, color='red', linestyle='--', linewidth=2)
            ax20.axhline(y=stability_threshold * 100, color='orange', linestyle=':', linewidth=1.5)
            ax20.set_xlabel('Gap Threshold (%)')
            ax20.set_ylabel('Mean Dominant Freq (%)')
            ax20.set_title('Cumulative Mean Dominant Freq')
            ax20.grid(True, alpha=0.3)
        else:
            ax20.text(0.5, 0.5, 'Not enough data', transform=ax20.transAxes, ha='center', va='center', fontsize=10)
            ax20.set_title('Cumulative Mean Dominant Freq')
    else:
        ax20.text(0.5, 0.5, 'HGS not available', transform=ax20.transAxes, ha='center', va='center', fontsize=10)
        ax20.set_title('Cumulative Mean Dominant Freq')

    
    plt.suptitle(f'Comprehensive Basin Analysis (run_id={run_id}, trial_id={trial_id})', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_file = os.path.join(basin_dir, f'comprehensive_analysis_run_{run_id}_trial_{trial_id}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Comprehensive visualization saved to: {output_file}")
    plt.close(fig)


def create_cross_trial_visualizations(all_trial_results, run_id, basin_dir):
    """Create visualizations across multiple trials for the same run_id."""
    os.makedirs(basin_dir, exist_ok=True)
    
    if not all_trial_results:
        return
    
    # Organize data by trial_id
    trial_data = {}
    for trial_id, results in all_trial_results.items():
        if not results:
            continue
        # Aggregate metrics per trial (use final solution's metrics)
        final_result = max(results, key=lambda x: x['local_iter'])
        trial_data[trial_id] = {
            'num_unique_basins': final_result['num_unique_basins'],
            'match_final_ratio': final_result['match_final_ratio'],
            'mean_final_cost': final_result['mean_final_cost'],
        }
    
    if not trial_data:
        return
    
    trial_ids = sorted(trial_data.keys())
    
    # Create figure
    fig = plt.figure(figsize=(18, 6))
    
    # 1. Unique basin count vs trial_id
    ax1 = plt.subplot(1, 3, 1)
    num_basins_by_trial = [trial_data[tid]['num_unique_basins'] for tid in trial_ids]
    ax1.plot(trial_ids, num_basins_by_trial, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('Trial ID')
    ax1.set_ylabel('Number of Unique Basins')
    ax1.set_title(f'Unique Basins vs Trial ID (run_id={run_id})')
    ax1.grid(True, alpha=0.3)
    
    # 2. Match ratio vs trial_id
    ax2 = plt.subplot(1, 3, 2)
    match_ratios_by_trial = [trial_data[tid]['match_final_ratio'] * 100 for tid in trial_ids]
    ax2.plot(trial_ids, match_ratios_by_trial, 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Trial ID')
    ax2.set_ylabel('Match Final Basin Ratio (%)')
    ax2.set_title(f'Match Ratio vs Trial ID (run_id={run_id})')
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3)
    
    # 3. Final basin cost vs trial_id
    ax3 = plt.subplot(1, 3, 3)
    final_costs_by_trial = [trial_data[tid]['mean_final_cost'] for tid in trial_ids]
    ax3.plot(trial_ids, final_costs_by_trial, 'o-', linewidth=2, markersize=8, color='red')
    ax3.set_xlabel('Trial ID')
    ax3.set_ylabel('Mean Final Basin Cost')
    ax3.set_title(f'Final Basin Cost vs Trial ID (run_id={run_id})')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = os.path.join(basin_dir, f'cross_trial_analysis_run_{run_id}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nCross-trial visualization saved to: {output_file}")
    plt.close(fig)


def find_critical_gap_for_basin_stability(results, basin_dir, run_id, trial_id, hgs_cost, 
                                         stability_threshold=0.8):
    """Find critical gap value: when gap is below this value, basin concentration significantly increases.
    Visualization is now included in create_comprehensive_visualization."""
    if not results or not hgs_cost:
        return None
    
    gaps, dominant_freqs = [], []
    for result in results:
        if not result['basin_counts']:
            continue
        frequencies = np.array(list(result['basin_counts'].values())) / result['num_runs']
        gaps.append(calculate_gap(result['initial_cost'], hgs_cost))
        dominant_freqs.append(np.max(frequencies))
    
    if len(gaps) < 3:
        return None
    
    gaps, dominant_freqs = np.array(gaps), np.array(dominant_freqs)
    sort_idx = np.argsort(gaps)
    gaps, dominant_freqs = gaps[sort_idx], dominant_freqs[sort_idx]
    
    critical_gap = None
    for i in range(len(gaps)):
        if np.all(dominant_freqs[i:] >= stability_threshold):
            critical_gap = gaps[i]
            break
    
    if critical_gap is not None:
        stable_count = np.sum(gaps <= critical_gap)
        print(f"Critical Gap: {critical_gap:.2f}% (solutions: {stable_count}/{len(gaps)})")
    
    # Return critical gap results for Excel export
    critical_gap_results = {
        'critical_gap': critical_gap,
        'stability_threshold': stability_threshold,
        'stable_count': np.sum(gaps <= critical_gap) if critical_gap is not None else 0,
        'total_count': len(gaps)
    }
    return critical_gap_results


def analyze_basin_entropy(results, basin_dir, run_id, trial_id, hgs_cost=None):
    """Basin Entropy Analysis: H(S_int) = -Σ p(i) log p(i), where p(i) is frequency of falling into i-th Basin.
    Visualization is now included in create_comprehensive_visualization."""
    if not results:
        return None
    
    entropy_data = []
    for result in results:
        if not result['basin_counts']:
            continue
        frequencies = np.array(list(result['basin_counts'].values())) / result['num_runs']
        freq_nonzero = frequencies[frequencies > 0]
        entropy = -np.sum(freq_nonzero * np.log2(freq_nonzero)) if len(freq_nonzero) > 0 else 0
        entropy_data.append({
            'cost': result['initial_cost'],
            'gap': calculate_gap(result['initial_cost'], hgs_cost) if hgs_cost else None,
            'entropy': entropy,
            'num_basins': result['num_unique_basins'],
        })
    
    if not entropy_data:
        return None
    
    entropies = np.array([d['entropy'] for d in entropy_data])
    
    if hgs_cost and all(d['gap'] is not None for d in entropy_data):
        x_values = np.array([d['gap'] for d in entropy_data])
        x_title = 'Gap'
    else:
        x_values = np.array([d['cost'] for d in entropy_data])
        x_title = 'Cost'
    
    r, p = stats.pearsonr(x_values, entropies) if (len(x_values) > 1 and np.std(x_values) > 1e-10 and np.std(entropies) > 1e-10) else (0, 1)
    print(f"\nBasin Entropy Analysis: {x_title} vs Entropy: r={r:.3f}, p={p:.4f}")
    print(f"  Mean: {np.mean(entropies):.3f}, Min: {np.min(entropies):.3f}, Max: {np.max(entropies):.3f} bits")
    
    # Return entropy data for Excel export
    entropy_results = {
        'entropy_data': entropy_data,
        'correlation': {'r': r, 'p': p, 'x_title': x_title},
        'statistics': {'mean': np.mean(entropies), 'min': np.min(entropies), 'max': np.max(entropies)}
    }
    return entropy_results


def save_results_to_excel(results, run_id, trial_id, basin_dir, final_edges_hash, hgs_cost,
                         entropy_results=None, critical_gap_results=None):
    """Save detailed results to Excel file with multiple sheets."""
    os.makedirs(basin_dir, exist_ok=True)
    excel_file = os.path.join(basin_dir, f'basin_analysis_run_{run_id}_trial_{trial_id}.xlsx')
    
    # Prepare summary data
    summary_data = []
    entropy_map = {}
    if entropy_results and 'entropy_data' in entropy_results:
        for ed in entropy_results['entropy_data']:
            key = (ed['cost'], ed.get('gap'))
            entropy_map[key] = ed['entropy']
    
    critical_gap = critical_gap_results.get('critical_gap') if critical_gap_results else None
    
    for result in results:
        entropy = entropy_map.get((result['initial_cost'], calculate_gap(result['initial_cost'], hgs_cost) if hgs_cost else None), None)
        initial_gap = calculate_gap(result['initial_cost'], hgs_cost) if hgs_cost else None
        is_stable = 'Yes' if (critical_gap is not None and initial_gap is not None and initial_gap <= critical_gap) else 'No'
        
        summary_data.append({
            'run_id': run_id,
            'trial_id': trial_id,
            'global_iter': result['global_iter'],
            'local_iter': result['local_iter'],
            'initial_cost': result['initial_cost'],
            'initial_edges_hash': result['initial_edges_hash'] if result['initial_edges_hash'] else '',
            'is_cycle_finder': 'Yes' if result.get('is_cycle_finder', False) else 'No',
            'num_unique_basins': result['num_unique_basins'],
            'mean_final_cost': result['mean_final_cost'],
            'matches_final': result['matches_final'],
            'match_final_ratio': f"{result['match_final_ratio']*100:.2f}%",
            'num_runs': result['num_runs'],
            'basin_entropy': f"{entropy:.3f}" if entropy is not None else '',
            'is_stable_region': is_stable if hgs_cost and critical_gap is not None else '',
        })
        if hgs_cost:
            summary_data[-1]['initial_gap_to_hgs'] = f"{initial_gap:.2f}%"
            summary_data[-1]['final_gap_to_hgs'] = f"{calculate_gap(result['mean_final_cost'], hgs_cost):.2f}%"
    
    summary_df = pd.DataFrame(summary_data)
    
    # Calculate correlation statistics first (for the entire trial)
    results_sorted = sorted(results, key=lambda x: (x['global_iter'], x['local_iter']))
    initial_costs = np.array([r['initial_cost'] for r in results_sorted])
    num_basins = np.array([r['num_unique_basins'] for r in results_sorted])
    match_ratios = np.array([r['match_final_ratio'] * 100 for r in results_sorted])
    final_costs = np.array([r['mean_final_cost'] for r in results_sorted])
    
    # Calculate correlation statistics
    corr_stats = {}
    if len(initial_costs) > 1 and np.std(initial_costs) > 1e-10:
        # Initial cost vs unique basin count
        r1, p1 = stats.pearsonr(initial_costs, num_basins) if np.std(num_basins) > 1e-10 else (0, 1)
        if np.std(num_basins) > 1e-10:
            slope1, intercept1, r_value1, p_val1, std_err1 = stats.linregress(initial_costs, num_basins)
        else:
            slope1, intercept1, r_value1, p_val1, std_err1 = 0, np.mean(num_basins), 0, 1, 0
        corr_stats['cost_vs_basins'] = {
            'pearson_r': r1, 'p_value': p1, 'r_squared': r_value1**2,
            'slope': slope1, 'intercept': intercept1, 'std_err': std_err1,
            'significance': '***' if p1 < 0.001 else ('**' if p1 < 0.01 else ('*' if p1 < 0.05 else 'ns')),
        }
        
        # Initial cost vs match ratio
        r2, p2 = stats.pearsonr(initial_costs, match_ratios) if np.std(match_ratios) > 1e-10 else (0, 1)
        if np.std(match_ratios) > 1e-10:
            slope2, intercept2, r_value2, p_val2, std_err2 = stats.linregress(initial_costs, match_ratios)
        else:
            slope2, intercept2, r_value2, p_val2, std_err2 = 0, np.mean(match_ratios), 0, 1, 0
        corr_stats['cost_vs_match'] = {
            'pearson_r': r2, 'p_value': p2, 'r_squared': r_value2**2,
            'slope': slope2, 'intercept': intercept2, 'std_err': std_err2,
            'significance': '***' if p2 < 0.001 else ('**' if p2 < 0.01 else ('*' if p2 < 0.05 else 'ns')),
        }
        
        # Initial cost vs final basin cost
        r3, p3 = stats.pearsonr(initial_costs, final_costs) if np.std(final_costs) > 1e-10 else (0, 1)
        if np.std(final_costs) > 1e-10:
            slope3, intercept3, r_value3, p_val3, std_err3 = stats.linregress(initial_costs, final_costs)
        else:
            slope3, intercept3, r_value3, p_val3, std_err3 = 0, np.mean(final_costs), 0, 1, 0
        corr_stats['cost_vs_final'] = {
            'pearson_r': r3, 'p_value': p3, 'r_squared': r_value3**2,
            'slope': slope3, 'intercept': intercept3, 'std_err': std_err3,
            'significance': '***' if p3 < 0.001 else ('**' if p3 < 0.01 else ('*' if p3 < 0.05 else 'ns')),
        }
    
    # Prepare detailed basin data (merged with frequency ranking)
    basin_details = []
    for result in results:
        # Calculate rank for each basin in this solution
        basin_items = sorted(result['basin_counts'].items(), key=lambda x: x[1], reverse=True)
        rank_map = {edges_hash: rank for rank, (edges_hash, _) in enumerate(basin_items, 1)}
        
        for edges_hash, count in result['basin_counts'].items():
            basin_info = result['basin_data'][edges_hash]
            costs = basin_info['costs']
            
            if costs and len(costs) > 0:
                mean_cost = np.mean(costs)
                std_cost = np.std(costs)
                min_cost = np.min(costs)
                max_cost = np.max(costs)
            else:
                mean_cost = result['mean_final_cost']
                std_cost = 0.0
                min_cost = mean_cost
                max_cost = mean_cost
            
            detail = {
                'run_id': run_id,
                'trial_id': trial_id,
                'source_global_iter': result['global_iter'],
                'source_local_iter': result['local_iter'],
                'source_initial_cost': result['initial_cost'],
                'source_is_cycle_finder': 'Yes' if result.get('is_cycle_finder', False) else 'No',
                'basin_rank': rank_map.get(edges_hash, ''),
                'basin_edges_hash': edges_hash if edges_hash else '',
                'basin_frequency': count,
                'frequency_percent': f"{count/result['num_runs']*100:.2f}%",
                'mean': mean_cost,
                'std': std_cost,
                'min': min_cost,
                'max': max_cost,
                'is_final_basin': 'Yes' if (final_edges_hash and edges_hash == final_edges_hash) else 'No',
            }
            
            # Add correlation statistics (same for all basins in the same trial)
            if corr_stats:
                detail['corr_cost_vs_basins_r'] = corr_stats['cost_vs_basins']['pearson_r']
                detail['corr_cost_vs_basins_p'] = corr_stats['cost_vs_basins']['p_value']
                detail['corr_cost_vs_basins_r2'] = corr_stats['cost_vs_basins']['r_squared']
                detail['corr_cost_vs_basins_sig'] = corr_stats['cost_vs_basins']['significance']
                
                detail['corr_cost_vs_match_r'] = corr_stats['cost_vs_match']['pearson_r']
                detail['corr_cost_vs_match_p'] = corr_stats['cost_vs_match']['p_value']
                detail['corr_cost_vs_match_r2'] = corr_stats['cost_vs_match']['r_squared']
                detail['corr_cost_vs_match_sig'] = corr_stats['cost_vs_match']['significance']
                
                detail['corr_cost_vs_final_r'] = corr_stats['cost_vs_final']['pearson_r']
                detail['corr_cost_vs_final_p'] = corr_stats['cost_vs_final']['p_value']
                detail['corr_cost_vs_final_r2'] = corr_stats['cost_vs_final']['r_squared']
                detail['corr_cost_vs_final_sig'] = corr_stats['cost_vs_final']['significance']
            else:
                detail['corr_cost_vs_basins_r'] = None
                detail['corr_cost_vs_basins_p'] = None
                detail['corr_cost_vs_basins_r2'] = None
                detail['corr_cost_vs_basins_sig'] = None
                detail['corr_cost_vs_match_r'] = None
                detail['corr_cost_vs_match_p'] = None
                detail['corr_cost_vs_match_r2'] = None
                detail['corr_cost_vs_match_sig'] = None
                detail['corr_cost_vs_final_r'] = None
                detail['corr_cost_vs_final_p'] = None
                detail['corr_cost_vs_final_r2'] = None
                detail['corr_cost_vs_final_sig'] = None
            
            # Add entropy and critical gap correlation statistics
            if entropy_results:
                corr = entropy_results.get('correlation', {})
                detail['corr_entropy_r'] = corr.get('r', '')
                detail['corr_entropy_p'] = corr.get('p', '')
                detail['corr_entropy_metric'] = f"{corr.get('x_title', 'Cost')} vs Entropy"
            
            if critical_gap_results and critical_gap_results.get('critical_gap') is not None:
                detail['critical_gap'] = critical_gap_results.get('critical_gap', '')
                detail['stability_threshold'] = critical_gap_results.get('stability_threshold', '')
            
            if hgs_cost:
                detail['basin_gap_to_hgs'] = f"{calculate_gap(mean_cost, hgs_cost):.2f}%"
            
            basin_details.append(detail)
    
    basin_df = pd.DataFrame(basin_details)
    
    # Write to Excel with multiple sheets
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        basin_df.to_excel(writer, sheet_name='Basin Details', index=False)
        
        # Auto-adjust column widths
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for idx, col in enumerate(worksheet.columns, 1):
                max_length = 0
                column = col[0].column_letter
                for cell in col:
                    if cell.value and len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column].width = adjusted_width
    
    print(f"\nExcel results saved to: {excel_file}")
    print(f"  Summary: {len(summary_df)} rows (with entropy and stability info)")
    print(f"  Basin Details: {len(basin_df)} rows (with frequency ranking, stats, and correlation)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze basin structure')
    parser.add_argument('--pkl', type=str, dest='instance_path', default="/home/jieyi/cvrp100_uniform.pkl",
                        help='Path to instance .pkl file')
    parser.add_argument('--idx', type=int, default=0, dest='instance_index',
                        help='Instance index')
    parser.add_argument('--run_id', type=str, dest='run_id',
                        help='Run ID (if not specified, analyze all runs)')
    parser.add_argument('--trial_id', type=int, dest='trial_id',
                        help='Trial ID (if not specified, analyze all trials)')
    parser.add_argument('--all', action='store_true', dest='analyze_all',
                        help='Analyze all run_id and trial_id combinations')
    parser.add_argument('--num_runs', type=int, default=100, dest='num_runs',
                        help='Number of local search runs per solution (default: 100)')
    parser.add_argument('--basin_dir', type=str, default="basin_datasets0", dest='basin_base_dir',
                        help='Base directory for basin datasets')
    parser.add_argument('--hgs', type=str, dest='hgs_solution_path', default="/home/jieyi/hgs_cvrp100_uniform.pkl",
                        help='Path to HGS solution .pkl file for gap comparison')
    parser.add_argument('--force', action='store_true', dest='force_rerun',
                        help='Force re-run even if results already exist (default: skip already analyzed pairs)')
    
    args = parser.parse_args()
    
    basin_paths = get_basin_paths(args.instance_path, args.instance_index, args.basin_base_dir)
    trajectory_path = basin_paths['trajectory_path']
    run_trial_pairs = get_all_run_trial_pairs(trajectory_path)
    
    # Determine which pairs to analyze
    if args.analyze_all or (args.run_id is None and args.trial_id is None):
        pairs_to_analyze = run_trial_pairs
    elif args.run_id is None:
        first_run_id = run_trial_pairs[0][0] if run_trial_pairs else None
        pairs_to_analyze = [(r, t) for r, t in run_trial_pairs if r == first_run_id] if first_run_id else []
    elif args.trial_id is None:
        pairs_to_analyze = [(r, t) for r, t in run_trial_pairs if r == args.run_id]
    else:
        pairs_to_analyze = [(args.run_id, args.trial_id)]
    
    if not pairs_to_analyze:
        print("No (run_id, trial_id) pairs found to analyze")
        sys.exit(0)
    
    # Filter out already analyzed pairs (unless --force is specified)
    # If a trial is partially completed, clean up the partial data before re-running
    pairs_to_analyze_filtered = []
    skipped_count = 0
    for run_id, trial_id in pairs_to_analyze:
        if not args.force_rerun and is_trial_already_analyzed(args.instance_path, args.instance_index, run_id, trial_id, args.basin_base_dir):
            print(f"Skipping run_id={run_id}, trial_id={trial_id} (already analyzed)")
            skipped_count += 1
        else:
            pairs_to_analyze_filtered.append((run_id, trial_id))
    
    print(f"\nTotal pairs: {len(pairs_to_analyze)}, Already analyzed: {skipped_count}, To analyze: {len(pairs_to_analyze_filtered)}")
    
    if not pairs_to_analyze_filtered:
        print("All pairs have already been analyzed. Exiting.")
        sys.exit(0)
    
    print(f"Analyzing {len(pairs_to_analyze_filtered)} (run_id, trial_id) combinations")
    
    # Collect results by run_id for cross-trial analysis
    all_results_by_run = defaultdict(dict)  # run_id -> {trial_id: results}
    
    for run_id, trial_id in pairs_to_analyze_filtered:
        print(f"\n{'='*80}")
        print(f"Analyzing run_id={run_id}, trial_id={trial_id}")
        print(f"{'='*80}")
        results = analyze_trial_basins(
            args.instance_path, args.instance_index, run_id, trial_id,
            args.num_runs, args.basin_base_dir, args.hgs_solution_path)
        if results:
            # Store results but clear large data structures to save memory
            # Only keep essential metadata for cross-trial visualization
            results_lightweight = []
            for r in results:
                # Clear basin_data to save memory (already saved to disk)
                if 'basin_data' in r:
                    for basin_info in r['basin_data'].values():
                        if 'solution_flat' in basin_info and basin_info['solution_flat']:
                            # Keep solution_flat but clear if too large
                            if len(basin_info['solution_flat']) > 1000:
                                basin_info['solution_flat'] = None
                    # Don't store full basin_data, only keep counts
                    r_light = {k: v for k, v in r.items() if k != 'basin_data'}
                    r_light['basin_counts'] = r.get('basin_counts', {})
                else:
                    r_light = r
                results_lightweight.append(r_light)
            
            all_results_by_run[run_id][trial_id] = results_lightweight
            
            # Cleanup original results after creating lightweight version
            del results
            aggressive_gc_cleanup()
    
    # Create cross-trial visualizations for each run_id
    for run_id, trial_results in all_results_by_run.items():
        if len(trial_results) > 1:  # Only create cross-trial plots if multiple trials
            basin_paths = get_basin_paths(args.instance_path, args.instance_index, args.basin_base_dir)
            basin_dir = os.path.join("basin_datasets0_analyze", basin_paths['instance_id'])
            create_cross_trial_visualizations(trial_results, run_id, basin_dir)
            
            # Cleanup after visualization
            del trial_results
            aggressive_gc_cleanup()
    
    # Final cleanup
    del all_results_by_run
    aggressive_gc_cleanup()