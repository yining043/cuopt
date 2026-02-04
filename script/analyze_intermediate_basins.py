#!/usr/bin/env python3
"""
Analyze basins from intermediate solutions in pybind runs.
For each trial's initial solution:
1. Run pybind once, recording intermediate solutions + candidate_nodes
2. For each intermediate solution, use it as initial + set candidate_nodes, run 100 times
3. Compare basins with original cuOpt basins and count unique basins
"""
import json
import os
import sys
import random
import gc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import datetime
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import networkx as nx

# Add the cuOpt build directory to Python path
cuopt_build_path = os.path.join(os.path.dirname(__file__), 'cpp', 'build', 'install', 'lib', 'python3', 'dist-packages')
if cuopt_build_path not in sys.path:
    sys.path.insert(0, cuopt_build_path)

from test_basin_pybind import (
    create_vrp_instance_from_pkl, validate_solution_feasibility
)
from test_load_data import load_hgs_solution_from_pkl, calculate_gap
from utils import (
    load_solution_from_trajectory, get_basin_paths,
    solution_flat_to_routes, routes_to_edges, edges_hash, routes_to_solution_flat
)
from analyze_basin import (
    run_local_search_silent,
    create_comprehensive_visualization,
    analyze_basin_entropy,
    find_critical_gap_for_basin_stability,
    save_results_to_excel
)

from compare_pybind_cuopt import visualize_solutions, plot_cost_curve_by_trial_pybind

# Try to import torch for CUDA cache clearing (optional)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def ensure_cuda_device():
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.set_device(0)


def aggressive_gc_cleanup():
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_local_search_with_intermediate_states(cuopt_env, initial_routes, vrp_instance, max_iterations=100, global_iter_start=0):
    """Run local search and record intermediate solutions + candidate_nodes at each step.
    
    Args:
        cuopt_env: CUDA optimization environment
        initial_routes: Initial solution routes
        vrp_instance: VRP instance dictionary
        max_iterations: Maximum number of iterations
        global_iter_start: Starting global_iter value (default: 0)
    
    Returns:
        intermediate_states: List of intermediate states with global_iter
        final_cost: Final cost
        final_edges_hash: Final edges hash
        final_edges: Final edges
        final_routes: Final routes
    """
    cuopt_env.initialize_search(initial_routes)
    weights = [10000., 10000., 100., 1000., 1000., 1000., 10000., 10000., 10000.]
    cuopt_env.set_weights(weights)
    cuopt_env.set_selection_weights(weights)
    cuopt_env.acquire_resource()
    cuopt_env.reset_move_candidates()
    cuopt_env.set_routes_to_search()
    cuopt_env.sync_streams()
    
    intermediate_states = []
    global_iter = global_iter_start  # Start from provided value
    
    # Extract nodes to search first to initialize candidate nodes
    cuopt_env.extract_nodes_to_search()
    
    # Record initial state
    initial_routes_copy = cuopt_env.get_solution_routes().copy()
    initial_cost = cuopt_env.get_cost()
    initial_edges = routes_to_edges(initial_routes_copy)
    initial_edges_hash = edges_hash(initial_edges)
    initial_candidates = cuopt_env.get_move_candidates()
    
    local_iter = 0
    intermediate_states.append({
        'local_iter': 0,
        'global_iter': global_iter,  # Record global_iter for initial state
        'cost': initial_cost,
        'routes': initial_routes_copy,
        'edges_hash': initial_edges_hash,
        'candidate_nodes': [list(cand) for cand in initial_candidates],  # Deep copy
        'is_cycle_finder': False,
        'move_found': False
    })
    
    for outer_iter in range(max_iterations):
        if outer_iter!=0: cuopt_env.extract_nodes_to_search()
        while True:
            candidates = cuopt_env.get_move_candidates()
            if not cuopt_env.sample_nodes_to_search(full_set=False):
                break # node pool exhausted
            fast_operators = ['vrp', 'sliding', 'two_opt']
            random.shuffle(fast_operators)
            improvements = []
            move_found = False
            for op in fast_operators:
                move_found_here = False
                if op == 'vrp':
                    move_found_here = cuopt_env.perform_vrp_search()
                elif op == 'sliding':
                    move_found_here = cuopt_env.run_sliding_search()
                elif op == 'two_opt':
                    move_found_here = cuopt_env.run_two_opt_search()
                if move_found_here:
                    improvements.append(op.upper())
                move_found = move_found or move_found_here
            cuopt_env.restore_found_nodes() # add back nodes to search
            if move_found: # only record state if a move was found
                local_iter += 1
                global_iter += 1  # Increment global_iter when move found
                routes = cuopt_env.get_solution_routes().copy()
                cost = cuopt_env.get_cost()
                edges = routes_to_edges(routes)
                edges_hash_val = edges_hash(edges)
                intermediate_states.append({
                    'local_iter': local_iter,
                    'global_iter': global_iter,  # Record global_iter
                    'cost': cost,
                    'routes': routes,
                    'edges_hash': edges_hash_val,
                    'candidate_nodes': [list(cand) for cand in candidates],  # Deep copy - captured before search
                    'is_cycle_finder': False,
                    'move_found': True
                })
        if not cuopt_env.run_cycle_finder():
            break
        # Record state after cycle finder if with improvement
        local_iter += 1
        global_iter += 1  # Increment global_iter after cycle finder
        routes = cuopt_env.get_solution_routes().copy()
        cost = cuopt_env.get_cost()
        edges = routes_to_edges(routes)
        edges_hash_val = edges_hash(edges)
        cuopt_env.extract_nodes_to_search()
        candidates = cuopt_env.get_move_candidates()
        intermediate_states.append({
            'local_iter': local_iter,
            'global_iter': global_iter,  # Record global_iter
            'cost': cost,
            'routes': routes,
            'edges_hash': edges_hash_val,
            'candidate_nodes': [list(cand) for cand in candidates],
            'is_cycle_finder': True,
            'move_found': False
        })
    
    cuopt_env.sync_streams()
    final_routes = cuopt_env.get_solution_routes().copy()
    final_cost = cuopt_env.get_cost()
    final_edges = routes_to_edges(final_routes)
    final_edges_hash = edges_hash(final_edges)
    cuopt_env.set_routes_to_search()
    cuopt_env.release_resource()
    cuopt_env.sync_streams()
    
    return intermediate_states, final_cost, final_edges_hash, final_edges, final_routes


def run_local_search_from_intermediate(cuopt_env, initial_routes, candidate_nodes, vrp_instance, max_iterations=100000):
    """Run local search from intermediate solution with pre-set candidate_nodes."""
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
        # run with pre-set candidate_nodes and the initial solution on first iteration 
        if outer_iter == 0: cuopt_env.set_move_candidates(candidate_nodes) 
        while True:
            if not cuopt_env.sample_nodes_to_search(full_set=False):
                break
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
            break
        # Periodic cleanup
        if (outer_iter + 1) % 50 == 0:
            gc.collect()
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    cuopt_env.sync_streams()
    final_cost = cuopt_env.get_cost()
    final_routes = cuopt_env.get_solution_routes().copy()
    final_edges = routes_to_edges(final_routes)
    edges_hash_val = edges_hash(final_edges)
    cuopt_env.set_routes_to_search()
    cuopt_env.release_resource()
    cuopt_env.sync_streams()
    
    return final_cost, edges_hash_val, final_edges, final_routes


def load_trajectory_data(trajectory_path, num_orders, run_id=None):
    """Load all needed data from trajectory.jsonl in a single pass, optionally filtered by run_id."""
    trial_ids = set()
    trial_solutions = defaultdict(list)  # trial_id -> list of solutions
    best_solution = None
    best_cost = float('inf')
    found_run_id = None
    
    with open(trajectory_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            # Filter by run_id if specified
            record_run_id = data.get('run_id')
            if run_id is not None and record_run_id != run_id:
                continue
            
            # Collect run_id (from first matching record)
            if found_run_id is None:
                found_run_id = record_run_id
            
            # Collect trial_ids
            trial_id = data.get('trial_id')
            if trial_id is not None:
                trial_ids.add(trial_id)
                trial_solutions[trial_id].append(data)
            
            # Track best solution (only for the specified run_id)
            cost = data.get('cost')
            if cost is not None and cost < best_cost:
                best_cost = cost
                best_solution = data
    
    # Find first solution for each trial and convert to routes
    first_solutions = {}
    for trial_id, solutions in trial_solutions.items():
        first_solution = min(solutions, key=lambda x: (x.get('global_iter', 0), x.get('local_iter', 0)))
        num_routes = first_solution.get('num_routes_after', first_solution.get('num_routes'))
        routes = solution_flat_to_routes(first_solution['solution_flat'], num_routes, num_orders)
        first_solution['routes'] = routes
        first_solutions[trial_id] = first_solution
    
    # Get final solution edges_hash for each trial
    final_solutions = {}
    for trial_id, solutions in trial_solutions.items():
        for sol in solutions:
            if sol.get('is_final_of_trial'):
                final_solutions[trial_id] = sol.get('edges_hash')
                break
    
    # Convert best solution to routes
    if best_solution:
        num_routes = best_solution.get('num_routes_after', best_solution.get('num_routes'))
        routes = solution_flat_to_routes(best_solution['solution_flat'], num_routes, num_orders)
        best_solution['routes'] = routes
    
    return sorted(trial_ids), first_solutions, best_solution, found_run_id, final_solutions


def get_max_global_iter_from_training_data(instance_path, instance_index, run_id, basin_base_dir="basin_datasets0"):
    """Get the maximum global_iter from training_data_pybind.jsonl for the given run_id.
    This is used to continue global_iter counting across trials.
    
    Returns:
        max_global_iter: Maximum global_iter found, or 0 if not found
    """
    basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
    output_dir = os.path.join("basin_datasets0_analyze", basin_paths['instance_id'])
    pybind_dir = os.path.join(output_dir, 'pybind_basin_analysis')
    training_file = os.path.join(pybind_dir, 'training_data_pybind.jsonl')
    
    max_global_iter = 0
    try:
        with open(training_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get('run_id') == run_id:
                    # Check if global_iter exists in the record
                    global_iter = data.get('global_iter')
                    if global_iter is not None:
                        # Ensure global_iter is a number (int or float)
                        try:
                            global_iter = int(global_iter) if isinstance(global_iter, (int, float, str)) else global_iter
                            max_global_iter = max(max_global_iter, global_iter)
                        except (ValueError, TypeError):
                            # Skip if global_iter cannot be converted to int
                            pass
    except FileNotFoundError:
        pass
    
    return max_global_iter


def save_intermediate_states(intermediate_states, final_cost, final_edges_hash, final_routes, 
                             run_id, trial_id, output_dir):
    """Save intermediate states to a JSON file for later basin analysis."""
    os.makedirs(output_dir, exist_ok=True)
    state_file = os.path.join(output_dir, f'intermediate_states_run_{run_id}_trial_{trial_id}.json')
    
    # Convert routes to solution_flat for JSON serialization
    num_orders = len([node for route in final_routes for node in route if node != 0])
    state_data = {
        'run_id': run_id,
        'trial_id': trial_id,
        'final_cost': final_cost,
        'final_edges_hash': final_edges_hash,
        'final_routes': final_routes,
        'intermediate_states': []
    }
    
    for state in intermediate_states:
        state_copy = state.copy()
        # Convert routes to solution_flat
        state_copy['solution_flat'] = routes_to_solution_flat(state['routes'], num_orders)
        # Don't store routes (too large, can reconstruct from solution_flat)
        del state_copy['routes']
        state_data['intermediate_states'].append(state_copy)
    
    with open(state_file, 'w') as f:
        json.dump(state_data, f, indent=2)
    
    print(f"  Saved intermediate states to {state_file}")
    return state_file


def load_intermediate_states(run_id, trial_id, output_dir, num_orders):
    """Load intermediate states from a JSON file."""
    state_file = os.path.join(output_dir, f'intermediate_states_run_{run_id}_trial_{trial_id}.json')
    
    if not os.path.exists(state_file):
        return None
    
    with open(state_file, 'r') as f:
        state_data = json.load(f)
    
    # Reconstruct routes from solution_flat
    intermediate_states = []
    for state in state_data['intermediate_states']:
        # Reconstruct routes from solution_flat
        num_routes = len([r for r in state['solution_flat'] if r == 0]) - 1
        routes = solution_flat_to_routes(state['solution_flat'], num_routes, num_orders)
        state['routes'] = routes
        del state['solution_flat']
        intermediate_states.append(state)
    
    # Reconstruct final_routes
    num_routes = len([r for r in state_data['final_routes'] if isinstance(r, list) and len(r) > 0])
    final_routes = state_data['final_routes']
    
    return {
        'intermediate_states': intermediate_states,
        'final_cost': state_data['final_cost'],
        'final_edges_hash': state_data['final_edges_hash'],
        'final_routes': final_routes
    }


def collect_intermediate_states_for_trial(instance_path, instance_index, run_id, trial_id,
                                         basin_base_dir, max_iterations, num_orders, output_dir):
    """Phase 1: Run pybind once for a trial and save intermediate states."""
    print(f"\nCollecting intermediate states for trial {trial_id}...")
    
    ensure_cuda_device()
    aggressive_gc_cleanup()
    
    # Get trajectory path
    basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
    trajectory_path = basin_paths['trajectory_path']
    
    # Load trajectory data
    trial_ids, first_solutions, best_solution, found_run_id, final_solutions = load_trajectory_data(
        trajectory_path, num_orders, run_id=run_id)
    
    if trial_id not in first_solutions:
        print(f"Trial {trial_id} not found in trajectory")
        return None
    
    # Get initial solution for this trial
    initial_solution = first_solutions[trial_id]
    initial_routes = initial_solution['routes']
    initial_cost = initial_solution['cost']
    
    print(f"  Initial solution cost: {initial_cost:.2f}")
    
    # Get starting global_iter from previous trials in this run
    global_iter_start = get_max_global_iter_from_training_data(instance_path, instance_index, run_id, basin_base_dir)
    if global_iter_start > 0:
        global_iter_start += 1  # Start from next value after max
    print(f"  Starting global_iter from: {global_iter_start}")
    
    # Run pybind once to get intermediate states
    ensure_cuda_device()
    vrp_instance = create_vrp_instance_from_pkl(instance_path, instance_index, num_vehicles=30)
    cuopt_env = vrp_instance['cuopt_env']
    
    intermediate_states = []
    final_cost = None
    final_edges_hash_run = None
    final_edges = None
    final_routes = None
    
    try:
        intermediate_states, final_cost, final_edges_hash_run, final_edges, final_routes = \
            run_local_search_with_intermediate_states(cuopt_env, initial_routes, vrp_instance, max_iterations, global_iter_start)
        print(f"  Collected {len(intermediate_states)} intermediate states")
        print(f"  Final cost: {final_cost:.2f}, edges_hash: {final_edges_hash_run}")
        if intermediate_states:
            print(f"  Global_iter range: {intermediate_states[0].get('global_iter', 'N/A')} to {intermediate_states[-1].get('global_iter', 'N/A')}")
    finally:
        del cuopt_env
        del vrp_instance
        aggressive_gc_cleanup()
    
    # Check if we got valid results
    if not intermediate_states or final_routes is None:
        print(f"Failed to collect intermediate states for trial {trial_id}")
        return None
    
    # Save intermediate states
    save_intermediate_states(intermediate_states, final_cost, final_edges_hash_run, final_routes,
                            run_id, trial_id, output_dir)
    
    # Build history for visualization
    history = []
    for state in intermediate_states:
        history.append({
            'local_iter': state['local_iter'],
            'cost_after': state['cost'],
            'is_circle_found': state['is_cycle_finder'],
            'move_found': state['move_found'],
            'edges_hash': state['edges_hash'],
            'routes': state['routes']
        })
    
    return {
        'rerun_id': 0,  # For compatibility with plot_cost_curve_by_trial_pybind
        'trial_id': trial_id,
        'initial_cost': initial_cost,
        'history': history,
        'final_cost': final_cost,
        'final_edges_hash': final_edges_hash_run,
        'final_routes': final_routes
    }


def analyze_intermediate_basins(instance_path, instance_index, run_id, trial_id, num_runs=100, 
                                basin_base_dir="basin_datasets0", hgs_solution_path=None, max_iterations=100,
                                num_orders=None, hgs_cost=None):
    """Analyze basins from intermediate solutions for a specific trial (single-stage processing)."""
    
    ensure_cuda_device()
    aggressive_gc_cleanup()
    
    # Get trajectory path
    basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
    trajectory_path = basin_paths['trajectory_path']
    basin_dir = basin_paths['basin_dir']
    
    # Get num_orders if not provided
    if num_orders is None:
        temp_instance = create_vrp_instance_from_pkl(instance_path, instance_index, num_vehicles=30)
        num_orders = temp_instance['num_orders']
        del temp_instance
        aggressive_gc_cleanup()
    
    # Load trajectory data
    trial_ids, first_solutions, best_solution, found_run_id, final_solutions = load_trajectory_data(
        trajectory_path, num_orders, run_id=run_id)
    
    if trial_id not in first_solutions:
        print(f"Trial {trial_id} not found in trajectory")
        return None
    
    # Get initial solution for this trial
    initial_solution = first_solutions[trial_id]
    initial_routes = initial_solution['routes']
    initial_cost = initial_solution['cost']
    
    # Get final solution edges_hash for comparison
    final_edges_hash = final_solutions.get(trial_id)
    print(f"Initial solution cost: {initial_cost:.2f}")
    print(f"Final solution edges_hash: {final_edges_hash}")
    
    # Step 1: Run pybind once to get intermediate states
    print(f"\nStep 1: Running pybind once to collect intermediate states...")
    ensure_cuda_device()
    vrp_instance = create_vrp_instance_from_pkl(instance_path, instance_index, num_vehicles=30)
    cuopt_env = vrp_instance['cuopt_env']
    
    # Get starting global_iter from previous trials in this run
    global_iter_start = get_max_global_iter_from_training_data(instance_path, instance_index, run_id, basin_base_dir)
    if global_iter_start > 0:
        global_iter_start += 1  # Start from next value after max
    print(f"  Starting global_iter from: {global_iter_start}")
    
    # Initialize variables in case of error
    intermediate_states = []
    final_cost = None
    final_edges_hash_run = None
    final_edges = None
    final_routes = None
    
    try:
        intermediate_states, final_cost, final_edges_hash_run, final_edges, final_routes = \
            run_local_search_with_intermediate_states(cuopt_env, initial_routes, vrp_instance, max_iterations, global_iter_start)
        print(f"  Collected {len(intermediate_states)} intermediate states")
        print(f"  Final cost: {final_cost:.2f}, edges_hash: {final_edges_hash_run}")
        if intermediate_states:
            print(f"  Global_iter range: {intermediate_states[0].get('global_iter', 'N/A')} to {intermediate_states[-1].get('global_iter', 'N/A')}")
    finally:
        del cuopt_env
        del vrp_instance
        aggressive_gc_cleanup()
    
    # Check if we got valid results
    if not intermediate_states or final_routes is None:
        print(f"Failed to collect intermediate states for trial {trial_id}")
        return None
    
    # Step 2: For each intermediate state, run 100 times and analyze basins
    print(f"\nStep 2: Analyzing basins for each intermediate state ({len(intermediate_states)} states)...")
    results = []
    
    for state_idx, state in enumerate(intermediate_states):
        global_iter_str = f", global_iter={state.get('global_iter', 'N/A')}" if 'global_iter' in state else ""
        print(f"\n  Intermediate state {state_idx + 1}/{len(intermediate_states)}: "
              f"local_iter={state['local_iter']}{global_iter_str}, "
              f"cost={state['cost']:.2f}, edges_hash={state['edges_hash'][:8]}")
        
        basin_counts = defaultdict(int)
        basin_data = {}  # edges_hash -> {edges, costs}
        final_costs = []
        matches_final = 0
        
        for run in range(num_runs):
            run_vrp_instance = None
            run_cuopt_env = None
            
            try:
                ensure_cuda_device()
                run_vrp_instance = create_vrp_instance_from_pkl(instance_path, instance_index, num_vehicles=30)
                run_cuopt_env = run_vrp_instance['cuopt_env']
                
                # Clear large arrays
                if 'cost_matrix' in run_vrp_instance:
                    del run_vrp_instance['cost_matrix']
                if 'node_coords' in run_vrp_instance:
                    del run_vrp_instance['node_coords']
                
                # Run with intermediate solution and candidate_nodes
                run_final_cost, edges_hash_val, run_final_edges, run_final_routes = run_local_search_from_intermediate(
                    run_cuopt_env, state['routes'], state['candidate_nodes'], run_vrp_instance, max_iterations=100000)
                
                if edges_hash_val not in basin_data:
                    basin_data[edges_hash_val] = {
                        'edges': sorted(run_final_edges),
                        'solution_flat': routes_to_solution_flat(run_final_routes, num_orders),
                        'costs': []
                    }
                basin_data[edges_hash_val]['costs'].append(run_final_cost)
                
                del run_final_routes
                del run_final_edges
                
                basin_counts[edges_hash_val] += 1
                final_costs.append(run_final_cost)
                if final_edges_hash and edges_hash_val == final_edges_hash:
                    matches_final += 1
            finally:
                if run_cuopt_env is not None:
                    try:
                        run_cuopt_env.set_routes_to_search()
                        run_cuopt_env.release_resource()
                        run_cuopt_env.sync_streams()
                    except:
                        pass
                    finally:
                        del run_cuopt_env
                    run_cuopt_env = None
                
                if run_vrp_instance is not None:
                    if 'cuopt_env' in run_vrp_instance:
                        del run_vrp_instance['cuopt_env']
                    del run_vrp_instance
                    run_vrp_instance = None
            
            # Progress report
            if (run + 1) % 20 == 0:
                print(f"    Run {run + 1}/{num_runs}: {len(basin_counts)} unique basins, "
                      f"{matches_final} matches final solution")
        
        mean_final_cost = sum(final_costs) / len(final_costs)
        match_ratio = matches_final / num_runs
        
        # Store initial solution info for later use
        initial_edges = routes_to_edges(state['routes'])
        initial_solution_flat = routes_to_solution_flat(state['routes'], num_orders)
        candidate_nodes = state['candidate_nodes']  # Save candidate_nodes
        
        results.append({
            'state_idx': state_idx,
            'local_iter': state['local_iter'],
            'global_iter': state.get('global_iter'),  # Add global_iter from intermediate state
            'intermediate_cost': state['cost'],
            'intermediate_edges_hash': state['edges_hash'],
            'initial_edges': initial_edges,
            'initial_solution_flat': initial_solution_flat,
            'candidate_nodes': candidate_nodes,  # Save candidate_nodes
            'is_cycle_finder': state['is_cycle_finder'],
            'move_found': state['move_found'],
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
        
    # Save results in training_data format
    output_dir = os.path.join("basin_datasets0_analyze", basin_paths['instance_id'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Create pybind_basin_analysis directory for visualizations and training data
    viz_output_dir = os.path.join(output_dir, 'pybind_basin_analysis')
    os.makedirs(viz_output_dir, exist_ok=True)
    output_file = os.path.join(viz_output_dir, 'training_data_pybind.jsonl')
    
    # Clean up incomplete data for this trial before processing
    # This ensures we can resume cleanly if the previous run was interrupted
    print(f"\nCleaning up incomplete data for run_id={run_id}, trial_id={trial_id}...")
    lines_to_keep = []
    removed_count = 0
    try:
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if not (data.get('run_id') == run_id and data.get('trial_id') == trial_id):
                            lines_to_keep.append(line)
                        else:
                            removed_count += 1
                    except:
                        lines_to_keep.append(line)  # Keep malformed lines
    except FileNotFoundError:
        pass
    
    if removed_count > 0:
        with open(output_file, 'w') as f:
            f.writelines(lines_to_keep)
        print(f"  Removed {removed_count} incomplete records from {output_file}")
    else:
        print(f"  No incomplete records found in {output_file}")
    
    # Check existing keys to avoid duplicates (should be empty after cleanup, but keep for safety)
    existing_keys = set()
    try:
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    existing_keys.add((data.get('run_id'), data.get('trial_id'), data.get('local_iter')))
    except FileNotFoundError:
        pass
    
    # Save in training_data format
    new_records = []
    with open(output_file, 'a') as f:
        for result in results:
            key = (run_id, trial_id, result['local_iter'])
            if key in existing_keys:
                continue
            
            # Get initial solution info (already stored in result)
            initial_edges = result['initial_edges']
            initial_solution_flat = result['initial_solution_flat']
            candidate_nodes = result['candidate_nodes']  # Get candidate_nodes
            
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
            
            # Create training record (same format as analyze_basin.py, plus candidate_nodes and global_iter)
            training_record = {
                'run_id': run_id,
                'trial_id': trial_id,
                'local_iter': result['local_iter'],
                'global_iter': result.get('global_iter'),  # Add global_iter from intermediate state
                'initial_solution': {
                    'edges': [[u, v] for u, v in initial_edges],
                    'solution_flat': initial_solution_flat,
                    'cost': result['intermediate_cost'],
                    'edges_hash': result['intermediate_edges_hash'],
                    'is_cycle_finder': result.get('is_cycle_finder', False),
                },
                'candidate_nodes': candidate_nodes,  # Additional field: candidate_nodes set
                'basin_distribution': basin_distribution,
                'basin_features': basin_features,
                'num_unique_basins': result['num_unique_basins'],
                'num_runs': result['num_runs'],
            }
            if hgs_cost:
                training_record['initial_solution']['gap_to_hgs'] = calculate_gap(result['intermediate_cost'], hgs_cost)
            
            f.write(json.dumps(training_record) + '\n')
            new_records.append(key)
    
    if new_records:
        print(f"\nTraining data saved: {len(new_records)} new records added to {output_file}")
    else:
        print(f"\nTraining data: no new records (all duplicates skipped)")
    
    # Cleanup basin_data from results to free memory (data already saved)
    for result in results:
        if 'basin_data' in result:
            for basin_info in result['basin_data'].values():
                if 'solution_flat' in basin_info and len(basin_info.get('solution_flat', [])) > 1000:
                    basin_info['solution_flat'] = None
        # Clear final_costs as well
        if 'final_costs' in result:
            del result['final_costs']
    
    # Adapt results for functions from analyze_basin.py (map intermediate_cost -> initial_cost)
    adapted_results = []
    for result in results:
        adapted_result = result.copy()
        adapted_result['initial_cost'] = result['intermediate_cost']
        adapted_result['initial_edges_hash'] = result['intermediate_edges_hash']
        adapted_results.append(adapted_result)
    
    # Basin entropy analysis
    entropy_results = analyze_basin_entropy(adapted_results, viz_output_dir, run_id, trial_id, hgs_cost)
    
    # Critical gap analysis (if HGS cost available)
    critical_gap_results = None
    if hgs_cost:
        critical_gap_results = find_critical_gap_for_basin_stability(adapted_results, viz_output_dir, run_id, trial_id, hgs_cost)
    
    # Save results to Excel
    save_results_to_excel(adapted_results, run_id, trial_id, viz_output_dir, final_edges_hash, hgs_cost, 
                         entropy_results, critical_gap_results)
    
    # Create visualization (use create_comprehensive_visualization from analyze_basin.py)
    create_comprehensive_visualization(adapted_results, final_edges_hash, run_id, trial_id, viz_output_dir, hgs_cost)
    
    # Rename output file to match our naming convention
    old_file = os.path.join(viz_output_dir, f'comprehensive_analysis_run_{run_id}_trial_{trial_id}.png')
    new_file = os.path.join(viz_output_dir, f'pybind_basins_run_{run_id}_trial_{trial_id}.png')
    if os.path.exists(old_file):
        os.rename(old_file, new_file)
        print(f"Visualization saved to: {new_file}")
    
    # Build history from intermediate_states for visualization (only once, at the end)
    history = []
    for state in intermediate_states:
        history.append({
            'local_iter': state['local_iter'],
            'cost_after': state['cost'],
            'is_circle_found': state['is_cycle_finder'],
            'move_found': state['move_found'],
            'edges_hash': state['edges_hash'],
            'routes': state['routes']
        })
    
    # Get initial cost from first intermediate state
    initial_cost = intermediate_states[0]['cost'] if intermediate_states else None
    
    # Return results with history for visualization
    # Add rerun_id=0 for compatibility with plot_cost_curve_by_trial_pybind
    return {
        'rerun_id': 0,  # For compatibility with plot_cost_curve_by_trial_pybind
        'trial_id': trial_id,
        'history': history,
        'final_cost': final_cost,
        'final_edges_hash': final_edges_hash_run,
        'final_routes': final_routes,
        'initial_cost': initial_cost,
        'basin_results': results  # Basin analysis results
    }

def main(instance_path, instance_index=0, num_vehicles=30, 
         basin_base_dir="basin_datasets0", hgs_solution_path=None,
         num_runs=100, max_iterations=100, run_id=None, trial_id=None, collect_only=False):
    """Main function - process all trials in a run.
    
    Args:
        collect_only: If True, only collect intermediate states and generate visualizations,
                      without running basin analysis.
    """
    print("=" * 80)
    if collect_only:
        print("Collect Intermediate States Only (Run-level)")
    else:
        print("Intermediate Basin Analysis (Run-level)")
    print("=" * 80)
    
    # Load instance
    print(f"\n1. Loading VRP instance from {instance_path} (index={instance_index})")
    vrp_instance = create_vrp_instance_from_pkl(instance_path, instance_index, num_vehicles)
    num_orders = vrp_instance['num_orders']
    node_coords = vrp_instance['node_coords']
    print(f"   Instance: {num_orders} orders, {num_vehicles} vehicles")
    del vrp_instance
    aggressive_gc_cleanup()
    
    # Get trajectory path
    basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
    trajectory_path = basin_paths['trajectory_path']
    basin_dir = basin_paths['basin_dir']
    instance_id = basin_paths['instance_id']
    print(f"   Trajectory path: {trajectory_path}")
    
    if not os.path.exists(trajectory_path):
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")
    
    # Load HGS solution
    hgs_cost = None
    hgs_routes = None
    if hgs_solution_path and os.path.exists(hgs_solution_path):
        print(f"\n2. Loading HGS solution from {hgs_solution_path}")
        hgs_solution = load_hgs_solution_from_pkl(hgs_solution_path, instance_index)
        hgs_cost = hgs_solution['hgs_cost']
        hgs_routes = hgs_solution['hgs_routes']
        print(f"   HGS cost: {hgs_cost:.2f}")
    
    # Load trajectory data
    print(f"\n3. Reading trajectory.jsonl...")
    trial_ids, first_solutions, best_original_solution, found_run_id, final_solutions = load_trajectory_data(
        trajectory_path, num_orders, run_id=run_id)
    print(f"   Found {len(trial_ids)} trials")
    
    # Get run_id from parameter or trajectory
    if run_id is None:
        run_id = found_run_id
        if run_id is None:
            raise ValueError("Could not find run_id in trajectory.jsonl. Please specify --run_id")
    print(f"   Run ID: {run_id}")
    
    # Get original best solution
    original_cost = None
    original_routes = None
    original_gap = None
    if best_original_solution:
        original_cost = best_original_solution['cost']
        original_routes = best_original_solution['routes']
        original_gap = calculate_gap(original_cost, hgs_cost) if hgs_cost else None
        print(f"   Original best cost: {original_cost:.2f}")
    
    # Determine which trials to analyze
    if trial_id is None:
        trials_to_analyze = sorted(trial_ids)
    else:
        if trial_id not in trial_ids:
            raise ValueError(f"Trial {trial_id} not found in trajectory")
        trials_to_analyze = [trial_id]
    
    # Create output directory for visualizations
    viz_dir = os.path.join("basin_datasets0_analyze", instance_id, f"{run_id}_trial_plot")
    os.makedirs(viz_dir, exist_ok=True)
    print(f"   Visualization directory: {viz_dir}")
    
    # Create output directory for intermediate states
    output_dir = os.path.join("basin_datasets0_analyze", instance_id)
    os.makedirs(output_dir, exist_ok=True)
    
    if collect_only:
        # Only collect intermediate states and generate visualizations
        print(f"\n4. Collecting intermediate states for {len(trials_to_analyze)} trial(s) in run {run_id}...")
        
        all_trial_results = []  # For visualization
        
        for trial_idx, trial_id in enumerate(trials_to_analyze):
            print(f"\n{'='*80}")
            print(f"Collecting intermediate states for trial_id={trial_id} ({trial_idx+1}/{len(trials_to_analyze)})")
            print(f"{'='*80}")
            
            result = collect_intermediate_states_for_trial(
                instance_path, instance_index, run_id, trial_id,
                basin_base_dir, max_iterations, num_orders, output_dir)
            
            if result:
                all_trial_results.append(result)
        
        # Generate visualizations
        if all_trial_results:
            print(f"\n{'='*80}")
            print("Generating visualizations...")
            print(f"{'='*80}")
            
            # Find best solution across all trials
            best_new_cost = float('inf')
            best_new_routes = None
            for result in all_trial_results:
                if result['final_cost'] < best_new_cost:
                    best_new_cost = result['final_cost']
                    best_new_routes = result['final_routes']
            
            # Solution comparison
            solution_comparison_path = os.path.join(viz_dir, f'run_{run_id}_collect_only_solution_comparison.png')
            visualize_solutions(hgs_routes, original_routes, best_new_routes,
                               node_coords, hgs_cost, original_cost, best_new_cost, 
                               solution_comparison_path)
            
            # Cost curve for all trials
            cost_curve_path = os.path.join(viz_dir, f'run_{run_id}_collect_only_cost_curve_by_trial_pybind.png')
            plot_cost_curve_by_trial_pybind(all_trial_results, hgs_cost, cost_curve_path)
            
            print(f"\n  Visualizations saved:")
            print(f"    - {solution_comparison_path}")
            print(f"    - {cost_curve_path}")
            print(f"  Best new cost: {best_new_cost:.2f}")
            if hgs_cost:
                print(f"  Best new gap: {calculate_gap(best_new_cost, hgs_cost):.2f}%")
        
        print(f"\n{'='*80}")
        print("Collection complete!")
        print(f"{'='*80}")
        print(f"  Processed {len(trials_to_analyze)} trial(s)")
        print(f"  Original best cost: {original_cost:.2f}" if original_cost else "  Original best cost: N/A")
        if hgs_cost:
            print(f"  HGS cost: {hgs_cost:.2f}")
            if original_gap is not None:
                print(f"  Original gap: {original_gap:.2f}%")
        
        return None
    
    # Process all trials in this run (single-stage: collect + analyze for each trial)
    print(f"\n4. Processing {len(trials_to_analyze)} trial(s) in run {run_id}...")
    
    # Process all trials in this run
    all_trial_results = []  # For visualization
    all_basin_results = {}  # For basin analysis
    
    for trial_idx, trial_id in enumerate(trials_to_analyze):
        print(f"\n{'='*80}")
        print(f"Processing trial_id={trial_id} ({trial_idx+1}/{len(trials_to_analyze)})")
        print(f"{'='*80}")
        
        result = analyze_intermediate_basins(
            instance_path, instance_index, run_id, trial_id,
            num_runs, basin_base_dir, hgs_solution_path, max_iterations, num_orders, hgs_cost)
        
        if result:
            all_trial_results.append(result)
            all_basin_results[trial_id] = result['basin_results']
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")
    print(f"  Processed {len(trials_to_analyze)} trial(s)")
    print(f"  Original best cost: {original_cost:.2f}" if original_cost else "  Original best cost: N/A")
    if hgs_cost:
        print(f"  HGS cost: {hgs_cost:.2f}")
        if original_gap is not None:
            print(f"  Original gap: {original_gap:.2f}%")
    
    return all_basin_results


def generate_run_visualizations_from_jsonl(instance_path, instance_index, run_id, 
                                         basin_base_dir, hgs_solution_path, num_orders, num_vehicles):
    """Generate solution comparison and cost curve visualizations from training_data_pybind.jsonl.
    
    This function should be called after all trials in a run are completed.
    It reads data from training_data_pybind.jsonl and generates the two run-level visualizations.
    """
    print(f"\n{'='*80}")
    print(f"Generating run-level visualizations for run_id={run_id}")
    print(f"{'='*80}")
    
    # Get paths
    basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
    instance_id = basin_paths['instance_id']
    output_dir = os.path.join("basin_datasets0_analyze", instance_id)
    pybind_dir = os.path.join(output_dir, 'pybind_basin_analysis')
    training_file = os.path.join(pybind_dir, 'training_data_pybind.jsonl')
    
    if not os.path.exists(training_file):
        print(f"  Training file not found: {training_file}")
        return
    
    # Load HGS solution
    hgs_cost = None
    hgs_routes = None
    if hgs_solution_path and os.path.exists(hgs_solution_path):
        hgs_solution = load_hgs_solution_from_pkl(hgs_solution_path, instance_index)
        hgs_cost = hgs_solution['hgs_cost']
        hgs_routes = hgs_solution['hgs_routes']
    
    # Load trajectory to get original best solution
    trajectory_path = basin_paths['trajectory_path']
    trial_ids, first_solutions, best_original_solution, found_run_id, final_solutions = load_trajectory_data(
        trajectory_path, num_orders, run_id=run_id)
    
    original_cost = None
    original_routes = None
    if best_original_solution:
        original_cost = best_original_solution['cost']
        original_routes = best_original_solution['routes']
    
    # Load node coordinates
    vrp_instance = create_vrp_instance_from_pkl(instance_path, instance_index, num_vehicles)
    node_coords = vrp_instance['node_coords']
    del vrp_instance
    aggressive_gc_cleanup()
    
    # Read all data from training_data_pybind.jsonl for this run_id
    print(f"  Reading data from {training_file}...")
    training_records = []
    with open(training_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data.get('run_id') == run_id:
                    training_records.append(data)
    
    if not training_records:
        print(f"  No data found for run_id={run_id} in {training_file}")
        return
    
    print(f"  Found {len(training_records)} records for run_id={run_id}")
    
    # Group by trial_id and build history for each trial
    trial_data = defaultdict(list)  # trial_id -> list of records sorted by global_iter
    for record in training_records:
        trial_id = record.get('trial_id')
        if trial_id is not None:
            trial_data[trial_id].append(record)
    
    # Sort each trial's records by global_iter
    for trial_id in trial_data:
        trial_data[trial_id].sort(key=lambda x: (x.get('global_iter', 0), x.get('local_iter', 0)))
    
    # Build results structure for plot_cost_curve_by_trial_pybind
    all_trial_results = []
    best_new_cost = float('inf')
    best_new_routes = None
    best_new_record = None  # Store the record with best cost for solution extraction
    
    for trial_id in sorted(trial_data.keys()):
        records = trial_data[trial_id]
        if not records:
            continue
        
        # Build history from records
        history = []
        for record in records:
            initial_sol = record.get('initial_solution', {})
            history.append({
                'local_iter': record.get('local_iter', 0),
                'cost_after': initial_sol.get('cost', 0),
                'is_circle_found': initial_sol.get('is_cycle_finder', False),
                'move_found': False,  # Not stored in training data
                'edges_hash': initial_sol.get('edges_hash', ''),
            })
        
        # Find best cost from history (this matches what plot_cost_curve_by_trial_pybind shows)
        best_history_cost = float('inf')
        best_history_record = None
        for record in records:
            initial_sol = record.get('initial_solution', {})
            cost = initial_sol.get('cost', float('inf'))
            if cost < best_history_cost:
                best_history_cost = cost
                best_history_record = record
        
        # Find best basin across ALL records for solution extraction
        # Check all basin_features from all records in this trial
        best_basin_hash = None
        best_basin_cost = float('inf')
        best_basin_info = None
        
        for record in records:
            basin_features = record.get('basin_features', {})
            for basin_hash, basin_info in basin_features.items():
                mean_cost = basin_info.get('mean_cost', float('inf'))
                if mean_cost < best_basin_cost:
                    best_basin_cost = mean_cost
                    best_basin_hash = basin_hash
                    best_basin_info = basin_info
        
        # Get solution from best basin (already found above)
        if best_basin_info and best_basin_hash:
            solution_flat = best_basin_info.get('solution_flat')
            if solution_flat:
                # Infer num_routes from solution_flat
                num_routes = len([r for r in solution_flat if r == 0]) - 1
                if num_routes > 0:
                    routes = solution_flat_to_routes(solution_flat, num_routes, num_orders)
                else:
                    routes = None
            else:
                routes = None
        else:
            routes = None
        
        # Track best solution across all trials (use history cost, not basin mean_cost)
        # This matches what plot_cost_curve_by_trial_pybind shows
        if best_history_cost < best_new_cost:
            best_new_cost = best_history_cost
            best_new_record = best_history_record
            # Try to get routes from the best history record's basin_features
            # First try: find basin with lowest mean_cost that matches or is close to best_history_cost
            if best_history_record:
                best_record_basin_features = best_history_record.get('basin_features', {})
                # Find the basin with lowest mean_cost in this record
                best_record_basin_info = None
                best_record_basin_cost = float('inf')
                for basin_hash, basin_info in best_record_basin_features.items():
                    mean_cost = basin_info.get('mean_cost', float('inf'))
                    if mean_cost < best_record_basin_cost:
                        best_record_basin_cost = mean_cost
                        best_record_basin_info = basin_info
                
                if best_record_basin_info:
                    solution_flat = best_record_basin_info.get('solution_flat')
                    if solution_flat and isinstance(solution_flat, (list, np.ndarray)) and len(solution_flat) > 0:
                        # Calculate num_routes: count sequences of 4 consecutive dummy depots (>= num_orders)
                        # Each route starts with 4 dummy depots
                        try:
                            solution_flat_arr = np.array(solution_flat, dtype=np.int32).flatten()
                            dummy_count = sum(1 for r in solution_flat_arr if r >= num_orders)
                            num_routes = dummy_count // 4
                            if num_routes > 0:
                                best_new_routes = solution_flat_to_routes(solution_flat_arr, num_routes, num_orders)
                                if not best_new_routes or len(best_new_routes) == 0:
                                    best_new_routes = None
                                else:
                                    # Validate routes: should have customer nodes
                                    total_customers = sum(len(r) - 2 for r in best_new_routes if len(r) > 2)  # -2 for depots
                                    if total_customers == 0:
                                        best_new_routes = None
                        except Exception as e:
                            print(f"    Warning: Failed to convert solution_flat to routes: {e}")
                            best_new_routes = None
        
        # Add result for visualization
        all_trial_results.append({
            'rerun_id': 0,  # For compatibility with plot_cost_curve_by_trial_pybind
            'trial_id': trial_id,
            'history': history,
            'final_cost': best_basin_cost,  # Keep for compatibility, but best cost is from history
            'final_edges_hash': best_basin_hash or '',
            'final_routes': routes,
            'initial_cost': history[0]['cost_after'] if history else None,
        })
    
    if not all_trial_results:
        print("  No valid trial results found")
        return
    
    # If best_new_routes is still None, try alternative approaches
    if best_new_routes is None:
        print("  Warning: best_new_routes is None, trying alternative extraction methods...")
        
        # Method 1: Try from best_history_record's initial_solution
        if best_new_record:
            initial_sol = best_new_record.get('initial_solution', {})
            initial_solution_flat = initial_sol.get('solution_flat')
            if initial_solution_flat and isinstance(initial_solution_flat, (list, np.ndarray)) and len(initial_solution_flat) > 0:
                try:
                    solution_flat_arr = np.array(initial_solution_flat, dtype=np.int32).flatten()
                    dummy_count = sum(1 for r in solution_flat_arr if r >= num_orders)
                    num_routes = dummy_count // 4
                    if num_routes > 0:
                        best_new_routes = solution_flat_to_routes(solution_flat_arr, num_routes, num_orders)
                        if best_new_routes and len(best_new_routes) > 0:
                            total_customers = sum(len(r) - 2 for r in best_new_routes if len(r) > 2)
                            if total_customers > 0:
                                print(f"  Found best solution from initial_solution (cost: {best_new_cost:.2f})")
                            else:
                                best_new_routes = None
                        else:
                            best_new_routes = None
                except Exception as e:
                    print(f"  Warning: Failed to convert initial_solution_flat: {e}")
        
        # Method 2: Search all basin_features for best solution
        if best_new_routes is None:
            print("  Trying to find solution from all basin_features...")
            best_solution_flat = None
            best_solution_cost = float('inf')
            for trial_id in sorted(trial_data.keys()):
                records = trial_data[trial_id]
                for record in records:
                    basin_features = record.get('basin_features', {})
                    for basin_hash, basin_info in basin_features.items():
                        mean_cost = basin_info.get('mean_cost', float('inf'))
                        if mean_cost < best_solution_cost:
                            best_solution_cost = mean_cost
                            best_solution_flat = basin_info.get('solution_flat')
            
            if best_solution_flat and isinstance(best_solution_flat, (list, np.ndarray)) and len(best_solution_flat) > 0:
                try:
                    solution_flat_arr = np.array(best_solution_flat, dtype=np.int32).flatten()
                    dummy_count = sum(1 for r in solution_flat_arr if r >= num_orders)
                    num_routes = dummy_count // 4
                    if num_routes > 0:
                        best_new_routes = solution_flat_to_routes(solution_flat_arr, num_routes, num_orders)
                        if best_new_routes and len(best_new_routes) > 0:
                            total_customers = sum(len(r) - 2 for r in best_new_routes if len(r) > 2)
                            if total_customers > 0:
                                print(f"  Found best solution from basin_features (cost: {best_solution_cost:.2f})")
                            else:
                                print(f"  Warning: Best solution from basin_features has no customer nodes")
                                best_new_routes = None
                        else:
                            best_new_routes = None
                except Exception as e:
                    print(f"  Warning: Failed to convert solution_flat from basin_features: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Method 3: Use final_routes from all_trial_results (already extracted)
        if best_new_routes is None:
            print("  Trying to use final_routes from all_trial_results...")
            # Find the trial result with best cost
            best_trial_result = None
            best_trial_cost = float('inf')
            for result in all_trial_results:
                trial_history = result.get('history', [])
                if trial_history:
                    trial_best_cost = min(r.get('cost_after', float('inf')) for r in trial_history)
                    if trial_best_cost < best_trial_cost:
                        best_trial_cost = trial_best_cost
                        best_trial_result = result
            
            if best_trial_result and best_trial_result.get('final_routes'):
                best_new_routes = best_trial_result['final_routes']
                print(f"  Found best solution from all_trial_results final_routes (cost: {best_trial_cost:.2f})")
        
        if best_new_routes is None:
            print("  Error: Could not extract best_new_routes from any source!")
            print(f"  Debug info: best_new_cost={best_new_cost:.2f}, best_new_record exists={best_new_record is not None}")
    
    # Create visualization directory
    viz_dir = os.path.join("basin_datasets0_analyze", instance_id, f"{run_id}_trial_plot")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Generate visualizations
    print(f"\n  Generating visualizations...")
    
    # Solution comparison
    solution_comparison_path = os.path.join(viz_dir, f'run_{run_id}_solution_comparison.png')
    if best_new_routes is None:
        print("  Warning: best_new_routes is None, cannot generate solution comparison")
    visualize_solutions(hgs_routes, original_routes, best_new_routes,
                       node_coords, hgs_cost, original_cost, best_new_cost, 
                       solution_comparison_path)
    
    # Cost curve for all trials
    cost_curve_path = os.path.join(viz_dir, f'run_{run_id}_cost_curve_by_trial_pybind.png')
    plot_cost_curve_by_trial_pybind(all_trial_results, hgs_cost, cost_curve_path)
    
    # Verify best cost matches what plot_cost_curve_by_trial_pybind calculates
    # (plot_cost_curve_by_trial_pybind calculates: min of all cost_after in all history records)
    all_costs_from_history = [r.get('cost_after') for result in all_trial_results 
                              for r in result.get('history', []) 
                              if r.get('cost_after') is not None]
    verified_best_cost = min(all_costs_from_history) if all_costs_from_history else best_new_cost
    
    if abs(verified_best_cost - best_new_cost) > 0.01:
        print(f"  Warning: best_new_cost ({best_new_cost:.2f}) doesn't match verified cost ({verified_best_cost:.2f})")
        best_new_cost = verified_best_cost  # Use verified cost
    
    print(f"\n  Visualizations saved:")
    print(f"    - {solution_comparison_path}")
    print(f"    - {cost_curve_path}")
    print(f"  Best new cost: {best_new_cost:.2f} (matches cost curve plot)")
    if hgs_cost:
        print(f"  Best new gap: {calculate_gap(best_new_cost, hgs_cost):.2f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze basins from intermediate solutions')
    parser.add_argument('--pkl', '--instance_path', type=str, 
                       default="/home/jieyi/cvrp100_uniform.pkl", dest='instance_path',
                       help='Path to CVRP instance .pkl file')
    parser.add_argument('--idx', '--instance_index', type=int, default=0, dest='instance_index',
                       help='Index of instance within the .pkl file')
    parser.add_argument('--vehicle', '--vehicles', type=int, default=30, dest='num_vehicles',
                       help='Number of vehicles')
    parser.add_argument('--basin_dir', type=str, default="basin_datasets0", dest='basin_base_dir',
                       help='Base directory for basin datasets')
    parser.add_argument('--hgs', '--hgs_solution_path', type=str, 
                       default="/home/jieyi/hgs_cvrp100_uniform.pkl", dest='hgs_solution_path',
                       help='Path to HGS solution .pkl file')
    parser.add_argument('--n_runs', type=int, default=100,
                       help='Number of runs per intermediate solution (default: 100)')
    parser.add_argument('--max_iter', type=int, default=100,
                       help='Maximum iterations for initial pybind run (default: 100)')
    parser.add_argument('--run_id', type=str, default=None, dest='run_id',
                       help='Run ID to use (if not specified, will read from trajectory.jsonl)')
    parser.add_argument('--trial_id', type=int, default=None, dest='trial_id',
                       help='Trial ID to analyze (if not specified, analyze all trials)')
    parser.add_argument('--collect_only', action='store_true',
                       help='Only collect intermediate states and generate visualizations, without basin analysis')
    
    args = parser.parse_args()
    
    main(
        args.instance_path, args.instance_index, args.num_vehicles,
        args.basin_base_dir, args.hgs_solution_path,
        args.n_runs, args.max_iter, args.run_id, args.trial_id, args.collect_only
    )
