#!/usr/bin/env python3
"""
Basin Collapsing Experiment

1. Load trajectories converging to super basin S* from training_data.jsonl
2. Find Commitment Point (S') as earliest intermediate state with 100% convergence to same basin
3. Diversity check for commitment points
4. Run CuOpt without cycle finder from S' to observe divergence into different basins
5. Calculate number of unique basins as Collapse Rate to quantify smoothing effect
"""
import json
import os
import sys
import random
import gc
import numpy as np
from collections import defaultdict

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def setup_cuopt_path():
    """Setup Python path for cuOpt build."""
    cuopt_build_path = os.path.join(
        os.path.dirname(__file__), 'cpp', 'build', 'install', 
        'lib', 'python3', 'dist-packages')
    
    if cuopt_build_path not in sys.path:
        sys.path.insert(0, cuopt_build_path)
    return cuopt_build_path


def cleanup():
    """Cleanup GPU memory."""
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_local_search_without_cycle_finder(cuopt_env, initial_routes, max_iterations=100000):
    """Run local search without cycle finder."""
    from utils import routes_to_edges, edges_hash
    
    cuopt_env.initialize_search(initial_routes)
    weights = [10000., 10000., 100., 1000., 1000., 1000., 10000., 10000., 10000.]
    cuopt_env.set_weights(weights)
    cuopt_env.set_selection_weights(weights)
    cuopt_env.acquire_resource()
    cuopt_env.reset_move_candidates()
    cuopt_env.set_routes_to_search()
    cuopt_env.sync_streams()
    
    cuopt_env.extract_nodes_to_search()
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
    cuopt_env.sync_streams()
    final_cost = cuopt_env.get_cost()
    final_routes = cuopt_env.get_solution_routes().copy()
    final_edges = routes_to_edges(final_routes)
    edges_hash_val = edges_hash(final_edges)
    cuopt_env.set_routes_to_search()
    cuopt_env.release_resource()
    cuopt_env.sync_streams()
    
    return final_cost, edges_hash_val, final_edges, final_routes


def load_training_data(training_data_path):
    """Load training data from JSONL file."""
    solutions = []
    with open(training_data_path, 'r') as f:
        for line in f:
            if line.strip():
                solutions.append(json.loads(line))
    return solutions


def find_basins_by_gap(training_data, max_gap=1.0):
    """
    Find all basins with gap <= max_gap and their statistics.
    Frequency is counted as the number of trials where this basin is the final basin.
    
    Args:
        training_data: List of solution records from training_data.jsonl
        max_gap: Maximum gap_to_hgs threshold (default: 1.0%)
    
    Returns:
        dict: {basin_id: {'frequency': int, 'gap': float, 'mean_cost': float, ...}}
    """
    trial_solutions = defaultdict(list)
    for sol in training_data:
        run_id = sol.get('run_id')
        trial_id = sol.get('trial_id')
        if run_id is not None and trial_id is not None:
            trial_solutions[(run_id, trial_id)].append(sol)
    
    basin_stats = defaultdict(lambda: {
        'frequency': 0,
        'gap': float('inf'),
        'mean_cost': None,
        'trials': []
    })
    
    for (run_id, trial_id), solutions in trial_solutions.items():
        solutions_sorted = sorted(solutions, 
                                 key=lambda x: (x.get('global_iter', 0), x.get('local_iter', 0)))
        final_solution = max(solutions_sorted, 
                           key=lambda x: (x.get('global_iter', 0), x.get('local_iter', 0)))
        
        # Find final basin (highest probability in basin_distribution)
        basin_dist = final_solution.get('basin_distribution', {})
        if not basin_dist:
            continue
        
        final_basin_id = max(basin_dist.items(), key=lambda x: x[1])[0]
        basin_features = final_solution.get('basin_features', {})
        features = basin_features.get(final_basin_id, {})
        gap = features.get('gap_to_hgs')
        
        if gap is not None and gap <= max_gap:
            basin_stats[final_basin_id]['frequency'] += 1
            if gap < basin_stats[final_basin_id]['gap']:
                basin_stats[final_basin_id]['gap'] = gap
            mean_cost = features.get('mean_cost')
            if mean_cost is not None:
                basin_stats[final_basin_id]['mean_cost'] = mean_cost
            basin_stats[final_basin_id]['trials'].append((run_id, trial_id))
    
    return dict(basin_stats)


def find_top_basin_by_frequency(training_data, max_gap=1.0):
    """
    Find the basin with highest frequency among basins with gap <= max_gap.
    
    Args:
        training_data: List of solution records from training_data.jsonl
        max_gap: Maximum gap_to_hgs threshold (default: 1.0%)
    
    Returns:
        tuple: (basin_id, frequency, gap) or (None, 0, None) if no basin found
    """
    basin_stats = find_basins_by_gap(training_data, max_gap)
    
    if not basin_stats:
        return None, 0, None
    
    top_basin = max(basin_stats.items(), key=lambda x: x[1]['frequency'])
    basin_id, stats = top_basin
    
    return basin_id, stats['frequency'], stats['gap']


def save_basins_to_excel(basin_stats, output_file):
    """
    Save basin statistics to Excel file.
    
    Args:
        basin_stats: Dict from find_basins_by_gap
        output_file: Path to output Excel file
    """
    if not HAS_PANDAS:
        print("Warning: pandas not available, cannot save to Excel")
        return
    
    data = []
    for basin_id, stats in sorted(basin_stats.items(), 
                                  key=lambda x: x[1]['frequency'], reverse=True):
        data.append({
            'basin_id': basin_id,
            'basin_id_short': basin_id[:8],
            'frequency': stats['frequency'],
            'gap_to_hgs': f"{stats['gap']:.2f}%" if stats['gap'] != float('inf') else 'N/A',
            'mean_cost': stats['mean_cost'] if stats['mean_cost'] is not None else 'N/A',
        })
    
    df = pd.DataFrame(data)
    
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Basins (gap <= 1%)', index=False)
        
        # Auto-adjust column widths
        worksheet = writer.sheets['Basins (gap <= 1%)']
        for idx, col in enumerate(worksheet.columns, 1):
            max_length = 0
            column = col[0].column_letter
            for cell in col:
                if cell.value and len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column].width = adjusted_width
    
    print(f"Basin statistics saved to: {output_file}")
    print(f"  Total basins with gap <= 1%: {len(df)}")


def find_commitment_points_for_basin(training_data, target_basin_id, convergence_threshold=1.0):
    """
    Find Commitment Points (S') for a specific basin across all trials.
    
    Args:
        training_data: List of solution records from training_data.jsonl
        target_basin_id: Basin ID (edges_hash) to find commitment points for
        convergence_threshold: Threshold for convergence (default: 1.0 = 100%)
    
    Returns:
        list: List of commitment_point_data for each trial that has this basin
    """
    trial_solutions = defaultdict(list)
    for sol in training_data:
        run_id = sol.get('run_id')
        trial_id = sol.get('trial_id')
        if run_id is not None and trial_id is not None:
            trial_solutions[(run_id, trial_id)].append(sol)
    
    commitment_points = []
    trials_with_basin = 0
    trials_with_100pct = 0
    trials_no_commitment = 0
    
    for (run_id, trial_id), solutions in trial_solutions.items():
        solutions_sorted = sorted(solutions, 
                                 key=lambda x: (x.get('global_iter', 0), x.get('local_iter', 0)))
        
        # Check if final solution contains this basin
        final_solution = max(solutions_sorted, 
                           key=lambda x: (x.get('global_iter', 0), x.get('local_iter', 0)))
        final_basin_dist = final_solution.get('basin_distribution', {})
        
        # Skip if final solution doesn't contain this basin at all
        if target_basin_id not in final_basin_dist:
            continue
        
        trials_with_basin += 1
        final_prob = final_basin_dist[target_basin_id]
        
        # Find earliest point with 100% convergence (forward traversal)
        commitment_point = None
        for sol in solutions_sorted:
            basin_dist = sol.get('basin_distribution', {})
            if target_basin_id in basin_dist:
                prob = basin_dist[target_basin_id]
                if prob >= convergence_threshold:
                    commitment_point = sol
                    break
        
        if final_prob >= convergence_threshold:
            trials_with_100pct += 1
        
        if commitment_point:
            # Count total local_iter in this trial
            total_local_iters = len(solutions_sorted)
            commitment_local_iter = commitment_point.get('local_iter', 0)
            
            commitment_points.append({
                'commitment_point': commitment_point,
                'target_basin_id': target_basin_id,
                'run_id': run_id,
                'trial_id': trial_id,
            })
            print(f"Found commitment point for basin {target_basin_id[:8]}: "
                  f"run_id={run_id}, trial_id={trial_id}, "
                  f"global_iter={commitment_point.get('global_iter')}, "
                  f"local_iter={commitment_local_iter} (total: {total_local_iters}), "
                  f"final_prob={final_prob:.2f}")
        else:
            trials_no_commitment += 1
    
    print(f"\nSummary for basin {target_basin_id[:8]}:")
    print(f"  Trials with this basin: {trials_with_basin}")
    print(f"  Trials with 100% convergence: {trials_with_100pct}")
    print(f"  Commitment points found: {len(commitment_points)}")
    print(f"  Trials without commitment point: {trials_no_commitment}")
    
    return commitment_points


def find_commitment_points(training_data, convergence_threshold=1.0, target_basins=None):
    """
    Find Commitment Points (S') for each trial or for specific basins.
    
    Args:
        training_data: List of solution records from training_data.jsonl
        convergence_threshold: Threshold for convergence (default: 1.0 = 100%)
        target_basins: List of basin IDs (edges_hash) to find commitment points for.
                      If None, finds commitment points for final basin of each trial.
    
    Returns:
        dict: {trial_key or basin_id: commitment_point_data}
    """
    trial_solutions = defaultdict(list)
    for sol in training_data:
        run_id = sol.get('run_id')
        trial_id = sol.get('trial_id')
        if run_id is not None and trial_id is not None:
            trial_solutions[(run_id, trial_id)].append(sol)
    
    commitment_points = {}
    
    if target_basins:
        # Find commitment points for specific basins
        target_basins_set = set(target_basins)
        
        for (run_id, trial_id), solutions in trial_solutions.items():
            solutions_sorted = sorted(solutions, 
                                     key=lambda x: (x.get('global_iter', 0), x.get('local_iter', 0)))
            
            for target_basin_id in target_basins_set:
                commitment_point = None
                for sol in solutions_sorted:
                    basin_dist = sol.get('basin_distribution', {})
                    if target_basin_id in basin_dist:
                        prob = basin_dist[target_basin_id]
                        if prob >= convergence_threshold:
                            commitment_point = sol
                            break
                
                if commitment_point:
                    # Count total local_iter records in this trial
                    total_local_iters = len(solutions_sorted)
                    commitment_local_iter = commitment_point.get('local_iter', 0)
                    
                    key = f"basin_{target_basin_id[:8]}_run_{run_id}_trial_{trial_id}"
                    commitment_points[key] = {
                        'commitment_point': commitment_point,
                        'target_basin_id': target_basin_id,
                        'run_id': run_id,
                        'trial_id': trial_id,
                    }
                    print(f"Found commitment point for basin {target_basin_id[:8]}: "
                          f"run_id={run_id}, trial_id={trial_id}, "
                          f"global_iter={commitment_point.get('global_iter')}, "
                          f"local_iter={commitment_local_iter} (total: {total_local_iters})")
    else:
        # Original logic: find commitment points for final basin of each trial
        for (run_id, trial_id), solutions in trial_solutions.items():
            solutions_sorted = sorted(solutions, 
                                     key=lambda x: (x.get('global_iter', 0), x.get('local_iter', 0)))
            
            final_solution = max(solutions_sorted, 
                               key=lambda x: (x.get('global_iter', 0), x.get('local_iter', 0)))
            
            final_basin_dist = final_solution.get('basin_distribution', {})
            if not final_basin_dist:
                continue
            
            final_basin_id = max(final_basin_dist.items(), key=lambda x: x[1])[0]
            final_basin_prob = final_basin_dist[final_basin_id]
            
            if final_basin_prob < convergence_threshold:
                continue
            
            commitment_point = None
            for sol in solutions_sorted:
                basin_dist = sol.get('basin_distribution', {})
                if final_basin_id in basin_dist:
                    prob = basin_dist[final_basin_id]
                    if prob >= convergence_threshold:
                        commitment_point = sol
                        break
            
            if commitment_point:
                # Count total local_iter records in this trial
                total_local_iters = len(solutions_sorted)
                commitment_local_iter = commitment_point.get('local_iter', 0)
                
                commitment_points[(run_id, trial_id)] = {
                    'commitment_point': commitment_point,
                    'final_basin_id': final_basin_id,
                    'final_solution': final_solution,
                }
                print(f"Found commitment point: run_id={run_id}, trial_id={trial_id}, "
                      f"global_iter={commitment_point.get('global_iter')}, "
                      f"local_iter={commitment_local_iter} (total: {total_local_iters})")
    
    return commitment_points


def compute_solution_similarity(sol1, sol2):
    """Compute Jaccard similarity between two solutions."""
    edges1 = set(tuple(e) for e in sol1.get('initial_solution', {}).get('edges', []))
    edges2 = set(tuple(e) for e in sol2.get('initial_solution', {}).get('edges', []))
    
    if not edges1 or not edges2:
        return 0.0
    
    intersection = len(edges1 & edges2)
    union = len(edges1 | edges2)
    return intersection / union if union > 0 else 0.0


def diversity_check(commitment_points):
    """Compute pairwise similarity statistics for commitment points."""
    if len(commitment_points) < 2:
        return None
    
    commitment_list = list(commitment_points.values())
    n = len(commitment_list)
    
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                sol1 = commitment_list[i]['commitment_point']
                sol2 = commitment_list[j]['commitment_point']
                similarity_matrix[i, j] = compute_solution_similarity(sol1, sol2)
    
    upper_triangle = similarity_matrix[np.triu_indices(n, k=1)]
    
    stats = {
        'mean_similarity': np.mean(upper_triangle),
        'std_similarity': np.std(upper_triangle),
        'min_similarity': np.min(upper_triangle),
        'max_similarity': np.max(upper_triangle),
    }
    
    print(f"Diversity Check: mean={stats['mean_similarity']:.3f}, "
          f"std={stats['std_similarity']:.3f}, "
          f"min={stats['min_similarity']:.3f}, "
          f"max={stats['max_similarity']:.3f}")
    
    return stats


def run_collapse_experiment_single_basin(commitment_points_list, basin_id, instance_path, instance_index, 
                                        trajectory_path, num_runs=100, num_vehicles=30):
    """Run collapse experiment for a single basin from all its commitment points."""
    from test_basin_pybind import create_vrp_instance_from_pkl
    from utils import load_solution_from_trajectory
    
    if not commitment_points_list:
        print(f"No commitment points found for basin {basin_id[:8]}")
        return None
    
    print(f"\nAnalyzing basin {basin_id[:8]} with {len(commitment_points_list)} commitment points")
    
    all_basin_counts = defaultdict(int)
    all_basin_data = {}
    all_final_costs = []
    commitment_point_results = []
    
    for cp_data in commitment_points_list:
        run_id = cp_data.get('run_id')
        trial_id = cp_data.get('trial_id')
        commitment_point = cp_data['commitment_point']
        global_iter = commitment_point.get('global_iter')
        local_iter = commitment_point.get('local_iter')
        
        print(f"  Processing: run_id={run_id}, trial_id={trial_id}, "
              f"global_iter={global_iter}, local_iter={local_iter}")
        
        temp_instance = create_vrp_instance_from_pkl(instance_path, instance_index, num_vehicles=num_vehicles)
        num_orders = temp_instance['num_orders']
        del temp_instance
        cleanup()
        
        solution_data = load_solution_from_trajectory(
            trajectory_path, trial_id, global_iter, local_iter, num_orders, run_id=run_id)
        if not solution_data:
            print(f"    Error: Could not load solution for run_id={run_id}, trial_id={trial_id}, "
                  f"global_iter={global_iter}, local_iter={local_iter}")
            continue
        initial_routes = solution_data['routes']
        
        # Get commitment point solution info
        from utils import routes_to_edges, edges_hash
        commitment_edges = routes_to_edges(initial_routes)
        commitment_edges_hash = edges_hash(commitment_edges)
        commitment_cost = solution_data.get('cost', None)
        
        cost_str = f"{commitment_cost:.2f}" if commitment_cost is not None else "N/A"
        print(f"    Commitment point: edges_hash={commitment_edges_hash[:8]}, cost={cost_str}")
        
        basin_counts = defaultdict(int)
        basin_data = {}
        final_costs = []
        
        for run in range(num_runs):
            run_vrp_instance = create_vrp_instance_from_pkl(
                instance_path, instance_index, num_vehicles=num_vehicles)
            cuopt_env = run_vrp_instance['cuopt_env']
            
            if 'cost_matrix' in run_vrp_instance:
                del run_vrp_instance['cost_matrix']
            if 'node_coords' in run_vrp_instance:
                del run_vrp_instance['node_coords']
            
            final_cost, edges_hash_val, final_edges, final_routes = \
                run_local_search_without_cycle_finder(cuopt_env, initial_routes)
            
            if edges_hash_val not in basin_data:
                basin_data[edges_hash_val] = {'edges': sorted(final_edges), 'costs': []}
            basin_data[edges_hash_val]['costs'].append(final_cost)
            
            basin_counts[edges_hash_val] += 1
            final_costs.append(final_cost)
            
            cuopt_env.set_routes_to_search()
            cuopt_env.release_resource()
            cuopt_env.sync_streams()
            
            del cuopt_env, run_vrp_instance, final_routes, final_edges
            cleanup()
            
            if (run + 1) % 20 == 0:
                print(f"    Run {run + 1}/{num_runs}: {len(basin_counts)} unique basins")
        
        # Print summary for this commitment point
        print(f"    Completed: {len(basin_counts)} unique basins from {num_runs} runs")
        if basin_counts:
            top_basin = max(basin_counts.items(), key=lambda x: x[1])
            top_basin_hash, top_basin_count = top_basin
            top_basin_freq = top_basin_count / num_runs
            top_basin_cost = np.mean(basin_data[top_basin_hash]['costs'])
            print(f"    Top basin: {top_basin_hash[:8]} (count={top_basin_count}, freq={top_basin_freq:.2%}, mean_cost={top_basin_cost:.2f})")
        
        # Store results for this commitment point
        cp_result = {
            'run_id': run_id,
            'trial_id': trial_id,
            'global_iter': global_iter,
            'local_iter': local_iter,
            'commitment_point': {
                'edges_hash': commitment_edges_hash,
                'edges': sorted(commitment_edges),
                'cost': commitment_cost,
            },
            'converged_basins': {
                edges_hash: {
                    'count': count,
                    'frequency': count / num_runs,
                    'edges': basin_data[edges_hash]['edges'],
                    'mean_cost': np.mean(basin_data[edges_hash]['costs']),
                    'costs': basin_data[edges_hash]['costs'],
                }
                for edges_hash, count in basin_counts.items()
            },
            'num_unique_basins': len(basin_counts),
            'num_runs': num_runs,
        }
        commitment_point_results.append(cp_result)
        
        # Aggregate results from all commitment points
        for edges_hash, count in basin_counts.items():
            all_basin_counts[edges_hash] += count
            if edges_hash not in all_basin_data:
                all_basin_data[edges_hash] = {'edges': basin_data[edges_hash]['edges'], 'costs': []}
            all_basin_data[edges_hash]['costs'].extend(basin_data[edges_hash]['costs'])
        
        all_final_costs.extend(final_costs)
    
    collapse_rate = len(all_basin_counts)
    mean_final_cost = np.mean(all_final_costs) if all_final_costs else 0.0
    
    result = {
        'basin_id': basin_id,
        'num_commitment_points': len(commitment_points_list),
        'collapse_rate': collapse_rate,
        'basin_counts': dict(all_basin_counts),
        'basin_data': {k: {'edges': v['edges'], 'mean_cost': np.mean(v['costs'])} 
                      for k, v in all_basin_data.items()},
        'total_runs': len(commitment_points_list) * num_runs,
        'mean_final_cost': mean_final_cost,
        'commitment_points': commitment_point_results,
    }
    
    print(f"\n  Basin {basin_id[:8]} Summary:")
    print(f"    Total commitment points: {len(commitment_points_list)}")
    print(f"    Total runs: {result['total_runs']}")
    print(f"    Collapse rate: {collapse_rate} unique basins")
    print(f"    Mean final cost: {mean_final_cost:.2f}")
    
    if all_basin_counts:
        print(f"    Converged basins distribution:")
        sorted_basins = sorted(all_basin_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (basin_hash, count) in enumerate(sorted_basins[:10], 1):  # Top 10
            freq = count / result['total_runs']
            mean_cost = np.mean(all_basin_data[basin_hash]['costs'])
            print(f"      {i}. {basin_hash[:8]}: count={count} ({freq:.2%}), mean_cost={mean_cost:.2f}")
        if len(sorted_basins) > 10:
            print(f"      ... and {len(sorted_basins) - 10} more basins")
    
    return result


def run_collapse_experiment(commitment_points, instance_path, instance_index, 
                           trajectory_path, num_runs=100, num_vehicles=30):
    """Run collapse experiment from each commitment point."""
    from test_basin_pybind import create_vrp_instance_from_pkl
    from utils import load_solution_from_trajectory
    
    results = {}
    
    for key, cp_data in commitment_points.items():
        # Handle both old format (run_id, trial_id) and new format (basin-based keys)
        if isinstance(key, tuple):
            run_id, trial_id = key
            basin_id = cp_data.get('final_basin_id')
        else:
            run_id = cp_data.get('run_id')
            trial_id = cp_data.get('trial_id')
            basin_id = cp_data.get('target_basin_id')
        
        if basin_id:
            print(f"\nRunning collapse experiment: basin={basin_id[:8]}, run_id={run_id}, trial_id={trial_id}")
        else:
            print(f"\nRunning collapse experiment: run_id={run_id}, trial_id={trial_id}")
        
        commitment_point = cp_data['commitment_point']
        global_iter = commitment_point.get('global_iter')
        local_iter = commitment_point.get('local_iter')
        
        temp_instance = create_vrp_instance_from_pkl(instance_path, instance_index, num_vehicles=num_vehicles)
        num_orders = temp_instance['num_orders']
        del temp_instance
        cleanup()
        
        solution_data = load_solution_from_trajectory(
            trajectory_path, trial_id, global_iter, local_iter, num_orders, run_id=run_id)
        if not solution_data:
            print(f"Error: Could not load solution for run_id={run_id}, trial_id={trial_id}, "
                  f"global_iter={global_iter}, local_iter={local_iter}")
            continue
        initial_routes = solution_data['routes']
        
        basin_counts = defaultdict(int)
        basin_data = {}
        final_costs = []
        
        for run in range(num_runs):
            run_vrp_instance = create_vrp_instance_from_pkl(
                instance_path, instance_index, num_vehicles=num_vehicles)
            cuopt_env = run_vrp_instance['cuopt_env']
            
            if 'cost_matrix' in run_vrp_instance:
                del run_vrp_instance['cost_matrix']
            if 'node_coords' in run_vrp_instance:
                del run_vrp_instance['node_coords']
            
            final_cost, edges_hash_val, final_edges, final_routes = \
                run_local_search_without_cycle_finder(cuopt_env, initial_routes)
            
            if edges_hash_val not in basin_data:
                basin_data[edges_hash_val] = {'edges': sorted(final_edges), 'costs': []}
            basin_data[edges_hash_val]['costs'].append(final_cost)
            
            basin_counts[edges_hash_val] += 1
            final_costs.append(final_cost)
            
            cuopt_env.set_routes_to_search()
            cuopt_env.release_resource()
            cuopt_env.sync_streams()
            
            del cuopt_env, run_vrp_instance, final_routes, final_edges
            cleanup()
            
            if (run + 1) % 20 == 0:
                print(f"  Run {run + 1}/{num_runs}: {len(basin_counts)} unique basins")
        
        collapse_rate = len(basin_counts)
        mean_final_cost = np.mean(final_costs)
        
        result_key = key if not isinstance(key, tuple) else (run_id, trial_id)
        results[result_key] = {
            'commitment_point': commitment_point,
            'basin_id': basin_id or cp_data.get('final_basin_id'),
            'run_id': run_id,
            'trial_id': trial_id,
            'collapse_rate': collapse_rate,
            'basin_counts': dict(basin_counts),
            'basin_data': {k: {'edges': v['edges'], 'mean_cost': np.mean(v['costs'])} 
                          for k, v in basin_data.items()},
            'num_runs': num_runs,
            'mean_final_cost': mean_final_cost,
        }
        
        print(f"Results: collapse_rate={collapse_rate}, mean_cost={mean_final_cost:.2f}")
    
    return results


def save_results(results, diversity_stats, output_dir):
    """Save experiment results."""
    os.makedirs(output_dir, exist_ok=True)
    
    results_serializable = {}
    for key, value in results.items():
        if isinstance(key, tuple):
            result_key = f"{key[0]}_{key[1]}"
        else:
            result_key = str(key)
        
        results_serializable[result_key] = {
            'run_id': value.get('run_id'),
            'trial_id': value.get('trial_id'),
            'basin_id': value.get('basin_id'),
            'commitment_point': {
                'global_iter': value['commitment_point'].get('global_iter'),
                'local_iter': value['commitment_point'].get('local_iter'),
                'cost': value['commitment_point'].get('initial_solution', {}).get('cost'),
            },
            'collapse_rate': value['collapse_rate'],
            'basin_counts': value['basin_counts'],
            'num_runs': value['num_runs'],
            'mean_final_cost': value['mean_final_cost'],
        }
    
    output_data = {
        'diversity_stats': diversity_stats,
        'results': results_serializable,
    }
    
    output_file = os.path.join(output_dir, 'basin_collapsing_results.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {output_file}")


def main():
    import argparse
    from utils import get_basin_paths
    
    parser = argparse.ArgumentParser(description='Basin Collapsing Experiment')
    parser.add_argument('--pkl', type=str, dest='instance_path', 
                       default="/home/jieyi/cvrp100_uniform.pkl",
                       help='Path to instance .pkl file')
    parser.add_argument('--idx', type=int, default=0, dest='instance_index',
                       help='Instance index')
    parser.add_argument('--num_runs', type=int, default=100,
                       help='Number of local search runs per commitment point')
    parser.add_argument('--vehicle', type=int, default=30, dest='num_vehicles',
                       help='Number of vehicles')
    parser.add_argument('--basin_dir', type=str, default="basin_datasets0", dest='basin_base_dir',
                       help='Base directory for basin datasets')
    parser.add_argument('--convergence_threshold', type=float, default=1.0,
                       help='Convergence threshold for commitment point')
    parser.add_argument('--max_gap', type=float, default=1.0,
                       help='Maximum gap_to_hgs (in %) for auto-selecting basins')
    parser.add_argument('--basin_edges_hash', type=str, nargs='+', default=None,
                       help='List of basin IDs (edges_hash) to analyze. '
                            'If provided, analyzes each basin separately. ')
    
    args = parser.parse_args()
    
    cuopt_build_path = setup_cuopt_path()
    
    if not os.path.exists(cuopt_build_path):
        print(f"Error: CuOpt build path does not exist: {cuopt_build_path}")
        sys.exit(1)
    
    # Auto-detect training_data path based on instance_path and instance_index
    basin_paths = get_basin_paths(args.instance_path, args.instance_index, args.basin_base_dir)
    instance_id = basin_paths['instance_id']
    training_data_path = os.path.join(
        os.path.dirname(__file__), 
        f"{args.basin_base_dir}_analyze", 
        instance_id, 
        'training_data.jsonl'
    )
    
    if not os.path.exists(training_data_path):
        print(f"Error: training_data.jsonl not found at: {training_data_path}")
        sys.exit(1)
    
    # Auto-generate output directory: {basin_base_dir}_analyze/{instance_id}/basin_collapsing_results
    output_dir = os.path.join(
        os.path.dirname(__file__),
        f"{args.basin_base_dir}_analyze",
        instance_id,
        'basin_collapsing_results'
    )
    print(f"Output directory: {output_dir}")
    
    print("Basin Collapsing Experiment")
    print("=" * 80)
    
    print(f"\nStep 1: Loading training data from {training_data_path}")
    training_data = load_training_data(training_data_path)
    print(f"  Loaded {len(training_data)} solution records")
    
    # Auto-select basin if not provided
    if not args.basin_edges_hash:
        print(f"\nStep 1.5: Finding basins with gap <= {args.max_gap}%")
        basin_stats = find_basins_by_gap(training_data, max_gap=args.max_gap)
        
        if not basin_stats:
            print("  Error: No basin found with gap <= 1%")
            sys.exit(1)
        
        # Save all basins to Excel
        excel_file = os.path.join(
            output_dir,
            f'basins_gap_le_{args.max_gap:g}pct.xlsx'
        )
        save_basins_to_excel(basin_stats, excel_file)
        
        # Select top basin by frequency
        top_basin = max(basin_stats.items(), key=lambda x: x[1]['frequency'])
        basin_id, stats = top_basin
        args.basin_edges_hash = [basin_id]
        print(f"  Selected basin: {basin_id[:8]}, frequency={stats['frequency']}, gap={stats['gap']:.2f}%")
    
    # Handle basin list analysis mode
    if args.basin_edges_hash:
        basin_paths = get_basin_paths(args.instance_path, args.instance_index, args.basin_base_dir)
        trajectory_path = basin_paths['trajectory_path']
        
        all_results = {}
        
        for basin_id in args.basin_edges_hash:
            print(f"\n{'='*80}")
            print(f"Analyzing basin: {basin_id[:8]}")
            print(f"{'='*80}")
            
            print(f"\nStep 2: Finding commitment points for basin {basin_id[:8]}")
            commitment_points_list = find_commitment_points_for_basin(
                training_data, basin_id, args.convergence_threshold)
            print(f"  Found {len(commitment_points_list)} commitment points")
            
            if not commitment_points_list:
                print(f"  No commitment points found for basin {basin_id[:8]}, skipping...")
                continue
            
            print(f"\nStep 3: Diversity check for basin {basin_id[:8]}")
            # Convert list to dict format for diversity_check
            commitment_points_dict = {
                f"{cp['run_id']}_{cp['trial_id']}": cp 
                for cp in commitment_points_list
            }
            diversity_stats = diversity_check(commitment_points_dict)
            
            print(f"\nStep 4: Running collapse experiment for basin {basin_id[:8]}")
            result = run_collapse_experiment_single_basin(
                commitment_points_list, basin_id, args.instance_path, args.instance_index,
                trajectory_path, args.num_runs, args.num_vehicles)
            
            if result:
                # Add diversity stats to result
                if diversity_stats:
                    result['diversity_stats'] = diversity_stats
                all_results[basin_id] = result
                
                print(f"\nStep 5: Saving results for basin {basin_id[:8]}")
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f'basin_{basin_id[:8]}_results.json')
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Results saved to: {output_file}")
        
        # Save summary of all basins
        if all_results:
            print(f"\n{'='*80}")
            print(f"Summary for all basins")
            print(f"{'='*80}")
            summary_file = os.path.join(output_dir, 'basin_collapsing_summary.json')
            summary_data = {
                'basins_analyzed': len(all_results),
                'basin_results': {k: {
                    'collapse_rate': v['collapse_rate'],
                    'mean_final_cost': v['mean_final_cost'],
                    'num_commitment_points': v['num_commitment_points'],
                    'total_runs': v['total_runs'],
                } for k, v in all_results.items()}
            }
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            print(f"Summary saved to: {summary_file}")
            
            collapse_rates = [r['collapse_rate'] for r in all_results.values()]
            total_commitment_points = sum(r['num_commitment_points'] for r in all_results.values())
            total_runs = sum(r['total_runs'] for r in all_results.values())
            
            print(f"\n{'='*80}")
            print(f"Experiment Complete!")
            print(f"{'='*80}")
            print(f"Total basins analyzed: {len(all_results)}")
            print(f"Total commitment points: {total_commitment_points}")
            print(f"Total runs: {total_runs}")
            print(f"\nCollapse Rate Statistics:")
            if collapse_rates:
                print(f"  Mean: {np.mean(collapse_rates):.2f}")
                print(f"  Std:  {np.std(collapse_rates):.2f}")
                print(f"  Min:  {np.min(collapse_rates)}")
                print(f"  Max:  {np.max(collapse_rates)}")
            else:
                print(f"  No collapse rates to compute")
            
            print(f"\nPer-Basin Results:")
            for basin_id, result in all_results.items():
                print(f"  Basin {basin_id[:8]}: {result['num_commitment_points']} commitment points, "
                      f"collapse_rate={result['collapse_rate']}, "
                      f"mean_cost={result['mean_final_cost']:.2f}")
    else:
        # Original mode: analyze all commitment points
        print(f"\nStep 2: Finding commitment points")
        commitment_points = find_commitment_points(
            training_data, args.convergence_threshold, target_basins=None)
        print(f"  Found {len(commitment_points)} commitment points")
        
        if not commitment_points:
            print("No commitment points found. Exiting.")
            return
        
        print(f"\nStep 3: Diversity check")
        diversity_stats = diversity_check(commitment_points)
        
        print(f"\nStep 4: Running collapse experiment")
        basin_paths = get_basin_paths(args.instance_path, args.instance_index, args.basin_base_dir)
        trajectory_path = basin_paths['trajectory_path']
        
        results = run_collapse_experiment(
            commitment_points, args.instance_path, args.instance_index,
            trajectory_path, args.num_runs, args.num_vehicles)
        
        print(f"\nStep 5: Saving results")
        save_results(results, diversity_stats, output_dir)
        
        print(f"\nExperiment Complete!")
        print(f"Total commitment points analyzed: {len(results)}")
        if results:
            collapse_rates = [r['collapse_rate'] for r in results.values()]
            print(f"Mean collapse rate: {np.mean(collapse_rates):.2f}")
            print(f"Std collapse rate: {np.std(collapse_rates):.2f}")


if __name__ == "__main__":
    main()
