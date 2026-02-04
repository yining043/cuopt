#!/usr/bin/env python3
"""
Compare pybind implementation of run_local_search_silent with original cuOpt results.
Read the first solution from each trial in trajectory.jsonl as initial solution,
run 30 times, and compare the results.
"""
import json
import os
import sys
import random
import gc
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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
from analyze_basin import run_local_search_silent
import datetime


def run_local_search_with_history(cuopt_env, initial_routes, global_iter, max_iterations=100):
    """Run local search with history tracking, return history records and final results."""
    cuopt_env.initialize_search(initial_routes)
    weights = [10000., 10000., 100., 1000., 1000., 1000., 10000., 10000., 10000.]
    cuopt_env.set_weights(weights)
    cuopt_env.set_selection_weights(weights)
    cuopt_env.acquire_resource()
    cuopt_env.reset_move_candidates()
    cuopt_env.set_routes_to_search()
    cuopt_env.sync_streams()
    
    def record_state(global_iter, local_iter, is_cycle, move_found):
        routes = cuopt_env.get_solution_routes().copy()
        return {
            'global_iter': global_iter,
            'local_iter': local_iter,
            'cost_after': cuopt_env.get_cost(),
            'is_circle_found': is_cycle,
            'move_found': move_found,
            'edges_hash': edges_hash(routes_to_edges(routes)),
            'routes': routes
        }
    
    history = []
    local_iter = 0
    history.append(record_state(global_iter, local_iter, False, False))
    
    for outer_iter in range(max_iterations):
        cuopt_env.extract_nodes_to_search()
        while True:
            if not cuopt_env.sample_nodes_to_search(full_set=False):
                break
            fast_operators = ['vrp', 'sliding', 'two_opt']
            random.shuffle(fast_operators)
            move_found = False
            for op in fast_operators:
                move_found_here = False
                if op == 'vrp':
                    move_found_here = cuopt_env.perform_vrp_search()
                elif op == 'sliding':
                    move_found_here = cuopt_env.run_sliding_search()
                elif op == 'two_opt':
                    move_found_here = cuopt_env.run_two_opt_search()
                move_found = move_found or move_found_here
            cuopt_env.restore_found_nodes()
            if move_found:
                local_iter += 1
                global_iter += 1
                history.append(record_state(global_iter, local_iter, False, True))
        if not cuopt_env.run_cycle_finder():
            break
        local_iter += 1
        global_iter += 1
        history.append(record_state(global_iter, local_iter, True, False))
    
    cuopt_env.sync_streams()
    final_routes = cuopt_env.get_solution_routes().copy()
    final_edges = routes_to_edges(final_routes)
    cuopt_env.set_routes_to_search()
    cuopt_env.release_resource()
    cuopt_env.sync_streams()
    
    return history, history[-1]['cost_after'], history[-1]['edges_hash'], final_edges, final_routes, global_iter


def get_first_solution_from_trial(trajectory_path, trial_id, num_orders):
    """Get the first solution from a trial (the one with smallest local_iter)."""
    solutions = []
    with open(trajectory_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get('trial_id') == trial_id:
                solutions.append(data)
    
    if not solutions:
        return None
    
    # Find the solution with smallest local_iter
    first_solution = min(solutions, key=lambda x: (x.get('global_iter', 0), x.get('local_iter', 0)))
    
    # Convert to routes
    num_routes = first_solution.get('num_routes_after', first_solution.get('num_routes'))
    routes = solution_flat_to_routes(first_solution['solution_flat'], num_routes, num_orders)
    first_solution['routes'] = routes
    
    return first_solution


def get_all_first_solutions(trajectory_path, trial_ids, num_orders):
    """Get first solution for all trials in one pass through the file."""
    # Map trial_id -> list of solutions
    trial_solutions = {trial_id: [] for trial_id in trial_ids}
    
    with open(trajectory_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            trial_id = data.get('trial_id')
            if trial_id in trial_solutions:
                trial_solutions[trial_id].append(data)
    
    # Find first solution for each trial and convert to routes
    first_solutions = {}
    for trial_id, solutions in trial_solutions.items():
        if not solutions:
            continue
        
        first_solution = min(solutions, key=lambda x: (x.get('global_iter', 0), x.get('local_iter', 0)))
        num_routes = first_solution.get('num_routes_after', first_solution.get('num_routes'))
        routes = solution_flat_to_routes(first_solution['solution_flat'], num_routes, num_orders)
        first_solution['routes'] = routes
        first_solutions[trial_id] = first_solution
    
    return first_solutions


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
    
    # Convert best solution to routes
    if best_solution:
        num_routes = best_solution.get('num_routes_after', best_solution.get('num_routes'))
        routes = solution_flat_to_routes(best_solution['solution_flat'], num_routes, num_orders)
        best_solution['routes'] = routes
    
    return sorted(trial_ids), first_solutions, best_solution, found_run_id


def get_best_solution_from_trajectory(trajectory_path, num_orders):
    """Get the solution with minimum cost from trajectory."""
    best_solution = None
    best_cost = float('inf')
    
    with open(trajectory_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            cost = data.get('cost')
            if cost is not None and cost < best_cost:
                best_cost = cost
                best_solution = data
    
    if best_solution:
        num_routes = best_solution.get('num_routes_after', best_solution.get('num_routes'))
        routes = solution_flat_to_routes(best_solution['solution_flat'], num_routes, num_orders)
        best_solution['routes'] = routes
    
    return best_solution


def get_all_trial_ids(trajectory_path):
    """Get all trial_ids from trajectory."""
    trial_ids = set()
    with open(trajectory_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            trial_id = data.get('trial_id')
            if trial_id is not None:
                trial_ids.add(trial_id)
    return sorted(trial_ids)


def get_run_id_from_trajectory(trajectory_path):
    """Get run_id from the first record in trajectory.jsonl."""
    with open(trajectory_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            run_id = data.get('run_id')
            if run_id is not None:
                return run_id
    return None


def get_all_run_ids(trajectory_path):
    """Get all unique run_ids from trajectory.jsonl."""
    run_ids = set()
    with open(trajectory_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            run_id = data.get('run_id')
            if run_id is not None:
                run_ids.add(run_id)
    return sorted(run_ids)


def is_run_id_completed(instance_path, instance_index, run_id, basin_base_dir="basin_datasets0", n_runs=30):
    """Check if a run_id has been completed by checking trajectory_pybind.jsonl.
    
    A run_id is considered completed if it has all rerun_ids from 0 to n_runs-1.
    """
    basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
    trajectory_pybind_path = os.path.join(basin_paths['basin_dir'], 'trajectory_pybind.jsonl')
    
    if not os.path.exists(trajectory_pybind_path):
        return False
    
    # Collect all rerun_ids for this run_id
    rerun_ids = set()
    with open(trajectory_pybind_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get('run_id') == run_id:
                rerun_id = data.get('rerun_id')
                if rerun_id is not None:
                    rerun_ids.add(rerun_id)
    
    # Check if we have all rerun_ids from 0 to n_runs-1
    expected_rerun_ids = set(range(n_runs))
    return expected_rerun_ids.issubset(rerun_ids)


def _convert_hgs_routes(hgs_routes):
    """Convert HGS routes to standard format: list of [0, ...nodes..., 0]."""
    if not hgs_routes or not isinstance(hgs_routes, list):
        return None
    
    if isinstance(hgs_routes[0], (int, np.integer)):
        # Flat sequence: [0, 1, 2, 0, 3, 4, 0] or [0, 1, 2, 0, 3, 4]
        routes, current = [], [0]
        for node in hgs_routes:
            if node == 0:
                if len(current) > 1:
                    # Ensure route ends with depot
                    if current[-1] != 0:
                        current.append(0)
                    routes.append(current)
                current = [0]
            else:
                current.append(int(node))
        # Handle last route if it doesn't end with 0
        if len(current) > 1:
            if current[-1] != 0:
                current.append(0)
            routes.append(current)
        return routes if routes else None
    else:
        # List of routes
        formatted = []
        for route in hgs_routes:
            if isinstance(route, list) and len(route) > 0:
                if len(route) >= 2 and route[0] == 0 and route[-1] == 0:
                    formatted.append(route)
                else:
                    clean = [n for n in route if n != 0]
                    if clean:
                        formatted.append([0] + clean + [0])
        return formatted if formatted else None


def visualize_solutions(hgs_routes, original_routes, new_routes, node_coords, 
                       hgs_cost, original_cost, new_cost, output_path='solution_comparison.png'):
    """Visualize HGS, original, and new solutions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Solution Comparison: HGS vs Original vs New', fontsize=16, fontweight='bold')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    x_pos, y_pos = node_coords[:, 0], node_coords[:, 1]
    
    def plot_routes(ax, routes, title, cost):
        if not routes or len(routes) == 0:
            ax.text(0.5, 0.5, 'No routes available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{title}\nCost: N/A', fontsize=12, fontweight='bold')
            return
        
        ax.scatter(x_pos[0], y_pos[0], c='black', s=200, marker='s', label='Depot', zorder=5)
        for route_idx, route in enumerate(routes):
            if isinstance(route, list) and len(route) >= 2:
                valid_nodes = [n for n in route if isinstance(n, (int, np.integer)) and 0 <= n < len(x_pos)]
                if len(valid_nodes) >= 2:
                    ax.plot(x_pos[valid_nodes], y_pos[valid_nodes], 'o-', 
                           color=colors[route_idx % len(colors)], linewidth=2, markersize=4, alpha=0.7)
        
        margin = 5
        ax.set_xlim(x_pos.min() - margin, x_pos.max() + margin)
        ax.set_ylim(y_pos.min() - margin, y_pos.max() + margin)
        ax.set_aspect('equal')
        ax.set_title(f'{title}\nCost: {cost:.2f}' if cost else f'{title}\nCost: N/A', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    hgs_routes = _convert_hgs_routes(hgs_routes) if hgs_routes else None
    plot_routes(axes[0], hgs_routes, 'HGS Solution', hgs_cost)
    plot_routes(axes[1], original_routes, 'Original Best Solution', original_cost)
    plot_routes(axes[2], new_routes, 'New Best Solution (Pybind)', new_cost)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Solution comparison saved to {output_path}")


def plot_cost_curve_by_trial_pybind(results, hgs_cost, output_path='cost_curve_by_trial_pybind.png'):
    """Plot cost evolution for all reruns, color by local optima (edges_hash).
    
    Args:
        results: List of result dicts, each containing 'rerun_id', 'trial_id', 'history', etc.
        hgs_cost: HGS cost for gap calculation
        output_path: Output file path
    """
    if not results:
        print("No results to plot")
        return
    
    # Group by rerun_id, then by trial_id
    # Use global_iter from history records (already calculated in main function)
    rerun_trial_data = defaultdict(lambda: defaultdict(list))  # rerun_id -> trial_id -> records
    
    for result in results:
        rerun_id = result.get('rerun_id')
        trial_id = result.get('trial_id')
        history = result.get('history', [])
        
        for record in history:
            # Use global_iter from record (already set in run_local_search_with_history)
            # This ensures consistency with the actual execution
            record_with_global = record.copy()
            record_with_global['rerun_id'] = rerun_id
            record_with_global['trial_id'] = trial_id
            rerun_trial_data[rerun_id][trial_id].append(record_with_global)
    
    # Group by final edges_hash to find duplicate optima
    edges_hash_to_trials = defaultdict(list)  # edges_hash -> list of (rerun_id, trial_id)
    for result in results:
        history = result.get('history', [])
        if history:
            final_edges_hash = history[-1].get('edges_hash', '')
            if final_edges_hash:
                edges_hash_to_trials[final_edges_hash].append((result['rerun_id'], result['trial_id']))
    
    # Build duplicate groups: edges_hash -> list of (rerun_id, trial_id)
    duplicate_groups = {}
    for edges_hash, trial_list in edges_hash_to_trials.items():
        if len(trial_list) > 1:
            duplicate_groups[edges_hash] = trial_list
    
    # Assign colors based on edges_hash (duplicate optima get same color)
    unique_color = 'black'
    cycle_finder_color_rgb = np.array([0xF1/255.0, 0x8F/255.0, 0x01/255.0])
    
    def filter_colors(colors_list):
        return [tuple(c) if isinstance(c, np.ndarray) else c for c in colors_list
                if len(c) >= 3 and np.sum(np.array(c[:3])) >= 0.3 and
                not np.allclose(np.array(c[:3]), cycle_finder_color_rgb, atol=0.15)]
    
    filtered_colors = filter_colors(plt.cm.tab10(np.linspace(0, 1, 10)))
    if not filtered_colors:
        filtered_colors = filter_colors(plt.cm.Set2(np.linspace(0, 1, 8)))
    if not filtered_colors:
        filtered_colors = [unique_color]
    
    # Map edges_hash to color
    edges_hash_to_color = {}
    color_idx = 0
    for edges_hash_val in sorted(edges_hash_to_trials.keys()):
        if edges_hash_val in duplicate_groups:
            edges_hash_to_color[edges_hash_val] = filtered_colors[color_idx % len(filtered_colors)]
            color_idx += 1
        else:
            edges_hash_to_color[edges_hash_val] = unique_color
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Track labeled colors and build legend info
    labeled_colors = set()
    cycle_finder_labeled = False
    color_to_trial_ids = defaultdict(list)  # color -> list of (rerun_id, trial_id)
    
    # Plot each (rerun_id, trial_id) combination
    for rerun_id in sorted(rerun_trial_data.keys()):
        for trial_id in sorted(rerun_trial_data[rerun_id].keys()):
            records = rerun_trial_data[rerun_id][trial_id]
            if not records:
                continue
            
            # Get final edges_hash for color assignment
            final_edges_hash = records[-1].get('edges_hash', '')
            trial_color = edges_hash_to_color.get(final_edges_hash, unique_color)
            trial_color = tuple(trial_color) if isinstance(trial_color, np.ndarray) else trial_color
            
            # Group points by type
            fast_search_points = []
            cycle_finder_points = []
            all_points = []
            
            for record in records:
                global_iter = record.get('global_iter', 0)
                cost = record.get('cost_after', 0)
                is_cycle = record.get('is_circle_found', False)
                
                all_points.append((global_iter, cost))
                if is_cycle:
                    cycle_finder_points.append((global_iter, cost))
                else:
                    fast_search_points.append((global_iter, cost))
            
            # Track trial_ids for this color (for legend)
            hashable_color = tuple(trial_color) if isinstance(trial_color, (list, np.ndarray)) else trial_color
            color_to_trial_ids[hashable_color].append((rerun_id, trial_id))
            
            # Plot trajectory line (connect all points in order of global_iter)
            if len(all_points) > 1:
                all_points_sorted = sorted(all_points, key=lambda x: x[0])
                iters, costs = zip(*all_points_sorted)
                
                # Determine label for this trajectory
                label = None
                if hashable_color not in labeled_colors:
                    if hashable_color == unique_color or hashable_color == 'black':
                        label = "Unique trials"
                    else:
                        # Get all (rerun_id, trial_id) pairs with this edges_hash
                        trial_list = edges_hash_to_trials.get(final_edges_hash, [])
                        # Format as (rerun_id, trial_id) for clarity
                        trial_pairs_sorted = sorted(trial_list)
                        if len(trial_pairs_sorted) > 1:
                            # Show as (rerun, trial) pairs
                            pairs_str = ', '.join([f"({r}, {t})" for r, t in trial_pairs_sorted])
                            label = f"Trials {pairs_str} (duplicate)"
                        else:
                            label = f"Trial ({rerun_id}, {trial_id})"
                    labeled_colors.add(hashable_color)
                
                # Plot main trajectory line with circle markers for all points
                ax.plot(iters, costs, marker='o', linestyle='-', linewidth=1.5, markersize=4,
                        color=trial_color, label=label, alpha=0.6, zorder=2)
            
            # Plot cycle finder points (on top of the trajectory line, as square markers)
            if cycle_finder_points:
                cycle_finder_sorted = sorted(cycle_finder_points, key=lambda x: x[0])
                iters, costs = zip(*cycle_finder_sorted)
                ax.scatter(iters, costs, marker='s', s=25, color=trial_color,
                          label='Cycle Finder' if not cycle_finder_labeled else None,
                          alpha=0.8, zorder=4, edgecolors='black', linewidths=0.5)
                cycle_finder_labeled = True
    
    # Setup axes
    ax.set_xlabel('Global Iteration', fontsize=13, fontweight='bold')
    ax.set_ylabel('Objective Value (Cost)', fontsize=13, fontweight='bold')
    
    # Calculate best cost
    all_costs = [r.get('cost_after') for result in results for r in result.get('history', []) 
                 if r.get('cost_after') is not None]
    best_cost = min(all_costs) if all_costs else None
    gap = calculate_gap(best_cost, hgs_cost) if (best_cost and hgs_cost) else None
    
    # Build title
    title = 'Pybind Local Search Cost Evolution'
    if best_cost is not None:
        title += f' | Best Cost: {best_cost:.2f}'
    if gap is not None:
        title += f' | Gap: {gap:.2f}%'
    
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend with automatic column adjustment
    legend = ax.legend(loc='best', fontsize=10, framealpha=0.9, shadow=True,
                      ncol=1, columnspacing=1.0, handlelength=1.5)
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    ax_bbox = ax.get_window_extent()
    if bbox.width > ax_bbox.width * 0.9:
        legend = ax.legend(loc='best', fontsize=9, framealpha=0.9, shadow=True,
                          ncol=2, columnspacing=0.8, handlelength=1.2)
        fig.canvas.draw()
        bbox = legend.get_window_extent()
        if bbox.width > ax_bbox.width * 0.9:
            legend = ax.legend(loc='best', fontsize=8, framealpha=0.9, shadow=True,
                              ncol=3, columnspacing=0.6, handlelength=1.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Cost curve by trial saved to {output_path}")


def create_boxplot_comparison(results, hgs_cost, original_cost, original_gap, output_path='boxplot_comparison.png'):
    """Create boxplots comparing gap and cost."""
    costs = [r['final_cost'] for r in results]
    gaps = [r['gap'] for r in results if r['gap'] is not None]
    
    num_runs = len(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Pybind Local Search Results ({num_runs} runs)', fontsize=16, fontweight='bold')
    
    # Cost boxplot
    bp1 = ax1.boxplot([costs], tick_labels=['Pybind Results'], patch_artist=True, 
                      showmeans=True, meanline=True)
    bp1['boxes'][0].set_facecolor('lightblue')
    if hgs_cost is not None:
        ax1.axhline(y=hgs_cost, color='green', linestyle='--', linewidth=2, label=f'HGS Cost: {hgs_cost:.2f}')
    if original_cost is not None:
        ax1.axhline(y=original_cost, color='orange', linestyle='--', linewidth=2, 
                    label=f'Original Best: {original_cost:.2f}')
    ax1.set_ylabel('Cost', fontsize=12)
    ax1.set_title('Cost Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    # Gap boxplot (only draw if gap data is available)
    if gaps:
        bp2 = ax2.boxplot([gaps], tick_labels=['Pybind Results'], patch_artist=True,
                          showmeans=True, meanline=True)
        bp2['boxes'][0].set_facecolor('lightcoral')
        ax2.axhline(y=0, color='green', linestyle='--', linewidth=2, label='HGS Gap: 0.00%')
        if original_gap is not None:
            ax2.axhline(y=original_gap, color='orange', linestyle='--', linewidth=2,
                        label=f'Original Best Gap: {original_gap:.2f}%')
        ax2.set_ylabel('Gap to HGS (%)', fontsize=12)
        ax2.set_title('Gap Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No gap data available\n(HGS solution not provided)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Gap Distribution', fontsize=14, fontweight='bold')
    
    # Add statistics
    mean_cost = np.mean(costs)
    std_cost = np.std(costs)
    textstr1 = f'Mean: {mean_cost:.2f}\nStd: {std_cost:.2f}'
    
    ax1.text(0.02, 0.98, textstr1, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if gaps:
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        textstr2 = f'Mean: {mean_gap:.2f}%\nStd: {std_gap:.2f}%'
        ax2.text(0.02, 0.98, textstr2, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Boxplot comparison saved to {output_path}")


def main(instance_path, instance_index=0, num_vehicles=30, 
         basin_base_dir="basin_datasets0", hgs_solution_path=None,
         n_runs=30, max_iterations=100, run_id=None):
    """Main function."""
    print("=" * 80)
    print("Pybind vs Original cuOpt Comparison")
    print("=" * 80)
    
    # Load instance
    print(f"\n1. Loading VRP instance from {instance_path} (index={instance_index})")
    vrp_instance = create_vrp_instance_from_pkl(instance_path, instance_index, num_vehicles)
    num_orders = vrp_instance['num_orders']
    node_coords = vrp_instance['node_coords']
    print(f"   Instance: {num_orders} orders, {num_vehicles} vehicles")
    
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
    hgs_edges = None
    if hgs_solution_path and os.path.exists(hgs_solution_path):
        print(f"\n2. Loading HGS solution from {hgs_solution_path}")
        hgs_solution = load_hgs_solution_from_pkl(hgs_solution_path, instance_index)
        hgs_cost = hgs_solution['hgs_cost']
        hgs_routes = hgs_solution['hgs_routes']
        # Convert HGS routes to edges format
        if hgs_routes:
            hgs_routes_formatted = []
            for route in hgs_routes:
                if isinstance(route, list) and len(route) > 0:
                    if len(route) >= 2 and route[0] == 0 and route[-1] == 0:
                        hgs_routes_formatted.append(route)
                    else:
                        route_clean = [node for node in route if node != 0]
                        if route_clean:
                            hgs_routes_formatted.append([0] + route_clean + [0])
            if hgs_routes_formatted:
                hgs_edges = set()
                for route in hgs_routes_formatted:
                    for i in range(len(route) - 1):
                        u, v = int(route[i]), int(route[i + 1])
                        hgs_edges.add((min(u, v), max(u, v)))
        print(f"   HGS cost: {hgs_cost:.2f}")
    
    # Load all trajectory data in a single pass (optimized)
    print(f"\n3. Reading trajectory.jsonl (single pass)...")
    # If run_id is specified, filter by it; otherwise load all and use first run_id found
    trial_ids, first_solutions, best_original_solution, trajectory_run_id = load_trajectory_data(trajectory_path, num_orders, run_id=run_id)
    print(f"   Found {len(trial_ids)} trials")
    
    # Get run_id from parameter or trajectory
    if run_id is None:
        run_id = trajectory_run_id
        if run_id is None:
            raise ValueError("Could not find run_id in trajectory.jsonl. Please specify --run_id")
    print(f"   Run ID: {run_id}")
    
    # Create output directory for visualizations
    timestamp = str(run_id) if run_id is not None else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_base_dir = os.path.join(basin_dir, 'trials_plot')
    viz_dir = os.path.join(viz_base_dir, f'callback_cost_curve_by_trial_trial_colored_cf_{timestamp}_pybind')
    os.makedirs(viz_dir, exist_ok=True)
    print(f"   Visualization directory: {viz_dir}")
    
    # Path for trajectory_pybind.jsonl
    trajectory_pybind_path = os.path.join(basin_dir, 'trajectory_pybind.jsonl')
    print(f"   Pybind trajectory path: {trajectory_pybind_path}")
    
    # Display first solutions
    print(f"\n4. First solutions from each trial:")
    for trial_id, first_sol in sorted(first_solutions.items()):
        print(f"   Trial {trial_id}: cost={first_sol['cost']:.2f}, "
              f"global_iter={first_sol.get('global_iter')}, local_iter={first_sol.get('local_iter')}")
    
    if not first_solutions:
        raise ValueError("No first solutions found in trajectory.jsonl")
    
    # Display best solution
    print(f"\n5. Best solution from trajectory:")
    if best_original_solution:
        original_cost = best_original_solution['cost']
        original_routes = best_original_solution['routes']
        original_gap = calculate_gap(original_cost, hgs_cost) if hgs_cost else None
        print(f"   Original best cost: {original_cost:.2f}")
        if original_gap is not None:
            print(f"   Original best gap: {original_gap:.2f}%")
    else:
        raise ValueError("No best solution found in trajectory.jsonl")
    
    # Run local search: repeat complete run (all trials) n_runs times
    trial_ids_sorted = sorted(first_solutions.keys())
    num_trials = len(trial_ids_sorted)
    
    print(f"\n6. Running {n_runs} complete runs, each with {num_trials} trials...")
    results = []
    best_new_solution = None
    best_new_cost = float('inf')
    all_runs_history = []
    
    for rerun_id in range(n_runs):
        print(f"\n   Rerun {rerun_id+1}/{n_runs}: Processing {num_trials} trials...")
        rerun_results = []  # Results for this rerun only
        global_iter = 0

        for trial_idx, trial_id in enumerate(trial_ids_sorted):
            sol = first_solutions[trial_id]
            cuopt_env = create_vrp_instance_from_pkl(instance_path, instance_index, num_vehicles)['cuopt_env']
            
            try:
                history, final_cost, final_edges_hash, _, final_routes, global_iter = run_local_search_with_history(
                    cuopt_env, sol['routes'], global_iter, max_iterations=max_iterations
                )
                gap = calculate_gap(final_cost, hgs_cost) if hgs_cost else None
                result = {
                    'rerun_id': rerun_id, 'trial_id': trial_id,
                    'initial_cost': sol['cost'], 'final_cost': final_cost,
                    'gap': gap, 'routes': final_routes,
                    'edges_hash': final_edges_hash, 'history': history
                }
                results.append(result)
                rerun_results.append(result)
                if final_cost < best_new_cost:
                    best_new_cost = final_cost
                    best_new_solution = result
                gap_str = f", gap={gap:.2f}%" if gap else ""
                print(f"      Trial {trial_id} ({trial_idx+1}/{num_trials}): "
                      f"{sol['cost']:.2f} -> {final_cost:.2f}{gap_str}")
            finally:
                del cuopt_env
                gc.collect()
        
        # Visualize after each rerun: solution_comparison and cost_curve
        if rerun_results:
            print(f"\n   Creating visualizations for rerun {rerun_id+1}...")
            
            # Find best solution for this rerun
            rerun_best = min(rerun_results, key=lambda r: r['final_cost'])
            
            # Solution comparison for this rerun
            solution_comparison_path = os.path.join(viz_dir, f'run{rerun_id+1}_solution_comparison.png')
            visualize_solutions(hgs_routes, original_routes, rerun_best['routes'],
                               node_coords, hgs_cost, original_cost, rerun_best['final_cost'], 
                               solution_comparison_path)
            
            # Cost curve for this rerun
            cost_curve_path = os.path.join(viz_dir, f'run{rerun_id+1}_cost_curve_by_trial_pybind.png')
            plot_cost_curve_by_trial_pybind(rerun_results, hgs_cost, cost_curve_path)
    
    if not results:
        raise ValueError("No successful runs!")
    
    # Save trajectory_pybind.jsonl (append mode to accumulate results from multiple run_ids)
    print(f"\n7. Saving trajectory_pybind.jsonl...")
    total_records = 0
    # Check if file exists and if this run_id already has data
    file_exists = os.path.exists(trajectory_pybind_path)
    existing_run_ids = set()
    if file_exists:
        with open(trajectory_pybind_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if data.get('run_id') == run_id:
                        existing_run_ids.add(data.get('rerun_id'))
    
    # If this run_id already has all reruns, skip saving (avoid duplicates)
    if len(existing_run_ids) >= n_runs:
        print(f"   Run_id {run_id} already has {len(existing_run_ids)} reruns in file, skipping...")
    else:
        # Append mode to accumulate results from multiple run_ids
        # Use global_iter from history records (already calculated in run_local_search_with_history)
        # This ensures consistency with the actual execution
        
        with open(trajectory_pybind_path, 'a') as f:
            for result in results:
                rerun_id = result.get('rerun_id')
                history = result.get('history', [])
                for record_idx, record in enumerate(history):
                    # Use global_iter from record (already set in run_local_search_with_history)
                    global_iter = record.get('global_iter', 0)
                    
                    routes = record.get('routes', [])
                    cost = record.get('cost_after', 0)
                    edges = routes_to_edges(routes)
                    edges_set = set(edges)
                    edges_list = [[int(u), int(v)] for (u, v) in sorted(edges)]
                    
                    # Calculate edge_diff_to_hgs: number of different edges between current solution and HGS
                    edge_diff_to_hgs = len(edges_set ^ hgs_edges) if hgs_edges else None
                    
                    # Calculate gap to HGS
                    gap = calculate_gap(cost, hgs_cost) if hgs_cost else None
                    
                    trajectory_record = {
                        'instance_id': instance_id, 'run_id': run_id,
                        'trial_id': result['trial_id'], 'rerun_id': result['rerun_id'],
                        'optimum_id': None,  # Pybind reruns don't have optimum_id mapping
                        'global_iter': global_iter,  # Use recalculated global_iter (consistent with cost_curve)
                        'local_iter': record.get('local_iter', 0),
                        'is_final_of_trial': (record_idx == len(history) - 1),
                        'cost': cost,
                        'gap': gap,  # Gap to HGS
                        'edge_diff_to_hgs': edge_diff_to_hgs,  # Number of different edges vs HGS
                        'cost_gap_to_hgs_pct': gap,  # Same as gap (for consistency with original format)
                        'is_cycle_finder': record.get('is_circle_found', False),
                        'move_found': record.get('move_found', False),
                        'num_routes_after': len(routes),
                        'solution_flat': routes_to_solution_flat(routes, num_orders),
                        'edges': edges_list,
                        'edges_hash': record.get('edges_hash', '')
                    }
                    f.write(json.dumps(trajectory_record) + '\n')
                    total_records += 1
        
        print(f"   Saved {total_records} records to {trajectory_pybind_path}")
    
    # Statistics
    print(f"\n8. Statistics:")
    costs = [r['final_cost'] for r in results]
    gaps = [r['gap'] for r in results if r['gap'] is not None]
    print(f"   Best cost: {min(costs):.2f}, Worst: {max(costs):.2f}, Mean: {np.mean(costs):.2f}, Std: {np.std(costs):.2f}")
    if gaps:
        print(f"   Best gap: {min(gaps):.2f}%, Worst: {max(gaps):.2f}%, Mean: {np.mean(gaps):.2f}%, Std: {np.std(gaps):.2f}%")
    
    # Final visualization: boxplot (statistics for all reruns)
    # Only use best result from each rerun for boxplot
    print(f"\n9. Creating final visualization (boxplot for all {n_runs} reruns)...")
    rerun_best_results = {}  # rerun_id -> best result
    for r in results:
        rid = r['rerun_id']
        if rid not in rerun_best_results or r['final_cost'] < rerun_best_results[rid]['final_cost']:
            rerun_best_results[rid] = r
    
    boxplot_results = [rerun_best_results[rid] for rid in sorted(rerun_best_results.keys())]
    boxplot_path = os.path.join(viz_dir, 'boxplot_comparison.png')
    create_boxplot_comparison(
        boxplot_results, hgs_cost, original_cost, original_gap,
        boxplot_path
    )
    
    print(f"\n✓ Comparison complete!")
    print(f"  Original best cost: {original_cost:.2f}")
    print(f"  New best cost: {best_new_cost:.2f}")
    if hgs_cost:
        print(f"  HGS cost: {hgs_cost:.2f}")
        print(f"  Original gap: {original_gap:.2f}%")
        print(f"  New best gap: {calculate_gap(best_new_cost, hgs_cost):.2f}%")
    
    return results, best_new_solution, best_original_solution


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare pybind local search with original cuOpt')
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
    parser.add_argument('--n_runs', type=int, default=30,
                       help='Number of runs to perform (default: 30)')
    parser.add_argument('--max_iter', type=int, default=100,
                       help='Maximum iterations for local search (default: 100)')
    parser.add_argument('--run_id', type=str, default=None, dest='run_id',
                       help='Run ID to use (if not specified, will read from trajectory.jsonl)')
    
    args = parser.parse_args()
    
    main(
        args.instance_path, args.instance_index, args.num_vehicles,
        args.basin_base_dir, args.hgs_solution_path,
        args.n_runs, args.max_iter, args.run_id
    )
