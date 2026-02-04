#!/usr/bin/env python3
"""
Visualize Commitment Points from Training Data

1. Find all commitment points (earliest 100% convergence to final basin)
2. Calculate relative positions (local_iter / total_local_iter)
3. Visualize:
   - Cost curves for each run_id with commitment points marked as stars
   - Bar chart: trial_id vs total_local_iter with commitment point position in orange
   - Bar chart: commitment point relative position vs trial_id
   - Boxplot: distribution of relative and absolute positions
"""
import json
import os
import sys
import argparse
from collections import defaultdict
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available, cannot generate plots")


def load_training_data(training_data_path):
    """Load training data from JSONL file."""
    solutions = []
    with open(training_data_path, 'r') as f:
        for line in f:
            if line.strip():
                solutions.append(json.loads(line))
    return solutions


def find_all_commitment_points(training_data, convergence_threshold=1.0):
    """
    Find all commitment points for each trial (based on final basin).
    
    Args:
        training_data: List of solution records from training_data.jsonl
        convergence_threshold: Threshold for convergence (default: 1.0 = 100%)
    
    Returns:
        dict: {(run_id, trial_id): commitment_point_data}
    """
    trial_solutions = defaultdict(list)
    for sol in training_data:
        run_id = sol.get('run_id')
        trial_id = sol.get('trial_id')
        if run_id is not None and trial_id is not None:
            trial_solutions[(run_id, trial_id)].append(sol)
    
    commitment_points = {}
    
    for (run_id, trial_id), solutions in trial_solutions.items():
        solutions_sorted = sorted(solutions, 
                                 key=lambda x: (x.get('global_iter', 0), x.get('local_iter', 0)))
        
        if not solutions_sorted:
            continue
        
        final_solution = max(solutions_sorted, 
                           key=lambda x: (x.get('global_iter', 0), x.get('local_iter', 0)))
        
        final_basin_dist = final_solution.get('basin_distribution', {})
        if not final_basin_dist:
            continue
        
        final_basin_id = max(final_basin_dist.items(), key=lambda x: x[1])[0]
        final_basin_prob = final_basin_dist[final_basin_id]
        
        if final_basin_prob < convergence_threshold:
            continue
        
        # Find earliest point with 100% convergence (forward traversal)
        # This is the commitment point: the earliest intermediate state where
        # the final basin reaches 100% probability
        commitment_point = None
        commitment_local_iter = None
        commitment_global_iter = None
        
        for sol in solutions_sorted:
            basin_dist = sol.get('basin_distribution', {})
            if final_basin_id in basin_dist:
                prob = basin_dist[final_basin_id]
                if prob >= convergence_threshold:
                    commitment_point = sol
                    commitment_local_iter = sol.get('local_iter', 0)
                    commitment_global_iter = sol.get('global_iter', 0)
                    break
        
        # Verify: check if commitment point is indeed the earliest
        # (This is a sanity check - the forward traversal should already find the earliest)
        if commitment_point:
            # Double-check: verify this is truly the earliest point
            earliest_verified = True
            for sol in solutions_sorted:
                if sol == commitment_point:
                    break  # Stop when we reach the commitment point
                basin_dist = sol.get('basin_distribution', {})
                if final_basin_id in basin_dist:
                    prob = basin_dist[final_basin_id]
                    if prob >= convergence_threshold:
                        # This should not happen if our logic is correct
                        print(f"WARNING: Found earlier commitment point for run_id={run_id}, trial_id={trial_id}: "
                              f"local_iter={sol.get('local_iter')} (found at {commitment_local_iter})")
                        earliest_verified = False
                        break
        
        if commitment_point:
            total_local_iters = len(solutions_sorted)
            commitment_local_iter = commitment_point.get('local_iter', 0)
            relative_position = (commitment_local_iter + 1) / total_local_iters if total_local_iters > 0 else 0.0
            
            commitment_points[(run_id, trial_id)] = {
                'commitment_point': commitment_point,
                'final_basin_id': final_basin_id,
                'final_solution': final_solution,
                'total_local_iters': total_local_iters,
                'commitment_local_iter': commitment_local_iter,
                'relative_position': relative_position,
                'global_iter': commitment_point.get('global_iter', 0),
            }
    
    return commitment_points


def visualize_commitment_points(training_data_path, output_dir, convergence_threshold=1.0):
    """Main visualization function."""
    if not HAS_MPL:
        print("Error: matplotlib not available, cannot generate plots")
        return
    
    print(f"Loading training data from: {training_data_path}")
    training_data = load_training_data(training_data_path)
    print(f"  Loaded {len(training_data)} solution records")
    
    print(f"\nFinding all commitment points (threshold={convergence_threshold})...")
    commitment_points = find_all_commitment_points(training_data, convergence_threshold)
    print(f"  Found {len(commitment_points)} commitment points")
    
    if not commitment_points:
        print("No commitment points found. Exiting.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Organize data by run_id
    data_by_run = defaultdict(list)
    for (run_id, trial_id), cp_data in commitment_points.items():
        data_by_run[run_id].append((trial_id, cp_data))
    
    # Sort trials within each run
    for run_id in data_by_run:
        data_by_run[run_id].sort(key=lambda x: x[0])
    
    # ===== Plot 1: Cost curves for each run_id with commitment points marked =====
    print("\nPlotting cost curves...")
    for run_id, trial_data_list in data_by_run.items():
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get all solutions for this run_id
        run_solutions = defaultdict(list)
        for sol in training_data:
            if sol.get('run_id') == run_id:
                trial_id = sol.get('trial_id')
                if trial_id is not None:
                    run_solutions[trial_id].append(sol)
        
        # Compute duplicate groups by comparing final solutions' edges (similar to utils.py)
        # Extract final solution for each trial
        trial_metadata = []
        final_solutions = {}
        for trial_id, solutions in sorted(run_solutions.items()):
            solutions_sorted = sorted(solutions,
                                    key=lambda x: (x.get('global_iter', 0), x.get('local_iter', 0)))
            final_sol = solutions_sorted[-1] if solutions_sorted else None
            if final_sol:
                final_solutions[trial_id] = final_sol
                trial_metadata.append({
                    'local_search_id': trial_id,
                    'trial_id': trial_id,
                })
        
        # Build duplicate_groups by comparing edges
        duplicate_groups = {}
        optimum_id_map = {}
        trial_idx_to_trial_id = {}
        for idx, meta in enumerate(trial_metadata):
            trial_id = meta['trial_id']
            trial_idx_to_trial_id[idx] = trial_id
            
            final_sol = final_solutions.get(trial_id)
            if final_sol:
                # Extract edges from initial_solution
                initial_sol = final_sol.get('initial_solution', {})
                edges = initial_sol.get('edges', [])
                if edges:
                    # Convert edges to sorted tuple for comparison
                    edges_tuple = tuple(sorted([tuple(e) if isinstance(e, list) else e for e in edges]))
                    
                    if edges_tuple not in optimum_id_map:
                        optimum_id = len(optimum_id_map)
                        optimum_id_map[edges_tuple] = optimum_id
                        duplicate_groups[edges_tuple] = [idx]
                    else:
                        duplicate_groups[edges_tuple].append(idx)
        
        # Build trial color mapping from duplicate_groups (similar to utils.py)
        trial_colors = {}
        unique_color = 'black'
        duplicate_colors_list = plt.cm.tab10(np.linspace(0, 1, 10))
        duplicate_colors = [tuple(c) if isinstance(c, np.ndarray) else c for c in duplicate_colors_list]
        # Remove orange and black/dark colors
        cycle_finder_color_rgb = (0xF1/255.0, 0x8F/255.0, 0x01/255.0)
        filtered_colors = []
        for c in duplicate_colors:
            if len(c) >= 3:
                c_rgb = np.array(c[:3])
                is_orange = np.allclose(c_rgb, np.array(cycle_finder_color_rgb), atol=0.15)
                is_black = np.sum(c_rgb) < 0.3
                if not is_orange and not is_black:
                    filtered_colors.append(c)
        duplicate_colors = filtered_colors
        if len(duplicate_colors) == 0:
            duplicate_colors = [tuple(c) if isinstance(c, np.ndarray) else c 
                               for c in plt.cm.Set2(np.linspace(0, 1, 8))]
            duplicate_colors = [c for c in duplicate_colors 
                               if len(c) >= 3 and np.sum(np.array(c[:3])) >= 0.3 and
                               not np.allclose(np.array(c[:3]), np.array(cycle_finder_color_rgb), atol=0.15)]
        
        color_idx = 0
        for edges_tuple, trial_indices in duplicate_groups.items():
            if isinstance(trial_indices, np.ndarray):
                trial_indices = trial_indices.tolist()
            elif not isinstance(trial_indices, list):
                trial_indices = list(trial_indices)
            
            trial_ids = []
            for idx in trial_indices:
                idx_int = int(idx) if not isinstance(idx, int) else idx
                if idx_int in trial_idx_to_trial_id:
                    trial_ids.append(trial_idx_to_trial_id[idx_int])
            
            if len(trial_ids) == 1:
                trial_colors[trial_ids[0]] = unique_color
            else:
                if len(duplicate_colors) == 0:
                    color = (0.2, 0.6, 0.8)
                else:
                    color = duplicate_colors[color_idx % len(duplicate_colors)]
                if isinstance(color, np.ndarray):
                    color = tuple(color)
                if color == unique_color or (isinstance(color, tuple) and len(color) >= 3 and np.sum(np.array(color[:3])) < 0.3):
                    color = (0.2, 0.6, 0.8)
                for tid in trial_ids:
                    trial_colors[tid] = color
                color_idx += 1
        
        # Build truly_duplicate_trial_ids set
        truly_duplicate_trial_ids = set()
        for edges_tuple, trial_indices in duplicate_groups.items():
            trial_ids = []
            for idx in trial_indices:
                idx_int = int(idx) if not isinstance(idx, int) else idx
                if idx_int in trial_idx_to_trial_id:
                    trial_ids.append(trial_idx_to_trial_id[idx_int])
            if len(trial_ids) > 1:
                truly_duplicate_trial_ids.update(trial_ids)
        
        # Track which colors have been labeled
        labeled_colors = set()
        
        # Helper function to get hashable color
        def get_hashable_color(color):
            if isinstance(color, np.ndarray):
                return tuple(color)
            return color
        
        # Helper function to get trial label (similar to utils.py)
        def get_trial_label(trial_id, trial_colors, labeled_colors, truly_duplicate_trial_ids, 
                           duplicate_groups, trial_idx_to_trial_id, unique_color):
            label = None
            if trial_id in trial_colors:
                color = trial_colors[trial_id]
                hashable_color = get_hashable_color(color)
                if hashable_color not in labeled_colors:
                    is_truly_duplicate = trial_id in truly_duplicate_trial_ids
                    
                    if is_truly_duplicate:
                        duplicate_trials = []
                        for edges_tuple, trial_indices in duplicate_groups.items():
                            trial_ids = []
                            for idx in trial_indices:
                                idx_int = int(idx) if not isinstance(idx, int) else idx
                                if idx_int in trial_idx_to_trial_id:
                                    trial_ids.append(trial_idx_to_trial_id[idx_int])
                            if trial_id in trial_ids and len(trial_ids) > 1:
                                duplicate_trials = sorted(trial_ids)
                                break
                        if len(duplicate_trials) > 1:
                            label = f"Trials {duplicate_trials} (duplicate)"
                        else:
                            label = f"Trial {trial_id}"
                    else:
                        if hashable_color == unique_color or (isinstance(hashable_color, str) and hashable_color == 'black'):
                            if unique_color not in labeled_colors:
                                label = "Unique trials"
                                labeled_colors.add(unique_color)
                        else:
                            label = f"Trial {trial_id}"
                    labeled_colors.add(hashable_color)
            else:
                if unique_color not in labeled_colors:
                    label = "Unique trials"
                    labeled_colors.add(unique_color)
            return label
        
        # Plot cost curve for each trial (similar to utils.py style)
        for trial_id, solutions in sorted(run_solutions.items()):
            solutions_sorted = sorted(solutions,
                                    key=lambda x: (x.get('global_iter', 0), x.get('local_iter', 0)))
            
            # Get color for this trial
            if trial_id in trial_colors:
                trial_color = trial_colors[trial_id]
                if isinstance(trial_color, np.ndarray):
                    trial_color = tuple(trial_color)
            else:
                trial_color = unique_color
            
            # Extract costs and global iterations
            trial_points = []
            for sol in solutions_sorted:
                cost = sol.get('initial_solution', {}).get('cost')
                if cost is not None:
                    global_iter = sol.get('global_iter', 0)
                    trial_points.append((global_iter, cost))
            
            if trial_points:
                # Plot fast search points (similar to utils.py)
                iters, costs = zip(*sorted(trial_points, key=lambda x: x[0]))
                label = get_trial_label(trial_id, trial_colors, labeled_colors, truly_duplicate_trial_ids,
                                       duplicate_groups, trial_idx_to_trial_id, unique_color)
                ax.plot(iters, costs, marker='o', linestyle='-', linewidth=1.5, markersize=4,
                       color=trial_color, label=label, alpha=0.6, zorder=2)
                
                # Plot trial trajectory line (similar to utils.py)
                if len(trial_points) > 1:
                    ax.plot(iters, costs, linestyle='-', linewidth=1.5, color=trial_color,
                           alpha=0.7, zorder=1, label=None)
        
        # Mark commitment points with stars (smaller size)
        commitment_labeled = False
        for trial_id, cp_data in trial_data_list:
            cp = cp_data['commitment_point']
            cp_global_iter = cp_data['global_iter']
            cp_cost = cp.get('initial_solution', {}).get('cost')
            
            # Verify commitment point is in the correct trial's cost curve
            if trial_id in run_solutions:
                trial_sols = run_solutions[trial_id]
                # Check if this commitment point exists in the trial's solutions
                found_in_trial = False
                for sol in trial_sols:
                    if (sol.get('global_iter') == cp_global_iter and 
                        sol.get('initial_solution', {}).get('cost') == cp_cost):
                        found_in_trial = True
                        break
                
                if not found_in_trial:
                    print(f"WARNING: Commitment point for trial {trial_id} (global_iter={cp_global_iter}, cost={cp_cost}) "
                          f"not found in trial's solutions")
            
            if cp_cost is not None:
                # Use smaller star marker (s=100 instead of 300)
                ax.scatter(cp_global_iter, cp_cost, marker='*', s=100, 
                          color='red', edgecolors='black', linewidths=1.0,
                          zorder=10, label='Commitment Point' if not commitment_labeled else '',
                          alpha=0.9)
                commitment_labeled = True
        
        # Setup axes and title (similar to utils.py)
        ax.set_xlabel('Global Iteration', fontsize=13, fontweight='bold')
        ax.set_ylabel('Objective Value (Cost)', fontsize=13, fontweight='bold')
        
        # Calculate best cost
        best_cost = None
        all_costs = []
        for sol in training_data:
            if sol.get('run_id') == run_id:
                cost = sol.get('initial_solution', {}).get('cost')
                if cost is not None:
                    all_costs.append(cost)
        if all_costs:
            best_cost = min(all_costs)
        
        # Build title
        title = f'Cost Evolution During Optimization - Run {run_id}'
        if best_cost is not None:
            title += f' | Cost: {best_cost:.2f}'
        
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend with automatic wrapping (similar to utils.py)
        legend = ax.legend(loc='best', fontsize=10, framealpha=0.9, shadow=True, 
                          ncol=1, columnspacing=1.0, handlelength=1.5)
        # Adjust legend to fit within plot
        fig.canvas.draw()
        bbox = legend.get_window_extent()
        ax_bbox = ax.get_window_extent()
        if bbox.width > ax_bbox.width * 0.9:
            # If legend is too wide, use multiple columns
            legend = ax.legend(loc='best', fontsize=9, framealpha=0.9, shadow=True,
                              ncol=2, columnspacing=0.8, handlelength=1.2)
            fig.canvas.draw()
            bbox = legend.get_window_extent()
            if bbox.width > ax_bbox.width * 0.9:
                # If still too wide, use 3 columns with smaller font
                legend = ax.legend(loc='best', fontsize=8, framealpha=0.9, shadow=True,
                                  ncol=3, columnspacing=0.6, handlelength=1.0)
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, f'run_{run_id}_cost_curves.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")
        
        # ===== Plot 2: Bar chart - trial_id vs total_local_iter with commitment point position =====
        print(f"\nPlotting bar chart: trial_id vs total_local_iter for run {run_id}...")
        
        # Collect trial data for this run_id, sorted by trial_id
        run_trial_data = []
        for trial_id, cp_data in trial_data_list:
            run_trial_data.append({
                'trial_id': trial_id,
                'total_local_iters': cp_data['total_local_iters'],
                'commitment_local_iter': cp_data['commitment_local_iter'],
            })
        
        # Sort by trial_id
        run_trial_data.sort(key=lambda x: x['trial_id'])
        
        trial_ids = [d['trial_id'] for d in run_trial_data]
        total_iters = [d['total_local_iters'] for d in run_trial_data]
        commitment_iters = [d['commitment_local_iter'] + 1 for d in run_trial_data]  # +1 because relative position is (local_iter+1)/total
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Plot total local_iters as blue bars (bottom part)
        # We'll use stacked bars: orange (commitment position) on bottom, blue (remaining) on top
        remaining_iters = [total - commit for total, commit in zip(total_iters, commitment_iters)]
        
        # Plot commitment point position as orange bars (bottom)
        commitment_bars = ax.bar(range(len(trial_ids)), commitment_iters, 
                               color='orange', edgecolor='darkorange', 
                               linewidth=1.5, label='Commitment Point Position')
        
        # Plot remaining iterations as blue bars (on top of orange)
        bars = ax.bar(range(len(trial_ids)), remaining_iters, bottom=commitment_iters,
                      color='lightblue', edgecolor='navy', linewidth=1.5, 
                      label='Remaining Iterations')
        
        ax.set_xlabel('Trial ID (Sorted)', fontsize=12)
        ax.set_ylabel('Local Iteration', fontsize=12)
        ax.set_title(f'Total Local Iterations per Trial - Run {run_id} (Orange = Commitment Point Position)', fontsize=14)
        ax.set_xticks(range(len(trial_ids)))
        ax.set_xticklabels(trial_ids, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, f'run_{run_id}_trial_id_vs_total_local_iter.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")
        
        # ===== Plot 3: Bar chart - commitment point relative position vs trial_id =====
        print(f"Plotting bar chart: commitment point relative position vs trial_id for run {run_id}...")
        
        relative_positions = [cp_data['relative_position'] for _, cp_data in trial_data_list]
        sorted_trial_ids = sorted([trial_id for trial_id, _ in trial_data_list])
        
        # Reorder relative_positions to match sorted_trial_ids
        relative_positions_sorted = []
        for trial_id in sorted_trial_ids:
            for tid, cp_data in trial_data_list:
                if tid == trial_id:
                    relative_positions_sorted.append(cp_data['relative_position'])
                    break
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        bars = ax.bar(range(len(sorted_trial_ids)), relative_positions_sorted, 
                     color='steelblue', edgecolor='navy', linewidth=1.5)
        
        ax.set_xlabel('Trial ID (Sorted)', fontsize=12)
        ax.set_ylabel('Relative Position (Commitment Point)', fontsize=12)
        ax.set_title(f'Commitment Point Relative Position vs Trial ID - Run {run_id}', fontsize=14)
        ax.set_xticks(range(len(sorted_trial_ids)))
        ax.set_xticklabels(sorted_trial_ids, rotation=45, ha='right')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal line at y=0.5 for reference
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, f'run_{run_id}_commitment_point_relative_position.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")
        
        # ===== Plot 4: Boxplot - relative and absolute positions =====
        print(f"Plotting boxplot: relative and absolute positions for run {run_id}...")
        
        relative_positions_run = [cp_data['relative_position'] for _, cp_data in trial_data_list]
        absolute_positions_run = [cp_data['commitment_local_iter'] for _, cp_data in trial_data_list]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Boxplot for relative positions
        bp1 = ax1.boxplot(relative_positions_run, vert=True, patch_artist=True,
                          tick_labels=['Relative Position'], showmeans=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
        ax1.set_ylabel('Relative Position', fontsize=12)
        ax1.set_title(f'Distribution of Commitment Point Relative Positions - Run {run_id}', fontsize=13)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 1.1])
        
        # Boxplot for absolute positions
        bp2 = ax2.boxplot(absolute_positions_run, vert=True, patch_artist=True,
                          tick_labels=['Absolute Position'], showmeans=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightcoral')
        ax2.set_ylabel('Local Iteration', fontsize=12)
        ax2.set_title(f'Distribution of Commitment Point Absolute Positions - Run {run_id}', fontsize=13)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, f'run_{run_id}_commitment_point_positions_boxplot.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")
    
    # ===== Save summary statistics =====
    # Collect all relative and absolute positions for summary
    relative_positions_all = [cp_data['relative_position'] for cp_data in commitment_points.values()]
    absolute_positions_all = [cp_data['commitment_local_iter'] for cp_data in commitment_points.values()]
    
    summary = {
        'total_commitment_points': len(commitment_points),
        'relative_positions': {
            'mean': float(np.mean(relative_positions_all)),
            'std': float(np.std(relative_positions_all)),
            'min': float(np.min(relative_positions_all)),
            'max': float(np.max(relative_positions_all)),
            'median': float(np.median(relative_positions_all)),
        },
        'absolute_positions': {
            'mean': float(np.mean(absolute_positions_all)),
            'std': float(np.std(absolute_positions_all)),
            'min': int(np.min(absolute_positions_all)),
            'max': int(np.max(absolute_positions_all)),
            'median': float(np.median(absolute_positions_all)),
        },
        'commitment_points': [
            {
                'run_id': run_id,
                'trial_id': trial_id,
                'global_iter': cp_data['global_iter'],
                'local_iter': cp_data['commitment_local_iter'],
                'total_local_iters': cp_data['total_local_iters'],
                'relative_position': float(cp_data['relative_position']),
            }
            for (run_id, trial_id), cp_data in sorted(commitment_points.items())
        ]
    }
    
    summary_file = os.path.join(output_dir, 'commitment_points_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")
    
    print(f"\n{'='*80}")
    print("Visualization Complete!")
    print(f"{'='*80}")
    print(f"Total commitment points: {len(commitment_points)}")
    print(f"\nRelative Position Statistics:")
    print(f"  Mean: {summary['relative_positions']['mean']:.4f}")
    print(f"  Std:  {summary['relative_positions']['std']:.4f}")
    print(f"  Min:  {summary['relative_positions']['min']:.4f}")
    print(f"  Max:  {summary['relative_positions']['max']:.4f}")
    print(f"  Median: {summary['relative_positions']['median']:.4f}")
    print(f"\nAbsolute Position Statistics:")
    print(f"  Mean: {summary['absolute_positions']['mean']:.2f}")
    print(f"  Std:  {summary['absolute_positions']['std']:.2f}")
    print(f"  Min:  {summary['absolute_positions']['min']}")
    print(f"  Max:  {summary['absolute_positions']['max']}")
    print(f"  Median: {summary['absolute_positions']['median']:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Commitment Points from Training Data')
    parser.add_argument('--training_data', type=str, 
                       default='/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#0_move_found_jump/training_data.jsonl',
                       help='Path to training_data.jsonl file')
    parser.add_argument('--convergence_threshold', type=float, default=1.0,
                       help='Convergence threshold for commitment point (default: 1.0 = 100%%)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.training_data):
        print(f"Error: Training data file not found: {args.training_data}")
        sys.exit(1)
    
    # Auto-generate output_dir from training_data path
    training_data_dir = os.path.dirname(os.path.abspath(args.training_data))
    output_dir = os.path.join(training_data_dir, 'commitment_points_visualization')
    print(f"Output directory: {output_dir}")
    
    visualize_commitment_points(args.training_data, output_dir, args.convergence_threshold)


if __name__ == "__main__":
    main()
