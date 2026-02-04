#!/usr/bin/env python3
"""
Visualize Basin Collapsing Experiment Results

1. Visualize commitment point positions (global_iter vs local_iter)
2. Visualize gap distribution of converged basins
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

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_training_data(training_data_path):
    """Load training data from JSONL file."""
    solutions = []
    with open(training_data_path, 'r') as f:
        for line in f:
            if line.strip():
                solutions.append(json.loads(line))
    return solutions


def get_basin_gaps_from_training_data(training_data, basin_hashes):
    """
    Extract gap_to_hgs for each basin hash from training_data.
    
    Returns:
        dict: {basin_hash: gap_to_hgs} (may be None if not found)
    """
    basin_gaps = {}
    
    for record in training_data:
        basin_features = record.get('basin_features', {})
        for basin_hash, features in basin_features.items():
            if basin_hash in basin_hashes and basin_hash not in basin_gaps:
                gap = features.get('gap_to_hgs')
                if gap is not None:
                    basin_gaps[basin_hash] = float(gap)
    
    return basin_gaps


def get_trial_total_local_iters(training_data_path, commitment_points):
    """Get total local iterations for each trial from training_data."""
    trial_total_iters = {}
    
    if not training_data_path or not os.path.exists(training_data_path):
        return trial_total_iters
    
    print(f"Loading training data to get total local iterations...")
    training_data = load_training_data(training_data_path)
    
    # Group by (run_id, trial_id)
    trial_solutions = defaultdict(list)
    for sol in training_data:
        run_id = sol.get('run_id')
        trial_id = sol.get('trial_id')
        if run_id is not None and trial_id is not None:
            trial_solutions[(run_id, trial_id)].append(sol)
    
    # Get max local_iter for each trial
    for (run_id, trial_id), solutions in trial_solutions.items():
        solutions_sorted = sorted(solutions, 
                                 key=lambda x: (x.get('global_iter', 0), x.get('local_iter', 0)))
        if solutions_sorted:
            max_local_iter = max(s.get('local_iter', 0) for s in solutions_sorted)
            trial_total_iters[(run_id, trial_id)] = max_local_iter + 1  # +1 because 0-indexed
    
    return trial_total_iters


def load_all_basin_results(results_file):
    """Load all basin results from summary file or individual basin files."""
    results_dir = os.path.dirname(os.path.abspath(results_file))
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Check if this is a summary file (has 'basin_results' key)
    if 'basin_results' in data:
        # This is a summary file, need to load individual basin files
        print(f"Loading individual basin result files from {results_dir}...")
        all_commitment_points = []
        all_basin_counts = {}
        all_basin_data = {}
        
        # Find all basin_*_results.json files
        import glob
        basin_files = glob.glob(os.path.join(results_dir, 'basin_*_results.json'))
        
        if not basin_files:
            print(f"Warning: No basin_*_results.json files found in {results_dir}")
            return [], {}, {}
        
        for basin_file in basin_files:
            with open(basin_file, 'r') as f:
                basin_data = json.load(f)
            
            # Collect commitment points
            cp_list = basin_data.get('commitment_points', [])
            # Flatten commitment point data
            for cp in cp_list:
                # Convert to int if string
                global_iter = cp.get('global_iter', 0)
                local_iter = cp.get('local_iter', 0)
                if isinstance(global_iter, str):
                    global_iter = int(global_iter)
                if isinstance(local_iter, str):
                    local_iter = int(local_iter)
                
                all_commitment_points.append({
                    'run_id': cp.get('run_id'),
                    'trial_id': cp.get('trial_id'),
                    'global_iter': global_iter,
                    'local_iter': local_iter,
                })
            
            # Aggregate basin counts and data
            basin_counts = basin_data.get('basin_counts', {})
            basin_data_dict = basin_data.get('basin_data', {})
            
            for basin_hash, count in basin_counts.items():
                all_basin_counts[basin_hash] = all_basin_counts.get(basin_hash, 0) + count
                if basin_hash not in all_basin_data:
                    all_basin_data[basin_hash] = basin_data_dict.get(basin_hash, {})
        
        print(f"  Loaded {len(basin_files)} basin files")
        print(f"  Total commitment points: {len(all_commitment_points)}")
        print(f"  Total unique basins: {len(all_basin_counts)}")
        
        return all_commitment_points, all_basin_counts, all_basin_data
    else:
        # This is an individual basin file or old format
        commitment_points = data.get('commitment_points', [])
        # Flatten commitment point data if needed
        if commitment_points and isinstance(commitment_points[0], dict):
            if 'commitment_point' in commitment_points[0]:
                # Old format with nested structure
                flattened = []
                for cp in commitment_points:
                    global_iter = cp.get('commitment_point', {}).get('global_iter', 0)
                    local_iter = cp.get('commitment_point', {}).get('local_iter', 0)
                    # Convert to int if string
                    if isinstance(global_iter, str):
                        global_iter = int(global_iter)
                    if isinstance(local_iter, str):
                        local_iter = int(local_iter)
                    flattened.append({
                        'run_id': cp.get('run_id'),
                        'trial_id': cp.get('trial_id'),
                        'global_iter': global_iter,
                        'local_iter': local_iter,
                    })
                commitment_points = flattened
            else:
                # Ensure int types
                for cp in commitment_points:
                    if isinstance(cp.get('global_iter'), str):
                        cp['global_iter'] = int(cp['global_iter'])
                    if isinstance(cp.get('local_iter'), str):
                        cp['local_iter'] = int(cp['local_iter'])
        
        basin_counts = data.get('basin_counts', {})
        basin_data_dict = data.get('basin_data', {})
        
        return commitment_points, basin_counts, basin_data_dict


def visualize_commitment_points(results_file, output_dir, training_data_path=None):
    """Visualize commitment point positions with relative and absolute positions."""
    commitment_points, _, _ = load_all_basin_results(results_file)
    
    if not commitment_points:
        print("No commitment points found in results file(s)")
        return
    
    # Get total local iterations for each trial
    trial_total_iters = get_trial_total_local_iters(training_data_path, commitment_points)
    
    global_iters = []
    local_iters = []
    relative_positions = []
    trial_ids = []
    
    for cp in commitment_points:
        run_id = cp.get('run_id')
        trial_id = cp.get('trial_id')
        global_iter = cp.get('global_iter', 0)
        local_iter = cp.get('local_iter', 0)
        
        global_iters.append(global_iter)
        local_iters.append(local_iter)
        trial_ids.append(trial_id)
        
        # Calculate relative position: (local_iter + 1) / total_iters
        # e.g., local_iter=0, total=5 -> 1/5 = 0.2 (first position)
        #       local_iter=1, total=10 -> 2/10 = 0.2 (second position)
        total_iters = trial_total_iters.get((run_id, trial_id))
        if total_iters and total_iters > 0:
            relative_pos = (local_iter + 1) / total_iters
            relative_positions.append(relative_pos)
        else:
            relative_positions.append(None)
    
    if not HAS_MPL:
        print("matplotlib not available, skipping visualization")
        return
    
    # Filter out None values for relative positions
    valid_relative = [r for r in relative_positions if r is not None]
    
    # Create figure with 2x2 subplots (removed scatter plot)
    fig = plt.figure(figsize=(16, 10))
    
    # Subplot 1: Box plot for absolute local_iter
    ax1 = plt.subplot(2, 2, 1)
    ax1.boxplot(local_iters, vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7),
               medianprops=dict(color='red', linewidth=2))
    ax1.set_ylabel('Local Iteration (Absolute)', fontsize=12)
    ax1.set_title('Box Plot of Absolute Local Iterations', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticklabels(['Commitment Points'])
    
    # Subplot 2: Histogram for absolute local_iter
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(local_iters, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.set_xlabel('Local Iteration (Absolute)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Histogram of Absolute Local Iterations', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = f"Mean: {np.mean(local_iters):.1f}\n"
    stats_text += f"Median: {np.median(local_iters):.1f}\n"
    stats_text += f"Min: {min(local_iters)}\n"
    stats_text += f"Max: {max(local_iters)}"
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)
    
    # Subplot 3: Box plot for relative positions (separate)
    if valid_relative:
        ax3 = plt.subplot(2, 2, 3)
        ax3.boxplot(valid_relative, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightgreen', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax3.set_ylabel('Relative Position ((local_iter+1) / total)', fontsize=12)
        ax3.set_title('Box Plot of Relative Positions', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_xticklabels(['Commitment Points'])
        
        # Add statistics
        stats_text = f"Mean: {np.mean(valid_relative):.3f}\n"
        stats_text += f"Median: {np.median(valid_relative):.3f}\n"
        stats_text += f"Min: {min(valid_relative):.3f}\n"
        stats_text += f"Max: {max(valid_relative):.3f}\n"
        stats_text += f"Valid: {len(valid_relative)}/{len(relative_positions)}"
        ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)
    else:
        ax3 = plt.subplot(2, 2, 3)
        ax3.text(0.5, 0.5, 'No relative position data\n(need training_data.jsonl)', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Box Plot of Relative Positions', fontsize=13, fontweight='bold')
    
    # Subplot 4: Histogram for relative positions (separate)
    if valid_relative:
        ax4 = plt.subplot(2, 2, 4)
        ax4.hist(valid_relative, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
        ax4.set_xlabel('Relative Position ((local_iter+1) / total)', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Histogram of Relative Positions', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        stats_text = f"Mean: {np.mean(valid_relative):.3f}\n"
        stats_text += f"Median: {np.median(valid_relative):.3f}\n"
        stats_text += f"Min: {min(valid_relative):.3f}\n"
        stats_text += f"Max: {max(valid_relative):.3f}"
        ax4.text(0.98, 0.98, stats_text, transform=ax4.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)
    else:
        ax4 = plt.subplot(2, 2, 4)
        ax4.text(0.5, 0.5, 'No relative position data\n(need training_data.jsonl)', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Histogram of Relative Positions', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'commitment_points_positions.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Commitment points visualization saved to: {output_file}")
    print(f"  Total commitment points: {len(commitment_points)}")
    print(f"  Absolute local_iter - Mean: {np.mean(local_iters):.2f}, Median: {np.median(local_iters):.2f}")
    if valid_relative:
        print(f"  Relative position - Mean: {np.mean(valid_relative):.3f}, Median: {np.median(valid_relative):.3f}")
    plt.close()


def calculate_gap(my_cost, hgs_cost):
    """Calculate gap between my solution and HGS solution"""
    if hgs_cost == 0:
        return float('inf') if my_cost > 0 else 0.0
    gap_percent = ((my_cost - hgs_cost) / hgs_cost) * 100.0
    return gap_percent


def visualize_gap_distribution(results_file, training_data_path, output_dir, 
                              hgs_solution_path=None, instance_index=0):
    """Visualize gap distribution of converged basins."""
    _, basin_counts, basin_data = load_all_basin_results(results_file)
    
    if not basin_counts:
        print("No basin counts found in results file(s)")
        return
    
    # Try to get hgs_cost if hgs_solution_path is provided
    hgs_cost = None
    if hgs_solution_path and os.path.exists(hgs_solution_path):
        try:
            import pickle
            with open(hgs_solution_path, 'rb') as f:
                solutions = pickle.load(f)
            
            if instance_index >= len(solutions):
                raise ValueError(f"Instance index {instance_index} out of range (max: {len(solutions)-1})")
            
            solution_tuple = solutions[instance_index]
            hgs_cost = solution_tuple[0] * 100.0  # scale factor
            print(f"Loaded HGS cost: {hgs_cost:.2f}")
        except Exception as e:
            print(f"Warning: Failed to load HGS solution: {e}")
    
    # Prepare data for visualization
    gaps = []
    counts = []
    basin_hash_labels = []
    
    # First, try to calculate gap from mean_cost in basin_data
    if hgs_cost is not None:
        for basin_hash, count in basin_counts.items():
            basin_info = basin_data.get(basin_hash, {})
            mean_cost = basin_info.get('mean_cost')
            if mean_cost is not None:
                gap = calculate_gap(mean_cost, hgs_cost)
                gaps.append(gap)
                counts.append(count)
                basin_hash_labels.append(basin_hash[:8])
    
    # If no gaps calculated yet, try to get from training_data
    if not gaps and training_data_path and os.path.exists(training_data_path):
        print(f"Loading training data from {training_data_path}...")
        training_data = load_training_data(training_data_path)
        print(f"Loaded {len(training_data)} training records")
        
        basin_hashes = set(basin_counts.keys())
        basin_gaps = get_basin_gaps_from_training_data(training_data, basin_hashes)
        
        for basin_hash, count in basin_counts.items():
            gap = basin_gaps.get(basin_hash)
            if gap is not None:
                gaps.append(gap)
                counts.append(count)
                basin_hash_labels.append(basin_hash[:8])
    
    if not gaps:
        print("Warning: No gap information found for converged basins")
        if hgs_cost is None:
            print("  Please provide --hgs_solution_path to calculate gaps from mean_cost")
        return
    
    if not HAS_MPL:
        print("matplotlib not available, skipping visualization")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Histogram of gap distribution
    ax1.hist(gaps, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Gap to HGS (%)', fontsize=12)
    ax1.set_ylabel('Number of Basins', fontsize=12)
    ax1.set_title('Gap Distribution of Converged Basins', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_gap = np.mean(gaps)
    median_gap = np.median(gaps)
    stats_text = f"Mean: {mean_gap:.2f}%\nMedian: {median_gap:.2f}%\n"
    stats_text += f"Min: {min(gaps):.2f}%\nMax: {max(gaps):.2f}%\n"
    stats_text += f"Total basins: {len(gaps)}"
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)
    
    # Subplot 2: Scatter plot of gap vs count (weighted by frequency)
    # Size of points represents count
    scatter = ax2.scatter(gaps, counts, s=[c*2 for c in counts], 
                         alpha=0.6, c=gaps, cmap='RdYlGn_r', 
                         edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Gap to HGS (%)', fontsize=12)
    ax2.set_ylabel('Count (Number of Runs)', fontsize=12)
    ax2.set_title('Gap vs Frequency of Converged Basins', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Gap to HGS (%)', fontsize=10)
    
    # Add top basins annotation
    sorted_indices = sorted(range(len(gaps)), key=lambda i: counts[i], reverse=True)
    top_n = min(5, len(gaps))
    for i in range(top_n):
        idx = sorted_indices[i]
        ax2.annotate(basin_hash_labels[idx], 
                    (gaps[idx], counts[idx]),
                    fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'converged_basins_gap_distribution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Gap distribution visualization saved to: {output_file}")
    plt.close()
    
    # Also save gap statistics to a text file
    stats_file = os.path.join(output_dir, 'converged_basins_gap_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("Converged Basins Gap Statistics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total basins with gap info: {len(gaps)}\n")
        f.write(f"Mean gap: {mean_gap:.4f}%\n")
        f.write(f"Median gap: {median_gap:.4f}%\n")
        f.write(f"Min gap: {min(gaps):.4f}%\n")
        f.write(f"Max gap: {max(gaps):.4f}%\n")
        f.write(f"Std gap: {np.std(gaps):.4f}%\n\n")
        f.write("Top 10 Basins by Count:\n")
        f.write("-" * 50 + "\n")
        sorted_data = sorted(zip(basin_hash_labels, gaps, counts), 
                           key=lambda x: x[2], reverse=True)
        for i, (label, gap, count) in enumerate(sorted_data[:10], 1):
            f.write(f"{i}. {label}: gap={gap:.4f}%, count={count}\n")
    
    print(f"Gap statistics saved to: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Basin Collapsing Experiment Results')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to basin collapsing results JSON file')
    parser.add_argument('--training_data', type=str, default=None,
                       help='Path to training_data.jsonl file (optional, used as fallback for gap info)')
    parser.add_argument('--hgs_solution_path', type=str, default="/home/jieyi/hgs_cvrp100_uniform.pkl",
                       help='Path to HGS solution .pkl file (recommended for gap calculation)')
    parser.add_argument('--instance_index', type=int, default=0,
                       help='Instance index in HGS solution file (default: 0)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for visualizations (default: same as results_file directory)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        sys.exit(1)
    
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.results_file))
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Visualizing Basin Collapsing Experiment Results")
    print("=" * 60)
    print(f"Results file: {args.results_file}")
    if args.hgs_solution_path:
        print(f"HGS solution: {args.hgs_solution_path} (index={args.instance_index})")
    if args.training_data:
        print(f"Training data: {args.training_data}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Visualize commitment point positions
    print("Step 1: Visualizing commitment point positions...")
    visualize_commitment_points(args.results_file, args.output_dir, args.training_data)
    
    # Visualize gap distribution
    print("\nStep 2: Visualizing gap distribution of converged basins...")
    visualize_gap_distribution(args.results_file, args.training_data, args.output_dir,
                               args.hgs_solution_path, args.instance_index)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
