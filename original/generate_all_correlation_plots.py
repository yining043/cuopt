#!/usr/bin/env python3
"""
Generate overall_correlation plots for all datasets in basin_datasets directory.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import shutil
import pandas as pd


def edge_set_key_for_basin(edges_list):
    """Helper function to create a hashable key from edge list for basin identification."""
    if not edges_list:
        return None
    norm_edges = [tuple(sorted((int(u), int(v)))) for (u, v) in edges_list]
    return frozenset(norm_edges)


def generate_overall_correlation_plot(data_dir):
    """
    Generate overall correlation plot with statistics.
    
    Args:
        data_dir: Path to the dataset directory
    
    Returns:
        str: Path to the saved figure
    """
    # File paths
    optima_jsonl = os.path.join(data_dir, 'optima.jsonl')
    trajectory_jsonl = os.path.join(data_dir, 'trajectory.jsonl')
    
    # Check if required files exist
    if not all(os.path.exists(f) for f in [optima_jsonl, trajectory_jsonl]):
        raise FileNotFoundError(f"Missing required files in {data_dir}")
    
    # Read optima.jsonl to build basin mapping
    optima_raw = []
    with open(optima_jsonl, 'r') as f:
        for line in f:
            optima_raw.append(json.loads(line))
    
    optima_by_basin = {}
    runopt_to_basin = {}
    for r in optima_raw:
        edges = r.get("edges")
        basin_key = edge_set_key_for_basin(edges)
        if basin_key is None:
            continue
        run_id = r.get("run_id")
        opt_id = r.get("optimum_id")
        if run_id is not None and opt_id is not None:
            runopt_to_basin[(run_id, opt_id)] = basin_key
        prev = optima_by_basin.get(basin_key)
        if prev is None:
            optima_by_basin[basin_key] = r
        else:
            c_new = r.get("final_cost")
            c_prev = prev.get("final_cost")
            if c_new is not None and (c_prev is None or c_new < c_prev):
                optima_by_basin[basin_key] = r
    
    # Calculate Convergence Complexity (average trajectory length per basin)
    basin_trial_lengths = defaultdict(list)
    current_trial = None
    current_length = 0
    
    with open(trajectory_jsonl, 'r') as f:
        for line in f:
            data = json.loads(line)
            run_id = data.get("run_id")
            opt_id = data.get("optimum_id")
            if run_id is None or opt_id is None:
                continue
            
            basin_key = runopt_to_basin.get((run_id, opt_id))
            if basin_key is None:
                continue
            
            trial_key = (run_id, opt_id)
            if current_trial != trial_key:
                if current_trial is not None and current_length > 0:
                    basin_trial_lengths[basin_key].append(current_length)
                current_trial = trial_key
                current_length = 1
            else:
                current_length += 1
        
        if current_trial is not None and current_length > 0:
            basin_key = runopt_to_basin.get(current_trial)
            if basin_key is not None:
                basin_trial_lengths[basin_key].append(current_length)
    
    # Calculate average trajectory length (Convergence Complexity)
    basin_to_complexity = {}
    for basin_key, lengths in basin_trial_lengths.items():
        if lengths:
            basin_to_complexity[basin_key] = np.mean(lengths)
    
    # Calculate Attraction Basin Width (number of trials per basin)
    basin_trials_count = defaultdict(set)
    for r in optima_raw:
        edges = r.get("edges")
        basin_key = edge_set_key_for_basin(edges)
        if basin_key is None:
            continue
        run_id = r.get("run_id")
        opt_id = r.get("optimum_id")
        if run_id is not None and opt_id is not None:
            basin_trials_count[basin_key].add((run_id, opt_id))
    
    # Calculate Exploration Volume (number of unique solutions per basin from trajectory)
    basin_solutions = defaultdict(set)
    with open(trajectory_jsonl, 'r') as f:
        for line in f:
            data = json.loads(line)
            run_id = data.get("run_id")
            opt_id = data.get("optimum_id")
            if run_id is None or opt_id is None:
                continue
            
            basin_key = runopt_to_basin.get((run_id, opt_id))
            if basin_key is None:
                continue
            
            edges_sol = data.get("edges")
            if edges_sol:
                sol_key = edge_set_key_for_basin(edges_sol)
                if sol_key is not None:
                    basin_solutions[basin_key].add(sol_key)
    
    # Build final basin data from computed statistics
    basin_data_final = []
    for basin_key in optima_by_basin.keys():
        opt_record = optima_by_basin[basin_key]
        cost = opt_record.get("final_cost")
        if cost is None:
            continue
        
        complexity = basin_to_complexity.get(basin_key)
        if complexity is None:
            continue
        
        attraction_basin_width = len(basin_trials_count.get(basin_key, set()))
        exploration_volume = len(basin_solutions.get(basin_key, set()))
        
        basin_data_final.append({
            'exploration_volume': exploration_volume,
            'attraction_basin_width': attraction_basin_width,
            'convergence_complexity': complexity,
            'cost': cost,
        })
    
    valid_final = [d for d in basin_data_final if d['cost'] is not None and d['convergence_complexity'] is not None]
    
    if len(valid_final) == 0:
        raise ValueError(f"No valid data found in {data_dir}")
    
    exploration_volume = np.array([d['exploration_volume'] for d in valid_final])
    attraction_basin_width = np.array([d['attraction_basin_width'] for d in valid_final])
    convergence_complexity = np.array([d['convergence_complexity'] for d in valid_final])
    costs = np.array([d['cost'] for d in valid_final])
    
    # Calculate correlations
    corr_exploration_cost = np.corrcoef(exploration_volume, costs)[0, 1]
    corr_attraction_cost = np.corrcoef(attraction_basin_width, costs)[0, 1]
    corr_complexity_cost = np.corrcoef(convergence_complexity, costs)[0, 1]
    
    # Calculate statistics
    trials_set = set()
    for r in optima_raw:
        run_id = r.get('run_id')
        opt_id = r.get('optimum_id')
        if run_id is not None and opt_id is not None:
            trials_set.add((run_id, opt_id))
    
    unique_solutions = set()
    trajectory_count = 0
    run_ids_traj = set()
    with open(trajectory_jsonl, 'r') as f:
        for line in f:
            trajectory_count += 1
            data = json.loads(line)
            run_id = data.get("run_id")
            if run_id is not None:
                run_ids_traj.add(run_id)
            edges_sol = data.get("edges")
            if edges_sol:
                sol_key = edge_set_key_for_basin(edges_sol)
                if sol_key is not None:
                    unique_solutions.add(sol_key)
    
    # Calculate basin_trials for statistics display (already computed above as basin_trials_count)
    basin_trials = basin_trials_count
    
    # Create 2x3 subplot figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Add statistics text box at the top
    stats_text = (
        f"Dataset Statistics:\n"
        f"Basin Count: {len(basin_trials):,}\n"
        f"Trial Count: {len(trials_set):,}\n"
        f"Unique Solutions: {len(unique_solutions):,}\n"
        f"Trajectory Records: {trajectory_count:,}\n"
        f"Unique Run IDs: {len(run_ids_traj):,}"
    )
    
    fig.text(0.5, 0.99, stats_text, fontsize=11, 
            verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            transform=fig.transFigure)
    
    # ========== Row 1: Line plots ==========
    
    # 1. Mean Cost by Attraction Basin Width (left)
    ax1 = axes[0, 0]
    trial_counts = sorted(set(attraction_basin_width))
    mean_costs_trial = []
    std_costs_trial = []
    valid_trials = []
    for t in trial_counts:
        group_costs = [d['cost'] for d in valid_final if d['attraction_basin_width'] == t]
        if group_costs:
            valid_trials.append(t)
            mean_costs_trial.append(np.mean(group_costs))
            std_costs_trial.append(np.std(group_costs))
    
    if valid_trials:
        ax1.errorbar(valid_trials, mean_costs_trial, yerr=std_costs_trial, fmt='o-', linewidth=2,
                    markersize=6, capsize=4, color='red')
    ax1.set_xlabel('Attraction Basin Width C(x*)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Cost f(x*)', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Cost by Attraction Basin Width', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, f'r = {corr_attraction_cost:.4f}', transform=ax1.transAxes,
            fontsize=11, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Mean Cost by Convergence Complexity (middle)
    ax2 = axes[0, 1]
    complexity_bins_edges = np.percentile(convergence_complexity, [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100])
    complexity_bins_list = []
    for i in range(len(complexity_bins_edges)-1):
        min_c = complexity_bins_edges[i]
        max_c = complexity_bins_edges[i+1]
        group_costs = [d['cost'] for d in valid_final if min_c <= d['convergence_complexity'] < max_c]
        if group_costs:
            complexity_bins_list.append({
                'center': (min_c + max_c) / 2,
                'mean': np.mean(group_costs),
                'std': np.std(group_costs)
            })
    
    bin_centers_c = [b['center'] for b in complexity_bins_list]
    mean_costs_c = [b['mean'] for b in complexity_bins_list]
    std_costs_c = [b['std'] for b in complexity_bins_list]
    
    if bin_centers_c:
        ax2.errorbar(bin_centers_c, mean_costs_c, yerr=std_costs_c, fmt='o-', linewidth=2,
                    markersize=6, capsize=4, color='green')
    ax2.set_xlabel('Convergence Complexity C(x*)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Cost f(x*)', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Cost by Convergence Complexity', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.98, f'r = {corr_complexity_cost:.4f}', transform=ax2.transAxes,
            fontsize=11, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 3. Mean Cost by Exploration Volume (right)
    ax3 = axes[0, 2]
    volume_bins = [
        (1, 1, "1"),
        (2, 5, "2-5"),
        (6, 10, "6-10"),
        (11, 20, "11-20"),
        (21, 50, "21-50"),
        (51, 100, "51-100"),
        (101, 500, "101-500"),
        (501, float('inf'), "500+"),
    ]
    
    bin_centers = []
    mean_costs_vol = []
    std_costs_vol = []
    for min_v, max_v, label in volume_bins:
        group_costs = [d['cost'] for d in valid_final if min_v <= d['exploration_volume'] <= max_v]
        if group_costs:
            bin_centers.append((min_v + max_v) / 2 if max_v != float('inf') else 750)
            mean_costs_vol.append(np.mean(group_costs))
            std_costs_vol.append(np.std(group_costs))
    
    if bin_centers:
        ax3.errorbar(bin_centers, mean_costs_vol, yerr=std_costs_vol, fmt='o-', linewidth=2, 
                    markersize=6, capsize=4, color='blue')
    ax3.set_xlabel('Exploration Volume', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mean Cost f(x*)', fontsize=12, fontweight='bold')
    ax3.set_title('Mean Cost by Exploration Volume', fontsize=13, fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.98, f'r = {corr_exploration_cost:.4f}', transform=ax3.transAxes,
            fontsize=11, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========== Row 2: Box plots ==========
    
    # 4. Cost Distribution by Attraction Basin Width (left)
    ax4 = axes[1, 0]
    trial_bins = [
        (1, 1, "1"),
        (2, 2, "2"),
        (3, 5, "3-5"),
        (6, 10, "6-10"),
        (11, 20, "11-20"),
        (21, 40, "21-40"),
    ]
    
    box_data_trial = []
    box_labels_trial = []
    for min_t, max_t, label in trial_bins:
        group_costs = [d['cost'] for d in valid_final if min_t <= d['attraction_basin_width'] <= max_t]
        if group_costs:
            box_data_trial.append(group_costs)
            box_labels_trial.append(label)
    
    if box_data_trial:
        bp4 = ax4.boxplot(box_data_trial, tick_labels=box_labels_trial, patch_artist=True)
        for patch in bp4['boxes']:
            patch.set_facecolor('lightcoral')
            patch.set_alpha(0.7)
    ax4.set_xlabel('Attraction Basin Width C(x*)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Cost f(x*)', fontsize=12, fontweight='bold')
    ax4.set_title('Cost Distribution by Attraction Basin Width', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Cost Distribution by Convergence Complexity (middle)
    ax5 = axes[1, 1]
    complexity_bins_for_box = []
    for i in range(len(complexity_bins_edges)-1):
        min_c = complexity_bins_edges[i]
        max_c = complexity_bins_edges[i+1]
        group_costs = [d['cost'] for d in valid_final if min_c <= d['convergence_complexity'] < max_c]
        if group_costs:
            complexity_bins_for_box.append({
                'costs': group_costs,
                'label': f'{int(min_c)}-{int(max_c)}'
            })
    
    box_data_c = [b['costs'] for b in complexity_bins_for_box]
    box_labels_c = [b['label'] for b in complexity_bins_for_box]
    
    if box_data_c:
        bp5 = ax5.boxplot(box_data_c, tick_labels=box_labels_c, patch_artist=True)
        for patch in bp5['boxes']:
            patch.set_facecolor('lightgreen')
            patch.set_alpha(0.7)
    ax5.set_xlabel('Convergence Complexity C(x*)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Cost f(x*)', fontsize=12, fontweight='bold')
    ax5.set_title('Cost Distribution by Convergence Complexity', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Cost Distribution by Exploration Volume (right)
    ax6 = axes[1, 2]
    box_data_vol = []
    box_labels_vol = []
    for min_v, max_v, label in volume_bins:
        group_costs = [d['cost'] for d in valid_final if min_v <= d['exploration_volume'] <= max_v]
        if group_costs:
            box_data_vol.append(group_costs)
            box_labels_vol.append(label)
    
    if box_data_vol:
        bp6 = ax6.boxplot(box_data_vol, tick_labels=box_labels_vol, patch_artist=True)
        for patch in bp6['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
    ax6.set_xlabel('Exploration Volume', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Cost f(x*)', fontsize=12, fontweight='bold')
    ax6.set_title('Cost Distribution by Exploration Volume', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for top text
    output_file = os.path.join(data_dir, 'overall_correlation.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Generate Excel statistics file
    excel_data = {
        'Instance': [os.path.basename(data_dir)],
        'Basin Count': [len(basin_trials)],
        'Trial Count': [len(trials_set)],
        'Unique Solutions': [len(unique_solutions)],
        'Trajectory Records': [trajectory_count],
        'Unique Run IDs': [len(run_ids_traj)],
        'Corr_AttractionBasinWidth_Cost': [corr_attraction_cost],
        'Corr_ConvergenceComplexity_Cost': [corr_complexity_cost],
        'Corr_ExplorationVolume_Cost': [corr_exploration_cost],
    }
    
    df = pd.DataFrame(excel_data)
    excel_file = os.path.join(data_dir, 'dataset_statistics.xlsx')
    df.to_excel(excel_file, index=False)
    
    return output_file, excel_file


def process_all_datasets(base_dir='basin_datasets', output_base='basin_stat'):
    """
    Process all datasets and generate overall_correlation plots for each.
    
    Args:
        base_dir: Base directory containing datasets
        output_base: Output directory for generated plots
    """
    # Create output directory
    os.makedirs(output_base, exist_ok=True)
    
    # Get all dataset directories
    datasets = []
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                # Check if required files exist
                required_files = ['optima.jsonl', 'trajectory.jsonl']
                if all(os.path.exists(os.path.join(item_path, f)) for f in required_files):
                    datasets.append(item_path)
    
    print(f"Found {len(datasets)} datasets")
    
    # Process each dataset
    results = []
    all_excel_data = []  # Collect data for summary Excel
    
    for i, dataset_path in enumerate(sorted(datasets), 1):
        dataset_name = os.path.basename(dataset_path)
        print(f"\n[{i}/{len(datasets)}] Processing dataset: {dataset_name}")
        
        try:
            # Generate plot and Excel
            output_file, excel_file = generate_overall_correlation_plot(dataset_path)
            
            # Copy to basin_stat directory
            plot_name = f"{dataset_name}_overall_correlation.png"
            plot_path = os.path.join(output_base, plot_name)
            shutil.copy2(output_file, plot_path)
            
            excel_name = f"{dataset_name}_dataset_statistics.xlsx"
            excel_path = os.path.join(output_base, excel_name)
            shutil.copy2(excel_file, excel_path)
            
            # Read Excel data for summary
            df_single = pd.read_excel(excel_file)
            all_excel_data.append(df_single)
            
            print(f"  ✓ Plot saved to: {plot_path}")
            print(f"  ✓ Excel saved to: {excel_path}")
            results.append({
                'dataset': dataset_name,
                'status': 'success',
                'plot': plot_path,
                'excel': excel_path
            })
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append({
                'dataset': dataset_name,
                'status': 'error',
                'error': str(e)
            })
    
    # Generate summary Excel file with all instances
    if all_excel_data:
        df_summary = pd.concat(all_excel_data, ignore_index=True)
        summary_excel_path = os.path.join(output_base, 'all_instances_statistics.xlsx')
        df_summary.to_excel(summary_excel_path, index=False)
        print(f"\n  ✓ Summary Excel saved to: {summary_excel_path}")
    
    # Print summary
    print(f"\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count
    print(f"Success: {success_count}")
    print(f"Failed: {error_count}")
    print(f"Output directory: {output_base}")
    
    return results


if __name__ == "__main__":
    results = process_all_datasets()

