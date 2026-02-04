#!/usr/bin/env python3
"""
Analyze basin statistics from Excel files or training_data.jsonl
"""
import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import glob
import re
import gc


def parse_excel_filename(filename):
    match = re.match(r'basin_analysis_run_(\d+)_trial_(\d+)\.xlsx', os.path.basename(filename))
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def load_data_from_excel(excel_file):
    usecols = ['run_id', 'trial_id', 'basin_edges_hash', 'basin_frequency', 'mean', 'basin_gap_to_hgs', 'source_global_iter', 'source_local_iter']
    df = pd.read_excel(excel_file, sheet_name='Basin Details', engine='openpyxl', usecols=usecols)
    if df.empty:
        return []
    
    run_id, trial_id = parse_excel_filename(excel_file)
    if run_id is None and 'run_id' in df.columns:
        run_id = str(df['run_id'].iloc[0])
    if trial_id is None and 'trial_id' in df.columns:
        trial_id = int(df['trial_id'].iloc[0])
    
    data = []
    for _, row in df.iterrows():
        basin_freq = row.get('basin_frequency', 0)
        if pd.isna(basin_freq):
            basin_freq = 0
        else:
            basin_freq = int(basin_freq)
        
        mean_cost = row.get('mean', 0)
        if pd.isna(mean_cost):
            mean_cost = 0
        
        gap_str = str(row.get('basin_gap_to_hgs', ''))
        gap_value = None
        if gap_str and gap_str != 'nan' and '%' in gap_str:
            try:
                gap_value = float(gap_str.replace('%', ''))
            except:
                pass
        
        data.append({
            'run_id': run_id or str(row.get('run_id', '')),
            'trial_id': trial_id if trial_id is not None else int(row.get('trial_id', 0)),
            'basin_edges_hash': str(row.get('basin_edges_hash', '')),
            'basin_frequency': basin_freq,
            'mean_cost': mean_cost,
            'gap_to_hgs': gap_value,
            'source_global_iter': int(row.get('source_global_iter', 0)) if not pd.isna(row.get('source_global_iter', 0)) else 0,
            'source_local_iter': int(row.get('source_local_iter', 0)) if not pd.isna(row.get('source_local_iter', 0)) else 0,
        })
    
    # Clear dataframe to free memory
    del df
    return data


def load_data_from_jsonl(jsonl_file):
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if 'basin_edges_hash' in record and 'basin_frequency' in record:
                    data.append({
                        'run_id': str(record.get('run_id', '')),
                        'trial_id': int(record.get('trial_id', 0)),
                        'basin_edges_hash': str(record.get('basin_edges_hash', '')),
                        'basin_frequency': int(record.get('basin_frequency', 0)),
                    })
    return data


def get_processed_run_ids(json_file):
    if not os.path.exists(json_file):
        return set()
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    processed = set()
    for stats in data.values():
        if 'runs' in stats:
            processed.update(stats['runs'].keys())
    return processed


def get_instance_name(path):
    for part in Path(path).parts:
        if 'pkl#' in part:
            return part
    parent = Path(path).parent.name
    return parent if 'pkl#' in parent else None


def analyze_basin_statistics(data_dir, instance_index, run_id=None, processed_run_ids=None):
    instance_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    basin_info = defaultdict(lambda: {'costs': [], 'gaps': [], 'trials': set(), 'global_iters': set()})
    trial_local_iters = defaultdict(set)  # Track (global_iter, local_iter) pairs for each trial
    global_iter_basins = defaultdict(set)  # Track global_iter -> basins mapping: (instance_name, global_iter) -> set of basin_hashes
    global_iter_basin_freqs = defaultdict(lambda: defaultdict(int))  # Track (instance_name, global_iter) -> {basin_hash: frequency}
    
    filter_str = f'pkl#{instance_index}'
    excel_files = [f for f in glob.glob(os.path.join(data_dir, '**', '*.xlsx'), recursive=True)
                   if 'basin_analysis_run_' in os.path.basename(f) and filter_str in f
                   and '_old' not in f and '_try' not in f
                   and 'pybind_basin_analysis' not in f]
    
    if run_id:
        excel_files = [f for f in excel_files if f'run_{run_id}_' in os.path.basename(f)]
    
    print(f"Found {len(excel_files)} Excel files")
    
    # Process files in batches to avoid memory issues
    BATCH_SIZE = 50
    total_batches = (len(excel_files) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(excel_files))
        batch_files = excel_files[start_idx:end_idx]
        
        if total_batches > 1:
            print(f"Processing batch {batch_idx + 1}/{total_batches} (files {start_idx + 1}-{end_idx})")
        
        for excel_file in batch_files:
            instance_name = get_instance_name(excel_file)
            if not instance_name:
                continue
            
            try:
                data = load_data_from_excel(excel_file)
                if not data:
                    continue
                
                record_run_id = data[0]['run_id']
                if processed_run_ids and record_run_id in processed_run_ids:
                    del data
                    continue
                
                for record in data:
                    if run_id and record['run_id'] != run_id:
                        continue
                    if record['basin_edges_hash'] and record['basin_frequency'] > 0:
                        basin_hash = record['basin_edges_hash']
                        trial_key = (record['run_id'], record['trial_id'])
                        instance_data[instance_name][record['run_id']][record['trial_id']][basin_hash] += record['basin_frequency']
                        
                        # Track local_iter for this trial
                        trial_local_iters[trial_key].add((record.get('source_global_iter', 0), record.get('source_local_iter', 0)))
                        
                        basin_key = (instance_name, basin_hash)
                        basin_info[basin_key]['trials'].add(trial_key)
                        if record['mean_cost'] > 0:
                            basin_info[basin_key]['costs'].append(record['mean_cost'])
                        if record['gap_to_hgs'] is not None:
                            basin_info[basin_key]['gaps'].append(record['gap_to_hgs'])
                        # Track global_iter for this basin
                        source_global_iter = record.get('source_global_iter', 0)
                        if source_global_iter > 0:
                            basin_info[basin_key]['global_iters'].add(source_global_iter)
                            # Track basin for this global_iter (for connectivity analysis)
                            global_iter_key = (instance_name, source_global_iter)
                            global_iter_basins[global_iter_key].add(basin_hash)
                            # Track basin frequency for this global_iter (for connectivity probability calculation)
                            global_iter_basin_freqs[global_iter_key][basin_hash] += record['basin_frequency']
                
                # Clear data after processing
                del data
            except Exception as e:
                print(f"Warning: Error processing {excel_file}: {e}")
                continue
        
        # Force garbage collection after each batch
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
            gc.collect()
    
    jsonl_files = [f for f in glob.glob(os.path.join(data_dir, '**', 'training_data.jsonl'), recursive=True)
                    if filter_str in f]
    
    for jsonl_file in jsonl_files:
        instance_name = get_instance_name(jsonl_file)
        if not instance_name or instance_name in instance_data:
            continue
        
        data = load_data_from_jsonl(jsonl_file)
        for record in data:
            if processed_run_ids and record['run_id'] in processed_run_ids:
                continue
            if run_id and record['run_id'] != run_id:
                continue
            basin_hash = record['basin_edges_hash']
            trial_key = (record['run_id'], record['trial_id'])
            instance_data[instance_name][record['run_id']][record['trial_id']][basin_hash] += record['basin_frequency']
            trial_key = (record['run_id'], record['trial_id'])
            trial_local_iters[trial_key].add((record.get('source_global_iter', 0), record.get('source_local_iter', 0)))
            basin_key = (instance_name, basin_hash)
            basin_info[basin_key]['trials'].add(trial_key)
    
    return instance_data, basin_info, trial_local_iters, global_iter_basins, global_iter_basin_freqs


def compute_statistics(instance_data, basin_info, trial_local_iters, global_iter_basins, global_iter_basin_freqs):
    results = {}
    
    for instance_name, runs in instance_data.items():
        all_basins_run_level = defaultdict(int)
        basin_details = defaultdict(lambda: {'frequency': 0, 'costs': [], 'gaps': [], 'trials': set(), 'global_iters': set()})
        run_stats_dict = {}
        
        for run_id, trials in runs.items():
            all_basins_trial_level = defaultdict(int)
            trial_stats_dict = {}
            
            for trial_id, basins in trials.items():
                frequencies = list(basins.values())
                trial_key = (run_id, trial_id)
                num_local_iters = len(trial_local_iters.get(trial_key, set()))
                
                trial_stats_dict[trial_id] = {
                    'trial_id': trial_id,
                    'num_unique_basins': len(basins),
                    'basin_frequencies': dict(basins),
                    'total_hits': sum(basins.values()),
                    'num_local_iters': num_local_iters,
                    'frequency_distribution': {
                        'min': min(frequencies),
                        'max': max(frequencies),
                        'mean': np.mean(frequencies),
                        'std': np.std(frequencies),
                        'median': np.median(frequencies),
                    },
                    'frequency_histogram': dict(sorted(Counter(frequencies).items())),
                }
                
                for basin_hash, freq in basins.items():
                    all_basins_trial_level[basin_hash] += freq
                    basin_key = (instance_name, basin_hash)
                    if basin_key in basin_info:
                        basin_details[basin_hash]['frequency'] += freq
                        basin_details[basin_hash]['trials'].add((run_id, trial_id))
            
            run_frequencies = list(all_basins_trial_level.values())
            # Count total global_iters for this run
            run_global_iters = set()
            for trial_id in trials.keys():
                trial_key = (run_id, trial_id)
                for g_iter, l_iter in trial_local_iters.get(trial_key, set()):
                    run_global_iters.add(g_iter)
            
            run_stats_dict[run_id] = {
                'run_id': run_id,
                'trials': trial_stats_dict,
                'run_summary': {
                    'num_trials': len(trials),
                    'num_unique_basins_all_trials': len(all_basins_trial_level),
                    'total_hits_all_trials': sum(all_basins_trial_level.values()),
                    'total_global_iters': len(run_global_iters),
                    'basin_frequencies': dict(all_basins_trial_level),
                    'frequency_distribution': {
                        'min': min(run_frequencies),
                        'max': max(run_frequencies),
                        'mean': np.mean(run_frequencies),
                        'std': np.std(run_frequencies),
                        'median': np.median(run_frequencies),
                    },
                    'frequency_histogram': dict(sorted(Counter(run_frequencies).items())),
                }
            }
            
            for basin_hash, freq in all_basins_trial_level.items():
                all_basins_run_level[basin_hash] += freq
        
        for basin_hash in all_basins_run_level:
            basin_key = (instance_name, basin_hash)
            if basin_key in basin_info:
                info = basin_info[basin_key]
                basin_details[basin_hash]['costs'].extend(info['costs'])
                basin_details[basin_hash]['gaps'].extend(info['gaps'])
                basin_details[basin_hash]['trials'].update(info['trials'])
                # Track global_iters for this basin
                if 'global_iters' not in basin_details[basin_hash]:
                    basin_details[basin_hash]['global_iters'] = set()
                basin_details[basin_hash]['global_iters'].update(info.get('global_iters', set()))
        
        # Count total global_iters for this instance (all runs)
        instance_global_iters = set()
        for run_id, run_stats in run_stats_dict.items():
            for trial_id in runs[run_id].keys():
                trial_key = (run_id, trial_id)
                for g_iter, l_iter in trial_local_iters.get(trial_key, set()):
                    instance_global_iters.add(g_iter)
        
        instance_frequencies = list(all_basins_run_level.values())
        basin_frequencies_dict = dict(all_basins_run_level)
        
        # Compute basin connectivity
        # 1. Co-occurrence count: number of global_iters where both basins appear
        basin_connectivity_count = defaultdict(int)  # (basin1, basin2) -> co-occurrence count
        for global_iter_key, basins in global_iter_basins.items():
            if global_iter_key[0] == instance_name:  # Only for current instance
                basins_list = list(basins)
                # Count all pairs of basins that co-occur in the same global_iter
                for i in range(len(basins_list)):
                    for j in range(i + 1, len(basins_list)):
                        basin1, basin2 = basins_list[i], basins_list[j]
                        # Ensure consistent ordering (smaller hash first)
                        if basin1 > basin2:
                            basin1, basin2 = basin2, basin1
                        basin_connectivity_count[(basin1, basin2)] += 1
        
        # 2. Connectivity probability: Cij = Σ P(Bi|sk) · P(Bj|sk)
        # where sk is intermediate solution (global_iter), P(Bi|sk) is probability of basin Bi from sk
        basin_connectivity_prob = defaultdict(float)  # (basin1, basin2) -> connectivity probability
        num_runs_per_global_iter = 100  # Each global_iter runs 100 times
        
        for global_iter_key, basin_freqs in global_iter_basin_freqs.items():
            if global_iter_key[0] == instance_name:  # Only for current instance
                # Calculate probabilities for each basin in this global_iter
                basin_probs = {}
                for basin_hash, freq in basin_freqs.items():
                    basin_probs[basin_hash] = freq / num_runs_per_global_iter
                
                # Calculate connectivity probability for all pairs
                basins_list = list(basin_probs.keys())
                for i in range(len(basins_list)):
                    for j in range(i + 1, len(basins_list)):
                        basin1, basin2 = basins_list[i], basins_list[j]
                        # Ensure consistent ordering (smaller hash first)
                        if basin1 > basin2:
                            basin1, basin2 = basin2, basin1
                        # Cij += P(Bi|sk) · P(Bj|sk)
                        prob_product = basin_probs[basins_list[i]] * basin_probs[basins_list[j]]
                        basin_connectivity_prob[(basin1, basin2)] += prob_product
        
        # 3. Jaccard Similarity: J(Bi, Bj) = |Si ∩ Sj| / |Si ∪ Sj|
        # where Si is the set of intermediate solutions (global_iters) that can flow to Bi
        basin_jaccard_similarity = defaultdict(float)  # (basin1, basin2) -> Jaccard similarity
        
        # Build mapping: basin -> set of global_iters that can flow to it
        basin_to_global_iters = defaultdict(set)
        for global_iter_key, basins in global_iter_basins.items():
            if global_iter_key[0] == instance_name:
                for basin_hash in basins:
                    basin_to_global_iters[basin_hash].add(global_iter_key[1])  # global_iter_key[1] is the global_iter number
        
        # Calculate Jaccard similarity for all basin pairs
        all_basin_hashes = list(basin_to_global_iters.keys())
        for i in range(len(all_basin_hashes)):
            for j in range(i + 1, len(all_basin_hashes)):
                basin1, basin2 = all_basin_hashes[i], all_basin_hashes[j]
                # Ensure consistent ordering (smaller hash first)
                if basin1 > basin2:
                    basin1, basin2 = basin2, basin1
                
                set1 = basin_to_global_iters[basin1]
                set2 = basin_to_global_iters[basin2]
                
                # J(Bi, Bj) = |Si ∩ Sj| / |Si ∪ Sj|
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                jaccard = intersection / union if union > 0 else 0.0
                basin_jaccard_similarity[(basin1, basin2)] = jaccard
        
        # Combine all metrics
        basin_connectivity = {
            'count': dict(basin_connectivity_count),
            'probability': dict(basin_connectivity_prob),
            'jaccard': dict(basin_jaccard_similarity)
        }
        
        basin_info_list = []
        for basin_hash, freq in sorted(basin_frequencies_dict.items(), key=lambda x: x[1], reverse=True):
            details = basin_details[basin_hash]
            costs = list(set(details['costs'])) if details['costs'] else []  # Remove duplicates
            gaps = list(set(details['gaps'])) if details['gaps'] else []  # Remove duplicates
            num_trials = len(details['trials'])
            num_global_iters = len(details.get('global_iters', set()))
            
            # Each basin is one solution, so cost and gap should be the same
            # If there are multiple values (shouldn't happen), take the first one
            basin_info_list.append({
                'basin_edges_hash': basin_hash,
                'frequency': freq,
                'num_trials': num_trials,
                'num_global_iters': num_global_iters,
                'cost': costs[0] if costs else None,
                'gap_to_hgs': gaps[0] if gaps else None,
            })
        
        results[instance_name] = {
            'instance': instance_name,
            'runs': run_stats_dict,
            'instance_summary': {
                'num_runs': len(runs),
                'num_unique_basins_all_runs': len(all_basins_run_level),
                'total_hits_all_runs': sum(all_basins_run_level.values()),
                'total_global_iters': len(instance_global_iters),
                'basin_frequencies': basin_frequencies_dict,
                'basin_details': basin_info_list,
                'basin_connectivity': dict(basin_connectivity),
                'frequency_distribution': {
                    'min': min(instance_frequencies),
                    'max': max(instance_frequencies),
                    'mean': np.mean(instance_frequencies),
                    'std': np.std(instance_frequencies),
                    'median': np.median(instance_frequencies),
                },
                'frequency_histogram': dict(sorted(Counter(instance_frequencies).items())),
            }
        }
    
    return results


def print_statistics(results):
    for instance_name, stats in results.items():
        print("\n" + "=" * 80)
        print(f"Instance: {instance_name}")
        print("=" * 80)
        
        summary = stats['instance_summary']
        print(f"\n[Instance-level Summary]")
        print(f"  Runs: {summary['num_runs']}, Unique basins: {summary['num_unique_basins_all_runs']}, Total hits: {summary['total_hits_all_runs']}")
        
        freq_dist = summary['frequency_distribution']
        print(f"  Frequency: min={freq_dist['min']}, max={freq_dist['max']}, mean={freq_dist['mean']:.2f}, median={freq_dist['median']:.2f}")


def save_statistics_to_excel(results, output_file):
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        instance_summary_data = []
        for instance_name, stats in results.items():
            summary = stats['instance_summary']
            freq_dist = summary['frequency_distribution']
            instance_summary_data.append({
                'instance': instance_name,
                'num_runs': summary['num_runs'],
                'num_unique_basins': summary['num_unique_basins_all_runs'],
                'total_hits': summary['total_hits_all_runs'],
                'total_global_iters': summary.get('total_global_iters', 0),
                'freq_min': freq_dist['min'],
                'freq_max': freq_dist['max'],
                'freq_mean': freq_dist['mean'],
                'freq_std': freq_dist['std'],
                'freq_median': freq_dist['median'],
            })
        pd.DataFrame(instance_summary_data).to_excel(writer, sheet_name='Instance Summary', index=False)
        
        run_summary_data = []
        for instance_name, stats in results.items():
            for run_id, run_stats in stats['runs'].items():
                run_summary = run_stats['run_summary']
                freq_dist = run_summary['frequency_distribution']
                run_summary_data.append({
                    'instance': instance_name,
                    'run_id': run_id,
                    'num_trials': run_summary['num_trials'],
                    'num_unique_basins': run_summary['num_unique_basins_all_trials'],
                    'total_hits': run_summary['total_hits_all_trials'],
                    'total_global_iters': run_summary.get('total_global_iters', 0),
                    'freq_min': freq_dist['min'],
                    'freq_max': freq_dist['max'],
                    'freq_mean': freq_dist['mean'],
                    'freq_std': freq_dist['std'],
                    'freq_median': freq_dist['median'],
                })
        pd.DataFrame(run_summary_data).to_excel(writer, sheet_name='Run Summary', index=False)
        
        trial_summary_data = []
        for instance_name, stats in results.items():
            for run_id, run_stats in stats['runs'].items():
                for trial_id, trial_stats in run_stats['trials'].items():
                    freq_dist = trial_stats['frequency_distribution']
                    trial_summary_data.append({
                        'instance': instance_name,
                        'run_id': run_id,
                        'trial_id': trial_id,
                        'num_unique_basins': trial_stats['num_unique_basins'],
                        'total_hits': trial_stats['total_hits'],
                        'num_local_iters': trial_stats.get('num_local_iters', 0),
                        'freq_min': freq_dist['min'],
                        'freq_max': freq_dist['max'],
                        'freq_mean': freq_dist['mean'],
                        'freq_std': freq_dist['std'],
                        'freq_median': freq_dist['median'],
                    })
        pd.DataFrame(trial_summary_data).to_excel(writer, sheet_name='Trial Summary', index=False)
        
        basin_details_data = []
        for instance_name, stats in results.items():
            summary = stats['instance_summary']
            total_hits = summary['total_hits_all_runs']
            for basin_info in summary.get('basin_details', []):
                basin_hash = basin_info['basin_edges_hash']
                frequency = basin_info['frequency']
                basin_details_data.append({
                    'instance': instance_name,
                    'basin_edges_hash': basin_hash,
                    'frequency': frequency,
                    'num_global_iters': basin_info.get('num_global_iters', 0),
                    'frequency_percent': f"{frequency / total_hits * 100:.4f}%" if total_hits > 0 else "0.0000%",
                    'num_trials': basin_info['num_trials'],
                    'cost': basin_info['cost'],
                    'gap_to_hgs': basin_info['gap_to_hgs'],
                })
        pd.DataFrame(basin_details_data).to_excel(writer, sheet_name='Basin Frequencies (Instance)', index=False)
        
        # Basin connectivity data
        basin_connectivity_data = []
        for instance_name, stats in results.items():
            summary = stats['instance_summary']
            connectivity = summary.get('basin_connectivity', {})
            basin_frequencies = summary.get('basin_frequencies', {})
            
            # Create a mapping from basin_hash to its details for easy lookup
            basin_details_map = {b['basin_edges_hash']: b for b in summary.get('basin_details', [])}
            
            # Get connectivity data (now has 'count', 'probability', and 'jaccard' keys)
            connectivity_count = connectivity.get('count', {}) if isinstance(connectivity, dict) and 'count' in connectivity else connectivity
            connectivity_prob = connectivity.get('probability', {}) if isinstance(connectivity, dict) and 'probability' in connectivity else {}
            connectivity_jaccard = connectivity.get('jaccard', {}) if isinstance(connectivity, dict) and 'jaccard' in connectivity else {}
            
            # Sort by connectivity probability (descending)
            all_pairs = set(connectivity_count.keys()) | set(connectivity_prob.keys()) | set(connectivity_jaccard.keys())
            sorted_pairs = sorted(all_pairs, key=lambda x: connectivity_prob.get(x, 0), reverse=True)
            
            # Limit to top 10000 pairs to avoid Excel size limit (1048576 rows max)
            # Also filter out pairs with very low connectivity (both prob and jaccard < 0.01)
            filtered_pairs = []
            for pair in sorted_pairs:
                prob = connectivity_prob.get(pair, 0.0)
                jaccard = connectivity_jaccard.get(pair, 0.0)
                if prob >= 0.01 or jaccard >= 0.01:
                    filtered_pairs.append(pair)
                if len(filtered_pairs) >= 10000:
                    break
            
            for (basin1, basin2) in filtered_pairs:
                basin1_freq = basin_frequencies.get(basin1, 0)
                basin2_freq = basin_frequencies.get(basin2, 0)
                
                # Get basin details
                basin1_details = basin_details_map.get(basin1, {})
                basin2_details = basin_details_map.get(basin2, {})
                
                co_occurrence_count = connectivity_count.get((basin1, basin2), 0)
                co_occurrence_probability = connectivity_prob.get((basin1, basin2), 0.0)
                jaccard_similarity = connectivity_jaccard.get((basin1, basin2), 0.0)
                
                basin_connectivity_data.append({
                    'instance': instance_name,
                    'basin1_hash': basin1,
                    'basin1_frequency': basin1_freq,
                    'basin1_cost': basin1_details.get('cost'),
                    'basin1_gap_to_hgs': basin1_details.get('gap_to_hgs'),
                    'basin2_hash': basin2,
                    'basin2_frequency': basin2_freq,
                    'basin2_cost': basin2_details.get('cost'),
                    'basin2_gap_to_hgs': basin2_details.get('gap_to_hgs'),
                    'co_occurrence_count': co_occurrence_count,
                    'co_occurrence_probability': f"{co_occurrence_probability * 100:.4f}%",
                    'jaccard_similarity': f"{jaccard_similarity:.4f}",
                })
        
        if basin_connectivity_data:
            pd.DataFrame(basin_connectivity_data).to_excel(writer, sheet_name='Basin Connectivity', index=False)
    
    print(f"Statistics saved to: {output_file}")


def visualize_basin_connectivity(results, output_file=None):
    """
    Visualize basin connectivity as a network graph.
    
    Args:
        results: Statistics results dictionary
        output_file: Optional output file path for the visualization image
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("Warning: networkx or matplotlib not available. Skipping visualization.")
        return
    
    for instance_name, stats in results.items():
        summary = stats['instance_summary']
        connectivity = summary.get('basin_connectivity', {})
        basin_frequencies = summary.get('basin_frequencies', {})
        basin_details_map = {b['basin_edges_hash']: b for b in summary.get('basin_details', [])}
        
        if not connectivity:
            print(f"No connectivity data for {instance_name}")
            continue
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        all_basins = set()
        # Handle both old format (dict) and new format (dict with 'count' and 'probability' keys)
        if isinstance(connectivity, dict) and 'count' in connectivity:
            connectivity_count_dict = connectivity.get('count', {})
            connectivity_keys = connectivity_count_dict.keys()
        else:
            # Old format: connectivity is directly the count dict
            connectivity_count_dict = connectivity if connectivity else {}
            connectivity_keys = connectivity_count_dict.keys()
        
        for (basin1, basin2) in connectivity_keys:
            all_basins.add(basin1)
            all_basins.add(basin2)
        
        # Filter basins: exclude those that appear only once, except the one with lowest gap
        # First, find the basin with the lowest gap
        min_gap_basin = None
        min_gap_value = float('inf')
        for basin_hash in all_basins:
            details = basin_details_map.get(basin_hash, {})
            gap = details.get('gap_to_hgs')
            if gap is not None and gap < min_gap_value:
                min_gap_value = gap
                min_gap_basin = basin_hash
        
        # Filter basins: keep only those with frequency > 1, or the min_gap_basin
        filtered_basins = []
        for basin_hash in all_basins:
            freq = basin_frequencies.get(basin_hash, 0)
            if freq > 1 or basin_hash == min_gap_basin:
                filtered_basins.append(basin_hash)
        
        # Sort by frequency (no limit – use all filtered basins)
        top_basins = sorted(filtered_basins, key=lambda x: basin_frequencies.get(x, 0), reverse=True)
        
        basin_id_map = {basin: i for i, basin in enumerate(top_basins)}
        
        # Add nodes with attributes
        node_colors = []
        node_color_categories = []
        node_sizes = []
        node_labels = {}
        
        # Define gap color categories
        gap_categories = [
            (0, 1, '0-1%'),
            (1, 2, '1-2%'),
            (2, 3, '2-3%'),
            (3, 5, '3-5%'),
            (5, 10, '5-10%'),
            (10, float('inf'), '>10%'),
        ]
        
        for basin_hash in top_basins:
            basin_id = basin_id_map[basin_hash]
            freq = basin_frequencies.get(basin_hash, 0)
            details = basin_details_map.get(basin_hash, {})
            cost = details.get('cost')
            gap = details.get('gap_to_hgs')
            
            G.add_node(basin_id, 
                      hash=basin_hash,
                      frequency=freq,
                      cost=cost,
                      gap=gap)
            
            # Node size based on frequency
            node_sizes.append(max(100, freq * 2))
            
            # Node color based on gap_to_hgs category
            if gap is not None:
                # Find which category this gap belongs to
                category_idx = len(gap_categories) - 1  # Default to last category
                for idx, (min_gap, max_gap, label) in enumerate(gap_categories):
                    if min_gap <= gap < max_gap:
                        category_idx = idx
                        break
                node_color_categories.append(category_idx)
                node_colors.append(gap)  # Keep original value for colorbar
            elif cost is not None:
                # If no gap, use a default category
                node_color_categories.append(-1)
                node_colors.append(cost)
            else:
                node_color_categories.append(-1)
                node_colors.append(0)
            
            # Short label
            node_labels[basin_id] = f"B{basin_id}\n{basin_hash[:6]}"
        
        # Add edges
        edge_weights = []
        max_co_occurrence = max(connectivity_count_dict.values()) if connectivity_count_dict else 1
        
        for (basin1, basin2), co_count in connectivity_count_dict.items():
            if basin1 in basin_id_map and basin2 in basin_id_map:
                id1 = basin_id_map[basin1]
                id2 = basin_id_map[basin2]
                weight = co_count / max_co_occurrence if max_co_occurrence > 0 else 0
                G.add_edge(id1, id2, weight=weight, count=co_count)
                edge_weights.append(weight)
        
        if len(G.nodes()) == 0:
            print(f"No nodes to visualize for {instance_name}")
            continue
        
        # Create visualization - Figure 1: Network graph and histogram
        fig1 = plt.figure(figsize=(20, 10))
        gs1 = fig1.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig1.add_subplot(gs1[0, :2])  # Network graph (spans 2 columns)
        ax2 = fig1.add_subplot(gs1[0, 2])    # Histogram
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Plot 1: Network graph with node colors based on gap categories
        has_gap_data = any(basin_details_map.get(b, {}).get('gap_to_hgs') is not None for b in top_basins)
        
        if has_gap_data:
            # Use discrete colors for gap categories
            from matplotlib.colors import ListedColormap
            category_colors = ['#c0392b', '#e74c3c', '#e67e22', '#9b59b6', '#3498db', '#2ecc71']  # Dark Red, Red, Orange, Purple, Blue, Green
            # Add gray for nodes without gap data
            category_colors_with_unknown = category_colors + ['#95a5a6']
            
            # Map category indices to colors
            node_color_values = [category_colors_with_unknown[cat_idx] if cat_idx >= 0 else category_colors_with_unknown[-1] 
                                for cat_idx in node_color_categories]
            
            cmap = ListedColormap(category_colors_with_unknown)
        else:
            # Fallback to continuous color scale for cost
            if node_colors and max(node_colors) > min(node_colors):
                vmin, vmax = min(node_colors), max(node_colors)
                cmap = cm.viridis
                node_color_values = node_colors
            else:
                vmin, vmax = 0, 1
                cmap = cm.Blues
                node_color_values = node_colors
        
        # Draw edges
        edges = G.edges()
        edge_widths = [G[u][v].get('weight', 0) * 3 for u, v in edges]
        nx.draw_networkx_edges(G, pos, ax=ax1, width=edge_widths, 
                              alpha=0.3, edge_color='gray')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax1, 
                              node_size=node_sizes,
                              node_color=node_color_values,
                              alpha=0.8)
        
        # Draw labels (only for smaller graphs)
        if len(G.nodes()) <= 30:
            nx.draw_networkx_labels(G, pos, ax=ax1, labels=node_labels, font_size=8)
        
        ax1.set_title(f'Basin Connectivity Network\n{instance_name}', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Add legend for gap categories
        if has_gap_data:
            import matplotlib.patches as mpatches
            legend_patches = []
            for min_gap, max_gap, label in gap_categories:
                if max_gap == float('inf'):
                    label_str = f'{label}'
                else:
                    label_str = f'{label}'
                color_idx = gap_categories.index((min_gap, max_gap, label))
                legend_patches.append(mpatches.Patch(color=category_colors[color_idx], label=label_str))
            # Add unknown category
            legend_patches.append(mpatches.Patch(color=category_colors_with_unknown[-1], label='No gap data'))
            ax1.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        else:
            # Add colorbar for cost
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax1)
            cbar.set_label('Cost', rotation=270, labelpad=15)
        
        # Plot 2: Connectivity strength histogram
        co_occurrence_counts = list(connectivity_count_dict.values())
        if co_occurrence_counts:
            ax2.hist(co_occurrence_counts, bins=min(20, len(set(co_occurrence_counts))), 
                    edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Co-occurrence Count', fontsize=12)
            ax2.set_ylabel('Number of Basin Pairs', fontsize=12)
            ax2.set_title('Distribution of Basin Co-occurrence', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # Save Figure 1: Network graph and histogram
        plt.tight_layout()
        if output_file:
            fig1_output = output_file.replace('.png', '_network.png')
        else:
            instance_safe = instance_name.replace('/', '_').replace('#', '_')
            fig1_output = f'basin_connectivity_{instance_safe}_network.png'
        plt.savefig(fig1_output, dpi=300, bbox_inches='tight')
        print(f"Network visualization saved to: {fig1_output}")
        plt.close(fig1)
        
        # Create visualization - Figure 2: All 6 heatmaps
        fig2 = plt.figure(figsize=(24, 16))
        gs2 = fig2.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        ax3 = fig2.add_subplot(gs2[0, 0])    # Jaccard Similarity heatmap - all basins
        ax4 = fig2.add_subplot(gs2[0, 1])    # Connectivity Probability heatmap - all basins
        ax5 = fig2.add_subplot(gs2[0, 2])    # Jaccard Similarity heatmap - freq>5
        ax6 = fig2.add_subplot(gs2[1, 0])    # Connectivity Probability heatmap - freq>5
        ax7 = fig2.add_subplot(gs2[1, 1])    # Jaccard Similarity heatmap - top 500
        ax8 = fig2.add_subplot(gs2[1, 2])    # Connectivity Probability heatmap - top 500
        
        # Plot heatmaps for Jaccard Similarity and Connectivity Probability
        # Get connectivity data
        connectivity_prob_dict = connectivity.get('probability', {}) if isinstance(connectivity, dict) and 'probability' in connectivity else {}
        connectivity_jaccard_dict = connectivity.get('jaccard', {}) if isinstance(connectivity, dict) and 'jaccard' in connectivity else {}
        
        # Find min_gap_basin for filtered heatmap
        min_gap_basin = None
        min_gap_value = float('inf')
        for basin_hash in top_basins:
            details = basin_details_map.get(basin_hash, {})
            gap = details.get('gap_to_hgs')
            if gap is not None and gap < min_gap_value:
                min_gap_value = gap
                min_gap_basin = basin_hash
        
        # Create matrices for heatmaps (all basins)
        sorted_basin_list = sorted(top_basins)
        n_basins = len(sorted_basin_list)
        
        if n_basins > 0:
            # Initialize matrices
            jaccard_matrix = np.zeros((n_basins, n_basins))
            prob_matrix = np.zeros((n_basins, n_basins))
            
            # Fill matrices
            for i, basin1 in enumerate(sorted_basin_list):
                for j, basin2 in enumerate(sorted_basin_list):
                    if i == j:
                        # Diagonal: same basin, similarity = 1.0
                        jaccard_matrix[i, j] = 1.0
                        prob_matrix[i, j] = 1.0
                    else:
                        # Ensure consistent ordering
                        if basin1 > basin2:
                            key = (basin2, basin1)
                        else:
                            key = (basin1, basin2)
                        
                        jaccard_matrix[i, j] = connectivity_jaccard_dict.get(key, 0.0)
                        prob_matrix[i, j] = connectivity_prob_dict.get(key, 0.0)
            
            # Plot Jaccard Similarity heatmap
            im1 = ax3.imshow(jaccard_matrix, cmap='YlOrRd', aspect='equal', vmin=0, vmax=1)
            ax3.set_title('Jaccard Similarity', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Basin Index', fontsize=10)
            ax3.set_ylabel('Basin Index', fontsize=10)
            # Set ticks to show fewer labels for readability
            step = max(1, n_basins // 10)
            ax3.set_xticks(range(0, n_basins, step))
            ax3.set_yticks(range(0, n_basins, step))
            ax3.set_xticklabels([sorted_basin_list[i][:6] for i in range(0, n_basins, step)], rotation=45, ha='right', fontsize=6)
            ax3.set_yticklabels([sorted_basin_list[i][:6] for i in range(0, n_basins, step)], fontsize=6)
            plt.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04)
            
            # Plot Connectivity Probability heatmap
            max_prob = np.max(prob_matrix) if np.max(prob_matrix) > 0 else 1.0
            im2 = ax4.imshow(prob_matrix, cmap='YlOrRd', aspect='equal', vmin=0, vmax=max_prob)
            ax4.set_title('Connectivity Probability', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Basin Index', fontsize=10)
            ax4.set_ylabel('Basin Index', fontsize=10)
            ax4.set_xticks(range(0, n_basins, step))
            ax4.set_yticks(range(0, n_basins, step))
            ax4.set_xticklabels([sorted_basin_list[i][:6] for i in range(0, n_basins, step)], rotation=45, ha='right', fontsize=6)
            ax4.set_yticklabels([sorted_basin_list[i][:6] for i in range(0, n_basins, step)], fontsize=6)
            plt.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04)
            
            # Plot 5 & 6: Heatmaps for basins with frequency > 5 (plus min_gap_basin)
            # Filter basins: frequency > 5 or is min_gap_basin
            filtered_basin_list = []
            for basin_hash in sorted_basin_list:
                freq = basin_frequencies.get(basin_hash, 0)
                if freq > 5 or basin_hash == min_gap_basin:
                    filtered_basin_list.append(basin_hash)
            
            n_filtered = len(filtered_basin_list)
            
            if n_filtered > 0:
                # Initialize matrices for filtered basins
                jaccard_matrix_filtered = np.zeros((n_filtered, n_filtered))
                prob_matrix_filtered = np.zeros((n_filtered, n_filtered))
                
                # Fill matrices
                for i, basin1 in enumerate(filtered_basin_list):
                    for j, basin2 in enumerate(filtered_basin_list):
                        if i == j:
                            # Diagonal: same basin, similarity = 1.0
                            jaccard_matrix_filtered[i, j] = 1.0
                            prob_matrix_filtered[i, j] = 1.0
                        else:
                            # Ensure consistent ordering
                            if basin1 > basin2:
                                key = (basin2, basin1)
                            else:
                                key = (basin1, basin2)
                            
                            jaccard_matrix_filtered[i, j] = connectivity_jaccard_dict.get(key, 0.0)
                            prob_matrix_filtered[i, j] = connectivity_prob_dict.get(key, 0.0)
                
                # Plot Jaccard Similarity heatmap (filtered)
                im3 = ax5.imshow(jaccard_matrix_filtered, cmap='YlOrRd', aspect='equal', vmin=0, vmax=1)
                ax5.set_title('Jaccard Similarity (Freq>5)', fontsize=14, fontweight='bold')
                ax5.set_xlabel('Basin Index', fontsize=10)
                ax5.set_ylabel('Basin Index', fontsize=10)
                step_filtered = max(1, n_filtered // 10)
                ax5.set_xticks(range(0, n_filtered, step_filtered))
                ax5.set_yticks(range(0, n_filtered, step_filtered))
                ax5.set_xticklabels([filtered_basin_list[i][:6] for i in range(0, n_filtered, step_filtered)], rotation=45, ha='right', fontsize=6)
                ax5.set_yticklabels([filtered_basin_list[i][:6] for i in range(0, n_filtered, step_filtered)], fontsize=6)
                plt.colorbar(im3, ax=ax5, fraction=0.046, pad=0.04)
                
                # Plot Connectivity Probability heatmap (filtered)
                max_prob_filtered = np.max(prob_matrix_filtered) if np.max(prob_matrix_filtered) > 0 else 1.0
                im4 = ax6.imshow(prob_matrix_filtered, cmap='YlOrRd', aspect='equal', vmin=0, vmax=max_prob_filtered)
                ax6.set_title('Connectivity Probability (Freq>5)', fontsize=14, fontweight='bold')
                ax6.set_xlabel('Basin Index', fontsize=10)
                ax6.set_ylabel('Basin Index', fontsize=10)
                ax6.set_xticks(range(0, n_filtered, step_filtered))
                ax6.set_yticks(range(0, n_filtered, step_filtered))
                ax6.set_xticklabels([filtered_basin_list[i][:6] for i in range(0, n_filtered, step_filtered)], rotation=45, ha='right', fontsize=6)
                ax6.set_yticklabels([filtered_basin_list[i][:6] for i in range(0, n_filtered, step_filtered)], fontsize=6)
                plt.colorbar(im4, ax=ax6, fraction=0.046, pad=0.04)
            else:
                ax5.axis('off')
                ax6.axis('off')
            
            # Plot 7 & 8: Heatmaps for top 500 basins by frequency (plus min_gap_basin)
            # Sort basins by frequency and take top 500
            sorted_by_freq = sorted(sorted_basin_list, key=lambda x: basin_frequencies.get(x, 0), reverse=True)
            top_500_basins = sorted_by_freq[:500]
            
            # Ensure min_gap_basin is included
            if min_gap_basin and min_gap_basin not in top_500_basins:
                top_500_basins.append(min_gap_basin)
            
            # Remove duplicates and sort
            top_500_basins = sorted(list(set(top_500_basins)))
            n_top500 = len(top_500_basins)
            
            if n_top500 > 0:
                # Initialize matrices for top 500 basins
                jaccard_matrix_top500 = np.zeros((n_top500, n_top500))
                prob_matrix_top500 = np.zeros((n_top500, n_top500))
                
                # Fill matrices
                for i, basin1 in enumerate(top_500_basins):
                    for j, basin2 in enumerate(top_500_basins):
                        if i == j:
                            # Diagonal: same basin, similarity = 1.0
                            jaccard_matrix_top500[i, j] = 1.0
                            prob_matrix_top500[i, j] = 1.0
                        else:
                            # Ensure consistent ordering
                            if basin1 > basin2:
                                key = (basin2, basin1)
                            else:
                                key = (basin1, basin2)
                            
                            jaccard_matrix_top500[i, j] = connectivity_jaccard_dict.get(key, 0.0)
                            prob_matrix_top500[i, j] = connectivity_prob_dict.get(key, 0.0)
                
                # Plot Jaccard Similarity heatmap (top 500)
                im5 = ax7.imshow(jaccard_matrix_top500, cmap='YlOrRd', aspect='equal', vmin=0, vmax=1)
                ax7.set_title('Jaccard Similarity (Top 500)', fontsize=14, fontweight='bold')
                ax7.set_xlabel('Basin Index', fontsize=10)
                ax7.set_ylabel('Basin Index', fontsize=10)
                step_top500 = max(1, n_top500 // 10)
                ax7.set_xticks(range(0, n_top500, step_top500))
                ax7.set_yticks(range(0, n_top500, step_top500))
                ax7.set_xticklabels([top_500_basins[i][:6] for i in range(0, n_top500, step_top500)], rotation=45, ha='right', fontsize=6)
                ax7.set_yticklabels([top_500_basins[i][:6] for i in range(0, n_top500, step_top500)], fontsize=6)
                plt.colorbar(im5, ax=ax7, fraction=0.046, pad=0.04)
                
                # Plot Connectivity Probability heatmap (top 500)
                max_prob_top500 = np.max(prob_matrix_top500) if np.max(prob_matrix_top500) > 0 else 1.0
                im6 = ax8.imshow(prob_matrix_top500, cmap='YlOrRd', aspect='equal', vmin=0, vmax=max_prob_top500)
                ax8.set_title('Connectivity Probability (Top 500)', fontsize=14, fontweight='bold')
                ax8.set_xlabel('Basin Index', fontsize=10)
                ax8.set_ylabel('Basin Index', fontsize=10)
                ax8.set_xticks(range(0, n_top500, step_top500))
                ax8.set_yticks(range(0, n_top500, step_top500))
                ax8.set_xticklabels([top_500_basins[i][:6] for i in range(0, n_top500, step_top500)], rotation=45, ha='right', fontsize=6)
                ax8.set_yticklabels([top_500_basins[i][:6] for i in range(0, n_top500, step_top500)], fontsize=6)
                plt.colorbar(im6, ax=ax8, fraction=0.046, pad=0.04)
            else:
                ax7.axis('off')
                ax8.axis('off')
            
        # Save Figure 2: All 6 heatmaps
        plt.tight_layout()
        if output_file:
            fig2_output = output_file.replace('.png', '_heatmaps.png')
        else:
            instance_safe = instance_name.replace('/', '_').replace('#', '_')
            fig2_output = f'basin_connectivity_{instance_safe}_heatmaps.png'
        plt.savefig(fig2_output, dpi=300, bbox_inches='tight')
        print(f"Heatmaps visualization saved to: {fig2_output}")
        plt.close(fig2)
        
        # Save individual heatmap figures
        base_output_file = output_file.replace('_connectivity.png', '') if output_file else None
        if not base_output_file:
            instance_safe = instance_name.replace('/', '_').replace('#', '_')
            base_output_file = f'basin_connectivity_{instance_safe}'
        
        # Save each heatmap separately
        # 1. All basins heatmaps
        if n_basins > 0:
            # Jaccard Similarity heatmap (all)
            fig_jaccard, ax_jaccard = plt.subplots(figsize=(10, 10))
            im_jaccard = ax_jaccard.imshow(jaccard_matrix, cmap='YlOrRd', aspect='equal', vmin=0, vmax=1)
            ax_jaccard.set_title('Jaccard Similarity (All)', fontsize=16, fontweight='bold')
            ax_jaccard.set_xlabel('Basin Index', fontsize=12)
            ax_jaccard.set_ylabel('Basin Index', fontsize=12)
            ax_jaccard.set_xticks(range(0, n_basins, step))
            ax_jaccard.set_yticks(range(0, n_basins, step))
            ax_jaccard.set_xticklabels([sorted_basin_list[i][:6] for i in range(0, n_basins, step)], rotation=45, ha='right', fontsize=8)
            ax_jaccard.set_yticklabels([sorted_basin_list[i][:6] for i in range(0, n_basins, step)], fontsize=8)
            plt.colorbar(im_jaccard, ax=ax_jaccard, fraction=0.046, pad=0.04)
            plt.tight_layout()
            jaccard_file = f'{base_output_file}_jaccard_all.png'
            plt.savefig(jaccard_file, dpi=300, bbox_inches='tight')
            print(f"Jaccard Similarity heatmap (all) saved to: {jaccard_file}")
            plt.close()
            
            # Connectivity Probability heatmap (all)
            fig_prob, ax_prob = plt.subplots(figsize=(10, 10))
            max_prob_val = np.max(prob_matrix) if np.max(prob_matrix) > 0 else 1.0
            im_prob = ax_prob.imshow(prob_matrix, cmap='YlOrRd', aspect='equal', vmin=0, vmax=max_prob_val)
            ax_prob.set_title('Connectivity Probability (All)', fontsize=16, fontweight='bold')
            ax_prob.set_xlabel('Basin Index', fontsize=12)
            ax_prob.set_ylabel('Basin Index', fontsize=12)
            ax_prob.set_xticks(range(0, n_basins, step))
            ax_prob.set_yticks(range(0, n_basins, step))
            ax_prob.set_xticklabels([sorted_basin_list[i][:6] for i in range(0, n_basins, step)], rotation=45, ha='right', fontsize=8)
            ax_prob.set_yticklabels([sorted_basin_list[i][:6] for i in range(0, n_basins, step)], fontsize=8)
            plt.colorbar(im_prob, ax=ax_prob, fraction=0.046, pad=0.04)
            plt.tight_layout()
            prob_file = f'{base_output_file}_prob_all.png'
            plt.savefig(prob_file, dpi=300, bbox_inches='tight')
            print(f"Connectivity Probability heatmap (all) saved to: {prob_file}")
            plt.close()
        
        # 2. Frequency > 5 heatmaps
        if n_filtered > 0:
            # Jaccard Similarity heatmap (freq>5)
            fig_jaccard, ax_jaccard = plt.subplots(figsize=(10, 10))
            im_jaccard = ax_jaccard.imshow(jaccard_matrix_filtered, cmap='YlOrRd', aspect='equal', vmin=0, vmax=1)
            ax_jaccard.set_title('Jaccard Similarity (Freq>5)', fontsize=16, fontweight='bold')
            ax_jaccard.set_xlabel('Basin Index', fontsize=12)
            ax_jaccard.set_ylabel('Basin Index', fontsize=12)
            ax_jaccard.set_xticks(range(0, n_filtered, step_filtered))
            ax_jaccard.set_yticks(range(0, n_filtered, step_filtered))
            ax_jaccard.set_xticklabels([filtered_basin_list[i][:6] for i in range(0, n_filtered, step_filtered)], rotation=45, ha='right', fontsize=8)
            ax_jaccard.set_yticklabels([filtered_basin_list[i][:6] for i in range(0, n_filtered, step_filtered)], fontsize=8)
            plt.colorbar(im_jaccard, ax=ax_jaccard, fraction=0.046, pad=0.04)
            plt.tight_layout()
            jaccard_file = f'{base_output_file}_jaccard_freq_gt_5.png'
            plt.savefig(jaccard_file, dpi=300, bbox_inches='tight')
            print(f"Jaccard Similarity heatmap (freq>5) saved to: {jaccard_file}")
            plt.close()
            
            # Connectivity Probability heatmap (freq>5)
            fig_prob, ax_prob = plt.subplots(figsize=(10, 10))
            max_prob_filtered_val = np.max(prob_matrix_filtered) if np.max(prob_matrix_filtered) > 0 else 1.0
            im_prob = ax_prob.imshow(prob_matrix_filtered, cmap='YlOrRd', aspect='equal', vmin=0, vmax=max_prob_filtered_val)
            ax_prob.set_title('Connectivity Probability (Freq>5)', fontsize=16, fontweight='bold')
            ax_prob.set_xlabel('Basin Index', fontsize=12)
            ax_prob.set_ylabel('Basin Index', fontsize=12)
            ax_prob.set_xticks(range(0, n_filtered, step_filtered))
            ax_prob.set_yticks(range(0, n_filtered, step_filtered))
            ax_prob.set_xticklabels([filtered_basin_list[i][:6] for i in range(0, n_filtered, step_filtered)], rotation=45, ha='right', fontsize=8)
            ax_prob.set_yticklabels([filtered_basin_list[i][:6] for i in range(0, n_filtered, step_filtered)], fontsize=8)
            plt.colorbar(im_prob, ax=ax_prob, fraction=0.046, pad=0.04)
            plt.tight_layout()
            prob_file = f'{base_output_file}_prob_freq_gt_5.png'
            plt.savefig(prob_file, dpi=300, bbox_inches='tight')
            print(f"Connectivity Probability heatmap (freq>5) saved to: {prob_file}")
            plt.close()
        
        # 3. Top 500 heatmaps
        if n_top500 > 0:
            # Jaccard Similarity heatmap (top 500)
            fig_jaccard, ax_jaccard = plt.subplots(figsize=(10, 10))
            im_jaccard = ax_jaccard.imshow(jaccard_matrix_top500, cmap='YlOrRd', aspect='equal', vmin=0, vmax=1)
            ax_jaccard.set_title('Jaccard Similarity (Top 500)', fontsize=16, fontweight='bold')
            ax_jaccard.set_xlabel('Basin Index', fontsize=12)
            ax_jaccard.set_ylabel('Basin Index', fontsize=12)
            ax_jaccard.set_xticks(range(0, n_top500, step_top500))
            ax_jaccard.set_yticks(range(0, n_top500, step_top500))
            ax_jaccard.set_xticklabels([top_500_basins[i][:6] for i in range(0, n_top500, step_top500)], rotation=45, ha='right', fontsize=8)
            ax_jaccard.set_yticklabels([top_500_basins[i][:6] for i in range(0, n_top500, step_top500)], fontsize=8)
            plt.colorbar(im_jaccard, ax=ax_jaccard, fraction=0.046, pad=0.04)
            plt.tight_layout()
            jaccard_file = f'{base_output_file}_jaccard_top_500.png'
            plt.savefig(jaccard_file, dpi=300, bbox_inches='tight')
            print(f"Jaccard Similarity heatmap (top 500) saved to: {jaccard_file}")
            plt.close()
            
            # Connectivity Probability heatmap (top 500)
            fig_prob, ax_prob = plt.subplots(figsize=(10, 10))
            max_prob_top500_val = np.max(prob_matrix_top500) if np.max(prob_matrix_top500) > 0 else 1.0
            im_prob = ax_prob.imshow(prob_matrix_top500, cmap='YlOrRd', aspect='equal', vmin=0, vmax=max_prob_top500_val)
            ax_prob.set_title('Connectivity Probability (Top 500)', fontsize=16, fontweight='bold')
            ax_prob.set_xlabel('Basin Index', fontsize=12)
            ax_prob.set_ylabel('Basin Index', fontsize=12)
            ax_prob.set_xticks(range(0, n_top500, step_top500))
            ax_prob.set_yticks(range(0, n_top500, step_top500))
            ax_prob.set_xticklabels([top_500_basins[i][:6] for i in range(0, n_top500, step_top500)], rotation=45, ha='right', fontsize=8)
            ax_prob.set_yticklabels([top_500_basins[i][:6] for i in range(0, n_top500, step_top500)], fontsize=8)
            plt.colorbar(im_prob, ax=ax_prob, fraction=0.046, pad=0.04)
            plt.tight_layout()
            prob_file = f'{base_output_file}_prob_top_500.png'
            plt.savefig(prob_file, dpi=300, bbox_inches='tight')
            print(f"Connectivity Probability heatmap (top 500) saved to: {prob_file}")
            plt.close()


def convert_to_native(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        # Convert tuple keys to strings for JSON compatibility
        result = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                # Convert tuple to string format: "(item1, item2)"
                key_str = str(k)
            else:
                key_str = k
            result[key_str] = convert_to_native(v)
        return result
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    if isinstance(obj, tuple):
        # Convert tuple to list for JSON compatibility
        return list(obj)
    return obj


def merge_results(existing, new):
    for instance_name, existing_stats in existing.items():
        if instance_name in new:
            existing_stats['runs'].update(new[instance_name]['runs'])
        else:
            new[instance_name] = existing_stats
    
    for instance_name in new:
        runs = new[instance_name]['runs']
        all_basins = defaultdict(int)
        for run_stats in runs.values():
            for basin_hash, freq in run_stats['run_summary']['basin_frequencies'].items():
                all_basins[basin_hash] += freq
        
        # Sum total_global_iters from all runs (cumulative count)
        total_global_iters = sum(run_stats['run_summary'].get('total_global_iters', 0) for run_stats in runs.values())
        
        instance_frequencies = list(all_basins.values())
        # Merge basin_connectivity from existing and new
        existing_connectivity = existing.get(instance_name, {}).get('instance_summary', {}).get('basin_connectivity', {}) if instance_name in existing else {}
        new_connectivity = new[instance_name].get('instance_summary', {}).get('basin_connectivity', {}) if 'instance_summary' in new[instance_name] else {}
        merged_connectivity = defaultdict(int)
        for k, v in existing_connectivity.items():
            merged_connectivity[k] += v
        for k, v in new_connectivity.items():
            merged_connectivity[k] += v
        
        new[instance_name]['instance_summary'] = {
            'num_runs': len(runs),
            'num_unique_basins_all_runs': len(all_basins),
            'total_hits_all_runs': sum(all_basins.values()),
            'total_global_iters': total_global_iters,
            'basin_frequencies': dict(all_basins),
            'basin_connectivity': dict(merged_connectivity),
            'frequency_distribution': {
                'min': min(instance_frequencies),
                'max': max(instance_frequencies),
                'mean': np.mean(instance_frequencies),
                'std': np.std(instance_frequencies),
                'median': np.median(instance_frequencies),
            },
            'frequency_histogram': dict(sorted(Counter(instance_frequencies).items())),
        }
    return new


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze basin statistics')
    parser.add_argument('--data_dir', type=str, default='/home/jieyi/cuopt/basin_datasets0_analyze')
    parser.add_argument('--instance_index', type=int, default=0, help='Instance index')
    parser.add_argument('--run_id', type=str, default=None, help='Run ID to process (if specified, only process this single run)')
    
    args = parser.parse_args()
    
    mode = 'single_run' if args.run_id else 'all_runs'
    
    # Generate output directory based on instance_index
    instance_dir = os.path.join(args.data_dir, f'cvrp100_uniform.pkl#{args.instance_index}')
    output_dir = os.path.join(instance_dir, 'basin_analysis_stats')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output file names based on instance_index and mode
    if mode == 'single_run':
        output_file = os.path.join(output_dir, f'basin_statistics_analysis_{args.instance_index}_run_{args.run_id}.xlsx')
        json_output_file = os.path.join(output_dir, f'basin_statistics_analysis_{args.instance_index}_run_{args.run_id}.json')
    else:
        output_file = os.path.join(output_dir, f'basin_statistics_analysis_{args.instance_index}.xlsx')
        json_output_file = os.path.join(output_dir, f'basin_statistics_analysis_{args.instance_index}.json')
    
    processed_run_ids = get_processed_run_ids(json_output_file) if mode == 'all_runs' else None
    
    instance_data, basin_info, trial_local_iters, global_iter_basins, global_iter_basin_freqs = analyze_basin_statistics(args.data_dir, args.instance_index, args.run_id, processed_run_ids)
    results = compute_statistics(instance_data, basin_info, trial_local_iters, global_iter_basins, global_iter_basin_freqs)
    
    if mode == 'all_runs' and os.path.exists(json_output_file):
        with open(json_output_file, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
        results = merge_results(existing_results, results)
    
    print_statistics(results)
    save_statistics_to_excel(results, output_file)
    
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(convert_to_native(results), f, indent=2, ensure_ascii=False)
    print(f"Statistics saved to JSON: {json_output_file}")
    
    # Generate visualization
    viz_output_file = output_file.replace('.xlsx', '_connectivity.png')
    visualize_basin_connectivity(results, viz_output_file)


if __name__ == "__main__":
    main()
