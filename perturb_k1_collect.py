#!/usr/bin/env python3
"""
Collect k=1 perturbation samples for training data.

For each local optima:
1. Load existing k=1 data from JSONL file
2. Check if we already have positive (return_prob >= 0.8) or negative (return_prob <= 0.1) samples
3. If not, run k=1 perturbations (max 5 times total, minus existing runs)
4. Collect:
   - Positive sample: highest return_prob among those >= 0.8
   - Negative sample: lowest return_prob among those <= 0.1
5. Stop when both positive and negative samples are collected
6. Record success/failure data

Output:
- JSONL file with full results (anchor, positive_sample, negative_sample)
- Excel file with summary (success/failure, hashes, costs, return_probs, jaccard, broken_pairs)

Usage:
    python perturb_k1_collect.py --operator_type remove_and_insert --instance_index 0
    ./run_perturb_k1_collect.sh remove_and_insert 0 [gpu_id]
"""

import glob
import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import pickle
from utils import edges_to_routes as edges_to_routes
from utils import routes_to_edges as routes_to_edges_set
from utils import get_basin_paths, edges_hash as edges_hash_func, routes_to_solution_flat

# Import functions from perturb.py
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perturb import (
    remove_and_insert_move,
    double_bridge_move,
    compute_cost_from_edges,
    edges_to_normalized_set,
    calculate_broken_pairs_distance,
    run_local_search_from_perturbed,
    load_optima_from_jsonl,
    load_trajectory_info,
)


def append_jsonl(path: str, record: dict):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def parse_run_log_for_outcomes(log_path: str, full_optima: List[dict]) -> Dict[str, str]:
    """
    Parse run_k1_collect.log and return anchor_hash -> last outcome: "success" | "failed" | "error".
    Same anchor may appear multiple times (e.g. retries); last occurrence wins.
    Batch identity uses batch start_idx and batch_id (S*_B*) to map "Processing N/M" to optima index.
    """
    import re
    outcome_by_hash = {}
    if not log_path or not os.path.exists(log_path) or not full_optima:
        return outcome_by_hash
    start_idx = 0
    current_anchor_hash = None
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip()
            # ---- batch start_idx=0 (batch_id=S0_B100) ----
            m = re.search(r'batch start_idx=(\d+)\s*\(batch_id=S\d+_B(\d+)\)', line)
            if m:
                start_idx = int(m.group(1))
                current_anchor_hash = None
                continue
            #   Processing 5/100 (hash=e33519c4...)
            m = re.search(r'Processing\s+(\d+)/(\d+)\s*\(hash=', line)
            if m:
                one_based = int(m.group(1))
                idx = start_idx + one_based - 1
                if 0 <= idx < len(full_optima):
                    current_anchor_hash = full_optima[idx].get('edges_hash')
                else:
                    current_anchor_hash = None
                continue
            #     Success: ...  or  Failed ...  or  Error (recorded ...
            if current_anchor_hash is not None:
                s = line.strip()
                if s.startswith('Success') or (s.startswith('Success:') and 'pos=' in line):
                    outcome_by_hash[current_anchor_hash] = 'success'
                    current_anchor_hash = None
                elif s.startswith('Failed') and 'Error' not in line:
                    outcome_by_hash[current_anchor_hash] = 'failed'
                    current_anchor_hash = None
                elif s.startswith('Error (') or 'Error (recorded' in line:
                    outcome_by_hash[current_anchor_hash] = 'error'
                    current_anchor_hash = None
    return outcome_by_hash


def load_results_anchor_hashes_from_dir(
    output_dir: str,
    operator_type: str,
    runs: int,
) -> set:
    """
    Load anchor hashes that have results data in results JSONL files.
    """
    hashes = set()
    pattern = os.path.join(output_dir, f"k1_collection_results_{operator_type}.batch_*_r{runs}.jsonl")
    all_paths = sorted(glob.glob(pattern))
    all_jsonl = os.path.join(output_dir, f"k1_collection_results_{operator_type}.ALL_r{runs}.jsonl")
    if os.path.exists(all_jsonl):
        all_paths.append(all_jsonl)

    for path in all_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                anchor = rec.get('anchor', {})
                h = anchor.get('edges_hash')
                if h:
                    hashes.add(h)
    return hashes


def iter_results_records_from_dir(output_dir: str, operator_type: str, runs: int):
    """
    Yield raw JSON records from all results JSONL in output_dir (batch + ALL).
    """
    pattern = os.path.join(output_dir, f"k1_collection_results_{operator_type}.batch_*_r{runs}.jsonl")
    all_paths = sorted(glob.glob(pattern))
    all_jsonl = os.path.join(output_dir, f"k1_collection_results_{operator_type}.ALL_r{runs}.jsonl")
    if os.path.exists(all_jsonl):
        all_paths.append(all_jsonl)

    for path in all_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                yield rec


def load_summary_state_by_anchor_hash_from_dir(
    output_dir: str,
    operator_type: str,
    runs: int,
    log_path: Optional[str] = None,
    full_optima: Optional[List[dict]] = None,
) -> dict:
    """
    Load the latest summary state per anchor_hash from all summary JSONL in output_dir.

    Returns:
      dict(anchor_hash -> dict(success: bool|None, error: bool|None, has_summary: bool))

    Notes:
    - If multiple rows exist for the same anchor_hash, the *last encountered* row wins.
      (We iterate files in sorted order; within a file, later lines win.)
    - For old runs without an "error" field, we parse run_k1_collect.log (last occurrence per
      anchor wins) and treat as error if the last log outcome is "error".
    """
    state: dict = {}
    pattern = os.path.join(output_dir, f"k1_collection_summary_{operator_type}.batch_*_r{runs}.jsonl")
    all_paths = sorted(glob.glob(pattern))
    all_jsonl = os.path.join(output_dir, f"k1_collection_summary_{operator_type}.ALL_r{runs}.jsonl")
    if os.path.exists(all_jsonl):
        all_paths.append(all_jsonl)

    log_outcomes = {}
    if log_path and full_optima is not None:
        log_outcomes = parse_run_log_for_outcomes(log_path, full_optima)

    for path in all_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                h = rec.get('anchor_hash')
                if not h:
                    continue

                # Determine error flag (handle old rows)
                if 'error' in rec:
                    err = rec.get('error') is True
                else:
                    err = (h in log_outcomes and log_outcomes[h] == 'error')

                state[h] = {
                    'has_summary': True,
                    'success': rec.get('success'),
                    'error': err,
                }

    return state


def synthesize_summary_row_from_results_record(result_rec: dict) -> dict:
    """
    Create a minimal-but-compatible summary row from a results JSONL record.
    results => must have been success.
    """
    anchor = result_rec.get('anchor', {}) or {}
    pos = result_rec.get('positive_sample') or {}
    neg = result_rec.get('negative_sample') or {}

    return {
        'success': True,
        'error': False,
        'anchor_hash': anchor.get('edges_hash'),
        'anchor_cost': anchor.get('cost'),
        'run_id': None,
        'frequency': None,
        'trial_id': None,
        'global_iter': None,
        'local_iter': None,
        'occurrence_ids': None,
        'num_runs': result_rec.get('num_runs'),
        'max_return_prob': None,
        'min_return_prob': None,
        'positive_hash': pos.get('edges_hash'),
        'positive_cost': pos.get('cost'),
        'positive_return_prob': pos.get('return_prob'),
        'positive_jaccard': pos.get('jaccard'),
        'positive_broken_pairs': pos.get('broken_pairs_distance'),
        'negative_hash': neg.get('edges_hash'),
        'negative_cost': neg.get('cost'),
        'negative_return_prob': neg.get('return_prob'),
        'negative_jaccard': neg.get('jaccard'),
        'negative_broken_pairs': neg.get('broken_pairs_distance'),
    }


def load_processed_anchor_hashes_from_dir(
    output_dir: str,
    operator_type: str,
    runs: int,
    log_path: Optional[str] = None,
    full_optima: Optional[List[dict]] = None,
) -> set:
    """
    Load anchor hashes that should be skipped when resuming.

    Implemented policy (per-anchor):
    - summary missing & results missing  => re-run
    - summary present & results present  => skip
    - summary success  & results missing => re-run (results lost)
    - summary failed   & results missing => skip (normal failure)
    - summary error    & results missing => re-run
    - summary missing  & results present => skip (results imply success); caller may backfill summary
    """
    results_hashes = load_results_anchor_hashes_from_dir(output_dir, operator_type, runs)
    summary_state = load_summary_state_by_anchor_hash_from_dir(
        output_dir,
        operator_type,
        runs,
        log_path=log_path,
        full_optima=full_optima,
    )

    processed = set()

    # Skip everything that already has results (results imply success)
    for h in results_hashes:
        processed.add(h)

    # Apply summary logic
    for h, st in summary_state.items():
        if st.get('error') is True:
            continue  # retry errors

        if st.get('success') is True:
            # success should have results; only skip if results exists
            if h in results_hashes:
                processed.add(h)
            continue

        # success=False => normal fail => skip
        processed.add(h)

    return processed


def write_summary_xlsx_from_jsonl(summary_jsonl_path: str, output_xlsx: str):
    rows = []
    if os.path.exists(summary_jsonl_path):
        with open(summary_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(['success', 'anchor_hash'], ascending=[False, True])
    with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Collection Results', index=False)
    if len(df) > 0 and 'success' in df.columns:
        print(f"Total: {len(df)}, Success: {df['success'].sum()}, Failed: {(~df['success']).sum()}")
        if 'error' in df.columns and df['error'].any():
            print(f"  (Errors recorded for retry: {df['error'].sum()})")


def load_existing_k1_data(jsonl_paths: List[str]) -> Dict[str, List[dict]]:
    existing_data = defaultdict(list)
    for jsonl_path in jsonl_paths:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record['perturbation_step'] != 1:
                    continue
                anchor_hash = record['initial_solution']['edges_hash']
                if anchor_hash:
                    existing_data[anchor_hash].append(record)
    return dict(existing_data)


def check_samples_status(existing_records: List[dict]) -> Tuple[Optional[dict], Optional[dict], int]:
    best_positive = None
    best_negative = None
    
    for record in existing_records:
        return_prob = record['return_to_original_ratio']
        
        if return_prob >= 0.8:
            if best_positive is None or return_prob > best_positive['return_to_original_ratio']:
                best_positive = record
        
        if return_prob <= 0.1:
            if best_negative is None or return_prob < best_negative['return_to_original_ratio']:
                best_negative = record
    
    return best_positive, best_negative, len(existing_records)


def _candidate_from_existing_record(
    record: dict,
    anchor_routes,
    anchor_edges_set,
    num_orders: int,
) -> dict:
    ps = record["perturbed_solution"]
    pert_edges = ps["edges"]
    pert_routes = edges_to_routes(pert_edges, num_orders)

    pert_edges_set = edges_to_normalized_set(pert_edges)
    inter = len(anchor_edges_set & pert_edges_set)
    union = len(anchor_edges_set | pert_edges_set)
    jaccard_distance = 1.0 - (inter / union) if union > 0 else 1.0

    _, _, broken_pairs_ratio = calculate_broken_pairs_distance(anchor_routes, pert_routes)

    return {
        "perturbed_solution": {
            "edges_hash": ps["edges_hash"],
            "edges": pert_edges,
            "cost": ps["cost"],
        },
        "return_to_original_ratio": record["return_to_original_ratio"],
        "jaccard_distance": jaccard_distance,
        "broken_pairs_ratio": broken_pairs_ratio,
        "solution_flat": routes_to_solution_flat(pert_routes, num_orders),
    }


def collect_k1_samples_for_optimum(
    optimum: dict,
    existing_records: List[dict],
    instance_path: str,
    instance_index: int,
    coordinates: List[List[float]],
    demands: Dict[int, float],
    vehicle_capacity: float,
    operator_type: str,
    num_local_search_runs: int,
    max_runs: int = 5,
    rng: np.random.Generator = None,
) -> Dict:
    if rng is None:
        rng = np.random.default_rng()
    
    anchor_hash = optimum['edges_hash']
    anchor_edges = optimum['edges']
    
    best_positive, best_negative, num_existing = check_samples_status(existing_records)
    
    max_node_id = max(max(e) for e in anchor_edges)
    num_orders = max_node_id + 1
    anchor_routes = edges_to_routes(anchor_edges, num_orders)
    anchor_solution_flat = routes_to_solution_flat(anchor_routes, num_orders)
    
    anchor_data = {
        'edges_hash': anchor_hash,
        'solution_flat': anchor_solution_flat,
        'cost': optimum['final_cost'],
        'edges': anchor_edges,
        'run_id': optimum.get('anchor_run_id') or optimum.get('run_id'),
        'trial_id': optimum.get('anchor_trial_id'),
        'global_iter': optimum.get('anchor_global_iter'),
        'local_iter': optimum.get('anchor_local_iter'),
        'frequency': optimum.get('anchor_frequency'),
        'occurrence_ids': optimum.get('anchor_occurrence_ids'),
    }
    
    remaining_runs = max(0, max_runs - num_existing)
    all_return_probs = [rec['return_to_original_ratio'] for rec in existing_records]
    anchor_edges_set = edges_to_normalized_set(anchor_edges)
    num_new_runs = 0

    current_positive = (
        _candidate_from_existing_record(best_positive, anchor_routes, anchor_edges_set, num_orders)
        if best_positive is not None
        else None
    )
    current_negative = (
        _candidate_from_existing_record(best_negative, anchor_routes, anchor_edges_set, num_orders)
        if best_negative is not None
        else None
    )
    
    for run_idx in range(remaining_runs):
        num_new_runs += 1
        if operator_type == 'double_bridge':
            perturbed_edges, _, _ = double_bridge_move(
                anchor_edges, num_orders, rng,
                demands=demands, vehicle_capacity=vehicle_capacity,
                require_feasible=False
            )
        else:
            perturbed_edges, _, _ = remove_and_insert_move(
                anchor_edges, num_orders, rng,
                demands=demands, vehicle_capacity=vehicle_capacity,
                require_feasible=False
            )
        
        perturbed_edges_set = edges_to_normalized_set(perturbed_edges)
        perturbed_hash = edges_hash_func(perturbed_edges_set)
        perturbed_cost = compute_cost_from_edges(perturbed_edges, coordinates, scale=100.0)
        
        inter = len(anchor_edges_set & perturbed_edges_set)
        union = len(anchor_edges_set | perturbed_edges_set)
        jaccard_distance = 1.0 - (inter / union) if union > 0 else 1.0
        
        perturbed_routes = edges_to_routes(perturbed_edges, num_orders)
        _, _, broken_pairs_ratio = calculate_broken_pairs_distance(anchor_routes, perturbed_routes)
        
        ls_results = run_local_search_from_perturbed(
            perturbed_edges, instance_path, instance_index,
            num_runs=num_local_search_runs,
            original_edges_hash=anchor_hash
        )
        
        return_prob = ls_results['return_to_original_ratio']
        all_return_probs.append(return_prob)
        
        if return_prob >= 0.8:
            if current_positive is None or return_prob > current_positive['return_to_original_ratio']:
                current_positive = {
                    'perturbed_solution': {
                        'edges_hash': perturbed_hash,
                        'edges': perturbed_edges,
                        'cost': perturbed_cost,
                    },
                    'return_to_original_ratio': return_prob,
                    'jaccard_distance': jaccard_distance,
                    'broken_pairs_ratio': broken_pairs_ratio,
                    'solution_flat': routes_to_solution_flat(perturbed_routes, num_orders),
                }
        
        if return_prob <= 0.1:
            if current_negative is None or return_prob < current_negative['return_to_original_ratio']:
                current_negative = {
                    'perturbed_solution': {
                        'edges_hash': perturbed_hash,
                        'edges': perturbed_edges,
                        'cost': perturbed_cost,
                    },
                    'return_to_original_ratio': return_prob,
                    'jaccard_distance': jaccard_distance,
                    'broken_pairs_ratio': broken_pairs_ratio,
                    'solution_flat': routes_to_solution_flat(perturbed_routes, num_orders),
                }
        
        if current_positive and current_negative:
            break
    
    success = current_positive is not None and current_negative is not None
    
    result = {
        'anchor': anchor_data,
        'positive_sample': None,
        'negative_sample': None,
        'num_runs': num_existing + num_new_runs,
        'max_return_prob': max(all_return_probs) if all_return_probs else None,
        'min_return_prob': min(all_return_probs) if all_return_probs else None,
        'success': success
    }
    
    if success:
        result['positive_sample'] = {
            'edges_hash': current_positive['perturbed_solution']['edges_hash'],
            'solution_flat': current_positive['solution_flat'],
            'cost': current_positive['perturbed_solution']['cost'],
            'return_prob': current_positive['return_to_original_ratio'],
            'jaccard': current_positive['jaccard_distance'],
            'broken_pairs_distance': current_positive['broken_pairs_ratio'],
        }
        
        result['negative_sample'] = {
            'edges_hash': current_negative['perturbed_solution']['edges_hash'],
            'solution_flat': current_negative['solution_flat'],
            'cost': current_negative['perturbed_solution']['cost'],
            'return_prob': current_negative['return_to_original_ratio'],
            'jaccard': current_negative['jaccard_distance'],
            'broken_pairs_distance': current_negative['broken_pairs_ratio'],
        }
    
    return result


def save_collection_results(results: List[dict], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            if result['success']:
                anchor = result['anchor']
                record_out = {
                    'anchor': {
                        'edges_hash': anchor['edges_hash'],
                        'solution_flat': anchor['solution_flat'],
                        'cost': anchor['cost'],
                    },
                    'positive_sample': result['positive_sample'],
                    'negative_sample': result['negative_sample'],
                    'num_runs': result['num_runs'],
                }
                f.write(json.dumps(record_out, ensure_ascii=False) + '\n')


def save_collection_summary(results: List[dict], output_path: str):
    rows = []
    for result in results:
        anchor = result['anchor']
        pos = result.get('positive_sample')
        neg = result.get('negative_sample')
        
        row = {
            'success': result['success'],
            'anchor_hash': anchor['edges_hash'],
            'anchor_cost': anchor['cost'],
            'run_id': anchor.get('run_id'),
            'frequency': anchor.get('frequency'),
            'trial_id': anchor.get('trial_id'),
            'global_iter': anchor.get('global_iter'),
            'local_iter': anchor.get('local_iter'),
            'occurrence_ids': anchor.get('occurrence_ids'),
            'num_runs': result['num_runs'],
            'max_return_prob': result['max_return_prob'],
            'min_return_prob': result['min_return_prob'],
            'positive_hash': pos['edges_hash'] if pos else None,
            'positive_cost': pos['cost'] if pos else None,
            'positive_return_prob': pos['return_prob'] if pos else None,
            'positive_jaccard': pos['jaccard'] if pos else None,
            'positive_broken_pairs': pos['broken_pairs_distance'] if pos else None,
            'negative_hash': neg['edges_hash'] if neg else None,
            'negative_cost': neg['cost'] if neg else None,
            'negative_return_prob': neg['return_prob'] if neg else None,
            'negative_jaccard': neg['jaccard'] if neg else None,
            'negative_broken_pairs': neg['broken_pairs_distance'] if neg else None,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['success', 'anchor_hash'], ascending=[False, True])
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Collection Results', index=False)
    
    print(f"Total: {len(df)}, Success: {df['success'].sum()}, Failed: {(~df['success']).sum()}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect k=1 perturbation samples for training data')
    parser.add_argument('--instance_path', type=str, default="/home/jieyi/cvrp100_uniform.pkl",
                        help='Path to instance pkl file')
    parser.add_argument('--instance_index', type=int, default=0,
                        help='Instance index (default: 0)')
    parser.add_argument('--basin_base_dir', type=str, default='basin_datasets0',
                        help='Input basin directory (default: basin_datasets0)')
    parser.add_argument('--output_base_dir', type=str, default='perturb_k1_collect',
                        help='Output directory (default: perturb_k1_collect)')
    parser.add_argument('--existing_data_dir', type=str, default='basin_datasets0_analyze',
                        help='Directory containing existing k=5 data (we extract k=1 from it)')
    parser.add_argument('--operator_type', type=str, default='remove_and_insert',
                        choices=['double_bridge', 'remove_and_insert'], help='Perturbation operator type (default: remove_and_insert)')
    parser.add_argument('--num_local_search_runs', type=int, default=30,
                        help='Number of local search runs per perturbation (default: 30)')
    parser.add_argument('--max_runs', type=int, default=10,
                        help='Maximum number of k=1 perturbations per optimum (default: 10)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index of optima to process')
    parser.add_argument('--max_optima', type=int, default=None,
                        help='Maximum number of optima to process')
    parser.add_argument('--batch_id', type=str, default=None,
                        help='Batch ID for output filenames')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing batch summary JSONL (skip processed anchor_hash)')
    parser.add_argument('--max_run_id', type=int, default=None,
                        help='(Deprecated: use --max_first_runs) Only process basins with run_id <= this')
    parser.add_argument('--max_first_runs', type=int, default=None,
                        help='Only process basins that appear in the first N distinct run_ids in this instance\'s trajectory')
    
    args = parser.parse_args()
    
    instance_id = f"{os.path.basename(args.instance_path)}#{args.instance_index}"
    
    basin_paths = get_basin_paths(args.instance_path, args.instance_index, args.basin_base_dir)
    jsonl_path = os.path.join(basin_paths['basin_dir'], 'optima.jsonl')
    trajectory_path = os.path.join(basin_paths['basin_dir'], 'trajectory.jsonl')
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_base_dir, instance_id)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(args.instance_path, 'rb') as f:
        instances = pickle.load(f)
    instance_tuple = instances[args.instance_index]
    coordinates = [instance_tuple[0][0]] + instance_tuple[1]
    demands = {i + 1: float(instance_tuple[2][i]) for i in range(len(instance_tuple[2]))}
    vehicle_capacity = float(instance_tuple[3])
    
    full_optima = load_optima_from_jsonl(jsonl_path)
    trajectory_info = load_trajectory_info(trajectory_path)

    if args.max_first_runs is not None:
        # First N runs = first N distinct run_ids in this instance's trajectory (sorted numerically)
        all_run_ids = set()
        for occs in trajectory_info.values():
            for occ in occs:
                rid = occ.get('run_id')
                if rid is not None:
                    all_run_ids.add(rid)
        def run_id_key(r):
            try:
                return (0, int(r))
            except (TypeError, ValueError):
                return (1, str(r))
        first_n_run_ids = set(sorted(all_run_ids, key=run_id_key)[: args.max_first_runs])
        full_optima = [
            opt for opt in full_optima
            if any(occ.get('run_id') in first_n_run_ids for occ in trajectory_info.get(opt.get('edges_hash'), []))
        ]
    elif args.max_run_id is not None:
        full_optima = [
            opt for opt in full_optima
            if any(int(occ.get('run_id', 0)) <= args.max_run_id for occ in trajectory_info.get(opt.get('edges_hash'), []))
        ]

    end_idx = len(full_optima) if args.max_optima is None else args.start_idx + args.max_optima
    optima = full_optima[args.start_idx:end_idx]
    
    existing_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.existing_data_dir)
    all_jsonl = os.path.join(existing_data_dir, instance_id, f"{args.operator_type}_training_data.ALL_r{args.num_local_search_runs}.jsonl")
    existing_jsonl_paths = [all_jsonl] if os.path.exists(all_jsonl) else []
    existing_data = load_existing_k1_data(existing_jsonl_paths) if existing_jsonl_paths else {}
    
    rng = np.random.default_rng(args.seed)

    # Use single ALL files (append-only) instead of per-batch files.
    results_jsonl = os.path.join(
        output_dir,
        f"k1_collection_results_{args.operator_type}.ALL_r{args.num_local_search_runs}.jsonl",
    )
    summary_jsonl = os.path.join(
        output_dir,
        f"k1_collection_summary_{args.operator_type}.ALL_r{args.num_local_search_runs}.jsonl",
    )
    output_xlsx = os.path.join(
        output_dir,
        f"k1_collection_summary_{args.operator_type}.ALL_r{args.num_local_search_runs}.xlsx",
    )

    log_path = os.path.join(output_dir, "run_k1_collect.log")
    processed = (
        load_processed_anchor_hashes_from_dir(
            output_dir, args.operator_type, args.num_local_search_runs,
            log_path=log_path, full_optima=full_optima,
        )
        if args.resume else set()
    )

    # Resume enhancement:
    # If results exist but summary is missing, we can infer success and backfill summary rows.
    # This prevents reprocessing and keeps downstream stats/merge consistent.
    if args.resume:
        summary_state = load_summary_state_by_anchor_hash_from_dir(
            output_dir,
            args.operator_type,
            args.num_local_search_runs,
            log_path=log_path,
            full_optima=full_optima,
        )
        summary_hashes = set(summary_state.keys())
        backfilled = 0
        for rec in iter_results_records_from_dir(output_dir, args.operator_type, args.num_local_search_runs):
            anchor = rec.get('anchor', {}) or {}
            h = anchor.get('edges_hash')
            if not h or h in summary_hashes:
                continue
            row = synthesize_summary_row_from_results_record(rec)
            if not row.get('anchor_hash'):
                continue
            append_jsonl(summary_jsonl, row)
            summary_hashes.add(h)
            backfilled += 1
        if backfilled:
            print(f"  Backfilled {backfilled} missing summary rows from existing results.")

    # Per-call batch statistics (only for this invocation)
    skipped_anchors = 0
    attempted_anchors = 0
    success_anchors = 0
    failed_anchors = 0
    error_anchors = 0

    for i, opt in enumerate(optima):
        anchor_hash = opt['edges_hash']
        if anchor_hash in processed:
            if args.resume:
                print(f"  Skipped (already done): {i+1}/{len(optima)} hash={anchor_hash[:8]}...")
                skipped_anchors += 1
            continue
        occurrences = trajectory_info.get(anchor_hash, [])
        if occurrences:
            best_occ = max(
                occurrences,
                key=lambda o: ((o.get('global_iter') or -1), (o.get('local_iter') or -1)),
            )
            opt['anchor_run_id'] = best_occ.get('run_id')
            opt['anchor_trial_id'] = best_occ.get('trial_id')
            opt['anchor_global_iter'] = best_occ.get('global_iter')
            opt['anchor_local_iter'] = best_occ.get('local_iter')
            opt['anchor_frequency'] = len(occurrences)
            opt['anchor_occurrence_ids'] = "; ".join(
                f"{o.get('run_id')}_{o.get('trial_id')}_{o.get('global_iter')}_{o.get('local_iter')}"
                for o in occurrences
            )
        print(f"  Processing {i+1}/{len(optima)} (hash={anchor_hash[:8]}...)")
        attempted_anchors += 1
        try:
            existing_records = existing_data.get(anchor_hash, [])
            result = collect_k1_samples_for_optimum(
                opt, existing_records,
                args.instance_path, args.instance_index,
                coordinates, demands, vehicle_capacity,
                args.operator_type, args.num_local_search_runs,
                max_runs=args.max_runs, rng=rng
            )
        except Exception as e:
            error_anchors += 1
            summary_row = {
                'success': False,
                'error': True,
                'error_message': str(e)[:500],
                'anchor_hash': anchor_hash,
                'anchor_cost': opt.get('final_cost'),
                'run_id': None,
                'frequency': None,
                'trial_id': None,
                'global_iter': None,
                'local_iter': None,
                'occurrence_ids': None,
                'num_runs': None,
                'max_return_prob': None,
                'min_return_prob': None,
                'positive_hash': None,
                'positive_cost': None,
                'positive_return_prob': None,
                'positive_jaccard': None,
                'positive_broken_pairs': None,
                'negative_hash': None,
                'negative_cost': None,
                'negative_return_prob': None,
                'negative_jaccard': None,
                'negative_broken_pairs': None,
            }
            append_jsonl(summary_jsonl, summary_row)
            print(f"    Error (recorded, will retry on next run): {e!r}")
            continue

        result['optimum_id'] = opt['optimum_id']
        result['anchor']['optimum_id'] = opt['optimum_id']

        anchor = result['anchor']
        pos = result.get('positive_sample')
        neg = result.get('negative_sample')
        summary_row = {
            'success': result['success'],
            'error': False,
            'anchor_hash': anchor['edges_hash'],
            'anchor_cost': anchor['cost'],
            'run_id': anchor.get('run_id'),
            'frequency': anchor.get('frequency'),
            'trial_id': anchor.get('trial_id'),
            'global_iter': anchor.get('global_iter'),
            'local_iter': anchor.get('local_iter'),
            'occurrence_ids': anchor.get('occurrence_ids'),
            'num_runs': result['num_runs'],
            'max_return_prob': result['max_return_prob'],
            'min_return_prob': result['min_return_prob'],
            'positive_hash': pos['edges_hash'] if pos else None,
            'positive_cost': pos['cost'] if pos else None,
            'positive_return_prob': pos['return_prob'] if pos else None,
            'positive_jaccard': pos['jaccard'] if pos else None,
            'positive_broken_pairs': pos['broken_pairs_distance'] if pos else None,
            'negative_hash': neg['edges_hash'] if neg else None,
            'negative_cost': neg['cost'] if neg else None,
            'negative_return_prob': neg['return_prob'] if neg else None,
            'negative_jaccard': neg['jaccard'] if neg else None,
            'negative_broken_pairs': neg['broken_pairs_distance'] if neg else None,
        }
        append_jsonl(summary_jsonl, summary_row)
        processed.add(anchor_hash)

        if result['success']:
            success_anchors += 1
            record_out = {
                'anchor': {
                    'edges_hash': anchor['edges_hash'],
                    'solution_flat': anchor['solution_flat'],
                    'cost': anchor['cost'],
                },
                'positive_sample': result['positive_sample'],
                'negative_sample': result['negative_sample'],
                'num_runs': result['num_runs'],
            }
            append_jsonl(results_jsonl, record_out)

        if result['success']:
            print(
                f"    Success: pos={result['positive_sample']['return_prob']:.3f}, "
                f"neg={result['negative_sample']['return_prob']:.3f}, "
                f"num_runs={result['num_runs']}"
            )
        else:
            failed_anchors += 1
            print(f"    Failed (num_runs={result['num_runs']})")
    # End of per-call batch; write Excel for up-to-date summary
    write_summary_xlsx_from_jsonl(summary_jsonl, output_xlsx)

    # Print per-call batch statistics summary
    total_considered = len(optima)
    print(
        f"Batch summary (this call only): "
        f"total_optima_slice={total_considered}, "
        f"skipped={skipped_anchors}, "
        f"attempted={attempted_anchors}, "
        f"success={success_anchors}, "
        f"failed={failed_anchors}, "
        f"errors={error_anchors}"
    )


if __name__ == '__main__':
    main()
