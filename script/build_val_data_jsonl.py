#!/usr/bin/env python3
"""
Build val_data.jsonl: 10 records from instances 50-79.

For each of 10 randomly chosen instances (50-79):
- Pick one local optimum from runs 11+ (exclude first 10 runs).
- Run k=1 collection: up to 150 perturbation trials, 30 local search runs each;
  collect 30 positive (P > 0.8) and 30 negative (P < 0.1).
- Write one line: instance_index, instance_data (pkl data for that instance),
  anchor (solution_flat, edges_hash, cost), positive_samples (30), negative_samples (30).
- Append every perturbation run to val_data_all_perturb_record.jsonl: edges_hash, solution_flat,
  cost, return_prob (P), return_count (count of LS runs that returned to anchor).

Usage:
  python script/build_val_data_jsonl.py [--instance_path PATH] [--basin_base_dir DIR] [--seed N] [--out val_data.jsonl]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle

from utils import get_basin_paths, edges_to_routes, routes_to_solution_flat, edges_hash as edges_hash_func
from perturb import (
    load_optima_from_jsonl,
    load_trajectory_info,
    remove_and_insert_move,
    double_bridge_move,
    compute_cost_from_edges,
    edges_to_normalized_set,
    calculate_broken_pairs_distance,
    run_local_search_from_perturbed,
)


def run_id_sort_key(r):
    try:
        return (0, int(r))
    except (TypeError, ValueError):
        return (1, str(r))


def get_anchor_hashes_from_runs_after_n(trajectory_info, n=10):
    """Return set of edges_hash that appear in at least one run after the first n runs."""
    all_run_ids = set()
    for occs in trajectory_info.values():
        for occ in occs:
            rid = occ.get("run_id")
            if rid is not None:
                all_run_ids.add(rid)
    sorted_run_ids = sorted(all_run_ids, key=run_id_sort_key)
    first_n_ids = set(sorted_run_ids[:n])
    after_n_hashes = set()
    for edges_hash, occs in trajectory_info.items():
        if any(occ.get("run_id") not in first_n_ids for occ in occs):
            after_n_hashes.add(edges_hash)
    return after_n_hashes


def instance_tuple_to_data(instance_tuple):
    """Serialize one instance from pkl to JSON-serializable dict."""
    depot = instance_tuple[0]
    customer_coords = instance_tuple[1]
    demands = instance_tuple[2]
    vehicle_capacity = float(instance_tuple[3])
    return {
        "depot": np.asarray(depot).tolist() if hasattr(depot, "tolist") else list(depot),
        "customer_coords": np.asarray(customer_coords).tolist()
        if hasattr(customer_coords, "tolist")
        else list(customer_coords),
        "demands": np.asarray(demands).tolist() if hasattr(demands, "tolist") else list(demands),
        "vehicle_capacity": vehicle_capacity,
    }


def collect_30_30_samples(
    optimum,
    instance_path,
    instance_index,
    coordinates,
    demands,
    vehicle_capacity,
    operator_type,
    num_local_search_runs,
    max_runs,
    rng,
    all_perturb_records=None,
):
    """
    Run up to max_runs perturbation trials; collect 30 positive (P > 0.8) and 30 negative (P < 0.1).
    If all_perturb_records is a list, append every perturbation result: edges_hash, solution_flat, cost, P, count.
    Returns (anchor_data, positive_list, negative_list) or None if cannot get 30+30.
    """
    anchor_hash = optimum["edges_hash"]
    anchor_edges = optimum["edges"]
    max_node_id = max(max(e) for e in anchor_edges)
    num_orders = max_node_id + 1
    anchor_routes = edges_to_routes(anchor_edges, num_orders)
    anchor_solution_flat = routes_to_solution_flat(anchor_routes, num_orders)
    anchor_edges_set = edges_to_normalized_set(anchor_edges)

    anchor_data = {
        "edges_hash": anchor_hash,
        "solution_flat": anchor_solution_flat,
        "cost": optimum["final_cost"],
    }

    positive_list = []
    negative_list = []
    _progress_interval = 25

    for run_idx in range(max_runs):
        if operator_type == "double_bridge":
            perturbed_edges, _, _ = double_bridge_move(
                anchor_edges,
                num_orders,
                rng,
                demands=demands,
                vehicle_capacity=vehicle_capacity,
                require_feasible=False,
            )
        else:
            perturbed_edges, _, _ = remove_and_insert_move(
                anchor_edges,
                num_orders,
                rng,
                demands=demands,
                vehicle_capacity=vehicle_capacity,
                require_feasible=False,
            )

        perturbed_edges_set = edges_to_normalized_set(perturbed_edges)
        perturbed_hash = edges_hash_func(perturbed_edges_set)
        perturbed_cost = compute_cost_from_edges(perturbed_edges, coordinates, scale=100.0)
        perturbed_routes = edges_to_routes(perturbed_edges, num_orders)
        solution_flat = routes_to_solution_flat(perturbed_routes, num_orders)

        ls_results = run_local_search_from_perturbed(
            perturbed_edges,
            instance_path,
            instance_index,
            num_runs=num_local_search_runs,
            original_edges_hash=anchor_hash,
        )
        return_prob = ls_results["return_to_original_ratio"]
        return_count = int(ls_results.get("return_to_original", 0))

        if all_perturb_records is not None:
            all_perturb_records.append({
                "instance_index": instance_index,
                "anchor_hash": anchor_hash,
                "edges_hash": perturbed_hash,
                "solution_flat": solution_flat,
                "cost": perturbed_cost,
                "return_prob": return_prob,
                "return_count": return_count,
            })

        if (run_idx + 1) % _progress_interval == 0:
            print(f"    perturb {run_idx + 1}/{max_runs}  pos={len(positive_list)} neg={len(negative_list)}", flush=True)

        sample = {"solution_flat": solution_flat, "edges_hash": perturbed_hash, "cost": perturbed_cost}
        if return_prob >= 0.8:
            positive_list.append(sample)
            if len(positive_list) >= 30:
                positive_list = positive_list[:30]
        if return_prob <= 0.1:
            negative_list.append(sample)
            if len(negative_list) >= 30:
                negative_list = negative_list[:30]

        if len(positive_list) >= 30 and len(negative_list) >= 30:
            break

    if len(positive_list) >= 30 and len(negative_list) >= 30:
        return anchor_data, positive_list[:30], negative_list[:30]
    return None


def build_one_val_record(
    instance_index,
    instance_path,
    basin_base_dir,
    operator_type,
    num_local_search_runs,
    max_runs,
    rng,
):
    """
    For one instance: pick one anchor from runs 11+, collect 30 positive + 30 negative samples.
    Return one record dict (instance_index, instance_data, anchor, positive_samples, negative_samples) or None.
    """
    basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
    basin_dir = basin_paths["basin_dir"]
    optima_path = os.path.join(basin_dir, "optima.jsonl")
    trajectory_path = os.path.join(basin_dir, "trajectory.jsonl")

    if not os.path.isfile(optima_path) or not os.path.isfile(trajectory_path):
        return None, []

    full_optima = load_optima_from_jsonl(optima_path)
    trajectory_info = load_trajectory_info(trajectory_path)
    after_10_hashes = get_anchor_hashes_from_runs_after_n(trajectory_info, n=10)
    candidate_optima = [
        o
        for o in full_optima
        if o.get("edges_hash") in after_10_hashes and o.get("edges")
    ]
    if not candidate_optima:
        return None, []

    with open(instance_path, "rb") as f:
        instances = pickle.load(f)
    if instance_index >= len(instances):
        return None, []
    instance_tuple = instances[instance_index]
    instance_data = instance_tuple_to_data(instance_tuple)
    coordinates = [instance_tuple[0][0]] + instance_tuple[1]
    demands = {i + 1: float(instance_tuple[2][i]) for i in range(len(instance_tuple[2]))}
    vehicle_capacity = float(instance_tuple[3])

    optimum = rng.choice(candidate_optima)
    all_perturb_records = []
    result = collect_30_30_samples(
        optimum,
        instance_path,
        instance_index,
        coordinates,
        demands,
        vehicle_capacity,
        operator_type=operator_type,
        num_local_search_runs=num_local_search_runs,
        max_runs=max_runs,
        rng=rng,
        all_perturb_records=all_perturb_records,
    )
    if result is None:
        return None, all_perturb_records

    anchor_data, positive_list, negative_list = result
    val_record = {
        "instance_index": instance_index,
        "instance_data": instance_data,
        "anchor": anchor_data,
        "positive_samples": positive_list,
        "negative_samples": negative_list,
    }
    return val_record, all_perturb_records


def main():
    parser = argparse.ArgumentParser(description="Build val_data.jsonl from instances 50-79")
    parser.add_argument(
        "--instance_path",
        type=str,
        default="/home/jieyi/cvrp100_uniform.pkl",
        help="Path to instance pkl",
    )
    parser.add_argument(
        "--basin_base_dir",
        type=str,
        default="basin_datasets0",
        help="Basin data directory",
    )
    parser.add_argument(
        "--operator_type",
        type=str,
        default="remove_and_insert",
        choices=["double_bridge", "remove_and_insert"],
    )
    parser.add_argument("--num_local_search_runs", type=int, default=30)
    parser.add_argument("--max_runs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="val_data.jsonl")
    parser.add_argument(
        "--out_all_perturb",
        type=str,
        default="val_data_all_perturb_record.jsonl",
        help="JSONL file to append every perturbation run: edges_hash, solution_flat, cost, return_prob, return_count",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    pool = list(range(50, 80))
    chosen = rng.choice(pool, size=10, replace=False).tolist()

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", out_path)
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    all_perturb_path = args.out_all_perturb
    if not os.path.isabs(all_perturb_path):
        all_perturb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", all_perturb_path)
    all_perturb_path = os.path.abspath(all_perturb_path)

    print(f"Instances to try: {sorted(chosen)}", flush=True)
    print(f"val_data.jsonl -> {out_path}", flush=True)
    print(f"val_data_all_perturb_record.jsonl -> {all_perturb_path}", flush=True)
    print("", flush=True)

    records = []
    tried_indices = set()
    candidate_list = list(chosen)
    pos = 0
    with open(all_perturb_path, "w", encoding="utf-8") as f_perturb:
        while len(records) < 10:
            if pos < len(candidate_list):
                instance_index = int(candidate_list[pos])
                pos += 1
            else:
                remaining = [i for i in pool if i not in tried_indices]
                if not remaining:
                    break
                instance_index = int(rng.choice(remaining))
            tried_indices.add(instance_index)
            print(f"[{len(records)+1}/10] Instance {instance_index} ...", flush=True)
            rec, perturb_list = build_one_val_record(
                instance_index,
                args.instance_path,
                args.basin_base_dir,
                args.operator_type,
                args.num_local_search_runs,
                args.max_runs,
                rng,
            )
            if rec is not None:
                records.append(rec)
                print(f"  ok anchor={rec['anchor']['edges_hash'][:12]}... ({len(perturb_list)} perturb records)", flush=True)
            else:
                print("  skip (no candidate anchor or collection failed)", flush=True)
            for row in perturb_list:
                f_perturb.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {out_path}", flush=True)
    print(f"Wrote all perturbation records (P, count, hash, solution_flat, cost) to {all_perturb_path}", flush=True)


if __name__ == "__main__":
    main()
