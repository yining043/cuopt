#!/usr/bin/env python3
"""
Generate Excel file with statistics for each instance:
- Number of runs (unique run_id in trials.jsonl)
- Number of trials (lines in trials.jsonl), successful_trials (optimum_id not None), failed_trials (optimum_id None)
- Number of intermediate solutions (lines in trajectory.jsonl)
- Also includes statistics for training_data.jsonl if present
- K1 collect: successful_local_optima / failed_local_optima from k1_collection_summary_*.jsonl (no trials.jsonl)
- Perturb (run_perturb.sh): lines in {operator}_training_data.ALL_r{N}.jsonl or batch_* files
"""

import argparse
import glob
import json
import os
import pandas as pd
from pathlib import Path


def analyze_trials_file(trials_path: str) -> dict:
    """
    Analyze trials.jsonl file in one pass.
    Returns dict with: runs, complete_runs, incomplete_runs, trials, successful_trials, failed_trials, run_trials.
    run_trials = run_id -> set of trial_ids (for comparing with perturb/k1 coverage).
    """
    if not os.path.isfile(trials_path):
        return {"runs": 0, "complete_runs": 0, "incomplete_runs": 0, "trials": 0, "successful_trials": 0, "failed_trials": 0, "run_trials": {}}
    
    seen_run_ids = set()
    run_trials = {}  # run_id -> set of trial_ids
    trial_count = 0
    successful_trials = 0
    failed_trials = 0
    
    # Single pass through the file
    with open(trials_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rid = rec.get("run_id")
                tid = rec.get("trial_id")
                opt_id = rec.get("optimum_id")
                
                if rid is not None:
                    seen_run_ids.add(rid)
                    if rid not in run_trials:
                        run_trials[rid] = set()
                    if tid is not None:
                        run_trials[rid].add(tid)
                trial_count += 1
                if opt_id is not None:
                    successful_trials += 1
                else:
                    failed_trials += 1
            except Exception:
                pass
    
    # Check completeness
    complete_runs = 0
    incomplete_runs = 0
    
    for run_id, trial_ids in run_trials.items():
        if not trial_ids:
            incomplete_runs += 1
            continue
        
        # Check if trial_ids are consecutive starting from 0
        sorted_trials = sorted(trial_ids)
        max_trial = sorted_trials[-1]
        
        # Check if we have consecutive trials from 0 to max_trial
        if sorted_trials[0] == 0 and len(trial_ids) == max_trial + 1:
            # Verify they are consecutive
            if all(i in trial_ids for i in range(max_trial + 1)):
                complete_runs += 1
            else:
                incomplete_runs += 1
        else:
            incomplete_runs += 1
    
    return {
        "runs": len(seen_run_ids),
        "complete_runs": complete_runs,
        "incomplete_runs": incomplete_runs,
        "trials": trial_count,
        "successful_trials": successful_trials,
        "failed_trials": failed_trials,
        "run_trials": run_trials,
    }


def count_lines(file_path: str) -> int:
    """Count non-empty lines in a file."""
    if not os.path.isfile(file_path):
        return 0
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def analyze_k1_collect_dir(
    instance_id: str,
    k1_collect_base_dir: str,
    operator_type: str = "remove_and_insert",
    runs: int = 30,
) -> dict:
    """
    Analyze perturb_k1_collect output for one instance.
    Reads only k1_collection_summary_*.jsonl (no trials.jsonl).
    Returns: k1_total_local_optima, k1_completed_local_optima, k1_incomplete_local_optima,
             k1_successful_local_optima, k1_failed_local_optima (local-optima level);
             k1_runs_with_data, k1_complete_runs, k1_incomplete_runs (run level, like training_data).
    Run: complete = run has data and no record has error=True; incomplete = has data but some error.
    """
    out_dir = os.path.join(k1_collect_base_dir, instance_id)
    if not os.path.isdir(out_dir):
        return {
            "k1_total_local_optima": 0,
            "k1_completed_local_optima": 0,
            "k1_incomplete_local_optima": 0,
            "k1_runs_with_data": 0,
            "k1_complete_runs": 0,
            "k1_incomplete_runs": 0,
            "k1_successful_local_optima": 0,
            "k1_failed_local_optima": 0,
        }
    pattern = os.path.join(out_dir, f"k1_collection_summary_{operator_type}.*_r{runs}.jsonl")
    paths = sorted(glob.glob(pattern))
    by_anchor = {}
    run_has_error = {}  # run_id -> True if any record has error=True
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    h = rec.get("anchor_hash")
                    if not h:
                        continue
                    run_id = rec.get("run_id")
                    if run_id is not None:
                        if run_id not in run_has_error:
                            run_has_error[run_id] = False
                        if rec.get("error") is True:
                            run_has_error[run_id] = True
                    if h not in by_anchor:
                        by_anchor[h] = {"completed": False, "success": False}
                    if rec.get("error") is not True:
                        by_anchor[h]["completed"] = True
                    if rec.get("success") is True:
                        by_anchor[h]["success"] = True
        except Exception:
            continue
    total = len(by_anchor)
    completed = sum(1 for v in by_anchor.values() if v["completed"])
    incomplete_lo = total - completed
    successful_local_optima = sum(1 for v in by_anchor.values() if v["success"])
    failed_local_optima = completed - successful_local_optima
    runs_with_data = len(run_has_error)
    k1_complete_runs = sum(1 for v in run_has_error.values() if not v)
    k1_incomplete_runs = runs_with_data - k1_complete_runs
    return {
        "k1_total_local_optima": total,
        "k1_completed_local_optima": completed,
        "k1_incomplete_local_optima": incomplete_lo,
        "k1_runs_with_data": runs_with_data,
        "k1_complete_runs": k1_complete_runs,
        "k1_incomplete_runs": k1_incomplete_runs,
        "k1_successful_local_optima": successful_local_optima,
        "k1_failed_local_optima": failed_local_optima,
    }


def analyze_training_data_file(training_data_path: str) -> dict:
    """
    Analyze training_data.jsonl file in one pass.
    Returns dict with: runs, complete_runs, incomplete_runs, trials, intermediate_solutions
    """
    if not os.path.isfile(training_data_path):
        return {
            "runs": 0,
            "complete_runs": 0,
            "incomplete_runs": 0,
            "trials": 0,
            "intermediate_solutions": 0
        }
    
    seen_run_ids = set()
    run_trials = {}  # run_id -> set of trial_ids
    total_lines = 0
    
    # Single pass through the file
    with open(training_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                total_lines += 1
                rid = rec.get("run_id")
                tid = rec.get("trial_id")
                
                if rid is not None:
                    seen_run_ids.add(rid)
                    if rid not in run_trials:
                        run_trials[rid] = set()
                    if tid is not None:
                        run_trials[rid].add(tid)
            except Exception:
                pass
    
    # Check completeness
    complete_runs = 0
    incomplete_runs = 0
    
    for run_id, trial_ids in run_trials.items():
        if not trial_ids:
            incomplete_runs += 1
            continue
        
        # Check if trial_ids are consecutive starting from 0
        sorted_trials = sorted(trial_ids)
        max_trial = sorted_trials[-1]
        
        # Check if we have consecutive trials from 0 to max_trial
        if sorted_trials[0] == 0 and len(trial_ids) == max_trial + 1:
            # Verify they are consecutive
            if all(i in trial_ids for i in range(max_trial + 1)):
                complete_runs += 1
            else:
                incomplete_runs += 1
        else:
            incomplete_runs += 1
    
    return {
        "runs": len(seen_run_ids),
        "complete_runs": complete_runs,
        "incomplete_runs": incomplete_runs,
        "trials": sum(len(trials) for trials in run_trials.values()),
        "intermediate_solutions": total_lines
    }


def analyze_perturb_output(
    instance_dir: str,
    operator_type: str = "remove_and_insert",
    runs: int = 30,
    expected_run_trials: dict = None,
) -> dict:
    """
    Analyze perturb (run_perturb.sh) output for one instance.
    Looks for {operator_type}_training_data.ALL_r{runs}.jsonl or batch_*_r{runs}.jsonl.
    Returns: perturb_records, perturb_unique_initial_optima;
             perturb_runs_with_data, perturb_complete_runs, perturb_incomplete_runs (like training_data).
    If expected_run_trials (run_id -> set of trial_ids from basin trials.jsonl) is given,
    complete = run has data and covers all expected trials; incomplete = has data but missing some.
    """
    if not os.path.isdir(instance_dir):
        return {
            "perturb_records": 0,
            "perturb_unique_initial_optima": 0,
            "perturb_runs_with_data": 0,
            "perturb_complete_runs": 0,
            "perturb_incomplete_runs": 0,
        }
    all_jsonl = os.path.join(instance_dir, f"{operator_type}_training_data.ALL_r{runs}.jsonl")
    batch_pattern = os.path.join(instance_dir, f"{operator_type}_training_data.batch_*_r{runs}.jsonl")
    paths = []
    if os.path.isfile(all_jsonl):
        paths = [all_jsonl]
    else:
        paths = sorted(glob.glob(batch_pattern))
    if not paths:
        return {
            "perturb_records": 0,
            "perturb_unique_initial_optima": 0,
            "perturb_runs_with_data": 0,
            "perturb_complete_runs": 0,
            "perturb_incomplete_runs": 0,
        }
    total_lines = 0
    unique_edges_hashes = set()
    run_trial_ids = {}  # run_id -> set of trial_ids seen in perturb data
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        total_lines += 1
                        init = rec.get("initial_solution") or {}
                        h = init.get("edges_hash")
                        if h is not None:
                            unique_edges_hashes.add(h)
                        rid = rec.get("run_id")
                        tid = rec.get("trial_id")
                        if rid is not None:
                            if rid not in run_trial_ids:
                                run_trial_ids[rid] = set()
                            if tid is not None:
                                run_trial_ids[rid].add(tid)
                    except Exception:
                        pass
        except Exception:
            continue
    runs_with_data = len(run_trial_ids)
    complete_runs = 0
    incomplete_runs = 0
    if expected_run_trials:
        for rid, perturb_trials in run_trial_ids.items():
            expected = expected_run_trials.get(rid)
            if expected is None:
                incomplete_runs += 1
                continue
            if perturb_trials >= expected:
                complete_runs += 1
            else:
                incomplete_runs += 1
    else:
        complete_runs = 0
        incomplete_runs = runs_with_data
    return {
        "perturb_records": total_lines,
        "perturb_unique_initial_optima": len(unique_edges_hashes),
        "perturb_runs_with_data": runs_with_data,
        "perturb_complete_runs": complete_runs,
        "perturb_incomplete_runs": incomplete_runs,
    }


def compute_instance_row(i, args, data_dir, training_data_dir, instance_basename, k1_collect_dir=None, k1_runs=30, perturb_operator_type="remove_and_insert", perturb_runs=30):
    """Compute statistics for a single instance. Returns (main_row_dict, training_row_dict or None)."""
    instance_id = f"{instance_basename}#{i}"
    instance_dir = os.path.join(data_dir, instance_id)
    trials_path = os.path.join(instance_dir, "trials.jsonl")
    trajectory_path = os.path.join(instance_dir, "trajectory.jsonl")

    trials_stats = analyze_trials_file(trials_path)
    runs = trials_stats["runs"]
    complete_runs = trials_stats["complete_runs"]
    incomplete_runs = trials_stats["incomplete_runs"]
    trials = trials_stats["trials"]
    successful_trials = trials_stats["successful_trials"]
    failed_trials = trials_stats["failed_trials"]
    run_trials = trials_stats.get("run_trials") or {}
    intermediate_solutions = count_lines(trajectory_path)

    training_data_runs = 0
    training_data_complete_runs = 0
    training_data_incomplete_runs = 0
    training_data_trials = 0
    training_data_intermediate = 0
    training_data_path = None
    training_row = None

    if training_data_dir and os.path.isdir(training_data_dir):
        training_data_path = os.path.join(training_data_dir, instance_id, "training_data.jsonl")
        if os.path.isfile(training_data_path):
            training_stats = analyze_training_data_file(training_data_path)
            training_data_runs = training_stats["runs"]
            training_data_complete_runs = training_stats["complete_runs"]
            training_data_incomplete_runs = training_stats["incomplete_runs"]
            training_data_trials = training_stats["trials"]
            training_data_intermediate = training_stats["intermediate_solutions"]
            training_row = {
                "instance_index": i,
                "instance_id": instance_id,
                "file_path": training_data_path,
                "runs": training_data_runs,
                "complete_runs": training_data_complete_runs,
                "incomplete_runs": training_data_incomplete_runs,
                "trials": training_data_trials,
                "intermediate_solutions": training_data_intermediate,
            }

    k1_total = 0
    k1_completed = 0
    k1_incomplete = 0
    k1_runs_with_data = 0
    k1_complete_runs = 0
    k1_incomplete_runs = 0
    k1_successful_local_optima = 0
    k1_failed_local_optima = 0
    if k1_collect_dir:
        k1_stats = analyze_k1_collect_dir(
            instance_id, k1_collect_dir, operator_type="remove_and_insert", runs=k1_runs
        )
        k1_total = k1_stats["k1_total_local_optima"]
        k1_completed = k1_stats["k1_completed_local_optima"]
        k1_incomplete = k1_stats["k1_incomplete_local_optima"]
        k1_runs_with_data = k1_stats["k1_runs_with_data"]
        k1_complete_runs = k1_stats["k1_complete_runs"]
        k1_incomplete_runs = k1_stats["k1_incomplete_runs"]
        k1_successful_local_optima = k1_stats["k1_successful_local_optima"]
        k1_failed_local_optima = k1_stats["k1_failed_local_optima"]

    perturb_records = 0
    perturb_unique_initial_optima = 0
    perturb_runs_with_data = 0
    perturb_complete_runs = 0
    perturb_incomplete_runs = 0
    if training_data_dir:
        perturb_dir = os.path.join(training_data_dir, instance_id)
        perturb_stats = analyze_perturb_output(
            perturb_dir,
            operator_type=perturb_operator_type,
            runs=perturb_runs,
            expected_run_trials=run_trials,
        )
        perturb_records = perturb_stats["perturb_records"]
        perturb_unique_initial_optima = perturb_stats["perturb_unique_initial_optima"]
        perturb_runs_with_data = perturb_stats["perturb_runs_with_data"]
        perturb_complete_runs = perturb_stats["perturb_complete_runs"]
        perturb_incomplete_runs = perturb_stats["perturb_incomplete_runs"]

    main_row = {
        "instance_index": i,
        "instance_id": instance_id,
        "runs": runs,
        "complete_runs": complete_runs,
        "incomplete_runs": incomplete_runs,
        "trials": trials,
        "successful_trials": successful_trials,
        "failed_trials": failed_trials,
        "intermediate_solutions": intermediate_solutions,
        "training_data_runs": training_data_runs,
        "training_data_complete_runs": training_data_complete_runs,
        "training_data_incomplete_runs": training_data_incomplete_runs,
        "training_data_trials": training_data_trials,
        "training_data_intermediate_solutions": training_data_intermediate,
        "perturb_records": perturb_records,
        "perturb_unique_initial_optima": perturb_unique_initial_optima,
        "perturb_runs_with_data": perturb_runs_with_data,
        "perturb_complete_runs": perturb_complete_runs,
        "perturb_incomplete_runs": perturb_incomplete_runs,
        "k1_total_local_optima": k1_total,
        "k1_completed_local_optima": k1_completed,
        "k1_incomplete_local_optima": k1_incomplete,
        "k1_runs_with_data": k1_runs_with_data,
        "k1_complete_runs": k1_complete_runs,
        "k1_incomplete_runs": k1_incomplete_runs,
        "k1_successful_local_optima": k1_successful_local_optima,
        "k1_failed_local_optima": k1_failed_local_optima,
    }
    return main_row, training_row


def main():
    parser = argparse.ArgumentParser(description="Generate Excel with basin statistics")
    parser.add_argument("--data_dir", type=str, default="basin_datasets0", help="Root directory (e.g. basin_datasets0)")
    parser.add_argument("--instance_basename", type=str, default="cvrp100_uniform.pkl", help="Instance basename")
    parser.add_argument("--start", type=int, default=0, help="Start instance index (inclusive)")
    parser.add_argument("--end", type=int, default=100, help="End instance index (exclusive)")
    parser.add_argument("--output", type=str, default="basin_statistics.xlsx", help="Output Excel file path")
    parser.add_argument("--training_data_dir", type=str, default="basin_datasets0_analyze", help="Directory for training_data.jsonl")
    parser.add_argument("--update_instances", type=int, nargs="+", default=None, help="Recompute only these instance indices and replace rows in existing Excel")
    parser.add_argument("--k1_collect_dir", type=str, default="perturb_k1_collect", help="Base directory for perturb_k1_collect output")
    parser.add_argument("--k1_runs", type=int, default=30, help="Number of local search runs used in k1 collection (for file pattern)")
    parser.add_argument("--perturb_operator_type", type=str, default="remove_and_insert", help="Operator type for run_perturb.sh output (file pattern)")
    parser.add_argument("--perturb_runs", type=int, default=30, help="Runs value for run_perturb.sh output (file pattern _r{N})")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    training_data_dir = os.path.abspath(args.training_data_dir) if args.training_data_dir else None
    k1_collect_dir = os.path.abspath(args.k1_collect_dir) if args.k1_collect_dir else None
    instance_basename = args.instance_basename

    if args.update_instances is not None:
        # Update only specified instances in existing Excel
        indices = sorted(set(args.update_instances))
        output_path = os.path.abspath(args.output)
        if not os.path.isfile(output_path):
            print(f"Error: {output_path} not found. Run full generation first.")
            return
        print(f"Loading {output_path} and recomputing instances {indices}...")
        df_main = pd.read_excel(output_path, sheet_name="Instance Statistics")
        try:
            df_training = pd.read_excel(output_path, sheet_name="Training Data Stats")
        except Exception:
            df_training = None

        for i in indices:
            main_row, training_row = compute_instance_row(
                i, args, data_dir, training_data_dir, instance_basename,
                k1_collect_dir=k1_collect_dir, k1_runs=args.k1_runs,
                perturb_operator_type=args.perturb_operator_type, perturb_runs=args.perturb_runs,
            )
            # Replace row in df_main (match by instance_index)
            loc = df_main["instance_index"] == i
            if loc.any():
                for k, v in main_row.items():
                    df_main.loc[loc, k] = v
                print(f"  Updated instance {i}: runs={main_row['runs']}, complete={main_row['complete_runs']}, incomplete={main_row['incomplete_runs']}")
            else:
                df_main = pd.concat([df_main, pd.DataFrame([main_row])], ignore_index=True)
                df_main = df_main.sort_values("instance_index").reset_index(drop=True)
            if df_training is not None and training_row is not None:
                tloc = df_training["instance_index"] == i
                if tloc.any():
                    for k, v in training_row.items():
                        df_training.loc[tloc, k] = v
                else:
                    df_training = pd.concat([df_training, pd.DataFrame([training_row])], ignore_index=True)

        # Ensure columns exist for backward compatibility
        for col in ("successful_trials", "failed_trials", "perturb_records", "perturb_unique_initial_optima", "perturb_runs_with_data", "perturb_complete_runs", "perturb_incomplete_runs", "k1_total_local_optima", "k1_completed_local_optima", "k1_incomplete_local_optima", "k1_runs_with_data", "k1_complete_runs", "k1_incomplete_runs", "k1_successful_local_optima", "k1_failed_local_optima"):
            if col not in df_main.columns:
                df_main[col] = 0
        # Rebuild summary row
        summary_row = {
            "instance_index": "TOTAL",
            "instance_id": "SUMMARY",
            "runs": df_main["runs"].sum(),
            "complete_runs": df_main["complete_runs"].sum(),
            "incomplete_runs": df_main["incomplete_runs"].sum(),
            "trials": df_main["trials"].sum(),
            "successful_trials": df_main["successful_trials"].sum(),
            "failed_trials": df_main["failed_trials"].sum(),
            "intermediate_solutions": df_main["intermediate_solutions"].sum(),
            "training_data_runs": df_main["training_data_runs"].sum(),
            "training_data_complete_runs": df_main["training_data_complete_runs"].sum(),
            "training_data_incomplete_runs": df_main["training_data_incomplete_runs"].sum(),
            "training_data_trials": df_main["training_data_trials"].sum(),
            "training_data_intermediate_solutions": df_main["training_data_intermediate_solutions"].sum(),
            "perturb_records": df_main["perturb_records"].sum(),
            "perturb_unique_initial_optima": df_main["perturb_unique_initial_optima"].sum(),
            "perturb_runs_with_data": df_main["perturb_runs_with_data"].sum(),
            "perturb_complete_runs": df_main["perturb_complete_runs"].sum(),
            "perturb_incomplete_runs": df_main["perturb_incomplete_runs"].sum(),
            "k1_total_local_optima": df_main["k1_total_local_optima"].sum(),
            "k1_completed_local_optima": df_main["k1_completed_local_optima"].sum(),
            "k1_incomplete_local_optima": df_main["k1_incomplete_local_optima"].sum(),
            "k1_runs_with_data": df_main["k1_runs_with_data"].sum(),
            "k1_complete_runs": df_main["k1_complete_runs"].sum(),
            "k1_incomplete_runs": df_main["k1_incomplete_runs"].sum(),
            "k1_successful_local_optima": df_main["k1_successful_local_optima"].sum(),
            "k1_failed_local_optima": df_main["k1_failed_local_optima"].sum(),
        }
        df_summary = pd.DataFrame([summary_row])
        if df_training is not None:
            training_summary = {
                "instance_index": "TOTAL",
                "instance_id": "SUMMARY",
                "file_path": "",
                "runs": df_training["runs"].sum(),
                "complete_runs": df_training["complete_runs"].sum(),
                "incomplete_runs": df_training["incomplete_runs"].sum(),
                "trials": df_training["trials"].sum(),
                "intermediate_solutions": df_training["intermediate_solutions"].sum(),
            }
            df_training_summary = pd.DataFrame([training_summary])

        print(f"\nWriting to {output_path}...")
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_main.to_excel(writer, sheet_name="Instance Statistics", index=False)
            df_summary.to_excel(writer, sheet_name="Summary", index=False)
            if df_training is not None:
                df_training.to_excel(writer, sheet_name="Training Data Stats", index=False)
                df_training_summary.to_excel(writer, sheet_name="Training Data Summary", index=False)
        print(f"Excel file saved. Summary: runs={summary_row['runs']}, complete={summary_row['complete_runs']}, incomplete={summary_row['incomplete_runs']}, trials: successful={summary_row['successful_trials']}, failed={summary_row['failed_trials']}")
        return

    rows = []
    training_data_rows = []
    
    print(f"Processing instances {args.start} to {args.end-1} in {data_dir}...")
    
    for i in range(args.start, args.end):
        main_row, training_row = compute_instance_row(
            i, args, data_dir, training_data_dir, instance_basename,
            k1_collect_dir=k1_collect_dir, k1_runs=args.k1_runs,
            perturb_operator_type=args.perturb_operator_type, perturb_runs=args.perturb_runs,
        )
        rows.append(main_row)
        if training_row is not None:
            training_data_rows.append(training_row)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{args.end - args.start} instances...")

    # Create DataFrame for main statistics
    df_main = pd.DataFrame(rows)
    
    # Print summary of training_data files found
    if training_data_rows:
        print(f"\nFound training_data.jsonl for {len(training_data_rows)} instances")

    # Create summary row
    summary_row = {
        "instance_index": "TOTAL",
        "instance_id": "SUMMARY",
        "runs": df_main["runs"].sum(),
        "complete_runs": df_main["complete_runs"].sum(),
        "incomplete_runs": df_main["incomplete_runs"].sum(),
        "trials": df_main["trials"].sum(),
        "successful_trials": df_main["successful_trials"].sum(),
        "failed_trials": df_main["failed_trials"].sum(),
        "intermediate_solutions": df_main["intermediate_solutions"].sum(),
        "training_data_runs": df_main["training_data_runs"].sum(),
        "training_data_complete_runs": df_main["training_data_complete_runs"].sum(),
        "training_data_incomplete_runs": df_main["training_data_incomplete_runs"].sum(),
        "training_data_trials": df_main["training_data_trials"].sum(),
        "training_data_intermediate_solutions": df_main["training_data_intermediate_solutions"].sum(),
        "perturb_records": df_main["perturb_records"].sum(),
        "perturb_unique_initial_optima": df_main["perturb_unique_initial_optima"].sum(),
        "perturb_runs_with_data": df_main["perturb_runs_with_data"].sum(),
        "perturb_complete_runs": df_main["perturb_complete_runs"].sum(),
        "perturb_incomplete_runs": df_main["perturb_incomplete_runs"].sum(),
        "k1_total_local_optima": df_main["k1_total_local_optima"].sum(),
        "k1_completed_local_optima": df_main["k1_completed_local_optima"].sum(),
        "k1_incomplete_local_optima": df_main["k1_incomplete_local_optima"].sum(),
        "k1_runs_with_data": df_main["k1_runs_with_data"].sum(),
        "k1_complete_runs": df_main["k1_complete_runs"].sum(),
        "k1_incomplete_runs": df_main["k1_incomplete_runs"].sum(),
        "k1_successful_local_optima": df_main["k1_successful_local_optima"].sum(),
        "k1_failed_local_optima": df_main["k1_failed_local_optima"].sum(),
    }
    df_summary = pd.DataFrame([summary_row])
    
    # Create DataFrame for training_data statistics
    df_training = None
    if training_data_rows:
        df_training = pd.DataFrame(training_data_rows)
        training_summary = {
            "instance_index": "TOTAL",
            "instance_id": "SUMMARY",
            "file_path": "",
            "runs": df_training["runs"].sum(),
            "complete_runs": df_training["complete_runs"].sum(),
            "incomplete_runs": df_training["incomplete_runs"].sum(),
            "trials": df_training["trials"].sum(),
            "intermediate_solutions": df_training["intermediate_solutions"].sum(),
        }
        df_training_summary = pd.DataFrame([training_summary])
    
    # Write to Excel with multiple sheets
    output_path = args.output
    print(f"\nWriting to {output_path}...")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_main.to_excel(writer, sheet_name='Instance Statistics', index=False)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        if df_training is not None:
            df_training.to_excel(writer, sheet_name='Training Data Stats', index=False)
            df_training_summary.to_excel(writer, sheet_name='Training Data Summary', index=False)
    
    print(f"Excel file saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Total instances processed: {len(df_main)}")
    print(f"  Total runs: {summary_row['runs']}")
    print(f"  Complete runs: {summary_row['complete_runs']}")
    print(f"  Incomplete runs: {summary_row['incomplete_runs']}")
    print(f"  Total trials: {summary_row['trials']} (successful: {summary_row['successful_trials']}, failed: {summary_row['failed_trials']})")
    print(f"  Total intermediate solutions: {summary_row['intermediate_solutions']}")
    print(f"  Perturb (run_perturb.sh) - records: {summary_row['perturb_records']}, unique initial optima: {summary_row['perturb_unique_initial_optima']}")
    print(f"    runs with data: {summary_row['perturb_runs_with_data']}, complete runs: {summary_row['perturb_complete_runs']}, incomplete runs: {summary_row['perturb_incomplete_runs']}")
    print(f"  K1 collect - total local optima: {summary_row['k1_total_local_optima']}, completed: {summary_row['k1_completed_local_optima']}, incomplete: {summary_row['k1_incomplete_local_optima']}")
    print(f"    runs with data: {summary_row['k1_runs_with_data']}, complete runs: {summary_row['k1_complete_runs']}, incomplete runs: {summary_row['k1_incomplete_runs']}")
    print(f"    successful local optima: {summary_row['k1_successful_local_optima']}, failed local optima: {summary_row['k1_failed_local_optima']}")
    if df_training is not None:
        print(f"\nTraining data (training_data.jsonl) - {len(training_data_rows)} instances:")
        print(f"  Total runs: {training_summary['runs']}")
        print(f"  Complete runs: {training_summary['complete_runs']}")
        print(f"  Incomplete runs: {training_summary['incomplete_runs']}")
        print(f"  Total trials: {training_summary['trials']}")
        print(f"  Total intermediate solutions: {training_summary['intermediate_solutions']}")


if __name__ == "__main__":
    main()
