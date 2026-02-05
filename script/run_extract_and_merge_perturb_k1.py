#!/usr/bin/env python3
"""
Extract first 10 runs' perturb data + stats per instance, then merge all stats xlsx.

- Single instance: --instance_dir DIR → perturb_data.jsonl + perturb_k1_stats.xlsx for that dir.
- Batch (default): --base_dir + --instances 0-49 → same per instance, then
  merged_perturb_k1_stats.xlsx in base_dir.

If an instance has only k1_collection_summary (no results), still writes perturb_k1_stats.xlsx
from summary + trajectory so it appears in the merged file.

Usage:
  # Batch: all instances 0-49, then merge
  python script/run_extract_and_merge_perturb_k1.py

  # Single instance
  python script/run_extract_and_merge_perturb_k1.py --instance_dir perturb_k1_collect/cvrp100_uniform.pkl#0
  python script/run_extract_and_merge_perturb_k1.py --instance_dir ... --trajectory_path basin_datasets0/.../trajectory.jsonl
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pandas as pd
except ImportError:
    pd = None

from perturb import load_trajectory_info


def get_first_n_run_ids_and_anchor_sets(trajectory_path, n=10):
    trajectory_info = load_trajectory_info(trajectory_path)
    run_to_anchors = defaultdict(set)
    run_id_order = []
    seen_runs = set()
    for edges_hash, occs in trajectory_info.items():
        for occ in occs:
            rid = occ.get("run_id")
            if rid is not None and rid not in seen_runs:
                seen_runs.add(rid)
                run_id_order.append(rid)
            if rid is not None:
                run_to_anchors[rid].add(edges_hash)
    first_n_run_ids = run_id_order[:n]
    return first_n_run_ids, dict(run_to_anchors), run_id_order


def run_stats_for_runs(summary_path, run_ids_set, run_to_expected_anchors):
    run_to_done = defaultdict(lambda: {"success": 0, "fail": 0, "anchors": set()})
    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rid = rec.get("run_id")
                if rid is None or rid not in run_ids_set:
                    continue
                ah = rec.get("anchor_hash")
                if ah:
                    run_to_done[rid]["anchors"].add(ah)
                    if rec.get("success") is True:
                        run_to_done[rid]["success"] += 1
                    else:
                        run_to_done[rid]["fail"] += 1
            except json.JSONDecodeError:
                continue

    total_optima_with_data = sum(len(r["anchors"]) for r in run_to_done.values())
    total_success_rows = sum(r["success"] for r in run_to_done.values())
    total_fail_rows = sum(r["fail"] for r in run_to_done.values())
    complete_runs = 0
    incomplete_runs = 0
    completed_success_runs = 0
    completed_fail_runs = 0
    for rid in run_ids_set:
        done = run_to_done.get(rid, {"anchors": set(), "success": 0, "fail": 0})
        expected = run_to_expected_anchors.get(rid, set())
        if not expected:
            continue
        if len(done["anchors"]) >= len(expected):
            complete_runs += 1
            if done["fail"] == 0:
                completed_success_runs += 1
            else:
                completed_fail_runs += 1
        else:
            incomplete_runs += 1

    return {
        "num_runs": len(run_ids_set),
        "num_local_optima_perturb_data": total_optima_with_data,
        "runs_complete": complete_runs,
        "runs_incomplete": incomplete_runs,
        "completed_runs_success": completed_success_runs,
        "completed_runs_fail": completed_fail_runs,
        "summary_rows_success": total_success_rows,
        "summary_rows_fail": total_fail_rows,
    }


def extract_one_instance(
    instance_dir,
    trajectory_path=None,
    operator="remove_and_insert",
    runs=30,
    first_n=10,
    out_jsonl="perturb_data.jsonl",
    out_xlsx="perturb_k1_stats.xlsx",
    quiet=False,
):
    """Run extract and stats for one instance. Returns (wrote_jsonl, wrote_xlsx, rows)."""
    instance_dir = os.path.abspath(instance_dir)
    if not os.path.isdir(instance_dir):
        if not quiet:
            print(f"Skip (not a dir): {instance_dir}")
        return False, False, []

    results_name = f"k1_collection_results_{operator}.ALL_r{runs}.jsonl"
    summary_name = f"k1_collection_summary_{operator}.ALL_r{runs}.jsonl"
    results_path = os.path.join(instance_dir, results_name)
    summary_path = os.path.join(instance_dir, summary_name)

    has_results = os.path.exists(results_path)
    if not has_results and not quiet:
        print(f"  Warning: no results file, will only write stats xlsx from summary")
    if not os.path.exists(summary_path):
        if not quiet:
            print(f"  Skip: summary not found {summary_path}")
        return False, False, []

    traj_path = trajectory_path or os.path.join(instance_dir, "trajectory.jsonl")
    first_n_run_ids = None
    run_to_anchors = {}
    all_run_ids_ordered = []

    if os.path.exists(traj_path):
        first_n_run_ids, run_to_anchors, all_run_ids_ordered = get_first_n_run_ids_and_anchor_sets(
            traj_path, n=first_n
        )
        anchor_hashes_first10 = set()
        for rid in first_n_run_ids:
            anchor_hashes_first10 |= run_to_anchors.get(rid, set())
    else:
        if not quiet:
            print(f"  Warning: trajectory not found, using run_id order from summary")
        run_to_anchors = defaultdict(set)
        all_run_ids_ordered = []
        seen_all = set()
        with open(summary_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    rid, ah = rec.get("run_id"), rec.get("anchor_hash")
                    if rid and ah:
                        if rid not in seen_all:
                            seen_all.add(rid)
                            all_run_ids_ordered.append(rid)
                        run_to_anchors[rid].add(ah)
                except json.JSONDecodeError:
                    continue
        run_to_anchors = dict(run_to_anchors)
        first_n_run_ids = all_run_ids_ordered[:first_n]
        anchor_hashes_first10 = set()
        for rid in first_n_run_ids:
            anchor_hashes_first10 |= run_to_anchors.get(rid, set())

    wrote_jsonl = False
    out_jsonl_path = os.path.join(instance_dir, out_jsonl)
    if has_results:
        written = 0
        with open(results_path, "r", encoding="utf-8") as fin, open(out_jsonl_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    anchor = rec.get("anchor") or {}
                    ah = anchor.get("edges_hash")
                    if ah and ah in anchor_hashes_first10:
                        fout.write(line + "\n")
                        written += 1
                except json.JSONDecodeError:
                    continue
        wrote_jsonl = True
        if not quiet:
            print(f"  Wrote {written} records to {out_jsonl}")
    else:
        with open(out_jsonl_path, "w", encoding="utf-8") as f:
            pass  # empty perturb_data.jsonl
        if not quiet:
            print(f"  Wrote 0 records to {out_jsonl} (no results file)")

    if all_run_ids_ordered:
        all_run_ids_set = set(all_run_ids_ordered)
    else:
        all_run_ids_set = set(run_to_anchors.keys())
    first10_set = set(first_n_run_ids) if first_n_run_ids else set()

    stats_all = run_stats_for_runs(summary_path, all_run_ids_set, run_to_anchors)
    stats_first10 = run_stats_for_runs(summary_path, first10_set, run_to_anchors) if first10_set else {}

    rows = [
        ("Metric", "All runs", "First 10 runs"),
        ("Number of runs", stats_all.get("num_runs", 0), stats_first10.get("num_runs", 0)),
        ("Local optima with perturb data", stats_all.get("num_local_optima_perturb_data", 0), stats_first10.get("num_local_optima_perturb_data", 0)),
        ("Runs completed (all local optima done)", stats_all.get("runs_complete", 0), stats_first10.get("runs_complete", 0)),
        ("Runs incomplete", stats_all.get("runs_incomplete", 0), stats_first10.get("runs_incomplete", 0)),
        ("Among completed: runs all success", stats_all.get("completed_runs_success", 0), stats_first10.get("completed_runs_success", 0)),
        ("Among completed: runs with at least one fail", stats_all.get("completed_runs_fail", 0), stats_first10.get("completed_runs_fail", 0)),
        ("Summary rows (success)", stats_all.get("summary_rows_success", 0), stats_first10.get("summary_rows_success", 0)),
        ("Summary rows (failure)", stats_all.get("summary_rows_fail", 0), stats_first10.get("summary_rows_fail", 0)),
    ]

    out_xlsx_path = os.path.join(instance_dir, out_xlsx)
    if pd is not None:
        df = pd.DataFrame(rows[1:], columns=rows[0])
        df.to_excel(out_xlsx_path, index=False, sheet_name="perturb_k1_stats")
    else:
        with open(out_xlsx_path.replace(".xlsx", ".csv"), "w", encoding="utf-8") as f:
            f.write(",".join(rows[0]) + "\n")
            for r in rows[1:]:
                f.write(",".join(str(x) for x in r) + "\n")
    if not quiet:
        print(f"  Wrote {out_xlsx}")
    return wrote_jsonl, True, rows


def merge_xlsx(base_dir, instance_id, indices, out_name):
    if pd is None:
        print("pandas required for merge")
        return
    all_runs_rows = []
    first10_runs_rows = []
    sheet_dfs = {}
    metric_order = None
    for i in indices:
        instance_dir = os.path.join(base_dir, f"{instance_id}#{i}")
        xlsx_path = os.path.join(instance_dir, "perturb_k1_stats.xlsx")
        if not os.path.exists(xlsx_path):
            continue
        try:
            df = pd.read_excel(xlsx_path, sheet_name=0)
        except Exception as e:
            print(f"  Skip merge {i}: {e}")
            continue
        sheet_dfs[str(i)] = df
        if metric_order is None:
            metric_order = [r.iloc[0] for _, r in df.iterrows() if not pd.isna(r.iloc[0])]
        row_all = {"instance": i}
        row_first10 = {"instance": i}
        for _, r in df.iterrows():
            metric = r.iloc[0]
            if pd.isna(metric):
                continue
            row_all[metric] = r.iloc[1]
            row_first10[metric] = r.iloc[2] if len(r) > 2 else None
        all_runs_rows.append(row_all)
        first10_runs_rows.append(row_first10)

    if not all_runs_rows:
        print("No xlsx files to merge.")
        return
    out_path = os.path.join(base_dir, out_name)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        cols_all = ["instance"] + metric_order
        pd.DataFrame(all_runs_rows, columns=cols_all).to_excel(w, index=False, sheet_name="All runs")
        pd.DataFrame(first10_runs_rows, columns=cols_all).to_excel(w, index=False, sheet_name="First 10 runs")
        for name, df in sorted(sheet_dfs.items(), key=lambda x: int(x[0])):
            df.to_excel(w, index=False, sheet_name=name)
    print(f"Merged {len(sheet_dfs)} instances -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract first 10 runs perturb data + stats; optional merge all xlsx")
    parser.add_argument("--instance_dir", type=str, default=None,
                        help="Single instance: run extract for this dir only (no merge)")
    parser.add_argument("--base_dir", type=str, default="perturb_k1_collect")
    parser.add_argument("--instance_id", type=str, default="cvrp100_uniform.pkl")
    parser.add_argument("--trajectory_path", type=str, default=None, help="For single-instance mode")
    parser.add_argument("--trajectory_base", type=str, default="basin_datasets0")
    parser.add_argument("--instances", type=str, default="0-49")
    parser.add_argument("--operator", type=str, default="remove_and_insert")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--first_n", type=int, default=10)
    parser.add_argument("--merged_out", type=str, default="merged_perturb_k1_stats.xlsx")
    args = parser.parse_args()

    if args.instance_dir is not None:
        # Single instance
        instance_dir = os.path.abspath(args.instance_dir)
        if not os.path.isdir(instance_dir):
            print(f"Error: instance_dir not found: {instance_dir}")
            sys.exit(1)
        summary_path = os.path.join(instance_dir, f"k1_collection_summary_{args.operator}.ALL_r{args.runs}.jsonl")
        if not os.path.exists(summary_path):
            print(f"Error: summary not found: {summary_path}")
            sys.exit(1)
        _, _, rows = extract_one_instance(
            instance_dir,
            trajectory_path=args.trajectory_path or os.path.join(instance_dir, "trajectory.jsonl"),
            operator=args.operator,
            runs=args.runs,
            first_n=args.first_n,
            quiet=False,
        )
        if rows:
            print("\nStats summary:")
            for r in rows[1:]:
                print(f"  {r[0]}: all={r[1]}, first10={r[2]}")
        return

    # Batch: run extract for each instance, then merge
    base_dir = os.path.abspath(args.base_dir)
    traj_base = os.path.abspath(args.trajectory_base)
    if "-" in args.instances:
        lo, hi = args.instances.split("-")
        indices = list(range(int(lo), int(hi) + 1))
    else:
        indices = [int(x) for x in args.instances.split(",")]

    for i in indices:
        instance_dir = os.path.join(base_dir, f"{args.instance_id}#{i}")
        trajectory_path = os.path.join(traj_base, f"{args.instance_id}#{i}", "trajectory.jsonl")
        print(f"=== Instance {i} ===")
        extract_one_instance(
            instance_dir,
            trajectory_path=trajectory_path if os.path.exists(trajectory_path) else None,
            operator=args.operator,
            runs=args.runs,
            first_n=args.first_n,
            out_jsonl="perturb_data.jsonl",
            out_xlsx="perturb_k1_stats.xlsx",
            quiet=False,
        )

    print("\n=== Merge xlsx ===")
    merge_xlsx(base_dir, args.instance_id, indices, args.merged_out)


if __name__ == "__main__":
    main()
