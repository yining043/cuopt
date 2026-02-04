#!/usr/bin/env python3
"""
Count runs per instance in a basin_datasets directory (number of unique run_id in trials.jsonl)
and list instances below target with how many runs to add.

Usage:
  python count_basin_runs.py --data_dir basin_datasets0
  python count_basin_runs.py --data_dir basin_datasets0 --export-todo   # print instance_index n_runs_to_add for those needing more
"""

import argparse
import json
import os


def count_runs(trials_path: str) -> int:
    """Return the number of unique run_id in trials.jsonl. Returns 0 if file is missing or empty."""
    if not os.path.isfile(trials_path):
        return 0
    seen = set()
    with open(trials_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rid = rec.get("run_id")
                if rid is not None:
                    seen.add(rid)
            except Exception:
                pass
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description="Count runs per instance in basin_datasets.")
    parser.add_argument("--data_dir", type=str, default="basin_datasets0", help="Root directory (e.g. basin_datasets0)")
    parser.add_argument("--instance_basename", type=str, default="cvrp100_uniform.pkl", help="Instance basename for instance_id")
    parser.add_argument("--start", type=int, default=0, help="Start instance index (inclusive)")
    parser.add_argument("--end", type=int, default=100, help="End instance index (exclusive)")
    parser.add_argument("--target", type=int, default=100, help="Target runs per instance")
    parser.add_argument("--export-todo", action="store_true", help="Print 'instance_index n_runs_to_add' for instances that need more runs")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"Data directory does not exist: {data_dir}")
        return

    rows = []
    need_more = []
    for i in range(args.start, args.end):
        instance_id = f"{args.instance_basename}#{i}"
        instance_dir = os.path.join(data_dir, instance_id)
        trials_path = os.path.join(instance_dir, "trials.jsonl")
        runs = count_runs(trials_path)
        to_add = max(0, args.target - runs) if runs < args.target else 0
        rows.append((i, runs, to_add))
        if to_add > 0:
            need_more.append((i, to_add))

    if args.export_todo:
        for i, n in need_more:
            print(f"{i} {n}")
        return

    # Print table
    print(f"data_dir={data_dir}  instance={args.instance_basename}  target={args.target}  range=[{args.start},{args.end})")
    print("-" * 60)
    for i, runs, to_add in rows:
        tag = "OK" if to_add == 0 else f"need +{to_add}"
        print(f"  instance {i:3d}: runs={runs:3d}  {tag}")
    print("-" * 60)
    ok = sum(1 for _, _, to_add in rows if to_add == 0)
    print(f"  Summary: {ok}/{len(rows)} instances have >={args.target} runs. {len(need_more)} need more.")
    if need_more:
        print(f"  To complete: run run_complete_basin_runs.sh (or use --export-todo to get the list).")


if __name__ == "__main__":
    main()
