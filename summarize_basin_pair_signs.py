#!/usr/bin/env python3
"""
Summarize, for each basin, how many positive / negative / zero co-occurrence pairs it has.

Per instance:
- Build (or reuse) basin co-occurrence matrix using the same logic as visualize_basin_cooccurrence_matrix.py
- For each basin, count:
  - num_pos_pairs: number of neighbor basins with co-occurrence value > 0
  - num_neg_pairs: number of neighbor basins with co-occurrence value < 0
  - num_zero_pairs: number of neighbor basins with co-occurrence value == 0

Note: "zero pairs" here are pairs that, within the active basin set, have no positive
or negative co-occurrence record (i.e. value 0 in the co-occurrence matrix), or
cancel out to 0 (rare in practice).
"""

import os
import json
from collections import defaultdict

import pandas as pd

# Excel single-sheet row limit (openpyxl / xlsx)
EXCEL_MAX_ROWS = 1_048_576

import visualize_basin_cooccurrence_matrix as vcm


def parse_instance_range(instance_str: str):
    """Parse instance range string into list of indices."""
    indices = set()
    parts = instance_str.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-')
            start, end = int(start.strip()), int(end.strip())
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part.strip()))
    return sorted(indices)


def load_basin_costs(instance_index: int):
    """Load basin hash -> cost mapping for one instance (if basin_info.jsonl exists)."""
    info_file = f"/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#{instance_index}/basin_info.jsonl"
    costs = {}
    if not os.path.exists(info_file):
        return costs

    with open(info_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                h = rec.get("hash")
                c = rec.get("cost")
                if h is not None and c is not None:
                    costs[h] = c
            except Exception:
                continue
    return costs


def summarize_instance(instance_index: int,
                       min_count: int = 10,
                       max_runs: int = 10,
                       max_records: int = None,
                       max_basins_per_record: int = None):
    """Summarize basin pair signs for a single instance."""
    training_data_file = (
        f"/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#{instance_index}/training_data.jsonl"
    )
    if not os.path.exists(training_data_file):
        print(f"Instance {instance_index}: training_data.jsonl not found, skipping")
        return []

    print(f"\n{'=' * 60}")
    print(f"Summarizing Instance {instance_index}")
    print(f"{'=' * 60}")

    # For instance 0, process all records from first 10 runs (no limit on records)
    # Use default values if not specified
    if max_records is None:
        max_records = None  # Process all records from the specified runs
    if max_basins_per_record is None:
        max_basins_per_record = None  # No limit on basins per record
    
    coocc = vcm.build_cooccurrence_matrix_from_training_data_fast(
        training_data_file,
        min_count=min_count,
        max_runs=max_runs,
        max_records=max_records,
        max_basins_per_record=max_basins_per_record,
    )
    print(f"  Co-occurrence pairs: {len(coocc)}")

    # Collect all basins that appear in any pair
    all_basins = set()
    for (b1, b2) in coocc.keys():
        all_basins.add(b1)
        all_basins.add(b2)
    all_basins = sorted(all_basins)

    # Initialize per-basin stats
    stats = {b: {"pos": 0, "neg": 0} for b in all_basins}

    # Count positive / negative pairs for each basin
    for (b1, b2), val in coocc.items():
        if val > 0:
            stats[b1]["pos"] += 1
            stats[b2]["pos"] += 1
        elif val < 0:
            stats[b1]["neg"] += 1
            stats[b2]["neg"] += 1
        # val == 0 is not counted; this case is rare

    # Compute zero pairs: for each basin, all neighbors in the active set except itself
    n_basins = len(all_basins)

    # Load cost info if available
    basin_costs = load_basin_costs(instance_index)

    rows = []
    for b in all_basins:
        pos = stats[b]["pos"]
        neg = stats[b]["neg"]
        total_neighbors = n_basins - 1
        zero = max(total_neighbors - pos - neg, 0)

        rows.append(
            {
                "instance_index": instance_index,
                "basin_hash": b,
                "cost": basin_costs.get(b),
                "positive_pairs": pos,
                "negative_pairs": neg,
                "zero_pairs": zero,
                "total_neighbors": total_neighbors,
            }
        )

    return rows


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize, for each basin, how many positive / negative / zero co-occurrence pairs it has."
    )
    parser.add_argument(
        "--instances",
        type=str,
        default="0-10",
        help='Instance indices to process (e.g., "0-10", "0,1,2"). Default: 0-10',
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=10,
        help="Minimum basin count threshold in one record to be considered positive (default: 10)",
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=10,
        help="Maximum number of runs to process per instance (for performance, default: 10)",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Maximum number of records per instance to process (default: None = process all)",
    )
    parser.add_argument(
        "--max_basins_per_record",
        type=int,
        default=None,
        help="Maximum number of basins per record to process (default: None = no limit)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="basin_pair_signs_summary.xlsx",
        help="Output Excel file path (default: basin_pair_signs_summary.xlsx)",
    )

    args = parser.parse_args()

    instance_indices = parse_instance_range(args.instances)
    print(f"Processing instances: {instance_indices}")

    all_rows = []
    for idx in instance_indices:
        rows = summarize_instance(
            idx,
            min_count=args.min_count,
            max_runs=args.max_runs,
            max_records=args.max_records,
            max_basins_per_record=args.max_basins_per_record,
        )
        all_rows.extend(rows)

    if not all_rows:
        print("No data collected, nothing to write.")
        return

    df = pd.DataFrame(all_rows)

    output_path = os.path.abspath(args.output)
    base, _ = os.path.splitext(output_path)
    detail_path = base + "_by_instance.csv"

    # Append mode: load existing "By Instance" from CSV (preferred) or Excel
    if os.path.exists(detail_path):
        try:
            existing = pd.read_csv(detail_path)
            df = pd.concat([existing, df], ignore_index=True)
            print(f"Appended to existing CSV; total rows in 'By Instance': {len(df)}")
        except Exception as e:
            print(f"Warning: could not load existing CSV ({e}), overwriting.")
    elif os.path.exists(output_path):
        try:
            existing = pd.read_excel(output_path, sheet_name="By Instance", engine="openpyxl")
            if "Unnamed: 0" in existing.columns:
                existing = existing.drop(columns=["Unnamed: 0"])
            df = pd.concat([existing, df], ignore_index=True)
            print(f"Appended to existing Excel; total rows in 'By Instance': {len(df)}")
        except Exception as e:
            print(f"Warning: could not load existing Excel ({e}), overwriting.")
    else:
        print(f"\nWriting output: {output_path}")

    # Recompute summary from combined "By Instance" data
    df_summary = (
        df.groupby("basin_hash")
        .agg(
            {
                "positive_pairs": "sum",
                "negative_pairs": "sum",
                "zero_pairs": "sum",
                "total_neighbors": "sum",
            }
        )
        .reset_index()
    )

    n_rows = len(df)
    if n_rows > EXCEL_MAX_ROWS:
        # Detail exceeds Excel limit: write detail to CSV, summary only to Excel
        df.to_csv(detail_path, index=False)
        print(f"Wrote {n_rows} rows to {detail_path} (Excel limit is {EXCEL_MAX_ROWS})")
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_summary.to_excel(writer, sheet_name="By Basin (All Instances)", index=False)
        print(f"Wrote summary to {output_path}")
    else:
        # Both fit in Excel
        if os.path.exists(detail_path):
            try:
                os.remove(detail_path)
            except OSError:
                pass
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="By Instance", index=False)
            df_summary.to_excel(writer, sheet_name="By Basin (All Instances)", index=False)
        print(f"Wrote Excel to {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()

