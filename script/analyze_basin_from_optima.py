#!/usr/bin/env python3
"""
Summarize basins from optima.jsonl and write per-instance Excel reports.

- Deduplicate basins by `edges_hash`
- Count basin frequency (# of times the same basin appears in optima.jsonl)
- Report cost stats and HGS deltas
"""

import json
import pandas as pd
from collections import defaultdict
from pathlib import Path


def load_optima_from_jsonl(jsonl_path):
    """Load records from a JSONL file."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def analyze_basins_for_instance(instance_dir):
    """Return a basin-level DataFrame for a single instance directory."""
    optima = load_optima_from_jsonl(Path(instance_dir) / "optima.jsonl")

    agg = defaultdict(lambda: {"frequency": 0, "costs": [], "edge_diff_to_hgs": None, "cost_gap_to_hgs_pct": None,
                               "num_routes": None, "num_orders": None})

    for r in optima:
        h = r["edges_hash"]
        a = agg[h]
        a["frequency"] += 1
        a["costs"].append(r["final_cost"])
        a["edge_diff_to_hgs"] = a["edge_diff_to_hgs"] if a["edge_diff_to_hgs"] is not None else r.get("edge_diff_to_hgs")
        a["cost_gap_to_hgs_pct"] = a["cost_gap_to_hgs_pct"] if a["cost_gap_to_hgs_pct"] is not None else r.get("cost_gap_to_hgs_pct")
        a["num_routes"] = a["num_routes"] if a["num_routes"] is not None else r.get("num_routes")
        a["num_orders"] = a["num_orders"] if a["num_orders"] is not None else r.get("num_orders")

    rows = []
    for h, a in agg.items():
        costs = a["costs"]
        rows.append({
            "edges_hash": h,
            "frequency": a["frequency"],
            "mean_cost": float(sum(costs) / len(costs)),
            "min_cost": float(min(costs)),
            "max_cost": float(max(costs)),
            "edge_diff_to_hgs": a["edge_diff_to_hgs"],
            "cost_gap_to_hgs_pct": a["cost_gap_to_hgs_pct"],
            "num_routes": a["num_routes"],
            "num_orders": a["num_orders"],
        })

    return (pd.DataFrame(rows)
            .sort_values(["frequency", "mean_cost"], ascending=[False, True])
            .reset_index(drop=True))


def analyze_all_instances(basin_base_dir):
    basin_base_dir = Path(basin_base_dir)
    instance_dirs = sorted([d for d in basin_base_dir.iterdir() if d.is_dir() and "cvrp" in d.name.lower()])

    for d in instance_dirs:
        df = analyze_basins_for_instance(d)
        with pd.ExcelWriter(d / "basin_statistics.xlsx", engine="openpyxl") as w:
            df.to_excel(w, sheet_name="Basin Statistics", index=False)

    with pd.ExcelWriter(basin_base_dir / "all_basin_statistics.xlsx", engine="openpyxl") as w:
        for d in instance_dirs:
            df = analyze_basins_for_instance(d)
            df.to_excel(w, sheet_name=d.name[:31], index=False)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Summarize basins from optima.jsonl and write Excel reports.")
    parser.add_argument(
        "--basin_base_dir",
        type=str,
        default="basin_datasets0",
        help="Base directory containing per-instance folders (default: basin_datasets0)",
    )
    
    args = parser.parse_args()
    analyze_all_instances(args.basin_base_dir)


if __name__ == '__main__':
    main()
