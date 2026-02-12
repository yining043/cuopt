#!/usr/bin/env python3
"""
Build return-probability data JSONL for analyze_return_probability.py.

We join two sources:
 1) perturb records with empirical return probability for many solutions:
    - val_data_all_perturb_record.jsonl
      Each line contains (at least):
        - instance_index: int
        - anchor_hash: str
        - solution_flat: list[int]           # start solution
        - cost: float                        # start solution cost
        - return_prob: float in [0,1]
        - return_count: int                  # optional, used only for sanity check
 2) validation metadata with anchor solutions:
    - val_data.jsonl
      Each line contains:
        - instance_index: int
        - instance_data: { ... }             # depot / coords / demands / capacity
        - anchor: {
              edges_hash: str,               # anchor_hash
              solution_flat: list[int],
              cost: float
          }
        - positive_samples: [...]
        - negative_samples: [...]

Output format (one JSON object per line):
  {
    "instance_index": int,
    "anchor_solution_flat": [...],
    "start_solution_flat": [...],
    "return_prob": float,
    "anchor_cost": float,
    "start_cost": float
  }

This matches the expectations of analyze_return_probability.py.
"""

import argparse
import json
from typing import Dict, Tuple


def load_anchor_index(val_data_path: str) -> Dict[Tuple[int, str], dict]:
    """Build mapping (instance_index, anchor_hash) -> anchor dict with solution_flat and cost."""
    mapping: Dict[Tuple[int, str], dict] = {}
    with open(val_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            inst_idx = int(rec["instance_index"])
            anchor = rec["anchor"]
            anchor_hash = str(anchor["edges_hash"])
            mapping[(inst_idx, anchor_hash)] = anchor
    return mapping


def build_return_prob_data(
    perturb_path: str,
    val_data_path: str,
    output_path: str,
) -> None:
    """Join perturb records with anchor solutions and write unified JSONL."""
    anchor_index = load_anchor_index(val_data_path)
    missing = 0
    total = 0

    with open(perturb_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            inst_idx = int(rec["instance_index"])
            anchor_hash = str(rec["anchor_hash"])
            key = (inst_idx, anchor_hash)
            anchor = anchor_index.get(key)
            if anchor is None:
                # Skip records whose anchor is not present in val_data.jsonl
                missing += 1
                continue

            out = {
                "instance_index": inst_idx,
                "anchor_solution_flat": anchor["solution_flat"],
                "start_solution_flat": rec["solution_flat"],
                "return_prob": float(rec["return_prob"]),
                "anchor_cost": float(anchor.get("cost", 0.0)),
                "start_cost": float(rec.get("cost", 0.0)),
            }
            fout.write(json.dumps(out) + "\n")

    print(f"Processed {total} perturb records; wrote {total - missing} joined records to {output_path}.")
    if missing:
        print(f"Warning: {missing} records skipped because (instance_index, anchor_hash) not found in {val_data_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build return-probability JSONL from perturb records and val_data.jsonl.")
    parser.add_argument(
        "--perturb_records",
        type=str,
        required=True,
        help="Path to val_data_all_perturb_record.jsonl (or similar) with return_prob per solution.",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        required=True,
        help="Path to val_data.jsonl containing anchor solutions and instance metadata.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path for analyze_return_probability.py.",
    )

    args = parser.parse_args()
    build_return_prob_data(args.perturb_records, args.val_data, args.output)


if __name__ == "__main__":
    main()

