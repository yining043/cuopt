#!/usr/bin/env python3
"""
Label each basin pair in basin_pairs.jsonl as positive / negative / zero
using the co-occurrence matrix computed from training_data.jsonl.

Definitions (per-record logic):
- "positive": in at least one record, the two basins both have
  count = round(prob * 100) >= min_count (they co-occur strongly).
- "negative": never have a positive co-occurrence, but appear together
  in some records where at least one side has count < min_count.
- "zero": never have any positive or negative contribution (true zero pair).

The sign is fully consistent with visualize_basin_cooccurrence_matrix.py /
build_cooccurrence_matrix_from_training_data_fast:
- coocc[(b1, b2)] > 0  -> positive
- coocc[(b1, b2)] < 0  -> negative
- pair not in coocc or value == 0 -> zero

Outputs:
- A jsonl file: each line of basin_pairs.jsonl is augmented with
  fields \"sign\" and \"coocc_value\".
- (Optional) a mask.npy: for all basins that appear in basin_pairs.jsonl,
  build an n×n matrix with values {1, -1, 0} for positive / negative / zero;
  diagonal entries are 0.
"""

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np

import visualize_basin_cooccurrence_matrix as vcm


def load_cooccurrence(
    training_data_file: str,
    min_count: int = 10,
    max_runs: int = 10,
    max_records: int = None,
    max_basins_per_record: int = None,
) -> Dict[Tuple[str, str], float]:
    """Call the existing fast implementation to build a pairwise basin co-occurrence dict."""
    coocc = vcm.build_cooccurrence_matrix_from_training_data_fast(
        training_data_file=training_data_file,
        min_count=min_count,
        max_runs=max_runs,
        max_records=max_records,
        max_basins_per_record=max_basins_per_record,
    )
    return coocc


def sign_from_value(val: float) -> str:
    """Return the sign label based on the co-occurrence value."""
    if val > 0:
        return "positive"
    if val < 0:
        return "negative"
    # Treat val == 0 or None as zero
    return "zero"


def build_mask_from_signed_pairs(
    coocc: Dict[Tuple[str, str], float],
    basin_pairs_path: str,
) -> Tuple[np.ndarray, list]:
    """
    Build an n×n sign mask over all basins that appear in basin_pairs.jsonl:
    - mask[i, j] =  1  -> positive
    - mask[i, j] = -1  -> negative
    - mask[i, j] =  0  -> zero
    Diagonal entries are set to 0.
    """
    basins = set()
    with open(basin_pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            basins.add(rec["anchor_basin"]["hash"])
            basins.add(rec["neighbor_basin"]["hash"])
    basin_list = sorted(basins)
    idx = {h: i for i, h in enumerate(basin_list)}

    n = len(basin_list)
    mask = np.zeros((n, n), dtype=np.int8)

    for (b1, b2), val in coocc.items():
        i = idx.get(b1)
        j = idx.get(b2)
        if i is None or j is None:
            continue
        s = sign_from_value(val)
        if s == "positive":
            v = 1
        elif s == "negative":
            v = -1
        else:
            v = 0
        mask[i, j] = v
        mask[j, i] = v

    # Set diagonal to 0 (self-relationships are not meaningful here)
    np.fill_diagonal(mask, 0)
    return mask, basin_list


def main():
    parser = argparse.ArgumentParser(
        description="Label each pair in basin_pairs.jsonl as positive/negative/zero using training_data.jsonl."
    )
    parser.add_argument(
        "--basin_pairs",
        type=str,
        required=True,
        help="Path to basin_pairs.jsonl",
    )
    parser.add_argument(
        "--training_data",
        type=str,
        required=True,
        help="Path to training_data.jsonl",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Output jsonl path (default: same dir as basin_pairs, name + '_with_sign.jsonl')",
    )
    parser.add_argument(
        "--output_mask",
        type=str,
        default=None,
        help="(Optional) Output .npy path for sign mask (default: same dir as basin_pairs, 'anchor_basin_sign_mask.npy')",
    )
    parser.add_argument(
        "--output_basin_list",
        type=str,
        default=None,
        help="(Optional) Output json path for basin order (default: same dir as basin_pairs, 'anchor_basin_sign_order.json')",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=10,
        help="min_count threshold used in co-occurrence computation (default 10)",
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=10,
        help="Maximum number of runs to use when building the co-occurrence matrix (default 10)",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Maximum number of records to read (default None = all)",
    )
    parser.add_argument(
        "--max_basins_per_record",
        type=int,
        default=None,
        help="Maximum number of basins to consider per record (default None = unlimited, can speed up)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.basin_pairs):
        raise FileNotFoundError(f"basin_pairs not found: {args.basin_pairs}")
    if not os.path.isfile(args.training_data):
        raise FileNotFoundError(f"training_data not found: {args.training_data}")

    # 1) Build co-occurrence matrix
    print("Building co-occurrence matrix from training_data.jsonl (fast version)...")
    coocc = load_cooccurrence(
        training_data_file=args.training_data,
        min_count=args.min_count,
        max_runs=args.max_runs,
        max_records=args.max_records,
        max_basins_per_record=args.max_basins_per_record,
    )
    print(f"  Number of co-occurring basin pairs: {len(coocc)}")

    # 2) Read basin_pairs.jsonl line by line, assign signs, and write a new jsonl
    if args.output_jsonl is None:
        base_dir = os.path.dirname(os.path.abspath(args.basin_pairs))
        base_name = os.path.basename(args.basin_pairs)
        name, _ = os.path.splitext(base_name)
        out_jsonl = os.path.join(base_dir, name + "_with_sign.jsonl")
    else:
        out_jsonl = args.output_jsonl

    print(f"Writing signed basin_pairs to: {out_jsonl}")
    num_pos = num_neg = num_zero = 0
    with open(args.basin_pairs, "r", encoding="utf-8") as fin, open(
        out_jsonl, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            h1 = rec["anchor_basin"]["hash"]
            h2 = rec["neighbor_basin"]["hash"]
            key = (h1, h2) if h1 < h2 else (h2, h1)
            val = coocc.get(key, 0.0)
            sign = sign_from_value(val)
            rec["coocc_value"] = float(val)
            rec["sign"] = sign
            if sign == "positive":
                num_pos += 1
            elif sign == "negative":
                num_neg += 1
            else:
                num_zero += 1
            fout.write(json.dumps(rec) + "\n")

    print(
        f"Stats over basin_pairs: positive={num_pos}, negative={num_neg}, zero={num_zero}, total={num_pos + num_neg + num_zero}"
    )

    # 3) Optional: build sign mask
    if args.output_mask is not None or args.output_basin_list is not None:
        if args.output_mask is None:
            base_dir = os.path.dirname(os.path.abspath(args.basin_pairs))
            out_mask = os.path.join(base_dir, "anchor_basin_sign_mask.npy")
        else:
            out_mask = args.output_mask
        if args.output_basin_list is None:
            base_dir = os.path.dirname(os.path.abspath(args.basin_pairs))
            out_order = os.path.join(base_dir, "anchor_basin_sign_order.json")
        else:
            out_order = args.output_basin_list

        print("Building sign mask matrix from the co-occurrence dictionary...")
        mask, basin_list = build_mask_from_signed_pairs(coocc, args.basin_pairs)
        np.save(out_mask, mask)
        with open(out_order, "w", encoding="utf-8") as f:
            json.dump(basin_list, f, indent=2)
        print(f"  Mask shape: {mask.shape}, saved to {out_mask}")
        print(f"  Basin order (total {len(basin_list)} basins) saved to {out_order}")


if __name__ == "__main__":
    main()

