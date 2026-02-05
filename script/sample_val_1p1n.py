#!/usr/bin/env python3
"""
Sample 20 records per instance from k1_collection_results (50-54) into val_data_1p1n.jsonl.

Each line: instance_index + anchor, positive_sample, negative_sample (1 anchor, 1 positive, 1 negative).
Total: 5 * 20 = 100 lines.

Usage:
  python script/sample_val_1p1n.py [--base_dir perturb_k1_collect] [--out perturb_k1_collect/val_data_1p1n.jsonl] [--seed N]
"""

import argparse
import json
import os
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="perturb_k1_collect", help="Base dir containing cvrp100_uniform.pkl#50 etc.")
    parser.add_argument("--out", type=str, default="perturb_k1_collect/val_data_1p1n.jsonl", help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.join(root, args.base_dir) if not os.path.isabs(args.base_dir) else args.base_dir
    out_path = os.path.join(root, args.out) if not os.path.isabs(args.out) else args.out
    out_path = os.path.abspath(out_path)

    if args.seed is not None:
        random.seed(args.seed)

    per_instance = 20
    instance_indices = list(range(50, 55))  # 50,51,52,53,54
    all_records = []

    for instance_index in instance_indices:
        rel_dir = f"cvrp100_uniform.pkl#{instance_index}"
        jsonl_name = "k1_collection_results_remove_and_insert.ALL_r30.jsonl"
        path = os.path.join(base_dir, rel_dir, jsonl_name)
        if not os.path.isfile(path):
            print(f"Skip instance {instance_index}: not found {path}", flush=True)
            continue
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        records = [json.loads(line) for line in lines]
        if len(records) < per_instance:
            chosen = records
            print(f"Instance {instance_index}: only {len(records)} records, using all", flush=True)
        else:
            chosen = random.sample(records, per_instance)
        for rec in chosen:
            rec["instance_index"] = instance_index
            all_records.append(rec)
        print(f"Instance {instance_index}: sampled {len(chosen)} -> total {len(all_records)}", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_records)} records to {out_path}", flush=True)


if __name__ == "__main__":
    main()
