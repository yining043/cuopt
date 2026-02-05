#!/usr/bin/env python3
"""
Build validation data with 1 anchor + 1 neighbor + 10 distant basins per record.

For each instance in 50-54:
- Load basin_info.jsonl (hash -> {solution_flat, cost})
- Load basin_pairs.jsonl (anchor_hash, neighbor_hash)
- Load distant_basins.jsonl (anchor_hash -> distant basins list)
- Sample 20 records per instance where anchor has >= 10 distant basins available in basin_info.

Output:
  /home/jieyi/cuopt/basin_datasets0_analyze/val_data_1a1n10d.jsonl

Each line includes:
  instance_index,
  anchor {hash, solution_flat, cost},
  neighbor {hash, solution_flat, cost},
  distant_basins: list of 10 {hash, solution_flat, cost}
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple


def _load_basin_info(info_path: str) -> Dict[str, dict]:
    info: Dict[str, dict] = {}
    with open(info_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            h = rec.get("hash")
            if not h:
                continue
            info[h] = {
                "hash": h,
                "solution_flat": rec.get("solution_flat", []),
                "cost": rec.get("cost"),
            }
    return info


def _iter_pairs(pairs_path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            a = (rec.get("anchor_basin") or {}).get("hash")
            n = (rec.get("neighbor_basin") or {}).get("hash")
            if a and n:
                pairs.append((a, n))
    return pairs


def _load_distant_map(distant_path: str) -> Dict[str, List[str]]:
    """
    Support both formats:
    - {anchor_basin_hash, distant_basin_hashes: [..]}
    - {anchor_basin_hash, distant_basins: [{basin_hash, ...}, ...]}
    """
    out: Dict[str, List[str]] = {}
    with open(distant_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ah = rec.get("anchor_basin_hash")
            if not ah:
                continue
            hashes = rec.get("distant_basin_hashes")
            if hashes is None:
                db = rec.get("distant_basins") or []
                hashes = [x.get("basin_hash") for x in db if x.get("basin_hash")]
            if hashes:
                out[ah] = list(hashes)
    return out


def build_for_instance(
    instance_index: int,
    base_dir: str,
    per_instance: int,
    rng: random.Random,
) -> List[dict]:
    inst_dir = os.path.join(base_dir, f"cvrp100_uniform.pkl#{instance_index}")
    info_path = os.path.join(inst_dir, "basin_info.jsonl")
    pairs_path = os.path.join(inst_dir, "basin_pairs.jsonl")
    distant_path = os.path.join(inst_dir, "distant_basins.jsonl")

    if not (os.path.isfile(info_path) and os.path.isfile(pairs_path) and os.path.isfile(distant_path)):
        return []

    basin_info = _load_basin_info(info_path)
    pairs = _iter_pairs(pairs_path)
    distant_map = _load_distant_map(distant_path)

    eligible: List[Tuple[str, str, List[str]]] = []
    for a, n in pairs:
        if a not in basin_info or n not in basin_info:
            continue
        d = distant_map.get(a) or []
        d_ok = [h for h in d if h in basin_info and h != a and h != n]
        if len(d_ok) >= 10:
            eligible.append((a, n, d_ok))

    if not eligible:
        return []

    chosen = eligible if len(eligible) <= per_instance else rng.sample(eligible, per_instance)

    out: List[dict] = []
    for a, n, d_ok in chosen:
        d_pick = rng.sample(d_ok, 10)
        out.append({
            "instance_index": instance_index,
            "anchor": basin_info[a],
            "neighbor": basin_info[n],
            "distant_basins": [basin_info[h] for h in d_pick],
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=str, default="50-54", help="Instance range, e.g. 50-54")
    parser.add_argument("--per_instance", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/jieyi/cuopt/basin_datasets0_analyze",
        help="Directory containing cvrp100_uniform.pkl#*/basin_info.jsonl etc.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/home/jieyi/cuopt/basin_datasets0_analyze/val_data_1a1n10d.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    if "-" in args.instances:
        a, b = args.instances.split("-")
        insts = list(range(int(a), int(b) + 1))
    else:
        insts = [int(x.strip()) for x in args.instances.split(",") if x.strip()]

    rng = random.Random(args.seed)

    all_rows: List[dict] = []
    for idx in insts:
        rows = build_for_instance(idx, args.base_dir, args.per_instance, rng)
        print(f"Instance {idx}: wrote {len(rows)} records", flush=True)
        all_rows.extend(rows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in all_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Total wrote {len(all_rows)} records -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

