#!/usr/bin/env python3
"""Compute (c_int - c_opt) / c_int from training_data.jsonl.

c_int = initial_solution.cost (intermediate solution cost)
c_opt = mean_cost of the basin with max probability in basin_distribution.
"""
import argparse


# (base) jieyi@wulab2:~/cuopt$ python stats_training_gap.py basin_datasets0_analyze --indices 0-49
# === (c_int - c_opt) / c_int ===
# File(s): 50
# Records: 1120411
# --- Gap == 0 (separate) ---
#   count  = 135367
#   frac   = 0.1208
# --- Non-zero gaps: quantiles ---
#   q10 = 0.000003
#   q20 = 0.076238
#   q30 = 0.250649
#   q40 = 0.516222
#   q50 = 0.890788
#   q60 = 1.415037
#   q70 = 2.184300
#   q80 = 3.458261
#   q90 = 6.214967
#   q95 = 10.040184
#   q99 = 23.977334
#   n_nonzero = 985044, mean = 2.476697, std = 4.692560
# --- By advantage buckets (interval = fraction, pct = %% of samples) ---
#   interval        count    pct(% of n)   mean_gap   median_gap
#   [0.00,0.05)       311816    27.83%   0.003763   0.000001
#   [0.05,0.10)        36828     3.29%   0.073978   0.073428
#   [0.10,0.20)        57868     5.16%   0.148161   0.147590
#   [0.20,0.50)       117786    10.51%   0.338209   0.332453
#   [0.50,1.00)       127005    11.34%   0.731799   0.722595
#   [1.00,1.50)        88104     7.86%   1.236632   1.229687
#   [1.50,2.00)        65205     5.82%   1.738895   1.732307
#   [2.00,3.50)       121320    10.83%   2.662186   2.620534
#   [3.50,5.00)        64013     5.71%   4.179531   4.142806
#   [5.00,8.00)        61400     5.48%   6.267686   6.158319
#   [8.00,10.00)        19494     1.74%   8.915421   8.878309
#   [10.00,25.00)        49572     4.42%   18.617660   15.695437
# c_int (anchor cost): mean=1634.6081, min=1318.6053, max=4180.7612
# c_opt (best basin):  mean=1594.8401, min=1317.7439, max=2116.6792

# Bucket edges (as fraction 0~1). Intervals: [T[i], T[i+1]). Last is [T[-1], inf).
ADVANTAGE_THRESHOLDS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.5, 5.0, 8.0, 10.0, 25.0]
import json
import os
import sys


def load_gaps_from_training_data(path: str, min_c_int: float = 1e-8):
    """Read training_data.jsonl and yield (c_int, c_opt, gap) per record."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            init = rec.get("initial_solution", {})
            c_int = init.get("cost")
            if c_int is None:
                continue
            c_int = float(c_int)
            if c_int < min_c_int:
                continue
            bd = rec.get("basin_distribution", {})
            if not bd:
                continue
            best_hash, _ = max(bd.items(), key=lambda x: float(x[1]))
            features = rec.get("basin_features", {})
            feat = features.get(best_hash, {})
            c_opt = feat.get("mean_cost")
            if c_opt is None:
                continue
            c_opt = float(c_opt)
            gap = (c_int - c_opt) / c_int
            gap = max(0.0, min(1.0, gap))  # clamp [0, 1]
            yield c_int, c_opt, gap


def main():
    parser = argparse.ArgumentParser(description="Stats for (c_int - c_opt) / c_int from training_data.jsonl")
    parser.add_argument("path", type=str, help="Path to training_data.jsonl (file or dir containing instance dirs)")
    parser.add_argument("--instance_prefix", type=str, default="cvrp100_uniform.pkl#", help="Instance dir prefix when path is a root dir")
    parser.add_argument("--indices", type=str, default=None, help="Instance indices, e.g. 0-49 or 0,1,2 (default: all found)")
    args = parser.parse_args()

    paths = []
    if os.path.isfile(args.path):
        paths = [args.path]
    elif os.path.isdir(args.path):
        if args.indices:
            parts = args.indices.replace(" ", "").split(",")
            indices = []
            for p in parts:
                if "-" in p:
                    a, b = map(int, p.split("-"))
                    indices.extend(range(a, b + 1))
                else:
                    indices.append(int(p))
        else:
            indices = []
            for name in os.listdir(args.path):
                if name.startswith(args.instance_prefix) and "#" in name:
                    try:
                        idx = int(name.split("#")[-1].split("/")[0].split("_")[0])
                        indices.append(idx)
                    except ValueError:
                        pass
            indices = sorted(set(indices)) if indices else list(range(200))

        for i in indices:
            p = os.path.join(args.path, f"{args.instance_prefix}{i}", "training_data.jsonl")
            if os.path.isfile(p):
                paths.append(p)
        if not paths:
            print("No training_data.jsonl found under directory.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Path is neither a file nor a directory.", file=sys.stderr)
        sys.exit(1)

    all_gaps = []
    all_c_int = []
    all_c_opt = []
    for p in paths:
        for c_int, c_opt, gap in load_gaps_from_training_data(p):
            all_gaps.append(gap)
            all_c_int.append(c_int)
            all_c_opt.append(c_opt)

    if not all_gaps:
        print("No valid records (need initial_solution.cost and basin_features with mean_cost).")
        return

    import numpy as np
    gaps = np.array(all_gaps) * 100.0  # convert to percentage
    c_int_arr = np.array(all_c_int)
    c_opt_arr = np.array(all_c_opt)
    n = len(gaps)

    # Gap == 0 (use small epsilon for float)
    eps = 1e-9
    mask_zero = gaps <= eps
    n_zero = int(mask_zero.sum())
    gaps_nonzero = gaps[~mask_zero]

    print("=== (c_int - c_opt) / c_int ===\n")
    print(f"File(s): {len(paths)}")
    print(f"Records: {n}\n")

    print("--- Gap == 0 (separate) ---")
    print(f"  count  = {n_zero}")
    print(f"  frac   = {n_zero / n:.4f}\n")

    print("--- Non-zero gaps: quantiles ---")
    if len(gaps_nonzero) == 0:
        print("  (no non-zero gaps)")
    else:
        q_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
        for p in q_levels:
            print(f"  q{p:2d} = {np.percentile(gaps_nonzero, p):.6f}")
        print(f"  n_nonzero = {len(gaps_nonzero)}, mean = {gaps_nonzero.mean():.6f}, std = {gaps_nonzero.std():.6f}")

    # Bucket by ADVANTAGE_THRESHOLDS: [T[i], T[i+1]), report pct and mean gap per bucket
    thresholds = np.array(ADVANTAGE_THRESHOLDS)
    bin_idx = np.searchsorted(thresholds, gaps, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(thresholds) - 2)

    print("--- By advantage buckets (interval = fraction, pct = %% of samples) ---")
    print("  interval        count    pct(% of n)   mean_gap   median_gap")
    for i in range(len(thresholds) - 1):
        mask = bin_idx == i
        cnt = int(mask.sum())
        if cnt == 0:
            pct = 0.0
            mean_gap = np.nan
            median_gap = np.nan
        else:
            pct = 100.0 * cnt / n
            b = gaps[mask]
            mean_gap = b.mean()
            median_gap = np.median(b)
        lo, hi = thresholds[i], thresholds[i + 1]
        print(f"  [{lo:.2f},{hi:.2f})     {cnt:8d}   {pct:6.2f}%   {mean_gap:.6f}   {median_gap:.6f}")

    print(f"\nc_int (anchor cost): mean={c_int_arr.mean():.4f}, min={c_int_arr.min():.4f}, max={c_int_arr.max():.4f}")
    print(f"c_opt (best basin):  mean={c_opt_arr.mean():.4f}, min={c_opt_arr.min():.4f}, max={c_opt_arr.max():.4f}")


if __name__ == "__main__":
    main()
