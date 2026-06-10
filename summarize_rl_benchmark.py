"""Summarize benchmark_rl.py text outputs.

Expected filenames are the ones written by run_benchmark_cvrp.sh:
  random_seed<seed>.txt
  policy_round<round>_<mode>_seed<seed>.txt   (mode = greedy | sample_t<temp>)

Legacy names (random_<seed>.txt, policy_r<round>_<seed>.txt) are still parsed.
"""

import argparse
import os
import re
from collections import defaultdict
from statistics import mean, pstdev


FINAL_COST_RE = re.compile(r"final_cost=([0-9.]+)")
WALL_TIME_RE = re.compile(r"wall_time_sec=([0-9.]+)")
POLICY_RE = re.compile(r"policy_r(\d+)_")
LEGACY_POLICY_RE = re.compile(r"policy_round(\d+)_")
ROUND_MODE_RE = re.compile(r"(?:policy_)?round(\d+)_(sample_t025|sample_t1|sample_t08|greedy)_")


def read_metrics(path):
    with open(path) as f:
        text = f.read()
    cost_match = FINAL_COST_RE.search(text)
    if not cost_match:
        return None, None
    wall_match = WALL_TIME_RE.search(text)
    wall_time = float(wall_match.group(1)) if wall_match else None
    return float(cost_match.group(1)), wall_time


def summarize(values):
    return {
        "n": len(values),
        "mean": mean(values),
        "min": min(values),
        "std": pstdev(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_dir")
    args = parser.parse_args()

    random_costs = []
    random_wall_times = []
    policy_costs = defaultdict(list)
    policy_wall_times = defaultdict(list)
    for name in sorted(os.listdir(args.benchmark_dir)):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(args.benchmark_dir, name)
        if not os.path.exists(path):
            continue
        cost, wall_time = read_metrics(path)
        if cost is None:
            continue
        if name.startswith("random_") or name.startswith("random_baseline_"):
            random_costs.append(cost)
            if wall_time is not None:
                random_wall_times.append(wall_time)
        else:
            match = ROUND_MODE_RE.search(name)
            if match:
                key = (int(match.group(1)), match.group(2))
                policy_costs[key].append(cost)
                if wall_time is not None:
                    policy_wall_times[key].append(wall_time)
                continue

            match = POLICY_RE.search(name)
            if not match:
                match = LEGACY_POLICY_RE.search(name)
            if match:
                key = (int(match.group(1)), "policy")
                policy_costs[key].append(cost)
                if wall_time is not None:
                    policy_wall_times[key].append(wall_time)

    if not random_costs:
        raise SystemExit(f"no random benchmark costs found in {args.benchmark_dir}")

    random_stats = summarize(random_costs)
    random_wall = summarize(random_wall_times) if random_wall_times else None
    random_wall_s = "NA" if random_wall is None else f"{random_wall['mean']:.3f}"
    random_wall_n = 0 if random_wall is None else random_wall["n"]
    print(
        f"random n={random_stats['n']} mean={random_stats['mean']:.6f} "
        f"min={random_stats['min']:.6f} std={random_stats['std']:.6f} "
        f"wall_n={random_wall_n} wall_mean_sec={random_wall_s}"
    )
    for round_id, mode in sorted(policy_costs):
        stats = summarize(policy_costs[(round_id, mode)])
        wall = summarize(policy_wall_times[(round_id, mode)]) if policy_wall_times[(round_id, mode)] else None
        wall_s = "NA" if wall is None else f"{wall['mean']:.3f}"
        wall_n = 0 if wall is None else wall["n"]
        gap = (random_stats["mean"] - stats["mean"]) / random_stats["mean"]
        print(
            f"policy_round={round_id} mode={mode} n={stats['n']} mean={stats['mean']:.6f} "
            f"min={stats['min']:.6f} std={stats['std']:.6f} "
            f"gap_vs_random={gap:+.4%} wall_n={wall_n} wall_mean_sec={wall_s}"
        )


if __name__ == "__main__":
    main()
