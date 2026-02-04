#!/usr/bin/env python3
"""
Analyze k=1 repeat experiments and plot variance boxplots.

Assumes you have run run_k1_variance.sh, which creates per-repeat JSONL files
under k1_variance_rep1/, k1_variance_rep2/, ... e.g.:

  k1_variance_rep{r}/instance_id/remove_and_insert_training_data.batch_rep*_r30.jsonl

For each local optimum (identified by initial_solution.edges_hash), this script:
  - collects metrics across repeats:
      * return_prob      = return_to_original_count / num_runs
      * jaccard_distance = rec["jaccard_distance"]
      * broken_pairs     = rec["broken_pairs_ratio"]
  - keeps only optima that appear in all repeats
  - computes per-optimum standard deviation across repeats
  - plots boxplots of these std values for each metric
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_metrics_from_jsonl(path: str) -> Dict[str, dict]:
    """
    Load metrics from a single JSONL file.

    Returns:
        dict: edges_hash -> metrics dict
    """
    metrics = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            init = rec.get("initial_solution") or {}
            edges_hash = init.get("edges_hash")
            if not edges_hash:
                continue

            num_runs = rec.get("num_local_search_runs") or rec.get(
                "num_runs", 30
            )
            if not num_runs:
                num_runs = 30

            # Try both field names for return count
            return_count = rec.get("return_to_original_count") or rec.get("return_to_original", 0)
            # Also try return_to_original_ratio directly
            return_prob = rec.get("return_to_original_prob") or rec.get("return_to_original_ratio")
            if return_prob is None:
                return_prob = return_count / float(num_runs) if num_runs > 0 else 0.0

            # jaccard_distance and broken_pairs_ratio are in perturbed_solution dict
            perturbed = rec.get("perturbed_solution") or {}
            jaccard = perturbed.get("jaccard_distance", None)
            broken_ratio = perturbed.get("broken_pairs_ratio", None)

            metrics[str(edges_hash)] = {
                "return_prob": return_prob,
                "jaccard": jaccard,
                "broken_pairs_ratio": broken_ratio,
            }
    return metrics


def collect_repeats_metrics(
    repeats_base_dir: str, instance_id: str, runs: int, repeats: int
) -> Dict[str, Dict[str, List[float]]]:
    """
    Collect metrics across repeats for each edges_hash.
    
    Each repeat is in a separate directory: repeats_base_dir_rep1, repeats_base_dir_rep2, etc.

    Returns:
        edges_hash -> {
            "return_prob": [v1, v2, ...],
            "jaccard": [v1, v2, ...],
            "broken_pairs_ratio": [v1, v2, ...],
        }
    """
    files = []
    for r in range(1, repeats + 1):
        rep_dir = f"{repeats_base_dir}_rep{r}"
        out_dir = os.path.join(rep_dir, instance_id)
        pattern = os.path.join(
            out_dir, f"remove_and_insert_training_data.batch_rep*_r{runs}.jsonl"
        )
        rep_files = sorted(glob.glob(pattern))
        if rep_files:
            files.extend(rep_files)
        else:
            print(f"Warning: No files found for repeat {r} in {out_dir}")
    
    if not files:
        raise FileNotFoundError(
            f"No repeat JSONL files found. Searched in: {repeats_base_dir}_rep*/{instance_id}/"
        )

    print(f"Found {len(files)} repeat files:")
    for p in files:
        print(f"  {p}")

    per_repeat = []
    for p in files:
        print(f"Loading metrics from: {p}")
        per_repeat.append(load_metrics_from_jsonl(p))

    # Keep only hashes that appear in all repeats
    all_hashes = set.intersection(*[set(m.keys()) for m in per_repeat])
    print(f"\nTotal hashes present in ALL repeats: {len(all_hashes)}")

    metrics_by_hash: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for h in all_hashes:
        for m in per_repeat:
            vals = m.get(h)
            if vals is None:
                continue
            metrics_by_hash[h]["return_prob"].append(vals["return_prob"])
            if vals["jaccard"] is not None:
                metrics_by_hash[h]["jaccard"].append(vals["jaccard"])
            if vals["broken_pairs_ratio"] is not None:
                metrics_by_hash[h]["broken_pairs_ratio"].append(
                    vals["broken_pairs_ratio"]
                )

    # Filter to those that have full repeats for each metric
    filtered: Dict[str, Dict[str, List[float]]] = {}
    for h, d in metrics_by_hash.items():
        if (
            len(d["return_prob"]) == repeats
            and len(d["jaccard"]) == repeats
            and len(d["broken_pairs_ratio"]) == repeats
        ):
            filtered[h] = d

    print(
        f"Hashes with complete data (all {repeats} repeats for all metrics): "
        f"{len(filtered)}"
    )
    return filtered


def compute_std_arrays(
    metrics_by_hash: Dict[str, Dict[str, List[float]]]
):
    """Compute per-optimum std arrays for each metric."""
    return_std = []
    jaccard_std = []
    broken_std = []

    for h, d in metrics_by_hash.items():
        return_std.append(float(np.std(d["return_prob"])))
        jaccard_std.append(float(np.std(d["jaccard"])))
        broken_std.append(float(np.std(d["broken_pairs_ratio"])))

    return (
        np.array(return_std),
        np.array(jaccard_std),
        np.array(broken_std),
    )


def collect_raw_values(
    metrics_by_hash: Dict[str, Dict[str, List[float]]]
):
    """Collect all raw values across all optima and repeats."""
    return_raw = []
    jaccard_raw = []
    broken_raw = []

    for h, d in metrics_by_hash.items():
        return_raw.extend(d["return_prob"])
        jaccard_raw.extend(d["jaccard"])
        broken_raw.extend(d["broken_pairs_ratio"])

    return (
        np.array(return_raw),
        np.array(jaccard_raw),
        np.array(broken_raw),
    )


def collect_raw_values(
    metrics_by_hash: Dict[str, Dict[str, List[float]]]
):
    """Collect all raw values across all optima and repeats."""
    return_raw = []
    jaccard_raw = []
    broken_raw = []

    for h, d in metrics_by_hash.items():
        return_raw.extend(d["return_prob"])
        jaccard_raw.extend(d["jaccard"])
        broken_raw.extend(d["broken_pairs_ratio"])

    return (
        np.array(return_raw),
        np.array(jaccard_raw),
        np.array(broken_raw),
    )


def plot_boxplot(data: np.ndarray, title: str, ylabel: str, out_path: str):
    plt.figure(figsize=(6, 5))
    plt.boxplot(data, vert=True, showfliers=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved boxplot to: {out_path}")


def plot_boxplot_by_optima(
    metrics_by_hash: Dict[str, Dict[str, List[float]]],
    metric_name: str,
    title: str,
    ylabel: str,
    out_path: str,
):
    """
    Plot boxplot where each box represents one optima's values across repeats.
    """
    # Prepare data: list of lists, one per optima
    data_by_optima = []
    labels = []
    
    for h, d in sorted(metrics_by_hash.items()):
        values = d[metric_name]
        if len(values) > 0:
            data_by_optima.append(values)
            labels.append(h[:8])  # Short hash for label
    
    if not data_by_optima:
        print(f"No data to plot for {metric_name}")
        return
    
    plt.figure(figsize=(max(8, len(data_by_optima) * 1.2), 5))
    bp = plt.boxplot(data_by_optima, vert=True, showfliers=True, tick_labels=labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Optima (edges_hash prefix)")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved boxplot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze k=1 repeat experiments and plot variance boxplots."
    )
    parser.add_argument(
        "--instance_index",
        type=int,
        default=1,
        help="Instance index (default: 1)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Number of local search runs per step (default: 30)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of repeats (default: 10)",
    )
    parser.add_argument(
        "--repeats_base_dir",
        type=str,
        default="/home/jieyi/cuopt/k1_variance",
        help="Base directory prefix for repeat outputs "
        "(each repeat is in repeats_base_dir_rep1, repeats_base_dir_rep2, etc.) "
        "(default: /home/jieyi/cuopt/k1_variance)",
    )
    parser.add_argument(
        "--instance_file",
        type=str,
        default="cvrp100_uniform.pkl",
        help="Instance file name (default: cvrp100_uniform.pkl)",
    )

    args = parser.parse_args()

    instance_id = f"{args.instance_file}#{args.instance_index}"

    print(f"Analyzing repeats from: {args.repeats_base_dir}_rep*/{instance_id}/")

    metrics_by_hash = collect_repeats_metrics(
        args.repeats_base_dir, instance_id, runs=args.runs, repeats=args.repeats
    )
    if not metrics_by_hash:
        print("No hashes with complete data; nothing to plot.")
        return

    return_std, jaccard_std, broken_std = compute_std_arrays(metrics_by_hash)

    print("\nSummary of std values (per optimum, across repeats):")
    print(
        f"  return_prob std: mean={return_std.mean():.4f}, "
        f"median={np.median(return_std):.4f}, "
        f"min={return_std.min():.4f}, max={return_std.max():.4f}"
    )
    print(
        f"  jaccard std:     mean={jaccard_std.mean():.4f}, "
        f"median={np.median(jaccard_std):.4f}, "
        f"min={jaccard_std.min():.4f}, max={jaccard_std.max():.4f}"
    )
    print(
        f"  broken_pairs std: mean={broken_std.mean():.4f}, "
        f"median={np.median(broken_std):.4f}, "
        f"min={broken_std.min():.4f}, max={broken_std.max():.4f}"
    )

    # Print detailed values for each optimum across repeats
    print("\n" + "="*80)
    print("Detailed values for each optimum across repeats:")
    print("="*80)
    for h, d in sorted(metrics_by_hash.items()):
        print(f"\nOptimum: {h[:16]}...")
        print(f"  Return probability: {[f'{v:.4f}' for v in d['return_prob']]}")
        print(f"  Jaccard distance:   {[f'{v:.4f}' for v in d['jaccard']]}")
        print(f"  Broken pairs ratio: {[f'{v:.4f}' for v in d['broken_pairs_ratio']]}")
        print(f"  (std: return_prob={np.std(d['return_prob']):.4f}, "
              f"jaccard={np.std(d['jaccard']):.4f}, "
              f"broken_pairs={np.std(d['broken_pairs_ratio']):.4f})")
    print("="*80 + "\n")

    # Output directory: use the first repeat's directory
    first_rep_dir = f"{args.repeats_base_dir}_rep1"
    out_dir = os.path.join(first_rep_dir, instance_id)
    os.makedirs(out_dir, exist_ok=True)

    # Boxplots by optima: each box shows one optima's raw values across repeats
    plot_boxplot_by_optima(
        metrics_by_hash,
        "return_prob",
        "Return probability across repeats (per optimum)",
        "Return probability",
        os.path.join(out_dir, "k1_repeats_return_prob_by_optima.png"),
    )
    plot_boxplot_by_optima(
        metrics_by_hash,
        "jaccard",
        "Jaccard distance across repeats (per optimum)",
        "Jaccard distance",
        os.path.join(out_dir, "k1_repeats_jaccard_by_optima.png"),
    )
    plot_boxplot_by_optima(
        metrics_by_hash,
        "broken_pairs_ratio",
        "Broken pairs ratio across repeats (per optimum)",
        "Broken pairs ratio",
        os.path.join(out_dir, "k1_repeats_broken_pairs_by_optima.png"),
    )


if __name__ == "__main__":
    main()

