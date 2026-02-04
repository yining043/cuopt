#!/usr/bin/env python3
"""
Compute critical gap from training_data.jsonl.

Definition:
- For each solution (one line in training_data.jsonl), we compute:
  - gap: gap_to_hgs of the dominant basin (the basin with highest probability)
  - dominant_freq: probability of the dominant basin (in [0, 1])
- Sort all solutions by gap (ascending).
- Find the largest gap G such that for all solutions with gap <= G,
  dominant_freq >= stability_threshold (default 0.8).

This script:
1. Reads training_data.jsonl
2. Extracts (gap, dominant_freq) for all solutions with valid data
3. Computes the critical gap G as defined above
4. Prints statistics and optionally visualizes the result
"""

import argparse
import json
import os
from typing import List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def extract_gap_and_dominant_freq(record: dict) -> Tuple[float, float]:
    """
    Extract (gap_to_hgs, dominant_probability) from a single JSON record.

    The training_data.jsonl lines have the following structure:
    - Top-level keys include many basin hashes, each mapping to a dict with:
        - 'probability': float (0-1)
        - 'gap_to_hgs': float (percent)
        - 'frequency', 'mean_cost', 'is_final_basin', ...
    - There are also meta keys like 'num_unique_basins', 'num_runs', etc.

    We treat any value that is a dict containing a 'probability' key
    as a basin entry.
    """
    basin_dist = record.get("basin_distribution", {})
    basin_features = record.get("basin_features", {})

    if not basin_dist or not basin_features:
        return None, None

    # Find dominant basin id by probability
    dominant_id, prob = max(basin_dist.items(), key=lambda kv: kv[1])
    features = basin_features.get(dominant_id, {})
    gap = features.get("gap_to_hgs")

    if gap is None:
        return None, None

    try:
        gap_val = float(gap)
    except (TypeError, ValueError):
        return None, None

    return gap_val, prob


def load_gaps_and_frequencies(training_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load all (gap, dominant_freq) pairs from training_data.jsonl."""
    gaps: List[float] = []
    freqs: List[float] = []

    with open(training_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            gap, freq = extract_gap_and_dominant_freq(record)
            if gap is None or freq is None:
                continue
            gaps.append(gap)
            freqs.append(freq)

    if not gaps:
        return np.array([]), np.array([])

    gaps_arr = np.array(gaps, dtype=float)
    freqs_arr = np.array(freqs, dtype=float)

    # Sort by gap ascending
    sort_idx = np.argsort(gaps_arr)
    gaps_arr = gaps_arr[sort_idx]
    freqs_arr = freqs_arr[sort_idx]

    return gaps_arr, freqs_arr


def compute_critical_gap(
    gaps: np.ndarray,
    dominant_freqs: np.ndarray,
    stability_threshold: float = 0.8,
) -> Tuple[float, int, int, int]:
    """
    Compute the critical gap G using the following definition:

    1. Find all solutions where dominant_freq >= stability_threshold.
    2. Among these stable solutions, find the maximum gap, set as critical_gap.
    3. Count how many solutions with gap <= critical_gap meet the threshold,
       and how many don't.

    Returns:
        critical_gap (float or None): maximum gap among stable solutions,
        stable_count (int): number of solutions with gap <= critical_gap that meet threshold,
        unstable_count (int): number of solutions with gap <= critical_gap that don't meet threshold,
        total_count (int): total number of solutions with gap <= critical_gap
    """
    if gaps.size == 0 or dominant_freqs.size == 0:
        return None, 0, 0, 0

    # Find all solutions that meet the threshold
    meets_threshold = dominant_freqs >= stability_threshold
    stable_mask = meets_threshold

    if not np.any(stable_mask):
        # No solution meets the threshold
        return None, 0, 0, len(gaps)

    # Among stable solutions, find the maximum gap
    stable_gaps = gaps[stable_mask]
    critical_gap = float(np.max(stable_gaps))

    # Count solutions with gap <= critical_gap
    within_range_mask = gaps <= critical_gap
    within_range_stable = np.sum(within_range_mask & stable_mask)
    within_range_unstable = np.sum(within_range_mask & (~stable_mask))
    within_range_total = int(np.sum(within_range_mask))

    return critical_gap, int(within_range_stable), int(within_range_unstable), within_range_total


def visualize_gap_vs_frequency(
    gaps: np.ndarray,
    freqs: np.ndarray,
    critical_gap: float,
    stability_threshold: float,
    output_file: str,
) -> None:
    """Create a scatter plot of gap vs dominant frequency with critical gap."""
    if not HAS_MPL or gaps.size == 0:
        return

    import matplotlib.pyplot as plt  # type: ignore

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        gaps,
        freqs * 100,
        s=20,
        alpha=0.5,
        color="steelblue",
        edgecolors="none",
        label="Solutions",
    )

    if critical_gap is not None:
        ax.axvline(
            x=critical_gap,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Critical gap = {critical_gap:.2f}%",
        )
        # Mark all solutions with gap <= critical_gap
        within_range_mask = gaps <= critical_gap
        meets_threshold = freqs >= stability_threshold
        
        # Stable solutions (meet threshold) within range
        stable_in_range = within_range_mask & meets_threshold
        if np.any(stable_in_range):
            ax.scatter(
                gaps[stable_in_range],
                freqs[stable_in_range] * 100,
                s=40,
                alpha=0.8,
                color="green",
                marker="*",
                label="Meet threshold (gap <= critical)",
            )
        
        # Unstable solutions (don't meet threshold) within range
        unstable_in_range = within_range_mask & (~meets_threshold)
        if np.any(unstable_in_range):
            ax.scatter(
                gaps[unstable_in_range],
                freqs[unstable_in_range] * 100,
                s=40,
                alpha=0.8,
                color="red",
                marker="x",
                label="Don't meet threshold (gap <= critical)",
            )

    ax.axhline(
        y=stability_threshold * 100,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label=f"Threshold = {stability_threshold * 100:.0f}%",
    )

    ax.set_xlabel("Gap to HGS (%)")
    ax.set_ylabel("Dominant Basin Frequency (%)")
    ax.set_title("Gap vs Dominant Basin Frequency (training_data.jsonl)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_boxplot_by_gap(
    gaps: np.ndarray,
    freqs: np.ndarray,
    gap_threshold: float = 5.0,
    output_file: str = None,
) -> None:
    """Create boxplots comparing dominant_freq for gap <= threshold vs gap > threshold."""
    if not HAS_MPL or gaps.size == 0:
        return

    import matplotlib.pyplot as plt  # type: ignore

    # Split data by gap threshold
    low_gap_mask = gaps <= gap_threshold
    high_gap_mask = gaps > gap_threshold
    
    low_gap_freqs = freqs[low_gap_mask] * 100  # Convert to percentage
    high_gap_freqs = freqs[high_gap_mask] * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Prepare data for boxplot
    data_to_plot = [low_gap_freqs, high_gap_freqs]
    labels = [
        f"Gap <= {gap_threshold}%\n(n={len(low_gap_freqs)})",
        f"Gap > {gap_threshold}%\n(n={len(high_gap_freqs)})"
    ]
    
    # Create boxplot
    bp = ax.boxplot(
        data_to_plot,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.6,
    )
    ax.set_xticklabels(labels)
    
    # Customize box colors
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize other elements
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        if element in bp:
            plt.setp(bp[element], color='black', linewidth=1.5)
    
    plt.setp(bp['means'], linestyle='--', linewidth=2)
    plt.setp(bp['medians'], linewidth=2)
    
    ax.set_ylabel("Dominant Basin Frequency (%)", fontsize=12)
    ax.set_title(
        f"Distribution of Dominant Basin Frequency\nby Gap Threshold ({gap_threshold}%)",
        fontsize=13,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    if len(low_gap_freqs) > 0 and len(high_gap_freqs) > 0:
        low_mean = np.mean(low_gap_freqs)
        low_median = np.median(low_gap_freqs)
        high_mean = np.mean(high_gap_freqs)
        high_median = np.median(high_gap_freqs)
        
        stats_text = (
            f"Gap <= {gap_threshold}%: mean={low_mean:.2f}%, median={low_median:.2f}%\n"
            f"Gap > {gap_threshold}%: mean={high_mean:.2f}%, median={high_median:.2f}%"
        )
        ax.text(
            0.02, 0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compute critical gap from training_data.jsonl"
    )
    parser.add_argument(
        "--training_file",
        type=str,
        default="/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#0/training_data.jsonl",
        help="Path to training_data.jsonl",
    )
    parser.add_argument(
        "--stability_threshold",
        type=float,
        default=0.8,
        help="Stability threshold for dominant basin frequency (default: 0.8)",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="Prefix for output files (default: derived from training_file)",
    )
    parser.add_argument(
        "--gap_threshold",
        type=float,
        default=5.0,
        help="Gap threshold (%) for boxplot comparison (default: 5.0)",
    )

    args = parser.parse_args()

    training_file = args.training_file
    threshold = args.stability_threshold

    if not os.path.isfile(training_file):
        print(f"Error: training file not found: {training_file}")
        return

    print(f"Loading training data from: {training_file}")
    gaps, freqs = load_gaps_and_frequencies(training_file)
    print(f"Loaded {len(gaps)} solutions with valid (gap, dominant_freq).")

    if len(gaps) == 0:
        print("No valid data found. Abort.")
        return

    critical_gap, stable_count, unstable_count, total_count = compute_critical_gap(
        gaps, freqs, stability_threshold=threshold
    )

    print("\n=== Critical Gap Result ===")
    print(f"Stability threshold (dominant_freq): {threshold:.2f}")
    if critical_gap is None:
        print("No solution meets the stability threshold.")
    else:
        stable_ratio = stable_count / total_count if total_count > 0 else 0.0
        unstable_ratio = unstable_count / total_count if total_count > 0 else 0.0
        print(f"Critical gap (max gap among stable solutions): {critical_gap:.4f}%")
        print(f"\nSolutions with gap <= {critical_gap:.4f}%:")
        print(f"  - Meet threshold (>= {threshold:.2f}): {stable_count} / {total_count} ({stable_ratio * 100:.2f}%)")
        print(f"  - Don't meet threshold (< {threshold:.2f}): {unstable_count} / {total_count} ({unstable_ratio * 100:.2f}%)")

    # Visualization
    if args.output_prefix is None:
        base = os.path.splitext(os.path.basename(training_file))[0]
        output_prefix = os.path.join(os.path.dirname(training_file), base)
    else:
        output_prefix = args.output_prefix

    if HAS_MPL:
        viz_file = f"{output_prefix}_critical_gap.png"
        print(f"\nSaving visualization to: {viz_file}")
        visualize_gap_vs_frequency(gaps, freqs, critical_gap, threshold, viz_file)
        
        # Boxplot comparison
        boxplot_file = f"{output_prefix}_boxplot_gap{args.gap_threshold:.1f}.png"
        print(f"Saving boxplot to: {boxplot_file}")
        visualize_boxplot_by_gap(gaps, freqs, gap_threshold=args.gap_threshold, output_file=boxplot_file)
    else:
        print("\nmatplotlib not available, skip visualization.")


if __name__ == "__main__":
    main()

