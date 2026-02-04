#!/usr/bin/env python3
"""
Gap-to-HGS distribution analysis for basin optima.

Inputs: per-instance `basin_statistics.xlsx` (generated from optima.jsonl)
Outputs:
- `basin_gap_statistics.xlsx`: per-instance summary incl. 5th percentile threshold
- `gap_distribution_overview.png`: overall histogram + CDF + p5 overview
- `gap_distribution_by_instance.png`: per-instance histograms (first 20)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_gap_distribution(basin_base_dir='basin_datasets0'):
    basin_base_dir = Path(basin_base_dir)
    instance_dirs = sorted([d for d in basin_base_dir.iterdir() if d.is_dir() and "cvrp" in d.name.lower()])

    rows = []
    all_gaps = []
    for d in instance_dirs:
        df = pd.read_excel(d / "basin_statistics.xlsx")
        gaps = df["cost_gap_to_hgs_pct"].dropna().to_numpy()

        rows.append({
            "instance": d.name,
            "num_basins": int(gaps.size),
            "min_gap": float(np.min(gaps)),
            "p5_gap": float(np.percentile(gaps, 5)),
            "p10_gap": float(np.percentile(gaps, 10)),
            "mean_gap": float(np.mean(gaps)),
            "median_gap": float(np.median(gaps)),
            "max_gap": float(np.max(gaps)),
            "std_gap": float(np.std(gaps)),
        })
        all_gaps.append(gaps)

    all_gaps = np.concatenate(all_gaps)
    stats_df = pd.DataFrame(rows).sort_values("instance").reset_index(drop=True)

    stats_df.to_excel(basin_base_dir / "basin_gap_statistics.xlsx", index=False)
    plot_gap_distribution(stats_df, all_gaps, basin_base_dir)

    p5_all = float(np.percentile(all_gaps, 5))
    print(f"Overall top-5% gap threshold: gap <= {p5_all:.4f}%")
    return stats_df, all_gaps


def plot_gap_distribution(stats_df, all_gaps, output_dir):
    output_dir = Path(output_dir)

    fig1, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    ax1.hist(all_gaps, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.axvline(np.percentile(all_gaps, 5), color='red', linestyle='--', linewidth=2, 
                label=f'5%分位数: {np.percentile(all_gaps, 5):.4f}%')
    ax1.axvline(np.percentile(all_gaps, 50), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {np.percentile(all_gaps, 50):.4f}%')
    ax1.set_xlabel('Gap to HGS (%)', fontsize=12)
    ax1.set_ylabel('Basin count', fontsize=12)
    ax1.set_title('Gap-to-HGS Histogram (All instances)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    sorted_gaps = np.sort(all_gaps)
    percentiles = np.arange(1, len(sorted_gaps) + 1) / len(sorted_gaps) * 100
    ax2.plot(sorted_gaps, percentiles, linewidth=2, color='blue')
    ax2.axvline(np.percentile(all_gaps, 5), color='red', linestyle='--', linewidth=2, 
                label=f'5th pct: {np.percentile(all_gaps, 5):.4f}%')
    ax2.axhline(5, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Gap to HGS (%)', fontsize=12)
    ax2.set_ylabel('Cumulative %', fontsize=12)
    ax2.set_title('CDF', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    p5_gaps = stats_df['p5_gap'].values
    ax3.boxplot(p5_gaps, vert=True)
    ax3.set_ylabel('Gap to HGS (%)', fontsize=12)
    ax3.set_title('Per-instance 5th percentile threshold', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticklabels(['All instances'])

    ax4 = axes[1, 1]
    instance_indices = range(len(stats_df))
    ax4.scatter(instance_indices, p5_gaps, alpha=0.6, s=50)
    ax4.axhline(np.mean(p5_gaps), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(p5_gaps):.4f}%')
    ax4.set_xlabel('Instance index', fontsize=12)
    ax4.set_ylabel('5th pct gap (%)', fontsize=12)
    ax4.set_title('Per-instance 5th percentile', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig1.savefig(output_dir / 'gap_distribution_overview.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    for _, row in stats_df.iterrows():
        instance_name = row['instance']
        excel_path = output_dir / instance_name / 'basin_statistics.xlsx'
        df = pd.read_excel(excel_path)
        gaps = df['cost_gap_to_hgs_pct'].dropna().values

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(gaps, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        p5 = np.percentile(gaps, 5)
        p50 = np.percentile(gaps, 50)
        ax.axvline(p5, color='red', linestyle='--', linewidth=2, label=f'5th pct: {p5:.4f}%')
        ax.axvline(p50, color='green', linestyle='--', linewidth=2, label=f'Median: {p50:.4f}%')
        ax.set_xlabel('Gap to HGS (%)', fontsize=12)
        ax.set_ylabel('Basin count', fontsize=12)
        ax.set_title(f'Gap Distribution: {instance_name}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        instance_safe = instance_name.replace('#', '_').replace('/', '_')
        fig.savefig(output_dir / instance_name / f'gap_distribution_{instance_safe}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze gap-to-HGS distributions for basin optima.")
    parser.add_argument(
        "--basin_base_dir",
        type=str,
        default="basin_datasets0",
        help="Base directory containing per-instance folders (default: basin_datasets0)",
    )
    
    args = parser.parse_args()
    analyze_gap_distribution(args.basin_base_dir)


if __name__ == '__main__':
    main()
