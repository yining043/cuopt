#!/usr/bin/env python3
"""Estimate how many states / pairs get filtered by min_pair_spread thresholds."""

import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to .npy directory")
    args = parser.parse_args()

    npy_dir = os.path.abspath(args.data)
    state_ids = np.load(os.path.join(npy_dir, "state_id_tensor.npy"))
    cost = np.load(os.path.join(npy_dir, "cost_tensor.npy"))

    target_ratio = cost[:, -1] / cost[:, 0]

    order = np.argsort(state_ids, kind="mergesort")
    sorted_sids = state_ids[order]
    breaks = np.flatnonzero(np.diff(sorted_sids)) + 1
    groups = np.split(order, breaks)
    unique_states = sorted_sids[np.concatenate(([0], breaks))]

    thresholds = [0.005, 0.02]

    total_states = len(unique_states)
    total_samples = len(state_ids)
    print(f"Dataset: {npy_dir}")
    print(f"Total samples: {total_samples}")
    print(f"Total states: {total_states}")
    print()

    state_spreads = []
    all_pair_diffs = []
    total_pairs = 0
    states_with_ge2 = 0

    for i, grp in enumerate(groups):
        ratios = target_ratio[grp]
        n = len(ratios)
        spread = ratios.max() - ratios.min()
        state_spreads.append(spread)

        if n >= 2:
            states_with_ge2 += 1
            diffs = np.abs(ratios[:, None] - ratios[None, :])
            upper_idx = np.triu_indices(n, k=1)
            pair_diffs = diffs[upper_idx]
            all_pair_diffs.append(pair_diffs)
            total_pairs += len(pair_diffs)

    state_spreads = np.array(state_spreads)
    all_pair_diffs = np.concatenate(all_pair_diffs) if all_pair_diffs else np.array([])

    print(f"States with >= 2 trails: {states_with_ge2}")
    print(f"Total pairs: {total_pairs}")
    print()

    print("=" * 70)
    print("State-level spread statistics:")
    print(f"  min:    {state_spreads.min():.6f}")
    print(f"  median: {np.median(state_spreads):.6f}")
    print(f"  mean:   {state_spreads.mean():.6f}")
    print(f"  max:    {state_spreads.max():.6f}")
    print(f"  std:    {state_spreads.std():.6f}")
    for pct in [10, 25, 50, 75, 90]:
        print(f"  p{pct:02d}:    {np.percentile(state_spreads, pct):.6f}")
    print()

    if len(all_pair_diffs) > 0:
        print("=" * 70)
        print("Pair-level |ratio_i - ratio_j| statistics:")
        print(f"  min:    {all_pair_diffs.min():.6f}")
        print(f"  median: {np.median(all_pair_diffs):.6f}")
        print(f"  mean:   {all_pair_diffs.mean():.6f}")
        print(f"  max:    {all_pair_diffs.max():.6f}")
        for pct in [10, 25, 50, 75, 90, 95, 99]:
            print(f"  p{pct:02d}:    {np.percentile(all_pair_diffs, pct):.6f}")
        print()

    print("=" * 70)
    print("min_pair_spread filter analysis (concordance_loss pair filtering):")
    print("-" * 70)
    for thr in thresholds:
        pairs_kept = int((all_pair_diffs > thr).sum())
        pairs_dropped = total_pairs - pairs_kept
        pct_dropped = 100.0 * pairs_dropped / total_pairs if total_pairs > 0 else 0

        states_all_dropped = 0
        for i, grp in enumerate(groups):
            ratios = target_ratio[grp]
            n = len(ratios)
            if n < 2:
                continue
            diffs = np.abs(ratios[:, None] - ratios[None, :])
            upper_idx = np.triu_indices(n, k=1)
            pair_diffs_s = diffs[upper_idx]
            if (pair_diffs_s > thr).sum() == 0:
                states_all_dropped += 1

        print(f"\n  threshold = {thr}")
        print(f"    Pairs dropped: {pairs_dropped}/{total_pairs} ({pct_dropped:.1f}%)")
        print(f"    Pairs kept:    {pairs_kept}/{total_pairs} ({100-pct_dropped:.1f}%)")
        print(f"    States with ALL pairs dropped (no valid pairs): "
              f"{states_all_dropped}/{states_with_ge2} "
              f"({100.0*states_all_dropped/states_with_ge2:.1f}%)")

    print()
    print("=" * 70)
    print("min_spread filter analysis (state-level, pre-training filter):")
    print("-" * 70)
    for thr in thresholds:
        dropped = int((state_spreads < thr).sum())
        print(f"\n  threshold = {thr}")
        print(f"    States dropped: {dropped}/{total_states} ({100.0*dropped/total_states:.1f}%)")
        print(f"    States kept:    {total_states-dropped}/{total_states} "
              f"({100.0*(total_states-dropped)/total_states:.1f}%)")


if __name__ == "__main__":
    main()
