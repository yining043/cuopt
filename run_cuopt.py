# -*- coding: utf-8 -*-
"""
cuOpt CVRP solver and log plotter.

Subcommands:
  solve  - Run cuOpt solver on CVRP instances (optionally auto-plot).
  plot   - Plot cost curves from solver log files.

Examples:
  python run_cuopt.py solve --n_instances 10 --time_limit 5 --plot
  python run_cuopt.py solve --n_instances 10 --time_limit 5
  python run_cuopt.py plot run1.log run2.log --ymin 1200 --ymax 1800
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
import math
import os
import random
import re
import socket
import subprocess
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import threading
import time

import matplotlib
matplotlib.use("Agg")  # save only, no GUI
import matplotlib.pyplot as plt

# ── Regex patterns for log parsing ──────────────────────────────────

# Match "cost before: X, cost after: Y"
COST_PAIR_RE = re.compile(
    r"cost before:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*"
    r"cost after:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)

# Match "Total time used: T ms, offset: O ms"
TIME_RE = re.compile(
    r"Total time used:\s*(\d+)\s*ms\s*,\s*offset:\s*(\d+)\s*ms",
    re.IGNORECASE,
)
# Match "[iter #N] offset: X ms" from C++ solver (local_search.cu)
OFFSET_LINE_RE = re.compile(r"\[iter\s*#\d+\]\s*offset:\s*(\d+)\s*ms")
# Match "[search #N] ..." from C++ (trial boundary; same as # --- break ---)
SEARCH_HEADER_RE = re.compile(r"\[search\s*#\s*\d+\]")
# Match "wall_clock: X.XXXX" appended to cost lines
WALL_CLOCK_RE = re.compile(r"wall_clock:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
# Match "is_feasible: 0|1" appended to cost lines (added 2026-05-06 to preserve feasibility on log reload)
IS_FEASIBLE_RE = re.compile(r"is_feasible:\s*([01])")
# Match similarity trace lines from C++ logging
SIM_TRACE_VALUE_RE = re.compile(
    r"\[SIM_TRACE\]\[VALUE\]\s*mode=(\S+)\s*sim=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)
SIM_TRACE_THRESHOLD_RE = re.compile(
    r"\[SIM_TRACE\]\[THRESHOLD\]\s*loc=(\S+)\s*metric=(\S+)\s*"
    r"sim=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*"
    r"threshold=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*decision=(\S+)"
)
# Match "Best Known: X.XX" from solve log (HGS reference cost)
BEST_KNOWN_RE = re.compile(r"Best Known:\s*([\d.]+)", re.IGNORECASE)

# ── Log parsing & series building ───────────────────────────────────

def parse_points(text: str):
    """
    Scan the log sequentially and build a point sequence:
      Each cost before/after pair -> new point (after value).
      If followed by time/offset -> bind to that point (first match only).
      Lines containing [INFEASIBLE] are tagged; only feasible costs update best_so_far.
    Returns: [{'after': float, 'best_so_far': float, 'time': float|None, 'offset': float|None, 'is_feasible': bool}, ...]
    """
    pts = []
    best_so_far = float('inf')

    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln == "# --- break ---" or SEARCH_HEADER_RE.search(ln):
            pts.append({
                'after': float('nan'),
                'best_so_far': float('nan'),
                'time': None,
                'offset': None,
                'is_feasible': True,
            })
            continue
        m_cost = COST_PAIR_RE.search(ln)
        if m_cost:
            # Prefer the explicit "is_feasible: 0|1" field if present (newer logs).
            # Fall back to "[INFEASIBLE]" tag (legacy, never actually written but kept
            # for back-compat). Default True if neither is present (oldest logs).
            m_feas = IS_FEASIBLE_RE.search(ln)
            if m_feas is not None:
                is_feasible = (m_feas.group(1) == '1')
            else:
                is_feasible = '[INFEASIBLE]' not in ln
            after_cost = float(m_cost.group(2))
            if is_feasible:
                best_so_far = min(best_so_far, after_cost)
            m_wc = WALL_CLOCK_RE.search(ln)
            pts.append({
                'after': after_cost,
                'best_so_far': best_so_far,
                'time': None,
                'offset': None,
                'wall_clock': float(m_wc.group(1)) if m_wc else None,
                'is_feasible': is_feasible,
            })
            continue
        m_time = TIME_RE.search(ln)
        if m_time and pts:
            if pts[-1]['time'] is None and pts[-1]['offset'] is None:
                pts[-1]['time'] = float(m_time.group(1))
                pts[-1]['offset'] = float(m_time.group(2))
    return pts


def build_series(pts, break_mode: str, max_segments, xshift: float,
                 data_key: str = 'after', use_percentage: bool = False):
    """
    Split points into segments by break_mode; keep only the first max_segments (None means all).
    X-axis is cumulative iteration i=1..N plus xshift; NaN is inserted before each new segment.
    If use_percentage=True, x-axis is converted to percentage (0-100%).
    Returns (xs, ys).
    """
    xs, ys = [], []
    seg_count = 0
    prev_key = None  # used for segment detection
    i_iter = 0       # cumulative iteration count (across segments)

    def key_of(p):
        if break_mode == "time":
            return p['time']
        elif break_mode == "offset":
            return p['offset']
        return None

    def is_new_seg(k, pk, sc, ii):
        if break_mode == "none":
            return sc == 0 and ii == 0
        if pk is None:
            return True
        if k is not None and pk is not None and k < pk:
            return True
        return False

    # Pre-compute total iterations when using percentage mode
    total_iterations = 0
    if use_percentage:
        t_sc, t_pk, t_ii = 0, None, 0
        for p in pts:
            k = key_of(p)
            if is_new_seg(k, t_pk, t_sc, t_ii):
                t_sc += 1
                if max_segments is not None and max_segments > 0 and t_sc > max_segments:
                    break
            t_pk = k if k is not None else t_pk
            t_ii += 1
        total_iterations = t_ii

    for p in pts:
        k = key_of(p)

        if is_new_seg(k, prev_key, seg_count, i_iter):
            if seg_count > 0:
                xs.append(math.nan)
                ys.append(math.nan)
            seg_count += 1
            if max_segments is not None and max_segments > 0 and seg_count > max_segments:
                break
        prev_key = k if k is not None else prev_key

        i_iter += 1

        if use_percentage and total_iterations > 0:
            x_value = (float(i_iter) / total_iterations) * 100.0
        else:
            x_value = float(i_iter) + xshift

        xs.append(x_value)
        ys.append(p[data_key])

    return xs, ys

# ── Plotting ────────────────────────────────────────────────────────

def _plot_one_figure(named_points, labels, break_mode, segments, xshift, dpi,
                     ymin, ymax, data_key, use_percentage, ylabel, out_path,
                     hlines=None, feasible_only=True, y_ref_cost=None):
    """Render one figure to *out_path*.

    Args:
        named_points: [(name, [point_dict, ...]), ...]
        hlines: Optional list of (value, label) for horizontal reference lines.
        feasible_only: If True (default), only use points with is_feasible=True (applies to all data_keys).
    Returns True if anything was plotted.
    """
    plt.figure(figsize=(9.5, 5.5))
    any_plotted = False

    for idx, (name, pts) in enumerate(named_points):
        if not pts:
            if not use_percentage and data_key == 'after':
                print(f"[INFO] '{name}' has no data points, skipping.")
            continue
        if feasible_only:
            pts = [p for p in pts if p.get('is_feasible', True)]
            if not pts:
                continue
        xs, ys = build_series(
            pts,
            break_mode=break_mode,
            max_segments=(segments if segments > 0 else None),
            xshift=xshift,
            data_key=data_key,
            use_percentage=use_percentage,
        )
        if y_ref_cost is not None and y_ref_cost != 0:
            ys = [
                ((float(y) / float(y_ref_cost)) * 100.0)
                if y is not None and not (isinstance(y, float) and math.isnan(y))
                else y
                for y in ys
            ]
        if xs:
            lab = labels[idx] if (labels and idx < len(labels)) else name
            plt.plot(xs, ys, marker=".", linewidth=1.2, label=lab)
            any_plotted = True

    if not any_plotted:
        print(f"No plottable {ylabel} data.")
        plt.close()
        return False

    # Draw HGS reference lines
    if hlines:
        for val, lab in hlines:
            plt.axhline(y=val, color='red', linestyle='--', linewidth=1.5, label=lab)

    if use_percentage:
        xlabel = "Iteration Percentage (%)"
    else:
        xlabel = "Cumulative Iteration" + (f" (+{xshift:g} shift)" if abs(xshift) > 0 else "")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(title="log")
    plt.tight_layout()

    if ymin is not None or ymax is not None:
        y0, y1 = plt.ylim()
        plt.ylim(ymin if ymin is not None else y0, ymax if ymax is not None else y1)

    plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close()
    print(f"Saved {ylabel} figure to: {out_path}")
    return True


def _plot_best_so_far_interval(config_curves, out_path, out_path_pct,
                               dpi=160, ymin=None, ymax=None, hgs_cost=None,
                               colors=None, feasible_only=True):
    """Plot best_so_far as mean ± std band per config (for multiple runs).

    config_curves: dict config_name -> list of [point_dict, ...] (one list per run).
    feasible_only: If True (default), only use points with is_feasible=True.
    """
    import numpy as np

    def _series_from_points(pts, data_key):
        ys = []
        last = None
        for p in pts:
            v = p.get(data_key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                if last is not None:
                    ys.append(last)
            else:
                last = v
                ys.append(v)
        return np.array(ys) if ys else np.array([], dtype=float)

    def _filter_feasible(pts):
        return [p for p in pts if p.get('is_feasible', True)] if feasible_only else pts

    colors = colors or ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for idx, (cfg_name, run_list) in enumerate(config_curves.items()):
        if not run_list:
            continue
        # (n_runs, max_len), pad with last value
        series_list = [_series_from_points(_filter_feasible(pts), 'best_so_far') for pts in run_list]
        max_len = max(len(s) for s in series_list)
        if max_len == 0:
            continue
        padded = np.full((len(series_list), max_len), np.nan)
        for i, s in enumerate(series_list):
            if len(s) > 0:
                padded[i, :len(s)] = s
                padded[i, len(s):] = s[-1]
        mean_y = np.nanmean(padded, axis=0)
        std_y = np.nanstd(padded, axis=0)
        x = np.arange(1, max_len + 1, dtype=float)
        c = colors[idx % len(colors)]
        ax.plot(x, mean_y, color=c, linewidth=1.5, label=cfg_name)
        ax.fill_between(x, mean_y - std_y, mean_y + std_y, color=c, alpha=0.25)

    if hgs_cost is not None:
        ax.axhline(hgs_cost, color='red', linestyle='--', linewidth=1.5, label="HGS")
    ax.set_xlabel("Cumulative Iteration")
    ax.set_ylabel("Cost")
    ax.legend()
    ax.grid(True)
    if ymin is not None or ymax is not None:
        y0, y1 = ax.get_ylim()
        ax.set_ylim(ymin if ymin is not None else y0, ymax if ymax is not None else y1)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close()
    print(f"Saved Best So Far (interval) to: {out_path}")

    if out_path_pct:
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        for idx, (cfg_name, run_list) in enumerate(config_curves.items()):
            if not run_list:
                continue
            series_list = [_series_from_points(_filter_feasible(pts), 'best_so_far') for pts in run_list]
            max_len = max(len(s) for s in series_list)
            if max_len == 0:
                continue
            padded = np.full((len(series_list), max_len), np.nan)
            for i, s in enumerate(series_list):
                if len(s) > 0:
                    padded[i, :len(s)] = s
                    padded[i, len(s):] = s[-1]
            mean_y = np.nanmean(padded, axis=0)
            std_y = np.nanstd(padded, axis=0)
            # Normalize each config's x-axis by its own total iterations so all curves end at 100%.
            x = (np.arange(1, max_len + 1, dtype=float) / float(max_len)) * 100.0
            c = colors[idx % len(colors)]
            ax.plot(x, mean_y, color=c, linewidth=1.5, label=cfg_name)
            ax.fill_between(x, mean_y - std_y, mean_y + std_y, color=c, alpha=0.25)
        if hgs_cost is not None:
            ax.axhline(hgs_cost, color='red', linestyle='--', linewidth=1.5, label="HGS")
        ax.set_xlabel("Iteration Progress (%)")
        ax.set_ylabel("Cost")
        ax.legend()
        ax.grid(True)
        if ymin is not None or ymax is not None:
            y0, y1 = ax.get_ylim()
            ax.set_ylim(ymin if ymin is not None else y0, ymax if ymax is not None else y1)
        plt.tight_layout()
        plt.savefig(out_path_pct, bbox_inches="tight", dpi=dpi)
        plt.close()
        print(f"Saved Best So Far % (interval) to: {out_path_pct}")


def _plot_best_so_far_wallclock(config_curves, out_path, out_path_pct,
                                time_limit, dpi=160, ymin=None, ymax=None,
                                hgs_cost=None, colors=None, feasible_only=True,
                                n_bins=200):
    """Plot best_so_far vs wall-clock seconds (mean ± std), truncated at time_limit.

    This shows the "with overhead" view: configs that spend time on neural
    network forward passes progress more slowly on the wall-clock x-axis.
    """
    import numpy as np
    colors = colors or ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    def _filter_feasible(pts):
        return [p for p in pts if p.get('is_feasible', True)] if feasible_only else pts

    t_edges = np.linspace(0, time_limit, n_bins + 1)
    t_centers = (t_edges[:-1] + t_edges[1:]) / 2.0

    def _wallclock_series(pts):
        """Return best_so_far sampled at t_centers using wall_clock timestamps."""
        fp = _filter_feasible(pts)
        fp = [p for p in fp if p.get('wall_clock') is not None and not math.isnan(p.get('after', float('nan')))]
        if not fp:
            return None
        times = np.array([p['wall_clock'] for p in fp])
        costs = np.array([p['best_so_far'] for p in fp])
        sampled = np.full(n_bins, np.nan)
        for bi in range(n_bins):
            mask = times <= t_edges[bi + 1]
            if mask.any():
                sampled[bi] = costs[mask][-1]
        return sampled

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for idx, (cfg_name, run_list) in enumerate(config_curves.items()):
        if not run_list:
            continue
        series_list = [_wallclock_series(pts) for pts in run_list]
        series_list = [s for s in series_list if s is not None]
        if not series_list:
            continue
        padded = np.stack(series_list, axis=0)
        mean_y = np.nanmean(padded, axis=0)
        std_y = np.nanstd(padded, axis=0)
        valid = ~np.isnan(mean_y)
        c = colors[idx % len(colors)]
        ax.plot(t_centers[valid], mean_y[valid], color=c, linewidth=1.5, label=cfg_name)
        ax.fill_between(t_centers[valid], (mean_y - std_y)[valid], (mean_y + std_y)[valid],
                        color=c, alpha=0.25)

    if hgs_cost is not None:
        ax.axhline(hgs_cost, color='red', linestyle='--', linewidth=1.5, label="HGS")
    ax.set_xlabel("Wall-clock Time (s)")
    ax.set_ylabel("Cost")
    ax.set_title("With Neural Network Overhead")
    ax.legend()
    ax.grid(True)
    if ymin is not None or ymax is not None:
        y0, y1 = ax.get_ylim()
        ax.set_ylim(ymin if ymin is not None else y0, ymax if ymax is not None else y1)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close()
    print(f"Saved Best So Far (wall-clock) to: {out_path}")

    if out_path_pct:
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        for idx, (cfg_name, run_list) in enumerate(config_curves.items()):
            if not run_list:
                continue
            series_list = [_wallclock_series(pts) for pts in run_list]
            series_list = [s for s in series_list if s is not None]
            if not series_list:
                continue
            padded = np.stack(series_list, axis=0)
            mean_y = np.nanmean(padded, axis=0)
            std_y = np.nanstd(padded, axis=0)
            valid = ~np.isnan(mean_y)
            c = colors[idx % len(colors)]
            x_pct = (t_centers / max(float(time_limit), 1e-9)) * 100.0
            ax.plot(x_pct[valid], mean_y[valid], color=c, linewidth=1.5, label=cfg_name)
            ax.fill_between(x_pct[valid], (mean_y - std_y)[valid], (mean_y + std_y)[valid],
                            color=c, alpha=0.25)
        if hgs_cost is not None:
            ax.axhline(hgs_cost, color='red', linestyle='--', linewidth=1.5, label="HGS")
        ax.set_xlabel("Wall-clock Progress (%)")
        ax.set_ylabel("Cost")
        ax.set_title("With Neural Network Overhead")
        ax.legend()
        ax.grid(True)
        if ymin is not None or ymax is not None:
            y0, y1 = ax.get_ylim()
            ax.set_ylim(ymin if ymin is not None else y0, ymax if ymax is not None else y1)
        plt.tight_layout()
        plt.savefig(out_path_pct, bbox_inches="tight", dpi=dpi)
        plt.close()
        print(f"Saved Best So Far % (wall-clock) to: {out_path_pct}")


def _write_endpoint_summary_table(config_curves, out_dir, time_limit, hgs_val=None,
                                  feasible_only=True):
    """Write per-config endpoint mean/std table for each plot.

    Endpoints:
      - best_so_far_iter:   last best_so_far per run (= each run's converged best)
      - best_so_far_wc:     best_so_far at latest wall_clock <= time_limit per run
      - after_last:         last 'after' cost per run
    """
    import csv
    import numpy as np

    def _filt(pts):
        return [p for p in pts if p.get('is_feasible', True)] if feasible_only else pts

    def _last_valid(values):
        for v in reversed(values):
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                return float(v)
        return None

    def _wc_at_limit(pts, limit):
        cur = None
        for p in pts:
            wc = p.get('wall_clock')
            v = p.get('best_so_far')
            if wc is None or v is None:
                continue
            if isinstance(v, float) and math.isnan(v):
                continue
            if wc > limit:
                break
            cur = float(v)
        return cur

    rows = []
    for cfg_name, run_list in config_curves.items():
        bsf_iter, bsf_wc, after_last = [], [], []
        for pts in run_list:
            fp = _filt(pts)
            v_bsf = _last_valid([p.get('best_so_far') for p in fp])
            if v_bsf is not None:
                bsf_iter.append(v_bsf)
            v_wc = _wc_at_limit(fp, time_limit)
            if v_wc is not None:
                bsf_wc.append(v_wc)
            v_after = _last_valid([p.get('after') for p in fp])
            if v_after is not None:
                after_last.append(v_after)

        def _mean_std(xs):
            if not xs:
                return (float('nan'), float('nan'))
            arr = np.asarray(xs, dtype=float)
            return (float(arr.mean()), float(arr.std(ddof=0)))

        bsf_iter_m, bsf_iter_s = _mean_std(bsf_iter)
        bsf_wc_m, bsf_wc_s = _mean_std(bsf_wc)
        after_m, after_s = _mean_std(after_last)
        gap_pct = (bsf_iter_m - hgs_val) / hgs_val * 100.0 if hgs_val else float('nan')

        rows.append({
            'config': cfg_name,
            'n_runs': len(run_list),
            'best_so_far_iter_mean': bsf_iter_m,
            'best_so_far_iter_std': bsf_iter_s,
            'best_so_far_wallclock_mean': bsf_wc_m,
            'best_so_far_wallclock_std': bsf_wc_s,
            'after_last_mean': after_m,
            'after_last_std': after_s,
            'gap_to_hgs_pct': gap_pct,
        })

    csv_path = os.path.join(out_dir, "endpoint_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            'config', 'n_runs',
            'best_so_far_iter_mean', 'best_so_far_iter_std',
            'best_so_far_wallclock_mean', 'best_so_far_wallclock_std',
            'after_last_mean', 'after_last_std', 'gap_to_hgs_pct',
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    md_path = os.path.join(out_dir, "endpoint_summary.md")
    headers = [
        "config", "n", "bsf_iter (mean±std)", "bsf_wc@tlim (mean±std)",
        "after_last (mean±std)", "gap_HGS %",
    ]
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for r in rows:
        lines.append(
            "| {config} | {n} | {bi_m:.4f} ± {bi_s:.4f} | {bw_m:.4f} ± {bw_s:.4f} | "
            "{a_m:.4f} ± {a_s:.4f} | {gap:.3f} |".format(
                config=r['config'], n=r['n_runs'],
                bi_m=r['best_so_far_iter_mean'], bi_s=r['best_so_far_iter_std'],
                bw_m=r['best_so_far_wallclock_mean'], bw_s=r['best_so_far_wallclock_std'],
                a_m=r['after_last_mean'], a_s=r['after_last_std'],
                gap=r['gap_to_hgs_pct'],
            )
        )
    if hgs_val is not None:
        lines.append(f"\nHGS reference: {hgs_val:.4f}  |  wall-clock cutoff: {time_limit:.1f}s")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\nEndpoint summary:")
    print("\n".join(lines))
    print(f"Saved endpoint summary to: {csv_path} / {md_path}")


def _points_to_trials_by_break(points):
    """Split points into trials (NaN = break). One trial = one C++ search."""
    trials, cur = [], []
    for p in points:
        if math.isnan(p.get('after', 0)):
            if cur:
                trials.append(cur)
                cur = []
        else:
            cur.append(p)
    if cur:
        trials.append(cur)
    return trials


def plot_cost_curve_by_trial_duplicates(named_points, out_path, dpi=150, hgs_cost=None, cost_ymax=None, min_cost=None):
    """
    Plot cost evolution with one line per trial; color by duplicate local optima.
    Trials that end at the same basin (same solution_hash or same cost) share a color; unique trials are black.
    If min_cost is set, trials with best cost > min_cost are not counted as distinct basins (one group).
    cost_ymax: if set, cap y-axis (cost) at this value so local search trajectory is easier to read.
    """
    import numpy as np
    _ABOVE_MIN_COST_KEY = "_above_min_cost_"
    all_pts = []
    for _label, pts in named_points:
        all_pts.extend(pts)
        all_pts.append({'after': float('nan'), 'best_so_far': float('nan')})
    trials = _points_to_trials_by_break(all_pts)
    if not trials:
        return
    # Build duplicate_groups: basin_key -> [trial_idx, ...]; cost > min_cost -> same key (not a basin).
    optimum_id_map = {}
    duplicate_groups = {}
    for t_idx, t in enumerate(trials):
        last = t[-1]
        best = last.get('best_so_far')
        if min_cost is not None and best is not None and best > min_cost:
            key = _ABOVE_MIN_COST_KEY
        else:
            key = last.get('solution_hash')
            if key is None:
                key = last.get('best_so_far')
        if key not in optimum_id_map:
            optimum_id_map[key] = len(optimum_id_map)
            duplicate_groups[key] = [t_idx]
        else:
            duplicate_groups[key].append(t_idx)
    # Trial index -> basin key (for legend)
    trial_to_key = {}
    for key, indices in duplicate_groups.items():
        for i in indices:
            trial_to_key[i] = key
    # Trial colors: unique = black, duplicate group = same color
    trial_colors = {}
    unique_color = 'black'
    duplicate_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    duplicate_colors = [tuple(c[:3]) for c in duplicate_colors if np.sum(c[:3]) >= 0.3]
    if not duplicate_colors:
        duplicate_colors = [(0.2, 0.6, 0.8)]
    color_idx = 0
    for key, indices in duplicate_groups.items():
        if len(indices) == 1:
            trial_colors[indices[0]] = unique_color
        else:
            c = duplicate_colors[color_idx % len(duplicate_colors)]
            for i in indices:
                trial_colors[i] = c
            color_idx += 1
    # Build (global_iter, cost) per point per trial
    fig, ax = plt.subplots(figsize=(12, 6))
    labeled_colors = set()
    global_iter = 0
    for t_idx, t in enumerate(trials):
        iters, costs = [], []
        for p in t:
            iters.append(global_iter)
            costs.append(p['after'])
            global_iter += 1
        color = trial_colors.get(t_idx, unique_color)
        hashable_c = tuple(color) if isinstance(color, (tuple, list)) else color
        if hasattr(color, 'tolist'):
            hashable_c = tuple(color.tolist())
        label = None
        if hashable_c not in labeled_colors:
            key = trial_to_key.get(t_idx)
            indices = duplicate_groups.get(key, [t_idx])
            if color == unique_color or len(indices) <= 1:
                label = 'Unique trials'
            else:
                label = f"Trials {sorted(indices)} (duplicate)"
            labeled_colors.add(hashable_c)
        ax.plot(iters, costs, marker='.', linestyle='-', linewidth=1.2, markersize=3,
                color=color, label=label, alpha=0.7)
    if hgs_cost is not None:
        ax.axhline(y=hgs_cost, color='red', linestyle='--', linewidth=1.5, label='HGS')
    best_cost = min(p['after'] for t in trials for p in t)
    ax.set_xlabel('Global Iteration')
    ax.set_ylabel('Cost')
    ax.set_title(f'Cost by trial (duplicate local optima same color) | Best: {best_cost:.4f}')
    if cost_ymax is not None:
        all_costs = [p['after'] for t in trials for p in t]
        y_min = min(all_costs) * 0.98 if all_costs else 0.0
        ax.set_ylim(y_min, float(cost_ymax))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7, ncol=1)
    fig.subplots_adjust(right=0.78)
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved duplicate-trial curve to: {out_path}")


def plot_from_points(named_points, labels=None, break_mode="none", segments=0,
                     xshift=0.0, dpi=160, ymin=None, ymax=None, out_dir=".",
                     hgs_costs=None, feasible_only=True):
    """Generate all 4 standard figures from pre-parsed point lists.

    Args:
        hgs_costs: Optional dict {instance_index: cost} for HGS reference lines.
        feasible_only: If True, best_so_far / best_so_far_pct use only points with is_feasible=True.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Build HGS horizontal lines
    hlines = None
    if hgs_costs:
        hlines = []
        for inst_idx, cost in sorted(hgs_costs.items()):
            lab = "HGS" if len(hgs_costs) == 1 else f"HGS inst{inst_idx}"
            hlines.append((cost, lab))

    common = dict(
        named_points=named_points, labels=labels,
        break_mode=break_mode, segments=segments, xshift=xshift,
        dpi=dpi, ymin=ymin, ymax=ymax, hlines=hlines,
        feasible_only=feasible_only,
    )
    _plot_one_figure(**common, data_key='after',       use_percentage=False,
                     ylabel="Cost",       out_path=os.path.join(out_dir, "after.png"))
    _plot_one_figure(**common, data_key='best_so_far', use_percentage=False,
                     ylabel="Cost", out_path=os.path.join(out_dir, "best_so_far.png"))
    _plot_one_figure(**common, data_key='after',       use_percentage=True,
                     ylabel="Cost",       out_path=os.path.join(out_dir, "after_pct.png"))
    _plot_one_figure(**common, data_key='best_so_far', use_percentage=True,
                     ylabel="Cost", out_path=os.path.join(out_dir, "best_so_far_pct.png"))

# ── Solver utilities ────────────────────────────────────────────────

def pairwise_euclidean_distance(x):
    """Batch pairwise Euclidean distance. x: (B, N, 2) -> (B, N, N)."""
    import torch
    x_square = (x ** 2).sum(dim=2, keepdim=True)
    dist_square = x_square + x_square.transpose(1, 2) - 2 * torch.bmm(x, x.transpose(1, 2))
    return torch.sqrt(torch.clamp(dist_square, min=1e-9))


def load_pkl_data(problem_path, solution_path, start_index=0, episode=1):
    """Load CVRP instances and HGS solutions from pkl files.

    Args:
        problem_path: Path to cvrp pkl (e.g. cvrp100_uniform.pkl).
            Each entry: (depot[[x,y]], customers[[x,y],...], demands[...], capacity).
        solution_path: Path to HGS solution pkl (e.g. hgs_cvrp100_uniform.pkl).
            Each entry: (cost, route_list).
        start_index: Starting instance index.
        episode: Number of instances to load.

    Returns:
        nodes, capacities, demands, costs, node_flags
    """
    import pickle
    import torch

    with open(problem_path, "rb") as f:
        problems = pickle.load(f)
    with open(solution_path, "rb") as f:
        solutions = pickle.load(f)

    subset_p = problems[start_index : start_index + episode]
    subset_s = solutions[start_index : start_index + episode]

    all_nodes, all_caps, all_demands, all_costs = [], [], [], []

    for (depot, customers, demands, capacity), (cost, _route) in zip(subset_p, subset_s):
        coords = [depot[0]] + customers
        all_nodes.append(coords)
        all_demands.append([0] + list(demands))
        all_caps.append(capacity)
        all_costs.append(cost)

    nodes = torch.tensor(all_nodes, dtype=torch.float32)         # (episode, n+1, 2)
    demands = torch.tensor(all_demands, dtype=torch.float32)     # (episode, n+1)
    capacities = torch.tensor(all_caps, dtype=torch.float32)     # (episode,)
    costs = torch.tensor(all_costs, dtype=torch.float32)         # (episode,)
    node_flags = torch.zeros(episode, nodes.shape[1], 2)

    return nodes, capacities, demands, costs, node_flags


def load_txt_data(data_path, start_index=0, episode=1):
    """Load CVRP data from the original txt format (via load_nco_data)."""
    from load_nco_data import load_raw_data
    return load_raw_data(data_path, episode=episode, start_index=start_index)


def _solution_to_edge_hash(sol):
    """Compute stable basin hash from flat solution [0, c1, c2, 0, c3, ...].

    Converts to undirected edges and hashes with SHA1, matching utils.edges_hash.
    """
    from utils import edges_hash
    edges = set()
    for i in range(len(sol) - 1):
        u, v = int(sol[i]), int(sol[i + 1])
        if u == 0 and v == 0:
            continue
        edges.add((min(u, v), max(u, v)))
    return edges_hash(edges)


def _make_callback_class(scale=1.0, early_stop=False, early_stop_base="random",
                         embedder=None, emb_proj=None, env=None, classifier=None,
                         check_interval=1, classifier_type="threshold",
                         callback_timeout=30.0,
                         demands=None, capacity=None,
                         es_improvement_pct=None,
                         depth_predictor=None, depth_env=None):
    """Build SolverCallback class that records iteration data.

    Args:
        scale: Divide objective by this to get real cost.
        early_stop: If True, enable early stopping.
        early_stop_base: "embedding" | "structure" | "random".
        embedder: Pre-loaded SolutionEmbedder (required for embedding mode).
        emb_proj: Optional DisentangledProjection loaded from checkpoint;
                  when present, ``proj.topo(raw_emb)`` is applied before distance
                  computation so that training-time and inference-time representations match.
        env: Pre-loaded CVRPEnv with instance data (required for embedding mode).
        classifier: Dict with 'tau_embed'/'tau_struct' and 'clf_embed'/'clf_struct'.
        check_interval: Check convergence every N iterations.
        classifier_type: "threshold" (d < tau) or "lr" (LogisticRegression).
        callback_timeout: Max seconds for early-stop decision; on timeout return False (no stop) to avoid hang.
        demands: 1-D array of node demands (index 0 = depot = 0). Used for feasibility check.
        capacity: Vehicle capacity scalar. Used for feasibility check.
        es_improvement_pct: If set (e.g. 0.05), depth early stop threshold.
        depth_predictor: Dict from _load_depth_predictor (embedder, proj, advantage_head, ...).
        depth_env: CVRPEnv for depth predictor (may differ from basin ES env).
    """
    import torch
    import numpy as np
    from cuopt.routing import CustomizeEarlyStopCallback

    class SolverCallback(CustomizeEarlyStopCallback):
        """Callback that records per-iteration cost and optionally stops early."""
        def __init__(self):
            super().__init__()
            self.points = []
            self._best_so_far = float('inf')
            self._scale = scale
            self._early_stop = early_stop
            self._early_stop_base = early_stop_base
            self._embedder = embedder
            self._emb_proj = emb_proj
            self._env = env
            self._classifier = classifier
            self._check_interval = check_interval
            self._classifier_type = classifier_type
            self._demands = np.asarray(demands, dtype=np.float64) if demands is not None else None
            self._capacity = float(capacity) if capacity is not None else None
            # Cache derived constants used by per-call feasibility check.
            self._n_customers = (len(self._demands) - 1) if self._demands is not None else 0
            # Whether any ES path needs _prev_solution_flat (skip the per-call list copy otherwise).
            self._needs_prev_flat = (early_stop and early_stop_base in ("embedding", "structure")) or (depth_predictor is not None)
            # Skip the depth-ES function call entirely when no depth predictor is configured.
            self._has_depth_es = (depth_predictor is not None) and (es_improvement_pct is not None)
            # Skip the embedding/structure-ES executor wrap when not configured.
            self._has_emb_es = early_stop and (early_stop_base in ("embedding", "structure"))
            # Embedding mode: store embeddings
            self._local_optima_embs = []   # [(embedding, cost), ...]
            # Structure mode: store (pairs_set, cost) for fast similarity
            self._local_optima_sols = []   # [(pairs, cost), ...]
            # Cache current embedding to avoid re-forward when solution unchanged
            self._last_flat = None
            self._last_emb = None
            # Shared state
            self._prev_iteration = -1
            self._prev_after = None
            self._prev_solution_flat = None
            self._prev_objective = None
            self._restart_detected = False
            self.n_early_stops = 0
            self.n_iterations = 0
            self.n_trials = 1  # starts at 1 (first trial)
            self._last_es_path = None  # for debug: skip_interval | skip_no_optima | cache_hit | full
            self._cached_optima_emb = None  # torch.cat of local_optima_embs; invalidate when list grows
            self._total_callback_time_ms = 0.0  # accumulated time inside customize_early_stop (Python side)
            self._callback_timeout = callback_timeout
            self._executor = ThreadPoolExecutor(max_workers=2)
            # Depth-based early stop (NN prediction only)
            self._es_improvement_pct = es_improvement_pct
            self.n_depth_early_stops = 0
            # Depth predictor (neural network)
            self._depth_predictor = depth_predictor
            self._depth_env = depth_env
            # Wall-clock tracking
            self._wall_clock_start = None

        # ── helpers: convert solution_flat ───────────────────────────

        def _to_route_solution(self, solution_flat):
            """Convert cuOpt solution_flat to route-format [0,n1,n2,...,0,...]."""
            from helper import solution_flat_to_solution
            return solution_flat_to_solution(list(solution_flat))

        def _check_feasible_flat(self, solution_flat):
            """CVRP feasibility directly on cuopt's flat representation (no route conversion).

            solution_flat[0] = customer_count + 1 (cuopt's encoding).
            solution_flat[1:] = visit sequence; values > customer_count are vehicle markers
            (treated as depot transitions).

            Checks: capacity per route + every customer visited exactly once.
            Single pass, no list/set allocation per call.
            """
            if self._demands is None or self._capacity is None:
                return True
            n = self._n_customers
            cap = self._capacity
            demands = self._demands
            visited_count = 0
            visited_mask = 0  # bitmask up to ~256 customers fits in int; for 100, fine
            route_load = 0.0
            # Skip the leading metadata cell (solution_flat[0]).
            for i in range(1, len(solution_flat)):
                v = int(solution_flat[i])
                if v == 0 or v > n:
                    # depot / vehicle marker
                    if route_load > cap:
                        return False
                    route_load = 0.0
                else:
                    bit = 1 << v
                    if visited_mask & bit:
                        return False  # duplicate visit
                    visited_mask |= bit
                    visited_count += 1
                    route_load += demands[v]
            if route_load > cap:
                return False
            if visited_count != n:
                return False
            return True

        # ── embedding mode ───────────────────────────────────────────

        def _embed_solution(self, solution_flat):
            """Embed a single solution. Returns (emb, from_cache). Caches when solution unchanged.

            If a DisentangledProjection was loaded from the same checkpoint,
            ``proj.topo(raw_emb)`` is applied automatically so that the distance
            space matches training.
            """
            flat = list(solution_flat)
            if self._last_flat is not None and len(self._last_flat) == len(flat):
                if all(a == b for a, b in zip(self._last_flat, flat)):
                    return self._last_emb, True
            sol = self._to_route_solution(solution_flat)
            h = "_cb_tmp"
            self._env._basin_info[h] = {"solution": sol}
            ctx = self._env.prepare_from_hashes([h])
            with torch.no_grad():
                raw = self._embedder(ctx, self._env)
                emb = self._emb_proj.topo(raw) if self._emb_proj is not None else raw
            del self._env._basin_info[h]
            self._last_flat = flat
            self._last_emb = emb
            return emb, False

        def _add_local_optimum_emb(self, solution_flat, objective):
            emb, _ = self._embed_solution(solution_flat)
            self._local_optima_embs.append((emb, objective / self._scale))
            self._cached_optima_emb = None  # invalidate so next _check_convergence_emb recomputes

        def _check_convergence_emb(self, current_emb):
            if not self._local_optima_embs:
                return False
            if self._cached_optima_emb is None:
                self._cached_optima_emb = torch.cat([e for e, _ in self._local_optima_embs], dim=0)
            dists = torch.cdist(current_emb, self._cached_optima_emb).squeeze(0)  # (K,)
            if self._classifier_type == "lr":
                # GPU prediction to avoid .cpu().numpy() sync
                coef = self._classifier["_clf_embed_coef"].to(dists.device)
                intercept = self._classifier["_clf_embed_intercept"].to(dists.device)
                logits = dists.unsqueeze(-1) @ coef.T + intercept
                preds = logits.squeeze(-1) >= 0
                return bool(preds.any())
            else:  # threshold
                return bool((dists < self._classifier["tau_embed"]).any())

        def _embedding_early_stop(self, solution_flat, objective, iteration):
            if self._restart_detected and self._prev_solution_flat is not None:
                self._add_local_optimum_emb(self._prev_solution_flat, self._prev_objective)
                self._restart_detected = False
            self._prev_solution_flat = list(solution_flat)
            self._prev_objective = objective
            if iteration % self._check_interval != 0:
                self._last_es_path = "skip_interval"
                return False
            if not self._local_optima_embs:
                self._last_es_path = "skip_no_optima"
                return False
            current_emb, from_cache = self._embed_solution(solution_flat)
            self._last_es_path = "cache_hit" if from_cache else "full"
            if self._check_convergence_emb(current_emb):
                self.n_early_stops += 1
                return True
            return False

        # ── structure (broken-pairs) mode ────────────────────────────
        # Store (pairs_set, cost) so we only compute current solution's pairs once per check.

        def _add_local_optimum_struct(self, solution_flat, objective):
            from helper import solution_to_pairs
            sol = self._to_route_solution(solution_flat)
            pairs = solution_to_pairs(sol)
            self._local_optima_sols.append((pairs, objective / self._scale))

        def _check_convergence_struct(self, current_sol):
            from helper import solution_to_pairs, broken_pairs_ratio_from_pairs
            if not self._local_optima_sols:
                return False
            pairs_a = solution_to_pairs(current_sol)
            if not pairs_a:
                return False
            if self._classifier_type == "lr":
                dists_list = [broken_pairs_ratio_from_pairs(pairs_a, p) for p, _ in self._local_optima_sols]
                device = self._classifier["_clf_struct_coef"].device
                dists = torch.tensor(dists_list, dtype=torch.float32, device=device).unsqueeze(-1)
                coef = self._classifier["_clf_struct_coef"]
                intercept = self._classifier["_clf_struct_intercept"]
                logits = dists @ coef.T + intercept
                preds = logits.squeeze(-1) >= 0
                return bool(preds.any())
            else:  # threshold
                tau = self._classifier["tau_struct"]
                for pairs_b, _ in self._local_optima_sols:
                    if broken_pairs_ratio_from_pairs(pairs_a, pairs_b) < tau:
                        return True
                return False

        def _structure_early_stop(self, solution_flat, objective, iteration):
            if self._restart_detected and self._prev_solution_flat is not None:
                self._add_local_optimum_struct(self._prev_solution_flat, self._prev_objective)
                self._restart_detected = False
            self._prev_solution_flat = list(solution_flat)
            self._prev_objective = objective
            if iteration % self._check_interval == 0 and self._local_optima_sols:
                current_sol = self._to_route_solution(solution_flat)
                if self._check_convergence_struct(current_sol):
                    self.n_early_stops += 1
                    return True
            return False

        def _run_early_stop_decision(self, solution_flat, objective, iteration):
            """Run early-stop decision (may be slow); used inside thread with timeout."""
            if self._early_stop_base == "embedding":
                return self._embedding_early_stop(solution_flat, objective, iteration)
            if self._early_stop_base == "structure":
                return self._structure_early_stop(solution_flat, objective, iteration)
            if self._early_stop_base == "random":
                return random.random() < 0.1
            return False

        # ── depth (NN prediction) mode ─────────────────────────────

        def _predict_gap_pct_nn(self, solution_flat):
            """Use depth predictor NN to predict gap% for current solution."""
            dp = self._depth_predictor
            dep_env = self._depth_env
            if dp is None or dep_env is None:
                return None
            dep_embedder = dp["embedder"]
            proj = dp["proj"]
            adv_head = dp["advantage_head"]
            gap_fn = dp["ordinal_probs_to_gap_pct"]

            sol = self._to_route_solution(solution_flat)
            h = "_depth_tmp"
            dep_env._basin_info[h] = {"solution": sol}
            ctx = dep_env.prepare_from_hashes([h])
            with torch.no_grad():
                emb_raw = dep_embedder(ctx, dep_env)
                emb_depth = proj.depth(emb_raw) if proj is not None else emb_raw
                logits = adv_head(emb_depth)
                probs = torch.sigmoid(logits)
                pred_gap = gap_fn(probs)
            del dep_env._basin_info[h]
            return pred_gap.item()

        def _check_depth_early_stop(self, val, iteration):
            """Early stop if NN-predicted local optimum won't beat best_so_far by >= es_improvement_pct."""
            if self._depth_predictor is None:
                return False
            if self._es_improvement_pct is None:
                return False
            if iteration < self._check_interval:
                return False
            if iteration % self._check_interval != 0:
                return False
            if self._best_so_far == float('inf'):
                return False

            target = self._best_so_far * (1.0 - self._es_improvement_pct)

            if val <= target:
                return False

            gap_pct = self._predict_gap_pct_nn(self._prev_solution_flat or [])
            if gap_pct is None:
                return False
            predicted_final = val / (1.0 + gap_pct / 100.0)
            if predicted_final > target:
                self.n_depth_early_stops += 1
                return True
            return False

        # ── main callback ────────────────────────────────────────────

        def customize_early_stop(self, solution_flat, objective, num_routes, iteration, phase=0):
            t0 = time.perf_counter()
            if self._wall_clock_start is None:
                self._wall_clock_start = t0
            val = objective / self._scale

            # cpp side now passes solution_flat as PyBytes (raw int buffer) to avoid
            # allocating ~200 PyLong objects per call. Decode as zero-copy numpy view.
            # Backward-compat: if it's still a list/sequence (older .so), pass through.
            if isinstance(solution_flat, (bytes, bytearray, memoryview)):
                solution_flat = np.frombuffer(solution_flat, dtype=np.int32)

            self.n_iterations += 1

            # One trial = one C++ search ([search #N]). C++ resets iter per search, so iteration drops when a new search starts.
            if self._prev_iteration >= 0 and iteration < self._prev_iteration:
                self.n_trials += 1
                self.points.append({
                    'after': float('nan'),
                    'best_so_far': float('nan'),
                    'time': None,
                    'offset': None,
                    'wall_clock': t0 - self._wall_clock_start,
                })
                self._restart_detected = True
                self._prev_after = None  # first point of new trial uses before=val

            self._prev_iteration = iteration

            cost_before = self._prev_after if self._prev_after is not None else val

            # Inline feasibility on solution_flat (no route conversion, no edge hash).
            # solution_hash field is no longer populated here — its only consumer
            # (plot_cost_curve_by_trial_duplicates) is dead code.
            is_feasible = self._check_feasible_flat(solution_flat)

            if is_feasible:
                if val < self._best_so_far:
                    self._best_so_far = val

            self.points.append({
                'before': cost_before,
                'after': val,
                'best_so_far': self._best_so_far,
                'time': None,
                'offset': None,
                'wall_clock': t0 - self._wall_clock_start,
                'is_feasible': is_feasible,
            })
            self._prev_after = val
            # Skip per-call O(n) list copy unless an ES path actually reads it.
            if self._needs_prev_flat:
                self._prev_solution_flat = list(solution_flat)
            self._prev_objective = objective

            out = False
            if self._has_emb_es:
                # Only pay the executor.submit + result.timeout cost when emb/struct ES
                # is configured. The sub-function does its own interval gating; the
                # ThreadPool wrap exists purely as timeout protection for the NN forward.
                future = self._executor.submit(
                    self._run_early_stop_decision, list(solution_flat), objective, iteration
                )
                try:
                    out = future.result(timeout=self._callback_timeout)
                except (FuturesTimeoutError, TimeoutError, Exception):
                    out = False  # don't block: continue search

            # Depth-based early stop (NN prediction, independent of embedding ES).
            # Skip the function call entirely when depth ES isn't configured.
            if not out and self._has_depth_es:
                out = self._check_depth_early_stop(val, iteration)

            # Ensure _restart_detected is consumed even when _embedding_early_stop is not active
            self._restart_detected = False

            self._total_callback_time_ms += (time.perf_counter() - t0) * 1000
            return out

    return SolverCallback


def _load_embedder(checkpoint_path, device="cuda"):
    """Load pre-trained SolutionEmbedder (and DisentangledProjection if present) from checkpoint.

    Returns:
        (embedder, proj_or_None) — proj is loaded & eval'd when ``proj_state`` exists in the checkpoint,
        guaranteeing that the same projection used during training is applied at inference time.
    """
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from net import SolutionEmbedder

    model_params = {
        "problem": "CVRP",
        "embedding_dim": 128,
        "encoder_layer_num": 3,
        "supplement_feature_dim": 5,
        "depot_feature_dim": 5,
        "node_feature_dim": 6,
        "head_num": 8,
        "qkv_dim": 16,
        "hidden_dim": 128,
        "use_l2_normalize": True,
    }
    embedder = SolutionEmbedder(model_params).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "embedder_state" in ckpt:
        embedder.load_state_dict(ckpt["embedder_state"])
    elif "encoder_state" in ckpt:
        embedder.encoder.load_state_dict(ckpt["encoder_state"])
        print("WARNING: legacy checkpoint has encoder_state only; pos_encoder weights are randomly initialized")
    embedder.eval()

    proj = None
    if "proj_state" in ckpt:
        from train_basin_contrastive import DisentangledProjection
        ps = ckpt["proj_state"]
        input_dim = ps["depth_head.0.weight"].shape[1]
        depth_dim = ps["depth_head.2.weight"].shape[0]
        volume_dim = ps["volume_head.2.weight"].shape[0]
        if "topo_head.0.weight" in ps:
            topo_dim = ps["topo_head.2.weight"].shape[0]
            n_heads = 3
        else:
            topo_dim = ps["intra_head.2.weight"].shape[0]
            n_heads = 4
        proj = DisentangledProjection(input_dim, topo_dim, depth_dim, volume_dim, n_heads=n_heads).to(device)
        proj.load_state_dict(ps)
        proj.eval()
        embedder.use_l2_normalize = False
        print(f"[embedder] Loaded DisentangledProjection from checkpoint (topo_dim={topo_dim}), "
              f"disabled embedder L2 normalize (proj handles it)")

    has_proj_tag = " +proj.topo" if proj is not None else ""
    print(f"Loaded embedder from {checkpoint_path} "
          f"(stage {ckpt.get('stage', '?')}, epoch {ckpt.get('epoch', '?')}){has_proj_tag}")
    return embedder, proj


def _load_depth_predictor(checkpoint_path, device="cuda"):
    """Load embedder + DisentangledProjection + AdvantageOrdinalHead for depth prediction.

    Returns dict with keys: embedder, proj (optional), advantage_head, thresholds, overflow_midpoint.
    """
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from train_basin_contrastive import (
        DisentangledProjection,
        AdvantageOrdinalHead,
        ADVANTAGE_THRESHOLDS,
        ADVANTAGE_OVERFLOW_MIDPOINT,
        ordinal_probs_to_gap_pct,
    )

    embedder, proj = _load_embedder(checkpoint_path, device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    advantage_head = None
    if "advantage_head_state" in ckpt:
        ahs = ckpt["advantage_head_state"]
        embed_dim = ahs["feature_net.0.weight"].shape[1]
        n_thresholds = ahs["bias_deltas"].shape[0]
        advantage_head = AdvantageOrdinalHead(embed_dim, n_thresholds=n_thresholds).to(device)
        advantage_head.load_state_dict(ahs)
        advantage_head.eval()
        print(f"[depth] Loaded AdvantageOrdinalHead (embed_dim={embed_dim}, n_thresholds={n_thresholds})")
    else:
        raise ValueError(f"Checkpoint {checkpoint_path} has no advantage_head_state; need a Stage-3 checkpoint with --quality_reg_weight > 0")

    return {
        "embedder": embedder,
        "proj": proj,
        "advantage_head": advantage_head,
        "ordinal_probs_to_gap_pct": ordinal_probs_to_gap_pct,
    }


def load_embedder_and_classifier(checkpoint_path, classifier_path, problem_size, device="cuda"):
    """Load pre-trained SolutionEmbedder and convergence classifier.

    Args:
        checkpoint_path: Path to embedder checkpoint (.pt).
        classifier_path: Path to convergence classifier (.pkl).
        problem_size: Number of customers (e.g. 100).
        device: torch device.

    Returns:
        (embedder, proj_or_None, classifier_dict)
    """
    import pickle
    import torch

    embedder, proj = _load_embedder(checkpoint_path, device=device)

    with open(classifier_path, "rb") as f:
        classifier = pickle.load(f)
    print(f"Loaded classifier from {classifier_path} (tau_embed={classifier.get('tau_embed', '?'):.4f})")

    # GPU-side LR weights to avoid dists.cpu().numpy() sync in early-stop hot path
    if "clf_embed" in classifier and hasattr(classifier["clf_embed"], "coef_"):
        clf = classifier["clf_embed"]
        classifier["_clf_embed_coef"] = torch.tensor(clf.coef_, dtype=torch.float32, device=device)
        classifier["_clf_embed_intercept"] = torch.tensor(clf.intercept_, dtype=torch.float32, device=device)
    if "clf_struct" in classifier and hasattr(classifier["clf_struct"], "coef_"):
        clf = classifier["clf_struct"]
        classifier["_clf_struct_coef"] = torch.tensor(clf.coef_, dtype=torch.float32, device=device)
        classifier["_clf_struct_intercept"] = torch.tensor(clf.intercept_, dtype=torch.float32, device=device)

    return embedder, proj, classifier


def setup_env_for_instance(instance_idx, raw_nodes, raw_demand, raw_cap, device="cuda"):
    """Create and load a CVRPEnv for one CVRP instance.

    Args:
        instance_idx: Index into the batch tensors.
        raw_nodes: (N, n+1, 2) all node coords (depot first).
        raw_demand: (N, n+1) demands (depot demand=0).
        raw_cap: (N,) capacities.
        device: torch device.

    Returns:
        CVRPEnv loaded with the instance data.
    """
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from CVRPEnv import CVRPEnv

    coords = raw_nodes[instance_idx]  # (n+1, 2)
    demands = raw_demand[instance_idx]  # (n+1,)
    capacity = raw_cap[instance_idx].item()

    depot_xy = coords[0:1].unsqueeze(0).to(device)  # (1, 1, 2)
    customer_xy = coords[1:]  # (problem_size, 2)
    customer_demand = demands[1:] / capacity  # normalize by capacity
    node_xy_demand = torch.cat([
        customer_xy.unsqueeze(0),
        customer_demand.unsqueeze(0).unsqueeze(-1),
    ], dim=-1).to(device)  # (1, problem_size, 3)

    problem_size = customer_xy.shape[0]
    env = CVRPEnv(problem_size, torch.device(device))
    basin_info = {}
    env.load(depot_xy, node_xy_demand, basin_info)
    return env


def make_cuopt_format(index, raw_data_dist, raw_data_demand, raw_data_capacity, n_vehicles, scale):
    import numpy as np
    import cudf
    distance_matrix_df = cudf.DataFrame(raw_data_dist[index].numpy() * scale)
    location_demand = cudf.Series(raw_data_demand[index].numpy(), dtype=np.int32)
    vehicle_capacity = cudf.Series([raw_data_capacity[index].item()] * n_vehicles, dtype=np.int32)
    return distance_matrix_df, location_demand, vehicle_capacity


def get_cuopt_model(index, raw_data_dist, raw_data_demand, raw_data_capacity, n_vehicles, scale):
    import cudf
    from cuopt import routing
    distance_matrix_df, location_demand, vehicle_capacity = make_cuopt_format(
        index, raw_data_dist, raw_data_demand, raw_data_capacity, n_vehicles, scale
    )
    n_locations = raw_data_dist.shape[1]
    data_model = routing.DataModel(n_locations, n_vehicles)
    data_model.add_cost_matrix(distance_matrix_df)
    data_model.add_capacity_dimension("demand", location_demand, vehicle_capacity)
    depot = cudf.Series([0] * n_vehicles)
    data_model.set_vehicle_locations(depot, depot)
    return data_model


def solve_cuopt(data_model, time_limit, callback=None):
    """Run the cuOpt solver and return the ssolve_cuoptolution (or None if infeasible)."""
    from cuopt import routing
    solver_settings = routing.SolverSettings()
    solver_settings.set_time_limit(time_limit)
    if callback:
        solver_settings.set_routing_callback(callback)
    solution = routing.Solve(data_model, solver_settings)
    return solution if solution.get_status() == 0 else None


def _run_solver_capture_stdout(data_model, time_limit, callback, log_path=None, need_capture=True):
    """Run solver with fd-1 redirected to a pipe; return (solution, captured_text).
    Avoids deadlock by using a single pipe (no pty). Streams to terminal in real-time.
    Optionally streams to log_path.

    If ``need_capture`` is False, bypass the pipe/reader-thread machinery entirely:
    cuopt prints go straight to the inherited stdout (terminal / outer ``tee``).
    Saves a reader-thread + per-chunk memcpy + decode for runs that don't need
    the captured text.
    """
    if not need_capture and not log_path:
        solution = solve_cuopt(data_model, time_limit, callback=callback)
        return solution, ""
    r, w = os.pipe()
    # Enlarge pipe buffer (Linux) to reduce solver blocking on write
    try:
        import fcntl
        fcntl.fcntl(r, getattr(fcntl, "F_SETPIPE_SZ", 1031), 1048576)  # 1MB
    except (ImportError, OSError, AttributeError):
        pass
    saved_stdout = os.dup(1)
    os.dup2(w, 1)
    os.close(w)
    captured = []
    log_file = open(log_path, "ab") if log_path else None

    def reader():
        while True:
            chunk = os.read(r, 65536)
            if not chunk:
                break
            captured.append(chunk)
            if log_file:
                log_file.write(chunk)
                log_file.flush()
            try:
                os.write(saved_stdout, chunk)
            except OSError:
                pass
        os.close(r)
        if log_file:
            log_file.close()

    reader_thread = threading.Thread(target=reader)
    reader_thread.start()
    try:
        solution = solve_cuopt(data_model, time_limit, callback=callback)
    finally:
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)
    reader_thread.join()
    text = b"".join(captured).decode("utf-8", errors="replace")
    return solution, text


def _parse_offset_ms_from_capture(text):
    """Return (sum, max, counts) from C++ log lines '[iter #N] offset: X ms'. counts: dict of X -> occurrence count."""
    values = [int(m.group(1)) for m in OFFSET_LINE_RE.finditer(text)]
    if not values:
        return 0, 0, {}
    counts = {}
    for x in values:
        counts[x] = counts.get(x, 0) + 1
    return sum(values), max(values), counts


def _parse_similarity_trace_from_capture(text):
    """Parse SIM_TRACE lines from C++ captured stdout.

    Uses finditer over the full text (no splitlines list allocation), so this is
    O(n) over text bytes without an intermediate ~1M-entry list when the run
    produces tens of megabytes of stdout.
    """
    values = [
        {"mode": m.group(1), "sim": float(m.group(2))}
        for m in SIM_TRACE_VALUE_RE.finditer(text)
    ]
    checks = [
        {
            "loc": m.group(1),
            "metric": m.group(2),
            "sim": float(m.group(3)),
            "threshold": float(m.group(4)),
            "decision": m.group(5),
        }
        for m in SIM_TRACE_THRESHOLD_RE.finditer(text)
    ]
    return values, checks


def _save_similarity_trace_run_artifacts(out_dir, run_tag, values, checks, dpi=140):
    """Save per-run similarity trace txt and png."""
    if not values and not checks:
        return
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, f"{run_tag}.txt")
    png_path = os.path.join(out_dir, f"{run_tag}.png")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# [SIM_TRACE][VALUE]\n")
        for i, v in enumerate(values):
            f.write(f"{i}\tmode={v['mode']}\tsim={v['sim']:.8f}\n")
        f.write("\n# [SIM_TRACE][THRESHOLD]\n")
        for i, c in enumerate(checks):
            f.write(
                f"{i}\tloc={c['loc']}\tmetric={c['metric']}\tsim={c['sim']:.8f}\t"
                f"threshold={c['threshold']:.8f}\tdecision={c['decision']}\n"
            )

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=False)
    ax0, ax1 = axes

    # Subsample for plotting only (txt has the full trace).
    # Plotting 300k+ points per run is the dominant cost in sim_trace mode.
    PLOT_MAX = 5000

    def _stride(n):
        return max(1, n // PLOT_MAX)

    if values:
        s = _stride(len(values))
        xs = list(range(0, len(values), s))
        ys = [values[i]["sim"] for i in xs]
        ax0.plot(xs, ys, color="#1f77b4", linewidth=1.0)
    ax0.set_title(f"Similarity Values ({run_tag})")
    ax0.set_ylabel("sim")
    ax0.grid(True, alpha=0.3)

    if checks:
        s = _stride(len(checks))
        xs = list(range(0, len(checks), s))
        sim_y = [checks[i]["sim"] for i in xs]
        th_y = [checks[i]["threshold"] for i in xs]
        ax1.plot(xs, sim_y, color="#ff7f0e", linewidth=1.0, label="sim@check")
        ax1.plot(xs, th_y, color="#d62728", linewidth=1.0, linestyle="--", label="threshold")
        hit_x = [i for i, c in enumerate(checks) if c["decision"] == "sim>threshold"]
        if hit_x and len(hit_x) > PLOT_MAX:
            hit_s = max(1, len(hit_x) // PLOT_MAX)
            hit_x = hit_x[::hit_s]
        hit_y = [checks[i]["sim"] for i in hit_x]
        if hit_x:
            ax1.scatter(hit_x, hit_y, s=10, color="#2ca02c", label="sim>threshold")
        ax1.legend()
    ax1.set_title("Threshold Checks")
    ax1.set_xlabel("check index")
    ax1.set_ylabel("value")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight", dpi=dpi)
    plt.close()


def _analyze_sim_trace_and_gen_excel(out_dir, dpi=140):
    """Scan sim_trace_*.txt files in *out_dir*, print summary, and generate an Excel report.

    Returns the path to the generated .xlsx file, or None if no data found.
    """
    import glob as _glob
    from collections import defaultdict

    txt_files = sorted(_glob.glob(os.path.join(out_dir, "sim_trace_*.txt")))
    if not txt_files:
        print("[sim_trace_analysis] No sim_trace_*.txt found — skipping.", flush=True)
        return None

    # ── Parse all txt files ──────────────────────────────────────────
    # Classify each file as baseline or landscape by filename prefix.
    groups = defaultdict(lambda: {"values": [], "checks": [], "runs": 0})
    per_run = []  # list of dicts for Per-Run Detail sheet

    for fpath in txt_files:
        fname = os.path.basename(fpath)
        # e.g. sim_trace_baseline_inst0_run3.txt, sim_trace_similarity_inst0_run7.txt
        if "_baseline_" in fname:
            gkey = "baseline"
        elif "_similarity_es_" in fname or "_landscape_es_" in fname:
            gkey = "similarity_es"
        elif "_similarity_depth_" in fname:
            gkey = "similarity_depth"
        elif "_depth_stop_" in fname:
            gkey = "depth_stop"
        elif "_early_stop_" in fname:
            gkey = "early_stop"
        elif "_similarity_" in fname or "_landscape_" in fname:
            gkey = "similarity"
        else:
            gkey = "other"

        vals, chks = [], []
        with open(fpath, encoding="utf-8") as f:
            section = None
            for line in f:
                line = line.strip()
                if line.startswith("# [SIM_TRACE][VALUE]"):
                    section = "V"
                    continue
                if line.startswith("# [SIM_TRACE][THRESHOLD]"):
                    section = "T"
                    continue
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if section == "V" and len(parts) >= 3:
                    mode = parts[1].split("=", 1)[1] if "=" in parts[1] else ""
                    sim = float(parts[2].split("=", 1)[1]) if "=" in parts[2] else 0.0
                    vals.append({"mode": mode, "sim": sim})
                elif section == "T" and len(parts) >= 6:
                    loc = parts[1].split("=", 1)[1] if "=" in parts[1] else ""
                    metric = parts[2].split("=", 1)[1] if "=" in parts[2] else ""
                    sim = float(parts[3].split("=", 1)[1]) if "=" in parts[3] else 0.0
                    thr = float(parts[4].split("=", 1)[1]) if "=" in parts[4] else 0.0
                    dec = parts[5].split("=", 1)[1] if "=" in parts[5] else ""
                    chks.append({"loc": loc, "metric": metric, "sim": sim,
                                 "threshold": thr, "decision": dec})

        g = groups[gkey]
        g["values"].extend(vals)
        g["checks"].extend(chks)
        g["runs"] += 1

        n_trigger = sum(1 for c in chks if c["decision"] == "sim>threshold")
        n_sim1 = sum(1 for v in vals if abs(v["sim"] - 1.0) < 1e-6)
        n_bsi = sum(1 for c in chks if c["loc"] == "best_similar_index" and c["decision"] == "sim>threshold")
        n_erad = sum(1 for c in chks if c["loc"] == "eradicate_similar" and c["decision"] == "sim>threshold")
        per_run.append({
            "file": fname, "group": gkey, "n_values": len(vals), "n_checks": len(chks),
            "n_trigger": n_trigger, "n_sim1": n_sim1,
            "n_bsi_trigger": n_bsi, "n_erad_trigger": n_erad,
        })

    _SIM_TRACE_ORDER = (
        "baseline", "similarity", "early_stop", "depth_stop",
        "similarity_es", "similarity_depth",
        "landscape", "landscape_es", "other",
    )
    _order_rank = {k: i for i, k in enumerate(_SIM_TRACE_ORDER)}
    excel_keys = sorted(
        [k for k in groups.keys() if groups[k]["runs"] or groups[k]["values"] or groups[k]["checks"]],
        key=lambda k: (_order_rank.get(k, 999), k),
    )

    # ── Print console summary ────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("  Similarity Trace Analysis", flush=True)
    print("=" * 60, flush=True)
    for gkey in excel_keys:
        g = groups.get(gkey)
        if g is None or (g["runs"] == 0 and not g["values"] and not g["checks"]):
            continue
        vals, chks = g["values"], g["checks"]
        n_trigger = sum(1 for c in chks if c["decision"] == "sim>threshold")
        n_sim1 = sum(1 for v in vals if abs(v["sim"] - 1.0) < 1e-6)
        sim_vals = [v["sim"] for v in vals]
        mean_sim = sum(sim_vals) / len(sim_vals) if sim_vals else 0.0
        print(f"\n[{gkey}]  runs={g['runs']}  VALUE calls={len(vals)}  "
              f"THRESHOLD checks={len(chks)}  triggers={n_trigger}  "
              f"sim=1 count={n_sim1}  mean_sim={mean_sim:.4f}", flush=True)
    print("=" * 60 + "\n", flush=True)

    # ── Build histogram buckets ──────────────────────────────────────
    bucket_edges = [(i / 10, (i + 1) / 10) for i in range(10)]

    def _bucket_counts(sim_list):
        counts = [0] * 10
        for s in sim_list:
            idx = min(int(s * 10), 9)
            if idx < 0:
                idx = 0
            counts[idx] += 1
        return counts

    # ── Generate Excel ───────────────────────────────────────────────
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    except ImportError:
        print("[sim_trace_analysis] openpyxl not installed — skipping Excel generation.", flush=True)
        return None

    wb = Workbook()
    header_font = Font(bold=True, size=11)
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_font_white = Font(bold=True, color="FFFFFF", size=11)
    thin_border = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin"),
    )

    def _write_header(ws, headers, row=1):
        for col_idx, h in enumerate(headers, 1):
            cell = ws.cell(row=row, column=col_idx, value=h)
            cell.font = header_font_white
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")
            cell.border = thin_border

    def _auto_width(ws):
        for col in ws.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
            ws.column_dimensions[col_letter].width = min(max_len + 4, 40)

    # ── Sheet 1: Overview ────────────────────────────────────────────
    ws = wb.active
    ws.title = "Overview"
    empty_g = {"values": [], "checks": [], "runs": 0}
    headers = ["Metric"] + excel_keys
    _write_header(ws, headers)
    metrics = [
        ("Runs", lambda g: g["runs"]),
        ("VALUE calls", lambda g: len(g["values"])),
        ("VALUE calls / run", lambda g: f"{len(g['values']) / g['runs']:.0f}" if g["runs"] else "0"),
        ("THRESHOLD checks", lambda g: len(g["checks"])),
        ("Triggers (sim>thr)", lambda g: sum(1 for c in g["checks"] if c["decision"] == "sim>threshold")),
        ("Trigger rate (%)", lambda g: f"{sum(1 for c in g['checks'] if c['decision'] == 'sim>threshold') / len(g['checks']) * 100:.1f}" if g["checks"] else "N/A"),
        ("sim=1.0 count", lambda g: sum(1 for v in g["values"] if abs(v["sim"] - 1.0) < 1e-6)),
        ("Mean sim", lambda g: f"{sum(v['sim'] for v in g['values']) / len(g['values']):.4f}" if g["values"] else "N/A"),
        ("best_similar triggers", lambda g: sum(1 for c in g["checks"] if c["loc"] == "best_similar_index" and c["decision"] == "sim>threshold")),
        ("eradicate triggers", lambda g: sum(1 for c in g["checks"] if c["loc"] == "eradicate_similar" and c["decision"] == "sim>threshold")),
    ]
    for row_idx, (mname, fn) in enumerate(metrics, 2):
        ws.cell(row=row_idx, column=1, value=mname).font = header_font
        for col, gkey in enumerate(excel_keys, start=2):
            ws.cell(row=row_idx, column=col, value=fn(groups.get(gkey, empty_g)))
    _auto_width(ws)

    # ── Sheet 2: Sim Distribution ────────────────────────────────────
    ws2 = wb.create_sheet("Sim Distribution")
    headers2 = ["Bucket"]
    for gkey in excel_keys:
        headers2.extend([f"{gkey} count", f"{gkey} %"])
    _write_header(ws2, headers2)
    for gkey in excel_keys:
        g = groups.get(gkey, empty_g)
        g["_bucket"] = _bucket_counts([v["sim"] for v in g["values"]])
    for i, (lo, hi) in enumerate(bucket_edges):
        r = i + 2
        ws2.cell(row=r, column=1, value=f"[{lo:.1f}, {hi:.1f})")
        col = 2
        for gkey in excel_keys:
            g = groups.get(gkey, empty_g)
            cnt = g["_bucket"][i]
            total = len(g["values"]) or 1
            ws2.cell(row=r, column=col, value=cnt)
            ws2.cell(row=r, column=col + 1, value=f"{cnt / total * 100:.1f}%")
            col += 2
    _auto_width(ws2)

    # ── Sheet 3: Threshold Breakdown ─────────────────────────────────
    ws3 = wb.create_sheet("Thresholds")
    headers3 = ["Group", "Threshold", "Count", "Triggers", "Trigger %",
                "Mean sim (trigger)", "Mean sim (no trigger)"]
    _write_header(ws3, headers3)
    row_idx = 2
    for gkey in excel_keys:
        g = groups.get(gkey, empty_g)
        thr_map = defaultdict(lambda: {"count": 0, "trigger": 0, "sim_trigger": [], "sim_no": []})
        for c in g["checks"]:
            t = round(c["threshold"], 6)
            thr_map[t]["count"] += 1
            if c["decision"] == "sim>threshold":
                thr_map[t]["trigger"] += 1
                thr_map[t]["sim_trigger"].append(c["sim"])
            else:
                thr_map[t]["sim_no"].append(c["sim"])
        for thr in sorted(thr_map.keys()):
            d = thr_map[thr]
            ws3.cell(row=row_idx, column=1, value=gkey)
            ws3.cell(row=row_idx, column=2, value=thr)
            ws3.cell(row=row_idx, column=3, value=d["count"])
            ws3.cell(row=row_idx, column=4, value=d["trigger"])
            ws3.cell(row=row_idx, column=5, value=f"{d['trigger'] / d['count'] * 100:.1f}%" if d["count"] else "N/A")
            ws3.cell(row=row_idx, column=6, value=f"{sum(d['sim_trigger']) / len(d['sim_trigger']):.4f}" if d["sim_trigger"] else "N/A")
            ws3.cell(row=row_idx, column=7, value=f"{sum(d['sim_no']) / len(d['sim_no']):.4f}" if d["sim_no"] else "N/A")
            row_idx += 1
    _auto_width(ws3)

    # ── Sheet 4: Per-Run Detail ──────────────────────────────────────
    ws4 = wb.create_sheet("Per-Run Detail")
    headers4 = ["File", "Group", "VALUES", "CHECKS", "Triggers",
                "sim=1.0", "bsi_trigger", "erad_trigger"]
    _write_header(ws4, headers4)
    for ri, pr in enumerate(per_run, 2):
        ws4.cell(row=ri, column=1, value=pr["file"])
        ws4.cell(row=ri, column=2, value=pr["group"])
        ws4.cell(row=ri, column=3, value=pr["n_values"])
        ws4.cell(row=ri, column=4, value=pr["n_checks"])
        ws4.cell(row=ri, column=5, value=pr["n_trigger"])
        ws4.cell(row=ri, column=6, value=pr["n_sim1"])
        ws4.cell(row=ri, column=7, value=pr["n_bsi_trigger"])
        ws4.cell(row=ri, column=8, value=pr["n_erad_trigger"])
    _auto_width(ws4)

    # ── Sheet 5: Action Consequences ─────────────────────────────────
    ws5 = wb.create_sheet("Action Consequences")
    headers5 = ["Group", "bsi: found similar (replaced)", "bsi: no similar (appended)",
                "erad: removed member", "erad: kept member"]
    _write_header(ws5, headers5)
    for ri, gkey in enumerate(excel_keys, 2):
        g = groups.get(gkey, empty_g)
        bsi_found = sum(1 for c in g["checks"] if c["loc"] == "best_similar_index" and c["decision"] == "sim>threshold")
        bsi_none = sum(1 for c in g["checks"] if c["loc"] == "best_similar_index" and c["decision"] != "sim>threshold")
        erad_rm = sum(1 for c in g["checks"] if c["loc"] == "eradicate_similar" and c["decision"] == "sim>threshold")
        erad_keep = sum(1 for c in g["checks"] if c["loc"] == "eradicate_similar" and c["decision"] != "sim>threshold")
        ws5.cell(row=ri, column=1, value=gkey)
        ws5.cell(row=ri, column=2, value=bsi_found)
        ws5.cell(row=ri, column=3, value=bsi_none)
        ws5.cell(row=ri, column=4, value=erad_rm)
        ws5.cell(row=ri, column=5, value=erad_keep)
    _auto_width(ws5)

    xlsx_path = os.path.join(out_dir, "sim_trace_analysis.xlsx")
    wb.save(xlsx_path)
    print(f"[sim_trace_analysis] Excel report saved to: {xlsx_path}", flush=True)

    # ── Also plot a combined comparison chart ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_hist, ax_trigger = axes

    colors = {
        "baseline": "#1f77b4",
        "similarity": "#d62728",
        "early_stop_dup": "#2ca02c",
        "early_stop_depth": "#ff7f0e",
        "sim_depth": "#8c564b",
        "sim_dup_depth": "#9467bd",
        "landscape": "#d62728",
        "landscape_es": "#9467bd",
        "other": "#999999",
    }
    hist_keys = [k for k in excel_keys if groups[k]["values"]]
    bar_w = min(0.22, 0.85 / max(1, len(hist_keys)))
    xs = list(range(10))
    for gi, gkey in enumerate(hist_keys):
        g = groups.get(gkey, empty_g)
        bucket = _bucket_counts([v["sim"] for v in g["values"]])
        total = len(g["values"]) or 1
        pcts = [c / total * 100 for c in bucket]
        offsets = [x + gi * bar_w for x in xs]
        ax_hist.bar(offsets, pcts, bar_w, label=gkey, color=colors.get(gkey, "#999"),
                    alpha=0.8)
    nh = len(hist_keys) if hist_keys else 1
    ax_hist.set_xticks([x + (nh - 1) * bar_w / 2 for x in xs])
    ax_hist.set_xticklabels([f"{i / 10:.1f}" for i in range(10)], fontsize=8)
    ax_hist.set_xlabel("Similarity bucket")
    ax_hist.set_ylabel("Percentage (%)")
    ax_hist.set_title("Similarity Value Distribution")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3, axis="y")

    cat_labels = ["bsi trigger", "bsi no-trigger", "erad remove", "erad keep"]
    trig_keys = [k for k in excel_keys if groups[k]["checks"]]
    bar_w2 = min(0.22, 0.85 / max(1, len(trig_keys)))
    for gi, gkey in enumerate(trig_keys):
        g = groups.get(gkey, empty_g)
        bsi_t = sum(1 for c in g["checks"] if c["loc"] == "best_similar_index" and c["decision"] == "sim>threshold")
        bsi_n = sum(1 for c in g["checks"] if c["loc"] == "best_similar_index" and c["decision"] != "sim>threshold")
        erad_t = sum(1 for c in g["checks"] if c["loc"] == "eradicate_similar" and c["decision"] == "sim>threshold")
        erad_n = sum(1 for c in g["checks"] if c["loc"] == "eradicate_similar" and c["decision"] != "sim>threshold")
        vals_bar = [bsi_t, bsi_n, erad_t, erad_n]
        offsets = [x + gi * bar_w2 for x in range(len(cat_labels))]
        ax_trigger.bar(offsets, vals_bar, bar_w2, label=gkey, color=colors.get(gkey, "#999"),
                       alpha=0.8)
    ntk = len(trig_keys) if trig_keys else 1
    ax_trigger.set_xticks([x + (ntk - 1) * bar_w2 / 2 for x in range(len(cat_labels))])
    ax_trigger.set_xticklabels(cat_labels, fontsize=8)
    ax_trigger.set_ylabel("Count")
    ax_trigger.set_title("Threshold Check Outcomes")
    ax_trigger.legend()
    ax_trigger.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    cmp_path = os.path.join(out_dir, "sim_trace_comparison.png")
    plt.savefig(cmp_path, bbox_inches="tight", dpi=dpi)
    plt.close()
    print(f"[sim_trace_analysis] Comparison plot saved to: {cmp_path}", flush=True)

    return xlsx_path


def _wait_unix_socket_ready(socket_path, timeout_s=20.0):
    """Wait until a Unix socket is connectable."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if os.path.exists(socket_path):
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                client.settimeout(0.2)
                client.connect(socket_path)
                client.close()
                return True
            except OSError:
                client.close()
        time.sleep(0.1)
    return False


def run_experiment(
    data_path=None,
    problem_path=None,
    solution_path=None,
    time_limit=10,
    n_instances=1,
    start_index=0,
    problem_type="CVRP",
    scale=1e2,
    n_vehicles=30,
    n_runs=1,
    use_callback=False,
    collect_data=False,
    early_stop_base="random",
    checkpoint_path=None,
    classifier_path=None,
    check_interval=1,
    classifier_type="threshold",
    callback_timeout=30.0,
    es_threshold=None,
    es_improvement_pct=None,
    depth_checkpoint_path=None,
    device="cuda",
    log_path=None,
    trace_dir=None,
    use_landscape_diversity=False,
    landscape_checkpoint_path=None,
    landscape_socket_dir="/tmp",
    landscape_device=None,
    sim_trace_out_dir=None,
    sim_trace_label="run",
):
    """Run cuOpt solver on CVRP instances.

    Returns:
        (best_costs, best_gaps, all_run_points, hgs_costs, early_stop_counts,
         iteration_counts, trial_counts)
        early_stop_counts/iteration_counts/trial_counts: per-instance totals.
    """
    from tqdm import tqdm

    # Load data from pkl or txt
    if problem_path and solution_path:
        raw_nodes, raw_cap, raw_demand, raw_cost, _ = load_pkl_data(
            problem_path, solution_path, start_index=start_index, episode=n_instances
        )
    elif data_path:
        raw_nodes, raw_cap, raw_demand, raw_cost, _ = load_txt_data(
            data_path, start_index=start_index, episode=n_instances
        )
    else:
        raise ValueError("Provide either --problem_path/--solution_path (pkl) or --data_path (txt)")

    raw_dist = pairwise_euclidean_distance(raw_nodes)

    # Set up evolution trace directory (for weights + routes visualization)
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
        os.environ["CUOPT_TRACE_DIR"] = trace_dir
        import numpy as np
        coords = raw_nodes[0].numpy()  # (n+1, 2) first instance
        coords_path = os.path.join(trace_dir, "coords.csv")
        with open(coords_path, "w") as cf:
            cf.write("node_id,x,y\n")
            for nid in range(coords.shape[0]):
                cf.write(f"{nid},{coords[nid, 0]:.8f},{coords[nid, 1]:.8f}\n")
        print(f"[trace] Saved {coords.shape[0]} node coordinates to {coords_path}", flush=True)
    else:
        os.environ.pop("CUOPT_TRACE_DIR", None)

    # Load embedding model and classifier if needed
    embedder, emb_proj, classifier = None, None, None
    if use_callback and early_stop_base == "embedding":
        if not checkpoint_path:
            raise ValueError("--checkpoint is required for embedding early stop")
        problem_size = raw_nodes.shape[1] - 1  # n+1 nodes, 1 depot
        if classifier_path:
            embedder, emb_proj, classifier = load_embedder_and_classifier(
                checkpoint_path, classifier_path, problem_size, device=device
            )
        elif es_threshold is not None:
            embedder, emb_proj = _load_embedder(checkpoint_path, device=device)
            classifier = {"tau_embed": es_threshold}
            print(f"[early_stop] threshold-only mode: tau_embed={es_threshold:.4f}")
        else:
            raise ValueError(
                "--classifier_pkl or --es_threshold is required for embedding early stop"
            )
    elif use_callback and early_stop_base == "structure":
        if not classifier_path:
            raise ValueError("--classifier_pkl is required for structure early stop")
        import pickle
        import torch
        with open(classifier_path, "rb") as f:
            classifier = pickle.load(f)
        print(f"Loaded classifier from {classifier_path} (tau_struct={classifier.get('tau_struct', '?'):.4f})")
        if "clf_struct" in classifier and hasattr(classifier["clf_struct"], "coef_"):
            clf = classifier["clf_struct"]
            dev = torch.device(device) if isinstance(device, str) else device
            classifier["_clf_struct_coef"] = torch.tensor(clf.coef_, dtype=torch.float32, device=dev)
            classifier["_clf_struct_intercept"] = torch.tensor(clf.intercept_, dtype=torch.float32, device=dev)

    # Load depth predictor if needed
    depth_pred = None
    if es_improvement_pct is not None and depth_checkpoint_path:
        depth_pred = _load_depth_predictor(depth_checkpoint_path, device=device)

    # Build callback class if needed
    need_callback = use_callback or collect_data or depth_pred is not None

    best_costs = []
    best_gaps = []
    all_run_points = []  # [(label, points), ...]
    hgs_costs = {}       # {instance_index: hgs_cost}
    early_stop_counts = []  # [total_es_per_instance, ...]
    iteration_counts = []   # [total_iterations_per_instance, ...]
    trial_counts = []       # [total_trials_per_instance, ...]

    total = n_instances * n_runs
    total_callback_time_ms = 0.0
    total_offset_cpp_ms = 0
    max_offset_cpp_ms = 0
    offset_value_counts = {}

    if sim_trace_out_dir:
        os.environ["CUOPT_LOG_SIMILARITY"] = "1"
        import ctypes
        _libc = ctypes.CDLL(None)
        _libc.setenv.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        _libc.setenv.restype = ctypes.c_int
        _libc.setenv(b"CUOPT_LOG_SIMILARITY", b"1", 1)
    if log_path and need_callback:
        open(log_path, "w").close()

    # Optional: automatic landscape diversity via embedding_server.py
    landscape_proc = None
    landscape_socket_path = None
    prev_metric_env = os.environ.get("CUOPT_DIVERSITY_METRIC")
    prev_socket_env = os.environ.get("CUOPT_EMBEDDING_SOCKET")
    if use_landscape_diversity:
        landscape_device = landscape_device or device
        if not problem_path:
            raise ValueError("--problem_path is required when --landscape_diversity is enabled")
        if not landscape_checkpoint_path:
            raise ValueError("--landscape_checkpoint is required when --landscape_diversity is enabled")
        server_script = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "embedding_server.py")
        )
        if not os.path.isfile(server_script):
            raise FileNotFoundError(f"embedding_server.py not found: {server_script}")
        os.makedirs(landscape_socket_dir, exist_ok=True)
        os.environ["CUOPT_DIVERSITY_METRIC"] = "landscape"
    else:
        os.environ.pop("CUOPT_DIVERSITY_METRIC", None)
        os.environ.pop("CUOPT_EMBEDDING_SOCKET", None)

    with tqdm(total=total, desc="Solving with cuOpt") as pbar:
        try:
            for i in range(n_instances):
                abs_index = start_index + i
                if use_landscape_diversity:
                    if landscape_proc is not None:
                        landscape_proc.terminate()
                        try:
                            landscape_proc.wait(timeout=5.0)
                        except subprocess.TimeoutExpired:
                            landscape_proc.kill()
                            landscape_proc.wait(timeout=5.0)
                        landscape_proc = None
                    # Include PID + nanosecond stamp in socket name. The PID prevents collision
                    # when multiple cuopt processes share the same instance index across panes;
                    # the time_ns stamp prevents the *previous* embedder's SIGTERM cleanup
                    # (which os.unlinks the socket file) from racing the *new* embedder that
                    # has just bound to the same path between configs in the same process.
                    landscape_socket_path = os.path.join(
                        landscape_socket_dir,
                        f"cuopt_embedding_{abs_index}_{os.getpid()}_{time.time_ns()}.sock",
                    )
                    launch_cmd = [
                        sys.executable,
                        server_script,
                        "--checkpoint",
                        landscape_checkpoint_path,
                        "--instance_pkl",
                        problem_path,
                        "--instance_index",
                        str(abs_index),
                        "--socket_path",
                        landscape_socket_path,
                        "--device",
                        "cuda",
                    ]
                    server_env = os.environ.copy()
                    _dev_match = re.match(r"cuda:(\d+)", landscape_device)
                    if _dev_match:
                        # Map parent's "cuda:N" to the corresponding PHYSICAL GPU id by
                        # looking up index N in parent's CUDA_VISIBLE_DEVICES list.
                        # Without this remap, child gets CUDA_VISIBLE_DEVICES=N which means
                        # physical GPU N, NOT parent's cuda:N device. With parent
                        # CUDA_VISIBLE_DEVICES="1,2" + landscape_device="cuda:1", child
                        # would otherwise see physical GPU 1 (collision with main solver).
                        _local_idx = int(_dev_match.group(1))
                        _parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                        _parent_gpus = [g.strip() for g in _parent_cvd.split(",") if g.strip()]
                        if _parent_gpus and _local_idx < len(_parent_gpus):
                            server_env["CUDA_VISIBLE_DEVICES"] = _parent_gpus[_local_idx]
                        else:
                            server_env["CUDA_VISIBLE_DEVICES"] = str(_local_idx)
                    landscape_proc = subprocess.Popen(launch_cmd, env=server_env)
                    if not _wait_unix_socket_ready(landscape_socket_path, timeout_s=20.0):
                        raise RuntimeError(
                            f"embedding server not ready for instance {abs_index}: {landscape_socket_path}"
                        )
                    os.environ["CUOPT_EMBEDDING_SOCKET"] = landscape_socket_path
                    print(
                        f"[landscape] instance={abs_index}, socket={landscape_socket_path}, "
                        f"pid={landscape_proc.pid}",
                        flush=True,
                    )

                raw_cost_value = raw_cost[i].item()
                hgs_costs[i] = raw_cost_value
                run_costs = []
                inst_es = 0
                inst_iters = 0
                inst_trials = 0

                # Set up CVRPEnv for this instance (embedding mode)
                env = None
                if embedder is not None:
                    env = setup_env_for_instance(i, raw_nodes, raw_demand, raw_cap, device=device)

                # Set up CVRPEnv for depth predictor (may use different checkpoint)
                dep_env = None
                if depth_pred is not None:
                    dep_env = setup_env_for_instance(i, raw_nodes, raw_demand, raw_cap, device=device)

                # Build callback class per instance (env is instance-specific)
                CallbackCls = _make_callback_class(
                    scale=scale, early_stop=use_callback,
                    early_stop_base=early_stop_base,
                    embedder=embedder, emb_proj=emb_proj,
                    env=env, classifier=classifier,
                    check_interval=check_interval, classifier_type=classifier_type,
                    callback_timeout=callback_timeout,
                    demands=raw_demand[i].numpy(),
                    capacity=raw_cap[i].item(),
                    es_improvement_pct=es_improvement_pct,
                    depth_predictor=depth_pred,
                    depth_env=dep_env,
                ) if need_callback else None

                for k in range(n_runs):
                    callback = CallbackCls() if need_callback else None
                    model = get_cuopt_model(i, raw_dist, raw_demand, raw_cap, n_vehicles, scale)
                    if need_callback:
                        # Only capture stdout if a downstream consumer needs it: sim_trace
                        # parsing or per-run log_path append. Skipping the pipe is a real
                        # win for non-sim_trace runs (no reader thread, no chunk copies).
                        _need_cap = bool(sim_trace_out_dir) or bool(log_path)
                        solution, captured = _run_solver_capture_stdout(
                            model, time_limit, callback,
                            log_path=log_path if k == 0 and i == 0 else None,
                            need_capture=_need_cap,
                        )
                        if log_path and (i > 0 or k > 0):
                            with open(log_path, "a", encoding="utf-8", errors="replace") as lf:
                                lf.write(captured)
                        run_sum, run_max, run_counts = _parse_offset_ms_from_capture(captured)
                        total_offset_cpp_ms += run_sum
                        if run_max > max_offset_cpp_ms:
                            max_offset_cpp_ms = run_max
                        for x, c in run_counts.items():
                            offset_value_counts[x] = offset_value_counts.get(x, 0) + c
                        if sim_trace_out_dir:
                            values, checks = _parse_similarity_trace_from_capture(captured)
                            n_trace = len(values) + len(checks)
                            run_tag = f"sim_trace_{sim_trace_label}_inst{i}_run{k}"
                            print(f"[sim_trace] {run_tag}: {len(values)} values, {len(checks)} checks "
                                  f"(captured {len(captured)} chars)", flush=True)
                            _save_similarity_trace_run_artifacts(sim_trace_out_dir, run_tag, values, checks)
                    else:
                        print(f"[run_cuopt] Instance {i} Run {k}: calling solver (time_limit={time_limit}s)...", file=sys.stderr, flush=True)
                        solution = solve_cuopt(model, time_limit, callback=callback)
                        print(f"[run_cuopt] Instance {i} Run {k}: solver returned.", file=sys.stderr, flush=True)

                    if solution:
                        cost = solution.get_total_objective() / scale
                        run_costs.append(cost)
                        gap = ((cost - raw_cost_value) / raw_cost_value) * 100
                        if callback:
                            depth_info = f" depth={callback.n_depth_early_stops}" if callback.n_depth_early_stops else ""
                            es_info = f" | EarlyStops: {callback.n_early_stops}{depth_info}"
                        else:
                            es_info = ""
                        print(f"[Instance {i} Run {k}] Cost: {cost:.2f} | Best Known: {raw_cost_value:.2f} | Gap: {gap:.2f}%{es_info}", flush=True)
                    else:
                        print(f"[Instance {i} Run {k}] No feasible solution.", flush=True)

                    if callback:
                        inst_es += callback.n_early_stops
                        inst_iters += callback.n_iterations
                        inst_trials += callback.n_trials
                        total_callback_time_ms += callback._total_callback_time_ms

                    # Collect callback data
                    if collect_data and callback and callback.points:
                        label = f"inst{i}" if n_runs == 1 else f"inst{i}_run{k}"
                        all_run_points.append((label, list(callback.points)))

                    pbar.update(1)

                early_stop_counts.append(inst_es)
                iteration_counts.append(inst_iters)
                trial_counts.append(inst_trials)

                if run_costs:
                    best = min(run_costs)
                    best_gap = ((best - raw_cost_value) / raw_cost_value) * 100
                    best_costs.append(best)
                    best_gaps.append(best_gap)
                    if n_runs > 1:
                        print(f"[Instance {i}] Best of {n_runs} runs: {best:.2f} | Gap: {best_gap:.2f}%")
                else:
                    best_costs.append(None)
                    best_gaps.append(None)
        finally:
            if landscape_proc is not None:
                landscape_proc.terminate()
                try:
                    landscape_proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    landscape_proc.kill()
                    landscape_proc.wait(timeout=5.0)
            if prev_metric_env is None:
                os.environ.pop("CUOPT_DIVERSITY_METRIC", None)
            else:
                os.environ["CUOPT_DIVERSITY_METRIC"] = prev_metric_env
            if prev_socket_env is None:
                os.environ.pop("CUOPT_EMBEDDING_SOCKET", None)
            else:
                os.environ["CUOPT_EMBEDDING_SOCKET"] = prev_socket_env

    if need_callback:
        print(
            f"[Summary] Time limit: {time_limit} s | "
            f"Total callback time (Python): {total_callback_time_ms:.1f} ms"
        )
    return best_costs, best_gaps, all_run_points, hgs_costs, early_stop_counts, iteration_counts, trial_counts

# ── Subcommand handlers ─────────────────────────────────────────────

def _write_log(named_points, hgs_costs, filepath):
    """Write log: [search #N] then cost before/after lines, plus optional is_feasible flag."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, 'w') as f:
        for label, points in named_points:
            m = re.search(r"inst(\d+)", label)
            inst_idx = int(m.group(1)) if m else 0
            hgs_cost = (hgs_costs or {}).get(inst_idx, 0)
            f.write(f"# === {label} (HGS: {hgs_cost}) ===\n")
            search_id = 1
            prev_was_break = True
            for p in points:
                if math.isnan(p['after']):
                    search_id += 1
                    prev_was_break = True
                    continue
                if prev_was_break:
                    f.write(f"[search #{search_id}]\n")
                    prev_was_break = False
                wc = p.get('wall_clock')
                wc_str = f", wall_clock: {wc:.4f}" if wc is not None else ""
                # Write is_feasible flag so reload preserves feasibility filtering.
                # Missing field on read = treated as feasible (back-compat with old logs).
                feas = p.get('is_feasible')
                feas_str = f", is_feasible: {1 if feas else 0}" if feas is not None else ""
                f.write(f"cost before: {p.get('before', 0)}, cost after: {p['after']}{wc_str}{feas_str}\n")
    print(f"Saved log to: {filepath}")


def _load_labeled_log(filepath):
    """Load a log written by _write_log and recover (named_points, hgs_costs)."""
    raw = open(filepath, "r", encoding="utf-8", errors="ignore").read()
    header_re = re.compile(r"^# === (.+?) \(HGS: ([^)]+)\) ===\s*$", re.M)

    matches = list(header_re.finditer(raw))
    if not matches:
        return [], {}

    named_points = []
    hgs_costs = {}
    for i, m in enumerate(matches):
        label = m.group(1).strip()
        hgs_str = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[body_start:body_end]
        pts = parse_points(body)
        named_points.append((label, pts))

        inst_m = re.search(r"inst(\d+)", label)
        if inst_m:
            inst_idx = int(inst_m.group(1))
            try:
                hgs_costs[inst_idx] = float(hgs_str)
            except ValueError:
                pass
    return named_points, hgs_costs


def _parse_csv_tokens(raw):
    """Parse comma-separated tokens into a clean list."""
    if not raw:
        return []
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _label_matches_any_prefix(label, prefixes):
    """Return True if label matches any prefix token."""
    if not prefixes:
        return True
    for p in prefixes:
        if label == p or label.startswith(p) or label.startswith(f"{p}_"):
            return True
    return False


def _filter_named_points_by_prefix(named_points, keep_prefixes):
    """Keep only labeled runs matching any given prefix token."""
    if not keep_prefixes:
        return list(named_points)
    return [(label, pts) for label, pts in named_points if _label_matches_any_prefix(label, keep_prefixes)]


def _timestamp_dir():
    """Return a directory name with current time (e.g. 20250212_143052)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class _TeeStdout:
    """Context manager: redirect fd 1 so all stdout (Python + C++) is written to both terminal and log file.
    Uses a pty when available so C++ sees a tty and uses line buffering (avoids appearing stuck with pipe)."""
    def __init__(self, log_path):
        self.log_path = log_path
        self._log_file = None
        self._saved_fd = None
        self._read_fd = None
        self._write_fd = None
        self._thread = None
        self._use_pty = False

    def __enter__(self):
        self._log_file = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self._saved_fd = os.dup(1)
        self._use_pty = False
        # Always use a pipe (not pty) to avoid the tiny 4KB pty buffer that
        # causes hangs when nested with _run_solver_capture_stdout.
        self._read_fd, self._write_fd = os.pipe()
        try:
            import fcntl
            fcntl.fcntl(self._read_fd, getattr(fcntl, "F_SETPIPE_SZ", 1031), 1048576)
        except (ImportError, OSError, AttributeError):
            pass
        os.dup2(self._write_fd, 1)

        def reader():
            while True:
                try:
                    data = os.read(self._read_fd, 65536)
                except OSError:
                    break
                if not data:
                    break
                try:
                    text = data.decode("utf-8", errors="replace")
                    self._log_file.write(text)
                    self._log_file.flush()
                except (OSError, AttributeError, TypeError):
                    pass
                try:
                    os.write(self._saved_fd, data)
                except OSError:
                    pass
        self._thread = threading.Thread(target=reader, daemon=False)
        self._thread.start()
        return self

    def _close_fd(self, fd):
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def __exit__(self, *exc):
        # Close ALL write ends of the pipe so reader thread sees EOF.
        # fd 1 and self._write_fd both reference the pipe write end.
        self._close_fd(self._write_fd)
        self._write_fd = None
        if self._saved_fd is not None:
            try:
                os.dup2(self._saved_fd, 1)  # closes fd 1 (pipe write) and restores terminal
            except OSError:
                pass
            self._close_fd(self._saved_fd)
            self._saved_fd = None
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._close_fd(self._read_fd)
        self._read_fd = None
        if self._log_file is not None:
            self._log_file.close()
        return False


def cmd_solve(args):
    """Execute the 'solve' subcommand. With --log, stdout (including C++ solver) is tee'd to the log file."""
    if getattr(args, "sim_trace", False):
        os.environ["CUOPT_LOG_SIMILARITY"] = "1"
        print("[sim_trace] Enabled CUOPT_LOG_SIMILARITY=1 in process environment", flush=True)
    effective_out = os.path.join(args.out_dir, "curves", args.problem_path.split("/")[-1].split(".")[0] + "_" + str(args.start_index) + "_tl_" + str(int(args.time_limit)) + "_" + _timestamp_dir())
    log_path = None
    if args.log is not None:
        os.makedirs(effective_out, exist_ok=True)
        log_path = os.path.join(effective_out, os.path.basename(args.log) or "log.txt")

    need_collect = args.plot or args.save_upper_bound_log
    # Enable evolution tracing when --trace is set
    trace_dir = effective_out if getattr(args, "trace", False) else None
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
    if args.landscape_diversity and not args.landscape_checkpoint:
        args.landscape_checkpoint = args.checkpoint

    run_kwargs = dict(
        data_path=args.data_path,
        problem_path=args.problem_path,
        solution_path=args.solution_path,
        time_limit=args.time_limit,
        n_instances=args.n_instances,
        start_index=args.start_index,
        problem_type=args.problem_type,
        scale=args.scale,
        n_vehicles=args.n_vehicles,
        n_runs=args.n_runs,
        use_callback=args.use_callback,
        collect_data=need_collect,
        early_stop_base=args.early_stop_base,
        checkpoint_path=args.checkpoint,
        classifier_path=args.classifier_pkl,
        check_interval=args.check_interval,
        classifier_type=args.classifier_type,
        callback_timeout=args.callback_timeout,
        device=args.device,
        log_path=log_path,
        trace_dir=trace_dir,
        use_landscape_diversity=args.landscape_diversity,
        landscape_checkpoint_path=args.landscape_checkpoint,
        landscape_socket_dir=args.landscape_socket_dir,
        landscape_device=args.landscape_device,
        sim_trace_out_dir=effective_out if os.environ.get("CUOPT_LOG_SIMILARITY") else None,
        sim_trace_label="solve",
    )

    # Launch live trace visualizer in background (before solver starts)
    _live_viz_proc = None
    if trace_dir:
        _viz_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualize_evolution_live.py")
        if os.path.isfile(_viz_script):
            import subprocess
            _live_viz_proc = subprocess.Popen(
                [sys.executable, _viz_script,
                 "--trace_dir", trace_dir, "--output_dir", trace_dir,
                 "--live", "--poll_interval", "3"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            print(f"[trace] Live visualizer started (pid={_live_viz_proc.pid}), "
                  f"updating {trace_dir}/weights_evolution.png + route_latest.png", flush=True)

    if log_path is not None:
        with _TeeStdout(log_path):
            costs, gaps, all_run_points, hgs_costs, *_ = run_experiment(**run_kwargs)
    else:
        costs, gaps, all_run_points, hgs_costs, *_ = run_experiment(**run_kwargs)

    # Stop live visualizer
    if _live_viz_proc is not None:
        _live_viz_proc.terminate()
        _live_viz_proc.wait(timeout=5)
        print(f"[trace] Live visualizer stopped.", flush=True)

    if log_path is not None:
        print(f"Saved log (terminal output) to: {log_path}", flush=True)
    if args.plot and all_run_points:
        os.makedirs(effective_out, exist_ok=True)
        ymin = args.ymin
        ymax = args.ymax
        if (ymin is None or ymax is None) and hgs_costs:
            hgs_min = min(hgs_costs.values())
            if ymin is None:
                ymin = hgs_min - 0.5
            if ymax is None:
                ymax = hgs_min + 4.5
        plot_from_points(
            all_run_points,
            dpi=args.dpi,
            ymin=ymin,
            ymax=ymax,
            out_dir=effective_out,
            hgs_costs=hgs_costs,
            feasible_only=not args.include_infeasible,
        )
        print(f"Outputs in: {effective_out}/")
    elif args.plot and not all_run_points:
        print("No callback data for plotting (run without --log or ensure callback is used).")
    if args.save_upper_bound_log and all_run_points:
        try:
            import json
            from upper_bound_analyzer import logs_from_run_cuopt_points
            label, points = all_run_points[0]
            logs = logs_from_run_cuopt_points(
                points,
                steps_to_time_ratio=1e-3,
                basin_key=args.ub_basin_key,
            )
            out_path = os.path.join(effective_out, "upper_bound_log.json")
            os.makedirs(effective_out, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2)
            print(f"Saved upper-bound trial log to: {out_path}")
        except Exception as e:
            print(f"[WARN] Failed to save upper_bound_log: {e}")

    # Run evolution trace visualization (post-hoc)
    if trace_dir and os.path.isfile(os.path.join(trace_dir, "trace.csv")):
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualize_evolution_live.py")
        if os.path.isfile(script):
            import subprocess
            print(f"[trace] Running evolution visualization on {trace_dir}/ ...", flush=True)
            subprocess.run([
                sys.executable, script,
                "--trace_dir", trace_dir,
                "--output_dir", trace_dir,
            ], check=False)
            print(f"[trace] Visualization outputs in: {trace_dir}/", flush=True)
        else:
            print(f"[trace] Visualization script not found: {script}", file=sys.stderr)

    # Auto-analyze sim_trace if enabled
    if os.environ.get("CUOPT_LOG_SIMILARITY") and effective_out:
        _analyze_sim_trace_and_gen_excel(effective_out, dpi=args.dpi)


def cmd_plot(args):
    """Execute the 'plot' subcommand."""
    # Read input files
    inputs = []
    if args.files:
        for fp in args.files:
            try:
                raw = open(fp, "r", encoding="utf-8", errors="ignore").read()
            except Exception as e:
                print(f"[WARN] Failed to read: {fp} ({e})")
                continue
            inputs.append((os.path.basename(fp), raw))
    else:
        raw = sys.stdin.read()
        inputs.append(("stdin", raw))

    if not inputs:
        print("No readable log provided.")
        sys.exit(1)

    # Parse text logs into point lists
    named_points = []
    scale = getattr(args, 'scale', 1.0)
    for name, raw in inputs:
        pts = parse_points(raw)
        if not pts:
            print(f"[INFO] '{name}' has no 'cost before/after' lines, skipping.")
        if scale != 1.0:
            for p in pts:
                for k in ('before', 'after', 'best_so_far'):
                    if k in p and p[k] is not None:
                        p[k] = p[k] / scale
        named_points.append((name, pts))

    # Process labels
    labels = None
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]
        if len(labels) != len(named_points):
            print(f"[WARN] Label count ({len(labels)}) != file count ({len(named_points)}); ignoring custom labels.")
            labels = None

    # Parse --hgs_costs into hlines; else try to parse "Best Known: X.XX" from logs
    hlines = None
    if args.hgs_costs:
        vals = [float(v.strip()) for v in args.hgs_costs.split(",")]
        hlines = []
        for i, v in enumerate(vals):
            lab = "HGS" if len(vals) == 1 else f"HGS inst{i}"
            hlines.append((v, lab))
    else:
        hgs_from_logs = []
        for _name, raw in inputs:
            for m in BEST_KNOWN_RE.finditer(raw):
                hgs_from_logs.append(float(m.group(1)))
                break  # one value per file
        if hgs_from_logs:
            hgs_min = min(hgs_from_logs)
            hlines = [(hgs_min, "HGS")]

    # Default ymin/ymax from HGS (same as solve --plot)
    ymin, ymax = args.ymin, args.ymax
    if (ymin is None or ymax is None) and hlines:
        hgs_vals = [v for v, _ in hlines]
        hgs_min = min(hgs_vals)
        if ymin is None:
            ymin = hgs_min - 0.5
        if ymax is None:
            ymax = hgs_min + 4.5

    common = dict(
        named_points=named_points, labels=labels,
        break_mode=args.break_mode, segments=args.segments, xshift=args.xshift,
        dpi=args.dpi, ymin=ymin, ymax=ymax, hlines=hlines,
        feasible_only=not args.include_infeasible,
    )
    _plot_one_figure(**common, data_key='after',       use_percentage=False, ylabel="Cost",       out_path=args.out)
    _plot_one_figure(**common, data_key='best_so_far', use_percentage=False, ylabel="Cost", out_path=args.out_best)
    _plot_one_figure(**common, data_key='after',       use_percentage=True,  ylabel="Cost",       out_path=args.out_pct)
    _plot_one_figure(**common, data_key='best_so_far', use_percentage=True,  ylabel="Cost", out_path=args.out_best_pct)

def cmd_compare(args):
    """Run 5 early-stop configs and plot per-instance boxplots of best costs.

    Outer loop is per-instance: after each instance finishes all configs × repeats,
    plots are saved immediately and data is dumped to JSON for crash recovery.
    """
    import json
    import numpy as np

    # (label, use_callback, early_stop_base, classifier_type)
    configs = [
        ("no ES",       False, "random",    "threshold"),
        ("emb+thr",     True,  "embedding", "threshold"),
        ("emb+lr",      True,  "embedding", "lr"),
        ("struct+thr",  True,  "structure", "threshold"),
        ("struct+lr",   True,  "structure", "lr"),
    ]
    labels = [c[0] for c in configs]
    colors = ['#AAAAAA', '#5B9BD5', '#2E75B6', '#ED7D31', '#C55A11']

    effective_out = os.path.join(args.out_dir, "curves", _timestamp_dir())
    os.makedirs(effective_out, exist_ok=True)
    args.out_dir = effective_out
    prefix = args.out  # e.g. "compare" -> compare_cost.png, compare_stats.png, ...
    n_inst = args.n_instances

    # ── Helper: draw & save for one instance ─────────────────────
    def _draw_and_save(global_idx, results, es, iters, trials, hgs_val):
        """Draw cost boxplot + 4-in-1 stats + save JSON for one instance."""
        # Build gap data
        gap = {name: [] for name in labels}
        if hgs_val and hgs_val != 0:
            for name in labels:
                for c in results[name]:
                    gap[name].append((c - hgs_val) / hgs_val * 100)

        suffix = "" if n_inst == 1 else f"_inst{global_idx}"

        def _out(tag, extension=".png"):
            return os.path.join(args.out_dir, f"{prefix}_{tag}{suffix}{extension}")

        # ── Cost boxplot ─────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        bp = ax.boxplot([results[l] for l in labels], labels=labels,
                        patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.8)
        if hgs_val is not None:
            ax.axhline(hgs_val, color='red', linestyle='--', linewidth=1,
                       label=f"HGS ({hgs_val:.2f})")
            ax.legend()
        ax.set_ylabel("Best Cost")
        ax.set_title(f"Instance {global_idx}  ({args.n_repeat} repeats, {args.time_limit}s/run)")
        ax.grid(axis='y', alpha=0.3)
        p = _out("cost")
        fig.tight_layout(); fig.savefig(p, dpi=args.dpi); plt.close(fig)
        print(f"  Saved: {p}")

        # ── 4-in-1 stats figure ──────────────────────────────────
        subplot_specs = [
            (gap, "Gap to HGS (%)"),
            (iters, "Total Iterations"),
            (trials, "Total Trials"),
            (es, "Early Stops"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax, (pdata, ylabel) in zip(axes.flat, subplot_specs):
            bp = ax.boxplot([pdata[l] for l in labels], labels=labels,
                            patch_artist=True, widths=0.5)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color); patch.set_alpha(0.8)
            ax.set_ylabel(ylabel)
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=15)
        fig.suptitle(f"Instance {global_idx}  ({args.n_repeat} repeats, {args.time_limit}s/run)",
                     fontsize=13)
        fig.tight_layout()
        p = _out("stats")
        fig.savefig(p, dpi=args.dpi); plt.close(fig)
        print(f"  Saved: {p}")

        # ── Save data to JSON ────────────────────────────────────
        data_dict = {
            "instance": global_idx, "hgs_cost": hgs_val,
            "n_repeat": args.n_repeat, "time_limit": args.time_limit,
        }
        for name in labels:
            data_dict[name] = {
                "cost": results[name], "gap": gap[name],
                "iters": iters[name], "trials": trials[name], "es": es[name],
            }
        p = _out("data", extension=".json")
        with open(p, "w") as f:
            json.dump(data_dict, f, indent=2)
        print(f"  Saved: {p}")

        # ── Print summary for this instance ──────────────────────
        print(f"\n  Instance {global_idx}" + (f"  (HGS: {hgs_val:.4f})" if hgs_val else "") + ":")
        for name in labels:
            vals = results[name]
            if not vals:
                continue
            arr = np.array(vals)
            extra = []
            it_arr = np.array(iters[name], dtype=float)
            tr_arr = np.array(trials[name], dtype=float)
            es_arr = np.array(es[name])
            if it_arr.sum() > 0:
                extra.append(f"iters={it_arr.mean():.0f}±{it_arr.std():.0f}")
            if tr_arr.sum() > 0:
                extra.append(f"trials={tr_arr.mean():.1f}±{tr_arr.std():.1f}")
            if es_arr.sum() > 0:
                extra.append(f"ES={es_arr.mean():.1f}±{es_arr.std():.1f}")
            extra_str = "  (" + ", ".join(extra) + ")" if extra else ""
            print(f"    {name:12s}: mean={arr.mean():.4f}, std={arr.std():.4f}, "
                  f"min={arr.min():.4f}, max={arr.max():.4f}{extra_str}")

    # ── Main loop: per instance ──────────────────────────────────
    for inst_idx in range(n_inst):
        global_idx = args.start_index + inst_idx
        print(f"\n{'#'*60}")
        print(f"# Instance {global_idx}  ({inst_idx+1}/{n_inst})")
        print(f"{'#'*60}")

        results = {c[0]: [] for c in configs}
        es_data = {c[0]: [] for c in configs}
        iters_data = {c[0]: [] for c in configs}
        trials_data = {c[0]: [] for c in configs}
        hgs_val = None

        for cfg_name, use_cb, es_base, clf_type in configs:
            for rep in range(args.n_repeat):
                print(f"  [{cfg_name}] repeat {rep+1}/{args.n_repeat}")
                best_costs, _, _, hgs_costs, es_counts, iter_counts, tri_counts = run_experiment(
                    data_path=args.data_path,
                    problem_path=args.problem_path,
                    solution_path=args.solution_path,
                    time_limit=args.time_limit,
                    n_instances=1,
                    start_index=global_idx,
                    problem_type=args.problem_type,
                    scale=args.scale,
                    n_vehicles=args.n_vehicles,
                    n_runs=1,
                    use_callback=use_cb,
                    collect_data=True,
                    early_stop_base=es_base,
                    checkpoint_path=args.checkpoint,
                    classifier_path=args.classifier_pkl,
                    check_interval=args.check_interval,
                    classifier_type=clf_type,
                    device=args.device,
                )
                if best_costs and best_costs[0] is not None:
                    results[cfg_name].append(best_costs[0])
                es_data[cfg_name].append(es_counts[0] if es_counts else 0)
                iters_data[cfg_name].append(iter_counts[0] if iter_counts else 0)
                trials_data[cfg_name].append(tri_counts[0] if tri_counts else 0)
                if hgs_val is None and hgs_costs:
                    hgs_val = hgs_costs.get(0)

        # Immediately plot + save for this instance
        _draw_and_save(global_idx, results, es_data, iters_data, trials_data, hgs_val)

    print(f"\nAll {n_inst} instances done. Results in: {args.out_dir}/")


def cmd_curves(args):
    """Run 3 configs (no ES, emb+lr, struct+lr) and plot after / best_so_far convergence curves."""
    # (label, use_callback, early_stop_base, classifier_type)
    configs = [
        ("no ES",    False, "random",    "threshold"),
        ("emb+lr",   True,  "embedding", "lr"),
        ("struct+lr", True, "structure", "lr"),
    ]
    labels = [c[0] for c in configs]
    named_points = []
    hgs_costs = None

    for cfg_name, use_cb, es_base, clf_type in configs:
        print(f"\n--- {cfg_name} ---")
        _, _, all_run_points, hgs_costs, *_ = run_experiment(
            data_path=args.data_path,
            problem_path=args.problem_path,
            solution_path=args.solution_path,
            time_limit=args.time_limit,
            n_instances=args.n_instances,
            start_index=args.start_index,
            problem_type=args.problem_type,
            scale=args.scale,
            n_vehicles=args.n_vehicles,
            n_runs=args.n_runs,
            use_callback=use_cb,
            collect_data=True,
            early_stop_base=es_base,
            checkpoint_path=args.checkpoint,
            classifier_path=args.classifier_pkl,
            check_interval=args.check_interval,
            classifier_type=clf_type,
            device=args.device,
        )
        for label, points in all_run_points:
            named_points.append((f"{cfg_name}_{label}", points))

    if not named_points:
        print("No data collected. Check problem/solution paths.")
        return

    effective_out = os.path.join(args.out_dir, "curves", _timestamp_dir())
    os.makedirs(effective_out, exist_ok=True)
    log_path = os.path.join(effective_out, "log.txt")
    _write_log(named_points, hgs_costs, log_path)

    n_per_config = args.n_instances * args.n_runs
    use_custom_labels = (args.n_instances == 1 and args.n_runs == 1 and len(named_points) == len(labels))

    if args.n_runs > 1:
        # Best-so-far: interval curve (mean ± std) per config
        config_curves = {}
        for j, cfg_name in enumerate(labels):
            start = j * n_per_config
            end = start + n_per_config
            if end <= len(named_points):
                config_curves[cfg_name] = [named_points[k][1] for k in range(start, end)]
        hgs_val = None
        if hgs_costs and 0 in hgs_costs:
            hgs_val = hgs_costs[0]
        eff_ymax_cv = args.ymax
        if eff_ymax_cv is None and hgs_val is not None:
            eff_ymax_cv = hgs_val + 1.0
        _plot_best_so_far_interval(
            config_curves,
            out_path=os.path.join(effective_out, "best_so_far.png"),
            out_path_pct=os.path.join(effective_out, "best_so_far_pct.png"),
            dpi=args.dpi, ymin=args.ymin, ymax=eff_ymax_cv, hgs_cost=hgs_val,
            colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
            feasible_only=not args.include_infeasible,
        )
        # After: still multi-line (all runs)
        _plot_one_figure(
            named_points=named_points, labels=None,
            break_mode="none", segments=0, xshift=0.0, dpi=args.dpi,
            ymin=args.ymin, ymax=eff_ymax_cv,
            data_key='after', use_percentage=False, ylabel="Cost",
            out_path=os.path.join(effective_out, "after.png"),
            hlines=[(hgs_val, "HGS")] if hgs_val else None,
        )
        _plot_one_figure(
            named_points=named_points, labels=None,
            break_mode="none", segments=0, xshift=0.0, dpi=args.dpi,
            ymin=args.ymin, ymax=eff_ymax_cv,
            data_key='after', use_percentage=True, ylabel="Cost",
            out_path=os.path.join(effective_out, "after_pct.png"),
            hlines=[(hgs_val, "HGS")] if hgs_val else None,
        )
    else:
        plot_from_points(
            named_points,
            labels=labels if use_custom_labels else None,
            break_mode="none",
            segments=0,
            xshift=0.0,
            dpi=args.dpi,
            ymin=args.ymin,
            ymax=args.ymax,
            out_dir=effective_out,
            hgs_costs=hgs_costs,
            feasible_only=not args.include_infeasible,
        )
    print(f"\nCurves saved to {effective_out}/ (log.txt, after.png, best_so_far.png, after_pct.png, best_so_far_pct.png)")


def cmd_landscape_curves(args):
    """Run six configs and plot mean±std curves."""
    if getattr(args, "sim_trace", False):
        os.environ["CUOPT_LOG_SIMILARITY"] = "1"
        import ctypes
        _libc = ctypes.CDLL(None)
        _libc.setenv.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        _libc.setenv.restype = ctypes.c_int
        _libc.getenv.argtypes = [ctypes.c_char_p]
        _libc.getenv.restype = ctypes.c_char_p
        _libc.setenv(b"CUOPT_LOG_SIMILARITY", b"1", 1)
        _val = _libc.getenv(b"CUOPT_LOG_SIMILARITY")
        print(f"[sim_trace] C-level getenv CUOPT_LOG_SIMILARITY = {_val}", flush=True)
    run_stamp = _timestamp_dir()
    curves_root = os.path.join(args.out_dir, "curves")
    os.makedirs(curves_root, exist_ok=True)
    # Each config: {name, diversity, emb_es, depth_es}
    all_configs = [
        {"name": "baseline",         "diversity": False, "emb_es": False, "depth_es": False},
        {"name": "similarity",       "diversity": True,  "emb_es": False, "depth_es": False},
        {"name": "early_stop_dup",   "diversity": False, "emb_es": True,  "depth_es": False},
        {"name": "early_stop_depth", "diversity": False, "emb_es": False, "depth_es": True},
        {"name": "sim_depth",        "diversity": True,  "emb_es": False, "depth_es": True},
        {"name": "sim_dup_depth",    "diversity": True,  "emb_es": True,  "depth_es": True},
    ]
    configs = list(all_configs)
    # Optional filter: --only_configs <csv> selects a subset of configs to run.
    only_cfgs = _parse_csv_tokens(getattr(args, "only_configs", None))
    if only_cfgs:
        before_names = [c["name"] for c in configs]
        configs = [c for c in configs if c["name"] in only_cfgs]
        if not configs:
            raise ValueError(
                f"--only_configs matched 0 configs. Got: {only_cfgs}. "
                f"Available: {before_names}"
            )
        print(f"[landscape] Filtered to configs (run): {[c['name'] for c in configs]}", flush=True)
    # `labels` controls which configs appear on plots/tables. Keep ALL config names
    # so that data loaded via --baseline_log (or any future --preload) still plots
    # alongside the actively-run configs. Without this, baseline curve disappears
    # from the plot even though baseline data IS in named_points.
    labels = [c["name"] for c in all_configs]
    named_points = []
    hgs_costs = None

    baseline_log_path = getattr(args, "baseline_log", None)
    if args.load_log:
        if baseline_log_path:
            print(
                "[landscape] --baseline_log ignored because --load_log provides the full combined log.",
                flush=True,
            )
        named_points, hgs_costs = _load_labeled_log(args.load_log)
        if not named_points:
            raise ValueError(f"No labeled runs found in log: {args.load_log}")
        effective_out = os.path.join(curves_root, f"landscape_replot_{run_stamp}")
        os.makedirs(effective_out, exist_ok=True)
        print(f"Loaded log from: {args.load_log}")
    else:
        # If --resume_dir is given, write into the existing dir (so this run's
        # configs get merged with previously-completed configs in same place).
        resume_dir = getattr(args, "resume_dir", None)
        if resume_dir:
            effective_out = resume_dir
            os.makedirs(effective_out, exist_ok=True)
            print(f"[landscape] Resume mode: writing into existing dir {effective_out}", flush=True)
        else:
            effective_out = os.path.join(curves_root, f"landscape_{run_stamp}")
            os.makedirs(effective_out, exist_ok=True)
        sim_ckpt = args.landscape_checkpoint or args.checkpoint
        es_ckpt = args.early_stop_checkpoint or args.landscape_checkpoint or args.checkpoint
        if not sim_ckpt:
            raise ValueError(
                "--landscape_checkpoint or --checkpoint is required for similarity / similarity_es "
                "(unless --load_log is provided)"
            )
        if not es_ckpt:
            raise ValueError(
                "--early_stop_checkpoint, --landscape_checkpoint, or --checkpoint is required "
                "for early_stop / similarity_es"
            )
        depth_ckpt = getattr(args, "depth_checkpoint", None)
        has_depth_cfg = any(c["depth_es"] for c in configs)
        if has_depth_cfg and not depth_ckpt:
            print(
                "[landscape] WARNING: early_stop_depth/sim_dup_depth configs require --depth_checkpoint "
                "(Stage-3 with advantage_head). Skipping depth configs.",
                flush=True,
            )
            configs = [c for c in configs if not c["depth_es"]]

        if args.early_stop_checkpoint:
            print(
                f"[landscape] similarity_ckpt (embedding_server)={sim_ckpt}\n"
                f"[landscape] early_stop_ckpt (Python callback)={es_ckpt}",
                flush=True,
            )
        else:
            print(
                f"[landscape] using same checkpoint for similarity + early_stop: {sim_ckpt}",
                flush=True,
            )
        if depth_ckpt:
            print(f"[landscape] depth_ckpt (advantage_head)={depth_ckpt}", flush=True)

        baseline_from_file = []
        if baseline_log_path:
            loaded_bp, hgs_from_base = _load_labeled_log(baseline_log_path)
            baseline_from_file = [(lab, pts) for lab, pts in loaded_bp if lab.startswith("baseline_")]
            if not baseline_from_file:
                raise ValueError(
                    f"No baseline_* sections in {baseline_log_path}. "
                    "Use a baseline.log.txt saved by landscape_curves, or a combined log containing baseline runs."
                )
            hgs_costs = hgs_from_base or {}
            print(
                f"[landscape] Loaded {len(baseline_from_file)} baseline run(s) from {baseline_log_path} "
                f"(solver will run similarity, early_stop, similarity_es only).",
                flush=True,
            )
            configs = [c for c in configs if c["name"] != "baseline"]
        else:
            hgs_costs = None

        # Set up timer compensation env var (C-level) once before any runs.
        # Safe to keep set during baseline: no IPC traffic → drain returns 0 → no-op.
        compensate = getattr(args, "compensate_time", False)
        time_mult = getattr(args, "landscape_time_multiplier", 1.0)
        if compensate:
            import ctypes as _ct
            _libc2 = _ct.CDLL(None)
            _libc2.setenv.argtypes = [_ct.c_char_p, _ct.c_char_p, _ct.c_int]
            _libc2.setenv.restype = _ct.c_int
            _libc2.setenv(b"CUOPT_LANDSCAPE_COMPENSATE_TIME", b"1", 1)
            os.environ["CUOPT_LANDSCAPE_COMPENSATE_TIME"] = "1"
            print("[landscape] Timer compensation enabled: embedding IPC time will NOT count against solver time limit.", flush=True)

        named_points = []
        if baseline_from_file:
            named_points.extend(baseline_from_file)

        imp_pct_val = getattr(args, "es_improvement_pct", None)

        for cfg in configs:
            cfg_name = cfg["name"]
            use_landscape = cfg["diversity"]
            use_emb = cfg["emb_es"]
            use_depth = cfg["depth_es"]
            use_cb = use_emb or use_depth  # callback needed for either embedding ES or depth ES

            print(f"\n--- {cfg_name} ---")
            cfg_time_limit = args.time_limit
            if use_landscape and not compensate and time_mult != 1.0:
                cfg_time_limit = args.time_limit * time_mult
                print(f"[landscape] time_limit={cfg_time_limit:.1f}s (base={args.time_limit}s × {time_mult}x)", flush=True)

            es_base = "embedding" if use_emb else "random"
            run_es_ckpt = es_ckpt if use_emb else args.checkpoint
            es_cls_path = args.classifier_pkl if use_emb else None
            es_threshold_val = getattr(args, "es_threshold", None) if use_emb else None
            es_interval = getattr(args, "es_check_interval", 1)
            es_cls_type = getattr(args, "classifier_type", "threshold") if use_emb else "threshold"
            es_timeout = getattr(args, "callback_timeout", 30.0)
            per_cfg_landscape_ckpt = sim_ckpt if use_landscape else None
            cfg_imp_pct = imp_pct_val if use_depth else None
            cfg_depth_ckpt = getattr(args, "depth_checkpoint", None) if use_depth else None

            _, _, all_run_points, hgs_new, *_ = run_experiment(
                data_path=args.data_path,
                problem_path=args.problem_path,
                solution_path=args.solution_path,
                time_limit=cfg_time_limit,
                n_instances=args.n_instances,
                start_index=args.start_index,
                problem_type=args.problem_type,
                scale=args.scale,
                n_vehicles=args.n_vehicles,
                n_runs=args.n_runs,
                use_callback=use_cb,
                collect_data=True,
                early_stop_base=es_base,
                checkpoint_path=run_es_ckpt,
                classifier_path=es_cls_path,
                check_interval=es_interval,
                classifier_type=es_cls_type,
                callback_timeout=es_timeout,
                es_threshold=es_threshold_val,
                es_improvement_pct=cfg_imp_pct,
                depth_checkpoint_path=cfg_depth_ckpt,
                device=args.device,
                use_landscape_diversity=use_landscape,
                landscape_checkpoint_path=per_cfg_landscape_ckpt,
                landscape_socket_dir=args.landscape_socket_dir,
                landscape_device=args.landscape_device,
                sim_trace_out_dir=effective_out if os.environ.get("CUOPT_LOG_SIMILARITY") else None,
                sim_trace_label=cfg_name,
            )
            if hgs_costs is None or not hgs_costs:
                hgs_costs = hgs_new
            else:
                for k, v in (hgs_new or {}).items():
                    hgs_costs.setdefault(k, v)
            for label, points in all_run_points:
                named_points.append((f"{cfg_name}_{label}", points))

            # Incremental dump: only write THIS config's log per loop iteration.
            # Skip rewriting the combined log here — for sim_dup_depth alone the points
            # list can be ~1GB; rewriting it 6 times wastes huge I/O. Per-config logs
            # cover crash recovery (combined is rebuilt at the end of cmd_landscape_curves).
            cfg_named = [(lab, pts) for lab, pts in named_points if lab.startswith(f"{cfg_name}_")]
            if cfg_named:
                _write_log(cfg_named, hgs_costs, os.path.join(effective_out, f"{cfg_name}.log.txt"))
            print(f"[landscape] {cfg_name} done; per-config log written.", flush=True)

        if not named_points:
            print("No data collected. Check problem/solution paths.")
            return

        # Keep legacy log in run folder for backward compatibility.
        _write_log(named_points, hgs_costs, os.path.join(effective_out, "log.txt"))

    # Optional label/prefix filtering for plotting/re-logging.
    keep_prefixes = _parse_csv_tokens(getattr(args, "keep_prefixes", None))
    if keep_prefixes:
        available = sorted({lab.split("_inst", 1)[0] for lab, _ in named_points})
        before_n = len(named_points)
        named_points = _filter_named_points_by_prefix(named_points, keep_prefixes)
        after_n = len(named_points)
        if after_n == 0:
            raise ValueError(
                f"--keep_prefixes matched 0 runs. prefixes={keep_prefixes}. "
                f"Available config prefixes in log: {available or 'N/A'}"
            )
        print(
            f"[landscape] Kept {after_n}/{before_n} runs for plotting "
            f"(prefixes: {', '.join(keep_prefixes)})",
            flush=True,
        )

    # Save timestamped combined log directly under curves/.
    combined_log_path = os.path.join(curves_root, f"log_{run_stamp}.txt")
    _write_log(named_points, hgs_costs, combined_log_path)

    # Also save per-setting logs for convenient reload/inspection.
    # In resume / only_configs mode, restrict the rewrite to configs that were
    # actually run this invocation. Otherwise the empty-cfg branch overwrites
    # previously-completed peer log files (similarity.log.txt etc.) with stubs,
    # destroying data from earlier resume passes.
    cfgs_run_this_call = set()
    if 'configs' in dir() and isinstance(configs, list):
        cfgs_run_this_call = {c["name"] for c in configs}
    only_cfgs_set = set(only_cfgs) if 'only_cfgs' in dir() and only_cfgs else set()
    is_partial_run = bool(only_cfgs_set) or bool(getattr(args, "resume_dir", None))
    for cfg_name in labels:
        cfg_points = [(lab, pts) for lab, pts in named_points if lab.startswith(f"{cfg_name}_")]
        cfg_log_path = os.path.join(effective_out, f"{cfg_name}.log.txt")
        if cfg_points:
            _write_log(cfg_points, hgs_costs, cfg_log_path)
        elif is_partial_run and cfg_name not in cfgs_run_this_call:
            # Skip stub-writing for configs we didn't run; keep any existing log file.
            continue
        else:
            with open(cfg_log_path, "w", encoding="utf-8") as f:
                f.write(f"# {cfg_name}: no callback points collected\n")
            print(f"Saved log to: {cfg_log_path} (empty — no callback points)", flush=True)

    # Group runs by config prefix to avoid dropping a config when some runs
    # have no callback points (and thus no entry in named_points).
    config_curves = {}
    for cfg_name in labels:
        run_list = [pts for lab, pts in named_points if lab.startswith(f"{cfg_name}_")]
        if run_list:
            config_curves[cfg_name] = run_list

    hgs_val = None
    if hgs_costs and 0 in hgs_costs:
        hgs_val = hgs_costs[0]

    # Auto ymax = HGS cost + 1 (unless user explicitly set --ymax)
    effective_ymax = args.ymax
    if effective_ymax is None and hgs_val is not None:
        effective_ymax = hgs_val + 1.0

    _plot_best_so_far_interval(
        config_curves,
        out_path=os.path.join(effective_out, "best_so_far.png"),
        out_path_pct=os.path.join(effective_out, "best_so_far_pct.png"),
        dpi=args.dpi,
        ymin=args.ymin,
        ymax=effective_ymax,
        hgs_cost=hgs_val,
        colors=["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#8c564b", "#9467bd"],
        feasible_only=not args.include_infeasible,
    )

    # Wall-clock version: shows real impact of neural network overhead
    has_wc = any(
        p.get('wall_clock') is not None
        for _, pts in named_points for p in pts
        if not math.isnan(p.get('after', float('nan')))
    )
    if has_wc:
        _plot_best_so_far_wallclock(
            config_curves,
            out_path=os.path.join(effective_out, "best_so_far_wallclock.png"),
            out_path_pct=os.path.join(effective_out, "best_so_far_wallclock_pct.png"),
            time_limit=args.time_limit,
            dpi=args.dpi,
            ymin=args.ymin,
            ymax=effective_ymax,
            hgs_cost=hgs_val,
            colors=["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#8c564b", "#9467bd"],
            feasible_only=not args.include_infeasible,
        )

    _plot_one_figure(
        named_points=named_points,
        labels=None,
        break_mode="none",
        segments=0,
        xshift=0.0,
        dpi=args.dpi,
        ymin=args.ymin,
        ymax=effective_ymax,
        data_key="after",
        use_percentage=True,
        ylabel="Cost",
        out_path=os.path.join(effective_out, "after_pct.png"),
        hlines=[(hgs_val, "HGS")] if hgs_val else None,
    )
    _write_endpoint_summary_table(
        config_curves,
        out_dir=effective_out,
        time_limit=args.time_limit,
        hgs_val=hgs_val,
        feasible_only=not args.include_infeasible,
    )

    print(
        f"\nLandscape comparison curves saved to {effective_out}/ "
        f"(best_so_far.png, best_so_far_pct.png, after_pct.png, endpoint_summary.{{csv,md}}). "
        f"Combined log: {combined_log_path}"
    )

    # Auto-analyze sim_trace if enabled
    if os.environ.get("CUOPT_LOG_SIMILARITY"):
        _analyze_sim_trace_and_gen_excel(effective_out, dpi=args.dpi)


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="cuOpt CVRP solver and log plotter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ---- solve ----
    sp = sub.add_parser("solve", help="Run cuOpt solver on CVRP instances")
    sp.add_argument("--data_path", default=None, type=str,
                    help="Path to txt dataset file (original format)")
    sp.add_argument("--problem_size", type=int, default=100, help="CVRP instance size (100 or 1000)")
    sp.add_argument("--problem_path", type=str, default=None,
                    help="Path to CVRP problem pkl file")
    sp.add_argument("--solution_path", type=str, default=None,
                    help="Path to HGS solution pkl file")
    # sp.add_argument("--problem_path", default="/home/jieyi/CaR-constraint/data/CVRP/cvrp1000_uniform_LV0.pkl", type=str,
    #                 help="Path to CVRP problem pkl file")
    # sp.add_argument("--solution_path", default="/home/jieyi/CaR-constraint/data/CVRP/hgs_cvrp1000_uniform_LV0.pkl", type=str,
    #                 help="Path to HGS solution pkl file")
    # sp.add_argument("--problem_path", default="/home/jieyi/cvrp100_uniform.pkl", type=str,
    #                 help="Path to CVRP problem pkl file")
    # sp.add_argument("--solution_path", default="/home/jieyi/hgs_cvrp100_uniform.pkl", type=str,
    #                 help="Path to HGS solution pkl file")
    sp.add_argument("--time_limit", type=float, default=5, help="Time limit per instance (seconds)")
    sp.add_argument("--n_instances", type=int, default=1, help="Number of instances to run")
    sp.add_argument("--start_index", type=int, default=60, help="Starting instance index")
    sp.add_argument("--problem_type", type=str, default="CVRP", help="Problem type (default: CVRP)")
    sp.add_argument("--n_vehicles", type=int, default=100, help="Number of vehicles")
    sp.add_argument("--scale", type=float, default=1e2, help="Coordinate scale")
    sp.add_argument("--n_runs", type=int, default=1, help="Number of runs per instance (best is kept)")
    sp.add_argument("--use_callback", action='store_true', help="Use early stop callback")
    # Early stop options
    sp.add_argument("--early_stop_base", type=str, default="random",
                    choices=["random", "embedding", "structure"],
                    help="Early stop strategy (default: random)")
    sp.add_argument("--checkpoint", type=str,
                    default="/home/jieyi/cuopt/out/20260210_204422_0-49_2stages_stage1e100_stage2e50/s1_epoch100.pt",
                    help="Path to embedder checkpoint .pt (required for --early_stop_base embedding)")
    sp.add_argument("--classifier_pkl", type=str,
                    default="/home/jieyi/cuopt/out/20260210_204422_0-49_2stages_stage1e100_stage2e50/convergence/trial_convergence_s1_epoch100_classifiers.pkl",
                    help="Path to convergence classifier .pkl (required for embedding/structure)")
    sp.add_argument("--check_interval", type=int, default=1,
                    help="Check convergence every N iterations")
    sp.add_argument("--classifier_type", type=str, default="lr",
                    choices=["threshold", "lr"],
                    help="Classifier method: 'threshold' (d<tau) or 'lr' (LogisticRegression)")
    sp.add_argument("--callback_timeout", type=float, default=30.0,
                    help="Max seconds for early-stop callback; on timeout solver continues (default 30)")
    sp.add_argument("--device", type=str, default="cuda", help="Torch device for embedder (default: cuda)")
    sp.add_argument("--landscape_device", type=str, default=None,
                    help="Torch device for embedding_server in landscape diversity mode (default: same as --device).")
    sp.add_argument("--landscape_diversity", action="store_true",
                    help="Use landscape-aware diversity metric in C++ and auto-manage embedding server per instance.")
    sp.add_argument("--landscape_checkpoint", type=str, default=None,
                    help="Path to checkpoint for landscape diversity embedding server (defaults to --checkpoint).")
    sp.add_argument("--landscape_socket_dir", type=str, default="/tmp",
                    help="Directory for per-instance Unix socket files (default: /tmp).")
    # Auto-plot options
    sp.add_argument("--plot", action='store_true', help="Auto-plot convergence after solving")
    sp.add_argument("--log", type=str, default=None, help="Save log to this file (for later re-plotting)")
    sp.add_argument("--save_upper_bound_log", action="store_true",
                    help="After solve, write upper_bound_log.json for UpperBoundAnalyzer (uses first run's callback data).")
    sp.add_argument(
        "--ub_basin_key",
        type=str,
        default="hash",
        choices=["hash", "cost"],
        help=(
            "Basin identity used in upper_bound_log.json: "
            "'hash' = callback solution_hash (edge-based, default); "
            "'cost' = per-trial best cost (coarser, matches tl_sensitivity)."
        ),
    )
    sp.add_argument("--out_dir", type=str, default=".", help="Output directory for plots and logs (default: cwd)")
    sp.add_argument("--dpi", type=int, default=160, help="Plot DPI (default 160)")
    sp.add_argument("--ymin", type=float, default=None, help="Y-axis lower bound (auto from HGS if omitted)")
    sp.add_argument("--ymax", type=float, default=None, help="Y-axis upper bound (auto from HGS if omitted)")
    sp.add_argument("--include-infeasible", action="store_true",
                    help="Include infeasible points in best_so_far / best_so_far_pct (default: feasible only).")
    sp.add_argument("--trace", action="store_true",
                    help="Enable evolution tracing: write weights & route data to trace.csv for visualization.")
    sp.add_argument("--sim_trace", action="store_true",
                    help="Enable per-run similarity trace logging and plotting (sets CUOPT_LOG_SIMILARITY=1).")

    # ---- plot ----
    pp = sub.add_parser("plot", help="Plot cost curves from solver log files")
    pp.add_argument("files", nargs="*", help="Log file(s). If omitted, reads from stdin.")
    pp.add_argument("--out", "-o", type=str, default="after_multi.png", help="Output image path.")
    pp.add_argument("--out-best", type=str, default="best_so_far_multi.png",
                    help="Best-so-far cost output image path.")
    pp.add_argument("--out-pct", type=str, default="after_multi_pct.png",
                    help="Cost-after percentage plot output path.")
    pp.add_argument("--out-best-pct", type=str, default="best_so_far_multi_pct.png",
                    help="Best-so-far cost percentage plot output path.")
    pp.add_argument("--dpi", type=int, default=160, help="Save resolution DPI (default 160).")
    pp.add_argument("--xshift", type=float, default=0.0,
                    help="Additive offset for x-axis cumulative iteration.")
    pp.add_argument("--break", dest="break_mode", choices=["none", "time", "offset"], default="time",
                    help="Segment detection mode: none|time|offset (default time).")
    pp.add_argument("--segments", "-s", type=int, default=10000000000,
                    help="Plot only the first N segments; N<=0 means all. Applied per file.")
    pp.add_argument("--labels", type=str, default=None,
                    help="Custom legend labels, comma-separated, must match file count.")
    pp.add_argument("--ymax", type=float, default=None, help="Y-axis upper bound (default: auto).")
    pp.add_argument("--ymin", type=float, default=None, help="Y-axis lower bound (default: auto).")
    pp.add_argument("--scale", type=float, default=1.0,
                    help="Divide all parsed cost values by this factor (e.g. 100 for C++ logs with scale=1e2).")
    pp.add_argument("--hgs_costs", type=str, default=None,
                    help="HGS reference costs, comma-separated (e.g. '1580.12,1690.50'). Draws red dashed lines.")
    pp.add_argument("--include-infeasible", action="store_true",
                    help="Include infeasible points in best_so_far / best_so_far_pct (default: feasible only).")

    # ---- compare ----
    cp = sub.add_parser("compare",
                        help="Run 5 early-stop configs and plot a boxplot of best costs")
    cp.add_argument("--data_path", default=None, type=str,
                    help="Path to txt dataset file (original format)")
    cp.add_argument("--problem_path", default="/home/jieyi/cvrp100_uniform.pkl", type=str,
                    help="Path to CVRP problem pkl file")
    cp.add_argument("--solution_path", default="/home/jieyi/hgs_cvrp100_uniform.pkl", type=str,
                    help="Path to HGS solution pkl file")
    cp.add_argument("--time_limit", type=float, default=5, help="Time limit per instance (seconds)")
    cp.add_argument("--n_instances", type=int, default=1, help="Number of instances to run")
    cp.add_argument("--start_index", type=int, default=0, help="Starting instance index")
    cp.add_argument("--problem_type", type=str, default="CVRP", help="Problem type (default: CVRP)")
    cp.add_argument("--n_vehicles", type=int, default=30, help="Number of vehicles")
    cp.add_argument("--scale", type=float, default=1e2, help="Coordinate scale")
    cp.add_argument("--n_repeat", type=int, default=10,
                    help="Number of independent repetitions per config (default 10)")
    cp.add_argument("--checkpoint", type=str,
                    default="/home/jieyi/cuopt/out/20260210_204422_0-49_2stages_stage1e100_stage2e50/s1_epoch100.pt",
                    help="Path to embedder checkpoint .pt")
    cp.add_argument("--classifier_pkl", type=str,
                    default="/home/jieyi/cuopt/out/20260210_204422_0-49_2stages_stage1e100_stage2e50/convergence/trial_convergence_s1_epoch100_classifiers.pkl",
                    help="Path to convergence classifier .pkl")
    cp.add_argument("--check_interval", type=int, default=1,
                    help="Check convergence every N iterations (default 1)")
    cp.add_argument("--device", type=str, default="cuda", help="Torch device (default: cuda)")
    cp.add_argument("--out", type=str, default="compare",
                    help="Output filename prefix (default 'compare' -> compare_cost.png, compare_stats.png, compare_data.json)")
    cp.add_argument("--out_dir", type=str, default=".", help="Output directory (default: cwd)")
    cp.add_argument("--dpi", type=int, default=160, help="Plot DPI (default 160)")

    # ---- curves ----
    cv = sub.add_parser("curves",
                        help="Run no ES / emb+lr / struct+lr and plot after & best_so_far curves")
    cv.add_argument("--data_path", default=None, type=str)
    cv.add_argument("--problem_path", default="/home/jieyi/cvrp100_uniform.pkl", type=str)
    cv.add_argument("--solution_path", default="/home/jieyi/hgs_cvrp100_uniform.pkl", type=str)
    cv.add_argument("--time_limit", type=float, default=5)
    cv.add_argument("--n_instances", type=int, default=1)
    cv.add_argument("--start_index", type=int, default=0)
    cv.add_argument("--problem_type", type=str, default="CVRP")
    cv.add_argument("--n_vehicles", type=int, default=21)
    cv.add_argument("--scale", type=float, default=1e2)
    cv.add_argument("--n_runs", type=int, default=1, help="Runs per instance (each = one curve per config)")
    cv.add_argument("--checkpoint", type=str,
                    default="/home/jieyi/cuopt/out/20260210_204422_0-49_2stages_stage1e100_stage2e50/s1_epoch100.pt")
    cv.add_argument("--classifier_pkl", type=str,
                    default="/home/jieyi/cuopt/out/20260210_204422_0-49_2stages_stage1e100_stage2e50/convergence/trial_convergence_s1_epoch100_classifiers.pkl")
    cv.add_argument("--check_interval", type=int, default=1)
    cv.add_argument("--device", type=str, default="cuda")
    cv.add_argument("--out_dir", type=str, default=".")
    cv.add_argument("--dpi", type=int, default=160)
    cv.add_argument("--ymin", type=float, default=14)
    cv.add_argument("--ymax", type=float, default=16)
    cv.add_argument("--include-infeasible", action="store_true",
                    help="Include infeasible points in best_so_far / best_so_far_pct (default: feasible only).")

    # ---- landscape_curves ----
    lc = sub.add_parser(
        "landscape_curves",
        help="Four-way compare: baseline, similarity (diversity), early_stop, similarity+early_stop",
    )
    lc.add_argument("--data_path", default=None, type=str)
    lc.add_argument("--problem_path", default="/home/jieyi/cvrp100_uniform.pkl", type=str)
    lc.add_argument("--solution_path", default="/home/jieyi/hgs_cvrp100_uniform.pkl", type=str)
    lc.add_argument("--time_limit", type=float, default=5)
    lc.add_argument("--n_instances", type=int, default=1)
    lc.add_argument("--start_index", type=int, default=0)
    lc.add_argument("--problem_type", type=str, default="CVRP")
    lc.add_argument("--n_vehicles", type=int, default=21)
    lc.add_argument("--scale", type=float, default=1e2)
    lc.add_argument("--n_runs", type=int, default=20, help="Runs per config (default: 20)")
    lc.add_argument("--checkpoint", type=str, default=None, help="Optional fallback checkpoint path")
    lc.add_argument("--landscape_checkpoint", type=str, required=False,
                    help="Checkpoint for embedding_server (C++ similarity / diversity metric)")
    lc.add_argument("--early_stop_checkpoint", type=str, required=False,
                    help="Checkpoint for Python early-stop embedder; default: same as --landscape_checkpoint "
                         "or --checkpoint")
    lc.add_argument("--load_log", type=str, default=None,
                    help="Path to a previously saved landscape log.txt; if set, skip solving and replot only.")
    lc.add_argument("--resume_dir", type=str, default=None,
                    help="Existing landscape_<TS> dir to write into (instead of creating a new one). "
                         "Pair with --only_configs and --baseline_log to resume an interrupted run.")
    lc.add_argument("--baseline_log", type=str, default=None,
                    help="Path to baseline.log.txt with baseline_* sections. "
                         "If set, baseline curves load from disk; solver runs similarity, early_stop, similarity_es.")
    lc.add_argument("--landscape_socket_dir", type=str, default="/tmp")
    lc.add_argument("--classifier_pkl", type=str, default=None,
                    help="Path to convergence classifier .pkl (optional; required for --early_stop with --classifier_type lr)")
    lc.add_argument("--es_threshold", type=float, default=0.6,
                    help="Embedding distance threshold for early stop (used when --classifier_pkl is not set). "
                         "Solutions with distance < threshold to a known local optimum trigger early stop. Default: 0.6")
    lc.add_argument("--es_check_interval", type=int, default=5,
                    help="Check early stop every N local-search iterations (default: 5)")
    lc.add_argument("--es_improvement_pct", type=float, default=0.05,
                    help="Depth-based early stop: skip trials whose predicted local optimum won't beat "
                         "best_so_far by this fraction. E.g. 0.05 = 5%%. Default: 0.05")
    lc.add_argument("--depth_checkpoint", type=str, default=None,
                    help="Stage-3 checkpoint with advantage_head for NN depth prediction. "
                         "Required for depth_stop / similarity_depth configs.")
    lc.add_argument("--classifier_type", type=str, default="threshold",
                    choices=["threshold", "lr"],
                    help="Early stop classifier method (default: threshold)")
    lc.add_argument("--callback_timeout", type=float, default=30.0,
                    help="Max seconds for early-stop callback; on timeout solver continues (default: 30)")
    lc.add_argument("--device", type=str, default="cuda")
    lc.add_argument("--landscape_device", type=str, default=None,
                    help="Torch device for embedding_server (default: same as --device).")
    lc.add_argument("--keep_prefixes", type=str, default=None,
                    help="Comma-separated run label prefixes to keep for plotting/re-logging. "
                         "Examples: baseline,similarity_es or baseline_inst0_run0.")
    lc.add_argument("--only_configs", type=str, default=None,
                    help="Comma-separated config names to actually run (skip others). "
                         "Useful when re-running a subset; pair with --baseline_log to reuse baseline. "
                         "Available: baseline,similarity,early_stop_dup,early_stop_depth,sim_depth,sim_dup_depth.")
    lc.add_argument("--sim_trace", action="store_true",
                    help="Enable per-run similarity trace logging and plotting (sets CUOPT_LOG_SIMILARITY=1).")
    lc.add_argument("--compensate_time", action="store_true",
                    help="Compensate solver timer for embedding IPC overhead (sets CUOPT_LANDSCAPE_COMPENSATE_TIME=1). "
                         "With this, embedding computation time does NOT count against the solver time limit.")
    lc.add_argument("--landscape_time_multiplier", type=float, default=1.0,
                    help="Multiply time_limit by this factor for landscape runs (e.g., 2.0 gives landscape 2x more time). "
                         "Ignored when --compensate_time is set.")
    lc.add_argument("--out_dir", type=str, default=".")
    lc.add_argument("--dpi", type=int, default=160)
    lc.add_argument("--ymin", type=float, default=None)
    lc.add_argument("--ymax", type=float, default=None)
    lc.add_argument("--include-infeasible", action="store_true")

    args = parser.parse_args()
    if hasattr(args, "problem_size") and not getattr(args, "problem_path", None):
        if args.problem_size == 100:
            args.problem_path = "/home/jieyi/cvrp100_uniform.pkl"
            args.solution_path = "/home/jieyi/hgs_cvrp100_uniform.pkl"
        elif args.problem_size == 1000:
            args.problem_path = "/home/jieyi/CaR-constraint/data/CVRP/cvrp1000_uniform_LV0.pkl"
            args.solution_path = "/home/jieyi/CaR-constraint/data/CVRP/hgs_cvrp1000_uniform_LV0.pkl"
        else:
            raise ValueError(f"Invalid problem size: {args.problem_size}")

    if args.command == "solve":
        cmd_solve(args)
    elif args.command == "plot":
        cmd_plot(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "curves":
        cmd_curves(args)
    elif args.command == "landscape_curves":
        cmd_landscape_curves(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
