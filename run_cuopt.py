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
            is_feasible = '[INFEASIBLE]' not in ln
            after_cost = float(m_cost.group(2))
            if is_feasible:
                best_so_far = min(best_so_far, after_cost)
            pts.append({
                'after': after_cost,
                'best_so_far': best_so_far,
                'time': None,
                'offset': None,
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
        feasible_only: If True (default) and data_key is 'best_so_far', only use points with is_feasible=True.
    Returns True if anything was plotted.
    """
    plt.figure(figsize=(9.5, 5.5))
    any_plotted = False

    for idx, (name, pts) in enumerate(named_points):
        if not pts:
            if not use_percentage and data_key == 'after':
                print(f"[INFO] '{name}' has no data points, skipping.")
            continue
        if feasible_only and data_key == 'best_so_far':
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

    if ymax is not None:
        plt.ylim(ymin, ymax)

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
    ax.set_ylabel("Best So Far Cost")
    ax.legend()
    ax.grid(True)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close()
    print(f"Saved Best So Far (interval) to: {out_path}")

    if out_path_pct and hgs_cost is not None and hgs_cost != 0:
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
                    pct = (s / hgs_cost) * 100.0
                    padded[i, :len(pct)] = pct
                    padded[i, len(pct):] = pct[-1]
            mean_y = np.nanmean(padded, axis=0)
            std_y = np.nanstd(padded, axis=0)
            # Normalize each config's x-axis by its own total iterations so all curves end at 100%.
            x = (np.arange(1, max_len + 1, dtype=float) / float(max_len)) * 100.0
            c = colors[idx % len(colors)]
            ax.plot(x, mean_y, color=c, linewidth=1.5, label=cfg_name)
            ax.fill_between(x, mean_y - std_y, mean_y + std_y, color=c, alpha=0.25)
        ax.axhline(100.0, color='red', linestyle='--', linewidth=1.5, label="HGS (100%)")
        ax.set_xlabel("Iteration Progress (%)")
        ax.set_ylabel("Best So Far Cost (% of HGS)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(out_path_pct, bbox_inches="tight", dpi=dpi)
        plt.close()
        print(f"Saved Best So Far % (interval) to: {out_path_pct}")


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
    ax.set_ylabel('Cost after')
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
                     ylabel="Cost after",       out_path=os.path.join(out_dir, "after.png"))
    _plot_one_figure(**common, data_key='best_so_far', use_percentage=False,
                     ylabel="Best So Far Cost", out_path=os.path.join(out_dir, "best_so_far.png"))
    _plot_one_figure(**common, data_key='after',       use_percentage=True,
                     ylabel="Cost after",       out_path=os.path.join(out_dir, "after_pct.png"))
    _plot_one_figure(**common, data_key='best_so_far', use_percentage=True,
                     ylabel="Best So Far Cost", out_path=os.path.join(out_dir, "best_so_far_pct.png"))

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
                         embedder=None, env=None, classifier=None,
                         check_interval=1, classifier_type="threshold",
                         callback_timeout=30.0,
                         demands=None, capacity=None):
    """Build SolverCallback class that records iteration data.

    Args:
        scale: Divide objective by this to get real cost.
        early_stop: If True, enable early stopping.
        early_stop_base: "embedding" | "structure" | "random".
        embedder: Pre-loaded SolutionEmbedder (required for embedding mode).
        env: Pre-loaded CVRPEnv with instance data (required for embedding mode).
        classifier: Dict with 'tau_embed'/'tau_struct' and 'clf_embed'/'clf_struct'.
        check_interval: Check convergence every N iterations.
        classifier_type: "threshold" (d < tau) or "lr" (LogisticRegression).
        callback_timeout: Max seconds for early-stop decision; on timeout return False (no stop) to avoid hang.
        demands: 1-D array of node demands (index 0 = depot = 0). Used for feasibility check.
        capacity: Vehicle capacity scalar. Used for feasibility check.
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
            self._env = env
            self._classifier = classifier
            self._check_interval = check_interval
            self._classifier_type = classifier_type
            self._demands = np.asarray(demands, dtype=np.float64) if demands is not None else None
            self._capacity = float(capacity) if capacity is not None else None
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

        # ── helpers: convert solution_flat ───────────────────────────

        def _to_route_solution(self, solution_flat):
            """Convert cuOpt solution_flat to route-format [0,n1,n2,...,0,...]."""
            from helper import solution_flat_to_solution
            return solution_flat_to_solution(list(solution_flat))

        def _check_feasible(self, sol):
            """Check capacity feasibility from route-format solution."""
            if self._demands is None or self._capacity is None:
                return True
            route_load = 0.0
            for node in sol:
                if node == 0:
                    if route_load > self._capacity:
                        return False
                    route_load = 0.0
                else:
                    route_load += self._demands[node]
            return route_load <= self._capacity

        # ── embedding mode ───────────────────────────────────────────

        def _embed_solution(self, solution_flat):
            """Embed a single solution. Returns (emb, from_cache). Caches when solution unchanged."""
            flat = list(solution_flat)
            if self._last_flat is not None and len(self._last_flat) == len(flat):
                if all(a == b for a, b in zip(self._last_flat, flat)):
                    return self._last_emb, True
            sol = self._to_route_solution(solution_flat)
            h = "_cb_tmp"
            self._env._basin_info[h] = {"solution": sol}
            ctx = self._env.prepare_from_hashes([h])
            with torch.no_grad():
                emb = self._embedder(ctx, self._env)
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

        # ── main callback ────────────────────────────────────────────

        def customize_early_stop(self, solution_flat, objective, num_routes, iteration):
            t0 = time.perf_counter()
            val = objective / self._scale

            self.n_iterations += 1

            # One trial = one C++ search ([search #N]). C++ resets iter per search, so iteration drops when a new search starts.
            if self._prev_iteration >= 0 and iteration < self._prev_iteration:
                self.n_trials += 1
                self.points.append({
                    'after': float('nan'),
                    'best_so_far': float('nan'),
                    'time': None,
                    'offset': None,
                })
                self._restart_detected = True
                self._prev_after = None  # first point of new trial uses before=val

            self._prev_iteration = iteration

            cost_before = self._prev_after if self._prev_after is not None else val

            sol = self._to_route_solution(solution_flat)
            solution_hash = _solution_to_edge_hash(sol)
            is_feasible = self._check_feasible(sol)

            if is_feasible:
                self._best_so_far = min(self._best_so_far, val)

            self.points.append({
                'before': cost_before,
                'after': val,
                'best_so_far': self._best_so_far,
                'time': None,
                'offset': None,
                'solution_hash': solution_hash,
                'is_feasible': is_feasible,
            })
            self._prev_after = val

            if self._early_stop:
                future = self._executor.submit(
                    self._run_early_stop_decision, list(solution_flat), objective, iteration
                )
                try:
                    out = future.result(timeout=self._callback_timeout)
                except (FuturesTimeoutError, TimeoutError, Exception):
                    out = False  # don't block: continue search
            else:
                out = False

            self._total_callback_time_ms += (time.perf_counter() - t0) * 1000
            return out

    return SolverCallback


def load_embedder_and_classifier(checkpoint_path, classifier_path, problem_size, device="cuda"):
    """Load pre-trained SolutionEmbedder and convergence classifier.

    Args:
        checkpoint_path: Path to embedder checkpoint (.pt).
        classifier_path: Path to convergence classifier (.pkl).
        problem_size: Number of customers (e.g. 100).
        device: torch device.

    Returns:
        (embedder, classifier_dict)
    """
    import pickle
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
    print(f"Loaded embedder from {checkpoint_path} (stage {ckpt.get('stage', '?')}, epoch {ckpt.get('epoch', '?')})")

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

    return embedder, classifier


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


def _run_solver_capture_stdout(data_model, time_limit, callback, log_path=None):
    """Run solver with fd-1 redirected to a pipe; return (solution, captured_text).
    Avoids deadlock by using a single pipe (no pty). Streams to terminal in real-time.
    Optionally streams to log_path."""
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
    device="cuda",
    log_path=None,
    trace_dir=None,
    use_landscape_diversity=False,
    landscape_checkpoint_path=None,
    landscape_socket_dir="/tmp",
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
    embedder, classifier = None, None
    if use_callback and early_stop_base == "embedding":
        if not checkpoint_path or not classifier_path:
            raise ValueError("--checkpoint and --classifier_pkl are required for embedding early stop")
        problem_size = raw_nodes.shape[1] - 1  # n+1 nodes, 1 depot
        embedder, classifier = load_embedder_and_classifier(
            checkpoint_path, classifier_path, problem_size, device=device
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

    # Build callback class if needed
    need_callback = use_callback or collect_data

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
    if log_path and need_callback:
        open(log_path, "w").close()

    # Optional: automatic landscape diversity via embedding_server.py
    landscape_proc = None
    landscape_socket_path = None
    prev_metric_env = os.environ.get("CUOPT_DIVERSITY_METRIC")
    prev_socket_env = os.environ.get("CUOPT_EMBEDDING_SOCKET")
    if use_landscape_diversity:
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
                    landscape_socket_path = os.path.join(
                        landscape_socket_dir, f"cuopt_embedding_{abs_index}.sock"
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
                        device,
                    ]
                    landscape_proc = subprocess.Popen(launch_cmd)
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

                # Build callback class per instance (env is instance-specific)
                CallbackCls = _make_callback_class(
                    scale=scale, early_stop=use_callback,
                    early_stop_base=early_stop_base,
                    embedder=embedder, env=env, classifier=classifier,
                    check_interval=check_interval, classifier_type=classifier_type,
                    callback_timeout=callback_timeout,
                    demands=raw_demand[i].numpy(),
                    capacity=raw_cap[i].item(),
                ) if need_callback else None

                for k in range(n_runs):
                    callback = CallbackCls() if need_callback else None
                    model = get_cuopt_model(i, raw_dist, raw_demand, raw_cap, n_vehicles, scale)
                    if need_callback:
                        solution, captured = _run_solver_capture_stdout(model, time_limit, callback, log_path=log_path if k == 0 and i == 0 else None)
                        if log_path and (i > 0 or k > 0):
                            with open(log_path, "a", encoding="utf-8", errors="replace") as lf:
                                lf.write(captured)
                        run_sum, run_max, run_counts = _parse_offset_ms_from_capture(captured)
                        total_offset_cpp_ms += run_sum
                        if run_max > max_offset_cpp_ms:
                            max_offset_cpp_ms = run_max
                        for x, c in run_counts.items():
                            offset_value_counts[x] = offset_value_counts.get(x, 0) + c
                    else:
                        print(f"[run_cuopt] Instance {i} Run {k}: calling solver (time_limit={time_limit}s)...", file=sys.stderr, flush=True)
                        solution = solve_cuopt(model, time_limit, callback=callback)
                        print(f"[run_cuopt] Instance {i} Run {k}: solver returned.", file=sys.stderr, flush=True)

                    if solution:
                        cost = solution.get_total_objective() / scale
                        run_costs.append(cost)
                        gap = ((cost - raw_cost_value) / raw_cost_value) * 100
                        es_info = f" | EarlyStops: {callback.n_early_stops}" if callback else ""
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
    """Write log: [search #N] then cost before/after lines."""
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
                f.write(f"cost before: {p.get('before', 0)}, cost after: {p['after']}\n")
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
    _plot_one_figure(**common, data_key='after',       use_percentage=False, ylabel="Cost after",       out_path=args.out)
    _plot_one_figure(**common, data_key='best_so_far', use_percentage=False, ylabel="Best So Far Cost", out_path=args.out_best)
    _plot_one_figure(**common, data_key='after',       use_percentage=True,  ylabel="Cost after",       out_path=args.out_pct)
    _plot_one_figure(**common, data_key='best_so_far', use_percentage=True,  ylabel="Best So Far Cost", out_path=args.out_best_pct)

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
        _plot_best_so_far_interval(
            config_curves,
            out_path=os.path.join(effective_out, "best_so_far.png"),
            out_path_pct=os.path.join(effective_out, "best_so_far_pct.png"),
            dpi=args.dpi, ymin=args.ymin, ymax=args.ymax, hgs_cost=hgs_val,
            colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
            feasible_only=not args.include_infeasible,
        )
        # After: still multi-line (all runs)
        _plot_one_figure(
            named_points=named_points, labels=None,
            break_mode="none", segments=0, xshift=0.0, dpi=args.dpi,
            ymin=args.ymin, ymax=args.ymax,
            data_key='after', use_percentage=False, ylabel="Cost after",
            out_path=os.path.join(effective_out, "after.png"),
            hlines=[(hgs_val, "HGS")] if hgs_val else None,
        )
        _plot_one_figure(
            named_points=named_points, labels=None,
            break_mode="none", segments=0, xshift=0.0, dpi=args.dpi,
            ymin=args.ymin, ymax=args.ymax,
            data_key='after', use_percentage=True, ylabel="Cost after (% of HGS)",
            out_path=os.path.join(effective_out, "after_pct.png"),
            hlines=[(100.0, "HGS (100%)")] if hgs_val else None,
            y_ref_cost=hgs_val,
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
    """Compare baseline vs landscape diversity over repeated runs and plot mean±std curves."""
    run_stamp = _timestamp_dir()
    curves_root = os.path.join(args.out_dir, "curves")
    os.makedirs(curves_root, exist_ok=True)
    configs = [
        ("baseline", False),
        ("landscape", True),
    ]
    labels = [c[0] for c in configs]
    named_points = []
    hgs_costs = None

    if args.load_log:
        named_points, hgs_costs = _load_labeled_log(args.load_log)
        if not named_points:
            raise ValueError(f"No labeled runs found in log: {args.load_log}")
        effective_out = os.path.join(curves_root, f"landscape_replot_{run_stamp}")
        os.makedirs(effective_out, exist_ok=True)
        print(f"Loaded log from: {args.load_log}")
    else:
        landscape_ckpt = args.landscape_checkpoint or args.checkpoint
        if not landscape_ckpt:
            raise ValueError(
                "--landscape_checkpoint or --checkpoint is required for landscape_curves "
                "(unless --load_log is provided)"
            )

        for cfg_name, use_landscape in configs:
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
                use_callback=False,
                collect_data=True,
                early_stop_base="random",
                checkpoint_path=args.checkpoint,
                classifier_path=args.classifier_pkl,
                check_interval=1,
                classifier_type="threshold",
                device=args.device,
                use_landscape_diversity=use_landscape,
                landscape_checkpoint_path=landscape_ckpt,
                landscape_socket_dir=args.landscape_socket_dir,
            )
            for label, points in all_run_points:
                named_points.append((f"{cfg_name}_{label}", points))

        if not named_points:
            print("No data collected. Check problem/solution paths.")
            return

        effective_out = os.path.join(curves_root, f"landscape_{run_stamp}")
        os.makedirs(effective_out, exist_ok=True)
        # Keep legacy log in run folder for backward compatibility.
        _write_log(named_points, hgs_costs, os.path.join(effective_out, "log.txt"))

    # Save timestamped combined log directly under curves/.
    combined_log_path = os.path.join(curves_root, f"log_{run_stamp}.txt")
    _write_log(named_points, hgs_costs, combined_log_path)

    # Also save per-setting logs for convenient reload/inspection.
    for cfg_name in labels:
        cfg_points = [(lab, pts) for lab, pts in named_points if lab.startswith(f"{cfg_name}_")]
        if cfg_points:
            cfg_log_path = os.path.join(effective_out, f"{cfg_name}.log.txt")
            _write_log(cfg_points, hgs_costs, cfg_log_path)

    n_per_config = args.n_instances * args.n_runs
    config_curves = {}
    for j, cfg_name in enumerate(labels):
        start = j * n_per_config
        end = start + n_per_config
        if end <= len(named_points):
            config_curves[cfg_name] = [named_points[k][1] for k in range(start, end)]

    hgs_val = None
    if hgs_costs and 0 in hgs_costs:
        hgs_val = hgs_costs[0]

    _plot_best_so_far_interval(
        config_curves,
        out_path=os.path.join(effective_out, "best_so_far.png"),
        out_path_pct=os.path.join(effective_out, "best_so_far_pct.png"),
        dpi=args.dpi,
        ymin=args.ymin,
        ymax=args.ymax,
        hgs_cost=hgs_val,
        colors=["#1f77b4", "#d62728"],
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
        ymax=args.ymax,
        data_key="after",
        use_percentage=True,
        ylabel="Cost after (% of HGS)",
        out_path=os.path.join(effective_out, "after_pct.png"),
        hlines=[(100.0, "HGS (100%)")] if hgs_val else None,
        y_ref_cost=hgs_val,
    )
    print(
        f"\nLandscape comparison curves saved to {effective_out}/ "
        f"(best_so_far.png, best_so_far_pct.png, after_pct.png). "
        f"Combined log: {combined_log_path}"
    )


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
        help="Compare baseline vs landscape diversity and plot mean±std curves",
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
                    help="Checkpoint used by embedding_server for landscape diversity")
    lc.add_argument("--load_log", type=str, default=None,
                    help="Path to a previously saved landscape log.txt; if set, skip solving and replot only.")
    lc.add_argument("--landscape_socket_dir", type=str, default="/tmp")
    lc.add_argument("--classifier_pkl", type=str, default=None, help="Unused here; kept for signature compatibility")
    lc.add_argument("--device", type=str, default="cuda")
    lc.add_argument("--out_dir", type=str, default=".")
    lc.add_argument("--dpi", type=int, default=160)
    lc.add_argument("--ymin", type=float, default=None)
    lc.add_argument("--ymax", type=float, default=None)
    lc.add_argument("--include-infeasible", action="store_true")

    args = parser.parse_args()
    if hasattr(args, "problem_size"):
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
