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
import math
import os
import random
import re
import sys

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

# ── Log parsing & series building ───────────────────────────────────

def parse_points(text: str):
    """
    Scan the log sequentially and build a point sequence:
      Each cost before/after pair -> new point (after value).
      If followed by time/offset -> bind to that point (first match only).
    Returns: [{'after': float, 'best_so_far': float, 'time': float|None, 'offset': float|None}, ...]
    """
    pts = []
    best_so_far = float('inf')

    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln == "# --- break ---":
            pts.append({
                'after': float('nan'),
                'best_so_far': float('nan'),
                'time': None,
                'offset': None,
            })
            continue
        m_cost = COST_PAIR_RE.search(ln)
        if m_cost:
            after_cost = float(m_cost.group(2))
            best_so_far = min(best_so_far, after_cost)
            pts.append({
                'after': after_cost,
                'best_so_far': best_so_far,
                'time': None,
                'offset': None,
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
                     hlines=None):
    """Render one figure to *out_path*.

    Args:
        named_points: [(name, [point_dict, ...]), ...]
        hlines: Optional list of (value, label) for horizontal reference lines.
    Returns True if anything was plotted.
    """
    plt.figure(figsize=(9.5, 5.5))
    any_plotted = False

    for idx, (name, pts) in enumerate(named_points):
        if not pts:
            if not use_percentage and data_key == 'after':
                print(f"[INFO] '{name}' has no data points, skipping.")
            continue
        xs, ys = build_series(
            pts,
            break_mode=break_mode,
            max_segments=(segments if segments > 0 else None),
            xshift=xshift,
            data_key=data_key,
            use_percentage=use_percentage,
        )
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


def plot_from_points(named_points, labels=None, break_mode="none", segments=0,
                     xshift=0.0, dpi=160, ymin=None, ymax=None, out_dir=".",
                     hgs_costs=None):
    """Generate all 4 standard figures from pre-parsed point lists.

    Args:
        hgs_costs: Optional dict {instance_index: cost} for HGS reference lines.
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


def _make_callback_class(scale=1.0, early_stop=False, early_stop_base="random",
                         embedder=None, env=None, classifier=None,
                         check_interval=1, classifier_type="threshold"):
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
    """
    import torch
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
            # Embedding mode: store embeddings
            self._local_optima_embs = []   # [(embedding, cost), ...]
            # Structure mode: store route-format solutions
            self._local_optima_sols = []   # [(solution_route, cost), ...]
            # Shared state
            self._prev_iteration = -1
            self._prev_solution_flat = None
            self._prev_objective = None
            self._restart_detected = False
            self.n_early_stops = 0
            self.n_iterations = 0
            self.n_trials = 1  # starts at 1 (first trial)

        # ── helpers: convert solution_flat ───────────────────────────

        def _to_route_solution(self, solution_flat):
            """Convert cuOpt solution_flat to route-format [0,n1,n2,...,0,...]."""
            from helper import solution_flat_to_solution
            return solution_flat_to_solution(list(solution_flat))

        # ── embedding mode ───────────────────────────────────────────

        def _embed_solution(self, solution_flat):
            """Embed a single solution. Returns (1, embedding_dim) tensor."""
            sol = self._to_route_solution(solution_flat)
            h = "_cb_tmp"
            self._env._basin_info[h] = {"solution": sol}
            ctx = self._env.prepare_from_hashes([h])
            with torch.no_grad():
                emb = self._embedder(ctx, self._env)
            del self._env._basin_info[h]
            return emb  # (1, embedding_dim)

        def _add_local_optimum_emb(self, solution_flat, objective):
            emb = self._embed_solution(solution_flat)
            self._local_optima_embs.append((emb, objective / self._scale))

        def _check_convergence_emb(self, current_emb):
            if not self._local_optima_embs:
                return False
            import numpy as np
            optima = torch.cat([e for e, _ in self._local_optima_embs], dim=0)
            dists = torch.cdist(current_emb, optima).squeeze(0)  # (K,)
            if self._classifier_type == "lr":
                d_np = dists.cpu().numpy().reshape(-1, 1)
                preds = self._classifier["clf_embed"].predict(d_np)
                return bool(preds.any())
            else:  # threshold
                return bool((dists < self._classifier["tau_embed"]).any())

        def _embedding_early_stop(self, solution_flat, objective, iteration):
            if self._restart_detected and self._prev_solution_flat is not None:
                self._add_local_optimum_emb(self._prev_solution_flat, self._prev_objective)
                self._restart_detected = False
            self._prev_solution_flat = list(solution_flat)
            self._prev_objective = objective
            if iteration % self._check_interval == 0 and self._local_optima_embs:
                current_emb = self._embed_solution(solution_flat)
                if self._check_convergence_emb(current_emb):
                    self.n_early_stops += 1
                    return True
            return False

        # ── structure (broken-pairs) mode ────────────────────────────

        def _add_local_optimum_struct(self, solution_flat, objective):
            sol = self._to_route_solution(solution_flat)
            self._local_optima_sols.append((sol, objective / self._scale))

        def _check_convergence_struct(self, current_sol):
            from helper import broken_pairs_ratio
            import numpy as np
            if not self._local_optima_sols:
                return False
            if self._classifier_type == "lr":
                dists = np.array([broken_pairs_ratio(current_sol, s)
                                  for s, _ in self._local_optima_sols]).reshape(-1, 1)
                preds = self._classifier["clf_struct"].predict(dists)
                return bool(preds.any())
            else:  # threshold
                tau = self._classifier["tau_struct"]
                for opt_sol, _ in self._local_optima_sols:
                    if broken_pairs_ratio(current_sol, opt_sol) < tau:
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

        # ── main callback ────────────────────────────────────────────

        def customize_early_stop(self, solution_flat, objective, num_routes, iteration):
            val = objective / self._scale
            self._best_so_far = min(self._best_so_far, val)

            self.n_iterations += 1

            # Detect restart: insert NaN gap to break the line between trials
            if self._prev_iteration >= 0 and iteration < self._prev_iteration:
                self.n_trials += 1
                self.points.append({
                    'after': float('nan'),
                    'best_so_far': float('nan'),
                    'time': None,
                    'offset': None,
                })
                self._restart_detected = True

            self._prev_iteration = iteration

            self.points.append({
                'after': val,
                'best_so_far': self._best_so_far,
                'time': None,
                'offset': None,
            })
            if self._early_stop:
                if self._early_stop_base == "embedding":
                    return self._embedding_early_stop(solution_flat, objective, iteration)
                elif self._early_stop_base == "structure":
                    return self._structure_early_stop(solution_flat, objective, iteration)
                elif self._early_stop_base == "random":
                    return random.random() < 0.1
            return False

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
    embedder.encoder.load_state_dict(ckpt["encoder_state"])
    embedder.eval()
    print(f"Loaded embedder from {checkpoint_path} (stage {ckpt.get('stage', '?')}, epoch {ckpt.get('epoch', '?')})")

    with open(classifier_path, "rb") as f:
        classifier = pickle.load(f)
    print(f"Loaded classifier from {classifier_path} (tau_embed={classifier.get('tau_embed', '?'):.4f})")

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
    """Run the cuOpt solver and return the solution (or None if infeasible)."""
    from cuopt import routing
    solver_settings = routing.SolverSettings()
    solver_settings.set_time_limit(time_limit)
    if callback:
        solver_settings.set_routing_callback(callback)
    solution = routing.Solve(data_model, solver_settings)
    return solution if solution.get_status() == 0 else None


def run_experiment(
    data_path=None,
    problem_path=None,
    solution_path=None,
    time_limit=10,
    n_instances=1,
    start_index=0,
    problem_type="CVRP",
    scale=1e2,
    n_vehicles=21,
    n_runs=1,
    use_callback=False,
    collect_data=False,
    early_stop_base="random",
    checkpoint_path=None,
    classifier_path=None,
    check_interval=1,
    classifier_type="threshold",
    device="cuda",
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
        with open(classifier_path, "rb") as f:
            classifier = pickle.load(f)
        print(f"Loaded classifier from {classifier_path} (tau_struct={classifier.get('tau_struct', '?'):.4f})")

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
    with tqdm(total=total, desc="Solving with cuOpt") as pbar:
        for i in range(n_instances):
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
            ) if need_callback else None

            for k in range(n_runs):
                callback = CallbackCls() if need_callback else None
                model = get_cuopt_model(i, raw_dist, raw_demand, raw_cap, n_vehicles, scale)
                solution = solve_cuopt(model, time_limit, callback=callback)

                if solution:
                    cost = solution.get_total_objective() / scale
                    run_costs.append(cost)
                    gap = ((cost - raw_cost_value) / raw_cost_value) * 100
                    es_info = ""
                    if callback and hasattr(callback, 'n_early_stops'):
                        es_info = f" | EarlyStops: {callback.n_early_stops}"
                    print(f"[Instance {i} Run {k}] Cost: {cost:.2f} | Best Known: {raw_cost_value:.2f} | Gap: {gap:.2f}%{es_info}")
                else:
                    print(f"[Instance {i} Run {k}] No feasible solution.")

                # Accumulate callback stats
                if callback and hasattr(callback, 'n_early_stops'):
                    inst_es += callback.n_early_stops
                    inst_iters += callback.n_iterations
                    inst_trials += callback.n_trials

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

    return best_costs, best_gaps, all_run_points, hgs_costs, early_stop_counts, iteration_counts, trial_counts

# ── Subcommand handlers ─────────────────────────────────────────────

def _write_log(all_run_points, hgs_costs, filepath):
    """Write all callback points to a single log file compatible with parse_points()."""
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(filepath, 'w') as f:
        for label, points in all_run_points:
            inst_idx = int(label.split("_")[0].replace("inst", ""))
            hgs_cost = hgs_costs.get(inst_idx, 0)
            f.write(f"# === {label} (HGS: {hgs_cost}) ===\n")
            for p in points:
                if math.isnan(p['after']):
                    f.write("# --- break ---\n")  # trial gap marker
                    continue
                f.write(f"cost before: 0, cost after: {p['after']}\n")
    print(f"Saved log to: {filepath}")


def cmd_solve(args):
    """Execute the 'solve' subcommand."""
    need_collect = args.plot or (args.log is not None)
    costs, gaps, all_run_points, hgs_costs, *_ = run_experiment(
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
        device=args.device,
    )

    # Save log file
    if args.log is not None and all_run_points:
        _write_log(all_run_points, hgs_costs, args.log)

    # Auto-plot from collected callback data
    if args.plot and all_run_points:
        out_dir = args.out_dir
        os.makedirs(out_dir, exist_ok=True)
        plot_from_points(
            all_run_points,
            dpi=args.dpi,
            ymin=args.ymin,
            ymax=args.ymax,
            out_dir=out_dir,
            hgs_costs=hgs_costs,
        )


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
    for name, raw in inputs:
        pts = parse_points(raw)
        if not pts:
            print(f"[INFO] '{name}' has no 'cost before/after' lines, skipping.")
        named_points.append((name, pts))

    # Process labels
    labels = None
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]
        if len(labels) != len(named_points):
            print(f"[WARN] Label count ({len(labels)}) != file count ({len(named_points)}); ignoring custom labels.")
            labels = None

    # Parse --hgs_costs into hlines
    hlines = None
    if args.hgs_costs:
        vals = [float(v.strip()) for v in args.hgs_costs.split(",")]
        hlines = []
        for i, v in enumerate(vals):
            lab = "HGS" if len(vals) == 1 else f"HGS inst{i}"
            hlines.append((v, lab))

    common = dict(
        named_points=named_points, labels=labels,
        break_mode=args.break_mode, segments=args.segments, xshift=args.xshift,
        dpi=args.dpi, ymin=args.ymin, ymax=args.ymax, hlines=hlines,
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

    os.makedirs(args.out_dir, exist_ok=True)
    base, ext = os.path.splitext(args.out)
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
        p = os.path.join(args.out_dir, f"{base}_cost{suffix}{ext}")
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
        p = os.path.join(args.out_dir, f"{base}_stats{suffix}{ext}")
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
        p = os.path.join(args.out_dir, f"{base}_data{suffix}.json")
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
    sp.add_argument("--problem_path", default="/home/jieyi/cvrp100_uniform.pkl", type=str,
                    help="Path to CVRP problem pkl file")
    sp.add_argument("--solution_path", default="/home/jieyi/hgs_cvrp100_uniform.pkl", type=str,
                    help="Path to HGS solution pkl file")
    sp.add_argument("--time_limit", type=float, default=5, help="Time limit per instance (seconds)")
    sp.add_argument("--n_instances", type=int, default=1, help="Number of instances to run")
    sp.add_argument("--start_index", type=int, default=60, help="Starting instance index")
    sp.add_argument("--problem_type", type=str, default="CVRP", help="Problem type (default: CVRP)")
    sp.add_argument("--n_vehicles", type=int, default=21, help="Number of vehicles")
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
    sp.add_argument("--device", type=str, default="cuda", help="Torch device for embedder (default: cuda)")
    # Auto-plot options
    sp.add_argument("--plot", action='store_true', help="Auto-plot convergence after solving")
    sp.add_argument("--log", type=str, default=None, help="Save log to this file (for later re-plotting)")
    sp.add_argument("--out_dir", type=str, default=".", help="Output directory for plots and logs (default: cwd)")
    sp.add_argument("--dpi", type=int, default=160, help="Plot DPI (default 160)")
    sp.add_argument("--ymin", type=float, default=14, help="Y-axis lower bound (auto if omitted)")
    sp.add_argument("--ymax", type=float, default=16, help="Y-axis upper bound (auto if omitted)")

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
    pp.add_argument("--ymax", type=float, default=1800, help="Y-axis upper bound.")
    pp.add_argument("--ymin", type=float, default=1200, help="Y-axis lower bound (default 1200).")
    pp.add_argument("--hgs_costs", type=str, default=None,
                    help="HGS reference costs, comma-separated (e.g. '1580.12,1690.50'). Draws red dashed lines.")

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
    cp.add_argument("--n_vehicles", type=int, default=21, help="Number of vehicles")
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
    cp.add_argument("--out", type=str, default="compare_boxplot.png",
                    help="Output boxplot path (default compare_boxplot.png)")
    cp.add_argument("--out_dir", type=str, default=".", help="Output directory (default: cwd)")
    cp.add_argument("--dpi", type=int, default=160, help="Plot DPI (default 160)")

    args = parser.parse_args()

    if args.command == "solve":
        cmd_solve(args)
    elif args.command == "plot":
        cmd_plot(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
