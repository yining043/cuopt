# -*- coding: utf-8 -*-
"""
Early-stop oracle experiment: simulate perfect vs noisy early stop with sunk cost.

Uses offline run data (log or points): split into trials, compute convergence
index per trial, then simulate:
  - Baseline: no early stop (run every trial to end).
  - Oracle 100%: stop exactly at convergence; measure speedup and quality.
  - Sunk cost: cap stop at max_stop_fraction of trial length (e.g. 0.2 = spend at most 20%
    per trial; if convergence is later, we stop at 20% and may get worse quality).
  - Error injection: oracle correct with probability `accuracy`; sweep to find acceptable accuracy.

Log format: use run_cuopt with --log to produce logs that contain "cost before/after" lines
and "# --- break ---" as trial boundaries.

Usage:
  python early_stop_oracle.py --log path/to/log.txt
  python early_stop_oracle.py --log run1.log run2.log --out_dir ./oracle_out
"""
import argparse
import math
import os
import random
import re
import sys

# Minimal log parsing (no heavy deps)
COST_PAIR_RE = re.compile(
    r"cost before:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*"
    r"cost after:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)
SEARCH_HEADER_RE = re.compile(r"\[search\s*#\s*\d+\]")


def parse_points(text: str):
    """Build point sequence from log text. Trial break = # --- break --- or [search #N]."""
    pts = []
    best_so_far = float("inf")
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln == "# --- break ---" or SEARCH_HEADER_RE.search(ln):
            pts.append({"after": float("nan"), "best_so_far": float("nan")})
            continue
        m = COST_PAIR_RE.search(ln)
        if m:
            after_cost = float(m.group(2))
            best_so_far = min(best_so_far, after_cost)
            pts.append({"after": after_cost, "best_so_far": best_so_far})
    return pts


def points_to_trials(points: list) -> list:
    """Split points into trials (each trial = list of point dicts, no NaN)."""
    trials = []
    current = []
    for p in points:
        if math.isnan(p.get("after", 0)) or math.isnan(p.get("best_so_far", 0)):
            if current:
                trials.append(current)
                current = []
        else:
            current.append(p)
    if current:
        trials.append(current)
    return trials


def convergence_index(trial_points: list, epsilon: float = 1e-9) -> int:
    """
    First index i where best_so_far will not improve further (trial has converged).
    Returns 0-based index; if we stop at this step we have spent (return_value + 1) iterations.
    """
    if not trial_points:
        return 0
    best = min(p["best_so_far"] for p in trial_points)
    for i, p in enumerate(trial_points):
        if abs(p["best_so_far"] - best) <= epsilon:
            return i
    return len(trial_points) - 1


def flatten_trials(trials: list) -> list:
    """Concatenate all trials in order. Returns list of point dicts."""
    out = []
    for t in trials:
        out.extend(t)
    return out


def count_oracle_real_checks(points: list) -> int:
    """
    In a sequence of points, 'check' only when best_so_far changes (or at first step).
    Like 111222 -> 2 checks (first 1, first 2). Returns 1 + (# times best_so_far changes).
    """
    if not points:
        return 0
    checks = 1
    for i in range(1, len(points)):
        if points[i]["best_so_far"] != points[i - 1]["best_so_far"]:
            checks += 1
    return checks


def get_required_checks_mask(points: list) -> list:
    """required[i]=True iff we must check at step i (first step or best_so_far changed)."""
    if not points:
        return []
    mask = [True]  # first step always required
    for i in range(1, len(points)):
        mask.append(points[i]["best_so_far"] != points[i - 1]["best_so_far"])
    return mask


def simulate_check_oracle_with_errors(
    flattened: list, n_window: int, accuracy: float, seed: int = 42
) -> tuple:
    """
    In window (first n_window steps), at each step we decide whether to 'check'.
    - Required step (first of state): check with prob=accuracy, skip with prob=(1-accuracy).
    - Skip step (same state as prev): check with prob=(1-accuracy), skip with prob=accuracy.
    Returns (actual_checks, all_required_done). all_required_done = we checked at every required step
    (e.g. 111222 with 2 required -> P(both correct) = accuracy^2 = 0.25 when accuracy=0.5).
    """
    window = flattened[:n_window] if n_window else []
    if not window:
        return 0, True
    required = get_required_checks_mask(window)
    rng = random.Random(seed)
    actual_checks = 0
    all_required_done = True
    for i in range(len(window)):
        need_check = required[i]
        do_check = rng.random() < (accuracy if need_check else (1.0 - accuracy))
        if do_check:
            actual_checks += 1
        if need_check and not do_check:
            all_required_done = False
    return actual_checks, all_required_done


def simulate_baseline(trials: list) -> tuple:
    """No early stop: run every trial to end. Returns (total_iterations, global_best_cost)."""
    total_it = 0
    global_best = float("inf")
    for t in trials:
        total_it += len(t)
        if t:
            global_best = min(global_best, t[-1]["best_so_far"])
    return total_it, global_best


def simulate_oracle_at_convergence(trials: list) -> tuple:
    """Stop each trial exactly at convergence step (no cap). Returns (total_iterations, global_best_cost)."""
    total_it = 0
    global_best = float("inf")
    for t in trials:
        if not t:
            continue
        conv_i = convergence_index(t)
        stop_i = conv_i
        total_it += stop_i + 1
        global_best = min(global_best, t[stop_i]["best_so_far"])
    return total_it, global_best


def simulate_oracle_perfect(trials: list, max_stop_fraction: float = 1.0) -> tuple:
    """
    Sunk cost: cap how much of each trial we run.
    max_stop_fraction=1.0: no early stop, run full trial (same as baseline).
    max_stop_fraction<1.0: at most run this fraction of each trial; stop at min(conv_i, cap).
    Returns (total_iterations, global_best_cost).
    """
    total_it = 0
    global_best = float("inf")
    for t in trials:
        if not t:
            continue
        L = len(t)
        conv_i = convergence_index(t)
        if max_stop_fraction >= 1.0:
            stop_i = L - 1  # run full trial, same as baseline
        else:
            cap = min(int(max_stop_fraction * L), L - 1) if L else 0
            stop_i = min(conv_i, cap)
        total_it += stop_i + 1
        global_best = min(global_best, t[stop_i]["best_so_far"])
    return total_it, global_best


def simulate_early_stop_with_precision(trials: list, precision: float, seed: int = 42) -> tuple:
    """
    Simulate early stop using only log: best_so_far = basin id (same value = same local optima).
    At each step we compare current solution to every previously seen local optima; ground truth
    'same' = (current best_so_far == that previous value). Each comparison: say 'same' with prob
    precision if truly same, else (1-precision). Early stop trial if any comparison says 'same'.
    No cuOpt/embedding; errors are random at each comparison. Returns (total_iterations, global_best_cost).
    """
    rng = random.Random(seed)
    total_it = 0
    global_best = float("inf")
    seen_optica = set()  # distinct best_so_far seen so far (global across trials)
    for t in trials:
        if not t:
            continue
        stopped = False
        for i in range(len(t)):
            v = t[i]["best_so_far"]
            if not seen_optica:
                seen_optica.add(v)
                continue
            any_say_same = False
            for s in seen_optica:
                same_basin = v == s
                say_same = rng.random() < (precision if same_basin else (1.0 - precision))
                if say_same:
                    any_say_same = True
                    break
            if any_say_same:
                total_it += i + 1
                global_best = min(global_best, v)
                stopped = True
                break
            seen_optica.add(v)
        if not stopped:
            total_it += len(t)
            if t:
                global_best = min(global_best, t[-1]["best_so_far"])
            for p in t:
                seen_optica.add(p["best_so_far"])
    return total_it, global_best


def run_oracle_experiment(
    trials: list,
    accuracies: list = None,
    max_stop_fractions: list = None,
    n_error_runs: int = 5,
) -> dict:
    """
    Run baseline, perfect oracle (with sunk-cost variants), and error-injection sweeps.
    Returns dict with metrics and series for plotting.
    """
    if accuracies is None:
        accuracies = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
    if max_stop_fractions is None:
        max_stop_fractions = [1.0, 0.5, 0.3, 0.2, 0.1]

    base_it, base_best = simulate_baseline(trials)
    L = base_it  # baseline = total iterations
    flattened = flatten_trials(trials)
    oracle_conv_it, oracle_conv_best = simulate_oracle_at_convergence(trials)
    results = {
        "n_trials": len(trials),
        "baseline_iterations": L,
        "baseline_best_cost": base_best,
        "oracle_at_convergence_iterations": oracle_conv_it,
        "oracle_at_convergence_best_cost": oracle_conv_best,
        "perfect_oracle": {},
        "accuracy_sweep": [],
        "sunk_cost_sweep": [],
        "check_oracle_sweep": [],
        "check_accuracy_sweep": [],
        "precision_sweep": [],  # inject error: per-comparison precision, 20 runs per precision
    }

    # Check oracle: for threshold frac, window = first frac*L steps; oracle_real = checks with skip
    # baseline cost = global best; oracle cost = best cost in window (best up to threshold)
    for frac in max_stop_fractions:
        n_window = max(0, int(frac * L))
        window = flattened[:n_window] if n_window else []
        oracle_real = count_oracle_real_checks(window)
        oracle_best_cost = min(p["best_so_far"] for p in window) if window else float("inf")
        results["check_oracle_sweep"].append({
            "max_stop_fraction": frac,
            "window_steps": n_window,
            "oracle_real": oracle_real,
            "oracle_best_cost": oracle_best_cost,
        })

    for frac in max_stop_fractions:
        it, best = simulate_oracle_perfect(trials, max_stop_fraction=frac)
        results["perfect_oracle"][frac] = {"iterations": it, "best_cost": best}
        results["sunk_cost_sweep"].append(
            {"max_stop_fraction": frac, "iterations": it, "best_cost": best}
        )

    # Inject error: at each step compare current to all previous local optima (best_so_far = basin id).
    # Each comparison says 'same' with prob precision (if truly same) else (1-precision). Early stop if any says 'same'.
    # Run n_error_runs (e.g. 20) per precision because errors are random at different positions.
    precisions = accuracies  # reuse same grid, e.g. [1.0, 0.95, ..., 0.5]
    for prec in precisions:
        it_list, best_list = [], []
        for run in range(n_error_runs):
            it, best = simulate_early_stop_with_precision(trials, precision=prec, seed=42 + run)
            it_list.append(it)
            best_list.append(best)
        import numpy as np
        results["precision_sweep"].append({
            "precision": prec,
            "mean_iterations": float(np.mean(it_list)),
            "std_iterations": float(np.std(it_list)),
            "mean_best_cost": float(np.mean(best_list)),
            "std_best_cost": float(np.std(best_list)),
        })

    # Check oracle with error: at frac=0.5, for each accuracy simulate; report actual_checks and P(all required done)
    n_window_50 = max(0, int(0.5 * L))
    oracle_real_50 = count_oracle_real_checks(flattened[:n_window_50]) if n_window_50 else 0
    for acc in accuracies:
        actual_list, all_done_list = [], []
        for run in range(n_error_runs):
            actual, all_done = simulate_check_oracle_with_errors(flattened, n_window_50, acc, seed=42 + run)
            actual_list.append(actual)
            all_done_list.append(1.0 if all_done else 0.0)
        import numpy as np
        results["check_accuracy_sweep"].append({
            "accuracy": acc,
            "mean_actual_checks": float(np.mean(actual_list)),
            "std_actual_checks": float(np.std(actual_list)),
            "P_all_required_done": float(np.mean(all_done_list)),
            "oracle_real_at_50": oracle_real_50,
            "window_steps": n_window_50,
        })

    return results


def load_log(path: str) -> list:
    """Load points from a log file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return parse_points(text)


def aggregate_trials_from_named_logs(log_paths: list) -> list:
    """Load one or more logs; concatenate all points then split into trials."""
    all_points = []
    for path in log_paths:
        pts = load_log(path)
        all_points.extend(pts)
    return points_to_trials(all_points)


def run_cuopt_and_get_trials(
    problem_path=None,
    solution_path=None,
    data_path=None,
    time_limit=10,
    n_instances=1,
    start_index=0,
    n_runs=1,
    scale=1e2,
    n_vehicles=21,
    device="cuda",
) -> tuple:
    """
    Run cuOpt via run_cuopt.run_experiment with collect_data=True (no early stop).
    Returns (trials, all_run_points, hgs_costs). all_run_points and hgs_costs can be used to write a log.
    """
    import sys
    run_cuopt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_cuopt.py")
    if os.path.dirname(run_cuopt_path) not in sys.path:
        sys.path.insert(0, os.path.dirname(run_cuopt_path))
    from run_cuopt import run_experiment

    _, _, all_run_points, hgs_costs, *_ = run_experiment(
        problem_path=problem_path,
        solution_path=solution_path,
        data_path=data_path,
        time_limit=time_limit,
        n_instances=n_instances,
        start_index=start_index,
        n_runs=n_runs,
        scale=scale,
        n_vehicles=n_vehicles,
        collect_data=True,
        use_callback=False,
        device=device,
    )
    # Merge all runs into one point list; insert break between runs (each run is one trial).
    all_points = []
    for _label, pts in all_run_points:
        all_points.extend(pts)
        all_points.append({"after": float("nan"), "best_so_far": float("nan")})
    trials = points_to_trials(all_points)
    return trials, all_run_points, hgs_costs


def write_capture_log(all_run_points, hgs_costs, filepath):
    """Write log in same shape as solver stdout: [search #N] then cost before/after lines."""
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(filepath, "w") as f:
        for label, points in all_run_points:
            try:
                inst_idx = int(label.split("_")[0].replace("inst", ""))
            except (ValueError, AttributeError):
                m = re.search(r"inst(\d+)", label) if label else None
                inst_idx = int(m.group(1)) if m else 0
            hgs = hgs_costs.get(inst_idx, 0)
            f.write(f"# === {label} (HGS: {hgs}) ===\n")
            search_id = 1
            prev_was_break = True
            for p in points:
                if math.isnan(p.get("after", 0)):
                    search_id += 1
                    prev_was_break = True
                    continue
                if prev_was_break:
                    f.write(f"[search #{search_id}]\n")
                    prev_was_break = False
                before = p.get("before", 0)
                f.write(f"cost before: {before}, cost after: {p['after']}\n")
    print(f"Saved capture log to: {filepath}")


def plot_results(results: dict, out_dir: str, baseline_best: float, baseline_it: int):
    """Generate plots for oracle experiment."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    sweep = results["sunk_cost_sweep"]
    acc_sweep = results["accuracy_sweep"]
    check_sweep = results.get("check_oracle_sweep", [])
    check_acc_sweep = results.get("check_accuracy_sweep", [])

    prec_sweep = results.get("precision_sweep", [])
    has_any = sweep or acc_sweep or check_sweep or check_acc_sweep or prec_sweep
    if has_any:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1) Fraction vs Steps (window_steps, oracle_real)
        ax = axes[0, 0]
        if check_sweep:
            fracs = [x["max_stop_fraction"] for x in check_sweep]
            window_steps = [x["window_steps"] for x in check_sweep]
            oracle_real = [x["oracle_real"] for x in check_sweep]
            ax.plot(fracs, window_steps, "o-", color="C0", label="window_steps")
            ax.plot(fracs, oracle_real, "s-", color="C2", label="oracle_real")
            for xv, yv in zip(fracs, window_steps):
                ax.annotate(f"{yv:.0f}", (xv, yv),
                            xytext=(5, 5), textcoords="offset points", fontsize=8, color="C0")
            for xv, yv in zip(fracs, oracle_real):
                ax.annotate(f"{yv:.0f}", (xv, yv),
                            xytext=(5, -10), textcoords="offset points", fontsize=8, color="C2")
        ax.set_xlabel("Threshold fraction")
        ax.set_ylabel("Steps / Checks")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True)

        # 2) Fraction vs Cost (oracle_best_cost)
        ax = axes[0, 1]
        if check_sweep:
            fracs = [x["max_stop_fraction"] for x in check_sweep]
            oracle_best_cost = [x.get("oracle_best_cost", baseline_best) for x in check_sweep]
            ax.plot(fracs, oracle_best_cost, "^-", color="C1", label="oracle cost")
            ax.axhline(baseline_best, color="C1", linestyle="--", alpha=0.6, label="baseline")
            for xv, yv in zip(fracs, oracle_best_cost):
                ax.annotate(f"{yv:.2f}", (xv, yv),
                            xytext=(5, 5), textcoords="offset points", fontsize=8, color="C1")
        ax.set_xlabel("Threshold fraction")
        ax.set_ylabel("Best cost")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True)

        # 3) Precision vs Steps (iterations)
        ax = axes[1, 0]
        if prec_sweep:
            precs = [x["precision"] for x in prec_sweep]
            mean_it = [x["mean_iterations"] for x in prec_sweep]
            std_it = [x["std_iterations"] for x in prec_sweep]
            ax.errorbar(precs, mean_it, yerr=std_it, fmt="o-", color="C0", capsize=3, label="iterations")
            ax.axhline(baseline_it, color="C0", linestyle="--", alpha=0.6, label="baseline")
            for xv, yv in zip(precs, mean_it):
                ax.annotate(f"{yv:.0f}", (xv, yv),
                            xytext=(5, 5), textcoords="offset points", fontsize=8, color="C0")
        ax.set_xlabel("Precision (per-comparison)")
        ax.set_ylabel("Iterations")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True)

        # 4) Precision vs Cost (best cost)
        ax = axes[1, 1]
        if prec_sweep:
            precs = [x["precision"] for x in prec_sweep]
            mean_best = [x["mean_best_cost"] for x in prec_sweep]
            std_best = [x["std_best_cost"] for x in prec_sweep]
            ax.errorbar(precs, mean_best, yerr=std_best, fmt="s-", color="C1", capsize=3, label="best cost")
            ax.axhline(baseline_best, color="C1", linestyle="--", alpha=0.6, label="baseline")
            for xv, yv in zip(precs, mean_best):
                ax.annotate(f"{yv:.2f}", (xv, yv),
                            xytext=(5, 5), textcoords="offset points", fontsize=8, color="C1")
        ax.set_xlabel("Precision (per-comparison)")
        ax.set_ylabel("Best cost")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True)

        plt.tight_layout()
        p_all = os.path.join(out_dir, "oracle_all.png")
        plt.savefig(p_all, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p_all}")


def main():
    ap = argparse.ArgumentParser(description="Early-stop oracle experiment")
    ap.add_argument("--log", nargs="+", default=None, help="Log file(s) with cost after + # --- break --- (required unless --run_cuopt)")
    ap.add_argument("--run_cuopt", action="store_true", help="Run cuOpt first, capture trajectory, then run oracle (no --log needed)")
    ap.add_argument("--save_capture", type=str, default=None, metavar="FILE", help="When --run_cuopt: save captured run to this log path (e.g. out/capture.log)")
    ap.add_argument("--out_dir", default="./oracle_out", help="Output directory for plots and summary")
    ap.add_argument("--accuracies", type=str, default="1.0,0.95,0.9,0.8,0.7,0.6,0.5",
                    help="Comma-separated precisions/accuracies for sweep")
    ap.add_argument("--max_stop_fractions", type=str, default="1.0,0.5,0.3,0.2,0.1",
                    help="Comma-separated max stop fractions (sunk cost)")
    ap.add_argument("--n_error_runs", type=int, default=20, help="Runs per precision for inject-error")
    ap.add_argument("--no_plot", action="store_true", help="Skip generating plots")
    # run_cuopt options (used when --run_cuopt)
    ap.add_argument("--problem_path", type=str, default=None, help="Pkl problem path (with --solution_path)")
    ap.add_argument("--solution_path", type=str, default=None, help="Pkl solution path (with --problem_path)")
    ap.add_argument("--data_path", type=str, default=None, help="Txt data path (alternative to pkl)")
    ap.add_argument("--time_limit", type=float, default=10, help="Solver time limit per run (s)")
    ap.add_argument("--n_instances", type=int, default=1, help="Number of instances")
    ap.add_argument("--start_index", type=int, default=0, help="Start instance index")
    ap.add_argument("--n_runs", type=int, default=1, help="Runs per instance")
    ap.add_argument("--scale", type=float, default=1e2, help="Cost scale")
    ap.add_argument("--n_vehicles", type=int, default=21, help="Vehicle count")
    ap.add_argument("--device", type=str, default="cuda", help="Device for run_cuopt")
    args = ap.parse_args()

    if args.run_cuopt:
        if not (args.problem_path and args.solution_path) and not args.data_path:
            ap.error("--run_cuopt requires --problem_path and --solution_path (pkl) or --data_path (txt)")
        print("Running cuOpt to capture trajectory (no early stop)...")
        trials, all_run_points, hgs_costs = run_cuopt_and_get_trials(
            problem_path=args.problem_path,
            solution_path=args.solution_path,
            data_path=args.data_path,
            time_limit=args.time_limit,
            n_instances=args.n_instances,
            start_index=args.start_index,
            n_runs=args.n_runs,
            scale=args.scale,
            n_vehicles=args.n_vehicles,
            device=args.device,
        )
        if args.save_capture:
            out_log = args.save_capture
            if not os.path.isabs(out_log):
                out_log = os.path.join(args.out_dir, out_log)
            write_capture_log(all_run_points, hgs_costs, out_log)
        print(f"Captured {len(trials)} trials, total points: {sum(len(t) for t in trials)}")
    else:
        if not args.log:
            ap.error("Provide --log or use --run_cuopt")
        trials = aggregate_trials_from_named_logs(args.log)

    if not trials:
        print("No trials found. With --log: ensure file has 'cost before/after' and '# --- break ---'. With --run_cuopt: check solver output.")
        sys.exit(1)

    accuracies = [float(x.strip()) for x in args.accuracies.split(",")]
    max_stop_fractions = [float(x.strip()) for x in args.max_stop_fractions.split(",")]

    results = run_oracle_experiment(
        trials,
        accuracies=accuracies,
        max_stop_fractions=max_stop_fractions,
        n_error_runs=args.n_error_runs,
    )

    base_it = results["baseline_iterations"]
    base_best = results["baseline_best_cost"]
    perf_it = results["oracle_at_convergence_iterations"]
    perf_best = results["oracle_at_convergence_best_cost"]

    print(f"\nTrials: {results['n_trials']}")
    print(f"Baseline: iterations={base_it}, best_cost={base_best:.4f}")
    print(f"Oracle (stop at convergence): iterations={perf_it}, best_cost={perf_best:.4f}")
    if base_it > 0:
        speedup = (base_it - perf_it) / base_it * 100
        print(f"Speedup (iteration reduction): {speedup:.1f}%")

    print("\n--- Check oracle (baseline L = total it; baseline cost = best cost; oracle cost = best in window) ---")
    print(f"  Baseline L = {base_it}, baseline cost = {base_best:.4f}")
    for s in results["check_oracle_sweep"]:
        f, w, r = s["max_stop_fraction"], s["window_steps"], s["oracle_real"]
        ob = s.get("oracle_best_cost", base_best)
        print(f"  frac={f:.2f}: window_steps={w}, oracle_real={r}, oracle_best_cost={ob:.4f}")

    if results.get("check_accuracy_sweep"):
        print("\n--- Check oracle error injection (frac=0.5: required steps check with prob=accuracy; P(all required done)=accuracy^num_required) ---")
        for s in results["check_accuracy_sweep"]:
            acc = s["accuracy"]
            ma, sa = s["mean_actual_checks"], s["std_actual_checks"]
            p_done = s["P_all_required_done"]
            o50 = s.get("oracle_real_at_50", 0)
            print(f"  accuracy={acc:.2f}: actual_checks={ma:.1f}±{sa:.1f}, P(all required done)={p_done:.3f}, oracle_real@50%={o50}")

    print("\n--- Sunk cost (max_stop_frac=1.0 = no early stop = baseline) ---")
    for s in results["sunk_cost_sweep"]:
        f, it, best = s["max_stop_fraction"], s["iterations"], s["best_cost"]
        print(f"  max_stop_frac={f:.2f}: it={it}, best_cost={best:.4f}")

    if results.get("precision_sweep"):
        print("\n--- Inject error (precision per comparison; current vs each previous local optima; 20 runs) ---")
        for s in results["precision_sweep"]:
            p, mi, si, mb, sb = s["precision"], s["mean_iterations"], s["std_iterations"], s["mean_best_cost"], s["std_best_cost"]
            print(f"  precision={p:.2f}: it={mi:.0f}±{si:.0f}, best_cost={mb:.4f}±{sb:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    if not args.no_plot:
        plot_results(results, args.out_dir, base_best, base_it)

    # Save summary JSON
    try:
        import json
        summary = {
            "n_trials": results["n_trials"],
            "baseline_iterations": base_it,
            "baseline_best_cost": base_best,
            "oracle_at_convergence_iterations": perf_it,
            "oracle_at_convergence_best_cost": perf_best,
            "check_oracle_sweep": results.get("check_oracle_sweep", []),
            "check_accuracy_sweep": results.get("check_accuracy_sweep", []),
            "precision_sweep": results.get("precision_sweep", []),
            "sunk_cost_sweep": results["sunk_cost_sweep"],
        }
        p = os.path.join(args.out_dir, "oracle_summary.json")
        with open(p, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {p}")
    except Exception as e:
        print(f"Could not save JSON: {e}")


if __name__ == "__main__":
    main()
