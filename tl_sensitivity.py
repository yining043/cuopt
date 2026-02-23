import argparse
import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


@dataclass
class StepRecord:
    """One row in the baseline log."""

    timestamp: float
    cost: float
    true_basin_id: Any
    is_new_basin: bool
    duration: float
    trial_id: int = 0


def generate_dummy_baseline_log(
    csv_path: str,
    total_time: float = 30.0,
    n_trials: int = 200,
    n_unique_basins: int = 50,
    seed: int = 42,
) -> None:
    """Generate a simulated baseline (no cuOpt run)."""
    rng = np.random.default_rng(seed)
    seen_basins: set[int] = set()
    all_basin_ids = list(range(n_unique_basins))
    wall_time = 0.0
    current_best_cost = 2000.0
    durations = rng.uniform(0.08, 0.22, n_trials)
    durations = durations * (total_time / durations.sum())

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "cost", "true_basin_id", "is_new_basin", "trial_id"])
        for i in range(n_trials):
            prob_new = max(0.05, 1.0 - i / n_trials)
            is_new = rng.random() < prob_new and len(seen_basins) < n_unique_basins
            if is_new:
                remaining = [b for b in all_basin_ids if b not in seen_basins]
                basin_id = int(rng.choice(remaining))
                seen_basins.add(basin_id)
                current_best_cost -= float(rng.uniform(1.0, 10.0))
            else:
                basin_id = int(rng.choice(list(seen_basins))) if seen_basins else 0
                current_best_cost += float(rng.normal(0.0, 0.5))
            current_best_cost = max(500.0, current_best_cost)
            wall_time += float(durations[i])
            w.writerow([f"{wall_time:.4f}", f"{current_best_cost:.4f}", basin_id, int(is_new), i + 1])
    print(f"Generated baseline log: {csv_path} ({n_trials} trials over {wall_time:.2f}s)")


def build_baseline_from_cuopt_log(
    cuopt_log_path: str,
    output_csv_path: str,
    total_time: float = 30.0,
) -> None:
    """Build baseline CSV (pure ground truth) from a cuOpt log."""
    import re

    cost_re = re.compile(
        r"cost before:\s*[^,]+\s*,\s*cost after:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        re.IGNORECASE,
    )
    search_re = re.compile(r"\[search\s*#\s*\d+\]")
    with open(cuopt_log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    points: List[Tuple[float, float]] = []
    best_so_far = math.inf
    for line in text.splitlines():
        line = line.strip()
        if line == "# --- break ---" or search_re.search(line):
            points.append((math.nan, math.nan))
            best_so_far = math.inf
            continue
        m = cost_re.search(line)
        if m:
            after = float(m.group(1))
            best_so_far = min(best_so_far, after)
            points.append((after, best_so_far))

    trials: List[List[Tuple[float, float]]] = []
    cur: List[Tuple[float, float]] = []
    for p in points:
        if math.isnan(p[0]):
            if cur:
                trials.append(cur)
                cur = []
        else:
            cur.append(p)
    if cur:
        trials.append(cur)
    if not trials:
        raise ValueError(f"No trials found in {cuopt_log_path}")

    cost_to_idx: Dict[float, int] = {}
    n = len(trials)
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "cost", "true_basin_id", "is_new_basin", "trial_id"])
        for i, t in enumerate(trials):
            cost = t[-1][1]
            is_new = cost not in cost_to_idx
            if is_new:
                cost_to_idx[cost] = len(cost_to_idx)
            basin_idx = cost_to_idx[cost]
            ts = total_time * (i + 1) / n
            w.writerow([f"{ts:.4f}", f"{cost:.4f}", basin_idx, int(is_new), i + 1])
    print(f"Built baseline from cuOpt log: {output_csv_path} ({n} trials over {total_time}s)")


def _points_to_trials(all_points: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Split point list into trials (NaN = break)."""
    trials, cur = [], []
    for p in all_points:
        if math.isnan(p.get("after", 0)):
            if cur:
                trials.append(cur)
                cur = []
        else:
            cur.append(p)
    if cur:
        trials.append(cur)
    return trials


def run_cuopt_30s_and_build_baseline(
    output_csv_path: str,
    problem_path: Optional[str] = None,
    solution_path: Optional[str] = None,
    data_path: Optional[str] = None,
    start_index: int = 0,
    time_limit: float = 30.0,
    scale: float = 1e2,
    n_vehicles: int = 30,
    device: str = "cuda",
    plot_dir: Optional[str] = None,
    cost_ymax: Optional[float] = None,
) -> None:
    """Run cuOpt solver for time_limit seconds, then build baseline CSV."""
    if not (problem_path and solution_path) and not data_path:
        raise ValueError("Provide problem_path+solution_path (pkl) or data_path (txt).")
    import sys

    this_dir = os.path.dirname(os.path.abspath(__file__))
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)
    from run_cuopt import run_experiment, plot_from_points, plot_cost_curve_by_trial_duplicates, _TeeStdout

    log_dir = os.path.dirname(output_csv_path) or "."
    os.makedirs(log_dir, exist_ok=True)
    baseline_run_log = os.path.join(log_dir, "baseline_run.log")
    print(f"[tl_sensitivity] Calling cuOpt solver for {time_limit}s ...")
    with _TeeStdout(baseline_run_log):
        _, _, all_run_points, hgs_costs, *_ = run_experiment(
            problem_path=problem_path,
            solution_path=solution_path,
            data_path=data_path,
            time_limit=time_limit,
            n_instances=1,
            start_index=start_index,
            n_runs=1,
            scale=scale,
            n_vehicles=n_vehicles,
            collect_data=True,
            use_callback=False,
            device=device,
        )

    build_baseline_from_cuopt_log(baseline_run_log, output_csv_path, total_time=time_limit)

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        plot_from_points(
            all_run_points, break_mode="none", segments=0, dpi=150,
            ymax=cost_ymax, out_dir=plot_dir, hgs_costs=hgs_costs,
        )
        hgs_val = (hgs_costs or {}).get(0) or (hgs_costs or {}).get(start_index)
        plot_cost_curve_by_trial_duplicates(
            all_run_points,
            out_path=os.path.join(plot_dir, "cost_curve_by_trial_duplicates.png"),
            dpi=150, hgs_cost=hgs_val, cost_ymax=cost_ymax,
        )
        print(f"Saved plots to: {plot_dir}")


def load_baseline_log(csv_path: str) -> List[StepRecord]:
    """Load baseline CSV (pure ground truth, no filtering)."""
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty log: {csv_path}")

    rows.sort(key=lambda r: float(r["timestamp"]))
    records: List[StepRecord] = []
    prev_t = 0.0
    for idx, row in enumerate(rows):
        t = float(row["timestamp"])
        cost = float(row["cost"])
        trial_id = int(row["trial_id"]) if row.get("trial_id") else idx + 1
        records.append(StepRecord(
            timestamp=t, cost=cost, true_basin_id=row["true_basin_id"],
            is_new_basin=(row.get("is_new_basin", "0") == "1"),
            duration=max(0.0, t - prev_t), trial_id=trial_id,
        ))
        prev_t = t
    return records


def compute_baseline_at_TL(records: List[StepRecord], TL: float) -> Tuple[int, float]:
    """Return (n_trials_within_TL, best_cost_at_TL) for baseline."""
    best_cost = math.inf
    n_steps = 0
    for r in records:
        if r.timestamp > TL:
            break
        n_steps += 1
        best_cost = min(best_cost, r.cost)
    if n_steps == 0:
        # No step finished before TL: treat best_cost as first cost.
        best_cost = records[0].cost
    return n_steps, best_cost


def estimate_fpr_for_PR(
    records: List[StepRecord], precision: float, recall: float,
    min_cost: Optional[float] = None,
) -> float:
    """Estimate FPR on new basins given target (precision, recall) on redundant-class.

    Trials with cost > min_cost are excluded from statistics (they're not real basins).
    TP = recall * pos, FP = FPR * neg
    precision = TP / (TP + FP) => FPR = recall * pos * (1 - precision) / (precision * neg)
    """
    recs = [r for r in records if min_cost is None or r.cost <= min_cost]
    pos = sum(1 for r in recs if not r.is_new_basin)
    neg = sum(1 for r in recs if r.is_new_basin)
    if pos == 0 or neg == 0:
        return 0.0
    fpr = recall * pos * (1.0 - precision) / (precision * neg)
    return min(max(fpr, 0.0), 1.0)


def simulate_TL_scenario(
    records: List[StepRecord],
    TL: float,
    precision: float,
    recall: float,
    fpr: float,
    reinvest: bool,
    overhead_skip_ratio: float = 0.2,
    T_infer: float = 1.0,
    min_cost: Optional[float] = None,
    rng: np.random.Generator | None = None,
) -> Tuple[int, float, int, float, float]:
    """Simulate online early stop under a time limit.

    Cutoff: cumulative search time + serial overhead >= TL.
    Parallel overhead is tracked but does not affect cutoff.
    Trials with cost > min_cost consume time but never enter archive or update best_cost.

    Returns (n_effective_steps, best_cost, n_trials_processed,
             total_serial_overhead, total_parallel_overhead).
    """
    if rng is None:
        rng = np.random.default_rng(123)

    max_idx_TL = 0
    for i, r in enumerate(records):
        if r.timestamp <= TL:
            max_idx_TL = i + 1
        else:
            break
    max_idx = len(records) if reinvest else max_idx_TL

    total_time = 0.0
    best_cost = math.inf
    steps_done = 0
    trials_processed = 0
    serial_overhead = 0.0
    parallel_overhead = 0.0
    archive: set = set()

    i = 0
    while i < max_idx and total_time < TL:
        r = records[i]

        above_min = min_cost is not None and r.cost > min_cost

        is_in_archive = r.true_basin_id in archive
        predict_skip = rng.random() < (recall if is_in_archive else fpr)

        serial_this = len(archive) * T_infer
        parallel_this = T_infer
        serial_overhead += serial_this
        parallel_overhead += parallel_this
        total_time += serial_this
        if total_time >= TL:
            break

        effective_dt = r.duration * (overhead_skip_ratio if predict_skip else 1.0)
        total_time += effective_dt
        trials_processed += 1
        if total_time > TL:
            break

        steps_done += 1
        if not predict_skip and not is_in_archive and not above_min:
            best_cost = min(best_cost, r.cost)
            archive.add(r.true_basin_id)
        i += 1

    if math.isinf(best_cost):
        best_cost = records[-1].cost
    return steps_done, best_cost, trials_processed, serial_overhead, parallel_overhead


def simulate_speedup_curve(
    records: List[StepRecord],
    precision: float,
    recall: float,
    fpr: float,
    overhead_skip_ratio: float = 0.2,
    T_infer: float = 1.0,
    min_cost: Optional[float] = None,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay full log, return per-trial cumulative times: (baseline, serial, parallel)."""
    if rng is None:
        rng = np.random.default_rng(123)

    n = len(records)
    baseline_arr = np.zeros(n)
    serial_arr = np.zeros(n)
    parallel_arr = np.zeros(n)
    baseline_cum = 0.0
    serial_cum = 0.0
    parallel_cum = 0.0
    archive: set = set()

    for j, r in enumerate(records):
        above_min = min_cost is not None and r.cost > min_cost
        is_in_archive = r.true_basin_id in archive
        predict_skip = rng.random() < (recall if is_in_archive else fpr)

        baseline_cum += r.duration
        search_dt = r.duration * (overhead_skip_ratio if predict_skip else 1.0)
        serial_cum += search_dt + len(archive) * T_infer
        parallel_cum += search_dt + T_infer

        if not predict_skip and not is_in_archive and not above_min:
            archive.add(r.true_basin_id)

        baseline_arr[j] = baseline_cum
        serial_arr[j] = serial_cum
        parallel_arr[j] = parallel_cum

    return baseline_arr, serial_arr, parallel_arr


def run_grid_for_TL(
    records: List[StepRecord],
    TL: float,
    precisions: List[float],
    recalls: List[float],
    n_repeat: int = 20,
    min_cost: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Run Exp 1/2/3 on a P/R grid for a single TL; each cell averaged over n_repeat runs.

    Exp 1 (vacuum): T_infer=0, reinvest=False  -> step_reduction, cost_degradation
    Exp 2 (overhead): T_infer=1, reinvest=False -> overhead_serial, overhead_parallel
    Exp 3 (reinvest): T_infer=0, reinvest=True  -> net_improvement
    """
    baseline_steps, baseline_cost = compute_baseline_at_TL(records, TL)

    P, R = len(precisions), len(recalls)
    step_reduction = np.full((R, P), np.nan)
    cost_degradation = np.full((R, P), np.nan)
    net_improvement = np.full((R, P), np.nan)
    overhead_serial = np.full((R, P), np.nan)
    overhead_parallel = np.full((R, P), np.nan)

    for i_r, recall in enumerate(recalls):
        for i_p, precision in enumerate(precisions):
            fpr = estimate_fpr_for_PR(records, precision, recall, min_cost=min_cost)

            sr, cd, ni, os_l, op_l = [], [], [], [], []
            for run_idx in range(n_repeat):
                seed = 42 + run_idx + (i_r * P + i_p) * 10000 + int(round(TL * 100))

                # Exp 1: vacuum (T_infer=0), no reinvest
                steps_v, cost_v, *_ = simulate_TL_scenario(
                    records, TL=TL, precision=precision, recall=recall,
                    fpr=fpr, reinvest=False, T_infer=0, min_cost=min_cost,
                    rng=np.random.default_rng(seed),
                )
                # Exp 2: overhead (T_infer=1), no reinvest
                _, _, _, serial_over, parallel_over = simulate_TL_scenario(
                    records, TL=TL, precision=precision, recall=recall,
                    fpr=fpr, reinvest=False, T_infer=1.0, min_cost=min_cost,
                    rng=np.random.default_rng(seed),
                )
                # Exp 3: vacuum (T_infer=0), reinvest
                _, cost_re, *_ = simulate_TL_scenario(
                    records, TL=TL, precision=precision, recall=recall,
                    fpr=fpr, reinvest=True, T_infer=0, min_cost=min_cost,
                    rng=np.random.default_rng(seed),
                )

                if baseline_steps > 0:
                    sr.append(1.0 - steps_v / baseline_steps)
                if baseline_cost > 0:
                    cd.append((cost_v - baseline_cost) / baseline_cost)
                    ni.append((baseline_cost - cost_re) / baseline_cost)
                os_l.append(serial_over / max(TL, 1e-9))
                op_l.append(parallel_over / max(TL, 1e-9))

            step_reduction[i_r, i_p] = float(np.mean(sr)) if sr else np.nan
            cost_degradation[i_r, i_p] = float(np.mean(cd)) if cd else np.nan
            net_improvement[i_r, i_p] = float(np.mean(ni)) if ni else np.nan
            overhead_serial[i_r, i_p] = float(np.mean(os_l))
            overhead_parallel[i_r, i_p] = float(np.mean(op_l))

    return {
        "step_reduction": step_reduction,
        "cost_degradation": cost_degradation,
        "net_improvement": net_improvement,
        "overhead_serial": overhead_serial,
        "overhead_parallel": overhead_parallel,
        "baseline_steps": float(baseline_steps),
        "baseline_cost": float(baseline_cost),
    }


def plot_heatmap(
    data: np.ndarray,
    precisions: List[float],
    recalls: List[float],
    title: str,
    cbar_label: str,
    center: float | None,
    out_path: str,
) -> None:
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        data,
        xticklabels=[f"{p:.2f}" for p in precisions],
        yticklabels=[f"{r:.2f}" for r in recalls],
        cmap="RdBu_r", center=center, annot=True, fmt=".2f",
    )
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title(title)
    ax.collections[0].colorbar.set_label(cbar_label)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_speedup_curve(
    records: List[StepRecord],
    precisions: List[float],
    recalls: List[float],
    TL: float,
    T_infer: float = 1.0,
    min_cost: Optional[float] = None,
    out_path: str = "speedup_curve.png",
) -> None:
    """Exp 2.1: Speedup vs Trial Index — serial O(N) collapses, parallel O(1) holds."""
    all_serial = []
    all_parallel = []

    for recall in recalls:
        for precision in precisions:
            fpr = estimate_fpr_for_PR(records, precision, recall, min_cost=min_cost)
            rng = np.random.default_rng(42)
            base, serial, parallel = simulate_speedup_curve(
                records, precision, recall, fpr,
                T_infer=T_infer, min_cost=min_cost, rng=rng,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                all_serial.append(np.where(serial > 0, base / serial, 1.0))
                all_parallel.append(np.where(parallel > 0, base / parallel, 1.0))

    mean_serial = np.mean(all_serial, axis=0)
    mean_parallel = np.mean(all_parallel, axis=0)
    trial_idx = np.arange(1, len(records) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(trial_idx, mean_serial, label="Serial O(N)", alpha=0.8)
    plt.plot(trial_idx, mean_parallel, label="Parallel O(1)", alpha=0.8)
    plt.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="Baseline (no early-stop)")

    n_within_TL = sum(1 for r in records if r.timestamp <= TL)
    if 0 < n_within_TL < len(records):
        plt.axvline(x=n_within_TL, color="red", linestyle=":", linewidth=1,
                     label=f"TL={TL:.0f}s ({n_within_TL} trials)")

    plt.xlabel("Trial Index")
    plt.ylabel("Speedup (baseline time / oracle time)")
    plt.title(f"Exp 2.1: Speedup vs Trial Index (T_infer={T_infer})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_overhead_heatmaps(
    precisions: List[float],
    recalls: List[float],
    overhead_serial: np.ndarray,
    overhead_parallel: np.ndarray,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, title in [
        (axes[0], overhead_serial, "Serial overhead / TL"),
        (axes[1], overhead_parallel, "Parallel overhead / TL"),
    ]:
        sns.heatmap(
            data, ax=ax,
            xticklabels=[f"{p:.2f}" for p in precisions],
            yticklabels=[f"{r:.2f}" for r in recalls],
            cmap="RdBu_r", center=0.0, annot=True, fmt=".2f",
        )
        ax.set_title(title)
        ax.set_xlabel("Precision")
        ax.set_ylabel("Recall")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Oracle TL Sensitivity")
    ap.add_argument("--baseline_log", type=str, default=None)
    ap.add_argument("--from_cuopt_log", type=str, default=None)
    ap.add_argument("--cuopt_log_total_time", type=float, default=30.0)
    ap.add_argument("--problem_path", type=str, default="/home/jieyi/cvrp100_uniform.pkl")
    ap.add_argument("--solution_path", type=str, default="/home/jieyi/hgs_cvrp100_uniform.pkl")
    ap.add_argument("--instance_index", type=int, default=0)
    ap.add_argument("--data_path", type=str, default=None)
    ap.add_argument("--run_cuopt_time_limit", type=float, default=30.0)
    ap.add_argument("--scale", type=float, default=1e2)
    ap.add_argument("--n_vehicles", type=int, default=30)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--time_limits", type=str, default="1,2,5,10,15")
    ap.add_argument("--precisions", type=str, default="0.8,0.85,0.9,0.95,1.0")
    ap.add_argument("--recalls", type=str, default="0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--out_root", type=str, default="oracle_out")
    ap.add_argument("--cost_ymax", type=float, default=20.0)
    ap.add_argument("--min_cost", type=float, default=None,
                    help="Trials with cost > min_cost are not counted as basins.")
    ap.add_argument("--n_repeat", type=int, default=20,
                    help="Repeated runs per (TL, P, R) cell; results are averaged.")
    args = ap.parse_args()

    baseline_log = args.baseline_log or os.path.join(args.out_root, "RUN_30S", "baseline_log.csv")
    min_cost = args.min_cost

    if args.from_cuopt_log:
        build_baseline_from_cuopt_log(
            args.from_cuopt_log, baseline_log, total_time=args.cuopt_log_total_time,
        )
    elif not os.path.exists(baseline_log):
        if (args.problem_path and args.solution_path) or args.data_path:
            run_cuopt_30s_and_build_baseline(
                output_csv_path=baseline_log,
                problem_path=args.problem_path,
                solution_path=args.solution_path,
                data_path=args.data_path,
                start_index=args.instance_index,
                time_limit=args.run_cuopt_time_limit,
                scale=args.scale,
                n_vehicles=args.n_vehicles,
                device=args.device,
                plot_dir=os.path.dirname(baseline_log) or ".",
                cost_ymax=args.cost_ymax if args.cost_ymax > 0 else None,
            )
        else:
            print("[tl_sensitivity] No baseline_log and no data path: generating dummy log.")
            generate_dummy_baseline_log(baseline_log)

    records = load_baseline_log(baseline_log)
    TLs = parse_float_list(args.time_limits)
    precisions = parse_float_list(args.precisions)
    recalls = parse_float_list(args.recalls)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(args.out_root, f"{ts}_30S")

    for TL in TLs:
        tl_dir = os.path.join(root, f"TL_{int(TL)}s")
        os.makedirs(tl_dir, exist_ok=True)

        grid = run_grid_for_TL(records, TL, precisions, recalls,
                              n_repeat=args.n_repeat, min_cost=min_cost)

        plot_heatmap(
            grid["step_reduction"] * 100.0, precisions, recalls,
            f"Exp 1.1: Step reduction % @ TL={TL:.1f}s", "Reduction (%)", 0.0,
            os.path.join(tl_dir, "exp1_1_step_reduction.png"),
        )
        plot_heatmap(
            grid["cost_degradation"] * 100.0, precisions, recalls,
            f"Exp 1.2: Cost degradation % @ TL={TL:.1f}s", "Degradation (%)", 0.0,
            os.path.join(tl_dir, "exp1_2_cost_degradation.png"),
        )
        plot_heatmap(
            grid["net_improvement"] * 100.0, precisions, recalls,
            f"Exp 3: Net cost improvement % @ TL={TL:.1f}s", "Improvement (%)", 0.0,
            os.path.join(tl_dir, "exp3_net_improvement.png"),
        )
        plot_speedup_curve(
            records, precisions, recalls, TL=TL,
            min_cost=min_cost,
            out_path=os.path.join(tl_dir, "exp2_1_speedup_curve.png"),
        )
        plot_overhead_heatmaps(
            precisions, recalls,
            grid["overhead_serial"], grid["overhead_parallel"],
            os.path.join(tl_dir, "exp2_2_overhead_heatmaps.png"),
        )


if __name__ == "__main__":
    main()

