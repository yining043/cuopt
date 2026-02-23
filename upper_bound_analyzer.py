import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TrialRecord:
    trial_id: int
    wall_clock_time: float
    duration: float
    steps: int
    final_cost: float
    basin_id: Any
    is_new_discovery: bool
    is_feasible: bool = True


@dataclass
class ExperimentResults:
    # Plot 1: Saturation
    total_trials: np.ndarray = field(default_factory=lambda: np.array([]))
    unique_basins: np.ndarray = field(default_factory=lambda: np.array([]))
    final_redundancy: float = 0.0

    # Shared
    time_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    baseline_total_time: float = 0.0
    oracle_total_time: float = 0.0

    # Plot 2: Best Cost vs Time
    baseline_cost_series: np.ndarray = field(default_factory=lambda: np.array([]))
    oracle_cost_series: np.ndarray = field(default_factory=lambda: np.array([]))

    # Plot 3: Unique Basins vs Time
    baseline_unique_series: np.ndarray = field(default_factory=lambda: np.array([]))
    oracle_unique_series: np.ndarray = field(default_factory=lambda: np.array([]))

    # Plot 4: Multiplier
    checkpoints: np.ndarray = field(default_factory=lambda: np.array([]))
    time_multipliers: np.ndarray = field(default_factory=lambda: np.array([]))
    trial_multipliers: np.ndarray = field(default_factory=lambda: np.array([]))


def _resample_step(
    times: np.ndarray, values: np.ndarray, grid: np.ndarray, default: float = np.inf,
) -> np.ndarray:
    """Step-wise resampling: last value at or before each grid point."""
    idx = np.searchsorted(times, grid, side="right") - 1
    return np.where(idx >= 0, values[np.clip(idx, 0, len(values) - 1)], default)


class UpperBoundAnalyzer:
    """Upper-bound analysis assuming a perfect oracle that skips all repeated basins.

    Given trial logs with basin identity, quantifies how much time / exploration
    can be saved and how much better cost the oracle reaches in the same wall-clock time.
    """

    def __init__(self) -> None:
        self.trials: List[TrialRecord] = []
        self.results: Optional[ExperimentResults] = None

    def load_logs(self, data: Sequence[Dict[str, Any]]) -> None:
        self.trials = [
            TrialRecord(
                trial_id=d["trial_id"],
                wall_clock_time=d["wall_clock_time"],
                duration=d["duration"],
                steps=d["steps"],
                final_cost=d["final_cost"],
                basin_id=d["basin_id"],
                is_new_discovery=d["is_new_discovery"],
                is_feasible=d.get("is_feasible", True),
            )
            for d in data
        ]
        self.trials.sort(key=lambda r: (r.wall_clock_time, r.trial_id))

    def run_analysis(
        self,
        overhead_ratio: float = 0.0,
        n_checkpoints: int = 50,
        dt: float = 0.1,
    ) -> ExperimentResults:
        """Run all four experiments.

        Args:
            overhead_ratio: Fraction of trial duration still paid on repeated basins
                (0.0 = ideal oracle / theoretical upper bound).
            n_checkpoints: Number of checkpoints for multiplier curve.
            dt: Time resolution for convergence curves.
        """
        assert self.trials, "No trials loaded. Call load_logs() first."
        res = ExperimentResults()
        n = len(self.trials)

        durations = np.array([t.duration for t in self.trials])
        costs = np.array([t.final_cost for t in self.trials])
        is_new = np.array([t.is_new_discovery for t in self.trials])
        feasible = np.array([t.is_feasible for t in self.trials])

        # Cumulative unique basin count — only feasible trials contribute
        seen: set = set()
        unique_count = np.zeros(n, dtype=int)
        for i, t in enumerate(self.trials):
            if t.is_feasible:
                seen.add(t.basin_id)
            unique_count[i] = len(seen)

        # Timelines
        # Oracle skips all redundant trials (feasible or infeasible)
        base_times = np.cumsum(durations)
        can_skip = ~is_new
        oracle_dur = np.where(can_skip, durations * overhead_ratio, durations)
        oracle_times = np.cumsum(oracle_dur)

        # Best cost — only feasible trials update it
        best_cost = np.full(n, np.inf)
        running_best = np.inf
        for i in range(n):
            if feasible[i]:
                running_best = min(running_best, costs[i])
            best_cost[i] = running_best

        res.baseline_total_time = float(base_times[-1])
        res.oracle_total_time = float(oracle_times[-1])

        # ── Plot 1: Saturation ──
        res.total_trials = np.arange(1, n + 1)
        res.unique_basins = unique_count
        res.final_redundancy = 1.0 - unique_count[-1] / n

        # ── Time grid (shared by Plots 2, 3) ──
        max_time = max(res.baseline_total_time, res.oracle_total_time)
        grid = np.arange(0.0, max_time + dt, dt)
        res.time_grid = grid

        # ── Plot 2: Best Cost vs Time ──
        res.baseline_cost_series = _resample_step(base_times, best_cost, grid, default=np.inf)
        res.oracle_cost_series = _resample_step(oracle_times, best_cost, grid, default=np.inf)

        # ── Plot 3: Unique Basins vs Time ──
        unique_f = unique_count.astype(float)
        res.baseline_unique_series = _resample_step(base_times, unique_f, grid, default=0.0)
        res.oracle_unique_series = _resample_step(oracle_times, unique_f, grid, default=0.0)

        # ── Plot 4: Multiplier ──
        checkpoints = np.linspace(
            res.baseline_total_time / n_checkpoints,
            res.baseline_total_time,
            n_checkpoints,
        )
        n_base = np.searchsorted(base_times, checkpoints, side="right")
        n_oracle = np.minimum(
            np.searchsorted(oracle_times, checkpoints, side="right"), n,
        )

        # Trial-count multiplier: at same wall-clock T, oracle processes more trials
        res.trial_multipliers = np.where(n_base > 0, n_oracle / n_base, 1.0)

        # Time-equivalent multiplier (with interpolation):
        #   At wall-clock T the oracle has completed k full trials and is
        #   partially through trial k+1.  Interpolate the fractional progress
        #   so that zero-skip ⇒ multiplier == 1.0 exactly.
        time_mults = np.ones(len(checkpoints))
        for i in range(len(checkpoints)):
            k = int(n_oracle[i])
            T = checkpoints[i]
            if k <= 0 or T <= 0:
                continue
            k_clamped = min(k, n)
            t_completed = oracle_times[k_clamped - 1]
            if k_clamped < n:
                trial_dur = oracle_dur[k_clamped]
                if trial_dur > 0:
                    frac = (T - t_completed) / trial_dur
                else:
                    frac = 1.0
                frac = min(frac, 1.0)
                effective_k = k_clamped + frac
            else:
                effective_k = float(k_clamped)
            ek_floor = int(effective_k)
            ek_frac = effective_k - ek_floor
            if ek_floor >= n:
                base_equiv = base_times[n - 1]
            elif ek_floor <= 0:
                base_equiv = ek_frac * durations[0]
            else:
                base_equiv = base_times[ek_floor - 1] + ek_frac * durations[min(ek_floor, n - 1)]
            time_mults[i] = base_equiv / T
        res.time_multipliers = time_mults
        res.checkpoints = checkpoints

        self.results = res
        return res

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_results(
        self,
        save_dir: str | Path | None = None,
        save_path: str | Path | None = None,
        wall_time_max: float | None = None,
        hgs_cost: float | None = None,
        dpi: int = 150,
    ) -> None:
        """Generate upper-bound plots.

        Args:
            save_dir: Save 4 individual PNGs here (preferred).
            save_path: Save a combined 2x2 PNG (fallback when save_dir is None).
            wall_time_max: X-axis upper limit for time-based plots.
            hgs_cost: HGS reference cost; costs below this are infeasible.
            dpi: Figure resolution.
        """
        assert self.results is not None, "Call run_analysis() first."
        res = self.results

        if save_dir is not None:
            d = Path(save_dir)
            d.mkdir(parents=True, exist_ok=True)
            self._plot_saturation(res, d / "plot1_saturation.png", dpi)
            self._plot_cost_convergence(res, d / "plot2_cost_convergence.png", wall_time_max, hgs_cost, dpi)
            self._plot_unique_convergence(res, d / "plot3_unique_convergence.png", wall_time_max, dpi)
            self._plot_multiplier(res, d / "plot4_multiplier.png", wall_time_max, dpi)
        else:
            self._plot_combined(res, save_path, wall_time_max, hgs_cost, dpi)

    def _plot_saturation(
        self, res: ExperimentResults, path: str | Path, dpi: int,
    ) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(res.total_trials, res.unique_basins, color="tab:blue", linewidth=1.5,
                label="Unique basins")
        ax.plot(res.total_trials, res.total_trials, color="gray", linestyle="--",
                linewidth=1, alpha=0.5, label="y = x (no redundancy)")
        ax.fill_between(
            res.total_trials, res.unique_basins, res.total_trials,
            alpha=0.12, color="tab:red",
        )
        n_total = int(res.total_trials[-1])
        n_unique = int(res.unique_basins[-1])
        ax.text(
            0.05, 0.95,
            f"{n_total} trials, {n_unique} unique\nRedundancy: {res.final_redundancy:.1%}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        ax.set_xlabel("Total Trials")
        ax.set_ylabel("Unique Local Optima Found")
        ax.set_title("Saturation Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    def _plot_cost_convergence(
        self, res: ExperimentResults, path: str | Path,
        wall_time_max: float | None, hgs_cost: float | None, dpi: int,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        t = res.time_grid
        base_c = res.baseline_cost_series
        ora_c = res.oracle_cost_series
        if wall_time_max is not None and len(t) > 0 and wall_time_max > t[-1]:
            t = np.append(t, wall_time_max)
            base_c = np.append(base_c, base_c[-1])
            ora_c = np.append(ora_c, ora_c[-1])
        m1 = np.isfinite(base_c)
        ax.plot(t[m1], base_c[m1], label="Baseline", color="tab:blue", linewidth=1.5)
        m2 = np.isfinite(ora_c)
        ax.plot(t[m2], ora_c[m2], label="Oracle (upper bound)", color="tab:orange", linewidth=1.5)
        if hgs_cost is not None:
            ax.axhline(hgs_cost, color="red", linestyle="--", linewidth=1.5,
                       label=f"HGS ({hgs_cost:.2f})")
            ax.axhspan(ax.get_ylim()[0], hgs_cost, alpha=0.06, color="red")
        # Auto ylim to highlight baseline–oracle gap
        valid = np.concatenate([base_c[m1], ora_c[m2]])
        if len(valid) > 0:
            y_min, y_max = np.nanmin(valid), np.nanmax(valid)
            margin = max((y_max - y_min) * 0.08, 0.05)
            if hgs_cost is not None:
                y_min = min(y_min, hgs_cost)
                y_max = max(y_max, hgs_cost)
            # ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_ylim(17.15, 17.3)
        if wall_time_max is not None:
            ax.set_xlim(0, wall_time_max)
        ax.set_xlabel("Wall-clock Time")
        ax.set_ylabel("Best Cost Found")
        ax.set_title("Performance–Time Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    def _plot_unique_convergence(
        self, res: ExperimentResults, path: str | Path,
        wall_time_max: float | None, dpi: int,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        t = res.time_grid
        base_u = res.baseline_unique_series
        ora_u = res.oracle_unique_series
        if wall_time_max is not None and len(t) > 0 and wall_time_max > t[-1]:
            t = np.append(t, wall_time_max)
            base_u = np.append(base_u, base_u[-1])
            ora_u = np.append(ora_u, ora_u[-1])
        ax.plot(t, base_u, label="Baseline", color="tab:blue", linewidth=1.5)
        # Oracle stops at oracle_total_time (no new optima → no need to keep running)
        mask_ora = t <= res.oracle_total_time
        ax.plot(t[mask_ora], ora_u[mask_ora], label="Oracle (upper bound)", color="tab:orange", linewidth=1.5)
        if wall_time_max is not None:
            ax.set_xlim(0, wall_time_max)
        ax.set_xlabel("Wall-clock Time")
        ax.set_ylabel("Unique Local Optima Found")
        ax.set_title("Exploration–Time Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    def _plot_multiplier(
        self, res: ExperimentResults, path: str | Path,
        wall_time_max: float | None, dpi: int,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        cp = res.checkpoints
        tm = res.time_multipliers
        trm = res.trial_multipliers

        if wall_time_max is not None:
            mask = cp <= wall_time_max
            cp, tm, trm = cp[mask], tm[mask], trm[mask]

        ax.plot(cp, tm, marker="o", markersize=4, linewidth=1.5,
                label="Time-equivalent multiplier", color="tab:green")
        ax.plot(cp, trm, marker="s", markersize=3, linewidth=1.5,
                label="Trial-count multiplier", color="tab:purple", alpha=0.7)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)

        if len(tm) > 0:
            peak_idx = int(np.argmax(tm))
            ax.annotate(
                f"{tm[peak_idx]:.2f}\u00d7",
                xy=(cp[peak_idx], tm[peak_idx]),
                xytext=(15, 10), textcoords="offset points",
                fontsize=11, fontweight="bold", color="tab:green",
                arrowprops=dict(arrowstyle="->", color="tab:green"),
            )

        ax.set_xlabel("Wall-clock Time")
        ax.set_ylabel("Effective Work Multiplier")
        ax.set_title("Time / Exploration Multiplier")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    def _plot_combined(
        self, res: ExperimentResults, save_path: str | Path | None,
        wall_time_max: float | None, hgs_cost: float | None, dpi: int,
    ) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Saturation
        ax = axes[0, 0]
        ax.plot(res.total_trials, res.unique_basins, color="tab:blue", linewidth=1.5)
        ax.plot(res.total_trials, res.total_trials, color="gray", linestyle="--",
                linewidth=1, alpha=0.5)
        ax.fill_between(res.total_trials, res.unique_basins, res.total_trials,
                        alpha=0.12, color="tab:red")
        ax.text(0.05, 0.95,
                f"Redundancy: {res.final_redundancy:.1%}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel("Total Trials")
        ax.set_ylabel("Unique Local Optima Found")
        ax.set_title("1. Saturation Curve")
        ax.grid(True, alpha=0.3)

        # 2. Cost convergence
        ax = axes[0, 1]
        m1 = np.isfinite(res.baseline_cost_series)
        ax.plot(res.time_grid[m1], res.baseline_cost_series[m1],
                label="Baseline", color="tab:blue", linewidth=1.5)
        m2 = np.isfinite(res.oracle_cost_series)
        ax.plot(res.time_grid[m2], res.oracle_cost_series[m2],
                label="Oracle", color="tab:orange", linewidth=1.5)
        if hgs_cost is not None:
            ax.axhline(hgs_cost, color="red", linestyle="--", linewidth=1.5,
                       label=f"HGS ({hgs_cost:.2f})")
            # ax.set_ylim(hgs_cost - 0.5, hgs_cost + 2)
        if wall_time_max is not None:
            ax.set_xlim(0, wall_time_max)
        ax.set_xlabel("Wall-clock Time")
        ax.set_ylabel("Best Cost Found")
        ax.set_title("2. Cost Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Unique basins convergence (oracle stops at oracle_total_time)
        ax = axes[1, 0]
        ax.plot(res.time_grid, res.baseline_unique_series,
                label="Baseline", color="tab:blue", linewidth=1.5)
        mask_ora = res.time_grid <= res.oracle_total_time
        ax.plot(res.time_grid[mask_ora], res.oracle_unique_series[mask_ora],
                label="Oracle", color="tab:orange", linewidth=1.5)
        if wall_time_max is not None:
            ax.set_xlim(0, wall_time_max)
        ax.set_xlabel("Wall-clock Time")
        ax.set_ylabel("Unique Local Optima Found")
        ax.set_title("3. Exploration Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Multiplier
        ax = axes[1, 1]
        cp = res.checkpoints
        tm = res.time_multipliers
        trm = res.trial_multipliers
        if wall_time_max is not None:
            mask = cp <= wall_time_max
            cp, tm, trm = cp[mask], tm[mask], trm[mask]
        ax.plot(cp, tm, marker="o", markersize=3, linewidth=1.5,
                label="Time multiplier", color="tab:green")
        ax.plot(cp, trm, marker="s", markersize=3, linewidth=1.5,
                label="Trial multiplier", color="tab:purple", alpha=0.7)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        if len(tm) > 0:
            peak_idx = int(np.argmax(tm))
            ax.annotate(f"{tm[peak_idx]:.2f}\u00d7",
                        xy=(cp[peak_idx], tm[peak_idx]),
                        xytext=(15, 8), textcoords="offset points",
                        fontsize=10, fontweight="bold", color="tab:green",
                        arrowprops=dict(arrowstyle="->", color="tab:green"))
        ax.set_xlabel("Wall-clock Time")
        ax.set_ylabel("Effective Work Multiplier")
        ax.set_title("4. Time / Exploration Multiplier")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.close(fig)


# --------------------------------------------------------------------------
# Log generation from run_cuopt data (unchanged)
# --------------------------------------------------------------------------

def _points_to_trials_by_break(points: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Split points into trials (NaN in 'after' or 'best_so_far' = boundary)."""
    trials: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    for p in points:
        a = p.get("after")
        b = p.get("best_so_far")
        if (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
            if cur:
                trials.append(cur)
                cur = []
        else:
            cur.append(p)
    if cur:
        trials.append(cur)
    return trials


def logs_from_run_cuopt_points(
    points: Sequence[Dict[str, Any]],
    steps_to_time_ratio: Optional[float] = 1e-3,
    basin_key: str = "hash",
) -> List[Dict[str, Any]]:
    """Build analyzer logs from run_cuopt callback points.

    Args:
        basin_key:
            - "hash": use solution_hash when available (default; edge-based hash from callback).
            - "cost": use per-trial best cost as basin identity (coarser, matches tl_sensitivity).
    """
    trials = _points_to_trials_by_break(points)
    seen_basins: set = set()
    wall_clock = 0.0
    logs: List[Dict[str, Any]] = []

    for trial_id, seg in enumerate(trials):
        steps = len(seg)
        duration = steps * steps_to_time_ratio
        wall_clock += duration

        # Per-trial best cost among feasible iterations (fall back to all if none feasible)
        feasible_costs = [float(p.get("after", float("inf")))
                          for p in seg if p.get("is_feasible", True)]
        all_costs = [float(p.get("after", float("inf"))) for p in seg]
        trial_is_feasible = len(feasible_costs) > 0
        trial_best_cost = min(feasible_costs) if feasible_costs else min(all_costs)

        if basin_key == "hash":
            bid = seg[-1].get("solution_hash", trial_best_cost)
        else:
            bid = trial_best_cost

        if trial_is_feasible:
            is_new = bid not in seen_basins
            seen_basins.add(bid)
        else:
            is_new = False

        logs.append({
            "trial_id": trial_id,
            "wall_clock_time": wall_clock,
            "duration": duration,
            "steps": steps,
            "final_cost": trial_best_cost,
            "basin_id": bid,
            "is_new_discovery": is_new,
            "is_feasible": trial_is_feasible,
        })
    return logs


def logs_from_baseline_csv(filepath: str) -> List[Dict[str, Any]]:
    """Load trial logs from the baseline_log.csv format produced by tl_sensitivity."""
    import csv

    with open(filepath, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {filepath}")

    rows.sort(key=lambda r: float(r["timestamp"]))
    logs: List[Dict[str, Any]] = []
    prev_t = 0.0
    for row in rows:
        t = float(row["timestamp"])
        logs.append({
            "trial_id": int(row.get("trial_id", len(logs))),
            "wall_clock_time": t,
            "duration": max(0.0, t - prev_t),
            "steps": 1,
            "final_cost": float(row["cost"]),
            "basin_id": row["true_basin_id"],
            "is_new_discovery": row.get("is_new_basin", "0") == "1",
        })
        prev_t = t
    return logs


def logs_from_run_cuopt_log_file(
    filepath: str,
    steps_to_time_ratio: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Parse run_cuopt text log into analyzer logs.

    Detects [INFEASIBLE] tag per iteration. A trial is feasible if its last
    iteration is feasible. basin_id = best feasible cost within the trial.

    Time calibration: if *steps_to_time_ratio* is ``None`` (default), the
    function tries to extract the actual time limit from the log's
    ``[Summary] Time limit: X s`` line and computes the ratio as
    ``time_limit / total_steps`` so that the synthetic timeline matches the
    real wall-clock duration.  Falls back to ``1e-3`` when no summary is found.
    """
    text = Path(filepath).read_text(encoding="utf-8", errors="replace")
    cost_re = re.compile(r"cost\s+before:\s*[\d.e+-]+\s*,\s*cost\s+after:\s*([\d.e+-]+)", re.I)
    break_re = re.compile(r"\[search\s+#\d+\]|#\s*---\s*break\s*---", re.I)

    # Each trial element: list of (cost, is_feasible) tuples
    trials: List[List[tuple]] = []
    current: List[tuple] = []
    for line in text.splitlines():
        line = line.strip()
        if break_re.search(line):
            if current:
                trials.append(current)
                current = []
            continue
        m = cost_re.search(line)
        if m:
            feas = "[INFEASIBLE]" not in line
            current.append((float(m.group(1)), feas))
    if current:
        trials.append(current)

    # Determine steps_to_time_ratio: auto-calibrate from log summary if not
    # explicitly provided.
    if steps_to_time_ratio is None:
        total_steps = sum(len(t) for t in trials)
        tl_match = re.search(r"Time limit:\s*([\d.]+)\s*s", text)
        if tl_match and total_steps > 0:
            steps_to_time_ratio = float(tl_match.group(1)) / total_steps
        else:
            steps_to_time_ratio = 1e-3

    seen_costs: set[float] = set()
    wall_clock = 0.0
    logs: List[Dict[str, Any]] = []

    for trial_id, iters in enumerate(trials):
        steps = len(iters)
        duration = steps * steps_to_time_ratio
        wall_clock += duration

        feasible_costs = [c for c, f in iters if f]
        trial_feasible = len(feasible_costs) > 0
        final_cost = min(feasible_costs) if feasible_costs else min(c for c, _ in iters)

        if trial_feasible:
            is_new = final_cost not in seen_costs
            seen_costs.add(final_cost)
        else:
            is_new = False

        logs.append({
            "trial_id": trial_id,
            "wall_clock_time": wall_clock,
            "duration": duration,
            "steps": steps,
            "final_cost": final_cost,
            "basin_id": final_cost,
            "is_new_discovery": is_new,
            "is_feasible": trial_feasible,
        })
    return logs


def _regroup_by_cost(logs: List[Dict[str, Any]]) -> None:
    """Re-assign basin_id / is_new_discovery using final_cost (in-place).

    Infeasible trials (is_feasible=False) are never counted as new basins.
    """
    seen: set = set()
    for entry in logs:
        cost = entry["final_cost"]
        entry["basin_id"] = cost
        if entry.get("is_feasible", True):
            entry["is_new_discovery"] = cost not in seen
            seen.add(cost)
        else:
            entry["is_new_discovery"] = False


def _generate_dummy_logs(n_trials=200, base_duration=0.1, n_unique_basins=50, seed=42):
    rng = np.random.default_rng(seed)
    logs = []
    seen = set()
    all_ids = list(range(n_unique_basins))
    wall_time = 0.0
    cost = 2000.0

    for tid in range(n_trials):
        prob_new = max(0.05, 1.0 - tid / n_trials)
        is_new = rng.random() < prob_new and len(seen) < n_unique_basins

        if is_new:
            remaining = [b for b in all_ids if b not in seen]
            bid = int(rng.choice(remaining))
            seen.add(bid)
            cost -= rng.uniform(1.0, 10.0)
        else:
            bid = int(rng.choice(list(seen))) if seen else 0
            cost += rng.normal(0.0, 0.5)

        cost = max(500.0, cost)
        duration = base_duration * rng.uniform(0.5, 1.5)
        wall_time += duration

        logs.append({
            "trial_id": tid,
            "wall_clock_time": wall_time,
            "duration": duration,
            "steps": int(rng.integers(80, 120)),
            "final_cost": cost,
            "basin_id": f"hash_{bid}",
            "is_new_discovery": bool(is_new),
        })
    return logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upper bound & time expansion analysis for VRP solver trial logs.",
    )
    parser.add_argument("--log", type=str, default=None,
                        help="run_cuopt text log (basin = cost, approximate)")
    parser.add_argument("--points_json", type=str, default=None,
                        help="JSON list of callback points")
    parser.add_argument("--upper_bound_log", type=str, default=None,
                        help="Pre-built trial log JSON")
    parser.add_argument("--baseline_csv", type=str, default=None,
                        help="baseline_log.csv from tl_sensitivity (has true_basin_id)")
    parser.add_argument("--basin_key", type=str, default="cost",
                        choices=["cost", "hash"],
                        help="Basin identity: 'cost' (per-trial best cost) or 'hash' (solution_hash)")
    parser.add_argument("--out_dir", type=str, default=".",
                        help="Output directory for plots. If left as '.', "
                             "an 'ub' subdirectory is created next to the "
                             "input log/baseline file.")
    parser.add_argument("--combined", action="store_true",
                        help="Save a single combined 2x2 figure instead of 4 individual plots")
    parser.add_argument("--save_log", type=str, default=None,
                        help="Save parsed trial log to JSON")
    parser.add_argument("--overhead", type=float, default=0.0,
                        help="Oracle overhead ratio (0.0 = ideal upper bound, 0.2 = physical)")
    parser.add_argument("--wall_time_max", type=float, default=None,
                        help="X-axis upper limit for time-based plots")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Figure resolution")
    parser.add_argument("--solution_pkl", type=str, default="/home/jieyi/hgs_cvrp100_uniform.pkl", #"/home/jieyi/CaR-constraint/data/CVRP/hgs_cvrp1000_uniform_LV0.pkl",
                        help="Path to HGS solution pkl (for auto HGS cost).")
    parser.add_argument("--instance_index", type=int, default=0,
                        help="Instance index into the solution pkl.")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Divide all costs by this value (e.g. 100 for raw C++ log)")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--steps_to_time", type=float, default=None,
                        help="Steps-to-time conversion ratio (default: auto-calibrate "
                             "from log's Time limit, fallback 1e-3)")
    args = parser.parse_args()

    # Decide default output dir: if user did not override --out_dir (still '.'),
    # place figures in a sibling 'ub' directory next to the main input log.
    auto_out_root: Path | None = None
    if args.out_dir == ".":
        base_path: str | None = None
        if args.baseline_csv:
            base_path = args.baseline_csv
        elif args.upper_bound_log:
            base_path = args.upper_bound_log
        elif args.log:
            base_path = args.log
        elif args.points_json:
            base_path = args.points_json
        if base_path:
            auto_out_root = Path(base_path).resolve().parent / f"ub_{str(int(args.wall_time_max))}"

    if args.baseline_csv:
        logs = logs_from_baseline_csv(args.baseline_csv)
    elif args.upper_bound_log:
        with open(args.upper_bound_log) as f:
            logs = json.load(f)
        if args.basin_key == "cost":
            _regroup_by_cost(logs)
    elif args.log:
        logs = logs_from_run_cuopt_log_file(args.log, steps_to_time_ratio=args.steps_to_time)
    elif args.points_json:
        with open(args.points_json) as f:
            logs = logs_from_run_cuopt_points(
                json.load(f),
                steps_to_time_ratio=args.steps_to_time or 1e-3,
                basin_key=args.basin_key,
            )
    else:
        logs = _generate_dummy_logs()

    # Apply cost scaling (e.g. raw C++ log costs are 100× larger)
    if args.scale != 1.0:
        for entry in logs:
            entry["final_cost"] /= args.scale
            if not isinstance(entry["basin_id"], str):
                entry["basin_id"] = entry["final_cost"]
        _regroup_by_cost(logs)

    # Auto HGS cost from solution_pkl
    hgs_cost = None
    if args.solution_pkl:
        try:
            import pickle

            with open(args.solution_pkl, "rb") as f:
                sols = pickle.load(f)
            if not isinstance(sols, (list, tuple)):
                raise TypeError("solution_pkl must contain a list/tuple of (cost, routes)")
            raw = sols[args.instance_index][0]
            # raw HGS cost is already in the same (unscaled) units as tl_sensitivity.
            # We do NOT divide by args.scale here; only the C++ log needs scaling.
            hgs_cost = float(raw)
            print(f"[upper_bound_analyzer] Loaded HGS cost {hgs_cost:.4f} "
                  f"from {args.solution_pkl} (index {args.instance_index})")
        except Exception as exc:  # noqa: BLE001
            print(f"[upper_bound_analyzer] WARNING: failed to load HGS cost from "
                  f"{args.solution_pkl}: {exc}")

    if args.save_log:
        Path(args.save_log).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_log, "w") as f:
            json.dump(logs, f, indent=2)
        print(f"Saved trial log to {args.save_log}")

    analyzer = UpperBoundAnalyzer()
    analyzer.load_logs(logs)
    analyzer.run_analysis(overhead_ratio=args.overhead)

    if auto_out_root is not None:
        args.out_dir = str(auto_out_root)

    if not args.no_plot:
        if args.combined:
            fig_path = Path(args.out_dir) / "upper_bound_analysis.png"
            analyzer.plot_results(
                save_path=fig_path,
                wall_time_max=args.wall_time_max,
                hgs_cost=hgs_cost,
                dpi=args.dpi,
            )
        else:
            analyzer.plot_results(
                save_dir=args.out_dir,
                wall_time_max=args.wall_time_max,
                hgs_cost=hgs_cost,
                dpi=args.dpi,
            )
