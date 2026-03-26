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
    inserted: bool = True


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

    # Skip strategy description (for plot labels)
    skip_label: str = "dup-only"
    n_skipped_dup: int = 0
    n_skipped_quality: int = 0


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
                inserted=d.get("inserted", True),
            )
            for d in data
        ]
        self.trials.sort(key=lambda r: (r.wall_clock_time, r.trial_id))

    def run_analysis(
        self,
        overhead_ratio: float = 0.0,
        n_checkpoints: int = 50,
        dt: float = 0.1,
        skip_margin_pct: float | None = None,
        top_k_basins: int | None = None,
        keep_best_pct: float | None = None,
        ancestry_basins: set | None = None,
        skip_not_inserted: bool = False,
    ) -> ExperimentResults:
        """Run all four experiments.

        Args:
            overhead_ratio: Fraction of trial duration still paid on skipped trials
                (0.0 = ideal oracle / theoretical upper bound).
            n_checkpoints: Number of checkpoints for multiplier curve.
            dt: Time resolution for convergence curves.
            skip_margin_pct: (Strategy A) Skip trials whose cost exceeds
                ``best_so_far * (1 + margin/100)``.  Causal — only uses info
                available up to the current trial.
            top_k_basins: (Strategy B) Retrospectively keep only the top-K
                lowest-cost unique basins; skip trials leading to all others.
            keep_best_pct: (Strategy C) Retrospectively keep only the best P%
                of unique basins by cost; skip the rest.
            ancestry_basins: (Strategy D) Set of basin costs that belong to
                the best solution's ancestry tree.  Trials whose basin is
                not in this set are skipped.
            skip_not_inserted: (Strategy E) Skip trials whose offspring was
                not inserted/replaced into the population (retrospective).
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

        # ── Compute can_skip mask ──
        base_times = np.cumsum(durations)
        can_skip_dup = ~is_new                        # duplicate trials
        can_skip_quality = np.zeros(n, dtype=bool)    # quality-based skip

        # Strategy A: causal margin-based skip
        if skip_margin_pct is not None:
            margin = skip_margin_pct / 100.0
            running_best = np.inf
            for i in range(n):
                if feasible[i]:
                    if costs[i] > running_best * (1 + margin):
                        can_skip_quality[i] = True
                    running_best = min(running_best, costs[i])

        # Strategies B & C share basin-cost ranking (retrospective)
        if top_k_basins is not None or keep_best_pct is not None:
            basin_best: dict[Any, float] = {}
            for t in self.trials:
                if t.is_feasible:
                    bid = t.basin_id
                    if bid not in basin_best or t.final_cost < basin_best[bid]:
                        basin_best[bid] = t.final_cost
            ranked = sorted(basin_best.items(), key=lambda x: x[1])

            keep_ids: set[Any] | None = None

            if top_k_basins is not None:
                keep_ids = {bid for bid, _ in ranked[:top_k_basins]}

            if keep_best_pct is not None:
                k = max(1, int(len(ranked) * keep_best_pct / 100.0))
                pct_ids = {bid for bid, _ in ranked[:k]}
                keep_ids = pct_ids if keep_ids is None else keep_ids & pct_ids

            if keep_ids is not None:
                for i, t in enumerate(self.trials):
                    if t.is_feasible and t.basin_id not in keep_ids:
                        can_skip_quality[i] = True

        # Strategy D: ancestry-based skip (retrospective)
        if ancestry_basins is not None:
            cost_precision = 4
            for i, t in enumerate(self.trials):
                if t.is_feasible:
                    rounded = round(float(t.basin_id), cost_precision) \
                        if isinstance(t.basin_id, (int, float)) else t.basin_id
                    if rounded not in ancestry_basins:
                        can_skip_quality[i] = True

        # Strategy E: skip trials not inserted into the population
        if skip_not_inserted:
            for i, t in enumerate(self.trials):
                if not t.inserted:
                    can_skip_quality[i] = True

        can_skip = can_skip_dup | can_skip_quality

        # Build descriptive label
        label_parts: list[str] = []
        if skip_margin_pct is not None:
            label_parts.append(f"margin>{skip_margin_pct}%")
        if top_k_basins is not None:
            label_parts.append(f"top-{top_k_basins}")
        if keep_best_pct is not None:
            label_parts.append(f"best-{keep_best_pct}%")
        if ancestry_basins is not None:
            label_parts.append(f"ancestry({len(ancestry_basins)})")
        if skip_not_inserted:
            label_parts.append("pop-only")
        res.skip_label = "dup + " + " + ".join(label_parts) if label_parts else "dup-only"
        res.n_skipped_dup = int(can_skip_dup.sum())
        res.n_skipped_quality = int((can_skip_quality & ~can_skip_dup).sum())

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
        ora_label = f"Oracle ({res.skip_label})"
        ax.plot(t[m2], ora_c[m2], label=ora_label, color="tab:orange", linewidth=1.5)
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
            ax.set_ylim(y_min - margin, 39.5) # y_max + margin
            # ax.set_ylim(37.0, 37.)
        if wall_time_max is not None:
            ax.set_xlim(0, wall_time_max)
        ax.set_xlabel("Wall-clock Time")
        ax.set_ylabel("Best Cost Found")
        ax.set_title("Performance–Time Convergence")
        ax.legend(loc="upper right")
        # Time to reach final best cost (baseline's last result)
        final_best = np.nanmin(base_c) if len(base_c) else np.nan
        if np.isfinite(final_best) and len(t) > 0:
            hit_b = np.where(base_c <= final_best)[0]
            hit_o = np.where(ora_c <= final_best)[0]
            t_baseline = float(t[hit_b[0]]) if len(hit_b) else float(t[-1])
            t_oracle = float(t[hit_o[0]]) if len(hit_o) else float(t[-1])
            pct = (t_oracle / t_baseline * 100.0) if t_baseline > 0 else 0.0
            time_txt = (f"Baseline time to final: {t_baseline:.2f}s\n"
                        f"Oracle time to final: {t_oracle:.2f}s ({pct:.1f}%)")
        else:
            time_txt = (f"Baseline total: {res.baseline_total_time:.2f}s, "
                        f"Oracle total: {res.oracle_total_time:.2f}s")
        ax.text(0.98, 0.28, time_txt, transform=ax.transAxes, fontsize=9,
                va="top", ha="right", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
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
        mask_ora = t <= res.oracle_total_time
        ora_label = f"Oracle ({res.skip_label})"
        ax.plot(t[mask_ora], ora_u[mask_ora], label=ora_label, color="tab:orange", linewidth=1.5)
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
    """Parse run_cuopt text log into analyzer logs (streaming).

    Detects [INFEASIBLE] tag per iteration. A trial is feasible if its last
    iteration is feasible. basin_id = best feasible cost within the trial.

    Also parses ``[POP] add_solution: INSERTED/REPLACED/REJECTED`` lines
    (printed by verbose population) to determine whether the trial's
    offspring entered the population.

    Time calibration: if *steps_to_time_ratio* is ``None`` (default), the
    function tries to extract the actual time limit from the log's
    ``[Summary] Time limit: X s`` line and computes the ratio as
    ``time_limit / total_steps`` so that the synthetic timeline matches the
    real wall-clock duration.  Falls back to ``1e-3`` when no summary is found.
    """
    cost_re = re.compile(r"cost\s+before:\s*[\d.e+-]+\s*,\s*cost\s+after:\s*([\d.e+-]+)", re.I)
    break_re = re.compile(r"\[search\s+#\d+\]|#\s*---\s*break\s*---", re.I)
    pop_insert_re = re.compile(r"\[POP\] add_solution: (INSERTED|REPLACED|REJECTED)")

    # Each trial: (list of (cost, is_feasible), inserted_flag)
    trials: List[tuple] = []   # [(iters, inserted), ...]
    current: List[tuple] = []  # [(cost, is_feasible), ...]
    current_inserted: bool = True  # default True (if no [POP] line found)

    time_limit_s: float | None = None

    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if break_re.search(line):
                if current:
                    trials.append((current, current_inserted))
                    current = []
                    current_inserted = True
                continue
            m = cost_re.search(line)
            if m:
                feas = "[INFEASIBLE]" not in line
                current.append((float(m.group(1)), feas))
                continue
            mp = pop_insert_re.search(line)
            if mp:
                current_inserted = mp.group(1) in ("INSERTED", "REPLACED")
                continue
            if time_limit_s is None:
                tl = re.search(r"Time limit:\s*([\d.]+)\s*s", line)
                if tl:
                    time_limit_s = float(tl.group(1))

    if current:
        trials.append((current, current_inserted))

    if steps_to_time_ratio is None:
        total_steps = sum(len(iters) for iters, _ in trials)
        if time_limit_s is not None and total_steps > 0:
            steps_to_time_ratio = time_limit_s / total_steps
        else:
            steps_to_time_ratio = 1e-3

    seen_costs: set[float] = set()
    wall_clock = 0.0
    logs: List[Dict[str, Any]] = []

    for trial_id, (iters, inserted) in enumerate(trials):
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
            "inserted": inserted,
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


def parse_lineage_from_log(
    filepath: str,
    scale: float = 1.0,
    cost_precision: int = 4,
) -> Dict[float, set]:
    """Parse [EVOLVE] lines from a verbose C++ log to build a lineage map.

    Returns a dict ``{child_basin_cost: {parent_cost_1, parent_cost_2, ...}}``.
    Costs are scaled by ``1/scale`` and rounded to ``cost_precision`` decimals
    to match the trial log representation.

    Handles ``recombine FAILED`` / ``SKIPPED`` cases where no child line
    follows a parent line — the pending parents are discarded when the next
    parent line or a FAILED/SKIPPED line is encountered.
    """
    parent_re = re.compile(
        r"\[EVOLVE\]\s+step=\d+\s+parents:\s+basin_A=([\d.e+-]+)\s+basin_B=([\d.e+-]+)"
    )
    child_re = re.compile(
        r"\[EVOLVE\]\s+crossover.*?new_basin=([\d.e+-]+)"
    )
    no_child_re = re.compile(
        r"\[EVOLVE\]\s+(recombine FAILED|recombine.*?SKIPPED)"
    )

    lineage: Dict[float, set] = {}
    pending_parents: tuple | None = None

    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "[EVOLVE]" not in line:
                continue
            mp = parent_re.search(line)
            if mp:
                pa = round(float(mp.group(1)) / scale, cost_precision)
                pb = round(float(mp.group(2)) / scale, cost_precision)
                pending_parents = (pa, pb)
                continue
            mc = child_re.search(line)
            if mc and pending_parents is not None:
                child = round(float(mc.group(1)) / scale, cost_precision)
                lineage.setdefault(child, set()).update(pending_parents)
                pending_parents = None
                continue
            if no_child_re.search(line):
                pending_parents = None

    return lineage


def compute_ancestry(
    lineage: Dict[float, set],
    target: float,
    cost_precision: int = 4,
) -> set:
    """BFS from *target* basin through the lineage map to collect all ancestors."""
    target = round(target, cost_precision)
    ancestry: set = {target}
    queue = [target]
    while queue:
        current = queue.pop()
        for parent in lineage.get(current, set()):
            if parent not in ancestry:
                ancestry.add(parent)
                queue.append(parent)
    return ancestry


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
    parser.add_argument("--solution_pkl", type=str, default="/home/jieyi/CaR-constraint/data/CVRP/hgs_cvrp1000_uniform_LV0.pkl", #"/home/jieyi/hgs_cvrp100_uniform.pkl",
                        help="Path to HGS solution pkl (for auto HGS cost).")
    parser.add_argument("--instance_index", type=int, default=9,
                        help="Instance index into the solution pkl.")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Divide all costs by this value (e.g. 100 for raw C++ log)")
    parser.add_argument("--skip_margin_pct", type=float, default=None,
                        help="(Strategy A) Skip new basins whose cost exceeds "
                             "best_so_far × (1+margin/100). Causal.")
    parser.add_argument("--top_k_basins", type=int, default=None,
                        help="(Strategy B) Retrospectively keep only the K best "
                             "unique basins; skip the rest.")
    parser.add_argument("--keep_best_pct", type=float, default=None,
                        help="(Strategy C) Retrospectively keep only the best P%% "
                             "of unique basins; skip the rest.")
    parser.add_argument("--ancestry", action="store_true",
                        help="(Strategy D) Keep only basins in the best solution's "
                             "ancestry tree. Requires [EVOLVE] lines in --log or --lineage_log.")
    parser.add_argument("--lineage_log", type=str, default=None,
                        help="Path to verbose C++ log containing [EVOLVE] lines "
                             "(for --ancestry). Defaults to --log if not specified.")
    parser.add_argument("--skip_not_inserted", action="store_true",
                        help="(Strategy E) Skip trials whose offspring was not inserted "
                             "into the population. Requires [EVOLVE] insertion_rank lines.")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--steps_to_time", type=float, default=None,
                        help="Steps-to-time conversion ratio (default: auto-calibrate "
                             "from log's Time limit, fallback 1e-3)")
    args = parser.parse_args()

    # Determine the "base path" for auto output dir (resolved after run_analysis).
    _auto_base_path: str | None = None
    if args.out_dir == ".":
        for candidate in (args.baseline_csv, args.upper_bound_log,
                          args.log, args.points_json):
            if candidate:
                _auto_base_path = candidate
                break

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

    # Strategy D: ancestry
    ancestry_set: set | None = None
    if args.ancestry:
        lineage_path = args.lineage_log or args.log
        if lineage_path is None:
            print("[upper_bound_analyzer] ERROR: --ancestry requires --log or --lineage_log "
                  "with [EVOLVE] lines.")
        else:
            lineage = parse_lineage_from_log(lineage_path, scale=args.scale)
            print(f"[upper_bound_analyzer] Parsed lineage: "
                  f"{len(lineage)} child basins from {lineage_path}")
            feasible_costs = [e["final_cost"] for e in logs
                              if e.get("is_feasible", True)]
            if feasible_costs:
                best = min(feasible_costs)
                ancestry_set = compute_ancestry(lineage, best)
                print(f"  Best basin cost: {best:.4f}, "
                      f"ancestry tree size: {len(ancestry_set)} basins")
            else:
                print("  WARNING: no feasible trials found; ancestry disabled.")

    analyzer = UpperBoundAnalyzer()
    analyzer.load_logs(logs)
    res = analyzer.run_analysis(
        overhead_ratio=args.overhead,
        skip_margin_pct=args.skip_margin_pct,
        top_k_basins=args.top_k_basins,
        keep_best_pct=args.keep_best_pct,
        ancestry_basins=ancestry_set,
        skip_not_inserted=args.skip_not_inserted,
    )
    print(f"[upper_bound_analyzer] Strategy: {res.skip_label}")
    print(f"  Trials: {len(analyzer.trials)}, "
          f"Skipped: {res.n_skipped_dup} dup + {res.n_skipped_quality} quality, "
          f"Redundancy: {res.final_redundancy:.1%}")
    print(f"  Baseline time: {res.baseline_total_time:.2f}s, "
          f"Oracle time: {res.oracle_total_time:.2f}s "
          f"({res.oracle_total_time / res.baseline_total_time:.1%})")

    if _auto_base_path is not None:
        suffix = res.skip_label.replace(" ", "").replace("+", "_")
        wt = f"_{int(args.wall_time_max)}" if args.wall_time_max else ""
        args.out_dir = str(Path(_auto_base_path).resolve().parent / f"ub_{suffix}{wt}")

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
