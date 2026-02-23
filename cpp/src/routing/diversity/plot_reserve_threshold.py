"""
Plot reserve_population.threshold vs normalized time in adjust_reserve_threshold().
Uses actual CVRP/VRP defaults: initial = 0.8 (multi-island), max = 0.99.
Formula: threshold = initial + t² × (0.99 − initial), t in [0, 1].
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Normalized time: 0 = reserve_start_time, 1 = time_limit
t = np.linspace(0, 1, 200)
max_threshold = 0.99

# CVRP multi-island default (generate_from_scratch): max(0.8, diversity_levels[threshold_index])
initial_cvrp = 0.8
reserve_time_ratio = t * t
threshold_cvrp = initial_cvrp + reserve_time_ratio * (max_threshold - initial_cvrp)

# Single-island case: threshold stays 0.99 (no adjustment needed, but show for reference)
threshold_single = np.full_like(t, 0.99)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t, threshold_cvrp, "b-", linewidth=2.5, label="CVRP multi-island (initial = 0.8)")
ax.plot(t, threshold_single, "gray", linestyle="--", linewidth=1.5, label="Single island (fixed 0.99)")
ax.set_xlabel("Normalized time  t  (0 = reserve_start, 1 = time_limit)", fontsize=11)
ax.set_ylabel("reserve_population.threshold", fontsize=11)
ax.set_title("adjust_reserve_threshold() for VRP/CVRP\nthreshold = initial + t² × (0.99 − initial)")
ax.axhline(initial_cvrp, color="blue", linestyle=":", alpha=0.5)
ax.axhline(max_threshold, color="gray", linestyle=":", alpha=0.5)
ax.legend(loc="lower right", fontsize=10)
ax.set_ylim(0.7, 1.02)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reserve_threshold_curve.png", dpi=120)
print("Saved reserve_threshold_curve.png (CVRP initial=0.8, max=0.99)")
