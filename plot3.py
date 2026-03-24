import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import matplotlib
import seaborn as sns

times_C = np.array([132.0, 256.8, 383.4, 509.4, 634.8, 762.6, 891.6, 1017.0])
gaps_C  = np.array([0]*len(times_C))

gaps_B = np.array([
    99.9, 97.7, 87.9, 80.4, 75.6, 71.3, 68.1, 65.0, 63.2, 61.7,
    59.9, 58.5, 57.8, 56.5, 55.5, 54.1, 53.1, 52.7, 51.7, 51.2,
    50.6, 50.2, 49.9, 49.7, 49.3, 48.7, 48.2, 47.7, 47.4, 47.3,
    47.1, 47.0, 47.0, 46.7, 46.5, 46.2, 45.8, 45.4, 45.2, 44.5,
    44.3, 44.1, 43.7, 43.3, 43.3, 43.2, 43.0, 42.6, 42.5, 42.5,
    42.5, 42.4, 42.4, 42.3, 42.2, 42.1, 42.1, 42.0, 42.0, 41.8,
    41.6, 41.6, 41.4, 41.2, 41.0, 40.9, 40.8, 40.8, 40.7, 40.6,
    40.6, 40.5, 40.4, 40.3, 40.3, 40.2, 40.1, 40.1, 40.0, 39.9,
    39.9, 39.8, 39.7, 39.7, 39.6, 39.6, 39.5, 39.5, 39.4, 39.4,
    39.3, 39.3, 39.2, 39.2, 39.2, 39.1, 39.1, 39.1, 39.1, 39.1
])[1::2]

# gaps_B = np.array([
#     99.9, 97.7, 87.9, 80.4, 75.6, 71.3, 68.1, 65.0, 63.2, 61.7,
#     59.9, 58.5, 57.8, 56.5, 55.5, 54.1, 53.1, 52.7, 51.7, 51.2,
#     50.6, 50.2, 49.9, 49.7, 49.3, 48.7, 48.2, 47.7, 47.4, 47.3,
#     47.1, 47.0, 47.0, 46.7, 46.5, 46.2, 45.8, 45.4, 45.2, 44.5,
#     44.3, 44.1, 43.7, 43.3, 43.3, 43.2, 43.0, 42.6, 42.5, 42.5,
#     42.5, 42.4, 42.4, 42.3, 42.2, 42.1, 42.1, 42.0, 42.0, 41.8,
#     41.6, 41.6, 41.4, 41.2, 41.0, 40.9, 40.8
# ])

# times_B = np.array([
#      9,  18,  27,  36,  45,  54,  63,  72,  81,  90,
#     99, 108, 117, 126, 135, 144, 153, 162, 171, 180,
#    189, 198, 207, 216, 225, 234, 243, 252, 261, 270,
#    279, 288, 297, 306, 315, 324, 333, 342, 351, 360,
#    369, 378, 387, 396, 405, 414, 423, 432, 441, 450,
#    459, 468, 477, 486, 495, 504, 513, 522, 531, 540,
#    549, 558, 567, 576, 585, 594, 603
# ])
T_array = np.array([
    50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
    550, 600, 650, 700, 750, 800, 850, 900, 950, 1000,
    1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500,
    1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000,
    2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500,
    2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950, 3000,
    3050, 3100, 3150, 3200, 3250, 3300, 3350, 3400, 3450, 3500,
    3550, 3600, 3650, 3700, 3750, 3800, 3850, 3900, 3950, 4000,
    4050, 4100, 4150, 4200, 4250, 4300, 4350, 4400, 4450, 4500,
    4550, 4600, 4650, 4700, 4750, 4800, 4850, 4900, 4950, 5000
])[1::2]
times_B = T_array * 762 / 5000

times_A = np.array([  5.83,  10.31,  11.77,  14.75,  19.18,  20.59,  23.26,  28.32,  41.7,   45.89,
  51.07,  59.6,   72.59,  82.23, 117.5,  117.71, 153.69, 188.59, 225.76, 259.87,
 296.45, 331.97, 369.88, 402.2,  480.82, 511.85, 619.04, 655.13])
gaps_A  = np.array([0]*len(times_A))


def normalized_pi(t, g, T_MAX):
    """Normalized performance index: integral of gap over [0, T_MAX] / T_MAX."""
    mask = t <= T_MAX
    if not np.any(mask):
        return np.nan
    t, g = t[mask].astype(float), g[mask].astype(float)
    if t[0] > 0:
        t = np.insert(t, 0, 0.0)
        g = np.insert(g, 0, g[0])
    if t[-1] < T_MAX:
        t = np.append(t, T_MAX)
        g = np.append(g, np.interp(T_MAX, t[:-1], g[:-1]))
    return np.trapz(g, t) / T_MAX


# ===================== 画图与指标 =====================
sns.set(style="white")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 18+5,
    'axes.labelsize': 18+5,
    'xtick.labelsize': 16+5,
    'ytick.labelsize': 16+5,
    'legend.fontsize': 16,
})
SMOOTH = False
T_MAX  = 600.0

methods = {
    "NeuOpt*": (times_B, gaps_B),  # 红
    "SGBS":    (times_C, gaps_C),  # 蓝
    "Ours":    (times_A, gaps_A),  # 绿（已替换为"reconstruction_aug_gap, eval_time"）
}

# 计算并绘图
plt.figure(figsize=(4, 3))
color_map = {"NeuOpt*": '#d62728', "SGBS": '#1f77b4', "Ours": '#2ca02c'}
markers = {k: '.' for k in methods}

pi_results = {}
for name, (t_raw, g_raw) in methods.items():
    t, g = t_raw, g_raw  # pareto_filter(t_raw, g_raw)
    pi_results[name] = normalized_pi(t, g, T_MAX)
    plt.plot(t, g, marker=markers[name], lw=1.4, color=color_map[name], label=name)
    if SMOOTH:
        dense_t = np.linspace(0, min(T_MAX, t[-1]), 600)
        smooth_g = make_interp_spline(t, g, k=3)(dense_t)
        plt.plot(dense_t, smooth_g, lw=1, alpha=0.6, color=color_map[name])

plt.xlim(0, T_MAX)
plt.ylim(-3, 100)
plt.xlabel("Time (s)", fontsize=18)
plt.ylabel("Infeasibility (%)",  fontsize=18)
plt.xticks(np.arange(0, T_MAX+1, 200), fontsize=16)
plt.yticks(np.arange(0, 101, 20), fontsize=16)
plt.grid(alpha=0.2)
plt.tight_layout()
# plt.legend(loc="best")
plt.savefig("PI_bltw_new_infsb_NEW.pdf", bbox_inches='tight')
plt.show()

print("PI results (lower is better):")
for k, v in pi_results.items():
    print(f"{k}: {v:.4f}")
