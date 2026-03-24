import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import make_interp_spline
import matplotlib
import seaborn as sns


def pareto_filter(times, gaps):
    order = np.argsort(times)
    t, g = times[order], gaps[order]
    keep_t, keep_g = [t[0]], [g[0]]
    best = g[0]
    for ti, gi in zip(t[1:], g[1:]):
        if gi < best:
            keep_t.append(ti); keep_g.append(gi); best = gi
    return np.array(keep_t), np.array(keep_g)


def normalized_pi(times, gaps, T):
    t = np.concatenate(([0.0], times))
    g = np.concatenate(([gaps[0]], gaps))
    area = 0.0
    for i in range(len(t)-1):
        if t[i] >= T:
            break
        next_t = min(t[i+1], T)
        area += (next_t - t[i]) * g[i]
        if t[i+1] >= T:
            break
    if t[-1] < T:
        area += (T - t[-1]) * g[-1]
    return 100 * area / T


# ==========================================
# 1. 数据准备 (Data) - from plot3.py
# ==========================================
# Ours (Green)
times_A = np.array([
    5.83, 10.31, 11.77, 14.75, 19.18, 20.59, 23.26, 28.32, 41.7, 45.89,
    51.07, 59.6, 72.59, 82.23, 117.5, 117.71, 153.69, 188.59, 225.76, 259.87,
    296.45, 331.97, 369.88, 402.2, 480.82, 511.85, 619.04, 655.13
])
gaps_A = np.array([0] * len(times_A))

# NeuOpt* (Red)
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

# SGBS (Blue)
times_C = np.array([0, 132.0, 256.8, 383.4, 509.4, 634.8, 762.6, 891.6, 1017.0])
gaps_C = np.array([0] * len(times_C))

# ==========================================
# 2. 画布与断轴设置
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1]})
ax1.set_facecolor('#ffffff')
ax2.set_facecolor('#ffffff')
fig.patch.set_facecolor('#ffffff')

T_MAX = 300.0  # X axis 0--300s

# Top: high infeasibility (NeuOpt*); bottom: low/zero (Ours, SGBS)
ax1.set_xlim(0, T_MAX)
ax2.set_xlim(0, T_MAX)
ax1.set_ylim(30, 100)
ax2.set_ylim(-0.1, 1)

# 隐藏边框来表现断轴
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)
ax2.xaxis.tick_bottom()

# 绘制断轴破折号标记
d = 0.015
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, lw=1.5)
ax1.plot((-d, +d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

ax1.grid(True, which="major", ls="--", alpha=0.5, color="#e0e0e0")
ax2.grid(True, which="major", ls="--", alpha=0.5, color="#e0e0e0")

ax2.set_xlabel("Inference Time (seconds)", fontsize=26, fontweight='bold', color="#333333")
fig.text(0.04, 0.5, 'Infeasibility (%)', va='center', rotation='vertical', fontsize=26, fontweight='bold', color="#333333")
fig.suptitle("Trajectories of Infeasibility (CVRPBLTW 100)", fontsize=28, fontweight='bold', y=0.95)

# Dominate callout (orange, bottom-right; only visible after animation settles)
dominate_text = ax2.text(0.98, 0.28, "CaR and SGBS\ndominate NeuOpt*", transform=ax2.transAxes,
                         color='#D35400', fontweight='bold', fontsize=26, ha='right', va='bottom', zorder=10,
                         bbox=dict(facecolor='white', alpha=0.9, edgecolor='#D35400', boxstyle='round,pad=0.4'))
dominate_text.set_visible(False)  # show only when frame >= frames

ax1.tick_params(axis='both', which='major', labelsize=22)
ax2.tick_params(axis='both', which='major', labelsize=22)

# Bottom-axis display offset so CaR and SGBS (both 0% infeasibility) are visible as two lines
OFFSET_OURS_BOTTOM = 0.01   # CaR drawn at +0.2 on ax2
OFFSET_SGBS_BOTTOM = -0.01  # SGBS drawn at -0.2 on ax2

# 目标绿区设在底轴 (good gap region)
ax2.fill_between([0, 120], -2.5, 0.5, color='limegreen', alpha=0.15, zorder=1)
ax2.text(10, 0.6, "Desired Region:\nFast & Feasible", color='forestgreen', fontsize=22, fontweight='bold', zorder=2)

# ==========================================
# 3. 动态曲线与组件初始化
# ==========================================
color_ours = '#2ca02c'  # Green
color_neu = '#d62728'   # Red
color_sgbs = '#1f77b4'  # Blue

# SGBS
line1_sgbs, = ax1.plot([], [], color=color_sgbs, linestyle='--', linewidth=3, alpha=0.8, label='SGBS')
line2_sgbs, = ax2.plot([], [], color=color_sgbs, linestyle='--', linewidth=3, alpha=0.8)
point1_sgbs = ax1.scatter([], [], s=180, color=color_sgbs, marker='X', edgecolors='black', linewidths=1.5, zorder=6)
point2_sgbs = ax2.scatter([], [], s=180, color=color_sgbs, marker='X', edgecolors='black', linewidths=1.5, zorder=6)

# NeuOpt*
line1_neu, = ax1.plot([], [], color=color_neu, linestyle='--', linewidth=3, alpha=0.8, label='NeuOpt*')
line2_neu, = ax2.plot([], [], color=color_neu, linestyle='--', linewidth=3, alpha=0.8)
point1_neu = ax1.scatter([], [], s=180, color=color_neu, marker='s', edgecolors='black', linewidths=1.5, zorder=6)
point2_neu = ax2.scatter([], [], s=180, color=color_neu, marker='s', edgecolors='black', linewidths=1.5, zorder=6)

# Ours
line1_ours, = ax1.plot([], [], color=color_ours, linestyle='-', linewidth=3.5, alpha=1.0, label='CaR')
line2_ours, = ax2.plot([], [], color=color_ours, linestyle='-', linewidth=3.5, alpha=1.0)
point1_ours = ax1.scatter([], [], s=250, color=color_ours, marker='P', edgecolors='black', linewidths=2.0, zorder=7)
point2_ours = ax2.scatter([], [], s=250, color=color_ours, marker='P', edgecolors='black', linewidths=2.0, zorder=7)

ax1.legend(loc='upper right', frameon=True, fontsize=22, edgecolor='gray')
# Inference time: top-center of top panel (same as right plot / bltw_gap)
time_text = ax1.text(0.22, 0.85, "", transform=ax1.transAxes, fontsize=26, fontweight='bold', color='#333333',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))

# Speedup 特效绘制于 ax2
speedup_line, = ax2.plot([], [], color='black', linestyle=':', linewidth=3, zorder=8)
speedup_arrow = ax2.annotate("", xy=(0, 0), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="->", color='black', lw=3), zorder=9, alpha=0)
speedup_text = ax2.text(0, 0, "", color='#D35400', fontweight='bold', fontsize=24, ha='center', va='bottom', zorder=10, alpha=0,
                       bbox=dict(facecolor='white', alpha=0.9, edgecolor='#D35400', boxstyle='round,pad=0.4'))

# ==========================================
# 4. 动画与渲染逻辑
# ==========================================
frames = 80
pause_frames = 30
max_time_val = T_MAX


def init():
    line1_sgbs.set_data([], [])
    line2_sgbs.set_data([], [])
    point1_sgbs.set_offsets(np.empty((0, 2)))
    point2_sgbs.set_offsets(np.empty((0, 2)))

    line1_neu.set_data([], [])
    line2_neu.set_data([], [])
    point1_neu.set_offsets(np.empty((0, 2)))
    point2_neu.set_offsets(np.empty((0, 2)))

    line1_ours.set_data([], [])
    line2_ours.set_data([], [])
    point1_ours.set_offsets(np.empty((0, 2)))
    point2_ours.set_offsets(np.empty((0, 2)))

    time_text.set_text("")
    speedup_line.set_data([], [])
    speedup_arrow.set_alpha(0)
    speedup_text.set_alpha(0)
    dominate_text.set_visible(False)
    return (line1_sgbs, line2_sgbs, point1_sgbs, point2_sgbs,
            line1_neu, line2_neu, point1_neu, point2_neu,
            line1_ours, line2_ours, point1_ours, point2_ours,
            time_text, speedup_line, speedup_arrow, speedup_text, dominate_text)


def update(frame):
    if frame < frames:
        current_t = (frame / frames) * max_time_val
    else:
        current_t = max_time_val

    def get_data(times, gaps):
        if current_t < times[0]:
            return [], []
        valid = times <= current_t
        x_val, y_val = times[valid], gaps[valid]
        if current_t < times[-1] and current_t > times[0]:
            curr_y = np.interp(current_t, times, gaps)
            x_val = np.append(x_val, current_t)
            y_val = np.append(y_val, curr_y)
        return x_val, y_val

    # SGBS (on ax2 use offset so distinct from CaR when both at 0)
    x_c, y_c = get_data(times_C, gaps_C)
    if len(x_c) > 0:
        line1_sgbs.set_data(x_c, y_c)
        line2_sgbs.set_data(x_c, np.asarray(y_c) + OFFSET_SGBS_BOTTOM)
        point1_sgbs.set_offsets([[x_c[-1], y_c[-1]]])
        point2_sgbs.set_offsets([[x_c[-1], y_c[-1] + OFFSET_SGBS_BOTTOM]])

    # NeuOpt*
    x_b, y_b = get_data(times_B, gaps_B)
    if len(x_b) > 0:
        line1_neu.set_data(x_b, y_b)
        line2_neu.set_data(x_b, y_b)
        point1_neu.set_offsets([[x_b[-1], y_b[-1]]])
        point2_neu.set_offsets([[x_b[-1], y_b[-1]]])

    # Ours (on ax2 use offset so distinct from SGBS when both at 0)
    x_a, y_a = get_data(times_A, gaps_A)
    if len(x_a) > 0:
        line1_ours.set_data(x_a, y_a)
        line2_ours.set_data(x_a, np.asarray(y_a) + OFFSET_OURS_BOTTOM)
        point1_ours.set_offsets([[x_a[-1], y_a[-1]]])
        point2_ours.set_offsets([[x_a[-1], y_a[-1] + OFFSET_OURS_BOTTOM]])

    time_text.set_text(f"Inference Time: {current_t:4.0f}s")

    # Orange dominate box: only show after animation has settled (final pause)
    dominate_text.set_visible(frame >= frames)

    # Speedup in final pause
    if frame >= frames:
        target_gap = -1.5
        t_ours = np.interp(target_gap, gaps_A[::-1], times_A[::-1])
        t_neu = np.interp(target_gap, gaps_B[::-1], times_B[::-1])
        if t_ours < t_neu and t_ours > 0:
            speedup_factor = t_neu / t_ours
            speedup_arrow.set_alpha(1)
            speedup_arrow.xy = (t_ours + 5, target_gap)
            speedup_arrow.set_position((t_neu - 5, target_gap))
            speedup_line.set_data([t_ours, t_neu], [target_gap, target_gap])
            speedup_text.set_text(f"~{speedup_factor:.1f}x Speedup\nto reach same gap")
            speedup_text.set_position(((t_ours + t_neu) / 2, target_gap + 0.25))
            speedup_text.set_alpha(1)

    return (line1_sgbs, line2_sgbs, point1_sgbs, point2_sgbs,
            line1_neu, line2_neu, point1_neu, point2_neu,
            line1_ours, line2_ours, point1_ours, point2_ours,
            time_text, speedup_line, speedup_arrow, speedup_text, dominate_text)


ani = animation.FuncAnimation(fig, update, frames=frames + pause_frames, init_func=init, blit=False, interval=60)
plt.subplots_adjust(hspace=0.1)

ani.save('bltw_inf.gif', writer='pillow', fps=20)
# plt.show()
