import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def pareto_filter(times, gaps):
    order = np.argsort(times)
    t, g = times[order], gaps[order]
    keep_t, keep_g = [t[0]], [g[0]]
    best = g[0]
    for ti, gi in zip(t[1:], g[1:]):
        if gi < best:
            keep_t.append(ti)
            keep_g.append(gi)
            best = gi
    return np.array(keep_t), np.array(keep_g)


def normalized_pi(times, gaps, T):
    t = np.concatenate(([0.0], times))
    g = np.concatenate(([gaps[0]], gaps))
    area = 0.0
    for i in range(len(t) - 1):
        if t[i] >= T:
            break
        next_t = min(t[i + 1], T)
        area += (next_t - t[i]) * g[i]
        if t[i + 1] >= T:
            break
    if t[-1] < T:
        area += (T - t[-1]) * g[-1]
    return 100 * area / T


# ==========================================
# 1. 数据准备 (Data)
# ==========================================
# Ours (Green) - raw then Pareto-filtered
gaps_A_raw = np.array([
    -1.8351, -2.1491, -2.3239, -2.4480, -2.5210,
    -2.5919, -2.6374, -2.6813, -2.7196, -2.7475,
    -2.7744, -2.7941, -2.8209, -2.8429, -2.8640,
    -2.8845, -2.8971, -2.9085, -2.9256, -2.9391,
    -2.2590, -2.6378, -2.7887, -2.9253, -2.9804, -3.1373, -3.1041, -3.1275, -3.2203, -3.2923,
    -3.3003, -3.3908, -3.3290, -3.3648, -3.4284, -3.4449, -3.4801, -3.4637, -3.4779, -3.5082,
    -2.9226, -3.1971, -3.4030, -3.4983, -3.5763, -3.6375, -3.7336, -3.7638, -3.8220, -3.7835,
    -3.8400, -3.8591, -3.8929, -3.9490, -3.9293, -3.9722, -4.0035, -3.9958, -3.9928, -4.1102,
    -3.0538, -3.3428, -3.5035, -3.5858, -3.7424, -3.7919, -3.7952, -3.8574, -3.9424, -3.9750,
    -4.0390, -4.0183, -3.9930, -4.0852, -4.1661, -4.1105, -4.1626, -4.1983, -4.1260, -4.1169,
    -3.3347, -3.7023, -3.8629, -3.8943, -4.0305, -4.0891, -4.1934, -4.1944, -4.2205, -4.2242,
    -4.3131, -4.3094, -4.3317, -4.4231, -4.3734, -4.3892, -4.4642, -4.4965, -4.4749, -4.4729
])
times_A_raw = np.array([
    5.83, 10.31, 14.75, 19.18, 23.62,
    28.07, 32.51, 36.89, 41.55, 45.89,
    50.39, 55.34, 59.76, 64.36, 68.69,
    73.38, 78.22, 82.47, 86.93, 92.39,
    11.77, 20.59, 29.50, 38.82, 47.88, 56.68, 65.96, 76.33, 83.96, 93.23,
    102.02, 111.17, 120.33, 129.04, 137.91, 149.16, 156.01, 167.80, 174.18, 185.83,
    23.26, 41.70, 59.60, 76.46, 95.12, 113.37, 130.25, 148.35, 166.49, 184.17,
    203.88, 219.45, 238.36, 257.60, 276.38, 293.68, 309.61, 326.37, 345.12, 362.11,
    28.32, 51.07, 72.59, 95.51, 117.50, 139.51, 161.47, 185.65, 208.59, 230.04,
    252.43, 273.96, 296.68, 318.57, 340.50, 366.44, 387.57, 407.59, 432.70, 457.05,
    45.89, 82.23, 117.71, 153.69, 188.59, 225.76, 259.87, 296.45, 331.97, 369.88,
    402.20, 437.41, 480.82, 511.85, 553.04, 581.15, 619.04, 655.13, 692.06, 723.08
])
times_A, gaps_A = pareto_filter(times_A_raw, gaps_A_raw)

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
    47.989325, 30.684825, 20.190002, 15.129232, 12.049896, 10.373374, 9.018443, 8.192071,
    7.417753, 6.731536, 5.942963, 5.561768, 5.126514, 4.782882, 4.566742, 4.319585,
    4.077605, 3.818421, 3.612182, 3.390018, 3.181861, 3.043384, 2.819611, 2.539998,
    2.345696, 2.258464, 2.122440, 2.063789, 1.944494, 1.790706, 1.663999, 1.585467,
    1.482020, 1.422641, 1.321130, 1.225501, 1.102794, 0.972198, 0.907986, 0.874785,
    0.822525, 0.719113, 0.597989, 0.551024, 0.494838, 0.453130, 0.336313, 0.289389,
    0.228399, 0.152158, 0.117658, 0.065045, 0.004933, -0.047711, -0.089072, -0.109055,
    -0.160633, -0.193366, -0.233315, -0.288556, -0.363023, -0.406865, -0.458420, -0.484315,
    -0.492444, -0.547420, -0.580006, -0.595682, -0.616492, -0.644007, -0.691756, -0.728591,
    -0.748315, -0.796211, -0.813771, -0.845349, -0.878295, -0.937360, -0.972449, -0.997074,
    -1.020026, -1.054222, -1.086244, -1.121180, -1.170239, -1.189438, -1.216632, -1.237751,
    -1.256970, -1.293470, -1.337502, -1.391508, -1.413543, -1.432170, -1.460843, -1.474711,
    -1.502498, -1.528067, -1.520032, -1.541339
])[1::2]

# SGBS (Blue)
times_C = np.array([132.0, 256.8, 383.4, 509.4, 634.8, 762.6, 891.6, 1017.0])
gaps_C = np.array([2.616, 2.198, 1.970, 1.821, 1.711, 1.614, 1.543, 1.472])

# ==========================================
# 2. Figure and single axes
# ==========================================
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_facecolor('#ffffff')
fig.patch.set_facecolor('#ffffff')

T_MAX = 300.0  # x-axis: 0–300 s
ax.set_xlim(0, T_MAX)
ax.set_ylim(-5.0, 5)

ax.grid(True, which="major", ls="--", alpha=0.5, color="#e0e0e0")
ax.set_xlabel("Inference Time (seconds)", fontsize=26, fontweight='bold', color="#333333")
ax.set_ylabel("Optimality Gap to OR-Tools (%)", fontsize=26, fontweight='bold', color="#333333", labelpad=12)
fig.suptitle("Trajectories of Optimality Gap (CVRPBLTW 100)", fontsize=28, fontweight='bold', y=0.95)
ax.tick_params(axis='both', which='major', labelsize=22)

# Desired region (low gap)
ax.fill_between([0, 120], -5, 0, color='limegreen', alpha=0.15, zorder=1)
ax.text(10, 0.6, "Desired Region:\nFast & Optimal", color='forestgreen', fontsize=22, fontweight='bold', zorder=2)

# ==========================================
# 3. 动态曲线与组件初始化
# ==========================================
color_ours = '#2ca02c'  # Green
color_neu = '#d62728'   # Red
color_sgbs = '#1f77b4'  # Blue

# SGBS
line_sgbs, = ax.plot([], [], color=color_sgbs, linestyle='--', linewidth=3, alpha=0.8, label='SGBS')
point_sgbs = ax.scatter([], [], s=180, color=color_sgbs, marker='X', edgecolors='black', linewidths=1.5, zorder=6)

# NeuOpt*
line_neu, = ax.plot([], [], color=color_neu, linestyle='--', linewidth=3, alpha=0.8, label='NeuOpt*')
point_neu = ax.scatter([], [], s=180, color=color_neu, marker='s', edgecolors='black', linewidths=1.5, zorder=6)

# Ours
line_ours, = ax.plot([], [], color=color_ours, linestyle='-', linewidth=3.5, alpha=1.0, label='CaR')
point_ours = ax.scatter([], [], s=250, color=color_ours, marker='P', edgecolors='black', linewidths=2.0, zorder=7)

ax.legend(loc='upper right', frameon=True, fontsize=22, edgecolor='gray')
time_text = ax.text(0.22, 0.90, "", transform=ax.transAxes, fontsize=26, fontweight='bold', color='#333333',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))

# Speedup effect
speedup_line, = ax.plot([], [], color='black', linestyle=':', linewidth=3, zorder=8)
speedup_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="->", color='black', lw=3), zorder=9, alpha=0)
speedup_text = ax.text(0, 0, "", color='#D35400', fontweight='bold', fontsize=24, ha='center', va='bottom', zorder=10, alpha=0,
                       bbox=dict(facecolor='white', alpha=0.9, edgecolor='#D35400', boxstyle='round,pad=0.4'))

# CaR dominates callout (same style as speedup box, shown only in final pause)
dominate_text = ax.text(0.98, 0.20, "CaR dominates\nother algorithms", transform=ax.transAxes,
                        color='#D35400', fontweight='bold', fontsize=26, ha='right', va='bottom', zorder=10,
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='#D35400', boxstyle='round,pad=0.4'))
dominate_text.set_visible(False)

# ==========================================
# 4. 动画与渲染逻辑
# ==========================================
frames = 80
pause_frames = 30
max_time_val = T_MAX


def init():
    line_sgbs.set_data([], [])
    point_sgbs.set_offsets(np.empty((0, 2)))

    line_neu.set_data([], [])
    point_neu.set_offsets(np.empty((0, 2)))

    line_ours.set_data([], [])
    point_ours.set_offsets(np.empty((0, 2)))

    time_text.set_text("")
    speedup_line.set_data([], [])
    speedup_arrow.set_alpha(0)
    speedup_text.set_alpha(0)
    dominate_text.set_visible(False)
    return (line_sgbs, point_sgbs, line_neu, point_neu, line_ours, point_ours,
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

    # SGBS
    x_c, y_c = get_data(times_C, gaps_C)
    if len(x_c) > 0:
        line_sgbs.set_data(x_c, y_c)
        point_sgbs.set_offsets([[x_c[-1], y_c[-1]]])

    # NeuOpt*
    x_b, y_b = get_data(times_B, gaps_B)
    if len(x_b) > 0:
        line_neu.set_data(x_b, y_b)
        point_neu.set_offsets([[x_b[-1], y_b[-1]]])

    # Ours
    x_a, y_a = get_data(times_A, gaps_A)
    if len(x_a) > 0:
        line_ours.set_data(x_a, y_a)
        point_ours.set_offsets([[x_a[-1], y_a[-1]]])

    time_text.set_text(f"Inference Time: {current_t:4.0f}s")

    # Speedup + "CaR dominates" callouts: only visible in final pause (like tw_gap.py)
    if frame >= frames:
        dominate_text.set_visible(True)
        target_gap = -1.5
        t_ours = np.interp(target_gap, gaps_A[::-1], times_A[::-1])
        t_neu = np.interp(target_gap, gaps_B[::-1], times_B[::-1])
        if t_ours < t_neu and t_ours > 0 and t_neu <= T_MAX:
            speedup_factor = t_neu / t_ours
            speedup_arrow.set_alpha(1)
            speedup_arrow.xy = (t_ours + 5, target_gap)
            speedup_arrow.set_position((t_neu - 5, target_gap))
            speedup_line.set_data([t_ours, t_neu], [target_gap, target_gap])
            speedup_text.set_text(f"~{speedup_factor:.1f}x Speedup\nto reach same gap")
            speedup_text.set_position(((t_ours + t_neu) / 2, target_gap + 0.25))
            speedup_text.set_alpha(1)
        else:
            speedup_arrow.set_alpha(0)
            speedup_line.set_data([], [])
            speedup_text.set_alpha(0)
    else:
        dominate_text.set_visible(False)
        speedup_arrow.set_alpha(0)
        speedup_line.set_data([], [])
        speedup_text.set_alpha(0)

    return (line_sgbs, point_sgbs, line_neu, point_neu, line_ours, point_ours,
            time_text, speedup_line, speedup_arrow, speedup_text, dominate_text)


ani = animation.FuncAnimation(fig, update, frames=frames + pause_frames, init_func=init, blit=False, interval=60)

ani.save('bltw_gap.gif', writer='pillow', fps=20)
# plt.show()
