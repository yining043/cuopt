import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# ==========================================
# 1. 数据准备 (已修复隐藏空格和长度对齐问题)
# ==========================================
# PIP
gaps_C = np.array([1.223, 1.1376, 1.1061, 1.0874, 1.0673, 1.0603, 1.0478, 1.0383, 1.0329, 1.0256])
times_C = np.array([32, 57.07, 85.68, 114.01, 142.02, 170.61, 199.02, 227, 255.9, 283.32])

# NEUOPT
gaps_A = np.array([4.9128, 3.271, 1.9185, 1.2267, 0.8888, 0.7312, 0.64, 0.5741, 0.5267, 0.4894, 0.4582, 0.4327, 0.4103, 0.3933, 0.3776])
times_A = np.array([52.85, 70.4667, 88.0833, 105.7, 123.3167, 140.9333, 158.55, 176.1667, 193.7833, 211.4, 229.0167, 246.6333, 264.25, 281.8667, 299.4833])

# CAR (Ours)
gaps_raw = np.array([0.6111, 0.3112, 0.1907, 0.1588, 0.1455, 0.1369, 0.1335, 0.1288, 0.1260, 0.1260])
times_raw = np.array([40.42, 58.27, 86.94, 115.57, 145.30, 176.87, 204.94, 233.50, 264.65, 288.14])

# ==========================================
# 2. 画布与基础元素设置 (match plot.py style)
# ==========================================
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_facecolor('#ffffff')
fig.patch.set_facecolor('#ffffff')

ax.set_xlim(0, 300)
ax.set_ylim(0.1, 1.3)
ax.grid(True, which="major", ls="--", alpha=0.5, color="#e0e0e0")

ax.set_xlabel("Inference Time (seconds)", fontsize=26, fontweight='bold', color="#333333")
ax.set_ylabel("Optimality Gap to LKH-3 (%)", fontsize=26, fontweight='bold', color="#333333", labelpad=12)
fig.suptitle("Trajectories of Optimality Gap (TSPTW 100)", fontsize=28, fontweight='bold', y=0.95)
ax.tick_params(axis='both', which='major', labelsize=22)

# Desired region (same style as plot.py)
ax.fill_between([0, 120], -0.1, 0.3, color='limegreen', alpha=0.15, zorder=1)
ax.text(10, 0.35, "Desired Region:\nFast & Optimal", color='forestgreen', fontsize=22, fontweight='bold', zorder=2)

# ==========================================
# 3. 动态曲线与组件初始化 (plot.py style: linewidth, scatter size, colors)
# ==========================================
color_pip = '#1f77b4'   # Blue (baseline)
color_neu = '#d62728'   # Red (baseline)
color_car = '#2ca02c'   # Green (Ours)

line_pip, = ax.plot([], [], color=color_pip, linestyle='--', linewidth=3, alpha=0.8, label='PIP', zorder=3)
point_pip = ax.scatter([], [], s=180, color=color_pip, marker='X', edgecolors='black', linewidths=1.5, zorder=6)

line_neu, = ax.plot([], [], color=color_neu, linestyle='--', linewidth=3, alpha=0.8, label='NeuOpt*', zorder=3)
point_neu = ax.scatter([], [], s=180, color=color_neu, marker='s', edgecolors='black', linewidths=1.5, zorder=6)

line_car, = ax.plot([], [], color=color_car, linestyle='-', linewidth=3.5, alpha=1.0, label='CaR', zorder=4)
point_car = ax.scatter([], [], s=250, color=color_car, marker='P', edgecolors='black', linewidths=2.0, zorder=7)

ax.legend(loc='upper right', frameon=True, fontsize=22, edgecolor='gray')

time_text = ax.text(0.22, 0.92, "", transform=ax.transAxes, fontsize=26, fontweight='bold', color='#333333',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))

# Speedup effect (same style as plot.py)
speedup_line, = ax.plot([], [], color='black', linestyle=':', linewidth=3, zorder=8)
speedup_arrow = ax.annotate("", xy=(0,0), xytext=(0,0),
                            arrowprops=dict(arrowstyle="->", color='black', lw=3), zorder=9, alpha=0)
speedup_text = ax.text(0, 0, "", color='#D35400', fontweight='bold', fontsize=24, ha='center', va='bottom', zorder=10, alpha=0,
                       bbox=dict(facecolor='white', alpha=0.9, edgecolor='#D35400', boxstyle='round,pad=0.4'))

# ==========================================
# 4. 动画与渲染逻辑
# ==========================================
frames = 80 # 动画主过程的帧数
pause_frames = 30 # 动画结束后的定格帧数 (用于展示 Speedup)
max_time = 300 # X轴最大时间

def init():
    line_pip.set_data([], [])
    point_pip.set_offsets(np.empty((0, 2)))
    line_neu.set_data([], [])
    point_neu.set_offsets(np.empty((0, 2)))
    line_car.set_data([], [])
    point_car.set_offsets(np.empty((0, 2)))
    time_text.set_text("")
    speedup_line.set_data([], [])
    speedup_arrow.set_alpha(0)
    speedup_text.set_alpha(0)
    return line_pip, point_pip, line_neu, point_neu, line_car, point_car, time_text, speedup_line, speedup_arrow, speedup_text

def update(frame):
    # 如果还在主过程内，正常推移时间；如果到了定格帧，保持在最大时间
    if frame < frames:
        current_t = (frame / frames) * max_time
    else:
        current_t = max_time 
        
    def get_data(times, gaps):
        if current_t < times[0]: return [], []
        valid = times <= current_t
        x_val, y_val = times[valid], gaps[valid]
        if current_t < times[-1] and current_t > times[0]:
            curr_y = np.interp(current_t, times, gaps)
            x_val = np.append(x_val, current_t)
            y_val = np.append(y_val, curr_y)
        return x_val, y_val

    # 更新三条曲线
    x_c, y_c = get_data(times_C, gaps_C)
    if len(x_c) > 0:
        line_pip.set_data(x_c, y_c)
        point_pip.set_offsets([[x_c[-1], y_c[-1]]])

    x_a, y_a = get_data(times_A, gaps_A)
    if len(x_a) > 0:
        line_neu.set_data(x_a, y_a)
        point_neu.set_offsets([[x_a[-1], y_a[-1]]])

    x_raw, y_raw = get_data(times_raw, gaps_raw)
    if len(x_raw) > 0:
        line_car.set_data(x_raw, y_raw)
        point_car.set_offsets([[x_raw[-1], y_raw[-1]]])
        
    time_text.set_text(f"Inference Time: {current_t:4.0f}s")

    # 【核心特效】动画进入定格阶段，绘制横跨图表的 Speedup 线
    if frame >= frames:
        # NeuOpt 在这300s内的最佳成绩是最后时刻 0.3776
        target_gap = gaps_A[-1] 
        t_neuopt = times_A[-1]  
        
        # 线性插值计算 CaR 达到同等 Gap 的时间 (约 54.3s)
        t_car = 40.42 + (target_gap - 0.6111) / (0.3112 - 0.6111) * (58.27 - 40.42)
        speedup_factor = t_neuopt / t_car # ~5.51x
        
        # 激活箭头与文本，设置坐标
        speedup_arrow.set_alpha(1)
        speedup_arrow.xy = (t_car + 2, target_gap)
        speedup_arrow.set_position((t_neuopt - 2, target_gap))
        
        speedup_line.set_data([t_car, t_neuopt], [target_gap, target_gap])
        
        speedup_text.set_text(f"~{speedup_factor:.1f}x Speedup\nto reach same gap")
        speedup_text.set_position(((t_car + t_neuopt)/2 - 25, target_gap + 0.03))
        speedup_text.set_alpha(1)
        
    return line_pip, point_pip, line_neu, point_neu, line_car, point_car, time_text, speedup_line, speedup_arrow, speedup_text

ani = animation.FuncAnimation(fig, update, frames=frames + pause_frames, init_func=init, blit=False, interval=60)
plt.subplots_adjust(top=0.88)

# 保存为 GIF 或 MP4
ani.save('racing_trajectories_speedup.gif', writer='pillow', fps=20)
# 如果需要在 Notebook 里实时预览，取消注释下一行：
plt.show()
