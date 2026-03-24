import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# ==========================================
# 1. 数据准备 (Data) - 已清理不可见空格
# ==========================================
# Ours (Green)
gaps_A = np.array([3.21, 1.09, 0.10, 0.03, 0.03, 0.02, 0.01, 0.01, 0.01, 0.00, 0.00,
               0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
times_A = np.array([34.99, 40.42, 57.77, 86.87, 115.53, 143.76, 177.67, 204.98, 232.82, 264.57, 287.83,
              322.98, 348.48, 377.09, 407.77, 443.84, 463.82, 492.32, 519.44, 553.26, 586.25, 605.05])

# NeuOpt* (Red)
times_B = np.array([52.85, 70.4667, 88.0833, 105.7, 123.3167, 140.9333, 158.55, 176.1667,
  193.7833, 211.4,   229.0167, 246.6333, 264.25,   281.8667, 299.4833, 317.1,
  334.7167, 352.3333, 369.95,   387.5667, 405.1833, 422.8,   440.4167, 458.0333,
  475.65,   493.2667, 510.8833, 528.5,   546.1167, 563.7333, 581.35,   598.9667,
  616.5833])

gaps_B = np.array([99.83, 79.94, 28.45, 6.21, 1.86, 0.88, 0.59, 0.44, 0.37, 0.32,
   0.27,  0.22,  0.22,  0.18,  0.17, 0.16, 0.15, 0.13, 0.11, 0.10,
   0.10,  0.10,  0.10,  0.10,  0.10, 0.09, 0.08, 0.08, 0.07, 0.06,
   0.06,  0.06,  0.06])

# PIP (Blue)
gaps_C = np.array([6.96, 5.68, 5.26, 4.94, 4.92, 4.76, 4.57, 4.57, 
    4.54, 4.47, 4.40, 4.36, 4.28, 4.16, 4.04, 4.01])

times_C = np.array([32, 57.07, 85.68, 114.01, 142.02, 170.61, 199.02, 227, 
    255.9, 283.32, 312.16, 340.66, 425.43, 509.91, 571.36, 709.21])

# ==========================================
# 2. 画布与断轴设置
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1]})
ax1.set_facecolor('#ffffff')
ax2.set_facecolor('#ffffff')
fig.patch.set_facecolor('#ffffff')

T_MAX = 300  # <--- 修改点：最大时间限制为 300s

# 上下半区范围设定
ax1.set_xlim(0, T_MAX)
ax2.set_xlim(0, T_MAX)
ax1.set_ylim(4, 7)     
ax2.set_ylim(0, 0.25)  

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
fig.suptitle("Trajectories of Infeasibility (TSPTW 100)", fontsize=28, fontweight='bold', y=0.95)

ax1.tick_params(axis='both', which='major', labelsize=22)
ax2.tick_params(axis='both', which='major', labelsize=22)

# 目标绿区设在底轴
ax2.fill_between([0, 120], -0.01, 0.1, color='limegreen', alpha=0.15, zorder=1)
ax2.text(10, 0.12, "Desired Region:\nFast & Feasible", color='forestgreen', fontsize=22, fontweight='bold', zorder=2)

# ==========================================
# 3. 动态曲线与组件初始化
# ==========================================
color_ours = '#2ca02c' # Green
color_neu = '#d62728'  # Red
color_PIP = '#1f77b4' # Blue

# PIP
line1_PIP, = ax1.plot([], [], color=color_PIP, linestyle='--', linewidth=3, alpha=0.8, label='PIP')
line2_PIP, = ax2.plot([], [], color=color_PIP, linestyle='--', linewidth=3, alpha=0.8)
point1_PIP = ax1.scatter([], [], s=180, color=color_PIP, marker='X', edgecolors='black', linewidths=1.5, zorder=6)
point2_PIP = ax2.scatter([], [], s=180, color=color_PIP, marker='X', edgecolors='black', linewidths=1.5, zorder=6)

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
time_text = ax1.text(0.22, 0.85, "", transform=ax1.transAxes, fontsize=26, fontweight='bold', color='#333333',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))

# Speedup 特效绘制于 ax2
speedup_line, = ax2.plot([], [], color='black', linestyle=':', linewidth=3, zorder=8)
speedup_arrow = ax2.annotate("", xy=(0,0), xytext=(0,0), 
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
    line1_PIP.set_data([], [])
    line2_PIP.set_data([], [])
    point1_PIP.set_offsets(np.empty((0, 2)))
    point2_PIP.set_offsets(np.empty((0, 2)))
    
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
    return (line1_PIP, line2_PIP, point1_PIP, point2_PIP, 
            line1_neu, line2_neu, point1_neu, point2_neu, 
            line1_ours, line2_ours, point1_ours, point2_ours, 
            time_text, speedup_line, speedup_arrow, speedup_text)

def update(frame):
    if frame < frames:
        current_t = (frame / frames) * max_time_val
    else:
        current_t = max_time_val 
        
    def get_data(times, gaps):
        if current_t < times[0]: return [], []
        valid = times <= current_t
        x_val, y_val = times[valid], gaps[valid]
        if current_t < times[-1] and current_t > times[0]:
            curr_y = np.interp(current_t, times, gaps)
            x_val = np.append(x_val, current_t)
            y_val = np.append(y_val, curr_y)
        return x_val, y_val

    # PIP: 上下轴传同一份数据，Matplotlib 自动裁剪
    x_c, y_c = get_data(times_C, gaps_C)
    if len(x_c) > 0:
        line1_PIP.set_data(x_c, y_c)
        line2_PIP.set_data(x_c, y_c)
        point1_PIP.set_offsets([[x_c[-1], y_c[-1]]])
        point2_PIP.set_offsets([[x_c[-1], y_c[-1]]])

    # NeuOpt*
    x_b, y_b = get_data(times_B, gaps_B)
    if len(x_b) > 0:
        line1_neu.set_data(x_b, y_b)
        line2_neu.set_data(x_b, y_b)
        point1_neu.set_offsets([[x_b[-1], y_b[-1]]])
        point2_neu.set_offsets([[x_b[-1], y_b[-1]]])

    # Ours
    x_a, y_a = get_data(times_A, gaps_A)
    if len(x_a) > 0:
        line1_ours.set_data(x_a, y_a)
        line2_ours.set_data(x_a, y_a)
        point1_ours.set_offsets([[x_a[-1], y_a[-1]]])
        point2_ours.set_offsets([[x_a[-1], y_a[-1]]])
        
    time_text.set_text(f"Inference Time: {current_t:4.0f}s")

    # 【终极特效】Speedup calculation (适应 300s 窗口的逻辑)
    if frame >= frames:
        # 在 <= 300s 的区间内，NeuOpt* 在 t=299.48s 时达到最佳 Infeasibility 0.17%
        target_gap = 0.17
        t_neuopt = 299.4833 
        
        # Ours 在 40.42s (1.09%) 和 57.77s (0.10%) 之间跨过 0.17%
        # 线性插值计算时间点
        t_car = 40.42 + (target_gap - 1.09) / (0.10 - 1.09) * (57.77 - 40.42) # 约合 56.54s
        speedup_factor = t_neuopt / t_car # 约合 5.3 倍
        
        speedup_arrow.set_alpha(1)
        speedup_arrow.xy = (t_car + 2, target_gap)
        speedup_arrow.set_position((t_neuopt - 2, target_gap))
        
        speedup_line.set_data([t_car, t_neuopt], [target_gap, target_gap])
        
        speedup_text.set_text(f"~{speedup_factor:.1f}x Speedup\nto reach same feasibility")
        speedup_text.set_position(((t_car + t_neuopt)/2, target_gap + 0.03))
        speedup_text.set_alpha(1)
        
    return (line1_PIP, line2_PIP, point1_PIP, point2_PIP, 
            line1_neu, line2_neu, point1_neu, point2_neu, 
            line1_ours, line2_ours, point1_ours, point2_ours, 
            time_text, speedup_line, speedup_arrow, speedup_text)

# 渲染间隔设定
ani = animation.FuncAnimation(fig, update, frames=frames + pause_frames, init_func=init, blit=False, interval=60)
plt.subplots_adjust(hspace=0.1)

# 保存为 GIF 
ani.save('infeasibility_broken_axis_300s.gif', writer='pillow', fps=20)
# plt.show()
