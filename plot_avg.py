# -*- coding: utf-8 -*-
"""
Plot Average 'Best So Far' Comparison (Multi-Group Support).
Automatically groups files by filename prefix (e.g., 'origin', 'model', 'oracle').
"""
import sys
import os
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import itertools

# 匹配 "cost before: X, cost after: Y"
COST_PAIR_RE = re.compile(
    r"cost before:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*"
    r"cost after:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)

# 匹配 "Total time used: T ms, offset: O ms"
TIME_RE = re.compile(
    r"Total time used:\s*(\d+)\s*ms\s*,\s*offset:\s*(\d+)\s*ms",
    re.IGNORECASE,
)

def parse_points(text: str):
    pts = []
    best_so_far = float('inf')
    
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln: continue
        m_cost = COST_PAIR_RE.search(ln)
        if m_cost:
            after_cost = float(m_cost.group(2))
            best_so_far = min(best_so_far, after_cost)
            pts.append({
                'after': after_cost, 
                'best_so_far': best_so_far,
                'time': None, 'offset': None
            })
            continue
        m_time = TIME_RE.search(ln)
        if m_time and pts:
            if pts[-1]['time'] is None:
                pts[-1]['time'] = float(m_time.group(1))
                pts[-1]['offset'] = float(m_time.group(2))
    return pts

def get_xy_data(pts, break_mode, max_segments, xshift, data_key, use_percentage=False):
    xs, ys = [], []
    seg_count = 0
    prev_key = None
    i_iter = 0

    def key_of(p):
        if break_mode == "time": return p['time']
        elif break_mode == "offset": return p['offset']
        return None

    # 计算总迭代数用于百分比
    total_iterations = 0
    if use_percentage:
        temp_seg = 0
        temp_prev = None
        temp_iter = 0
        for p in pts:
            k = key_of(p)
            new_seg = False
            if break_mode == "none":
                new_seg = (temp_seg == 0 and temp_iter == 0)
            else:
                if temp_prev is None: new_seg = True
                elif k is not None and temp_prev is not None and k < temp_prev: new_seg = True
            
            if new_seg:
                temp_seg += 1
                if max_segments is not None and max_segments > 0 and temp_seg > max_segments: break
            temp_prev = k if k is not None else temp_prev
            temp_iter += 1
        total_iterations = temp_iter

    # 实际提取
    for p in pts:
        k = key_of(p)
        new_segment = False
        if break_mode == "none":
            new_segment = (seg_count == 0 and i_iter == 0)
        else:
            if prev_key is None: new_segment = True
            elif k is not None and prev_key is not None and k < prev_key: new_segment = True

        if new_segment:
            seg_count += 1
            if max_segments is not None and max_segments > 0 and seg_count > max_segments: break
        
        prev_key = k if k is not None else prev_key
        i_iter += 1
        
        if use_percentage and total_iterations > 0:
            x_val = (float(i_iter) / total_iterations) * 100.0
        else:
            x_val = float(i_iter) + xshift
            
        xs.append(x_val)
        ys.append(p[data_key])

    return xs, ys

def extract_group_name(filename):
    """
    智能推断组名：
    origin_now.txt -> origin
    model2.txt -> model
    oracle_v1.txt -> oracle
    """
    base = os.path.basename(filename)
    # 提取第一个由字母组成的单词作为组名
    match = re.match(r"([a-zA-Z_]+)", base)
    if match:
        return match.group(1).capitalize()
    return "Other"

def plot_averaged_group(ax, series_list, label, color, use_percentage):
    """
    绘制一组数据的均值和标准差
    """
    if not series_list: return
    
    # 根据是否使用百分比决定 X 轴范围
    if use_percentage:
        common_x = np.linspace(0, 100, 1000)
    else:
        # 提取当前组所有系列中的最大 X 值（最大迭代次数）
        max_x = max([xs[-1] for xs, ys in series_list if xs])
        common_x = np.linspace(0, max_x, 1000)
        
    interpolated_ys = []

    for xs, ys in series_list:
        if not xs: continue
        # 线性插值，对于超出范围的会自动保持最后的值
        y_interp = np.interp(common_x, xs, ys)
        interpolated_ys.append(y_interp)
    
    if not interpolated_ys: return

    stack = np.vstack(interpolated_ys)
    mean_y = np.mean(stack, axis=0)
    std_y = np.std(stack, axis=0)

    # 绘制
    ax.plot(common_x, mean_y, label=label, color=color, linewidth=2.5)
    ax.fill_between(common_x, mean_y - std_y, mean_y + std_y, color=color, alpha=0.15)

def main():
    ap = argparse.ArgumentParser(description="Plot Average Best-So-Far Cost (Auto-Grouping).")
    ap.add_argument("files", nargs="*", help="Input log files")
    ap.add_argument("--out", "-o", type=str, default="data.png")
    ap.add_argument("--out-best", type=str, default="data_best.png")
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--ymin", type=float, default=0.0)
    ap.add_argument("--segments", "--seg", "-s", type=int, default=0)
    ap.add_argument("--xshift", type=float, default=0.0)
    ap.add_argument("--break", dest="break_mode", default="time")
    # 新增开关：是否使用绝对迭代次数
    ap.add_argument("--abs", action="store_true", help="Use absolute iteration count instead of percentage for X axis")
    
    # 兼容性参数 (不使用但防止报错)
    ap.add_argument("--labels", type=str, default=None)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--out-pct", type=str, default="ignore.png")
    ap.add_argument("--out-best-pct", type=str, default="ignore.png")

    args = ap.parse_args()

    if not args.files:
        print("Need input files")
        sys.exit(1)

    # 决定是否使用百分比
    use_pct = not args.abs

    # 1. 解析数据并按组分类
    # 结构: groups = { 'Origin': [(xs, ys), ...], 'Model': [...] }
    groups = {}
    
    # 记录出现顺序，保证图例顺序一致
    group_order = [] 

    for fp in args.files:
        try:
            raw = open(fp, "r", encoding="utf-8", errors="ignore").read()
        except:
            continue
            
        pts = parse_points(raw)
        if not pts: continue
        
        xs, ys = get_xy_data(pts, args.break_mode, args.segments, args.xshift, 'best_so_far', use_percentage=use_pct)
        
        # 智能获取组名
        g_name = extract_group_name(fp)
        
        if g_name not in groups:
            groups[g_name] = []
            group_order.append(g_name)
        
        groups[g_name].append((xs, ys))

    print(f"Detected Groups: {group_order}")

    # 2. 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 使用 Matplotlib 的颜色循环 (蓝, 橙, 绿, 红, 紫...)
    colors = itertools.cycle(plt.cm.tab10.colors)

    for g_name in group_order:
        series_data = groups[g_name]
        color = next(colors)
        # 如果是 Origin 组，强制用蓝色；Model 用红色；Oracle 用绿色（可选项）
        if "Origin" in g_name: color = "#1f77b4" # Blue
        elif "Model" in g_name: color = "#d62728"  # Red
        elif "Oracle" in g_name: color = "#2ca02c" # Green
        
        plot_averaged_group(ax, series_data, g_name, color, use_pct)

    # 根据开关设置 X 轴标签
    if use_pct:
        ax.set_xlabel("Iteration Percentage (%)")
    else:
        ax.set_xlabel("Iteration Count")
        
    ax.set_ylabel("Best So Far Cost (Average)")
    ax.set_title("Average Performance Comparison")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    if args.ymax is not None:
        ax.set_ylim(args.ymin, args.ymax)

    out_path = args.out
    if not out_path.endswith(".png"): out_path += ".png"
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    print(f"Saved Average Figure to: {out_path}")

if __name__ == "__main__":
    main()
