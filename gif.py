#!/usr/bin/env python3
"""
Visualize N Groups of Consecutive Iterations (Direct to GIF).
Modified for Single Trail (Trail #0) with Expanded Columns.
Columns:
1. Route (Deepened Color)
2. Candidates
3. Search Scope (nodes_to_search)
4. Anchors (VRP, SLIDING, TWO_OPT, Overlaps in Black)
5. h_active_nodes_impacted
6. Sol Changed Nodes
7. Sol Intersection Nodes
"""

import re
import sys
import os
import random
import argparse
import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

# Try to import the data loader
try:
    from load_nco_data import load_raw_data
except ImportError:
    print("Warning: load_nco_data not found. Make sure it's in the same directory.")
    load_raw_data = None

# ==========================================
# Parsing Logic
# ==========================================

def parse_solution(line):
    """Parses routes from format: [0,1,2] [0,3,4] ..."""
    routes = []
    for match in re.finditer(r'\[([\d\s,]+)\]', line):
        try:
            content = match.group(1)
            route = [int(x) for x in content.replace(',', ' ').split() if x.strip()]
            if route:
                routes.append(route)
        except ValueError:
            continue
    return routes

def parse_nodes_list(line):
    """Parses lists like: candidates: [1, 2, 3] or nodes: 1, 2, 3"""
    if ':' in line:
        content = line.split(':', 1)[1]
    else:
        content = line
    nums = re.findall(r'\d+', content)
    return set([int(x) for x in nums])

def read_log_file(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    data_store = {}
    current_search = 0
    current_iter = -1

    for line in lines:
        line = line.strip()
        if not line: continue

        # --- Hierarchy Identification ---
        search_match = re.search(r'\[search #(\d+)\]', line)
        if search_match:
            current_search = int(search_match.group(1))
            current_iter = -1
            continue

        iter_match = re.search(r'\[iter #(\d+)\]', line)
        if iter_match:
            current_iter = int(iter_match.group(1))
            continue
        
        # Ensure we have a storage entry
        if current_iter == -1: continue
        
        key = (current_search, current_iter)
        if key not in data_store:
            data_store[key] = {}
        
        # Initialize Trail #0 if not exists
        if 0 not in data_store[key]:
            data_store[key][0] = {
                'routes_before': [], 'routes_after': [],
                'candidates': set(), 'search': set(),
                'anchor_vrp': set(), 'anchor_sliding': set(), 'anchor_two_opt': set(), # NEW ANCHOR FIELDS
                'h_active': set(), # REPLACED 'changed'
                'intersection': set(), # kept for safety, though maybe unused now
                'sol_changed': set(), 'sol_intersection': set(), 
                'score': -float('inf')
            }
        
        ref = data_store[key][0]

        # --- Data Parsing ---
        
        # Capture Trail specific info (usually Trail #0)
        if '[trail #0]' in line:
            if 'nodes_to_search:' in line:
                ref['search'] = parse_nodes_list(line)
            elif 'candidates:' in line:
                ref['candidates'] = parse_nodes_list(line)
            elif '[before_search]' in line and 'sol:' in line:
                ref['routes_before'] = parse_solution(line)

        # Capture Anchor Info
        if '[anchor]' in line:
            if 'types=VRP:' in line:
                ref['anchor_vrp'] = parse_nodes_list(line)
            elif 'types=SLIDING:' in line:
                ref['anchor_sliding'] = parse_nodes_list(line)
            elif 'types=TWO_OPT:' in line:
                ref['anchor_two_opt'] = parse_nodes_list(line)

        # Capture Global/After Search info
        if '[after_search]' in line:
            if 'sol_changed_nodes:' in line:
                ref['sol_changed'] = parse_nodes_list(line)
            elif 'label_nodes:' in line:
                ref['sol_intersection'] = parse_nodes_list(line)
            elif 'h_active_nodes_impacted:' in line:
                ref['h_active'] = parse_nodes_list(line)
            elif 'sol:' in line:
                ref['routes_after'] = parse_solution(line)

        # Score parsing
        if 'score:' in line and 'best score:' in line:
            score_match = re.search(r'score:\s*([\d\.-]+)', line)
            if score_match:
                try: ref['score'] = float(score_match.group(1))
                except ValueError: pass

    return data_store

# ==========================================
# Visualization Helpers
# ==========================================

def get_safe_index(node_id, num_coords):
    if node_id >= num_coords: return 0
    return node_id

def plot_single_graph(ax, routes, coords, title, hl_nodes=None, hl_color='red', hl_size=40, is_main_sol=False):
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')
    
    num_coords = len(coords)

    # Plot Routes (Background)
    for route in routes:
        if len(route) < 2: continue
        
        is_relevant = False
        if hl_nodes and any(n in hl_nodes for n in route):
            is_relevant = True
            
        # Apply specific deepening logic for the first plot
        if is_main_sol:
            alpha = 0.8
            color = 'black'  # Deepen sol color
            lw = 1.5
        else:
            alpha = 0.6 if is_relevant else 0.15
            color = 'gray'
            lw = 1.2 if is_relevant else 0.8
        
        r_idx = [get_safe_index(n, num_coords) for n in route]
        ax.plot(coords[r_idx, 0], coords[r_idx, 1], c=color, alpha=alpha, lw=lw, zorder=1)

    # Plot All Nodes (faint background)
    ax.scatter(coords[:, 0], coords[:, 1], c='#e0e0e0', s=10, zorder=0)
    # Depot
    ax.scatter(coords[0, 0], coords[0, 1], c='black', marker='s', s=50, zorder=2)

    # Plot Highlights
    if hl_nodes:
        valid_hl = [n for n in hl_nodes if n < num_coords]
        if valid_hl:
            h_coords = coords[valid_hl]
            ax.scatter(h_coords[:, 0], h_coords[:, 1], c=hl_color, s=hl_size, zorder=3, edgecolors='white', linewidth=0.5)

def plot_anchor_graph(ax, routes, coords, title, vrp_nodes, sliding_nodes, two_opt_nodes):
    """Specialized function to plot the 3 anchor types with overlap handling."""
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')
    
    num_coords = len(coords)

    # Plot Background Routes
    for route in routes:
        if len(route) < 2: continue
        r_idx = [get_safe_index(n, num_coords) for n in route]
        ax.plot(coords[r_idx, 0], coords[r_idx, 1], c='gray', alpha=0.15, lw=0.8, zorder=1)

    # Plot All Nodes
    ax.scatter(coords[:, 0], coords[:, 1], c='#e0e0e0', s=10, zorder=0)
    # Depot
    ax.scatter(coords[0, 0], coords[0, 1], c='black', marker='s', s=50, zorder=2)

    # Calculate overlaps (Intersection of any two sets)
    overlap = (vrp_nodes & sliding_nodes) | (vrp_nodes & two_opt_nodes) | (sliding_nodes & two_opt_nodes)
    
    # Calculate pure sets
    vrp_only = vrp_nodes - overlap
    sliding_only = sliding_nodes - overlap
    two_opt_only = two_opt_nodes - overlap

    # Define color mappings (3 colors + black for overlap)
    plot_sets = {
        '#1f77b4': vrp_only,      # Blue for VRP
        '#ff7f0e': sliding_only,  # Orange for SLIDING
        '#2ca02c': two_opt_only,  # Green for TWO_OPT
        'black': overlap          # Black for overlap
    }

    # Plot each set
    for color, nodes in plot_sets.items():
        if not nodes: continue
        valid = [n for n in nodes if n < num_coords]
        if valid:
            h_coords = coords[valid]
            ax.scatter(h_coords[:, 0], h_coords[:, 1], c=color, s=40, zorder=3, edgecolors='white', linewidth=0.5)

# ==========================================
# Core Logic: GIF Generation
# ==========================================

def create_multi_group_visualization(data_store, coordinates, output_dir, num_groups=10, seq_len=3, duration=800):
    if not data_store: return
    
    os.makedirs(output_dir, exist_ok=True)
    
    if hasattr(coordinates, 'numpy'): coords = coordinates.numpy()
    elif hasattr(coordinates, 'cpu'): coords = coordinates.cpu().numpy()
    else: coords = np.array(coordinates)

    all_keys = sorted(data_store.keys(), key=lambda x: (x[0], x[1]))
    if not all_keys: return

    # 1. 提取所有不重复的 search_id
    unique_search_ids = list(set([k[0] for k in all_keys]))
    
    # 2. 随机抽取一个 search_id
    selected_search_id = random.choice(unique_search_ids)
    
    # 3. 过滤出属于这个 search_id 的所有 keys，此时它们已经是按 iter 排序好的
    target_keys = [k for k in all_keys if k[0] == selected_search_id]
    
    print(f"Randomly selected Search #{selected_search_id}. Total iterations: {len(target_keys)}")
    if not target_keys: return

    # 4. 找到 iter = 0 的准确位置（安全起见，防止日志里缺失前面的 iter）
    start_idx = 0
    for i, k in enumerate(target_keys):
        if k[1] == 0:  # k[1] 即为 iter_id
            start_idx = i
            break
            
    # 5. 从 iter=0 开始往后截取 seq_len 的长度
    group_keys = target_keys[start_idx : start_idx + seq_len]
    
    if not group_keys:
        print("Error: No valid iterations found.")
        return

    print(f"Drawing Search #{selected_search_id} from Iter #0, length: {len(group_keys)}...")
    frames = []

    # 6. 直接开始遍历截取好的 key 生成画面（去掉了原来外层的随机分组循环）
    for seq_i, (search_id, iter_id) in enumerate(group_keys):
        if 0 not in data_store[(search_id, iter_id)]: continue
        
        data = data_store[(search_id, iter_id)][0]
        
        fig, axes = plt.subplots(1, 7, figsize=(35, 5), constrained_layout=True)
        if not isinstance(axes, np.ndarray): axes = [axes]
        
        r_before = data['routes_before']
        r_after  = data.get('routes_after', r_before)
        score_val = data['score']

        # ------------------ 画图逻辑保持不变 ------------------
        plot_single_graph(axes[0], r_before, coords, 
                          f"Trail #0 | Score: {score_val:.2f}", hl_nodes=None, is_main_sol=True)
        plot_single_graph(axes[1], r_before, coords, 
                          f"Candidates ({len(data['candidates'])})", data['candidates'], 'mediumorchid', 30)
        plot_single_graph(axes[2], r_before, coords, 
                          f"Search Scope ({len(data['search'])})", data['search'], 'orange', 50)
        
        anchor_total = len(data['anchor_vrp']) + len(data['anchor_sliding']) + len(data['anchor_two_opt'])
        anchor_title = f"Anchors ({anchor_total}) (V:{len(data['anchor_vrp'])} S:{len(data['anchor_sliding'])} T:{len(data['anchor_two_opt'])})"
        plot_anchor_graph(axes[3], r_before, coords, anchor_title, 
                          data['anchor_vrp'], data['anchor_sliding'], data['anchor_two_opt'])
        
        plot_single_graph(axes[4], r_after, coords, 
                          f"h_active ({len(data['h_active'])})", data['h_active'], 'red', 40)
        plot_single_graph(axes[5], r_after, coords, 
                          f"Sol Changed ({len(data['sol_changed'])})", data['sol_changed'], 'dodgerblue', 40)
        plot_single_graph(axes[6], r_after, coords, 
                          f"Sol Intersection ({len(data['sol_intersection'])})", data['sol_intersection'], 'deeppink', 60)

        fig.suptitle(f'Search #{search_id} - Iter #{iter_id} (Step {seq_i+1}/{len(group_keys)})', 
                     fontsize=18, fontweight='bold')
        
        handles = [
            mpatches.Patch(color='mediumorchid', label='Candidates'),
            mpatches.Patch(color='orange', label='Search Scope'),
            mpatches.Patch(color='#1f77b4', label='VRP (Anchor)'),
            mpatches.Patch(color='#ff7f0e', label='SLIDING (Anchor)'),
            mpatches.Patch(color='#2ca02c', label='TWO_OPT (Anchor)'),
            mpatches.Patch(color='black', label='Overlap (Anchor)'),
            mpatches.Patch(color='red', label='h_active'),
            mpatches.Patch(color='dodgerblue', label='Sol Changed'),
            mpatches.Patch(color='deeppink', label='Sol Intersection')
        ]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=9, fontsize=12)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=70)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        plt.close(fig)
        buf.close()

    # 7. 保存 GIF，文件名更新为带有 search_id 和序列长度的信息
    if frames:
        gif_filename = f'search_{selected_search_id:04d}_iter_0_to_{seq_len-1}.gif'
        save_path = os.path.join(output_dir, gif_filename)
        frames[0].save(save_path, format='GIF', save_all=True, append_images=frames[1:], duration=duration, loop=0)
        print(f"  -> Saved GIF: {gif_filename}")
# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Visualize Groups as GIFs (Single Trail, Expanded Cols)')
    parser.add_argument('--log_file', type=str, required=True, help='Path to log file')
    parser.add_argument('--instance_file', type=str, required=True, help='Path to instance file')
    parser.add_argument('--index', type=int, required=True, help='Instance index (for coords)')
    parser.add_argument('--output_dir', type=str, default='viz_gifs_output', help='Output directory')
    parser.add_argument('--num_groups', type=int, default=5, help='Number of random groups')
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length per group')
    parser.add_argument('--duration', type=int, default=500, help='Frame duration in ms')
    
    args = parser.parse_args()
    
    if load_raw_data is None:
        print("Error: load_nco_data module missing.")
        sys.exit(1)
        
    print(f"Loading coordinates for Instance Index {args.index}...")
    try:
        nodes_t, _, _, _, _ = load_raw_data(args.instance_file, episode=1, begin_index=args.index)
        coordinates = nodes_t[0] 
    except Exception as e:
        print(f"Data Load Error: {e}")
        sys.exit(1)
    
    print("Parsing log file...")
    data_store = read_log_file(args.log_file)
    
    if not data_store:
        print("No valid data found in log.")
        sys.exit(1)
        
    create_multi_group_visualization(data_store, coordinates, args.output_dir, 
                                     args.num_groups, args.seq_len, args.duration)
    print(f"Done! GIFs saved to {args.output_dir}")

if __name__ == '__main__':
    main()