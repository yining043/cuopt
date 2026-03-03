#!/usr/bin/env python3
"""
解析 local_search.cu 输出的日志文件（扩展版本）
包含10次random运行的所有数据
"""

import re
import sys
import os
import glob
import torch
from dataclasses import dataclass, field
from load_nco_data import load_raw_data
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
    

def visulize_analysis(sorted_costs_np):
    """
    画all_costs_tensor最后一个step的箱型图
    """
    
    # 准备箱型图数据：每个trail有10个runs的值
    boxplot_data = [sorted_costs_np[i, :] for i in range(10)]

    # 画箱型图
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(boxplot_data, patch_artist=True)
    
    # 美化箱型图
    colors = plt.cm.viridis(np.linspace(0, 1, 10))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('all_costs_boxplot.png', dpi=300, bbox_inches='tight')
    print("Plot saved to all_costs_boxplot.png")
    plt.close()

@dataclass
class SearchInfo:
    search_id: int
    n_nodes_w_dummy: int
    n_nodes: int
    iterations: list = field(default_factory=list)
    instance_index: int = 0
    instance_data: object = None


@dataclass
class IterationInfo:
    iter_id: int
    best_node_size: int
    sample_size: int
    current_solution: dict = field(default_factory=dict)
    trails: list = field(default_factory=list)
    executed: dict = field(default_factory=dict)


def parse_route_list(route_str):
    routes = []
    for match in re.findall(r'\[([^\]]+)\]', route_str):
        nodes = [int(x.strip()) for x in match.split(',') if x.strip()]
        routes.append(nodes)
    return routes


def parse_output_file(file_path):
    results = []
    current_search = None
    current_iter = None
    current_trail = None
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # [search #X] N_nodes_w_dummy: Y, N_nodes: Z
        m = re.match(r'\[search #(\d+)\]\s*N_nodes_w_dummy:\s*(\d+),\s*N_nodes:\s*(\d+)', line)
        if m:
            if current_search:
                results.append(current_search)
            current_search = SearchInfo(
                search_id=int(m.group(1)),
                n_nodes_w_dummy=int(m.group(2)),
                n_nodes=int(m.group(3))
            )
            current_iter = None
            i += 1
            continue
        
        # [iter #X] best_node_size: Y, sample_size: Z
        m = re.match(r'\[iter #(\d+)\]\s*best_node_size:\s*(\d+),\s*sample_size:\s*(\d+)', line)
        if m and current_search:
            if current_iter:
                current_search.iterations.append(current_iter)
            current_iter = IterationInfo(
                iter_id=int(m.group(1)),
                best_node_size=int(m.group(2)),
                sample_size=int(m.group(3))
            )
            i += 1
            continue
        
        # [current_solution] number_of_routes: X, [route1] [route2] ...
        m = re.match(r'\[current_solution\]\s*number_of_routes:\s*(\d+),\s*(.+)', line)
        if m and current_iter:
            current_iter.current_solution = {
                'number_of_routes': int(m.group(1)),
                'routes': parse_route_list(m.group(2))
            }
            i += 1
            continue
        
        # [trail #X] candidates: [...] selected: [...] [10 runs stats] mean: X, std: Y
        # 注意：需要转义方括号
        m = re.match(r'\[trail #(\d+)\]\s*candidates:\s*\[([^\]]+)\]\s*selected:\s*\[([^\]]+)\]\s*\[10 runs stats\]\s*mean:\s*([\d.]+),\s*std:\s*([\d.]+)', line)
        if m and current_iter:
            trail_id = int(m.group(1))
            candidates = [int(x.strip()) for x in m.group(2).split(',') if x.strip()]
            selected = [int(x.strip()) for x in m.group(3).split(',') if x.strip()]
            mean_score = float(m.group(4))
            std_score = float(m.group(5))
            
            current_trail = {
                'trail_id': trail_id,
                'candidates': candidates,
                'selected': selected,
                'mean_score': mean_score,
                'std_score': std_score,
                'score': None,  # 将在下一行读取
                'best_score': None,  # 将在下一行读取
                'all_costs': None  # 将在后续行读取
            }
            i += 1
            continue
        
        # score: X best score: Y
        m = re.match(r'score:\s*([\d.]+)\s*best score:\s*([\d.]+)', line)
        if m and current_trail:
            current_trail['score'] = float(m.group(1))
            current_trail['best_score'] = float(m.group(2))
            i += 1
            continue
        
        # all_costs:
        if line == 'all_costs:' and current_trail:
            i += 1
            # 读取接下来的10行 run[X]: [...]
            all_costs = []
            for run_idx in range(10):
                if i >= len(lines):
                    break
                run_line = lines[i].strip()
                # 匹配 run[X]: [cost1, cost2, ..., cost10]
                m = re.match(r'\s*run\[(\d+)\]:\s*\[([^\]]+)\]', run_line)
                if m:
                    run_id = int(m.group(1))
                    costs_str = m.group(2)
                    # 解析costs，处理'---'为-1
                    costs = []
                    for cost_str in costs_str.split(','):
                        cost_str = cost_str.strip()
                        if cost_str == '---':
                            costs.append(-1.0)
                        else:
                            try:
                                costs.append(float(cost_str))
                            except ValueError:
                                costs.append(-1.0)
                    all_costs.append(costs)
                    i += 1
                else:
                    # 如果格式不匹配，跳出循环
                    break
            
            # 保存all_costs（可能少于10个runs）
            if len(all_costs) > 0:
                current_trail['all_costs'] = all_costs
            # 保存trail到iteration（无论是否有完整的all_costs）
            if current_iter:
                current_iter.trails.append(current_trail)
                current_trail = None
            continue
        
        # [executed] cost before: X, cost after: Y, move_found: Z
        m = re.match(r'\[executed\]\s*cost before:\s*([\d.]+),\s*cost after:\s*([\d.]+),\s*move_found:\s*(\d+)', line)
        if m and current_iter:
            current_iter.executed = {
                'cost_before': float(m.group(1)),
                'cost_after': float(m.group(2)),
                'move_found': int(m.group(3)) == 1
            }
            i += 1
            continue
        
        i += 1
    
    if current_iter and current_search:
        current_search.iterations.append(current_iter)
    if current_search:
        results.append(current_search)
    
    return results


def to_dict(search):
    return {
        'search_id': search.search_id,
        'n_nodes_w_dummy': search.n_nodes_w_dummy,
        'n_nodes': search.n_nodes,
        'iterations': [
            {
                'iter_id': iter_info.iter_id,
                'best_node_size': iter_info.best_node_size,
                'sample_size': iter_info.sample_size,
                'current_solution': iter_info.current_solution,
                'trails': iter_info.trails,
                'executed': iter_info.executed
            }
            for iter_info in search.iterations
        ]
    }

if __name__ == '__main__':
    dir_path = sys.argv[1]
    instance_file = sys.argv[2]
    
    # 1. 解析文件 (这一步本身很快，不需要改)
    all_results = []
    pattern = os.path.join(dir_path, 'instance_*.txt')
    files = sorted(glob.glob(pattern), key=lambda x: int(re.search(r'instance_(\d+)\.txt', x).group(1)))
    
    for file_path in tqdm(files[:8], desc="Parsing files"):
        match = re.search(r'instance_(\d+)\.txt', file_path)
        index = int(match.group(1))
        results = parse_output_file(file_path)
        instance_data = load_raw_data(instance_file, episode=1, begin_index=index)
        for search in results:
            search.instance_index = index
            search.instance_data = instance_data
        all_results.extend(results)
    
    results = all_results
    total = sum(len(r.iterations) for r in results)
    print(f"Total searches loaded: {len(results)}")
    print(f"Total data point loaded: {total}")
    
    # 打印第一个trail的信息以验证解析
    if results and results[0].iterations and results[0].iterations[0].trails:
        first_trail = results[0].iterations[0].trails[0]
        print(f"\nFirst trail info:")
        print(f"  trail_id: {first_trail['trail_id']}")
        print(f"  mean_score: {first_trail['mean_score']}")
        print(f"  std_score: {first_trail['std_score']}")
        print(f"  score: {first_trail['score']}")
        print(f"  best_score: {first_trail['best_score']}")
        if first_trail['all_costs']:
            print(f"  all_costs shape: {len(first_trail['all_costs'])} runs x {len(first_trail['all_costs'][0])} steps")
            print(f"  all_costs[0]: {first_trail['all_costs'][0]}")

    max_candidates_length = max([r.n_nodes_w_dummy for r in results])
    dummy_depot_start = 1001
    
    # 2. 准备容器
    batch_nodes = []
    batch_demands = []
    batch_current_sol = []
    batch_candidates = []
    batch_selected = []
    batch_score = []
    batch_all_costs = []  # 存储所有10次run的cost数据
    
    # 适当调大 batch_size 以利用向量化优势
    batch_size = 2000 
    
    import gc
    pbar = tqdm(total=len(results), desc="Processing batches (Vectorized)")
    
    while len(results) > 0:
        current_batch_size = min(batch_size, len(results))
        batch_results = results[:current_batch_size]
        del results[:current_batch_size] # 内存优化：处理完即删
        
        ml_data = {
            'nodes_tensor': [],
            'demands_tensor': [],
            'current_sol': [],
            'candidates': [],
            'selected': [],
            'score': [],
            'all_costs': []
        }
        
        for search in batch_results:
            nodes_t, capacities_t, demands_t, costs_t, node_flags_t = search.instance_data
            nodes = nodes_t[0]
            demands = demands_t[0] / capacities_t[0]
            
            for iter_info in search.iterations:
                if not iter_info.current_solution or not iter_info.trails:
                    continue
                if len(iter_info.trails[0]['candidates']) == len(iter_info.trails[0]['selected']):
                    continue
                
                # --- 优化点：Solution 构建逻辑不变 ---
                current_sol_routes = iter_info.current_solution.get('routes', [])
                flat_sol = []
                for route_idx, route in enumerate(current_sol_routes):
                    if route_idx == 0:
                        dummy_depot_nodes = list(range(dummy_depot_start, dummy_depot_start + 4))
                    else:
                        dummy_depot_base = dummy_depot_start + 4 + (route_idx - 1) * 4
                        dummy_depot_nodes = list(range(dummy_depot_base, dummy_depot_base + 4))
                    flat_sol.extend(dummy_depot_nodes)
                    flat_sol.extend(route[1:])
                if len(flat_sol) < max_candidates_length:
                    flat_sol.extend([-1] * (max_candidates_length - len(flat_sol)))
                
                ml_data['nodes_tensor'].append(nodes)
                ml_data['demands_tensor'].append(demands)
                ml_data['current_sol'].append(flat_sol)

                # --- 核心提速改动开始 (Vectorized) ---
                for trail in iter_info.trails:
                    candidates = trail['candidates']
                    selected = trail['selected']
                    # 使用mean_score代替score
                    score = trail.get('mean_score', 0.0)
                    all_costs = trail.get('all_costs', [])
                    
                    # 极速优化：直接用 Tensor 操作代替 Python List 循环
                    c_tensor = torch.zeros(max_candidates_length, dtype=torch.bool)
                    if candidates:
                        c_tensor[candidates] = True # 瞬间完成赋值
                    
                    s_tensor = torch.zeros(max_candidates_length, dtype=torch.bool)
                    if selected:
                        s_tensor[selected] = True   # 瞬间完成赋值
                    
                    ml_data['candidates'].append(c_tensor)
                    ml_data['selected'].append(s_tensor)
                    ml_data['score'].append(score)
                    
                    # 处理all_costs：转换为tensor，-1表示无效值
                    if all_costs:
                        # all_costs是list of lists，形状为(10, 10)
                        # 转换为tensor (10, 10)
                        costs_tensor = torch.tensor(all_costs, dtype=torch.float32)
                        ml_data['all_costs'].append(costs_tensor)
                    else:
                        # 如果没有all_costs数据，创建一个全-1的tensor
                        costs_tensor = torch.full((10, 10), -1.0, dtype=torch.float32)
                        ml_data['all_costs'].append(costs_tensor)
                # --- 核心提速改动结束 ---
        
        del batch_results
        
        if len(ml_data['nodes_tensor']) > 0:
            batch_nodes.append(torch.stack(ml_data['nodes_tensor']).view(-1, 1001, 2))
            batch_demands.append(torch.stack(ml_data['demands_tensor']).view(-1, 1001, 1))
            batch_current_sol.append(torch.tensor(ml_data['current_sol']).view(-1, max_candidates_length))
            
            # 由于 ml_data 里已经是 tensor 了，这里直接 stack
            batch_candidates.append(torch.stack(ml_data['candidates']).view(-1, 10, max_candidates_length))
            batch_selected.append(torch.stack(ml_data['selected']).view(-1, 10, max_candidates_length))
            batch_score.append(torch.tensor(ml_data['score']).view(-1, 10))
            batch_all_costs.append(torch.stack(ml_data['all_costs']).view(-1, 10, 10, 10))  # (batch, trails, runs, steps)
            
        pbar.update(current_batch_size)
        gc.collect()

    pbar.close()
    del results
    gc.collect()

    # 内存优化：Smart Cat
    def smart_cat(tensor_list, dtype=None):
        if not tensor_list: return torch.empty(0)
        total_rows = sum(t.shape[0] for t in tensor_list)
        shape = (total_rows,) + tensor_list[0].shape[1:]
        if dtype is None: dtype = tensor_list[0].dtype
        final_tensor = torch.empty(shape, dtype=dtype)
        start = 0
        for i in range(len(tensor_list)):
            t = tensor_list[i]
            end = start + t.shape[0]
            final_tensor[start:end] = t
            start = end
            tensor_list[i] = None 
        return final_tensor

    print("Concatenating tensors safely...")
    nodes_tensor = smart_cat(batch_nodes)
    del batch_nodes; gc.collect()
    
    demands_tensor = smart_cat(batch_demands)
    del batch_demands; gc.collect()
    
    current_sol_tensor = smart_cat(batch_current_sol, dtype=torch.int16)
    del batch_current_sol; gc.collect()
    
    candidates_tensor = smart_cat(batch_candidates)
    del batch_candidates; gc.collect()
    
    selected_tensor = smart_cat(batch_selected)
    del batch_selected; gc.collect()
    
    score_tensor = smart_cat(batch_score)
    del batch_score; gc.collect()
    
    all_costs_tensor = smart_cat(batch_all_costs)  # (N, 10, 10, 10)
    del batch_all_costs; gc.collect()

    # filter: 使用原始逻辑
    candidates_tensor = candidates_tensor[:, :1, :]
    index = score_tensor.std(1) > 0
    
    nodes_tensor = nodes_tensor[index]
    demands_tensor = demands_tensor[index]
    current_sol_tensor = current_sol_tensor[index]
    candidates_tensor = candidates_tensor[index]
    selected_tensor = selected_tensor[index]
    score_tensor = score_tensor[index]
    all_costs_tensor = all_costs_tensor[index]
    score_tensor = all_costs_tensor[:,:,:,0].mean(-1)
    
    print(f"ML Data shapes:")
    print(f"  nodes_tensor: {nodes_tensor.shape}")
    print(f"  demands_tensor: {demands_tensor.shape}")
    print(f"  current_sol_tensor: {current_sol_tensor.shape}")
    print(f"  candidates_tensor: {candidates_tensor.shape}")
    print(f"  selected_tensor: {selected_tensor.shape}")
    print(f"  score_tensor: {score_tensor.shape} (使用mean_score填充)")
    print(f"  all_costs_tensor: {all_costs_tensor.shape} (N iterations x 10 trails x 10 runs x 10 steps)")
    print(f"  max_candidates_length: {max_candidates_length}")

    output_file = sys.argv[3] if len(sys.argv) > 3 else 'ml_data_extended.pt'
    ml_data_dict = {
        'nodes_tensor': nodes_tensor,
        'demands_tensor': demands_tensor,
        'current_sol_tensor': current_sol_tensor,
        'candidates_tensor': candidates_tensor,
        'selected_tensor': selected_tensor,
        'score_tensor': score_tensor,
        # 'all_costs_tensor': all_costs_tensor,
        'max_candidates_length': max_candidates_length
    }
    import pdb; pdb.set_trace()
    all_costs_tensor = ml_data_dict['all_costs_tensor']  # (N, 10, 10, 10)
    last_step_costs = all_costs_tensor[:, :, :, 1]  # (N, 10, 10)
    trail_means_per_iter = last_step_costs.mean(dim=-1)  # (N, 10)
    sorted_indices = torch.argsort(trail_means_per_iter, -1)
    sorted_costs = last_step_costs.gather(dim=1, index=sorted_indices.unsqueeze(-1).expand_as(last_step_costs))
    # mean_costs = sorted_costs.mean(dim=0)  # (10, 10) - 对iterations维度求平均
    mean_costs = sorted_costs[0]
    visulize_analysis(mean_costs.cpu().numpy())
    # torch.save(ml_data_dict, output_file)
    # print(f"ML data saved to: {output_file}")

