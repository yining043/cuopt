#!/usr/bin/env python3
"""
解析 local_search.cu 输出的日志文件
"""

import re
import sys
import os
import glob
import torch
from dataclasses import dataclass, field
from load_nco_data import load_raw_data
from tqdm import tqdm


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
        
        # [trail #X] candidates: [...] selected: [...] score: X best score: Y (同一行)
        m = re.match(r'\[trail #(\d+)\]\s*candidates:\s*\[([^\]]+)\]\s*selected:\s*\[([^\]]+)\]\s*score:\s*([\d.]+)\s*best score:\s*([\d.]+)', line)
        if m and current_iter:
            trail_id = int(m.group(1))
            candidates = [int(x.strip()) for x in m.group(2).split(',') if x.strip()]
            selected = [int(x.strip()) for x in m.group(3).split(',') if x.strip()]
            current_trail = {
                'trail_id': trail_id,
                'candidates': candidates,
                'selected': selected,
                'score': float(m.group(4)),
                'best_score': float(m.group(5))
            }
            i += 1
            continue
        
        # [basin_solution #X] number_of_routes: Y, [route1] [route2] ...
        m = re.match(r'\[basin_solution #(\d+)\]\s*number_of_routes:\s*(\d+),\s*(.+)', line)
        if m and current_trail:
            current_trail['basin_solution'] = {
                'number_of_routes': int(m.group(2)),
                'routes': parse_route_list(m.group(3))
            }
            current_iter.trails.append(current_trail)
            current_trail = None
            i += 1
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
    
    all_results = []
    
    pattern = os.path.join(dir_path, 'instance_*.txt')
    files = sorted(glob.glob(pattern), key=lambda x: int(re.search(r'instance_(\d+)\.txt', x).group(1)))
    
    for file_path in tqdm(files[:50], desc="Parsing files"):
        match = re.search(r'instance_(\d+)\.txt', file_path)
        index = int(match.group(1))
        
        results = parse_output_file(file_path)
        instance_data = load_raw_data(instance_file, episode=1, begin_index=index)
        
        for search in results:
            search.instance_index = index
            search.instance_data = instance_data
        
        all_results.extend(results)
    
    results = all_results
    total = 0
    for result in results:
        total += len(result.iterations)
    print(f"Total searches loaded: {len(results)}")
    print(f"Total data point loaded: {total}")

    # make for ML
    max_candidates_length = max([r.n_nodes_w_dummy for r in results])
    dummy_depot_start = 1001
    
    ml_data = {
        'nodes_tensor': [],
        'demands_tensor': [],
        'current_sol': [],
        'candidates': [],
        'selected': [],
        'score': []
    }
    
    for search in results:
        nodes_t, capacities_t, demands_t, costs_t, node_flags_t = search.instance_data
        nodes = nodes_t[0]
        demands = demands_t[0] / capacities_t[0]
        
        for iter_info in search.iterations:
            if not iter_info.current_solution or not iter_info.trails:
                continue

            if len(iter_info.trails[0]['candidates']) == len(iter_info.trails[0]['selected']):
                continue
            
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

            for trail in iter_info.trails:
                candidates = trail['candidates']
                selected = trail['selected']
                score = trail['score']
                
                candidates_bool = [False] * max_candidates_length
                for idx in candidates: candidates_bool[idx] = True
                
                selected_bool = [False] * max_candidates_length
                for idx in selected: selected_bool[idx] = True
                
                ml_data['candidates'].append(candidates_bool)
                ml_data['selected'].append(selected_bool)
                ml_data['score'].append(score)
    
    nodes_tensor = torch.stack(ml_data['nodes_tensor']).view(-1, 1001, 2)
    demands_tensor = torch.stack(ml_data['demands_tensor']).view(-1, 1001, 1)
    current_sol_tensor = torch.tensor(ml_data['current_sol'], dtype=torch.int32).view(-1, max_candidates_length)
    candidates_tensor = torch.tensor(ml_data['candidates'], dtype=torch.bool).view(-1, 10, max_candidates_length)
    selected_tensor = torch.tensor(ml_data['selected'], dtype=torch.bool).view(-1, 10, max_candidates_length)
    score_tensor = torch.tensor(ml_data['score']).view(-1, 10)
    
    print(f"ML Data shapes:")
    print(f"  nodes_tensor: {nodes_tensor.shape}")
    print(f"  demands_tensor: {demands_tensor.shape}")
    print(f"  current_sol_tensor: {current_sol_tensor.shape}")
    print(f"  candidates_tensor: {candidates_tensor.shape}")
    print(f"  selected_tensor: {selected_tensor.shape}")
    print(f"  score_tensor: {score_tensor.shape}")
    print(f"  max_candidates_length: {max_candidates_length}")

    # store ML data
    output_file = sys.argv[3] if len(sys.argv) > 3 else 'ml_data.pt'
    ml_data_dict = {
        'nodes_tensor': nodes_tensor,
        'demands_tensor': demands_tensor,
        'current_sol_tensor': current_sol_tensor,
        'candidates_tensor': candidates_tensor,
        'selected_tensor': selected_tensor,
        'score_tensor': score_tensor,
        'max_candidates_length': max_candidates_length
    }
    torch.save(ml_data_dict, output_file)
    print(f"ML data saved to: {output_file}")