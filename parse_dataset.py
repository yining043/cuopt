#!/usr/bin/env python3
"""
解析 local_search.cu 输出的日志文件 (适配 anchor 格式)
每个 state (= instance+search+iteration) 保留最多 12 个代表性 trails，
并分配 state_id 供 GRPO 训练使用。
"""

import re
import sys
import os
import glob
import gc
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
    candidate_size: int
    anchor_nodes: set = field(default_factory=set)
    current_solution: dict = field(default_factory=dict)
    trails: list = field(default_factory=list)


def parse_route_list(route_str):
    routes = []
    for match in re.findall(r'\[([^\]]+)\]', route_str):
        nodes = [int(x.strip()) for x in match.split(',') if x.strip()]
        routes.append(nodes)
    return routes


def parse_int_list(s):
    """Parse a comma-separated int list string like '1,2,3' into a list of ints."""
    return [int(x.strip()) for x in s.split(',') if x.strip()]


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
            if current_iter and current_search:
                current_search.iterations.append(current_iter)
            if current_search:
                results.append(current_search)
            current_search = SearchInfo(
                search_id=int(m.group(1)),
                n_nodes_w_dummy=int(m.group(2)),
                n_nodes=int(m.group(3))
            )
            current_iter = None
            current_trail = None
            i += 1
            continue

        # [iter #X] candidate_size: Y
        m = re.match(r'\[iter #(\d+)\]\s*candidate_size:\s*(\d+)', line)
        if m and current_search:
            if current_iter:
                current_search.iterations.append(current_iter)
            current_iter = IterationInfo(
                iter_id=int(m.group(1)),
                candidate_size=int(m.group(2))
            )
            current_trail = None
            i += 1
            continue

        # [before_search] sol: [route1] [route2] ...
        m = re.match(r'\[before_search\]\s*sol:\s*(.+)', line)
        if m and current_iter:
            routes = parse_route_list(m.group(1))
            current_iter.current_solution = {
                'number_of_routes': len(routes),
                'routes': routes
            }
            i += 1
            continue

        # [before_search] candidates: [...] — skip
        if line.startswith('[before_search] candidates:'):
            i += 1
            continue

        # [anchor] types=VRP: [...] / SLIDING: [...] / TWO_OPT: [...]
        m = re.match(r'\[anchor\]\s*types=\w+:\s*\[([^\]]*)\]', line)
        if m and current_iter:
            if m.group(1).strip():
                nodes = parse_int_list(m.group(1))
                current_iter.anchor_nodes.update(nodes)
            i += 1
            continue

        # [trail #X] last_estimated_cost: ..., cost: ...
        m = re.match(r'\[trail #(\d+)\]\s*last_estimated_cost:', line)
        if m and current_iter:
            current_trail = {
                'trail_id': int(m.group(1)),
                'executed_anchors_all': [],
                'previous_cost': [],
                'number_of_anchors': 0
            }
            i += 1
            continue

        # executed_anchors (all): [...]
        m = re.match(r'executed_anchors \(all\):\s*\[([^\]]*)\]', line)
        if m and current_trail is not None:
            if m.group(1).strip():
                current_trail['executed_anchors_all'] = parse_int_list(m.group(1))
            i += 1
            continue

        # executed_anchors (sliding/vrp/recycle_vrp/two_opt): [...] — skip
        if line.startswith('executed_anchors ('):
            i += 1
            continue

        # previous_cost: [v1,v2,v3,v4]
        m = re.match(r'previous_cost:\s*\[([^\]]+)\]', line)
        if m and current_trail is not None:
            current_trail['previous_cost'] = [float(x.strip()) for x in m.group(1).split(',') if x.strip()]
            i += 1
            continue

        # [trail #X] full size: ..., number_of_anchors: Y, ... — trail end marker
        m = re.match(r'\[trail #(\d+)\]\s*full size:.*number_of_anchors:\s*(\d+)', line)
        if m and current_trail is not None and current_iter:
            current_trail['number_of_anchors'] = int(m.group(2))
            current_iter.trails.append(current_trail)
            current_trail = None
            i += 1
            continue

        i += 1

    if current_iter and current_search:
        current_search.iterations.append(current_iter)
    if current_search:
        results.append(current_search)

    return results


if __name__ == '__main__':
    dir_path = sys.argv[1]
    instance_file = sys.argv[2]

    all_results = []
    pattern = os.path.join(dir_path, 'instance_*.txt')
    files = sorted(glob.glob(pattern), key=lambda x: int(re.search(r'instance_(\d+)\.txt', x).group(1)))

    for file_path in tqdm(files, desc="Parsing files"):
        match = re.search(r'instance_(\d+)\.txt', file_path)
        index = int(match.group(1))
        results = parse_output_file(file_path)
        instance_data = load_raw_data(instance_file, episode=1, begin_index=index)
        for search in results:
            search.instance_index = index
            search.instance_data = instance_data
        all_results.extend(results)

    results = all_results
    total_iters = sum(len(r.iterations) for r in results)
    total_trails = sum(len(it.trails) for r in results for it in r.iterations)
    print(f"Total searches loaded: {len(results)}")
    print(f"Total iterations loaded: {total_iters}")
    print(f"Total trails loaded: {total_trails}")

    if len(results) == 0:
        print("No data parsed. Exiting.")
        sys.exit(0)

    max_candidates_length = max(r.n_nodes_w_dummy for r in results)
    dummy_depot_start = 1001

    batch_nodes = []
    batch_demands = []
    batch_current_sol = []
    batch_anchor = []
    batch_selected = []
    batch_previous_cost = []
    batch_num_anchors = []
    batch_state_id = []

    batch_size = 2000
    state_counter = 0
    max_trails_per_state = 12
    skipped_no_anchor = 0
    skipped_empty_selected = 0
    skipped_no_improvement = 0
    skipped_low_std = 0
    skipped_few_trails = 0
    kept = 0

    pbar = tqdm(total=len(results), desc="Processing batches")

    while len(results) > 0:
        current_batch_size = min(batch_size, len(results))
        batch_results = results[:current_batch_size]
        del results[:current_batch_size]

        ml_data = {
            'nodes_tensor': [],
            'demands_tensor': [],
            'current_sol': [],
            'anchor': [],
            'selected': [],
            'previous_cost': [],
            'number_of_anchors': [],
            'state_id': []
        }

        for search in batch_results:
            nodes_t, capacities_t, demands_t, costs_t, node_flags_t = search.instance_data
            nodes = nodes_t[0]
            demands = demands_t[0] / capacities_t[0]

            for iter_info in search.iterations:
                if not iter_info.current_solution or not iter_info.trails:
                    continue

                if not iter_info.anchor_nodes:
                    skipped_no_anchor += len(iter_info.trails)
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

                anchor_tensor = torch.zeros(max_candidates_length, dtype=torch.bool)
                for idx in iter_info.anchor_nodes:
                    if idx < max_candidates_length:
                        anchor_tensor[idx] = True

                valid_trails = []
                for trail in iter_info.trails:
                    anchors_all = trail['executed_anchors_all']
                    prev_cost = trail['previous_cost']
                    n_anchors = trail['number_of_anchors']

                    if not anchors_all:
                        skipped_empty_selected += 1
                        continue

                    if len(prev_cost) < 2 or prev_cost[0] - prev_cost[-1] <= 0:
                        skipped_no_improvement += 1
                        continue

                    cost_tensor = torch.zeros(4, dtype=torch.float32)
                    for ci, cv in enumerate(prev_cost[:4]):
                        cost_tensor[ci] = cv

                    if cost_tensor.std(-1).item() <= 1.0:
                        skipped_low_std += 1
                        continue

                    s_tensor = torch.zeros(max_candidates_length, dtype=torch.bool)
                    for idx in anchors_all:
                        if idx < max_candidates_length:
                            s_tensor[idx] = True

                    ratio = prev_cost[-1] / prev_cost[0]
                    valid_trails.append({
                        's_tensor': s_tensor,
                        'cost_tensor': cost_tensor,
                        'n_anchors': n_anchors,
                        'ratio': ratio,
                    })

                if len(valid_trails) < 2:
                    skipped_few_trails += len(valid_trails)
                    continue

                valid_trails.sort(key=lambda x: x['ratio'])

                if len(valid_trails) <= max_trails_per_state:
                    selected_trails = valid_trails
                else:
                    best = valid_trails[0]
                    worst = valid_trails[-1]
                    middle = valid_trails[1:-1]
                    n_mid = max_trails_per_state - 2
                    indices = [int(i * (len(middle) - 1) / (n_mid - 1)) for i in range(n_mid)]
                    selected_trails = [best] + [middle[i] for i in indices] + [worst]

                for t in selected_trails:
                    ml_data['nodes_tensor'].append(nodes)
                    ml_data['demands_tensor'].append(demands)
                    ml_data['current_sol'].append(flat_sol)
                    ml_data['anchor'].append(anchor_tensor)
                    ml_data['selected'].append(t['s_tensor'])
                    ml_data['previous_cost'].append(t['cost_tensor'])
                    ml_data['number_of_anchors'].append(t['n_anchors'])
                    ml_data['state_id'].append(state_counter)
                    kept += 1

                state_counter += 1

        del batch_results

        if len(ml_data['nodes_tensor']) > 0:
            batch_nodes.append(torch.stack(ml_data['nodes_tensor']))
            batch_demands.append(torch.stack(ml_data['demands_tensor']))
            batch_current_sol.append(torch.tensor(ml_data['current_sol']))
            batch_anchor.append(torch.stack(ml_data['anchor']))
            batch_selected.append(torch.stack(ml_data['selected']))
            batch_previous_cost.append(torch.stack(ml_data['previous_cost']))
            batch_num_anchors.append(torch.tensor(ml_data['number_of_anchors']))
            batch_state_id.append(torch.tensor(ml_data['state_id'], dtype=torch.int64))

        pbar.update(current_batch_size)
        gc.collect()

    pbar.close()
    del results
    gc.collect()

    print(f"\nFiltering summary:")
    print(f"  Kept: {kept}")
    print(f"  Total states: {state_counter}")
    print(f"  Avg trails per state: {kept / max(state_counter, 1):.1f}")
    print(f"  Skipped (anchor union empty): {skipped_no_anchor}")
    print(f"  Skipped (executed_anchors_all empty): {skipped_empty_selected}")
    print(f"  Skipped (no improvement): {skipped_no_improvement}")
    print(f"  Skipped (cost std <= 1): {skipped_low_std}")
    print(f"  Skipped (fewer than 2 trails per state): {skipped_few_trails}")

    def smart_cat(tensor_list, dtype=None):
        if not tensor_list:
            return torch.empty(0)
        total_rows = sum(t.shape[0] for t in tensor_list)
        shape = (total_rows,) + tensor_list[0].shape[1:]
        if dtype is None:
            dtype = tensor_list[0].dtype
        final_tensor = torch.empty(shape, dtype=dtype)
        start = 0
        for j in range(len(tensor_list)):
            t = tensor_list[j]
            end = start + t.shape[0]
            final_tensor[start:end] = t
            start = end
            tensor_list[j] = None
        return final_tensor

    print("Concatenating tensors...")
    nodes_tensor = smart_cat(batch_nodes)
    del batch_nodes; gc.collect()

    demands_tensor = smart_cat(batch_demands)
    del batch_demands; gc.collect()

    current_sol_tensor = smart_cat(batch_current_sol, dtype=torch.int16)
    del batch_current_sol; gc.collect()

    anchor_tensor = smart_cat(batch_anchor)
    del batch_anchor; gc.collect()

    selected_tensor = smart_cat(batch_selected)
    del batch_selected; gc.collect()

    cost_tensor = smart_cat(batch_previous_cost)
    del batch_previous_cost; gc.collect()

    number_of_anchors_tensor = smart_cat(batch_num_anchors)
    del batch_num_anchors; gc.collect()

    state_id_tensor = smart_cat(batch_state_id, dtype=torch.int64)
    del batch_state_id; gc.collect()

    print(f"\nML Data shapes:")
    print(f"  nodes_tensor:              {nodes_tensor.shape}")
    print(f"  demands_tensor:            {demands_tensor.shape}")
    print(f"  current_sol_tensor:        {current_sol_tensor.shape}")
    print(f"  anchor_tensor:             {anchor_tensor.shape}")
    print(f"  selected_tensor:           {selected_tensor.shape}")
    print(f"  cost_tensor:               {cost_tensor.shape}")
    print(f"  number_of_anchors_tensor:  {number_of_anchors_tensor.shape}")
    print(f"  state_id_tensor:           {state_id_tensor.shape}")
    print(f"  max_candidates_length:     {max_candidates_length}")

    output_file = sys.argv[3] if len(sys.argv) > 3 else 'ml_data.pt'
    ml_data_dict = {
        'nodes_tensor': nodes_tensor,
        'demands_tensor': demands_tensor,
        'current_sol_tensor': current_sol_tensor,
        'anchor_tensor': anchor_tensor,
        'selected_tensor': selected_tensor,
        'cost_tensor': cost_tensor,
        'number_of_anchors_tensor': number_of_anchors_tensor,
        'state_id_tensor': state_id_tensor,
        'max_candidates_length': max_candidates_length
    }
    torch.save(ml_data_dict, output_file)
    print(f"ML data saved to: {output_file}")
