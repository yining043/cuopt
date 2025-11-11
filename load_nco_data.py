import torch
from tqdm import tqdm   

def load_raw_data(data_path: str, episode: int = 1, begin_index: int = 0):
    def tow_col_nodeflag(node_flag):
        V = int(len(node_flag) / 2)
        return [[node_flag[i], node_flag[V + i]] for i in range(V)]

    nodes, capacities, demands, costs, node_flags = [], [], [], [], []

    with open(data_path, "r") as f:
        lines = f.readlines()[begin_index:begin_index + episode]

    for line in tqdm(lines, desc="Loading data"):
        line = line.strip().split(",")

        depot_index = line.index('depot')
        customer_index = line.index('customer')
        capacity_index = line.index('capacity')
        demand_index = line.index('demand')
        cost_index = line.index('cost')
        node_flag_index = line.index('node_flag')

        depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
        customer = [[float(line[idx]), float(line[idx + 1])]
                    for idx in range(customer_index + 1, capacity_index, 2)]
        loc = depot + customer

        capacity = int(float(line[capacity_index + 1]))
        if int(line[demand_index + 1]) == 0:
            demand = [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
        else:
            demand = [0] + [int(line[idx]) for idx in range(demand_index + 1, cost_index)]

        cost = float(line[cost_index + 1])
        node_flag = [int(line[idx]) for idx in range(node_flag_index + 1, len(line))]
        node_flag = tow_col_nodeflag(node_flag)

        nodes.append(loc)
        capacities.append(capacity)
        demands.append(demand)
        costs.append(cost)
        node_flags.append(node_flag)

    return (
        torch.tensor(nodes, requires_grad=False),
        torch.tensor(capacities, requires_grad=False),
        torch.tensor(demands, requires_grad=False),
        torch.tensor(costs, requires_grad=False),
        torch.tensor(node_flags, requires_grad=False)
    )
