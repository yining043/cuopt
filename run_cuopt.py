import argparse
import math
from gc import set_debug
from os import pread
from statistics import mean
import numpy as np
import cudf
from cuopt import routing
from cuopt.routing import CustomizeNodesCallback
from tqdm import tqdm
from load_nco_data import load_raw_data
import torch
import random
from model import CostPredictor

def pairwise_euclidean_distance(x: torch.Tensor) -> torch.Tensor:
    x_square = (x ** 2).sum(dim=2, keepdim=True)
    dist_square = x_square + x_square.transpose(1, 2) - 2 * torch.bmm(x, x.transpose(1, 2))
    return torch.sqrt(torch.clamp(dist_square, min=1e-9))

class CostPredictorCallback(CustomizeNodesCallback):
    """Callback that uses CostPredictor to score executed_anchors from K trails."""
    def __init__(self, model, coordinates, demand, vehicle_capacity,
                 device='cpu', top_frac: float = 0.05):
        super().__init__()
        self.model = model.eval()
        self.coordinates = coordinates.to(device)
        self.demand = (demand / vehicle_capacity).to(device)
        self.device = device
        self.top_frac = top_frac

    def customize_nodes_to_search(self, solution_flat, num_routes,
                                  solution_cost, trail_masks_flat, num_trails, iter):
        max_length = len(trail_masks_flat) // num_trails
        K = num_trails
        trail_masks = torch.tensor(trail_masks_flat, dtype=torch.bool, device=self.device
                                   ).reshape(K, max_length)

        non_empty = trail_masks.any(dim=1)
        if not non_empty.any():
            return [0] * max_length

        sol_tensor = torch.tensor(
            solution_flat + [-1] * (max_length - len(solution_flat)),
            dtype=torch.long, device=self.device
        ).unsqueeze(0).expand(K, -1)
        nodes = self.coordinates.expand(K, -1, -1)
        demands = self.demand.expand(K, -1)
        cost_0 = torch.full((K,), solution_cost,
                            dtype=torch.float32, device=self.device)

        with torch.no_grad():
            predicted_ratios = self.model(
                nodes, demands, sol_tensor, trail_masks, cost_0)

        predicted_ratios[~non_empty] = float('inf')
        valid_idx = torch.where(non_empty)[0]
        n_valid = int(valid_idx.numel())
        scores = predicted_ratios[valid_idx]
        k = max(1, int(math.ceil(self.top_frac * n_valid)))
        k = min(k, n_valid)
        _, local_top = torch.topk(scores, k, largest=False)
        pool = valid_idx[local_top].cpu().tolist()
        best_idx = random.choice(pool)
        return trail_masks[best_idx].int().cpu().tolist()

def make_cuopt_format(index, raw_data_dist, raw_data_demand, raw_data_capacity, n_vehicles, scale):
    distance_matrix_df = cudf.DataFrame(raw_data_dist[index].numpy() * scale)
    location_demand = cudf.Series(raw_data_demand[index].numpy(), dtype=np.int32)
    vehicle_capacity = cudf.Series([raw_data_capacity[index].item()] * n_vehicles, dtype=np.int32)
    return distance_matrix_df, location_demand, vehicle_capacity

def get_cuopt_model(index, raw_data_dist, raw_data_demand, raw_data_capacity, n_vehicles, scale):
    distance_matrix_df, location_demand, vehicle_capacity = make_cuopt_format(
        index, raw_data_dist, raw_data_demand, raw_data_capacity, n_vehicles, scale
    )
    n_locations = raw_data_dist.shape[1]
    data_model = routing.DataModel(n_locations, n_vehicles)
    data_model.add_cost_matrix(distance_matrix_df)
    data_model.add_capacity_dimension("demand", location_demand, vehicle_capacity)
    depot = cudf.Series([0] * n_vehicles)
    data_model.set_vehicle_locations(depot, depot)
    return data_model

def run_cuopt(data_model, time_limit, callback=None):
    solver_settings = routing.SolverSettings()
    solver_settings.set_time_limit(time_limit)
    if callback:
        solver_settings.set_routing_callback(callback)
    solution = routing.Solve(data_model, solver_settings)
    return solution if solution.get_status() == 0 else None

def run_experiment(
    data_path="data/test_cvrp1000_hgs_n128_C250.txt",
    time_limit=10,
    problem_size=1,
    index=0,
    problem_type="CVRP",
    scale=1e2,
    n_vehicles=21,
    use_callback=False,
    policy_model_path=None,
    mode='new',
    callback_top_frac: float = 0.3,
):
    raw_nodes, raw_cap, raw_demand, raw_cost, raw_flag = load_raw_data(data_path, episode=problem_size, begin_index=index)
    raw_dist = pairwise_euclidean_distance(raw_nodes)

    # 自动检测 device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Device] Using {device}")
    
    # 加载 Policy 模型（如果需要）
    policy = None
    if use_callback:
        policy = CostPredictor(device=device, mode=mode)
        if policy_model_path:
            print(f"[Callback] Loading CostPredictor model from {policy_model_path}")
            policy.load_state_dict(torch.load(policy_model_path, map_location=device)['model_state_dict'])

    costs = []
    gaps = []

    for index in tqdm(range(problem_size), desc="Solving with cuOpt"):
        # 创建 callback（使用当前问题的数据）
        callback = None
        if use_callback:
            callback = CostPredictorCallback(
                policy, raw_nodes, raw_demand, raw_cap, device,
                top_frac=callback_top_frac,
            )
        
        model = get_cuopt_model(index, raw_dist, raw_demand, raw_cap, n_vehicles, scale)
        solution = run_cuopt(model, time_limit, callback=callback)
        raw_cost_value = raw_cost[index].item()

        if solution:
            cost = solution.get_total_objective() / scale
            gap = ((cost - raw_cost_value) / raw_cost_value) * 100
            costs.append(cost)
            gaps.append(gap)
            print(f"[Index {index}] cuOpt Cost: {cost:.2f} | Best Cost: {raw_cost_value:.2f} | Gap: {gap:.2f}%")
        else:
            print(f"[Index {index}] No feasible solution.")
            costs.append(None)
            gaps.append(None)

    return costs, gaps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cuOpt experiments with custom config.")
    parser.add_argument("--data_path", default="../cuopt-examples/data/test_cvrp1000_hgs_n128_C250.txt", type=str, help="Path to dataset file")
    parser.add_argument("--time_limit", type=float, default=5, help="Time limit per instance (seconds)")
    parser.add_argument("--problem_size", type=int, default=1, help="Number of instances to run")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--problem_type", type=str, default="CVRP", help="Problem type (default: CVRP)")
    parser.add_argument("--n_vehicles", type=int, default=21, help="Number of vehicles")
    parser.add_argument("--scale", type=float, default=1e2, help="Coordinate scale")
    parser.add_argument("--use_callback", action='store_true', help="Use callback for node selection (requires pred_with_NN=true in C++ code)")
    parser.add_argument("--policy_model_path", type=str, default=None, help="Path to Policy model checkpoint (.pt file)")
    parser.add_argument("--v", type=str, default="new", choices=["old", "new", "v2"],
                        help="CostPredictor mode: old=ratio, new=new head, v2=RegressionHeadV2（与 test --mode 一致）")
    parser.add_argument(
        "--callback-top-frac",
        type=float,
        default=0.3,
        help="Callback：在 pred 最小的 ceil(frac×K_valid) 条 trail 中随机选一条（默认 0.3=前30%%）",
    )

    args = parser.parse_args()

    if args.v == "old":
        mode = "ratio"
    elif args.v == "v2":
        mode = "v2"
    else:
        mode = "new"
    run_experiment(
        data_path=args.data_path,
        time_limit=args.time_limit,
        problem_size=args.problem_size,
        index=args.index,
        problem_type=args.problem_type,
        scale=args.scale,
        n_vehicles=args.n_vehicles,
        use_callback=args.use_callback,
        policy_model_path=args.policy_model_path,
        mode=mode,
        callback_top_frac=args.callback_top_frac,
    )
