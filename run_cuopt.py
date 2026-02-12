import argparse
import numpy as np
import cudf
from cuopt import routing
from cuopt.routing import CustomizeEarlyStopCallback
from tqdm import tqdm
from load_nco_data import load_raw_data
import torch
import random

def pairwise_euclidean_distance(x: torch.Tensor) -> torch.Tensor:
    x_square = (x ** 2).sum(dim=2, keepdim=True)
    dist_square = x_square + x_square.transpose(1, 2) - 2 * torch.bmm(x, x.transpose(1, 2))
    return torch.sqrt(torch.clamp(dist_square, min=1e-9))

class EarlyStopCallback(CustomizeEarlyStopCallback):
    """Callback for early stop decision. Override customize_early_stop to implement custom logic."""
    def customize_early_stop(self, solution_flat, objective, num_routes, iteration):
        """Return True to stop local search early, False to continue."""
        random_number = random.random()
        return random_number < 0.1

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
):
    raw_nodes, raw_cap, raw_demand, raw_cost, raw_flag = load_raw_data(data_path, episode=problem_size, begin_index=index)
    raw_dist = pairwise_euclidean_distance(raw_nodes)

    costs = []
    gaps = []

    for index in tqdm(range(problem_size), desc="Solving with cuOpt"):
        # 创建 callback（使用当前问题的数据）
        callback = None
        if use_callback:
            callback = EarlyStopCallback()
        
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
    parser.add_argument("--use_callback", action='store_true', help="Use early stop callback")

    args = parser.parse_args()

    run_experiment(
        data_path=args.data_path,
        time_limit=args.time_limit,
        problem_size=args.problem_size,
        index=args.index,
        problem_type=args.problem_type,
        scale=args.scale,
        n_vehicles=args.n_vehicles,
        use_callback=args.use_callback,
    )
