import argparse
import pickle
import numpy as np
import cudf
from cuopt import routing
from cuopt.routing import CustomizeEarlyStopCallback
from tqdm import tqdm
import torch
import random


def pairwise_euclidean_distance(x: torch.Tensor) -> torch.Tensor:
    x_square = (x ** 2).sum(dim=2, keepdim=True)
    dist_square = x_square + x_square.transpose(1, 2) - 2 * torch.bmm(x, x.transpose(1, 2))
    return torch.sqrt(torch.clamp(dist_square, min=1e-9))


def load_pkl_data(problem_path, solution_path, begin_index=0, episode=1):
    """Load CVRP instances and HGS solutions from pkl files.

    Args:
        problem_path: Path to cvrp pkl (e.g. cvrp100_uniform.pkl).
            Each entry: (depot[[x,y]], customers[[x,y],...], demands[...], capacity).
        solution_path: Path to HGS solution pkl (e.g. hgs_cvrp100_uniform.pkl).
            Each entry: (cost, route_list).
        begin_index: Starting instance index.
        episode: Number of instances to load.

    Returns:
        nodes, capacities, demands, costs, node_flags  (same interface as load_raw_data)
    """
    with open(problem_path, "rb") as f:
        problems = pickle.load(f)
    with open(solution_path, "rb") as f:
        solutions = pickle.load(f)

    subset_p = problems[begin_index : begin_index + episode]
    subset_s = solutions[begin_index : begin_index + episode]

    all_nodes, all_caps, all_demands, all_costs = [], [], [], []

    for (depot, customers, demands, capacity), (cost, _route) in zip(subset_p, subset_s):
        # depot first, then customers
        coords = [depot[0]] + customers
        # demand: prepend 0 for depot so length matches nodes
        all_nodes.append(coords)
        all_demands.append([0] + list(demands))
        all_caps.append(capacity)
        all_costs.append(cost)

    nodes = torch.tensor(all_nodes, dtype=torch.float32)         # (episode, n+1, 2)
    demands = torch.tensor(all_demands, dtype=torch.float32)     # (episode, n+1)
    capacities = torch.tensor(all_caps, dtype=torch.float32)     # (episode,)
    costs = torch.tensor(all_costs, dtype=torch.float32)         # (episode,)
    # node_flags not available in pkl; return zeros as placeholder
    node_flags = torch.zeros(episode, nodes.shape[1], 2)

    return nodes, capacities, demands, costs, node_flags


def load_txt_data(data_path, begin_index=0, episode=1):
    """Load CVRP data from the original txt format (via load_nco_data)."""
    from load_nco_data import load_raw_data
    return load_raw_data(data_path, episode=episode, begin_index=begin_index)


class EarlyStopCallback(CustomizeEarlyStopCallback):
    """Callback for early stop decision."""
    def customize_early_stop(self, solution_flat, objective, num_routes, iteration):
        """Return True to stop local search early, False to continue."""
        return random.random() < 0.1


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
    data_path=None,
    problem_path=None,
    solution_path=None,
    time_limit=10,
    problem_size=1,
    index=0,
    problem_type="CVRP",
    scale=1e2,
    n_vehicles=21,
    use_callback=False,
):
    # Load data from pkl or txt
    if problem_path and solution_path:
        raw_nodes, raw_cap, raw_demand, raw_cost, _ = load_pkl_data(
            problem_path, solution_path, begin_index=index, episode=problem_size
        )
    elif data_path:
        raw_nodes, raw_cap, raw_demand, raw_cost, _ = load_txt_data(
            data_path, begin_index=index, episode=problem_size
        )
    else:
        raise ValueError("Provide either --problem_path/--solution_path (pkl) or --data_path (txt)")

    raw_dist = pairwise_euclidean_distance(raw_nodes)

    costs = []
    gaps = []

    for i in tqdm(range(problem_size), desc="Solving with cuOpt"):
        callback = EarlyStopCallback() if use_callback else None

        model = get_cuopt_model(i, raw_dist, raw_demand, raw_cap, n_vehicles, scale)
        solution = run_cuopt(model, time_limit, callback=callback)
        raw_cost_value = raw_cost[i].item()

        if solution:
            cost = solution.get_total_objective() / scale
            gap = ((cost - raw_cost_value) / raw_cost_value) * 100
            costs.append(cost)
            gaps.append(gap)
            print(f"[Index {i}] cuOpt Cost: {cost:.2f} | Best Cost: {raw_cost_value:.2f} | Gap: {gap:.2f}%")
        else:
            print(f"[Index {i}] No feasible solution.")
            costs.append(None)
            gaps.append(None)

    return costs, gaps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cuOpt experiments with custom config.")
    # Data source: either pkl pair or txt file
    parser.add_argument("--data_path", default=None, type=str,
                        help="Path to txt dataset file (original format)")
    parser.add_argument("--problem_path", default="/home/jieyi/cvrp100_uniform.pkl", type=str,
                        help="Path to CVRP problem pkl file")
    parser.add_argument("--solution_path", default="/home/jieyi/hgs_cvrp100_uniform.pkl", type=str,
                        help="Path to HGS solution pkl file")
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
        problem_path=args.problem_path,
        solution_path=args.solution_path,
        time_limit=args.time_limit,
        problem_size=args.problem_size,
        index=args.index,
        problem_type=args.problem_type,
        scale=args.scale,
        n_vehicles=args.n_vehicles,
        use_callback=args.use_callback,
    )
