import argparse
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
from model import Policy

def pairwise_euclidean_distance(x: torch.Tensor) -> torch.Tensor:
    x_square = (x ** 2).sum(dim=2, keepdim=True)
    dist_square = x_square + x_square.transpose(1, 2) - 2 * torch.bmm(x, x.transpose(1, 2))
    return torch.sqrt(torch.clamp(dist_square, min=1e-9))

class PolicyCustomizeNodesCallback(CustomizeNodesCallback):
    """Callback that uses Policy model to score and select best trail candidate."""
    def __init__(self, policy_model, coordinates, demand, vehicle_capacity, device='cpu'):
        super().__init__()
        self.policy = policy_model.eval()
        self.coordinates = coordinates.to(device)
        self.demand = (demand / vehicle_capacity).to(device)
        self.device = device

    # def customize_nodes_to_search(self, solution_flat, num_routes, 
    #                               solution_cost, candidate_mask, iter):
    #     """random change the trail candidates"""
    #     num_candidates = sum(candidate_mask)
    #     # print(f"[Callback] Num candidates: {num_candidates}")
    #     if num_candidates == 0:
    #         return [0] * len(candidate_mask)
        
    #     # 并行计算采样大小和生成 trail candidates
    #     sample_size = num_candidates if num_candidates < 40 else (num_candidates // 2 if num_candidates < 80 else 40)
        
    #     candidate_nodes = [node_id for node_id in range(len(candidate_mask)) if candidate_mask[node_id] == 1]
    #     selection_mask = np.zeros(len(candidate_mask), dtype=np.int32)
    #     sampled = random.sample(candidate_nodes,min(sample_size, len(candidate_nodes)))
    #     selection_mask[sampled] = 1
    #     selection_mask = selection_mask.tolist()
    #     return selection_mask

    def customize_nodes_to_search(self, solution_flat, num_routes, 
                                  solution_cost, candidate_mask, iter):
        """使用 Policy 对 100 个随机 trail candidates 并行打分，选择得分最高的"""
        num_candidates = sum(candidate_mask)
        # print(f"[Callback] Num candidates: {num_candidates}")
        if num_candidates == 0:
            return [0] * len(candidate_mask)
        
        # # 并行计算采样大小和生成 trail candidates
        max_length = len(candidate_mask)
        # sample_size = num_candidates if num_candidates < 40 else (num_candidates // 2 if num_candidates < 80 else 40)
        # num_samples = 100
        
        # # 并行生成所有 trail candidates
        # candidate_mask_t = torch.tensor(candidate_mask, dtype=torch.float32, device=self.device)
        # probs = candidate_mask_t.unsqueeze(0).expand(num_samples, -1)
        # sampled_indices = torch.multinomial(probs, sample_size, replacement=False)
        # selected = torch.zeros_like(probs, dtype=torch.int32)
        # selected.scatter_(1, sampled_indices, 1)  # (num_samples, max_length)

        # 并行准备所有 Policy 输入
        nodes_tensor = self.coordinates
        demands_tensor = self.demand.unsqueeze(-1)  # (1, N, 1)
        solution_flat_tensor = torch.tensor(solution_flat + [-1] * (max_length - len(solution_flat)), dtype=torch.long, device=self.device)
        solution_flat_tensor = solution_flat_tensor.unsqueeze(0)
        candidates_tensor = torch.tensor(candidate_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        # selected = selected.unsqueeze(0)  # (1, num_samples, max_length)

        # Policy 并行打分
        with torch.no_grad():
            scores = self.policy(nodes_tensor, demands_tensor, solution_flat_tensor, candidates_tensor)
            scores = torch.sigmoid(scores)
            scores = scores * candidates_tensor.view_as(scores)
            size_candidates = max(50, int(candidates_tensor.sum().item() * 0.8))
            threshold = torch.topk(scores, size_candidates)[0].min().item()
            pred = (scores >= threshold).bool()

        # 返回得分最高的 trail candidate
        # best_idx = scores.argmax(dim=-1).item()
        # return selected[0, best_idx].cpu().numpy().tolist()
        return pred[0].cpu().numpy().tolist()
        # return result.cpu().numpy().tolist()

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
):
    raw_nodes, raw_cap, raw_demand, raw_cost, raw_flag = load_raw_data(data_path, episode=problem_size, begin_index=index)
    raw_dist = pairwise_euclidean_distance(raw_nodes)

    # 自动检测 device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Device] Using {device}")
    
    # 加载 Policy 模型（如果需要）
    policy = None
    if use_callback:
        policy = Policy(device=device)
        if policy_model_path:
            print(f"[Callback] Loading Policy model from {policy_model_path}")
            policy.load_state_dict(torch.load(policy_model_path, map_location=device)['model_state_dict'])

    costs = []
    gaps = []

    for index in tqdm(range(problem_size), desc="Solving with cuOpt"):
        # 创建 callback（使用当前问题的数据）
        callback = None
        if use_callback:
            callback = PolicyCustomizeNodesCallback(policy, raw_nodes, raw_demand, raw_cap, device)
        
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
        policy_model_path=args.policy_model_path
    )
