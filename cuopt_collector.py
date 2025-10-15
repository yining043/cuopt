"""
CuOpt Trajectory Collector
Synchronously collects (state, action, reward) trajectories using fixed sampling policy
"""
import numpy as np
import cudf
import torch
from cuopt import routing
from cuopt.routing import CustomizeNodesCallback, RewardCallback
from set_transformer_policy import NodeCandidateSelectionPolicy


class CuOptCollector:
    """Synchronous trajectory collector for CuOpt"""
    
    def __init__(self, n_locations=100, n_vehicles=10, 
                 time_limit=10.0, seed=42, policy=None, use_policy=False):
        self.n_locations = n_locations
        self.n_vehicles = n_vehicles
        self.time_limit = time_limit
        self.rng = np.random.default_rng(seed)
        
        self.trajectory = {'states': [], 'actions': [], 'rewards': [], 'logps': []}
        self.problem_data = None
        self.solution = None
        self.customize_cb = None
        self.reward_cb = None
        
        # Policy network
        self.use_policy = use_policy
        if use_policy and policy is None:
            self.policy = NodeCandidateSelectionPolicy(
                node_feature_dim=3,
                d_model=128,
                num_heads=4,
                num_graph_layers=2,
                num_sab_layers=2
            )
            self.policy.eval()
        else:
            self.policy = policy
    
    def reset(self):
        """Generate new problem and prepare solver"""
        self.problem_data = self._generate_problem()
        self.trajectory = {'states': [], 'actions': [], 'rewards': [], 'logps': []}
        
        dm = routing.DataModel(self.problem_data['n_locations'], 
                               self.problem_data['n_vehicles'])
        dm.add_cost_matrix(self.problem_data['cost_matrix'])
        dm.add_capacity_dimension("demand", self.problem_data['demand'], 
                                 self.problem_data['vehicle_capacity'])
        
        settings = routing.SolverSettings()
        settings.set_time_limit(self.time_limit)
        self.customize_cb = _CustomizeCallback(self)
        self.reward_cb = _RewardCallback(self)
        settings.set_routing_callback(self.customize_cb)
        settings.set_routing_callback(self.reward_cb)
        return dm, settings
    
    def run_solver(self, dm, settings):
        """Run solver and return collected trajectory"""
        self.solution = routing.Solve(dm, settings)
        return {
            'trajectory': self.trajectory,
            'solution': self.solution,
            'problem': self.problem_data,
            'num_steps': len(self.trajectory['states'])
        }
    
    def _generate_problem(self):
        """Generate random VRP (same as test_callback_minimal)"""
        n = self.n_locations
        coords = self.rng.random((n, 2)) * 100
        distances = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)
        demand = np.concatenate([[0], self.rng.integers(5, 15, n - 1)])
        vehicle_capacity = np.full(self.n_vehicles, 100, dtype=np.int32)
        
        return {
            'n_locations': n,
            'n_vehicles': self.n_vehicles,
            'cost_matrix': cudf.DataFrame(distances),
            'demand': cudf.Series(demand),
            'vehicle_capacity': cudf.Series(vehicle_capacity),
            'coordinates': coords
        }


class _CustomizeCallback(CustomizeNodesCallback):
    """Records state and generates action with adaptive sampling policy or neural network"""
    
    def __init__(self, collector):
        super().__init__()
        self.collector = collector
    
    def customize_nodes_to_search(self, routes_2d, candidate_node_ids, 
                                  solution_cost, num_routes):
        # Record state
        state = {
            'routes_2d': routes_2d,
            'candidate_node_ids': candidate_node_ids,
            'solution_cost': solution_cost,
            'num_routes': num_routes,
            'num_candidates': len(candidate_node_ids),
        }
        self.collector.trajectory['states'].append(state)
        
        # Generate action
        if self.collector.use_policy:
            # Use policy network
            selected_indices, logp = self.collector.policy.sample(
                state, 
                self.collector.problem_data,
                deterministic = False
            )
            action = selected_indices
            # print(f"logp: {logp.item() if torch.is_tensor(logp) else logp}")
            self.collector.trajectory['logps'].append(logp.item() if torch.is_tensor(logp) else logp)
        else:
            # Use adaptive sampling (n<40→all, n<80→half, n≥80→40)
            n = len(candidate_node_ids)
            if n == 0:
                action = []
            elif n < 40:
                sample_size = n
            elif n < 80:
                sample_size = n // 2
            else:
                sample_size = 40
            
            if n > 0 and sample_size > 0:
                action = self.collector.rng.choice(n, sample_size, replace=False).tolist()
            else:
                action = []
            self.collector.trajectory['logps'].append(0.0)
        
        self.collector.trajectory['actions'].append(action)
        print(f" {len(action)}/{len(candidate_node_ids)} action selected")
        return action


class _RewardCallback(RewardCallback):
    """Records reward as cost improvement"""
    
    def __init__(self, collector):
        super().__init__()
        self.collector = collector
    
    def receive_reward(self, improvement_found, solution_cost):
        prev_cost = self.collector.trajectory['states'][-1]['solution_cost']
        reward = float(prev_cost - solution_cost)
        self.collector.trajectory['rewards'].append(reward)


if __name__ == "__main__":
    import sys
    
    use_policy = "--policy" in sys.argv
    
    if use_policy:
        print("Using Policy Network")
        collector = CuOptCollector(n_locations=50, n_vehicles=5, time_limit=2.0, seed=42, use_policy=True)
    else:
        print("Using Random Sampling")
        collector = CuOptCollector(n_locations=50, n_vehicles=5, time_limit=2.0, seed=42, use_policy=False)

    for i in range(3):
        dm, settings = collector.reset()
        result = collector.run_solver(dm, settings)
        
        if result['num_steps'] > 0:
            states = result['trajectory']['states']
            rewards = result['trajectory']['rewards']
            logps = result['trajectory']['logps']
            print(f"--------------------------------")
            print(f"Episode {i+1}")
            print(f"Steps: {result['num_steps']}")
            print(f"Cost: {states[0]['solution_cost']:.2f} → {states[-1]['solution_cost']:.2f} "
                f"(Δ={states[0]['solution_cost'] - states[-1]['solution_cost']:.2f})")
            print(f"Total reward: {sum(rewards):.2f}")
            print(f"Avg candidates: {np.mean([s['num_candidates'] for s in states]):.1f}")
            if use_policy:
                print(f"Avg logp: {np.mean(logps):.4f}")
            print(f"--------------------------------\n")
        else:
            print("No steps recorded (callback not triggered)")