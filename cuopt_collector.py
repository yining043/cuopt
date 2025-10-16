"""
CuOpt Trajectory Collector
Synchronously collects (state, action, reward) trajectories using fixed sampling policy
"""
import numpy as np
import cudf
import torch
from cuopt import routing
from cuopt.routing import CustomizeNodesCallback, RewardCallback
from transformer_policy import TransformerCandidatePolicy


class CuOptCollector:
    """Synchronous trajectory collector for CuOpt"""
    
    def __init__(self, n_locations=100, n_vehicles=10, 
                 time_limit=10.0, seed=42, policy=None, use_policy=False, 
                 temperature=1.0, policy_device=None):
        self.n_locations = n_locations
        self.n_vehicles = n_vehicles
        self.time_limit = time_limit
        self.rng = np.random.default_rng(seed)
        self.temperature = temperature  # Sampling temperature
        
        self.trajectory = {'states': [], 'actions': [], 'rewards': [], 'logps': []}
        self.problem_data = None
        self.solution = None
        self.customize_cb = None
        self.reward_cb = None
        
        # Policy network
        self.use_policy = use_policy
        self.policy = policy
        if use_policy and policy is None:
            # Auto-detect device
            if policy_device is None:
                if torch.cuda.is_available():
                    # If multiple GPUs visible, use GPU 1 for policy (GPU 0 for CuOpt)
                    if torch.cuda.device_count() > 1:
                        policy_device = 'cuda:1'
                    else:
                        policy_device = 'cuda'  # Single GPU, share with CuOpt
                else:
                    policy_device = 'cpu'
            
            print(f"Policy device: {policy_device} (total GPUs: {torch.cuda.device_count()})")
            
            self.policy = TransformerCandidatePolicy(
                d_model=128,
                num_heads=8,
                num_encoder_layers=3,
                max_candidates=40,
                device=policy_device
            )
    
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
    
    def customize_nodes_to_search(self, solution_flat, candidate_mask, 
                                  solution_cost, num_routes):
        # Compute number of candidates
        num_candidates = sum(candidate_mask)
        assert num_candidates > 0, "No candidates"
        
        # Record state (without candidate_node_ids - redundant with candidate_mask)
        state = {
            'solution_flat': solution_flat,
            'candidate_mask': candidate_mask,
            'solution_cost': solution_cost,
            'num_routes': num_routes,
            'num_candidates': num_candidates,
        }
        self.collector.trajectory['states'].append(state)
        
        # Use adaptive sampling
        if num_candidates < 40:
            sample_size = num_candidates
        else:
            sample_size = 40
        
        # Generate action (returns selection_mask)
        if self.collector.use_policy:
            # Use policy network (respects policy.training state)
            # Always sample, never greedy
            selection_mask, _logp = self.collector.policy.sample(
                state, 
                self.collector.problem_data,
                sample_size,
                temperature=self.collector.temperature
            )
            logp = _logp.item() if torch.is_tensor(_logp) else _logp
        else:
            # Random sampling: extract candidates locally
            candidate_nodes = [node_id for node_id in range(len(candidate_mask)) if candidate_mask[node_id] == 1]
            selection_mask = np.zeros(len(candidate_mask), dtype=np.int32)
            sampled = self.collector.rng.choice(candidate_nodes, sample_size, replace=False)
            selection_mask[sampled] = 1
            selection_mask = selection_mask.tolist()
            logp = 0.0

        self.collector.trajectory['logps'].append(logp)
        self.collector.trajectory['actions'].append(selection_mask)
        return selection_mask  # Return selection mask (fixed length)


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
        collector = CuOptCollector(n_locations=50, n_vehicles=5, time_limit=10.0, seed=42, use_policy=True)
        # Set policy to eval mode for inference
        collector.policy.eval()
    else:
        print("Using Random Sampling")
        collector = CuOptCollector(n_locations=50, n_vehicles=5, time_limit=10.0, seed=42, use_policy=False)

    for i in range(3):
        dm, settings = collector.reset()
        collector.run_solver(dm, settings)  
        
        if len(collector.trajectory['states']) > 0:
            states = collector.trajectory['states']
            rewards = collector.trajectory['rewards']
            actions = collector.trajectory['actions']
            logps = collector.trajectory['logps']
            print(f"--------------------------------")
            print(f"Episode {i+1}")
            print(f"Steps: {len(states)}")
            print(f"Cost: {states[0]['solution_cost']:.2f} → {states[-1]['solution_cost']:.2f} "
                f"(Δ={states[0]['solution_cost'] - states[-1]['solution_cost']:.2f})")
            print(f"Total reward: {sum(rewards):.2f}")
            print(f"Avg candidates: {np.mean([s['num_candidates'] for s in states]):.1f}")
            if use_policy:
                print(f"Avg logp: {np.mean(logps):.4f}")
            # now print the number of state, reward, logp
            print(f"Number of states: {len(states)}")
            print(f"Number of rewards: {len(rewards)}")
            print(f"Number of actions: {len(actions)}")
            if use_policy: print(f"Number of logps: {len(logps)}")
            print(f"--------------------------------\n")
        else:
            print("No steps recorded (callback not triggered)")