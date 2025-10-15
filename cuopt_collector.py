"""
CuOpt Trajectory Collector
Synchronously collects (state, action, reward) trajectories using fixed sampling policy
"""
import numpy as np
import cudf
from typing import Dict
from cuopt import routing
from cuopt.routing import CustomizeNodesCallback, RewardCallback


class CuOptCollector:
    """Synchronous trajectory collector for CuOpt"""
    
    def __init__(self, n_locations: int = 100, n_vehicles: int = 10, 
                 time_limit: float = 10.0, seed: int = 42):
        self.n_locations = n_locations
        self.n_vehicles = n_vehicles
        self.time_limit = time_limit
        self.rng = np.random.default_rng(seed)
        
        self.trajectory = {'states': [], 'actions': [], 'rewards': []}
        self.problem_data = None
        self.solution = None
        self.customize_cb = None
        self.reward_cb = None
    
    def reset(self) -> Dict:
        """Generate new problem and prepare solver"""
        self.problem_data = self._generate_problem()
        self.trajectory = {'states': [], 'actions': [], 'rewards': []}
        
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
    
    def _generate_problem(self) -> Dict:
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
    """Records state and generates action with adaptive sampling policy"""
    
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
        
        # Generate action: adaptive sampling (n<40→all, n<80→half, n≥80→40)
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
        
        self.collector.trajectory['actions'].append(action)
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
    collector = CuOptCollector(n_locations=100, n_vehicles=10, time_limit=2.0, seed=42)

    for i in range(3):
        dm, settings = collector.reset()
        result = collector.run_solver(dm, settings)
        
        if result['num_steps'] > 0:
            states = result['trajectory']['states']
            rewards = result['trajectory']['rewards']
            print(f"--------------------------------")
            print(f"Steps: {result['num_steps']}")
            print(f"Cost: {states[0]['solution_cost']:.2f} → {states[-1]['solution_cost']:.2f} "
                f"(Δ={states[0]['solution_cost'] - states[-1]['solution_cost']:.2f})")
            print(f"Total reward: {sum(rewards):.2f}")
            print(f"Avg candidates: {np.mean([s['num_candidates'] for s in states]):.1f}")
            print(f"number of states: {len(states)}")
            print(f"number of actions: {len(result['trajectory']['actions'])}")
            print(f"number of rewards: {len(result['trajectory']['rewards'])}")
            print(f"--------------------------------\n")
        else:
            print("No steps recorded (callback not triggered)")