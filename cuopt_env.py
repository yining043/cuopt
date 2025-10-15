import threading
import numpy as np
import cudf
from typing import Tuple, Dict, List
from cuopt import routing
from cuopt.routing import CustomizeNodesCallback, RewardCallback


class CuOptEnv:
    """Minimal RL Environment for CuOpt - generates new random problem each reset"""
    
    def __init__(self, n_locations: int = 100, n_vehicles: int = 10, 
                 time_limit: float = 5.0, seed: int = 42):
        self.n_locations = n_locations
        self.n_vehicles = n_vehicles
        self.time_limit = time_limit
        self.rng = np.random.default_rng(seed)
        
        self.state_ready = threading.Event()
        self.action_ready = threading.Event()
        self.reward_ready = threading.Event()
        
        self.state = None
        self.action = None
        self.current_reward = 0.0
        self.done = False
        self.thread = None
        self.solution = None
        self.problem_data = None
        self.sync_errors = []
    
    def reset(self) -> Dict:
        """Reset with new random problem"""
        if self.thread:
            self.done = True
            self.action_ready.set()
            self.thread.join(timeout=2.0)
        
        self.problem_data = self._generate_problem()
        self.current_reward = 0.0
        self.done = False
        self.sync_errors = []
        self.state_ready.clear()
        self.action_ready.clear()
        self.reward_ready.clear()
        
        self.thread = threading.Thread(target=self._run_solver)
        self.thread.start()
        
        if not self.state_ready.wait(timeout=10):
            self.sync_errors.append("reset: state_ready timeout")
        self.state_ready.clear()
        return self.state if self.state else {}
    
    def _generate_problem(self) -> Dict:
        """Generate random VRP (same as test_callback_minimal)"""
        n, v = self.n_locations, self.n_vehicles
        
        coords = self.rng.random((n, 2)) * 100
        dists = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
        demands = np.concatenate([[0], self.rng.integers(5, 15, n-1)])
        
        return {
            'cost_matrix': cudf.DataFrame(dists),
            'demand': cudf.Series(demands),
            'vehicle_capacity': cudf.Series([100] * v),
            'n_locations': n,
            'n_vehicles': v
        }
    
    def step(self, action: List[int]) -> Tuple[Dict, float, bool, Dict]:
        """Execute action and get next state"""
        self.action = action
        self.action_ready.set()
        
        if not self.state_ready.wait(timeout=1):
            self.sync_errors.append("step: state_ready timeout")
        self.state_ready.clear()
        
        if not self.reward_ready.wait(timeout=1):
            self.sync_errors.append("step: reward_ready timeout")
        self.reward_ready.clear()
        
        return self.state, self.current_reward, self.done, {}
    
    def _run_solver(self):
        """Solver thread"""
        print("[SOLVER] Starting...")
        dm = routing.DataModel(self.problem_data['n_locations'], 
                               self.problem_data['n_vehicles'])
        dm.add_cost_matrix(self.problem_data['cost_matrix'])
        dm.add_capacity_dimension("demand", self.problem_data['demand'], 
                                 self.problem_data['vehicle_capacity'])
        
        settings = routing.SolverSettings()
        settings.set_time_limit(self.time_limit)
        
        customize_cb = _CustomizeCallback(self)
        reward_cb = _RewardCallback(self)
        settings.set_routing_callback(customize_cb)
        settings.set_routing_callback(reward_cb)
        
        self.solution = routing.Solve(dm, settings)
        self.done = True
        self.state_ready.set()
    
    def close(self):
        """Wait for solver to finish and report sync errors"""
        if self.thread:
            self.thread.join(timeout=10.0)
        
        if self.sync_errors:
            print(f"⚠️  Sync timeouts: {len(self.sync_errors)}")
            for err in self.sync_errors:
                print(f"  - {err}")


class _CustomizeCallback(CustomizeNodesCallback):
    def __init__(self, env):
        super().__init__()
        self.env = env
    
    def customize_nodes_to_search(self, routes_2d, candidate_node_ids, 
                                  solution_cost, num_routes):  
        self.env.state = {
            'routes_2d': routes_2d,
            'candidate_node_ids': candidate_node_ids,
            'solution_cost': solution_cost,
            'num_routes': num_routes,
            'num_candidates': len(candidate_node_ids)
        }
        self.env.state_ready.set()
        
        if not self.env.action_ready.wait(timeout=5):
            self.env.sync_errors.append("customize: action_ready timeout")
            return []
        
        self.env.action_ready.clear()
        return self.env.action


class _RewardCallback(RewardCallback):
    def __init__(self, env):
        super().__init__()
        self.env = env
    
    def receive_reward(self, improvement_found, solution_cost):
        if self.env.state:
            self.env.current_reward = float(self.env.state['solution_cost'] - solution_cost)
            self.env.reward_ready.set()


# Example
if __name__ == "__main__":
    print("Creating env...")
    env = CuOptEnv(n_locations=100, n_vehicles=10, time_limit=10.0, seed=42)
    
    print("Resetting env...")
    state = env.reset()
    
    step_count = 0
    while not env.done:
        n = state.get('num_candidates', 0)        
        size = min(40, n//2 if n >= 80 else n)
        action = env.rng.choice(n, size, replace=False).tolist() if size > 0 else []
        state, reward, done, _ = env.step(action)
        step_count += 1
    print(f"Done: {step_count} steps, final={state.get('solution_cost', 0):.2f}")
    
    env.close()