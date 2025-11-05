"""
CuOpt Trajectory Collector
Synchronously collects (state, action, reward) trajectories using fixed sampling policy
"""
from hmac import new
import numpy as np
import cudf
import torch
import time
import random
import tqdm
import argparse
from cuopt import routing
from cuopt.routing import (
    CustomizeNodesCallback, 
    RewardCallback,
    LocalSearchStartCallback,
    BeforeCycleFinderCallback,
    AfterCycleFinderCallback
)
from transformer_policy import TransformerCandidatePolicy
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

class Problem:
    """Unified problem generator for VRP instances"""
    
    def __init__(self, n_locations=100, n_vehicles=50, 
                 coordinate_range=100.0, capacity=100.0, 
                 demand_range=(1, 10)):
        self.n_locations = n_locations
        self.n_vehicles = n_vehicles
        self.coordinate_range = coordinate_range
        self.capacity = capacity
        self.demand_range = demand_range
        self._problem_data = None
    
    def generate(self):
        n = self.n_locations
        coords = np.random.random((n, 2)) * self.coordinate_range
        distances = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)
        demand = np.concatenate([[0], np.random.randint(low=self.demand_range[0], high=self.demand_range[1] + 1, size=n - 1)])
        vehicle_capacity = np.full(self.n_vehicles, self.capacity, dtype=np.int32)
        
        self._problem_data = {
            'n_locations': n,
            'n_vehicles': self.n_vehicles,
            'cost_matrix': cudf.DataFrame(distances),
            'demand': cudf.Series(demand),
            'vehicle_capacity': cudf.Series(vehicle_capacity),
            'coordinates': coords,
            'problem_scale': self.coordinate_range,
            'capacity_scale': self.capacity
        }
        
        return self._problem_data
    
    def create_data_model(self, problem_data=None):
        """
        Create CuOpt DataModel from problem data
        """
        if problem_data is None:
            if self._problem_data is None:
                self.generate()
            problem_data = self._problem_data
        
        dm = routing.DataModel(problem_data['n_locations'], 
                              problem_data['n_vehicles'])
        dm.add_cost_matrix(problem_data['cost_matrix'])
        dm.add_capacity_dimension("demand", problem_data['demand'], 
                                 problem_data['vehicle_capacity'])
        return dm
    
    def get_problem_data(self):
        """Get cached problem data"""
        if self._problem_data is None:
            self.generate()
        return self._problem_data


class CuOptCollector:
    """Synchronous trajectory collector for CuOpt"""
    
    def __init__(self, use_policy=False, 
                 policy_device=None, checkpoint_path=None):
    
        # Policy network
        self.use_policy = use_policy
        self.policy = None
        self.policy_device = policy_device
        self.checkpoint_path = checkpoint_path
        
        # Auto-detect device if use_policy is True
        if use_policy and policy_device is None:
            if torch.cuda.is_available():
                # If multiple GPUs visible, use GPU 1 for policy (GPU 0 for CuOpt)
                if torch.cuda.device_count() > 1:
                    self.policy_device = 'cuda:1'
                else:
                    self.policy_device = 'cuda'  # Single GPU, share with CuOpt
            else:
                self.policy_device = 'cpu'

    
    def run_solver(self, problem_data, time_limit):
        self.problem_data = problem_data
        self.time_limit = time_limit
        
        # Initialize problem generator if needed
        n_locs = problem_data['n_locations']
        n_vehs = problem_data['n_vehicles']
        if not hasattr(self, '_problem_generator') or self._problem_generator is None:
            self._problem_generator = Problem(n_locations=n_locs, n_vehicles=n_vehs)
        
        # Initialize policy if use_policy is True
        if self.use_policy and self.policy is None:
            if self.policy_device is None:
                if torch.cuda.is_available():
                    if torch.cuda.device_count() > 1:
                        self.policy_device = 'cuda:1'
                    else:
                        self.policy_device = 'cuda'
                else:
                    self.policy_device = 'cpu'
            
            print(f"Policy device: {self.policy_device} (total GPUs: {torch.cuda.device_count()})")
            
            self.policy = TransformerCandidatePolicy(
                d_model=64,
                num_heads=4,
                num_encoder_layers=3,
                max_vehicles=n_vehs,
                N=n_locs,
                problem_scale=100.0,
                capacity_scale=50.0,
                device=self.policy_device
            )
            
            # Load checkpoint if provided
            if self.checkpoint_path is not None:
                checkpoint = torch.load(self.checkpoint_path, weights_only=False)
                self.policy.load_state_dict(checkpoint['policy_state_dict'])
                print(f"Loaded trained policy from: {self.checkpoint_path}")
                print(f"  Trained for {checkpoint.get('epoch', 0)} epochs")
                print(f"  Best cost: {checkpoint.get('best_cost', 0):.2f}")
            
            self.policy.eval()
        elif self.use_policy and self.policy is not None:
            # Check if dimensions match
            if self.policy.N != n_locs or self.policy.max_vehicles != n_vehs:
                raise ValueError(
                    f"Policy dimensions mismatch: policy expects N={self.policy.N}, max_vehicles={self.policy.max_vehicles}, "
                    f"but problem has n_locations={n_locs}, n_vehicles={n_vehs}"
                )
        
        self.global_history = {
            'history': [],
            'current_local_search_id': -1,
            'current_global_iter': -1,
            'current_local_iter': -1,
            'pending_state': None,
            'initial_cost': None,
            'local_best_cost_so_far': None
        }
        
        # Reset performance tracking
        self.policy_sample_time = 0.0
        self.policy_call_count = 0
        
        dm = self._problem_generator.create_data_model(self.problem_data)
        
        settings = routing.SolverSettings()
        settings.set_time_limit(self.time_limit)
        self.customize_cb = _CustomizeCallback(self)
        self.reward_cb = _RewardCallback(self)
        self.start_cb = _LocalSearchStartCallback(self)
        self.before_cb = _BeforeCycleFinderCallback(self)
        self.after_cb = _AfterCycleFinderCallback(self)
        settings.set_routing_callback(self.customize_cb)
        settings.set_routing_callback(self.reward_cb)
        settings.set_routing_callback(self.start_cb)
        settings.set_routing_callback(self.before_cb)
        settings.set_routing_callback(self.after_cb)

        self.solution = routing.Solve(dm, settings)
        return self.solution
    
class _CustomizeCallback(CustomizeNodesCallback):
    """Records state and generates action with adaptive sampling policy or neural network"""
    
    def __init__(self, collector):
        super().__init__()
        self.collector = collector
    
    def customize_nodes_to_search(self, solution_flat, num_routes, 
                                  solution_cost, candidate_mask, iter):
        local_iter = self.collector.global_history['current_local_iter'] = self.collector.global_history['current_local_iter'] + 1
        global_iter = self.collector.global_history['current_global_iter'] = self.collector.global_history['current_global_iter'] + 1
        local_search_id = self.collector.global_history['current_local_search_id']
        
        # Compute number of candidates
        num_candidates = sum(candidate_mask)
        if num_candidates < 40:
            sample_size = num_candidates
        elif num_candidates < 80:
            sample_size = num_candidates // 2
        else:
            sample_size = 40

        # Store before state in pending_state
        self.collector.global_history['pending_state'] = {
            'local_search_id': local_search_id,
            'global_iter': global_iter,
            'local_iter': local_iter,
            'sol_before': solution_flat.copy(),
            'num_routes_before': num_routes,
            'cost_before': solution_cost,
            'candidate_mask': candidate_mask.copy(),
            'num_candidates': num_candidates,
            'is_circle_found': False
        }
        
        if self.collector.use_policy:
            policy_start = time.perf_counter()

            # Prepare state for policy
            state = {
                'solution_flat': solution_flat,
                'candidate_mask': candidate_mask,
                'solution_cost': solution_cost,
                'num_routes': num_routes,
                'num_candidates': num_candidates,
                'sample_size': sample_size,
            }        
            with torch.no_grad():
                selected_node_ids, log_prob = self.collector.policy(
                    state, 
                    self.collector.problem_data,
                    temperature=1.0
                )
            
            policy_end = time.perf_counter()
            self.collector.policy_sample_time += (policy_end - policy_start)
            self.collector.policy_call_count += 1
            
            selected_node_ids = selected_node_ids[0].tolist()
            selection_mask = np.zeros(len(candidate_mask), dtype=np.int32)
            selection_mask[selected_node_ids] = 1
            selection_mask = selection_mask.tolist()
            logp = log_prob.item()
            selected_sequence = selected_node_ids
        else:
            # Random sampling
            candidate_nodes = [node_id for node_id in range(len(candidate_mask)) if candidate_mask[node_id] == 1]
            selection_mask = np.zeros(len(candidate_mask), dtype=np.int32)
            sampled = random.sample(candidate_nodes, min(sample_size, len(candidate_nodes)))
            selection_mask[sampled] = 1
            selection_mask = selection_mask.tolist()
            logp = 0.0
            selected_sequence = sampled
        
        # Add policy outputs to pending_state
        self.collector.global_history['pending_state']['action'] = selection_mask
        self.collector.global_history['pending_state']['logp'] = logp
        self.collector.global_history['pending_state']['selected_sequence'] = selected_sequence
        
        return selection_mask


class _RewardCallback(RewardCallback):
    """Records reward as cost improvement with historical best tracking"""
    
    def __init__(self, collector):
        super().__init__()
        self.collector = collector
    
    def receive_reward(self, improvement_found, solution_cost, iter, solution_flat, num_routes):
        # Update both best_cost_so_far (local and global)
        initial_cost = self.collector.global_history['initial_cost']
        old_bsf = self.collector.global_history['local_best_cost_so_far']
        new_bsf = min(old_bsf, solution_cost)
        self.collector.global_history['local_best_cost_so_far'] = new_bsf
        

        local_bsf_improvement = old_bsf - new_bsf
        pending = self.collector.global_history['pending_state']
        step_improvement = pending['cost_before'] - solution_cost
        reward = local_bsf_improvement / pending['cost_before'] * pending['local_iter']
        print(f"Local Search Reward: {reward}")
        
        record = {
            'local_search_id': pending['local_search_id'],
            'global_iter': pending['global_iter'],
            'local_iter': pending['local_iter'],
            'sol_before': pending['sol_before'],
            'sol_after': solution_flat.copy(),
            'num_routes_before': pending['num_routes_before'],
            'num_routes_after': num_routes,
            'cost_before': pending['cost_before'],
            'cost_after': solution_cost,
            'move_found': improvement_found,
            'is_circle_found': False,
            'action': pending.get('action'),
            'logp': pending.get('logp'),
            'selected_sequence': pending.get('selected_sequence'),
            'candidate_mask': pending.get('candidate_mask'),
            'num_candidates': pending.get('num_candidates'),
            'initial_cost': initial_cost,
            'local_best_cost_so_far': new_bsf,
            'local_bsf_improvement': local_bsf_improvement,
            'step_improvement': step_improvement,
            'reward': reward
        }
        
        self.collector.global_history['history'].append(record)
        self.collector.global_history['pending_state'] = None


class _LocalSearchStartCallback(LocalSearchStartCallback):
    def __init__(self, collector):
        super().__init__()
        self.collector = collector
    
    def on_local_search_start(self, solution_flat, num_routes, solution_cost, weights, selection_weights, should_all_nodes_be_served):
        self.collector.global_history['current_local_search_id'] = self.collector.global_history['current_local_search_id'] + 1
        self.collector.global_history['current_local_iter'] = -1
        self.collector.global_history['initial_cost'] = solution_cost
        self.collector.global_history['local_best_cost_so_far'] = solution_cost
        if not should_all_nodes_be_served:
            print("Not all nodes are served")
            # assert False, "Not all nodes are served"
        print(f"Local Search Start: {solution_cost}")


class _BeforeCycleFinderCallback(BeforeCycleFinderCallback):
    def __init__(self, collector):
        super().__init__()
        self.collector = collector
    
    def on_before_cycle_finder(self, solution_flat, num_routes, solution_cost, iter):
        local_iter = self.collector.global_history['current_local_iter'] = self.collector.global_history['current_local_iter'] + 1
        global_iter = self.collector.global_history['current_global_iter'] = self.collector.global_history['current_global_iter'] + 1
        local_search_id = self.collector.global_history['current_local_search_id']
        self.collector.global_history['pending_state'] = {
            'local_search_id': local_search_id,
            'global_iter': global_iter,
            'local_iter': local_iter,
            'sol_before': solution_flat.copy(),
            'num_routes_before': num_routes,
            'cost_before': solution_cost,
            'is_circle_found': True
        }


class _AfterCycleFinderCallback(AfterCycleFinderCallback):
    def __init__(self, collector):
        super().__init__()
        self.collector = collector
    
    def on_after_cycle_finder(self, solution_flat, num_routes, solution_cost, iter, improved):
        initial_cost = self.collector.global_history['initial_cost']
        old_bsf = self.collector.global_history['local_best_cost_so_far']
        new_bsf = min(old_bsf, solution_cost)
        self.collector.global_history['local_best_cost_so_far'] = new_bsf

        local_bsf_improvement = old_bsf - new_bsf
        pending = self.collector.global_history['pending_state']
        step_improvement = pending['cost_before'] - solution_cost
        reward = local_bsf_improvement / pending['cost_before'] * pending['local_iter']
        print(f"Cycle Finder Reward: {reward}")

        record = {
            'local_search_id': pending['local_search_id'],
            'global_iter': pending['global_iter'],
            'local_iter': pending['local_iter'],
            'sol_before': pending['sol_before'],
            'sol_after': solution_flat.copy(),
            'num_routes_before': pending['num_routes_before'],
            'num_routes_after': num_routes,
            'cost_before': pending['cost_before'],
            'cost_after': solution_cost,
            'move_found': improved,
            'is_circle_found': True,
            'action': None,
            'logp': None,
            'selected_sequence': None,
            'candidate_mask': None,
            'num_candidates': None,
            'initial_cost': self.collector.global_history['initial_cost'],
            'local_best_cost_so_far': new_bsf,
            'local_bsf_improvement': local_bsf_improvement,
            'step_improvement': step_improvement,
            'reward': reward
        }
        self.collector.global_history['history'].append(record)
        self.collector.global_history['pending_state'] = None


def plot_reward_vs_cost(episode_avg_rewards, episode_final_costs, all_rewards, all_local_bsf_improvements, save_path='reward_vs_cost.png'):
    """Plot reward vs cost analysis from collected data"""
    print(f"Plotting {len(episode_final_costs)} episodes, {len(all_rewards)} steps")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Episode-level: Avg Reward vs Final Cost
    ax1 = axes[0]
    ax1.scatter(episode_avg_rewards, episode_final_costs, alpha=0.6, s=60, c='blue', edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Avg Reward (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Final Cost', fontsize=12, fontweight='bold')
    ax1.set_title('Episode: Avg Reward vs Final Cost', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    if len(episode_final_costs) > 2:
        corr1, p_val1 = pearsonr(episode_avg_rewards, episode_final_costs)
        ax1.text(0.05, 0.95, f'r={corr1:.3f}\np={p_val1:.2e}', 
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        print(f"Episode correlation: r={corr1:.3f}, p={p_val1:.2e}")
    
    # Plot 2: Step-level: Reward vs Local BSF Improvement
    ax2 = axes[1]
    ax2.scatter(all_rewards, all_local_bsf_improvements, alpha=0.4, s=30, c='steelblue', edgecolors='navy', linewidth=0.3)
    ax2.set_xlabel('Reward (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Local BSF Improvement', fontsize=12, fontweight='bold')
    ax2.set_title('Step: Reward vs Local BSF Improvement', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    if len(all_rewards) > 2:
        corr2, p_val2 = pearsonr(all_rewards, all_local_bsf_improvements)
        ax2.text(0.05, 0.95, f'r={corr2:.3f}\np={p_val2:.2e}', 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        print(f"Step correlation: r={corr2:.3f}, p={p_val2:.2e}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CuOpt Collector Test')
    parser.add_argument('--policy', action='store_true', help='Use policy network')
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='Path to trained policy checkpoint')
    parser.add_argument('--plot', action='store_true', help='Generate reward vs cost plot')
    parser.add_argument('--n_locations', type=int, default=1000, help='Number of locations')
    parser.add_argument('--n_vehicles', type=int, default=300, help='Number of vehicles')
    parser.add_argument('--capacity', type=float, default=200.0, help='Capacity')
    parser.add_argument('--n_episodes', type=int, default=3, help='Number of episodes to run')
    parser.add_argument('--time_limit', type=float, default=10.0, help='Time limit per episode')
    args = parser.parse_args()
    
    collector = CuOptCollector(use_policy=args.policy, checkpoint_path=args.checkpoint)
    if collector.policy is not None:
        collector.policy.eval()
    
    print(f"Running {args.n_episodes} episodes with {args.n_locations} locations, {args.n_vehicles} vehicles")
    if args.plot:
        print("Collecting data for plotting...")
    print("=" * 80)
    
    # Collect data
    episode_avg_rewards = []
    episode_final_costs = []
    all_rewards = []
    all_local_bsf_improvements = []
    
    for i in tqdm.tqdm(range(args.n_episodes)) if args.plot else range(args.n_episodes):
        problem_data = Problem(n_locations=args.n_locations, n_vehicles=args.n_vehicles).generate()
        collector.run_solver(problem_data, args.time_limit)
        
        history = collector.global_history['history']
        if len(history) > 0:
            rewards = [r['reward'] for r in history if r.get('reward') is not None]
            local_bsf_improvements = [r['local_bsf_improvement'] for r in history if r.get('local_bsf_improvement') is not None]
            initial = history[0]['initial_cost']
            final = history[-1]['local_best_cost_so_far']
            
            # Collect data for plotting
            if args.plot:
                episode_avg_rewards.append(np.mean(rewards) if rewards else 0.0)
                episode_final_costs.append(final)
                all_rewards.extend(rewards)
                all_local_bsf_improvements.extend(local_bsf_improvements)
            
            # Print episode summary
            if not args.plot:
                print(f"Episode {i+1}: Steps={len(history)}, "
                      f"Cost {initial:.1f}→{final:.1f}, "
                      f"AvgReward={np.mean(rewards):.2f}")
                
                if args.policy and collector.policy_call_count > 0:
                    avg_time = collector.policy_sample_time / collector.policy_call_count * 1000
                    print(f"  Policy time: {avg_time:.2f}ms/step")
        else:
            if not args.plot:
                print(f"Episode {i+1}: No steps recorded")
    
    # Plot if requested
    if args.plot:
        plot_reward_vs_cost(episode_avg_rewards, episode_final_costs, 
                           all_rewards, all_local_bsf_improvements,
                           save_path='reward_vs_cost.png')