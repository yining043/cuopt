"""
CuOpt Trajectory Collector
Synchronously collects (state, action, reward) trajectories using fixed sampling policy
"""
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
        seed = 42
        np.random.seed(seed)
        random.seed(seed)
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
                 policy_device=None, checkpoint_path=None, policy = None):
    
        # Policy network
        self.use_policy = use_policy
        self.policy = policy
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

    
    def run_solver(self, problem_data, dm, time_limit):
        # Store problem_data for callback access
        self.problem_data = problem_data
        
        # Get problem data
        n_locs = problem_data['n_locations']
        n_vehs = problem_data['n_vehicles'] 

        # Initialize policy if use_policy is True
        if self.use_policy and self.policy is None:
            print("Initializing policy from scratch")

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
            
        elif self.use_policy and self.policy is not None:
            # Check if dimensions match
            if self.policy.N != n_locs or self.policy.max_vehicles != n_vehs:
                raise ValueError(
                    f"Policy dimensions mismatch: policy expects N={self.policy.N}, max_vehicles={self.policy.max_vehicles}, "
                    f"but problem has n_locations={n_locs}, n_vehicles={n_vehs}"
                )

        self.global_history = {
            'history': [],
            'initial_cost_set': [],
            'should_all_nodes_be_served_set': [],
            'local_bsf_set': [],
            'return_set': [],
            'global_bsf': None,
            'current_local_search_id': -1,
            'current_global_iter': -1,
            'current_local_iter': -1,
            'pending_state': None,
        }
        
        # Reset performance tracking
        if self.use_policy:
            self.policy_sample_time = 0.0
            self.policy_call_count = 0
            self.policy.eval()

        settings = routing.SolverSettings()
        settings.set_time_limit(time_limit)
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
        # if num_candidates < 40:
        #     sample_size = num_candidates
        # elif num_candidates < 80:
        #     sample_size = num_candidates // 2
        # else:
        #     sample_size = 40
        
        sample_size = min(num_candidates, 10)

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
            'is_circle_found': False,
            'sample_size': sample_size,
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
    
    def receive_reward(self, improvement_found, solution_cost, trail_cost, iter, solution_flat, num_routes):
        # Update both best_cost_so_far (local and global)
        current_local_search_id = self.collector.global_history['current_local_search_id']
        old_bsf = self.collector.global_history['local_bsf_set'][current_local_search_id]
        new_bsf = min(old_bsf, solution_cost)
        self.collector.global_history['local_bsf_set'][current_local_search_id] = new_bsf
        old_global_bsf = self.collector.global_history['global_bsf']
        new_global_bsf = min(old_global_bsf, new_bsf)
        self.collector.global_history['global_bsf'] = new_global_bsf

        local_bsf_improvement = old_bsf - new_bsf
        global_bsf_improvement = old_global_bsf - new_global_bsf
        pending = self.collector.global_history['pending_state']
        step_improvement = pending['cost_before'] - solution_cost
        ### Reward: use pure local BSF improvement (no scaling by iter or initial cost)
        reward = (trail_cost - solution_cost) 

        # print(f"Reward: {reward}, solution_cost: {solution_cost}, trail_cost: {trail_cost}")
        self.collector.global_history['return_set'][current_local_search_id] += reward

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
            'trail_cost': trail_cost,
            'move_found': improvement_found,
            'is_circle_found': False,
            'action': pending.get('action'),
            'logp': pending.get('logp'),
            'selected_sequence': pending.get('selected_sequence'),
            'candidate_mask': pending.get('candidate_mask'),
            'num_candidates': pending.get('num_candidates'),
            'local_bsf_improvement': local_bsf_improvement,
            'global_bsf_improvement': global_bsf_improvement,
            'step_improvement': step_improvement,
            'reward': reward,
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
        self.collector.global_history['initial_cost_set'].append(solution_cost)
        self.collector.global_history['local_bsf_set'].append(solution_cost)
        self.collector.global_history['return_set'].append(0.0)
        if self.collector.global_history['global_bsf'] is None:
            self.collector.global_history['global_bsf'] = solution_cost
        else:
            self.collector.global_history['global_bsf'] = min(
                self.collector.global_history['global_bsf'], solution_cost
            )
        self.collector.global_history['should_all_nodes_be_served_set'].append(should_all_nodes_be_served)
        if not should_all_nodes_be_served:
            print("Not all nodes are served")
            assert False, "Not all nodes are served"


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
        current_local_search_id = self.collector.global_history['current_local_search_id']
        old_bsf = self.collector.global_history['local_bsf_set'][current_local_search_id]
        new_bsf = min(old_bsf, solution_cost)
        self.collector.global_history['local_bsf_set'][current_local_search_id] = new_bsf
        old_global_bsf = self.collector.global_history['global_bsf']
        new_global_bsf = min(old_global_bsf, new_bsf)
        self.collector.global_history['global_bsf'] = new_global_bsf

        local_bsf_improvement = old_bsf - new_bsf
        global_bsf_improvement = old_global_bsf - new_global_bsf
        pending = self.collector.global_history['pending_state']
        step_improvement = pending['cost_before'] - solution_cost
        
        ### Reward: use pure local BSF improvement (no scaling by iter or initial cost)
        reward = 0
        self.collector.global_history['return_set'][current_local_search_id] += reward

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
            'local_bsf_improvement': local_bsf_improvement,
            'global_bsf_improvement': global_bsf_improvement,
            'step_improvement': step_improvement,
            'reward': reward,
        }
        self.collector.global_history['history'].append(record)
        self.collector.global_history['pending_state'] = None


def plot_reward_vs_cost(history, save_path='reward_vs_cost.png'):
    """Plot figures
    subfigure1: avg return vs final_bsf analysis from history data
    subfigure2: avg reward vs final_bsf analysis from history data
    subfigure3: return vs local_bsf analysis from history data
    subfigure4: reward vs local_bsf analysis from history data
    """
    if not history:
        print("No history data to plot")
        return
    
    # Extract data for each episode (for subplots 1 and 2)
    episodes_avg_return = []
    episodes_avg_reward = []
    episodes_final_bsf = []
    episodes_avg_local_bsf = []
    
    # Extract data for each step (for subplots 3 and 4)
    all_returns = []
    all_rewards = []
    all_local_bsf = []
    
    for ep_history in history:
        # Extract episode-level data
        final_bsf = ep_history.get('global_bsf')
        return_set = ep_history.get('return_set', [])
        local_bsf_set = ep_history.get('local_bsf_set', [])
        history_records = ep_history.get('history', [])
        
        # Calculate averages for episode-level plots
        avg_return = np.mean(return_set) if return_set else 0.0
        avg_local_bsf = np.mean(local_bsf_set) if local_bsf_set else 0.0
        
        # Extract rewards from history records
        rewards = [r.get('reward', 0) for r in history_records if r.get('reward') is not None]
        avg_reward = np.mean(rewards) if rewards else 0.0
        
        # Store episode data
        episodes_avg_return.append(avg_return)
        episodes_avg_reward.append(avg_reward)
        episodes_final_bsf.append(final_bsf)
        episodes_avg_local_bsf.append(avg_local_bsf)
        
        # Extract step-level data for subplots 3 and 4
        for record in history_records:
            local_search_id = record.get('local_search_id')
            reward = record.get('reward')
            local_bsf_improvement = record.get('local_bsf_improvement')
            
            # Get cumulative return for this local_search_id
            if local_search_id is not None and local_search_id < len(return_set):
                current_return = return_set[local_search_id]
            else:
                current_return = None
            
            # Only add if we have valid data
            if reward is not None and local_bsf_improvement is not None and current_return is not None:
                all_returns.append(current_return)
                all_rewards.append(reward)
                all_local_bsf.append(local_bsf_improvement)
    
    # Convert to numpy arrays
    episodes_avg_return = np.array(episodes_avg_return)
    episodes_avg_reward = np.array(episodes_avg_reward)
    episodes_final_bsf = np.array(episodes_final_bsf)
    episodes_avg_local_bsf = np.array(episodes_avg_local_bsf)
    
    all_returns = np.array(all_returns)
    all_rewards = np.array(all_rewards)
    all_local_bsf = np.array(all_local_bsf)
    
    print(f"Plotting {len(history)} episodes, {len(all_returns)} steps")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: avg return vs final_bsf
    ax1 = axes[0, 0]
    ax1.scatter(episodes_avg_return, episodes_final_bsf, alpha=0.6, s=60, c='blue', edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Avg Return', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Final BSF', fontsize=12, fontweight='bold')
    ax1.set_title('Avg Return vs Final BSF', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    if len(episodes_avg_return) > 2:
        corr1, p_val1 = pearsonr(episodes_avg_return, episodes_final_bsf)
        ax1.text(0.05, 0.95, f'r={corr1:.3f}\np={p_val1:.2e}', 
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        print(f"Avg Return vs Final BSF: r={corr1:.3f}, p={p_val1:.2e}")
    
    # Subplot 2: avg reward vs final_bsf
    ax2 = axes[0, 1]
    ax2.scatter(episodes_avg_reward, episodes_final_bsf, alpha=0.6, s=60, c='green', edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Avg Reward', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Final BSF', fontsize=12, fontweight='bold')
    ax2.set_title('Avg Reward vs Final BSF', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    if len(episodes_avg_reward) > 2:
        corr2, p_val2 = pearsonr(episodes_avg_reward, episodes_final_bsf)
        ax2.text(0.05, 0.95, f'r={corr2:.3f}\np={p_val2:.2e}', 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        print(f"Avg Reward vs Final BSF: r={corr2:.3f}, p={p_val2:.2e}")
    
    # Subplot 3: return vs local_bsf (step-level)
    ax3 = axes[1, 0]
    ax3.scatter(all_returns, all_local_bsf, alpha=0.4, s=30, c='red', edgecolors='darkred', linewidth=0.3)
    ax3.set_xlabel('Return', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Local BSF Improvement', fontsize=12, fontweight='bold')
    ax3.set_title('Return vs Local BSF Improvement', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    if len(all_returns) > 2:
        corr3, p_val3 = pearsonr(all_returns, all_local_bsf)
        ax3.text(0.05, 0.95, f'r={corr3:.3f}\np={p_val3:.2e}', 
                transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        print(f"Return vs Local BSF: r={corr3:.3f}, p={p_val3:.2e}")
    
    # Subplot 4: reward vs local_bsf (step-level)
    ax4 = axes[1, 1]
    ax4.scatter(all_rewards, all_local_bsf, alpha=0.4, s=30, c='purple', edgecolors='darkviolet', linewidth=0.3)
    ax4.set_xlabel('Reward', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Local BSF Improvement', fontsize=12, fontweight='bold')
    ax4.set_title('Reward vs Local BSF Improvement', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    if len(all_rewards) > 2:
        corr4, p_val4 = pearsonr(all_rewards, all_local_bsf)
        ax4.text(0.05, 0.95, f'r={corr4:.3f}\np={p_val4:.2e}', 
                transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))
        print(f"Reward vs Local BSF: r={corr4:.3f}, p={p_val4:.2e}")
    
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
    parser.add_argument('--n_locations', type=int, default=100, help='Number of locations')
    parser.add_argument('--n_vehicles', type=int, default=30, help='Number of vehicles')
    parser.add_argument('--capacity', type=float, default=100.0, help='Capacity')
    parser.add_argument('--n_episodes', type=int, default=3, help='Number of episodes to run')
    parser.add_argument('--time_limit', type=float, default=2.0, help='Time limit per episode')
    args = parser.parse_args()
    
    collector = CuOptCollector(use_policy=args.policy, checkpoint_path=args.checkpoint)
    
    print(f"Running {args.n_episodes} episodes with {args.n_locations} locations, {args.n_vehicles} vehicles")
    print("=" * 80)
    
    # Collect data
    all_history = []
    all_initial_costs = []
    all_final_costs = []
    all_rewards = []
    all_returns = []
    
    for i in range(args.n_episodes):
        problem_gen = Problem(n_locations=args.n_locations, n_vehicles=args.n_vehicles)
        problem_data = problem_gen.generate()
        dm = problem_gen.create_data_model(problem_data)
        collector.run_solver(problem_data, dm, args.time_limit)
        history = collector.global_history
        
        initial_cost = history['initial_cost_set'][0]
        final_cost = history['global_bsf']
        rewards = [r.get('reward', 0) for r in history['history']]
        returns = history['return_set']
        all_initial_costs.append(initial_cost)
        all_final_costs.append(final_cost)
        all_rewards.append(np.mean(rewards))
        all_returns.append(np.mean(returns))
        print(f"Episode {i+1}: Steps={history['current_global_iter']}, "
                      f"Cost {initial_cost:.2f}→{final_cost:.2f}, "
                      f"AvgReward={np.mean(rewards):.2f}, "
                      f"AvgReturn={np.mean(returns):.2f}",
                      f"PolicyTime={collector.policy_sample_time * 1000 / collector.policy_call_count if collector.policy is not None else 0.0}ms")
        if args.plot:
            all_history.append(history)

    all_initial_costs = np.array(all_initial_costs)
    all_final_costs = np.array(all_final_costs)
    all_rewards = np.array(all_rewards)
    all_returns = np.array(all_returns)
    print("=" * 80)
    print(f"Episode Avg: "
                      f"Cost {all_initial_costs.mean():.2f}→{all_final_costs.mean():.2f}, "
                      f"AvgReward={all_rewards.mean():.2f}, "
                      f"AvgReturn={all_returns.mean():.2f}")
    print("=" * 80)

    # Plot if requested
    if args.plot:
        plot_reward_vs_cost(all_history, save_path='reward_vs_cost.png')