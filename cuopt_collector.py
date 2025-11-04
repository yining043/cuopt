"""
CuOpt Trajectory Collector
Synchronously collects (state, action, reward) trajectories using fixed sampling policy
"""
import numpy as np
import cudf
import torch
import time
import tqdm
from cuopt import routing
from cuopt.routing import CustomizeNodesCallback, RewardCallback
from transformer_policy import TransformerCandidatePolicy


class CuOptCollector:
    """Synchronous trajectory collector for CuOpt"""
    
    def __init__(self, n_locations=100, n_vehicles=50, 
                 time_limit=10.0, seed=42, policy=None, use_policy=False, 
                 temperature=1.0, policy_device=None, checkpoint_path=None):
        self.n_locations = n_locations
        self.n_vehicles = n_vehicles
        self.time_limit = time_limit
        self.rng = np.random.default_rng(seed)
        self.temperature = temperature  # Sampling temperature
        
        self.trajectory = {'states': [], 'actions': [], 'rewards': [], 'logps': [], 'selected_sequences': [], 'step_improvements': []}
        self.problem_data = None
        self.solution = None
        self.customize_cb = None
        self.reward_cb = None
        
        # Performance tracking
        self.policy_sample_time = 0.0
        self.policy_call_count = 0
        
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
            
            # Load trained policy or create new one
            if checkpoint_path is not None:
                self.policy = self._load_trained_policy(checkpoint_path, policy_device, self.n_vehicles, self.n_locations)
            else:
                self.policy = TransformerCandidatePolicy(
                    d_model=64,
                    num_heads=4,
                    num_encoder_layers=3,
                    max_vehicles=self.n_vehicles,
                    N=self.n_locations,
                    problem_scale=100.0,
                    capacity_scale=50.0,
                    device=policy_device
                )
    
    def _load_trained_policy(self, checkpoint_path, policy_device, max_vehicles, n_locations):
        """Load a trained policy from checkpoint"""
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        # Get problem size from checkpoint
        checkpoint_args = checkpoint.get('args', {})

        # Create policy with checkpoint parameters
        policy = TransformerCandidatePolicy(
            d_model=64,
            num_heads=4,
            num_encoder_layers=3,
            max_vehicles=max_vehicles,
            N=n_locations,
            problem_scale=100.0,
            capacity_scale=50.0,
            device=policy_device
        )
        policy.load_state_dict(checkpoint['policy_state_dict'])
        policy.eval()
        
        print(f"Loaded trained policy from: {checkpoint_path}")
        print(f"  Trained for {checkpoint.get('epoch', 0)} epochs")
        print(f"  Best cost: {checkpoint.get('best_cost', 0):.2f}")
        
        return policy
    
    def reset(self):
        """Generate new problem and prepare solver"""
        self.problem_data = self._generate_problem()
        self.trajectory = {'best_cost': None, 'states': [], 'actions': [], 'rewards': [], 'logps': [], 'selected_sequences': [], 'step_improvements': []}
        
        # Reset performance tracking
        self.policy_sample_time = 0.0
        self.policy_call_count = 0
        
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
        seed = 42
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        """Generate random VRP (same as test_callback_minimal)"""
        n = self.n_locations
        coords = self.rng.random((n, 2)) * 100
        # print(f"coords: {coords[:3]}")
        distances = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)
        demand = np.concatenate([[0], self.rng.integers(1, 10, n - 1)])
        vehicle_capacity = np.full(self.n_vehicles, 50, dtype=np.int32)
        
        return {
            'n_locations': n,
            'n_vehicles': self.n_vehicles,
            'cost_matrix': cudf.DataFrame(distances),
            'demand': cudf.Series(demand),
            'vehicle_capacity': cudf.Series(vehicle_capacity),
            'coordinates': coords,
            'problem_scale': 100.0,  # Coordinate range [0, 100]
            'capacity_scale': 50.0   # Max vehicle capacity
        }


class _CustomizeCallback(CustomizeNodesCallback):
    """Records state and generates action with adaptive sampling policy or neural network"""
    
    def __init__(self, collector):
        super().__init__()
        self.collector = collector
    
    def customize_nodes_to_search(self, solution_flat, num_routes, 
                                  solution_cost, candidate_mask):
        # Compute number of candidates
        num_candidates = sum(candidate_mask)
        sample_size = 40
        # if num_candidates < 40:
        #     sample_size = num_candidates
        # elif num_candidates < 80:
        #     sample_size = num_candidates // 2
        # else:
        #     sample_size = 40

        # Record state
        state = {
            'solution_flat': solution_flat,
            'candidate_mask': candidate_mask,
            'solution_cost': solution_cost,
            'num_routes': num_routes,
            'num_candidates': num_candidates,
            'sample_size': sample_size,
        }
        self.collector.trajectory['states'].append(state)
        # print(f"number of candidates: {num_candidates}, sample size: {sample_size}")

        
        if self.collector.use_policy:
            policy_start = time.perf_counter()
            with torch.no_grad():
                selected_node_ids, log_prob = self.collector.policy(
                    state, 
                    self.collector.problem_data,
                    temperature=self.collector.temperature
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
            sampled = self.collector.rng.choice(candidate_nodes, sample_size, replace=False)
            selection_mask[sampled] = 1
            selection_mask = selection_mask.tolist()
            logp = 0.0
            selected_sequence = sampled.tolist()  # Random order (for consistency)
        
        self.collector.trajectory['logps'].append(logp)
        self.collector.trajectory['actions'].append(selection_mask)
        self.collector.trajectory['selected_sequences'].append(selected_sequence)  # Save ordering!
        
        return selection_mask


class _RewardCallback(RewardCallback):
    """Records reward as cost improvement with historical best tracking"""
    
    def __init__(self, collector):
        super().__init__()
        self.collector = collector
    
    def receive_reward(self, improvement_found, solution_cost):

        prev_cost = self.collector.trajectory['states'][-1]['solution_cost']
        step_improvement = (prev_cost - solution_cost)

        # if self.collector.trajectory['best_cost'] is not None:
        #     best_improvement = float(self.collector.trajectory['best_cost'] - solution_cost)
        #     reward = float(np.clip(best_improvement, 0.0, 100000.0)) / self.collector.trajectory['best_cost'] * 100
        # else:
        #     reward = 0.0
        reward = step_improvement / prev_cost * 100 #+ 10 * best_improvement #- prev_cost * 0.0001
        # reward = - solution_cost * 0.0001

        self.collector.trajectory['rewards'].append(reward)
        self.collector.trajectory['step_improvements'].append(step_improvement)
        if self.collector.trajectory['best_cost'] is None:
            self.collector.trajectory['best_cost'] = solution_cost
        else:
            self.collector.trajectory['best_cost'] = min(self.collector.trajectory['best_cost'], solution_cost)

def plot_reward_vs_cost(n_episodes=10, n_locations=100, n_vehicles=10, time_limit=5.0, use_policy=False, save_path='reward_vs_cost.png'):
    """
    Simple plot function: AVG reward vs Final cost + Step-level analysis
    """
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    
    print("=" * 50)
    print(f"Testing Reward vs Cost ({n_episodes} episodes)")
    print("=" * 50)
    
    # Create collector
    if use_policy:
        collector = CuOptCollector(n_locations=n_locations, n_vehicles=n_vehicles, time_limit=time_limit, seed=42, use_policy=True)
        collector.policy.eval()
    else:
        collector = CuOptCollector(n_locations=n_locations, n_vehicles=n_vehicles, time_limit=time_limit, seed=42, use_policy=False)
    
    # Collect data (both episode-level and step-level)
    total_rewards = []
    final_costs = []
    all_step_rewards = []
    all_step_improvements = []
    
    print("Collecting episodes...")
    for i in tqdm.tqdm(range(n_episodes)):
        collector.rng = np.random.default_rng(42 + i)
        dm, settings = collector.reset()
        collector.run_solver(dm, settings)
        
        if len(collector.trajectory['states']) > 0:
            states = collector.trajectory['states']
            rewards = collector.trajectory['rewards']
            step_improvements = collector.trajectory['step_improvements']
            
            # Episode-level
            total_reward = np.mean(rewards)
            final_cost = collector.trajectory['best_cost']
            total_rewards.append(total_reward)
            final_costs.append(final_cost)
            
            # Step-level
            all_step_rewards.extend(rewards)
            all_step_improvements.extend(step_improvements)
    
    print(f"Collected {len(final_costs)} valid episodes")
    print(f"Collected {len(all_step_rewards)} step-level data points")
    
    # Two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Total Reward vs Final Cost (Episode-level)
    ax1 = axes[0]
    ax1.scatter(total_rewards, final_costs, alpha=0.6, s=60, c='blue', edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('AVG Reward (Episode)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Final Cost', fontsize=12, fontweight='bold')
    ax1.set_title('Episode: Total Reward vs Final Cost', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add correlation
    if len(final_costs) > 2:
        corr1, p_val1 = pearsonr(total_rewards, final_costs)
        ax1.text(0.05, 0.95, f'Pearson r={corr1:.3f}\np={p_val1:.2e}', 
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        print(f"Episode correlation: r={corr1:.3f}, p={p_val1:.2e}")
    
    # Plot 2: Step Reward vs Step Improvement (Step-level)
    ax2 = axes[1]
    ax2.scatter(all_step_rewards, all_step_improvements, alpha=0.4, s=30, c='steelblue', edgecolors='navy', linewidth=0.3)
    ax2.set_xlabel('Step Reward', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Step Improvement', fontsize=12, fontweight='bold')
    ax2.set_title('Step: Reward vs Immediate Improvement', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation
    if len(all_step_rewards) > 2:
        corr2, p_val2 = pearsonr(all_step_rewards, all_step_improvements)
        ax2.text(0.05, 0.95, f'Pearson r={corr2:.3f}\np={p_val2:.2e}', 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        print(f"Step correlation: r={corr2:.3f}, p={p_val2:.2e}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()
    
    print("=" * 50)


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='CuOpt Collector Test')
    parser.add_argument('--policy', action='store_true', help='Use policy network')
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='Path to trained policy checkpoint (e.g., output/xxx/best_policy.pt)')
    parser.add_argument('--plot', action='store_true', help='Generate reward vs cost plot')
    parser.add_argument('--n_locations', type=int, default=100, help='Number of locations')
    parser.add_argument('--n_vehicles', type=int, default=30, help='Number of vehicles')
    parser.add_argument('--n_episodes', type=int, default=3, help='Number of episodes to run')
    parser.add_argument('--time_limit', type=float, default=10.0, help='Time limit per episode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    if args.plot:
        # Run plotting test
        plot_reward_vs_cost(
            n_episodes=10,
            n_locations=args.n_locations,
            n_vehicles=args.n_vehicles,
            time_limit=args.time_limit,
            use_policy=args.policy,
            save_path='reward_vs_cost.png'
        )
    else:
        # Run normal test
        if args.policy or args.checkpoint:
            if args.checkpoint:
                print(f"Loading trained policy from: {args.checkpoint}")
            else:
                print("Using untrained Policy Network")
            
            collector = CuOptCollector(
                n_locations=args.n_locations, 
                n_vehicles=args.n_vehicles, 
                time_limit=args.time_limit, 
                seed=args.seed, 
                use_policy=True,
                checkpoint_path=args.checkpoint
            )
            if collector.policy is not None:
                collector.policy.eval()
        else:
            print("Using Random Sampling")
            collector = CuOptCollector(
                n_locations=args.n_locations, 
                n_vehicles=args.n_vehicles, 
                time_limit=args.time_limit, 
                seed=args.seed, 
                use_policy=False
            )

        print(f"Running {args.n_episodes} episodes with {args.n_locations} locations, {args.n_vehicles} vehicles")
        print("=" * 80)
        
        for i in range(args.n_episodes):
            dm, settings = collector.reset()
            collector.run_solver(dm, settings)  
            
            if len(collector.trajectory['states']) > 0:
                states = collector.trajectory['states']
                rewards = collector.trajectory['rewards']
            
                print(f"Episode {i+1}: "
                      f"Steps={len(states)}, "
                      f"Cost {states[0]['solution_cost']:.1f}→{collector.trajectory['best_cost']:.1f}, "
                      f"Reward={np.mean(rewards):.2f}")
                
                if (args.policy or args.checkpoint) and collector.policy_call_count > 0:
                    avg_time = collector.policy_sample_time / collector.policy_call_count * 1000
                    print(f"  Policy time: {avg_time:.2f}ms/step")
            else:
                print(f"Episode {i+1}: No steps recorded")