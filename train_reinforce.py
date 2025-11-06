"""
PPO Training for Node Candidate Selection in VRP
Uses discounted returns as advantages (computed in collect_episodes)
"""
import os
import time
import gc
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, RandomSampler

from cuopt_collector import CuOptCollector, Problem
from transformer_policy import TransformerCandidatePolicy


# ============================================================================
# Constants
# ============================================================================
GAMMA = 0.99          # Discount factor for return computation


# ============================================================================
# Plotting
# ============================================================================
def plot_training_progress(test_costs, entropies, selection_rates, avg_improvements, best_cost, save_path='training_progress.png'):
    """Plot and save training progress"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    epochs = list(range(1, len(test_costs) + 1))
    
    ax1.plot(epochs, test_costs, 'b-o', label='Test Cost', linewidth=2, markersize=4)
    ax1.axhline(y=best_cost, color='g', linestyle='--', label=f'Best Cost: {best_cost:.2f}', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Test Cost', fontsize=12)
    ax1.set_title('PPO Training: Test Cost', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    if entropies:
        ax2.plot(epochs, entropies, 'r-o', label='Policy Entropy', linewidth=2, markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Entropy', fontsize=12)
        ax2.set_title('PPO Training: Policy Entropy', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    if selection_rates:
        ax3.plot(epochs, selection_rates, 'g-o', label='Selection Rate', linewidth=2, markersize=4)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Selection Rate (%)', fontsize=12)
        ax3.set_title('PPO Training: Node Selection Rate', fontsize=14, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
    
    if avg_improvements:
        ax4.plot(epochs, avg_improvements, 'm-o', label='Avg Step Improvement', linewidth=2, markersize=4)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Avg Step Improvement', fontsize=12)
        ax4.set_title('PPO Training: Average Step Improvement', fontsize=14, fontweight='bold')
        ax4.legend(loc='best', fontsize=10)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


# ============================================================================
# Data Collection
# ============================================================================
def collect_episodes(policy, n_episodes, n_locations, n_vehicles, time_limit):
    """Collect trajectories and return flat training data"""
    flat_data = []
    costs = []
    selection_rates = []
    step_improvements = []
    policy_device = str(policy.device) if policy is not None else 'cpu'
    episode_max_local_search_ids = []

    for _ in tqdm(range(n_episodes), desc="Collecting episodes", ncols=80):
        # Generate problem
        problem_gen = Problem(n_locations=n_locations, n_vehicles=n_vehicles)
        problem_data = problem_gen.generate()
        dm = problem_gen.create_data_model(problem_data)
        
        # Create collector
        collector = CuOptCollector(
            use_policy=True,
            policy_device=policy_device,
            checkpoint_path=None,
            policy=policy
        )
        
        # Run solver
        collector.run_solver(problem_data, dm, time_limit)
        
        # Extract history
        history_full = collector.global_history['history']
        global_bsf = collector.global_history['global_bsf']
        
        if global_bsf is not None:
            costs.append(global_bsf)
        
        # Find max local_search_id for this episode
        episode_max_ls_id = -1
        trajectories_by_ls_id = {}
        for record in history_full:
            ls_id = record['local_search_id']
            if ls_id is not None:
                episode_max_ls_id = max(episode_max_ls_id, ls_id)
            if ls_id not in trajectories_by_ls_id:
                trajectories_by_ls_id[ls_id] = []
            trajectories_by_ls_id[ls_id].append(record)
        
        episode_max_local_search_ids.append(episode_max_ls_id)
        
        # Process each trajectory and build flat_data directly
        for ls_id, records in trajectories_by_ls_id.items():
            if not records:
                continue

            first_circle_found_idx = -1
            for idx, record in enumerate(records):
                if record['is_circle_found'] == True:
                    first_circle_found_idx = idx
                    break
            records=records[:first_circle_found_idx]
            
            # Compute discounted return backwards
            Return = 0.0
            for record in reversed(records):
                Return = record['reward'] + GAMMA * Return
                has_action = record['is_circle_found'] == False
                
                if has_action:
                    state = {
                        'solution_flat': record['sol_before'],
                        'candidate_mask': record['candidate_mask'],
                        'solution_cost': record['cost_before'],
                        'num_routes': record['num_routes_before'],
                        'num_candidates': record['num_candidates'],
                        'sample_size': sum(record['action']) if record['action'] else 0
                    }
                    
                    # Build flat_data entry directly
                    flat_data.append({
                        'state': state,
                        'problem': problem_data,
                        'action': record['action'],
                        'selected_sequence': record['selected_sequence'],
                        'reward': Return,
                        'old_logp': record['logp']
                    })
                    
                    # Compute selection rate
                    n_candidates = sum(state['candidate_mask'])
                    n_selected = sum(record['action'])
                    if n_candidates > 0:
                        selection_rates.append(n_selected / n_candidates * 100)
                    
                    step_improvements.append(record['step_improvement'])
    
    if not flat_data:
        raise RuntimeError("All episodes failed to collect!")
    
    # Calculate average max local_search_id across episodes
    avg_max_local_search_id = episode_max_local_search_ids
    
    return flat_data, costs, selection_rates, step_improvements, avg_max_local_search_id

# ============================================================================
# Testing
# ============================================================================
def test_policy(policy, n_episodes=3, n_locations=50, n_vehicles=5, time_limit=5.0):
    """Test policy performance"""
    policy.eval()
    _, costs, _, step_improvements, _ = collect_episodes(policy, n_episodes, n_locations, n_vehicles, time_limit)
    
    if not costs:
        print("  Warning: No valid episodes collected!")
        return float('inf'), 0.0
    
    avg_cost = np.mean(costs)
    std_cost = np.std(costs)
    avg_step_improvement = np.mean(step_improvements) if step_improvements else 0.0
    print(f"  Cost: {avg_cost:.2f}±{std_cost:.2f}, Avg Step Impr: {avg_step_improvement:.4f}")
    return avg_cost, avg_step_improvement


# ============================================================================
# PPO Training
# ============================================================================
def train_one_epoch(policy, optimizer, n_episodes=5, n_locations=50, n_vehicles=5,
                    grad_clip=1.0, ppo_epochs=4, clip_epsilon=0.2, batch_size=256, time_limit=5.0):
    """Train policy for one epoch using PPO"""
    
    # Collect trajectories (returns flat_data directly)
    print(f"Collecting {n_episodes} episodes...")
    policy.eval()
    flat_data, costs, selection_rates, step_improvements, avg_max_local_search_id = collect_episodes(
        policy, n_episodes, n_locations, n_vehicles, time_limit
    )
    
    # Print collection statistics
    avg_cost = np.mean(costs) if costs else 0.0
    std_cost = np.std(costs) if costs else 0.0
    avg_selection_rate = np.mean(selection_rates) if selection_rates else 0.0
    avg_step_improvement = np.mean(step_improvements) if step_improvements else 0.0
    print(f"  Collected {len(costs)} episodes: cost={avg_cost:.2f}±{std_cost:.2f}, "
          f"select={avg_selection_rate:.1f}%, avg_step_impr={avg_step_improvement:.4f}, "
          f"avg_max_ls_id={avg_max_local_search_id}")
    
    n_samples = len(flat_data)
    print(f"  Training samples: {n_samples} (avg {n_samples/len(costs):.1f} steps/episode)")
    
    # PPO Training
    print(f"PPO updates ({ppo_epochs} epochs, batch_size={batch_size})...")
    policy.train()
    
    class FlatDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, i):
            return self.data[i]
    
    def collate_fn(batch):
        batch_states = [b['state'] for b in batch]
        batch_problems = [b['problem'] for b in batch]
        seqs = [b['selected_sequence'] for b in batch]
        max_k = max(len(s) for s in seqs) if seqs else 1
        selected_indices = torch.tensor([s + [-1] * (max_k - len(s)) for s in seqs],
                                        dtype=torch.long, device=policy.device)
        rewards_batch = torch.tensor([b['reward'] for b in batch], dtype=torch.float32, device=policy.device)
        old_logps_batch = torch.tensor([b['old_logp'] for b in batch], dtype=torch.float32, device=policy.device)
        return batch_states, batch_problems, selected_indices, rewards_batch, old_logps_batch, max_k
    
    ds = FlatDataset(flat_data)
    ppo_stats = []
    
    for ppo_epoch in tqdm(range(ppo_epochs), desc="PPO updates", ncols=80, leave=False):
        sampler = RandomSampler(ds, replacement=True, num_samples=len(ds))
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
        
        batch_stats = []
        for batch_states, batch_problems, selected_indices, batch_rewards, batch_old_logps, max_k in loader:
            batch_new_logps, batch_entropies = policy(
                batch_states, batch_problems, k=max_k, given_sequence=selected_indices
            )
            
            # batch rewards
            batch_advantages = batch_rewards
            
            # PPO clipped objective
            ratio = torch.exp(batch_new_logps - batch_old_logps)
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)
            optimizer.step()
            
            batch_stats.append({
                'loss': loss.item(),
                'ratio': ratio.mean().item(),
                'clipped_frac': (ratio != clipped_ratio).float().mean().item(),
                'entropy': batch_entropies.mean().item(),
                'advantages_avg': batch_advantages.mean().item()
            })
        
        ppo_stats.append({
            'loss': np.mean([s['loss'] for s in batch_stats]),
            'ratio_mean': np.mean([s['ratio'] for s in batch_stats]),
            'ratio_std': np.std([s['ratio'] for s in batch_stats]),
            'clipped_frac': np.mean([s['clipped_frac'] for s in batch_stats]),
            'entropy': np.mean([s['entropy'] for s in batch_stats]),
            'advantages_avg': np.mean([s['advantages_avg'] for s in batch_stats])
        })
    
    # Print PPO statistics
    first, last = ppo_stats[0], ppo_stats[-1]
    print(f"  PPO loss: {first['loss']:.4f} → {last['loss']:.4f}")
    print(f"  Ratio: {first['ratio_mean']:.3f}±{first['ratio_std']:.3f} → {last['ratio_mean']:.3f}±{last['ratio_std']:.3f}")
    print(f"  Clipped: {first['clipped_frac']*100:.1f}% → {last['clipped_frac']*100:.1f}%")
    print(f"  Entropy: {first['entropy']:.3f} → {last['entropy']:.3f}")
    print(f"  Advantages avg: {first['advantages_avg']:.4f} → {last['advantages_avg']:.4f}\n")
    
    return first['loss'], last['entropy'], avg_selection_rate, avg_step_improvement


# ============================================================================
# Utilities
# ============================================================================
def _save_checkpoint(policy, optimizer, epoch, best_cost, before_cost, test_costs,
                     entropies, selection_rates, avg_improvements, losses, args,
                     output_dir, filename='best_policy.pt', after_cost=None):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'policy_state_dict': policy.state_dict(),
        'policy_optimizer_state_dict': optimizer.state_dict(),
        'best_cost': best_cost,
        'before_cost': before_cost,
        'test_costs': test_costs,
        'entropies': entropies,
        'selection_rates': selection_rates,
        'avg_improvements': avg_improvements,
        'losses': losses,
        'args': vars(args)
    }
    if after_cost is not None:
        checkpoint['after_cost'] = after_cost
    path = os.path.join(output_dir, filename)
    torch.save(checkpoint, path)
    return path


def _cleanup_old_checkpoints(output_dir, keep_last_n=5):
    """Remove old checkpoint files, keeping only the last N epochs"""
    import glob
    checkpoint_files = glob.glob(os.path.join(output_dir, 'checkpoint_epoch_*.pt'))
    
    if len(checkpoint_files) <= keep_last_n:
        return
    
    # Extract epoch numbers from filenames
    def get_epoch(filename):
        try:
            basename = os.path.basename(filename)
            epoch_str = basename.replace('checkpoint_epoch_', '').replace('.pt', '')
            return int(epoch_str)
        except:
            return -1
    
    # Sort by epoch number
    checkpoint_files.sort(key=get_epoch)
    
    # Remove oldest checkpoints (keep last N)
    files_to_remove = checkpoint_files[:-keep_last_n]
    for f in files_to_remove:
        try:
            os.remove(f)
        except:
            pass


def _get_device():
    """Auto-detect device"""
    if torch.cuda.is_available():
        return 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda'
    return 'cpu'


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='PPO Training for VRP')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--n_episodes', type=int, default=10, help='Episodes per epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n_locations', type=int, default=100, help='Number of locations')
    parser.add_argument('--n_vehicles', type=int, default=30, help='Number of vehicles')
    parser.add_argument('--grad_clip', type=float, default=1, help='Gradient clipping max norm')
    parser.add_argument('--ppo_epochs', type=int, default=10, help='PPO update epochs per batch')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--batch_size', type=int, default=256, help='PPO mini-batch size')
    parser.add_argument('--time_limit', type=float, default=2, help='Time limit per episode (seconds)')
    parser.add_argument('--d_model', type=int, default=64, help='Transformer model dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Transformer attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='Transformer encoder layers')
    parser.add_argument('--problem_scale', type=float, default=100.0, help='Coordinate scaling factor')
    parser.add_argument('--capacity_scale', type=float, default=50.0, help='Capacity scaling factor')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--test_only', action='store_true', help='Only test the loaded model')
    parser.add_argument('--test_episodes', type=int, default=10, help='Number of episodes for testing')
    parser.add_argument('--output_base', type=str, default='output', help='Base directory for outputs')
    args = parser.parse_args()
    
    # Setup output directory
    if args.resume and not args.test_only:
        output_dir = os.path.dirname(args.resume)
    elif not args.test_only:
        output_dir = os.path.join(args.output_base, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(args.resume) if args.resume else 'output'
    
    print("=" * 60)
    print("PPO Training")
    print("=" * 60)
    print(f"Epochs: {args.n_epochs}, Episodes/epoch: {args.n_episodes}")
    print(f"PPO: {args.ppo_epochs} epochs, batch_size={args.batch_size}, clip={args.clip_epsilon}")
    print(f"LR: {args.lr}, Grad clip: {args.grad_clip}")
    print(f"Problem: {args.n_locations} locations, {args.n_vehicles} vehicles")
    print(f"Output: {output_dir}\n")
    
    # Initialize policy
    device = _get_device()
    print(f"Device: {device} (GPUs: {torch.cuda.device_count()})")
    
    policy = TransformerCandidatePolicy(
        d_model=args.d_model, num_heads=args.num_heads, num_encoder_layers=args.num_encoder_layers,
        max_vehicles=args.n_vehicles, N=args.n_locations,
        problem_scale=args.problem_scale, capacity_scale=args.capacity_scale, device=device
    )
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}\n")
    
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    
    # Load checkpoint
    start_epoch = 0
    checkpoint = None
    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, weights_only=False)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint.get('policy_optimizer_state_dict', optimizer.state_dict()))
            start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"  Epoch: {checkpoint.get('epoch', 0)}, Best cost: {checkpoint.get('best_cost', 0):.2f}\n")
        
        if args.test_only:
            print("=" * 60)
            print(f"TEST ONLY - {args.test_episodes} episodes")
            print("=" * 60)
            test_cost, test_impr = test_policy(policy, args.test_episodes, args.n_locations, args.n_vehicles, args.time_limit)
            print(f"\nTest cost: {test_cost:.2f}, impr: {test_impr:.4f}")
            print(f"Checkpoint best: {checkpoint.get('best_cost', 0):.2f}")
            print("=" * 60)
            return
    
    # Initialize training state
    test_kwargs = {
        'n_episodes': args.test_episodes,
        'n_locations': args.n_locations,
        'n_vehicles': args.n_vehicles,
        'time_limit': args.time_limit
    }
    
    if not args.resume:
        print("Testing before training...")
        before_cost, _ = test_policy(policy, **test_kwargs)
        print()
        best_cost, losses, test_costs, entropies, selection_rates, avg_improvements = before_cost, [], [], [], [], []
    else:
        before_cost = checkpoint.get('before_cost', 0)
        best_cost = checkpoint.get('best_cost', before_cost)
        losses = checkpoint.get('losses', [])
        test_costs = checkpoint.get('test_costs', [])
        entropies = checkpoint.get('entropies', [])
        selection_rates = checkpoint.get('selection_rates', [])
        avg_improvements = checkpoint.get('avg_improvements', [])
    
    # Training loop
    print("=" * 60)
    print(f"Training: epoch {start_epoch} → {args.n_epochs}")
    print("=" * 60)
    
    train_kwargs = {
        'n_episodes': args.n_episodes,
        'n_locations': args.n_locations,
        'n_vehicles': args.n_vehicles,
        'grad_clip': args.grad_clip,
        'ppo_epochs': args.ppo_epochs,
        'clip_epsilon': args.clip_epsilon,
        'batch_size': args.batch_size,
        'time_limit': args.time_limit
    }
    
    for epoch in range(start_epoch, args.n_epochs):
        print(f"\n--- Epoch {epoch+1}/{args.n_epochs} ---")
        
        loss, entropy, selection_rate, train_impr = train_one_epoch(policy, optimizer, **train_kwargs)
        losses.append(loss)
        entropies.append(entropy)
        selection_rates.append(selection_rate)
        
        test_cost, test_impr = test_policy(policy, **test_kwargs)
        avg_improvements.append(test_impr)
        test_costs.append(test_cost)
        print(f"Epoch {epoch+1}: test_cost={test_cost:.2f}, test_impr={test_impr:.4f}, "
              f"train_impr={train_impr:.4f}, loss={loss:.4f}")
        
        plot_training_progress(test_costs, entropies, selection_rates, avg_improvements, best_cost,
                              save_path=os.path.join(output_dir, 'training_progress.png'))
        
        # Save checkpoint for this epoch (keep last 5)
        epoch_filename = f'checkpoint_epoch_{epoch}.pt'
        _save_checkpoint(policy, optimizer, epoch, best_cost, before_cost,
                        test_costs, entropies, selection_rates, avg_improvements, losses, args,
                        output_dir, epoch_filename, after_cost=test_cost)
        _cleanup_old_checkpoints(output_dir, keep_last_n=5)
        
        # Save best checkpoint if improved
        if test_cost < best_cost:
            best_cost = test_cost
            improvement = before_cost - best_cost
            improvement_pct = (improvement / before_cost * 100) if before_cost > 0 else 0
            print(f"  * New best: {best_cost:.2f} (improved {improvement:.2f} / {improvement_pct:.2f}%)")
            path = _save_checkpoint(policy, optimizer, epoch, best_cost, before_cost,
                                   test_costs, entropies, selection_rates, avg_improvements, losses, args,
                                   output_dir, 'best_policy.pt', after_cost=test_cost)
            print(f"  Saved to: {path}")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final test and summary
    print("\n" + "=" * 60)
    print(f"Final testing ({args.test_episodes} episodes)...")
    print("=" * 60)
    after_cost, after_impr = test_policy(policy, **test_kwargs)
    print()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Before: {before_cost:.2f}, After: {after_cost:.2f}, Best: {best_cost:.2f}")
    
    cost_improvement = before_cost - after_cost
    best_improvement = before_cost - best_cost
    cost_improvement_pct = (cost_improvement / before_cost * 100) if before_cost > 0 else 0
    best_improvement_pct = (best_improvement / before_cost * 100) if before_cost > 0 else 0
    print(f"Final improvement: {cost_improvement:.2f} ({cost_improvement_pct:+.2f}%)")
    print(f"Best improvement: {best_improvement:.2f} ({best_improvement_pct:+.2f}%)")
    print(f"Avg loss: {np.mean(losses):.4f}")
    print("=" * 60)
    
    print(f"\nOutput: {output_dir}")
    print(f"  Files: best_policy.pt, checkpoint_epoch_*.pt (last 5), training_progress.png")
    print(f"  Costs: Best={best_cost:.2f}, Final={after_cost:.2f}")


if __name__ == "__main__":
    main()
