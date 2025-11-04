"""
PPO Training for Node Candidate Selection in VRP
Uses step-level rewards with advantage normalization
"""
import torch
import torch.optim as optim
import numpy as np
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
from cuopt_collector import CuOptCollector
from transformer_policy import TransformerCandidatePolicy
from torch.utils.data import Dataset, DataLoader, RandomSampler
from math import ceil

def plot_training_progress(test_costs, entropies, selection_rates, avg_improvements, best_cost, save_path='training_progress.png'):
    """Plot and save training progress"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = list(range(1, len(test_costs) + 1))
    
    # Plot test cost
    ax1.plot(epochs, test_costs, 'b-o', label='Test Cost', linewidth=2, markersize=4)
    ax1.axhline(y=best_cost, color='g', linestyle='--', label=f'Best Cost: {best_cost:.2f}', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Test Cost', fontsize=12)
    ax1.set_title('PPO Training: Test Cost', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot entropy
    if len(entropies) > 0:
        ax2.plot(epochs, entropies, 'r-o', label='Policy Entropy', linewidth=2, markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Entropy', fontsize=12)
        ax2.set_title('PPO Training: Policy Entropy', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    # Plot selection rate
    if len(selection_rates) > 0:
        ax3.plot(epochs, selection_rates, 'g-o', label='Selection Rate', linewidth=2, markersize=4)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Selection Rate (%)', fontsize=12)
        ax3.set_title('PPO Training: Node Selection Rate', fontsize=14, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
    
    # Plot avg step improvement
    if len(avg_improvements) > 0:
        ax4.plot(epochs, avg_improvements, 'm-o', label='Avg Step Improvement', linewidth=2, markersize=4)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Avg Step Improvement', fontsize=12)
        ax4.set_title('PPO Training: Average Step Improvement', fontsize=14, fontweight='bold')
        ax4.legend(loc='best', fontsize=10)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def collect_episodes(policy, use_policy, n_episodes, n_locations, n_vehicles, time_limit):
    """Collect multiple episodes sequentially"""
    results = []
    policy_device = str(policy.device) if policy is not None else 'cpu'

    for i in tqdm(range(n_episodes), desc="Collecting episodes", ncols=80):
        seed = np.random.randint(0, 100000)
        collector = CuOptCollector(
            n_locations=n_locations,
            n_vehicles=n_vehicles,
            time_limit=time_limit,
            seed=seed,
            policy=policy,
            use_policy=use_policy,
            temperature=1.0,
            policy_device=policy_device,
            checkpoint_path=None
        )
        dm, settings = collector.reset()
        collector.run_solver(dm, settings)
        results.append({
            'trajectory': collector.trajectory,
            'problem': collector.problem_data
        })
    
    if len(results) == 0:
        raise RuntimeError("All episodes failed to collect!")

    return results


def test_policy(policy, n_episodes=3, n_locations=50, n_vehicles=5, time_limit=5.0):
    """Test policy performance"""
    policy.eval()
    
    results = collect_episodes(
        policy=policy,
        use_policy=True,
        n_episodes=n_episodes,
        n_locations=n_locations,
        n_vehicles=n_vehicles,
        time_limit=time_limit
    )
    
    costs = []
    all_step_improvements = []
    
    for result in results:
        traj = result['trajectory']
        final_cost = traj['best_cost']
        costs.append(final_cost)
        all_step_improvements.extend(traj['step_improvements'])
    
    avg_cost = np.mean(costs)
    std_cost = np.std(costs)
    avg_step_improvement = np.mean(all_step_improvements) if all_step_improvements else 0.0
    print(f"  Cost: {avg_cost:.2f}±{std_cost:.2f}, Avg Step Impr: {avg_step_improvement:.4f}")
    
    return avg_cost, avg_step_improvement


def train_one_epoch(policy, policy_optimizer, n_episodes=5, n_locations=50, n_vehicles=5,
                    grad_clip=1.0, ppo_epochs=4, clip_epsilon=0.2, batch_size=256, time_limit=5.0):
    """Train policy for one epoch using PPO"""
    
    # Collect data
    print(f"Collecting {n_episodes} episodes...")
    policy.eval()
    results = collect_episodes(
        policy=policy,
        use_policy=True,
        n_episodes=n_episodes,
        n_locations=n_locations,
        n_vehicles=n_vehicles,
        time_limit=time_limit
    )
    
    # Process collected data
    all_rewards = []
    all_trajectories = []
    all_old_logps = []
    costs = []
    selection_rates = []
    all_step_improvements = []
    
    for result in results:
        traj = result['trajectory']
        n_steps = len(traj['states'])
        
        if n_steps == 0:
            print(f"  Warning: Episode has no steps, skipping...")
            continue
        
        rewards = torch.tensor(traj['rewards'], dtype=torch.float32)
        all_rewards.append(rewards)
        
        all_trajectories.append({
            'states': traj['states'],
            'actions': traj['actions'],
            'selected_sequences': traj['selected_sequences'],  # Critical for PPO!
            'problem': result['problem']
        })
        
        # Save old logps (without gradient)
        old_logps = torch.tensor(traj['logps'], dtype=torch.float32)
        all_old_logps.append(old_logps)
        
        # Statistics
        final_cost = traj['best_cost']
        costs.append(final_cost)
        
        # Compute selection rate
        for state, action in zip(traj['states'], traj['actions']):
            n_candidates = sum(state['candidate_mask'])
            n_selected = sum(action)
            if n_candidates > 0:
                selection_rates.append(n_selected / n_candidates * 100)
        
        # Collect step improvements
        all_step_improvements.extend(traj['step_improvements'])
    
    # Print statistics
    avg_cost = np.mean(costs)
    std_cost = np.std(costs)
    avg_selection_rate = np.mean(selection_rates)
    avg_step_improvement = np.mean(all_step_improvements) if all_step_improvements else 0.0
    print(f"  Collected {len(costs)} episodes: cost={avg_cost:.2f}±{std_cost:.2f}, select={avg_selection_rate:.1f}%, avg_step_impr={avg_step_improvement:.4f}")
    
    # Concatenate rewards and old logps
    rewards = torch.cat(all_rewards)
    old_logps = torch.cat(all_old_logps)
    
    # Move to same device as policy
    device = policy.device
    rewards = rewards.to(device)
    old_logps = old_logps.to(device)
    
    # Debug: Print data statistics
    n_samples = len(old_logps)
    print(f"  Training samples: {n_samples} (from {len(costs)} episodes, avg {n_samples/len(costs):.1f} steps/episode)")
    
    # Use rewards directly as advantages (no baseline)
    print("  Computing advantages...")
    print(f"  Reward stats: mean={rewards.mean().item():.4f}, std={rewards.std().item():.4f}, min={rewards.min().item():.4f}, max={rewards.max().item():.4f}")
    
    # Normalize advantages
    advantages = rewards
    print(f"  Advantage stats: mean={advantages.mean().item():.4f}, std={advantages.std().item():.4f}, range=[{advantages.min().item():.2f}, {advantages.max().item():.2f}]")
    
    # PPO: Multiple updates with mini-batches
    print(f"PPO updates ({ppo_epochs} epochs, batch_size={batch_size})...")
    policy.train()
    
    # Save initial parameters for comparison
    initial_params = {name: param.clone().detach() for name, param in policy.named_parameters()}
    
    # Build flat data structure for batching
    print("  Preparing training data...")
    flat_data = []
    idx = 0
    for traj_idx, traj_data in enumerate(all_trajectories):
        states = traj_data['states']
        actions = traj_data['actions']
        selected_sequences = traj_data['selected_sequences']
        problem = traj_data['problem']
        
        for step_idx, (state, action, seq) in enumerate(zip(states, actions, selected_sequences)):
            flat_data.append({
                'state': state,
                'problem': problem,
                'action': action,  # selection_mask [total_nodes]
                'selected_sequence': seq,  # ordered node_ids (critical!)
                'advantage': advantages[idx],
                'old_logp': old_logps[idx]
            })
            idx += 1
    ppo_stats = []
    total_grad_norm = 0
    
    # Build Dataset/DataLoader to simplify batching
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
        max_k = max(len(s) for s in seqs)
        selected_indices = torch.tensor([s + [-1] * (max_k - len(s)) for s in seqs],
                                        dtype=torch.long, device=policy.device)
        advantages_batch = torch.stack([b['advantage'] for b in batch]).to(policy.device)
        old_logps_batch = torch.stack([b['old_logp'] for b in batch]).to(policy.device)
        return batch_states, batch_problems, selected_indices, advantages_batch, old_logps_batch, max_k

    ds = FlatDataset(flat_data)
    N = len(ds)
    default_updates = 1

    for ppo_epoch in tqdm(range(ppo_epochs), desc="PPO updates", ncols=80, leave=False):
        updates_per_epoch = default_updates
        sampler = RandomSampler(ds, replacement=True, num_samples=max(0, updates_per_epoch * batch_size))
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)

        epoch_losses = []
        epoch_ratios = []
        epoch_clipped_fracs = []
        epoch_entropies = []

        for batch_idx, (batch_states, batch_problems, selected_indices, batch_advantages, batch_old_logps, max_k) in enumerate(loader):
            batch_new_logps, batch_entropies_tensor = policy(
                batch_states, batch_problems,
                k=max_k,
                given_sequence=selected_indices
            )

            ratio = torch.exp(batch_new_logps - batch_old_logps)
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            # entropy_coef = 0.01
            # pg_obj = torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages)
            # loss = -(pg_obj - entropy_coef * batch_entropies_tensor.mean(1, True)).mean()
            # print("policy loss and entropy loss:", pg_obj.mean().item(), entropy_coef * batch_entropies_tensor.mean(1, True).mean().item())
            ### check！！
            loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()

            policy_optimizer.zero_grad()
            loss.backward()
            if ppo_epoch == 0 and batch_idx == 0:
                grad_norms = []
                for name, param in policy.named_parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm().item())
                total_grad_norm = sum(g**2 for g in grad_norms)**0.5 if grad_norms else 0
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)
            policy_optimizer.step()

            epoch_losses.append(loss.item())
            epoch_ratios.append(ratio.mean().item())
            epoch_clipped_fracs.append((ratio != clipped_ratio).float().mean().item())
            epoch_entropies.append(batch_entropies_tensor.mean().item())

        # Aggregate epoch statistics
        ppo_stats.append({
            'loss': np.mean(epoch_losses),
            'ratio_mean': np.mean(epoch_ratios),
            'ratio_std': np.std(epoch_ratios),
            'clipped_frac': np.mean(epoch_clipped_fracs),
            'entropy': np.mean(epoch_entropies)
        })
    
    # Print PPO statistics
    first = ppo_stats[0]
    last = ppo_stats[-1]
    print(f"  PPO loss: {first['loss']:.4f} → {last['loss']:.4f}")
    print(f"  Ratio: {first['ratio_mean']:.3f}±{first['ratio_std']:.3f} → {last['ratio_mean']:.3f}±{last['ratio_std']:.3f}")
    print(f"  Clipped: {first['clipped_frac']*100:.1f}% → {last['clipped_frac']*100:.1f}%")
    print(f"  Entropy: {first['entropy']:.3f} → {last['entropy']:.3f}")
    print(f"  Gradient norm: {total_grad_norm:.4f}\n")
    
    return first['loss'], last['entropy'], avg_selection_rate, avg_step_improvement


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='REINFORCE Training')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--n_episodes', type=int, default=5, help='Episodes per epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n_locations', type=int, default=100, help='Number of locations')
    parser.add_argument('--n_vehicles', type=int, default=30, help='Number of vehicles')
    parser.add_argument('--grad_clip', type=float, default=0.2, help='Gradient clipping max norm')
    parser.add_argument('--ppo_epochs', type=int, default=3, help='PPO update epochs per batch')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--batch_size', type=int, default=512, help='PPO mini-batch size')
    parser.add_argument('--time_limit', type=float, default=5.0, help='Time limit per episode (seconds)')
    # Policy hyperparameters (use passed args instead of relying on constructor defaults)
    parser.add_argument('--d_model', type=int, default=64, help='Transformer model dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Transformer attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='Transformer encoder layers')
    parser.add_argument('--problem_scale', type=float, default=100.0, help='Coordinate scaling factor')
    parser.add_argument('--capacity_scale', type=float, default=50.0, help='Capacity scaling factor')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--test_only', action='store_true', help='Only test the loaded model, no training')
    parser.add_argument('--test_episodes', type=int, default=5, help='Number of episodes for testing')
    parser.add_argument('--output_base', type=str, default='output', help='Base directory for outputs')
    args = parser.parse_args()
    
    # Create or reuse output directory
    import os
    import time
    if args.resume and not args.test_only:
        # Resume training: use the checkpoint's directory
        output_dir = os.path.dirname(args.resume)
    elif not args.test_only:
        # New training: create unique directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_base, timestamp)
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Test-only: use checkpoint's directory
        output_dir = os.path.dirname(args.resume) if args.resume else 'output'
    
    print("=" * 60)
    print("PPO Training")
    print("=" * 60)
    print(f"Epochs: {args.n_epochs}, Episodes/epoch: {args.n_episodes}")
    print(f"PPO: {args.ppo_epochs} update epochs, batch_size={args.batch_size}, clip={args.clip_epsilon}")
    print(f"Learning rate: {args.lr}, Grad clip: {args.grad_clip}")
    print(f"Problem size: {args.n_locations} locations, {args.n_vehicles} vehicles")
    print(f"Output directory: {output_dir}")
    print()
    
    # Initialize policy
    print("Initializing policy...")
    
    # Auto-detect device
    if torch.cuda.is_available():
        # If multiple GPUs visible, use GPU 1 for policy (GPU 0 for CuOpt workers)
        if torch.cuda.device_count() > 1:
            device = 'cuda:1'
        else:
            device = 'cuda'
    else:
        device = 'cpu'
    
    print(f"Using device: {device} (total GPUs: {torch.cuda.device_count()})")
    
    policy = TransformerCandidatePolicy(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        max_vehicles=args.n_vehicles,
        N=args.n_locations,
        problem_scale=args.problem_scale,
        capacity_scale=args.capacity_scale,
        device=device
    )
    
    policy_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy parameters: {policy_params:,}")
    
    # Only policy optimizer needed
    policy_optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, weights_only=False)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        
        if not args.test_only:
            policy_optimizer.load_state_dict(checkpoint.get('policy_optimizer_state_dict', policy_optimizer.state_dict()))
            start_epoch = checkpoint.get('epoch', 0) + 1
        
        print(f"  Loaded from epoch {checkpoint.get('epoch', 0)}")
        print(f"  Best cost: {checkpoint.get('best_cost', 0):.2f}")
        print()
        
        # Test-only mode
        if args.test_only:
            print("=" * 60)
            print(f"TEST ONLY MODE - Testing {args.test_episodes} episodes")
            print("=" * 60)
            
            test_cost, test_impr = test_policy(
                policy,
                n_episodes=args.test_episodes,
                n_locations=args.n_locations,
                n_vehicles=args.n_vehicles,
                time_limit=args.time_limit
            )
            
            print("\n" + "=" * 60)
            print("TEST RESULTS")
            print("=" * 60)
            print(f"Test cost: {test_cost:.2f}, avg step impr: {test_impr:.4f}")
            print(f"Checkpoint best: {checkpoint.get('best_cost', 0):.2f}")
            print("=" * 60)
            return
    
    # Test before training (if not resuming)
    if not args.resume:
        print("Testing before training...")
        before_cost, _ = test_policy(
            policy,
            n_episodes=args.test_episodes,
            n_locations=args.n_locations,
            n_vehicles=args.n_vehicles,
            time_limit=args.time_limit
        )
        print()
        best_cost = before_cost
        losses = []
        test_costs = []
        entropies = []
        selection_rates = []
        avg_improvements = []
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
    print(f"Training from epoch {start_epoch} to {args.n_epochs}...")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.n_epochs):
        print(f"\n--- Epoch {epoch+1}/{args.n_epochs} ---")
        
        loss, entropy, selection_rate, train_impr = train_one_epoch(
            policy,
            policy_optimizer,
            n_episodes=args.n_episodes,
            n_locations=args.n_locations,
            n_vehicles=args.n_vehicles,
            grad_clip=args.grad_clip,
            ppo_epochs=args.ppo_epochs,
            clip_epsilon=args.clip_epsilon,
            batch_size=args.batch_size,
            time_limit=args.time_limit
        )
        losses.append(loss)
        entropies.append(entropy)
        selection_rates.append(selection_rate)
        
        # Test after every epoch
        test_cost, test_impr = test_policy(
            policy,
            n_episodes=args.test_episodes,
            n_locations=args.n_locations,
            n_vehicles=args.n_vehicles,
            time_limit=args.time_limit
        )
        avg_improvements.append(test_impr)
        print(f"Epoch {epoch+1}: test_cost={test_cost:.2f}, test_impr={test_impr:.4f}, train_impr={train_impr:.4f}, loss={loss:.4f}")
        test_costs.append(test_cost)
        
        # Plot training progress
        plot_path = os.path.join(output_dir, 'training_progress.png')
        plot_training_progress(test_costs, entropies, selection_rates, avg_improvements, best_cost, save_path=plot_path)
        
        if test_cost < best_cost:
            best_cost = test_cost
            improvement = before_cost - best_cost
            improvement_pct = (improvement / before_cost * 100) if before_cost > 0 else 0
            print(f"  * New best cost: {best_cost:.2f} (improved {improvement:.2f} / {improvement_pct:.2f}%)")
            
            # Save best model checkpoint
            best_checkpoint_path = os.path.join(output_dir, 'best_policy.pt')
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'policy_optimizer_state_dict': policy_optimizer.state_dict(),
                'best_cost': best_cost,
                'before_cost': before_cost,
                'test_costs': test_costs,
                'entropies': entropies,
                'selection_rates': selection_rates,
                'avg_improvements': avg_improvements,
                'losses': losses,
                'args': vars(args)
            }, best_checkpoint_path)
            print(f"  Saved best model to: {best_checkpoint_path}")
        
        # Clean up memory after each epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final test
    print("\n" + "=" * 60)
    print(f"Final testing ({args.test_episodes} episodes)...")
    print("=" * 60)
    after_cost, after_impr = test_policy(
        policy,
        n_episodes=args.test_episodes,
        n_locations=args.n_locations,
        n_vehicles=args.n_vehicles,
        time_limit=args.time_limit
    )
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Before training:   cost={before_cost:.2f}")
    print(f"After training:    cost={after_cost:.2f}")
    print(f"Best cost:         {best_cost:.2f}")
    
    cost_improvement = before_cost - after_cost
    best_improvement = before_cost - best_cost
    cost_improvement_pct = (cost_improvement / before_cost * 100) if before_cost > 0 else 0
    best_improvement_pct = (best_improvement / before_cost * 100) if before_cost > 0 else 0
    print(f"\nFinal improvement: {cost_improvement:.2f} cost units ({cost_improvement_pct:+.2f}%)")
    print(f"Best improvement:  {best_improvement:.2f} cost units ({best_improvement_pct:+.2f}%)")
    print(f"Avg loss: {np.mean(losses):.4f}")
    print("=" * 60)
    
    # Save final model
    final_checkpoint_path = os.path.join(output_dir, 'final_policy.pt')
    
    torch.save({
        'epoch': args.n_epochs - 1,
        'policy_state_dict': policy.state_dict(),
        'policy_optimizer_state_dict': policy_optimizer.state_dict(),
        'test_costs': test_costs,
        'entropies': entropies,
        'selection_rates': selection_rates,
        'avg_improvements': avg_improvements,
        'losses': losses,
        'best_cost': best_cost,
        'before_cost': before_cost,
        'after_cost': after_cost,
        'args': vars(args)
    }, final_checkpoint_path)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"  Final model: final_policy.pt")
    print(f"  Best model: best_policy.pt")
    print(f"  Training plot: training_progress.png (cost + entropy + selection rate + avg step improvement)")
    print(f"  Best cost: {best_cost:.2f}")
    print(f"  Final cost: {after_cost:.2f}")


if __name__ == "__main__":
    main()

