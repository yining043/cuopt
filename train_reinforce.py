"""
PPO Training for NodeCandidateSelectionPolicy
Uses Proximal Policy Optimization to reuse collected data
"""
import torch
import torch.optim as optim
import numpy as np
import gc
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
from cuopt_collector import CuOptCollector
from transformer_policy import TransformerCandidatePolicy


def plot_training_progress(test_costs, entropies, selection_rates, best_cost, save_path='training_progress.png'):
    """Plot and save training progress with entropy and selection rate"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))
    
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
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def load_trained_policy(checkpoint_path, policy_device=None):
    """
    Load a trained policy from checkpoint
    
    Example:
        policy = load_trained_policy('checkpoints/best_policy.pt')
        # Use policy for inference
        selection_mask, logp = policy.sample(state, problem_data, deterministic=True)
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Auto-detect device
    if policy_device is None:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                policy_device = 'cuda:1'
            else:
                policy_device = 'cuda'
        else:
            policy_device = 'cpu'
    
    policy = TransformerCandidatePolicy(
        d_model=128,
        num_heads=8,
        num_encoder_layers=3,
        max_candidates=40,
        device=policy_device
    )
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    print(f"Loaded policy from: {checkpoint_path}")
    print(f"  Trained for {checkpoint.get('epoch', 0)} epochs")
    print(f"  Best cost: {checkpoint.get('best_cost', 0):.2f}")
    
    return policy


def _collect_episode_worker(args):
    """Worker function for multiprocessing"""
    policy_state_dict, use_policy, n_locations, n_vehicles, seed, temperature, training_mode, policy_device = args
    
    # Create policy in subprocess if needed
    if use_policy and policy_state_dict is not None:
        # Use same device as parent
        policy = TransformerCandidatePolicy(
            d_model=128,
            num_heads=8,
            num_encoder_layers=3,
            max_candidates=40,
            device=policy_device
        )
        policy.load_state_dict(policy_state_dict)
        # Set same mode as parent process
        if training_mode:
            policy.train()
        else:
            policy.eval()
    else:
        policy = None
    
    # Collect episode
    return collect_episode(
        policy=policy,
        use_policy=use_policy,
        n_locations=n_locations,
        n_vehicles=n_vehicles,
        seed=seed,
        temperature=temperature,
        policy_device=policy_device
    )


def collect_episode(policy=None, use_policy=True, n_locations=50, n_vehicles=5, seed=None, temperature=1.0, policy_device=None):
    """Collect a single episode trajectory"""
    collector = CuOptCollector(
        n_locations=n_locations,
        n_vehicles=n_vehicles,
        time_limit=5.0,
        seed=seed if seed is not None else np.random.randint(0, 10000),
        policy=policy,
        use_policy=use_policy,
        temperature=temperature,
        policy_device=policy_device
    )
    
    dm, settings = collector.reset()
    collector.run_solver(dm, settings)
    
    # Return trajectory and problem data
    return {
        'trajectory': collector.trajectory,
        'problem': collector.problem_data
    }


def collect_episodes_parallel(policy, use_policy, n_episodes, n_locations, n_vehicles, n_workers=4, temperature=1.0):
    """Collect multiple episodes in parallel using multiprocessing"""
    # Get policy state dict for serialization
    policy_state_dict = policy.state_dict() if policy is not None else None
    
    # Get current training mode and device
    training_mode = policy.training if policy is not None else False
    policy_device = str(policy.device) if policy is not None else 'cpu'
    
    # Prepare arguments for each worker
    seeds = [np.random.randint(0, 100000) for _ in range(n_episodes)]
    args_list = [
        (policy_state_dict, use_policy, n_locations, n_vehicles, seed, temperature, training_mode, policy_device)
        for seed in seeds
    ]
    
    # Collect episodes in parallel with progress bar
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(_collect_episode_worker, args_list),
            total=n_episodes,
            desc="Collecting episodes",
            ncols=80
        ))
    
    return results


def compute_returns(rewards, gamma=0.99, reward_scale=0.01):
    """Compute discounted returns with reward scaling"""
    returns = []
    G = 0
    for r in reversed(rewards):
        # Scale reward to reasonable range
        r_scaled = r * reward_scale
        G = r_scaled + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)




def test_policy(policy, n_episodes=3, n_locations=50, n_vehicles=5, n_workers=3, reward_scale=0.01, temperature=1.0):
    """Test policy performance (parallel)"""
    policy.eval()
    
    # Collect episodes in parallel
    results = collect_episodes_parallel(
        policy=policy,
        use_policy=True,
        n_episodes=n_episodes,
        n_locations=n_locations,
        n_vehicles=n_vehicles,
        n_workers=n_workers,
        temperature=temperature  # Lower temperature during evaluation
    )
    
    costs = []
    returns_list = []
    
    for result in results:
        traj = result['trajectory']
        final_cost = traj['states'][-1]['solution_cost']
        costs.append(final_cost)
        
        returns = compute_returns(traj['rewards'], reward_scale=reward_scale)
        total_return = returns[0].item()
        returns_list.append(total_return)
    
    avg_cost = np.mean(costs)
    std_cost = np.std(costs)
    avg_return = np.mean(returns_list)
    print(f"  Cost: {avg_cost:.2f}±{std_cost:.2f}, Return: {avg_return:.2f}")
    
    return avg_cost, avg_return


def train_one_epoch(policy, optimizer, n_episodes=5, n_locations=50, n_vehicles=5, n_workers=5, 
                    reward_scale=0.01, grad_clip=1.0, ppo_epochs=4, clip_epsilon=0.2, batch_size=512, temperature=1.0):
    """Train policy for one epoch using PPO (parallel collection)"""
    
    # Collect data
    print(f"Collecting data ({n_workers} workers)...")
    policy.train()  # Set to training mode for data collection
    results = collect_episodes_parallel(
        policy=policy,
        use_policy=True,
        n_episodes=n_episodes,
        n_locations=n_locations,
        n_vehicles=n_vehicles,
        n_workers=n_workers,
        temperature=temperature  # Higher temperature during training for exploration
    )
    
    # Process collected data
    all_returns = []
    all_trajectories = []
    all_old_logps = []
    costs = []
    selection_rates = []
    
    for result in results:
        traj = result['trajectory']
        returns = compute_returns(traj['rewards'], reward_scale=reward_scale)
        all_returns.append(returns)
        all_trajectories.append({
            'states': traj['states'],
            'actions': traj['actions'],
            'problem': result['problem']
        })
        
        # Save old logps (without gradient)
        old_logps = torch.tensor(traj['logps'], dtype=torch.float32)
        all_old_logps.append(old_logps)
        
        states = traj['states']
        final_cost = states[-1]['solution_cost']
        costs.append(final_cost)
        
        # Compute selection rate
        for state, action in zip(traj['states'], traj['actions']):
            n_candidates = sum(state['candidate_mask'])
            n_selected = sum(action)  # action is now selection_mask
            if n_candidates > 0:
                selection_rates.append(n_selected / n_candidates * 100)
    
    # Print statistics
    avg_cost = np.mean(costs)
    std_cost = np.std(costs)
    avg_selection_rate = np.mean(selection_rates)
    print(f"  Collected {len(costs)} episodes: cost={avg_cost:.2f}±{std_cost:.2f}, select={avg_selection_rate:.1f}%")
    
    # Concatenate returns and old logps
    returns = torch.cat(all_returns)
    old_logps = torch.cat(all_old_logps)
    
    # Move to same device as policy
    device = policy.device
    returns = returns.to(device)
    old_logps = old_logps.to(device)
    
    # Debug: Print data statistics
    n_samples = len(old_logps)
    print(f"  Training samples: {n_samples} (from {len(costs)} episodes, avg {n_samples/len(costs):.1f} steps/episode)")
    
    # Count total states, actions, logps
    total_states = sum(len(traj['states']) for traj in all_trajectories)
    total_actions = sum(len(traj['actions']) for traj in all_trajectories)
    total_logps = len(old_logps)
    total_rewards = sum(len(result['trajectory']['rewards']) for result in results)
    print(f"  Data sizes - States: {total_states}, Actions: {total_actions}, Logps: {total_logps}, Rewards: {total_rewards}")
    
    # Normalize advantages (no external baseline)
    advantages = returns
    
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
        problem = traj_data['problem']
        
        for step_idx, (state, action) in enumerate(zip(states, actions)):
            flat_data.append({
                'state': state,
                'problem': problem,
                'action': action,  # selection_mask [total_nodes]
                'advantage': advantages[idx],
                'old_logp': old_logps[idx]
            })
            idx += 1
    
    ppo_stats = []
    total_grad_norm = 0
    
    for ppo_epoch in tqdm(range(ppo_epochs), desc="PPO updates", ncols=80, leave=False):
        # Shuffle data for each epoch
        indices = torch.randperm(len(flat_data)).tolist()
        
        # Process mini-batches
        n_batches = (len(flat_data) + batch_size - 1) // batch_size
        epoch_losses = []
        epoch_ratios = []
        epoch_clipped_fracs = []
        epoch_entropies = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(flat_data))
            batch_indices = indices[start_idx:end_idx]
            
            # Collect batch data
            batch_states = []
            batch_problems = []
            batch_selection_masks = []
            batch_advantages = []
            batch_old_logps = []
            
            for idx in batch_indices:
                sample = flat_data[idx]
                batch_states.append(sample['state'])
                batch_problems.append(sample['problem'])
                batch_selection_masks.append(sample['action'])  # Already selection_mask
                batch_advantages.append(sample['advantage'])
                batch_old_logps.append(sample['old_logp'])
            
            # Batch evaluate using new API
            batch_new_logps, batch_entropies = policy.evaluate(
                batch_states, batch_problems, batch_selection_masks
            )
            
            # Stack advantages and old_logps
            batch_advantages = torch.stack(batch_advantages)
            batch_old_logps = torch.stack(batch_old_logps)
            
            # PPO clipped objective
            ratio = torch.exp(batch_new_logps - batch_old_logps)
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            
            loss1 = ratio * batch_advantages
            loss2 = clipped_ratio * batch_advantages
            loss = -torch.min(loss1, loss2).mean()
            
            # Update policy
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients (first batch of first epoch)
            if ppo_epoch == 0 and batch_idx == 0:
                grad_norms = []
                for name, param in policy.named_parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm().item())
                total_grad_norm = sum(g**2 for g in grad_norms)**0.5 if grad_norms else 0
            
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)
            optimizer.step()
            
            # Collect batch statistics
            epoch_losses.append(loss.item())
            epoch_ratios.append(ratio.mean().item())
            epoch_clipped_fracs.append((ratio != clipped_ratio).float().mean().item())
            epoch_entropies.append(batch_entropies.mean().item())
        
        # Aggregate epoch statistics
        ppo_stats.append({
            'loss': np.mean(epoch_losses),
            'ratio_mean': np.mean(epoch_ratios),
            'ratio_std': np.std(epoch_ratios),
            'clipped_frac': np.mean(epoch_clipped_fracs),
            'entropy': np.mean(epoch_entropies)
        })
    
    # Check if parameters actually changed
    param_changes = []
    for name, param in policy.named_parameters():
        change = (param - initial_params[name]).abs().max().item()
        param_changes.append(change)
    
    max_param_change = max(param_changes)
    
    # Print PPO statistics
    first = ppo_stats[0]
    last = ppo_stats[-1]
    print(f"  PPO loss: {first['loss']:.4f} → {last['loss']:.4f}")
    print(f"  Ratio: {first['ratio_mean']:.3f}±{first['ratio_std']:.3f} → {last['ratio_mean']:.3f}±{last['ratio_std']:.3f}")
    print(f"  Clipped: {first['clipped_frac']*100:.1f}% → {last['clipped_frac']*100:.1f}%")
    print(f"  Entropy: {first['entropy']:.3f} → {last['entropy']:.3f}")
    print(f"  Gradient norm: {total_grad_norm:.4f}, Max param change: {max_param_change:.6f}\n")
    
    return first['loss'], last['entropy'], avg_selection_rate


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='REINFORCE Training')
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--n_episodes', type=int, default=5, help='Episodes per epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n_locations', type=int, default=50, help='Number of locations')
    parser.add_argument('--n_vehicles', type=int, default=5, help='Number of vehicles')
    parser.add_argument('--train_workers', type=int, default=1, help='Workers for training')
    parser.add_argument('--test_workers', type=int, default=1, help='Workers for testing')
    parser.add_argument('--reward_scale', type=float, default=0.01, help='Reward scaling factor')
    parser.add_argument('--grad_clip', type=float, default=0.1, help='Gradient clipping max norm')
    parser.add_argument('--ppo_epochs', type=int, default=5, help='PPO update epochs per batch')
    parser.add_argument('--clip_epsilon', type=float, default=0.1, help='PPO clip epsilon')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--test_only', action='store_true', help='Only test the loaded model, no training')
    parser.add_argument('--test_episodes', type=int, default=3, help='Number of episodes for test-only mode')
    parser.add_argument('--output_base', type=str, default='output', help='Base directory for outputs')
    parser.add_argument('--train_temperature', type=float, default=1.0, help='Sampling temperature during training')
    parser.add_argument('--test_temperature', type=float, default=1.0, help='Sampling temperature during testing')
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
    print("PPO Training (Parallel)")
    print("=" * 60)
    print(f"Epochs: {args.n_epochs}, Episodes/epoch: {args.n_episodes}")
    print(f"PPO: {args.ppo_epochs} update epochs, clip={args.clip_epsilon}")
    print(f"Learning rate: {args.lr}, Reward scale: {args.reward_scale}, Grad clip: {args.grad_clip}")
    print(f"Problem size: {args.n_locations} locations, {args.n_vehicles} vehicles")
    print(f"Workers: training={args.train_workers}, testing={args.test_workers}")
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
        d_model=128,
        num_heads=8,
        num_encoder_layers=3,
        max_candidates=40,
        device=device
    )
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, weights_only=False)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        
        if not args.test_only:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
        
        print(f"  Loaded from epoch {checkpoint.get('epoch', 0)}")
        print(f"  Best cost: {checkpoint.get('best_cost', 0):.2f}")
        print()
        
        # Test-only mode
        if args.test_only:
            print("=" * 60)
            print(f"TEST ONLY MODE - Testing {args.test_episodes} episodes")
            print("=" * 60)
            
            test_cost, test_return = test_policy(
                policy,
                n_episodes=args.test_episodes,
                n_locations=args.n_locations,
                n_vehicles=args.n_vehicles,
                n_workers=args.test_workers,
                reward_scale=args.reward_scale,
                temperature=args.test_temperature
            )
            
            print("\n" + "=" * 60)
            print("TEST RESULTS")
            print("=" * 60)
            print(f"Test cost: {test_cost:.2f}, return: {test_return:.2f}")
            print(f"Checkpoint best: {checkpoint.get('best_cost', 0):.2f}")
            print("=" * 60)
            return
    
    # Test before training (if not resuming)
    if not args.resume:
        print("Testing before training...")
        before_cost, before_return = test_policy(
            policy,
            n_episodes=args.test_episodes,
            n_locations=args.n_locations,
            n_vehicles=args.n_vehicles,
            n_workers=args.test_workers,
            reward_scale=args.reward_scale,
            temperature=args.test_temperature
        )
        print()
        best_cost = before_cost
        losses = []
        test_costs = []
        entropies = []
        selection_rates = []
    else:
        before_cost = checkpoint.get('before_cost', 0)
        best_cost = checkpoint.get('best_cost', before_cost)
        losses = checkpoint.get('losses', [])
        test_costs = checkpoint.get('test_costs', [])
        entropies = checkpoint.get('entropies', [])
        selection_rates = checkpoint.get('selection_rates', [])
    
    # Training loop
    print("=" * 60)
    print(f"Training from epoch {start_epoch} to {args.n_epochs}...")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.n_epochs):
        print(f"\n--- Epoch {epoch+1}/{args.n_epochs} ---")
        
        loss, entropy, selection_rate = train_one_epoch(
            policy,
            optimizer,
            n_episodes=args.n_episodes,
            n_locations=args.n_locations,
            n_vehicles=args.n_vehicles,
            n_workers=args.train_workers,
            reward_scale=args.reward_scale,
            grad_clip=args.grad_clip,
            ppo_epochs=args.ppo_epochs,
            clip_epsilon=args.clip_epsilon,
            temperature=args.train_temperature
        )
        losses.append(loss)
        entropies.append(entropy)
        selection_rates.append(selection_rate)
        
        # Test after every epoch
        test_cost, test_return = test_policy(
            policy,
            n_episodes=args.test_episodes,
            n_locations=args.n_locations,
            n_vehicles=args.n_vehicles,
            n_workers=args.test_workers,
            reward_scale=args.reward_scale,
            temperature=args.test_temperature
        )
        print(f"Epoch {epoch+1}: test_cost={test_cost:.2f}, loss={loss:.4f}")
        test_costs.append(test_cost)
        
        # Plot training progress
        plot_path = os.path.join(output_dir, 'training_progress.png')
        plot_training_progress(test_costs, entropies, selection_rates, best_cost, save_path=plot_path)
        
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
                'optimizer_state_dict': optimizer.state_dict(),
                'best_cost': best_cost,
                'before_cost': before_cost,
                'test_costs': test_costs,
                'entropies': entropies,
                'selection_rates': selection_rates,
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
    after_cost, after_return = test_policy(
        policy,
        n_episodes=args.test_episodes,
        n_locations=args.n_locations,
        n_vehicles=args.n_vehicles,
        n_workers=args.test_workers,
        reward_scale=args.reward_scale,
        temperature=args.test_temperature
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
        'optimizer_state_dict': optimizer.state_dict(),
        'test_costs': test_costs,
        'entropies': entropies,
        'selection_rates': selection_rates,
        'losses': losses,
        'best_cost': best_cost,
        'before_cost': before_cost,
        'after_cost': after_cost,
        'args': vars(args)
    }, final_checkpoint_path)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"  Final model: final_policy.pt")
    print(f"  Best model: best_policy.pt")
    print(f"  Training plot: training_progress.png (cost + entropy + selection rate)")
    print(f"  Best cost: {best_cost:.2f}")
    print(f"  Final cost: {after_cost:.2f}")


if __name__ == "__main__":
    import multiprocessing as mp
    # Use 'spawn' instead of 'fork' for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    main()

