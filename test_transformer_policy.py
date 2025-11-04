"""
Test TransformerCandidatePolicy: Verify logp consistency for PPO
"""
import numpy as np
import torch
import cudf
from transformer_policy import TransformerCandidatePolicy


def create_test_problem(n_locations=100, n_routes=10, seed=42):
    """Create a single test problem (matches cuopt_collector format)"""
    np.random.seed(seed)
    total_nodes = n_locations + n_routes * 4
    
    # Match cuopt_collector.py format
    demand = np.concatenate([[0], np.random.randint(1, 10, n_locations - 1)])
    
    problem_data = {
        'coordinates': np.random.rand(n_locations, 2) * 100,  # numpy array
        'demand': cudf.Series(demand)  # cudf.Series (like cuopt_collector)
    }
    
    # Create candidate_mask
    num_candidates = min(35, n_locations - 1)
    candidate_mask = [0] * total_nodes
    candidate_indices = np.random.choice(range(1, n_locations), size=num_candidates, replace=False)
    for idx in candidate_indices:
        candidate_mask[idx] = 1
    
    # Create solution_flat (route structure)
    solution_flat = []
    available_nodes = list(range(1, n_locations))
    np.random.shuffle(available_nodes)
    nodes_per_route = len(available_nodes) // n_routes
    
    for route_id in range(n_routes):
        # Add 4 dummy depots
        for batch in range(4):
            solution_flat.append(n_locations + route_id * 4 + batch)
        # Add nodes
        start_idx = route_id * nodes_per_route
        end_idx = start_idx + nodes_per_route if route_id < n_routes - 1 else len(available_nodes)
        solution_flat.extend(available_nodes[start_idx:end_idx])
    
    num_candidates_count = sum(candidate_mask)
    sample_size = min(20, num_candidates_count)
    
    state = {
        'solution_flat': solution_flat,
        'candidate_mask': candidate_mask,
        'num_routes': n_routes,
        'solution_cost': 2500.0,
        'num_candidates': num_candidates_count,
        'sample_size': sample_size
    }
    
    return state, problem_data


def test_logp_consistency(n_locations=50, n_routes=5, num_problems=5):
    """Test that sample() and evaluate() produce identical logp (critical for PPO)"""
    print("=" * 70)
    print("Testing Logp Consistency (PPO Requirement)")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_vehicles = max(n_routes * 2, 100)
    policy = TransformerCandidatePolicy(
        d_model=128, 
        num_heads=8, 
        num_encoder_layers=3,
        max_vehicles=max_vehicles,
        N=n_locations,
        problem_scale=100.0,
        capacity_scale=50.0,
        device=device
    )
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"Device: {device}, Parameters: {num_params:,}\n")
    
    # Collect samples
    policy.eval()
    results = []
    
    with torch.no_grad():
        for i in range(num_problems):
            state, problem_data = create_test_problem(n_locations, n_routes, seed=42+i)
            torch.manual_seed(42 + i)
            
            selected_node_ids, log_prob = policy(state, problem_data, k=20)
            
            selected_node_ids = selected_node_ids[0].tolist()
            selection_mask = np.zeros(len(state['candidate_mask']), dtype=np.int32)
            selection_mask[selected_node_ids] = 1
            selection_mask = selection_mask.tolist()
            
            results.append({
                'state': state,
                'problem_data': problem_data,
                'selection_mask': selection_mask,
                'selected_sequence': selected_node_ids,
                'logp_sample': log_prob.item()
            })
    
    # Batch evaluate
    policy.train()
    states = [r['state'] for r in results]
    problem_data_list = [r['problem_data'] for r in results]
    selected_sequences = [r['selected_sequence'] for r in results]
    
    max_k = max(len(seq) for seq in selected_sequences)
    selected_indices_padded = [seq + [-1] * (max_k - len(seq)) for seq in selected_sequences]
    selected_indices = torch.tensor(selected_indices_padded, dtype=torch.long, device=policy.device)
    
    logps_batch, _ = policy(
        states, problem_data_list, 
        k=max_k, 
        given_sequence=selected_indices
    )
    
    # Verify consistency
    print(f"{'ID':<5} {'Sample logp':<15} {'Batch logp':<15} {'Diff':<12} {'Status':<8}")
    print("-" * 70)
    
    max_diff = 0.0
    all_passed = True
    
    for i, result in enumerate(results):
        logp_sample = result['logp_sample']
        logp_batch = logps_batch[i].item()
        diff = abs(logp_sample - logp_batch)
        max_diff = max(max_diff, diff)
        
        passed = diff < 1e-3
        all_passed = all_passed and passed
        status = "✓" if passed else "✗"
        
        print(f"{i+1:<5} {logp_sample:<15.6f} {logp_batch:<15.6f} {diff:<12.2e} {status:<8}")
    
    print("-" * 70)
    print(f"Max difference: {max_diff:.2e}")
    print(f"\n{'✅ PASS' if all_passed else '❌ FAIL'}: Logp consistency test\n")
    
    return all_passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test Transformer Policy')
    parser.add_argument('--n_locations', type=int, default=50, help='Number of locations')
    parser.add_argument('--n_routes', type=int, default=5, help='Number of routes')
    parser.add_argument('--num_problems', type=int, default=5, help='Number of test problems')
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("Transformer Policy Test")
    print("=" * 70)
    
    passed = test_logp_consistency(args.n_locations, args.n_routes, args.num_problems)
    
    if passed:
        print("🎉 Policy is ready for PPO training!\n")
    else:
        print("⚠️  Logp mismatch detected - check implementation\n")

