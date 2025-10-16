"""
Quick test script for TransformerCandidatePolicy
Tests the new autoregressive decoder with the callback interface
"""
import numpy as np
import torch
from transformer_policy import TransformerCandidatePolicy


def test_policy_interface():
    """Test that policy works with the new interface"""
    print("=" * 60)
    print("Testing TransformerCandidatePolicy")
    print("=" * 60)
    
    # Create policy
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    policy = TransformerCandidatePolicy(
        d_model=128,
        num_heads=8,
        num_encoder_layers=3,
        max_candidates=40,
        device=device
    )
    
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"\nPolicy created successfully!")
    print(f"Total parameters: {num_params:,}")
    
    # Create dummy problem data
    n_locations = 50
    n_routes = 5
    total_nodes = n_locations + n_routes * 4  # 50 + 20 = 70
    
    problem_data = {
        'coordinates': np.random.rand(n_locations, 2) * 100,
        'demand': np.concatenate([[0], np.random.randint(5, 15, n_locations - 1)])
    }
    
    # Create dummy state
    # Simulate that 30 nodes are candidates
    candidate_mask = [0] * total_nodes
    candidate_indices = np.random.choice(range(1, n_locations), size=30, replace=False)
    for idx in candidate_indices:
        candidate_mask[idx] = 1
    
    # Add some dummy depot candidates
    for i in range(n_locations, n_locations + 5):
        candidate_mask[i] = 1
    
    # Create dummy solution_flat (route structure)
    solution_flat = []
    for route_id in range(n_routes):
        # Add 4 dummy depots
        for batch in range(4):
            solution_flat.append(n_locations + route_id * 4 + batch)
        # Add some nodes
        nodes_in_route = np.random.choice(range(1, n_locations), size=5, replace=False)
        solution_flat.extend(nodes_in_route.tolist())
    
    state = {
        'solution_flat': solution_flat,
        'candidate_mask': candidate_mask,
        'num_routes': n_routes
    }
    
    print(f"\nTest state:")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Candidates: {sum(candidate_mask)}")
    print(f"  Solution flat length: {len(solution_flat)}")
    
    # Test sample()
    print("\n" + "-" * 60)
    print("Testing sample() method...")
    print("-" * 60)
    
    policy.eval()
    selection_mask, logp = policy.sample(state, problem_data, sample_size=20, temperature=1.0)
    
    print(f"Selection mask length: {len(selection_mask)}")
    print(f"Selected nodes: {sum(selection_mask)}")
    print(f"Log probability: {logp:.4f}")
    print(f"Sample successful: ✓")
    
    # Verify selection_mask format
    assert len(selection_mask) == total_nodes, "Selection mask length mismatch"
    assert all(v in [0, 1] for v in selection_mask), "Selection mask should be binary"
    
    # Verify selected nodes are from candidates
    selected_node_ids = [i for i, v in enumerate(selection_mask) if v == 1]
    candidate_node_ids = [i for i, v in enumerate(candidate_mask) if v == 1]
    assert all(nid in candidate_node_ids for nid in selected_node_ids), "Selected nodes should be candidates"
    
    print("All assertions passed: ✓")
    
    # Test evaluate()
    print("\n" + "-" * 60)
    print("Testing evaluate() method...")
    print("-" * 60)
    
    # Create batch of states
    states = [state, state]  # Duplicate for batch
    problem_data_list = [problem_data, problem_data]
    selection_masks = [selection_mask, selection_mask]
    
    policy.train()
    logps, entropies = policy.evaluate(states, problem_data_list, selection_masks)
    
    print(f"Batch size: {len(states)}")
    print(f"Log probs: {logps}")
    print(f"Entropies: {entropies}")
    print(f"Evaluate successful: ✓")
    
    # Verify output shapes
    assert logps.shape == (2,), "Log probs shape mismatch"
    assert entropies.shape == (2,), "Entropies shape mismatch"
    
    print("All assertions passed: ✓")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_policy_interface()

