"""
Test for routing callback functionality

Demonstrates the callback types:
- CustomizeNodesCallback: Customizes node sampling for local search
- RewardCallback: Receives iteration feedback
- LocalSearchStartCallback: Observes state before local search begins
- BeforeCycleFinderCallback: Observes state before cycle finder execution
- AfterCycleFinderCallback: Observes state after cycle finder execution
"""

import os
import time
from utils import get_initial_solutions, flat_solution_to_initial_solutions

conda_prefix = os.environ["CONDA_PREFIX"]
cuda_root = f"{conda_prefix}/targets/x86_64-linux"

def prepend(name, val):
    os.environ[name] = f"{val}:{os.environ[name]}" if name in os.environ else val

os.environ["CUDA_ROOT"] = cuda_root
os.environ["CUDA_PATH"] = cuda_root
os.environ["CUDA_HOME"] = cuda_root
os.environ["CUPY_CUDA_PATH"] = cuda_root

prepend("CPATH", f"{cuda_root}/include")
prepend("LIBRARY_PATH", f"{conda_prefix}/lib:{cuda_root}/lib64:{cuda_root}/lib")
prepend("LD_LIBRARY_PATH", f"{conda_prefix}/lib:{cuda_root}/lib64:{cuda_root}/lib")

os.environ.setdefault("LD_PRELOAD", f"{conda_prefix}/lib/libstdc++.so.6")

import numpy as np
import pickle
from cuopt import routing
from cuopt.routing import (
    CustomizeNodesCallback, 
    RewardCallback,
    LocalSearchStartCallback,
    BeforeCycleFinderCallback,
    AfterCycleFinderCallback
)
from test_callback_minimal import (
    TestCustomizeNodesCallback,
    TestRewardCallback,
    TestBeforeCycleFinderCallback,
    TestAfterCycleFinderCallback
)
from cuopt_collector import Problem
import matplotlib.pyplot as plt
import random
import cudf
import warnings
from utils import *
warnings.filterwarnings("ignore")

# Import callbacks from test_callback_minimal.py
# TestLocalSearchStartCallback is kept here because test_callback_minimal.py version has print/assert
# which we don't want in test_landscape.py
class TestLocalSearchStartCallback(LocalSearchStartCallback):
    """Manages global iteration offset and local search ID."""
    
    def __init__(self, global_history, customize_callback=None, reward_callback=None):
        super().__init__()
        self.global_history = global_history
        self.customize_callback = customize_callback
        self.reward_callback = reward_callback
    
    def on_local_search_start(self, solution_flat, num_routes, solution_cost, weights, selection_weights, should_all_nodes_be_served):
        """Update global iter and increment local search ID."""
        current_id = self.global_history.get('current_local_search_id', -1) + 1
        self.global_history['current_local_search_id'] = current_id
        self.global_history['current_local_iter'] = -1
        self.global_history['current_global_iter'] = self.global_history.get('current_global_iter', -1) + 1
        
        # Record initial solution at start of local search
        initial_record = {
            'local_search_id': current_id,
            'global_iter': self.global_history['current_global_iter'],
            'local_iter': -1,  # Before first iteration
            'sol_before': None,  # No previous solution
            'sol_after': solution_flat.copy() if solution_flat is not None else None,  # Initial solution
            'num_routes_before': None,
            'num_routes_after': num_routes,
            'cost_before': None,
            'cost_after': solution_cost,
            'move_found': False,
            'is_circle_found': False,
            'is_initial_solution': True  # Mark as initial solution
        }
        self.global_history['history'].append(initial_record)

    # def on_local_search_start(self, solution_flat, num_routes, solution_cost, weights, selection_weights, should_all_nodes_be_served):
    #     """Update global iter and increment local search ID."""
    #     self.global_history['current_local_search_id'] = self.global_history.get('current_local_search_id', -1) + 1
    #     self.global_history['current_local_iter'] = -1
    #     # print(f"Global iter: {self.global_history['current_global_iter']}, Local search count: {self.global_history['current_local_search_id']}")
    #     # print(f"Weights: {weights}")
    #     # print(f"Selection weights: {selection_weights}")
    #     # print(f"Should all nodes be served: {should_all_nodes_be_served}")
    #     # assert should_all_nodes_be_served, "All nodes should be served"

class SingleLocalSearchStartCallback(LocalSearchStartCallback):
    """Callback that tracks if we've started a second local search."""
    
    def __init__(self, global_history):
        super().__init__()
        self.global_history = global_history
    
    def on_local_search_start(self, solution_flat, num_routes, solution_cost, weights, selection_weights, should_all_nodes_be_served):
        """Track local search ID and mark if second one starts."""
        current_id = self.global_history.get('current_local_search_id', -1) + 1
        self.global_history['current_local_search_id'] = current_id
        self.global_history['current_local_iter'] = -1
        self.global_history['current_global_iter'] = self.global_history.get('current_global_iter', -1) + 1
        
        # Record initial solution at start of local search
        initial_record = {
            'local_search_id': current_id,
            'global_iter': self.global_history['current_global_iter'],
            'local_iter': -1,  # Before first iteration
            'sol_before': None,  # No previous solution
            'sol_after': solution_flat.copy() if solution_flat is not None else None,  # Initial solution
            'num_routes_before': None,
            'num_routes_after': num_routes,
            'cost_before': None,
            'cost_after': solution_cost,
            'move_found': False,
            'is_circle_found': False,
            'is_initial_solution': True  # Mark as initial solution
        }
        self.global_history['history'].append(initial_record)
        
        # Mark if this is the second local search (we want to stop after first)
        if current_id > 0:
            self.global_history['second_local_search_started'] = True

def perturb_solution_simple(solution_flat, num_routes, num_orders, perturbation_strength=0.1, seed=None):
    """
    Simple perturbation: randomly swap a fraction of customer nodes between routes.
    
    Args:
        solution_flat: Original solution
        num_routes: Number of routes
        num_orders: Number of customer orders
        perturbation_strength: Fraction of customers to move (0.0 to 1.0)
        seed: Random seed
    
    Returns:
        Perturbed solution_flat
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Infer correct num_routes if provided value might be incorrect
    inferred_num_routes = infer_num_routes_from_solution_flat(solution_flat, num_orders)
    if inferred_num_routes > 0 and inferred_num_routes != num_routes:
        num_routes = inferred_num_routes
    
    # Validate solution first
    validate_cuopt_solution(solution_flat, num_orders, " (input to perturb_solution_simple)")
    
    # Convert cuOpt format to routes (standard format: [0,1,2,3,0], [0,4,5,6,0], ...)
    routes = solution_flat_to_routes(solution_flat, num_routes, num_orders)
    
    # Collect all customers from routes
    all_customers = []
    for route in routes:
        all_customers.extend(route[1:-1])
    
    # Validate all customers are present (validation already done above, but double-check)
    expected_customers = set(range(1, num_orders + 1))
    actual_customers = set(all_customers)
    if expected_customers != actual_customers:
        missing = expected_customers - actual_customers
        extra = actual_customers - expected_customers
        raise ValueError(f"Invalid solution after conversion: missing nodes {sorted(missing)}, extra nodes {sorted(extra)}")
    
    # Calculate number to perturb
    n_to_perturb = max(1, int(len(all_customers) * perturbation_strength))
    customers_to_move = random.sample(all_customers, min(n_to_perturb, len(all_customers)))
    
    # Shuffle the customers to move for random redistribution
    customers_to_redistribute = customers_to_move.copy()
    random.shuffle(customers_to_redistribute)
    
    # Create a mapping: original customer -> replacement customer
    # This ensures each customer in customers_to_move gets replaced by exactly one customer from customers_to_redistribute
    customer_replacement_map = {}
    for i, original_customer in enumerate(customers_to_move):
        replacement_customer = customers_to_redistribute[i]
        customer_replacement_map[original_customer] = replacement_customer
    
    # Create perturbed routes by replacing customers according to the mapping
    # IMPORTANT: We must preserve num_routes routes, even if some become empty
    perturbed_routes = []
    
    for route in routes:
        perturbed_route = [0]  # Start with depot
        for node in route[1:-1]:  # Skip depot nodes
            if node in customer_replacement_map:
                # Replace with mapped customer
                perturbed_route.append(customer_replacement_map[node])
            else:
                # Keep original position
                perturbed_route.append(node)
        perturbed_route.append(0)  # End with depot
        # Always add route, even if empty (only depot nodes)
        perturbed_routes.append(perturbed_route)
    
    # Ensure we have exactly num_routes routes
    if len(perturbed_routes) != num_routes:
        raise ValueError(f"Route count mismatch: got {len(perturbed_routes)}, expected {num_routes}")
    
    # Ensure all customers are still present
    perturbed_customers = []
    for route in perturbed_routes:
        perturbed_customers.extend(route[1:-1])
    
    if set(perturbed_customers) != expected_customers:
        missing = expected_customers - set(perturbed_customers)
        extra = set(perturbed_customers) - expected_customers
        raise ValueError(f"Perturbation lost nodes: missing {sorted(missing)}, extra {sorted(extra)}")
    
    # Check for duplicates
    if len(perturbed_customers) != len(set(perturbed_customers)):
        from collections import Counter
        duplicates = [node for node, count in Counter(perturbed_customers).items() if count > 1]
        raise ValueError(f"Perturbation created duplicate nodes: {duplicates}")
    
    # Convert perturbed routes back to cuOpt format
    return routes_to_solution_flat(perturbed_routes, num_routes, num_orders)


def run_single_local_search(data_model, solver_settings, initial_solution_flat, 
                            num_routes, num_orders, time_limit=0.2):
    """
    Run cuopt with initial solution, but only run ONE local search (trial).
    Uses very short time limit to ensure only one local search completes.
    
    Args:
        data_model: CuOpt data model
        solver_settings: Solver settings
        initial_solution_flat: Initial solution in flat format
        num_routes: Number of routes
        num_orders: Number of customer orders
        time_limit: Very short time limit (default 0.2 seconds)
    
    Returns:
        dict: Final solution and history from single local search
    """
    # Validate and correct num_routes if needed
    inferred_num_routes = infer_num_routes_from_solution_flat(initial_solution_flat, num_orders)
    if inferred_num_routes > 0 and inferred_num_routes != num_routes:
        print(f"Warning: num_routes mismatch. Provided: {num_routes}, Inferred: {inferred_num_routes}. Using inferred value.")
        num_routes = inferred_num_routes
    
    # Convert to cuopt format
    vehicle_ids, routes, types, sol_offsets = solution_flat_to_cuopt_initial_solution(
        initial_solution_flat, num_routes, num_orders
    )
    
    # Validate before adding
    if len(routes) != len(vehicle_ids) or len(routes) != len(types):
        raise ValueError(f"Length mismatch: routes={len(routes)}, vehicle_ids={len(vehicle_ids)}, types={len(types)}")
    
    if len(sol_offsets) != num_routes + 1:
        raise ValueError(f"sol_offsets length mismatch: got {len(sol_offsets)}, expected {num_routes + 1}")
    
    # Check that all routes values are in valid range [0, num_orders-1] (0-based for cuOpt API)
    # Note: We converted from 1-based (1 to num_orders) to 0-based (0 to num_orders-1) above
    if len(routes) > 0:
        invalid_routes = routes[(routes < 0) | (routes >= num_orders)]
        if len(invalid_routes) > 0:
            raise ValueError(f"Invalid route node IDs: {invalid_routes.to_pandas().tolist()}. Valid range: 0 to {num_orders-1}")
        
        # Check for duplicates
        unique_routes = set(routes.to_pandas().tolist())
        if len(unique_routes) != len(routes):
            from collections import Counter
            duplicates = [node for node, count in Counter(routes.to_pandas().tolist()).items() if count > 1]
            raise ValueError(f"Duplicate node IDs in routes: {duplicates}")
        
        # Check all nodes are present (0 to num_orders-1 after conversion to 0-based)
        expected_nodes = set(range(0, num_orders))
        actual_nodes = set(routes.to_pandas().tolist())
        missing_nodes = expected_nodes - actual_nodes
        if missing_nodes:
            raise ValueError(f"Missing node IDs: {sorted(missing_nodes)}")
    
    # Clear any previous initial solutions if supported to avoid accumulation across runs
    if hasattr(data_model, "clear_initial_solutions"):
        data_model.clear_initial_solutions()
    
    # Add initial solution
    data_model.add_initial_solutions(vehicle_ids, routes, types, sol_offsets)
    
    # Extra sanity checks before Solve
    # Note: routes are now 0-based (0 to num_orders-1) for cuOpt API
    if len(routes) > 0:
        rlist = routes.to_pandas().tolist()
        expected_nodes = set(range(0, num_orders))
        actual_nodes = set(rlist)
        if expected_nodes != actual_nodes:
            missing_nodes = sorted(expected_nodes - actual_nodes)
            extra_nodes = sorted(actual_nodes - expected_nodes)
            print(f"[Pre-Solve] Node mismatch detected.")
            print(f"  expected 0..{num_orders-1} (0-based)")
            print(f"  missing: {missing_nodes[:20]}{' ...' if len(missing_nodes)>20 else ''}")
            print(f"  extra: {extra_nodes[:20]}{' ...' if len(extra_nodes)>20 else ''}")
        # Offsets final value must equal number of customers
        total_customers = len(rlist)
        last_offset = int(sol_offsets.iloc[-1])
        if last_offset != total_customers:
            print(f"[Pre-Solve] sol_offsets last value {last_offset} != total customers {total_customers}")
    
    # Set up callbacks to track single local search
    global_history = {
        'history': [],
        'current_local_search_id': -1,
        'current_global_iter': -1,
        'current_local_iter': -1,
        'second_local_search_started': False
    }
    
    customize_callback = TestCustomizeNodesCallback(global_history)
    reward_callback = TestRewardCallback(global_history)
    start_callback = SingleLocalSearchStartCallback(global_history)
    before_callback = TestBeforeCycleFinderCallback(global_history)
    after_callback = TestAfterCycleFinderCallback(global_history)
    
    solver_settings.set_routing_callback(customize_callback)
    solver_settings.set_routing_callback(reward_callback)
    solver_settings.set_routing_callback(start_callback)
    solver_settings.set_routing_callback(before_callback)
    solver_settings.set_routing_callback(after_callback)
    
    # Set VERY short time limit to ensure only one local search
    solver_settings.set_time_limit(time_limit)
    
    # Run solver
    solution = routing.Solve(data_model, solver_settings)
    
    # Extract last solution from the first local search
    history = global_history['history']
    
    # Check if second local search started (shouldn't happen with short time limit)
    if global_history.get('second_local_search_started', False):
        print("Warning: Second local search started despite short time limit")
    
    if history:
        # Get the last record from the first local search
        # Filter by local_search_id == 0 (first local search)
        # Handle None values in local_search_id
        first_local_search_records = [r for r in history 
                                     if r.get('local_search_id') is not None and r.get('local_search_id') == 0]
        if first_local_search_records:
            last_record = first_local_search_records[-1]
            final_solution_flat = last_record.get('sol_after')
            final_cost = last_record.get('cost_after')
            final_num_routes = last_record.get('num_routes_after', 0)
            if final_num_routes is None:
                final_num_routes = 0
        else:
            # Fallback: use last record
            last_record = history[-1]
            final_solution_flat = last_record.get('sol_after')
            final_cost = last_record.get('cost_after')
            final_num_routes = last_record.get('num_routes_after', 0)
            if final_num_routes is None:
                final_num_routes = 0
    else:
        # Fallback: use solution from solver
        route_np = solution.get_route()['route'].to_numpy().astype(int)
        final_routes = convert_node_sequence_to_routes(route_np)
        final_solution_flat = None
        final_cost = solution.get_total_objective()
        final_num_routes = len(final_routes)
    
    return {
        'solution': solution,
        'final_solution_flat': final_solution_flat,
        'final_cost': final_cost,
        'final_num_routes': final_num_routes,
        'history': history,
        'second_local_search_started': global_history.get('second_local_search_started', False)
    }

def test_callback(instance_path=None, hgs_solution_path=None, instance_index=0, n_locations=1001, time_limit=1, initial_solution_flat=None):
    """
    Test routing callback functionality with global history tracking.
    """
    print("=" * 60)
    print("Testing Routing Callback")
    print("=" * 60)
    
    # Initialize shared global history dictionary
    # This dictionary is passed to all callbacks for coordinated state tracking
    global_history = {
        'history': [],
        'current_local_search_id': -1,
        'current_global_iter': -1,
        'current_local_iter': -1
    }
    
    # Load HGS solution if provided
    hgs_solution = None
    if hgs_solution_path:
        print(f"\nLoading HGS solution from: {hgs_solution_path}")
        hgs_solution = load_hgs_solution_from_pkl(hgs_solution_path, instance_index)
        print(f"HGS optimal cost: {hgs_solution['hgs_cost']:.2f}")
    
    n_vehicles = 30
    
    # Setup problem - load from pkl if provided, otherwise generate
    instance_data, problem_data, problem_gen = None, None, Problem()  # For later visualization
    if instance_path:
        print(f"\nLoading instance from: {instance_path}")
        instance_data = load_instance_from_pkl(instance_path, instance_index)
        problem_gen = Problem(
            n_locations=instance_data['n_locations'],
            n_vehicles=n_vehicles,
            coordinate_range=100.0,
            capacity=instance_data['vehicle_capacity'],
            demand_range=(1, 10)
        )
        problem_data = {
            'n_locations': instance_data['n_locations'],
            'n_vehicles': n_vehicles,  # Default number of vehicles
            'cost_matrix': cudf.DataFrame(instance_data['cost_matrix']),
            'demand': cudf.Series(instance_data['demand']),
            'vehicle_capacity': cudf.Series(np.full(n_vehicles, instance_data['vehicle_capacity'], dtype=np.int32)),
            'coordinates': instance_data['coordinates'],
            'problem_scale': 100.0,
            'capacity_scale': instance_data['vehicle_capacity']
        }
    else:
        problem_gen = Problem(
            n_locations=n_locations,
            n_vehicles=n_vehicles,
            coordinate_range=100.0,
            capacity=100.0,
            demand_range=(1, 10)
        )
        problem_data = problem_gen.generate()
    data_model = problem_gen.create_data_model(problem_data)
    if initial_solution_flat is not None:
        vehicle_ids, routes, types, sol_offsets = flat_solution_to_initial_solutions(initial_solution_flat)
        data_model.add_initial_solutions(vehicle_ids, routes, types, sol_offsets)
    
    # Setup callbacks with shared global_history
    # Note: All callbacks receive the same global_history dictionary reference
    customize_callback = TestCustomizeNodesCallback(global_history)
    reward_callback = TestRewardCallback(global_history)
    start_callback = TestLocalSearchStartCallback(
        global_history, 
        customize_callback,  # Pass references for offset management
        reward_callback
    )
    before_callback = TestBeforeCycleFinderCallback(global_history)
    after_callback = TestAfterCycleFinderCallback(global_history)
    
    solver_settings = routing.SolverSettings()
    solver_settings.set_time_limit(time_limit)
    solver_settings.set_routing_callback(customize_callback)
    solver_settings.set_routing_callback(reward_callback)
    solver_settings.set_routing_callback(start_callback)
    solver_settings.set_routing_callback(before_callback)
    solver_settings.set_routing_callback(after_callback)
    
    # Solve
    print("\nSolving...")
    solution = routing.Solve(data_model, solver_settings)

    # cu_status = solution.get_status()
    # assert cu_status == 0
    # original_cost = solution.get_total_objective()
    # vehicle_ids, routes, types, sol_offsets = get_initial_solutions(solution)
    # data_model.add_initial_solutions(vehicle_ids, routes, types, sol_offsets)
    # solver_settings.set_time_limit(1)
    # solution = routing.Solve(data_model, solver_settings)
    # cu_status = solution.get_status()
    # assert cu_status == 0
    # new_cost = solution.get_total_objective()
    # assert new_cost <= original_cost
    # print(f"New cost: {new_cost:.2f}")
    # print(f"Original cost: {original_cost:.2f}")

    # Results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Solution status: {solution.get_status()}")
    
    # Statistics from global history
    history = global_history['history']
    total_history_records = len(history)
    local_search_count = global_history['current_local_search_id'] + 1
    cycle_finder_count = sum(1 for r in history if r['is_circle_found'])
    improvements = sum(1 for r in history if r['move_found'])
    
    print(f"Total local searches: {local_search_count}")
    print(f"Total history records: {total_history_records}")
    print(f"Cycle finder used: {cycle_finder_count} times")
    print(f"Total improvements: {improvements}")
    if history:
        all_costs = [r['cost_after'] for r in history if r['cost_after'] is not None]
        if all_costs:
            best_cost = min(all_costs)
            print(f"Best cost in history: {best_cost:.2f}")
    print("=" * 60)
    
    # Sample history output
    print("\n" + "=" * 60)
    print("SAMPLE HISTORY RECORDS (first 1)")
    print("=" * 60)
    for i, record in enumerate(history[:1]):
        print(f"\n Record {i}:")
        print(f"  Local Search ID: {record['local_search_id']}")
        print(f"  Global Iter: {record['global_iter']}")
        print(f"  Local Iter: {record['local_iter']}")
        print(f"  Cost Before: {record['cost_before']:.2f}" if record['cost_before'] else "  Cost Before: None")
        print(f"  Cost After: {record['cost_after']:.2f}" if record['cost_after'] else "  Cost After: None")
        print(f"  Move Found: {record['move_found']}")
        print(f"  Cycle Finder: {record['is_circle_found']}")
        if record['sol_after']:
            flat_preview = record['sol_after'][:20] if len(record['sol_after']) > 20 else record['sol_after']
            print(f"  Solution After (preview): {flat_preview}...")
    
    # Detailed analysis with enhanced visualization
    if history:
        # Get num_orders for distance computation
        num_orders = None
        if instance_data is not None:
            num_orders = instance_data['n_customers']
        elif 'problem_gen' in locals():
            num_orders = problem_gen.n_locations - 1  # Subtract depot
        
        # # Compute distance matrix between all solutions
        # print("\n" + "=" * 60)
        # print("Computing Solution Distance Matrix")
        # print("=" * 60)
        # distance_matrix, all_solutions, global_iter_list, cost_list, local_search_id_list = compute_solution_distances(
        #     global_history, num_orders=num_orders
        # )
        # if distance_matrix is not None:
        #     print(f"\nDistance matrix computed successfully!")
        #     print(f"  Shape: {distance_matrix.shape}")
        #     print(f"  Min distance: {np.min(distance_matrix):.4f}")
        #     print(f"  Max distance: {np.max(distance_matrix):.4f}")
        #     print(f"  Mean distance: {np.mean(distance_matrix):.4f}")
        #     print(f"  Std distance: {np.std(distance_matrix):.4f}")
        # else:
        #     print(f"\nDistance matrix computation returned None")

        best_record = min([r for r in history if r['cost_after'] is not None], key=lambda x: x['cost_after'])
        final_cost = solution.get_total_objective()
        print(f"\nBest Record from History:")
        print(f"  Cost: {best_record['cost_after']:.2f} (global iter {best_record['global_iter']}, local iter {best_record['local_iter']})")
        print(f"  Move Found: {best_record['move_found']}")
        print(f"  Cycle Finder: {best_record['is_circle_found']}")
        print(f"\nFinal Solution:")
        print(f"  Cost: {final_cost:.2f}")
        print(f"  Gap from best: {abs(final_cost - best_record['cost_after']):.2f}")
        
        # Initialize result dictionary
        result = {
            'instance_index': instance_index,
            'final_cost': final_cost,
            'best_cost_in_history': best_record['cost_after'],
            'hgs_cost': None,
            'gap': None,
            'num_orders': num_orders,

        }
        
        # Calculate gap with HGS solution if available
        if hgs_solution is not None:
            gap = calculate_gap(final_cost, hgs_solution['hgs_cost'])
            result['hgs_cost'] = hgs_solution['hgs_cost']
            result['gap'] = gap
            print(f"\n" + "=" * 60)
            print("HGS COMPARISON")
            print("=" * 60)
            print(f"HGS optimal cost: {hgs_solution['hgs_cost']:.2f}")
            print(f"My solution cost: {final_cost:.2f}")
            print(f"Gap: {gap:.2f}%")
            if gap < 0:
                print(f"  (My solution is {abs(gap):.2f}% better than HGS!)")
            elif gap == 0:
                print(f"  (My solution matches HGS optimal!)")
            else:
                print(f"  (My solution is {gap:.2f}% worse than HGS)")
            print("=" * 60)
            
        # # Extract last solution from each trial and compute distance matrix
        # print("\n" + "=" * 60)
        # print("Extracting Last Solution from Each Trial")
        # print("=" * 60)
        # trial_distance_matrix, trial_solutions, trial_metadata = extract_last_solution_per_trial(
        #     global_history, num_orders=num_orders
        # )
        # if trial_distance_matrix is not None:
        #     print(f"\nTrial distance matrix computed successfully!")
        #     print(f"  Shape: {trial_distance_matrix.shape}")
        #     print(f"  Number of trials: {len(trial_metadata)}")
    else:
        # No history, create minimal result
        result = {
            'instance_index': instance_index,
            'final_cost': None,
            'best_cost_in_history': None,
            'hgs_cost': None,
            'gap': None,
            'num_orders': None,
        }
    
    # # Calculate average diversity (mean of all distances in lower triangle)
    # avg_diversity = None
    # if distance_matrix is not None and distance_matrix.size > 0:
    #     # Get lower triangle (excluding diagonal)
    #     lower_triangle_mask = np.tril(np.ones_like(distance_matrix, dtype=bool), k=-1)
    #     lower_triangle_distances = distance_matrix[lower_triangle_mask]
    #     if len(lower_triangle_distances) > 0:
    #         avg_diversity = np.mean(lower_triangle_distances)
    
    # # Add diversity to result
    # if result:
    #     result['avg_diversity'] = avg_diversity
    
    # Return complete global history and result for external analysis
    return {
        'history': global_history['history'],
        'result': result,
        # 'avg_diversity': avg_diversity
    }


def extract_perturbations_from_trial_history(global_history, num_orders=None):
    """
    Extract perturbations from trial history.
    Type 1: All solutions in a trial (except the last one) as perturbations of the last solution.
    
    Args:
        global_history: Dictionary containing history records
        num_orders: Number of customer orders
    
    Returns:
        List of (local_optimum, perturbed_solutions, metadata) tuples
    """
    history = global_history['history']
    if not history:
        return []
    
    # Group records by local_search_id (trial)
    trials_dict = {}
    for record in history:
        trial_id = record.get('local_search_id', -1)
        if trial_id is None:
            trial_id = -1
        if trial_id not in trials_dict:
            trials_dict[trial_id] = []
        trials_dict[trial_id].append(record)
    
    perturbations = []
    
    for trial_id in sorted(trials_dict.keys()):
        trial_records = trials_dict[trial_id]
        # Sort by local_iter
        trial_records_sorted = sorted(trial_records, 
                                     key=lambda x: x.get('local_iter') if x.get('local_iter') is not None else -1)
        
        # Last record is the local optimum (last solution in this trial)
        last_record = trial_records_sorted[-1]
        local_optimum = last_record.get('sol_after')
        num_routes = last_record.get('num_routes_after', 0)
        if num_routes is None:
            num_routes = 0
        
        if local_optimum is None or len(local_optimum) == 0:
            continue
        
        # Validate local optimum
        if num_orders is not None:
            validate_cuopt_solution(local_optimum, num_orders, 
                                   f" in trial {trial_id} (local_iter={last_record.get('local_iter')})")
        
        # Extract all solutions except the last one as perturbations (regardless of type)
        perturbed_solutions = []
        for record in trial_records_sorted[:-1]:  # All except the last one
            perturbed_sol = record.get('sol_after')
            if perturbed_sol is not None and len(perturbed_sol) > 0:
                # Validate the solution before adding
                if num_orders is not None:
                    validate_cuopt_solution(perturbed_sol, num_orders,
                                           f" in history (local_iter={record.get('local_iter')})")
                perturbed_solutions.append({
                    'solution': perturbed_sol,
                    'local_iter': record.get('local_iter', -1),
                    'global_iter': record.get('global_iter', -1),
                    'cost': record.get('cost_after', None),
                    'is_circle_found': record.get('is_circle_found', False),
                    'move_found': record.get('move_found', False)
                })
        
        if perturbed_solutions:
            perturbations.append({
                'local_optimum': local_optimum,
                'local_optimum_metadata': {
                    'local_search_id': trial_id,
                    'local_iter': last_record.get('local_iter', -1),
                    'global_iter': last_record.get('global_iter', -1),
                    'cost': last_record.get('cost_after', None),
                    'num_routes_after': num_routes,
                    'is_circle_found': last_record.get('is_circle_found', False),
                    'move_found': last_record.get('move_found', False)
                },
                'perturbed_solutions': perturbed_solutions,
                'perturbation_type': 'trial_intermediate'
            })
    
    return perturbations

def extract_perturbations_after_last_cycle_finder(global_history, num_orders=None):
    """
    Extract perturbations from after the last cycle finder in each trial.
    Type 2: All fast search solutions after the last cycle finder as perturbations.
    
    Args:
        global_history: Dictionary containing history records
        num_orders: Number of customer orders
    
    Returns:
        List of (local_optimum, perturbed_solutions, metadata) tuples
    """
    history = global_history['history']
    if not history:
        return []
    
    # Group records by local_search_id (trial)
    trials_dict = {}
    for record in history:
        trial_id = record.get('local_search_id', -1)
        if trial_id is None:
            trial_id = -1
        if trial_id not in trials_dict:
            trials_dict[trial_id] = []
        trials_dict[trial_id].append(record)
    
    perturbations = []
    
    for trial_id in sorted(trials_dict.keys()):
        trial_records = trials_dict[trial_id]
        # Sort by local_iter
        trial_records_sorted = sorted(trial_records, 
                                     key=lambda x: x.get('local_iter') if x.get('local_iter') is not None else -1)
        
        if len(trial_records_sorted) < 2:
            continue
        
        # Find the last cycle finder (is_circle_found = True)
        last_cycle_finder_record = None
        last_cycle_finder_idx = -1
        for i in range(len(trial_records_sorted) - 1, -1, -1):
            if trial_records_sorted[i].get('is_circle_found', False):
                last_cycle_finder_record = trial_records_sorted[i]
                last_cycle_finder_idx = i
                break
        
        if last_cycle_finder_record is None:
            # No cycle finder found, skip this trial
            continue
        
        # Local optimum is the last solution in this trial (regardless of type)
        last_record = trial_records_sorted[-1]
        local_optimum = last_record.get('sol_after')
        num_routes = last_record.get('num_routes_after', 0)
        if num_routes is None:
            num_routes = 0
        
        if local_optimum is None or len(local_optimum) == 0:
            continue
        
        # Validate local optimum
        if num_orders is not None:
            validate_cuopt_solution(local_optimum, num_orders,
                                   f" in trial {trial_id} (local_iter={last_record.get('local_iter')})")
        
        # Determine which cycle finder to use as the boundary:
        # - If the last record is a cycle finder, use the second-to-last cycle finder as boundary
        # - Otherwise, use the last cycle finder as boundary
        cycle_finder_indices = [i for i, rec in enumerate(trial_records_sorted) if rec.get('is_circle_found', False)]
        if not cycle_finder_indices:
            continue
        last_is_cycle = last_record.get('is_circle_found', False)
        if last_is_cycle and len(cycle_finder_indices) >= 2:
            target_cf_idx = cycle_finder_indices[-2]
            next_cf_idx = cycle_finder_indices[-1]
        else:
            target_cf_idx = cycle_finder_indices[-1]
            next_cf_idx = len(trial_records_sorted)
        
        # Extract all fast search solutions after target cycle finder and before next cycle finder (exclusive)
        # Exclude cycle finder records (is_circle_found = True)
        perturbed_solutions = []
        for i in range(target_cf_idx + 1, next_cf_idx):
            record = trial_records_sorted[i]
            if not record.get('is_circle_found', False):
                perturbed_sol = record.get('sol_after')
                if perturbed_sol is not None and len(perturbed_sol) > 0:
                    if num_orders is not None:
                        validate_cuopt_solution(perturbed_sol, num_orders,
                                               f" after cycle finder (local_iter={record.get('local_iter')})")
                    perturbed_solutions.append({
                        'solution': perturbed_sol,
                        'local_iter': record.get('local_iter', -1),
                        'global_iter': record.get('global_iter', -1),
                        'cost': record.get('cost_after', None),
                        'is_circle_found': False,
                        'move_found': record.get('move_found', False)
                    })
        
        if perturbed_solutions:
            perturbations.append({
                'local_optimum': local_optimum,
                'local_optimum_metadata': {
                    'local_search_id': trial_id,
                    'local_iter': last_record.get('local_iter', -1),
                    'global_iter': last_record.get('global_iter', -1),
                    'cost': last_record.get('cost_after', None),
                    'num_routes_after': num_routes,
                    'is_circle_found': last_record.get('is_circle_found', False),
                    'move_found': last_record.get('move_found', False),
                    'last_cycle_finder_local_iter': last_cycle_finder_record.get('local_iter', -1)
                },
                'perturbed_solutions': perturbed_solutions,
                'perturbation_type': 'after_cycle_finder'
            })
    
    return perturbations


def analyze_perturbed_solution(perturbed_sol, local_optimum, optimum_id, num_routes, num_orders,
                               unique_optima, instance_data, n_runs_per_perturbation, 
                               time_limit_per_run, perturbation_type='manual'):
    """
    Analyze a single perturbed solution: run local search multiple times and record convergence.
    
    Args:
        perturbed_sol: Perturbed solution (solution_flat)
        local_optimum: Original local optimum (solution_flat)
        optimum_id: ID of the original local optimum
        num_routes: Number of routes
        num_orders: Number of customer orders
        unique_optima: List of unique local optima
        instance_data: Instance data
        n_runs_per_perturbation: Number of local search runs
        time_limit_per_run: Time limit for each run
        perturbation_type: Type of perturbation ('manual', 'trial_intermediate', 'after_cycle_finder')
    
    Returns:
        dict: Analysis results
    """
    convergence_results = []
    
    for run_idx in range(n_runs_per_perturbation):
        if (run_idx + 1) % 20 == 0:
            print(f"      Run {run_idx + 1}/{n_runs_per_perturbation}...")
        
        # Recreate data_model for each run
        data_model = create_data_model_from_instance(instance_data)
        solver_settings = routing.SolverSettings()
        
        try:
            result = run_single_local_search(
                data_model, solver_settings, perturbed_sol,
                num_routes, num_orders, time_limit=time_limit_per_run
            )
        except Exception as e:
            # Skip this run if there's an error (e.g., "Inconsistent order ids")
            error_msg = str(e)
            if "Inconsistent order ids" in error_msg or "ValidationError" in error_msg:
                # Silently skip validation errors
                continue
            else:
                # For other errors, print and skip
                if (run_idx + 1) % 20 == 0:  # Only print occasionally to avoid spam
                    print(f"      Warning: Error in run {run_idx + 1}: {error_msg[:100]}")
                continue
        
        if result['final_solution_flat'] is not None:
            # Check which local optimum this converged to
            final_sol = result['final_solution_flat']
            final_routes = solution_flat_to_routes(final_sol, num_routes, num_orders)
            final_edges = extract_edges_from_routes(final_routes)
            final_edges_tuple = tuple(sorted(final_edges))
            
            # Find which unique optimum this matches
            converged_to_id = None
            for unique_opt in unique_optima:
                if tuple(sorted(unique_opt['edges'])) == final_edges_tuple:
                    converged_to_id = unique_opt['optimum_id']
                    break
            
            # If not found in unique optima, it's a new optimum
            if converged_to_id is None:
                converged_to_id = -1  # New/unseen optimum
            
            convergence_results.append({
                'run_idx': run_idx,
                'converged_to_id': converged_to_id,
                'final_cost': result['final_cost'],
                'final_solution': final_sol
            })
    
    if not convergence_results:
        return None
    
    # Count frequencies
    convergence_counts = {}
    for res in convergence_results:
        opt_id = res['converged_to_id']
        convergence_counts[opt_id] = convergence_counts.get(opt_id, 0) + 1
    
    # Calculate return frequency to original optimum
    return_frequency = convergence_counts.get(optimum_id, 0) / len(convergence_results)
    
    return {
        'return_frequency': return_frequency,
        'convergence_distribution': convergence_counts,
        'n_runs': len(convergence_results),
        'convergence_results': convergence_results,
        'perturbation_type': perturbation_type
    }


def collect_contrastive_dataset_extended(instance_path, hgs_solution_path, instance_index=0,
                                        n_cuopt_runs=1, n_perturbations_per_optimum=1,
                                        n_runs_per_perturbation=100, perturbation_strength=0.1,
                                        output_dir='plot', time_limit_per_run=0.2,
                                        collect_manual=True):
    """
    Collect contrastive learning dataset with THREE types of perturbations:
    1. Manual perturbations (original method) - optional, controlled by collect_manual
    2. Trial intermediate solutions (all solutions in a trial except the last)
    3. Fast search after last cycle finder
    
    Args:
        instance_path: Path to instance file
        hgs_solution_path: Path to HGS solution file
        instance_index: Instance index
        n_cuopt_runs: Number of times to run full cuopt
        n_perturbations_per_optimum: Number of manual perturbations per local optimum
        n_runs_per_perturbation: Number of local search runs per perturbation
        perturbation_strength: Strength of manual perturbation (0.0 to 1.0)
        output_dir: Output directory
        time_limit_per_run: Time limit for each local search run
        collect_manual: Whether to collect manual perturbations (default: True)
    
    Returns:
        dict: Combined dataset with perturbation types (manual is optional)
    """
    import datetime
    import pickle
    
    # Create timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_output_dir = os.path.join(output_dir, f'dataset_{timestamp}')
    os.makedirs(timestamped_output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"Collecting EXTENDED contrastive learning dataset for instance {instance_index}")
    print(f"  CuOpt runs: {n_cuopt_runs}")
    print(f"  Collect manual perturbations: {collect_manual}")
    if collect_manual:
        print(f"  Manual perturbations per optimum: {n_perturbations_per_optimum}")
        print(f"  Manual perturbation strength: {perturbation_strength}")
    print(f"  Runs per perturbation: {n_runs_per_perturbation}")
    print(f"  Output directory: {timestamped_output_dir}")
    print("=" * 60)
    
    # Load instance data
    instance_data = load_instance_from_pkl(instance_path, instance_index)
    
    # Step 1: Run cuopt and collect history
    print("\nStep 1: Running cuopt and collecting history...")
    all_histories = []

    # Todo: currently only support one cuopt run for one instance
    # for run_idx in range(n_cuopt_runs):
    # print(f"  Running cuopt {run_idx + 1}/{n_cuopt_runs}...")
    result_data = test_callback(
        instance_path=instance_path,
        hgs_solution_path=hgs_solution_path,
        instance_index=instance_index
    )

    history = result_data.get('history', [])
    if history:
        all_histories.append(history)
        print(f"    Collected {len(history)} history records")
    hgs_cost = result_data['result'].get('hgs_cost')
    gap = result_data['result'].get('gap')
    num_orders = result_data['result'].get('num_orders')

    # Combine all histories
    combined_history = []
    for hist in all_histories:
        combined_history.extend(hist)
    
    # Step 2: Extract local optima
    print("\nStep 2: Extracting local optima...")
    temp_global_history = {'history': combined_history}
    _, trial_solutions, trial_metadata = extract_last_solution_per_trial(
        temp_global_history, num_orders=None
    )
    print(f"  Found {len(trial_solutions)} local optima")
    
    # Step 3: Identify unique local optima
    print("\nStep 3: Identifying unique local optima...")
    unique_optima = []
    optimum_id_map = {}
    duplicate_groups = {}  # Map from edges_tuple to list of original indices
    
    for idx, (opt_sol, opt_meta) in enumerate(zip(trial_solutions, trial_metadata)):
        num_routes = opt_meta.get('num_routes_after', 0)
        if num_routes is None:
            num_routes = 0
        # Infer correct num_routes if needed
        inferred_num_routes = infer_num_routes_from_solution_flat(opt_sol, num_orders)
        if inferred_num_routes > 0 and inferred_num_routes != num_routes:
            num_routes = inferred_num_routes
        
        routes = solution_flat_to_routes(opt_sol, num_routes, num_orders)
        edges = extract_edges_from_routes(routes)
        edges_tuple = tuple(sorted(edges))
        
        if edges_tuple not in optimum_id_map:
            optimum_id = len(unique_optima)
            optimum_id_map[edges_tuple] = optimum_id
            duplicate_groups[edges_tuple] = [idx]
            unique_optima.append({
                'solution': opt_sol,
                'metadata': opt_meta,
                'edges': edges,
                'optimum_id': optimum_id
            })
        else:
            # This is a duplicate
            duplicate_groups[edges_tuple].append(idx)
    
    print(f"  Found {len(unique_optima)} unique local optima")
    
    # Print duplicate information and verify solutions are actually the same
    duplicates_found = False
    for edges_tuple, indices in duplicate_groups.items():
        if len(indices) > 1:
            duplicates_found = True
            unique_id = optimum_id_map[edges_tuple]
            print(f"  Duplicate: IDs {indices} map to unique optimum ID {unique_id}")
            
            # Verify solutions are actually the same
            if len(indices) >= 2:
                idx1, idx2 = indices[0], indices[1]
                sol1 = trial_solutions[idx1]
                sol2 = trial_solutions[idx2]
                meta1 = trial_metadata[idx1]
                meta2 = trial_metadata[idx2]
                
                # Compare edge sets (this is what matters for solution similarity)
                num_routes1 = meta1.get('num_routes_after', 0) or 0
                num_routes2 = meta2.get('num_routes_after', 0) or 0
                # Infer num_routes if needed
                inferred_num_routes1 = infer_num_routes_from_solution_flat(sol1, num_orders)
                inferred_num_routes2 = infer_num_routes_from_solution_flat(sol2, num_orders)
                if inferred_num_routes1 > 0:
                    num_routes1 = inferred_num_routes1
                if inferred_num_routes2 > 0:
                    num_routes2 = inferred_num_routes2
                
                routes1 = solution_flat_to_routes(sol1, num_routes1, num_orders)
                routes2 = solution_flat_to_routes(sol2, num_routes2, num_orders)
                edges1 = extract_edges_from_routes(routes1)
                edges2 = extract_edges_from_routes(routes2)
                
                if edges1 == edges2:
                    print(f"    ✓ Solutions {idx1} and {idx2} have identical edge sets ({len(edges1)} edges)")
                else:
                    only_in_1 = edges1 - edges2
                    only_in_2 = edges2 - edges1
                    intersection = edges1 & edges2
                    jaccard = len(intersection) / len(edges1 | edges2) if len(edges1 | edges2) > 0 else 0.0
                    print(f"    ✗ Solutions {idx1} and {idx2} have different edge sets:")
                    print(f"      Edges only in {idx1}: {len(only_in_1)}")
                    print(f"      Edges only in {idx2}: {len(only_in_2)}")
                    print(f"      Common edges: {len(intersection)}")
                    print(f"      Jaccard similarity: {jaccard:.4f}")
                
                # Compare costs
                cost1 = meta1.get('cost', None)
                cost2 = meta2.get('cost', None)
                if cost1 is not None and cost2 is not None:
                    if abs(cost1 - cost2) < 1e-6:
                        print(f"    ✓ Costs are identical: {cost1:.2f}")
                    else:
                        print(f"    ✗ Costs differ: {cost1:.2f} vs {cost2:.2f}")
                
                # Compare num_routes
                if num_routes1 == num_routes2:
                    print(f"    ✓ num_routes are identical: {num_routes1}")
                else:
                    print(f"    ✗ num_routes differ: {num_routes1} vs {num_routes2}")
    
    if not duplicates_found:
        print("  No duplicates found (all local optima are unique)")
    
    # Plot cost curve by trial BEFORE perturbation (after identifying unique optima)
    # Use the results from Step 3 to avoid recomputing
    print("\nPlotting cost curve by trial (before perturbation)...")
    temp_global_history = {'history': combined_history}
    # num_orders already loaded above for Step 3
    plot_filename = os.path.join(timestamped_output_dir, 'callback_cost_curve_by_trial.png')
    plot_cost_curve_by_trial(temp_global_history, num_orders=num_orders, filename=plot_filename,
                             trial_metadata=trial_metadata, duplicate_groups=duplicate_groups,
                             optimum_id_map=optimum_id_map, gap=gap)
    print(f"  Saved to: {plot_filename}")
    
    # Step 5: Collect datasets for each perturbation type
    all_datasets = {}
    
    # Type 2: Trial intermediate solutions (prioritize this)
    print("\n" + "=" * 60)
    print("Type 2: Trial Intermediate Solutions")
    print("=" * 60)
    dataset_trial = collect_trial_intermediate_perturbations(
        combined_history, unique_optima, num_orders, instance_data,
        n_runs_per_perturbation, time_limit_per_run
    )
    all_datasets['trial_intermediate'] = dataset_trial
    
    # Type 3: Fast search after last cycle finder
    print("\n" + "=" * 60)
    print("Type 3: Fast Search After Last Cycle Finder")
    print("=" * 60)
    dataset_cycle = collect_after_cycle_finder_perturbations(
        combined_history, unique_optima, num_orders, instance_data,
        n_runs_per_perturbation, time_limit_per_run
    )
    all_datasets['after_cycle_finder'] = dataset_cycle
    
    # Type 1: Manual perturbations (optional, controlled by collect_manual flag)
    if collect_manual:
        print("\n" + "=" * 60)
        print("Type 1: Manual Perturbations")
        print("=" * 60)
        dataset_manual = collect_manual_perturbations(
            unique_optima, num_orders, instance_data,
            n_perturbations_per_optimum, n_runs_per_perturbation,
            perturbation_strength, time_limit_per_run
        )
        all_datasets['manual'] = dataset_manual
    else:
        print("\n" + "=" * 60)
        print("Type 1: Manual Perturbations (SKIPPED)")
        print("=" * 60)
        all_datasets['manual'] = {'pairs': [], 'labels': [], 'metadata': []}
    
    # Step 6: Save combined dataset
    output_file = os.path.join(timestamped_output_dir, f'contrastive_dataset_extended_instance{instance_index}.pkl')
    
    output_data = {
        'instance_index': instance_index,
        'n_cuopt_runs': n_cuopt_runs,
        'n_perturbations_per_optimum': n_perturbations_per_optimum,
        'n_runs_per_perturbation': n_runs_per_perturbation,
        'perturbation_strength': perturbation_strength,
        'n_unique_optima': len(unique_optima),
        'unique_optima': unique_optima,
        'datasets': all_datasets,
        'timestamp': timestamp
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
    
    # Step 7: Print summary
    print(f"\n{'='*60}")
    print(f"Extended Dataset Collection Completed!")
    print(f"  Output directory: {timestamped_output_dir}")
    print(f"  File saved to: {output_file}")
    for pert_type, dataset in all_datasets.items():
        if dataset and dataset.get('pairs'):
            n_pairs = len(dataset['pairs'])
            similar = sum(1 for l in dataset.get('labels', []) if l.get('similarity_label') == 'similar')
            distant = sum(1 for l in dataset.get('labels', []) if l.get('similarity_label') == 'distant')
            print(f"  {pert_type}: {n_pairs} pairs (similar: {similar}, distant: {distant})")
    print("=" * 60)
    
    return output_data


def collect_manual_perturbations(unique_optima, num_orders, instance_data,
                                 n_perturbations_per_optimum, n_runs_per_perturbation,
                                 perturbation_strength, time_limit_per_run):
    """Collect dataset from manual perturbations."""
    dataset = {'pairs': [], 'labels': [], 'metadata': []}
    
    for opt_idx, optimum in enumerate(unique_optima):
        print(f"  Processing local optimum {opt_idx + 1}/{len(unique_optima)} (ID: {optimum['optimum_id']})...")
        
        local_optimum = optimum['solution']
        num_routes = optimum['metadata'].get('num_routes_after', 0)
        if num_routes is None:
            num_routes = 0
        # Infer correct num_routes if needed
        inferred_num_routes = infer_num_routes_from_solution_flat(local_optimum, num_orders)
        if inferred_num_routes > 0 and inferred_num_routes != num_routes:
            num_routes = inferred_num_routes
        
        original_cost = optimum['metadata'].get('cost', None)
        
        for pert_idx in range(n_perturbations_per_optimum):
            print(f"    Manual perturbation {pert_idx + 1}/{n_perturbations_per_optimum}...")
            
            perturbed_sol = perturb_solution_simple(
                local_optimum, num_routes, num_orders,
                perturbation_strength=perturbation_strength,
                seed=opt_idx * 1000 + pert_idx
            )
            
            result = analyze_perturbed_solution(
                perturbed_sol, local_optimum, optimum['optimum_id'],
                num_routes, num_orders, unique_optima, instance_data,
                n_runs_per_perturbation, time_limit_per_run, 'manual'
            )
            
            if result:
                dataset['pairs'].append({
                    'original_optimum': local_optimum,
                    'perturbed_solution': perturbed_sol,
                    'original_optimum_id': optimum['optimum_id'],
                    'original_cost': original_cost
                })
                
                # Build similarity label
                return_freq = result['return_frequency']
                if return_freq > 0.5:
                    similarity_label = 'similar'
                elif return_freq < 0.1:
                    similarity_label = 'distant'
                else:
                    similarity_label = 'moderate'
                
                dataset['labels'].append({
                    'return_frequency': return_freq,
                    'convergence_distribution': result['convergence_distribution'],
                    'n_runs': result['n_runs'],
                    'perturbation_strength': perturbation_strength,
                    'similarity_label': similarity_label,
                    'perturbation_type': 'manual'
                })
                
                dataset['metadata'].append({
                    'optimum_idx': opt_idx,
                    'perturbation_idx': pert_idx,
                    'convergence_results': result['convergence_results']
                })
                
                print(f"      Return frequency: {return_freq:.2%} ({similarity_label})")
    
    return dataset

def collect_trial_intermediate_perturbations(combined_history, unique_optima, num_orders, instance_data,
                                            n_runs_per_perturbation, time_limit_per_run):
    """Collect dataset from trial intermediate solutions."""
    dataset = {'pairs': [], 'labels': [], 'metadata': []}
    
    temp_global_history = {'history': combined_history}
    perturbations = extract_perturbations_from_trial_history(temp_global_history, num_orders)
    
    print(f"  Found {len(perturbations)} trials with intermediate solutions")
    
    # Map local optima to unique optima
    optimum_map = {}
    for opt in unique_optima:
        edges_tuple = tuple(sorted(opt['edges']))
        optimum_map[edges_tuple] = opt
    
    pair_idx = 0
    for pert_data in perturbations:
        local_optimum = pert_data['local_optimum']
        num_routes = pert_data['local_optimum_metadata'].get('num_routes_after', 0)
        if num_routes is None:
            num_routes = 0
        
        # Find the unique optimum ID
        routes = solution_flat_to_routes(local_optimum, num_routes, num_orders)
        edges = extract_edges_from_routes(routes)
        edges_tuple = tuple(sorted(edges))
        unique_opt = optimum_map.get(edges_tuple)
        
        if unique_opt is None:
            continue
        
        optimum_id = unique_opt['optimum_id']
        original_cost = pert_data['local_optimum_metadata'].get('cost', None)
        
        # Analyze each perturbed solution
        for pert_sol_data in pert_data['perturbed_solutions']:
            perturbed_sol = pert_sol_data['solution']
            
            # Infer num_routes for this specific perturbed solution
            # Each intermediate solution may have different num_routes
            pert_num_routes = infer_num_routes_from_solution_flat(perturbed_sol, num_orders)
            if pert_num_routes <= 0:
                # Skip if we can't infer num_routes
                print(f"    Warning: Skipping perturbed solution (cannot infer num_routes)")
                continue
            
            # Validate the perturbed solution before using it
            try:
                validate_cuopt_solution(perturbed_sol, num_orders, 
                                       f" (perturbed solution in trial {pert_data['local_optimum_metadata'].get('local_search_id', -1)})")
            except ValueError as e:
                print(f"    Warning: Skipping invalid perturbed solution: {e}")
                continue
            
            result = analyze_perturbed_solution(
                perturbed_sol, local_optimum, optimum_id,
                pert_num_routes, num_orders, unique_optima, instance_data,
                n_runs_per_perturbation, time_limit_per_run, 'trial_intermediate'
            )
            
            if result:
                dataset['pairs'].append({
                    'original_optimum': local_optimum,
                    'perturbed_solution': perturbed_sol,
                    'original_optimum_id': optimum_id,
                    'original_cost': original_cost
                })
                
                return_freq = result['return_frequency']
                if return_freq > 0.5:
                    similarity_label = 'similar'
                elif return_freq < 0.1:
                    similarity_label = 'distant'
                else:
                    similarity_label = 'moderate'
                
                dataset['labels'].append({
                    'return_frequency': return_freq,
                    'convergence_distribution': result['convergence_distribution'],
                    'n_runs': result['n_runs'],
                    'similarity_label': similarity_label,
                    'perturbation_type': 'trial_intermediate'
                })
                
                dataset['metadata'].append({
                    'pair_idx': pair_idx,
                    'convergence_results': result['convergence_results']
                })
                
                pair_idx += 1
                if pair_idx % 10 == 0:
                    print(f"    Processed {pair_idx} pairs...")
    
    print(f"  Total pairs collected: {len(dataset['pairs'])}")
    return dataset

def collect_after_cycle_finder_perturbations(combined_history, unique_optima, num_orders, instance_data,
                                             n_runs_per_perturbation, time_limit_per_run):
    """Collect dataset from fast search after last cycle finder."""
    dataset = {'pairs': [], 'labels': [], 'metadata': []}
    
    temp_global_history = {'history': combined_history}
    perturbations = extract_perturbations_after_last_cycle_finder(temp_global_history, num_orders)
    
    print(f"  Found {len(perturbations)} trials with fast search after cycle finder")
    
    # Map local optima to unique optima
    optimum_map = {}
    for opt in unique_optima:
        edges_tuple = tuple(sorted(opt['edges']))
        optimum_map[edges_tuple] = opt
    
    pair_idx = 0
    for pert_data in perturbations:
        local_optimum = pert_data['local_optimum']
        num_routes = pert_data['local_optimum_metadata'].get('num_routes_after', 0)
        if num_routes is None:
            num_routes = 0
        
        # Find the unique optimum ID
        routes = solution_flat_to_routes(local_optimum, num_routes, num_orders)
        edges = extract_edges_from_routes(routes)
        edges_tuple = tuple(sorted(edges))
        unique_opt = optimum_map.get(edges_tuple)
        
        if unique_opt is None:
            continue
        
        optimum_id = unique_opt['optimum_id']
        original_cost = pert_data['local_optimum_metadata'].get('cost', None)
        
        # Analyze each perturbed solution
        for pert_sol_data in pert_data['perturbed_solutions']:
            perturbed_sol = pert_sol_data['solution']
            
            # Infer num_routes for this specific perturbed solution
            # Each intermediate solution may have different num_routes
            pert_num_routes = infer_num_routes_from_solution_flat(perturbed_sol, num_orders)
            if pert_num_routes <= 0:
                # Skip if we can't infer num_routes
                print(f"    Warning: Skipping perturbed solution (cannot infer num_routes)")
                continue
            
            # Validate the perturbed solution before using it
            try:
                validate_cuopt_solution(perturbed_sol, num_orders, 
                                       f" (perturbed solution after cycle finder in trial {pert_data['local_optimum_metadata'].get('local_search_id', -1)})")
            except ValueError as e:
                print(f"    Warning: Skipping invalid perturbed solution: {e}")
                continue
            
            result = analyze_perturbed_solution(
                perturbed_sol, local_optimum, optimum_id,
                pert_num_routes, num_orders, unique_optima, instance_data,
                n_runs_per_perturbation, time_limit_per_run, 'after_cycle_finder'
            )
            
            if result:
                dataset['pairs'].append({
                    'original_optimum': local_optimum,
                    'perturbed_solution': perturbed_sol,
                    'original_optimum_id': optimum_id,
                    'original_cost': original_cost
                })
                
                return_freq = result['return_frequency']
                if return_freq > 0.5:
                    similarity_label = 'similar'
                elif return_freq < 0.1:
                    similarity_label = 'distant'
                else:
                    similarity_label = 'moderate'
                
                dataset['labels'].append({
                    'return_frequency': return_freq,
                    'convergence_distribution': result['convergence_distribution'],
                    'n_runs': result['n_runs'],
                    'similarity_label': similarity_label,
                    'perturbation_type': 'after_cycle_finder'
                })
                
                dataset['metadata'].append({
                    'pair_idx': pair_idx,
                    'convergence_results': result['convergence_results']
                })
                
                pair_idx += 1
                if pair_idx % 10 == 0:
                    print(f"    Processed {pair_idx} pairs...")
    
    print(f"  Total pairs collected: {len(dataset['pairs'])}")
    return dataset


def run_multiple_instances(instance_path, hgs_solution_path, instance_indices='all', max_instances=None):
    """
    Run test_callback on multiple instances.

    Args:
        instance_path: Path to pkl file with CVRP instances
        hgs_solution_path: Path to pkl file with HGS solutions
        instance_indices: List of instance indices to run, or 'all' for all instances
        max_instances: Maximum number of instances to run (None for all)

    Returns:
        list: List of result dictionaries for each instance
    """
    # Load instances to get total count
    with open(instance_path, 'rb') as f:
        instances = pickle.load(f)
    total_instances = len(instances)

    # Determine which instances to run
    if instance_indices == 'all':
        instance_indices = list(range(total_instances))
    elif isinstance(instance_indices, int):
        instance_indices = [instance_indices]

    # Limit number of instances if specified
    if max_instances is not None:
        instance_indices = instance_indices[:max_instances]

    print("=" * 60)
    print(f"Running {len(instance_indices)} instances (indices: {instance_indices})")
    print("=" * 60)

    all_results = []

    for idx, instance_idx in enumerate(instance_indices):
        print(f"\n{'=' * 60}")
        print(f"Instance {idx + 1}/{len(instance_indices)}: Instance Index {instance_idx}")
        print(f"{'=' * 60}")

        result = test_callback(
            instance_path=instance_path,
            hgs_solution_path=hgs_solution_path,
            instance_index=instance_idx
        )
        all_results.append(result['result'])

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    valid_results = [r for r in all_results if r is not None and 'gap' in r and r['gap'] is not None]

    if valid_results:
        gaps = [r['gap'] for r in valid_results]
        costs = [r['final_cost'] for r in valid_results]
        hgs_costs = [r['hgs_cost'] for r in valid_results]

        print(f"Total instances processed: {len(valid_results)}")
        print(f"\nCost Statistics:")
        print(f"  My solution - Min: {min(costs):.2f}, Max: {max(costs):.2f}, Avg: {np.mean(costs):.2f}")
        if hgs_costs:
            print(
                f"  HGS solution - Min: {min(hgs_costs):.2f}, Max: {max(hgs_costs):.2f}, Avg: {np.mean(hgs_costs):.2f}")
        print(f"\nGap Statistics:")
        print(f"  Min gap: {min(gaps):.2f}%")
        print(f"  Max gap: {max(gaps):.2f}%")
        print(f"  Average gap: {np.mean(gaps):.2f}%")
        print(f"  Median gap: {np.median(gaps):.2f}%")
        print(f"  Better than HGS: {sum(1 for g in gaps if g < 0)} instances")
        print(f"  Equal to HGS: {sum(1 for g in gaps if g == 0)} instances")
        print(f"  Worse than HGS: {sum(1 for g in gaps if g > 0)} instances")
    else:
        print("No valid results to summarize.")

    print("=" * 60)

    return all_results


def run_1000_instances_and_plot_gap(instance_path, hgs_solution_path, n_instances=1000, output_dir='plot',
                                    delay_seconds=1):
    import time
    from datetime import datetime

    # Load instances to get total count
    with open(instance_path, 'rb') as f:
        instances = pickle.load(f)
    total_instances = len(instances)

    # Limit to available instances
    n_instances = min(n_instances, total_instances)
    instance_indices = list(range(n_instances))

    print("=" * 60)
    print(f"Running {n_instances} instances to compute gap with HGS")
    print("=" * 60)

    # Checkpoint file
    checkpoint_file = os.path.join(output_dir, f'gap_checkpoint_{n_instances}instances.pkl')
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint if exists
    all_results = []
    start_idx = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
            all_results = checkpoint_data.get('all_results', [])
            start_idx = len(all_results)
            print(f"Loaded checkpoint: {start_idx} instances already processed")

    # Run instances
    for idx, instance_idx in enumerate(instance_indices[start_idx:], start=start_idx):
        print(f"\n{'=' * 60}")
        print(f"Instance {idx + 1}/{n_instances}: Instance Index {instance_idx}")
        print(f"{'=' * 60}")

        result = test_callback(
            instance_path=instance_path,
            hgs_solution_path=hgs_solution_path,
            instance_index=instance_idx
        )

        if result and result.get('result') and result['result'].get('gap') is not None:
            all_results.append({
                'instance_index': instance_idx,
                'gap': result['result']['gap'],
                'final_cost': result['result'].get('final_cost'),
                'hgs_cost': result['result'].get('hgs_cost')
            })
            print(f"  Gap: {result['result']['gap']:.2f}%")
        else:
            print(f"  Warning: No gap computed for instance {instance_idx}")
            all_results.append({
                'instance_index': instance_idx,
                'gap': None,
                'error': 'No gap computed'
            })

        # Save checkpoint every 10 successful instances
        if (idx + 1) % 10 == 0:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({'all_results': all_results}, f)
            print(f"  Checkpoint saved: {idx + 1} instances processed")

        # Delay between instances
        if delay_seconds > 0 and idx < n_instances - 1:
            time.sleep(delay_seconds)

    # Filter valid results (with gap)
    valid_results = [r for r in all_results if r.get('gap') is not None]

    if not valid_results:
        print("\nNo valid results with gap computed!")
        return {'all_results': all_results, 'valid_results': []}

    # Extract gap values
    gaps = [r['gap'] for r in valid_results]

    # Statistics
    print("\n" + "=" * 60)
    print("GAP STATISTICS")
    print("=" * 60)
    print(f"Total instances processed: {len(all_results)}")
    print(f"Valid results (with gap): {len(valid_results)}")
    print(f"Gap - Min: {min(gaps):.2f}%")
    print(f"Gap - Max: {max(gaps):.2f}%")
    print(f"Gap - Mean: {np.mean(gaps):.2f}%")
    print(f"Gap - Median: {np.median(gaps):.2f}%")
    print(f"Gap - Std: {np.std(gaps):.2f}%")
    print("=" * 60)

    # Plot box plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'gap_boxplot_{n_instances}instances_{timestamp}.png')

    plt.figure(figsize=(8, 6))
    bp = plt.boxplot(gaps, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5))

    plt.ylabel('Gap (%)', fontsize=12)
    plt.title(f'Gap Distribution (n={len(valid_results)} instances)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Add statistics text
    stats_text = f'Mean: {np.mean(gaps):.2f}%\nMedian: {np.median(gaps):.2f}%\nStd: {np.std(gaps):.2f}%'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nBox plot saved to: {output_file}")
    plt.close()

    return {
        'all_results': all_results,
        'valid_results': valid_results,
        'gaps': gaps,
        'statistics': {
            'min': min(gaps),
            'max': max(gaps),
            'mean': np.mean(gaps),
            'median': np.median(gaps),
            'std': np.std(gaps)
        }
    }


def save_full_history(instance_path, hgs_solution_path, instance_index=0, output_dir='plot'):
    """
    Run test_callback and save the complete history to a pickle file.
    
    Args:
        instance_path: Path to pkl file with CVRP instances
        hgs_solution_path: Path to pkl file with HGS solutions
        instance_index: Index of instance to use (default: 0)
        output_dir: Output directory for saving history file
    
    Returns:
        str: Path to the saved history file
    """
    import datetime
    import pickle
    
    print("=" * 60)
    print(f"Saving full history for instance (index={instance_index})")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load instance data
    instance_data = load_instance_from_pkl(instance_path, instance_index)
    
    # Run test_callback to get full history
    print("\nRunning test_callback...")
    result_data = test_callback(
        instance_path=instance_path,
        hgs_solution_path=hgs_solution_path,
        instance_index=instance_index
    )
    
    # Extract history and other data
    history = result_data.get('history', [])
    result = result_data.get('result', {})
    avg_diversity = result_data.get('avg_diversity', None)
    
    # Create output dictionary with all data
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_data = {
        'instance_index': instance_index,
        'timestamp': timestamp,
        'history': history,
        'result': result,
        'avg_diversity': avg_diversity,
        'n_history_records': len(history),
        'instance_data': instance_data
    }
    
    # Save to file
    output_file = os.path.join(output_dir, f'full_history_instance{instance_index}_{timestamp}.pkl')
    
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\n{'='*60}")
    print(f"Full history saved successfully!")
    print(f"  Instance index: {instance_index}")
    print(f"  Total history records: {len(history)}")
    print(f"  File saved to: {output_file}")
    print("=" * 60)
    
    # Print some statistics
    if history:
        costs = [r.get('cost_after') for r in history if r.get('cost_after') is not None]
        if costs:
            print(f"\nHistory statistics:")
            print(f"  Min cost: {min(costs):.2f}")
            print(f"  Max cost: {max(costs):.2f}")
            print(f"  Mean cost: {np.mean(costs):.2f}")
            print(f"  Median cost: {np.median(costs):.2f}")
    
    return output_file


def run_100_trials_and_compare_diversity(instance_path, hgs_solution_path, instance_index=0, 
                                         n_trials=100, output_dir='plot', delay_seconds=3):
    """
    Run 100 trials, collect gap and diversity, then compare top 20% and bottom 20% with boxplot.
    
    Args:
        instance_path: Path to pkl file with CVRP instances
        hgs_solution_path: Path to pkl file with HGS solutions
        instance_index: Index of instance to use
        n_trials: Number of trials to run (default: 100)
        output_dir: Output directory for saving plots
        delay_seconds: Delay between trials in seconds (default: 3)
    """
    import datetime
    import random
    import time
    import pickle
    
    print("=" * 60)
    print(f"Running {n_trials} trials for gap and diversity analysis")
    print(f"Instance Index: {instance_index}")
    print(f"Delay between trials: {delay_seconds} seconds")
    print("=" * 60)
    
    # Collect results from all trials
    all_results = []
    error_count = 0
    
    # Create checkpoint file for saving progress
    checkpoint_file = os.path.join(output_dir, f'trial_checkpoint_instance{instance_index}.pkl')
    os.makedirs(output_dir, exist_ok=True)
    
    for trial_idx in range(n_trials):
        print(f"\n{'='*60}")
        print(f"Trial {trial_idx + 1}/{n_trials}")
        print(f"{'='*60}")
        
        try:
            # Use different random seed for each trial
            # Note: We need to modify test_callback to accept seed parameter
            # For now, we'll use trial_idx as a seed offset
            print(f"  Starting test_callback...")
            import time as time_module
            start_time = time_module.time()
            
            result_data = test_callback(
                instance_path=instance_path,
                hgs_solution_path=hgs_solution_path,
                instance_index=instance_index
            )
            
            elapsed_time = time_module.time() - start_time
            print(f"  test_callback completed in {elapsed_time:.2f} seconds")
            
            result = result_data['result']
            avg_diversity = result_data.get('avg_diversity', None)
            
            if result and result.get('gap') is not None and avg_diversity is not None:
                all_results.append({
                    'trial': trial_idx,
                    'gap': result['gap'],
                    'avg_diversity': avg_diversity,
                    'final_cost': result.get('final_cost'),
                    'hgs_cost': result.get('hgs_cost')
                })
                print(f"  Gap: {result['gap']:.2f}%, Diversity: {avg_diversity:.4f}")
                
                # Save checkpoint every 10 successful trials
                if len(all_results) % 10 == 0:
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump({
                            'all_results': all_results,
                            'trial_idx': trial_idx,
                            'error_count': error_count,
                            'instance_index': instance_index
                        }, f)
                    print(f"  Checkpoint saved: {len(all_results)} valid trials")
            else:
                print(f"  Skipped: Missing gap or diversity data")
                if result:
                    print(f"    - Gap: {result.get('gap')}")
                    print(f"    - Diversity: {avg_diversity}")
                else:
                    print(f"    - Result is None")
            
            # Add delay between trials (except for the last one)
            if trial_idx < n_trials - 1:
                print(f"\n  Waiting {delay_seconds} seconds before next trial...")
                time.sleep(delay_seconds)
                
        except KeyboardInterrupt:
            print(f"\n\n  Trial interrupted by user. Stopping...")
            print(f"  Collected {len(all_results)} valid trials so far.")
            # Save checkpoint before stopping
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'all_results': all_results,
                    'trial_idx': trial_idx,
                    'error_count': error_count,
                    'instance_index': instance_index
                }, f)
            print(f"  Checkpoint saved to: {checkpoint_file}")
            break
    
    print(f"\n{'='*60}")
    print(f"Trial Summary:")
    print(f"  Total trials: {n_trials}")
    print(f"  Successful: {len(all_results)}")
    print(f"  Errors: {error_count}")
    print(f"  Success rate: {len(all_results)/n_trials*100:.1f}%")
    print("=" * 60)
    
    # Save final checkpoint
    with open(checkpoint_file, 'wb') as f:
        pickle.dump({
            'all_results': all_results,
            'trial_idx': n_trials - 1,
            'error_count': error_count,
            'instance_index': instance_index,
            'completed': True
        }, f)
    print(f"Final checkpoint saved to: {checkpoint_file}")
    
    if len(all_results) < 2:
        print(f"\nError: Only {len(all_results)} valid trials collected. Need at least 2 to compare.")
        print(f"  Cannot generate comparison plot.")
        return
    elif len(all_results) < 10:
        print(f"\nWarning: Only {len(all_results)} valid trials collected. Recommended at least 10.")
        print(f"Continuing with available data...")
    
    # Sort by gap (ascending: best gap is smallest)
    all_results_sorted = sorted(all_results, key=lambda x: x['gap'])
    
    # Get top 20% (best gap) and bottom 20% (worst gap)
    n_top = max(1, int(len(all_results_sorted) * 0.2))
    n_bottom = max(1, int(len(all_results_sorted) * 0.2))
    
    top_20_percent = all_results_sorted[:n_top]
    bottom_20_percent = all_results_sorted[-n_bottom:]
    
    print(f"\nTop 20% (best gap): {len(top_20_percent)} trials")
    top_gap_range = f"{top_20_percent[0]['gap']:.2f}% to {top_20_percent[-1]['gap']:.2f}%"
    print(f"  Gap range: {top_gap_range}")
    print(f"  Avg diversity: {np.mean([r['avg_diversity'] for r in top_20_percent]):.4f}")
    
    print(f"\nBottom 20% (worst gap): {len(bottom_20_percent)} trials")
    bottom_gap_range = f"{bottom_20_percent[0]['gap']:.2f}% to {bottom_20_percent[-1]['gap']:.2f}%"
    print(f"  Gap range: {bottom_gap_range}")
    print(f"  Avg diversity: {np.mean([r['avg_diversity'] for r in bottom_20_percent]):.4f}")
    
    # Extract diversity values for boxplot
    top_diversity = [r['avg_diversity'] for r in top_20_percent]
    bottom_diversity = [r['avg_diversity'] for r in bottom_20_percent]
    
    # Statistical test for significance
    from scipy import stats
    # Use Mann-Whitney U test (non-parametric, more robust)
    statistic, p_value = stats.mannwhitneyu(top_diversity, bottom_diversity, alternative='two-sided')
    
    # Determine significance level
    if p_value < 0.001:
        sig_level = '***'
        sig_text = 'p < 0.001 ***'
    elif p_value < 0.01:
        sig_level = '**'
        sig_text = f'p = {p_value:.4f} **'
    elif p_value < 0.05:
        sig_level = '*'
        sig_text = f'p = {p_value:.4f} *'
    else:
        sig_level = 'ns'
        sig_text = f'p = {p_value:.4f} (ns)'
    
    print(f"\nStatistical Test (Mann-Whitney U):")
    print(f"  Statistic: {statistic:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significance: {sig_text}")
    
    # Create boxplot
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f'diversity_comparison_instance{instance_index}_{timestamp}.png')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for boxplot
    data_to_plot = [top_diversity, bottom_diversity]
    labels = [f'Top 20%\n(Best Gap)\nn={len(top_diversity)}', 
              f'Bottom 20%\n(Worst Gap)\nn={len(bottom_diversity)}']
    
    # Create boxplot
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                    widths=0.6, showmeans=True, meanline=True)
    
    # Customize boxplot colors
    colors = ['#2E86AB', '#E63946']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize other elements
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    
    plt.setp(bp['means'], linestyle='--', linewidth=2)
    
    ax.set_ylabel('Average Diversity (Jaccard Distance)', fontsize=13, fontweight='bold')
    title = 'Diversity Comparison: Top 20% vs Bottom 20% by Gap'
    if sig_level and sig_level != 'ns':
        title += f' ({sig_level})'
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add significance bar if significant
    if p_value is not None and p_value < 0.05:
        # Draw significance bar
        y_max = max(max(top_diversity), max(bottom_diversity))
        y_min = min(min(top_diversity), min(bottom_diversity))
        y_range = y_max - y_min
        y_bar = y_max + y_range * 0.1
        
        # Draw horizontal line
        ax.plot([1, 2], [y_bar, y_bar], 'k-', linewidth=1.5)
        # Draw vertical lines
        ax.plot([1, 1], [y_max + y_range * 0.05, y_bar], 'k-', linewidth=1.5)
        ax.plot([2, 2], [y_max + y_range * 0.05, y_bar], 'k-', linewidth=1.5)
        # Add significance text
        ax.text(1.5, y_bar + y_range * 0.02, sig_text, 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add statistics text with gap ranges
    stats_text = (f"Top 20% Gap Range: {top_gap_range}\n"
                  f"Bottom 20% Gap Range: {bottom_gap_range}\n\n"
                  f"Top 20% Mean: {np.mean(top_diversity):.4f}\n"
                  f"Bottom 20% Mean: {np.mean(bottom_diversity):.4f}\n"
                  f"Difference: {np.mean(bottom_diversity) - np.mean(top_diversity):.4f}")
    if p_value is not None:
        stats_text += f"\n\n{sig_text}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n  Boxplot saved to: {filename}")
    
    # Return results for further analysis
    return {
        'all_results': all_results,
        'top_20_percent': top_20_percent,
        'bottom_20_percent': bottom_20_percent,
        'top_diversity': top_diversity,
        'bottom_diversity': bottom_diversity
    }

if __name__ == "__main__":
    import sys
    import os
    
    # load paths
    instance_path = "/home/jieyi/cvrp100_uniform.pkl"
    hgs_solution_path = "/home/jieyi/hgs_cvrp100_uniform.pkl"
    if not os.path.exists(instance_path):
        print(f"Warning: Instance file not found: {instance_path}")
        print("Using default problem generation instead.")
        instance_path = None
    if not os.path.exists(hgs_solution_path):
        print(f"Warning: HGS solution file not found: {hgs_solution_path}")
        print("Skipping HGS comparison.")
        hgs_solution_path = None
    
    # Run single instance by default, or multiple if specified
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all' or sys.argv[1] == 'all':
            # Run all instances
            run_multiple_instances(instance_path, hgs_solution_path, instance_indices='all')
        elif sys.argv[1] == '--multiple':
            # Run multiple specific instances
            indices = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else [0, 1, 2]
            run_multiple_instances(instance_path, hgs_solution_path, instance_indices=indices)
        elif sys.argv[1].startswith('--max='):
            # Run up to N instances
            max_n = int(sys.argv[1].split('=')[1])
            run_multiple_instances(instance_path, hgs_solution_path, instance_indices='all', max_instances=max_n)
        elif sys.argv[1] == '--diversity':
            # Run 100 trials and compare diversity
            instance_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            n_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 100
            delay_seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 3
            run_100_trials_and_compare_diversity(
                instance_path=instance_path,
                hgs_solution_path=hgs_solution_path,
                instance_index=instance_idx,
                n_trials=n_trials,
                delay_seconds=delay_seconds
            )
        elif sys.argv[1] == '--gap':
            # Run 1000 instances and plot gap boxplot
            n_instances = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
            delay_seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 1
            run_1000_instances_and_plot_gap(
                instance_path=instance_path,
                hgs_solution_path=hgs_solution_path,
                n_instances=n_instances,
                delay_seconds=delay_seconds
            )
        elif sys.argv[1] == '--history':
            # Save full history to pickle file
            instance_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            save_full_history(
                instance_path=instance_path,
                hgs_solution_path=hgs_solution_path,
                instance_index=instance_idx
            )
        elif sys.argv[1] == '--contrastive':
            # Collect contrastive learning dataset (using extended version which includes manual perturbations)
            # Usage: --contrastive <instance_idx> [n_cuopt_runs] [n_perturbations] [n_runs] [pert_strength] [collect_manual]
            # collect_manual: 1 to collect manual perturbations, 0 to skip (default: 0, skip manual to prioritize others)
            instance_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            n_cuopt_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
            n_perturbations = int(sys.argv[4]) if len(sys.argv) > 4 else 1
            n_runs = int(sys.argv[5]) if len(sys.argv) > 5 else 100
            pert_strength = float(sys.argv[6]) if len(sys.argv) > 6 else 0.1
            collect_manual = bool(int(sys.argv[7])) if len(sys.argv) > 7 else False  # Default: False to prioritize others
            result = collect_contrastive_dataset_extended(
                instance_path=instance_path,
                hgs_solution_path=hgs_solution_path,
                instance_index=instance_idx,
                n_cuopt_runs=n_cuopt_runs,
                n_perturbations_per_optimum=n_perturbations,
                n_runs_per_perturbation=n_runs,
                perturbation_strength=pert_strength,
                collect_manual=collect_manual
            )
            # Automatically visualize the dataset
            if result:
                timestamp = result.get('timestamp', '')
                dataset_dir = os.path.join('plot', f'dataset_{timestamp}')
                dataset_file = os.path.join(dataset_dir, f'contrastive_dataset_extended_instance{instance_idx}.pkl')
                if os.path.exists(dataset_file):
                    print("\nGenerating visualizations...")
                    visualize_contrastive_dataset(dataset_file, output_dir=dataset_dir)
        elif sys.argv[1] == '--visualize':
            # Visualize a saved dataset file
            dataset_file = sys.argv[2] if len(sys.argv) > 2 else None
            if dataset_file and os.path.exists(dataset_file):
                visualize_contrastive_dataset(dataset_file, output_dir='plot')
            else:
                print(f"Error: Dataset file not found: {dataset_file}")
                print("Usage: python test_landscape.py --visualize <dataset_file.pkl>")
        elif sys.argv[1] == '--random':
            # Generate random CVRP instance (no HGS solution)
            # Usage: --random <n_customers> (default: 1000)
            n_customers = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
            n_locations = n_customers + 1  # customers + 1 depot
            print(f"Generating random CVRP instance with {n_customers} customers ({n_locations} locations including depot)...")
            test_callback(
                instance_path=None,
                hgs_solution_path=None,
                instance_index=0,
                n_locations=n_locations
            )
        else:
            # Single instance with specified index
            instance_index = int(sys.argv[1])
            test_callback(
                instance_path=instance_path,
                hgs_solution_path=hgs_solution_path,
                instance_index=instance_index
            )
    else:
        # Default: single instance
        test_callback(
            instance_path=instance_path,
            hgs_solution_path=hgs_solution_path,
            instance_index=0
        )

