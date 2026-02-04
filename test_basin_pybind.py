#!/usr/bin/env python3
"""
cuOpt VRP Local Search Test
Replicates cuOpt's run_best_local_search logic with pybind interface
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import random
from test_load_data import load_instance_from_pkl, load_hgs_solution_from_pkl, calculate_gap
from utils import (
    solution_flat_to_routes,
    load_solution_from_trajectory,
    get_basin_paths
)

# Add the cuOpt build directory to Python path
cuopt_build_path = os.path.join(os.path.dirname(__file__), 'cpp', 'build', 'install', 'lib', 'python3', 'dist-packages')
if cuopt_build_path not in sys.path:
    sys.path.insert(0, cuopt_build_path)


def create_vrp_instance(num_locations=20, num_vehicles=5, vehicle_capacity=100):
    """Create VRP instance: location 0 = depot, locations 1..n-1 = orders"""
    import cuopt_pybind
    
    num_orders = num_locations
    
    node_coords = np.random.rand(num_locations, 2) * 10
    
    cost_matrix = np.zeros((num_locations, num_locations), dtype=np.float32)
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                distance = np.sqrt(np.sum((node_coords[i] - node_coords[j])**2))
                cost_matrix[i, j] = distance
    
    demands = [0] + list(np.random.randint(1, 10, num_locations-1))
    capacities = [vehicle_capacity] * num_vehicles
    
    cuopt_env = cuopt_pybind.VrpLS(num_locations, num_vehicles, num_orders)
    cuopt_env.add_cost_matrix(cost_matrix)
    cuopt_env.add_capacity_dimension("weight", demands, capacities)
    
    # Debug: verify capacity dimension was added
    print(f"Debug: Added capacity dimension - demands: {len(demands)}, capacities: {len(capacities)}")
    print(f"Debug: Capacity values: {capacities[:5]}..." if len(capacities) > 5 else f"Debug: Capacity values: {capacities}")
    
    return {
        'cuopt_env': cuopt_env,
        'node_coords': node_coords,
        'demands': demands,
        'capacities': capacities,
        'cost_matrix': cost_matrix,
        'num_locations': num_locations,
        'num_vehicles': num_vehicles,
        'num_orders': num_orders,
        'vehicle_capacity': vehicle_capacity
    }


def create_random_initial_solution(vrp_instance):
    """Create a random initial solution respecting capacity constraints"""
    demands = vrp_instance['demands']
    num_orders = vrp_instance['num_orders']
    num_vehicles = vrp_instance['num_vehicles']
    vehicle_capacity = vrp_instance['vehicle_capacity']
    
    order_nodes = list(range(1, num_orders))
    random.shuffle(order_nodes)
    
    routes = []
    remaining_orders = order_nodes.copy()
    
    for vehicle_id in range(num_vehicles):
        route = [0]
        current_load = 0
        
        for order in remaining_orders.copy():
            order_demand = demands[order]
            if current_load + order_demand <= vehicle_capacity:
                route.append(order)
                current_load += order_demand
                remaining_orders.remove(order)
        
        random.shuffle(route[1:-1] if len(route) > 2 else route[1:])
        routes.append(route if route[-1] == 0 else route + [0])
    
    # Assign remaining orders to existing routes or create new routes
    if remaining_orders:
        for order in remaining_orders:
            order_demand = demands[order]
            placed = False
            for route in routes:
                if len(route) > 1:  # Skip empty routes [0, 0]
                    route_load = sum(demands[node] for node in route if node != 0)
                    if route_load + order_demand <= vehicle_capacity:
                        route.insert(-1, order)
                        placed = True
                        break
            if not placed:
                routes.append([0, order, 0])
    
    return routes


def validate_solution_feasibility(routes, vrp_instance):
    """Validate solution feasibility: check capacity constraints and all orders served"""
    demands = vrp_instance['demands']
    vehicle_capacity = vrp_instance['vehicle_capacity']
    num_orders = vrp_instance['num_orders']
    
    # Check all orders are served
    served_orders = set()
    for route in routes:
        for node in route:
            if node != 0:  # Skip depot
                if node < 1 or node > num_orders - 1:
                    return False, f"Node {node} out of range [1, {num_orders-1}]"
                if node in served_orders:
                    return False, f"Order {node} served multiple times"
                served_orders.add(node)
    
    if len(served_orders) != num_orders - 1:
        missing = set(range(1, num_orders - 1)) - served_orders
        return False, f"{len(missing)} orders not served: {sorted(missing)}"
    
    # Check capacity constraints
    violations = []
    for route_idx, route in enumerate(routes):
        if len(route) <= 2 and route == [0, 0]:
            continue  # Empty route
        
        route_load = sum(demands[node] for node in route if node != 0)
        if route_load > vehicle_capacity:
            violations.append(f"Route {route_idx}: load {route_load} > capacity {vehicle_capacity}")
    
    if violations:
        return False, "; ".join(violations)
    
    return True, "Solution is feasible"

def create_vrp_instance_from_pkl(instance_path, instance_index=0, num_vehicles=20):
    """Create VRP instance from pkl file"""
    import cuopt_pybind

    instance_data = load_instance_from_pkl(instance_path, instance_index)

    node_coords = instance_data['coordinates']  # (n_locations, 2), including depot
    cost_matrix = instance_data['cost_matrix']  # (n_locations, n_locations)
    demands = [int(x) for x in instance_data['demand'].tolist()]  # len = n_locations, demands[0] = 0
    vehicle_capacity = int(instance_data['vehicle_capacity'])

    num_locations = instance_data['n_locations']
    num_orders = num_locations
    capacities = [vehicle_capacity] * num_vehicles

    cuopt_env = cuopt_pybind.VrpLS(num_locations, num_vehicles, num_orders)
    cuopt_env.add_cost_matrix(cost_matrix.astype(np.float32))
    cuopt_env.add_capacity_dimension("weight", demands, capacities)
    
    # Debug: verify capacity dimension was added
    # print(f"Debug: Added capacity dimension - demands: {len(demands)}, capacities: {len(capacities)}")
    # print(f"Debug: Capacity values: {capacities[:5]}..." if len(capacities) > 5 else f"Debug: Capacity values: {capacities}")

    return {
        'cuopt_env': cuopt_env,
        'node_coords': node_coords,
        'demands': demands,
        'capacities': capacities,
        'cost_matrix': cost_matrix,
        'num_locations': num_locations,
        'num_vehicles': num_vehicles,
        'num_orders': num_orders,
        'vehicle_capacity': vehicle_capacity,
        'instance_path': instance_path,
        'instance_index': instance_index,
    }


def customize_nodes_to_search(cuopt_env):
    """
    Custom node selection logic
    Users can modify nodes_to_search before sampling
    """
    nodes = cuopt_env.get_move_candidates()
    # Example: sort according to x [[x, ..., ...], [x, ..., ...], ...]
    nodes.sort(key=lambda x: x[0])
    cuopt_env.set_move_candidates(nodes)
    print(f"Number of nodes to search: {len(nodes)}")


def run_local_search(cuopt_env, num_orders, node_coords, use_custom_sampling=False, vrp_instance=None):
    """Run local search replicating cuOpt's run_best_local_search logic"""
    cuopt_env.acquire_resource()
    cuopt_env.reset_move_candidates()
    cuopt_env.set_routes_to_search()
    cuopt_env.sync_streams()
    
    solutions = [cuopt_env.get_solution_routes().copy()]
    costs = [cuopt_env.get_cost()]
    cycle_finder_iterations = []
    
    max_outer_iterations = 100000
    total_iterations = 0
    
    for outer_iter in range(max_outer_iterations):
        print(f"\nOuter Iteration {outer_iter + 1}")
        cuopt_env.extract_nodes_to_search()
        
        fast_search_iter = 0     
        while True:
            fast_search_iter += 1
            total_iterations += 1

            if use_custom_sampling:
                customize_nodes_to_search(cuopt_env)
            
            if not cuopt_env.sample_nodes_to_search(full_set=use_custom_sampling):
                print(f"  Fast search: Node pool exhausted")
                break
            
            fast_operators = ['vrp', 'sliding', 'two_opt']
            random.shuffle(fast_operators)
            
            improvements = []
            move_found = False
            for op in fast_operators:
                move_found_here = False
                if op == 'vrp':
                    move_found_here = cuopt_env.perform_vrp_search()
                elif op == 'sliding':
                    move_found_here = cuopt_env.run_sliding_search()
                elif op == 'two_opt':
                    move_found_here = cuopt_env.run_two_opt_search()
                
                if move_found_here:
                    improvements.append(op.upper())
                move_found = move_found or move_found_here

                tmp_route = cuopt_env.get_solution_routes()
                if validate_solution_feasibility(tmp_route, vrp_instance)[0] == False:
                    print("Infeasible solution! Found at iteration {total_iterations}!")
                    print(tmp_route)
                    print(vrp_instance["demands"])
                    print(validate_solution_feasibility(tmp_route, vrp_instance))
            
            cuopt_env.restore_found_nodes()
            
            if move_found:
                current_cost = cuopt_env.get_cost()
                print(f"  Fast search {fast_search_iter}: ✓ Cost = {current_cost:.2f} ({'+'.join(improvements)})")
                solutions.append(cuopt_env.get_solution_routes().copy())
                costs.append(current_cost)
            else:
                print(f"  Fast search {fast_search_iter}: No improvement")
        
        print(f"  Running Cycle Finder...")
        cycle_found = cuopt_env.run_cycle_finder()
        
        if cycle_found:
            current_cost = cuopt_env.get_cost()
            print(f"  Cycle Finder: ✓ Cost = {current_cost:.2f}")
            solutions.append(cuopt_env.get_solution_routes().copy())
            costs.append(current_cost)
            cycle_finder_iterations.append(len(costs) - 1)
        else:
            print(f"  Cycle Finder: No improvement")
            break

    cuopt_env.set_routes_to_search()
    cuopt_env.release_resource()
    cuopt_env.sync_streams()
    
    return solutions, costs, total_iterations, cycle_finder_iterations


def create_local_search_visualization(solutions, costs, node_coords, cycle_finder_iterations=None):
    """Create visualization of the local search process"""
    
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('VRP Local Search Results', fontsize=14, fontweight='bold')
    
    # Cost evolution
    ax1[0].plot(range(len(costs)), costs, 'bo-', linewidth=2, markersize=6, label='Cost')
    ax1[0].set_xlabel('Iteration')
    ax1[0].set_ylabel('Cost')
    ax1[0].set_title('Cost Evolution')
    ax1[0].grid(True, alpha=0.3)
    
    # Mark Cycle Finder iterations
    if cycle_finder_iterations:
        for cf_iter in cycle_finder_iterations:
            ax1[0].plot(cf_iter, costs[cf_iter], 'r*', markersize=15, label='Cycle Finder' if cf_iter == cycle_finder_iterations[0] else '')
    
    # Mark best solution
    best_point = costs.index(min(costs))
    ax1[0].axvline(x=best_point, color='green', linestyle='--', alpha=0.7, label='Best solution')
    ax1[0].legend()
    
    # Route visualization
    x_pos = node_coords[:, 0]
    y_pos = node_coords[:, 1]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    def plot_routes(ax, iter_idx, title):
        if iter_idx >= len(solutions):
            return
        
        for route_idx, route in enumerate(solutions[iter_idx]):
            if len(route) >= 2:
                valid_nodes = [node for node in route if 0 <= node < len(x_pos)]
                if len(valid_nodes) >= 2:
                    route_x = [x_pos[node] for node in valid_nodes]
                    route_y = [y_pos[node] for node in valid_nodes]
                    color = colors[route_idx % len(colors)]
                    ax.plot(route_x, route_y, 'o-', color=color, linewidth=2, markersize=4)
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.set_title(f'{title}\nCost: {costs[iter_idx]:.1f}')
        ax.grid(True, alpha=0.3)
    
    plot_routes(ax1[1], 0, 'Initial Solution')
    plot_routes(ax2[0], best_point, 'Best Solution')
    plot_routes(ax2[1], len(costs)-1, 'Final Solution')
    
    plt.tight_layout()
    plt.savefig('local_search_process.png', dpi=300, bbox_inches='tight')
    print(f"  Visualization saved as 'local_search_process.png'")
    
    if cycle_finder_iterations:
        print(f"  Cycle Finder executed {len(cycle_finder_iterations)} times at iterations: {cycle_finder_iterations}")


def main(num_locations=20, num_vehicles=5, vehicle_capacity=100, use_custom_sampling=False,
         instance_path=None, instance_index=0, hgs_solution_path=None,
         trial_id=0, global_iter=0, local_iter=0, basin_base_dir="basin_datasets0"):
    if num_vehicles < 1:
        raise ValueError(f"num_vehicles must be >= 1 (got {num_vehicles})")
    print(f"🚀 cuOpt VRP Local Search Test")

    # load/generate instance
    if instance_path is not None:
        print(f"Using instance from: {instance_path} (index={instance_index})")
        vrp_instance = create_vrp_instance_from_pkl(instance_path, instance_index, num_vehicles)
        num_locations = vrp_instance['num_locations']
        vehicle_capacity = vrp_instance['vehicle_capacity']
        basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
        trajectory_path = basin_paths['trajectory_path']
        print(f"Instance ID: {basin_paths['instance_id']}")
    else:
        if num_locations < 2:
            raise ValueError(f"num_locations must be >= 2 (got {num_locations})")
        print("Using randomly generated instance")
        vrp_instance = create_vrp_instance(num_locations, num_vehicles, vehicle_capacity)
        trajectory_path = None
    print(f"Configuration: {num_locations} locations, {num_vehicles} vehicles, capacity {vehicle_capacity}")
    print(f"Custom sampling: {'Enabled' if use_custom_sampling else 'Disabled'}")
    print("=" * 80)
    cuopt_env = vrp_instance['cuopt_env']
    node_coords = vrp_instance['node_coords']
    
    # Load initial solution from trajectory.jsonl
    if trajectory_path and os.path.exists(trajectory_path):
        print(f"\nLoading initial solution from trajectory.jsonl (trial_id={trial_id}, global_iter={global_iter}, local_iter={local_iter})")
        solution_data = load_solution_from_trajectory(trajectory_path, trial_id, global_iter, local_iter, num_locations)
        initial_routes = solution_data['routes']
        print(f"  Loaded {len(initial_routes)} routes, cost={solution_data['cost']:.2f}")
    else:
        print(f"\nCreating random initial solution")
        initial_routes = create_random_initial_solution(vrp_instance)
    # Validate initial solution feasibility
    is_feasible, message = validate_solution_feasibility(initial_routes, vrp_instance)
    if not is_feasible:
        print(f"❌ Initial solution infeasible: {message}")
        for i, route in enumerate(initial_routes):
            print(f"   Route {i}: {route}")
        raise ValueError(f"Initial solution infeasible: {message}")
    else:
        print(f"✓ Initial solution validated: {message}")
    # Add initial solution to cuopt env
    cuopt_env.initialize_search(initial_routes)
    initial_cost = cuopt_env.get_cost()
    print(f"✓ Initial solution - Cost: {initial_cost:.2f}")
    
    # Set initial weight for each constraint as the original cuopt
    print(f"Debug: Initial weights: {cuopt_env.get_weights()}")
    weights = [10000., 10000., 100., 1000., 1000., 1000., 10000., 10000., 10000.]
    cuopt_env.set_weights(weights)
    cuopt_env.set_selection_weights(weights)
    print(f"Debug: Updated weights: {cuopt_env.get_weights()}")

    # Run local search
    solutions, costs, total_iterations, cycle_finder_iterations = run_local_search(
        cuopt_env, vrp_instance['num_orders'], node_coords, use_custom_sampling, vrp_instance)
    
    # Create visualization
    print(f"\n📊 Creating visualization...")
    create_local_search_visualization(solutions, costs, node_coords, cycle_finder_iterations)
    # Print summary
    final_cost = cuopt_env.get_cost()
    final_routes = cuopt_env.get_solution_routes()
    # Validate final solution feasibility
    is_feasible, message = validate_solution_feasibility(final_routes, vrp_instance)
    if not is_feasible:
        print(f"\n❌ Final solution infeasible: {message}")
        for i, route in enumerate(final_routes):
            print(f"   Route {i}: {route}")
    else:
        print(f"\n✓ Final solution validated: {message}")
    improvement = initial_cost - final_cost
    improvement_pct = (improvement / initial_cost * 100)
    print(f"\n✓ Search completed")
    print(f"  Fast search iterations: {total_iterations}")
    print(f"  Cycle Finder improvements: {len(cycle_finder_iterations)}")
    print(f"  Solutions recorded: {len(solutions)}")
    print(f"  Initial: {initial_cost:.2f} → Final: {final_cost:.2f}")
    print(f"  Improvement: {improvement:.2f} ({improvement_pct:.1f}%)")
    
    # Compare with HGS solution if provided
    if hgs_solution_path is not None:
        hgs_solution = load_hgs_solution_from_pkl(hgs_solution_path, instance_index)
        hgs_cost = hgs_solution["hgs_cost"]
        gap_final = calculate_gap(final_cost, hgs_cost)
        gap_initial = calculate_gap(initial_cost, hgs_cost)
        print("\n📐 Comparison with HGS solution")
        print(f"  HGS cost: {hgs_cost:.2f}")
        print(f"  Initial gap to HGS: {gap_initial:.2f}%")
        print(f"  Final   gap to HGS: {gap_final:.2f}%")
    
    return cuopt_env


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='cuOpt VRP Local Search Test')
    parser.add_argument('--lo', '--locations', type=int, default=101, dest='locations', 
                        help='Number of locations (depot + orders)')
    parser.add_argument('--vehicle', '--vehicles', type=int, default=30, dest='vehicles', 
                        help='Number of vehicles')
    parser.add_argument('--capacity', type=int, default=50, help='Vehicle capacity')
    parser.add_argument('--diy', action='store_true', 
                        help='Enable custom node sampling (use customize_nodes_to_search)')
    parser.add_argument('--pkl', '--instance_path', type=str, default="/home/jieyi/cvrp100_uniform.pkl", dest='instance_path',
                        help='Path to CVRP instance .pkl file; if set, load existing instance instead of random')
    parser.add_argument('--idx', '--instance_index', type=int, default=0, dest='instance_index',
                        help='Index of instance within the .pkl file')
    parser.add_argument('--hgs', '--hgs_solution_path', type=str, default="/home/jieyi/hgs_cvrp100_uniform.pkl", dest='hgs_solution_path',
                        help='Path to HGS solution .pkl file for gap comparison')
    parser.add_argument('--trial_id', type=int, default=0, dest='trial_id',
                        help='Trial ID to load from trajectory.jsonl')
    parser.add_argument('--global_iter', default=0, type=int, dest='global_iter',
                        help='Global iteration to load from trajectory.jsonl')
    parser.add_argument('--local_iter', default=0, type=int, dest='local_iter',
                        help='Local iteration to load from trajectory.jsonl')
    parser.add_argument('--basin_dir', type=str, default="basin_datasets0", dest='basin_base_dir',
                        help='Base directory for basin datasets (default: basin_datasets0)')
    args = parser.parse_args()

    main(args.locations, args.vehicles, args.capacity, args.diy,
         args.instance_path, args.instance_index, args.hgs_solution_path,
         args.trial_id, args.global_iter, args.local_iter, args.basin_base_dir)
