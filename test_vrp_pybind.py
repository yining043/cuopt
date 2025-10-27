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

# Add the cuOpt build directory to Python path
cuopt_build_path = os.path.join(os.path.dirname(__file__), 'cpp', 'build', 'install', 'lib', 'python3', 'dist-packages')
if cuopt_build_path not in sys.path:
    sys.path.insert(0, cuopt_build_path)


def create_vrp_instance(num_locations=20, num_vehicles=5, vehicle_capacity=100):
    """Create VRP instance: location 0 = depot, locations 1..n-1 = orders"""
    import cuopt_pybind
    
    num_orders = num_locations - 1
    
    node_coords = np.random.rand(num_locations, 2) * 10
    
    cost_matrix = np.zeros((num_locations, num_locations), dtype=np.float32)
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                distance = np.sqrt(np.sum((node_coords[i] - node_coords[j])**2))
                cost_matrix[i, j] = distance
    
    demands = [0] + list(np.random.randint(1, 10, num_orders))
    capacities = [vehicle_capacity] * num_vehicles
    
    cuopt_env = cuopt_pybind.VrpLS(num_locations, num_vehicles, num_orders)
    cuopt_env.add_cost_matrix(cost_matrix)
    cuopt_env.add_capacity_dimension("weight", demands, capacities)
    
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
    
    order_nodes = list(range(1, num_orders + 1))
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
    
    if remaining_orders:
        last_route = routes[-1]
        if last_route[-1] == 0:
            last_route.pop()
        last_route.extend(remaining_orders)
        last_route.append(0)
    
    return routes


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


def run_local_search(cuopt_env, num_orders, node_coords, use_custom_sampling=False):
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
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
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


def main(num_locations=20, num_vehicles=5, vehicle_capacity=100, use_custom_sampling=False):
    if num_locations < 2:
        raise ValueError(f"num_locations must be >= 2 (got {num_locations})")
    if num_vehicles < 1:
        raise ValueError(f"num_vehicles must be >= 1 (got {num_vehicles})")
    
    num_orders = num_locations - 1
    print(f"🚀 cuOpt VRP Local Search Test")
    print(f"Configuration: {num_locations} locations (1 depot + {num_orders} orders), {num_vehicles} vehicles, capacity {vehicle_capacity}")
    print(f"Custom sampling: {'Enabled' if use_custom_sampling else 'Disabled'}")
    print("="*80)
    
    vrp_instance = create_vrp_instance(num_locations, num_vehicles, vehicle_capacity)
    cuopt_env = vrp_instance['cuopt_env']
    node_coords = vrp_instance['node_coords']
    
    initial_routes = create_random_initial_solution(vrp_instance)
    cuopt_env.initialize_search(initial_routes)
    initial_cost = cuopt_env.get_cost()
    print(f"✓ Initial solution - Cost: {initial_cost:.2f}")
    
    cuopt_env.set_selection_weights(cuopt_env.get_weights())
    
    solutions, costs, total_iterations, cycle_finder_iterations = run_local_search(
        cuopt_env, vrp_instance['num_orders'], node_coords, use_custom_sampling)
    
    # Create visualization
    print(f"\n📊 Creating visualization...")
    create_local_search_visualization(solutions, costs, node_coords, cycle_finder_iterations)
    
    # Print summary
    final_cost = cuopt_env.get_cost()
    improvement = initial_cost - final_cost
    improvement_pct = (improvement / initial_cost * 100)
    
    print(f"\n✓ Search completed")
    print(f"  Fast search iterations: {total_iterations}")
    print(f"  Cycle Finder improvements: {len(cycle_finder_iterations)}")
    print(f"  Solutions recorded: {len(solutions)}")
    print(f"  Initial: {initial_cost:.2f} → Final: {final_cost:.2f}")
    print(f"  Improvement: {improvement:.2f} ({improvement_pct:.1f}%)")
    
    return cuopt_env


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='cuOpt VRP Local Search Test')
    parser.add_argument('--lo', '--locations', type=int, default=20, dest='locations', 
                        help='Number of locations (depot + orders)')
    parser.add_argument('--vehicle', '--vehicles', type=int, default=5, dest='vehicles', 
                        help='Number of vehicles')
    parser.add_argument('--capacity', type=int, default=100, help='Vehicle capacity')
    parser.add_argument('--diy', action='store_true', 
                        help='Enable custom node sampling (use customize_nodes_to_search)')
    
    args = parser.parse_args()
    
    try:
        main(args.locations, args.vehicles, args.capacity, args.diy)
        print("\n✅ Test completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)
