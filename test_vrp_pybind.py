#!/usr/bin/env python3
"""
Test script for cuOpt Local Search (VrpLS) interface - 20 points comprehensive test
"""

import numpy as np
import sys
import os

# Add the cuOpt build directory to Python path
cuopt_build_path = os.path.join(os.path.dirname(__file__), 'cpp', 'build', 'install', 'lib', 'python3', 'dist-packages')
if cuopt_build_path not in sys.path:
    sys.path.insert(0, cuopt_build_path)

def create_vrp_instance(num_locations=20, num_vehicles=5, num_orders=15, vehicle_capacity=100):
    """Create a complete VRP instance with all data"""
    import cuopt_pybind
    
    # Generate node coordinates first
    node_coords = np.random.rand(num_locations, 2) * 10  # Random coordinates in [0,10] x [0,10]
    
    # Calculate cost matrix based on Euclidean distances
    cost_matrix = np.zeros((num_locations, num_locations), dtype=np.float32)
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                # Euclidean distance between nodes
                distance = np.sqrt(np.sum((node_coords[i] - node_coords[j])**2))
                cost_matrix[i, j] = distance
            else:
                cost_matrix[i, j] = 0
    
    # Generate demands for all nodes
    demands = [0] + list(np.random.randint(10, 31, num_orders)) + [0]  # depot has 0 demand
    capacities = [vehicle_capacity] * num_vehicles  # all vehicles have same capacity
    
    # Create VRP problem with configurable parameters
    cuopt_env = cuopt_pybind.VrpLS(num_locations, num_vehicles, num_orders)
    cuopt_env.add_cost_matrix(cost_matrix)
    cuopt_env.add_capacity_dimension("weight", demands, capacities)
    
    # Return the complete VRP instance
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
    """Create a random initial solution for VRP with capacity constraints"""
    import random
    
    # Extract data from VRP instance
    demands = vrp_instance['demands']
    num_orders = vrp_instance['num_orders']
    num_vehicles = vrp_instance['num_vehicles']
    vehicle_capacity = vrp_instance['vehicle_capacity']
    
    # Create list of all order nodes (1 to num_orders) with their demands
    order_nodes = list(range(1, num_orders + 1))
    random.shuffle(order_nodes)
    
    # Distribute orders across vehicles while respecting capacity constraints
    routes = []
    remaining_orders = order_nodes.copy()
    
    for vehicle_id in range(num_vehicles):
        route = [0]  # Start with depot
        current_load = 0
        
        # Try to add orders to this vehicle while respecting capacity
        orders_for_vehicle = []
        temp_remaining = remaining_orders.copy()
        
        for order in temp_remaining:
            order_demand = demands[order]
            if current_load + order_demand <= vehicle_capacity:
                orders_for_vehicle.append(order)
                current_load += order_demand
                remaining_orders.remove(order)
        
        # Shuffle orders within this route
        random.shuffle(orders_for_vehicle)
        route.extend(orders_for_vehicle)
        route.append(0)  # End with depot
        routes.append(route)
    
    # Handle any remaining orders that couldn't fit in previous vehicles
    # Add them to the last vehicle (even if it exceeds capacity - this will be handled by the solver)
    if remaining_orders:
        print(f"Adding remaining orders to last vehicle: {remaining_orders}")
        last_route = routes[-1]
        # Remove the final depot
        last_route.pop()
        # Add remaining orders
        random.shuffle(remaining_orders)
        last_route.extend(remaining_orders)
        # Add depot back
        last_route.append(0)
    
    return routes


def main(num_locations=20, num_vehicles=5, num_orders=15, vehicle_capacity=100):
    print(f"🚀 Starting cuOpt VRP Local Search Test")
    print(f"Configuration: {num_locations} locations, {num_vehicles} vehicles, {num_orders} orders, capacity {vehicle_capacity}")
    print("="*80)
    
    vrp_instance = create_vrp_instance(num_locations, num_vehicles, num_orders, vehicle_capacity)
    print("✓ VRP instance created successfully")

    """Test basic VRP functionality"""
    print("="*60)
    print("Testing Basic VRP Functionality")
    print(f"Problem: {vrp_instance['num_locations']} locations, {vrp_instance['num_vehicles']} vehicles, {vrp_instance['num_orders']} orders")
    print("="*60)
    
    cuopt_env = vrp_instance['cuopt_env']
    node_coords = vrp_instance['node_coords']
    num_orders = vrp_instance['num_orders']
    initial_routes = create_random_initial_solution(vrp_instance)
    cuopt_env.initialize_search(initial_routes)
    initial_cost = cuopt_env.get_cost()
    print(f"✓ Initial solution - Cost: {initial_cost:.2f}")

    # Handle weights
    current_weights = cuopt_env.get_weights()
    cuopt_env.set_selection_weights(current_weights)
    
    solutions = []
    costs = []
    
    def perform_search_round(round_name):
        """Perform a round of local search"""
        print(f"\n🔍 {round_name}...")
        cuopt_env.acquire_resource()
        cuopt_env.reset_move_candidates()
        cuopt_env.set_routes_to_search()
        cuopt_env.extract_nodes_to_search()
        
        # Store initial state
        solutions.append(cuopt_env.get_solution_routes().copy())
        costs.append(cuopt_env.get_cost())
        
        # Perform search iterations
        move_found = True
        i = 0
        while move_found:
            i += 1
            print(f"      Before sample_nodes_to_search: {len(cuopt_env.get_move_candidates())} move candidates")
            sampled_nodes = cuopt_env.sample_nodes_to_search()
            cuopt_env.sync_streams()
            print(f"      After sample_nodes_to_search: {len(cuopt_env.get_move_candidates())} move candidates")
            
            # Test move candidates get/set functionality
            print(f"    Testing move candidates get/set for iteration {i+1}:")
            
            # Test set/get functionality without detailed printing
            # if len(move_candidates) > 0:
            #     # Test setting modified candidates (filter out some if we have many)
            #     if len(move_candidates) > 5:
            #         # Keep only first 5 candidates as a test
            #         modified_candidates = move_candidates[:5]
            #         cuopt_env.set_move_candidates(modified_candidates)
            #         cuopt_env.sync_streams()
            #         print(f"      ✓ Set {len(modified_candidates)} modified candidates")
            #     else:
            #         # If we have few candidates, just set them back as-is
            #         cuopt_env.set_move_candidates(move_candidates)
            #         print(f"      ✓ Set back all {len(move_candidates)} candidates unchanged")
            # else:
            #     print(f"      No candidates available for testing")
            
            # Compare with sampled nodes (the actual data used by perform_vrp_search)
            sampled_nodes = cuopt_env.get_sampled_nodes()
            print(f"      Sampled nodes: {len(sampled_nodes)} candidates")
            
            move_found = cuopt_env.perform_vrp_search()
            cuopt_env.restore_found_nodes()
            print(f"      After restore_found_nodes: Restored {len(cuopt_env.get_move_candidates())} move candidates")
            cuopt_env.sync_streams()
            
            current_cost = cuopt_env.get_cost()
            current_routes = cuopt_env.get_solution_routes()
            
            # Check node coverage
            current_nodes = set()
            for route in current_routes:
                current_nodes.update(route)
            current_nodes.discard(0)
            missing_nodes = set(range(1, num_orders + 1)) - current_nodes
            
            if move_found:
                print(f"  Iteration {i+1}: ✓ Cost = {current_cost:.2f}")
                if missing_nodes:
                    print(f"    ⚠️  Missing nodes: {sorted(missing_nodes)}")
            else:
                print(f"  Iteration {i+1}: ✗ No improvement, Cost = {current_cost:.2f}")
                if missing_nodes:
                    print(f"    ⚠️  Missing nodes: {sorted(missing_nodes)}")
            
            if move_found:
                solutions.append(current_routes.copy())
                costs.append(current_cost)
            else:
                break

        cuopt_env.release_resource()
        cuopt_env.sync_streams()
    
        return current_cost, current_routes
    
    # Two rounds of search
    final_cost_1, final_routes_1 = perform_search_round("First Round")
    first_round_end = len(solutions)
    
    print(f"\n🔄 Resetting solution...")
    reset_routes = final_routes_1 #create_random_initial_solution(vrp_instance)
    cuopt_env.setup_solution(reset_routes)
    reset_cost = cuopt_env.get_cost()
    
    final_cost_2, _ = perform_search_round("Second Round")
    
    # Create visualization
    print(f"\n📊 Creating visualization...")
    create_local_search_visualization(solutions, costs, node_coords, reset_point=first_round_end)
    
    print(f"\n✓ Search completed")
    print(f"  Initial: {initial_cost:.2f} → First: {final_cost_1:.2f} → Reset: {reset_cost:.2f} → Final: {final_cost_2:.2f}")
    
    return cuopt_env

def create_local_search_visualization(solutions, costs, node_coords, reset_point=None):
    """Create simplified visualization of the local search process"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('VRP Local Search Results', fontsize=14, fontweight='bold')
    
    # Cost evolution
    ax1[0].plot(range(len(costs)), costs, 'bo-', linewidth=2, markersize=6)
    ax1[0].set_xlabel('Iteration')
    ax1[0].set_ylabel('Cost')
    ax1[0].set_title('Cost Evolution')
    ax1[0].grid(True, alpha=0.3)
    
    # Find reset point more accurately
    if reset_point is None:
        # Look for significant cost increases as potential reset points
        for i in range(1, len(costs)):
            # Look for cost increases > 10% as potential reset points
            if costs[i] > costs[i-1] * 1.1:
                reset_point = i
                break
        
        # If no clear reset point found, use the middle point
        if reset_point is None:
            reset_point = len(costs) // 2
    
    best_point = costs.index(min(costs))
    
    # Add vertical lines and labels
    ax1[0].axvline(x=reset_point, color='red', linestyle='--', alpha=0.7, label='Second Round Start')
    ax1[0].axvline(x=best_point, color='green', linestyle='--', alpha=0.7, label='Best solution')
    ax1[0].legend()
    
    # Add text annotations
    ax1[0].text(reset_point, max(costs), 'Round 2', rotation=90, va='top', ha='right', color='red', fontweight='bold')
    ax1[0].text(0, max(costs), 'Round 1', va='top', ha='left', color='blue', fontweight='bold')
    
    # Node coordinates
    x_pos = node_coords[:, 0]
    y_pos = node_coords[:, 1]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    def plot_routes(ax, iter_idx, title):
        """Plot routes for an iteration"""
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
    
    # Plot key iterations
    plot_routes(ax1[1], 0, 'Initial Solution')
    if reset_point > 1:
        plot_routes(ax2[0], reset_point-1, 'First Round End')
    else:
        plot_routes(ax2[0], 0, 'First Round End')
    plot_routes(ax2[1], len(costs)-1, 'Final Solution')
    
    plt.tight_layout()
    plt.savefig('local_search_process.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"  📊 Visualization saved as 'local_search_process.png'")
    print(f"  📈 Total improvement: {costs[0] - costs[-1]:.1f} ({((costs[0] - costs[-1])/costs[0]*100):.1f}%)")
    
    # Print round statistics
    if reset_point < len(costs):
        first_round_costs = costs[:reset_point]
        second_round_costs = costs[reset_point:]
        
        if first_round_costs:
            first_improvement = first_round_costs[0] - first_round_costs[-1]
            print(f"  🔵 Round 1: {len(first_round_costs)} iterations, improvement: {first_improvement:.1f}")
        
        if second_round_costs:
            second_improvement = second_round_costs[0] - second_round_costs[-1]
            print(f"  🔴 Round 2: {len(second_round_costs)} iterations, improvement: {second_improvement:.1f}")


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='cuOpt VRP Local Search Test')
    parser.add_argument('--locations', type=int, default=20, help='Number of locations (default: 20)')
    parser.add_argument('--vehicles', type=int, default=5, help='Number of vehicles (default: 5)')
    parser.add_argument('--orders', type=int, default=15, help='Number of orders (default: 15)')
    parser.add_argument('--capacity', type=int, default=100, help='Vehicle capacity (default: 100)')
    
    args = parser.parse_args()
    
    try:
        cuopt_env = main(args.locations, args.vehicles, args.orders, args.capacity)
        print("\n✅ Test completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)