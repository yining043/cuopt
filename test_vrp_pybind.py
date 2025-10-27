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
    
    # Perform local search
    print(f"\n🔍 Starting Local Search...")
    cuopt_env.acquire_resource()
    cuopt_env.reset_move_candidates()
    cuopt_env.set_routes_to_search()
    cuopt_env.extract_nodes_to_search()
    
    # Store initial state
    solutions.append(cuopt_env.get_solution_routes().copy())
    costs.append(cuopt_env.get_cost())
    
    # Perform search iterations
    iteration = 0
    while True:
        iteration += 1
        sampled = cuopt_env.sample_nodes_to_search()
        if not sampled:
            break
            
        cuopt_env.sync_streams()
        
        # Perform VRP, Sliding, and Two-Opt searches
        vrp_found = cuopt_env.perform_vrp_search()
        sliding_found = cuopt_env.run_sliding_search()
        two_opt_found = cuopt_env.run_two_opt_search()
        move_found = vrp_found or sliding_found or two_opt_found
        
        cuopt_env.restore_found_nodes()
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
            improvements = []
            if vrp_found:
                improvements.append("VRP")
            if sliding_found:
                improvements.append("Sliding")
            if two_opt_found:
                improvements.append("2-Opt")
            print(f"  Iteration {iteration}: ✓ Cost = {current_cost:.2f} ({'+'.join(improvements)})")
            if missing_nodes:
                print(f"      ⚠️  Missing nodes: {sorted(missing_nodes)}")
            solutions.append(current_routes.copy())
            costs.append(current_cost)
        else:
            print(f"  Iteration {iteration}: ✗ No improvement, Cost = {current_cost:.2f}")
            if missing_nodes:
                print(f"      ⚠️  Missing nodes: {sorted(missing_nodes)}")
            break

    cuopt_env.release_resource()
    cuopt_env.sync_streams()
    
    final_cost = current_cost
    
    # Create visualization
    print(f"\n📊 Creating visualization...")
    create_local_search_visualization(solutions, costs, node_coords)
    
    print(f"\n✓ Search completed")
    print(f"  Initial: {initial_cost:.2f} → Final: {final_cost:.2f}")
    print(f"  Improvement: {initial_cost - final_cost:.2f} ({((initial_cost - final_cost)/initial_cost*100):.1f}%)")
    
    return cuopt_env

def create_local_search_visualization(solutions, costs, node_coords):
    """Create simplified visualization of the local search process"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('VRP Local Search Results (VRP + Sliding + Two-Opt)', fontsize=14, fontweight='bold')
    
    # Cost evolution
    ax1[0].plot(range(len(costs)), costs, 'bo-', linewidth=2, markersize=6)
    ax1[0].set_xlabel('Iteration')
    ax1[0].set_ylabel('Cost')
    ax1[0].set_title('Cost Evolution')
    ax1[0].grid(True, alpha=0.3)
    
    # Mark best solution
    best_point = costs.index(min(costs))
    ax1[0].axvline(x=best_point, color='green', linestyle='--', alpha=0.7, label='Best solution')
    ax1[0].legend()
    
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
    plot_routes(ax2[0], best_point, 'Best Solution')
    plot_routes(ax2[1], len(costs)-1, 'Final Solution')
    
    plt.tight_layout()
    plt.savefig('local_search_process.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"  📊 Visualization saved as 'local_search_process.png'")
    print(f"  📈 Total improvement: {costs[0] - costs[-1]:.1f} ({((costs[0] - costs[-1])/costs[0]*100):.1f}%)")
    print(f"  🔍 Total iterations: {len(costs)}, Best at iteration: {best_point}")


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