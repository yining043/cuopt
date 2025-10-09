#!/usr/bin/env python3
"""
cuOpt VRP Core Demo
==================
Minimal demo showcasing cuOpt VRP solver core functionality.
"""

import numpy as np
import sys
import os

# Setup cuOpt path
cuopt_build_path = os.path.join(os.path.dirname(__file__), 'cpp', 'build', 'install', 'lib', 'python3', 'dist-packages')
if cuopt_build_path not in sys.path:
    sys.path.insert(0, cuopt_build_path)


def create_vrp_problem(num_locations=20, num_vehicles=5, num_orders=15, vehicle_capacity=100):
    """Create VRP problem instance"""
    import cuopt_pybind
    
    # Generate random coordinates and cost matrix
    node_coords = np.random.rand(num_locations, 2) * 10
    cost_matrix = np.zeros((num_locations, num_locations), dtype=np.float32)
    
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                cost_matrix[i, j] = np.linalg.norm(node_coords[i] - node_coords[j])
    
    # Setup demands and capacities
    demands = [0] + list(np.random.randint(10, 31, num_orders))
    capacities = [vehicle_capacity] * num_vehicles
    
    # Initialize cuOpt environment
    cuopt_env = cuopt_pybind.VrpLS(num_locations, num_vehicles, num_orders)
    cuopt_env.add_cost_matrix(cost_matrix)
    cuopt_env.add_capacity_dimension("weight", demands, capacities)
    
    return {
        'cuopt_env': cuopt_env,
        'num_vehicles': num_vehicles,
        'num_orders': num_orders
    }


def generate_initial_solution(vrp_instance):
    """Generate random initial solution"""
    import random
    
    num_orders = vrp_instance['num_orders']
    num_vehicles = vrp_instance['num_vehicles']
    
    # Randomly distribute orders across vehicles
    orders = list(range(1, num_orders + 1))
    random.shuffle(orders)
    
    routes = []
    for v in range(num_vehicles):
        route = [0]  # Start at depot
        route.extend(orders[v::num_vehicles])  # Distribute orders
        route.append(0)  # Return to depot
        routes.append(route)
    
    return routes


def perform_local_search(cuopt_env):
    """Perform local search optimization"""
    
    cuopt_env.acquire_resource()
    cuopt_env.reset_move_candidates()
    cuopt_env.set_routes_to_search()
    cuopt_env.extract_nodes_to_search()
    
    move_found = True
    while move_found:
        cuopt_env.sample_nodes_to_search()
        cuopt_env.sync_streams()
        
        move_found = cuopt_env.perform_vrp_search()
        cuopt_env.restore_found_nodes()
        cuopt_env.sync_streams()
        
        if not move_found:
            break
    
    cuopt_env.sync_streams()
    cuopt_env.release_resource()
    
    return cuopt_env.get_cost(), cuopt_env.get_solution_routes()




def main():
    """Main function: demonstrate cuOpt VRP solving process"""
    
    # Create and solve VRP problem
    vrp_instance = create_vrp_problem()
    initial_routes = generate_initial_solution(vrp_instance)
    
    cuopt_env = vrp_instance['cuopt_env']
    cuopt_env.initialize_search(initial_routes)
    initial_cost = cuopt_env.get_cost()
    
    cuopt_env.set_selection_weights(cuopt_env.get_weights())
    final_cost, final_routes = perform_local_search(cuopt_env)
    
    # Output results
    print("cuOpt VRP Demo Results:")
    print(f"  Initial Cost: {initial_cost:.2f}")
    print(f"  Final Cost:   {final_cost:.2f}")
    print(f"  Improvement:  {initial_cost - final_cost:.2f} ({((initial_cost - final_cost)/initial_cost*100):.1f}%)")
    print(f"  Vehicles:     {vrp_instance['num_vehicles']}")
    print(f"  Orders:       {vrp_instance['num_orders']}")
    
    return cuopt_env


if __name__ == "__main__":
    try:
        cuopt_env = main()
        print("Demo completed successfully!")
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
