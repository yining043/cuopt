"""
Minimal test for routing callback functionality
"""
import numpy as np
import cudf
from cuopt import routing
from cuopt.routing import ObservationCallback
import matplotlib.pyplot as plt
import random

class TestCallback(ObservationCallback):
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.solutions = []
    
    def get_observation_and_sample(self, routes_2d, node_ids_to_search, objective_value, n_routes):
        """
        Custom callback that logs solutions and returns indices for sampling
        
        NOTE: Must return INDICES into node_ids_to_search, not node IDs!
        """
        self.call_count += 1
        
        # Store observation for analysis
        self.solutions.append({
            'iteration': self.call_count,
            'cost': objective_value,
            'n_routes': n_routes,
            'routes': routes_2d,
            'nodes_to_search': list(node_ids_to_search)
        })
        
        # Determine sample size
        n_available = len(node_ids_to_search)
        if n_available < 40:
            sample_size = n_available
        elif n_available < 80:
            sample_size = n_available // 2
        else:
            sample_size = 40
        
        # Return random INDICES (not node IDs!)
        sampled_indices = random.sample(range(n_available), sample_size) if sample_size > 0 else []
        
        return sampled_indices


def generate_random_vrp(n_locations, n_vehicles, seed=42):
    """Generate a random VRP problem"""
    # np.random.seed(seed)
    
    coords = np.random.rand(n_locations, 2) * 100
    
    distances = np.zeros((n_locations, n_locations))
    for i in range(n_locations):
        for j in range(n_locations):
            distances[i, j] = np.linalg.norm(coords[i] - coords[j])
    
    distance_matrix = cudf.DataFrame(distances)
    
    demands_np = np.zeros(n_locations, dtype=np.int32)
    demands_np[1:] = np.random.randint(5, 15, n_locations - 1)
    demand_array = cudf.Series(demands_np)
    
    vehicle_capacity = cudf.Series([100] * n_vehicles)
    
    return {
        'cost_matrix': distance_matrix,
        'demand': demand_array,
        'vehicle_capacity': vehicle_capacity,
        'n_locations': n_locations,
        'n_vehicles': n_vehicles,
        'coordinates': coords  # Save coordinates for visualization
    }


def test_callback():
    print("=" * 60)
    print("Testing Routing Callback")
    print("=" * 60)
    
    n_locations = 100
    n_vehicles = 10
    
    problem_data = generate_random_vrp(n_locations, n_vehicles)
    
    data_model = routing.DataModel(n_locations, n_vehicles)
    data_model.add_cost_matrix(problem_data['cost_matrix'])
    data_model.add_capacity_dimension(
        "demand",
        problem_data['demand'],
        problem_data['vehicle_capacity']
    )
    
    callback = TestCallback()
    
    solver_settings = routing.SolverSettings()
    solver_settings.set_time_limit(10)
    solver_settings.set_routing_callback(callback)
    
    print("\nSolving...")
    routing_solution = routing.Solve(data_model, solver_settings)
    
    print("\n" + "=" * 60)
    print(f"Solution status: {routing_solution.get_status()}")
    print(f"Total callback invocations: {callback.call_count}")
    print("=" * 60)
    
    if callback.solutions:
        costs = [s['cost'] for s in callback.solutions]
        plt.figure(figsize=(10, 6))
        plt.plot(costs, marker='o', linestyle='-', linewidth=2, markersize=4)
        plt.xlabel('Callback Iteration', fontsize=12)
        plt.ylabel('Objective Value (Cost)', fontsize=12)
        plt.title('Cost Evolution During Optimization', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('callback_cost_curve.png', dpi=150)
        print(f"\nCost curve saved to: callback_cost_curve.png")
        print(f"Best cost: {min(costs):.2f}")
        print(f"Initial cost: {costs[0]:.2f}")
        print(f"Improvement: {costs[0] - min(costs):.2f} ({(1 - min(costs)/costs[0])*100:.1f}%)")
    
    # Visualize and compare solutions
    print("\n" + "=" * 60)
    print("Comparing Callback Best vs Final Solution")
    print("=" * 60)
    
    coords = problem_data['coordinates']
    
    # Get callback best solution
    if callback.solutions:
        best_callback_sol = min(callback.solutions, key=lambda x: x['cost'])
        print(f"Callback best cost: {best_callback_sol['cost']:.2f}")
        print(f"Callback best at iteration: {best_callback_sol['iteration']}")
    
    # Get final solution from get_route()
    route_df = routing_solution.get_route()
    route_np = route_df['route'].to_numpy().astype(int)
    final_cost = routing_solution.get_total_objective()
    print(f"Final solution cost: {final_cost:.2f}")
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, n_vehicles))
    
    # Plot 1: Callback Best Solution
    ax = axes[0]
    ax.scatter(coords[0, 0], coords[0, 1], c='red', s=300, marker='s', 
               edgecolors='black', linewidths=2, label='Depot', zorder=5)
    ax.scatter(coords[1:, 0], coords[1:, 1], c='lightblue', s=100, 
               edgecolors='black', linewidths=1, label='Customers', zorder=3)
    
    if callback.solutions:
        for route_idx, route in enumerate(best_callback_sol['routes']):
            if len(route) > 2:  # Skip empty routes
                route_coords = coords[route]
                ax.plot(route_coords[:, 0], route_coords[:, 1], 
                       color=colors[route_idx % len(colors)], 
                       linewidth=2, alpha=0.7, zorder=2)
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(f'Callback Best Solution\nCost: {best_callback_sol["cost"]:.2f} (Iteration {best_callback_sol["iteration"]})', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: Final Solution from get_route()
    ax = axes[1]
    ax.scatter(coords[0, 0], coords[0, 1], c='red', s=300, marker='s', 
               edgecolors='black', linewidths=2, label='Depot', zorder=5)
    ax.scatter(coords[1:, 0], coords[1:, 1], c='lightblue', s=100, 
               edgecolors='black', linewidths=1, label='Customers', zorder=3)
    
    current_route = []
    route_idx = 0
    for node in route_np:
        if node == 0:  # Depot
            if len(current_route) > 1:
                current_route.append(0)
                route_coords = coords[current_route]
                ax.plot(route_coords[:, 0], route_coords[:, 1], 
                       color=colors[route_idx % len(colors)], 
                       linewidth=2, alpha=0.7, zorder=2)
                route_idx += 1
                current_route = [0]
            elif len(current_route) == 0:
                current_route = [0]
        else:
            current_route.append(node)
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(f'Final Solution (get_route())\nCost: {final_cost:.2f}', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('solution_comparison.png', dpi=150)
    print(f"\nSolution comparison saved to: solution_comparison.png")
    
    # Print comparison
    if callback.solutions:
        cost_diff = abs(final_cost - best_callback_sol['cost'])
        print(f"\nCost difference: {cost_diff:.2f}")
        if cost_diff < 0.01:
            print("✓ Solutions match! (callback best == final)")
        else:
            print(f"✗ Solutions differ by {cost_diff:.2f}")


if __name__ == "__main__":
    test_callback()
