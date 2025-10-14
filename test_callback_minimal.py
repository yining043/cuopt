"""
Test for routing callback functionality (Customize-Nodes and Reward)

Demonstrates the two callback types:
- CustomizeNodesCallback: Customizes node sampling for local search
- RewardCallback: Receives iteration feedback
"""
import numpy as np
import cudf
from cuopt import routing
from cuopt.routing import CustomizeNodesCallback, RewardCallback
import matplotlib.pyplot as plt
import random


class TestCustomizeNodesCallback(CustomizeNodesCallback):
    """Example callback - customizes node sampling with observation tracking"""
    
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.observations = []
    
    def customize_nodes_to_search(self, routes_2d, candidate_node_ids, solution_cost, num_routes):
        """Customize sampling and record state"""
        self.call_count += 1
        
        # Record observation
        num_candidates = len(candidate_node_ids)
        self.observations.append({
            'iteration': self.call_count,
            'cost': solution_cost,
            'num_routes': num_routes,
            'routes': routes_2d,
            'num_candidates': num_candidates
        })
        
        # Adaptive sampling strategy
        if num_candidates < 40:
            sample_size = num_candidates
        elif num_candidates < 80:
            sample_size = num_candidates // 2
        else:
            sample_size = 40
        
        sampled_indices = random.sample(range(num_candidates), sample_size) if sample_size > 0 else []
        print(f"  [Customize {self.call_count}] Sampled {sample_size}/{num_candidates} nodes (cost={solution_cost:.2f})")
        
        return sampled_indices


class TestRewardCallback(RewardCallback):
    """Example reward callback - tracks improvement signals"""
    
    def __init__(self):
        super().__init__()
        self.reward_count = 0
        self.improvements = []
        self.cost_history = []
    
    def receive_reward(self, improvement_found, solution_cost):
        """Process reward signal"""
        self.reward_count += 1
        self.cost_history.append(solution_cost)
        
        if improvement_found:
            self.improvements.append({
                'iteration': self.reward_count,
                'cost': solution_cost
            })
            print(f"  [Reward {self.reward_count}] ✓ Improvement! cost={solution_cost:.2f}")
        else:
            print(f"  [Reward {self.reward_count}] ✗ No improvement, cost={solution_cost:.2f}")


def generate_random_vrp(n_locations, n_vehicles, seed=42):
    """Generate a random VRP problem"""
    # np.random.seed(seed)
    coords = np.random.rand(n_locations, 2) * 100
    distances = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)
    
    return {
        'cost_matrix': cudf.DataFrame(distances),
        'demand': cudf.Series(np.concatenate([[0], np.random.randint(5, 15, n_locations - 1)])),
        'vehicle_capacity': cudf.Series([100] * n_vehicles),
        'coordinates': coords
    }


def plot_cost_curve(solutions, filename='callback_cost_curve.png'):
    """Plot cost evolution during optimization"""
    costs = [s['cost'] for s in solutions]
    plt.figure(figsize=(10, 6))
    plt.plot(costs, marker='o', linestyle='-', linewidth=2, markersize=4)
    plt.xlabel('Callback Iteration', fontsize=12)
    plt.ylabel('Objective Value (Cost)', fontsize=12)
    plt.title('Cost Evolution During Optimization', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    
    print(f"\nCost Statistics:")
    print(f"  Initial: {costs[0]:.2f}")
    print(f"  Best: {min(costs):.2f}")
    print(f"  Improvement: {costs[0] - min(costs):.2f} ({(1 - min(costs)/costs[0])*100:.1f}%)")
    print(f"  Saved to: {filename}")


def plot_route(ax, coords, route, color, depot_idx=0):
    """Plot a single route"""
    if len(route) > 2:
        route_coords = coords[route]
        ax.plot(route_coords[:, 0], route_coords[:, 1], 
                color=color, linewidth=2, alpha=0.7, zorder=2)


def plot_solution(routes_data, coords, title, ax, colors):
    """Plot routing solution"""
    ax.scatter(coords[0, 0], coords[0, 1], c='red', s=300, marker='s', 
               edgecolors='black', linewidths=2, label='Depot', zorder=5)
    ax.scatter(coords[1:, 0], coords[1:, 1], c='lightblue', s=100, 
               edgecolors='black', linewidths=1, label='Customers', zorder=3)
    
    for route_idx, route in enumerate(routes_data):
        plot_route(ax, coords, route, colors[route_idx % len(colors)])
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')


def parse_solution_to_routes(route_df):
    """Convert route dataframe to list of routes"""
    route_np = route_df['route'].to_numpy().astype(int)
    routes = []
    current_route = []
    
    for node in route_np:
        if node == 0:
            if len(current_route) > 1:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0]
            elif len(current_route) == 0:
                current_route = [0]
        else:
            current_route.append(node)
    
    return routes


def compare_solutions(callback_sol, final_routes, final_cost, coords, filename='solution_comparison.png'):
    """Create comparison visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    plot_solution(callback_sol['routes'], coords, 
                  f'Callback Best\nCost: {callback_sol["cost"]:.2f} (Iter {callback_sol["iteration"]})',
                  axes[0], colors)
    
    plot_solution(final_routes, coords,
                  f'Final Solution\nCost: {final_cost:.2f}',
                  axes[1], colors)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"  Solution comparison saved to: {filename}")


def test_callback():
    print("=" * 60)
    print("Testing Routing Callback")
    print("=" * 60)
    
    # Setup problem
    n_locations, n_vehicles = 100, 10
    problem_data = generate_random_vrp(n_locations, n_vehicles)
    
    data_model = routing.DataModel(n_locations, n_vehicles)
    data_model.add_cost_matrix(problem_data['cost_matrix'])
    data_model.add_capacity_dimension("demand", problem_data['demand'], 
                                      problem_data['vehicle_capacity'])
    
    # Setup callbacks
    customize_callback = TestCustomizeNodesCallback()
    reward_callback = TestRewardCallback()
    
    solver_settings = routing.SolverSettings()
    solver_settings.set_time_limit(10)
    solver_settings.set_routing_callback(customize_callback)
    solver_settings.set_routing_callback(reward_callback)
    
    # Solve
    print("\nSolving...")
    solution = routing.Solve(data_model, solver_settings)
    
    # Results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Solution status: {solution.get_status()}")
    print(f"Customize nodes calls: {customize_callback.call_count}")
    print(f"Reward calls: {reward_callback.reward_count}")
    print(f"Total improvements: {len(reward_callback.improvements)}")
    print("=" * 60)
    
    # Detailed analysis
    if customize_callback.observations:
        plot_cost_curve(customize_callback.observations)
        
        best_obs = min(customize_callback.observations, key=lambda x: x['cost'])
        final_routes = parse_solution_to_routes(solution.get_route())
        final_cost = solution.get_total_objective()
        
        print(f"\nCustomize Nodes Callback:")
        print(f"  Best observed cost: {best_obs['cost']:.2f} (iteration {best_obs['iteration']})")
        
        print(f"\nReward Callback:")
        print(f"  Total reward signals: {reward_callback.reward_count}")
        print(f"  Positive rewards (improvements): {len(reward_callback.improvements)}")
        if reward_callback.improvements:
            best_improvement = min(imp['cost'] for imp in reward_callback.improvements)
            print(f"  Best improvement cost: {best_improvement:.2f}")
        
        print(f"\nFinal Solution:")
        print(f"  Cost: {final_cost:.2f}")
        print(f"  Gap from best observed: {abs(final_cost - best_obs['cost']):.2f}")
        
        compare_solutions(best_obs, final_routes, final_cost, problem_data['coordinates'])


if __name__ == "__main__":
    test_callback()
