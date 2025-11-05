"""
Test for routing callback functionality

Demonstrates the callback types:
- CustomizeNodesCallback: Customizes node sampling for local search
- RewardCallback: Receives iteration feedback
- LocalSearchStartCallback: Observes state before local search begins
- BeforeCycleFinderCallback: Observes state before cycle finder execution
- AfterCycleFinderCallback: Observes state after cycle finder execution
"""
import numpy as np
from cuopt import routing
from cuopt.routing import (
    CustomizeNodesCallback, 
    RewardCallback,
    LocalSearchStartCallback,
    BeforeCycleFinderCallback,
    AfterCycleFinderCallback
)
from cuopt_collector import Problem
import matplotlib.pyplot as plt
import random


class TestCustomizeNodesCallback(CustomizeNodesCallback):
    """Stores before state and implements adaptive node sampling."""
    
    def __init__(self, global_history):
        super().__init__()
        self.global_history = global_history
    
    def customize_nodes_to_search(self, solution_flat, num_routes, solution_cost, candidate_mask, iter):
        """Store before state and implement adaptive node sampling."""
        local_iter = self.global_history['current_local_iter'] = self.global_history['current_local_iter'] + 1
        global_iter = self.global_history['current_global_iter'] = self.global_history['current_global_iter'] + 1
        local_search_id = self.global_history['current_local_search_id']
        self.global_history['pending_state'] = {
            'local_search_id': local_search_id,
            'global_iter': global_iter,
            'local_iter': local_iter,
            'sol_before': solution_flat.copy(),
            'num_routes_before': num_routes,
            'cost_before': solution_cost,
            'is_circle_found': False
        }
        
        num_candidates = sum(candidate_mask)
        sample_size = min(40, num_candidates) if num_candidates >= 80 else (num_candidates // 2 if num_candidates >= 40 else num_candidates)
        
        candidate_nodes = [i for i, m in enumerate(candidate_mask) if m == 1]
        selection_mask = np.zeros(len(candidate_mask), dtype=np.int32)
        selection_mask[random.sample(candidate_nodes, min(sample_size, len(candidate_nodes)))] = 1
        
        return selection_mask.tolist()


class TestRewardCallback(RewardCallback):
    """Completes fast search record with after state."""
    
    def __init__(self, global_history):
        super().__init__()
        self.global_history = global_history
    
    def receive_reward(self, improvement_found, solution_cost, iter, solution_flat, num_routes):
        """Complete fast search record with after state and move result."""

        pending = self.global_history.get('pending_state', None)
        if pending:
            record = {
                'local_search_id': pending['local_search_id'],
                'global_iter': pending['global_iter'],
                'local_iter': pending['local_iter'],
                'sol_before': pending['sol_before'],
                'sol_after': solution_flat.copy(),
                'num_routes_before': pending['num_routes_before'],
                'num_routes_after': num_routes,
                'cost_before': pending['cost_before'],
                'cost_after': solution_cost,
                'move_found': improvement_found,
                'is_circle_found': False
            }
            self.global_history['pending_state'] = None
        else:
            assert False, "No pending record found"
        self.global_history['history'].append(record)


class TestLocalSearchStartCallback(LocalSearchStartCallback):
    """Manages global iteration offset and local search ID."""
    
    def __init__(self, global_history, customize_callback=None, reward_callback=None):
        super().__init__()
        self.global_history = global_history
        self.customize_callback = customize_callback
        self.reward_callback = reward_callback
    
    def on_local_search_start(self, solution_flat, num_routes, solution_cost, weights, selection_weights, should_all_nodes_be_served):
        """Update global iter and increment local search ID."""
        self.global_history['current_local_search_id'] = self.global_history.get('current_local_search_id', -1) + 1
        self.global_history['current_local_iter'] = -1
        print(f"Global iter: {self.global_history['current_global_iter']}, Local search count: {self.global_history['current_local_search_id']}")
        # print(f"Weights: {weights}")
        # print(f"Selection weights: {selection_weights}")
        # print(f"Should all nodes be served: {should_all_nodes_be_served}")
        assert should_all_nodes_be_served, "All nodes should be served"


class TestBeforeCycleFinderCallback(BeforeCycleFinderCallback):
    """Stores before state for cycle finder."""
    
    def __init__(self, global_history):
        super().__init__()
        self.global_history = global_history
    
    def on_before_cycle_finder(self, solution_flat, num_routes, solution_cost, iter):
        """Store before state for cycle finder."""
        local_iter = self.global_history['current_local_iter'] = self.global_history['current_local_iter'] + 1
        global_iter = self.global_history['current_global_iter'] = self.global_history['current_global_iter'] + 1
        local_search_id = self.global_history['current_local_search_id']
        self.global_history['pending_state'] = {
            'local_search_id': local_search_id,
            'global_iter': global_iter,
            'local_iter': local_iter,
            'sol_before': solution_flat.copy(),
            'num_routes_before': num_routes,
            'cost_before': solution_cost,
            'is_circle_found': True
        }


class TestAfterCycleFinderCallback(AfterCycleFinderCallback):
    """Completes cycle finder record with after state."""
    
    def __init__(self, global_history):
        super().__init__()
        self.global_history = global_history
    
    def on_after_cycle_finder(self, solution_flat, num_routes, solution_cost, iter, improved):
        """Complete cycle finder record with after state and improvement result."""

        pending = self.global_history.get('pending_state', None)
        if pending:
            record = {
                'local_search_id': pending['local_search_id'],
                'global_iter': pending['global_iter'],
                'local_iter': pending['local_iter'],
                'sol_before': pending['sol_before'],
                'sol_after': solution_flat.copy(),
                'num_routes_before': pending['num_routes_before'],
                'num_routes_after': num_routes,
                'cost_before': pending['cost_before'],
                'cost_after': solution_cost,
                'move_found': improved,
                'is_circle_found': True
            }
            self.global_history['pending_state'] = None
        else:
            assert False, "No pending record found"
        self.global_history['history'].append(record)

def plot_cost_curve(global_history, filename='callback_cost_curve.png'):
    """Plot cost evolution using records from global_history."""
    history = global_history['history']
    if not history:
        print("No records to plot")
        return
    
    # Separate by is_circle_found
    all_search = [(r['global_iter'], r['cost_after']) for r in history if r['cost_after'] is not None]
    cycle_finder = [(r['global_iter'], r['cost_after']) for r in history if r['is_circle_found'] and r['cost_after'] is not None]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot fast search trajectory
    if all_search:
        iters, costs = zip(*all_search)
        ax.plot(iters, costs, marker='o', linestyle='-', linewidth=1.5, markersize=4,
               color='#2E86AB', label='Fast Search', alpha=0.6, zorder=2)
    
    # Plot cycle finder points
    if cycle_finder:
        iters, costs = zip(*cycle_finder)
        ax.scatter(iters, costs, marker='o', s=20, color='#F18F01',
                  label='Cycle Finder', alpha=0.9, zorder=4, edgecolors='none')
    
    ax.set_xlabel('Global Iteration', fontsize=13, fontweight='bold')
    ax.set_ylabel('Objective Value (Cost)', fontsize=13, fontweight='bold')
    ax.set_title('Cost Evolution During Optimization', fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11, framealpha=0.9, shadow=True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    
    # Statistics
    all_costs = [r['cost_after'] for r in history if r['cost_after'] is not None]
    print(f"\nCost Statistics:")
    print(f"  Initial: {all_costs[0]:.2f}")
    print(f"  Best: {min(all_costs):.2f}")
    print(f"  Improvement: {all_costs[0] - min(all_costs):.2f} ({(1 - min(all_costs)/all_costs[0])*100:.1f}%)")
    print(f"  Total iterations: {len(history)}")
    print(f"  All search: {len(all_search)}")
    print(f"  Cycle finder: {len(cycle_finder)}")
    print(f"  Saved to: {filename}")


def plot_route(ax, coords, route, color, depot_idx=0):
    """Plot a single route"""
    if len(route) > 2:
        route_coords = coords[route]
        ax.plot(route_coords[:, 0], route_coords[:, 1], 
                color=color, linewidth=2, alpha=0.7, zorder=2)


def solution_flat_to_routes_2d(solution_flat, num_routes, num_orders):
    """Convert solution_flat to 2D routes for visualization"""
    routes_2d = []
    depot_node_id = 0  # Assuming depot_included = true
    
    idx = 0
    for route_id in range(num_routes):
        route = [depot_node_id]
        
        # Skip 4 dummy depot nodes
        idx += 4
        
        # Collect real nodes until next route's dummy or end
        while idx < len(solution_flat):
            node_id = solution_flat[idx]
            # Stop if we hit next dummy depot
            if node_id >= num_orders:
                break
            route.append(node_id)
            idx += 1
        
        # Add return depot
        if len(route) > 1:
            route.append(depot_node_id)
            routes_2d.append(route)
    
    return routes_2d


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
    """
    Test routing callback functionality with global history tracking.
    
    Creates a shared global history dictionary that tracks:
    - Local search session IDs
    - Global and local iteration counts
    - Event types (customize, reward, cycle_finder, etc.)
    - Solution states at each iteration
    - Whether cycle finder was used
    
    Returns:
        list: Complete global history of all search events
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
    
    # Setup problem using Problem class
    problem_gen = Problem(
        n_locations=100,
        n_vehicles=50,
        seed=42,
        coordinate_range=100.0,
        capacity=100.0,
        demand_range=(1, 10)
    )
    problem_data = problem_gen.generate()
    data_model = problem_gen.create_data_model(problem_data)
    
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
    solver_settings.set_time_limit(1)
    solver_settings.set_routing_callback(customize_callback)
    solver_settings.set_routing_callback(reward_callback)
    solver_settings.set_routing_callback(start_callback)
    solver_settings.set_routing_callback(before_callback)
    solver_settings.set_routing_callback(after_callback)
    
    # Note: callback uses policy.eval() mode by default, with sampling (not greedy)
    
    # Solve
    print("\nSolving...")
    solution = routing.Solve(data_model, solver_settings)
    
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
    print("SAMPLE HISTORY RECORDS (first 10)")
    print("=" * 60)
    for i, record in enumerate(history[:10]):
        print(f"\nRecord {i}:")
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
        plot_cost_curve(global_history)
        
        best_record = min([r for r in history if r['cost_after'] is not None], key=lambda x: x['cost_after'])
        
        final_routes = parse_solution_to_routes(solution.get_route())
        final_cost = solution.get_total_objective()
        
        print(f"\nBest Record from History:")
        print(f"  Cost: {best_record['cost_after']:.2f} (global iter {best_record['global_iter']}, local iter {best_record['local_iter']})")
        print(f"  Move Found: {best_record['move_found']}")
        print(f"  Cycle Finder: {best_record['is_circle_found']}")
        
        print(f"\nFinal Solution:")
        print(f"  Cost: {final_cost:.2f}")
        print(f"  Gap from best: {abs(final_cost - best_record['cost_after']):.2f}")
        
        # Visualize best solution from history vs final solution
        if best_record['sol_after'] and best_record['num_routes_after']:
            best_routes_2d = solution_flat_to_routes_2d(
                best_record['sol_after'],
                best_record['num_routes_after'],
                problem_gen.n_locations
            )
            compare_solutions(
                {'routes': best_routes_2d, 'cost': best_record['cost_after'], 'iteration': best_record['global_iter']},
                final_routes,
                final_cost,
                problem_data['coordinates']
            )
    # Return complete global history for external analysis
    return global_history['history']


if __name__ == "__main__":
    test_callback()
