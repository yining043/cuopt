#!/usr/bin/env python3
"""
Apply perturbation operators (Double-Bridge Move or Remove-and-Insert) to local optima 
and compute normalized Jaccard distance.

Features:
- Support for Double-Bridge Move and Remove-and-Insert operators
- Configurable number of operator applications (k)
- Optional feasibility requirement
- Use NumPy vectorized operations for parallel processing
- Save perturbed solutions and original local optima information to JSON and Excel files
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Set, Dict
import pickle
from utils import edges_to_routes as edges_to_routes
from utils import routes_to_edges as routes_to_edges_set
from utils import get_basin_paths, edges_hash as edges_hash_func, routes_to_solution_flat


def _calculate_gap(my_cost: float, hgs_cost: float) -> float:
    """Calculate gap percentage between my solution and HGS solution."""
    if hgs_cost is None or hgs_cost == 0:
        return None
    return ((my_cost - hgs_cost) / hgs_cost) * 100.0


def _convert_hgs_routes(hgs_routes):
    """Convert HGS routes to standard format: list of [0, ...nodes..., 0]."""
    if not hgs_routes or not isinstance(hgs_routes, list):
        return None
    
    if isinstance(hgs_routes[0], (int, np.integer)):
        # Flat sequence: [0, 1, 2, 0, 3, 4, 0] or [0, 1, 2, 0, 3, 4]
        routes, current = [], [0]
        for node in hgs_routes:
            if node == 0:
                if len(current) > 1:
                    # Ensure route ends with depot
                    if current[-1] != 0:
                        current.append(0)
                    routes.append(current)
                current = [0]
            else:
                current.append(int(node))
        # Handle last route if it doesn't end with 0
        if len(current) > 1:
            if current[-1] != 0:
                current.append(0)
            routes.append(current)
        return routes if routes else None
    else:
        # List of routes
        formatted = []
        for route in hgs_routes:
            if isinstance(route, list) and len(route) > 0:
                if len(route) >= 2 and route[0] == 0 and route[-1] == 0:
                    formatted.append(route)
                else:
                    clean = [n for n in route if n != 0]
                    if clean:
                        formatted.append([0] + clean + [0])
        return formatted if formatted else None


def is_routes_feasible_basic(
    routes: List[List[int]], 
    num_orders: int,
    demands: Dict[int, float] = None,
    vehicle_capacity: float = None
) -> bool:
    """
    CVRP feasibility check:
    - Each route starts/ends with depot 0, no depot in middle
    - Each customer [1, num_orders-1] appears exactly once
    - Route demand <= vehicle capacity (required)
    
    Args:
        num_orders: Total nodes (1 depot + num_orders-1 customers), e.g., 101
        demands: Dict mapping customer node -> demand (required)
        vehicle_capacity: Vehicle capacity (required)
    """
    if not routes:
        return False
    
    # Capacity check is required
    if demands is None:
        raise ValueError("demands is required for feasibility check")
    if vehicle_capacity is None:
        raise ValueError("vehicle_capacity is required for feasibility check")

    served = set()
    for route in routes:
        if len(route) < 2:
            return False
        # Each route must start and end with depot 0
        if route[0] != 0 or route[-1] != 0:
            return False
        
        # Check route capacity constraint (required)
        route_demand = sum(demands.get(node, 0) for node in route[1:-1])
        if route_demand > vehicle_capacity:
            return False
        
        # Check each customer node in the route
        for node in route[1:-1]:
            # Depot 0 must not appear in the middle
            if node == 0:
                return False
            # Customer nodes: [1, num_orders-1]
            if node < 1 or node >= num_orders:
                return False
            # Each customer must appear exactly once
            if node in served:
                return False
            served.add(node)

    # All customers 1..(num_orders-1) must be served exactly once
    return served == set(range(1, num_orders))


def double_bridge_move(
    edges: List[List[int]],
    num_orders: int = None,
    rng: np.random.Generator = None,
    max_attempts: int = 100,
    avoid_depot_boundaries: bool = False,
    demands: Dict[int, float] = None,
    vehicle_capacity: float = None,
    require_feasible: bool = True,
) -> Tuple[List[List[int]], int, bool]:
    """
    Apply Double-Bridge Move perturbation to a solution.
    Treat CVRP as TSP by connecting all routes through depot, then perform double-bridge move on the entire tour.
    
    Double-Bridge Move: A-B-C-D-E -> A-D-C-B-E (4 cut points, 4 edges removed, 4 edges added)
    
    Args:
        edges: List of edges (undirected, each edge is [u, v])
        num_orders: Total nodes (1 depot + num_orders-1 customers). If None, inferred from edges (max_id + 1)
        rng: NumPy random number generator
        max_attempts: Maximum retry attempts for feasible solution
        avoid_depot_boundaries: If True, avoid cutting at depot boundaries
        demands: Optional dict mapping customer node -> demand (for capacity check)
        vehicle_capacity: Optional vehicle capacity (for capacity check)
        require_feasible: If True, retry until feasible; If False, return first attempt result

    Returns:
        (perturbed_edges, attempts_used, success)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Infer num_orders from edges if not provided
    if num_orders is None:
        max_node_id = max(max(e) for e in edges) if edges else 0
        num_orders = max_node_id + 1
    
    # Convert edges to routes
    routes = edges_to_routes(edges, num_orders)
    
    if len(routes) == 0:
        return edges.copy(), 1, False

    # Retry until feasible (or max_attempts) if require_feasible is True
    # Otherwise, just do one attempt and return result regardless of feasibility
    attempts_limit = max_attempts if require_feasible else 1
    
    for attempt in range(1, attempts_limit + 1):
        # Connect all routes into a single tour (depot 0 as separator)
        tour: List[int] = []
        for i, route in enumerate(routes):
            if i == 0:
                tour.extend(route)  # First route: keep all nodes including depots
            else:
                tour.extend(route[1:])  # Subsequent routes: skip first depot (already connected)
        
        if len(tour) < 8:
            return edges.copy(), attempt, False
        
        n = len(tour)
        # Choose 4 cut points
        if avoid_depot_boundaries:
            # Avoid cutting at depot boundaries to reduce degeneracy
            valid_positions = [i for i in range(1, n) if tour[i] != 0 and tour[i - 1] != 0]
            if len(valid_positions) < 4:
                valid_positions = list(range(1, n))  # Fall back to all positions
        else:
            # Allow cutting anywhere
            valid_positions = list(range(1, n))
        
        if len(valid_positions) < 4:
            return edges.copy(), attempt, False
        
        p1, p2, p3, p4 = sorted(rng.choice(valid_positions, size=4, replace=False))
        part_A, part_B, part_C, part_D, part_E = tour[0:p1], tour[p1:p2], tour[p2:p3], tour[p3:p4], tour[p4:]
        new_tour = part_A + part_D + part_C + part_B + part_E  # A-B-C-D-E -> A-D-C-B-E
        
        # Convert tour back to routes (split at depot 0)
        new_routes: List[List[int]] = []
        current_route: List[int] = [0]
        for node in new_tour:
            if node == 0:
                if len(current_route) > 1:
                    current_route.append(0)
                    new_routes.append(current_route)
                current_route = [0]
            else:
                current_route.append(node)
        if len(current_route) > 1:
            current_route.append(0)
            new_routes.append(current_route)
        
        # Check feasibility
        is_feasible = False
        if demands is not None and vehicle_capacity is not None:
            is_feasible = is_routes_feasible_basic(new_routes, num_orders, demands, vehicle_capacity)
        else:
            # If no capacity constraints provided, assume feasible (basic structure check)
            is_feasible = True
        
        # Convert to edges
        edges_set = routes_to_edges_set(new_routes)
        new_edges = [[int(u), int(v)] for (u, v) in sorted(edges_set)]
        
        # If require_feasible is False, return immediately
        if not require_feasible:
            return new_edges, attempt, is_feasible
        
        # If require_feasible is True, only return if feasible
        if is_feasible:
            return new_edges, attempt, True
    
    # Failed to find feasible perturbation
    return edges.copy(), max_attempts, False


def remove_and_insert_move(
    edges: List[List[int]],
    num_orders: int = None,
    rng: np.random.Generator = None,
    max_attempts: int = 100,
    demands: Dict[int, float] = None,
    vehicle_capacity: float = None,
    require_feasible: bool = True,
) -> Tuple[List[List[int]], int, bool]:
    """
    Apply Remove-and-Insert Move perturbation to a solution.
    Randomly select a customer node, remove it from its current route, and insert it into a feasible position.
    This move removes 2 edges and adds 2 edges.
    
    Args:
        edges: List of edges (undirected, each edge is [u, v])
        num_orders: Total nodes (1 depot + num_orders-1 customers). If None, inferred from edges (max_id + 1)
        rng: NumPy random number generator
        max_attempts: Maximum retry attempts for feasible solution
        demands: Dict mapping customer node -> demand (required for capacity check)
        vehicle_capacity: Vehicle capacity (required for capacity check)
        require_feasible: If True, retry until feasible; If False, return first attempt result

    Returns:
        (perturbed_edges, attempts_used, success)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if demands is None or vehicle_capacity is None:
        raise ValueError("demands and vehicle_capacity are required for remove_and_insert_move")
    
    # Infer num_orders from edges if not provided
    if num_orders is None:
        max_node_id = max(max(e) for e in edges) if edges else 0
        num_orders = max_node_id + 1
    
    # Convert edges to routes
    routes = edges_to_routes(edges, num_orders)
    
    if len(routes) == 0:
        return edges.copy(), 1, False
    
    # Get all customer nodes
    all_customers = set(range(1, num_orders))
    
    # Retry until feasible (or max_attempts) if require_feasible is True
    attempts_limit = max_attempts if require_feasible else 1
    
    for attempt in range(1, attempts_limit + 1):
        # Make a copy of routes for modification
        new_routes = [route.copy() for route in routes]
        
        # Randomly select a customer node to remove
        customer_nodes = []
        for route in new_routes:
            customer_nodes.extend([node for node in route[1:-1] if node in all_customers])
        
        if len(customer_nodes) == 0:
            return edges.copy(), attempt, False
        
        node_to_remove = rng.choice(customer_nodes)
        
        # Find and remove the node from its current route
        removed = False
        original_route_idx = None  # Record original route index before removal
        route_to_remove_idx = None
        for route_idx, route in enumerate(new_routes):
            if node_to_remove in route:
                original_route_idx = route_idx  # Save original route index
                # Remove the node
                route.remove(node_to_remove)
                # If route becomes empty (only depots), mark for removal
                if len(route) <= 2:  # Only [0, 0] or [0]
                    route_to_remove_idx = route_idx
                removed = True
                break
        
        # Remove empty route if needed
        if route_to_remove_idx is not None:
            new_routes.pop(route_to_remove_idx)
            # If the original route was removed, no need to exclude it
            if original_route_idx == route_to_remove_idx:
                original_route_idx = None  # Route was removed, no need to exclude
            # Update original_route_idx if it was after the removed route
            elif original_route_idx is not None and original_route_idx > route_to_remove_idx:
                original_route_idx -= 1  # Adjust index after removal
        
        if not removed:
            return edges.copy(), attempt, False
        
        # Find all feasible insertion positions (optimized: store route indices and position counts)
        # Instead of storing all positions, we store (route_idx, num_positions) pairs
        # This reduces memory usage and speeds up selection
        # IMPORTANT: Exclude the original route to ensure meaningful perturbation
        feasible_routes = []  # List of (route_idx, num_positions) tuples
        node_demand = demands.get(node_to_remove, 0)
        
        for route_idx, route in enumerate(new_routes):
            # Skip the original route to avoid inserting back to the same position
            if original_route_idx is not None and route_idx == original_route_idx:
                continue
            
            route_demand = sum(demands.get(node, 0) for node in route[1:-1])
            
            # Check if node can be inserted into this route (capacity check)
            if route_demand + node_demand <= vehicle_capacity:
                # All positions in this route are feasible
                num_positions = len(route) - 1  # Positions: 1 to len(route)-1
                if num_positions > 0:
                    feasible_routes.append((route_idx, num_positions))
        
        # Also consider creating a new route with just this node
        if node_demand <= vehicle_capacity:
            feasible_routes.append((len(new_routes), 1))  # New route, 1 position
        
        if len(feasible_routes) == 0:
            # No feasible position found, restore original
            if require_feasible:
                continue  # Try again
            else:
                # Return original edges if not requiring feasible
                return edges.copy(), attempt, False
        
        # Randomly select a feasible route and position
        # Use weighted selection based on number of positions in each route
        total_positions = sum(num_pos for _, num_pos in feasible_routes)
        if total_positions == 0:
            if require_feasible:
                continue
            else:
                return edges.copy(), attempt, False
        
        # Select a random position index (0 to total_positions-1)
        selected_pos_idx = rng.integers(0, total_positions)
        
        # Find which route this position belongs to
        current_pos = 0
        route_idx = None
        insert_pos = None
        for r_idx, num_pos in feasible_routes:
            if selected_pos_idx < current_pos + num_pos:
                route_idx = r_idx
                # Calculate the actual position within this route
                pos_in_route = selected_pos_idx - current_pos
                if route_idx < len(new_routes):
                    # Position in existing route: 1 + pos_in_route
                    insert_pos = 1 + pos_in_route
                else:
                    # New route: always position 1
                    insert_pos = 1
                break
            current_pos += num_pos
        
        # Insert the node
        if route_idx < len(new_routes):
            new_routes[route_idx].insert(insert_pos, node_to_remove)
        else:
            # Create new route
            new_routes.append([0, node_to_remove, 0])
        
        # Check feasibility
        is_feasible = is_routes_feasible_basic(new_routes, num_orders, demands, vehicle_capacity)
        
        # Convert to edges
        edges_set = routes_to_edges_set(new_routes)
        new_edges = [[int(u), int(v)] for (u, v) in sorted(edges_set)]
        
        # If require_feasible is False, return immediately
        if not require_feasible:
            return new_edges, attempt, is_feasible
        
        # If require_feasible is True, only return if feasible
        if is_feasible:
            return new_edges, attempt, True
    
    # Failed to find feasible perturbation
    return edges.copy(), max_attempts, False


def compute_cost_from_edges(edges: List[List[int]], coordinates: List[List[float]], scale: float = 100.0) -> float:
    """
    Compute total cost from edges using Euclidean distance.
    Coordinates are scaled to [0, 100] range, so we multiply by scale factor.
    
    Args:
        edges: List of edges (undirected, each edge is [u, v])
        coordinates: List of coordinates in [0, 1] range, coordinates[0] is depot, coordinates[1:] are customers
        scale: Scaling factor to convert coordinates to [0, 100] range (default: 100.0)
    
    Returns:
        Total cost (sum of edge distances, scaled to [0, 100] range)
    """
    total_cost = 0.0
    
    for edge in edges:
        u, v = edge[0], edge[1]
        # Get coordinates
        coord_u = coordinates[u] if u < len(coordinates) else coordinates[0]
        coord_v = coordinates[v] if v < len(coordinates) else coordinates[0]
        
        # Scale coordinates to [0, 100] range
        scaled_u = [c * scale for c in coord_u]
        scaled_v = [c * scale for c in coord_v]
        
        # Compute Euclidean distance
        dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(scaled_u, scaled_v)))
        total_cost += dist
    
    return total_cost


def edges_to_normalized_set(edges: List[List[int]]) -> Set[Tuple[int, int]]:
    """Convert edge list to normalized edge set (for Jaccard distance calculation)"""
    edge_set = set()
    for edge in edges:
        u, v = edge[0], edge[1]
        # Normalize edge (smaller node first)
        normalized_edge = (min(u, v), max(u, v))
        edge_set.add(normalized_edge)
    return edge_set


def jaccard_distance_vectorized(edges_sets_list: List[Set[Tuple[int, int]]]) -> np.ndarray:
    """
    Batch compute Jaccard distance using NumPy vectorized operations
    Compute Jaccard distance between each perturbed solution and corresponding original solution
    
    Args:
        edges_sets_list: List of tuples, each element is (original_edges_set, perturbed_edges_set)
    
    Returns:
        Array of Jaccard distances
    """
    n = len(edges_sets_list)
    jaccard_dists = np.zeros(n, dtype=np.float64)
    
    for i, (original_set, perturbed_set) in enumerate(edges_sets_list):
        if len(original_set) == 0 and len(perturbed_set) == 0:
            jaccard_dists[i] = 0.0
            continue
        
        intersection = len(original_set & perturbed_set)
        union = len(original_set | perturbed_set)
        
        if union == 0:
            jaccard_dists[i] = 1.0
        else:
            jaccard_sim = intersection / union
            jaccard_dists[i] = 1.0 - jaccard_sim
    
    return jaccard_dists


def calculate_broken_pairs_distance(original_routes: List[List[int]], 
                                     perturbed_routes: List[List[int]]) -> Tuple[int, int, float]:
    """
    Calculate Broken Pairs Distance (BPD)
    
    Broken Pairs Distance is the number of adjacent node pairs in the original solution
    that are no longer adjacent in the perturbed solution.
    
    Note: CVRP is undirected, so we only care about adjacency itself, not direction.
    Adjacent pairs (u, v) and (v, u) are treated as the same, normalized to (min(u,v), max(u,v)).
    
    Args:
        original_routes: List of routes in the original solution
        perturbed_routes: List of routes in the perturbed solution
    
    Returns:
        (broken_pairs_count, total_pairs_count, broken_pairs_ratio)
    """
    # Extract all adjacent node pairs from original solution (including depot)
    # CVRP is undirected, so normalize to (min, max) form
    original_pairs = set()
    for route in original_routes:
        for i in range(len(route) - 1):
            # Adjacent node pair, normalized to undirected form
            pair = (route[i], route[i + 1])
            normalized_pair = (min(pair[0], pair[1]), max(pair[0], pair[1]))
            original_pairs.add(normalized_pair)
    
    # Extract all adjacent node pairs from perturbed solution
    perturbed_pairs = set()
    for route in perturbed_routes:
        for i in range(len(route) - 1):
            pair = (route[i], route[i + 1])
            normalized_pair = (min(pair[0], pair[1]), max(pair[0], pair[1]))
            perturbed_pairs.add(normalized_pair)
    
    # Calculate broken pairs: pairs that are adjacent in original but not in perturbed
    broken_pairs = original_pairs - perturbed_pairs
    
    total_pairs_count = len(original_pairs)
    broken_pairs_count = len(broken_pairs)
    broken_pairs_ratio = broken_pairs_count / total_pairs_count if total_pairs_count > 0 else 0.0
    
    return broken_pairs_count, total_pairs_count, broken_pairs_ratio


def run_local_search_from_perturbed(perturbed_edges: List[List[int]], 
                                    instance_path: str, instance_index: int,
                                    num_runs: int = 100,
                                    original_edges_hash: str = None) -> Dict:
    """
    Run local search from perturbed solution multiple times and collect basin statistics.
    
    Args:
        perturbed_edges: Edges of the perturbed solution
        instance_path: Path to instance pkl file
        instance_index: Instance index
        num_runs: Number of local search runs (default: 100)
        original_edges_hash: Hash of original solution (for counting returns)
    
    Returns:
        Dictionary with basin statistics
    """
    # Heavy imports live inside the function to keep module import light.
    from analyze_basin import ensure_cuda_device, aggressive_gc_cleanup, run_local_search_silent
    from test_basin_pybind import create_vrp_instance_from_pkl
    
    # Convert edges to routes
    max_node_id = max(max(e) for e in perturbed_edges) if perturbed_edges else 0
    num_orders = max_node_id + 1
    perturbed_routes = edges_to_routes(perturbed_edges, num_orders)
    
    if not perturbed_routes:
        return {
            'basin_counts': {},
            'basin_data': {},
            'num_unique_basins': 0,
            'return_to_original': 0,
            'return_to_original_ratio': 0.0,
            'final_costs': [],
            'mean_final_cost': 0.0
        }
    
    # Accumulators
    basin_counts = defaultdict(int)
    basin_data = {}  # edges_hash -> {edges, solution_flat, costs}
    final_costs = []
    return_to_original = 0
    
    for run in range(num_runs):
        run_vrp_instance = None
        cuopt_env = None
        
        try:
            ensure_cuda_device()
            run_vrp_instance = create_vrp_instance_from_pkl(instance_path, instance_index, num_vehicles=30)
            cuopt_env = run_vrp_instance['cuopt_env']
            
            # Clear large arrays to reduce memory
            if 'cost_matrix' in run_vrp_instance:
                del run_vrp_instance['cost_matrix']
            if 'node_coords' in run_vrp_instance:
                del run_vrp_instance['node_coords']
            
            final_cost, edges_hash_val, final_edges, final_routes = run_local_search_silent(
                cuopt_env, perturbed_routes, run_vrp_instance, max_iterations=100000)
            
            if edges_hash_val not in basin_data:
                basin_data[edges_hash_val] = {
                    'edges': sorted(final_edges),
                    'solution_flat': routes_to_solution_flat(final_routes, num_orders),
                    'costs': []
                }
            basin_data[edges_hash_val]['costs'].append(final_cost)
            
            del final_routes
            del final_edges
            
            basin_counts[edges_hash_val] += 1
            final_costs.append(final_cost)
            
            if original_edges_hash and edges_hash_val == original_edges_hash:
                return_to_original += 1
                
        except Exception as e:
            error_msg = str(e)
            # Check for specific CUDA errors that might need special handling
            if "misaligned" in error_msg.lower() or "cuda" in error_msg.lower():
                print(f"    Warning: Local search run {run+1} failed (CUDA error): {error_msg[:200]}")
                # For CUDA errors, do extra aggressive cleanup
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except:
                    pass
            else:
                print(f"    Warning: Local search run {run+1} failed: {error_msg[:200]}")
        finally:
            if cuopt_env is not None:
                try:
                    cuopt_env.set_routes_to_search()
                    cuopt_env.release_resource()
                    cuopt_env.sync_streams()
                except:
                    pass
                finally:
                    del cuopt_env
                cuopt_env = None
            
            if run_vrp_instance is not None:
                if 'cuopt_env' in run_vrp_instance:
                    del run_vrp_instance['cuopt_env']
                del run_vrp_instance
                run_vrp_instance = None
            
            aggressive_gc_cleanup()
    
    total_runs = len(final_costs)
    mean_final_cost = sum(final_costs) / total_runs if total_runs else 0.0
    return_to_original_ratio = return_to_original / total_runs if total_runs > 0 else 0.0
    
    return {
        'basin_counts': dict(basin_counts),
        'basin_data': basin_data,
        'num_unique_basins': len(basin_counts),
        'return_to_original': return_to_original,
        'return_to_original_ratio': return_to_original_ratio,
        'num_runs': total_runs,
        'final_costs': final_costs,
        'mean_final_cost': mean_final_cost
    }


def process_optima_vectorized(optima: List[dict], seed: int = None, 
                              demands: Dict[int, float] = None,
                              vehicle_capacity: float = None,
                              operator_type: str = 'double_bridge',
                              k: int = 1,
                              require_feasible: bool = False,
                              coordinates: List[List[float]] = None,
                              instance_path: str = None,
                              instance_index: int = 0,
                              num_local_search_runs: int = 30,
                              jsonl_file=None, optima_stats: Dict[str, dict] = None,
                              hgs_cost: float = None, processed_hashes: set = None) -> List[dict]:
    """
    Batch process all optima: apply perturbation k times, then run local search from each perturbed solution.
    
    Args:
        optima: List of optima
        seed: Random seed (for reproducibility)
        demands: Dict mapping customer node -> demand (required for capacity check)
        vehicle_capacity: Vehicle capacity (required for capacity check)
        operator_type: 'double_bridge' or 'remove_and_insert'
        k: Number of times to apply the operator
        require_feasible: If True, only return feasible solutions; If False, allow infeasible
        coordinates: List of coordinates [depot, customer1, customer2, ...] for cost calculation
        instance_path: Path to instance pkl file (required for local search)
        instance_index: Instance index (required for local search)
        num_local_search_runs: Number of local search runs per perturbed solution (default: 30)
        jsonl_file: Optional file handle for streaming JSONL writes (if provided, writes immediately after each optimum)
        optima_stats: Optional stats dict for JSONL writes (required if jsonl_file is provided)
        hgs_cost: Optional HGS cost for gap calculation (required if jsonl_file is provided)
        processed_hashes: Optional set of edges_hash to skip (for resume functionality)
    
    Returns:
        List of processing results, each element contains original optima info, 
        perturbed solutions at each step, and local search results
    """
    n = len(optima)
    operator_name = 'Double-Bridge Move' if operator_type == 'double_bridge' else 'Remove-and-Insert'
    print(f"  Processing {n} optima with {operator_name} (k={k}, require_feasible={require_feasible})...")
    print(f"  Running {num_local_search_runs} local search runs from each perturbed solution...")
    
    # Create random number generators (use different seed for each optimum to ensure independence)
    if seed is not None:
        rngs = [np.random.default_rng(seed + i) for i in range(n)]
    else:
        rngs = [np.random.default_rng() for _ in range(n)]
    
    results = []
    
    # Feasibility / retry statistics
    total = 0
    feasible_success = 0
    failed_feasible = 0
    
    # Process each optimum
    skipped_count = 0
    for i, opt in enumerate(optima):
        try:
            total += 1
            edges = opt.get('edges', [])
            optimum_id = opt.get('optimum_id', i)
            original_edges_hash = opt.get('edges_hash')
            
            # Skip if already processed (resume functionality)
            if processed_hashes and original_edges_hash and original_edges_hash in processed_hashes:
                skipped_count += 1
                print(f"  Skipping optimum {i+1}/{n} (optimum_id={optimum_id}, hash={original_edges_hash[:8]}): already processed")
                continue
            
            if not edges:
                result = {
                    'optimum_id': optimum_id,
                    'original': opt,
                    'perturbed_steps': [],
                    'error': 'No edges found'
                }
                results.append(result)
                # Still write to JSONL even if error (for consistency)
                if jsonl_file and optima_stats is not None:
                    write_result_to_jsonl_stream(result, optima_stats, jsonl_file, hgs_cost)
                continue
            
            # Infer num_orders from edges
            max_node_id = max(max(e) for e in edges) if edges else 0
            num_orders = max_node_id + 1
            original_edges_set = edges_to_normalized_set(edges)
            
            # Apply operator k times, record each step
            current_edges = edges
            perturbed_steps = []
            all_feasible = True
            
            for step in range(1, k + 1):
                if operator_type == 'double_bridge':
                    perturbed_edges, attempts_used, feasible = double_bridge_move(
                        current_edges, num_orders, rngs[i], 
                        demands=demands, vehicle_capacity=vehicle_capacity,
                        require_feasible=require_feasible
                    )
                elif operator_type == 'remove_and_insert':
                    perturbed_edges, attempts_used, feasible = remove_and_insert_move(
                        current_edges, num_orders, rngs[i],
                        demands=demands, vehicle_capacity=vehicle_capacity,
                        require_feasible=require_feasible
                    )
                else:
                    raise ValueError(f"Unknown operator_type: {operator_type}")
                
                if not feasible:
                    all_feasible = False
                    if require_feasible:
                        break
                
                # Compute hash, Jaccard distance, Broken Pairs Distance, and cost for this step
                perturbed_edges_set = edges_to_normalized_set(perturbed_edges)
                perturbed_hash = edges_hash_func(perturbed_edges_set)
                
                # Jaccard distance between original basin and this perturbed solution
                if len(original_edges_set) == 0 and len(perturbed_edges_set) == 0:
                    jaccard_distance = 0.0
                else:
                    inter = len(original_edges_set & perturbed_edges_set)
                    union = len(original_edges_set | perturbed_edges_set)
                    jaccard_distance = 1.0 - (inter / union) if union > 0 else 1.0
                
                # Broken Pairs Distance between original basin and this perturbed solution
                original_routes_for_bpd = edges_to_routes(edges, num_orders)
                perturbed_routes_for_bpd = edges_to_routes(perturbed_edges, num_orders)
                broken_pairs_count, total_pairs_count, broken_pairs_ratio = calculate_broken_pairs_distance(
                    original_routes_for_bpd, perturbed_routes_for_bpd
                )
                
                perturbed_cost = compute_cost_from_edges(perturbed_edges, coordinates, scale=100.0) if coordinates else None
                
                # Run local search from this perturbed solution
                print(
                    f"    Optimum {i+1}/{n}, Step {step}/{k}: "
                    f"Jaccard={jaccard_distance:.4f}, "
                    f"BrokenPairs={int(broken_pairs_count)}/{int(total_pairs_count)} ({broken_pairs_ratio:.4f}), "
                    f"perturb_cost={(f'{perturbed_cost:.2f}' if perturbed_cost is not None else 'N/A')}, "
                    f"running {num_local_search_runs} local search runs..."
                )
                ls_results = run_local_search_from_perturbed(
                    perturbed_edges, instance_path, instance_index,
                    num_runs=num_local_search_runs,
                    original_edges_hash=original_edges_hash
                )
                bc = ls_results.get('basin_counts', {}) or {}
                num_runs_done = ls_results.get('num_runs', len(ls_results.get('final_costs', [])) or num_local_search_runs)
                ret_cnt = int(ls_results.get('return_to_original') or 0)
                ret_prob = float(ls_results.get('return_to_original_ratio') or 0.0)
                num_basins = int(ls_results.get('num_unique_basins') or len(bc))
                # Top-3 basins (excluding original) for a quick glance
                other = [(h, c) for h, c in bc.items() if h != original_edges_hash]
                other.sort(key=lambda x: x[1], reverse=True)
                top_str = ", ".join(f"{h[:6]}:{c}" for h, c in other[:3]) if other else "-"
                print(
                    f"      -> done: runs={num_runs_done}, "
                    f"return_to_original={ret_cnt} ({ret_prob:.3f}), "
                    f"num_basins={num_basins}, top_other=[{top_str}]"
                )
                
                perturbed_steps.append({
                    'step': step,
                    'edges': perturbed_edges,
                    'edges_hash': perturbed_hash,
                    'jaccard_distance': jaccard_distance,
                    'broken_pairs_count': broken_pairs_count,
                    'broken_pairs_total': total_pairs_count,
                    'broken_pairs_ratio': broken_pairs_ratio,
                    'cost': perturbed_cost,
                    'feasible': feasible,
                    'attempts_used': int(attempts_used),
                    'local_search_results': ls_results
                })
                
                current_edges = perturbed_edges
            
            if all_feasible:
                feasible_success += 1
            else:
                failed_feasible += 1
            
            result = {
                'optimum_id': optimum_id,
                'original': opt,
                'perturbed_steps': perturbed_steps,
                'error': None,
                'operator_type': operator_type,
                'operator_k': k,
            }
            results.append(result)
            
            # Write to JSONL immediately if streaming mode is enabled
            if jsonl_file and optima_stats is not None:
                write_result_to_jsonl_stream(result, optima_stats, jsonl_file, hgs_cost)
            
            # Visual separator between optima in logs
            hash_short = original_edges_hash[:8] if original_edges_hash else "N/A"
            original_cost = opt.get("final_cost")
            print(
                f"  ---- Finished optimum {i+1}/{n} "
                f"(optimum_id={optimum_id}, hash={hash_short}, original_cost={original_cost}) ----\n"
            )
            
            # Progress report
            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{n} optima...")
        
        except Exception as e:
            print(f"  Error processing optimum {i+1}/{n}: {e}")
            import traceback
            traceback.print_exc()
            result = {
                'optimum_id': opt.get('optimum_id', i),
                'original': opt,
                'perturbed_steps': [],
                'error': str(e),
                'operator_type': operator_type,
                'operator_k': k,
            }
            results.append(result)
            # Write error result to JSONL if streaming
            if jsonl_file and optima_stats is not None:
                write_result_to_jsonl_stream(result, optima_stats, jsonl_file, hgs_cost)
    
    # Print statistics
    if total > 0:
        print("\nPerturbation statistics:")
        print(f"  Total optima processed: {total}")
        print(f"  Fully feasible: {feasible_success} ({100.0 * feasible_success / total:.2f}%)")
        print(f"  Failed feasibility: {failed_feasible} ({100.0 * failed_feasible / total:.2f}%)")
        if skipped_count > 0:
            print(f"  Skipped (already processed): {skipped_count}")
    
    return results


def load_optima_from_jsonl(jsonl_path: str) -> List[dict]:
    """Load all local optima data from JSONL file"""
    if not os.path.exists(jsonl_path):
        return []
    
    optima = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                optima.append(json.loads(line))
    return optima


def load_trial_info(trials_path: str) -> Dict[int, dict]:
    """
    Load trial information from trials.jsonl
    Returns: dict mapping optimum_id to trial info
    """
    trial_info = {}
    if not os.path.exists(trials_path):
        print(f"  Warning: Trials file does not exist: {trials_path}")
        return trial_info
    
    try:
        with open(trials_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    optimum_id = record.get('optimum_id')
                    if optimum_id is not None:
                        trial_info[optimum_id] = {
                            'run_id': record.get('run_id'),
                            'trial_id': record.get('trial_id'),
                            'n_steps': record.get('n_steps'),
                            'optimum_edges_hash': record.get('optimum_edges_hash')
                        }
                except json.JSONDecodeError as e:
                    print(f"  Warning: Failed to parse JSON: {e}")
                    continue
    except Exception as e:
        print(f"  Error: Failed to read trials file {trials_path}: {e}")
    
    return trial_info


def load_last_solutions_per_trial(trajectory_path: str) -> Dict[tuple, dict]:
    """
    Load the last solution from each trial in trajectory.jsonl.
    Local optima are the last solutions of each trial.
    
    Returns: dict mapping (run_id, trial_id) -> last solution record
    """
    last_solutions = {}
    if not os.path.exists(trajectory_path):
        print(f"  Warning: Trajectory file does not exist: {trajectory_path}")
        return last_solutions
    
    try:
        with open(trajectory_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    run_id = record.get('run_id')
                    trial_id = record.get('trial_id')
                    
                    if run_id is None or trial_id is None:
                        continue
                    
                    key = (run_id, trial_id)
                    
                    # Update last solution (compare by global_iter, then local_iter)
                    if key not in last_solutions:
                        last_solutions[key] = record
                    else:
                        current = last_solutions[key]
                        current_global = current.get('global_iter', 0) or 0
                        current_local = current.get('local_iter', 0) or 0
                        record_global = record.get('global_iter', 0) or 0
                        record_local = record.get('local_iter', 0) or 0
                        
                        if (record_global > current_global or 
                            (record_global == current_global and record_local > current_local)):
                            last_solutions[key] = record
                            
                except json.JSONDecodeError as e:
                    print(f"  Warning: Failed to parse JSON: {e}")
                    continue
    except Exception as e:
        print(f"  Error: Failed to read trajectory file {trajectory_path}: {e}")
    
    return last_solutions


def load_trajectory_info(trajectory_path: str) -> Dict[str, List[dict]]:
    """
    Load trajectory information from trajectory.jsonl.
    Build a mapping from edges_hash to list of occurrences (from last solutions per trial).
    
    Returns: dict mapping edges_hash to list of occurrences (run_id, trial_id, global_iter, local_iter)
    """
    trajectory_info = defaultdict(list)
    
    # Get last solutions per trial
    last_solutions = load_last_solutions_per_trial(trajectory_path)
    
    # Build edges_hash -> occurrences mapping
    for (run_id, trial_id), record in last_solutions.items():
        edges_hash = record.get('edges_hash')
        if edges_hash:
            trajectory_info[edges_hash].append({
                'run_id': run_id,
                'trial_id': trial_id,
                'global_iter': record.get('global_iter'),
                'local_iter': record.get('local_iter'),
                'optimum_id': record.get('optimum_id')
            })
    
    return dict(trajectory_info)


def compute_optima_frequency_and_occurrences(optima: List[dict], trajectory_info: Dict[str, List[dict]]) -> Dict[str, dict]:
    """
    Compute frequency and occurrences for each optimum
    Returns: dict mapping edges_hash to frequency and occurrence details
    Note: Uses edges_hash as key (not optimum_id) because optimum_id is only unique within each run,
          but edges_hash is globally unique across all runs.
    Frequency = number of trials whose last solution is this local optima (same as count in optima.jsonl).
    """
    optima_stats = {}
    
    for opt in optima:
        optimum_id = opt.get('optimum_id')
        edges_hash = opt.get('edges_hash')
        run_id = opt.get('run_id')
        
        if edges_hash is None:
            continue
        
        # Get occurrences from trajectory (edges_hash is the key)
        # Each occurrence is a trial's last solution
        occurrences = trajectory_info.get(edges_hash, [])
        
        # Collect unique occurrence IDs: (run_id, trial_id, global_iter, local_iter)
        occurrence_ids = []
        seen_ids = set()
        for occ in occurrences:
            occ_run_id = occ.get('run_id')
            trial_id = occ.get('trial_id')
            global_iter = occ.get('global_iter')
            local_iter = occ.get('local_iter')
            
            # Create occurrence ID: run_id_trial_id_global_iter_local_iter
            if occ_run_id is not None and trial_id is not None:
                global_iter_str = str(global_iter) if global_iter is not None else 'None'
                local_iter_str = str(local_iter) if local_iter is not None else 'None'
                
                occurrence_id = f"{occ_run_id}_{trial_id}_{global_iter_str}_{local_iter_str}"
                
                if occurrence_id not in seen_ids:
                    seen_ids.add(occurrence_id)
                    occurrence_ids.append({
                        'occurrence_id': occurrence_id,
                        'run_id': occ_run_id,
                        'trial_id': trial_id,
                        'global_iter': global_iter,
                        'local_iter': local_iter
                    })
        
        # Frequency = number of trials whose last solution is this local optima
        # This should equal the count in optima.jsonl (before dedup)
        frequency = len(occurrence_ids)
        
        # Use edges_hash as key (globally unique), not optimum_id (only unique within each run)
        optima_stats[edges_hash] = {
            'optimum_id': optimum_id,  # Keep for reference, but may vary across runs
            'frequency': frequency,
            'run_id': run_id,  # Keep first seen run_id for reference
            'occurrence_ids': sorted(occurrence_ids, key=lambda x: (
                x['run_id'] or '',
                x['trial_id'] or -1,
                x['global_iter'] or -1,
                x['local_iter'] or -1
            ))
        }
    
    return optima_stats


def load_processed_hashes_from_jsonl(jsonl_path: str) -> set:
    """Load already processed edges_hash from existing JSONL file."""
    if not os.path.exists(jsonl_path):
        return set()
    
    processed_hashes = set()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            edges_hash = record.get('initial_solution', {}).get('edges_hash')
            if edges_hash:
                processed_hashes.add(edges_hash)
    return processed_hashes


def load_processed_hashes_from_excel(excel_path: str) -> set:
    """Load already processed edges_hash from existing Excel file."""
    if not os.path.exists(excel_path):
        return set()
    
    df = pd.read_excel(excel_path)
    if 'original_edges_hash' not in df.columns:
        return set()
    
    return {str(h) for h in df['original_edges_hash'].dropna() if h}


def save_excel_from_training_jsonl(
    jsonl_path: str,
    output_xlsx_path: str,
    optima_stats: Dict[str, dict] = None,
):
    """
    Build (or rebuild) the per-optimum Excel summary from a training_data JSONL file.

    This is used for fault-tolerant resume: we treat JSONL as the source of truth.
    If Excel is missing or partial, we regenerate it from JSONL.

    JSONL format is the one written by write_result_to_jsonl_stream():
      - initial_solution.edges_hash (original basin id)
      - perturbation_step (1..k)
      - perturbed_solution.* (hash/cost/feasible/jaccard/broken_pairs)
      - basin_distribution / num_unique_basins / return_to_original / return_to_original_ratio / num_runs / mean_final_cost
    """
    if not os.path.exists(jsonl_path):
        print(f"  Warning: JSONL not found, cannot build Excel: {jsonl_path}")
        return

    # Aggregate by original_edges_hash
    per_opt = {}
    max_step = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            init = rec.get("initial_solution") or {}
            original_hash = init.get("edges_hash")
            if not original_hash:
                continue

            step = int(rec.get("perturbation_step") or 0)
            if step <= 0:
                continue
            max_step = max(max_step, step)

            ent = per_opt.get(original_hash)
            if ent is None:
                ent = {
                    "optimum_id": rec.get("optimum_id"),
                    "run_id": rec.get("run_id"),
                    "trial_id": rec.get("trial_id"),
                    "global_iter": rec.get("global_iter"),
                    "local_iter": rec.get("local_iter"),
                    "operator_type": rec.get("operator_type", None),  # might be absent in JSONL
                    "operator_k": rec.get("operator_k", None),        # might be absent in JSONL
                    "original_cost": init.get("cost"),
                    "original_edges_hash": original_hash,
                    "steps": {},
                }
                per_opt[original_hash] = ent

            # Step payload
            ps = rec.get("perturbed_solution") or {}
            ls_num_runs = rec.get("num_runs") or 0
            try:
                ls_num_runs = int(ls_num_runs)
            except Exception:
                ls_num_runs = 0

            basin_dist = rec.get("basin_distribution") or {}
            # other basins list for Excel
            other_basins_list = []
            if isinstance(basin_dist, dict) and basin_dist:
                for h, c in basin_dist.items():
                    if h == original_hash:
                        continue
                    try:
                        c_int = int(c)
                    except Exception:
                        c_int = 0
                    prob = (c_int / ls_num_runs) if ls_num_runs > 0 else 0.0
                    other_basins_list.append({"hash": h, "count": c_int, "prob": prob})
                other_basins_list.sort(key=lambda x: x["count"], reverse=True)

            ent["steps"][step] = {
                "hash": ps.get("edges_hash"),
                "cost": ps.get("cost"),
                "feasible": bool(ps.get("feasible", False)),
                "jaccard_distance": ps.get("jaccard_distance"),
                "broken_pairs_count": ps.get("broken_pairs_count"),
                "broken_pairs_total": ps.get("broken_pairs_total"),
                "broken_pairs_ratio": ps.get("broken_pairs_ratio"),
                "return_to_original_count": int(rec.get("return_to_original") or 0),
                "return_to_original_prob": float(rec.get("return_to_original_ratio") or 0.0),
                "other_basins": other_basins_list,
                "num_unique_basins": int(rec.get("num_unique_basins") or 0),
                "mean_final_cost": rec.get("mean_final_cost"),
                "num_runs": ls_num_runs,
            }

    rows = []
    for original_hash, ent in per_opt.items():
        stats = (optima_stats or {}).get(original_hash, {}) if optima_stats else {}
        occurrence_ids = stats.get("occurrence_ids", []) or []
        occurrence_ids_str = "; ".join([occ.get("occurrence_id", "") for occ in occurrence_ids if occ.get("occurrence_id")])
        first_occ = occurrence_ids[0] if occurrence_ids else None

        row = {
            "optimum_id": stats.get("optimum_id", ent.get("optimum_id")),
            "run_id": stats.get("run_id", ent.get("run_id")),
            "frequency": stats.get("frequency", 0),
            "trial_id": (first_occ.get("trial_id") if first_occ else ent.get("trial_id")),
            "global_iter": (first_occ.get("global_iter") if first_occ else ent.get("global_iter")),
            "local_iter": (first_occ.get("local_iter") if first_occ else ent.get("local_iter")),
            "occurrence_ids": occurrence_ids_str,
            "operator_type": ent.get("operator_type") or "unknown",
            "operator_k": ent.get("operator_k") or max_step or 1,
            "original_cost": ent.get("original_cost"),
            "original_edges_hash": ent.get("original_edges_hash"),
        }

        for step in range(1, (max_step or 1) + 1):
            sd = ent["steps"].get(step)
            if sd is None:
                row[f"step_{step}_hash"] = None
                row[f"step_{step}_cost"] = None
                row[f"step_{step}_feasible"] = False
                row[f"step_{step}_jaccard_distance"] = None
                row[f"step_{step}_broken_pairs_count"] = None
                row[f"step_{step}_broken_pairs_total"] = None
                row[f"step_{step}_broken_pairs_ratio"] = None
                row[f"step_{step}_return_to_original_count"] = 0
                row[f"step_{step}_return_to_original_prob"] = 0.0
                row[f"step_{step}_other_basins"] = "[]"
                row[f"step_{step}_num_unique_basins"] = 0
                row[f"step_{step}_mean_final_cost"] = None
                continue

            row[f"step_{step}_hash"] = sd.get("hash")
            row[f"step_{step}_cost"] = sd.get("cost")
            row[f"step_{step}_feasible"] = sd.get("feasible", False)
            row[f"step_{step}_jaccard_distance"] = sd.get("jaccard_distance")
            row[f"step_{step}_broken_pairs_count"] = sd.get("broken_pairs_count")
            row[f"step_{step}_broken_pairs_total"] = sd.get("broken_pairs_total")
            row[f"step_{step}_broken_pairs_ratio"] = sd.get("broken_pairs_ratio")
            row[f"step_{step}_return_to_original_count"] = sd.get("return_to_original_count", 0)
            row[f"step_{step}_return_to_original_prob"] = sd.get("return_to_original_prob", 0.0)
            row[f"step_{step}_other_basins"] = json.dumps(sd.get("other_basins") or [], ensure_ascii=False)
            row[f"step_{step}_num_unique_basins"] = sd.get("num_unique_basins", 0)
            row[f"step_{step}_mean_final_cost"] = sd.get("mean_final_cost")

        rows.append(row)

    if not rows:
        if os.path.exists(output_xlsx_path):
            print(f"  No rows found in JSONL; keeping existing Excel: {output_xlsx_path}")
            return
        df = pd.DataFrame([])
        with pd.ExcelWriter(output_xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Perturbation Results", index=False)
        print(f"  Results saved to Excel (empty): {output_xlsx_path}")
        return

    df = pd.DataFrame(rows)
    if "frequency" in df.columns and "optimum_id" in df.columns:
        df = df.sort_values(["frequency", "optimum_id"], ascending=[False, True])

    with pd.ExcelWriter(output_xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Perturbation Results", index=False)
    print(f"  Excel rebuilt from JSONL: {output_xlsx_path} (rows={len(df)})")


def write_result_to_jsonl_stream(result: dict, optima_stats: Dict[str, dict], 
                                  jsonl_file, hgs_cost: float = None):
    """
    Write a single result to JSONL file (streaming mode).
    This function writes immediately after processing each optimum.
    """
    if result.get('error'):
        return
    
    optimum_id = result['optimum_id']
    original = result['original']
    original_edges_hash = original.get('edges_hash')
    # Use edges_hash as key (globally unique), not optimum_id (only unique within each run)
    stats = optima_stats.get(original_edges_hash, {}) if original_edges_hash else {}
    
    # Get first occurrence for run_id, trial_id, etc.
    first_occurrence = stats.get('occurrence_ids', [{}])[0] if stats.get('occurrence_ids') else {}
    
    # Prepare initial solution info
    original_edges = original.get('edges', [])
    original_cost = original.get('final_cost', 0.0)
    
    # Convert edges to solution_flat
    max_node_id = max(max(e) for e in original_edges) if original_edges else 0
    num_orders = max_node_id + 1
    original_routes = edges_to_routes(original_edges, num_orders)
    original_solution_flat = routes_to_solution_flat(original_routes, num_orders) if original_routes else []
    
    # For each perturbed step, create a training_data entry
    for step_data in result.get('perturbed_steps', []):
        step = step_data['step']
        perturbed_edges = step_data['edges']
        perturbed_edges_hash = step_data['edges_hash']
        perturbed_cost = step_data['cost']
        ls_results = step_data.get('local_search_results', {})
        
        # Build basin distribution (edges_hash -> count)
        basin_distribution = ls_results.get('basin_counts', {})
        basin_data = ls_results.get('basin_data', {})
        
        # Build final_solutions list (all unique basins reached)
        final_solutions = []
        for basin_hash, basin_info in basin_data.items():
            final_solutions.append({
                'edges': basin_info['edges'],
                'edges_hash': basin_hash,
                'solution_flat': basin_info['solution_flat'],
                'cost': sum(basin_info['costs']) / len(basin_info['costs']) if basin_info['costs'] else 0.0,
                'count': basin_distribution.get(basin_hash, 0)
            })
        
        # Create training_data entry
        training_entry = {
            'run_id': stats.get('run_id') or first_occurrence.get('run_id'),
            'trial_id': first_occurrence.get('trial_id'),
            'global_iter': first_occurrence.get('global_iter'),
            'local_iter': first_occurrence.get('local_iter'),
            'optimum_id': optimum_id,
            'perturbation_step': step,
            'initial_solution': {
                'edges': original_edges,
                'solution_flat': original_solution_flat,
                'cost': original_cost,
                'edges_hash': original_edges_hash,
                'gap_to_hgs': _calculate_gap(original_cost, hgs_cost)
            },
            'perturbed_solution': {
                'edges': perturbed_edges,
                'edges_hash': perturbed_edges_hash,
                'cost': perturbed_cost,
                'feasible': step_data.get('feasible', False),
                'jaccard_distance': step_data.get('jaccard_distance'),
                'broken_pairs_count': step_data.get('broken_pairs_count'),
                'broken_pairs_total': step_data.get('broken_pairs_total'),
                'broken_pairs_ratio': step_data.get('broken_pairs_ratio')
            },
            'final_solutions': final_solutions,
            'basin_distribution': basin_distribution,
            'num_unique_basins': ls_results.get('num_unique_basins', 0),
            'return_to_original': ls_results.get('return_to_original', 0),
            'return_to_original_ratio': ls_results.get('return_to_original_ratio', 0.0),
            'num_runs': len(ls_results.get('final_costs', [])),
            'mean_final_cost': ls_results.get('mean_final_cost', 0.0)
        }
        
        # Write as JSONL line immediately
        jsonl_file.write(json.dumps(training_entry, ensure_ascii=False) + '\n')
        jsonl_file.flush()  # Ensure data is written to disk immediately


def save_results_to_jsonl(results: List[dict], optima_stats: Dict[str, dict], output_path: str, hgs_cost: float = None):
    """
    Save results to JSONL file in training_data format.
    Each line contains: original solution, perturbed solutions at each step, and local search results.
    Note: For streaming writes, use write_result_to_jsonl_stream instead.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            if result.get('error'):
                continue
            
            optimum_id = result['optimum_id']
            original = result['original']
            original_edges_hash = original.get('edges_hash')
            # Use edges_hash as key (globally unique), not optimum_id (only unique within each run)
            stats = optima_stats.get(original_edges_hash, {}) if original_edges_hash else {}
            
            # Get first occurrence for run_id, trial_id, etc.
            first_occurrence = stats.get('occurrence_ids', [{}])[0] if stats.get('occurrence_ids') else {}
            
            # Prepare initial solution info
            original_edges = original.get('edges', [])
            original_edges_hash = original.get('edges_hash')
            original_cost = original.get('final_cost', 0.0)
            
            # Convert edges to solution_flat
            max_node_id = max(max(e) for e in original_edges) if original_edges else 0
            num_orders = max_node_id + 1
            original_routes = edges_to_routes(original_edges, num_orders)
            original_solution_flat = routes_to_solution_flat(original_routes, num_orders) if original_routes else []
            
            # For each perturbed step, create a training_data entry
            for step_data in result.get('perturbed_steps', []):
                step = step_data['step']
                perturbed_edges = step_data['edges']
                perturbed_edges_hash = step_data['edges_hash']
                perturbed_cost = step_data['cost']
                ls_results = step_data.get('local_search_results', {})
                
                # Build basin distribution (edges_hash -> count)
                basin_distribution = ls_results.get('basin_counts', {})
                basin_data = ls_results.get('basin_data', {})
                
                # Build final_solutions list (all unique basins reached)
                final_solutions = []
                for basin_hash, basin_info in basin_data.items():
                    final_solutions.append({
                        'edges': basin_info['edges'],
                        'edges_hash': basin_hash,
                        'solution_flat': basin_info['solution_flat'],
                        'cost': sum(basin_info['costs']) / len(basin_info['costs']) if basin_info['costs'] else 0.0,
                        'count': basin_distribution.get(basin_hash, 0)
                    })
                
                # Create training_data entry
                training_entry = {
                    'run_id': stats.get('run_id') or first_occurrence.get('run_id'),
                    'trial_id': first_occurrence.get('trial_id'),
                    'global_iter': first_occurrence.get('global_iter'),
                    'local_iter': first_occurrence.get('local_iter'),
                    'optimum_id': optimum_id,
                    'perturbation_step': step,
                    'initial_solution': {
                        'edges': original_edges,
                        'solution_flat': original_solution_flat,
                        'cost': original_cost,
                        'edges_hash': original_edges_hash,
                        'gap_to_hgs': _calculate_gap(original_cost, hgs_cost)
                    },
                    'perturbed_solution': {
                        'edges': perturbed_edges,
                        'edges_hash': perturbed_edges_hash,
                        'cost': perturbed_cost,
                        'feasible': step_data.get('feasible', False)
                    },
                    'final_solutions': final_solutions,
                    'basin_distribution': basin_distribution,
                    'num_unique_basins': ls_results.get('num_unique_basins', 0),
                    'return_to_original': ls_results.get('return_to_original', 0),
                    'return_to_original_ratio': ls_results.get('return_to_original_ratio', 0.0),
                    'num_runs': len(ls_results.get('final_costs', [])),
                    'mean_final_cost': ls_results.get('mean_final_cost', 0.0)
                }
                
                # Write as JSONL line
                f.write(json.dumps(training_entry, ensure_ascii=False) + '\n')
    
    print(f"  Results saved to JSONL: {output_path}")


def save_results_to_excel(results: List[dict], optima_stats: Dict[str, dict], last_solutions: Dict[tuple, dict], output_path: str):
    """
    Save results to Excel file: each row is an original basin, with columns for each perturbation step.
    Records hash, cost, and return probabilities for each step.
    """
    rows = []
    
    for result in results:
        if result.get('error'):
            continue
        
        optimum_id = result['optimum_id']
        original = result['original']
        original_edges_hash = original.get('edges_hash')
        # Use edges_hash as key (globally unique), not optimum_id (only unique within each run)
        stats = optima_stats.get(original_edges_hash, {}) if original_edges_hash else {}
        
        # Get all occurrences from stats
        occurrence_ids = stats.get('occurrence_ids', [])
        occurrence_ids_str = '; '.join([occ['occurrence_id'] for occ in occurrence_ids])
        
        # For trial_id, global_iter, local_iter: use the first occurrence if available
        first_occurrence = occurrence_ids[0] if occurrence_ids else None
        
        # Get run_id from stats (which may filter by original run_id) or original
        run_id = stats.get('run_id') or original.get('run_id')
        
        # Base row with original basin info
        row = {
            'optimum_id': optimum_id,
            'run_id': run_id,
            'frequency': stats.get('frequency', 0),
            'trial_id': first_occurrence.get('trial_id') if first_occurrence else None,
            'global_iter': first_occurrence.get('global_iter') if first_occurrence else None,
            'local_iter': first_occurrence.get('local_iter') if first_occurrence else None,
            'occurrence_ids': occurrence_ids_str,
            'operator_type': result.get('operator_type', 'unknown'),
            'operator_k': result.get('operator_k', 1),
            'original_cost': original.get('final_cost'),
            'original_edges_hash': original.get('edges_hash'),
        }
        
        # Get original edges set for Jaccard distance calculation
        original_edges_set = edges_to_normalized_set(original.get('edges', []))
        
        # Add columns for each perturbation step (1, 2, ..., k)
        perturbed_steps = result.get('perturbed_steps', [])
        max_k = result.get('operator_k', len(perturbed_steps))
        
        for step in range(1, max_k + 1):
            step_data = next((s for s in perturbed_steps if s['step'] == step), None)
            
            if step_data:
                ls_results = step_data.get('local_search_results', {})
                basin_distribution = ls_results.get('basin_counts', {})
                original_hash = original.get('edges_hash')
                
                # Count returns to original basin
                return_to_original_count = basin_distribution.get(original_hash, 0)
                total_runs = ls_results.get('num_runs', 0)
                return_to_original_prob = return_to_original_count / total_runs if total_runs > 0 else 0.0
                
                # Get other basins (excluding original) as a list
                other_basins_list = []
                for hash_val, count in basin_distribution.items():
                    if hash_val != original_hash:
                        prob = count / total_runs if total_runs > 0 else 0.0
                        other_basins_list.append({
                            'hash': hash_val,
                            'count': count,
                            'prob': prob
                        })
                # Sort by count (descending)
                other_basins_list.sort(key=lambda x: x['count'], reverse=True)
                
                # Get Jaccard distance and Broken Pairs Distance from step_data (already calculated)
                # If not present, calculate Jaccard distance (for backward compatibility)
                jaccard_distance = step_data.get('jaccard_distance')
                if jaccard_distance is None:
                    # Fallback: calculate Jaccard distance if not in step_data
                    perturbed_edges = step_data.get('edges', [])
                    perturbed_edges_set = edges_to_normalized_set(perturbed_edges)
                    if len(original_edges_set) == 0 and len(perturbed_edges_set) == 0:
                        jaccard_distance = 0.0
                    else:
                        intersection = len(original_edges_set & perturbed_edges_set)
                        union = len(original_edges_set | perturbed_edges_set)
                        if union == 0:
                            jaccard_distance = 1.0
                        else:
                            jaccard_sim = intersection / union
                            jaccard_distance = 1.0 - jaccard_sim
                
                row[f'step_{step}_hash'] = step_data.get('edges_hash')
                row[f'step_{step}_cost'] = step_data.get('cost')
                row[f'step_{step}_feasible'] = step_data.get('feasible', False)
                row[f'step_{step}_jaccard_distance'] = jaccard_distance
                row[f'step_{step}_broken_pairs_count'] = step_data.get('broken_pairs_count')
                row[f'step_{step}_broken_pairs_total'] = step_data.get('broken_pairs_total')
                row[f'step_{step}_broken_pairs_ratio'] = step_data.get('broken_pairs_ratio')
                row[f'step_{step}_return_to_original_count'] = return_to_original_count
                row[f'step_{step}_return_to_original_prob'] = return_to_original_prob
                # Store other basins as a list of dicts (JSON string for Excel compatibility)
                row[f'step_{step}_other_basins'] = json.dumps(other_basins_list, ensure_ascii=False) if other_basins_list else '[]'
                row[f'step_{step}_num_unique_basins'] = ls_results.get('num_unique_basins', 0)
                row[f'step_{step}_mean_final_cost'] = ls_results.get('mean_final_cost', 0.0)
            else:
                # No data for this step
                row[f'step_{step}_hash'] = None
                row[f'step_{step}_cost'] = None
                row[f'step_{step}_feasible'] = False
                row[f'step_{step}_jaccard_distance'] = None
                row[f'step_{step}_broken_pairs_count'] = None
                row[f'step_{step}_broken_pairs_total'] = None
                row[f'step_{step}_broken_pairs_ratio'] = None
                row[f'step_{step}_return_to_original_count'] = 0
                row[f'step_{step}_return_to_original_prob'] = 0.0
                row[f'step_{step}_other_basins'] = '[]'  # Empty list
                row[f'step_{step}_num_unique_basins'] = 0
                row[f'step_{step}_mean_final_cost'] = None
        
        rows.append(row)

    # If there are no new rows, do NOT overwrite an existing Excel during resume.
    # This avoids clobbering a previously completed batch when this run only skipped everything.
    if not rows:
        if os.path.exists(output_path):
            print(f"  No new rows to write; keeping existing Excel: {output_path}")
            return
        # Best-effort: create an empty Excel file if it doesn't exist
        df = pd.DataFrame([])
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Perturbation Results', index=False)
        print(f"  Results saved to Excel (empty): {output_path}")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(['frequency', 'optimum_id'], ascending=[False, True])
    
    # Determine sheet name based on operator type
    operator_type = rows[0].get('operator_type', 'unknown') if rows else 'unknown'
    if operator_type == 'double_bridge':
        sheet_name = 'Double-Bridge Move Results'
    elif operator_type == 'remove_and_insert':
        sheet_name = 'Remove-and-Insert Results'
    else:
        sheet_name = 'Perturbation Results'
    
    # If resuming and the Excel exists, merge (append) new rows and de-duplicate by original_edges_hash.
    if os.path.exists(output_path):
        try:
            existing = pd.read_excel(output_path)
            if 'original_edges_hash' in existing.columns:
                combined = pd.concat([existing, df], ignore_index=True)
                combined = combined.drop_duplicates(subset=['original_edges_hash'], keep='first')
                df = combined
        except Exception as e:
            print(f"  Warning: Failed to merge with existing Excel ({output_path}): {e}")

    # Save to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"  Results saved to Excel: {output_path}")


def visualize_perturbation_results(excel_path: str, output_dir: str = None):
    """
    Visualize perturbation results for step=1..k:
    - Distribution of Jaccard similarity (1 - jaccard_distance) per step
    - Distribution of cost changes (step_cost - original_cost) per step
    Saved as a single figure with 2 rows and k columns.
    """
    print("\nGenerating visualization (per-step distributions)...")

    df = pd.read_excel(excel_path)

    # Determine k from columns: step_{i}_cost / step_{i}_jaccard_distance
    step_cost_cols = [c for c in df.columns if c.startswith("step_") and c.endswith("_cost")]
    step_ids = []
    for c in step_cost_cols:
        # c = "step_{i}_cost"
        try:
            step_ids.append(int(c.split("_")[1]))
        except Exception:
            continue
    k = max(step_ids) if step_ids else int(df.get("operator_k", pd.Series([1])).max())
    k = max(1, int(k))

    # Output dir
    if output_dir is None:
        out_dir = Path(excel_path).parent
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure: 2 rows × k cols
    fig_w = max(6, 4 * k)
    fig, axes = plt.subplots(2, k, figsize=(fig_w, 9), squeeze=False)
    fig.suptitle('Perturbation Analysis (Per Step): Jaccard Similarity and Cost Changes', fontsize=16, fontweight='bold')

    original_cost = df["original_cost"]

    for step in range(1, k + 1):
        # --- Jaccard similarity ---
        jd_col = f"step_{step}_jaccard_distance"
        ax_j = axes[0, step - 1]
        if jd_col in df.columns:
            jaccard_dist = df[jd_col].dropna()
            if len(jaccard_dist) > 0:
                jaccard_sim = 1 - jaccard_dist
                ax_j.hist(jaccard_sim, bins=50, edgecolor="black", alpha=0.7, color="skyblue")
                ax_j.axvline(jaccard_sim.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {jaccard_sim.mean():.4f}")
                ax_j.axvline(jaccard_sim.median(), color="green", linestyle="--", linewidth=2, label=f"Median: {jaccard_sim.median():.4f}")
                ax_j.legend(fontsize=9)
        ax_j.set_title(f"Step {step}: Jaccard Similarity", fontsize=12, fontweight="bold")
        ax_j.set_xlabel("Jaccard Similarity (1 - distance)")
        ax_j.set_ylabel("Frequency")
        ax_j.grid(True, alpha=0.3)

        # --- Cost changes ---
        cost_col = f"step_{step}_cost"
        ax_c = axes[1, step - 1]
        if cost_col in df.columns:
            step_cost = df[cost_col]
            cost_change = (step_cost - original_cost).dropna()
            if len(cost_change) > 0:
                ax_c.hist(cost_change, bins=50, edgecolor="black", alpha=0.7, color="lightcoral")
                ax_c.axvline(cost_change.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {cost_change.mean():.2f}")
                ax_c.axvline(cost_change.median(), color="green", linestyle="--", linewidth=2, label=f"Median: {cost_change.median():.2f}")
                ax_c.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
                ax_c.legend(fontsize=9)
        ax_c.set_title(f"Step {step}: Cost Change", fontsize=12, fontweight="bold")
        ax_c.set_xlabel("Cost Change (step_cost - original_cost)")
        ax_c.set_ylabel("Frequency")
        ax_c.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path = out_dir / "perturbation_steps_analysis.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Visualization saved to: {out_path}")


def analyze_double_bridge_move(instance_path: str, instance_index: int = 0,
                                basin_base_dir: str = "basin_datasets0",
                                output_base_dir: str = "basin_datasets0_analyze",
                                hgs_solution_path: str = None,
                                output_path: str = None, seed: int = None,
                                operator_type: str = 'double_bridge',
                                k: int = 1,
                                require_feasible: bool = False,
                                num_local_search_runs: int = 30,
                                start_idx: int = 0,
                                max_optima: int = None,
                                batch_id: str = None,
                                visualize: bool = False):
    """
    Analyze perturbation using Double-Bridge Move or Remove-and-Insert
    
    Args:
        instance_path: Path to instance pkl file
        instance_index: Instance index (default: 0)
        basin_base_dir: Base directory for basin datasets
        hgs_solution_path: Path to HGS solution pkl file (optional)
        output_path: Path to output JSON file, auto-generated if None
        seed: Random seed (for reproducibility)
        operator_type: 'double_bridge' or 'remove_and_insert'
        k: Number of times to apply the operator
        require_feasible: If True, only return feasible solutions; If False, allow infeasible
    """
    # Get instance filename from path
    instance_filename = os.path.basename(instance_path)
    instance_id = f"{instance_filename}#{instance_index}"
    
    # Get basin paths
    if not os.path.exists(instance_path):
        raise FileNotFoundError(f"Instance file not found: {instance_path}")
    
    # Input basin dir (contains optima.jsonl / trajectory.jsonl)
    basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
    basin_dir = basin_paths['basin_dir']
    jsonl_path = os.path.join(basin_dir, 'optima.jsonl')
    trials_path = os.path.join(basin_dir, 'trials.jsonl')
    trajectory_path = os.path.join(basin_dir, 'trajectory.jsonl')

    # Output dir (store results here)
    instance_id = basin_paths['instance_id']
    output_dir = os.path.join(
        output_base_dir if os.path.isabs(output_base_dir) else os.path.join(os.path.dirname(os.path.abspath(__file__)), output_base_dir),
        instance_id
    )
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Instance ID: {instance_id}")
    print(f"Instance path: {instance_path}")
    print(f"Input basin directory: {basin_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load instance data for capacity check
    print(f"\nLoading instance data...")
    with open(instance_path, 'rb') as f:
        instances = pickle.load(f)
    if instance_index >= len(instances):
        raise ValueError(f"Instance index {instance_index} out of range (max: {len(instances)-1})")
    
    instance_tuple = instances[instance_index]
    depot_coords = instance_tuple[0]  # Depot coordinates (in [0, 1] range)
    customer_coords = instance_tuple[1]  # Customer coordinates (in [0, 1] range)
    demands_array = np.array(instance_tuple[2])  # Customer demands
    vehicle_capacity = float(instance_tuple[3])  # Vehicle capacity
    
    # Combine coordinates: [depot, customer1, customer2, ...]
    # coordinates[0] is depot, coordinates[1..n] are customers
    # Note: coordinates are in [0, 1] range, will be scaled to [0, 100] when computing cost
    coordinates = [depot_coords[0]] + customer_coords  # depot + customers
    
    # Create demands dict: customer node -> demand (customers are 1..n, demands array is 0..n-1)
    demands = {i + 1: float(demands_array[i]) for i in range(len(demands_array))}
    
    print(f"  Vehicle capacity: {vehicle_capacity}")
    print(f"  Number of customers: {len(demands)}")
    
    # Load optima data
    print(f"\nLoading optima data: {jsonl_path}")
    optima = load_optima_from_jsonl(jsonl_path)
    
    # Get HGS cost for gap calculation (initialize to None)
    hgs_cost = None
    
    # Load HGS solution and add it as a local optimum
    if hgs_solution_path and os.path.exists(hgs_solution_path):
        print(f"\nLoading HGS solution: {hgs_solution_path}")
        with open(hgs_solution_path, 'rb') as f:
            hgs_solutions = pickle.load(f)
        if instance_index >= len(hgs_solutions):
            print(f"  Warning: HGS solution index {instance_index} out of range, skipping HGS solution")
        else:
            hgs_solution_tuple = hgs_solutions[instance_index]
            hgs_cost_raw = float(hgs_solution_tuple[0])
            # Scale HGS cost by 100 (coordinates are in [0, 1] range, scaled to [0, 100])
            hgs_cost = hgs_cost_raw * 100.0
            hgs_routes_raw = hgs_solution_tuple[1]
            
            # Convert HGS routes to standard format (list of [0, ...nodes..., 0])
            hgs_routes = _convert_hgs_routes(hgs_routes_raw)
            if hgs_routes is None:
                print(f"  Warning: Failed to convert HGS routes, skipping HGS solution")
            else:
                # Convert HGS routes to edges format
                hgs_edges_set = routes_to_edges_set(hgs_routes)
                hgs_edges = [[int(u), int(v)] for (u, v) in sorted(hgs_edges_set)]
                hgs_edges_hash = edges_hash_func(hgs_edges_set)
                
                # Infer num_orders from HGS edges
                max_node_id = max(max(e) for e in hgs_edges) if hgs_edges else 0
                hgs_num_orders = max_node_id + 1
                
                # Create HGS optimum entry
                hgs_optimum = {
                    'instance_id': instance_id,
                    'run_id': 'HGS',
                    'optimum_id': -1,  # Special ID for HGS solution
                    'final_cost': hgs_cost,
                    'num_routes': len(hgs_routes),
                    'num_orders': hgs_num_orders,
                    'edges': hgs_edges,
                    'edges_hash': hgs_edges_hash,
                    'edge_diff_to_hgs': 0,
                    'cost_gap_to_hgs_pct': 0.0
                }
                
                # Add HGS solution as first optimum
                optima.insert(0, hgs_optimum)
                print(f"  Added HGS solution (cost: {hgs_cost:.2f}, {len(hgs_routes)} routes)")
                print(f"  HGS routes (first 3): {hgs_routes[:3]}")
                if len(hgs_routes) > 3:
                    print(f"  ... (total {len(hgs_routes)} routes)")
    
    if not optima:
        print("  Error: No optima data found")
        return
    
    print(f"  Found {len(optima)} local optima (including HGS if provided)")
    
    # Deduplicate optima by edges_hash before perturbation
    print(f"\nDeduplicating optima by edges_hash...")
    seen_hashes = {}
    unique_optima = []
    duplicate_count = 0
    
    for opt in optima:
        opt_edges_hash = opt.get('edges_hash')
        if opt_edges_hash is None:
            # Keep optima without edges_hash (e.g., HGS)
            unique_optima.append(opt)
        elif opt_edges_hash not in seen_hashes:
            seen_hashes[opt_edges_hash] = opt
            unique_optima.append(opt)
        else:
            duplicate_count += 1
    
    optima = unique_optima
    if duplicate_count > 0:
        print(f"  After deduplication: {len(optima)} unique optima (removed {duplicate_count} duplicates)")
    else:
        print(f"  After deduplication: {len(optima)} unique optima (no duplicates)")

    # Slice optima for batching (by index after dedup)
    total_unique_optima = len(optima)
    if start_idx < 0:
        start_idx = 0
    if start_idx >= total_unique_optima:
        print(f"  Warning: start_idx={start_idx} out of range (total_unique_optima={total_unique_optima}), nothing to do.")
        return []
    end_idx = total_unique_optima if max_optima is None else min(total_unique_optima, start_idx + max_optima)
    optima = optima[start_idx:end_idx]
    print(f"  Batch slice: optima[{start_idx}:{end_idx}] (batch_size={len(optima)}, total_unique_optima={total_unique_optima})")
    
    # Print one local optima example (skip HGS if it's the first one)
    if optima and len(optima) > 1:
        example_opt = optima[1] if optima[0].get('optimum_id') == -1 else optima[0]
        example_edges = example_opt.get('edges', [])
        if example_edges:
            # Convert edges to routes for display
            example_num_orders = example_opt.get('num_orders', None)
            if example_num_orders is None:
                max_node_id = max(max(e) for e in example_edges) if example_edges else 0
                example_num_orders = max_node_id + 1
            example_routes = edges_to_routes(example_edges, example_num_orders)
            print(f"\n  Example local optima (optimum_id={example_opt.get('optimum_id')}):")
            print(f"    Cost: {example_opt.get('final_cost', 'N/A')}")
            print(f"    Routes (first 3): {example_routes[:3]}")
            if len(example_routes) > 3:
                print(f"    ... (total {len(example_routes)} routes)")
    
    # Load last solutions per trial from trajectory (these are the local optima)
    print(f"\nLoading last solutions per trial from trajectory...")
    last_solutions = load_last_solutions_per_trial(str(trajectory_path))
    trajectory_info = load_trajectory_info(str(trajectory_path))
    
    print(f"  Found {len(last_solutions)} trials (last solutions)")
    print(f"  Loaded trajectory info for {len(trajectory_info)} unique edges_hash")
    
    # Compute frequency and occurrences (for the sliced optima)
    # Frequency = number of trials whose last solution is this local optima (same as count in optima.jsonl)
    print(f"  Computing frequency and occurrences...")
    optima_stats = compute_optima_frequency_and_occurrences(optima, trajectory_info)
    
    # Reassign optimum_id based on frequency (high frequency -> small ID)
    # This ensures globally unique optimum_id across all runs
    print(f"  Reassigning optimum_id based on frequency...")
    optima_with_freq = []
    for opt in optima:
        edges_hash = opt.get('edges_hash')
        if edges_hash:
            stats = optima_stats.get(edges_hash, {})
            frequency = stats.get('frequency', 0)
            optima_with_freq.append((opt, frequency, edges_hash))
        else:
            # For optima without edges_hash (e.g., HGS), use frequency=0 or a special value
            optima_with_freq.append((opt, 0, None))
    
    # Sort by frequency (descending), then by edges_hash for stability
    optima_with_freq.sort(key=lambda x: (-x[1], x[2] or ''))
    
    # Reassign optimum_id: higher-frequency optima get smaller IDs, starting from 0
    for new_id, (opt, freq, edges_hash) in enumerate(optima_with_freq):
        opt['optimum_id'] = new_id
        # Also update optima_stats to include the new optimum_id
        if edges_hash and edges_hash in optima_stats:
            optima_stats[edges_hash]['optimum_id'] = new_id
    
    # Reconstruct optima list in the new order
    optima = [opt for opt, _, _ in optima_with_freq]
    print(f"  Reassigned {len(optima)} optimum_id (sorted by frequency, high->low)")
    
    # Determine output file paths (consistent with save_results logic)
    if output_path is None:
        batch_tag = ""
        if batch_id is not None:
            batch_tag = f".batch_{batch_id}"
        else:
            # Stable tag from slice range
            batch_tag = f".idx_{start_idx}_{end_idx}"
        jsonl_output_path = os.path.join(output_dir, f"{operator_type}_training_data{batch_tag}_r{num_local_search_runs}.jsonl")
        excel_output_path = os.path.join(output_dir, f"{operator_type}_results{batch_tag}_r{num_local_search_runs}.xlsx")
    else:
        jsonl_output_path = str(Path(output_path).with_suffix('.jsonl'))
        excel_output_path = str(Path(output_path).with_suffix('.xlsx'))
    
    # Resume functionality (JSONL is the source of truth):
    # Only recognize completion based on training_data JSONL.
    # For batch processing, scan ALL.jsonl and all batch JSONL files in output_dir (global resume).
    processed_hashes = set()
    import glob

    # Check ALL.jsonl first (merged file, highest priority)
    all_jsonl_path = os.path.join(output_dir, f"{operator_type}_training_data.ALL_r{num_local_search_runs}.jsonl")
    if os.path.exists(all_jsonl_path):
        processed_hashes.update(load_processed_hashes_from_jsonl(all_jsonl_path))

    # Current batch JSONL
    if os.path.exists(jsonl_output_path):
        processed_hashes.update(load_processed_hashes_from_jsonl(jsonl_output_path))

    # Other batch JSONLs (same operator_type and runs)
    batch_pattern_jsonl = os.path.join(output_dir, f"{operator_type}_training_data.batch_*_r{num_local_search_runs}.jsonl")
    for jf in glob.glob(batch_pattern_jsonl):
        if jf != jsonl_output_path:
            processed_hashes.update(load_processed_hashes_from_jsonl(jf))

    if processed_hashes:
        print(f"  Resuming: {len(processed_hashes)} optima already processed (JSONL-only, global across batches)")
    
    # Use NumPy vectorized processing (with capacity check and local search)
    print(f"\nStarting vectorized processing...")
    
    # Open JSONL file for streaming writes (append mode if resuming)
    jsonl_mode = 'a' if processed_hashes else 'w'
    with open(jsonl_output_path, jsonl_mode, encoding='utf-8') as jsonl_file:
        results = process_optima_vectorized(
            optima, seed, demands, vehicle_capacity,
            operator_type=operator_type, k=k, require_feasible=require_feasible,
            coordinates=coordinates,
            instance_path=instance_path, instance_index=instance_index,
            num_local_search_runs=num_local_search_runs,
            jsonl_file=jsonl_file, optima_stats=optima_stats,
            hgs_cost=hgs_cost, processed_hashes=processed_hashes,
        )
    
    print(f"  Results written to JSONL (streaming): {jsonl_output_path}")

    # Print return-to-original distribution (per step) for quick sanity check
    try:
        print("\nReturn-to-original distribution (per step):")
        step_max = int(k)
        for step in range(1, step_max + 1):
            counts = []
            probs = []
            for r in results:
                if r.get('error') is not None:
                    continue
                step_data = next((s for s in (r.get('perturbed_steps') or []) if s.get('step') == step), None)
                if not step_data:
                    continue
                ls = step_data.get('local_search_results') or {}
                c = ls.get('return_to_original')
                p = ls.get('return_to_original_ratio')
                if c is not None:
                    counts.append(int(c))
                if p is not None:
                    probs.append(float(p))

            if not probs:
                print(f"  Step {step}: (no data)")
                continue

            probs_arr = np.array(probs, dtype=np.float64)
            counts_arr = np.array(counts, dtype=np.int64) if counts else None

            # Simple probability histogram buckets
            bins = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0 + 1e-9]
            hist = np.histogram(probs_arr, bins=bins)[0]
            bucket_labels = ["[0,0.1)", "[0.1,0.25)", "[0.25,0.5)", "[0.5,0.75)", "[0.75,0.9)", "[0.9,1.0]"]
            bucket_str = ", ".join(f"{lbl}:{int(n)}" for lbl, n in zip(bucket_labels, hist))

            if counts_arr is not None and len(counts_arr) == len(probs_arr):
                print(
                    f"  Step {step}: n={len(probs_arr)} | "
                    f"return_count mean={counts_arr.mean():.2f} median={np.median(counts_arr):.0f} | "
                    f"return_prob mean={probs_arr.mean():.3f} median={np.median(probs_arr):.3f} "
                    f"p10={np.quantile(probs_arr, 0.10):.3f} p90={np.quantile(probs_arr, 0.90):.3f} | "
                    f"bins {bucket_str}"
                )
            else:
                print(
                    f"  Step {step}: n={len(probs_arr)} | "
                    f"return_prob mean={probs_arr.mean():.3f} median={np.median(probs_arr):.3f} "
                    f"p10={np.quantile(probs_arr, 0.10):.3f} p90={np.quantile(probs_arr, 0.90):.3f} | "
                    f"bins {bucket_str}"
                )
    except Exception as e:
        print(f"  Warning: Failed to print return-to-original distribution: {e}")
    
    # Statistics
    successful = [r for r in results if r.get('error') is None]
    failed = [r for r in results if r.get('error') is not None]
    
    print(f"\nProcessing completed:")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailure examples:")
        for f in failed[:5]:
            print(f"    Optimum {f.get('optimum_id')}: {f.get('error')}")
    
    # JSONL is already written via streaming in process_optima_vectorized
    # Only generate Excel summary (from results, which may be partial if resuming)
    # Note: jsonl_output_path and excel_output_path are already defined above
    print(f"\nGenerating Excel summary from JSONL (source of truth)...")
    save_excel_from_training_jsonl(jsonl_output_path, excel_output_path, optima_stats=optima_stats)

    # Visualization (per-step) - usually run once after merging batches
    if visualize:
        visualize_perturbation_results(str(excel_output_path), output_dir)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply perturbation operators (Double-Bridge Move or Remove-and-Insert) to local optima and compute Jaccard distance')
    parser.add_argument('--instance_path', '--pkl', type=str, default="/home/jieyi/cvrp100_uniform.pkl", dest='instance_path', help='Path to instance pkl file')
    parser.add_argument('--instance_index', '--idx', type=int, default=0, dest='instance_index', help='Instance index (default: 0)')
    parser.add_argument('--basin_base_dir', type=str, default='basin_datasets0',
                        help='Input basin directory containing optima/trajectory (default: basin_datasets0)')
    parser.add_argument('--output_base_dir', type=str, default='basin_datasets0_analyze',
                        help='Output directory for results (default: basin_datasets0_analyze)')
    parser.add_argument('--hgs_solution_path', type=str, default="/home/jieyi/hgs_cvrp100_uniform.pkl", help='Path to HGS solution pkl file (optional, will add HGS solution as a local optimum)')
    parser.add_argument('--output_path', type=str, default=None, help='Path to output JSON file (default: operator_results.json in basin directory)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (for reproducibility)')
    parser.add_argument('--operator_type', type=str, default='double_bridge', choices=['double_bridge', 'remove_and_insert'],
                        help='Perturbation operator type: double_bridge or remove_and_insert (default: double_bridge)')
    parser.add_argument('--k', type=int, default=1, help='Number of times to apply the operator (default: 1)')
    parser.add_argument('--require_feasible', action='store_true', 
                        help='If set, only return feasible solutions; otherwise allow infeasible solutions')
    parser.add_argument('--num_local_search_runs', type=int, default=30,
                        help='Number of local search runs per perturbed solution step (default: 30)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index (after dedup) of optima to process for batching (default: 0)')
    parser.add_argument('--max_optima', type=int, default=None,
                        help='Maximum number of optima to process in this run (default: all remaining)')
    parser.add_argument('--batch_id', type=str, default=None,
                        help='Optional batch identifier appended to output filenames (default: idx_{start}_{end})')
    parser.add_argument('--visualize', action='store_true',
                        help='If set, generate visualization PNG for this batch (default: off; run after merging)')
    
    args = parser.parse_args()
    
    analyze_double_bridge_move(
        instance_path=args.instance_path,
        instance_index=args.instance_index,
        basin_base_dir=args.basin_base_dir,
        output_base_dir=args.output_base_dir,
        hgs_solution_path=args.hgs_solution_path,
        output_path=args.output_path,
        seed=args.seed,
        operator_type=args.operator_type,
        k=args.k,
        require_feasible=args.require_feasible,
        num_local_search_runs=args.num_local_search_runs,
        start_idx=args.start_idx,
        max_optima=args.max_optima,
        batch_id=args.batch_id,
        visualize=args.visualize,
    )


if __name__ == '__main__':
    main()
