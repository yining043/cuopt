"""
Utility functions for solution format conversion, distance calculation, data loading, and visualization.
These functions are used by test_landscape.py for data generation.
"""

import numpy as np
import pickle
import cudf
import os
import matplotlib.pyplot as plt
import datetime
import wandb


# ============================================================================
# Solution Validation
# ============================================================================

def validate_cuopt_solution(solution_flat, num_orders, context=""):
    """
    Validate that a cuOpt solution_flat contains all expected customers.
    cuOpt format: customer IDs are 1 to num_orders, depot IDs are > num_orders
    
    Args:
        solution_flat: cuOpt solution_flat array
        num_orders: Number of customer orders
        context: Context string for error messages
    
    Raises:
        ValueError if solution is invalid
    """
    if solution_flat is None or len(solution_flat) == 0:
        raise ValueError(f"Empty solution{context}")
    
    customers = [n for n in solution_flat if 1 <= n <= num_orders]
    expected_customers = set(range(1, num_orders + 1))
    actual_customers = set(customers)
    
    if len(actual_customers) != num_orders or actual_customers != expected_customers:
        missing = sorted(expected_customers - actual_customers)
        extra = sorted(actual_customers - expected_customers)
        raise ValueError(f"Invalid solution{context}: expected {num_orders} customers, "
                        f"got {len(actual_customers)}. Missing: {missing}, Extra: {extra}")


# ============================================================================
# Solution Format Conversion
# ============================================================================

def infer_num_routes_from_solution_flat(solution_flat, num_orders):
    """
    Infer the actual number of routes from cuOpt solution_flat by counting dummy depot sequences.
    cuOpt format: customer IDs are 1 to num_orders, depot IDs are > num_orders (4 consecutive dummies per route)
    
    Args:
        solution_flat: cuOpt solution_flat array
        num_orders: Number of customer orders
    
    Returns:
        Inferred number of routes
    """
    if solution_flat is None or len(solution_flat) == 0:
        return 0
    
    # Convert to numpy array if needed and ensure it's 1D
    if not isinstance(solution_flat, np.ndarray):
        solution_flat = np.array(solution_flat, dtype=np.int32)
    solution_flat = np.asarray(solution_flat).flatten()
    
    # Count sequences of 4 consecutive dummy nodes (> num_orders)
    num_routes = 0
    idx = 0
    while idx < len(solution_flat):
        # Check if we have 4 consecutive dummy nodes (depot pattern)
        if idx + 3 < len(solution_flat):
            # Use item() to get scalar values for comparison
            if all(int(solution_flat[idx + i].item()) > num_orders for i in range(4)):
                num_routes += 1
                idx += 4
                # Skip customer nodes (1 to num_orders) until next dummy
                while idx < len(solution_flat) and 1 <= int(solution_flat[idx].item()) <= num_orders:
                    idx += 1
            else:
                idx += 1
        else:
            break
    
    return num_routes


def solution_flat_to_routes(solution_flat, num_routes, num_orders):
    """
    Convert cuOpt solution_flat to list of routes.
    cuOpt format: customer IDs are 1 to num_orders, depot IDs are > num_orders (4 consecutive dummies per route)
    
    Args:
        solution_flat: cuOpt solution_flat array
        num_routes: Number of routes
        num_orders: Number of customer orders
    
    Returns:
        List of routes, each route is [0, node1, node2, ..., 0] (standard format)
    """
    if num_routes is None:
        num_routes = 0
    if num_orders is None:
        num_orders = 0
    
    # Convert to numpy array if needed and ensure it's 1D
    if not isinstance(solution_flat, np.ndarray):
        solution_flat = np.array(solution_flat, dtype=np.int32)
    solution_flat = np.asarray(solution_flat).flatten()
    
    routes = []
    idx = 0
    
    # Helper function to get scalar value from array element
    def get_scalar(arr, idx):
        val = arr[idx]
        if hasattr(val, 'item'):
            return int(val.item())
        return int(val)
    
    for route_id in range(num_routes):
        route = [0]  # Start with depot
        
        # Skip 4 dummy depot nodes (> num_orders)
        if idx + 4 <= len(solution_flat):
            # Check if these are dummy nodes (all > num_orders)
            # Use get_scalar to get scalar values for comparison
            if all(get_scalar(solution_flat, idx + i) > num_orders for i in range(4)):
                idx += 4
            else:
                idx += 4  # Still skip even if pattern is wrong
        else:
            # Not enough nodes, create empty route
            route.append(0)
            routes.append(route)
            continue
        
        # Collect customer nodes (1 to num_orders) until next route's 4 dummies or end
        while idx < len(solution_flat):
            node_id = get_scalar(solution_flat, idx)  # Get scalar value
            
            # Stop if we hit next route's 4 dummy nodes
            if idx + 3 < len(solution_flat):
                if all(get_scalar(solution_flat, idx + i) > num_orders for i in range(4)):
                    # This is the start of next route, stop here
                    break
            
            # Add customer nodes (1 to num_orders)
            if 1 <= node_id <= num_orders:
                route.append(node_id)
            # Skip dummy nodes (> num_orders) and 0s
            
            idx += 1
        
        # Add return depot
        route.append(0)
        routes.append(route)
    
    # Ensure we have exactly num_routes routes
    while len(routes) < num_routes:
        routes.append([0, 0])
    
    return routes

def routes_to_solution_flat(routes, num_routes, num_orders):
    """
    Convert routes back to solution_flat format.
    
    Args:
        routes: List of routes, each route is [0, node1, node2, ..., 0]
        num_routes: Number of routes
        num_orders: Number of customer orders
    
    Returns:
        solution_flat: Solution in flat format
    """
    solution_flat = []
    for route_id in range(num_routes):
        # Add 4 dummy depot nodes at start of each route
        solution_flat.extend([num_orders + 1, num_orders + 2, num_orders + 3, num_orders + 4])
        
        if route_id < len(routes):
            route = routes[route_id]
            # Add customer nodes (skip depot at start and end)
            for node in route[1:-1]:
                solution_flat.append(node)
    
    return np.array(solution_flat, dtype=np.int32)


def cuopt_to_standard(solution_flat, num_routes, num_orders):
    """
    Convert cuOpt solution format to standard format.
    
    cuOpt format: customer IDs are 1 to num_orders, depot IDs are > num_orders (4 consecutive dummies per route)
    Standard format: 0 is depot, 1 to num_orders are customers, format like [0,1,2,3,0,4,5,6,0,...]
    
    Args:
        solution_flat: cuOpt solution_flat array
        num_routes: Number of routes
        num_orders: Number of customer orders
    
    Returns:
        Standard format solution: flat array with 0 as depot separator
    """
    routes = solution_flat_to_routes(solution_flat, num_routes, num_orders)
    # Flatten routes: [0,1,2,3,0], [0,4,5,6,0] -> [0,1,2,3,0,4,5,6,0]
    standard = []
    cnt = 0
    for route in routes:
        standard.extend(route[1:]) if cnt >=1 else standard.extend(route)
        cnt += 1
    return np.array(standard, dtype=np.int32)


def standard_solution_to_routes(standard_solution_flat):
    """
    Parse standard format solution into routes.
    Standard format: 0 is depot separator, e.g., [0,1,2,3,0,4,5,6,0]
    
    Args:
        standard_solution_flat: Standard format solution (flat array with 0 as depot separator)
    
    Returns:
        List of routes, each route is [0, node1, node2, ..., 0] (with Python int, not np.int32)
    """
    if not isinstance(standard_solution_flat, np.ndarray):
        standard_solution_flat = np.array(standard_solution_flat, dtype=np.int32)
    standard_solution_flat = np.asarray(standard_solution_flat).flatten()
    
    routes = []
    current_route = [0]  # Start with depot
    
    for node in standard_solution_flat:
        # Convert numpy scalar to Python int
        node_val = int(node.item() if hasattr(node, 'item') else node)
        if node_val == 0:
            if len(current_route) > 1:  # Route has at least [0, ...]
                current_route.append(0)  # Close the route
                routes.append(current_route)
            current_route = [0]  # Start new route
        else:
            current_route.append(node_val)
    
    # Handle last route if it doesn't end with 0
    if len(current_route) > 1:
        if current_route[-1] != 0:
            current_route.append(0)
        routes.append(current_route)
    
    return routes


def standard_to_cuopt(standard_solution, num_routes, num_orders):
    """
    Convert standard format to cuOpt solution format.
    
    Standard format: 0 is depot, 1 to num_orders are customers, format like [0,1,2,3,0,4,5,6,0,...]
    cuOpt format: customer IDs are 1 to num_orders, depot IDs are > num_orders (4 consecutive dummies per route)
    
    Args:
        standard_solution: Standard format solution (flat array with 0 as depot separator)
        num_routes: Number of routes
        num_orders: Number of customer orders
    
    Returns:
        cuOpt solution_flat format
    """
    # Parse standard format into routes
    routes = []
    current_route = []
    for node in standard_solution:
        if node == 0:
            if len(current_route) == 0:
                # Starting a new route
                current_route = [0]
            else:
                # Ending current route
                current_route.append(0)
                if len(current_route) > 2:  # At least [0, 0]
                    routes.append(current_route)
                current_route = []
        else:
            if len(current_route) == 0:
                current_route = [0]
            current_route.append(node)
    
    # Handle last route if it doesn't end with 0
    if len(current_route) > 0:
        if current_route[-1] != 0:
            current_route.append(0)
        if len(current_route) > 2:
            routes.append(current_route)
    
    # Ensure we have exactly num_routes routes
    while len(routes) < num_routes:
        routes.append([0, 0])
    routes = routes[:num_routes]  # Trim if too many
    
    # Convert routes to cuOpt format
    return routes_to_solution_flat(routes, num_routes, num_orders)


def convert_node_sequence_to_routes(node_sequence, num_routes=None, num_orders=None):
    """
    General function: Convert node sequence to routes list.
    Handles both cuopt solution_flat format and HGS solution format.
    
    Args:
        node_sequence: Node sequence (list or numpy array)
        num_routes: Number of routes (for cuopt format, if None assumes HGS format)
        num_orders: Number of customer orders (for cuopt format, if None assumes HGS format)
    
    Returns:
        List of routes, each route is [0, node1, node2, ..., 0]
    """
    # If num_routes and num_orders are provided, use cuopt format processing
    if num_routes is not None and num_orders is not None:
        return solution_flat_to_routes(node_sequence, num_routes, num_orders)
    
    # Otherwise, use HGS format processing (0 represents depot separator)
    # HGS format: node sequence where 0 represents depot
    # Routes are separated by 0, and each route should start and end with 0 (depot)
    routes = []
    current_route = []
    start_new_route = True
    
    for node in node_sequence:
        if node == 0:
            # Depot encountered
            if start_new_route:
                # Starting a new route, begin with depot
                current_route = [0]
                start_new_route = False
            else:
                # Ending current route, close with depot
                if len(current_route) > 0:
                    current_route.append(0)
                    if len(current_route) > 1:  # Only add non-empty routes
                        routes.append(current_route)
                    current_route = []
                    start_new_route = True
        else:
            # Customer node
            if len(current_route) == 0:
                # Route should start with depot
                current_route = [0]
            current_route.append(node)
            start_new_route = False
    
    # Handle case where last route doesn't end with depot
    if len(current_route) > 0:
        if current_route[0] != 0:
            current_route = [0] + current_route
        if len(current_route) > 1:
            current_route.append(0)
            routes.append(current_route)
    
    # Filter out empty routes (only depot)
    routes = [r for r in routes if len(r) > 1]
    
    return routes


# ============================================================================
# Initial Solutions format conversion
# ============================================================================

def get_initial_solutions(routing_solution, n_initial_sols=1):
    """
    Get initial solutions from routing solution (output of cuopt.routing.Solve).
    """
    initial_sol = routing_solution.get_route()
    sol_offsets = [0]
    vehicle_ids = cudf.Series()
    routes = cudf.Series()
    types = cudf.Series()
    # simply expand the same solution for convenience
    for i in range(0, n_initial_sols):
        vehicle_ids = cudf.concat([vehicle_ids, initial_sol["truck_id"]])
        routes = cudf.concat([routes, initial_sol["route"]])
        types = cudf.concat([types, initial_sol["type"]])
        sol_offsets.append(sol_offsets[i] + initial_sol["route"].shape[0])
    sol_offsets = cudf.Series(sol_offsets)
    return vehicle_ids, routes, types, sol_offsets


def flat_solution_to_initial_solutions(solution_flat):
    """
    Convert standard flat solution format to initial solutions format for add_initial_solutions.
    
    Standard flat format: [0, node1, node2, ..., 0, node3, node4, ..., 0]
    where 0 is depot separator, and nodes are customer IDs (1 to num_orders).
    
    Example: [0, 1, 2, 0, 4, 5, 3, 0] represents:
        Route 0: Depot -> Customer 1 -> Customer 2 -> Depot
        Route 1: Depot -> Customer 4 -> Customer 5 -> Customer 3 -> Depot
    
    Args:
        solution_flat: numpy array or list in standard flat format, e.g., [0, 1, 2, 3, 0, 4, 5, 6, 0]
    
    Returns:
        tuple: (vehicle_ids, routes, types, sol_offsets)
            - vehicle_ids: cudf.Series of vehicle IDs for each node
            - routes: cudf.Series of node IDs (0 for depot, customer IDs otherwise)
            - types: cudf.Series of node types ("Depot", "Delivery", etc.)
            - sol_offsets: cudf.Series of solution offsets
    """
    # Convert to numpy array if needed
    if not isinstance(solution_flat, np.ndarray):
        solution_flat = np.array(solution_flat, dtype=np.int32)
    solution_flat = np.asarray(solution_flat).flatten()
    
    # Parse flat format into routes
    routes_list = []
    current_route = [0]
    for node in solution_flat:
        if node == 0:
            if len(current_route) > 1:  # Route has at least [0, 0]
                current_route.append(0)
                routes_list.append(current_route)
            current_route = [0]
        else:
            current_route.append(node)
    if len(current_route) > 1:
        if current_route[-1] != 0:
            current_route.append(0)
        routes_list.append(current_route)
    
    # Convert routes to vehicle_ids, routes, types format
    vehicle_ids_list, routes_list_flat, types_list = [], [], []
    for vehicle_id, route in enumerate(routes_list):
        for node in route:
            vehicle_ids_list.append(vehicle_id)
            routes_list_flat.append(node)
            types_list.append("Depot" if node == 0 else "Delivery")

    # For add_initial_solutions we currently provide ONE solution,
    # so sol_offsets should contain exactly two elements: [0, total_length]
    total_len = len(routes_list_flat)
    sol_offsets = [0, total_len]
    
    return (cudf.Series(vehicle_ids_list, dtype=np.int32),
            cudf.Series(routes_list_flat, dtype=np.int32),
            cudf.Series(types_list),
            cudf.Series(sol_offsets, dtype=np.int32))

# ============================================================================
# Distance Calculation
# ============================================================================

def extract_edges_from_routes(routes):
    """
    Extract all undirected edges from routes using vectorized operations.
    Each route [0, 1, 2, 3, 0] produces edges: (0,1), (1,2), (2,3), (3,0)
    
    For VRP: Expected edge count = sum(len(route) - 1 for route in routes)
    This equals total nodes (including depot) - number of routes
    
    Args:
        routes: List of routes, each route is [0, node1, node2, ..., 0]
    
    Returns:
        Set of undirected edges, each edge is a tuple (min_node, max_node)
    """
    edges = set()
    
    for route in routes:
        if len(route) < 2:
            continue
        
        # Convert to numpy array for vectorized operations
        route_arr = np.array(route, dtype=np.int32)
        
        # Extract consecutive pairs using slicing: [:-1] and [1:]
        node1_arr = route_arr[:-1]
        node2_arr = route_arr[1:]
        
        # Normalize edges to ensure undirected: always use (min, max)
        # Use numpy's minimum and maximum for vectorized operation
        min_nodes = np.minimum(node1_arr, node2_arr)
        max_nodes = np.maximum(node1_arr, node2_arr)
        
        # Convert to tuples and add to set
        # Convert to Python int to avoid numpy types
        edges.update((int(min_n), int(max_n)) for min_n, max_n in zip(min_nodes, max_nodes))
    
    return edges


def jaccard_distance(set1, set2):
    """
    Calculate Jaccard distance between two sets.
    Jaccard distance = 1 - Jaccard similarity
    
    Args:
        set1: First set
        set2: Second set
    
    Returns:
        Jaccard distance (0 = identical, 1 = completely different)
    """
    if len(set1) == 0 and len(set2) == 0:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 1.0
    
    jaccard_sim = intersection / union
    return 1.0 - jaccard_sim


def solution_distance_matrix(solutions_list, num_routes_list, num_orders):
    """
    Calculate distance matrix between all solutions using Jaccard distance on edge sets.
    Extracts all undirected edges from routes and computes Jaccard distance on edge sets.
    
    Args:
        solutions_list: List of solution_flat arrays
        num_routes_list: List of number of routes for each solution
        num_orders: Number of customer orders
    
    Returns:
        Distance matrix of shape (n_solutions, n_solutions)
    """
    n_solutions = len(solutions_list)
    
    # Convert solutions to route representations and extract edge sets
    edge_sets_list = []
    
    for sol, num_routes in zip(solutions_list, num_routes_list):
        if sol is None or len(sol) == 0:
            edge_sets_list.append(set())
            continue
        
        # Handle None values
        if num_routes is None:
            num_routes = 0
        
        routes = solution_flat_to_routes(sol, num_routes, num_orders)
        
        # Extract all undirected edges from routes
        edges = extract_edges_from_routes(routes)
        edge_sets_list.append(edges)
    
    # Initialize distance matrix
    distance_matrix = np.zeros((n_solutions, n_solutions))
    
    print(f"Computing distance matrix for {n_solutions} solutions...")
    print(f"  Using Jaccard distance on edge sets")
    
    for i in range(n_solutions):
        for j in range(i + 1, n_solutions):
            # Jaccard distance on edge sets
            jaccard_dist = jaccard_distance(edge_sets_list[i], edge_sets_list[j])
            
            distance_matrix[i, j] = jaccard_dist
            distance_matrix[j, i] = jaccard_dist
    
    print(f"Distance matrix computation completed.")
    return distance_matrix


def compute_solution_distances(global_history, num_orders=None):
    """
    Compute distance matrix between all solutions using Jaccard distance.
    
    Args:
        global_history: Dictionary containing history records
        num_orders: Number of customer orders
    
    Returns:
        distance_matrix: Distance matrix of shape (n_solutions, n_solutions)
        all_solutions: List of all solutions in chronological order
        global_iter_list: List of global_iter for each solution in chronological order
        cost_list: List of cost_after for each solution in chronological order
        local_search_id_list: List of local_search_id for each solution in chronological order
    """
    history = global_history['history']
    if not history:
        print("No records to compute distances")
        return None, None, None, None, None
    
    # Collect all solutions in chronological order by global_iter (time sequence)
    # history is already sorted by global_iter
    all_solutions = []
    num_routes_list = []
    global_iter_list = []
    cost_list = []
    local_search_id_list = []
    
    for idx, record in enumerate(history):
        # For the first record, also record sol_before
        if idx == 0:
            if record.get('sol_before') is not None:
                all_solutions.append(record['sol_before'])
                num_routes_list.append(record.get('num_routes_before', 0))
                global_iter_list.append(record.get('global_iter', -1))
                cost_list.append(record.get('cost_before', None))
                local_search_id_list.append(record.get('local_search_id', -1))
        
        # For all records, record sol_after (sol_before is same as previous sol_after)
        if record.get('sol_after') is not None:
            all_solutions.append(record['sol_after'])
            num_routes_list.append(record.get('num_routes_after', 0))
            global_iter_list.append(record.get('global_iter', -1))
            cost_list.append(record.get('cost_after', None))
            local_search_id_list.append(record.get('local_search_id', -1))

    unique_solutions = set(tuple(item) for item in all_solutions)
    unique_trials = set(local_search_id_list)
    print(f">>> Number of non-duplicate solutions: {len(unique_solutions)}")
    print(f">>> Number of trials (unique local_search_id): {len(unique_trials)}")
    
    # Compute distance matrix
    distance_matrix = solution_distance_matrix(all_solutions, num_routes_list, num_orders)
    
    return distance_matrix, all_solutions, global_iter_list, cost_list, local_search_id_list


def log_basin_stats_wandb(
    history,
    trial_solutions,
    trial_metadata,
    unique_optima,
    optimum_id_map,
    num_orders,
    hgs_solution_path,
    instance_index,
    timestamp,
    project_name="cuopt_basins",
):
    """
    Log basin-level statistics to wandb:
    - For each basin (optimum_id), number of unique solutions (all intermediates + local optima)
    - For each basin's local optimum, edge-distance to HGS and gap (%) to HGS
    """
    # Map trial_id -> optimum_id based on final solutions
    trial_id_to_optimum_id = {}
    for sol, meta in zip(trial_solutions, trial_metadata):
        trial_id = meta.get("local_search_id", -1)
        if trial_id is None:
            trial_id = -1

        num_routes_trial = meta.get("num_routes_after", 0) or 0
        inferred_num_routes_trial = infer_num_routes_from_solution_flat(sol, num_orders)
        if inferred_num_routes_trial > 0:
            num_routes_trial = inferred_num_routes_trial

        routes_trial = solution_flat_to_routes(sol, num_routes_trial, num_orders)
        edges_trial = extract_edges_from_routes(routes_trial)
        edges_tuple_trial = tuple(sorted(edges_trial))
        opt_id = optimum_id_map.get(edges_tuple_trial)
        if opt_id is not None:
            trial_id_to_optimum_id[trial_id] = opt_id

    # Count unique solutions per basin using all intermediate solutions
    basin_solutions = {}  # optimum_id -> set of solution keys
    for record in history:
        sol = record.get("sol_after")
        if sol is None or len(sol) == 0:
            continue
        trial_id = record.get("local_search_id", -1)
        if trial_id is None:
            trial_id = -1
        opt_id = trial_id_to_optimum_id.get(trial_id)
        if opt_id is None:
            continue
        sol_key = tuple(int(x) for x in sol)
        if opt_id not in basin_solutions:
            basin_solutions[opt_id] = set()
        basin_solutions[opt_id].add(sol_key)

    basin_sizes = {opt_id: len(sol_set) for opt_id, sol_set in basin_solutions.items()}

    # Distance of each basin's local optimum to HGS
    hgs_edges = None
    hgs_cost = None
    if hgs_solution_path:
        try:
            hgs_solution_full = load_hgs_solution_from_pkl(hgs_solution_path, instance_index)
            hgs_cost = hgs_solution_full["hgs_cost"]
            hgs_routes = convert_hgs_routes_to_format(hgs_solution_full["hgs_routes"])
            hgs_edges = extract_edges_from_routes(hgs_routes)
        except Exception as e:
            print(f"  Failed to load HGS solution for wandb stats: {e}")

    basin_edge_diff = {}
    basin_gap_pct = {}
    if hgs_edges is not None and hgs_cost is not None:
        for opt in unique_optima:
            opt_id = opt.get("optimum_id")
            edges = opt.get("edges", set())
            meta = opt.get("metadata", {})
            cost_opt = meta.get("cost")
            edge_diff = len(edges ^ hgs_edges)
            gap_pct = None
            if cost_opt is not None:
                gap_pct = calculate_gap(cost_opt, hgs_cost)
            basin_edge_diff[opt_id] = edge_diff
            basin_gap_pct[opt_id] = gap_pct

    # Log to wandb
    try:
        # Use a fresh run each time; group by instance so runs are organized
        run_name = f"basin_{timestamp}_instance{instance_index}"
        wandb.init(
            project=project_name,
            name=run_name,
            group=f"instance{instance_index}",
            reinit=True,
        )

        # Bar chart: unique solutions per basin
        table_size = wandb.Table(columns=["basin_id", "n_unique_solutions"])
        for opt_id, size in sorted(basin_sizes.items()):
            table_size.add_data(int(opt_id), int(size))
        wandb.log({
            "basin_unique_solution_count": wandb.plot.bar(
                table_size,
                "basin_id",
                "n_unique_solutions",
                title="Unique solutions per basin",
            )
        })

        # Bar chart: local optimum edge distance to HGS
        if basin_edge_diff:
            table_edge = wandb.Table(columns=["basin_id", "edge_diff_to_hgs"])
            for opt_id, diff in sorted(basin_edge_diff.items()):
                if diff is None:
                    continue
                table_edge.add_data(int(opt_id), int(diff))
            wandb.log({
                "basin_optimum_edge_diff_to_hgs": wandb.plot.bar(
                    table_edge,
                    "basin_id",
                    "edge_diff_to_hgs",
                    title="Local optimum edge distance to HGS (per basin)",
                )
            })

        # Bar chart: local optimum gap (%) to HGS
        if basin_gap_pct:
            table_gap = wandb.Table(columns=["basin_id", "gap_pct_to_hgs"])
            for opt_id, gap_val in sorted(basin_gap_pct.items()):
                if gap_val is None:
                    continue
                table_gap.add_data(int(opt_id), float(gap_val))
            wandb.log({
                "basin_optimum_gap_to_hgs_pct": wandb.plot.bar(
                    table_gap,
                    "basin_id",
                    "gap_pct_to_hgs",
                    title="Local optimum gap (%) to HGS (per basin)",
                )
            })

    except Exception as e:
        print(f"  Warning: failed to log wandb basin stats: {e}")


def extract_last_solution_per_trial(global_history, num_orders=None):
    """
    Extract the last solution from each trial (local_search_id).
    
    Args:
        global_history: Dictionary containing history records
        num_orders: Number of customer orders
    
    Returns:
        trial_solutions: List of last solutions for each trial
        trial_metadata: List of metadata (local_search_id, cost, etc.) for each trial
    """
    history = global_history['history']
    if not history:
        print("No records to extract trial solutions")
        return None, None
    
    # Group records by local_search_id
    trials_dict = {}
    for record in history:
        trial_id = record.get('local_search_id', -1)
        # Handle None values
        if trial_id is None:
            trial_id = -1
        if trial_id not in trials_dict:
            trials_dict[trial_id] = []
        trials_dict[trial_id].append(record)
    
    # Extract last solution from each trial
    trial_solutions = []
    trial_num_routes = []
    trial_metadata = []
    
    for trial_id in sorted(trials_dict.keys()):
        trial_records = trials_dict[trial_id]
        # Sort by local_iter to get the last one
        # Handle None values in local_iter
        trial_records_sorted = sorted(trial_records, 
                                     key=lambda x: x.get('local_iter') if x.get('local_iter') is not None else -1)
        
        # Last record is the last solution in this trial (regardless of type)
        last_record = trial_records_sorted[-1]
        
        if last_record.get('sol_after') is not None:
            trial_solutions.append(last_record['sol_after'])
            # Handle None values in num_routes_after
            num_routes_val = last_record.get('num_routes_after', 0)
            if num_routes_val is None:
                num_routes_val = 0
            trial_num_routes.append(num_routes_val)
            trial_metadata.append({
                'local_search_id': trial_id,
                'local_iter': last_record.get('local_iter', -1),
                'global_iter': last_record.get('global_iter', -1),
                'cost': last_record.get('cost_after', None),
                'move_found': last_record.get('move_found', False),
                'is_circle_found': last_record.get('is_circle_found', False),
                'num_routes_after': num_routes_val  # Store the cleaned value
            })
    
    print(f">>> Extracted {len(trial_solutions)} trials (last solution from each)")
    
    # Compute distance matrix for trial solutions
    if trial_solutions:
        distance_matrix = solution_distance_matrix(trial_solutions, trial_num_routes, num_orders)
        return distance_matrix, trial_solutions, trial_metadata
    else:
        return None, None, None


# ============================================================================
# Data Loading
# ============================================================================

def load_instance_from_pkl(instance_path, instance_index=0, scale=100.0):
    """Load CVRP instance from pickle file"""
    with open(instance_path, 'rb') as f:
        instances = pickle.load(f)
    
    if instance_index >= len(instances):
        raise ValueError(f"Instance index {instance_index} out of range (max: {len(instances)-1})")
    
    instance_tuple = instances[instance_index]
    depot_coords = np.array(instance_tuple[0]) * scale # (2,) - depot coordinates
    node_coords = np.array(instance_tuple[1]) * scale   # (100, 2) - customer coordinates
    demands = np.array(instance_tuple[2])       # (100,) - customer demands
    vehicle_capacity = instance_tuple[3]         # float - vehicle capacity
    
    # Combine depot and customer coordinates (depot at index 0)
    all_coords = np.vstack([depot_coords.reshape(1, -1), node_coords])
    
    # Calculate cost matrix (Euclidean distance) using vectorized operations
    num_locations = len(all_coords)
    # Use broadcasting to compute all pairwise distances at once
    # all_coords[i] - all_coords[j] for all i,j pairs
    diff = all_coords[:, np.newaxis, :] - all_coords[np.newaxis, :, :]  # (n, n, 2)
    cost_matrix = np.sqrt(np.sum(diff ** 2, axis=2)).astype(np.float32)  # (n, n)
    # Set diagonal to 0 (distance from node to itself)
    np.fill_diagonal(cost_matrix, 0.0)
    
    # Create demand vector (depot has 0 demand, customers have their demands)
    demand_vector = np.concatenate([[0], demands])
    
    return {
        'coordinates': all_coords,
        'cost_matrix': cost_matrix,
        'demand': demand_vector,
        'vehicle_capacity': vehicle_capacity,
        'n_locations': num_locations,
        'n_customers': len(node_coords)
    }


def load_hgs_solution_from_pkl(solution_path, instance_index=0, scale=100.0):
    """Load HGS solution from pickle file"""
    with open(solution_path, 'rb') as f:
        solutions = pickle.load(f)
    
    if instance_index >= len(solutions):
        raise ValueError(f"Solution index {instance_index} out of range (max: {len(solutions)-1})")
    
    solution_tuple = solutions[instance_index]
    hgs_cost = solution_tuple[0] * scale # float - HGS optimal cost
    hgs_routes = solution_tuple[1]  # list - route representation
    
    return {
        'hgs_cost': hgs_cost,
        'hgs_routes': hgs_routes,
        'instance_index': instance_index
    }


def calculate_gap(my_cost, hgs_cost):
    """Calculate gap between my solution and HGS solution"""
    if hgs_cost == 0:
        return float('inf') if my_cost > 0 else 0.0
    gap_percent = ((my_cost - hgs_cost) / hgs_cost) * 100.0
    return gap_percent


def convert_hgs_routes_to_format(hgs_routes):
    """Convert HGS route format to our route format."""
    return convert_node_sequence_to_routes(hgs_routes)


# ============================================================================
# Data Model Creation
# ============================================================================

def create_data_model_from_instance(instance_data, n_vehicles=30):
    """
    Create cuopt data_model from instance data.
    
    Args:
        instance_data: Dictionary with instance data (from load_instance_from_pkl)
        n_vehicles: Number of vehicles
    
    Returns:
        data_model: CuOpt DataModel
    """
    from cuopt_collector import Problem
    
    problem_data = {
        'n_locations': instance_data['n_locations'],
        'n_vehicles': n_vehicles,
        'cost_matrix': cudf.DataFrame(instance_data['cost_matrix']),
        'demand': cudf.Series(instance_data['demand']),
        'vehicle_capacity': cudf.Series(np.full(n_vehicles, instance_data['vehicle_capacity'], dtype=np.int32)),
        'coordinates': instance_data['coordinates'],
        'problem_scale': 100.0,
        'capacity_scale': instance_data['vehicle_capacity']
    }
    
    problem_gen = Problem(
        n_locations=instance_data['n_locations'],
        n_vehicles=n_vehicles,
        seed=42,
        coordinate_range=100.0,
        capacity=instance_data['vehicle_capacity'],
        demand_range=(1, 10)
    )
    
    data_model = problem_gen.create_data_model(problem_data)
    return data_model




# ============================================================================
# Visualization Functions
# ============================================================================

def plot_solution(routes, coords, title, ax, colors=None):
    """
    Plot a VRP solution (one or multiple routes) on given axes.
    
    Args:
        routes: List of routes, each route is [0, node1, ..., 0]
        coords: (N, 2) array of coordinates, index 0 is depot
        title: Title for this subplot
        ax: Matplotlib Axes
        colors: Optional list/array of colors; if None use tab10
    """
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Plot depot
    depot = coords[0]
    ax.scatter(depot[0], depot[1], c='red', s=80, marker='s', label='Depot', zorder=3)

    # Plot customer nodes
    if coords.shape[0] > 1:
        customers = coords[1:]
        ax.scatter(customers[:, 0], customers[:, 1], c='gray', s=20, alpha=0.6, label='Customers', zorder=2)

    # Plot routes
    for ridx, route in enumerate(routes):
        if not route or len(route) < 2:
            continue
        color = colors[ridx % len(colors)]
        pts = coords[np.array(route, dtype=int)]
        ax.plot(pts[:, 0], pts[:, 1], '-o', color=color, linewidth=1.5, markersize=4, alpha=0.9, zorder=4)

    ax.legend(fontsize=9)


def compare_solutions(final_routes, final_cost, coords, hgs_solution=None,
                      filename='solution_comparison.png'):
    """
    Create comparison visualization with HGS solution.
    
    Args:
        final_routes: List of routes for our best solution (standard format)
        final_cost: Cost of our best solution
        coords: (N, 2) array of coordinates
        hgs_solution: Optional dict with keys 'hgs_cost', 'hgs_routes'
        filename: Output filename (can include directory)
    """
    # Ensure output directory exists
    out_dir = os.path.dirname(filename) or '.'
    os.makedirs(out_dir, exist_ok=True)

    if hgs_solution is not None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        # Our solution
        plot_solution(
            final_routes,
            coords,
            f'My Best Solution\nCost: {final_cost:.2f}',
            axes[0],
            colors
        )

        # HGS solution (convert to our format)
        hgs_routes = convert_hgs_routes_to_format(hgs_solution['hgs_routes'])
        gap = calculate_gap(final_cost, hgs_solution['hgs_cost'])
        gap_text = f"Gap: {gap:.2f}%" if gap is not None else ""
        plot_solution(
            hgs_routes,
            coords,
            f'HGS Optimal\nCost: {hgs_solution["hgs_cost"]:.2f}\n{gap_text}',
            axes[1],
            colors
        )
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 9))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        plot_solution(
            final_routes,
            coords,
            f'Final Solution\nCost: {final_cost:.2f}',
            axes,
            colors
        )

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Solution comparison saved to: {filename}")


def plot_unique_optima_solutions(unique_optima, num_orders, instance_data,
                                 filename='unique_optima_routes.png'):
    """
    Plot all unique local optima solutions in a grid, ordered by discovery.
    
    Args:
        unique_optima: List of dicts with at least keys:
            - 'solution': cuopt solution_flat
            - 'metadata': metadata dict with 'local_search_id', 'cost', 'global_iter'
            - 'optimum_id': integer id
        num_orders: Number of customer orders
        instance_data: Dict from load_instance_from_pkl (must contain 'coordinates')
        filename: Output filename
    """
    if not unique_optima:
        print("No unique optima to plot.")
        return

    coords = instance_data['coordinates']

    # Prepare routes and titles
    routes_list = []
    titles = []
    for opt in unique_optima:
        sol = opt.get('solution')
        meta = opt.get('metadata', {})
        if sol is None:
            continue
        # Infer routes
        num_routes = meta.get('num_routes_after', 0) or 0
        inferred_num_routes = infer_num_routes_from_solution_flat(sol, num_orders)
        if inferred_num_routes > 0:
            num_routes = inferred_num_routes
        routes = solution_flat_to_routes(sol, num_routes, num_orders)
        routes_list.append(routes)

        trial_id = meta.get('local_search_id', -1)
        cost = meta.get('cost', None)
        global_iter = meta.get('global_iter', -1)
        title = f"Opt {opt.get('optimum_id', 0)} | trial {trial_id}"
        if cost is not None:
            title += f"\nCost {cost:.2f}"
        if global_iter is not None and global_iter >= 0:
            title += f" @ iter {global_iter}"
        titles.append(title)

    n = len(routes_list)
    if n == 0:
        print("No routes extracted for unique optima.")
        return

    # Determine grid size
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = np.reshape(axes, (rows, cols))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if idx < n:
                plot_solution(routes_list[idx], coords, titles[idx], ax, colors)
            else:
                ax.axis('off')
            idx += 1

    plt.tight_layout()
    out_dir = os.path.dirname(filename) or '.'
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Unique optima routes plot saved to: {filename}")
    plt.close(fig)


def visualize_contrastive_dataset(dataset_file, output_dir='plot'):
    """
    Visualize contrastive learning dataset with multiple plots.
    Supports both regular and extended datasets (with three perturbation types).
    Also supports loading and plotting saved history files.
    
    Args:
        dataset_file: Path to the saved dataset pickle file or history file
        output_dir: Output directory for saving plots
    """
    # Load dataset
    with open(dataset_file, 'rb') as f:
        data = pickle.load(f)
    
    # Check if this is a history file (saved by save_full_history)
    if 'history' in data and 'result' in data:
        # This is a history file, plot cost curve by trial
        print("Detected history file, plotting cost curve by trial...")
        history = data.get('history', [])
        result = data.get('result', {})
        instance_index = data.get('instance_index', 0)
        timestamp = data.get('timestamp', '')
        
        # Infer num_orders
        num_orders = None
        if 'instance_data' in data:
            num_orders = data['instance_data'].get('n_customers')
        if num_orders is None and history:
            first_sol = history[0].get('sol_after')
            if first_sol is not None:
                num_orders = len([x for x in first_sol if x > 0])
        
        # Extract trial metadata and duplicate groups
        temp_global_history = {'history': history}
        _, trial_solutions, trial_metadata = extract_last_solution_per_trial(
            temp_global_history, num_orders=num_orders
        )
        
        # Identify unique local optima
        unique_optima = []
        optimum_id_map = {}
        duplicate_groups = {}
        
        for idx, (opt_sol, opt_meta) in enumerate(zip(trial_solutions, trial_metadata)):
            num_routes = opt_meta.get('num_routes_after', 0)
            if num_routes is None:
                num_routes = 0
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
            else:
                optimum_id = optimum_id_map[edges_tuple]
                duplicate_groups[edges_tuple].append(idx)
        
        # Get HGS cost and gap from result
        gap = result.get('gap')
        
        # Plot cost curve by trial
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = os.path.join(output_dir, f'callback_cost_curve_by_trial_loaded_{timestamp}.png')
        
        plot_cost_curve_by_trial(
            temp_global_history,
            num_orders=num_orders,
            filename=plot_filename,
            trial_metadata=trial_metadata,
            duplicate_groups=duplicate_groups,
            optimum_id_map=optimum_id_map,
            gap=gap
        )
        print(f"  Saved to: {plot_filename}")
        return
    
    # Check if this is an extended dataset
    if 'datasets' in data:
        # Extended dataset with multiple perturbation types
        visualize_extended_dataset(data, output_dir)
    else:
        # Regular dataset
        dataset = data['dataset']
        unique_optima = data['unique_optima']
        visualize_single_dataset(dataset, unique_optima, output_dir, 'all')


def visualize_extended_dataset(data, output_dir='plot'):
    """Visualize extended dataset with three perturbation types."""
    unique_optima = data['unique_optima']
    all_datasets = data['datasets']
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize each perturbation type separately
    for pert_type, dataset in all_datasets.items():
        if dataset and dataset.get('pairs'):
            print(f"\nVisualizing {pert_type} perturbations...")
            visualize_single_dataset(dataset, unique_optima, output_dir, pert_type, timestamp)
    
    # Also create a combined comparison plot
    # TODO: Implement create_combined_comparison_plot if needed
    # create_combined_comparison_plot(all_datasets, output_dir, timestamp)


def visualize_single_dataset(dataset, unique_optima, output_dir, pert_type='all', timestamp=None):
    """Visualize a single dataset (one perturbation type)."""
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== Plot 1: Return Frequency Distribution ==========
    return_frequencies = [label['return_frequency'] for label in dataset['labels']]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Histogram of return frequencies
    ax1 = axes[0, 0]
    ax1.hist(return_frequencies, bins=20, edgecolor='black', alpha=0.7, color='#2E86AB')
    ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Similar threshold (0.5)')
    ax1.axvline(0.1, color='orange', linestyle='--', linewidth=2, label='Distant threshold (0.1)')
    ax1.set_xlabel('Return Frequency', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    title_suffix = f" ({pert_type})" if pert_type != 'all' else ""
    ax1.set_title(f'Distribution of Return Frequencies{title_suffix}', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Similarity label distribution
    ax2 = axes[0, 1]
    similarity_labels = [label['similarity_label'] for label in dataset['labels']]
    label_counts = {
        'similar': similarity_labels.count('similar'),
        'distant': similarity_labels.count('distant'),
        'moderate': similarity_labels.count('moderate')
    }
    colors = ['#2E86AB', '#E63946', '#F18F01']
    ax2.bar(label_counts.keys(), label_counts.values(), color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title(f'Similarity Label Distribution{title_suffix}', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for label, count in label_counts.items():
        ax2.text(label, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    # Subplot 3: Return frequency vs perturbation strength (if available)
    ax3 = axes[1, 0]
    if any('perturbation_strength' in label for label in dataset['labels']):
        pert_strengths = [label.get('perturbation_strength', 0) for label in dataset['labels']]
        ax3.scatter(pert_strengths, return_frequencies, alpha=0.6, s=50, color='#2E86AB')
        ax3.set_xlabel('Perturbation Strength', fontsize=12, fontweight='bold')
    else:
        # Just show distribution
        ax3.hist(return_frequencies, bins=20, edgecolor='black', alpha=0.7, color='#F18F01')
        ax3.set_xlabel('Return Frequency', fontsize=12, fontweight='bold')
    ax3.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axhline(0.1, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_ylabel('Return Frequency', fontsize=12, fontweight='bold')
    ax3.set_title(f'Return Frequency Analysis{title_suffix}', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Convergence diversity
    ax4 = axes[1, 1]
    n_different_optima = [len(label['convergence_distribution']) for label in dataset['labels']]
    if n_different_optima:
        ax4.hist(n_different_optima, bins=range(1, max(n_different_optima)+2), 
                 edgecolor='black', alpha=0.7, color='#F18F01', align='left')
    ax4.set_xlabel('Number of Different Optima Reached', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax4.set_title(f'Convergence Diversity{title_suffix}', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename1 = os.path.join(output_dir, f'dataset_overview_{pert_type}_{timestamp}.png')
    plt.savefig(filename1, dpi=150, bbox_inches='tight')
    print(f"  Overview plot saved to: {filename1}")
    plt.close()
    
    # ========== Plot 2: Convergence Distribution Heatmap ==========
    if len(unique_optima) > 1:
        # Create matrix: rows = perturbations, cols = unique optima
        n_pairs = len(dataset['pairs'])
        n_optima = len(unique_optima)
        
        convergence_matrix = np.zeros((n_pairs, n_optima))
        for i, label in enumerate(dataset['labels']):
            for opt_id, count in label['convergence_distribution'].items():
                if opt_id >= 0 and opt_id < n_optima:
                    convergence_matrix[i, opt_id] = count / label['n_runs']
        
        fig, ax = plt.subplots(figsize=(max(12, n_optima), max(8, n_pairs/5)))
        im = ax.imshow(convergence_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Convergence Frequency', fontsize=12, fontweight='bold')
        
        # Set labels
        ax.set_xlabel('Local Optimum ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('Perturbation Index', fontsize=12, fontweight='bold')
        ax.set_title('Convergence Distribution Heatmap\n(Frequency of converging to each optimum)', 
                     fontsize=13, fontweight='bold', pad=15)
        
        # Set ticks
        ax.set_xticks(np.arange(n_optima))
        ax.set_xticklabels([f"Opt {i}" for i in range(n_optima)])
        
        # Only show y-ticks if not too many
        if n_pairs <= 50:
            ax.set_yticks(np.arange(n_pairs))
            ax.set_yticklabels([f"P{i}" for i in range(n_pairs)])
        else:
            step = max(1, n_pairs // 20)
            ax.set_yticks(np.arange(0, n_pairs, step))
            ax.set_yticklabels([f"P{i}" for i in range(0, n_pairs, step)])
        
        plt.tight_layout()
        filename2 = os.path.join(output_dir, f'convergence_heatmap_{pert_type}_{timestamp}.png')
        plt.savefig(filename2, dpi=150, bbox_inches='tight')
        print(f"  Convergence heatmap saved to: {filename2}")
        plt.close()
    
    # ========== Plot 3: Return Frequency by Original Optimum ==========
    if len(unique_optima) > 1:
        # Group by original optimum
        optimum_return_freqs = {}
        for pair, label in zip(dataset['pairs'], dataset['labels']):
            opt_id = pair['original_optimum_id']
            if opt_id not in optimum_return_freqs:
                optimum_return_freqs[opt_id] = []
            optimum_return_freqs[opt_id].append(label['return_frequency'])
        
        fig, ax = plt.subplots(figsize=(max(12, len(optimum_return_freqs)*0.8), 8))
        
        opt_ids = sorted(optimum_return_freqs.keys())
        data_to_plot = [optimum_return_freqs[oid] for oid in opt_ids]
        positions = range(len(opt_ids))
        
        bp = ax.boxplot(data_to_plot, positions=positions, patch_artist=True, 
                       widths=0.6, showmeans=True, meanline=True)
        
        # Color boxes
        colors = plt.cm.tab10(np.linspace(0, 1, len(opt_ids)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Customize other elements
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)
        
        plt.setp(bp['means'], linestyle='--', linewidth=2)
        
        ax.set_xticks(positions)
        ax.set_xticklabels([f"Opt {oid}" for oid in opt_ids])
        ax.set_ylabel('Return Frequency', fontsize=12, fontweight='bold')
        ax.set_xlabel('Original Local Optimum ID', fontsize=12, fontweight='bold')
        ax.set_title('Return Frequency Distribution by Original Optimum', 
                     fontsize=13, fontweight='bold', pad=15)
        ax.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Similar threshold')
        ax.axhline(0.1, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Distant threshold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filename3 = os.path.join(output_dir, f'return_freq_by_optimum_{pert_type}_{timestamp}.png')
        plt.savefig(filename3, dpi=150, bbox_inches='tight')
        print(f"  Return frequency by optimum plot saved to: {filename3}")
        plt.close()
    
    # ========== Print Summary Statistics ==========
    print(f"\n{'='*60}")
    print(f"Dataset Visualization Summary")
    print(f"{'='*60}")
    print(f"Total pairs: {len(dataset['pairs'])}")
    print(f"Unique optima: {len(unique_optima)}")
    print(f"\nReturn Frequency Statistics:")
    print(f"  Min: {min(return_frequencies):.4f}")
    print(f"  Max: {max(return_frequencies):.4f}")
    print(f"  Mean: {np.mean(return_frequencies):.4f}")
    print(f"  Median: {np.median(return_frequencies):.4f}")
    print(f"\nSimilarity Label Distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(similarity_labels)*100:.1f}%)")
    print("=" * 60)


def plot_cost_curve_by_trial(global_history, num_orders=None, filename='callback_cost_curve_by_trial.png',
                              trial_metadata=None, duplicate_groups=None, optimum_id_map=None,
                              gap=None):
    """
    Plot cost evolution - exactly same as plot_cost_curve, but overlay trial lines with colors from Step 3.
    
    Args:
        global_history: Dictionary containing history records
        num_orders: Number of customer orders (not used, kept for compatibility)
        filename: Output filename
        trial_metadata: List of trial metadata (from extract_last_solution_per_trial)
        duplicate_groups: Dict mapping edges_tuple to list of trial indices (from Step 3)
        optimum_id_map: Dict mapping edges_tuple to optimum_id (from Step 3)
    """
    history = global_history['history']
    if not history:
        print("No records to plot")
        return
    
    # Build trial color mapping from Step 3 results
    trial_colors = {}  # Map trial_id to color
    if duplicate_groups is not None and trial_metadata is not None:
        # Map from trial index (in trial_metadata) to trial_id (local_search_id)
        trial_idx_to_trial_id = {}
        for idx, meta in enumerate(trial_metadata):
            trial_id = meta.get('local_search_id', -1)
            if trial_id is None:
                trial_id = -1
            trial_idx_to_trial_id[idx] = trial_id
        
        # Assign colors based on duplicate_groups
        unique_color = 'black'
        duplicate_colors_list = plt.cm.tab10(np.linspace(0, 1, 10))
        # Convert numpy array colors to tuples for hashing
        duplicate_colors = [tuple(c) if isinstance(c, np.ndarray) else c for c in duplicate_colors_list]
        # Remove orange color (#F18F01) and black/dark colors from duplicate colors to avoid conflicts
        # Cycle finder uses #F18F01 = RGB(241, 143, 1) = (0.945, 0.561, 0.004)
        # Unique trials use black = RGB(0, 0, 0) = (0.0, 0.0, 0.0)
        cycle_finder_color_rgb = (0xF1/255.0, 0x8F/255.0, 0x01/255.0)
        filtered_colors = []
        for c in duplicate_colors:
            if len(c) >= 3:
                c_rgb = np.array(c[:3])
                # Check if color is too close to orange (cycle finder)
                is_orange = np.allclose(c_rgb, np.array(cycle_finder_color_rgb), atol=0.15)
                # Check if color is too dark/black (sum of RGB < 0.3 means very dark)
                is_black = np.sum(c_rgb) < 0.3
                if not is_orange and not is_black:
                    filtered_colors.append(c)
        duplicate_colors = filtered_colors
        # If all colors were filtered out, use a fallback color palette (excluding black and orange)
        if len(duplicate_colors) == 0:
            # Use Set2 or Set3 colormap as fallback (these don't have black or orange)
            duplicate_colors = [tuple(c) if isinstance(c, np.ndarray) else c 
                               for c in plt.cm.Set2(np.linspace(0, 1, 8))]
            # Filter out any dark colors (RGB sum < 0.3) and orange-like colors
            duplicate_colors = [c for c in duplicate_colors 
                               if len(c) >= 3 and np.sum(np.array(c[:3])) >= 0.3 and
                               not np.allclose(np.array(c[:3]), np.array(cycle_finder_color_rgb), atol=0.15)]
        
        color_idx = 0
        
        for edges_tuple, trial_indices in duplicate_groups.items():
            # Convert trial_indices to list of integers
            if isinstance(trial_indices, np.ndarray):
                trial_indices = trial_indices.tolist()
            elif not isinstance(trial_indices, list):
                trial_indices = list(trial_indices)
            
            # Map indices to trial_ids
            trial_ids = []
            for idx in trial_indices:
                idx_int = int(idx) if not isinstance(idx, int) else idx
                if idx_int in trial_idx_to_trial_id:
                    trial_ids.append(trial_idx_to_trial_id[idx_int])
            
            if len(trial_ids) == 1:
                # Unique trial - use black
                trial_colors[trial_ids[0]] = unique_color
            else:
                # Duplicate trials - use same color (convert to tuple for hashing)
                # IMPORTANT: Duplicate trials should NEVER use black
                if len(duplicate_colors) == 0:
                    # Last resort: use a default non-black color
                    color = (0.2, 0.6, 0.8)  # Light blue
                else:
                    color = duplicate_colors[color_idx % len(duplicate_colors)]
                if isinstance(color, np.ndarray):
                    color = tuple(color)
                # Double-check: ensure duplicate trials don't get black
                if color == unique_color or (isinstance(color, tuple) and len(color) >= 3 and np.sum(np.array(color[:3])) < 0.3):
                    # Use a fallback color instead
                    color = (0.2, 0.6, 0.8)  # Light blue
                for tid in trial_ids:
                    trial_colors[tid] = color
                color_idx += 1
    
    # Group records by local_search_id (trial)
    trials_dict = {}
    for record in history:
        trial_id = record.get('local_search_id', -1)
        if trial_id is None:
            trial_id = -1
        if trial_id not in trials_dict:
            trials_dict[trial_id] = []
        trials_dict[trial_id].append(record)
    
    # Build duplicate groups for legend (map color to list of trial_ids)
    # Convert colors to hashable types (tuples) for dictionary keys
    color_to_trials = {}  # Map color to list of trial_ids
    for trial_id, color in trial_colors.items():
        # Convert numpy array colors to tuples for hashing
        hashable_color = tuple(color) if isinstance(color, np.ndarray) else color
        if hashable_color not in color_to_trials:
            color_to_trials[hashable_color] = []
        color_to_trials[hashable_color].append(trial_id)
        # Also update trial_colors to use hashable color
        if hashable_color != color:
            trial_colors[trial_id] = hashable_color
    
    # Build a set of truly duplicate trial_ids (from duplicate_groups)
    truly_duplicate_trial_ids = set()
    if duplicate_groups is not None:
        for edges_tuple, trial_indices in duplicate_groups.items():
            # Map indices to trial_ids
            trial_ids = []
            for idx in trial_indices:
                idx_int = int(idx) if not isinstance(idx, int) else idx
                if idx_int in trial_idx_to_trial_id:
                    trial_ids.append(trial_idx_to_trial_id[idx_int])
            # Only mark as duplicate if there are multiple trials with same optimum
            if len(trial_ids) > 1:
                truly_duplicate_trial_ids.update(trial_ids)
    
    # Create figure for trial-colored cycle finder
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each trial with its own color
    unique_color = 'black'
    
    # Track which colors have been labeled
    labeled_colors = set()
    cycle_finder_labeled = False
    
    # Helper function to convert color to hashable type
    def get_hashable_color(color):
        if isinstance(color, np.ndarray):
            return tuple(color)
        return color
    
    for trial_id in sorted(trials_dict.keys()):
        trial_records = trials_dict[trial_id]
        trial_records_sorted = sorted(trial_records, 
                                     key=lambda x: x.get('local_iter') if x.get('local_iter') is not None else -1)
        
        # Get color for this trial
        if trial_id in trial_colors:
            trial_color = trial_colors[trial_id]
            # Convert to hashable if needed
            if isinstance(trial_color, np.ndarray):
                trial_color = tuple(trial_color)
        else:
            trial_color = unique_color  # Unique trial uses black
        
        # Collect all points for this trial
        trial_points = []
        fast_search_points = []
        cycle_finder_points = []
        
        for record in trial_records_sorted:
            if record.get('cost_after') is not None:
                global_iter = record.get('global_iter', -1)
                cost = record['cost_after']
                is_cycle = record.get('is_circle_found', False)
                
                trial_points.append((global_iter, cost))
                if is_cycle:
                    cycle_finder_points.append((global_iter, cost))
                else:
                    fast_search_points.append((global_iter, cost))
        
        # Helper function to get label for this trial
        def get_trial_label(trial_id, trial_colors, labeled_colors, truly_duplicate_trial_ids, 
                           duplicate_groups, trial_idx_to_trial_id, unique_color):
            label = None
            if trial_id in trial_colors:
                color = trial_colors[trial_id]
                hashable_color = get_hashable_color(color)
                if hashable_color not in labeled_colors:
                    # Check if this trial is truly duplicate (from duplicate_groups)
                    is_truly_duplicate = trial_id in truly_duplicate_trial_ids
                    
                    if is_truly_duplicate:
                        # Get all trials with same optimum (from duplicate_groups)
                        duplicate_trials = []
                        if duplicate_groups is not None:
                            for edges_tuple, trial_indices in duplicate_groups.items():
                                trial_ids = []
                                for idx in trial_indices:
                                    idx_int = int(idx) if not isinstance(idx, int) else idx
                                    if idx_int in trial_idx_to_trial_id:
                                        trial_ids.append(trial_idx_to_trial_id[idx_int])
                                if trial_id in trial_ids and len(trial_ids) > 1:
                                    duplicate_trials = sorted(trial_ids)
                                    break
                        if len(duplicate_trials) > 1:
                            label = f"Trials {duplicate_trials} (duplicate)"
                        else:
                            label = f"Trial {trial_id}"
                    else:
                        # Unique trial - use generic label for black color
                        if hashable_color == unique_color or (isinstance(hashable_color, str) and hashable_color == 'black'):
                            if unique_color not in labeled_colors:
                                label = "Unique trials"
                                labeled_colors.add(unique_color)
                        else:
                            # Other unique trial with different color (shouldn't happen, but handle it)
                            label = f"Trial {trial_id}"
                    labeled_colors.add(hashable_color)
            else:
                # Unique trial (no color assigned)
                if unique_color not in labeled_colors:
                    label = "Unique trials"
                    labeled_colors.add(unique_color)
            return label
        
        # Plot fast search points
        if fast_search_points:
            iters, costs = zip(*sorted(fast_search_points, key=lambda x: x[0]))
            label = get_trial_label(trial_id, trial_colors, labeled_colors, truly_duplicate_trial_ids,
                                   duplicate_groups, trial_idx_to_trial_id, unique_color)
            ax.plot(iters, costs, marker='o', linestyle='-', linewidth=1.5, markersize=4,
                    color=trial_color, label=label, alpha=0.6, zorder=2)
        
        # Plot cycle finder points same color as trial
        if cycle_finder_points:
            iters, costs = zip(*cycle_finder_points)
            ax.scatter(iters, costs, marker='s', s=25, color=trial_color,
                       label='Cycle Finder' if not cycle_finder_labeled else None, 
                       alpha=0.8, zorder=4, edgecolors='black', linewidths=0.5)
            cycle_finder_labeled = True
        
        # Plot trial trajectory line
        if len(trial_points) > 1:
            trial_points_sorted = sorted(trial_points, key=lambda x: x[0])
            iters, costs = zip(*trial_points_sorted)
            ax.plot(iters, costs, linestyle='-', linewidth=1.5, color=trial_color,
                    alpha=0.7, zorder=1, label=None)
    
    # Helper function to set up axes and save figure
    def setup_and_save_figure(fig, ax, filename, title_base):
        ax.set_xlabel('Global Iteration', fontsize=13, fontweight='bold')
        ax.set_ylabel('Objective Value (Cost)', fontsize=13, fontweight='bold')
        
        # Calculate best cost (minimum cost) from history
        best_cost = None
        if history:
            all_costs = []
            for r in history:
                cost_after = r.get('cost_after')
                if cost_after is not None:
                    # Convert to scalar if it's a numpy array
                    if hasattr(cost_after, 'item'):
                        cost_after = cost_after.item()
                    elif isinstance(cost_after, np.ndarray):
                        cost_after = float(cost_after[0]) if len(cost_after) > 0 else None
                    if cost_after is not None:
                        all_costs.append(cost_after)
            if all_costs:
                best_cost = min(all_costs)
        
        # Build title with best Cost and Gap if available
        title = title_base
        if best_cost is not None:
            title += f' | Cost: {best_cost:.2f}'
        if gap is not None:
            title += f' | Gap: {gap:.2f}%'
        
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        # Legend with automatic wrapping to prevent it from exceeding plot boundaries
        legend = ax.legend(loc='best', fontsize=10, framealpha=0.9, shadow=True, 
                          ncol=1, columnspacing=1.0, handlelength=1.5)
        # Adjust legend to fit within plot
        fig.canvas.draw()
        bbox = legend.get_window_extent()
        ax_bbox = ax.get_window_extent()
        if bbox.width > ax_bbox.width * 0.9:
            # If legend is too wide, use multiple columns
            legend = ax.legend(loc='best', fontsize=9, framealpha=0.9, shadow=True,
                              ncol=2, columnspacing=0.8, handlelength=1.2)
            fig.canvas.draw()
            bbox = legend.get_window_extent()
            if bbox.width > ax_bbox.width * 0.9:
                # If still too wide, use 3 columns with smaller font
                legend = ax.legend(loc='best', fontsize=8, framealpha=0.9, shadow=True,
                                  ncol=3, columnspacing=0.6, handlelength=1.0)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Save figure
    base_filename = filename.replace('.png', '')
    output_filename = f"{base_filename}_trial_colored_cf.png"
    
    setup_and_save_figure(fig, ax, output_filename, 'Cost Evolution During Optimization')
    
    # Statistics
    all_costs = [r['cost_after'] for r in history if r['cost_after'] is not None]
    all_search_count = sum(len([r for r in trials_dict[tid] if r.get('cost_after') is not None and not r.get('is_circle_found', False)]) 
                          for tid in trials_dict.keys())
    cycle_finder_count = sum(len([r for r in trials_dict[tid] if r.get('is_circle_found') and r.get('cost_after') is not None]) 
                            for tid in trials_dict.keys())
    
    print(f"\nCost Statistics:")
    if all_costs:
        print(f"  Initial: {all_costs[0]:.2f}")
        print(f"  Best: {min(all_costs):.2f}")
        print(f"  Improvement: {all_costs[0] - min(all_costs):.2f} ({(1 - min(all_costs)/all_costs[0])*100:.1f}%)")
    print(f"  Total iterations: {len(history)}")
    print(f"  All search: {all_search_count}")
    print(f"  Cycle finder: {cycle_finder_count}")
    print(f"  Total trials: {len(trials_dict)}")
    print(f"  Saved to: {output_filename}")


# ============================================================================
# Overall Correlation Plot Generation
# ============================================================================

def edge_set_key_for_basin(edges_list):
    """Helper function to create a hashable key from edge list for basin identification."""
    if not edges_list:
        return None
    norm_edges = [tuple(sorted((int(u), int(v)))) for (u, v) in edges_list]
    return frozenset(norm_edges)


def generate_overall_correlation_plot(data_dir):
    """
    Generate overall correlation plot with statistics.
    
    This function creates a 2x3 subplot figure showing:
    - Row 1: Mean Cost by Attraction Basin Width, Convergence Complexity, and Exploration Volume (line plots)
    - Row 2: Cost Distribution by the same three metrics (box plots)
    - Statistics text box with dataset information
    
    Args:
        data_dir: Path to the dataset directory (e.g., 'basin_datasets0/cvrp100_uniform.pkl#0')
    
    Returns:
        str: Path to the saved figure
    """
    import json
    import csv
    from collections import defaultdict
    
    # File paths
    basin_statistics_csv = os.path.join(data_dir, 'basin_statistics.csv')
    basin_statistics_json = os.path.join(data_dir, 'basin_statistics.json')
    optima_jsonl = os.path.join(data_dir, 'optima.jsonl')
    trajectory_jsonl = os.path.join(data_dir, 'trajectory.jsonl')
    
    # Read basin statistics
    basin_data = []
    with open(basin_statistics_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            basin_data.append({
                'exploration_volume': int(row['num_solutions']),
                'attraction_basin_width': int(row['num_trials']),
                'cost': float(row['cost']) if row['cost'] else None,
            })
    
    # Read optima.jsonl to build basin mapping
    optima_raw = []
    with open(optima_jsonl, 'r') as f:
        for line in f:
            optima_raw.append(json.loads(line))
    
    optima_by_basin = {}
    runopt_to_basin = {}
    for r in optima_raw:
        edges = r.get("edges")
        basin_key = edge_set_key_for_basin(edges)
        if basin_key is None:
            continue
        run_id = r.get("run_id")
        opt_id = r.get("optimum_id")
        if run_id is not None and opt_id is not None:
            runopt_to_basin[(run_id, opt_id)] = basin_key
        prev = optima_by_basin.get(basin_key)
        if prev is None:
            optima_by_basin[basin_key] = r
        else:
            c_new = r.get("final_cost")
            c_prev = prev.get("final_cost")
            if c_new is not None and (c_prev is None or c_new < c_prev):
                optima_by_basin[basin_key] = r
    
    # Calculate Convergence Complexity (average trajectory length per basin)
    basin_trial_lengths = defaultdict(list)
    current_trial = None
    current_length = 0
    
    with open(trajectory_jsonl, 'r') as f:
        for line in f:
            data = json.loads(line)
            run_id = data.get("run_id")
            opt_id = data.get("optimum_id")
            if run_id is None or opt_id is None:
                continue
            
            basin_key = runopt_to_basin.get((run_id, opt_id))
            if basin_key is None:
                continue
            
            trial_key = (run_id, opt_id)
            if current_trial != trial_key:
                if current_trial is not None and current_length > 0:
                    basin_trial_lengths[basin_key].append(current_length)
                current_trial = trial_key
                current_length = 1
            else:
                current_length += 1
        
        if current_trial is not None and current_length > 0:
            basin_key = runopt_to_basin.get(current_trial)
            if basin_key is not None:
                basin_trial_lengths[basin_key].append(current_length)
    
    # Calculate average trajectory length (Convergence Complexity)
    basin_to_complexity = {}
    for basin_key, lengths in basin_trial_lengths.items():
        if lengths:
            basin_to_complexity[basin_key] = np.mean(lengths)
    
    # Match basin_data with convergence_complexity
    basin_to_edges_hash = {}
    for r in optima_raw:
        edges = r.get("edges")
        basin_key = edge_set_key_for_basin(edges)
        if basin_key:
            edges_hash = r.get("edges_hash", "")
            basin_to_edges_hash[basin_key] = edges_hash
    
    # Read from basin_statistics.json
    with open(basin_statistics_json, 'r') as f:
        basin_stats_json = json.load(f)
    
    basin_data_final = []
    for stat in basin_stats_json:
        edges_hash = stat.get('edges_hash', '')
        matching_basin = None
        for basin_key, hash_val in basin_to_edges_hash.items():
            if hash_val.startswith(edges_hash[:16]):
                matching_basin = basin_key
                break
        
        complexity = basin_to_complexity.get(matching_basin) if matching_basin else None
        if complexity is None:
            continue
        
        basin_data_final.append({
            'exploration_volume': stat.get('num_solutions', 0),
            'attraction_basin_width': stat.get('num_trials', 0),
            'convergence_complexity': complexity,
            'cost': stat.get('cost'),
        })
    
    valid_final = [d for d in basin_data_final if d['cost'] is not None and d['convergence_complexity'] is not None]
    
    exploration_volume = np.array([d['exploration_volume'] for d in valid_final])
    attraction_basin_width = np.array([d['attraction_basin_width'] for d in valid_final])
    convergence_complexity = np.array([d['convergence_complexity'] for d in valid_final])
    costs = np.array([d['cost'] for d in valid_final])
    
    # Calculate correlations
    corr_exploration_cost = np.corrcoef(exploration_volume, costs)[0, 1]
    corr_attraction_cost = np.corrcoef(attraction_basin_width, costs)[0, 1]
    corr_complexity_cost = np.corrcoef(convergence_complexity, costs)[0, 1]
    
    # Calculate statistics
    trials_set = set()
    for r in optima_raw:
        run_id = r.get('run_id')
        opt_id = r.get('optimum_id')
        if run_id is not None and opt_id is not None:
            trials_set.add((run_id, opt_id))
    
    unique_solutions = set()
    trajectory_count = 0
    run_ids_traj = set()
    with open(trajectory_jsonl, 'r') as f:
        for line in f:
            trajectory_count += 1
            data = json.loads(line)
            run_id = data.get("run_id")
            if run_id is not None:
                run_ids_traj.add(run_id)
            edges_sol = data.get("edges")
            if edges_sol:
                sol_key = edge_set_key_for_basin(edges_sol)
                if sol_key is not None:
                    unique_solutions.add(sol_key)
    
    basin_trials = defaultdict(set)
    for r in optima_raw:
        edges = r.get("edges")
        basin_key = edge_set_key_for_basin(edges)
        if basin_key is None:
            continue
        run_id = r.get("run_id")
        opt_id = r.get("optimum_id")
        if run_id is not None and opt_id is not None:
            basin_trials[basin_key].add((run_id, opt_id))
    
    # Create 2x3 subplot figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # ========== Row 1: Line plots ==========
    
    # 1. Mean Cost by Attraction Basin Width (left)
    ax1 = axes[0, 0]
    trial_counts = sorted(set(attraction_basin_width))
    mean_costs_trial = []
    std_costs_trial = []
    valid_trials = []
    for t in trial_counts:
        group_costs = [d['cost'] for d in valid_final if d['attraction_basin_width'] == t]
        if group_costs:
            valid_trials.append(t)
            mean_costs_trial.append(np.mean(group_costs))
            std_costs_trial.append(np.std(group_costs))
    
    ax1.errorbar(valid_trials, mean_costs_trial, yerr=std_costs_trial, fmt='o-', linewidth=2,
                markersize=6, capsize=4, color='red')
    ax1.set_xlabel('Attraction Basin Width C(x*)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Cost f(x*)', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Cost by Attraction Basin Width', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, f'r = {corr_attraction_cost:.4f}', transform=ax1.transAxes,
            fontsize=11, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Mean Cost by Convergence Complexity (middle)
    ax2 = axes[0, 1]
    complexity_bins_edges = np.percentile(convergence_complexity, [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100])
    complexity_bins_list = []
    for i in range(len(complexity_bins_edges)-1):
        min_c = complexity_bins_edges[i]
        max_c = complexity_bins_edges[i+1]
        group_costs = [d['cost'] for d in valid_final if min_c <= d['convergence_complexity'] < max_c]
        if group_costs:
            complexity_bins_list.append({
                'center': (min_c + max_c) / 2,
                'mean': np.mean(group_costs),
                'std': np.std(group_costs)
            })
    
    bin_centers_c = [b['center'] for b in complexity_bins_list]
    mean_costs_c = [b['mean'] for b in complexity_bins_list]
    std_costs_c = [b['std'] for b in complexity_bins_list]
    
    ax2.errorbar(bin_centers_c, mean_costs_c, yerr=std_costs_c, fmt='o-', linewidth=2,
                markersize=6, capsize=4, color='green')
    ax2.set_xlabel('Convergence Complexity C(x*)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Cost f(x*)', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Cost by Convergence Complexity', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.98, f'r = {corr_complexity_cost:.4f}', transform=ax2.transAxes,
            fontsize=11, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 3. Mean Cost by Exploration Volume (right)
    ax3 = axes[0, 2]
    volume_bins = [
        (1, 1, "1"),
        (2, 5, "2-5"),
        (6, 10, "6-10"),
        (11, 20, "11-20"),
        (21, 50, "21-50"),
        (51, 100, "51-100"),
        (101, 500, "101-500"),
        (501, float('inf'), "500+"),
    ]
    
    bin_centers = []
    mean_costs_vol = []
    std_costs_vol = []
    for min_v, max_v, label in volume_bins:
        group_costs = [d['cost'] for d in valid_final if min_v <= d['exploration_volume'] <= max_v]
        if group_costs:
            bin_centers.append((min_v + max_v) / 2 if max_v != float('inf') else 750)
            mean_costs_vol.append(np.mean(group_costs))
            std_costs_vol.append(np.std(group_costs))
    
    ax3.errorbar(bin_centers, mean_costs_vol, yerr=std_costs_vol, fmt='o-', linewidth=2, 
                markersize=6, capsize=4, color='blue')
    ax3.set_xlabel('Exploration Volume', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mean Cost f(x*)', fontsize=12, fontweight='bold')
    ax3.set_title('Mean Cost by Exploration Volume', fontsize=13, fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.98, f'r = {corr_exploration_cost:.4f}', transform=ax3.transAxes,
            fontsize=11, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========== Row 2: Box plots ==========
    
    # 4. Cost Distribution by Attraction Basin Width (left)
    ax4 = axes[1, 0]
    trial_bins = [
        (1, 1, "1"),
        (2, 2, "2"),
        (3, 5, "3-5"),
        (6, 10, "6-10"),
        (11, 20, "11-20"),
        (21, 40, "21-40"),
    ]
    
    box_data_trial = []
    box_labels_trial = []
    for min_t, max_t, label in trial_bins:
        group_costs = [d['cost'] for d in valid_final if min_t <= d['attraction_basin_width'] <= max_t]
        if group_costs:
            box_data_trial.append(group_costs)
            box_labels_trial.append(label)
    
    bp4 = ax4.boxplot(box_data_trial, tick_labels=box_labels_trial, patch_artist=True)
    for patch in bp4['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)
    ax4.set_xlabel('Attraction Basin Width C(x*)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Cost f(x*)', fontsize=12, fontweight='bold')
    ax4.set_title('Cost Distribution by Attraction Basin Width', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Cost Distribution by Convergence Complexity (middle)
    ax5 = axes[1, 1]
    complexity_bins_for_box = []
    for i in range(len(complexity_bins_edges)-1):
        min_c = complexity_bins_edges[i]
        max_c = complexity_bins_edges[i+1]
        group_costs = [d['cost'] for d in valid_final if min_c <= d['convergence_complexity'] < max_c]
        if group_costs:
            complexity_bins_for_box.append({
                'costs': group_costs,
                'label': f'{int(min_c)}-{int(max_c)}'
            })
    
    box_data_c = [b['costs'] for b in complexity_bins_for_box]
    box_labels_c = [b['label'] for b in complexity_bins_for_box]
    
    bp5 = ax5.boxplot(box_data_c, tick_labels=box_labels_c, patch_artist=True)
    for patch in bp5['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)
    ax5.set_xlabel('Convergence Complexity C(x*)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Cost f(x*)', fontsize=12, fontweight='bold')
    ax5.set_title('Cost Distribution by Convergence Complexity', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Cost Distribution by Exploration Volume (right)
    ax6 = axes[1, 2]
    box_data_vol = []
    box_labels_vol = []
    for min_v, max_v, label in volume_bins:
        group_costs = [d['cost'] for d in valid_final if min_v <= d['exploration_volume'] <= max_v]
        if group_costs:
            box_data_vol.append(group_costs)
            box_labels_vol.append(label)
    
    bp6 = ax6.boxplot(box_data_vol, tick_labels=box_labels_vol, patch_artist=True)
    for patch in bp6['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax6.set_xlabel('Exploration Volume', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Cost f(x*)', fontsize=12, fontweight='bold')
    ax6.set_title('Cost Distribution by Exploration Volume', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text box at the top
    stats_text = (
        f"Dataset Statistics:\n"
        f"Basin Count: {len(basin_trials):,}\n"
        f"Trial Count: {len(trials_set):,}\n"
        f"Unique Solutions: {len(unique_solutions):,}\n"
        f"Trajectory Records: {trajectory_count:,}\n"
        f"Unique Run IDs: {len(run_ids_traj):,}"
    )
    
    fig.text(0.5, 0.99, stats_text, fontsize=11, 
            verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            transform=fig.transFigure)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for top text
    output_file = os.path.join(data_dir, 'overall_correlation.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_file
