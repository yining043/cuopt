"""
Utility functions for loading and converting VRP solutions.
"""
import json
import hashlib


def edges_to_routes(edges, num_orders):
    """
    Convert edge list to route list.
    - Edges are undirected: (u, v) == (v, u)
    - num_orders: Total nodes (1 depot + num_orders-1 customers), e.g., 101
    - Customers: [1, num_orders-1], depot: 0
    - Returns routes in format [0, ..., 0]
    """
    # Build adjacency list for undirected graph
    adj = {}
    for u, v in edges:
        u, v = int(u), int(v)
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    routes, used = [], set()

    # Build routes starting from depot (0)
    for start in adj.get(0, []):
        if start in used:
            continue
        route, curr = [0, start], start
        used.add(start)

        # Traverse until returning to depot or no unvisited neighbors
        while curr != 0 and curr in adj:
            next_nodes = [n for n in adj[curr] if n not in used and n != 0]
            if not next_nodes:
                if 0 in adj[curr] and route[-1] != 0:
                    route.append(0)
                break
            nxt = next_nodes[0]
            route.append(nxt)
            used.add(nxt)
            curr = nxt

        # Ensure route ends with depot
        if route[-1] != 0:
            route.append(0)

        if len(route) > 2:
            routes.append(route)

    # Add missing customers as single-node routes
    all_nodes_in_routes = {n for r in routes for n in r if n != 0}
    missing = set(range(1, int(num_orders))) - all_nodes_in_routes
    routes.extend([[0, n, 0] for n in sorted(missing)])
    return routes


def solution_flat_to_routes(solution_flat, num_routes, num_orders):
    """Convert flat solution list to route list. Dummy depots (>=num_orders) mark route boundaries."""
    routes, i, current_route = [], 0, []
    
    while i < len(solution_flat):
        node = solution_flat[i]
        if node >= num_orders:
            if current_route:
                routes.append([0] + current_route + [0])
                current_route = []
            i += 4
            continue
        if 1 <= node < num_orders:
            current_route.append(node)
        i += 1
    
    if current_route:
        routes.append([0] + current_route + [0])
    
    return routes


def load_solution_from_trajectory(trajectory_path, trial_id, global_iter=None, local_iter=None, num_orders=None, run_id=None):
    """Load solution from trajectory.jsonl. Returns None if not found.
    
    Args:
        trajectory_path: Path to trajectory.jsonl file
        trial_id: Trial ID to match
        global_iter: Global iteration to match (optional)
        local_iter: Local iteration to match (optional)
        num_orders: Number of orders (optional, will use from data if not provided)
        run_id: Run ID to match (optional, but recommended to avoid ambiguity)
    """
    with open(trajectory_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Check all matching criteria
            if data['trial_id'] != trial_id:
                continue
            if run_id is not None and data.get('run_id') != run_id:
                continue
            if global_iter is not None and data.get('global_iter') != global_iter:
                continue
            if local_iter is not None and data.get('local_iter') != local_iter:
                continue
            # All criteria matched
            num_orders = num_orders or data.get('num_orders')
            if num_orders is None:
                # Try to infer from solution_flat if available
                solution_flat = data.get('solution_flat', [])
                if solution_flat:
                    # Rough estimate: count non-zero values (excluding depot)
                    num_orders = len([x for x in solution_flat if x > 0])
            if num_orders is None:
                print(f"Warning: Could not determine num_orders for solution")
                return None
            data['routes'] = solution_flat_to_routes(data['solution_flat'], 
                data.get('num_routes_after', data.get('num_routes')), num_orders)
            return data
    return None


def load_optimum_id_from_trial(trials_path, trial_id):
    """Get optimum_id for trial_id from trials.jsonl. Returns None if not found."""
    with open(trials_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['trial_id'] == trial_id:
                return data.get('optimum_id')
    return None


def load_solution_from_optima(optima_path, optimum_id):
    """Load solution from optima.jsonl and convert edges to routes. Returns None if not found."""
    with open(optima_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['optimum_id'] == optimum_id:
                data['routes'] = edges_to_routes(data['edges'], data.get('num_orders', 100))
                return data
    return None


def edges_hash(edges):
    """Compute stable hash for edge set."""
    if not edges:
        return ""
    edge_str = ";".join(f"{int(u)}-{int(v)}" for (u, v) in sorted(edges))
    return hashlib.sha1(edge_str.encode("utf-8")).hexdigest()


def routes_to_edges(routes):
    """Convert routes to edge set (undirected edges: normalized with small id first).
    
    The normalization (min(u, v), max(u, v)) makes edges undirected:
    both (u, v) and (v, u) are stored as the same tuple.
    """
    edges = set()
    for route in routes:
        if len(route) < 2:
            continue
        for i in range(len(route) - 1):
            u, v = int(route[i]), int(route[i + 1])
            # Normalize: small id first (undirected edge representation)
            edges.add((min(u, v), max(u, v)))
    return edges


def routes_to_solution_flat(routes, num_orders):
    """Convert routes to solution_flat format (with dummy depots). Each route uses different dummy depot IDs."""
    solution_flat = []
    dummy_base = num_orders
    for route_idx, route in enumerate(routes):
        # Each route uses 4 consecutive dummy depots, starting from num_orders + route_idx * 4
        dummy_start = dummy_base + route_idx * 4
        solution_flat.extend([dummy_start, dummy_start + 1, dummy_start + 2, dummy_start + 3])
        # Add customer nodes (skip depot 0 at start and end)
        for node in route[1:-1]:
            solution_flat.append(node)
    return solution_flat


def get_basin_paths(instance_path, instance_index, basin_base_dir="basin_datasets0"):
    """Get basin dataset paths based on instance_path and instance_index."""
    import os
    
    instance_id = f"{os.path.basename(instance_path)}#{instance_index}"
    if not os.path.isabs(basin_base_dir):
        basin_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), basin_base_dir)
    
    basin_dir = os.path.join(basin_base_dir, instance_id)
    return {
        'instance_id': instance_id,
        'basin_dir': basin_dir,
        'trajectory_path': os.path.join(basin_dir, 'trajectory.jsonl'),
    }

