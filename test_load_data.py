"""
Quick test to show problem_data structure
"""
import numpy as np
import pickle
import cudf

def load_instance_from_pkl(instance_path, instance_index=0, scale=100.0):
    """Load CVRP instance from pickle file"""
    with open(instance_path, 'rb') as f:
        instances = pickle.load(f)
    
    if instance_index >= len(instances):
        raise ValueError(f"Instance index {instance_index} out of range (max: {len(instances)-1})")
    
    instance_tuple = instances[instance_index]
    depot_coords = np.array(instance_tuple[0]) * scale  # (2,) - depot coordinates
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
    hgs_cost = solution_tuple[0] * scale  # float - HGS optimal cost
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


if __name__ == "__main__":
    instance_path = "/data/jieyi/unified_solver_1/data/CVRP/cvrp100_uniform.pkl"
    instance_index = 0
    n_vehicles = 30
    
    print("=" * 60)
    print("LOADING INSTANCE DATA")
    print("=" * 60)
    
    instance_data = load_instance_from_pkl(instance_path, instance_index)
    
    print(f"\nLoaded instance data:")
    print(f"  n_locations: {instance_data['n_locations']}")
    print(f"  n_customers: {instance_data['n_customers']}")
    print(f"  vehicle_capacity: {instance_data['vehicle_capacity']}")
    print(f"  coordinates shape: {instance_data['coordinates'].shape}")
    print(f"  cost_matrix shape: {instance_data['cost_matrix'].shape}")
    print(f"  demand length: {len(instance_data['demand'])}")
    
    # Create problem_data dictionary compatible with Problem class
    problem_data = {
        'n_locations': instance_data['n_locations'],
        'n_vehicles': n_vehicles,
        'cost_matrix': cudf.DataFrame(instance_data['cost_matrix']),
        'demand': cudf.Series(instance_data['demand']),
        'vehicle_capacity': cudf.Series(np.full(n_vehicles, instance_data['vehicle_capacity'], dtype=np.int32)),
        'coordinates': instance_data['coordinates'],
        'problem_scale': np.max(instance_data['coordinates']),
        'capacity_scale': instance_data['vehicle_capacity']
    }
    
    # Print problem_data structure for debugging
    print("\n" + "=" * 60)
    print("PROBLEM_DATA STRUCTURE")
    print("=" * 60)
    print(f"n_locations: {problem_data['n_locations']} (type: {type(problem_data['n_locations'])})")
    print(f"n_vehicles: {problem_data['n_vehicles']} (type: {type(problem_data['n_vehicles'])})")
    print(f"\ncost_matrix:")
    print(f"  Type: {type(problem_data['cost_matrix'])}")
    print(f"  Shape: {problem_data['cost_matrix'].shape}")
    print(f"  First 3x3 values:\n{problem_data['cost_matrix'].iloc[:3, :3]}")
    print(f"\ndemand:")
    print(f"  Type: {type(problem_data['demand'])}")
    print(f"  Length: {len(problem_data['demand'])}")
    print(f"  First 10 values: {problem_data['demand'].head(10).to_arrow().to_pylist()}")
    print(f"\nvehicle_capacity:")
    print(f"  Type: {type(problem_data['vehicle_capacity'])}")
    print(f"  Length: {len(problem_data['vehicle_capacity'])}")
    print(f"  Values: {problem_data['vehicle_capacity'].to_arrow().to_pylist()}")
    print(f"\ncoordinates:")
    print(f"  Type: {type(problem_data['coordinates'])}")
    print(f"  Shape: {problem_data['coordinates'].shape}")
    print(f"  First 5 rows:\n{problem_data['coordinates'][:5]}")
    print(f"\nproblem_scale: {problem_data['problem_scale']} (type: {type(problem_data['problem_scale'])})")
    print(f"capacity_scale: {problem_data['capacity_scale']} (type: {type(problem_data['capacity_scale'])})")
    print("=" * 60)

