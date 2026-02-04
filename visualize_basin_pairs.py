#!/usr/bin/env python3
"""
Aggregate basin pairs by summing weights for identical pairs (unordered)
Visualize basin network graph
Filter out weak edges if needed
"""

import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def read_basin_pairs(file_path):
    """
    Read basin pairs from JSONL file
    """
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
                pairs.append(pair)
            except Exception as e:
                print(f"Error parsing line: {e}")
                continue
    return pairs


def aggregate_pairs(pairs):
    """
    Aggregate pairs: for identical unordered pairs, sum their weights
    Returns: dict mapping (basin1, basin2) -> total_weight
    """
    pair_weights = defaultdict(float)
    
    for pair in pairs:
        anchor_hash = pair['anchor_basin']['hash']
        neighbor_hash = pair['neighbor_basin']['hash']
        
        # Create unordered pair (always put smaller hash first for consistency)
        if anchor_hash < neighbor_hash:
            pair_key = (anchor_hash, neighbor_hash)
        else:
            pair_key = (neighbor_hash, anchor_hash)
        
        weight = pair['weight']
        pair_weights[pair_key] += weight
    
    return pair_weights


def read_basin_info(file_path):
    """
    Read basin info mapping
    """
    basin_info = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                info = json.loads(line)
                basin_info[info['hash']] = info
            except Exception as e:
                print(f"Error parsing line: {e}")
                continue
    return basin_info


def visualize_top_weight_network(pair_weights, basin_info, output_dir, top_n=100):
    """
    Visualize top N weight basin network graph
    Node size: inversely proportional to cost (lower cost = larger node)
    Edge width: proportional to weight
    """
    # Sort pairs by weight and take top N
    sorted_pairs = sorted(pair_weights.items(), key=lambda x: x[1], reverse=True)
    top_pairs = sorted_pairs[:top_n]
    
    print(f"\nVisualizing top {top_n} weight pairs...")
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes and edges for top pairs
    for (basin1, basin2), weight in top_pairs:
        G.add_edge(basin1, basin2, weight=weight)
    
    print(f"Nodes in graph: {G.number_of_nodes()}")
    print(f"Edges in graph: {G.number_of_edges()}")
    
    if G.number_of_edges() == 0:
        print("No edges to visualize!")
        return
    
    # Get node costs for sizing
    node_costs = {}
    for node in G.nodes():
        if node in basin_info and basin_info[node].get('cost') is not None:
            node_costs[node] = basin_info[node]['cost']
        else:
            # If cost is missing, use a default large value (small node)
            node_costs[node] = 10000
    
    # Calculate node sizes: cost越小节点越大
    # Invert the cost relationship: larger cost -> smaller node
    costs = list(node_costs.values())
    min_cost = min(costs)
    max_cost = max(costs)
    cost_range = max_cost - min_cost if max_cost > min_cost else 1
    
    # Node size: inversely proportional to cost
    # Scale: min_size + (max_cost - cost) / cost_range * (max_size - min_size)
    min_node_size = 50
    max_node_size = 500
    node_sizes = []
    for node in G.nodes():
        cost = node_costs[node]
        # Invert: higher cost -> smaller node
        normalized = (max_cost - cost) / cost_range if cost_range > 0 else 0.5
        size = min_node_size + normalized * (max_node_size - min_node_size)
        node_sizes.append(size)
    
    # Get edge weights for visualization
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    min_weight = min(edge_weights) if edge_weights else 0
    weight_range = max_weight - min_weight if max_weight > min_weight else 1
    
    # Edge widths: proportional to weight
    min_edge_width = 0.5
    max_edge_width = 5.0
    edge_widths = []
    for u, v in G.edges():
        weight = G[u][v]['weight']
        normalized = (weight - min_weight) / weight_range if weight_range > 0 else 0.5
        width = min_edge_width + normalized * (max_edge_width - min_edge_width)
        edge_widths.append(width)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw edges with width proportional to weight
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, 
                           edge_color='gray', ax=ax)
    
    # Draw nodes with size proportional to inverse cost
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                           node_color='lightblue', alpha=0.7, ax=ax)
    
    # Label important nodes with specific criteria:
    # 1. Top 10 nodes with lowest cost (best solutions)
    # 2. Top 10 nodes with highest degree (most connected)
    # Combine and deduplicate
    node_degrees = dict(G.degree())
    
    # Get top 10 lowest cost nodes
    nodes_by_cost = sorted(G.nodes(), key=lambda n: node_costs.get(n, 10000))[:10]
    
    # Get top 10 highest degree nodes
    nodes_by_degree = sorted(G.nodes(), key=lambda n: node_degrees.get(n, 0), reverse=True)[:10]
    
    # Combine and deduplicate
    important_nodes = list(set(nodes_by_cost + nodes_by_degree))
    
    if len(important_nodes) > 0:
        labels = {}
        for n in important_nodes:
            if n in node_costs:
                cost = node_costs[n]
                degree = node_degrees.get(n, 0)
                # Show hash, cost, and degree
                labels[n] = f"{n[:8]}\n{cost:.1f}\nd:{degree}"
        nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax)
    
    ax.set_title(f'Top {top_n} Weight Basin Network\n'
                 f'Node size ∝ 1/cost (lower cost = larger node)\n'
                 f'Edge width ∝ weight\n'
                 f'({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)', 
                 fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'basin_network_top100.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Top {top_n} weight network visualization saved to: {output_path}")
    plt.close()
    
    # Print statistics
    print(f"\nTop {top_n} weight statistics:")
    print(f"  Min weight: {min(edge_weights):.6f}")
    print(f"  Max weight: {max(edge_weights):.6f}")
    print(f"  Mean weight: {np.mean(edge_weights):.6f}")
    print(f"\nNode cost statistics:")
    print(f"  Min cost: {min(costs):.2f}")
    print(f"  Max cost: {max(costs):.2f}")
    print(f"  Mean cost: {np.mean(costs):.2f}")


def process_instance(instance_index, top_n=100):
    """
    Process visualization for a single instance
    """
    output_dir = f"/home/jieyi/cuopt/basin_datasets0_analyze/cvrp100_uniform.pkl#{instance_index}"
    pairs_file = os.path.join(output_dir, "basin_pairs.jsonl")
    basin_info_file = os.path.join(output_dir, "basin_info.jsonl")
    
    if not os.path.exists(pairs_file) or not os.path.exists(basin_info_file):
        print(f"Instance {instance_index}: Required files not found, skipping...")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing Instance {instance_index}")
    print(f"{'='*60}")
    
    print("Reading basin pairs...")
    pairs = read_basin_pairs(pairs_file)
    print(f"Read {len(pairs)} basin pairs")
    
    if len(pairs) == 0:
        print(f"Instance {instance_index}: No basin pairs found, skipping...")
        return
    
    print("\nAggregating pairs (summing weights for identical pairs)...")
    pair_weights = aggregate_pairs(pairs)
    print(f"Unique basin pairs: {len(pair_weights)}")
    
    print("\nReading basin info...")
    basin_info = read_basin_info(basin_info_file)
    print(f"Read info for {len(basin_info)} basins")
    
    # Visualize top 100 weight network
    print("\n" + "="*60)
    print(f"Visualization: Top {top_n} weight pairs")
    print("="*60)
    visualize_top_weight_network(pair_weights, basin_info, output_dir, top_n=top_n)
    
    print(f"Instance {instance_index} done!")


def parse_instance_range(instance_str):
    """
    Parse instance range string into list of indices
    Examples:
        "0-10" -> [0, 1, 2, ..., 10]
        "0,1,2" -> [0, 1, 2]
        "0-5,8,10" -> [0, 1, 2, 3, 4, 5, 8, 10]
    """
    indices = set()
    parts = instance_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range
            start, end = part.split('-')
            start, end = int(start.strip()), int(end.strip())
            indices.update(range(start, end + 1))
        else:
            # Single number
            indices.add(int(part.strip()))
    return sorted(list(indices))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize basin pairs for specified instances')
    parser.add_argument('--instances', type=str, default='0-10',
                       help='Instance indices to process (e.g., "0-10", "0,1,2", "0-5,8,10"). Default: 0-10')
    parser.add_argument('--top_n', type=int, default=100,
                       help='Number of top weight pairs to visualize (default: 100)')
    
    args = parser.parse_args()
    
    # Parse instance range
    instance_indices = parse_instance_range(args.instances)
    
    print(f"Processing {len(instance_indices)} instances: {instance_indices}")
    print(f"Top N: {args.top_n}")
    
    for instance_index in instance_indices:
        try:
            process_instance(instance_index, top_n=args.top_n)
        except Exception as e:
            print(f"Error processing instance {instance_index}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("All instances processed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
