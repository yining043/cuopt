#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a Search Trajectory Network (STN) from trajectory.jsonl and produce:
1. A final static figure (highly modular, clearly separated communities)
2. A search-trajectory GIF with edges added over time

Dependencies:
    pip install networkx matplotlib python-louvain imageio
"""

import argparse
import base64
import io
import json
import math
from collections import Counter, defaultdict

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import community as community_louvain  # python-louvain


def load_trajectory_build_graph(path, max_runs=None):
    """
    Read trajectory.jsonl, build STN (directed graph), and return:
    - G: nx.DiGraph
    - edge_sequence: [(u, v), ...], transitions in chronological order for GIF
    - node_freq: Counter(hash -> frequency)
    - node_cost: dict(hash -> cost)

    max_runs: if set, only use the first max_runs runs (by order of appearance).

    Supports two formats:
    1. Nested: each line is a Run with key "trials" (list of trials; each trial is a list
       of steps; last step has "hash" and "cost").
    2. Flat: each line is one step with run_id, trial_id, edges_hash (or "hash"), cost.
       Last record per (run_id, trial_id) is the local optimum.
    """
    node_freq = Counter()
    node_cost = {}
    edge_weights = Counter()
    edge_sequence = []

    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line:
            return _build_graph_from_parsed(node_freq, node_cost, edge_weights, edge_sequence)
        data0 = json.loads(first_line)

    # Nested format: one JSON per Run with "trials" list
    if "trials" in data0:
        _parse_nested(path, node_freq, node_cost, edge_weights, edge_sequence, data0, first_line, max_runs)
    else:
        # Flat format: one line per step; group by (run_id, trial_id)
        _parse_flat(path, node_freq, node_cost, edge_weights, edge_sequence, data0, first_line, max_runs)

    return _build_graph_from_parsed(node_freq, node_cost, edge_weights, edge_sequence)


def _build_graph_from_parsed(node_freq, node_cost, edge_weights, edge_sequence):
    G = nx.DiGraph()
    for h, freq in node_freq.items():
        G.add_node(h, freq=freq, cost=node_cost[h])
    for (u, v), w in edge_weights.items():
        G.add_edge(u, v, weight=w)
    return G, edge_sequence, node_freq, node_cost


def _parse_nested(path, node_freq, node_cost, edge_weights, edge_sequence, data0, first_line, max_runs=None):
    def process_line(data):
        trials = data.get("trials", [])
        last_hashes = []
        for trial in trials:
            if not trial:
                continue
            last_sol = trial[-1]
            h = last_sol.get("hash") or last_sol.get("edges_hash")
            if h is None:
                continue
            c = last_sol["cost"]
            node_freq[h] += 1
            if h not in node_cost or c < node_cost[h]:
                node_cost[h] = c
            last_hashes.append(h)
        for i in range(len(last_hashes) - 1):
            u, v = last_hashes[i], last_hashes[i + 1]
            edge_weights[(u, v)] += 1
            edge_sequence.append((u, v))

    process_line(data0)
    runs_done = 1
    with open(path, "r", encoding="utf-8") as f:
        f.readline()  # skip first
        for line in f:
            if max_runs is not None and runs_done >= max_runs:
                break
            line = line.strip()
            if not line:
                continue
            process_line(json.loads(line))
            runs_done += 1


def _parse_flat(path, node_freq, node_cost, edge_weights, edge_sequence, data0, first_line, max_runs=None):
    # Group by (run_id, trial_id); each group = list of records in order. Last = local optimum.
    # Hash key: "edges_hash" or "hash"
    # Only keep runs that are among the first max_runs (by order of first appearance).
    groups = defaultdict(list)
    run_order = []  # order of first-seen run_id
    seen_runs = set()

    def add_record(data):
        run_id = data.get("run_id")
        trial_id = data.get("trial_id")
        h = data.get("edges_hash") or data.get("hash")
        if run_id is None or trial_id is None or h is None:
            return
        if max_runs is not None and run_id not in seen_runs and len(seen_runs) >= max_runs:
            return  # skip runs beyond first max_runs
        if run_id not in seen_runs:
            seen_runs.add(run_id)
            run_order.append(run_id)
        try:
            c = float(data.get("cost", 0))
        except (TypeError, ValueError):
            return
        groups[(run_id, trial_id)].append((h, c))

    add_record(data0)
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                add_record(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Restrict to first max_runs run_ids
    if max_runs is not None:
        allowed_runs = set(run_order[:max_runs])
    else:
        allowed_runs = None

    # Per run_id: sort by trial_id, get last (h, c) per trial, build edges
    runs = defaultdict(list)
    for (run_id, trial_id), recs in groups.items():
        if allowed_runs is not None and run_id not in allowed_runs:
            continue
        if not recs:
            continue
        h, c = recs[-1]
        node_freq[h] += 1
        if h not in node_cost or c < node_cost[h]:
            node_cost[h] = c
        runs[run_id].append((trial_id, h))

    for run_id in (run_order if max_runs is None else run_order[:max_runs]):
        if run_id not in runs:
            continue
        trial_list = runs[run_id]
        trial_list.sort(key=lambda x: x[0])
        hashes = [h for _, h in trial_list]
        for i in range(len(hashes) - 1):
            u, v = hashes[i], hashes[i + 1]
            edge_weights[(u, v)] += 1
            edge_sequence.append((u, v))


def community_layout(
    G,
    partition,
    community_scale=6.0,
    intra_scale=1.0,
    k_community=None,
    k_intra=None,
    seed=42,
    iterations=200,
    placement="spring",
):
    """
    Layout based on Louvain communities:
    1. Treat each community as a "super-node", run spring_layout on the community graph.
    2. Run spring_layout inside each community, then scale and translate to community center.

    Returns: pos: dict(node -> (x, y))
    """
    H = G.to_undirected()

    # Community partition
    communities = defaultdict(list)
    for node, comm in partition.items():
        communities[comm].append(node)

    comm_ids = sorted(communities.keys())

    if placement == "circle":
        # Place communities on a big circle (maximally separated)
        n_comms = max(len(comm_ids), 1)
        pos_communities = {}
        for i, comm in enumerate(comm_ids):
            angle = 2.0 * math.pi * i / n_comms
            pos_communities[comm] = (math.cos(angle), math.sin(angle))
    else:
        # Community super-graph (weighted by inter-community connectivity)
        community_graph = nx.Graph()
        for comm in comm_ids:
            community_graph.add_node(comm)
        for u, v in H.edges():
            cu = partition[u]
            cv = partition[v]
            if cu != cv:
                w = H[u][v].get("weight", 1)
                if community_graph.has_edge(cu, cv):
                    community_graph[cu][cv]["weight"] += w
                else:
                    community_graph.add_edge(cu, cv, weight=w)

        # Community-level layout: spread communities apart
        if k_community is None:
            n_comms = max(len(community_graph), 1)
            k_community = 2.5 / math.sqrt(n_comms)
        pos_communities = nx.spring_layout(
            community_graph,
            k=k_community,
            seed=seed,
            weight="weight",
            iterations=iterations,
        )

    # Intra-community layout
    pos = {}
    for comm, nodes in communities.items():
        subgraph = H.subgraph(nodes)
        if k_intra is None:
            k_in = 1.5 / math.sqrt(max(len(subgraph), 1))
        else:
            k_in = k_intra
        sub_pos = nx.spring_layout(
            subgraph,
            k=k_in,
            seed=seed,
            weight="weight",
            scale=intra_scale,
            iterations=iterations,
        )
        cx, cy = pos_communities[comm]
        for n in subgraph.nodes():
            sx, sy = sub_pos[n]
            pos[n] = (
                cx + community_scale * sx,
                cy + community_scale * sy,
            )  # translate to community center and separate communities

    return pos


def compute_partition_and_layout(
    G,
    resolution=1.0,
    seed=42,
    community_scale=6.0,
    intra_scale=1.0,
    k_community=None,
    k_intra=None,
    iterations=200,
    placement="spring",
):
    """
    Run Louvain community detection on G and return:
    - partition: dict(node -> community_id)
    - pos: dict(node -> (x, y)) from community_layout for well-separated communities
    """
    undirected = G.to_undirected()
    partition = community_louvain.best_partition(
        undirected,
        weight="weight",
        resolution=resolution,
        random_state=seed,
    )

    pos = community_layout(
        G,
        partition,
        community_scale=community_scale,
        intra_scale=intra_scale,
        k_community=k_community,
        k_intra=k_intra,
        seed=seed,
        iterations=iterations,
        placement=placement,
    )
    return partition, pos


def _compute_node_sizes_and_colors(
    G,
    node_freq,
    node_cost,
    partition,
    color_mode="cost",
):
    """Helper: compute node sizes and colors consistently."""
    nodes = list(G.nodes())

    # Node size: frequency -> scale mapping
    freqs = np.array([node_freq[n] for n in nodes], dtype=float)
    if len(freqs) == 0:
        return nodes, np.array([]), []
    freq_min, freq_max = freqs.min(), freqs.max()
    if freq_max == freq_min:
        sizes = np.full_like(freqs, 200.0)
    else:
        sizes = 100 + 900 * (freqs - freq_min) / (freq_max - freq_min)

    # Node colors
    if color_mode == "cost":
        costs = np.array([node_cost[n] for n in nodes], dtype=float)
        cmin, cmax = costs.min(), costs.max()
        cmap = plt.cm.Oranges_r  # lower cost -> darker color
        if cmax == cmin:
            normed = np.full_like(costs, 0.5)
        else:
            normed = (costs - cmin) / (cmax - cmin)
        node_colors = cmap(normed)
    else:
        # Option B: color by community ID to highlight clusters
        # Use by passing color_mode="community" when calling
        comm_ids = np.array([partition[n] for n in nodes])
        unique_comms = sorted(set(comm_ids))
        cmap = plt.cm.tab20
        color_map = {c: cmap(i % cmap.N) for i, c in enumerate(unique_comms)}
        node_colors = [color_map[c] for c in comm_ids]

    return nodes, sizes, node_colors


def draw_static_stn(
    G,
    pos,
    node_freq,
    node_cost,
    partition,
    out_path="stn_final.png",
    color_mode="cost",
):
    """
    Draw the final STN static figure.

    color_mode:
    - "cost" (recommended): color by cost; lower cost -> darker.
    - "community": color by community ID to emphasize cluster structure.
    """
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.set_axis_off()

    nodes, sizes, node_colors = _compute_node_sizes_and_colors(
        G, node_freq, node_cost, partition, color_mode=color_mode
    )
    if len(nodes) == 0:
        print("Graph is empty; nothing to draw.")
        return

    # Draw edges (thicker than before)
    edges = list(G.edges())
    edge_weights = [
        0.8 + 2.5 * math.log1p(G[u][v].get("weight", 1)) for u, v in edges
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        width=edge_weights,
        alpha=0.45,
        edge_color="#666666",
        arrows=False,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodes,
        node_size=sizes,
        node_color=node_colors,
        linewidths=0.2,
        edgecolors="black",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Static STN figure saved to: {out_path}")


def _node_order_from_trajectory(edge_sequence, all_nodes):
    """Order nodes by first appearance in edge_sequence; append the rest."""
    seen = set()
    order = []
    for u, v in edge_sequence:
        for n in (u, v):
            if n not in seen:
                seen.add(n)
                order.append(n)
    for n in all_nodes:
        if n not in seen:
            order.append(n)
    return order


def _get_trajectory_frame_indices(node_order, nodes_per_frame, frames_max=None):
    """Return list of node indices (show up to idx+1 nodes) for each frame."""
    n_nodes = len(node_order)
    if n_nodes == 0:
        return []
    idx_list = list(range(nodes_per_frame - 1, n_nodes, nodes_per_frame))
    if not idx_list or idx_list[-1] != n_nodes - 1:
        idx_list.append(n_nodes - 1)
    if frames_max is not None and len(idx_list) > frames_max:
        sel = np.linspace(0, len(idx_list) - 1, num=frames_max, dtype=int)
        return [idx_list[i] for i in sel]
    return idx_list


def _draw_one_trajectory_frame(
    G, pos, node_order, idx, node_to_size, node_to_color
):
    """Draw one frame (nodes up to node_order[:idx+1]) and return RGB array (H,W,3) uint8."""
    visible_nodes = node_order[: idx + 1]
    visible_set = set(visible_nodes)
    current_edges = [(u, v) for u, v in G.edges() if u in visible_set and v in visible_set]

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.set_axis_off()

    edge_widths = [
        0.8 + 2.0 * math.log1p(G[u][v].get("weight", 1)) for u, v in current_edges
    ]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=current_edges,
        width=edge_widths,
        alpha=0.45,
        edge_color="#666666",
        arrows=False,
    )
    vis_sizes = [node_to_size[n] for n in visible_nodes]
    vis_colors = [node_to_color[n] for n in visible_nodes]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=visible_nodes,
        node_size=vis_sizes,
        node_color=vis_colors,
        linewidths=0.2,
        edgecolors="black",
    )

    plt.tight_layout()
    fig = plt.gcf()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    image = buf[:, :, :3].copy()
    plt.close()
    return image


def make_trajectory_gif(
    G,
    pos,
    edge_sequence,
    node_freq,
    node_cost,
    partition,
    gif_path="stn_trajectory.gif",
    color_mode="cost",
    frames=None,
    nodes_per_frame=2,
):
    """
    Build a GIF by drawing nodes in trajectory order.
    Each frame adds nodes_per_frame nodes (and edges between already-visible nodes).
    frames: if set, cap total number of frames (subsample).
    nodes_per_frame: how many new nodes to add per frame (default 2).
    """
    nodes, sizes, node_colors = _compute_node_sizes_and_colors(
        G, node_freq, node_cost, partition, color_mode=color_mode
    )
    if len(nodes) == 0:
        return

    node_order = _node_order_from_trajectory(edge_sequence, nodes)
    indices = _get_trajectory_frame_indices(node_order, nodes_per_frame, frames)

    if not indices:
        return

    node_to_size = dict(zip(nodes, sizes))
    node_to_color = dict(zip(nodes, node_colors))

    images = []
    for idx in indices:
        image = _draw_one_trajectory_frame(
            G, pos, node_order, idx, node_to_size, node_to_color
        )
        images.append(image)

    imageio.mimsave(gif_path, images, fps=8)
    print(f"Search trajectory GIF saved to: {gif_path}")


def make_trajectory_html(
    G,
    pos,
    edge_sequence,
    node_freq,
    node_cost,
    partition,
    html_path="stn_trajectory.html",
    color_mode="cost",
    frames=None,
    nodes_per_frame=2,
):
    """
    Build an HTML file with a slider to scrub through trajectory frames.
    Open in a browser and drag the progress bar to any frame.
    """
    nodes, sizes, node_colors = _compute_node_sizes_and_colors(
        G, node_freq, node_cost, partition, color_mode=color_mode
    )
    if len(nodes) == 0:
        return

    node_order = _node_order_from_trajectory(edge_sequence, nodes)
    indices = _get_trajectory_frame_indices(node_order, nodes_per_frame, frames)

    if not indices:
        return

    node_to_size = dict(zip(nodes, sizes))
    node_to_color = dict(zip(nodes, node_colors))

    # Encode each frame as PNG base64
    b64_frames = []
    for idx in indices:
        image = _draw_one_trajectory_frame(
            G, pos, node_order, idx, node_to_size, node_to_color
        )
        buf = io.BytesIO()
        imageio.imwrite(buf, image, format="png")
        b64_frames.append(base64.standard_b64encode(buf.getvalue()).decode("ascii"))

    n_frames = len(b64_frames)
    frames_js = json.dumps(b64_frames)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>STN Trajectory</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }}
    .controls {{ display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }}
    input[type="range"] {{ flex: 1; min-width: 200px; accent-color: #4a9; }}
    #frameInfo {{ min-width: 100px; }}
    #img {{ max-width: 100%; height: auto; display: block; }}
  </style>
</head>
<body>
  <h2>STN Trajectory (drag slider to scrub)</h2>
  <div class="controls">
    <input type="range" id="slider" min="0" max="{n_frames - 1}" value="0" step="1">
    <span id="frameInfo">Frame 1 / {n_frames}</span>
  </div>
  <img id="img" alt="frame">
  <script>
    const frames = {frames_js};
    const slider = document.getElementById("slider");
    const img = document.getElementById("img");
    const frameInfo = document.getElementById("frameInfo");
    function showFrame(i) {{
      i = Math.max(0, Math.min(i, frames.length - 1));
      img.src = "data:image/png;base64," + frames[i];
      frameInfo.textContent = "Frame " + (i + 1) + " / " + frames.length;
    }}
    slider.addEventListener("input", function() {{ showFrame(parseInt(this.value, 10)); }});
    showFrame(0);
  </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Interactive HTML (with slider) saved to: {html_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build and visualize Search Trajectory Network from trajectory.jsonl"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="trajectory.jsonl",
        help="Path to trajectory.jsonl",
    )
    parser.add_argument(
        "--static_out",
        type=str,
        default="stn_final.png",
        help="Output path for the final static figure",
    )
    parser.add_argument(
        "--gif_out",
        type=str,
        default="stn_trajectory.gif",
        help="Output path for the GIF",
    )
    parser.add_argument(
        "--html_out",
        type=str,
        default=None,
        metavar="PATH",
        help="Output path for interactive HTML with slider (e.g. stn_trajectory.html)",
    )
    parser.add_argument(
        "--color_mode",
        type=str,
        default="cost",
        choices=["cost", "community"],
        help="Node coloring: 'cost' or 'community'",
    )
    parser.add_argument(
        "--louvain_resolution",
        type=float,
        default=1.0,
        help="Louvain resolution (higher -> more/smaller communities)",
    )
    parser.add_argument(
        "--layout_iterations",
        type=int,
        default=300,
        help="spring_layout iterations (higher -> more stable, slower)",
    )
    parser.add_argument(
        "--community_scale",
        type=float,
        default=10.0,
        help="How far apart to separate communities (higher -> less overlap)",
    )
    parser.add_argument(
        "--intra_scale",
        type=float,
        default=1.0,
        help="Scale inside each community (higher -> less intra-community overlap)",
    )
    parser.add_argument(
        "--k_community",
        type=float,
        default=None,
        help="spring_layout k for community super-graph (default: auto)",
    )
    parser.add_argument(
        "--k_intra",
        type=float,
        default=None,
        help="spring_layout k inside each community (default: auto)",
    )
    parser.add_argument(
        "--community_placement",
        type=str,
        default="circle",
        choices=["spring", "circle"],
        help="How to place communities globally: 'spring' or 'circle' (circle = maximally separated)",
    )
    parser.add_argument(
        "--gif_frames",
        type=int,
        default=None,
        metavar="N",
        help="Max GIF frames (subsample if more than N)",
    )
    parser.add_argument(
        "--nodes_per_frame",
        type=int,
        default=2,
        metavar="K",
        help="How many new nodes to add per GIF frame (default 2)",
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N runs from the trajectory (default: all)",
    )
    parser.add_argument(
        "--top_nodes",
        type=int,
        default=None,
        metavar="K",
        help="Keep only the top-K most frequent nodes (to reduce clutter)",
    )
    args = parser.parse_args()

    G, edge_sequence, node_freq, node_cost = load_trajectory_build_graph(
        args.input, max_runs=args.max_runs
    )

    # Optionally keep only top-k most frequent nodes to reduce clutter
    if hasattr(args, "top_nodes") and args.top_nodes is not None:
        k = args.top_nodes
        if k > 0 and len(node_freq) > k:
            # Select top-k by frequency
            sorted_nodes = sorted(
                node_freq.items(), key=lambda kv: kv[1], reverse=True
            )
            keep = set([n for n, _ in sorted_nodes[:k]])
            G = G.subgraph(keep).copy()
            node_freq = {n: node_freq[n] for n in keep}
            node_cost = {n: node_cost[n] for n in keep}
            edge_sequence = [
                (u, v) for (u, v) in edge_sequence if u in keep and v in keep
            ]

    if len(G) == 0:
        print("Graph is empty (no nodes); check trajectory.jsonl.")
        return

    partition, pos = compute_partition_and_layout(
        G,
        resolution=args.louvain_resolution,
        community_scale=args.community_scale,
        intra_scale=args.intra_scale,
        k_community=args.k_community,
        k_intra=args.k_intra,
        iterations=args.layout_iterations,
        placement=args.community_placement,
    )

    draw_static_stn(
        G,
        pos,
        node_freq,
        node_cost,
        partition,
        out_path=args.static_out,
        color_mode=args.color_mode,
    )

    make_trajectory_gif(
        G,
        pos,
        edge_sequence,
        node_freq,
        node_cost,
        partition,
        gif_path=args.gif_out,
        color_mode=args.color_mode,
        frames=args.gif_frames,
        nodes_per_frame=args.nodes_per_frame,
    )

    if args.html_out:
        make_trajectory_html(
            G,
            pos,
            edge_sequence,
            node_freq,
            node_cost,
            partition,
            html_path=args.html_out,
            color_mode=args.color_mode,
            frames=args.gif_frames,
            nodes_per_frame=args.nodes_per_frame,
        )


if __name__ == "__main__":
    main()

