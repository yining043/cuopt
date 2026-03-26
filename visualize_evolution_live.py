#!/usr/bin/env python3
"""Visualize cuOpt evolution trace: weight changes + parent/offspring routes.

Reads trace.csv and coords.csv produced by diverse_solver's trace output.

Modes:
  Post-hoc (default): read all data, generate weights plot + route GIF/HTML.
  Live (--live):      poll trace.csv in a loop, update weights plot and latest
                      route frame image in real time.  Does NOT block the solver.

Generates:
  - weights_evolution.png: line chart of each weight dimension over steps
  - route_latest.png: most recent 4-panel route snapshot (live mode)
  - route_evolution.gif / route_evolution.html: animated route viz (post-hoc)
  - evolution_summary.csv: per-step cost summary
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIM_NAMES = ["DIST", "TIME", "CAP", "PRIZE", "TASKS", "SVCT", "MISMATCH", "BREAK", "VFCOST"]
ROLE_ORDER = ["parent1", "parent2", "offspring_pre_ls", "offspring_post_ls"]
ROLE_TITLES = {
    "parent1": "Parent A",
    "parent2": "Parent B",
    "offspring_pre_ls": "Offspring (pre-LS)",
    "offspring_post_ls": "Offspring (post-LS)",
}
ROLE_COLORS = {
    "parent1": "#1f77b4",
    "parent2": "#ff7f0e",
    "offspring_pre_ls": "#2ca02c",
    "offspring_post_ls": "#d62728",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_coords(trace_dir):
    """Load node coordinates from coords.csv. Returns dict {node_id: (x, y)}."""
    path = os.path.join(trace_dir, "coords.csv")
    coords = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            coords[int(row["node_id"])] = (float(row["x"]), float(row["y"]))
    return coords


def parse_trace(trace_dir):
    """Parse trace.csv into weight records and route records."""
    path = os.path.join(trace_dir, "trace.csv")
    weights = []
    routes = defaultdict(lambda: defaultdict(list))
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if parts[0] == "W":
                step = int(parts[1])
                phase = parts[2]
                dims = [float(x) for x in parts[3:12]]
                coeff = float(parts[12]) if len(parts) > 12 else 0.0
                weights.append({"step": step, "phase": phase, "dims": dims, "coeff": coeff})
            elif parts[0] == "R":
                step = int(parts[1])
                role = parts[2]
                cost = float(parts[3])
                feasible = int(parts[4])
                n_routes = int(parts[5])
                route_id = int(parts[6])
                vehicle_id = int(parts[7])
                nodes = [int(x) for x in parts[8:] if x]
                routes[step][role].append((route_id, vehicle_id, cost, feasible, nodes))
    return weights, dict(routes)


def parse_trace_incremental(trace_path, file_offset):
    """Read new lines from trace.csv starting at file_offset.

    Returns (new_weights, new_routes, new_offset).
    """
    new_weights = []
    new_routes = defaultdict(lambda: defaultdict(list))
    try:
        with open(trace_path, "r") as f:
            f.seek(file_offset)
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                try:
                    if parts[0] == "W":
                        step = int(parts[1])
                        phase = parts[2]
                        dims = [float(x) for x in parts[3:12]]
                        coeff = float(parts[12]) if len(parts) > 12 else 0.0
                        new_weights.append({"step": step, "phase": phase, "dims": dims, "coeff": coeff})
                    elif parts[0] == "R":
                        step = int(parts[1])
                        role = parts[2]
                        cost = float(parts[3])
                        feasible = int(parts[4])
                        n_routes = int(parts[5])
                        route_id = int(parts[6])
                        vehicle_id = int(parts[7])
                        nodes = [int(x) for x in parts[8:] if x]
                        new_routes[step][role].append((route_id, vehicle_id, cost, feasible, nodes))
                except (ValueError, IndexError):
                    continue
            new_offset = f.tell()
    except FileNotFoundError:
        return [], {}, file_offset
    return new_weights, dict(new_routes), new_offset


# ---------------------------------------------------------------------------
# Weight plotting
# ---------------------------------------------------------------------------

def plot_weights(weights, output_dir):
    """Plot weight evolution over steps."""
    if not weights:
        return

    before = [w for w in weights if w["phase"] == "before"]
    after = [w for w in weights if w["phase"] == "after"]
    if not after:
        return

    steps_a = [w["step"] for w in after]
    steps_b = [w["step"] for w in before]

    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
    fig.suptitle("Weight Evolution During Optimization (log scale)", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(range(9))
    for idx, (ax, name) in enumerate(zip(axes.flat, DIM_NAMES)):
        vals_before = [w["dims"][idx] for w in before]
        vals_after = [w["dims"][idx] for w in after]
        ax.plot(steps_b, vals_before, "o-", color=colors[idx], alpha=0.3, markersize=2, label="before")
        ax.plot(steps_a, vals_after, "s-", color=colors[idx], markersize=2, label="after")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    for ax in axes[-1]:
        ax.set_xlabel("Step")
    for ax in axes[:, 0]:
        ax.set_ylabel("Weight")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "weights_evolution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # adjust_coeff_weights
    coeffs = [w["coeff"] for w in after]
    if any(c != 0 for c in coeffs):
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(steps_a, coeffs, "k-", linewidth=1)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("adjust_coeff_weights")
        ax2.set_title("Adjustment Coefficient Over Time")
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(os.path.join(output_dir, "adjust_coeff_evolution.png"), dpi=150, bbox_inches="tight")
        plt.close(fig2)


# ---------------------------------------------------------------------------
# Route drawing
# ---------------------------------------------------------------------------

def _route_data_edges(route_data, depot_id):
    """Build set of directed edges (a, b) from route_data (depot->n1->...->depot)."""
    edges = set()
    for _, _, _, _, nodes in route_data:
        if not nodes:
            continue
        path = [depot_id] + list(nodes) + [depot_id]
        for i in range(len(path) - 1):
            edges.add((path[i], path[i + 1]))
    return edges


def draw_routes_panel(ax, coords, route_data, role, step, depot_id=0, step_data=None):
    """Draw routes on a single axes panel.

    For offspring_pre_ls and offspring_post_ls, if step_data is provided, the full route
    is drawn with low alpha and nodes/edges that differ from both parents are drawn
    opaque in the same route color.
    """
    title = ROLE_TITLES.get(role, role)

    if not route_data:
        ax.set_title(f"{title}\n(no data)", fontsize=9)
        ax.set_aspect("equal")
        return

    cost = route_data[0][2]
    feasible = route_data[0][3]
    feas_str = "F" if feasible else "INF"

    all_x = [c[0] for c in coords.values()]
    all_y = [c[1] for c in coords.values()]
    ax.scatter(all_x, all_y, s=8, c="#cccccc", zorder=1)

    if depot_id in coords:
        dx, dy = coords[depot_id]
        ax.scatter([dx], [dy], s=80, c="red", marker="*", zorder=5,
                   edgecolors="black", linewidths=0.5)

    # Offspring panels: highlight edges/nodes not in either parent
    is_offspring = role in ("offspring_pre_ls", "offspring_post_ls")
    parent_edges = set()
    if is_offspring and step_data:
        for pr in ("parent1", "parent2"):
            pr_data = step_data.get(pr, [])
            if pr_data:
                parent_edges |= _route_data_edges(pr_data, depot_id)
    offspring_edges = _route_data_edges(route_data, depot_id)
    different_edges = offspring_edges - parent_edges if parent_edges else set()
    different_nodes = set()
    for (a, b) in different_edges:
        different_nodes.add(a)
        different_nodes.add(b)
    if depot_id in different_nodes:
        different_nodes.discard(depot_id)

    route_colors = plt.cm.Set2(np.linspace(0, 1, max(len(route_data), 1)))
    alpha_full = 0.25 if (is_offspring and different_edges) else 0.8
    alpha_diff = 1.0

    for ridx, (route_id, vid, _, _, nodes) in enumerate(route_data):
        if not nodes:
            continue
        rc = route_colors[ridx % len(route_colors)]
        path_ids = [depot_id] + nodes + [depot_id]
        px = [coords[n][0] for n in path_ids if n in coords]
        py = [coords[n][1] for n in path_ids if n in coords]
        ax.plot(px, py, "-", color=rc, linewidth=1.2, alpha=alpha_full, zorder=2)
        nx = [coords[n][0] for n in nodes if n in coords]
        ny = [coords[n][1] for n in nodes if n in coords]
        ax.scatter(nx, ny, s=15, c=[rc], zorder=3, edgecolors="black", linewidths=0.3, alpha=alpha_full)

        # Overlay different edges and nodes for offspring panels
        if is_offspring and different_edges:
            for i in range(len(path_ids) - 1):
                u, v = path_ids[i], path_ids[i + 1]
                if (u, v) in different_edges and u in coords and v in coords:
                    ax.plot([coords[u][0], coords[v][0]], [coords[u][1], coords[v][1]],
                            "-", color=rc, linewidth=2.0, alpha=alpha_diff, zorder=4)
            diff_nx = [coords[n][0] for n in nodes if n in different_nodes and n in coords]
            diff_ny = [coords[n][1] for n in nodes if n in different_nodes and n in coords]
            if diff_nx and diff_ny:
                ax.scatter(diff_nx, diff_ny, s=22, c=rc, zorder=5,
                           edgecolors="black", linewidths=0.5, alpha=alpha_diff)

    ax.set_title(f"{title}\ncost={cost:.1f} [{feas_str}] R={len(route_data)}", fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)


def draw_route_frame(coords, step_data, step, out_path, depot_id=0):
    """Draw a single 4-panel route frame and save to out_path."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Step {step}", fontsize=13, fontweight="bold")
    for ax, role in zip(axes, ROLE_ORDER):
        draw_routes_panel(ax, coords, step_data.get(role, []), role, step, depot_id=depot_id, step_data=step_data)
    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def generate_route_frames(coords, routes, output_dir, max_frames=200):
    """Generate individual frame PNGs for route evolution."""
    frames_dir = os.path.join(output_dir, "_route_frames")
    os.makedirs(frames_dir, exist_ok=True)

    sorted_steps = sorted(routes.keys())
    if not sorted_steps:
        return [], frames_dir

    if max_frames > 0 and len(sorted_steps) > max_frames:
        indices = np.linspace(0, len(sorted_steps) - 1, max_frames, dtype=int)
        sorted_steps = [sorted_steps[i] for i in indices]

    frame_paths = []
    for frame_idx, step in enumerate(sorted_steps):
        fpath = os.path.join(frames_dir, f"frame_{frame_idx:05d}.png")
        draw_route_frame(coords, routes[step], step, fpath)
        frame_paths.append(fpath)
        if (frame_idx + 1) % 50 == 0:
            print(f"[viz] Generated {frame_idx + 1}/{len(sorted_steps)} route frames", flush=True)

    print(f"[viz] Generated {len(frame_paths)} route frames in {frames_dir}/")
    return frame_paths, frames_dir


# ---------------------------------------------------------------------------
# GIF / HTML
# ---------------------------------------------------------------------------

def create_gif(frame_paths, output_dir, fps=4):
    if not frame_paths:
        return
    try:
        from PIL import Image
    except ImportError:
        print("[viz] PIL not available, skipping GIF. pip install Pillow", file=sys.stderr)
        return

    out_path = os.path.join(output_dir, "route_evolution.gif")
    duration = int(1000 / fps)

    def append_gen():
        for fp in frame_paths[1:]:
            im = Image.open(fp)
            yield im
            im.close()

    first = Image.open(frame_paths[0])
    try:
        first.save(out_path, save_all=True, append_images=append_gen(),
                   duration=duration, loop=0, optimize=True)
    finally:
        first.close()
    print(f"[viz] Saved GIF: {out_path} ({len(frame_paths)} frames, {fps} fps)")


# Threshold above which we use local image paths in HTML instead of base64 (avoids huge file + too many open files)
HTML_EMBED_FRAME_LIMIT = 2000


def create_html(frame_paths, output_dir, fps=4, embed_images=False):
    if not frame_paths:
        return

    out_path = os.path.join(output_dir, "route_evolution.html")
    n = len(frame_paths)

    if embed_images or n <= HTML_EMBED_FRAME_LIMIT:
        if embed_images and n > 10000:
            print(f"[viz] Embedding {n} images (may take several minutes, HTML will be large)...", flush=True)
        import base64
        encoded = []
        for i, fp in enumerate(frame_paths):
            with open(fp, "rb") as f:
                encoded.append(base64.b64encode(f.read()).decode("ascii"))
            if embed_images and n > 5000 and (i + 1) % 5000 == 0:
                print(f"[viz] Embedded {i + 1}/{n} images...", flush=True)
        frames_js = str(encoded)
        src_prefix = '"data:image/png;base64," + frames[idx]'
        open_hint = ""
    else:
        # Use local paths (forward slashes for URLs)
        rel_paths = [os.path.relpath(fp, output_dir).replace(os.sep, "/") for fp in frame_paths]
        frames_js = str(rel_paths)
        src_prefix = "frames[idx]"
        out_abs = os.path.abspath(output_dir)
        open_hint = (
            '<p style="font-size:13px;color:#c00;margin:8px;padding:8px;background:#ffe0e0;border-radius:6px;">'
            '<b>Images load only over HTTP.</b> In a terminal run: '
            '<code>cd &quot;' + out_abs + '&quot; &amp;&amp; python -m http.server 8000</code>, '
            'then open <a href="http://localhost:8000/route_evolution.html" target="_blank">http://localhost:8000/route_evolution.html</a> in your browser.'
            '</p>'
        )
    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Route Evolution Animation</title>
<style>
  body {{ font-family: sans-serif; text-align: center; background: #f5f5f5; }}
  #frame {{ max-width: 100%; border: 1px solid #ccc; margin: 10px auto; display: block; }}
  .controls {{ margin: 10px; }}
  button {{ padding: 8px 16px; margin: 0 4px; cursor: pointer; font-size: 14px; }}
  #slider {{ width: 60%; }}
  #info {{ font-size: 14px; color: #555; margin: 5px; }}
</style>
</head><body>
<h2>Route Evolution: Parent &rarr; Offspring &rarr; Post-LS</h2>
{open_hint}
<img id="frame" src="">
<div id="info">Frame 1 / {n}</div>
<div class="controls">
  <button onclick="prev()">&#9664; Prev</button>
  <button id="playBtn" onclick="togglePlay()">&#9654; Play</button>
  <button onclick="next()">Next &#9654;</button>
  <input id="slider" type="range" min="0" max="{n-1}" value="0" oninput="goTo(this.value)">
  <label>Speed: <input id="speed" type="range" min="1" max="20" value="{fps}" oninput="updateSpeed(this.value)"> <span id="fpsLabel">{fps}</span> fps</label>
</div>
<script>
const frames = {frames_js};
let idx = 0, playing = false, timer = null, fps = {fps};
function show(i) {{
  idx = Math.max(0, Math.min(frames.length-1, i));
  document.getElementById("frame").src = {src_prefix};
  document.getElementById("slider").value = idx;
  document.getElementById("info").textContent = "Frame " + (idx+1) + " / " + frames.length;
}}
function next() {{ show(idx+1); }}
function prev() {{ show(idx-1); }}
function goTo(v) {{ show(parseInt(v)); }}
function togglePlay() {{
  playing = !playing;
  document.getElementById("playBtn").textContent = playing ? "\\u23F8 Pause" : "\\u25B6 Play";
  if (playing) timer = setInterval(() => {{ if(idx >= frames.length-1) {{ idx=-1; }} next(); }}, 1000/fps);
  else clearInterval(timer);
}}
function updateSpeed(v) {{ fps=parseInt(v); document.getElementById("fpsLabel").textContent=fps; if(playing){{ clearInterval(timer); timer=setInterval(()=>{{ if(idx>=frames.length-1){{idx=-1;}} next(); }}, 1000/fps); }} }}
show(0);
</script>
</body></html>"""

    with open(out_path, "w") as f:
        f.write(html)
    print(f"[viz] Saved HTML animation: {out_path} ({n} frames)"
          + (" (images embedded, open file directly)" if (embed_images or n <= HTML_EMBED_FRAME_LIMIT) else ""))
    if not embed_images and n > HTML_EMBED_FRAME_LIMIT:
        print("[viz] Open via HTTP so images load: cd to that dir, run "
              "python -m http.server 8000, then open http://localhost:8000/route_evolution.html")


# ---------------------------------------------------------------------------
# Live mode: poll trace.csv, update plots incrementally
# ---------------------------------------------------------------------------

def run_live(trace_dir, output_dir, poll_interval=2.0):
    """Poll trace.csv and update plots as new data arrives."""
    trace_path = os.path.join(trace_dir, "trace.csv")
    coords_path = os.path.join(trace_dir, "coords.csv")

    # Wait for coords.csv to appear
    while not os.path.isfile(coords_path):
        time.sleep(0.5)
    coords = load_coords(trace_dir)
    print(f"[viz-live] Loaded {len(coords)} node coordinates", flush=True)

    all_weights = []
    all_routes = {}
    file_offset = 0
    last_step = -1
    n_updates = 0

    while True:
        if not os.path.isfile(trace_path):
            time.sleep(poll_interval)
            continue

        new_w, new_r, new_offset = parse_trace_incremental(trace_path, file_offset)
        if new_offset == file_offset and not new_w and not new_r:
            time.sleep(poll_interval)
            continue

        file_offset = new_offset
        all_weights.extend(new_w)
        for step, roles in new_r.items():
            if step not in all_routes:
                all_routes[step] = {}
            for role, rdata in roles.items():
                all_routes[step][role] = rdata

        n_updates += 1

        # Update weights plot every time
        try:
            plot_weights(all_weights, output_dir)
        except Exception as e:
            print(f"[viz-live] Weight plot error: {e}", file=sys.stderr, flush=True)

        # Update route snapshot for the latest step
        if all_routes:
            latest_step = max(all_routes.keys())
            if latest_step != last_step:
                last_step = latest_step
                out_path = os.path.join(output_dir, "route_latest.png")
                try:
                    draw_route_frame(coords, all_routes[latest_step], latest_step, out_path)
                except Exception as e:
                    print(f"[viz-live] Route frame error: {e}", file=sys.stderr, flush=True)

        if n_updates % 10 == 0:
            print(f"[viz-live] {len(all_weights)} weight records, "
                  f"{len(all_routes)} route steps, latest step={last_step}", flush=True)

        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Post-hoc mode
# ---------------------------------------------------------------------------

def run_from_existing_frames(output_dir, fps=4, no_gif=False, no_html=False, embed_images=False):
    """Build GIF/HTML from existing _route_frames/*.png (recovery after failure)."""
    frames_dir = os.path.join(output_dir, "_route_frames")
    if not os.path.isdir(frames_dir):
        print(f"[viz] No _route_frames dir in {output_dir}", file=sys.stderr)
        return
    frame_paths = sorted(
        [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png")],
        key=lambda p: os.path.basename(p),
    )
    if not frame_paths:
        print(f"[viz] No PNG frames in {frames_dir}", file=sys.stderr)
        return
    print(f"[viz] Using {len(frame_paths)} existing frames from {frames_dir}", flush=True)
    if not no_gif:
        create_gif(frame_paths, output_dir, fps=fps)
    if not no_html:
        create_html(frame_paths, output_dir, fps=fps, embed_images=embed_images)


def run_posthoc(trace_dir, output_dir, max_frames=200, fps=4,
                no_gif=False, no_html=False, embed_images=False):
    """Read all trace data and generate final outputs."""
    coords = load_coords(trace_dir)
    weights, routes = parse_trace(trace_dir)
    print(f"[viz] Loaded {len(coords)} nodes, {len(weights)} weight records, "
          f"{len(routes)} route steps", flush=True)

    plot_weights(weights, output_dir)

    if routes and coords:
        frame_paths, frames_dir = generate_route_frames(
            coords, routes, output_dir, max_frames=max_frames)
        if frame_paths:
            if not no_gif:
                create_gif(frame_paths, output_dir, fps=fps)
            if not no_html:
                create_html(frame_paths, output_dir, fps=fps, embed_images=embed_images)

            summary_path = os.path.join(output_dir, "evolution_summary.csv")
            with open(summary_path, "w") as sf:
                sf.write("step,role,cost,feasible,n_routes\n")
                for step in sorted(routes.keys()):
                    for role in ROLE_ORDER:
                        if role in routes[step] and routes[step][role]:
                            r0 = routes[step][role][0]
                            sf.write(f"{step},{role},{r0[2]:.4f},{r0[3]},"
                                     f"{len(routes[step][role])}\n")
            print(f"[viz] Saved summary: {summary_path}")
    else:
        print("[viz] No route/coordinate data for route visualization.", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize cuOpt evolution trace")
    parser.add_argument("--trace_dir", required=False, default=None,
                        help="Directory containing trace.csv and coords.csv (or output dir for --use_existing_frames)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: same as trace_dir)")
    parser.add_argument("--max_frames", type=int, default=-1,
                        help="Max route frames for GIF/HTML (post-hoc). Use -1 for no limit.")
    parser.add_argument("--fps", type=int, default=4, help="Animation FPS")
    parser.add_argument("--no_gif", action="store_true", help="Skip GIF generation")
    parser.add_argument("--no_html", action="store_true", help="Skip HTML generation")
    parser.add_argument("--embed_images", action="store_true",
                        help="Embed all frames as base64 in HTML (single file, open anywhere; slow and large for 45k+ frames)")
    parser.add_argument("--live", action="store_true",
                        help="Live mode: poll trace.csv and update plots in real time")
    parser.add_argument("--poll_interval", type=float, default=2.0,
                        help="Live mode poll interval in seconds (default 2.0)")
    parser.add_argument("--use_existing_frames", action="store_true",
                        help="Recovery: build GIF/HTML only from existing _route_frames/*.png (no trace parsing)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.trace_dir
    if not output_dir:
        parser.error("--trace_dir or --output_dir required (or both; for --use_existing_frames, pass dir that contains _route_frames)")
    os.makedirs(output_dir, exist_ok=True)

    if args.use_existing_frames:
        print(f"[viz] Recovery mode: building GIF/HTML from existing frames in {output_dir}/", flush=True)
        run_from_existing_frames(output_dir, fps=args.fps, no_gif=args.no_gif, no_html=args.no_html,
                                 embed_images=args.embed_images)
    elif args.live:
        if not args.trace_dir:
            parser.error("--trace_dir required for --live")
        print(f"[viz-live] Starting live monitor on {args.trace_dir}/", flush=True)
        run_live(args.trace_dir, output_dir, poll_interval=args.poll_interval)
    else:
        if not args.trace_dir:
            parser.error("--trace_dir required for post-hoc mode")
        print(f"[viz] Loading trace from {args.trace_dir}/", flush=True)
        run_posthoc(args.trace_dir, output_dir,
                    max_frames=args.max_frames, fps=args.fps,
                    no_gif=args.no_gif, no_html=args.no_html, embed_images=args.embed_images)
    print("[viz] Done.", flush=True)


if __name__ == "__main__":
    main()
