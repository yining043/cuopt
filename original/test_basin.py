import sys
import os

# IMPORTANT: CUDA_VISIBLE_DEVICES must be set BEFORE importing cuOpt
# If not set, cuOpt will use the default device (usually GPU 0)
# Check if CUDA_VISIBLE_DEVICES is set, if not, print a warning
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    print("WARNING: CUDA_VISIBLE_DEVICES is not set. cuOpt will use the default GPU (usually GPU 0).")
    print("To use a specific GPU, set it before running: CUDA_VISIBLE_DEVICES=3 python original/test_basin.py ...")
else:
    print(f"Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

# Set library paths to ensure correct nvJitLink library is found
# This fixes the ImportError: undefined symbol: __nvJitLinkGetErrorLogSize_12_9
conda_env_path = os.environ.get('CONDA_PREFIX', '/home/jieyi/.conda/envs/cuopt')
nvjitlink_lib = os.path.join(conda_env_path, 'targets/x86_64-linux/lib/libnvJitLink.so.12.9.86')
lib_path = os.path.join(conda_env_path, 'targets/x86_64-linux/lib')
env_lib_path = os.path.join(conda_env_path, 'lib')

# Set LD_LIBRARY_PATH
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
new_ld_path = f"{lib_path}:{env_lib_path}"
if current_ld_path:
    new_ld_path = f"{new_ld_path}:{current_ld_path}"
os.environ['LD_LIBRARY_PATH'] = new_ld_path

# Preload the correct nvJitLink library using ctypes
if os.path.exists(nvjitlink_lib):
    try:
        import ctypes
        ctypes.CDLL(nvjitlink_lib, ctypes.RTLD_GLOBAL)
    except Exception as e:
        print(f"Warning: Could not preload {nvjitlink_lib}: {e}", file=sys.stderr)
import datetime
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from test_landscape import test_callback, run_multiple_instances
from utils import *
import argparse


def _edges_hash(edges):
    """Helper function to compute stable hash for edge set."""
    if not edges:
        return ""
    edge_str = ";".join(f"{int(u)}-{int(v)}" for (u, v) in sorted(edges))
    return hashlib.sha1(edge_str.encode("utf-8")).hexdigest()


def _save_basin_jsonl_dataset(
    history,
    trial_solutions,
    trial_metadata,
    unique_optima,
    optimum_id_map,
    num_orders,
    instance_path,
    instance_index,
    hgs_solution,
    result_summary,
    output_dir,
    run_id,
):
    """Save basin dataset as three JSONL files: optima, trials, trajectory.

    Also appends to global instance-level JSONL files so multiple runs on the
    same instance accumulate into one dataset.
    """
    os.makedirs(output_dir, exist_ok=True)

    instance_id = (
        f"{os.path.basename(instance_path)}#{instance_index}"
        if instance_path
        else f"random#{instance_index}"
    )

    # Prepare HGS reference (edges + cost) if available
    hgs_edges = None
    hgs_cost = None
    if hgs_solution is not None:
        hgs_routes = convert_hgs_routes_to_format(hgs_solution["hgs_routes"])
        hgs_edges = extract_edges_from_routes(hgs_routes)
        hgs_cost = hgs_solution["hgs_cost"]

    # Use module-level _edges_hash function

    # Map optimum_id -> edges / cost / hash
    optimum_edges = {}
    optimum_cost = {}
    optimum_edges_hash = {}
    for opt in unique_optima:
        opt_id = opt.get("optimum_id")
        edges = opt.get("edges", set())
        meta = opt.get("metadata", {})
        cost = meta.get("cost")
        optimum_edges[opt_id] = edges
        optimum_cost[opt_id] = cost
        optimum_edges_hash[opt_id] = _edges_hash(edges)

    # Map trial_id -> optimum_id
    trial_id_to_optimum_id = {}
    for sol, meta in zip(trial_solutions, trial_metadata):
        trial_id = meta.get("local_search_id", -1)
        if trial_id is None:
            trial_id = -1

        num_routes = meta.get("num_routes_after", 0) or 0
        inferred_num_routes = infer_num_routes_from_solution_flat(sol, num_orders)
        if inferred_num_routes > 0:
            num_routes = inferred_num_routes

        routes = solution_flat_to_routes(sol, num_routes, num_orders)
        edges = extract_edges_from_routes(routes)
        edges_tuple = tuple(sorted(edges))
        opt_id = optimum_id_map.get(edges_tuple)
        if opt_id is not None:
            trial_id_to_optimum_id[trial_id] = opt_id

    # Per-trial statistics from history
    trial_id_to_last_local_iter = {}
    trial_id_to_step_count = {}
    for record in history:
        trial_id = record.get("local_search_id", -1)
        if trial_id is None:
            trial_id = -1
        local_iter = record.get("local_iter", -1)
        if local_iter is None:
            local_iter = -1
        trial_id_to_last_local_iter[trial_id] = max(
            trial_id_to_last_local_iter.get(trial_id, -1), local_iter
        )
        trial_id_to_step_count[trial_id] = trial_id_to_step_count.get(trial_id, 0) + 1

    # Local (per-run) JSONL paths
    optima_path = os.path.join(output_dir, "optima.jsonl")
    trials_path = os.path.join(output_dir, "trials.jsonl")
    traj_path = os.path.join(output_dir, "trajectory.jsonl")

    # Global (across runs for this instance) JSONL paths
    global_root = os.path.join(os.getcwd(), "basin_datasets")
    global_instance_dir = os.path.join(global_root, instance_id)
    os.makedirs(global_instance_dir, exist_ok=True)
    global_optima_path = os.path.join(global_instance_dir, "optima.jsonl")
    global_trials_path = os.path.join(global_instance_dir, "trials.jsonl")
    global_traj_path = os.path.join(global_instance_dir, "trajectory.jsonl")

    # 1) optima.jsonl
    with open(optima_path, "w", encoding="utf-8") as f_opt, open(
        global_optima_path, "a", encoding="utf-8"
    ) as f_opt_global:
        for opt in unique_optima:
            opt_id = opt.get("optimum_id")
            edges = opt.get("edges", set())
            meta = opt.get("metadata", {})

            edges_list = [[int(u), int(v)] for (u, v) in sorted(edges)]

            edge_diff_to_hgs = None
            cost_gap_to_hgs_pct = None
            if hgs_edges is not None:
                edge_diff_to_hgs = len(edges ^ hgs_edges)
            if hgs_cost is not None:
                cost = meta.get("cost")
                if cost is not None:
                    cost_gap_to_hgs_pct = calculate_gap(cost, hgs_cost)

            record = {
                "instance_id": instance_id,
                "run_id": run_id,
                "optimum_id": opt_id,
                "final_cost": meta.get("cost"),
                "num_routes": meta.get("num_routes_after", 0) or 0,
                "num_orders": num_orders,
                "edges": edges_list,
                "edges_hash": optimum_edges_hash.get(opt_id, ""),
                "edge_diff_to_hgs": edge_diff_to_hgs,
                "cost_gap_to_hgs_pct": cost_gap_to_hgs_pct,
            }
            f_opt.write(json.dumps(record) + "\n")
            f_opt_global.write(json.dumps(record) + "\n")

    # 2) trials.jsonl
    with open(trials_path, "w", encoding="utf-8") as f_trials, open(
        global_trials_path, "a", encoding="utf-8"
    ) as f_trials_global:
        for meta in trial_metadata:
            trial_id = meta.get("local_search_id", -1)
            if trial_id is None:
                trial_id = -1
            opt_id = trial_id_to_optimum_id.get(trial_id)

            final_cost = meta.get("cost")
            final_edge_diff_to_hgs = None
            final_cost_gap_to_hgs_pct = None
            if opt_id is not None and hgs_edges is not None:
                edges = optimum_edges.get(opt_id, set())
                final_edge_diff_to_hgs = len(edges ^ hgs_edges)
            if opt_id is not None and hgs_cost is not None:
                opt_cost = optimum_cost.get(opt_id)
                if opt_cost is not None:
                    final_cost_gap_to_hgs_pct = calculate_gap(opt_cost, hgs_cost)

            record = {
                "instance_id": instance_id,
                "run_id": run_id,
                "trial_id": trial_id,
                "optimum_id": opt_id,
                "final_cost": final_cost,
                "final_edge_diff_to_hgs": final_edge_diff_to_hgs,
                "final_cost_gap_to_hgs_pct": final_cost_gap_to_hgs_pct,
                "n_steps": trial_id_to_step_count.get(trial_id, 0),
                "optimum_edges_hash": optimum_edges_hash.get(opt_id, None),
            }
            f_trials.write(json.dumps(record) + "\n")
            f_trials_global.write(json.dumps(record) + "\n")

    # 3) trajectory.jsonl
    with open(traj_path, "w", encoding="utf-8") as f_traj, open(
        global_traj_path, "a", encoding="utf-8"
    ) as f_traj_global:
        sorted_history = sorted(
            history,
            key=lambda r: r.get("global_iter") if r.get("global_iter") is not None else -1,
        )

        for record in sorted_history:
            sol = record.get("sol_after")
            if sol is None or len(sol) == 0:
                continue

            trial_id = record.get("local_search_id", -1)
            if trial_id is None:
                trial_id = -1
            local_iter = record.get("local_iter", -1)
            if local_iter is None:
                local_iter = -1

            num_routes = record.get("num_routes_after", 0) or 0
            inferred_num_routes = infer_num_routes_from_solution_flat(sol, num_orders)
            if inferred_num_routes > 0:
                num_routes = inferred_num_routes

            routes = solution_flat_to_routes(sol, num_routes, num_orders)
            edges = extract_edges_from_routes(routes)
            edges_list = [[int(u), int(v)] for (u, v) in sorted(edges)]

            edge_diff_to_hgs = None
            cost_gap_to_hgs_pct = None
            cost = record.get("cost_after")
            if hgs_edges is not None:
                edge_diff_to_hgs = len(edges ^ hgs_edges)
            if hgs_cost is not None and cost is not None:
                cost_gap_to_hgs_pct = calculate_gap(cost, hgs_cost)

            record_out = {
                "instance_id": instance_id,
                "run_id": run_id,
                "trial_id": trial_id,
                "optimum_id": trial_id_to_optimum_id.get(trial_id),
                "global_iter": record.get("global_iter", -1),
                "local_iter": local_iter,
                "is_final_of_trial": local_iter
                == trial_id_to_last_local_iter.get(trial_id, -1),
                "cost": cost,
                "gap": result_summary.get("gap"),
                "edge_diff_to_hgs": edge_diff_to_hgs,
                "cost_gap_to_hgs_pct": cost_gap_to_hgs_pct,
                "is_cycle_finder": record.get("is_circle_found", False),
                "move_found": record.get("move_found", False),
                "num_routes_after": num_routes,
                "solution_flat": sol,
                "edges": edges_list,
                "edges_hash": _edges_hash(edges),
                "raw_history": record,
            }
            f_traj.write(json.dumps(record_out) + "\n")
            f_traj_global.write(json.dumps(record_out) + "\n")


def visualize_basin_from_jsonl(instance_path, instance_index, global_root="basin_datasets"):
    """Visualize basin datasets from aggregated JSONL files."""
    instance_id = (
        f"{os.path.basename(instance_path)}#{instance_index}"
        if instance_path
        else f"random#{instance_index}"
    )
    instance_dir = os.path.join(global_root, instance_id)
    optima_path = os.path.join(instance_dir, "optima.jsonl")
    traj_path = os.path.join(instance_dir, "trajectory.jsonl")

    if not os.path.exists(optima_path) or not os.path.exists(traj_path):
        print(f"No JSONL dataset found for instance_id={instance_id} under {global_root}")
        return

    # Load optima and trajectory records
    optima_raw = []
    with open(optima_path, "r", encoding="utf-8") as f:
        for line in f:
            optima_raw.append(json.loads(line))

    traj = []
    with open(traj_path, "r", encoding="utf-8") as f:
        for line in f:
            traj.append(json.loads(line))

    if not traj or not optima_raw:
        print("Empty dataset, nothing to visualize.")
        return

    # Helper: canonicalize edge list into a hashable basin key (undirected edge set)
    def edge_set_key(edges_list):
        if not edges_list:
            return None
        # ensure undirected and sorted
        norm_edges = [tuple(sorted((int(u), int(v)))) for (u, v) in edges_list]
        return frozenset(norm_edges)

    # ------------------------------------------------------------------
    # Step 1: merge local optima across runs by edge set (global basins)
    # ------------------------------------------------------------------
    # basin_key -> best-cost representative optima record
    optima_by_basin = {}
    # (run_id, optimum_id) -> basin_key
    runopt_to_basin = {}
    # Per-run stats for debugging/reporting
    per_run_stats = {}

    for r in optima_raw:
        edges = r.get("edges")
        basin_key = edge_set_key(edges)
        if basin_key is None:
            continue

        run_id = r.get("run_id")
        opt_id = r.get("optimum_id")
        if run_id is not None and opt_id is not None:
            runopt_to_basin[(run_id, opt_id)] = basin_key
            st = per_run_stats.setdefault(
                run_id, {"optima_count": 0, "basin_keys": set()}
            )
            st["optima_count"] += 1
            st["basin_keys"].add(basin_key)

        prev = optima_by_basin.get(basin_key)
        if prev is None:
            optima_by_basin[basin_key] = r
        else:
            c_new = r.get("final_cost")
            c_prev = prev.get("final_cost")
            if c_new is not None and (c_prev is None or c_new < c_prev):
                optima_by_basin[basin_key] = r

    # Print per-run basin stats before/after merging across runs
    if per_run_stats:
        print("\nBasin statistics per run (before/after merging across runs):")
        for run_id in sorted(per_run_stats.keys()):
            st = per_run_stats[run_id]
            print(
                f"  run_id={run_id}: "
                f"optima entries={st['optima_count']}, "
                f"unique basins in this run={len(st['basin_keys'])}"
            )
        print(f"Global merged basins across runs: {len(optima_by_basin)}")

    # ------------------------------------------------------------------
    # Step 2: prepare data for fitness-distance plot
    # ------------------------------------------------------------------
    # Intermediates: all trajectory points with valid distance
    traj_edge_diff = np.array(
        [r.get("edge_diff_to_hgs") for r in traj if r.get("edge_diff_to_hgs") is not None],
        dtype=float,
    )
    traj_cost = np.array(
        [r.get("cost") for r in traj if r.get("edge_diff_to_hgs") is not None],
        dtype=float,
    )

    # Local optima: one per basin (merged across runs)
    optima_merged = list(optima_by_basin.values())
    opt_edge_diff = np.array(
        [
            r.get("edge_diff_to_hgs")
            for r in optima_merged
            if r.get("edge_diff_to_hgs") is not None
        ],
        dtype=float,
    )
    opt_cost = np.array(
        [
            r.get("final_cost")
            for r in optima_merged
            if r.get("edge_diff_to_hgs") is not None
        ],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Step 3: basin widths = unique solutions per basin (across runs)
    # ------------------------------------------------------------------
    # basin_key -> set of unique solution edge sets
    basin_solutions = {}
    for r in traj:
        run_id = r.get("run_id")
        opt_id = r.get("optimum_id")
        if run_id is None or opt_id is None:
            continue
        basin_key = runopt_to_basin.get((run_id, opt_id))
        if basin_key is None:
            # this trial's optimum wasn't in optima_raw (shouldn't happen, but be safe)
            continue

        edges_sol = r.get("edges")
        sol_key = edge_set_key(edges_sol)
        if sol_key is None:
            continue

        basin_solutions.setdefault(basin_key, set()).add(sol_key)

    basin_keys = sorted(basin_solutions.keys(), key=lambda k: optima_by_basin[k].get("final_cost", float("inf")))
    basin_widths = [len(basin_solutions[k]) for k in basin_keys]
    basin_indices = list(range(len(basin_keys)))  # compressed basin ids for x-axis

    # Ensure output directory exists for saving visualizations
    os.makedirs(instance_dir, exist_ok=True)

    # Plot: Fitness-Distance landscape (2D)
    plt.figure(figsize=(7, 5))
    plt.scatter(
        traj_edge_diff,
        traj_cost,
        s=5,
        alpha=0.2,
        label="intermediate",
        color="tab:blue",
    )
    if len(opt_edge_diff) > 0:
        plt.scatter(
            opt_edge_diff,
            opt_cost,
            s=40,
            alpha=0.9,
            label="local optima",
            color="red",
            edgecolors="black",
        )
    plt.xlabel("Distance to HGS (edge_diff_to_hgs)")
    plt.ylabel("Cost")
    plt.title(f"Fitness–Distance Landscape ({instance_id})")
    plt.legend()
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    fd_path = os.path.join(instance_dir, "fitness_distance_landscape.png")
    plt.savefig(fd_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Plot: Basin width bar chart
    plt.figure(figsize=(7, 4))
    plt.bar(basin_indices, basin_widths, color="tab:green", alpha=0.7)
    plt.xlabel("Basin index (merged by edges_hash)")
    plt.ylabel("Unique solutions in basin")
    plt.title(f"Basin widths ({instance_id})")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    bw_path = os.path.join(instance_dir, "basin_widths.png")
    plt.savefig(bw_path, dpi=150, bbox_inches="tight")
    plt.close()


def analyze_basin_structure(result_data, instance_path, hgs_solution_path, instance_index, output_dir='plot', collect_dataset=False):
    """Analyze basin structure: extract local optima, identify unique ones, and plot."""
    # Create timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_output_dir = os.path.join(output_dir, f'basin_{timestamp}')
    os.makedirs(timestamped_output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Analyzing basin structure for instance {instance_index}")
    print(f"  Output directory: {timestamped_output_dir}")
    print("=" * 60)

    # Step 1: Get history and metadata
    history = result_data.get('history', [])
    result = result_data.get('result', {}) or {}
    gap = result.get('gap')
    num_orders = result.get('num_orders')
    print(f"\nStep 1: Collected {len(history)} history records")
    instance_data = load_instance_from_pkl(instance_path, instance_index) # Load instance data (for coordinates)

    # Step 2: Extract local optima
    print("\nStep 2: Extracting local optima...")
    temp_global_history = {'history': history}
    _, trial_solutions, trial_metadata = extract_last_solution_per_trial(temp_global_history, num_orders=None)
    print(f"  Found {len(trial_solutions)} local optima")

    # Step 3: Identify unique local optima
    print("\nStep 3: Identifying unique local optima...")
    unique_optima = []
    optimum_id_map = {}
    duplicate_groups = {}
    for idx, (opt_sol, opt_meta) in enumerate(zip(trial_solutions, trial_metadata)):
        num_routes = opt_meta.get('num_routes_after', 0) or 0
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
            unique_optima.append({
                'solution': opt_sol,
                'metadata': opt_meta,
                'edges': edges,
                'optimum_id': optimum_id,
            })
        else:
            duplicate_groups[edges_tuple].append(idx)
    print(f"  Found {len(unique_optima)} unique local optima")
    # Print duplicate information
    duplicates_found = False
    for edges_tuple, indices in duplicate_groups.items():
        if len(indices) > 1:
            duplicates_found = True
            unique_id = optimum_id_map[edges_tuple]
            print(f"  Duplicate: IDs {indices} map to unique optimum ID {unique_id}")
    if not duplicates_found: print("  No duplicates found (all local optima are unique)")

    # # Basin statistics + wandb logging (unique solutions per basin, distance/gap to HGS)
    # log_basin_stats_wandb(
    #     history=history,
    #     trial_solutions=trial_solutions,
    #     trial_metadata=trial_metadata,
    #     unique_optima=unique_optima,
    #     optimum_id_map=optimum_id_map,
    #     num_orders=num_orders,
    #     hgs_solution_path=hgs_solution_path,
    #     instance_index=instance_index,
    #     timestamp=timestamp,
    # )

    # Step 4: Plot cost curve
    print("\nStep 4: Plotting cost curve by trial...")
    plot_filename = os.path.join(timestamped_output_dir, 'callback_cost_curve_by_trial.png')
    plot_cost_curve_by_trial(
        temp_global_history,
        num_orders=num_orders,
        filename=plot_filename,
        trial_metadata=trial_metadata,
        duplicate_groups=duplicate_groups,
        optimum_id_map=optimum_id_map,
        gap=gap,
    )

    # Step 5: Plot best solution vs HGS solution
    print("\nStep 5: Plotting best solution vs HGS solution...")
    # Find best trial (minimum cost)
    best_idx = None
    best_cost = None
    for i, meta in enumerate(trial_metadata):
        c = meta.get('cost')
        if c is None:
            continue
        if best_cost is None or c < best_cost:
            best_cost = c
            best_idx = i
    if best_idx is not None and best_cost is not None:
        best_sol = trial_solutions[best_idx]
        best_meta = trial_metadata[best_idx]
        num_routes_best = best_meta.get('num_routes_after', 0) or 0
        inferred_num_routes_best = infer_num_routes_from_solution_flat(best_sol, num_orders)
        if inferred_num_routes_best > 0:
            num_routes_best = inferred_num_routes_best
        best_routes = solution_flat_to_routes(best_sol, num_routes_best, num_orders)
        hgs_solution = None
        if hgs_solution_path: hgs_solution = load_hgs_solution_from_pkl(hgs_solution_path, instance_index)
        comparison_filename = os.path.join(timestamped_output_dir, 'solution_comparison_hgs.png')
        compare_solutions(
            best_routes,
            best_cost,
            instance_data['coordinates'],
            hgs_solution=hgs_solution,
            filename=comparison_filename,
        )
    else:
        print("  No valid best trial found for comparison.")

    # # Step 6: Plot all unique local optima solutions in one figure
    # print("\nStep 6: Plotting all unique local optima solutions...")
    # unique_optima_filename = os.path.join(timestamped_output_dir, 'unique_optima_routes.png')
    # plot_unique_optima_solutions(
    #     unique_optima,
    #     num_orders=num_orders,
    #     instance_data=instance_data,
    #     filename=unique_optima_filename,
    # )

    # Step 7: Save JSONL dataset if requested
    if collect_dataset:
        print("\nStep 7: Saving basin dataset to JSONL...")
        hgs_solution_for_dataset = None
        if hgs_solution_path:
            try:
                hgs_solution_for_dataset = load_hgs_solution_from_pkl(
                    hgs_solution_path, instance_index
                )
            except Exception as e:
                print(f"  Failed to load HGS solution for dataset: {e}")

        dataset_dir = os.path.join(timestamped_output_dir, "dataset")
        _save_basin_jsonl_dataset(
            history=history,
            trial_solutions=trial_solutions,
            trial_metadata=trial_metadata,
            unique_optima=unique_optima,
            optimum_id_map=optimum_id_map,
            num_orders=num_orders,
            instance_path=instance_path,
            instance_index=instance_index,
            hgs_solution=hgs_solution_for_dataset,
            result_summary=result,
            output_dir=dataset_dir,
            run_id=timestamp,
        )
        print(f"  Basin dataset saved under: {dataset_dir}")


def analyze_basin_transitions_from_dataset(
    args,
    instance_path, 
    instance_index, 
    global_root="basin_datasets",
    time_limit=1,
    n_runs_per_initial_solution=10,
    top_n_optima=30,
    max_initial_solutions=None,
):
    """
    Extract initial solutions from existing dataset's trajectory.jsonl,
    re-run cuOpt from these initial solutions, and analyze basin transitions.
    
    This function:
    1. Selects top N local optima (lowest cost) from optima.jsonl
    2. Finds initial solutions from trajectory.jsonl that lead to these top optima
    3. Re-runs cuOpt from each initial solution multiple times
    4. Tracks which basin each run converges to
    5. Calculates basin transition probabilities
    6. Extends the dataset with transition information
    
    Args:
        instance_path: Path to instance pkl file
        instance_index: Instance index
        global_root: Root directory for basin datasets
        time_limit: Time limit for each cuOpt run
        n_runs_per_initial_solution: Number of times to run cuOpt from each initial solution
        top_n_optima: Number of top (lowest cost) local optima to select for analysis
    
    Returns:
        dict: Transition statistics and extended dataset info
    """
    from test_landscape import test_callback
    
    instance_id = (
        f"{os.path.basename(instance_path)}#{instance_index}"
        if instance_path
        else f"random#{instance_index}"
    )
    instance_dir = os.path.join(global_root, instance_id)
    traj_path = os.path.join(instance_dir, "trajectory.jsonl")
    optima_path = os.path.join(instance_dir, "optima.jsonl")
    
    print("=" * 80)
    print(f"Analyzing basin transitions for {instance_id}")
    print("=" * 80)
    
    # Load trajectory and optima data
    traj = []
    with open(traj_path, "r", encoding="utf-8") as f:
        for line in f:
            traj.append(json.loads(line))
    
    optima_raw = []
    with open(optima_path, "r", encoding="utf-8") as f:
        for line in f:
            optima_raw.append(json.loads(line))
    
    def edge_set_key(edges_list):
        norm_edges = [tuple(sorted((int(u), int(v)))) for (u, v) in edges_list]
        return frozenset(norm_edges)
    
    # Build basin mapping and find top N optima by cost
    basin_by_edges_hash = {}
    optima_with_cost = []
    for r in optima_raw:
        edges = r.get("edges")
        edges_hash = r.get("edges_hash", "")
        cost = r.get("final_cost")
        basin_key = edge_set_key(edges)
        basin_by_edges_hash[edges_hash] = basin_key
        if cost is not None:
            optima_with_cost.append({
                "edges_hash": edges_hash,
                "cost": cost,
                "run_id": r.get("run_id"),
                "optimum_id": r.get("optimum_id")
            })
    
    # Sort by cost and select top N
    optima_with_cost.sort(key=lambda x: x["cost"])
    top_optima = optima_with_cost[:top_n_optima]
    top_edges_hashes = set(opt["edges_hash"] for opt in top_optima)
    
    print(f"\nStep 1: Selected top {len(top_optima)} local optima (lowest cost)")
    print(f"  Cost range: {top_optima[0]['cost']:.2f} to {top_optima[-1]['cost']:.2f}")
    
    # Map trial to its final optimum
    trial_to_optimum = {}
    for r in traj:
        run_id = r.get("run_id")
        trial_id = r.get("trial_id")
        optimum_id = r.get("optimum_id")
        if run_id is not None and trial_id is not None and optimum_id is not None:
            key = (run_id, trial_id)
            if key not in trial_to_optimum:
                # Find the edges_hash for this optimum
                for opt in optima_raw:
                    if opt.get("run_id") == run_id and opt.get("optimum_id") == optimum_id:
                        trial_to_optimum[key] = opt.get("edges_hash", "")
                        break
    
    # Extract initial solutions from trajectory that lead to top optima
    print("\nStep 2: Extracting initial solutions for top optima...")
    initial_solutions_by_trial = {}
    for r in traj:
        run_id = r.get("run_id")
        trial_id = r.get("trial_id")
        local_iter = r.get("local_iter")
        solution_flat = r.get("solution_flat")
        
        key = (run_id, trial_id)
        trial_final_optimum_hash = trial_to_optimum.get(key)
        
        # Only keep initial solutions for trials that converge to top optima
        if trial_final_optimum_hash in top_edges_hashes:
            if key not in initial_solutions_by_trial or local_iter < initial_solutions_by_trial[key]["local_iter"]:
                initial_solutions_by_trial[key] = {
                    "solution_flat": solution_flat,
                    "local_iter": local_iter,
                    "num_routes": r.get("num_routes_after", 0),
                    "cost": r.get("cost"),
                    "edges_hash": r.get("edges_hash", ""),
                    "target_optimum_hash": trial_final_optimum_hash,
                }
    
    # Deduplicate initial solutions by edges_hash (keep one per unique initial solution)
    print(f"  Found {len(initial_solutions_by_trial)} initial solutions before deduplication")
    initial_solutions_dedup = {}
    seen_edges_hash = set()
    for key, sol_info in initial_solutions_by_trial.items():
        init_edges_hash = sol_info["edges_hash"]
        if init_edges_hash not in seen_edges_hash:
            seen_edges_hash.add(init_edges_hash)
            initial_solutions_dedup[key] = sol_info
    
    initial_solutions_by_trial = initial_solutions_dedup
    print(f"  Found {len(initial_solutions_by_trial)} unique initial solutions for top {top_n_optima} optima")

    # Optional: randomly sample at most max_initial_solutions initial solutions
    if max_initial_solutions is not None and len(initial_solutions_by_trial) > max_initial_solutions:
        import random
        keys = list(initial_solutions_by_trial.keys())
        random.shuffle(keys)
        selected_keys = keys[:max_initial_solutions]
        initial_solutions_by_trial = {k: initial_solutions_by_trial[k] for k in selected_keys}
        print(f"  Sampled {len(initial_solutions_by_trial)} initial solutions (max_initial_solutions={max_initial_solutions})")
    
    # Get num_orders from instance data
    instance_data = load_instance_from_pkl(instance_path, instance_index)
    num_orders = instance_data['n_locations'] - 1
    
    # Run cuOpt from each initial solution and track basin transitions
    print(f"\nStep 3: Running cuOpt from {len(initial_solutions_by_trial)} initial solutions...")
    print(f"  Each initial solution will be run {n_runs_per_initial_solution} times")
    
    transition_records = []
    transition_matrix = {}  # (source_basin_hash, target_basin_hash) -> count
    
    for idx, ((run_id, trial_id), init_sol_info) in enumerate(initial_solutions_by_trial.items(), 1):
        if idx % 10 == 0:
            print(f"  Processing initial solution {idx}/{len(initial_solutions_by_trial)}...")
        
        source_solution_flat = init_sol_info["solution_flat"]
        source_edges_hash = init_sol_info["edges_hash"]
        source_num_routes = init_sol_info["num_routes"]
        standard_solution_flat = cuopt_to_standard(source_solution_flat, source_num_routes, num_orders)
        # print(">>> source solution: ", source_solution_flat)
        # print(">>> initial solution: ", standard_solution_flat)

        # Calculate edges_hash of the initial solution we're setting
        # standard_solution_flat is in standard format (0-separated), use standard_solution_to_routes
        routes_initial = standard_solution_to_routes(standard_solution_flat)
        # print(">>> initial routes: ", routes_initial)
        edges_initial = extract_edges_from_routes(routes_initial)
        # print(">>> initial edges: ", edges_initial)
        initial_edges_hash = _edges_hash(edges_initial)
        initial_edges_set = set(edges_initial)
        
        # Run cuOpt multiple times from this initial solution using test_callback
        for run_idx in range(n_runs_per_initial_solution):
            result_data = test_callback(
                instance_path=args.instance_path if not args.random else None,
                hgs_solution_path=None,
                instance_index=args.instance_index,
                n_locations=args.n_locations,
                time_limit=0.5,
                initial_solution_flat=standard_solution_flat
            )
            
            # Extract first trial's local optima from history
            history = result_data.get('history', [])
            first_trial_records = [r for r in history if r.get('local_search_id') is not None and r.get('local_search_id') == 0]
            if first_trial_records:
                # Get the initial solution from history (what cuOpt actually used)
                initial_record = [r for r in first_trial_records if r.get('local_iter', -1) == -1 or r.get('is_initial_solution', False)]
                if initial_record:
                    initial_record = initial_record[0]
                    history_initial_sol = initial_record.get('sol_after')
                    history_initial_num_routes = initial_record.get('num_routes_after', source_num_routes)
                    if history_initial_sol is not None:
                        # Calculate edges_hash of the actual initial solution cuOpt used
                        routes_history_initial = solution_flat_to_routes(history_initial_sol, history_initial_num_routes, num_orders)
                        edges_history_initial = extract_edges_from_routes(routes_history_initial)
                        actual_source_edges_hash = _edges_hash(edges_history_initial)
                    else:
                        # Fallback to original source_edges_hash if history initial solution is None
                        actual_source_edges_hash = source_edges_hash
                else:
                    # Fallback to original source_edges_hash if no initial record found
                    actual_source_edges_hash = source_edges_hash
                
                # Get the last record (final solution after first trial)
                # Skip initial solution record (local_iter == -1) if present
                non_initial_records = [r for r in first_trial_records if r.get('local_iter', -1) >= 0]
                if non_initial_records:
                    last_record = non_initial_records[-1]
                else:
                    # If only initial solution record exists, use it
                    last_record = first_trial_records[-1]
                
                final_solution_flat = last_record.get('sol_after')
                final_cost = last_record.get('cost_after')
                final_num_routes = last_record.get('num_routes_after', source_num_routes) or source_num_routes
                routes_final = solution_flat_to_routes(final_solution_flat, final_num_routes, num_orders)
                final_edges = extract_edges_from_routes(routes_final)
                final_edges_hash = _edges_hash(final_edges)
            else:
                assert 0, "First trial not finished!"

            transition_records.append({
                "source_run_id": run_id,
                "source_trial_id": trial_id,
                "source_edges_hash": actual_source_edges_hash,  # Use actual initial solution from history
                "target_edges_hash": final_edges_hash,
                "target_cost": final_cost,
                "transition_run_idx": run_idx,
            })
            
            key = (actual_source_edges_hash, final_edges_hash)
            transition_matrix[key] = transition_matrix.get(key, 0) + 1
    
    print(f"\nStep 4: Analyzing transition statistics...")
    print(f"  Total transition runs: {len(transition_records)}")
    
    # Calculate transition probabilities
    transition_probs = {}
    source_counts = {}
    for (source_hash, target_hash), count in transition_matrix.items():
        source_counts[source_hash] = source_counts.get(source_hash, 0) + count
    for (source_hash, target_hash), count in transition_matrix.items():
        transition_probs[(source_hash, target_hash)] = count / source_counts[source_hash]
    
    # Save transition results
    print(f"\nStep 5: Saving transition results...")
    transition_file = os.path.join(instance_dir, "basin_transitions.jsonl")
    with open(transition_file, "w", encoding="utf-8") as f:
        for record in transition_records:
            f.write(json.dumps(record) + "\n")
    print(f"  Saved {len(transition_records)} transition records to {transition_file}")
    
    # Save transition matrix summary
    transition_summary_file = os.path.join(instance_dir, "basin_transition_summary.json")
    summary = {
        "instance_id": instance_id,
        "top_n_optima": top_n_optima,
        "n_initial_solutions": len(initial_solutions_by_trial),
        "n_runs_per_initial_solution": n_runs_per_initial_solution,
        "total_transitions": len(transition_records),
        "transition_matrix": {f"{s}->{t}": count for (s, t), count in transition_matrix.items()},
        "transition_probabilities": {f"{s}->{t}": prob for (s, t), prob in transition_probs.items()},
    }
    with open(transition_summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved transition summary to {transition_summary_file}")
    
    # Print sample transition statistics
    print(f"\nStep 6: Sample transition statistics:")
    print(f"  Unique source basins: {len(set(r['source_edges_hash'] for r in transition_records))}")
    print(f"  Unique target basins: {len(set(r['target_edges_hash'] for r in transition_records))}")
    
    # Show top transitions
    sorted_transitions = sorted(transition_matrix.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 10 most frequent transitions:")
    for (source_hash, target_hash), count in sorted_transitions[:10]:
        prob = transition_probs.get((source_hash, target_hash), 0.0)
        print(f"    {source_hash[:8]}... -> {target_hash[:8]}... : {count} times ({prob*100:.1f}%)")
    
    return {
        "transition_records": transition_records,
        "transition_matrix": transition_matrix,
        "transition_probabilities": transition_probs,
        "summary": summary
    }


def visualize_basin_transitions(
    instance_path,
    instance_index,
    global_root="basin_datasets",
    transition_file=None
):
    """
    Visualize basin transition network and probabilities.
    
    Args:
        instance_path: Path to instance pkl file
        instance_index: Instance index
        global_root: Root directory for basin datasets
        transition_file: Path to basin_transitions.jsonl (if None, will look in instance_dir)
    """
    instance_id = (
        f"{os.path.basename(instance_path)}#{instance_index}"
        if instance_path
        else f"random#{instance_index}"
    )
    instance_dir = os.path.join(global_root, instance_id)
    
    if transition_file is None:
        transition_file = os.path.join(instance_dir, "basin_transitions.jsonl")
    
    if not os.path.exists(transition_file):
        print(f"Transition file not found: {transition_file}")
        return
    
    # Load transition data
    transition_records = []
    with open(transition_file, "r", encoding="utf-8") as f:
        for line in f:
            transition_records.append(json.loads(line))
    
    if not transition_records:
        print("No transition records found.")
        return
    
    # Load optima to get basin costs
    optima_path = os.path.join(instance_dir, "optima.jsonl")
    basin_costs = {}
    basin_edges_hash_to_id = {}
    if os.path.exists(optima_path):
        with open(optima_path, "r", encoding="utf-8") as f:
            for line in f:
                opt = json.loads(line)
                edges_hash = opt.get("edges_hash", "")
                cost = opt.get("final_cost")
                if edges_hash:
                    basin_costs[edges_hash] = cost
                    if edges_hash not in basin_edges_hash_to_id:
                        basin_edges_hash_to_id[edges_hash] = len(basin_edges_hash_to_id)
    
    # Build transition matrix
    transition_matrix = {}
    source_counts = {}
    for r in transition_records:
        source_hash = r.get("source_edges_hash", "")
        target_hash = r.get("target_edges_hash", "")
        if source_hash and target_hash:
            key = (source_hash, target_hash)
            transition_matrix[key] = transition_matrix.get(key, 0) + 1
            source_counts[source_hash] = source_counts.get(source_hash, 0) + 1
    
    # Calculate probabilities
    transition_probs = {}
    for (source_hash, target_hash), count in transition_matrix.items():
        total = source_counts.get(source_hash, 1)
        prob = count / total if total > 0 else 0.0
        transition_probs[(source_hash, target_hash)] = prob
    
    # Get unique basins
    all_basins = set()
    for r in transition_records:
        all_basins.add(r.get("source_edges_hash", ""))
        all_basins.add(r.get("target_edges_hash", ""))
    all_basins = sorted([b for b in all_basins if b])
    
    # Create basin ID mapping
    basin_id_map = {hash_val: idx for idx, hash_val in enumerate(all_basins)}
    
    # Prepare data for visualization
    print(f"\nVisualizing basin transitions for {instance_id}")
    print(f"  Total basins: {len(all_basins)}")
    print(f"  Total transitions: {len(transition_records)}")
    
    # Plot 1: Transition Network Graph
    try:
        import networkx as nx
        import matplotlib.patches as mpatches
        
        G = nx.DiGraph()
        
        # Add nodes with basin information
        for basin_hash in all_basins:
            cost = basin_costs.get(basin_hash, None)
            G.add_node(basin_id_map[basin_hash], 
                      hash=basin_hash,
                      cost=cost,
                      label=f"B{basin_id_map[basin_hash]}\n{basin_hash[:8]}")
        
        # Add edges with weights (probabilities)
        edge_weights = []
        for (source_hash, target_hash), prob in transition_probs.items():
            if source_hash in basin_id_map and target_hash in basin_id_map:
                source_id = basin_id_map[source_hash]
                target_id = basin_id_map[target_hash]
                count = transition_matrix.get((source_hash, target_hash), 0)
                G.add_edge(source_id, target_id, weight=prob, count=count)
                edge_weights.append(prob)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Subplot 1: Network graph
        ax1 = plt.subplot(2, 2, 1)
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw nodes colored by cost
        node_colors = []
        for node_id in G.nodes():
            cost = G.nodes[node_id].get('cost')
            if cost is not None:
                node_colors.append(cost)
            else:
                node_colors.append(0)
        
        if node_colors:
            vmin, vmax = min(node_colors), max(node_colors) if max(node_colors) > 0 else 1
            nodes = nx.draw_networkx_nodes(G, pos, ax=ax1, 
                                          node_color=node_colors,
                                          node_size=500,
                                          cmap=plt.cm.viridis,
                                          vmin=vmin, vmax=vmax,
                                          alpha=0.8)
            if nodes:
                plt.colorbar(nodes, ax=ax1, label='Cost')
        else:
            nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=500, alpha=0.8)
        
        # Draw edges with width proportional to probability
        edges = G.edges()
        edge_widths = [G[u][v].get('weight', 0.1) * 5 for u, v in edges]
        nx.draw_networkx_edges(G, pos, ax=ax1, 
                              width=edge_widths,
                              alpha=0.6,
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20,
                              arrowstyle='->')
        
        # Draw labels
        labels = {node_id: f"B{node_id}" for node_id in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax1, font_size=8)
        
        ax1.set_title("Basin Transition Network\n(Node size/color = Cost, Edge width = Transition Probability)", 
                     fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Subplot 2: Transition Probability Heatmap
        ax2 = plt.subplot(2, 2, 2)
        n_basins = len(all_basins)
        prob_matrix = np.zeros((n_basins, n_basins))
        
        for (source_hash, target_hash), prob in transition_probs.items():
            if source_hash in basin_id_map and target_hash in basin_id_map:
                i = basin_id_map[source_hash]
                j = basin_id_map[target_hash]
                prob_matrix[i, j] = prob
        
        im = ax2.imshow(prob_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax2.set_xlabel('Target Basin', fontsize=10)
        ax2.set_ylabel('Source Basin', fontsize=10)
        ax2.set_title('Transition Probability Matrix', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(n_basins))
        ax2.set_yticks(range(n_basins))
        ax2.set_xticklabels([f'B{i}' for i in range(n_basins)], fontsize=6)
        ax2.set_yticklabels([f'B{i}' for i in range(n_basins)], fontsize=6)
        plt.colorbar(im, ax=ax2, label='Probability')
        
        # Subplot 3: Top transitions bar chart
        ax3 = plt.subplot(2, 2, 3)
        sorted_transitions = sorted(transition_probs.items(), key=lambda x: x[1], reverse=True)[:20]
        if sorted_transitions:
            labels_list = []
            probs_list = []
            counts_list = []
            for (source_hash, target_hash), prob in sorted_transitions:
                source_id = basin_id_map.get(source_hash, -1)
                target_id = basin_id_map.get(target_hash, -1)
                count = transition_matrix.get((source_hash, target_hash), 0)
                labels_list.append(f"B{source_id}→B{target_id}")
                probs_list.append(prob)
                counts_list.append(count)
            
            x_pos = np.arange(len(labels_list))
            bars = ax3.bar(x_pos, probs_list, alpha=0.7, color='steelblue')
            ax3.set_xlabel('Transition', fontsize=10)
            ax3.set_ylabel('Probability', fontsize=10)
            ax3.set_title('Top 20 Transitions by Probability', fontsize=12, fontweight='bold')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=7)
            ax3.grid(axis='y', alpha=0.3)
            
            # Add count labels on bars
            for i, (bar, count) in enumerate(zip(bars, counts_list)):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'n={count}',
                        ha='center', va='bottom', fontsize=6)
        
        # Subplot 4: Source basin transition distribution (pie chart for top source)
        ax4 = plt.subplot(2, 2, 4)
        
        # Find source basin with most transitions
        source_transition_counts = {}
        for r in transition_records:
            source_hash = r.get("source_edges_hash", "")
            if source_hash:
                source_transition_counts[source_hash] = source_transition_counts.get(source_hash, 0) + 1
        
        if source_transition_counts:
            top_source = max(source_transition_counts.items(), key=lambda x: x[1])[0]
            top_source_id = basin_id_map.get(top_source, -1)
            
            # Get transitions from this source
            source_transitions = {}
            for (s_hash, t_hash), count in transition_matrix.items():
                if s_hash == top_source:
                    target_id = basin_id_map.get(t_hash, -1)
                    source_transitions[f"B{target_id}"] = count
            
            if source_transitions:
                labels = list(source_transitions.keys())
                sizes = list(source_transitions.values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
                
                ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
                ax4.set_title(f'Transition Distribution from\nSource Basin B{top_source_id} (Most Active)', 
                             fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(instance_dir, "basin_transition_visualization.png")
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"  Saved visualization to {output_file}")
        plt.close()
        
    except ImportError:
        print("  Warning: networkx not available, skipping network graph visualization")
        print("  Install with: pip install networkx")
        
        # Fallback: simple bar chart visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart of top transitions
        sorted_transitions = sorted(transition_probs.items(), key=lambda x: x[1], reverse=True)[:20]
        if sorted_transitions:
            labels_list = []
            probs_list = []
            counts_list = []
            for (source_hash, target_hash), prob in sorted_transitions:
                labels_list.append(f"{source_hash[:8]}→{target_hash[:8]}")
                probs_list.append(prob)
                counts_list.append(transition_matrix.get((source_hash, target_hash), 0))
            
            x_pos = np.arange(len(labels_list))
            axes[0].bar(x_pos, probs_list, alpha=0.7)
            axes[0].set_xlabel('Transition', fontsize=10)
            axes[0].set_ylabel('Probability', fontsize=10)
            axes[0].set_title('Top 20 Transitions by Probability', fontsize=12, fontweight='bold')
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(labels_list, rotation=45, ha='right', fontsize=7)
            axes[0].grid(axis='y', alpha=0.3)
        
        # Transition count distribution
        transition_counts = list(transition_matrix.values())
        axes[1].hist(transition_counts, bins=20, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Transition Count', fontsize=10)
        axes[1].set_ylabel('Frequency', fontsize=10)
        axes[1].set_title('Distribution of Transition Counts', fontsize=12, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(instance_dir, "basin_transition_visualization.png")
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"  Saved visualization to {output_file}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_limit", type=int, default=1)
    parser.add_argument("--instance_path", type=str, default="/home/jieyi/cvrp100_uniform.pkl")
    parser.add_argument("--hgs_solution_path", type=str, default="/home/jieyi/hgs_cvrp100_uniform.pkl")
    parser.add_argument("--instance_index", type=int, default=0)
    parser.add_argument("--load_all", action="store_true", default=False)
    parser.add_argument("--random", action="store_true", default=False)
    parser.add_argument("--n_locations", type=int, default=101)
    parser.add_argument("--contrastive", action="store_true", default=True)
    parser.add_argument("--n_cuopt_runs", type=int, default=1)
    parser.add_argument("--visualize", action="store_true", default=False, help="Visualize after generating dataset")
    parser.add_argument("--visualize_only", action="store_true", default=False, help="Only visualize from existing JSONL, do not run cuOpt")
    parser.add_argument("--analyze_transitions", action="store_true", default=False, help="Analyze basin transitions from existing dataset")
    parser.add_argument("--visualize_transitions", action="store_true", default=False, help="Visualize basin transitions from existing dataset")
    parser.add_argument("--n_runs_per_initial", type=int, default=10, help="Number of runs per initial solution for transition analysis")
    parser.add_argument("--top_n_optima", type=int, default=30, help="Number of top (lowest cost) local optima to use for transition analysis")
    parser.add_argument("--max_initial_solutions", type=int, default=None, help="Maximum number of initial solutions to sample after deduplication")
    # parser.add_argument("--manual_perturbation", action="store_true", default=False)
    # parser.add_argument("--manual_perturbation_epsilon", type=float, default=0.1)
    # parser.add_argument("--n_perturbations_per_optimum", type=int, default=1)
    # parser.add_argument("--n_runs_per_perturbation", type=int, default=100)
   
    args = parser.parse_args()

    # Only visualize from existing JSONL (no new runs)
    if args.visualize_only:
        visualize_basin_from_jsonl(args.instance_path, args.instance_index)
    elif args.analyze_transitions:
        # Analyze basin transitions from existing dataset
        analyze_basin_transitions_from_dataset(
            args=args,
            instance_path=args.instance_path,
            instance_index=args.instance_index,
            time_limit=args.time_limit,
            n_runs_per_initial_solution=args.n_runs_per_initial,
            top_n_optima=args.top_n_optima,
            max_initial_solutions=args.max_initial_solutions,
        )
    elif args.visualize_transitions:
        # Visualize basin transitions from existing dataset
        visualize_basin_transitions(
            instance_path=args.instance_path,
            instance_index=args.instance_index
        )
    elif args.load_all:
        run_multiple_instances(args.instance_path, args.hgs_solution_path, instance_indices='all')
    else:
        # Run cuOpt multiple times; each run gets its own PNGs and per-run basin analysis.
        import time
        for run_idx in range(args.n_cuopt_runs):
            print(f"\n=== cuOpt run {run_idx + 1}/{args.n_cuopt_runs} ===")
            start_time = time.time()
            result_data = test_callback(
                instance_path=args.instance_path if not args.random else None,
                hgs_solution_path=args.hgs_solution_path if not args.random else None,
                instance_index=args.instance_index,
                n_locations=args.n_locations,
                time_limit=args.time_limit,
            )
            elapsed_time = time.time() - start_time
            if result_data:
                history = result_data.get('history', [])
                if history:
                    max_global_iter = max(r.get('global_iter', -1) for r in history if r.get('global_iter') is not None)
                    print(f"\nPerformance Summary:")
                    print(f"  Actual runtime: {elapsed_time:.2f} seconds")
                    print(f"  Time limit: {args.time_limit} seconds")
                    print(f"  Max global_iter: {max_global_iter}")
                    print(f"  Total history records: {len(history)}")
                    if max_global_iter > 0:
                        iter_per_sec = max_global_iter / elapsed_time if elapsed_time > 0 else 0
                        print(f"  Global iter per second: {iter_per_sec:.1f}")
                        print(f"  Expected iter in {args.time_limit}s: {iter_per_sec * args.time_limit:.0f}")
            if not result_data:
                continue
            if not args.random:
                analyze_basin_structure(
                    result_data,
                    args.instance_path,
                    args.hgs_solution_path,
                    args.instance_index,
                    collect_dataset=args.contrastive,
                )

        # Optionally visualize from aggregated JSONL after these runs
        if args.visualize and not args.random:
            visualize_basin_from_jsonl(args.instance_path, args.instance_index)
