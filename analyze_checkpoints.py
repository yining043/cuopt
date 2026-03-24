#!/usr/bin/env python3
"""
Offline visualization and validation for saved checkpoints.

Usage (example):
  python analyze_checkpoints.py \
    --checkpoint_dir /home/jieyi/cuopt/out/20260204_113802_0-49_bsz512_inbatch \
    --stage both

For each checkpoint in the directory:
  - Stage 1 (if present):
      * Evaluate on fixed val set (val_data_1a1n10d.jsonl)
      * Plot distance histogram d(A, neighbour) vs d(A, distant)
      * Plot 2D embeddings (anchor, neighbour, distant)
  - Stage 2 (if present):
      * Evaluate on fixed val set (val_data_1p1n.jsonl)
      * Plot distance histogram d(A,P) vs d(A,N)
      * Plot 2D embeddings (anchor, positive, negative)
"""

import argparse
import base64
import glob
import io
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from CVRPEnv import CVRPEnv
from dataloader import (
    TripletBatch,
    load_val_data_1a1n10d,
    load_val_data_1p1n,
)
from helper import (
    copy_all_src,
    load_instances_pkl,
    plot_distance_histogram,
    plot_embedding_2d,
    make_triplet_gif,
    solution_flat_to_solution,
)
from net import SolutionEmbedder
from callback.utils import load_hgs_solution_from_pkl, convert_hgs_routes_to_format


def embed_solutions(
    embedder: SolutionEmbedder,
    solutions: List[List[int]],
    env: CVRPEnv,
    basin_info_cache: Dict[str, dict],
) -> torch.Tensor:
    """Embed a list of solutions via CVRPEnv (same as train script)."""
    hashes: List[str] = []
    for i, sol in enumerate(solutions):
        h = f"_tmp_{i}"
        basin_info_cache[h] = {"solution": sol}
        hashes.append(h)
    context = env.prepare_from_hashes(hashes)
    emb = embedder(context, env)
    for h in hashes:
        del basin_info_cache[h]
    return emb


def _get_encoder_state(ckpt: dict) -> Optional[Dict[str, torch.Tensor]]:
    """Extract encoder-only state dict from either new (embedder_state) or legacy (encoder_state) format."""
    if "embedder_state" in ckpt:
        prefix = "encoder."
        return {k[len(prefix):]: v for k, v in ckpt["embedder_state"].items() if k.startswith(prefix)}
    return ckpt.get("encoder_state")


def _load_embedder(embedder: SolutionEmbedder, ckpt: dict) -> None:
    """Load weights into embedder from either new (embedder_state) or legacy (encoder_state) format."""
    if "embedder_state" in ckpt:
        embedder.load_state_dict(ckpt["embedder_state"])
    elif "encoder_state" in ckpt:
        embedder.encoder.load_state_dict(ckpt["encoder_state"])
        print("WARNING: legacy checkpoint has encoder_state only; pos_encoder weights are randomly initialized")


def build_model(
    args: argparse.Namespace,
    device: torch.device,
    encoder_state: Optional[Dict[str, torch.Tensor]] = None,
) -> SolutionEmbedder:
    """Build SolutionEmbedder, optionally inferring dims from a checkpoint encoder_state.

    This makes analysis robust even if supplement_feature_dim / embedding_dim changed in training.
    """
    # Defaults from args (used when no checkpoint info)
    embedding_dim = args.embedding_dim
    supplement_feature_dim = args.supplement_feature_dim
    hidden_dim = args.hidden_dim

    if encoder_state is not None:
        # Infer embedding_dim and supplement_feature_dim from checkpoint weights
        w_depot = encoder_state.get("embedding_depot.weight")
        if w_depot is not None:
            # shape: (embedding_dim, 2 + supplement_feature_dim)
            embedding_dim = int(w_depot.shape[0])
            in_dim = int(w_depot.shape[1])
            supplement_feature_dim = max(in_dim - 2, 0)

        w_node = encoder_state.get("embedding_node.weight")
        if w_node is not None:
            # shape: (embedding_dim, 3 + supplement_feature_dim); use it to sanity-check
            in_dim_node = int(w_node.shape[1])
            supp_from_node = max(in_dim_node - 3, 0)
            supplement_feature_dim = min(supplement_feature_dim, supp_from_node)

        # Infer hidden_dim from first FF layer if available
        w_ff1 = encoder_state.get("layers.0.ff.W1.weight")
        if w_ff1 is not None:
            hidden_dim = int(w_ff1.shape[0])

    model_params = {
        "problem": "CVRP",
        "embedding_dim": embedding_dim,
        "encoder_layer_num": args.n_layers,
        "supplement_feature_dim": supplement_feature_dim,
        # Encoder internally uses (2 + supplement_feature_dim) and (3 + supplement_feature_dim),
        # these are kept for completeness but not used directly.
        "depot_feature_dim": 5,
        "node_feature_dim": 6,
        "head_num": args.n_heads,
        "qkv_dim": args.embedding_dim // args.n_heads,
        "hidden_dim": hidden_dim,
        # For analysis we default to raw (unnormalized) embeddings unless user explicitly enables it.
        "use_l2_normalize": args.use_l2_normalize,
    }
    return SolutionEmbedder(model_params).to(device)


def parse_instance_index_from_trajectory_path(trajectory_path: str) -> int:
    """Infer instance index from parent dir name, e.g. .../cvrp100_uniform.pkl#3/trajectory.jsonl -> 3."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(trajectory_path)))
    if "#" in parent:
        try:
            return int(parent.split("#")[-1].strip())
        except ValueError:
            pass
    return 0


def load_trajectory_trials(trajectory_path: str) -> List[Tuple[Any, List[Dict]]]:
    """
    Load trajectory.jsonl and return list of (final_edges_hash, list of records) per trial.
    Each record has at least: solution_flat, cost, edges_hash, global_iter, local_iter.
    """
    trials: Dict[Tuple[Any, Any], List[Dict]] = {}
    if not os.path.isfile(trajectory_path):
        return []
    with open(trajectory_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            run_id = rec.get("run_id")
            trial_id = rec.get("trial_id")
            if run_id is None or trial_id is None:
                continue
            key = (run_id, trial_id)
            if key not in trials:
                trials[key] = []
            trials[key].append(rec)
    # Sort each trial by global_iter, local_iter
    out: List[Tuple[Any, List[Dict]]] = []
    for key, recs in trials.items():
        recs_sorted = sorted(
            recs,
            key=lambda r: (r.get("global_iter", 0) or 0, r.get("local_iter", 0) or 0),
        )
        final_hash = recs_sorted[-1].get("edges_hash") if recs_sorted else None
        out.append((final_hash, recs_sorted))
    return out


MIN_POINTS_PER_TRIAL = 5


def select_N_trials_different_optima(
    trajectory_path: str,
    N: int = 10,
    seed: int = 42,
    min_points: int = MIN_POINTS_PER_TRIAL,
) -> Optional[List[Tuple[int, List[Dict]]]]:
    """
    From one trajectory.jsonl (single instance), select up to N trials with distinct final local optima.
    Only considers trials with at least min_points solutions.
    Returns list of (instance_index, list of records), or None if no valid trials.
    """
    trials_with_hash = load_trajectory_trials(trajectory_path)
    if not trials_with_hash:
        return None
    rng = random.Random(seed)
    by_hash: Dict[Any, List[Tuple[Any, List[Dict]]]] = {}
    for final_hash, recs in trials_with_hash:
        if len(recs) < min_points:
            continue
        h = final_hash if final_hash else id(recs)
        if h not in by_hash:
            by_hash[h] = []
        by_hash[h].append((final_hash, recs))
    distinct_trials = [rng.choice(entries)[1] for entries in by_hash.values() if entries]
    if not distinct_trials:
        return None
    k = min(N, len(distinct_trials))
    chosen = rng.sample(distinct_trials, k)
    instance_index = parse_instance_index_from_trajectory_path(trajectory_path)
    return [(instance_index, recs) for recs in chosen]


def select_five_trials_different_optima(
    trajectory_path: str,
    seed: int = 42,
    min_points: int = MIN_POINTS_PER_TRIAL,
) -> Optional[List[Tuple[int, List[Dict]]]]:
    """Select 5 trials with distinct local optima (convenience wrapper)."""
    return select_N_trials_different_optima(trajectory_path, N=5, seed=seed, min_points=min_points)


def select_N_trials_stratified_by_run(
    trajectory_path: str,
    K: int = 10,
    run_id: Optional[Any] = None,
    min_points: int = MIN_POINTS_PER_TRIAL,
) -> Optional[List[Tuple[int, List[Dict]]]]:
    """
    From one run: (1) keep only trials that end at distinct local optima (necessary) and have >= min_points.
    (2) Order these by trial index, divide into K equal segments, pick one per segment (near center).
    Returns list of (instance_index, list of records), or None if no valid selection.
    """
    one_run = load_one_run_from_trajectory(trajectory_path, run_id=run_id)
    if one_run is None:
        return None
    _run_id, instance_index, full_sequence, local_optima_indices = one_run
    num_trials = len(local_optima_indices)
    if num_trials == 0:
        return None
    trial_starts = [0] + [local_optima_indices[i] + 1 for i in range(num_trials - 1)]
    trial_ends = list(local_optima_indices)
    # (trial_index, recs, final_hash) for each trial with >= min_points
    candidates: List[Tuple[int, List[Dict], Any]] = []
    for i in range(num_trials):
        recs = full_sequence[trial_starts[i] : trial_ends[i] + 1]
        if len(recs) < min_points:
            continue
        final_hash = recs[-1].get("edges_hash") if recs else None
        candidates.append((i, recs, final_hash))
    # One representative per distinct local optimum (by final edges_hash), keep order by trial index
    by_hash: Dict[Any, List[Tuple[int, List[Dict]]]] = {}
    for idx, recs, h in candidates:
        key = h if h is not None else id(recs)
        if key not in by_hash:
            by_hash[key] = []
        by_hash[key].append((idx, recs))
    distinct_trials = [(idx, recs) for _, group in by_hash.items() for (idx, recs) in [min(group, key=lambda x: x[0])]]
    # Each element: (trial_index_in_run, recs)
    distinct_trials.sort(key=lambda x: x[0])
    if not distinct_trials:
        return None

    # Work on positions in distinct_trials list (0..n-1)
    n = len(distinct_trials)

    # Best-cost trial among distinct-optima trials (must be included)
    def final_cost_from_pair(pair: Tuple[int, List[Dict]]) -> float:
        recs = pair[1]
        c = recs[-1].get("cost") if recs else None
        return float(c) if c is not None else float("inf")

    best_pos = min(range(n), key=lambda i: final_cost_from_pair(distinct_trials[i]))

    # K segments over positions 0..n-1; if a segment contains best_pos, it's represented by best trial
    chosen_positions: List[int] = [best_pos]
    for j in range(K):
        lo = j * n // K
        hi = (j + 1) * n // K
        if lo >= hi:
            continue
        seg_positions = list(range(lo, hi))
        if best_pos in seg_positions:
            # This segment already represented by best trial
            continue
        mid_idx = seg_positions[(len(seg_positions) - 1) // 2]
        chosen_positions.append(mid_idx)

    # Unique and ordered by trial index (search order)
    chosen_positions = sorted(set(chosen_positions))
    return [(instance_index, distinct_trials[p][1]) for p in chosen_positions]


def select_five_trials_from_paths(
    trajectory_paths: List[str],
    seed: int = 42,
    min_points: int = MIN_POINTS_PER_TRIAL,
) -> Optional[List[Tuple[int, List[Dict]]]]:
    """
    From multiple trajectory paths (e.g. one per instance), pick one random trial per path, up to 5.
    Only picks trials with at least min_points solutions.
    """
    if len(trajectory_paths) < 5:
        return None
    rng = random.Random(seed)
    out: List[Tuple[int, List[Dict]]] = []
    for path in trajectory_paths[:5]:
        trials_with_hash = load_trajectory_trials(path)
        valid = [(h, recs) for h, recs in trials_with_hash if len(recs) >= min_points]
        if not valid:
            continue
        _, recs = rng.choice(valid)
        inst_idx = parse_instance_index_from_trajectory_path(path)
        out.append((inst_idx, recs))
    return out if len(out) == 5 else None


def parse_stage_epoch(basename: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse stage (1 or 2) and epoch from checkpoint basename, e.g. s1_epoch10 -> (1, 10)."""
    m = re.match(r"s([12])_epoch(\d+)", basename, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def get_last_epoch_checkpoints_per_stage(checkpoint_dir: str) -> List[str]:
    """Return one checkpoint path per stage: the one with the largest epoch (e.g. last s1_epoch*.pt, last s2_epoch*.pt)."""
    all_pt = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    by_stage: Dict[int, List[Tuple[int, str]]] = {}
    for path in all_pt:
        base = os.path.splitext(os.path.basename(path))[0]
        stage, epoch = parse_stage_epoch(base)
        if stage is not None and epoch is not None:
            if stage not in by_stage:
                by_stage[stage] = []
            by_stage[stage].append((epoch, path))
    out: List[str] = []
    for stage in sorted(by_stage.keys()):
        best = max(by_stage[stage], key=lambda x: x[0])
        out.append(best[1])
    return out


def load_one_run_from_trajectory(
    trajectory_path: str,
    run_id: Optional[Any] = None,
) -> Optional[Tuple[Any, int, List[Dict], List[int]]]:
    """
    Load one run from trajectory.jsonl: all trials for one run_id in order.
    Returns (run_id, instance_index, full_sequence_records, local_optima_indices).
    full_sequence = all solutions in order (trial1_sol1..trial1_final, trial2_sol1..trial2_final, ...).
    local_optima_indices = indices into full_sequence of the last solution of each trial (each local optimum).
    """
    if not os.path.isfile(trajectory_path):
        return None
    trials_by_key: Dict[Tuple[Any, Any], List[Dict]] = {}
    with open(trajectory_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = rec.get("run_id")
            tid = rec.get("trial_id")
            if rid is None or tid is None:
                continue
            key = (rid, tid)
            if key not in trials_by_key:
                trials_by_key[key] = []
            trials_by_key[key].append(rec)

    if not trials_by_key:
        return None
    # Pick run_id: use first seen or the one requested
    if run_id is None:
        run_id = next(iter(trials_by_key))[0]
    # All trials for this run_id, sorted by trial_id (match int/str)
    def run_match(k0: Any) -> bool:
        return k0 == run_id or str(k0) == str(run_id)
    run_trials = [(k[1], trials_by_key[k]) for k in trials_by_key if run_match(k[0])]
    if not run_trials:
        return None
    run_trials.sort(key=lambda x: x[0])
    full_sequence: List[Dict] = []
    local_optima_indices: List[int] = []
    for _, recs in run_trials:
        recs_sorted = sorted(
            recs,
            key=lambda r: (r.get("global_iter", 0) or 0, r.get("local_iter", 0) or 0),
        )
        for r in recs_sorted:
            if r.get("solution_flat") is not None:
                full_sequence.append(r)
        if recs_sorted:
            local_optima_indices.append(len(full_sequence) - 1)
    if not full_sequence or not local_optima_indices:
        return None
    instance_index = parse_instance_index_from_trajectory_path(trajectory_path)
    return (run_id, instance_index, full_sequence, local_optima_indices)


def embed_run_full_sequence(
    embedder: Any,
    instance_index: int,
    full_sequence: List[Dict],
    instance_data_by_idx: Dict[int, Dict],
    env: CVRPEnv,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Embed all solutions in full_sequence (one instance). Returns (N, D) tensor."""
    solutions = []
    for r in full_sequence:
        sol_flat = r.get("solution_flat")
        if sol_flat is None:
            continue
        try:
            sol = solution_flat_to_solution(sol_flat)
        except Exception:
            continue
        if sol:
            solutions.append(sol)
    if not solutions:
        return torch.empty(0, 1, device=device)
    inst_data = instance_data_by_idx.get(instance_index)
    if inst_data is None:
        return torch.empty(0, 1, device=device)
    depot_xy = inst_data["depot_xy"].to(device)
    node_xy_demand = inst_data["node_xy_demand"].to(device)
    basin_cache: Dict[str, dict] = {}
    env.load(depot_xy, node_xy_demand, basin_cache)
    embs: List[torch.Tensor] = []
    for start in range(0, len(solutions), batch_size):
        batch = solutions[start : start + batch_size]
        emb = embed_solutions(embedder, batch, env, basin_cache)
        embs.append(emb)
    return torch.cat(embs, dim=0)


def eval_stage1_on_val(
    embedder: SolutionEmbedder,
    env: CVRPEnv,
    val_records: List[Tuple[int, dict]],
    val_instance_data_by_idx: Dict[int, dict],
    device: torch.device,
) -> Tuple[float, float, float, List[float], List[float], Optional[Tuple[int, dict, dict]]]:
    """Eval Stage 1 on fixed val set (anchor, neighbour, 10 distant)."""
    if not val_records:
        return 0.0, 0.0, 0.0, [], [], None
    list_d_ap: List[float] = []
    list_d_ad: List[float] = []
    first_triplet: Optional[Tuple[int, dict, dict]] = None
    basin_cache: Dict[str, dict] = {}

    embedder.eval()
    with torch.no_grad():
        for idx, rec in val_records:
            inst = val_instance_data_by_idx[idx]
            env.load(inst["depot_xy"].to(device), inst["node_xy_demand"].to(device), basin_cache)
            emb_a = embed_solutions(embedder, [rec["anchor_solution"]], env, basin_cache)
            emb_p = embed_solutions(embedder, [rec["neighbor_solution"]], env, basin_cache)
            if rec["distant_solutions"]:
                emb_d = embed_solutions(embedder, rec["distant_solutions"], env, basin_cache)
                d_ad = F.pairwise_distance(emb_a.expand_as(emb_d), emb_d, p=2)
                list_d_ad.extend(d_ad.cpu().tolist())
            d_ap = F.pairwise_distance(emb_a, emb_p, p=2)[0].item()
            list_d_ap.append(d_ap)
            if first_triplet is None:
                first_triplet = (idx, rec, inst)

    mean_d_ap = sum(list_d_ap) / len(list_d_ap) if list_d_ap else 0.0
    mean_d_ad = sum(list_d_ad) / len(list_d_ad) if list_d_ad else 0.0
    ratio = mean_d_ad / (mean_d_ap + 1e-8) if list_d_ap and list_d_ad else 0.0
    return mean_d_ap, mean_d_ad, ratio, list_d_ap, list_d_ad, first_triplet


def compute_s1_val_embeddings(
    embedder: SolutionEmbedder,
    env: CVRPEnv,
    val_records: List[Tuple[int, dict]],
    val_instance_data_by_idx: Dict[int, dict],
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Embed all Stage 1 val records and collect instance_ids, roles, and triplet_ids.

    roles: 0 = anchor, 1 = neighbour, 2 = distant.
    triplet_ids: index of record for each row.
    """
    if not val_records:
        return None, None, None, None

    embedder.eval()
    all_embs: List[torch.Tensor] = []
    instance_ids: List[int] = []
    roles: List[int] = []
    triplet_ids: List[int] = []

    basin_cache: Dict[str, dict] = {}
    with torch.no_grad():
        for t_idx, (idx, rec) in enumerate(val_records):
            inst = val_instance_data_by_idx[idx]
            env.load(inst["depot_xy"].to(device), inst["node_xy_demand"].to(device), basin_cache)

            # Anchor
            emb_a = embed_solutions(embedder, [rec["anchor_solution"]], env, basin_cache)
            all_embs.append(emb_a.squeeze(0))
            instance_ids.append(idx)
            roles.append(0)
            triplet_ids.append(t_idx)

            # Neighbour
            emb_p = embed_solutions(embedder, [rec["neighbor_solution"]], env, basin_cache)
            all_embs.append(emb_p.squeeze(0))
            instance_ids.append(idx)
            roles.append(1)
            triplet_ids.append(t_idx)

            # Distant solutions (may be empty)
            distant_solutions = rec.get("distant_solutions") or []
            if distant_solutions:
                emb_d = embed_solutions(embedder, distant_solutions, env, basin_cache)
                for j in range(emb_d.size(0)):
                    all_embs.append(emb_d[j])
                    instance_ids.append(idx)
                    roles.append(2)
                    triplet_ids.append(t_idx)

    if not all_embs:
        return None, None, None, None

    embs_tensor = torch.stack(all_embs, dim=0)
    return (
        embs_tensor,
        np.asarray(instance_ids, dtype=np.int64),
        np.asarray(roles, dtype=np.int64),
        np.asarray(triplet_ids, dtype=np.int64),
    )


def eval_stage2_on_val(
    embedder: SolutionEmbedder,
    env: CVRPEnv,
    val_triplets: List[Tuple[int, dict]],
    val_instance_data_by_idx: Dict[int, dict],
    device: torch.device,
    batch_size: int,
) -> Tuple[float, float, float, List[float], List[float], Optional[TripletBatch]]:
    """Eval Stage 2 on fixed val set (anchor, positive, negative)."""
    if not val_triplets:
        return 0.0, 0.0, 0.0, [], [], None

    list_d_ap: List[float] = []
    list_d_an: List[float] = []
    sum_ratio = 0.0
    n_batches = 0
    first_batch: Optional[TripletBatch] = None
    basin_cache: Dict[str, dict] = {}

    embedder.eval()
    with torch.no_grad():
        batch_start = 0
        while batch_start < len(val_triplets):
            batch_items = val_triplets[batch_start : batch_start + batch_size]
            batch_start += batch_size
            inst_idx = batch_items[0][0]
            inst = val_instance_data_by_idx[inst_idx]
            batch = TripletBatch(
                anchor_solutions=[t["anchor_solution"] for _, t in batch_items],
                positive_solutions=[t["positive_solution"] for _, t in batch_items],
                negative_solutions=[t["negative_solution"] for _, t in batch_items],
                instance_idx=inst_idx,
                depot_xy=inst["depot_xy"].to(device),
                node_xy_demand=inst["node_xy_demand"].to(device),
            )
            if first_batch is None:
                first_batch = batch
            env.load(batch.depot_xy, batch.node_xy_demand, basin_cache)
            emb_a = embed_solutions(embedder, batch.anchor_solutions, env, basin_cache)
            emb_p = embed_solutions(embedder, batch.positive_solutions, env, basin_cache)
            emb_n = embed_solutions(embedder, batch.negative_solutions, env, basin_cache)
            d_ap = F.pairwise_distance(emb_a, emb_p, p=2)
            d_an = F.pairwise_distance(emb_a, emb_n, p=2)
            list_d_ap.extend(d_ap.cpu().tolist())
            list_d_an.extend(d_an.cpu().tolist())
            mean_ap, mean_an = d_ap.mean().item(), d_an.mean().item()
            sum_ratio += mean_an / (mean_ap + 1e-8)
            n_batches += 1

    mean_d_ap = sum(list_d_ap) / len(list_d_ap) if list_d_ap else 0.0
    mean_d_an = sum(list_d_an) / len(list_d_an) if list_d_an else 0.0
    mean_ratio = sum_ratio / n_batches if n_batches else 0.0
    return mean_d_ap, mean_d_an, mean_ratio, list_d_ap, list_d_an, first_batch


def compute_s2_val_embeddings(
    embedder: SolutionEmbedder,
    env: CVRPEnv,
    val_triplets: List[Tuple[int, dict]],
    val_instance_data_by_idx: Dict[int, dict],
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Embed all Stage 2 val triplets and collect instance_ids, roles, and triplet_ids."""
    if not val_triplets:
        return None, None, None, None

    embedder.eval()
    all_embs: List[torch.Tensor] = []
    instance_ids: List[int] = []
    roles: List[int] = []        # 0 = anchor, 1 = positive, 2 = negative
    triplet_ids: List[int] = []  # index of triplet for each row

    with torch.no_grad():
        basin_cache: Dict[str, dict] = {}
        for t_idx, (inst_idx, record) in enumerate(val_triplets):
            inst = val_instance_data_by_idx[inst_idx]
            depot_xy = inst["depot_xy"].to(device)
            node_xy_demand = inst["node_xy_demand"].to(device)
            env.load(depot_xy, node_xy_demand, basin_cache)

            sols = [
                record["anchor_solution"],
                record["positive_solution"],
                record["negative_solution"],
            ]
            for role, sol in enumerate(sols):
                emb = embed_solutions(embedder, [sol], env, basin_cache)
                all_embs.append(emb.squeeze(0))
                instance_ids.append(inst_idx)
                roles.append(role)
                triplet_ids.append(t_idx)

    if not all_embs:
        return None, None, None, None

    embs_tensor = torch.stack(all_embs, dim=0)  # (3*T, D)
    return (
        embs_tensor,
        np.asarray(instance_ids, dtype=np.int64),
        np.asarray(roles, dtype=np.int64),
        np.asarray(triplet_ids, dtype=np.int64),
    )


def analyze_stage1(args: argparse.Namespace, device: torch.device) -> None:
    # Load fixed val data for S1
    val_records = load_val_data_1a1n10d(args.val_data_1a1n10d)
    if not val_records:
        print(f"[S1] No val records found in {args.val_data_1a1n10d}, skip Stage 1 analysis.")
        return
    val_indices = sorted(set(idx for idx, _ in val_records))
    val_instance_list = load_instances_pkl(args.instance_pkl, device, val_indices, {})
    val_instance_data_by_idx = dict(zip(val_indices, val_instance_list))
    env = CVRPEnv(problem_size=args.problem_size, device=device)

    # Prepare output dir
    plot_dir = os.path.join(args.checkpoint_dir, "analysis_s1")
    os.makedirs(plot_dir, exist_ok=True)

    # Find checkpoints (any *.pt)
    ckpts = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pt")))
    if not ckpts:
        print(f"[S1] No *.pt checkpoints found in {args.checkpoint_dir}")
        return

    print(f"[S1] Found {len(ckpts)} checkpoints.")

    epochs: List[float] = []
    vals_ap: List[float] = []
    vals_ad: List[float] = []
    vals_ratio: List[float] = []

    for ckpt_path in ckpts:
        print(f"[S1] Analyzing {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        embedder = build_model(args, device, encoder_state=_get_encoder_state(ckpt))
        _load_embedder(embedder, ckpt)

        mean_d_ap, mean_d_ad, ratio, d_ap_list, d_ad_list, first_triplet = eval_stage1_on_val(
            embedder, env, val_records, val_instance_data_by_idx, device
        )
        epoch = ckpt.get("epoch", "?")
        base = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(
            f"[S1] Epoch {epoch} val: d(A,neighbour)={mean_d_ap:.4f}, "
            f"d(A,distant)={mean_d_ad:.4f}, ratio={ratio:.4f}"
        )

        try:
            e_float = float(epoch)
        except Exception:
            e_float = float(len(epochs))
        epochs.append(e_float)
        vals_ap.append(mean_d_ap)
        vals_ad.append(mean_d_ad)
        vals_ratio.append(ratio)

        # Distance histogram
        if d_ap_list and d_ad_list:
            hist_path = os.path.join(plot_dir, f"distance_hist_s1_{base}.png")
            plot_distance_histogram(d_ap_list, d_ad_list, save_path=hist_path)
            print(f"[S1] Saved {hist_path}")

        # Embedding 2D & GIF over the entire fixed val set (all instances, all anchors/neighbours/distant)
        embs_tensor, inst_ids_np, roles_np, triplet_ids_np = compute_s1_val_embeddings(
            embedder, env, val_records, val_instance_data_by_idx, device
        )
        if embs_tensor is not None:
            emb_path = os.path.join(plot_dir, f"embedding_2d_s1_{base}.png")
            coords = plot_embedding_2d(
                embs_tensor,
                instance_ids=inst_ids_np,
                group_labels=roles_np,
                method="pca",
                save_path=emb_path,
            )
            print(f"[S1] Saved {emb_path}")

            # GIF: 每一帧高亮一条 (anchor, neighbour, all distant) 的数据
            gif_path = os.path.join(plot_dir, f"embedding_2d_s1_triplets_{base}.gif")
            make_triplet_gif(
                coords=coords,
                roles=roles_np,
                triplet_ids=triplet_ids_np,
                role_label_map={0: "anchor", 1: "neighbour", 2: "distant"},
                role_color_map={0: "tab:blue", 1: "tab:orange", 2: "tab:red"},
                title_prefix="S1 record",
                gif_path=gif_path,
                duration=2.0,
            )

    # Line plots over epochs for S1 metrics
    if epochs:
        try:
            import matplotlib.pyplot as plt

            order = sorted(range(len(epochs)), key=lambda i: epochs[i])
            xs = [epochs[i] for i in order]

            plt.figure(figsize=(6, 4))
            plt.plot(xs, [vals_ap[i] for i in order], "-o", label="d(A, neighbour)")
            plt.plot(xs, [vals_ad[i] for i in order], "-o", label="d(A, distant)")
            plt.xlabel("epoch")
            plt.ylabel("distance")
            plt.legend()
            plt.title("Stage1 val distances over checkpoints")
            plt.tight_layout()
            path1 = os.path.join(plot_dir, "s1_val_distances_over_epochs.png")
            plt.savefig(path1, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[S1] Saved {path1}")

            plt.figure(figsize=(6, 4))
            plt.plot(xs, [vals_ratio[i] for i in order], "-o", label="d(A,D)/d(A,P)")
            plt.xlabel("epoch")
            plt.ylabel("ratio")
            plt.legend()
            plt.title("Stage1 val d(A,D)/d(A,P) over checkpoints")
            plt.tight_layout()
            path2 = os.path.join(plot_dir, "s1_val_ratio_over_epochs.png")
            plt.savefig(path2, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[S1] Saved {path2}")
        except ImportError:
            print("[S1] matplotlib not available, skip line plots.")


def analyze_stage2(args: argparse.Namespace, device: torch.device) -> None:
    # Load fixed val data for S2
    val_triplets = load_val_data_1p1n(args.val_data_1p1n)
    if not val_triplets:
        print(f"[S2] No val records found in {args.val_data_1p1n}, skip Stage 2 analysis.")
        return
    val_indices = sorted(set(idx for idx, _ in val_triplets))
    val_instance_list = load_instances_pkl(args.instance_pkl, device, val_indices, {})
    val_instance_data_by_idx = dict(zip(val_indices, val_instance_list))
    env = CVRPEnv(problem_size=args.problem_size, device=device)

    plot_dir = os.path.join(args.checkpoint_dir, "analysis_s2")
    os.makedirs(plot_dir, exist_ok=True)

    ckpts = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pt")))
    if not ckpts:
        print(f"[S2] No *.pt checkpoints found in {args.checkpoint_dir}")
        return

    print(f"[S2] Found {len(ckpts)} checkpoints.")

    epochs: List[float] = []
    vals_ap: List[float] = []
    vals_an: List[float] = []
    vals_ratio: List[float] = []

    for ckpt_path in ckpts:
        print(f"[S2] Analyzing {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        embedder = build_model(args, device, encoder_state=_get_encoder_state(ckpt))
        _load_embedder(embedder, ckpt)

        mean_d_ap, mean_d_an, ratio, d_ap_list, d_an_list, first_batch = eval_stage2_on_val(
            embedder, env, val_triplets, val_instance_data_by_idx, device, args.batch_size2
        )
        epoch = ckpt.get("epoch", "?")
        base = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(
            f"[S2] Epoch {epoch} val: d(A,P)={mean_d_ap:.4f}, "
            f"d(A,N)={mean_d_an:.4f}, ratio={ratio:.4f}"
        )

        try:
            e_float = float(epoch)
        except Exception:
            e_float = float(len(epochs))
        epochs.append(e_float)
        vals_ap.append(mean_d_ap)
        vals_an.append(mean_d_an)
        vals_ratio.append(ratio)

        if d_ap_list and d_an_list:
            hist_path = os.path.join(plot_dir, f"distance_hist_s2_{base}.png")
            plot_distance_histogram(d_ap_list, d_an_list, save_path=hist_path)
            print(f"[S2] Saved {hist_path}")

        # Embedding 2D and GIF over the entire fixed val set (all instances, all triplets)
        embs_list: List[torch.Tensor] = []
        inst_ids: List[int] = []
        roles: List[int] = []
        triplet_ids: List[int] = []

        embedder.eval()
        basin_cache: Dict[str, dict] = {}
        with torch.no_grad():
            for t_idx, (inst_idx, record) in enumerate(val_triplets):
                inst = val_instance_data_by_idx[inst_idx]
                depot_xy = inst["depot_xy"].to(device)
                node_xy_demand = inst["node_xy_demand"].to(device)
                env.load(depot_xy, node_xy_demand, basin_cache)

                sols = [
                    record["anchor_solution"],
                    record["positive_solution"],
                    record["negative_solution"],
                ]
                for role, sol in enumerate(sols):
                    emb = embed_solutions(embedder, [sol], env, basin_cache)
                    embs_list.append(emb.squeeze(0))
                    inst_ids.append(inst_idx)
                    roles.append(role)
                    triplet_ids.append(t_idx)

        if embs_list:
            embs_tensor = torch.stack(embs_list, dim=0)
            inst_ids_np = np.asarray(inst_ids, dtype=np.int64)
            roles_np = np.asarray(roles, dtype=np.int64)
            triplet_ids_np = np.asarray(triplet_ids, dtype=np.int64)

            emb_path = os.path.join(plot_dir, f"embedding_2d_s2_{base}.png")
            coords = plot_embedding_2d(
                embs_tensor,
                instance_ids=inst_ids_np,
                group_labels=roles_np,
                method="pca",
                save_path=emb_path,
            )
            print(f"[S2] Saved {emb_path}")

            # GIF: same 2D coordinates, each frame highlights a single (A, P, N) triplet
            gif_path = os.path.join(plot_dir, f"embedding_2d_s2_triplets_{base}.gif")
            make_triplet_gif(
                coords=coords,
                roles=roles_np,
                triplet_ids=triplet_ids_np,
                role_label_map={0: "anchor", 1: "positive", 2: "negative"},
                role_color_map={0: "tab:blue", 1: "tab:orange", 2: "tab:green"},
                title_prefix="S2 triplet",
                gif_path=gif_path,
                duration=2.0,
            )

    # Line plots over epochs for S2 metrics
    if epochs:
        try:
            import matplotlib.pyplot as plt

            order = sorted(range(len(epochs)), key=lambda i: epochs[i])
            xs = [epochs[i] for i in order]

            plt.figure(figsize=(6, 4))
            plt.plot(xs, [vals_ap[i] for i in order], "-o", label="d(A,P)")
            plt.plot(xs, [vals_an[i] for i in order], "-o", label="d(A,N)")
            plt.xlabel("epoch")
            plt.ylabel("distance")
            plt.legend()
            plt.title("Stage2 val distances over checkpoints")
            plt.tight_layout()
            path1 = os.path.join(plot_dir, "s2_val_distances_over_epochs.png")
            plt.savefig(path1, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[S2] Saved {path1}")

            plt.figure(figsize=(6, 4))
            plt.plot(xs, [vals_ratio[i] for i in order], "-o", label="d(A,N)/d(A,P)")
            plt.xlabel("epoch")
            plt.ylabel("ratio")
            plt.legend()
            plt.title("Stage2 val d(A,N)/d(A,P) over checkpoints")
            plt.tight_layout()
            path2 = os.path.join(plot_dir, "s2_val_ratio_over_epochs.png")
            plt.savefig(path2, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[S2] Saved {path2}")
        except ImportError:
            print("[S2] matplotlib not available, skip line plots.")


def embed_trajectory_trials(
    embedder: Any,
    five_trials: List[Tuple[int, List[Dict]]],
    instance_data_by_idx: Dict[int, Dict],
    env: CVRPEnv,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Embed all solutions from 5 trials. Returns (embeddings (N,D), trial_ids (N,), costs (N,)).
    """
    all_solutions: List[List[int]] = []
    trial_ids: List[int] = []
    costs: List[float] = []
    index_to_inst: List[int] = []
    for trial_idx, (inst_idx, recs) in enumerate(five_trials):
        for r in recs:
            sol_flat = r.get("solution_flat")
            cost = r.get("cost")
            if sol_flat is None:
                continue
            try:
                sol = solution_flat_to_solution(sol_flat)
            except Exception:
                continue
            if not sol:
                continue
            all_solutions.append(sol)
            trial_ids.append(trial_idx)
            costs.append(float(cost) if cost is not None else 0.0)
            index_to_inst.append(inst_idx)
    if not all_solutions:
        return torch.empty(0, 1, device=device), np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    # Group by instance: list of (global_index, sol) per instance
    by_inst: Dict[int, List[Tuple[int, List[int]]]] = {}
    for i, (inst_idx, sol) in enumerate(zip(index_to_inst, all_solutions)):
        if inst_idx not in by_inst:
            by_inst[inst_idx] = []
        by_inst[inst_idx].append((i, sol))
    basin_cache: Dict[str, dict] = {}
    # Embed per instance, collect (index, emb) then sort by index
    all_emb_index: List[Tuple[int, torch.Tensor]] = []
    for inst_idx in sorted(by_inst.keys()):
        inst_data = instance_data_by_idx.get(inst_idx)
        if inst_data is None:
            continue
        depot_xy = inst_data["depot_xy"].to(device)
        node_xy_demand = inst_data["node_xy_demand"].to(device)
        env.load(depot_xy, node_xy_demand, basin_cache)
        items = by_inst[inst_idx]
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            indices = [x[0] for x in batch]
            sols = [x[1] for x in batch]
            emb = embed_solutions(embedder, sols, env, basin_cache)
            for j, idx in enumerate(indices):
                all_emb_index.append((idx, emb[j : j + 1]))
    if not all_emb_index:
        return torch.empty(0, 1, device=device), np.array(trial_ids), np.array(costs)
    all_emb_index.sort(key=lambda x: x[0])
    embeddings = torch.cat([e for _, e in all_emb_index], dim=0)
    return embeddings, np.array(trial_ids), np.array(costs)


def make_trajectory_embedding_gif_and_html(
    frames: List[np.ndarray],
    frame_labels: List[str],
    gif_path: str,
    html_path: str,
    duration: float = 1.5,
) -> None:
    """Write GIF and HTML with slider from list of RGB frames."""
    try:
        import imageio
    except ImportError:
        print("[Trajectory] imageio not available, skip GIF/HTML.")
        return
    if not frames:
        return
    os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
    imageio.mimsave(gif_path, frames, duration=duration)
    print(f"[Trajectory] Saved GIF: {gif_path}")

    b64_frames = []
    for arr in frames:
        buf = io.BytesIO()
        imageio.imwrite(buf, arr, format="png")
        b64_frames.append(base64.standard_b64encode(buf.getvalue()).decode("ascii"))
    n_frames = len(b64_frames)
    frames_js = json.dumps(b64_frames)
    labels_js = json.dumps(frame_labels)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Trajectory embedding by epoch</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }}
    .controls {{ display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }}
    input[type="range"] {{ flex: 1; min-width: 200px; accent-color: #4a9; }}
    #frameInfo {{ min-width: 120px; }}
    #img {{ max-width: 100%; height: auto; display: block; }}
  </style>
</head>
<body>
  <h2>Trajectory embedding (PCA 2D) by epoch</h2>
  <div class="controls">
    <input type="range" id="slider" min="0" max="{n_frames - 1}" value="0" step="1">
    <span id="frameInfo">Frame 1 / {n_frames}</span>
  </div>
  <img id="img" alt="frame">
  <script>
    const frames = {frames_js};
    const labels = {labels_js};
    const slider = document.getElementById("slider");
    const img = document.getElementById("img");
    const frameInfo = document.getElementById("frameInfo");
    function showFrame(i) {{
      i = Math.max(0, Math.min(i, frames.length - 1));
      img.src = "data:image/png;base64," + frames[i];
      frameInfo.textContent = (labels[i] || ("Frame " + (i + 1))) + " (" + (i + 1) + " / " + frames.length + ")";
    }}
    slider.addEventListener("input", function() {{ showFrame(parseInt(this.value, 10)); }});
    showFrame(0);
  </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[Trajectory] Saved HTML: {html_path}")


def analyze_trajectory_embedding(args: argparse.Namespace, device: torch.device) -> None:
    """
    Load N trials from a single trajectory.jsonl (same instance, distinct local optima).
    Embed at each checkpoint, PCA to 2D, plot two panels (color by trial, color by cost).
    Build GIF and HTML over epochs. Only runs when --trajectory_embed_viz is set.
    """
    path = getattr(args, "trajectory_embed_path", None) or getattr(args, "trajectory_path", None)
    if not path and getattr(args, "trajectory_paths", None):
        path = args.trajectory_paths[0]
    if not path or not os.path.isfile(path):
        return
    N = getattr(args, "trajectory_embed_n_trials", 10)
    strategy = getattr(args, "trajectory_embed_strategy", "distinct_optima")
    min_points = getattr(args, "trajectory_embed_min_points", 5)
    if strategy == "stratified_run":
        K = getattr(args, "trajectory_embed_stratify_k", 10)
        selected_trials = select_N_trials_stratified_by_run(
            path, K=K, run_id=getattr(args, "run_id", None), min_points=min_points
        )
        if selected_trials is None:
            print(f"[TrajectoryEmbed] Could not select stratified trials (distinct optima + K={K} segments, >={min_points} points) from {path}. Skip.")
            return
    else:
        selected_trials = select_N_trials_different_optima(
            path, N=N, seed=args.trajectory_seed, min_points=min_points
        )
        if selected_trials is None:
            print(f"[TrajectoryEmbed] Could not select {N} trials with distinct local optima from {path}. Skip.")
            return
    num_trials = len(selected_trials)
    print(f"[TrajectoryEmbed] Strategy={strategy}, using {num_trials} trials from same instance: {path}")
    indices = sorted({inst_idx for inst_idx, _ in selected_trials})
    instance_list = load_instances_pkl(args.instance_pkl, device, indices, {})
    instance_data_by_idx = dict(zip(indices, instance_list))

    # Optional: load HGS global optimum for this instance (single instance assumed).
    # We assume /home/jieyi/hgs_cvrp100_uniform.pkl format: list[(cost, node_sequence)].
    hgs_solution_seq: Optional[List[int]] = None
    hgs_cost: Optional[float] = None
    hgs_path = getattr(args, "trajectory_embed_hgs_path", None)
    if hgs_path:
        import pickle

        inst_idx_for_hgs = indices[0]
        with open(hgs_path, "rb") as f:
            hgs_data = pickle.load(f)
        cost_val, routes_val = hgs_data[inst_idx_for_hgs]

        # Convert HGS node sequence to list of routes, then to our 1D route-like sequence.
        routes = convert_hgs_routes_to_format(routes_val)
        seq: List[int] = [0]
        for r in routes:
            if not r:
                continue
            for node in r[1:]:
                seq.append(int(node))
        if not seq or seq[-1] != 0:
            seq.append(0)
        hgs_solution_seq = []
        for x in seq:
            if x == 0 and hgs_solution_seq and hgs_solution_seq[-1] == 0:
                continue
            hgs_solution_seq.append(x)

        hgs_cost = float(cost_val)

    all_ckpts = glob.glob(os.path.join(args.checkpoint_dir, "*.pt"))
    if not all_ckpts:
        print("[Trajectory] No checkpoints found.")
        return
    # Sort by (stage, epoch) so GIF order is s1_epoch10, s1_epoch20, ..., s2_epoch10, ...
    def ckpt_sort_key(p: str) -> Tuple[int, int]:
        base = os.path.splitext(os.path.basename(p))[0]
        stage, epoch = parse_stage_epoch(base)
        return (stage if stage is not None else 99, epoch if epoch is not None else -1)
    ckpts = sorted(all_ckpts, key=ckpt_sort_key)
    plot_dir = os.path.join(args.checkpoint_dir, "analysis_trajectory_embed")
    os.makedirs(plot_dir, exist_ok=True)
    env = CVRPEnv(problem_size=args.problem_size, device=device)

    # Separate frames per stage (s1, s2)
    frames_s1: List[np.ndarray] = []
    labels_s1: List[str] = []
    frames_s2: List[np.ndarray] = []
    labels_s2: List[str] = []
    
    import matplotlib.pyplot as plt

    for ckpt_path in ckpts:
        base = os.path.splitext(os.path.basename(ckpt_path))[0]
        stage, epoch = parse_stage_epoch(base)
        print(f"[Trajectory] Epoch {base}")
        ckpt = torch.load(ckpt_path, map_location=device)
        embedder = build_model(args, device, encoder_state=_get_encoder_state(ckpt))
        _load_embedder(embedder, ckpt)
        embedder.eval()

        with torch.no_grad():
            emb, trial_ids, costs = embed_trajectory_trials(
                embedder, selected_trials, instance_data_by_idx, env, device,
                batch_size=getattr(args, "trajectory_batch_size", 64),
            )
        if emb.size(0) == 0:
            continue
        X = emb.cpu().numpy()
        # PCA on trial embeddings
        X_mean = X.mean(axis=0)
        X_centered = X - X_mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        coords = (X_centered @ Vt.T[:, :2]).astype(np.float64)

        # If HGS solution is available, embed and project to same PCA space
        coords_hgs: Optional[np.ndarray] = None
        if hgs_solution_seq is not None:
            inst_idx_for_hgs = indices[0]
            inst_data = instance_data_by_idx[inst_idx_for_hgs]
            depot_xy = inst_data["depot_xy"].to(device)
            node_xy_demand = inst_data["node_xy_demand"].to(device)
            basin_cache_hgs: Dict[str, dict] = {}
            env.load(depot_xy, node_xy_demand, basin_cache_hgs)
            emb_hgs = embed_solutions(embedder, [hgs_solution_seq], env, basin_cache_hgs)
            x_hgs = emb_hgs.detach().cpu().numpy()[0]
            x_hgs_centered = x_hgs - X_mean
            coords_hgs = (x_hgs_centered @ Vt.T[:, :2]).astype(np.float64)

        # Index of each trial's last point (local optimum) in coords/costs
        local_optima_indices: List[int] = []
        for tid in range(num_trials):
            idx = np.where(trial_ids == tid)[0]
            if len(idx) > 0:
                local_optima_indices.append(int(idx[-1]))
            else:
                local_optima_indices.append(-1)
        # Consistent color per trial
        colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(num_trials, 1)))

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # Left: color by trial (same trial = same color)
        star_points: List[Tuple[float, float]] = []
        for tid in range(num_trials):
            mask = trial_ids == tid
            if not np.any(mask):
                continue
            axes[0].scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=8,
                alpha=0.7,
                label=f"Trial {tid}",
                color=colors[tid],
            )
        for tid in range(num_trials):
            i = local_optima_indices[tid]
            if i < 0:
                continue
            x, y = coords[i, 0], coords[i, 1]
            axes[0].scatter(
                x,
                y,
                marker="*",
                s=30,
                color=colors[tid],
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
            )
            axes[0].annotate(
                f"{costs[i]:.1f}",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
            )
            star_points.append((x, y))
        # Optional: HGS global optimum
        if coords_hgs is not None and hgs_cost is not None:
            xh, yh = float(coords_hgs[0]), float(coords_hgs[1])
            axes[0].scatter(
                xh,
                yh,
                marker="*",
                s=80,
                color="red",
                edgecolors="black",
                linewidths=1.0,
                zorder=6,
                label="HGS",
            )
            axes[0].annotate(
                f"HGS {hgs_cost * 100:.1f}",
                (xh, yh),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
            )
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        # axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_title("By trial")
        axes[0].legend(loc="best", fontsize=7)
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        # Right: color by cost (lower = darker)
        sc = axes[1].scatter(coords[:, 0], coords[:, 1], c=costs, s=8, alpha=0.7, cmap="viridis")
        for tid in range(num_trials):
            i = local_optima_indices[tid]
            if i < 0:
                continue
            x, y = coords[i, 0], coords[i, 1]
            axes[1].scatter(
                x,
                y,
                marker="*",
                s=30,
                color=colors[tid],
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
            )
            axes[1].annotate(f"{costs[i]:.1f}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=7)
        if coords_hgs is not None and hgs_cost is not None:
            xh, yh = float(coords_hgs[0]), float(coords_hgs[1])
            axes[1].scatter(
                xh,
                yh,
                marker="*",
                s=80,
                color="red",
                edgecolors="black",
                linewidths=1.0,
                zorder=6,
                label="HGS (global optima)",
            )
            axes[1].annotate(
                f"HGS {hgs_cost * 100:.1f}",
                (xh, yh),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
            )
        # Arrows connecting local optima in search order (trial 0 -> 1 -> ... )
        for (x0, y0), (x1, y1) in zip(star_points[:-1], star_points[1:]):
            axes[0].annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="black", linewidth=0.8, alpha=0.7),
                zorder=4,
            )
            axes[1].annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="black", linewidth=0.8, alpha=0.7),
                zorder=4,
            )
        plt.colorbar(sc, ax=axes[1], label="Cost")
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")
        # axes[1].set_aspect("equal", adjustable="box")
        axes[1].set_title("By cost (lower = darker)")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        fig.suptitle(f"Epoch: {base}", fontsize=10)
        plt.tight_layout()

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf, dtype=np.uint8)[..., :3].copy()
        if stage == 1:
            frames_s1.append(frame)
            labels_s1.append(base)
        elif stage == 2:
            frames_s2.append(frame)
            labels_s2.append(base)
        plt.close(fig)

    duration = getattr(args, "trajectory_embed_gif_duration", 3.0)
    if frames_s1:
        gif_path = os.path.join(plot_dir, "trajectory_embed_s1_by_epoch.gif")
        html_path = os.path.join(plot_dir, "trajectory_embed_s1_by_epoch.html")
        make_trajectory_embedding_gif_and_html(frames_s1, labels_s1, gif_path, html_path, duration=duration)
    if frames_s2:
        gif_path = os.path.join(plot_dir, "trajectory_embed_s2_by_epoch.gif")
        html_path = os.path.join(plot_dir, "trajectory_embed_s2_by_epoch.html")
        make_trajectory_embedding_gif_and_html(frames_s2, labels_s2, gif_path, html_path, duration=duration)


def compute_and_plot_trajectory_diversity(
    coords: np.ndarray,
    local_optima_indices: List[int],
    base: str,
    plot_dir: str,
    normalize: bool = False,
) -> Dict[str, float]:
    """
    Compute diversity metrics for local optima in 2D embedding space:
    pairwise distance stats and convex hull area. Save histogram and scatter+hull figure.
    If normalize=True, scale by trajectory radius (max distance from centroid of all points)
    so values are comparable across runs/checkpoints.
    """
    opt_xy = coords[np.asarray(local_optima_indices)]
    n = len(opt_xy)
    if n < 2:
        return {"num_optima": n, "mean_pairwise_dist": 0.0, "convex_hull_area": 0.0}

    try:
        from scipy.spatial.distance import pdist
        from scipy.spatial import ConvexHull
    except ImportError:
        return {"num_optima": n}

    # Scale = trajectory radius (max dist from centroid of full trajectory)
    centroid = np.mean(coords, axis=0)
    radii = np.linalg.norm(coords - centroid, axis=1)
    scale = float(np.max(radii)) if len(radii) > 0 else 1.0
    if scale <= 0:
        scale = 1.0

    d = pdist(opt_xy)
    mean_d = float(np.mean(d))
    std_d = float(np.std(d)) if len(d) > 1 else 0.0
    min_d = float(np.min(d))
    max_d = float(np.max(d))

    hull_area = 0.0
    if n >= 3:
        try:
            hull = ConvexHull(opt_xy)
            hull_area = float(hull.volume)  # in 2D, volume = area
        except Exception:
            pass

    metrics = {
        "num_optima": n,
        "mean_pairwise_dist": mean_d,
        "std_pairwise_dist": std_d,
        "min_pairwise_dist": min_d,
        "max_pairwise_dist": max_d,
        "convex_hull_area": hull_area,
    }
    if normalize:
        metrics["scale"] = scale
        metrics["mean_pairwise_dist_norm"] = mean_d / scale
        metrics["std_pairwise_dist_norm"] = std_d / scale
        metrics["min_pairwise_dist_norm"] = min_d / scale
        metrics["max_pairwise_dist_norm"] = max_d / scale
        metrics["convex_hull_area_norm"] = hull_area / (scale * scale) if scale > 0 else 0.0
        print(
            f"[RunTrajectory] {base} diversity (normalized by scale={scale:.4f}): "
            f"n_opt={n}, mean_d_norm={metrics['mean_pairwise_dist_norm']:.4f}, "
            f"std_d_norm={metrics['std_pairwise_dist_norm']:.4f}, hull_area_norm={metrics['convex_hull_area_norm']:.4f}"
        )
    else:
        print(f"[RunTrajectory] {base} diversity: n_opt={n}, mean_d={mean_d:.4f}, std_d={std_d:.4f}, hull_area={hull_area:.4f}")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(d, bins=min(30, max(5, len(d) // 2)), color="steelblue", edgecolor="white", alpha=0.8)
        axes[0].set_xlabel("Pairwise distance (2D)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Local optima pairwise distances")
        axes[0].axvline(mean_d, color="red", linestyle="--", label=f"mean={mean_d:.3f}")
        axes[0].legend()

        axes[1].scatter(opt_xy[:, 0], opt_xy[:, 1], s=40, c="tab:orange", alpha=0.8, zorder=2)
        if n >= 3 and hull_area > 0:
            try:
                hull = ConvexHull(opt_xy)
                for simplex in hull.simplices:
                    axes[1].plot(opt_xy[simplex, 0], opt_xy[simplex, 1], "k-", alpha=0.5, linewidth=1)
            except Exception:
                pass
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")
        hull_title = f"hull_area_norm={hull_area / (scale * scale):.3f}" if normalize and scale > 0 else f"hull area={hull_area:.2f}"
        axes[1].set_title(f"Local optima ({hull_title})")
        axes[1].set_aspect("equal", adjustable="datalim")
        fig.suptitle(f"{base} — trajectory diversity", fontsize=10)
        plt.tight_layout()
        hist_path = os.path.join(plot_dir, f"diversity_{base}.png")
        plt.savefig(hist_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[RunTrajectory] Saved {hist_path}")
    except Exception as e:
        print(f"[RunTrajectory] Diversity plot failed: {e}")
    return metrics


def _segmented_diversity_one_mode(
    target_xy: np.ndarray, all_bg_xy: np.ndarray, base: str,
    plot_dir: str, n_segments: int, normalize: bool,
    mode_label: str, mode_suffix: str,
    point_size: int = 25,
) -> None:
    """Internal: compute and plot segmented diversity for one mode (optima or all)."""
    from scipy.spatial.distance import pdist
    from scipy.spatial import ConvexHull
    import matplotlib.pyplot as plt

    n = len(target_xy)
    if n < 3:
        print(f"[SegDiversity-{mode_suffix}] Too few points ({n}), skip.")
        return

    centroid = np.mean(all_bg_xy, axis=0)
    scale = float(np.max(np.linalg.norm(all_bg_xy - centroid, axis=1)))
    if scale <= 0:
        scale = 1.0

    seg_hull_areas: List[float] = []
    seg_max_dists: List[float] = []
    seg_pts: List[np.ndarray] = []
    for j in range(n_segments):
        lo = j * n // n_segments
        hi = (j + 1) * n // n_segments
        if lo >= hi:
            seg_pts.append(np.empty((0, 2)))
            seg_hull_areas.append(0.0)
            seg_max_dists.append(0.0)
            continue
        pts = target_xy[lo:hi]
        seg_pts.append(pts)
        seg_max_dists.append(float(np.max(pdist(pts))) if len(pts) >= 2 else 0.0)
        if len(pts) >= 3:
            try:
                seg_hull_areas.append(float(ConvexHull(pts).volume))
            except Exception:
                seg_hull_areas.append(0.0)
        else:
            seg_hull_areas.append(0.0)

    if normalize:
        seg_hull_areas = [a / (scale * scale) for a in seg_hull_areas]
        seg_max_dists = [d / scale for d in seg_max_dists]

    # --- Figure 1: subplots ---
    ncols = min(5, n_segments)
    nrows = (n_segments + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = np.asarray(axes).flatten() if n_segments > 1 else [axes]
    xmin, xmax = all_bg_xy[:, 0].min(), all_bg_xy[:, 0].max()
    ymin, ymax = all_bg_xy[:, 1].min(), all_bg_xy[:, 1].max()
    pad = max(xmax - xmin, ymax - ymin) * 0.1
    for j in range(n_segments):
        ax = axes_flat[j]
        pts = seg_pts[j]
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.scatter(all_bg_xy[:, 0], all_bg_xy[:, 1], s=2, c="lightgray", alpha=0.3, zorder=1)
        if len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], s=point_size, c="tab:orange", zorder=3)
        if len(pts) >= 3 and seg_hull_areas[j] > 0:
            try:
                hull = ConvexHull(pts)
                for simplex in hull.simplices:
                    ax.plot(pts[simplex, 0], pts[simplex, 1], "k-", alpha=0.6, linewidth=1)
            except Exception:
                pass
        lo_idx = j * n // n_segments
        hi_idx = (j + 1) * n // n_segments
        ax.set_title(f"Seg {j} ({lo_idx}–{hi_idx-1})\narea={seg_hull_areas[j]:.3f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n_segments, len(axes_flat)):
        axes_flat[j].set_visible(False)
    norm_str = " (normalized)" if normalize else ""
    fig.suptitle(f"{base} — seg diversity ({mode_label}){norm_str}", fontsize=11)
    plt.tight_layout()
    path1 = os.path.join(plot_dir, f"seg_diversity_subplots_{mode_suffix}_{base}.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SegDiversity] Saved {path1}")

    # --- Figure 2: trend ---
    fig2, ax1 = plt.subplots(figsize=(7, 4))
    xs = list(range(n_segments))
    color1 = "tab:blue"
    ax1.plot(xs, seg_hull_areas, "-o", color=color1, label="Hull area")
    ax1.set_xlabel("Segment index (search order)")
    ax1.set_ylabel("Hull area", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.plot(xs, seg_max_dists, "-s", color=color2, label="Max pairwise dist")
    ax2.set_ylabel("Max pairwise dist", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    fig2.suptitle(f"{base} — diversity trend ({mode_label}){norm_str}", fontsize=11)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    plt.tight_layout()
    path2 = os.path.join(plot_dir, f"seg_diversity_trend_{mode_suffix}_{base}.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[SegDiversity] Saved {path2}")

    for j in range(n_segments):
        lo_idx = j * n // n_segments
        hi_idx = (j + 1) * n // n_segments
        print(f"  Seg {j} ({lo_idx}–{hi_idx-1}): hull_area={seg_hull_areas[j]:.4f}, max_dist={seg_max_dists[j]:.4f}, n={len(seg_pts[j])}")


def compute_and_plot_segmented_diversity(
    coords: np.ndarray, local_optima_indices: List[int], base: str,
    plot_dir: str, n_segments: int = 10, normalize: bool = False,
) -> None:
    """Segmented diversity for both local optima and all solutions."""
    opt_xy = coords[np.asarray(local_optima_indices)]
    # Version 1: local optima only
    _segmented_diversity_one_mode(
        opt_xy, coords, base, plot_dir, n_segments, normalize,
        mode_label="local optima", mode_suffix="optima", point_size=25,
    )
    # Version 2: all solutions
    _segmented_diversity_one_mode(
        coords, coords, base, plot_dir, n_segments, normalize,
        mode_label="all solutions", mode_suffix="all", point_size=5,
    )


def load_all_runs_from_trajectory(
    trajectory_path: str, max_runs: int = 10,
) -> List[Tuple[Any, int, List[Dict], List[int]]]:
    """Load up to max_runs runs from trajectory.jsonl.
    Returns list of (run_id, instance_index, full_sequence, local_optima_indices)."""
    trials_by_key: Dict[Tuple[Any, Any], List[Dict]] = {}
    with open(trajectory_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid, tid = rec.get("run_id"), rec.get("trial_id")
            if rid is None or tid is None:
                continue
            key = (rid, tid)
            if key not in trials_by_key:
                trials_by_key[key] = []
            trials_by_key[key].append(rec)
    # Group by run_id
    by_run: Dict[Any, List[Tuple[Any, List[Dict]]]] = {}
    for (rid, tid), recs in trials_by_key.items():
        if rid not in by_run:
            by_run[rid] = []
        by_run[rid].append((tid, recs))
    instance_index = parse_instance_index_from_trajectory_path(trajectory_path)
    results = []
    for rid in sorted(by_run.keys())[:max_runs]:
        run_trials = sorted(by_run[rid], key=lambda x: x[0])
        full_sequence: List[Dict] = []
        local_optima_indices: List[int] = []
        for _, recs in run_trials:
            recs_sorted = sorted(recs, key=lambda r: (r.get("global_iter", 0) or 0, r.get("local_iter", 0) or 0))
            for r in recs_sorted:
                if r.get("solution_flat") is not None:
                    full_sequence.append(r)
            if recs_sorted:
                local_optima_indices.append(len(full_sequence) - 1)
        if full_sequence and local_optima_indices:
            results.append((rid, instance_index, full_sequence, local_optima_indices))
    return results


def analyze_multi_run_diversity(args: argparse.Namespace, device: torch.device) -> None:
    """Load 10 runs from one instance, embed ALL solutions jointly (PCA 2D).
    Plot 1: all solutions colored by run. Plot 2: local optima colored by run + per-run hull.
    Compute per-run hull_area / max_dist for both cases, correlate with per-run best cost."""
    path = getattr(args, "multi_run_path", None) or getattr(args, "trajectory_path", None)
    if not path and getattr(args, "trajectory_paths", None):
        path = args.trajectory_paths[0]
    if not path or not os.path.isfile(path):
        return
    n_runs = getattr(args, "multi_run_n", 10)
    all_runs = load_all_runs_from_trajectory(path, max_runs=n_runs)
    if len(all_runs) < 2:
        print(f"[MultiRunDiv] Only {len(all_runs)} run(s) found, need >=2. Skip.")
        return
    instance_index = all_runs[0][1]
    print(f"[MultiRunDiv] Loaded {len(all_runs)} runs for instance {instance_index} from {path}")

    indices = [instance_index]
    instance_list = load_instances_pkl(args.instance_pkl, device, indices, {})
    instance_data_by_idx = dict(zip(indices, instance_list))
    ckpts = get_last_epoch_checkpoints_per_stage(args.checkpoint_dir)
    if not ckpts:
        print("[MultiRunDiv] No checkpoints found.")
        return
    plot_dir = os.path.join(args.checkpoint_dir, "analysis_multi_run")
    os.makedirs(plot_dir, exist_ok=True)
    env = CVRPEnv(problem_size=args.problem_size, device=device)
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import pdist
    from scipy.spatial import ConvexHull
    from scipy.stats import spearmanr

    for ckpt_path in ckpts:
        base = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(f"[MultiRunDiv] Checkpoint {base}")
        ckpt = torch.load(ckpt_path, map_location=device)
        embedder = build_model(args, device, encoder_state=_get_encoder_state(ckpt))
        _load_embedder(embedder, ckpt)
        embedder.eval()

        # Embed all runs, track per-run ranges
        all_embs: List[torch.Tensor] = []
        run_ids_per_pt: List[int] = []       # run index (0..N-1) per point
        is_optima_per_pt: List[bool] = []     # True if this point is a local optimum
        run_best_costs: List[float] = []      # best cost per run
        run_optima_global_indices: List[List[int]] = []  # global indices of optima per run
        run_all_global_indices: List[List[int]] = []     # global indices of all pts per run
        global_idx = 0

        with torch.no_grad():
            for run_idx, (rid, inst_idx, full_seq, opt_indices) in enumerate(all_runs):
                emb = embed_run_full_sequence(embedder, inst_idx, full_seq, instance_data_by_idx, env, device)
                n_pts = emb.size(0)
                all_embs.append(emb)
                run_all_global_indices.append(list(range(global_idx, global_idx + n_pts)))
                opt_global = [global_idx + i for i in opt_indices if i < n_pts]
                run_optima_global_indices.append(opt_global)
                for i in range(n_pts):
                    run_ids_per_pt.append(run_idx)
                    is_optima_per_pt.append(i in opt_indices)
                # Best cost in this run
                costs_in_run = [float(r.get("cost", 1e9)) for r in full_seq if r.get("cost") is not None]
                run_best_costs.append(min(costs_in_run) if costs_in_run else 1e9)
                global_idx += n_pts

        if not all_embs:
            continue
        X = torch.cat(all_embs, dim=0).cpu().numpy()
        X_centered = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        coords = (U[:, :2] * S[:2]).astype(np.float64)

        n_runs_actual = len(all_runs)
        colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(n_runs_actual, 1)))

        # --- Plot 1: all solutions colored by run ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for r_idx in range(n_runs_actual):
            gi = run_all_global_indices[r_idx]
            axes[0].scatter(coords[gi, 0], coords[gi, 1], s=3, alpha=0.4, color=colors[r_idx], label=f"Run {r_idx}")
        axes[0].set_title("All solutions by run")
        axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
        axes[0].legend(loc="best", fontsize=6, markerscale=3)
        axes[0].set_xticks([]); axes[0].set_yticks([])

        # --- Plot 2: local optima colored by run + per-run hull ---
        for r_idx in range(n_runs_actual):
            gi = run_optima_global_indices[r_idx]
            if not gi:
                continue
            opt_xy = coords[gi]
            axes[1].scatter(opt_xy[:, 0], opt_xy[:, 1], s=20, alpha=0.7, color=colors[r_idx], label=f"Run {r_idx}")
            if len(opt_xy) >= 3:
                try:
                    hull = ConvexHull(opt_xy)
                    for simplex in hull.simplices:
                        axes[1].plot(opt_xy[simplex, 0], opt_xy[simplex, 1], "-", color=colors[r_idx], alpha=0.5, linewidth=1)
                except Exception:
                    pass
        axes[1].set_title("Local optima by run + hull")
        axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
        axes[1].legend(loc="best", fontsize=6, markerscale=2)
        axes[1].set_xticks([]); axes[1].set_yticks([])
        fig.suptitle(f"{base} — {n_runs_actual} runs (instance {instance_index})", fontsize=11)
        plt.tight_layout()
        path_scatter = os.path.join(plot_dir, f"multi_run_scatter_{base}.png")
        plt.savefig(path_scatter, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[MultiRunDiv] Saved {path_scatter}")

        # --- Per-run metrics ---
        run_opt_hull: List[float] = []
        run_opt_maxd: List[float] = []
        run_all_hull: List[float] = []
        run_all_maxd: List[float] = []
        for r_idx in range(n_runs_actual):
            # Optima
            gi_opt = run_optima_global_indices[r_idx]
            opt_xy = coords[gi_opt] if gi_opt else np.empty((0, 2))
            if len(opt_xy) >= 2:
                d = pdist(opt_xy)
                run_opt_maxd.append(float(np.max(d)))
            else:
                run_opt_maxd.append(0.0)
            if len(opt_xy) >= 3:
                try:
                    run_opt_hull.append(float(ConvexHull(opt_xy).volume))
                except Exception:
                    run_opt_hull.append(0.0)
            else:
                run_opt_hull.append(0.0)
            # All solutions
            gi_all = run_all_global_indices[r_idx]
            all_xy = coords[gi_all]
            if len(all_xy) >= 2:
                d = pdist(all_xy)
                run_all_maxd.append(float(np.max(d)))
            else:
                run_all_maxd.append(0.0)
            if len(all_xy) >= 3:
                try:
                    run_all_hull.append(float(ConvexHull(all_xy).volume))
                except Exception:
                    run_all_hull.append(0.0)
            else:
                run_all_hull.append(0.0)

        # --- Correlation with best cost ---
        best_costs_arr = np.array(run_best_costs)
        metric_pairs = [
            ("optima_hull_area", np.array(run_opt_hull)),
            ("optima_max_dist", np.array(run_opt_maxd)),
            ("all_sol_hull_area", np.array(run_all_hull)),
            ("all_sol_max_dist", np.array(run_all_maxd)),
        ]
        print(f"[MultiRunDiv] {base} — per-run metrics vs best_cost:")
        corr_results: Dict[str, float] = {}
        for name, vals in metric_pairs:
            rho, pval = spearmanr(vals, best_costs_arr)
            corr_results[name] = rho
            print(f"  {name}: Spearman rho={rho:.4f}, p={pval:.4f}")

        # --- Correlation scatter plots (2x2) ---
        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
        for ax, (name, vals) in zip(axes2.flatten(), metric_pairs):
            ax.scatter(vals, best_costs_arr, s=30, alpha=0.7)
            for r_idx in range(n_runs_actual):
                ax.annotate(str(r_idx), (vals[r_idx], best_costs_arr[r_idx]), fontsize=7)
            rho = corr_results[name]
            ax.set_xlabel(name)
            ax.set_ylabel("Best cost")
            ax.set_title(f"{name} vs best_cost (ρ={rho:.3f})")
        fig2.suptitle(f"{base} — diversity vs best cost", fontsize=11)
        plt.tight_layout()
        path_corr = os.path.join(plot_dir, f"multi_run_corr_{base}.png")
        plt.savefig(path_corr, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"[MultiRunDiv] Saved {path_corr}")

        # Print table
        print(f"  {'Run':>4} {'best_cost':>10} {'opt_hull':>10} {'opt_maxd':>10} {'all_hull':>10} {'all_maxd':>10}")
        for r_idx in range(n_runs_actual):
            print(f"  {r_idx:>4} {run_best_costs[r_idx]:>10.1f} {run_opt_hull[r_idx]:>10.3f} {run_opt_maxd[r_idx]:>10.3f} {run_all_hull[r_idx]:>10.3f} {run_all_maxd[r_idx]:>10.3f}")

        # --- Segmented diversity (two versions: all solutions & local optima only) ---
        n_segments = getattr(args, "diversity_n_segments", 10)
        for mode, mode_label, mode_suffix in [("all", "all solutions", "all"), ("optima", "local optima", "optima")]:
            fig_seg, axes_seg = plt.subplots(n_runs_actual, n_segments, figsize=(2 * n_segments, 2 * n_runs_actual))
            if n_runs_actual == 1:
                axes_seg = axes_seg[np.newaxis, :]
            seg_hull_runs: List[List[float]] = []
            seg_maxd_runs: List[List[float]] = []
            for r_idx in range(n_runs_actual):
                if mode == "all":
                    gi = run_all_global_indices[r_idx]
                else:
                    gi = run_optima_global_indices[r_idx]
                pts_xy = coords[gi] if gi else np.empty((0, 2))
                n_pts = len(pts_xy)
                seg_size = max(1, n_pts // n_segments)
                seg_hulls: List[float] = []
                seg_maxds: List[float] = []
                for s in range(n_segments):
                    start = s * seg_size
                    end = n_pts if s == n_segments - 1 else (s + 1) * seg_size
                    seg_xy = pts_xy[start:end]
                    ax = axes_seg[r_idx, s]
                    pt_size = 3 if mode == "all" else 15
                    ax.scatter(seg_xy[:, 0], seg_xy[:, 1], s=pt_size, alpha=0.5, color=colors[r_idx])
                    hull_area = 0.0
                    if len(seg_xy) >= 3:
                        try:
                            hull = ConvexHull(seg_xy)
                            hull_area = float(hull.volume)
                            for simplex in hull.simplices:
                                ax.plot(seg_xy[simplex, 0], seg_xy[simplex, 1], "-", color=colors[r_idx], alpha=0.7, lw=0.8)
                        except Exception:
                            pass
                    seg_hulls.append(hull_area)
                    max_d = float(np.max(pdist(seg_xy))) if len(seg_xy) >= 2 else 0.0
                    seg_maxds.append(max_d)
                    ax.set_title(f"R{r_idx}S{s}", fontsize=6)
                    ax.set_xticks([]); ax.set_yticks([])
                seg_hull_runs.append(seg_hulls)
                seg_maxd_runs.append(seg_maxds)
            fig_seg.suptitle(f"{base} — segmented diversity ({mode_label})", fontsize=10)
            plt.tight_layout()
            path_seg_sub = os.path.join(plot_dir, f"multi_run_seg_subplots_{mode_suffix}_{base}.png")
            plt.savefig(path_seg_sub, dpi=120, bbox_inches="tight")
            plt.close(fig_seg)
            print(f"[MultiRunDiv] Saved {path_seg_sub}")

            # Trend plot
            fig_trend, axes_trend = plt.subplots(1, 2, figsize=(12, 5))
            seg_x = np.arange(n_segments)
            for r_idx in range(n_runs_actual):
                axes_trend[0].plot(seg_x, seg_hull_runs[r_idx], marker="o", markersize=3, label=f"Run {r_idx}", color=colors[r_idx])
                axes_trend[1].plot(seg_x, seg_maxd_runs[r_idx], marker="o", markersize=3, label=f"Run {r_idx}", color=colors[r_idx])
            axes_trend[0].set_xlabel("Segment"); axes_trend[0].set_ylabel("Hull area")
            axes_trend[0].set_title(f"Hull area vs segment ({mode_label})")
            axes_trend[0].legend(fontsize=6, loc="best")
            axes_trend[1].set_xlabel("Segment"); axes_trend[1].set_ylabel("Max pairwise dist")
            axes_trend[1].set_title(f"Max dist vs segment ({mode_label})")
            axes_trend[1].legend(fontsize=6, loc="best")
            fig_trend.suptitle(f"{base} — segmented diversity trend ({mode_label})", fontsize=11)
            plt.tight_layout()
            path_seg_trend = os.path.join(plot_dir, f"multi_run_seg_trend_{mode_suffix}_{base}.png")
            plt.savefig(path_seg_trend, dpi=150, bbox_inches="tight")
            plt.close(fig_trend)
            print(f"[MultiRunDiv] Saved {path_seg_trend}")


def analyze_run_trajectory_embedding(args: argparse.Namespace, device: torch.device) -> None:
    """
    Pick one cuopt run from trajectory.jsonl. Use only the last-epoch checkpoint per stage (s1, s2).
    For each such checkpoint: one GIF/HTML; each frame = one trial: left = that trial's trajectory, right = that trial's local optimum (one point).
    """
    trajectory_path = getattr(args, "trajectory_path", None) or (getattr(args, "run_trajectory_path", None))
    if not trajectory_path or not os.path.isfile(trajectory_path):
        return
    run_id = getattr(args, "run_id", None)
    one_run = load_one_run_from_trajectory(trajectory_path, run_id=run_id)
    if one_run is None:
        print("[RunTrajectory] No run found in trajectory, skip.")
        return
    run_id_val, instance_index, full_sequence, local_optima_indices = one_run
    print(f"[RunTrajectory] Run id={run_id_val}, instance={instance_index}, {len(full_sequence)} solutions, {len(local_optima_indices)} trials (local optima).")

    indices = [instance_index]
    instance_list = load_instances_pkl(args.instance_pkl, device, indices, {})
    instance_data_by_idx = dict(zip(indices, instance_list))
    ckpts = get_last_epoch_checkpoints_per_stage(args.checkpoint_dir)
    if not ckpts:
        print("[RunTrajectory] No s1_epoch* / s2_epoch* checkpoints found.")
        return
    plot_dir = os.path.join(args.checkpoint_dir, "analysis_run_trajectory")
    os.makedirs(plot_dir, exist_ok=True)
    env = CVRPEnv(problem_size=args.problem_size, device=device)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[RunTrajectory] matplotlib not available, skip.")
        return

    num_trials = len(local_optima_indices)
    trial_starts = [0] + [local_optima_indices[i] + 1 for i in range(num_trials - 1)]
    trial_ends = list(local_optima_indices)

    for ckpt_path in ckpts:
        base = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(f"[RunTrajectory] Checkpoint {base}")
        ckpt = torch.load(ckpt_path, map_location=device)
        embedder = build_model(args, device, encoder_state=_get_encoder_state(ckpt))
        _load_embedder(embedder, ckpt)
        embedder.eval()

        with torch.no_grad():
            emb = embed_run_full_sequence(
                embedder, instance_index, full_sequence, instance_data_by_idx, env, device,
                batch_size=getattr(args, "trajectory_batch_size", 64),
            )
        if emb.size(0) == 0:
            continue
        X = emb.cpu().numpy()
        X_centered = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        coords = (U[:, :2] * S[:2]).astype(np.float64)

        compute_and_plot_trajectory_diversity(
            coords, local_optima_indices, base, plot_dir,
            normalize=getattr(args, "diversity_normalize", False),
        )
        compute_and_plot_segmented_diversity(
            coords, local_optima_indices, base, plot_dir,
            n_segments=getattr(args, "diversity_n_segments", 10),
            normalize=getattr(args, "diversity_normalize", False),
        )

        frames: List[np.ndarray] = []
        frame_labels: List[str] = []
        for t in range(num_trials):
            # Left: cumulative trajectory from trial 0 to trial t (frame t builds on previous)
            end_cumul = trial_ends[t] + 1
            cumul_coords = coords[0:end_cumul]
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            if len(cumul_coords) > 1:
                axes[0].plot(cumul_coords[:, 0], cumul_coords[:, 1], "b-", alpha=0.5, linewidth=0.8)
            axes[0].scatter(cumul_coords[:, 0], cumul_coords[:, 1], s=10, c="tab:blue", alpha=0.7)
            axes[0].set_xlabel("PC1")
            axes[0].set_ylabel("PC2")
            axes[0].set_title(f"Trials 0–{t} trajectory")
            axes[0].set_aspect("equal", adjustable="datalim")
            axes[0].set_xticks([])
            axes[0].set_yticks([])

            # Right: local optima 0..t with arrows between consecutive optima
            opt_xy_t = coords[[local_optima_indices[i] for i in range(t + 1)]]
            axes[1].scatter(opt_xy_t[:, 0], opt_xy_t[:, 1], s=60, c="tab:orange", alpha=0.9, zorder=2)
            for i in range(len(opt_xy_t) - 1):
                axes[1].annotate(
                    "",
                    xy=(opt_xy_t[i + 1, 0], opt_xy_t[i + 1, 1]),
                    xytext=(opt_xy_t[i, 0], opt_xy_t[i, 1]),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
                )
            axes[1].set_xlabel("PC1")
            axes[1].set_ylabel("PC2")
            axes[1].set_title("Local optima (search order)")
            axes[1].set_aspect("equal", adjustable="datalim")
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            fig.suptitle(f"{base} — Trials 0–{t}", fontsize=10)
            plt.tight_layout()

            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf, dtype=np.uint8)[..., :3].copy()
            frames.append(frame)
            frame_labels.append(f"Trials 0–{t}")
            plt.close(fig)

        if frames:
            gif_path = os.path.join(plot_dir, f"run_trajectory_{base}.gif")
            html_path = os.path.join(plot_dir, f"run_trajectory_{base}.html")
            make_trajectory_embedding_gif_and_html(
                frames, frame_labels, gif_path, html_path,
                duration=getattr(args, "trajectory_gif_duration", 1.5),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline visualization/validation for checkpoints.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/home/jieyi/cuopt/out/20260205_095131_0-49_2stages",
        help="Directory containing s1_epoch*.pt / s2_epoch*.pt checkpoints.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["1", "2", "both", "trajectory"],
        default="both",
        help="Which to run: 1 / 2 / both (S1+S2 val), or trajectory (only trajectory embedding viz).",
    )

    # Model / data settings (should match training)
    parser.add_argument("--problem_size", type=int, default=100)
    parser.add_argument("--instance_pkl", type=str, default="/home/jieyi/cvrp100_uniform.pkl")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--supplement_feature_dim", type=int, default=5)
    parser.add_argument("--use_l2_normalize", action="store_true", default=True, help="L2-normalize pooled embeddings in SolutionEmbedder.forward during analysis (default off).")
    parser.add_argument("--val_data_1a1n10d", type=str, default="/home/jieyi/cuopt/basin_datasets0_analyze/val_data_1a1n10d.jsonl"); parser.add_argument("--val_data_1p1n", type=str, default="/home/jieyi/cuopt/perturb_k1_collect/val_data_1p1n.jsonl")
    parser.add_argument("--batch_size2", type=int, default=128, help="Batch size for Stage 2 val.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Trajectory paths (used by run-trajectory viz and optionally by trajectory-embed viz)
    parser.add_argument("--trajectory_path", type=str, default=None, help="Single trajectory.jsonl (e.g. for run-trajectory or trajectory-embed).")
    _default_trajectory_dir = "/home/jieyi/cuopt/basin_datasets0"
    _default_trajectory_basename = "cvrp100_uniform.pkl"
    _default_trajectory_paths = [
        os.path.join(_default_trajectory_dir, f"{_default_trajectory_basename}#{i}", "trajectory.jsonl")
        for i in (50, 51, 52, 53, 54)
    ]
    parser.add_argument("--trajectory_paths", type=str, nargs="*", default=_default_trajectory_paths, help="Multiple trajectory paths (for other uses).")
    # N trials from ONE instance: only runs when --trajectory_embed_viz is set
    parser.add_argument("--trajectory_embed_viz", action="store_true", help="Run N-trials embedding viz: same instance, N trials, PCA by epoch -> GIF+HTML.")
    parser.add_argument("--trajectory_embed_n_trials", type=int, default=10, help="Number of trials to use in trajectory_embed_viz (default 10).")
    parser.add_argument("--trajectory_embed_path", type=str, default=None, help="trajectory.jsonl for N-trials viz (default: --trajectory_path or first of --trajectory_paths).")
    parser.add_argument("--trajectory_embed_strategy", type=str, default="distinct_optima", choices=("distinct_optima", "stratified_run"),
        help="distinct_optima: N trials with distinct local optima; stratified_run: distinct local optima (required), then K equal segments, one per segment (>=5 points).")
    parser.add_argument("--trajectory_embed_stratify_k", type=int, default=10, help="Number of segments for stratified_run (K-way split over distinct-optima trials).")
    parser.add_argument("--trajectory_embed_min_points", type=int, default=5, help="Minimum number of solutions per trial for trajectory_embed selection (default 5).")
    parser.add_argument("--trajectory_embed_hgs_path", type=str, default="/home/jieyi/hgs_cvrp100_uniform.pkl", help="Optional: pickle with HGS solutions; if set, overlay global optimum in trajectory-embed plots.")
    parser.add_argument("--trajectory_seed", type=int, default=42, help="Random seed for selecting trials.")
    parser.add_argument("--trajectory_batch_size", type=int, default=64, help="Batch size for embedding trajectory solutions.")
    parser.add_argument("--trajectory_gif_duration", type=float, default=1.5, help="Seconds per frame in run-trajectory GIF.")
    parser.add_argument("--trajectory_embed_gif_duration", type=float, default=3.0, help="Seconds per frame in N-trials trajectory embed GIF (default 3.0).")

    # One-run trajectory viz: one run's full search path in 2D + local optima arrows
    parser.add_argument("--run_trajectory_path", type=str, default=None, help="trajectory.jsonl for run trajectory viz (default: same as --trajectory_path).")
    parser.add_argument("--run_id", type=int, default=None, help="run_id to use for run trajectory (default: first run in file).")
    parser.add_argument("--diversity_normalize", action="store_true", help="Normalize diversity metrics by trajectory radius for cross-run/checkpoint comparison.")
    parser.add_argument("--diversity_n_segments", type=int, default=10, help="Number of segments for segmented diversity analysis.")

    # Multi-run diversity: 10 runs from ONE instance, joint embedding, hull/maxdist vs best cost
    parser.add_argument("--multi_run_viz", action="store_true", help="Multi-run diversity analysis: load N runs from one instance, joint PCA, per-run diversity vs best cost.")
    parser.add_argument("--multi_run_path", type=str, default=None, help="trajectory.jsonl for multi-run analysis (default: --trajectory_path or first of --trajectory_paths).")
    parser.add_argument("--multi_run_n", type=int, default=10, help="Number of runs to load for multi-run analysis (default 10).")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f">> Using device: {device}")

    if args.stage in ("1", "both"):
        analyze_stage1(args, device)
    if args.stage in ("2", "both"):
        analyze_stage2(args, device)
    if getattr(args, "trajectory_embed_viz", False):
        analyze_trajectory_embedding(args, device)
    run_path = args.trajectory_path or args.run_trajectory_path or (args.trajectory_paths[0] if args.trajectory_paths else None)
    run_trajectory_requested = run_path and (args.stage == "trajectory" or args.trajectory_path or args.run_trajectory_path)
    if run_trajectory_requested and not getattr(args, "trajectory_embed_viz", False):
        if not args.run_trajectory_path and not args.trajectory_path:
            args.run_trajectory_path = run_path
        analyze_run_trajectory_embedding(args, device)
    if getattr(args, "multi_run_viz", False):
        analyze_multi_run_diversity(args, device)


if __name__ == "__main__":
    main()

