#!/usr/bin/env python3
"""
Train solution embeddings with contrastive learning.

Stage 1: Weighted InfoNCE with basin pairs.
Stage 2: Triplet Margin Loss with perturb data (anchor, positive, negative).
"""

import argparse
import hashlib
import math
import os
import pickle
import random
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import wasserstein_distance
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from dataloader import (
    TrainBatch,
    TripletBatch,
    build_loader_from_args,
    load_perturb_data,
    load_training_data_pairs,
    load_val_data_1a1n10d,
    load_val_data_1p1n,
    parse_instance_indices,
)
from CVRPEnv import CVRPEnv
from net import SolutionEmbedder
from helper import (
    copy_all_src,
    load_instances_pkl,
    plot_distance_histogram,
    plot_embedding_2d,
    seed_everything,
    make_triplet_gif,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _default_instance_pkl_path() -> str:
    """Prefer project-local instance pkl, fallback to home directory."""
    project_path = os.path.join(PROJECT_ROOT, "cvrp100_uniform.pkl")
    if os.path.isfile(project_path):
        return project_path
    return os.path.expanduser("~/cvrp100_uniform.pkl")


def _extract_cfg_value(config_text: str, key: str) -> Optional[str]:
    """Extract simple 'key: value' from wandb config.yaml text."""
    m = re.search(rf"{re.escape(key)}:\n\s+value:\s*(.+)", config_text)
    if not m:
        return None
    v = m.group(1).strip()
    if v in ("null", "None"):
        return None
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1]
    return v


def _parse_run_name_components(run_name: str) -> Tuple[Optional[datetime], Optional[str], Optional[str]]:
    """Parse run name like 'YYYYMMDD_HHMMSS_<instance_indices>_<note>'."""
    m = re.match(r"^(\d{8}_\d{6})(?:_(.*))?$", run_name)
    if not m:
        return None, None, None
    ts = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    rest = m.group(2) or ""
    if not rest:
        return ts, None, None
    if "_" in rest:
        inst_idx, note = rest.split("_", 1)
    else:
        inst_idx, note = rest, None
    return ts, inst_idx or None, note or None


def _infer_wandb_run_id_from_local(load_checkpoint: str, project_root: str) -> Optional[str]:
    """Infer historical wandb run id from local wandb folders."""
    prev_run_dir = os.path.basename(os.path.dirname(os.path.abspath(load_checkpoint)))
    target_ts, target_inst, target_note = _parse_run_name_components(prev_run_dir)
    if target_ts is None:
        return None

    best_id = None
    best_score = float("inf")
    run_pat = re.compile(r"^run-(\d{8}_\d{6})-([a-z0-9]+)$")

    for wb_root_name in ("wandb", "wandb_runtime"):
        wb_root = os.path.join(project_root, wb_root_name)
        if not os.path.isdir(wb_root):
            continue
        try:
            entries = os.listdir(wb_root)
        except OSError:
            continue

        for d in entries:
            mm = run_pat.match(d)
            if mm is None:
                continue
            run_ts = datetime.strptime(mm.group(1), "%Y%m%d_%H%M%S")
            run_id = mm.group(2)
            # Base score: temporal closeness (seconds) to output run timestamp.
            score = abs((run_ts - target_ts).total_seconds())

            cfg_path = os.path.join(wb_root, d, "files", "config.yaml")
            if os.path.isfile(cfg_path):
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg_txt = f.read()
                    cfg_inst = _extract_cfg_value(cfg_txt, "instance_indices")
                    cfg_note = _extract_cfg_value(cfg_txt, "note")
                    if target_inst is not None and cfg_inst == target_inst:
                        score -= 120.0
                    if target_note is not None and cfg_note == target_note:
                        score -= 120.0
                except OSError:
                    pass

            if score < best_score:
                best_score = score
                best_id = run_id

    # Conservative guard: skip if time too far away (> 12h).
    if best_id is not None and best_score < 12 * 3600:
        return best_id
    return None


def _extract_wandb_run_id_from_checkpoint(load_checkpoint: Optional[str]) -> Optional[str]:
    """Read wandb_run_id from checkpoint metadata if present."""
    if not load_checkpoint or not os.path.isfile(load_checkpoint):
        return None
    try:
        ckpt = torch.load(load_checkpoint, map_location="cpu")
    except Exception:
        return None
    run_id = ckpt.get("wandb_run_id")
    if isinstance(run_id, str) and run_id.strip():
        return run_id.strip()
    return None


def _parse_train_stages(train_stages: str) -> List[str]:
    """Parse comma-separated stage list like '1,3'."""
    if not train_stages:
        return []
    out: List[str] = []
    seen = set()
    for tok in train_stages.split(","):
        s = tok.strip()
        if s not in {"1", "2", "3"}:
            raise ValueError(f"Invalid stage '{s}' in --train_stages. Use comma-separated subset of 1,2,3.")
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _legacy_stage_to_stages(stage: str) -> List[str]:
    """Map legacy --stage argument to explicit stage list."""
    if stage == "1":
        return ["1"]
    if stage == "2":
        return ["2"]
    if stage == "3":
        return ["3"]
    if stage == "both":
        return ["1", "2"]
    if stage in {"all", "sequential", "joint"}:
        return ["1", "2", "3"]
    raise ValueError(f"Unsupported --stage value: {stage}")


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    """Warmup + cosine decay. Returns None if warmup_steps <= 0."""
    if warmup_steps <= 0:
        return None

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def embed_solutions(
    embedder: SolutionEmbedder,
    solutions: List[List[int]],
    env: CVRPEnv,
    basin_info_cache: Dict[str, dict],
) -> torch.Tensor:
    """Embed a list of solutions."""
    hashes = []
    for i, sol in enumerate(solutions):
        h = f"_tmp_{i}"
        basin_info_cache[h] = {"solution": sol}
        hashes.append(h)
    context = env.prepare_from_hashes(hashes)
    emb = embedder(context, env)
    for h in hashes:
        del basin_info_cache[h]
    return emb


def embed_multi_instance(
    embedder: SolutionEmbedder,
    inst_solutions: List[Tuple[int, List[int]]],
    instance_data_by_idx: Dict[int, dict],
    env: CVRPEnv,
    basin_info_cache: Dict[str, dict],
) -> torch.Tensor:
    """Embed solutions from potentially multiple instances, preserving input order."""
    by_inst: Dict[int, List[Tuple[int, List[int]]]] = {}
    for i, (idx, sol) in enumerate(inst_solutions):
        by_inst.setdefault(idx, []).append((i, sol))
    result: List[Optional[torch.Tensor]] = [None] * len(inst_solutions)
    for idx, items in by_inst.items():
        inst = instance_data_by_idx[idx]
        env.load(inst["depot_xy"], inst["node_xy_demand"], basin_info_cache)
        emb = embed_solutions(embedder, [sol for _, sol in items], env, basin_info_cache)
        for j, (orig_i, _) in enumerate(items):
            result[orig_i] = emb[j]
    return torch.stack(result)


def weighted_infonce_loss(
    embeddings: torch.Tensor,
    pair_indices: torch.Tensor,
    weights: torch.Tensor,
    temperature: float = 0.07,
    include_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Weighted InfoNCE:
      L = - sum_{(i,j)} W_ij * log( exp(sim(i,j)/tau) / sum_k exp(sim(i,k)/tau) )
    embeddings: (N, D), pair_indices: [(i,j), ...], weights: [W_ij, ...]
    If include_mask is given: (n_pairs, N), denominator for pair i is sum over k
    where include_mask[i,k]==1 of exp(sim(anchor_i, k)/tau). Used for masked_in_batch neg.
    """
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim = (embeddings @ embeddings.t()) / temperature
    anchors = pair_indices[:, 0]
    positives = pair_indices[:, 1]

    if include_mask is not None:
        # include_mask: (n_pairs, N), mask out co-occurred (invalid negatives).
        # We additionally drop self-similarity from the denominator.
        sim_for_anchors = sim.index_select(0, anchors)  # (n_pairs, N)
        n_pairs, n_emb = sim_for_anchors.size()

        # Start from include_mask, then zero-out self positions (k == anchor)
        mask = include_mask.clone()
        row_idx = torch.arange(n_pairs, device=mask.device)
        mask[row_idx, anchors] = 0.0  # remove self from denominator

        log_exp_sim = sim_for_anchors.masked_fill(mask == 0, float("-inf"))
        log_denom = torch.logsumexp(log_exp_sim, dim=1)
    else:
        # Standard InfoNCE: exclude self-similarity from denominator.
        sim_no_self = sim.clone()
        sim_no_self.fill_diagonal_(float("-inf"))
        log_denom = torch.logsumexp(sim_no_self, dim=1).index_select(0, anchors)

    log_num = sim[anchors, positives]
    loss = -(weights * (log_num - log_denom)).sum() / weights.sum()
    return loss


def train_one_batch(
    embedder: SolutionEmbedder,
    optimizer: torch.optim.Optimizer,
    batch: TrainBatch,
    env: CVRPEnv,
    device: torch.device,
    temperature: float,
    grad_clip: float = 0.0,
) -> float:

    context = env.prepare_from_hashes(batch.hashes)

    embedder.train()
    optimizer.zero_grad()
    emb = embedder(context, env)

    pair_idx = torch.tensor(batch.pair_indices, dtype=torch.long, device=device)
    w = torch.tensor(batch.weights, dtype=torch.float32, device=device)
    include_mask = batch.include_mask.to(device) if batch.include_mask is not None else None
    loss = weighted_infonce_loss(emb, pair_idx, w, temperature, include_mask=include_mask)

    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(embedder.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


def _compute_masked_neg_stats(batch: TrainBatch) -> Optional[Dict[str, float]]:
    """Compute valid negative counts for masked in-batch InfoNCE."""
    if batch.include_mask is None or not batch.pair_indices:
        return None
    mask = batch.include_mask
    pair_idx = torch.tensor(batch.pair_indices, dtype=torch.long)
    n_pairs, n_emb = mask.size()
    if pair_idx.size(0) != n_pairs or pair_idx.size(1) != 2:
        return None
    row_idx = torch.arange(n_pairs, dtype=torch.long)
    anchors = pair_idx[:, 0]
    positives = pair_idx[:, 1]
    # Keep only true negatives: remove anchor and positive from candidate set.
    neg_mask = mask.clone()
    neg_mask[row_idx, anchors] = 0.0
    neg_mask[row_idx, positives] = 0.0
    neg_counts = neg_mask.sum(dim=1)
    return {
        "mean": float(neg_counts.mean().item()),
        "min": float(neg_counts.min().item()),
        "max": float(neg_counts.max().item()),
        "zero_ratio": float((neg_counts == 0).float().mean().item()),
        "n_pairs": float(n_pairs),
        "n_emb": float(n_emb),
    }


def _log_histogram(
    list_a: List[float], list_b: List[float],
    hist_path: str, tag: str,
    wb_run, global_step: int,
    hist_key: str, wasserstein_key: str,
) -> None:
    """Plot distance/sim histogram, compute Wasserstein, optionally log to wandb."""
    plot_distance_histogram(list_a, list_b, save_path=hist_path)
    w = wasserstein_distance(list_a, list_b)
    print(f"[{tag}] Saved histogram to {hist_path} | Wasserstein={w:.6f}")
    if wb_run is not None:
        wandb.log({hist_key: wandb.Image(hist_path), wasserstein_key: w}, step=global_step)


def _embed_val_triplets_2d(
    val_records: List[Tuple[int, dict]],
    instance_data_by_idx: Dict[int, dict],
    embedder: SolutionEmbedder,
    env: CVRPEnv,
    basin_info_cache: Dict[str, dict],
    sol_keys: List[str],
    role_label_map: Dict[int, str],
    role_color_map: Dict[int, str],
    epoch_label: str,
    stage_prefix: str,
    plot_dir: str,
    wb_run, global_step: int,
) -> None:
    """Embed val records, plot 2D PCA + GIF. Shared by S1 and S2."""
    basin_cache: Dict[str, dict] = {}
    all_emb: List[torch.Tensor] = []
    all_inst_ids: List[int] = []
    all_groups: List[int] = []
    all_triplet_ids: List[int] = []
    embedder.eval()
    with torch.no_grad():
        for t_idx, (inst_idx, rec) in enumerate(val_records):
            inst = instance_data_by_idx[inst_idx]
            env.load(inst["depot_xy"], inst["node_xy_demand"], basin_cache)
            for role, key in enumerate(sol_keys):
                val = rec[key]
                if isinstance(val, list) and val and isinstance(val[0], list):
                    sols = val
                elif isinstance(val, list) and not val:
                    continue
                else:
                    sols = [val]
                emb = embed_solutions(embedder, sols, env, basin_info_cache)
                for j in range(emb.size(0)):
                    all_emb.append(emb[j])
                    all_inst_ids.append(inst_idx)
                    all_groups.append(role)
                    all_triplet_ids.append(t_idx)
    if not all_emb:
        return
    emb_t = torch.stack(all_emb, dim=0)
    emb_path = os.path.join(plot_dir, f"embedding_2d_{stage_prefix}_{epoch_label}.png")
    coords = plot_embedding_2d(emb_t, instance_ids=all_inst_ids, group_labels=all_groups, method="pca", save_path=emb_path)
    print(f"[{stage_prefix.upper()}] Saved embedding 2D to {emb_path}")
    if wb_run is not None:
        wandb.log({f"plot_{stage_prefix}/embedding_2d": wandb.Image(emb_path)}, step=global_step)
    gif_path = os.path.join(plot_dir, f"embedding_2d_{stage_prefix}_{epoch_label}.gif")
    make_triplet_gif(
        coords=coords,
        roles=np.asarray(all_groups, dtype=np.int64),
        triplet_ids=np.asarray(all_triplet_ids, dtype=np.int64),
        role_label_map=role_label_map,
        role_color_map=role_color_map,
        title_prefix=f"{stage_prefix.upper()} record",
        gif_path=gif_path,
        duration=2.0,
    )


def _collect_s3_sim_hist(
    pairs_by_inst: Dict[int, List[dict]],
    instance_data_by_idx: Dict[int, dict],
    embedder: SolutionEmbedder,
    env: CVRPEnv,
    basin_info_cache: Dict[str, dict],
    batch_size: int,
    sample_limit: int = 500,
) -> Tuple[List[float], List[float]]:
    """Collect pos/neg cosine similarities for S3 histogram (per-instance)."""
    list_pos: List[float] = []
    list_neg: List[float] = []
    sampled = 0
    for idx, recs in pairs_by_inst.items():
        if sampled >= sample_limit:
            break
        chunk = recs[: sample_limit - sampled]
        inst = instance_data_by_idx[idx]
        env.load(inst["depot_xy"], inst["node_xy_demand"], basin_info_cache)
        for i in range(0, len(chunk), batch_size):
            sub = chunk[i : i + batch_size]
            ea = embed_solutions(embedder, [t["anchor_solution"] for t in sub], env, basin_info_cache)
            ep = embed_solutions(embedder, [t["positive_solution"] for t in sub], env, basin_info_cache)
            sim_mat = torch.mm(ea, ep.t())
            list_pos.extend(sim_mat.diag().cpu().tolist())
            pos_hashes = [t["positive_basin_hash"] for t in sub]
            reachable_sets = [t["reachable_basin_hashes"] for t in sub]
            bsz = ea.size(0)

            all_hashes: set = set(pos_hashes)
            for rs in reachable_sets:
                all_hashes.update(rs)
            hash_to_id = {h: j for j, h in enumerate(all_hashes)}

            pos_hash_ids = torch.tensor([hash_to_id[ph] for ph in pos_hashes], device=ea.device)
            reachable_mat = torch.zeros(bsz, len(all_hashes), dtype=torch.bool, device=ea.device)
            for row, rs in enumerate(reachable_sets):
                if rs:
                    reachable_mat[row, [hash_to_id[h] for h in rs]] = True

            valid_neg = ~reachable_mat[:, pos_hash_ids]
            valid_neg.fill_diagonal_(False)
            list_neg.extend(sim_mat.masked_select(valid_neg).cpu().tolist())
        sampled += len(chunk)
    return list_pos, list_neg


def run_stage1(
    args: argparse.Namespace,
    embedder: SolutionEmbedder,
    save_dir: str,
    plot_dir: str,
    wb_run,
    global_step: int,
) -> int:
    """Stage 1: InfoNCE with basin pairs. Returns updated global_step."""
    print("\n" + "=" * 60)
    print("Stage 1: InfoNCE Training")
    print("=" * 60)

    optimizer = torch.optim.AdamW(embedder.parameters(), lr=args.lr1, weight_decay=args.weight_decay)
    loader, basin_data = build_loader_from_args(args, args.device)
    start_epoch1 = int(getattr(args, "resume_epoch1", 0) or 0)
    if start_epoch1 >= args.epochs1:
        print(f"[S1] Resume epoch {start_epoch1} >= target epochs {args.epochs1}, skip Stage 1.")
        return global_step
    total_steps_s1 = max(args.epochs1 - start_epoch1, 0) * len(loader)
    scheduler = make_lr_scheduler(optimizer, args.warmup_steps, total_steps_s1)
    env = CVRPEnv(problem_size=args.problem_size, device=args.device)

    # Fixed validation set for Stage 1: anchor + neighbour + 10 distant per line
    val_s1_records: List[Tuple[int, dict]] = []
    val_s1_instance_data_by_idx: Dict[int, dict] = {}
    if os.path.isfile(args.val_data_1a1n10d):
        val_s1_records = load_val_data_1a1n10d(args.val_data_1a1n10d)
        val_s1_indices = sorted(set(idx for idx, _ in val_s1_records))
        val_s1_instance_list = load_instances_pkl(args.instance_pkl, args.device, val_s1_indices, {})
        val_s1_instance_data_by_idx = dict(zip(val_s1_indices, val_s1_instance_list))
        print(
            f"Loaded Stage-1 val set: {args.val_data_1a1n10d} -> "
            f"{len(val_s1_records)} anchors, instances {val_s1_indices}"
        )
    else:
        print(f"Stage-1 val file not found: {args.val_data_1a1n10d}, skipping S1 validation/plots")

    def eval_stage1_on_val() -> Tuple[float, float, float, List[float], List[float], Optional[Tuple[int, dict, dict]]]:
        """Eval S1 on fixed val set. Returns (mean_d_ap, mean_d_an, ratio, list_d_ap, list_d_an, first_triplet)."""
        if not val_s1_records:
            return 0.0, 0.0, 0.0, [], [], None
        list_d_ap: List[float] = []
        list_d_an: List[float] = []
        first_triplet: Optional[Tuple[int, dict, dict]] = None
        for idx, rec in val_s1_records:
            inst = val_s1_instance_data_by_idx[idx]
            basin_cache: Dict[str, dict] = {}
            env.load(inst["depot_xy"], inst["node_xy_demand"], basin_cache)
            emb_a = embed_solutions(embedder, [rec["anchor_solution"]], env, basin_cache)
            emb_p = embed_solutions(embedder, [rec["neighbor_solution"]], env, basin_cache)
            if rec["distant_solutions"]:
                emb_d = embed_solutions(embedder, rec["distant_solutions"], env, basin_cache)
                d_ad = F.pairwise_distance(emb_a.expand_as(emb_d), emb_d, p=2)
                list_d_an.extend(d_ad.cpu().tolist())
            d_ap = F.pairwise_distance(emb_a, emb_p, p=2)[0].item()
            list_d_ap.append(d_ap)
            if first_triplet is None:
                first_triplet = (idx, rec, inst)
        mean_d_ap = sum(list_d_ap) / len(list_d_ap) if list_d_ap else 0.0
        mean_d_an = sum(list_d_an) / len(list_d_an) if list_d_an else 0.0
        ratio = mean_d_an / (mean_d_ap + 1e-8) if list_d_ap and list_d_an else 0.0
        return mean_d_ap, mean_d_an, ratio, list_d_ap, list_d_an, first_triplet

    # Optional: eval & plot once at random initialization (epoch 0)
    if start_epoch1 == 0 and val_s1_records and args.plot_interval > 0:
        embedder.eval()
        with torch.no_grad():
            val_d_ap, val_d_an, val_ratio, val_list_d_ap, val_list_d_an, first_triplet = eval_stage1_on_val()
        embedder.train()
        print(f"[S1] Epoch 0 (init) Val d(A,P)={val_d_ap:.4f} d(A,D)={val_d_an:.4f} d(A,D)/d(A,P)={val_ratio:.4f}")
        if wb_run is not None:
            wandb.log(
                {"val_s1/d_ap": val_d_ap, "val_s1/d_an": val_d_an, "val_s1/d_an_over_d_ap": val_ratio},
                step=global_step,
            )
        if val_list_d_ap and val_list_d_an:
            _log_histogram(val_list_d_ap, val_list_d_an,
                           os.path.join(plot_dir, "distance_hist_s1_epoch0.png"), "S1",
                           wb_run, global_step, "plot_s1/distance_hist", "plot_s1/hist_wasserstein")
        if first_triplet is not None:
            inst_idx, rec, inst = first_triplet
            basin_cache: Dict[str, dict] = {}
            embedder.eval()
            with torch.no_grad():
                env.load(inst["depot_xy"], inst["node_xy_demand"], basin_cache)
                emb_a = embed_solutions(embedder, [rec["anchor_solution"]], env, basin_cache)
                emb_p = embed_solutions(embedder, [rec["neighbor_solution"]], env, basin_cache)
                emb_d = (
                    embed_solutions(embedder, rec["distant_solutions"], env, basin_cache)
                    if rec["distant_solutions"]
                    else None
                )
                if emb_d is not None:
                    emb = torch.cat([emb_a, emb_p, emb_d], dim=0)
                    group_labels = [0, 1] + [2] * emb_d.size(0)
                else:
                    emb = torch.cat([emb_a, emb_p], dim=0)
                    group_labels = [0, 1]
                instance_ids = [inst_idx] * len(group_labels)
                emb_path = os.path.join(plot_dir, f"embedding_2d_s1_epoch0.png")
                plot_embedding_2d(
                    emb,
                    instance_ids=instance_ids,
                    group_labels=group_labels,
                    method="pca",
                    save_path=emb_path,
                )
            embedder.train()
            print(f"[S1] Saved plot (val, init) to {emb_path}")
            if wb_run is not None:
                wandb.log({"plot_s1/embedding_2d": wandb.Image(emb_path)}, step=global_step)

    for epoch in range(start_epoch1, args.epochs1):
        total_loss = 0.0
        n_batches = 0
        first_batch_for_plot = None
        masked_neg_mean_sum = 0.0
        masked_neg_zero_sum = 0.0
        masked_neg_min = float("inf")
        masked_neg_max = 0.0
        masked_neg_stat_batches = 0

        pbar = tqdm(loader, desc=f"[S1] Epoch {epoch+1}/{args.epochs1}", unit="batch")
        for batch_idx, batch in enumerate(pbar, start=1):
            env.load(batch.depot_xy, batch.node_xy_demand, basin_data.basin_info)
            loss = train_one_batch(embedder, optimizer, batch, env, args.device, args.temperature,
                                   grad_clip=args.grad_clip)
            neg_stats = _compute_masked_neg_stats(batch)
            if scheduler is not None:
                scheduler.step()

            total_loss += loss
            n_batches += 1
            if neg_stats is not None:
                masked_neg_mean_sum += neg_stats["mean"]
                masked_neg_zero_sum += neg_stats["zero_ratio"]
                masked_neg_min = min(masked_neg_min, neg_stats["min"])
                masked_neg_max = max(masked_neg_max, neg_stats["max"])
                masked_neg_stat_batches += 1

            if (epoch + 1) % args.plot_interval == 0 and first_batch_for_plot is None:
                first_batch_for_plot = batch

            if wb_run is not None:
                log_dict = {"s1/step_loss": loss, "s1/epoch": epoch + 1}
                if neg_stats is not None and batch_idx % 20 == 0:
                    log_dict.update(
                        {
                            "s1/masked_neg_mean": neg_stats["mean"],
                            "s1/masked_neg_min": neg_stats["min"],
                            "s1/masked_neg_max": neg_stats["max"],
                            "s1/masked_neg_zero_ratio": neg_stats["zero_ratio"],
                        }
                    )
                wandb.log(log_dict, step=global_step)
            if batch_idx % 20 == 0:
                postfix = {"loss": f"{loss:.4f}"}
                if neg_stats is not None:
                    postfix["nneg"] = f"{neg_stats['mean']:.1f}"
                    postfix["zero"] = f"{neg_stats['zero_ratio']:.2f}"
                pbar.set_postfix(postfix)
            global_step += 1

        # Stage-1 validation + plots: only every plot_interval epochs, and using fixed val set
        if (epoch + 1) % args.plot_interval == 0 and val_s1_records:
            embedder.eval()
            with torch.no_grad():
                val_d_ap, val_d_an, val_ratio, val_list_d_ap, val_list_d_an, first_triplet = eval_stage1_on_val()
            embedder.train()
            print(f"[S1] Val (fixed) d(A,P)={val_d_ap:.4f} d(A,D)={val_d_an:.4f} d(A,D)/d(A,P)={val_ratio:.4f}")
            if wb_run is not None:
                wandb.log(
                    {"val_s1/d_ap": val_d_ap, "val_s1/d_an": val_d_an, "val_s1/d_an_over_d_ap": val_ratio},
                    step=global_step,
                )

            if val_list_d_ap and val_list_d_an:
                _log_histogram(val_list_d_ap, val_list_d_an,
                               os.path.join(plot_dir, f"distance_hist_s1_epoch{epoch+1}.png"), "S1",
                               wb_run, global_step, "plot_s1/distance_hist", "plot_s1/hist_wasserstein")

            if val_s1_records:
                _embed_val_triplets_2d(
                    val_s1_records, val_s1_instance_data_by_idx,
                    embedder, env, basin_data.basin_info,
                    sol_keys=["anchor_solution", "neighbor_solution", "distant_solutions"],
                    role_label_map={0: "anchor", 1: "neighbour", 2: "distant"},
                    role_color_map={0: "tab:blue", 1: "tab:orange", 2: "tab:red"},
                    epoch_label=f"epoch{epoch+1}", stage_prefix="s1",
                    plot_dir=plot_dir, wb_run=wb_run, global_step=global_step,
                )

        avg_loss = total_loss / max(n_batches, 1)
        print(f"[S1] Epoch {epoch+1}/{args.epochs1} loss={avg_loss:.6f}")
        if wb_run is not None:
            epoch_log = {"s1/epoch_loss": avg_loss}
            if masked_neg_stat_batches > 0:
                epoch_masked_neg_zero_ratio = masked_neg_zero_sum / masked_neg_stat_batches
                epoch_log.update(
                    {
                        "s1/epoch_masked_neg_mean": masked_neg_mean_sum / masked_neg_stat_batches,
                        "s1/epoch_masked_neg_zero_ratio": epoch_masked_neg_zero_ratio,
                        "s1/epoch_masked_neg_min": masked_neg_min,
                        "s1/epoch_masked_neg_max": masked_neg_max,
                    }
                )
                if epoch_masked_neg_zero_ratio > args.masked_neg_zero_ratio_warn:
                    print(
                        f"[S1][WARN] epoch {epoch+1}: masked_neg_zero_ratio={epoch_masked_neg_zero_ratio:.4f} "
                        f"> threshold={args.masked_neg_zero_ratio_warn:.4f}. "
                        "Many pairs may have no valid negatives."
                    )
            wandb.log(epoch_log, step=global_step)

        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs1:
            ckpt_path = os.path.join(save_dir, f"s1_epoch{epoch+1}.pt")
            save_dict = {
                "embedder_state": embedder.state_dict(),
                "epoch": epoch + 1,
                "stage": 1,
                "global_step": global_step,
                "wandb_run_id": (wb_run.id if wb_run is not None else None),
            }
            torch.save(save_dict, ckpt_path)
            print(f"Saved {ckpt_path}")

    return global_step


def run_stage2(
    args: argparse.Namespace,
    embedder: SolutionEmbedder,
    save_dir: str,
    plot_dir: str,
    wb_run,
    global_step: int,
) -> int:
    """Stage 2: Triplet Margin Loss with perturb data. Returns updated global_step."""
    print("\n" + "=" * 60)
    print("Stage 2: Triplet Margin Loss Training")
    print("=" * 60)

    optimizer = torch.optim.AdamW(embedder.parameters(), lr=args.lr2, weight_decay=args.weight_decay)

    indices = parse_instance_indices(args.instance_indices) if args.instance_indices else [0]
    basin_info_cache: Dict[str, dict] = {}
    instance_data_list = load_instances_pkl(args.instance_pkl, args.device, indices, basin_info_cache)
    instance_data_by_idx = dict(zip(indices, instance_data_list))
    print(f"Loaded {len(instance_data_list)} instances")

    # Load perturb data
    all_triplets: List[Tuple[int, dict]] = []
    for idx in indices:
        perturb_path = os.path.join(args.perturb_root, f"{args.instance_prefix}{idx}", "perturb_data.jsonl")
        triplets = load_perturb_data(perturb_path)
        for t in triplets:
            all_triplets.append((idx, t))
        print(f"  instance {idx}: {len(triplets)} triplets")
    print(f"Total triplets: {len(all_triplets)}")

    env = CVRPEnv(problem_size=args.problem_size, device=args.device)

    # Fixed validation set (same data every epoch so plots/metrics are comparable)
    val_triplets: List[Tuple[int, dict]] = []
    val_instance_data_by_idx: Dict[int, dict] = {}
    if os.path.isfile(args.val_data_1p1n):
        val_triplets = load_val_data_1p1n(args.val_data_1p1n)
        val_indices = sorted(set(t[0] for t in val_triplets))
        val_instance_list = load_instances_pkl(args.instance_pkl, args.device, val_indices, {})
        val_instance_data_by_idx = dict(zip(val_indices, val_instance_list))
        print(f"Loaded fixed val set: {args.val_data_1p1n} -> {len(val_triplets)} triplets, instances {val_indices}")
    else:
        print(f"Val file not found: {args.val_data_1p1n}, skipping validation and fixed plots")

    def eval_on_val():
        """Run embedder on fixed val set; return (mean_d_ap, mean_d_an, mean_ratio, list_d_ap, list_d_an, first_batch)."""
        if not val_triplets:
            return 0.0, 0.0, 0.0, [], [], None
        list_d_ap, list_d_an = [], []
        sum_ratio = 0.0
        n = 0
        first_batch = None
        batch_start = 0
        while batch_start < len(val_triplets):
            batch_items = val_triplets[batch_start : batch_start + args.batch_size2]
            batch_start += args.batch_size2
            emb_a = embed_multi_instance(
                embedder,
                [(idx, t["anchor_solution"]) for idx, t in batch_items],
                val_instance_data_by_idx, env, basin_info_cache,
            )
            emb_p = embed_multi_instance(
                embedder,
                [(idx, t["positive_solution"]) for idx, t in batch_items],
                val_instance_data_by_idx, env, basin_info_cache,
            )
            emb_n = embed_multi_instance(
                embedder,
                [(idx, t["negative_solution"]) for idx, t in batch_items],
                val_instance_data_by_idx, env, basin_info_cache,
            )
            d_ap = F.pairwise_distance(emb_a, emb_p, p=2)
            d_an = F.pairwise_distance(emb_a, emb_n, p=2)
            list_d_ap.extend(d_ap.cpu().tolist())
            list_d_an.extend(d_an.cpu().tolist())
            mean_ap, mean_an = d_ap.mean().item(), d_an.mean().item()
            sum_ratio += mean_an / (mean_ap + 1e-8)
            n += 1
            if first_batch is None:
                inst_idx = batch_items[0][0]
                inst = val_instance_data_by_idx[inst_idx]
                first_batch = TripletBatch(
                    anchor_solutions=[t["anchor_solution"] for _, t in batch_items],
                    positive_solutions=[t["positive_solution"] for _, t in batch_items],
                    negative_solutions=[t["negative_solution"] for _, t in batch_items],
                    instance_idx=inst_idx,
                    depot_xy=inst["depot_xy"],
                    node_xy_demand=inst["node_xy_demand"],
                )
        mean_d_ap = sum(list_d_ap) / len(list_d_ap) if list_d_ap else 0.0
        mean_d_an = sum(list_d_an) / len(list_d_an) if list_d_an else 0.0
        mean_ratio = sum_ratio / n if n else 0.0
        return mean_d_ap, mean_d_an, mean_ratio, list_d_ap, list_d_an, first_batch

    # Optional: eval & plot once at random initialization (epoch 0)
    if int(getattr(args, "resume_epoch2", 0) or 0) == 0 and val_triplets and args.plot_interval > 0:
        embedder.eval()
        with torch.no_grad():
            val_d_ap, val_d_an, val_ratio, val_list_d_ap, val_list_d_an, val_first_batch = eval_on_val()
        embedder.train()
        print(f"[S2] Epoch 0 (init) Val d(A,P)={val_d_ap:.4f} d(A,N)={val_d_an:.4f} d(A,N)/d(A,P)={val_ratio:.4f}")
        if wb_run is not None:
            wandb.log(
                {"val_s2/d_ap": val_d_ap, "val_s2/d_an": val_d_an, "val_s2/d_an_over_d_ap": val_ratio},
                step=global_step,
            )
        if val_list_d_ap and val_list_d_an:
            _log_histogram(val_list_d_ap, val_list_d_an,
                           os.path.join(plot_dir, "distance_hist_s2_epoch0.png"), "S2",
                           wb_run, global_step, "plot_s2/distance_hist", "plot_s2/hist_wasserstein")
        if val_triplets:
            _embed_val_triplets_2d(
                val_triplets, val_instance_data_by_idx,
                embedder, env, basin_info_cache,
                sol_keys=["anchor_solution", "positive_solution", "negative_solution"],
                role_label_map={0: "anchor", 1: "positive", 2: "negative"},
                role_color_map={0: "tab:blue", 1: "tab:orange", 2: "tab:green"},
                epoch_label="epoch0", stage_prefix="s2",
                plot_dir=plot_dir, wb_run=wb_run, global_step=global_step,
            )
            embedder.train()

    n_batches_per_epoch = (len(all_triplets) + args.batch_size2 - 1) // args.batch_size2
    start_epoch2 = int(getattr(args, "resume_epoch2", 0) or 0)
    if start_epoch2 >= args.epochs2:
        print(f"[S2] Resume epoch {start_epoch2} >= target epochs {args.epochs2}, skip Stage 2.")
        return global_step
    total_steps_s2 = max(args.epochs2 - start_epoch2, 0) * n_batches_per_epoch
    scheduler = make_lr_scheduler(optimizer, args.warmup_steps, total_steps_s2)

    for epoch in range(start_epoch2, args.epochs2):
        random.shuffle(all_triplets)
        total_loss, total_d_ap, total_d_an, total_ratio = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        pbar = tqdm(total=n_batches_per_epoch,
                    desc=f"[S2] Epoch {epoch+1}/{args.epochs2}", unit="batch")
        batch_start = 0

        while batch_start < len(all_triplets):
            batch_items = all_triplets[batch_start : batch_start + args.batch_size2]
            batch_start += args.batch_size2

            embedder.train()
            optimizer.zero_grad()

            emb_a = embed_multi_instance(
                embedder,
                [(idx, t["anchor_solution"]) for idx, t in batch_items],
                instance_data_by_idx, env, basin_info_cache,
            )
            emb_p = embed_multi_instance(
                embedder,
                [(idx, t["positive_solution"]) for idx, t in batch_items],
                instance_data_by_idx, env, basin_info_cache,
            )
            emb_n = embed_multi_instance(
                embedder,
                [(idx, t["negative_solution"]) for idx, t in batch_items],
                instance_data_by_idx, env, basin_info_cache,
            )

            d_ap = F.pairwise_distance(emb_a, emb_p, p=2)
            d_an = F.pairwise_distance(emb_a, emb_n, p=2)
            loss = F.relu(d_ap - d_an + args.margin).mean()

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(embedder.parameters(), args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            mean_d_ap = d_ap.mean().item()
            mean_d_an = d_an.mean().item()
            ratio = mean_d_an / (mean_d_ap + 1e-8)

            total_loss += loss.item()
            total_d_ap += mean_d_ap
            total_d_an += mean_d_an
            total_ratio += ratio
            n_batches += 1

            if wb_run is not None:
                wandb.log({"s2/step_loss": loss.item(), "s2/step_d_ap": mean_d_ap, "s2/step_d_an": mean_d_an, "s2/step_d_an_over_d_ap": ratio}, step=global_step)
            if n_batches % 10 == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "d_ap": f"{mean_d_ap:.3f}",
                    "d_an": f"{mean_d_an:.3f}",
                })
            pbar.update(1)
            global_step += 1

        pbar.close()
        avg_loss = total_loss / max(n_batches, 1)
        avg_d_ap = total_d_ap / max(n_batches, 1)
        avg_d_an = total_d_an / max(n_batches, 1)
        avg_ratio = total_ratio / max(n_batches, 1)
        print(f"[S2] Epoch {epoch+1}/{args.epochs2} loss={avg_loss:.6f} d(A,P)={avg_d_ap:.4f} d(A,N)={avg_d_an:.4f} d(A,N)/d(A,P)={avg_ratio:.4f}")

        if wb_run is not None:
            wandb.log({"s2/epoch_loss": avg_loss, "s2/epoch_d_ap": avg_d_ap, "s2/epoch_d_an": avg_d_an, "s2/epoch_d_an_over_d_ap": avg_ratio}, step=global_step)

        # Fixed validation (run every plot_interval epochs)
        if (epoch + 1) % args.plot_interval == 0 and val_triplets:
            embedder.eval()
            with torch.no_grad():
                val_d_ap, val_d_an, val_ratio, val_list_d_ap, val_list_d_an, _ = eval_on_val()
            embedder.train()

            print(f"[S2] Val (fixed) d(A,P)={val_d_ap:.4f} d(A,N)={val_d_an:.4f} d(A,N)/d(A,P)={val_ratio:.4f}")
            if wb_run is not None:
                wandb.log(
                    {"val_s2/d_ap": val_d_ap, "val_s2/d_an": val_d_an, "val_s2/d_an_over_d_ap": val_ratio},
                    step=global_step,
                )

            if val_list_d_ap and val_list_d_an:
                _log_histogram(val_list_d_ap, val_list_d_an,
                               os.path.join(plot_dir, f"distance_hist_s2_epoch{epoch+1}.png"), "S2",
                               wb_run, global_step, "plot_s2/distance_hist", "plot_s2/hist_wasserstein")

            _embed_val_triplets_2d(
                val_triplets, val_instance_data_by_idx,
                embedder, env, basin_info_cache,
                sol_keys=["anchor_solution", "positive_solution", "negative_solution"],
                role_label_map={0: "anchor", 1: "positive", 2: "negative"},
                role_color_map={0: "tab:blue", 1: "tab:orange", 2: "tab:green"},
                epoch_label=f"epoch{epoch+1}", stage_prefix="s2",
                plot_dir=plot_dir, wb_run=wb_run, global_step=global_step,
            )

        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs2:
            ckpt_path = os.path.join(save_dir, f"s2_epoch{epoch+1}.pt")
            save_dict = {
                "embedder_state": embedder.state_dict(),
                "epoch": epoch + 1,
                "stage": 2,
                "global_step": global_step,
                "wandb_run_id": (wb_run.id if wb_run is not None else None),
            }
            torch.save(save_dict, ckpt_path)
            print(f"Saved {ckpt_path}")

    return global_step


class PmaxHead(torch.nn.Module):
    """Predict P_max (basin certainty) from solution embedding."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


ADVANTAGE_THRESHOLDS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.5, 5.0, 8.0, 10.0, 25.0]
ADVANTAGE_OVERFLOW_MIDPOINT = 35.0




class AdvantageOrdinalHead(torch.nn.Module):
    """CORAL-style ordinal regression for gap% = (cost_int - cost_opt) / cost_opt * 100.

    Uses shared latent score + ordered biases to guarantee monotonicity:
      logit_k = z - b_k,  b_k = cumsum(softplus(delta_k))
    This ensures P(gap% > t_k) is non-increasing as t_k increases.
    """

    def __init__(self, embed_dim: int, n_thresholds: int = 13, hidden_dim: int = 64):
        super().__init__()
        self.feature_net = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.bias_deltas = torch.nn.Parameter(torch.zeros(n_thresholds))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.feature_net(x)  # (B, 1)
        biases = torch.cumsum(F.softplus(self.bias_deltas), dim=0)  # (K,) non-decreasing
        return z - biases.unsqueeze(0)  # (B, K)


def _compute_bin_midpoints() -> List[float]:
    """Derive bin midpoints from ADVANTAGE_THRESHOLDS automatically.

    Bins: [0, t_0], (t_0, t_1], ..., (t_{K-1}, inf).
    Midpoints: t/2 for first bin, (t_k + t_{k+1})/2, OVERFLOW for last.
    """
    t = ADVANTAGE_THRESHOLDS
    mids = [t[0] / 2.0]
    for i in range(len(t) - 1):
        mids.append((t[i] + t[i + 1]) / 2.0)
    mids.append(ADVANTAGE_OVERFLOW_MIDPOINT)
    return mids


_ADVANTAGE_BIN_MIDPOINTS = _compute_bin_midpoints()


def ordinal_probs_to_gap_pct(probs: torch.Tensor) -> torch.Tensor:
    """Convert cumulative ordinal probabilities to predicted gap%.

    Args:
        probs: (B, K) where probs[:, k] ≈ P(gap% > THRESHOLDS[k]).
    Returns:
        (B,) predicted gap% via bin midpoint expectation.
    """
    midpoints = torch.tensor(_ADVANTAGE_BIN_MIDPOINTS, device=probs.device, dtype=probs.dtype)
    ones = torch.ones(probs.shape[0], 1, device=probs.device, dtype=probs.dtype)
    zeros = torch.zeros(probs.shape[0], 1, device=probs.device, dtype=probs.dtype)
    extended = torch.cat([ones, probs, zeros], dim=1)  # (B, K+2)
    bin_probs = (extended[:, :-1] - extended[:, 1:]).clamp(min=0)  # (B, K+1)
    return (bin_probs * midpoints.unsqueeze(0)).sum(dim=1)


def run_stage3(
    args: argparse.Namespace,
    embedder: SolutionEmbedder,
    save_dir: str,
    plot_dir: str,
    wb_run,
    global_step: int,
) -> int:
    """Stage 3: InfoNCE contrastive + multi-task training. Returns updated global_step.

    Data: training_data.jsonl (100-run basin distribution per intermediate solution).

    Main contrastive loss: In-batch masked InfoNCE (cosine sim, masked by basin reachability).
    Optional auxiliary heads (independently toggled):
      - P_max regression:        --use_regression          (predict basin certainty)
      - Advantage ordinal reg:   --quality_reg_weight > 0  (ordinal BCE on gap% bins)
    """
    print("\n" + "=" * 60)
    print("Stage 3: InfoNCE Contrastive + Multi-Task Training")
    print("=" * 60)

    indices = parse_instance_indices(args.instance_indices) if args.instance_indices else [0]
    basin_info_cache: Dict[str, dict] = {}
    instance_data_list = load_instances_pkl(args.instance_pkl, args.device, indices, basin_info_cache)
    instance_data_by_idx = dict(zip(indices, instance_data_list))
    print(f"Loaded {len(instance_data_list)} instances")

    env = CVRPEnv(problem_size=args.problem_size, device=args.device)

    # ── Data loading (training_data.jsonl only, with disk cache) ──
    chaotic_samples: List[Tuple[int, dict]] = []
    td_paths = [
        os.path.join(args.training_data_root, f"{args.instance_prefix}{idx}", "training_data.jsonl")
        for idx in indices
    ]
    cache_key_dict = {
        "td_paths": sorted(td_paths),
        "certainty_threshold": args.certainty_threshold,
        "seed": args.seed,
        "max_runs": args.max_traj_runs,
    }
    cache_hash = hashlib.md5(repr(sorted(cache_key_dict.items())).encode()).hexdigest()[:12]
    cache_dir = getattr(args, "s3_cache_dir", "s3_data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"s3_pairs_{cache_hash}.pkl")

    if os.path.isfile(cache_path):
        print(f"[S3] Loading cached data from {cache_path} ...")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("key") == cache_key_dict:
            contrastive_pairs = cached["contrastive"]
            chaotic_samples = cached["chaotic"]
            print(f"[S3] Cache hit: {len(contrastive_pairs)} contrastive, {len(chaotic_samples)} chaotic")
        else:
            print("[S3] Cache key mismatch, regenerating ...")
            cached = None
    else:
        cached = None

    if cached is None:
        contrastive_pairs, chaotic_samples = load_training_data_pairs(
            td_paths,
            certainty_threshold=args.certainty_threshold,
            seed=args.seed,
            max_runs=args.max_traj_runs,
        )
        print(f"Contrastive pairs (P_max >= {args.certainty_threshold}): {len(contrastive_pairs)}")
        print(f"Chaotic regression samples (P_max < {args.certainty_threshold}): {len(chaotic_samples)}")
        print(f"[S3] Saving cache to {cache_path} ...")
        with open(cache_path, "wb") as f:
            pickle.dump({
                "key": cache_key_dict,
                "contrastive": contrastive_pairs,
                "chaotic": chaotic_samples,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[S3] Cache saved ({os.path.getsize(cache_path) / 1024 / 1024:.1f} MB)")

    # ── Fixed S3 evaluation set (built with same pipeline as training data) ──
    s3_val_pairs: List[Tuple[int, dict]] = []
    s3_val_chaotic: List[Tuple[int, dict]] = []
    val_instance_data_by_idx: Dict[int, dict] = {}
    val_indices = parse_instance_indices(args.s3_val_instance_indices) if args.s3_val_instance_indices else []
    if val_indices:
        val_instance_list = load_instances_pkl(args.instance_pkl, args.device, val_indices, basin_info_cache)
        val_instance_data_by_idx = dict(zip(val_indices, val_instance_list))
        val_td_paths = [
            os.path.join(args.training_data_root, f"{args.instance_prefix}{idx}", "training_data.jsonl")
            for idx in val_indices
        ]
        val_key_dict = {
            "val_td_paths": sorted(val_td_paths),
            "certainty_threshold": args.certainty_threshold,
            "seed": args.seed,
            "max_runs": args.s3_val_max_traj_runs,
        }
        val_hash = hashlib.md5(repr(sorted(val_key_dict.items())).encode()).hexdigest()[:12]
        val_cache_path = os.path.join(cache_dir, f"s3_val_pairs_{val_hash}.pkl")
        val_cached = None
        if os.path.isfile(val_cache_path):
            print(f"[S3] Loading fixed eval cache from {val_cache_path} ...")
            with open(val_cache_path, "rb") as f:
                val_cached = pickle.load(f)
            if val_cached.get("key") == val_key_dict:
                s3_val_pairs = val_cached["contrastive"]
                s3_val_chaotic = val_cached["chaotic"]
                print(f"[S3] Fixed eval cache hit: {len(s3_val_pairs)} contrastive, {len(s3_val_chaotic)} chaotic")
            else:
                print("[S3] Fixed eval cache key mismatch, regenerating ...")
                val_cached = None
        if val_cached is None:
            s3_val_pairs, s3_val_chaotic = load_training_data_pairs(
                val_td_paths,
                certainty_threshold=args.certainty_threshold,
                seed=args.seed,
                max_runs=args.s3_val_max_traj_runs,
            )
            with open(val_cache_path, "wb") as f:
                pickle.dump({
                    "key": val_key_dict,
                    "contrastive": s3_val_pairs,
                    "chaotic": s3_val_chaotic,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(
                f"[S3] Fixed eval set built from instances {val_indices}: "
                f"{len(s3_val_pairs)} contrastive, {len(s3_val_chaotic)} chaotic"
            )
        # Keep S3 fixed eval lightweight: persist a sampled subset once, then reuse.
        val_sample_size = 100
        sampled_key_dict = {
            "base_key": val_key_dict,
            "sample_size": val_sample_size,
            "sample_seed": args.seed,
        }
        sampled_hash = hashlib.md5(repr(sorted(sampled_key_dict.items())).encode()).hexdigest()[:12]
        sampled_cache_path = os.path.join(cache_dir, f"s3_val_sampled_{sampled_hash}.pkl")
        sampled_cached = None
        if os.path.isfile(sampled_cache_path):
            print(f"[S3] Loading fixed eval sampled cache from {sampled_cache_path} ...")
            with open(sampled_cache_path, "rb") as f:
                sampled_cached = pickle.load(f)
            if sampled_cached.get("key") == sampled_key_dict:
                s3_val_pairs = sampled_cached["contrastive"]
                s3_val_chaotic = sampled_cached["chaotic"]
                print(
                    f"[S3] Fixed eval sampled cache hit: "
                    f"{len(s3_val_pairs)} contrastive, {len(s3_val_chaotic)} chaotic"
                )
            else:
                print("[S3] Fixed eval sampled cache key mismatch, regenerating ...")
                sampled_cached = None
        if sampled_cached is None:
            val_rng = random.Random(args.seed)
            if len(s3_val_pairs) > val_sample_size:
                s3_val_pairs = val_rng.sample(s3_val_pairs, val_sample_size)
            if len(s3_val_chaotic) > val_sample_size:
                s3_val_chaotic = val_rng.sample(s3_val_chaotic, val_sample_size)
            with open(sampled_cache_path, "wb") as f:
                pickle.dump({
                    "key": sampled_key_dict,
                    "contrastive": s3_val_pairs,
                    "chaotic": s3_val_chaotic,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(
                f"[S3] Fixed eval sampled and saved: "
                f"{len(s3_val_pairs)} contrastive + {len(s3_val_chaotic)} chaotic "
                f"(max {val_sample_size} each)"
            )

    if not contrastive_pairs:
        print("[S3] No contrastive pairs found, skipping Stage 3.")
        return global_step

    # ── Model & optimizer ──
    pmax_head: Optional[PmaxHead] = None
    advantage_head: Optional[AdvantageOrdinalHead] = None
    params = list(embedder.parameters())

    if args.use_regression:
        pmax_head = PmaxHead(args.embedding_dim).to(args.device)
        params += list(pmax_head.parameters())
        print(f"[S3] P_max regression head enabled (λ={args.regression_weight})")

    if args.quality_reg_weight > 0:
        n_thresh = len(ADVANTAGE_THRESHOLDS)
        advantage_head = AdvantageOrdinalHead(args.embedding_dim, n_thresholds=n_thresh).to(args.device)
        params += list(advantage_head.parameters())
        print(f"[S3] Advantage ordinal head enabled (λ={args.quality_reg_weight}, {n_thresh} thresholds)")

    print(f"[S3] InfoNCE contrastive loss (λ={args.infonce_weight}, T={args.infonce_temperature})")

    optimizer = torch.optim.AdamW(params, lr=args.lr3, weight_decay=args.weight_decay)

    # Group by instance: same instance per batch (consistent with S1 GroupByInstanceBatchSampler)
    pairs_by_inst: Dict[int, List[dict]] = {}
    for idx, rec in contrastive_pairs:
        pairs_by_inst.setdefault(idx, []).append(rec)

    chaotic_by_inst: Dict[int, List[dict]] = {}
    for idx, rec in chaotic_samples:
        chaotic_by_inst.setdefault(idx, []).append(rec)

    val_pairs_by_inst: Dict[int, List[dict]] = {}
    for idx, rec in s3_val_pairs:
        val_pairs_by_inst.setdefault(idx, []).append(rec)
    val_chaotic_by_inst: Dict[int, List[dict]] = {}
    for idx, rec in s3_val_chaotic:
        val_chaotic_by_inst.setdefault(idx, []).append(rec)

    def eval_s3_fixed() -> Optional[Dict[str, float]]:
        if not val_pairs_by_inst:
            return None
        total_infonce, total_pos_sim, total_neg_sim, total_n_valid_neg = 0.0, 0.0, 0.0, 0.0
        n_batches = 0
        total_pmax_sse, total_pmax_n = 0.0, 0
        total_adv_loss, total_gap_gt, total_gap_pred, total_gap_mae = 0.0, 0.0, 0.0, 0.0
        n_adv_batches = 0

        embedder.eval()
        if pmax_head is not None:
            pmax_head.eval()
        if advantage_head is not None:
            advantage_head.eval()
        with torch.no_grad():
            for idx, recs in val_pairs_by_inst.items():
                inst = val_instance_data_by_idx[idx]
                env.load(inst["depot_xy"], inst["node_xy_demand"], basin_info_cache)
                for i in range(0, len(recs), args.batch_size3):
                    batch_records = recs[i : i + args.batch_size3]
                    emb_a = embed_solutions(embedder, [t["anchor_solution"] for t in batch_records], env, basin_info_cache)
                    emb_p = embed_solutions(embedder, [t["positive_solution"] for t in batch_records], env, basin_info_cache)
                    bsz = emb_a.size(0)
                    pos_hashes = [t["positive_basin_hash"] for t in batch_records]
                    reachable_sets = [t["reachable_basin_hashes"] for t in batch_records]
                    all_hashes: set = set(pos_hashes)
                    for rs in reachable_sets:
                        all_hashes.update(rs)
                    hash_to_id = {h: j for j, h in enumerate(all_hashes)}
                    pos_hash_ids = torch.tensor([hash_to_id[ph] for ph in pos_hashes], device=emb_a.device)
                    reachable_mat = torch.zeros(bsz, len(all_hashes), dtype=torch.bool, device=emb_a.device)
                    for bi, rs in enumerate(reachable_sets):
                        for rh in rs:
                            reachable_mat[bi, hash_to_id[rh]] = True
                    valid_neg = ~reachable_mat[:, pos_hash_ids]
                    valid_neg.fill_diagonal_(False)
                    sim_raw = torch.mm(emb_a, emb_p.t())
                    sim = sim_raw / args.infonce_temperature
                    logit_mask = valid_neg.clone()
                    logit_mask.fill_diagonal_(True)
                    sim = sim.masked_fill(~logit_mask, -1e9)
                    labels = torch.arange(bsz, device=emb_a.device)
                    infonce_val = F.cross_entropy(sim, labels).item()
                    n_neg_total = valid_neg.float().sum()
                    neg_sims = sim_raw.masked_fill(~valid_neg, 0.0)
                    batch_neg_sim = (neg_sims.sum() / n_neg_total).item() if n_neg_total > 0 else 0.0
                    total_infonce += infonce_val
                    total_pos_sim += sim_raw.diag().mean().item()
                    total_neg_sim += batch_neg_sim
                    total_n_valid_neg += valid_neg.float().sum(dim=1).mean().item()
                    n_batches += 1

                    if pmax_head is not None and args.use_regression and "p_max" in batch_records[0]:
                        pmax_t = torch.tensor([t["p_max"] for t in batch_records], dtype=torch.float32, device=emb_a.device)
                        pmax_pred = pmax_head(emb_a)
                        total_pmax_sse += F.mse_loss(pmax_pred, pmax_t, reduction="sum").item()
                        total_pmax_n += pmax_t.numel()

                    if advantage_head is not None and args.quality_reg_weight > 0:
                        c_int = torch.tensor([t["anchor_cost"] for t in batch_records], dtype=torch.float32, device=emb_a.device)
                        c_opt = torch.tensor([t["positive_cost"] for t in batch_records], dtype=torch.float32, device=emb_a.device)
                        gap_pct = ((c_int - c_opt) / c_opt.clamp(min=1e-8) * 100.0).clamp(min=0.0)
                        thresholds_t = torch.tensor(ADVANTAGE_THRESHOLDS, dtype=torch.float32, device=emb_a.device).unsqueeze(0)
                        ordinal_targets = (gap_pct.unsqueeze(1) > thresholds_t).float()
                        logits = advantage_head(emb_a)
                        total_adv_loss += F.binary_cross_entropy_with_logits(logits, ordinal_targets).item()
                        probs = torch.sigmoid(logits)
                        pred_gap = ordinal_probs_to_gap_pct(probs)
                        total_gap_gt += gap_pct.mean().item()
                        total_gap_pred += pred_gap.mean().item()
                        total_gap_mae += (pred_gap - gap_pct).abs().mean().item()
                        n_adv_batches += 1

            if pmax_head is not None and args.use_regression and val_chaotic_by_inst:
                for idx, recs in val_chaotic_by_inst.items():
                    inst = val_instance_data_by_idx[idx]
                    env.load(inst["depot_xy"], inst["node_xy_demand"], basin_info_cache)
                    for i in range(0, len(recs), args.batch_size3):
                        batch_records = recs[i : i + args.batch_size3]
                        emb_a = embed_solutions(embedder, [t["anchor_solution"] for t in batch_records], env, basin_info_cache)
                        pmax_t = torch.tensor([t["p_max"] for t in batch_records], dtype=torch.float32, device=emb_a.device)
                        pmax_pred = pmax_head(emb_a)
                        total_pmax_sse += F.mse_loss(pmax_pred, pmax_t, reduction="sum").item()
                        total_pmax_n += pmax_t.numel()
        embedder.train()
        if pmax_head is not None:
            pmax_head.train()
        if advantage_head is not None:
            advantage_head.train()

        n = max(n_batches, 1)
        result: Dict[str, float] = {
            "val_s3/infonce_loss": total_infonce / n,
            "val_s3/pos_sim": total_pos_sim / n,
            "val_s3/neg_sim": total_neg_sim / n,
            "val_s3/n_valid_neg": total_n_valid_neg / n,
        }
        if total_pmax_n > 0:
            result["val_s3/pmax_mse"] = total_pmax_sse / total_pmax_n
        if n_adv_batches > 0:
            result["val_s3/adv_bce"] = total_adv_loss / n_adv_batches
            result["val_s3/gap_gt_pct"] = total_gap_gt / n_adv_batches
            result["val_s3/gap_pred_pct"] = total_gap_pred / n_adv_batches
            result["val_s3/gap_mae_pct"] = total_gap_mae / n_adv_batches
        return result

    if int(getattr(args, "resume_epoch3", 0) or 0) == 0 and args.plot_interval > 0:
        embedder.eval()
        with torch.no_grad():
            hist_pairs = val_pairs_by_inst if val_pairs_by_inst else pairs_by_inst
            hist_instances = val_instance_data_by_idx if val_pairs_by_inst else instance_data_by_idx
            list_pos_init, list_neg_init = _collect_s3_sim_hist(
                hist_pairs, hist_instances, embedder, env, basin_info_cache, args.batch_size3)
        embedder.train()
        if list_pos_init and list_neg_init:
            _log_histogram(list_pos_init, list_neg_init,
                           os.path.join(plot_dir, "sim_hist_s3_epoch0.png"), "S3",
                           wb_run, global_step, "plot_s3/sim_hist_epoch0", "plot_s3/hist_wasserstein")
        val_init_metrics = eval_s3_fixed()
        if val_init_metrics is not None:
            print(
                "[S3] Fixed eval epoch0 "
                f"infonce={val_init_metrics['val_s3/infonce_loss']:.6f} "
                f"pos_sim={val_init_metrics['val_s3/pos_sim']:.4f} "
                f"neg_sim={val_init_metrics['val_s3/neg_sim']:.4f}"
            )
            if wb_run is not None:
                wandb.log(val_init_metrics, step=global_step)

    n_batches_per_epoch = sum(
        (len(recs) + args.batch_size3 - 1) // args.batch_size3
        for recs in pairs_by_inst.values()
    )
    start_epoch3 = int(getattr(args, "resume_epoch3", 0) or 0)
    if start_epoch3 >= args.epochs3:
        print(f"[S3] Resume epoch {start_epoch3} >= target epochs {args.epochs3}, skip Stage 3.")
        return global_step
    total_steps_s3 = max(args.epochs3 - start_epoch3, 0) * n_batches_per_epoch
    scheduler = make_lr_scheduler(optimizer, args.warmup_steps, total_steps_s3)

    # ── Training loop ──
    for epoch in range(start_epoch3, args.epochs3):
        epoch_batches: List[Tuple[int, List[dict]]] = []
        for idx, recs in pairs_by_inst.items():
            random.shuffle(recs)
            for i in range(0, len(recs), args.batch_size3):
                epoch_batches.append((idx, recs[i:i + args.batch_size3]))
        random.shuffle(epoch_batches)

        total_infonce_loss, total_pmax_reg, total_adv_loss = 0.0, 0.0, 0.0
        total_pos_sim, total_neg_sim = 0.0, 0.0
        total_n_valid_neg = 0.0
        total_anchor_cost, total_optima_cost = 0.0, 0.0
        total_adv_gt, total_adv_pred, total_adv_mae = 0.0, 0.0, 0.0
        total_n_over, total_n_under = 0, 0
        total_over_amount, total_under_amount = 0.0, 0.0
        total_n_adv_samples = 0
        total_pmax_gt, total_pmax_pred, total_pmax_mae = 0.0, 0.0, 0.0
        n_batches = 0
        n_adv_batches, n_pmax_batches = 0, 0

        chaotic_iters: Dict[int, int] = {}
        for idx in chaotic_by_inst:
            random.shuffle(chaotic_by_inst[idx])
            chaotic_iters[idx] = 0

        pbar = tqdm(total=len(epoch_batches), desc=f"[S3] Epoch {epoch+1}/{args.epochs3}", unit="batch")

        for inst_idx, batch_records in epoch_batches:
            inst = instance_data_by_idx[inst_idx]

            embedder.train()
            if pmax_head is not None:
                pmax_head.train()
            if advantage_head is not None:
                advantage_head.train()
            optimizer.zero_grad()

            env.load(inst["depot_xy"], inst["node_xy_demand"], basin_info_cache)

            # ── Embed anchor / positive (single instance) ──
            emb_a = embed_solutions(embedder, [t["anchor_solution"] for t in batch_records], env, basin_info_cache)
            emb_p = embed_solutions(embedder, [t["positive_solution"] for t in batch_records], env, basin_info_cache)

            # ── InfoNCE with in-batch masked negatives (vectorized) ──
            B = emb_a.size(0)
            pos_hashes = [t["positive_basin_hash"] for t in batch_records]
            reachable_sets = [t["reachable_basin_hashes"] for t in batch_records]

            all_hashes: set = set(pos_hashes)
            for rs in reachable_sets:
                all_hashes.update(rs)
            hash_to_id = {h: i for i, h in enumerate(all_hashes)}

            pos_hash_ids = torch.tensor([hash_to_id[ph] for ph in pos_hashes], device=emb_a.device)
            reachable_mat = torch.zeros(B, len(all_hashes), dtype=torch.bool, device=emb_a.device)
            for i, rs in enumerate(reachable_sets):
                for rh in rs:
                    reachable_mat[i, hash_to_id[rh]] = True
            valid_neg = ~reachable_mat[:, pos_hash_ids]
            valid_neg.fill_diagonal_(False)

            sim_raw = torch.mm(emb_a, emb_p.t())
            sim = sim_raw / args.infonce_temperature

            logit_mask = valid_neg.clone()
            logit_mask.fill_diagonal_(True)
            sim = sim.masked_fill(~logit_mask, -1e9)

            labels = torch.arange(B, device=emb_a.device)
            infonce_loss = F.cross_entropy(sim, labels)
            loss = args.infonce_weight * infonce_loss
            infonce_val = infonce_loss.item()

            with torch.no_grad():
                batch_n_valid_neg = valid_neg.float().sum(dim=1).mean().item()
                batch_pos_sim = sim_raw.diag().mean().item()
                neg_sims = sim_raw.masked_fill(~valid_neg, 0.0)
                n_neg_total = valid_neg.float().sum()
                batch_neg_sim = (neg_sims.sum() / n_neg_total).item() if n_neg_total > 0 else 0.0
            total_n_valid_neg += batch_n_valid_neg

            # ── P_max regression (certainty) ──
            pmax_reg_val = 0.0
            batch_pmax_gt, batch_pmax_pred = 0.0, 0.0
            batch_pmax_mae = 0.0
            if args.use_regression and pmax_head is not None and "p_max" in batch_records[0]:
                p_max_targets = torch.tensor(
                    [t["p_max"] for t in batch_records],
                    dtype=torch.float32, device=emb_a.device,
                )
                emb_for_pmax = emb_a.detach() if not args.regression_grad_encoder else emb_a
                p_max_pred_con = pmax_head(emb_for_pmax)
                pmax_reg_con = F.mse_loss(p_max_pred_con, p_max_targets)

                with torch.no_grad():
                    batch_pmax_gt = p_max_targets.mean().item()
                    batch_pmax_pred = p_max_pred_con.mean().item()
                    batch_pmax_mae = (p_max_pred_con - p_max_targets).abs().mean().item()

                pmax_reg_chaotic = torch.tensor(0.0, device=emb_a.device)
                if inst_idx in chaotic_by_inst and chaotic_by_inst[inst_idx]:
                    c_pool = chaotic_by_inst[inst_idx]
                    c_start = chaotic_iters.get(inst_idx, 0)
                    c_batch = c_pool[c_start : c_start + args.batch_size3]
                    if not c_batch:
                        chaotic_iters[inst_idx] = 0
                        c_batch = c_pool[: args.batch_size3]
                    chaotic_iters[inst_idx] = (c_start + len(c_batch)) % max(len(c_pool), 1)

                    if args.regression_grad_encoder:
                        emb_chaotic = embed_solutions(
                            embedder, [r["anchor_solution"] for r in c_batch], env, basin_info_cache,
                        )
                    else:
                        with torch.no_grad():
                            emb_chaotic = embed_solutions(
                                embedder, [r["anchor_solution"] for r in c_batch], env, basin_info_cache,
                            )
                    p_max_chaotic_t = torch.tensor(
                        [r["p_max"] for r in c_batch],
                        dtype=torch.float32, device=emb_chaotic.device,
                    )
                    emb_ch = emb_chaotic if args.regression_grad_encoder else emb_chaotic.detach()
                    pmax_reg_chaotic = F.mse_loss(pmax_head(emb_ch), p_max_chaotic_t)

                pmax_reg_total = pmax_reg_con + pmax_reg_chaotic
                loss = loss + args.regression_weight * pmax_reg_total
                pmax_reg_val = pmax_reg_total.item()
                n_pmax_batches += 1
                total_pmax_gt += batch_pmax_gt
                total_pmax_pred += batch_pmax_pred
                total_pmax_mae += batch_pmax_mae

            # ── Advantage ordinal regression: gap% = (cost_int - cost_opt) / cost_opt * 100 ──
            adv_val = 0.0
            batch_gap_gt, batch_gap_pred, batch_gap_mae = 0.0, 0.0, 0.0
            batch_anchor_cost, batch_optima_cost = 0.0, 0.0
            batch_over_ratio, batch_under_ratio = 0.0, 0.0
            batch_over_mean_pct, batch_under_mean_pct = 0.0, 0.0
            if advantage_head is not None and args.quality_reg_weight > 0:
                c_int = torch.tensor(
                    [t["anchor_cost"] for t in batch_records],
                    dtype=torch.float32, device=emb_a.device,
                )
                c_opt = torch.tensor(
                    [t["positive_cost"] for t in batch_records],
                    dtype=torch.float32, device=emb_a.device,
                )
                gap_pct = ((c_int - c_opt) / c_opt.clamp(min=1e-8) * 100.0).clamp(min=0.0)

                thresholds_t = torch.tensor(
                    ADVANTAGE_THRESHOLDS, dtype=torch.float32, device=emb_a.device,
                ).unsqueeze(0)
                ordinal_targets = (gap_pct.unsqueeze(1) > thresholds_t).float()

                emb_for_adv = emb_a.detach() if not args.advantage_grad_encoder else emb_a
                logits = advantage_head(emb_for_adv)
                adv_loss = F.binary_cross_entropy_with_logits(logits, ordinal_targets)
                loss = loss + args.quality_reg_weight * adv_loss
                adv_val = adv_loss.item()

                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    pred_gap = ordinal_probs_to_gap_pct(probs)
                    batch_anchor_cost = c_int.mean().item()
                    batch_optima_cost = c_opt.mean().item()
                    batch_gap_gt = gap_pct.mean().item()
                    batch_gap_pred = pred_gap.mean().item()
                    batch_gap_mae = (pred_gap - gap_pct).abs().mean().item()

                    over_mask = pred_gap > gap_pct
                    under_mask = pred_gap < gap_pct
                    n_b = pred_gap.numel()
                    n_over = over_mask.sum().item()
                    n_under = under_mask.sum().item()
                    batch_over_ratio = n_over / n_b if n_b else 0.0
                    batch_under_ratio = n_under / n_b if n_b else 0.0
                    batch_over_mean_pct = (pred_gap[over_mask] - gap_pct[over_mask]).mean().item() if n_over > 0 else 0.0
                    batch_under_mean_pct = (gap_pct[under_mask] - pred_gap[under_mask]).mean().item() if n_under > 0 else 0.0

                    total_n_adv_samples += n_b
                    total_n_over += n_over
                    total_n_under += n_under
                    total_over_amount += (pred_gap[over_mask] - gap_pct[over_mask]).sum().item()
                    total_under_amount += (gap_pct[under_mask] - pred_gap[under_mask]).sum().item()
                n_adv_batches += 1
                total_anchor_cost += batch_anchor_cost
                total_optima_cost += batch_optima_cost
                total_adv_gt += batch_gap_gt
                total_adv_pred += batch_gap_pred
                total_adv_mae += batch_gap_mae

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_infonce_loss += infonce_val
            total_pos_sim += batch_pos_sim
            total_neg_sim += batch_neg_sim
            total_pmax_reg += pmax_reg_val
            total_adv_loss += adv_val
            n_batches += 1

            if wb_run is not None:
                log_dict = {
                    "s3/step_infonce_loss": infonce_val,
                    "s3/step_pos_sim": batch_pos_sim,
                    "s3/step_neg_sim": batch_neg_sim,
                    "s3/step_n_valid_neg": batch_n_valid_neg,
                }
                if args.use_regression:
                    log_dict["s3/step_pmax_loss"] = pmax_reg_val
                    log_dict["s3/step_pmax_gt"] = batch_pmax_gt
                    log_dict["s3/step_pmax_pred"] = batch_pmax_pred
                    log_dict["s3/step_pmax_mae"] = batch_pmax_mae
                if args.quality_reg_weight > 0:
                    log_dict["s3/step_adv_loss"] = adv_val
                    log_dict["s3/step_anchor_cost"] = batch_anchor_cost
                    log_dict["s3/step_optima_cost"] = batch_optima_cost
                    log_dict["s3/step_gap_gt_pct"] = batch_gap_gt
                    log_dict["s3/step_gap_pred_pct"] = batch_gap_pred
                    log_dict["s3/step_gap_mae_pct"] = batch_gap_mae
                    log_dict["s3/step_over_ratio"] = batch_over_ratio
                    log_dict["s3/step_under_ratio"] = batch_under_ratio
                    log_dict["s3/step_over_mean_pct"] = batch_over_mean_pct
                    log_dict["s3/step_under_mean_pct"] = batch_under_mean_pct
                wandb.log(log_dict, step=global_step)
            if n_batches % 10 == 0:
                postfix = {
                    "infonce": f"{infonce_val:.4f}",
                    "pos": f"{batch_pos_sim:.3f}",
                    "neg": f"{batch_neg_sim:.3f}",
                    "nneg": f"{batch_n_valid_neg:.1f}",
                }
                if args.use_regression and pmax_head is not None:
                    postfix["pmax"] = f"{pmax_reg_val:.4f}"
                if args.quality_reg_weight > 0 and advantage_head is not None:
                    postfix["adv"] = f"{adv_val:.4f}"
                pbar.set_postfix(postfix)
            pbar.update(1)
            global_step += 1

        pbar.close()
        n = max(n_batches, 1)
        parts = [
            f"[S3] Epoch {epoch+1}/{args.epochs3}",
            f"infonce={total_infonce_loss/n:.6f}",
            f"pos_sim={total_pos_sim/n:.4f} neg_sim={total_neg_sim/n:.4f} n_neg={total_n_valid_neg/n:.1f}",
        ]
        if args.use_regression and n_pmax_batches > 0:
            np_ = n_pmax_batches
            parts.append(
                f"pmax_loss={total_pmax_reg/n:.6f} "
                f"pmax_gt={total_pmax_gt/np_:.3f} pred={total_pmax_pred/np_:.3f} mae={total_pmax_mae/np_:.4f}"
            )
        if args.quality_reg_weight > 0 and n_adv_batches > 0:
            na = n_adv_batches
            n_adv = max(total_n_adv_samples, 1)
            over_ratio = total_n_over / n_adv
            under_ratio = total_n_under / n_adv
            over_mean_pct = total_over_amount / max(total_n_over, 1)
            under_mean_pct = total_under_amount / max(total_n_under, 1)
            parts.append(
                f"adv_bce={total_adv_loss/n:.6f} "
                f"gap_gt={total_adv_gt/na:.3f}% pred={total_adv_pred/na:.3f}% mae={total_adv_mae/na:.3f}% "
                f"over_ratio={over_ratio:.3f} over_mean={over_mean_pct:.3f}% "
                f"under_ratio={under_ratio:.3f} under_mean={under_mean_pct:.3f}% "
                f"cost_int={total_anchor_cost/na:.1f} cost_opt={total_optima_cost/na:.1f}"
            )
        print(" ".join(parts))

        if wb_run is not None:
            log_dict = {
                "s3/epoch_infonce_loss": total_infonce_loss / n,
                "s3/epoch_pos_sim": total_pos_sim / n,
                "s3/epoch_neg_sim": total_neg_sim / n,
                "s3/epoch_n_valid_neg": total_n_valid_neg / n,
            }
            if args.use_regression and n_pmax_batches > 0:
                np_ = n_pmax_batches
                log_dict.update({
                    "s3/epoch_pmax_loss": total_pmax_reg / n,
                    "s3/epoch_pmax_gt": total_pmax_gt / np_,
                    "s3/epoch_pmax_pred": total_pmax_pred / np_,
                    "s3/epoch_pmax_mae": total_pmax_mae / np_,
                })
            if args.quality_reg_weight > 0 and n_adv_batches > 0:
                na = n_adv_batches
                n_adv = max(total_n_adv_samples, 1)
                log_dict.update({
                    "s3/epoch_adv_bce": total_adv_loss / n,
                    "s3/epoch_anchor_cost": total_anchor_cost / na,
                    "s3/epoch_optima_cost": total_optima_cost / na,
                    "s3/epoch_gap_gt_pct": total_adv_gt / na,
                    "s3/epoch_gap_pred_pct": total_adv_pred / na,
                    "s3/epoch_gap_mae_pct": total_adv_mae / na,
                    "s3/epoch_over_ratio": total_n_over / n_adv,
                    "s3/epoch_under_ratio": total_n_under / n_adv,
                    "s3/epoch_over_mean_pct": total_over_amount / max(total_n_over, 1),
                    "s3/epoch_under_mean_pct": total_under_amount / max(total_n_under, 1),
                })
            wandb.log(log_dict, step=global_step)

        if (epoch + 1) % args.plot_interval == 0:
            embedder.eval()
            with torch.no_grad():
                hist_pairs = val_pairs_by_inst if val_pairs_by_inst else pairs_by_inst
                hist_instances = val_instance_data_by_idx if val_pairs_by_inst else instance_data_by_idx
                list_pos_sim, list_neg_sim = _collect_s3_sim_hist(
                    hist_pairs, hist_instances, embedder, env, basin_info_cache, args.batch_size3)
            embedder.train()
            if list_pos_sim and list_neg_sim:
                _log_histogram(list_pos_sim, list_neg_sim,
                               os.path.join(plot_dir, f"sim_hist_s3_epoch{epoch+1}.png"), "S3",
                               wb_run, global_step, "plot_s3/sim_hist", "plot_s3/hist_wasserstein")
            val_metrics = eval_s3_fixed()
            if val_metrics is not None:
                print(
                    f"[S3] Fixed eval epoch{epoch+1} "
                    f"infonce={val_metrics['val_s3/infonce_loss']:.6f} "
                    f"pos_sim={val_metrics['val_s3/pos_sim']:.4f} "
                    f"neg_sim={val_metrics['val_s3/neg_sim']:.4f}"
                )
                if wb_run is not None:
                    wandb.log(val_metrics, step=global_step)

        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs3:
            ckpt_path = os.path.join(save_dir, f"s3_epoch{epoch+1}.pt")
            save_dict = {
                "embedder_state": embedder.state_dict(),
                "epoch": epoch + 1,
                "stage": 3,
                "global_step": global_step,
                "wandb_run_id": (wb_run.id if wb_run is not None else None),
            }
            if pmax_head is not None:
                save_dict["pmax_head_state"] = pmax_head.state_dict()
            if advantage_head is not None:
                save_dict["advantage_head_state"] = advantage_head.state_dict()
            torch.save(save_dict, ckpt_path)
            print(f"Saved {ckpt_path}")

    return global_step


def run_joint(
    args: argparse.Namespace,
    embedder: SolutionEmbedder,
    save_dir: str,
    plot_dir: str,
    wb_run,
    global_step: int,
    selected_stages: Optional[List[str]] = None,
) -> int:
    """Joint training: interleave S1/S2/S3 batches with a shared optimizer."""
    selected = set(selected_stages or ["1", "2", "3"])
    use_s1 = "1" in selected
    use_s2 = "2" in selected
    use_s3 = "3" in selected
    if not (use_s1 or use_s2 or use_s3):
        print("[Joint] No selected stages, skip.")
        return global_step

    print("\n" + "=" * 60)
    print(f"Joint Training: Interleaved stages {sorted(selected)}")
    print("=" * 60)

    # ── S1 data ──
    loader_s1, basin_data = (None, None)
    if use_s1:
        loader_s1, basin_data = build_loader_from_args(args, args.device)
    env = CVRPEnv(problem_size=args.problem_size, device=args.device)

    # ── S2 / S3 shared instance data ──
    indices = parse_instance_indices(args.instance_indices) if args.instance_indices else [0]
    basin_info_cache: Dict[str, dict] = {}
    instance_data_list = load_instances_pkl(args.instance_pkl, args.device, indices, basin_info_cache)
    instance_data_by_idx = dict(zip(indices, instance_data_list))

    # S2 triplets
    all_triplets: List[Tuple[int, dict]] = []
    if use_s2:
        for idx in indices:
            perturb_path = os.path.join(args.perturb_root, f"{args.instance_prefix}{idx}", "perturb_data.jsonl")
            if os.path.isfile(perturb_path):
                triplets = load_perturb_data(perturb_path)
                for t in triplets:
                    all_triplets.append((idx, t))
    print(f"[Joint] S2 triplets: {len(all_triplets)}")

    # ── S3 data (disk-cached) ──
    td_paths = [
        os.path.join(args.training_data_root, f"{args.instance_prefix}{idx}", "training_data.jsonl")
        for idx in indices
    ]
    cache_key_dict = {
        "td_paths": sorted(td_paths),
        "certainty_threshold": args.certainty_threshold,
        "seed": args.seed,
        "max_runs": args.max_traj_runs,
    }
    cache_hash = hashlib.md5(repr(sorted(cache_key_dict.items())).encode()).hexdigest()[:12]
    cache_dir = getattr(args, "s3_cache_dir", "s3_data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"s3_pairs_{cache_hash}.pkl")

    contrastive_pairs: List[Tuple[int, dict]] = []
    chaotic_samples: List[Tuple[int, dict]] = []
    if use_s3:
        cached = None
        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("key") == cache_key_dict:
                contrastive_pairs = cached["contrastive"]
                chaotic_samples = cached["chaotic"]
                print(f"[Joint] S3 cache hit: {len(contrastive_pairs)} contrastive, {len(chaotic_samples)} chaotic")
            else:
                cached = None
        if cached is None:
            contrastive_pairs, chaotic_samples = load_training_data_pairs(
                td_paths, certainty_threshold=args.certainty_threshold,
                seed=args.seed, max_runs=args.max_traj_runs,
            )
            with open(cache_path, "wb") as f:
                pickle.dump({"key": cache_key_dict, "contrastive": contrastive_pairs, "chaotic": chaotic_samples}, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[Joint] S3: {len(contrastive_pairs)} contrastive, {len(chaotic_samples)} chaotic")

    pairs_by_inst: Dict[int, List[dict]] = {}
    for idx, rec in contrastive_pairs:
        pairs_by_inst.setdefault(idx, []).append(rec)
    chaotic_by_inst: Dict[int, List[dict]] = {}
    for idx, rec in chaotic_samples:
        chaotic_by_inst.setdefault(idx, []).append(rec)

    # ── Model heads & optimizer ──
    pmax_head: Optional[PmaxHead] = None
    advantage_head: Optional[AdvantageOrdinalHead] = None
    params = list(embedder.parameters())
    if use_s3 and args.use_regression:
        pmax_head = PmaxHead(args.embedding_dim).to(args.device)
        params += list(pmax_head.parameters())
    if use_s3 and args.quality_reg_weight > 0:
        advantage_head = AdvantageOrdinalHead(args.embedding_dim, n_thresholds=len(ADVANTAGE_THRESHOLDS)).to(args.device)
        params += list(advantage_head.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr_joint, weight_decay=args.weight_decay)

    n_s1_batches = len(loader_s1) if use_s1 and loader_s1 is not None else 0
    n_s2_batches = max((len(all_triplets) + args.batch_size2 - 1) // args.batch_size2, 0)
    n_s3_batches = sum((len(r) + args.batch_size3 - 1) // args.batch_size3 for r in pairs_by_inst.values())
    batches_per_epoch = n_s1_batches + n_s2_batches + n_s3_batches
    start_epoch_joint = int(getattr(args, "resume_epoch_joint", 0) or 0)
    if start_epoch_joint >= args.epochs_joint:
        print(f"[Joint] Resume epoch {start_epoch_joint} >= target epochs {args.epochs_joint}, skip Joint.")
        return global_step
    total_steps = max(args.epochs_joint - start_epoch_joint, 0) * batches_per_epoch
    scheduler = make_lr_scheduler(optimizer, args.warmup_steps, total_steps)
    print(f"[Joint] Per-epoch batches: S1={n_s1_batches} S2={n_s2_batches} S3={n_s3_batches} total={batches_per_epoch}")

    # ── Training loop ──
    for epoch in range(start_epoch_joint, args.epochs_joint):
        epoch_batches: List[Tuple[str, Any]] = []

        if use_s1 and loader_s1 is not None:
            for batch in loader_s1:
                epoch_batches.append(("s1", batch))

        if use_s2:
            random.shuffle(all_triplets)
            for i in range(0, len(all_triplets), args.batch_size2):
                epoch_batches.append(("s2", all_triplets[i : i + args.batch_size2]))

        if use_s3:
            for idx, recs in pairs_by_inst.items():
                random.shuffle(recs)
                for i in range(0, len(recs), args.batch_size3):
                    epoch_batches.append(("s3", (idx, recs[i : i + args.batch_size3])))

        random.shuffle(epoch_batches)

        total_s1, total_s2, total_s3 = 0.0, 0.0, 0.0
        n_s1, n_s2, n_s3 = 0, 0, 0

        chaotic_iters: Dict[int, int] = {}
        if use_s3:
            for idx in chaotic_by_inst:
                random.shuffle(chaotic_by_inst[idx])
                chaotic_iters[idx] = 0

        pbar = tqdm(total=len(epoch_batches), desc=f"[Joint] Epoch {epoch+1}/{args.epochs_joint}", unit="batch")

        for tag, data in epoch_batches:
            embedder.train()
            if pmax_head is not None:
                pmax_head.train()
            if advantage_head is not None:
                advantage_head.train()
            optimizer.zero_grad()

            if tag == "s1":
                batch = data
                env.load(batch.depot_xy, batch.node_xy_demand, basin_data.basin_info)
                context = env.prepare_from_hashes(batch.hashes)
                emb = embedder(context, env)
                pair_idx = torch.tensor(batch.pair_indices, dtype=torch.long, device=args.device)
                w = torch.tensor(batch.weights, dtype=torch.float32, device=args.device)
                inc_mask = batch.include_mask.to(args.device) if batch.include_mask is not None else None
                loss = args.joint_s1_weight * weighted_infonce_loss(emb, pair_idx, w, args.temperature, include_mask=inc_mask)
                total_s1 += loss.item(); n_s1 += 1

            elif tag == "s2":
                batch_items = data
                emb_a = embed_multi_instance(embedder, [(idx, t["anchor_solution"]) for idx, t in batch_items], instance_data_by_idx, env, basin_info_cache)
                emb_p = embed_multi_instance(embedder, [(idx, t["positive_solution"]) for idx, t in batch_items], instance_data_by_idx, env, basin_info_cache)
                emb_n = embed_multi_instance(embedder, [(idx, t["negative_solution"]) for idx, t in batch_items], instance_data_by_idx, env, basin_info_cache)
                d_ap = F.pairwise_distance(emb_a, emb_p, p=2)
                d_an = F.pairwise_distance(emb_a, emb_n, p=2)
                loss = args.joint_s2_weight * F.relu(d_ap - d_an + args.margin).mean()
                total_s2 += loss.item(); n_s2 += 1

            elif tag == "s3":
                inst_idx, batch_records = data
                inst = instance_data_by_idx[inst_idx]
                env.load(inst["depot_xy"], inst["node_xy_demand"], basin_info_cache)

                emb_a = embed_solutions(embedder, [t["anchor_solution"] for t in batch_records], env, basin_info_cache)
                emb_p = embed_solutions(embedder, [t["positive_solution"] for t in batch_records], env, basin_info_cache)
                B = emb_a.size(0)

                pos_hashes = [t["positive_basin_hash"] for t in batch_records]
                reachable_sets = [t["reachable_basin_hashes"] for t in batch_records]
                all_hashes: set = set(pos_hashes)
                for rs in reachable_sets:
                    all_hashes.update(rs)
                hash_to_id = {h: j for j, h in enumerate(all_hashes)}
                pos_hash_ids = torch.tensor([hash_to_id[ph] for ph in pos_hashes], device=emb_a.device)
                reachable_mat = torch.zeros(B, len(all_hashes), dtype=torch.bool, device=emb_a.device)
                for bi, rs in enumerate(reachable_sets):
                    for rh in rs:
                        reachable_mat[bi, hash_to_id[rh]] = True
                valid_neg = ~reachable_mat[:, pos_hash_ids]
                valid_neg.fill_diagonal_(False)

                sim_raw = torch.mm(emb_a, emb_p.t())
                sim = sim_raw / args.infonce_temperature
                logit_mask = valid_neg.clone()
                logit_mask.fill_diagonal_(True)
                sim = sim.masked_fill(~logit_mask, -1e9)
                labels = torch.arange(B, device=emb_a.device)
                infonce_loss = F.cross_entropy(sim, labels)
                loss = args.joint_s3_weight * args.infonce_weight * infonce_loss

                if args.use_regression and pmax_head is not None and "p_max" in batch_records[0]:
                    p_max_t = torch.tensor([t["p_max"] for t in batch_records], dtype=torch.float32, device=emb_a.device)
                    emb_pm = emb_a.detach() if not args.regression_grad_encoder else emb_a
                    pmax_loss = F.mse_loss(pmax_head(emb_pm), p_max_t)
                    if inst_idx in chaotic_by_inst and chaotic_by_inst[inst_idx]:
                        c_pool = chaotic_by_inst[inst_idx]
                        c_start = chaotic_iters.get(inst_idx, 0)
                        c_batch = c_pool[c_start : c_start + args.batch_size3]
                        if not c_batch:
                            chaotic_iters[inst_idx] = 0
                            c_batch = c_pool[: args.batch_size3]
                        chaotic_iters[inst_idx] = (c_start + len(c_batch)) % max(len(c_pool), 1)
                        if args.regression_grad_encoder:
                            emb_ch = embed_solutions(embedder, [r["anchor_solution"] for r in c_batch], env, basin_info_cache)
                        else:
                            with torch.no_grad():
                                emb_ch = embed_solutions(embedder, [r["anchor_solution"] for r in c_batch], env, basin_info_cache)
                        p_max_ch_t = torch.tensor([r["p_max"] for r in c_batch], dtype=torch.float32, device=emb_ch.device)
                        emb_ch_d = emb_ch if args.regression_grad_encoder else emb_ch.detach()
                        pmax_loss = pmax_loss + F.mse_loss(pmax_head(emb_ch_d), p_max_ch_t)
                    loss = loss + args.joint_s3_weight * args.regression_weight * pmax_loss

                if advantage_head is not None and args.quality_reg_weight > 0:
                    c_int = torch.tensor([t["anchor_cost"] for t in batch_records], dtype=torch.float32, device=emb_a.device)
                    c_opt = torch.tensor([t["positive_cost"] for t in batch_records], dtype=torch.float32, device=emb_a.device)
                    gap_pct = ((c_int - c_opt) / c_opt.clamp(min=1e-8) * 100.0).clamp(min=0.0)
                    thresholds_t = torch.tensor(ADVANTAGE_THRESHOLDS, dtype=torch.float32, device=emb_a.device).unsqueeze(0)
                    ordinal_targets = (gap_pct.unsqueeze(1) > thresholds_t).float()
                    emb_adv = emb_a.detach() if not args.advantage_grad_encoder else emb_a
                    logits = advantage_head(emb_adv)
                    loss = loss + args.joint_s3_weight * args.quality_reg_weight * F.binary_cross_entropy_with_logits(logits, ordinal_targets)

                total_s3 += loss.item(); n_s3 += 1

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if wb_run is not None:
                step_log = {f"joint/step_{tag}_loss": loss.item()}
                # Mirror stage-style keys for direct comparison across runs.
                if tag == "s1":
                    step_log["s1/step_loss"] = loss.item()
                elif tag == "s2":
                    step_log["s2/step_loss"] = loss.item()
                elif tag == "s3":
                    step_log["s3/step_loss"] = loss.item()
                wandb.log(step_log, step=global_step)
            if (n_s1 + n_s2 + n_s3) % 10 == 0:
                pbar.set_postfix({
                    "tag": tag,
                    "loss": f"{loss.item():.4f}",
                })
            pbar.update(1)
            global_step += 1

        pbar.close()
        ns1, ns2, ns3 = max(n_s1, 1), max(n_s2, 1), max(n_s3, 1)
        print(
            f"[Joint] Epoch {epoch+1}/{args.epochs_joint} "
            f"s1={total_s1/ns1:.6f}({n_s1}) s2={total_s2/ns2:.6f}({n_s2}) s3={total_s3/ns3:.6f}({n_s3})"
        )
        if wb_run is not None:
            epoch_log = {
                "joint/epoch_s1_loss": total_s1 / ns1,
                "joint/epoch_s2_loss": total_s2 / ns2,
                "joint/epoch_s3_loss": total_s3 / ns3,
            }
            if use_s1 and n_s1 > 0:
                epoch_log["s1/epoch_loss"] = total_s1 / ns1
            if use_s2 and n_s2 > 0:
                epoch_log["s2/epoch_loss"] = total_s2 / ns2
            if use_s3 and n_s3 > 0:
                epoch_log["s3/epoch_loss"] = total_s3 / ns3
            wandb.log(epoch_log, step=global_step)

        if use_s3 and (epoch + 1) % args.plot_interval == 0 and pairs_by_inst:
            embedder.eval()
            with torch.no_grad():
                list_pos_sim, list_neg_sim = _collect_s3_sim_hist(
                    pairs_by_inst, instance_data_by_idx, embedder, env, basin_info_cache, args.batch_size3)
            embedder.train()
            if list_pos_sim and list_neg_sim:
                _log_histogram(list_pos_sim, list_neg_sim,
                               os.path.join(plot_dir, f"sim_hist_joint_epoch{epoch+1}.png"), "Joint",
                               wb_run, global_step, "plot_joint/sim_hist", "plot_joint/hist_wasserstein")

        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs_joint:
            save_dict = {
                "embedder_state": embedder.state_dict(),
                "epoch": epoch + 1,
                "stage": "joint",
                "global_step": global_step,
                "wandb_run_id": (wb_run.id if wb_run is not None else None),
            }
            if pmax_head is not None:
                save_dict["pmax_head_state"] = pmax_head.state_dict()
            if advantage_head is not None:
                save_dict["advantage_head_state"] = advantage_head.state_dict()
            ckpt_path = os.path.join(save_dir, f"joint_epoch{epoch+1}.pt")
            torch.save(save_dict, ckpt_path)
            print(f"Saved {ckpt_path}")

    return global_step


def trainer(args: argparse.Namespace) -> None:
    """Main trainer"""
    args.resume_epoch1 = 0
    args.resume_epoch2 = 0
    args.resume_epoch3 = 0
    args.resume_epoch_joint = 0

    if args.resume_save_dir:
        save_dir = os.path.abspath(args.resume_save_dir)
        run_name = os.path.basename(os.path.normpath(save_dir))
        print(f"[resume] Using existing save_dir: {save_dir}")
    elif args.load_checkpoint:
        save_dir = os.path.dirname(os.path.abspath(args.load_checkpoint))
        run_name = os.path.basename(os.path.normpath(save_dir))
        print(f"[resume] Auto-using checkpoint save_dir: {save_dir}")
    else:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{time_str}_{args.instance_indices}" if args.instance_indices else time_str
        if args.note:
            run_name = f"{run_name}_{args.note}"
        save_dir = os.path.join(args.save, run_name)
    os.makedirs(save_dir, exist_ok=True)
    plot_dir = os.path.join(save_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)
    copy_all_src(save_dir, home_dir=os.path.dirname(os.path.abspath(__file__)))

    # Initialize wandb
    wb_run = None
    resumed_wandb_step = 0
    ckpt_step = 0
    if not args.disable_wandb:
        init_kwargs = {"project": "landscape", "name": run_name, "config": vars(args)}
        resume_id = args.wandb_run_id
        if resume_id is None and args.load_checkpoint and not args.no_wandb_auto_resume:
            resume_id = _extract_wandb_run_id_from_checkpoint(args.load_checkpoint)
            prev_save_dir = os.path.dirname(os.path.abspath(args.load_checkpoint))
            id_path = os.path.join(prev_save_dir, "wandb_run_id.txt")
            if resume_id is None and os.path.isfile(id_path):
                try:
                    with open(id_path, "r", encoding="utf-8") as f:
                        resume_id = f.read().strip() or None
                except OSError:
                    resume_id = None
            if resume_id is None:
                resume_id = _infer_wandb_run_id_from_local(
                    load_checkpoint=args.load_checkpoint,
                    project_root=os.path.dirname(os.path.abspath(__file__)),
                )

        if resume_id is not None:
            init_kwargs["id"] = resume_id
            init_kwargs["resume"] = args.wandb_resume_mode
            init_kwargs["name"] = os.path.basename(os.path.dirname(os.path.abspath(args.load_checkpoint)))
            print(f"[wandb] Resuming run id={resume_id} (mode={args.wandb_resume_mode})")

        wb_run = wandb.init(**init_kwargs)
        if wb_run is not None:
            # Prevent step going backwards when resuming the same run.
            resumed_wandb_step = int(getattr(wb_run, "step", 0) or 0)
            try:
                with open(os.path.join(save_dir, "wandb_run_id.txt"), "w", encoding="utf-8") as f:
                    f.write(wb_run.id)
            except OSError:
                pass

    # Initialize model
    model_params = {
        "problem": "CVRP",
        "embedding_dim": args.embedding_dim,
        "encoder_layer_num": args.n_layers,
        "supplement_feature_dim": args.supplement_feature_dim,
        "depot_feature_dim": 5,
        "node_feature_dim": 6,
        "head_num": args.n_heads,
        "qkv_dim": args.embedding_dim // args.n_heads,
        "hidden_dim": args.hidden_dim,
        "use_l2_normalize": args.use_l2_normalize,
    }
    embedder = SolutionEmbedder(model_params).to(args.device)

    global_step = 0

    # Load checkpoint if provided
    if args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location=args.device)
        if "embedder_state" in ckpt:
            embedder.load_state_dict(ckpt["embedder_state"])
        elif "encoder_state" in ckpt:
            embedder.encoder.load_state_dict(ckpt["encoder_state"])
            print("WARNING: legacy checkpoint has encoder_state only; pos_encoder weights are randomly initialized")
        ckpt_step = int(ckpt.get("global_step", 0) or 0)
        global_step = ckpt_step
        ckpt_epoch = int(ckpt.get("epoch", 0) or 0)
        ckpt_stage = str(ckpt.get("stage", ""))
        if ckpt_stage == "1":
            args.resume_epoch1 = ckpt_epoch
        elif ckpt_stage == "2":
            args.resume_epoch2 = ckpt_epoch
        elif ckpt_stage == "3":
            args.resume_epoch3 = ckpt_epoch
        elif ckpt_stage.lower() == "joint":
            args.resume_epoch_joint = ckpt_epoch
        print(f"Loaded checkpoint: {args.load_checkpoint} (epoch {ckpt.get('epoch', '?')}, stage {ckpt.get('stage', '?')})")

    if wb_run is not None:
        global_step = max(global_step, resumed_wandb_step)
    if args.print_resume_state:
        print(
            "[resume] "
            f"ckpt_global_step={ckpt_step} "
            f"wandb_step={resumed_wandb_step if wb_run is not None else 'N/A'} "
            f"start_global_step={global_step}"
        )

    # Run stages: new API (train_mode + train_stages), fallback to legacy --stage
    selected_stages = (
        _parse_train_stages(args.train_stages)
        if args.train_stages is not None
        else _legacy_stage_to_stages(args.stage)
    )
    train_mode = (
        args.train_mode
        if args.train_mode is not None
        else ("joint" if args.stage == "joint" else "sequential")
    )
    print(f"[train] mode={train_mode} stages={selected_stages}")

    if train_mode == "sequential":
        if "1" in selected_stages:
            global_step = run_stage1(args, embedder, save_dir, plot_dir, wb_run, global_step)
        if "2" in selected_stages:
            global_step = run_stage2(args, embedder, save_dir, plot_dir, wb_run, global_step)
        if "3" in selected_stages:
            global_step = run_stage3(args, embedder, save_dir, plot_dir, wb_run, global_step)
    else:
        global_step = run_joint(args, embedder, save_dir, plot_dir, wb_run, global_step, selected_stages=selected_stages)

    if wb_run is not None:
        wandb.finish()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train solution embeddings with contrastive learning.")

    # instance related
    parser.add_argument("--problem_size", type=int, default=100, help="Problem size for CVRPEnv (number of customers).")
    parser.add_argument("--instance_root", type=str, default="basin_datasets0_analyze")
    parser.add_argument("--instance_prefix", type=str, default="cvrp100_uniform.pkl#", help="Prefix inside instance_root, final dir is prefix + index (default: cvrp100_uniform.pkl#).")
    parser.add_argument("--instance_indices", type=str, default=None, help="Data dir indices to merge, e.g. '0-49' or '0,1,2'. Graph from --instance_pkl.")
    parser.add_argument("--instance_pkl", type=str, default=_default_instance_pkl_path(), help="Path to NeuOpt-style CVRP instance pkl.")

    # model related
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--supplement_feature_dim", type=int, default=5, help="Extra feature dim for encoder (from CVRPEnv.get_dynamic_feature).")
    parser.add_argument("--use_l2_normalize", action="store_true", default=True, help="L2-normalize pooled embeddings in SolutionEmbedder.forward (default: True).")

    # stage selection
    parser.add_argument("--stage", type=str, choices=["1", "2", "3", "both", "all", "sequential", "joint"], default="both", help="Training stage: 1/2/3=single stage, both=S1+S2, all/sequential=S1->S2->S3 sequentially, joint=interleaved S1+S2+S3.")
    parser.add_argument("--train_mode", type=str, choices=["sequential", "joint"], default=None, help="Training mode (new API). If set, overrides mode implied by --stage.")
    parser.add_argument("--train_stages", type=str, default=None, help="Comma-separated subset of stages to train, e.g. '1,3' (new API).")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to load before training.")

    # stage 1 specific
    parser.add_argument("--epochs1", type=int, default=100, help="Target total epochs for stage 1 (if resuming S1 from epoch k, runs k+1..epochs1).")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for stage 1.")
    parser.add_argument("--lr1", type=float, default=5e-4, help="Learning rate for stage 1.")
    parser.add_argument("--neg_mode", type=str, choices=["distant", "masked_in_batch"], default="masked_in_batch", help="Negative sampling for stage 1.")
    parser.add_argument("--max_negatives", type=int, default=64, help="Max distant basins per anchor (stage 1).")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for InfoNCE loss (stage 1).")
    parser.add_argument("--masked_neg_zero_ratio_warn", type=float, default=0.1, help="Warn if epoch masked-neg zero ratio exceeds this threshold (stage 1, masked_in_batch).")

    # stage 2 specific
    parser.add_argument("--epochs2", type=int, default=50, help="Target total epochs for stage 2 (if resuming S2 from epoch k, runs k+1..epochs2).")
    parser.add_argument("--batch_size2", type=int, default=128, help="Batch size for stage 2.")
    parser.add_argument("--lr2", type=float, default=5e-5, help="Learning rate for stage 2.")
    parser.add_argument("--margin", type=float, default=0.1, help="Triplet margin (stage 2). With L2-normalized embeddings, max distance is 2.0.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW (both stages).")
    parser.add_argument("--perturb_root", type=str, default="perturb_k1_collect", help="Root dir for perturb_data.jsonl files (stage 2).")
    parser.add_argument("--val_data_1a1n10d", type=str, default=os.path.join(PROJECT_ROOT, "basin_datasets0_analyze", "val_data_1a1n10d.jsonl"), help="Stage-1 fixed validation set (anchor, neighbour, 10 distant basins).")
    parser.add_argument("--val_data_1p1n", type=str, default=os.path.join(PROJECT_ROOT, "perturb_k1_collect", "val_data_1p1n.jsonl"), help="Fixed validation set for stage 2 (1 anchor, 1 positive, 1 negative per line); instance_index maps to pkl.")

    # stage 3 specific
    parser.add_argument("--epochs3", type=int, default=100, help="Target total epochs for stage 3 (if resuming S3 from epoch k, runs k+1..epochs3).")
    parser.add_argument("--batch_size3", type=int, default=512, help="Batch size for stage 3.")
    parser.add_argument("--lr3", type=float, default=5e-5, help="Learning rate for stage 3.")
    parser.add_argument("--max_traj_runs", type=int, default=10, help="Max runs to load per instance from training_data.jsonl (stage 3).")
    parser.add_argument("--training_data_root", type=str, default="basin_datasets0_analyze", help="Root dir for training_data.jsonl files.")
    parser.add_argument("--s3_cache_dir", type=str, default="s3_data_cache", help="Dir to cache Stage 3 processed pairs (avoids re-parsing on repeated runs).")
    parser.add_argument("--certainty_threshold", type=float, default=0.8, help="P_max >= this -> contrastive pair; below -> regression only.")
    parser.add_argument("--use_regression", action="store_true", help="Enable Task 2: regress P_max from embedding.")
    parser.add_argument("--regression_weight", type=float, default=1.0, help="Lambda weight for regression loss.")
    parser.add_argument("--regression_grad_encoder", action="store_true", help="Let regression loss backprop into encoder (default: detach).")
    parser.add_argument("--infonce_weight", type=float, default=1.0, help="Lambda for InfoNCE contrastive loss (stage 3).")
    parser.add_argument("--infonce_temperature", type=float, default=0.07, help="Temperature for InfoNCE loss (stage 3). Independent from --temperature (stage 1).")
    parser.add_argument("--quality_reg_weight", type=float, default=0.0, help="Lambda for advantage ordinal BCE loss: predict gap%% bins. 0=disabled.")
    parser.add_argument("--advantage_grad_encoder", action="store_true", help="Let advantage loss backprop into encoder (default: detach).")
    parser.add_argument("--s3_val_instance_indices", type=str, default="50-55", help="Fixed eval instances for Stage 3, e.g. '50-55'.")
    parser.add_argument("--s3_val_max_traj_runs", type=int, default=10, help="Max runs to load per fixed Stage-3 eval instance.")

    # joint training
    parser.add_argument("--epochs_joint", type=int, default=200, help="Target total epochs for joint training (if resuming joint from epoch k, runs k+1..epochs_joint).")
    parser.add_argument("--lr_joint", type=float, default=5e-4, help="Learning rate for joint training.")
    parser.add_argument("--joint_s1_weight", type=float, default=1.0, help="Loss weight for S1 in joint training.")
    parser.add_argument("--joint_s2_weight", type=float, default=1.0, help="Loss weight for S2 in joint training.")
    parser.add_argument("--joint_s3_weight", type=float, default=1.0, help="Loss weight for S3 in joint training.")

    # common
    parser.add_argument("--warmup_steps", type=int, default=0, help="LR warmup steps then cosine decay. 0=no scheduling.")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Max gradient norm for clipping. 0=disabled.")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--save", type=str, default="out", help="Root dir for checkpoints.")
    parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoint every N epochs.")
    parser.add_argument("--plot_interval", type=int, default=5, help="Plot embeddings and distance histogram every N epochs (saved to save_dir/plot).")
    parser.add_argument("--note", type=str, default=None, help="Note appended to run name.")
    parser.add_argument("--resume_save_dir", type=str, default=None, help="If set, write outputs into this existing directory instead of creating a new timestamped run folder.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="Explicit W&B run id to resume.")
    parser.add_argument("--wandb_resume_mode", type=str, choices=["allow", "must", "never"], default="allow", help="W&B resume mode when run id is set or auto-detected.")
    parser.add_argument("--no_wandb_auto_resume", action="store_true", help="Disable auto-detection of W&B run id from --load_checkpoint.")
    parser.add_argument("--print_resume_state", action="store_true", default=True, help="Print checkpoint/wandb/final starting global_step for resume sanity check.")
    parser.add_argument("--no_print_resume_state", action="store_true", help="Disable resume-state print.")

    args = parser.parse_args()
    if args.no_print_resume_state:
        args.print_resume_state = False

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.cuda.set_device(0)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        args.device = torch.device('cpu')
    print(">> USE_CUDA: {}, CUDA_DEVICE_NUM: {}".format(torch.cuda.is_available(), args.gpu_id))

    torch.set_printoptions(threshold=1000000)
    seed_everything(args.seed)

    trainer(args)

