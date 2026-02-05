#!/usr/bin/env python3
"""
Train solution embeddings with contrastive learning.

Stage 1: Weighted InfoNCE with basin pairs.
Stage 2: Triplet Margin Loss with perturb data (anchor, positive, negative).
"""

import argparse
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from dataloader import (
    TrainBatch,
    TripletBatch,
    build_loader_from_args,
    load_perturb_data,
    load_val_data_1a1n10d,
    load_val_data_1p1n,
    parse_instance_indices,
)
from CVRPEnv import CVRPEnv
from net import SolutionEmbedder
from helper import copy_all_src, load_instances_pkl, plot_distance_histogram, plot_embedding_2d, seed_everything


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


def train_one_triplet_batch(
    embedder: SolutionEmbedder,
    optimizer: torch.optim.Optimizer,
    batch: TripletBatch,
    env: CVRPEnv,
    basin_info_cache: Dict[str, dict],
    margin: float,
) -> Tuple[float, float, float, float]:
    """Train on one triplet batch. Returns (loss, mean_d_ap, mean_d_an, mean_d_an_over_d_ap)."""
    embedder.train()
    optimizer.zero_grad()

    env.load(batch.depot_xy, batch.node_xy_demand, basin_info_cache)

    emb_a = embed_solutions(embedder, batch.anchor_solutions, env, basin_info_cache)
    emb_p = embed_solutions(embedder, batch.positive_solutions, env, basin_info_cache)
    emb_n = embed_solutions(embedder, batch.negative_solutions, env, basin_info_cache)

    d_ap = F.pairwise_distance(emb_a, emb_p, p=2)
    d_an = F.pairwise_distance(emb_a, emb_n, p=2)
    loss = F.relu(d_ap - d_an + margin).mean()

    loss.backward()
    optimizer.step()

    mean_d_ap = d_ap.mean().item()
    mean_d_an = d_an.mean().item()
    ratio = mean_d_an / (mean_d_ap + 1e-8)
    return loss.item(), mean_d_ap, mean_d_an, ratio


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
    optimizer.step()
    return loss.item()


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

    for epoch in range(args.epochs1):
        total_loss = 0.0
        n_batches = 0
        first_batch_for_plot = None

        for batch_idx, batch in enumerate(
            tqdm(loader, desc=f"[S1] Epoch {epoch+1}/{args.epochs1}", unit="batch"), start=1
        ):
            env.load(batch.depot_xy, batch.node_xy_demand, basin_data.basin_info)
            loss = train_one_batch(embedder, optimizer, batch, env, args.device, args.temperature)

            total_loss += loss
            n_batches += 1

            if (epoch + 1) % args.plot_interval == 0 and first_batch_for_plot is None:
                first_batch_for_plot = batch

            if wb_run is not None:
                wandb.log({"s1/step_loss": loss, "s1/epoch": epoch + 1}, step=global_step)
            if batch_idx % 100 == 0:
                print(f"[S1 Epoch {epoch+1}] batch {batch_idx}, loss={loss:.6f}")
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
                plot_distance_histogram(
                    val_list_d_ap,
                    val_list_d_an,
                    save_path=os.path.join(plot_dir, f"distance_hist_s1_epoch{epoch+1}.png"),
                )
                print(f"[S1] Saved plot (val) to {plot_dir}/distance_hist_s1_epoch{epoch+1}.png")

            if first_triplet is not None:
                inst_idx, rec, inst = first_triplet
                basin_cache: Dict[str, dict] = {}
                embedder.eval()
                with torch.no_grad():
                    env.load(inst["depot_xy"], inst["node_xy_demand"], basin_cache)
                    emb_a = embed_solutions(embedder, [rec["anchor_solution"]], env, basin_cache)
                    emb_p = embed_solutions(embedder, [rec["neighbor_solution"]], env, basin_cache)
                    emb_d = embed_solutions(embedder, rec["distant_solutions"], env, basin_cache) if rec["distant_solutions"] else None
                    if emb_d is not None:
                        emb = torch.cat([emb_a, emb_p, emb_d], dim=0)
                        group_labels = [0, 1] + [2] * emb_d.size(0)
                    else:
                        emb = torch.cat([emb_a, emb_p], dim=0)
                        group_labels = [0, 1]
                    instance_ids = [inst_idx] * len(group_labels)
                    plot_embedding_2d(
                        emb,
                        instance_ids=instance_ids,
                        group_labels=group_labels,
                        method="pca",
                        save_path=os.path.join(plot_dir, f"embedding_2d_s1_epoch{epoch+1}.png"),
                    )
                embedder.train()
                print(f"[S1] Saved plot (val) to {plot_dir}/embedding_2d_s1_epoch{epoch+1}.png")

        avg_loss = total_loss / max(n_batches, 1)
        print(f"[S1] Epoch {epoch+1}/{args.epochs1} loss={avg_loss:.6f}")
        if wb_run is not None:
            wandb.log({"s1/epoch_loss": avg_loss}, step=global_step)

        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs1:
            ckpt_path = os.path.join(save_dir, f"s1_epoch{epoch+1}.pt")
            torch.save({"encoder_state": embedder.encoder.state_dict(), "epoch": epoch + 1, "stage": 1}, ckpt_path)
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
            inst_idx = batch_items[0][0]
            inst = val_instance_data_by_idx[inst_idx]
            batch = TripletBatch(
                anchor_solutions=[t["anchor_solution"] for _, t in batch_items],
                positive_solutions=[t["positive_solution"] for _, t in batch_items],
                negative_solutions=[t["negative_solution"] for _, t in batch_items],
                instance_idx=inst_idx,
                depot_xy=inst["depot_xy"],
                node_xy_demand=inst["node_xy_demand"],
            )
            if first_batch is None:
                first_batch = batch
            env.load(batch.depot_xy, batch.node_xy_demand, basin_info_cache)
            emb_a = embed_solutions(embedder, batch.anchor_solutions, env, basin_info_cache)
            emb_p = embed_solutions(embedder, batch.positive_solutions, env, basin_info_cache)
            emb_n = embed_solutions(embedder, batch.negative_solutions, env, basin_info_cache)
            d_ap = F.pairwise_distance(emb_a, emb_p, p=2)
            d_an = F.pairwise_distance(emb_a, emb_n, p=2)
            list_d_ap.extend(d_ap.cpu().tolist())
            list_d_an.extend(d_an.cpu().tolist())
            mean_ap, mean_an = d_ap.mean().item(), d_an.mean().item()
            sum_ratio += mean_an / (mean_ap + 1e-8)
            n += 1
        mean_d_ap = sum(list_d_ap) / len(list_d_ap) if list_d_ap else 0.0
        mean_d_an = sum(list_d_an) / len(list_d_an) if list_d_an else 0.0
        mean_ratio = sum_ratio / n if n else 0.0
        return mean_d_ap, mean_d_an, mean_ratio, list_d_ap, list_d_an, first_batch

    for epoch in range(args.epochs2):
        random.shuffle(all_triplets)
        total_loss, total_d_ap, total_d_an, total_ratio = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        pbar = tqdm(total=(len(all_triplets) + args.batch_size2 - 1) // args.batch_size2,
                    desc=f"[S2] Epoch {epoch+1}/{args.epochs2}", unit="batch")
        batch_start = 0

        while batch_start < len(all_triplets):
            batch_items = all_triplets[batch_start : batch_start + args.batch_size2]
            batch_start += args.batch_size2

            inst_idx = batch_items[0][0]
            triplets_only = [item[1] for item in batch_items]
            inst = instance_data_by_idx[inst_idx]
            batch = TripletBatch(
                anchor_solutions=[t["anchor_solution"] for t in triplets_only],
                positive_solutions=[t["positive_solution"] for t in triplets_only],
                negative_solutions=[t["negative_solution"] for t in triplets_only],
                instance_idx=inst_idx,
                depot_xy=inst["depot_xy"],
                node_xy_demand=inst["node_xy_demand"],
            )

            loss, d_ap, d_an, ratio = train_one_triplet_batch(
                embedder, optimizer, batch, env, basin_info_cache, args.margin
            )

            total_loss += loss
            total_d_ap += d_ap
            total_d_an += d_an
            total_ratio += ratio
            n_batches += 1

            if wb_run is not None:
                wandb.log({"s2/step_loss": loss, "s2/step_d_ap": d_ap, "s2/step_d_an": d_an, "s2/step_d_an_over_d_ap": ratio}, step=global_step)
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
                val_d_ap, val_d_an, val_ratio, val_list_d_ap, val_list_d_an, val_first_batch = eval_on_val()
            embedder.train()

            print(f"[S2] Val (fixed) d(A,P)={val_d_ap:.4f} d(A,N)={val_d_an:.4f} d(A,N)/d(A,P)={val_ratio:.4f}")
            if wb_run is not None:
                wandb.log(
                    {"val_s2/d_ap": val_d_ap, "val_s2/d_an": val_d_an, "val_s2/d_an_over_d_ap": val_ratio},
                    step=global_step,
                )

            # Plots from same fixed val set
            if val_list_d_ap and val_list_d_an:
                plot_distance_histogram(
                    val_list_d_ap, val_list_d_an,
                    save_path=os.path.join(plot_dir, f"distance_hist_epoch{epoch+1}.png"),
                )
                print(f"[S2] Saved plot (val) to {plot_dir}/distance_hist_epoch{epoch+1}.png")
            if val_first_batch is not None:
                embedder.eval()
                with torch.no_grad():
                    env.load(val_first_batch.depot_xy, val_first_batch.node_xy_demand, basin_info_cache)
                    emb_a = embed_solutions(embedder, val_first_batch.anchor_solutions, env, basin_info_cache)
                    emb_p = embed_solutions(embedder, val_first_batch.positive_solutions, env, basin_info_cache)
                    emb_n = embed_solutions(embedder, val_first_batch.negative_solutions, env, basin_info_cache)
                    emb = torch.cat([emb_a, emb_p, emb_n], dim=0)
                    B = emb_a.size(0)
                    group_labels = [0] * B + [1] * B + [2] * B
                    instance_ids = [val_first_batch.instance_idx] * (3 * B)
                    plot_embedding_2d(
                        emb,
                        instance_ids=instance_ids,
                        group_labels=group_labels,
                        method="pca",
                        save_path=os.path.join(plot_dir, f"embedding_2d_s2_epoch{epoch+1}.png"),
                    )
                embedder.train()
                print(f"[S2] Saved plot (val) to {plot_dir}/embedding_2d_s2_epoch{epoch+1}.png")

        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs2:
            ckpt_path = os.path.join(save_dir, f"s2_epoch{epoch+1}.pt")
            torch.save({"encoder_state": embedder.encoder.state_dict(), "epoch": epoch + 1, "stage": 2}, ckpt_path)
            print(f"Saved {ckpt_path}")

    return global_step


def trainer(args: argparse.Namespace) -> None:
    """Main trainer: runs stage 1, stage 2, or both."""
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{time_str}_{args.instance_indices}" if args.instance_indices else time_str
    if args.note: run_name = f"{run_name}_{args.note}"
    save_dir = os.path.join(args.save, run_name)
    os.makedirs(save_dir, exist_ok=True)
    plot_dir = os.path.join(save_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)
    copy_all_src(save_dir, home_dir=os.path.dirname(os.path.abspath(__file__)))

    # Initialize wandb
    wb_run = None
    if not args.disable_wandb:
        wb_run = wandb.init(project="landscape", name=run_name, config=vars(args))

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

    # Load checkpoint if provided (for stage 2 only mode)
    if args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location=args.device)
        embedder.encoder.load_state_dict(ckpt["encoder_state"])
        print(f"Loaded checkpoint: {args.load_checkpoint} (epoch {ckpt.get('epoch', '?')}, stage {ckpt.get('stage', '?')})")

    global_step = 0

    # Run stages
    if args.stage in ("1", "both"):
        global_step = run_stage1(args, embedder, save_dir, plot_dir, wb_run, global_step)

    if args.stage in ("2", "both"):
        global_step = run_stage2(args, embedder, save_dir, plot_dir, wb_run, global_step)

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
    parser.add_argument("--instance_pkl", type=str, default="/home/jieyi/cvrp100_uniform.pkl", help="Path to NeuOpt-style CVRP instance pkl.")

    # model related
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--supplement_feature_dim", type=int, default=5, help="Extra feature dim for encoder (from CVRPEnv.get_dynamic_feature).")
    parser.add_argument(
        "--use_l2_normalize",
        action="store_true",
        help="If set, L2-normalize pooled embeddings in SolutionEmbedder.forward (default off; triplet loss sees raw vectors).",
    )

    # stage selection
    parser.add_argument("--stage", type=str, choices=["1", "2", "both"], default="both", help="Training stage: 1=InfoNCE only, 2=Triplet only, both=run stage1 then stage2.")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to load before training.")

    # stage 1 specific
    parser.add_argument("--epochs1", type=int, default=50, help="Epochs for stage 1.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for stage 1.")
    parser.add_argument("--lr1", type=float, default=5e-4, help="Learning rate for stage 1.")
    parser.add_argument("--neg_mode", type=str, choices=["distant", "masked_in_batch"], default="masked_in_batch", help="Negative sampling for stage 1.")
    parser.add_argument("--max_negatives", type=int, default=64, help="Max distant basins per anchor (stage 1).")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for InfoNCE loss (stage 1).")

    # stage 2 specific
    parser.add_argument("--epochs2", type=int, default=100, help="Epochs for stage 2.")
    parser.add_argument("--batch_size2", type=int, default=128, help="Batch size for stage 2.")
    parser.add_argument("--lr2", type=float, default=5e-5, help="Learning rate for stage 2.")
    parser.add_argument("--margin", type=float, default=0.1, help="Triplet margin (stage 2).")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW (both stages).")
    parser.add_argument("--perturb_root", type=str, default="perturb_k1_collect", help="Root dir for perturb_data.jsonl files (stage 2).")
    parser.add_argument(
        "--val_data_1a1n10d",
        type=str,
        default="/home/jieyi/cuopt/basin_datasets0_analyze/val_data_1a1n10d.jsonl",
        help="Stage-1 fixed validation set (anchor, neighbour, 10 distant basins).",
    )
    parser.add_argument("--val_data_1p1n", type=str, default="/home/jieyi/cuopt/perturb_k1_collect/val_data_1p1n.jsonl", help="Fixed validation set for stage 2 (1 anchor, 1 positive, 1 negative per line); instance_index maps to pkl.")

    # common
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--save", type=str, default="out", help="Root dir for checkpoints.")
    parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoint every N epochs.")
    parser.add_argument("--plot_interval", type=int, default=5, help="Plot embeddings and distance histogram every N epochs (saved to save_dir/plot).")
    parser.add_argument("--note", type=str, default=None, help="Note appended to run name.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging.")

    args = parser.parse_args()

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

