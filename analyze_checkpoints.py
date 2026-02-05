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
import glob
import os
from typing import Dict, List, Optional, Tuple

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
)
from net import SolutionEmbedder


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
        # Infer embedding_dim and supplement_feature_dim from embedding_depot.weight
        w_depot = encoder_state.get("embedding_depot.weight")
        if w_depot is not None:
            embedding_dim = int(w_depot.shape[0])
            in_dim = int(w_depot.shape[1])
            supplement_feature_dim = max(in_dim - 2, 0)
        # Infer hidden_dim from first FF layer if available
        w_ff1 = encoder_state.get("layers.0.ff.W1.weight")
        if w_ff1 is not None:
            hidden_dim = int(w_ff1.shape[0])

    model_params = {
        "problem": "CVRP",
        "embedding_dim": embedding_dim,
        "encoder_layer_num": args.n_layers,
        "supplement_feature_dim": supplement_feature_dim,
        "depot_feature_dim": 5,
        "node_feature_dim": 6,
        "head_num": args.n_heads,
        "qkv_dim": args.embedding_dim // args.n_heads,
        "hidden_dim": hidden_dim,
        # For analysis we default to raw (unnormalized) embeddings unless user explicitly enables it.
        "use_l2_normalize": args.use_l2_normalize,
    }
    return SolutionEmbedder(model_params).to(device)


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
        embedder = build_model(args, device, encoder_state=ckpt.get("encoder_state"))
        embedder.encoder.load_state_dict(ckpt["encoder_state"])

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
        embedder = build_model(args, device, encoder_state=ckpt.get("encoder_state"))
        embedder.encoder.load_state_dict(ckpt["encoder_state"])

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline visualization/validation for checkpoints.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing s1_epoch*.pt / s2_epoch*.pt checkpoints.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["1", "2", "both"],
        default="both",
        help="Which stages to analyze.",
    )

    # Model / data settings (should match training)
    parser.add_argument("--problem_size", type=int, default=100)
    parser.add_argument("--instance_pkl", type=str, default="/home/jieyi/cvrp100_uniform.pkl")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--supplement_feature_dim", type=int, default=5)
    parser.add_argument("--use_l2_normalize", action="store_true", help="L2-normalize pooled embeddings in SolutionEmbedder.forward during analysis (default off).")

    # Validation files
    parser.add_argument(
        "--val_data_1a1n10d",
        type=str,
        default="/home/jieyi/cuopt/basin_datasets0_analyze/val_data_1a1n10d.jsonl",
    )
    parser.add_argument(
        "--val_data_1p1n",
        type=str,
        default="/home/jieyi/cuopt/perturb_k1_collect/val_data_1p1n.jsonl",
    )

    parser.add_argument("--batch_size2", type=int, default=128, help="Batch size for Stage 2 val.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f">> Using device: {device}")

    if args.stage in ("1", "both"):
        analyze_stage1(args, device)
    if args.stage in ("2", "both"):
        analyze_stage2(args, device)


if __name__ == "__main__":
    main()

