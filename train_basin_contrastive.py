#!/usr/bin/env python3
"""
Train solution embeddings with contrastive learning.

Inputs:
- basin_pairs.jsonl: positive pairs (anchor, neighbor) with weights
- distant_basins.jsonl: negative candidates per anchor (optional; used as in-batch negatives)
- basin_info.jsonl: basin hash -> solution (route-like visit sequence with depot=0)

We use the Encoder to produce node embeddings, then mean-pool to get a solution-level
embedding. Training objective is weighted InfoNCE:

  L = - sum_{(i,j) in Pairs} W_ij * log( exp(sim(i,j)/tau) / sum_k exp(sim(i,k)/tau) )
"""

import argparse
import os
from datetime import datetime
from typing import Optional
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from dataloader import TrainBatch, build_loader_from_args
from CVRPEnv import CVRPEnv
from net import SolutionEmbedder
from helper import copy_all_src, seed_everything


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

    pair_idx = torch.tensor(batch.pair_indices, dtype=torch.long, device=args.device)
    w = torch.tensor(batch.weights, dtype=torch.float32, device=args.device)
    include_mask = batch.include_mask.to(args.device) if batch.include_mask is not None else None
    loss = weighted_infonce_loss(emb, pair_idx, w, temperature, include_mask=include_mask)

    loss.backward()
    optimizer.step()
    return loss.item()


def trainer(args: argparse.Namespace) -> None:
    # Run name & save dir
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{time_str}_{args.instance_indices}" if args.instance_indices else time_str
    if args.note: run_name = f"{run_name}_{args.note}"
    save_dir = os.path.join(args.save, run_name)
    os.makedirs(save_dir, exist_ok=True)
    copy_all_src(save_dir, home_dir=os.path.dirname(os.path.abspath(__file__)))

    # Initialize wandb
    wb_run = None
    if not args.disable_wandb:
        wb_run = wandb.init(
            project="landscape",
            name=run_name,
            config={
                "problem_size": args.problem_size,
                "instance_root": args.instance_root,
                "instance_indices": args.instance_indices,
                "neg_mode": args.neg_mode,
                "instance_pkl": args.instance_pkl,
                "embedding_dim": args.embedding_dim,
                "hidden_dim": args.hidden_dim,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "temperature": args.temperature,
                "batch_size": args.batch_size,
                "max_negatives": args.max_negatives,
                "epochs": args.epochs,
                "lr": args.lr,
                "device": args.device,
            },
        )

    # Initialize model and optimizer
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
    }
    embedder = SolutionEmbedder(model_params).to(args.device)
    optimizer = torch.optim.Adam(embedder.parameters(), lr=args.lr)

    # Data loader
    loader, basin_data = build_loader_from_args(args, args.device)

    env = CVRPEnv(problem_size=args.problem_size, device=args.device)

    # Train loop
    global_step = 0
    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(
            tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch"), start=1
        ):
            
            env.load(batch.depot_xy, batch.node_xy_demand, basin_data.basin_info)

            loss = train_one_batch(embedder, optimizer, batch, env, args.device, args.temperature)

            total_loss += loss
            n_batches += 1

            if wb_run is not None:
                wandb.log(
                    {
                        "train/step_loss": loss,
                        "train/epoch": epoch + 1,
                        "train/batch": batch_idx,
                        "train/instance_idx": batch.instance_idx,
                    },
                    step=global_step,
                )
            if batch_idx % 100 == 0:
                print(
                    f"[Epoch {epoch+1}/{args.epochs}] batch {batch_idx} "
                    f"(instance={batch.instance_idx}), loss={loss:.6f}"
                )
            global_step += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs} loss={avg_loss:.6f}")
        if wb_run is not None:
            wandb.log({"train/epoch_loss": avg_loss}, step=global_step)

        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt")
            torch.save({"encoder_state": embedder.encoder.state_dict(), "epoch": epoch + 1}, ckpt_path)
            print(f"Saved {ckpt_path}")
            if wb_run is not None:
                wandb.log({"artifact/checkpoint": ckpt_path}, step=global_step)

    if wb_run is not None:
        wandb.finish()


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

    # training related
    parser.add_argument("--neg_mode", type=str, choices=["distant", "masked_in_batch"], default="distant", help="Negative sampling: distant=use distant_basins in batch; masked_in_batch=use sign mask (inferred from instance dir).")
    parser.add_argument("--max_negatives", type=int, default=64, help="Max distant basins per anchor (only when neg_mode=distant); ignored for masked_in_batch (negatives = in-batch only, set by batch_size).")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for InfoNCE loss.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")

    # save related
    parser.add_argument("--save", type=str, default="out", help="Root dir for checkpoints; a subdir named like run name will be created (default: out).")
    parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoint every N epochs.")

    # wandb related
    parser.add_argument("--note", type=str, default=None, help='If set, wandb run name = date_index_note (e.g. "20260203_222410_0-49_myrun").')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable_wandb", dest="disable_wandb", action="store_true", help="Disable wandb logging when set.")

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

