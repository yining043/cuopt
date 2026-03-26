#!/usr/bin/env python3
"""Cross-task evaluation matrix for additivity analysis.

Given multiple checkpoints, evaluate each model on:
  - T1: S1 fixed validation (anchor-neighbour-distant)
  - T2: S2 fixed validation (anchor-positive-negative)
  - T3: S3 fixed validation (masked in-batch InfoNCE on fixed held-out instances)

Outputs a CSV table for downstream additivity analysis.
"""

import argparse
import csv
import hashlib
import os
import pickle
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance

from CVRPEnv import CVRPEnv
from dataloader import (
    load_training_data_pairs,
    load_val_data_1a1n10d,
    load_val_data_1p1n,
    parse_instance_indices,
)
from helper import load_instances_pkl, seed_everything
from net import SolutionEmbedder
from train_basin_contrastive import embed_multi_instance, embed_solutions


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def build_embedder(args: argparse.Namespace) -> SolutionEmbedder:
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
    return SolutionEmbedder(model_params).to(args.device)


@torch.no_grad()
def eval_t1(
    embedder: SolutionEmbedder,
    env: CVRPEnv,
    records: List[Tuple[int, dict]],
    inst_by_idx: Dict[int, dict],
) -> Dict[str, float]:
    d_ap_all: List[float] = []
    d_ad_all: List[float] = []
    for idx, rec in records:
        inst = inst_by_idx[idx]
        cache: Dict[str, dict] = {}
        env.load(inst["depot_xy"], inst["node_xy_demand"], cache)
        emb_a = embed_solutions(embedder, [rec["anchor_solution"]], env, cache)
        emb_p = embed_solutions(embedder, [rec["neighbor_solution"]], env, cache)
        d_ap_all.append(F.pairwise_distance(emb_a, emb_p, p=2)[0].item())
        if rec["distant_solutions"]:
            emb_d = embed_solutions(embedder, rec["distant_solutions"], env, cache)
            d_ad = F.pairwise_distance(emb_a.expand_as(emb_d), emb_d, p=2)
            d_ad_all.extend(d_ad.cpu().tolist())
    d_ap = sum(d_ap_all) / len(d_ap_all) if d_ap_all else 0.0
    d_ad = sum(d_ad_all) / len(d_ad_all) if d_ad_all else 0.0
    ratio = d_ad / (d_ap + 1e-8) if d_ap_all and d_ad_all else 0.0
    w = float(wasserstein_distance(d_ap_all, d_ad_all)) if d_ap_all and d_ad_all else 0.0
    return {
        "t1_d_ap": d_ap,
        "t1_d_ad": d_ad,
        "t1_ratio_ad_over_ap": ratio,
        "t1_hist_wasserstein": w,
    }


@torch.no_grad()
def eval_t2(
    embedder: SolutionEmbedder,
    env: CVRPEnv,
    records: List[Tuple[int, dict]],
    inst_by_idx: Dict[int, dict],
    batch_size2: int,
) -> Dict[str, float]:
    d_ap_all: List[float] = []
    d_an_all: List[float] = []
    for i in range(0, len(records), batch_size2):
        batch = records[i : i + batch_size2]
        emb_a = embed_multi_instance(
            embedder,
            [(idx, t["anchor_solution"]) for idx, t in batch],
            inst_by_idx,
            env,
            {},
        )
        emb_p = embed_multi_instance(
            embedder,
            [(idx, t["positive_solution"]) for idx, t in batch],
            inst_by_idx,
            env,
            {},
        )
        emb_n = embed_multi_instance(
            embedder,
            [(idx, t["negative_solution"]) for idx, t in batch],
            inst_by_idx,
            env,
            {},
        )
        d_ap = F.pairwise_distance(emb_a, emb_p, p=2)
        d_an = F.pairwise_distance(emb_a, emb_n, p=2)
        d_ap_all.extend(d_ap.cpu().tolist())
        d_an_all.extend(d_an.cpu().tolist())
    d_ap = sum(d_ap_all) / len(d_ap_all) if d_ap_all else 0.0
    d_an = sum(d_an_all) / len(d_an_all) if d_an_all else 0.0
    ratio = d_an / (d_ap + 1e-8) if d_ap_all and d_an_all else 0.0
    w = float(wasserstein_distance(d_ap_all, d_an_all)) if d_ap_all and d_an_all else 0.0
    return {
        "t2_d_ap": d_ap,
        "t2_d_an": d_an,
        "t2_ratio_an_over_ap": ratio,
        "t2_hist_wasserstein": w,
    }


@torch.no_grad()
def eval_t3(
    embedder: SolutionEmbedder,
    env: CVRPEnv,
    pairs_by_inst: Dict[int, List[dict]],
    inst_by_idx: Dict[int, dict],
    batch_size3: int,
    infonce_temperature: float,
) -> Dict[str, float]:
    total_infonce = 0.0
    total_pos_sim = 0.0
    total_neg_sim = 0.0
    total_n_valid_neg = 0.0
    pos_list: List[float] = []
    neg_list: List[float] = []
    n_batches = 0

    for idx, recs in pairs_by_inst.items():
        inst = inst_by_idx[idx]
        basin_info_cache: Dict[str, dict] = {}
        env.load(inst["depot_xy"], inst["node_xy_demand"], basin_info_cache)
        for i in range(0, len(recs), batch_size3):
            batch_records = recs[i : i + batch_size3]
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
            sim = sim_raw / infonce_temperature
            logit_mask = valid_neg.clone()
            logit_mask.fill_diagonal_(True)
            sim = sim.masked_fill(~logit_mask, -1e9)
            labels = torch.arange(bsz, device=emb_a.device)
            infonce_loss = F.cross_entropy(sim, labels).item()

            n_neg_total = valid_neg.float().sum()
            neg_sims = sim_raw.masked_fill(~valid_neg, 0.0)
            batch_neg_sim = (neg_sims.sum() / n_neg_total).item() if n_neg_total > 0 else 0.0

            total_infonce += infonce_loss
            total_pos_sim += sim_raw.diag().mean().item()
            total_neg_sim += batch_neg_sim
            total_n_valid_neg += valid_neg.float().sum(dim=1).mean().item()
            n_batches += 1

            pos_list.extend(sim_raw.diag().cpu().tolist())
            if n_neg_total > 0:
                neg_list.extend(sim_raw.masked_select(valid_neg).cpu().tolist())

    n = max(n_batches, 1)
    w = float(wasserstein_distance(pos_list, neg_list)) if pos_list and neg_list else 0.0
    return {
        "t3_infonce_loss": total_infonce / n,
        "t3_pos_sim": total_pos_sim / n,
        "t3_neg_sim": total_neg_sim / n,
        "t3_n_valid_neg": total_n_valid_neg / n,
        "t3_hist_wasserstein": w,
    }


def parse_ckpt_specs(specs: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for s in specs:
        if "=" not in s:
            raise ValueError(f"--ckpt must be label=path, got: {s}")
        label, path = s.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Invalid --ckpt spec: {s}")
        out.append((label, path))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint additivity matrix on T1/T2/T3.")
    parser.add_argument("--ckpt", action="append", required=True, help="Checkpoint spec: label=/abs/or/rel/path.pt (repeatable).")
    parser.add_argument("--out_csv", type=str, default="eval_additivity_matrix.csv")
    parser.add_argument("--problem_size", type=int, default=100)
    parser.add_argument("--instance_pkl", type=str, default=os.path.join(PROJECT_ROOT, "cvrp100_uniform.pkl"))
    parser.add_argument("--instance_prefix", type=str, default="cvrp100_uniform.pkl#")
    parser.add_argument("--training_data_root", type=str, default="basin_datasets0_analyze")
    parser.add_argument("--s3_cache_dir", type=str, default="s3_data_cache")
    parser.add_argument("--s3_val_instance_indices", type=str, default="50-55")
    parser.add_argument("--s3_val_max_traj_runs", type=int, default=10)
    parser.add_argument("--certainty_threshold", type=float, default=0.8)
    parser.add_argument("--infonce_temperature", type=float, default=0.07)
    parser.add_argument("--val_data_1a1n10d", type=str, default=os.path.join(PROJECT_ROOT, "basin_datasets0_analyze", "val_data_1a1n10d.jsonl"))
    parser.add_argument("--val_data_1p1n", type=str, default=os.path.join(PROJECT_ROOT, "perturb_k1_collect", "val_data_1p1n.jsonl"))
    parser.add_argument("--batch_size2", type=int, default=128)
    parser.add_argument("--batch_size3", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # model args
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--supplement_feature_dim", type=int, default=5)
    parser.add_argument("--use_l2_normalize", action="store_true", default=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    args.device = torch.device(args.device)
    ckpt_specs = parse_ckpt_specs(args.ckpt)
    env = CVRPEnv(problem_size=args.problem_size, device=args.device)

    # Prepare T1 data once
    t1_records = load_val_data_1a1n10d(args.val_data_1a1n10d)
    t1_indices = sorted(set(idx for idx, _ in t1_records))
    t1_inst_list = load_instances_pkl(args.instance_pkl, args.device, t1_indices, {})
    t1_inst_by_idx = dict(zip(t1_indices, t1_inst_list))

    # Prepare T2 data once
    t2_records = load_val_data_1p1n(args.val_data_1p1n)
    t2_indices = sorted(set(idx for idx, _ in t2_records))
    t2_inst_list = load_instances_pkl(args.instance_pkl, args.device, t2_indices, {})
    t2_inst_by_idx = dict(zip(t2_indices, t2_inst_list))

    # Prepare T3 fixed val once (cached)
    val_indices = parse_instance_indices(args.s3_val_instance_indices) if args.s3_val_instance_indices else []
    t3_inst_list = load_instances_pkl(args.instance_pkl, args.device, val_indices, {})
    t3_inst_by_idx = dict(zip(val_indices, t3_inst_list))
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
    cache_hash = hashlib.md5(repr(sorted(val_key_dict.items())).encode()).hexdigest()[:12]
    os.makedirs(args.s3_cache_dir, exist_ok=True)
    cache_path = os.path.join(args.s3_cache_dir, f"s3_val_pairs_{cache_hash}.pkl")
    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("key") == val_key_dict:
            t3_pairs = cached["contrastive"]
        else:
            t3_pairs, _ = load_training_data_pairs(
                val_td_paths,
                certainty_threshold=args.certainty_threshold,
                seed=args.seed,
                max_runs=args.s3_val_max_traj_runs,
            )
    else:
        t3_pairs, _ = load_training_data_pairs(
            val_td_paths,
            certainty_threshold=args.certainty_threshold,
            seed=args.seed,
            max_runs=args.s3_val_max_traj_runs,
        )
        with open(cache_path, "wb") as f:
            pickle.dump({"key": val_key_dict, "contrastive": t3_pairs, "chaotic": []}, f, protocol=pickle.HIGHEST_PROTOCOL)
    t3_pairs_by_inst: Dict[int, List[dict]] = {}
    for idx, rec in t3_pairs:
        t3_pairs_by_inst.setdefault(idx, []).append(rec)

    rows: List[Dict[str, object]] = []
    for label, ckpt_path in ckpt_specs:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        embedder = build_embedder(args)
        ckpt = torch.load(ckpt_path, map_location=args.device)
        if "embedder_state" in ckpt:
            embedder.load_state_dict(ckpt["embedder_state"])
        elif "encoder_state" in ckpt:
            embedder.encoder.load_state_dict(ckpt["encoder_state"])
        else:
            raise KeyError(f"No embedder_state/encoder_state in {ckpt_path}")
        embedder.eval()

        row: Dict[str, object] = {
            "model_label": label,
            "ckpt_path": ckpt_path,
            "stage": ckpt.get("stage", ""),
            "epoch": ckpt.get("epoch", ""),
        }
        row.update(eval_t1(embedder, env, t1_records, t1_inst_by_idx))
        row.update(eval_t2(embedder, env, t2_records, t2_inst_by_idx, args.batch_size2))
        row.update(eval_t3(embedder, env, t3_pairs_by_inst, t3_inst_by_idx, args.batch_size3, args.infonce_temperature))
        rows.append(row)
        print(f"[done] {label}: T1/T2/T3 evaluated")

    fieldnames = list(rows[0].keys()) if rows else []
    out_csv = args.out_csv
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(PROJECT_ROOT, out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved matrix CSV: {out_csv}")


if __name__ == "__main__":
    main()

