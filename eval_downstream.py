#!/usr/bin/env python3
"""Downstream evaluation of trained solution embeddings.

Tasks
-----
(a) Basin Prediction Accuracy
    Given an intermediate solution, predict which basin it will fall into
    using kNN or a linear probe on the learned embedding.

(d) Landscape Probing (visualisation)
    Embed all solutions from held-out instances and project via PCA / t-SNE.
    Colour by basin label (separation) and by cost (funnel structure).

Usage
-----
    python eval_downstream.py \
        --checkpoint out/<run>/joint_epoch50.pt \
        --instance_indices 50-55 \
        --max_traj_runs 10
"""

import argparse
import hashlib
import os
import pickle
import random
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from CVRPEnv import CVRPEnv
from dataloader import load_training_data_pairs, parse_instance_indices
from helper import load_instances_pkl, seed_everything
from net import SolutionEmbedder
from train_basin_contrastive import embed_solutions


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _build_embedder(args, device) -> SolutionEmbedder:
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
        "use_l2_normalize": True,
    }
    embedder = SolutionEmbedder(model_params).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "embedder_state" in ckpt:
        embedder.load_state_dict(ckpt["embedder_state"])
    elif "encoder_state" in ckpt:
        embedder.encoder.load_state_dict(ckpt["encoder_state"])
    stage = ckpt.get("stage", "?")
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint: {args.checkpoint}  (stage={stage}, epoch={epoch})")
    embedder.eval()
    return embedder


# ---------------------------------------------------------------------------
# Batch embedding helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _embed_all(
    embedder: SolutionEmbedder,
    solutions: List[List[int]],
    env: CVRPEnv,
    cache: Dict[str, dict],
    batch_size: int = 256,
) -> np.ndarray:
    """Embed a list of solutions in batches, returns (N, D) numpy array."""
    parts = []
    for i in range(0, len(solutions), batch_size):
        emb = embed_solutions(embedder, solutions[i : i + batch_size], env, cache)
        parts.append(emb.cpu().numpy())
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Data loading (with disk cache, same key as training)
# ---------------------------------------------------------------------------

def _load_eval_data(args):
    """Load contrastive pairs for held-out instances. Returns (contrastive, chaotic)."""
    indices = parse_instance_indices(args.instance_indices)
    td_paths = [
        os.path.join(args.training_data_root, f"{args.instance_prefix}{idx}", "training_data.jsonl")
        for idx in indices
    ]
    cache_key = {
        "td_paths": sorted(td_paths),
        "certainty_threshold": args.certainty_threshold,
        "seed": args.seed,
        "max_runs": args.max_traj_runs,
    }
    cache_hash = hashlib.md5(repr(sorted(cache_key.items())).encode()).hexdigest()[:12]
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, f"eval_pairs_{cache_hash}.pkl")

    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("key") == cache_key:
            print(f"Cache hit: {cache_path}")
            return cached["contrastive"], cached["chaotic"], indices
    contrastive, chaotic = load_training_data_pairs(
        td_paths,
        certainty_threshold=args.certainty_threshold,
        seed=args.seed,
        max_runs=args.max_traj_runs,
    )
    with open(cache_path, "wb") as f:
        pickle.dump({"key": cache_key, "contrastive": contrastive, "chaotic": chaotic}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return contrastive, chaotic, indices


# ---------------------------------------------------------------------------
# Task (a): Basin Prediction Accuracy
# ---------------------------------------------------------------------------

def task_basin_prediction(embeddings: np.ndarray, labels: np.ndarray, seed: int = 2026):
    """kNN and linear-probe basin prediction on a 70/30 stratified split.

    Returns dict[method_name -> {accuracy, f1_weighted}], n_classes.
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    # Filter classes with < 2 samples (cannot stratify)
    counts = Counter(y)
    keep_mask = np.array([counts[yi] >= 2 for yi in y])
    if not keep_mask.all():
        n_drop = (~keep_mask).sum()
        print(f"  Dropping {n_drop} samples from singleton basins for stratified split")
        embeddings = embeddings[keep_mask]
        y = y[keep_mask]
        le2 = LabelEncoder()
        y = le2.fit_transform(y)
        n_classes = len(le2.classes_)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, test_idx = next(splitter.split(embeddings, y))
    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    results = {}

    for k in [1, 5, 10]:
        k_eff = min(k, len(X_train))
        knn = KNeighborsClassifier(n_neighbors=k_eff, metric="cosine")
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        results[f"kNN-{k}"] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

    lr = LogisticRegression(max_iter=2000, random_state=seed, multi_class="multinomial", solver="lbfgs")
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results["LinearProbe"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    return results, n_classes


def task_target_optimum_prediction(embeddings: np.ndarray, binary_labels: np.ndarray, seed: int = 2026):
    """Binary prediction: whether S converges to a known target optimum S*.

    Returns dict[method_name -> {accuracy, f1_weighted}], class_counts.
    """
    y = binary_labels.astype(np.int64)
    counts = Counter(y.tolist())
    if len(counts) < 2:
        raise ValueError(
            f"Binary labels contain only one class: {dict(counts)}. "
            "Please choose another target basin hash or label mode."
        )

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, test_idx = next(splitter.split(embeddings, y))
    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    results = {}
    for k in [1, 5, 10]:
        k_eff = min(k, len(X_train))
        knn = KNeighborsClassifier(n_neighbors=k_eff, metric="cosine")
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        results[f"kNN-{k}"] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

    lr = LogisticRegression(max_iter=2000, random_state=seed, solver="lbfgs")
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results["LinearProbe"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    return results, counts


# ---------------------------------------------------------------------------
# Task (d): Landscape Probing (PCA + t-SNE)
# ---------------------------------------------------------------------------

def task_landscape_probing(
    embeddings: np.ndarray,
    basin_labels: np.ndarray,
    costs: np.ndarray,
    save_dir: str,
    tag: str = "",
    max_tsne: int = 5000,
    seed: int = 2026,
):
    """PCA and t-SNE coloured by basin and by cost."""
    os.makedirs(save_dir, exist_ok=True)
    le = LabelEncoder()
    y = le.fit_transform(basin_labels)
    n_basins = len(le.classes_)

    # Assign top-k basins distinct colours, rest grey
    counts = Counter(y)
    top_k = 20
    top_labels = {l for l, _ in counts.most_common(top_k)}
    c_basin = np.array([y_i if y_i in top_labels else -1 for y_i in y])

    # ── PCA ──
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(embeddings)
    ev = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    _scatter_basin(axes[0], X_pca, c_basin, n_basins, top_k)
    axes[0].set_title(f"PCA — {n_basins} basins (top-{top_k} coloured)")
    axes[0].set_xlabel(f"PC1 ({ev[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({ev[1]:.1%})")

    if costs is not None and len(costs) > 0:
        sc = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=costs, cmap="viridis", s=6, alpha=0.5)
        plt.colorbar(sc, ax=axes[1], label="Cost")
        axes[1].set_title("PCA — Cost (funnel structure)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    plt.tight_layout()
    path_pca = os.path.join(save_dir, f"pca_{tag}.png")
    plt.savefig(path_pca, dpi=150)
    plt.close()
    print(f"  Saved {path_pca}")

    # ── t-SNE (subsample if too large) ──
    if len(embeddings) > max_tsne:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(embeddings), max_tsne, replace=False)
        X_sub, y_sub, c_sub = embeddings[idx], c_basin[idx], (costs[idx] if costs is not None else None)
    else:
        X_sub, y_sub, c_sub = embeddings, c_basin, costs

    perp = min(30, len(X_sub) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=seed, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X_sub)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    _scatter_basin(axes[0], X_tsne, y_sub, n_basins, top_k)
    axes[0].set_title(f"t-SNE — {n_basins} basins (top-{top_k} coloured)")

    if c_sub is not None and len(c_sub) > 0:
        sc = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=c_sub, cmap="viridis", s=6, alpha=0.5)
        plt.colorbar(sc, ax=axes[1], label="Cost")
        axes[1].set_title("t-SNE — Cost (funnel structure)")
    plt.tight_layout()
    path_tsne = os.path.join(save_dir, f"tsne_{tag}.png")
    plt.savefig(path_tsne, dpi=150)
    plt.close()
    print(f"  Saved {path_tsne}")

    return ev


def _scatter_basin(ax, X_2d, colour_ids, n_basins, top_k):
    """Scatter plot: top-k basins in distinct colours, rest grey."""
    mask_other = colour_ids == -1
    if mask_other.any():
        ax.scatter(X_2d[mask_other, 0], X_2d[mask_other, 1], c="lightgrey", s=4, alpha=0.3, label="other")
    mask_top = ~mask_other
    if mask_top.any():
        cmap = plt.cm.get_cmap("tab20", top_k)
        ax.scatter(X_2d[mask_top, 0], X_2d[mask_top, 1], c=colour_ids[mask_top], cmap=cmap, s=6, alpha=0.6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Downstream evaluation of trained embeddings.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt).")
    parser.add_argument("--instance_indices", type=str, default="50-55", help="Held-out instance indices (e.g. '50-55').")
    parser.add_argument("--instance_pkl", type=str, default="/home/jieyi/cvrp100_uniform.pkl")
    parser.add_argument("--instance_prefix", type=str, default="cvrp100_uniform.pkl#")
    parser.add_argument("--training_data_root", type=str, default="basin_datasets0_analyze")
    parser.add_argument("--problem_size", type=int, default=100)
    parser.add_argument("--max_traj_runs", type=int, default=10)
    parser.add_argument("--certainty_threshold", type=float, default=0.8)
    parser.add_argument("--cache_dir", type=str, default="eval_cache")
    parser.add_argument("--out_dir", type=str, default="eval_results")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=0, help="Cap total samples (0=unlimited). Useful for quick checks.")
    parser.add_argument("--seed", type=int, default=2026)
    # model arch (must match checkpoint)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--supplement_feature_dim", type=int, default=5)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--tasks", type=str, default="all", choices=["all", "predict", "target", "probe"], help="Which tasks to run.")
    parser.add_argument(
        "--target_basin_hash",
        type=str,
        default="",
        help="Known local optimum S* basin hash for binary prediction task.",
    )
    parser.add_argument(
        "--target_label_mode",
        type=str,
        default="dominant",
        choices=["dominant", "reachable"],
        help=(
            "How to define positive label y=1 for target task: "
            "dominant => positive_basin_hash == target, "
            "reachable => target in reachable_basin_hashes."
        ),
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    # ── Load model ──
    embedder = _build_embedder(args, device)

    # ── Load held-out data ──
    contrastive, chaotic, indices = _load_eval_data(args)
    print(f"Held-out data: {len(contrastive)} contrastive, {len(chaotic)} chaotic, instances={indices}")

    if args.max_samples > 0 and len(contrastive) > args.max_samples:
        rng = random.Random(args.seed)
        contrastive = rng.sample(contrastive, args.max_samples)
        print(f"  Subsampled to {args.max_samples} contrastive records")

    # ── Load instances ──
    basin_info_cache: Dict[str, dict] = {}
    instance_data_list = load_instances_pkl(args.instance_pkl, device, indices, basin_info_cache)
    instance_data_by_idx = dict(zip(indices, instance_data_list))
    env = CVRPEnv(problem_size=args.problem_size, device=device)

    # ── Embed all anchor solutions (per instance) ──
    print("\nEmbedding solutions ...")
    all_embs, all_labels, all_target_labels, all_costs = [], [], [], []
    pairs_by_inst: Dict[int, List[dict]] = {}
    for idx, rec in contrastive:
        pairs_by_inst.setdefault(idx, []).append(rec)

    for idx in sorted(pairs_by_inst.keys()):
        recs = pairs_by_inst[idx]
        inst = instance_data_by_idx[idx]
        env.load(inst["depot_xy"], inst["node_xy_demand"], basin_info_cache)
        solutions = [r["anchor_solution"] for r in recs]
        embs = _embed_all(embedder, solutions, env, basin_info_cache, batch_size=args.batch_size)
        labels = [r["positive_basin_hash"] for r in recs]
        if args.target_basin_hash:
            if args.target_label_mode == "reachable":
                target_labels = [int(args.target_basin_hash in r["reachable_basin_hashes"]) for r in recs]
            else:
                target_labels = [int(r["positive_basin_hash"] == args.target_basin_hash) for r in recs]
            all_target_labels.extend(target_labels)
        costs = [r["anchor_cost"] for r in recs]
        all_embs.append(embs)
        all_labels.extend(labels)
        all_costs.extend(costs)
        print(f"  Instance {idx}: {len(recs)} solutions, {len(set(labels))} basins")

    X = np.concatenate(all_embs, axis=0)
    y_labels = np.array(all_labels)
    y_target = np.array(all_target_labels, dtype=np.int64) if all_target_labels else None
    y_costs = np.array(all_costs, dtype=np.float32)
    print(f"Total: {X.shape[0]} samples, {len(set(all_labels))} unique basins, dim={X.shape[1]}")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_tag = os.path.splitext(os.path.basename(args.checkpoint))[0]

    # ── Task (a): Basin Prediction ──
    if args.tasks in ("all", "predict"):
        print("\n" + "=" * 60)
        print("Task (a): Basin Prediction Accuracy")
        print("=" * 60)
        results, n_cls = task_basin_prediction(X, y_labels, seed=args.seed)
        for method, metrics in results.items():
            print(f"  {method:15s}  acc={metrics['accuracy']:.4f}  f1={metrics['f1_weighted']:.4f}")
        print(f"  ({n_cls} classes after filtering)")

        # Save results
        res_path = os.path.join(args.out_dir, f"basin_pred_{ckpt_tag}.txt")
        with open(res_path, "w") as f:
            f.write(f"checkpoint: {args.checkpoint}\n")
            f.write(f"instances: {args.instance_indices}\n")
            f.write(f"samples: {X.shape[0]}, classes: {n_cls}\n\n")
            for method, metrics in results.items():
                f.write(f"{method:15s}  acc={metrics['accuracy']:.4f}  f1={metrics['f1_weighted']:.4f}\n")
        print(f"  Saved {res_path}")

    # ── Task (b): Target optimum convergence prediction (binary) ──
    if args.tasks in ("all", "target"):
        print("\n" + "=" * 60)
        print("Task (b): Predict Convergence to Known Optimum S* (binary)")
        print("=" * 60)
        if not args.target_basin_hash:
            print("  Skipped: please set --target_basin_hash for task=target/all.")
        elif y_target is None or len(y_target) != len(X):
            print("  Skipped: target labels were not prepared.")
        else:
            try:
                t_results, t_counts = task_target_optimum_prediction(X, y_target, seed=args.seed)
                pos = int(t_counts.get(1, 0))
                neg = int(t_counts.get(0, 0))
                print(
                    f"  target={args.target_basin_hash}, mode={args.target_label_mode}, "
                    f"pos={pos}, neg={neg}"
                )
                for method, metrics in t_results.items():
                    print(f"  {method:15s}  acc={metrics['accuracy']:.4f}  f1={metrics['f1_weighted']:.4f}")

                t_path = os.path.join(
                    args.out_dir,
                    f"target_pred_{args.target_label_mode}_{ckpt_tag}.txt",
                )
                with open(t_path, "w") as f:
                    f.write(f"checkpoint: {args.checkpoint}\n")
                    f.write(f"instances: {args.instance_indices}\n")
                    f.write(f"target_basin_hash: {args.target_basin_hash}\n")
                    f.write(f"target_label_mode: {args.target_label_mode}\n")
                    f.write(f"samples: {X.shape[0]}, pos: {pos}, neg: {neg}\n\n")
                    for method, metrics in t_results.items():
                        f.write(f"{method:15s}  acc={metrics['accuracy']:.4f}  f1={metrics['f1_weighted']:.4f}\n")
                print(f"  Saved {t_path}")
            except ValueError as e:
                print(f"  Skipped: {e}")

    # ── Task (d): Landscape Probing ──
    if args.tasks in ("all", "probe"):
        print("\n" + "=" * 60)
        print("Task (d): Landscape Probing (PCA / t-SNE)")
        print("=" * 60)
        probe_dir = os.path.join(args.out_dir, f"probe_{ckpt_tag}")
        ev = task_landscape_probing(X, y_labels, y_costs, save_dir=probe_dir, tag=ckpt_tag, seed=args.seed)
        print(f"  PCA explained variance: PC1={ev[0]:.1%}, PC2={ev[1]:.1%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
