#!/usr/bin/env python3
"""
Evaluate how well learned embedding distance predicts basin return probability.

Two main experiments per checkpoint:
1) Distance–Probability Correlation:
   - Spearman correlation between embedding distance d(A, S) and true return probability P(S -> basin(A)).
2) Boundary Classification Accuracy:
   - Given (A, S), classify whether S can return to basin(A) using a scalar distance threshold.
   - Compare embedding distance vs broken-pairs distance vs cost difference.

Expected outcome:
 - Embedding distance should correlate strongly (negatively) with return probability,
   and should be a better decision boundary than structural (broken-pairs) or cost-based baselines.
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from CVRPEnv import CVRPEnv
from analyze_checkpoints import build_model, embed_solutions  # reuse model construction logic
from helper import load_instances_pkl, solution_flat_to_solution


def load_return_prob_data(path: str) -> List[Dict]:
    """Load JSONL: instance_index, anchor/start_solution_flat, return_prob; optional anchor_cost, start_cost."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def broken_pairs_distance(a: List[int], b: List[int]) -> float:
    """Fraction of consecutive pairs in a that are not consecutive in b."""
    if len(a) < 2:
        return 0.0
    pairs_a = set((a[i], a[i + 1]) for i in range(len(a) - 1))
    pairs_b = set((b[i], b[i + 1]) for i in range(len(b) - 1))
    return sum(1 for p in pairs_a if p not in pairs_b) / len(pairs_a)


def parse_stage_epoch(basename: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse s1_epoch10 -> (1, 10)."""
    m = re.match(r"s([12])_epoch(\d+)", basename, re.IGNORECASE)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def load_return_prob_classifiers(path: str) -> Dict[str, Any]:
    """Load saved thresholds from analyze_return_probability. Returns dict with tau_embed, tau_bp, tau_cost, boundary_prob, checkpoint_base.

    Usage:
      c = load_return_prob_classifiers("analysis_return_prob/s1_epoch50_classifiers.pkl")
      pred_embed = (d_embed < c["tau_embed"]).astype(np.int32)
      pred_bp = (d_bp < c["tau_bp"]).astype(np.int32)
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def analyze_checkpoint(
    ckpt_path: str,
    args: argparse.Namespace,
    device: torch.device,
    data: List[Dict],
    instance_data_by_idx: Dict[int, Dict],
    out_dir: str,
) -> Dict[str, Any]:
    """Correlation + boundary classification for one checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    embedder = build_model(args, device, encoder_state=ckpt.get("encoder_state"))
    embedder.encoder.load_state_dict(ckpt["encoder_state"])
    embedder.eval()

    env = CVRPEnv(problem_size=args.problem_size, device=device)
    basin_cache: Dict[str, dict] = {}

    d_embed_list: List[float] = []
    d_bp_list: List[float] = []
    d_cost_list: List[float] = []
    p_list: List[float] = []

    with torch.no_grad():
        for rec in data:
            inst_idx = int(rec["instance_index"])
            inst = instance_data_by_idx[inst_idx]
            depot_xy = inst["depot_xy"].to(device)
            node_xy_demand = inst["node_xy_demand"].to(device)
            env.load(depot_xy, node_xy_demand, basin_cache)

            sol_a = solution_flat_to_solution(rec["anchor_solution_flat"])
            sol_s = solution_flat_to_solution(rec["start_solution_flat"])

            emb_a = embed_solutions(embedder, [sol_a], env, basin_cache)
            emb_s = embed_solutions(embedder, [sol_s], env, basin_cache)
            d_embed = F.pairwise_distance(emb_a, emb_s, p=2)[0].item()

            d_bp = broken_pairs_distance(sol_a, sol_s)

            d_cost = None
            if "anchor_cost" in rec and "start_cost" in rec:
                d_cost = abs(float(rec["anchor_cost"]) - float(rec["start_cost"]))

            p = float(rec["return_prob"])

            d_embed_list.append(d_embed)
            d_bp_list.append(d_bp)
            if d_cost is not None:
                d_cost_list.append(d_cost)
            p_list.append(p)

    d_embed_arr = np.asarray(d_embed_list, dtype=np.float64)
    d_bp_arr = np.asarray(d_bp_list, dtype=np.float64)
    p_arr = np.asarray(p_list, dtype=np.float64)
    has_cost = len(d_cost_list) == len(p_list)
    d_cost_arr = np.asarray(d_cost_list, dtype=np.float64) if has_cost else None

    base = os.path.splitext(os.path.basename(ckpt_path))[0]
    stage, epoch = parse_stage_epoch(base)
    result: Dict[str, Any] = {"base": base, "stage": stage, "epoch": epoch}

    # 1) Distance–Probability Spearman correlation
    sp_embed = float(spearmanr(d_embed_arr, p_arr)[0]) if d_embed_arr.size >= 2 else 0.0
    sp_bp = float(spearmanr(d_bp_arr, p_arr)[0]) if d_bp_arr.size >= 2 else 0.0
    sp_cost = float(spearmanr(d_cost_arr, p_arr)[0]) if has_cost and d_cost_arr.size >= 2 else 0.0
    result["sp_embed"] = sp_embed
    result["sp_bp"] = sp_bp
    result["sp_cost"] = sp_cost if has_cost else None

    print(f"[{base}] Spearman(d_embed, P) = {sp_embed:.4f}")
    print(f"[{base}] Spearman(d_broken_pairs, P) = {sp_bp:.4f}")
    if has_cost:
        print(f"[{base}] Spearman(|Δcost|, P) = {sp_cost:.4f}")

    # 2) Boundary classification: P >= threshold => "can return" (positive)
    y = (p_arr >= args.boundary_prob).astype(np.int32)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    ratio_pos = n_pos / len(y) if y.size else 0.0
    print(f"[{base}] pos/neg: {n_pos} / {n_neg} (ratio pos: {ratio_pos:.2%})")

    def eval_boundary(dist: np.ndarray, name: str) -> Dict[str, Any]:
        auc = roc_auc_score(y, -dist)
        taus = np.percentile(dist, np.linspace(0, 100, 101))
        best_acc, best_tau = max(
            (accuracy_score(y, (dist < tau).astype(np.int32)), float(tau)) for tau in taus
        )
        y_pred = (dist < best_tau).astype(np.int32)
        acc_pos = float(np.mean(y_pred[y == 1])) if n_pos else 0.0
        acc_neg = float(np.mean((1 - y_pred)[y == 0])) if n_neg else 0.0
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        prec = precision_score(y, y_pred, zero_division=0.0)
        rec = recall_score(y, y_pred, zero_division=0.0)
        # PR curve: scores = -dist (higher => more likely positive)
        pr_prec, pr_rec, _ = precision_recall_curve(y, -dist)
        ap = average_precision_score(y, -dist)
        print(
            f"[{base}] {name}: AUC={auc:.4f}, acc={best_acc:.4f} (pos={acc_pos:.4f}, neg={acc_neg:.4f}) at tau={best_tau:.4f}"
        )
        print(f"[{base}] {name} confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
        print(f"[{base}] {name} precision={prec:.4f}, recall={rec:.4f}, AP={ap:.4f}")
        return {
            "auc": float(auc), "best_acc": float(best_acc),
            "acc_pos": acc_pos, "acc_neg": acc_neg,
            "cm": cm, "tau": best_tau,
            "prec": float(prec), "rec": float(rec), "ap": float(ap),
            "pr_prec": pr_prec, "pr_rec": pr_rec,
        }

    eb = eval_boundary(d_embed_arr, "embed_dist")
    bp = eval_boundary(d_bp_arr, "broken_pairs_dist")
    cm_embed, tau_embed = eb["cm"], eb["tau"]
    cm_bp, tau_bp = bp["cm"], bp["tau"]
    result["auc_embed"], result["best_acc_embed"] = eb["auc"], eb["best_acc"]
    result["acc_pos_embed"], result["acc_neg_embed"] = eb["acc_pos"], eb["acc_neg"]
    result["prec_embed"], result["rec_embed"], result["ap_embed"] = eb["prec"], eb["rec"], eb["ap"]
    result["auc_bp"], result["best_acc_bp"] = bp["auc"], bp["best_acc"]
    result["acc_pos_bp"], result["acc_neg_bp"] = bp["acc_pos"], bp["acc_neg"]
    result["prec_bp"], result["rec_bp"], result["ap_bp"] = bp["prec"], bp["rec"], bp["ap"]
    if has_cost:
        cb = eval_boundary(d_cost_arr, "abs_cost_diff")
        cm_cost, tau_cost = cb["cm"], cb["tau"]
        result["auc_cost"], result["best_acc_cost"] = cb["auc"], cb["best_acc"]
        result["acc_pos_cost"], result["acc_neg_cost"] = cb["acc_pos"], cb["acc_neg"]
        result["prec_cost"], result["rec_cost"], result["ap_cost"] = cb["prec"], cb["rec"], cb["ap"]
    else:
        result["auc_cost"] = result["best_acc_cost"] = None
        result["acc_pos_cost"] = result["acc_neg_cost"] = None
        result["prec_cost"] = result["rec_cost"] = result["ap_cost"] = None
        tau_cost = None
        cm_cost = None
        cb = None

    # Combined figure: row1 = [pos/neg bar, embed CM, bp CM, cost CM]; row2 = [dist vs return_prob, Spearman bar]
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax5 = fig.add_subplot(gs[1, 2:4])

    # Row 0: pos/neg count + 3 confusion matrices
    ax0.bar([0], [n_pos], color="C0", width=0.5, label="pos")
    ax0.bar([1], [n_neg], color="C1", width=0.5, label="neg")
    ax0.set_xticks([0, 1])
    ax0.set_xticklabels(["pos", "neg"])
    ax0.set_ylabel("Count")
    ax0.set_title(f"Sample count (pos {ratio_pos:.1%} / neg {1 - ratio_pos:.1%})")
    ax0.legend()
    for i, v in enumerate([n_pos, n_neg]):
        ax0.text(i, v + max(n_pos, n_neg) * 0.02, str(v), ha="center", va="bottom")

    def _draw_cm(ax, cm: np.ndarray, title: str) -> None:
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred neg", "pred pos"], fontsize=11)
        ax.set_yticklabels(["true neg", "true pos"], fontsize=11)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)
        ax.set_title(title, fontsize=12)

    _draw_cm(ax1, cm_embed, "embed_dist @ best τ")
    _draw_cm(ax2, cm_bp, "broken_pairs @ best τ")
    if cm_cost is not None:
        _draw_cm(ax3, cm_cost, "|Δcost| @ best τ")
    else:
        ax3.set_visible(False)

    # Row 1 left: dist vs return prob (binned mean, embed only)
    d_embed = d_embed_arr.flatten()
    n_bins = 30
    bins = np.percentile(d_embed, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        bins = np.linspace(d_embed.min(), d_embed.max(), n_bins + 1)
    bin_ix = np.searchsorted(bins[1:-1], d_embed)
    bin_means_d, bin_means_p = [], []
    for b in range(len(bins) - 1):
        mask = bin_ix == b
        if mask.sum() > 0:
            bin_means_d.append(d_embed[mask].mean())
            bin_means_p.append(p_arr[mask].mean())
    if bin_means_d:
        ax4.plot(bin_means_d, bin_means_p, "o-", color="steelblue", linewidth=2, markersize=6)
    ax4.set_xlabel("Embedding distance (bin mean)")
    ax4.set_ylabel("Return probability P")
    ax4.set_title(f"Distance vs return prob (Spearman={sp_embed:.3f})")
    ax4.grid(True, alpha=0.3)

    # Row 1 right: Spearman bar for 2 or 3 metrics
    if has_cost:
        names, sp_vals = ["embed", "broken_pairs", "|Δcost|"], [sp_embed, sp_bp, sp_cost]
    else:
        names, sp_vals = ["embed", "broken_pairs"], [sp_embed, sp_bp]
    colors = ["C0", "C1", "C2"]
    x_pos = np.arange(len(names))
    ax5.bar(x_pos, sp_vals, color=colors[: len(names)])
    ax5.axhline(0, color="gray", linewidth=0.8)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(names)
    ax5.set_ylabel("Spearman(d, P)")
    ax5.set_title("Spearman correlation with return probability")
    for i, v in enumerate(sp_vals):
        ax5.text(i, v + 0.02 if v >= 0 else v - 0.05, f"{v:.3f}", ha="center", va="bottom")

    fig.suptitle(f"[{base}] Pos/neg, confusion matrices, dist vs P, Spearman")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{base}_pos_neg_confusion.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[{base}] Saved combined figure to {base}_pos_neg_confusion.png")

    # PR curve (left) + Precision/Recall bar (right) in one figure
    fig_pr, (ax_pr, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    # Left: PR curves
    ax_pr.plot(eb["pr_rec"], eb["pr_prec"], "-", color="C0", label=f"embed (AP={eb['ap']:.3f})")
    ax_pr.plot(bp["pr_rec"], bp["pr_prec"], "-", color="C1", label=f"broken_pairs (AP={bp['ap']:.3f})")
    if cb is not None:
        ax_pr.plot(cb["pr_rec"], cb["pr_prec"], "-", color="C2", label=f"|Δcost| (AP={cb['ap']:.3f})")
    ax_pr.plot(eb["rec"], eb["prec"], "o", color="C0", markersize=8)
    ax_pr.plot(bp["rec"], bp["prec"], "o", color="C1", markersize=8)
    if cb is not None:
        ax_pr.plot(cb["rec"], cb["prec"], "o", color="C2", markersize=8)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall curve")
    ax_pr.legend()
    ax_pr.grid(True, alpha=0.3)
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.02)
    # Right: Precision & Recall bar (color = metric, hatch = prec/rec)
    metric_colors = ["C0", "C1", "C2"]
    prec_vals = [eb["prec"], bp["prec"]] + ([cb["prec"]] if has_cost else [])
    rec_vals = [eb["rec"], bp["rec"]] + ([cb["rec"]] if has_cost else [])
    bar_names = ["embed", "broken_pairs"] + (["|Δcost|"] if has_cost else [])
    x_idx = np.arange(len(bar_names))
    w = 0.35
    for i in range(len(bar_names)):
        b_p = ax_bar.bar(x_idx[i] - w / 2, prec_vals[i], w, color=metric_colors[i], edgecolor="black", linewidth=0.5)
        b_r = ax_bar.bar(x_idx[i] + w / 2, rec_vals[i], w, color=metric_colors[i], edgecolor="black", linewidth=0.5, hatch="//", alpha=0.7)
        ax_bar.text(x_idx[i] - w / 2, prec_vals[i] + 0.02, f"{prec_vals[i]:.3f}", ha="center", va="bottom", fontsize=9)
        ax_bar.text(x_idx[i] + w / 2, rec_vals[i] + 0.02, f"{rec_vals[i]:.3f}", ha="center", va="bottom", fontsize=9)
    # Legend: solid = Precision, hatched = Recall
    from matplotlib.patches import Patch
    ax_bar.legend(handles=[Patch(facecolor="gray", label="Precision"), Patch(facecolor="gray", hatch="//", alpha=0.7, label="Recall")])
    ax_bar.set_xticks(x_idx)
    ax_bar.set_xticklabels(bar_names)
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("Precision & Recall @ best τ")
    ax_bar.set_ylim(0, 1.15)
    ax_bar.grid(True, alpha=0.3, axis="y")
    fig_pr.suptitle(f"[{base}] PR curve & Precision/Recall")
    plt.tight_layout()
    pr_path = os.path.join(out_dir, f"{base}_pr_curve.png")
    plt.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[{base}] Saved PR curve + bar to {pr_path}")

    classifier_path = os.path.join(out_dir, f"{base}_classifiers.pkl")
    with open(classifier_path, "wb") as f:
        pickle.dump(
            {
                "tau_embed": tau_embed,
                "tau_bp": tau_bp,
                "tau_cost": tau_cost,
                "boundary_prob": args.boundary_prob,
                "checkpoint_base": base,
            },
            f,
        )
    print(f"[{base}] Classifiers saved: {classifier_path}")
    return result


def plot_metrics_over_epochs(
    results: List[Dict[str, Any]],
    out_dir: str,
) -> None:
    """Plot Spearman, AUC, best_acc vs epoch per stage."""
    valid = [r for r in results if r.get("stage") in (1, 2) and r.get("epoch") is not None]
    by_stage: Dict[int, List[Dict[str, Any]]] = {1: [], 2: []}
    for r in valid:
        by_stage[r["stage"]].append(r)
    for s in (1, 2):
        by_stage[s].sort(key=lambda x: x["epoch"])

    has_cost = any(r.get("sp_cost") is not None for r in valid)

    # Exp1: Spearman(d_embed, d_bp, d_cost) vs epoch — one figure per stage, 3 lines together
    for stage in (1, 2):
        rows = by_stage[stage]
        if not rows:
            continue
        epochs = [r["epoch"] for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, [r["sp_embed"] for r in rows], "o-", label="Spearman(d_embed, P)", color="C0")
        ax.plot(epochs, [r["sp_bp"] for r in rows], "s-", label="Spearman(d_broken_pairs, P)", color="C1")
        if has_cost:
            ax.plot(epochs, [r["sp_cost"] for r in rows], "^-", label="Spearman(|Δcost|, P)", color="C2")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Spearman correlation")
        ax.set_title(f"Stage {stage}: Distance–probability correlation vs epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, f"exp1_spearman_vs_epoch_stage{stage}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")

    # Exp2: one metric per figure, one figure per stage — AUC (3 lines), Best accuracy (3 lines)
    for stage in (1, 2):
        rows = by_stage[stage]
        rows = [r for r in rows if "auc_embed" in r]  # only where boundary was computed
        if not rows:
            continue
        epochs = [r["epoch"] for r in rows]

        # Combined: row1 = [Best acc, pos/neg×3], row2 = [Precision, Recall, AP]
        n_cols = 4 if has_cost else 3
        fig = plt.figure(figsize=(5 * n_cols, 8))
        gs = fig.add_gridspec(2, n_cols)

        # Row 1: Best accuracy + pos/neg bars
        ax_acc = fig.add_subplot(gs[0, 0])
        ax_acc.plot(epochs, [r["best_acc_embed"] for r in rows], "o-", label="embed", color="C0")
        ax_acc.plot(epochs, [r["best_acc_bp"] for r in rows], "s-", label="broken_pairs", color="C1")
        if has_cost:
            ax_acc.plot(epochs, [r["best_acc_cost"] for r in rows], "^-", label="|Δcost|", color="C2")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Best accuracy")
        ax_acc.set_title("Best accuracy")
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

        def _bar_pos_neg(ax, acc_pos: float, acc_neg: float, color: str):
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.bar([0], [acc_pos], color=color, width=0.5, label="pos")
            ax.bar([1], [-acc_neg], color=color, width=0.5, alpha=0.7, label="neg")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["pos", "neg"])
            ax.set_ylabel("Accuracy")
            ax.set_ylim(-1.05, 1.05)
            ax.text(0, acc_pos + 0.03, f"{acc_pos:.3f}", ha="center", va="bottom", fontsize=9)
            ax.text(1, -acc_neg - 0.03, f"{acc_neg:.3f}", ha="center", va="top", fontsize=9)

        r_last = rows[-1]
        ax_pn1 = fig.add_subplot(gs[0, 1])
        ax_pn1.set_title("embed_dist (pos/neg)")
        _bar_pos_neg(ax_pn1, r_last["acc_pos_embed"], r_last["acc_neg_embed"], "C0")
        ax_pn2 = fig.add_subplot(gs[0, 2])
        ax_pn2.set_title("broken_pairs_dist (pos/neg)")
        _bar_pos_neg(ax_pn2, r_last["acc_pos_bp"], r_last["acc_neg_bp"], "C1")
        if has_cost:
            ax_pn3 = fig.add_subplot(gs[0, 3])
            ax_pn3.set_title("|Δcost| (pos/neg)")
            _bar_pos_neg(ax_pn3, r_last["acc_pos_cost"], r_last["acc_neg_cost"], "C2")

        # Row 2: Precision, Recall, AP vs epoch
        ax_p = fig.add_subplot(gs[1, 0])
        ax_p.plot(epochs, [r["prec_embed"] for r in rows], "o-", label="embed", color="C0")
        ax_p.plot(epochs, [r["prec_bp"] for r in rows], "s-", label="broken_pairs", color="C1")
        if has_cost:
            ax_p.plot(epochs, [r["prec_cost"] for r in rows], "^-", label="|Δcost|", color="C2")
        ax_p.set_xlabel("Epoch")
        ax_p.set_ylabel("Precision")
        ax_p.set_title("Precision @ best τ")
        ax_p.legend()
        ax_p.grid(True, alpha=0.3)

        ax_r = fig.add_subplot(gs[1, 1])
        ax_r.plot(epochs, [r["rec_embed"] for r in rows], "o-", label="embed", color="C0")
        ax_r.plot(epochs, [r["rec_bp"] for r in rows], "s-", label="broken_pairs", color="C1")
        if has_cost:
            ax_r.plot(epochs, [r["rec_cost"] for r in rows], "^-", label="|Δcost|", color="C2")
        ax_r.set_xlabel("Epoch")
        ax_r.set_ylabel("Recall")
        ax_r.set_title("Recall @ best τ")
        ax_r.legend()
        ax_r.grid(True, alpha=0.3)

        ax_ap = fig.add_subplot(gs[1, 2])
        ax_ap.plot(epochs, [r["ap_embed"] for r in rows], "o-", label="embed", color="C0")
        ax_ap.plot(epochs, [r["ap_bp"] for r in rows], "s-", label="broken_pairs", color="C1")
        if has_cost:
            ax_ap.plot(epochs, [r["ap_cost"] for r in rows], "^-", label="|Δcost|", color="C2")
        ax_ap.set_xlabel("Epoch")
        ax_ap.set_ylabel("Average Precision")
        ax_ap.set_title("AP vs epoch")
        ax_ap.legend()
        ax_ap.grid(True, alpha=0.3)

        # Hide extra cell in row2 if has_cost (4 cols but only 3 plots in row2)
        if has_cost:
            fig.add_subplot(gs[1, 3]).set_visible(False)

        fig.suptitle(f"Stage {stage}: Boundary classification metrics vs epoch")
        plt.tight_layout()
        path = os.path.join(out_dir, f"exp2_metrics_vs_epoch_stage{stage}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")


class _Tee:
    """Write to multiple file-like objects (e.g. stdout + log file)."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze relation between embedding distance and basin return probability.")
    parser.add_argument("--checkpoint_dir", type=str, default="/home/jieyi/cuopt/out/20260205_095131_0-49_2stages", help="Directory containing s1_epoch*.pt / s2_epoch*.pt checkpoints."); parser.add_argument("--return_prob_data", type=str, required=True, help="JSONL file with fields: instance_index, anchor_solution_flat, start_solution_flat, return_prob."); parser.add_argument("--instance_pkl", type=str, default="/home/jieyi/cvrp100_uniform.pkl", help="Path to NeuOpt-style CVRP instance pkl.")
    parser.add_argument("--problem_size", type=int, default=100)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--supplement_feature_dim", type=int, default=5)
    parser.add_argument("--use_l2_normalize", action=argparse.BooleanOptionalAction, default=True, help="Enable L2 normalization in SolutionEmbedder.forward during analysis. (default: True)")
    parser.add_argument("--boundary_prob", type=float, default=0.5, help="Threshold on return_prob for boundary classification (>= boundary_prob => inside)."); parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    out_dir = os.path.join(args.checkpoint_dir, "analysis_return_prob")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")

    device = torch.device("cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(f"\n--- {datetime.now().isoformat()} ---\n")
        log_f.flush()
        old_stdout = sys.stdout
        sys.stdout = _Tee(old_stdout, log_f)
        try:
            print(f">> Using device: {device}")

            data = load_return_prob_data(args.return_prob_data)
            if not data:
                return

            indices = sorted({int(rec["instance_index"]) for rec in data})
            instance_list = load_instances_pkl(args.instance_pkl, device, indices, {})
            instance_data_by_idx = dict(zip(indices, instance_list))

            ckpt_paths = glob.glob(os.path.join(args.checkpoint_dir, "*.pt"))
            if not ckpt_paths:
                return

            def _ckpt_order(p: str) -> Tuple[int, int]:
                base = os.path.splitext(os.path.basename(p))[0]
                s, e = parse_stage_epoch(base)
                return (s if s is not None else 0, e if e is not None else 0)

            ckpts = sorted(ckpt_paths, key=_ckpt_order)
            print(f"Found {len(ckpts)} checkpoints (by stage, epoch).")
            all_results: List[Dict[str, Any]] = []
            for ckpt_path in ckpts:
                print(f"\n=== Analyzing {ckpt_path} ===")
                res = analyze_checkpoint(ckpt_path, args, device, data, instance_data_by_idx, out_dir)
                all_results.append(res)

            plot_metrics_over_epochs(all_results, out_dir)
        finally:
            sys.stdout = old_stdout


if __name__ == "__main__":
    main()

