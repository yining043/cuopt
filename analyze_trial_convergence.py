#!/usr/bin/env python3
"""
Analyze whether intermediate solutions in a trial are predicted (by learned embeddings)
to converge to their own local optima.

For each trajectory.jsonl:
  - Each (run_id, trial_id) defines one trial.
  - The last solution of a trial (last record in that trial) is its local optimum.
  - For every solution in that trial (optionally excluding the final one), we create:
      * Positive pair: (intermediate_solution, local_optimum_of_this_trial), label = 1
      * Negative pairs: (intermediate_solution, local_optimum_of_other_trials_of_same_instance), label = 0

Given a checkpoint, we:
  - Embed both sides of each pair using SolutionEmbedder.
  - Compute L2 distance between the two embeddings.
  - Learn a scalar threshold tau that maximizes classification accuracy on this data
    (predict "yes / converge to this optimum" when distance < tau).
We then report:
  - Overall accuracy, positive accuracy, negative accuracy, ROC-AUC.
  - Especially: negative accuracy = proportion of "non-ground-truth local optima" predicted as NO.
"""

import argparse
import json
import glob
import os
import pickle
import re
import random
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from CVRPEnv import CVRPEnv
from analyze_checkpoints import (
    build_model,
    embed_solutions,
    load_trajectory_trials,
    parse_instance_index_from_trajectory_path,
)
from helper import (
    broken_pairs_ratio,
    load_instances_pkl,
    solution_flat_to_solution,
)


def load_trajectory_trials_first_k_runs(
    trajectory_path: str,
    max_runs: int = 10,
) -> List[Tuple[Any, List[Dict]]]:
    """Like load_trajectory_trials but only keeps trials from the first max_runs run_ids (by first appearance)."""
    trials: Dict[Tuple[Any, Any], List[Dict]] = {}
    run_id_order: List[Any] = []
    run_id_seen: Dict[Any, bool] = {}

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
            if run_id not in run_id_seen:
                if len(run_id_order) >= max_runs:
                    continue
                run_id_seen[run_id] = True
                run_id_order.append(run_id)
            if run_id not in run_id_order[:max_runs]:
                continue
            key = (run_id, trial_id)
            if key not in trials:
                trials[key] = []
            trials[key].append(rec)

    out: List[Tuple[Any, List[Dict]]] = []
    for key, recs in trials.items():
        recs_sorted = sorted(
            recs,
            key=lambda r: (r.get("global_iter", 0) or 0, r.get("local_iter", 0) or 0),
        )
        final_hash = recs_sorted[-1].get("edges_hash") if recs_sorted else None
        out.append((final_hash, recs_sorted))
    return out


def build_pair_dataset(
    trajectory_paths: List[str],
    min_points_per_trial: int = 5,
    max_neg_per_intermediate: int = 3,
    only_strict_intermediate: bool = True,
    seed: int = 42,
    num_hard_neg_per_intermediate: int = 1,
    num_easy_neg_per_intermediate: int = 2,
    hard_neg_top_k: int = 5,
    max_runs_per_instance: Optional[int] = 10,
) -> List[Dict[str, Any]]:
    """Build (inter_solution, opt_solution, label, progress) pairs from trajectory.jsonl."""
    rng = random.Random(seed)
    all_pairs: List[Dict[str, Any]] = []
    n_paths = len(trajectory_paths)
    load_fn = (lambda p: load_trajectory_trials_first_k_runs(p, max_runs_per_instance)) if (max_runs_per_instance is not None and max_runs_per_instance > 0) else load_trajectory_trials

    for path_i, traj_path in enumerate(trajectory_paths):
        if not os.path.isfile(traj_path):
            print(f"  [build_pairs {path_i+1}/{n_paths}] skip (not found): {traj_path}")
            continue
        print(f"  [build_pairs {path_i+1}/{n_paths}] load {os.path.basename(os.path.dirname(traj_path))}" + (f" (first {max_runs_per_instance} runs)" if (max_runs_per_instance and max_runs_per_instance > 0) else " (all runs)") + " ...", flush=True)
        trials_with_hash = load_fn(traj_path)
        if not trials_with_hash:
            continue

        inst_idx = parse_instance_index_from_trajectory_path(traj_path)
        # Collect each trial's local optimum (solution_flat) and full sequence
        trial_opt_solutions: List[List[int]] = []
        trial_opt_routes: List[List[int]] = []
        trial_sequences: List[List[Dict]] = []

        for _, recs in trials_with_hash:
            # Keep only trials with enough solutions
            valid_recs = [r for r in recs if r.get("solution_flat") is not None]
            if len(valid_recs) < min_points_per_trial:
                continue

            # Local optimum = last record with solution_flat
            last_rec = valid_recs[-1]
            opt_sol_flat = last_rec["solution_flat"]
            trial_opt_solutions.append(opt_sol_flat)
            # Convert to route-like sequence (with depot=0 separators)
            opt_route = solution_flat_to_solution(opt_sol_flat)
            trial_opt_routes.append(opt_route)
            trial_sequences.append(valid_recs)

        n_trials = len(trial_opt_solutions)
        if n_trials <= 1:
            continue

        before = len(all_pairs)
        # Pre-compute broken-pairs distances between local optima of this instance
        broken_mat = [[0.0 for _ in range(n_trials)] for _ in range(n_trials)]
        for i in range(n_trials):
            for j in range(n_trials):
                if i == j:
                    continue
                broken_mat[i][j] = broken_pairs_ratio(
                    trial_opt_routes[i], trial_opt_routes[j]
                )

        # Split negative budget into hard / easy parts
        total_budget = max_neg_per_intermediate
        hard_budget = min(num_hard_neg_per_intermediate, total_budget)
        easy_budget = max(0, min(num_easy_neg_per_intermediate, total_budget - hard_budget))

        # Now create (intermediate, optimum) pairs for this instance
        for t_idx, seq in enumerate(trial_sequences):
            opt_flat_this = trial_opt_solutions[t_idx]

            other_indices = [j for j in range(n_trials) if j != t_idx]
            # Hard negative candidates = nearest in broken-pairs space
            if hard_neg_top_k > 0:
                sorted_others = sorted(
                    other_indices,
                    key=lambda j: broken_mat[t_idx][j],
                )
                hard_candidates = sorted_others[: min(hard_neg_top_k, len(sorted_others))]
            else:
                hard_candidates = []
            easy_candidates = [j for j in other_indices if j not in hard_candidates]

            seq_len = len(seq)

            for k, rec in enumerate(seq):
                # Optionally exclude the final local optimum itself as "intermediate"
                if only_strict_intermediate and k == seq_len - 1:
                    continue

                sol_flat = rec.get("solution_flat")
                if sol_flat is None:
                    continue

                # Normalized progress in this trial; step = absolute iteration index
                progress = float(k) / float(max(seq_len - 1, 1))
                step = k

                # Positive pair: (intermediate, its own local optimum)
                all_pairs.append(
                    {
                        "instance_index": inst_idx,
                        "inter_solution_flat": sol_flat,
                        "opt_solution_flat": opt_flat_this,
                        "label": 1,
                        "progress": progress,
                        "step": step,
                    }
                )

                # Hard negatives: same intermediate + "nearby" other-trial local optima
                n_hard = min(hard_budget, len(hard_candidates))
                if n_hard > 0:
                    chosen_hard = rng.sample(hard_candidates, n_hard)
                    for idx_h in chosen_hard:
                        opt_flat_other = trial_opt_solutions[idx_h]
                        all_pairs.append(
                            {
                                "instance_index": inst_idx,
                                "inter_solution_flat": sol_flat,
                                "opt_solution_flat": opt_flat_other,
                                "label": 0,
                                "progress": progress,
                                "step": step,
                            }
                        )

                # Easy negatives: random other-trial local optima (far or arbitrary)
                n_easy = min(easy_budget, len(easy_candidates))
                if n_easy > 0:
                    chosen_easy = rng.sample(easy_candidates, n_easy)
                    for idx_e in chosen_easy:
                        opt_flat_other = trial_opt_solutions[idx_e]
                        all_pairs.append(
                            {
                                "instance_index": inst_idx,
                                "inter_solution_flat": sol_flat,
                                "opt_solution_flat": opt_flat_other,
                                "label": 0,
                                "progress": progress,
                                "step": step,
                            }
                        )
        print(f"  [build_pairs {path_i+1}/{n_paths}] instance {inst_idx}: {n_trials} trials -> +{len(all_pairs) - before} pairs (total {len(all_pairs)})", flush=True)

    return all_pairs


def embed_pairs_for_checkpoint(
    ckpt_path: str,
    args: argparse.Namespace,
    device: torch.device,
    pairs: List[Dict[str, Any]],
    instance_data_by_idx: Dict[int, Dict],
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Embed pairs, return (d_embed, d_struct, y, progress, step) arrays."""
    n_pairs = len(pairs)
    ckpt = torch.load(ckpt_path, map_location=device)
    embedder = build_model(args, device, encoder_state=ckpt.get("encoder_state"))
    embedder.encoder.load_state_dict(ckpt["encoder_state"])
    embedder.eval()

    env = CVRPEnv(problem_size=args.problem_size, device=device)
    basin_cache: Dict[str, dict] = {}

    by_inst: Dict[int, List[Dict[str, Any]]] = {}
    for rec in pairs:
        by_inst.setdefault(int(rec["instance_index"]), []).append(rec)

    d_embed_list: List[float] = []
    d_struct_list: List[float] = []
    y_list: List[int] = []
    prog_list: List[float] = []
    step_list: List[int] = []
    done = 0

    with torch.no_grad():
        for inst_idx, inst_pairs in by_inst.items():
            inst = instance_data_by_idx[inst_idx]
            env.load(inst["depot_xy"].to(device), inst["node_xy_demand"].to(device), basin_cache)

            # Unique solutions per instance -> embed once per unique (inter or opt)
            inter_flat_to_idx: Dict[Tuple, int] = {}
            opt_flat_to_idx: Dict[Tuple, int] = {}
            unique_inter_flat: List[List[int]] = []
            unique_opt_flat: List[List[int]] = []
            for rec in inst_pairs:
                if rec["inter_solution_flat"] is None:
                    continue
                k_inter = tuple(rec["inter_solution_flat"])
                if k_inter not in inter_flat_to_idx:
                    inter_flat_to_idx[k_inter] = len(unique_inter_flat)
                    unique_inter_flat.append(rec["inter_solution_flat"])
                k_opt = tuple(rec["opt_solution_flat"])
                if k_opt not in opt_flat_to_idx:
                    opt_flat_to_idx[k_opt] = len(unique_opt_flat)
                    unique_opt_flat.append(rec["opt_solution_flat"])

            unique_inter_sol = []
            for flat in unique_inter_flat:
                s = solution_flat_to_solution(flat)
                if s:
                    unique_inter_sol.append(s)
                else:
                    unique_inter_sol.append(None)
            unique_opt_sol = []
            for flat in unique_opt_flat:
                s = solution_flat_to_solution(flat)
                if s:
                    unique_opt_sol.append(s)
                else:
                    unique_opt_sol.append(None)

            n_uniq_inter = sum(1 for s in unique_inter_sol if s is not None)
            n_uniq_opt = sum(1 for s in unique_opt_sol if s is not None)
            print(f"    embed instance {inst_idx}: {len(inst_pairs)} pairs -> {n_uniq_inter} unique inter, {n_uniq_opt} unique opt ...", flush=True)

            # Embed unique inter (batched)
            emb_inter_arr: List[torch.Tensor] = []
            for start in range(0, len(unique_inter_sol), batch_size):
                batch = [s for s in unique_inter_sol[start:start + batch_size] if s is not None]
                if not batch:
                    continue
                emb = embed_solutions(embedder, batch, env, basin_cache)
                emb_inter_arr.append(emb)
            emb_inter_all = torch.cat(emb_inter_arr, dim=0) if emb_inter_arr else torch.empty(0, 1, device=device)
            # Re-index: valid inter index -> row in emb_inter_all (skip None)
            inter_idx_to_row: List[int] = []
            row = 0
            for s in unique_inter_sol:
                if s is not None:
                    inter_idx_to_row.append(row)
                    row += 1
                else:
                    inter_idx_to_row.append(-1)

            emb_opt_arr: List[torch.Tensor] = []
            for start in range(0, len(unique_opt_sol), batch_size):
                batch = [s for s in unique_opt_sol[start:start + batch_size] if s is not None]
                if not batch:
                    continue
                emb = embed_solutions(embedder, batch, env, basin_cache)
                emb_opt_arr.append(emb)
            emb_opt_all = torch.cat(emb_opt_arr, dim=0) if emb_opt_arr else torch.empty(0, 1, device=device)
            opt_idx_to_row: List[int] = []
            row = 0
            for s in unique_opt_sol:
                if s is not None:
                    opt_idx_to_row.append(row)
                    row += 1
                else:
                    opt_idx_to_row.append(-1)

            struct_cache: Dict[Tuple[int, int], float] = {}
            ri_list: List[int] = []
            rj_list: List[int] = []
            pair_labels: List[int] = []
            pair_prog: List[float] = []
            pair_step: List[int] = []

            for rec in inst_pairs:
                i = inter_flat_to_idx.get(tuple(rec["inter_solution_flat"]), -1)
                j = opt_flat_to_idx.get(tuple(rec["opt_solution_flat"]), -1)
                if i < 0 or j < 0:
                    continue
                ri, rj = inter_idx_to_row[i], opt_idx_to_row[j]
                if ri < 0 or rj < 0:
                    continue
                ri_list.append(ri)
                rj_list.append(rj)
                pair_labels.append(int(rec["label"]))
                pair_prog.append(float(rec["progress"]))
                pair_step.append(int(rec.get("step", 0)))
                key = (i, j)
                if key not in struct_cache:
                    sol_i, sol_j = unique_inter_sol[i], unique_opt_sol[j]
                    struct_cache[key] = broken_pairs_ratio(sol_i, sol_j) if sol_i and sol_j else 0.0
                d_struct_list.append(struct_cache[key])

            if ri_list:
                ri_t = torch.tensor(ri_list, device=device)
                rj_t = torch.tensor(rj_list, device=device)
                d_embed_list.extend(F.pairwise_distance(emb_inter_all[ri_t], emb_opt_all[rj_t], p=2).cpu().tolist())
                y_list.extend(pair_labels)
                prog_list.extend(pair_prog)
                step_list.extend(pair_step)
                done += len(ri_list)
            print(f"    instance {inst_idx} done (running total {done}/{n_pairs})", flush=True)

    d_embed_arr = np.asarray(d_embed_list, dtype=np.float64)
    d_struct_arr = np.asarray(d_struct_list, dtype=np.float64)
    y_arr = np.asarray(y_list, dtype=np.int32)
    prog_arr = np.asarray(prog_list, dtype=np.float64)
    step_arr = np.asarray(step_list, dtype=np.int32)
    return d_embed_arr, d_struct_arr, y_arr, prog_arr, step_arr


def choose_best_threshold(d: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Choose tau that maximizes accuracy for (d < tau) -> 1. Returns (best_tau, metrics)."""
    taus = np.percentile(d, np.linspace(0, 100, 201))
    best_acc, best_tau, best_pos_acc, best_neg_acc = -1.0, float(taus[0]), 0.0, 0.0
    mask_pos, mask_neg = (y == 1), (y == 0)
    n_pos, n_neg = int(mask_pos.sum()), int(mask_neg.sum())

    for tau in taus:
        y_pred = (d < tau).astype(np.int32)
        correct = (y_pred == y)
        acc = float(correct.mean())
        pos_acc = (correct & mask_pos).sum() / n_pos if n_pos else 0.0
        neg_acc = (correct & mask_neg).sum() / n_neg if n_neg else 0.0
        if acc > best_acc:
            best_acc, best_tau, best_pos_acc, best_neg_acc = acc, float(tau), pos_acc, neg_acc

    return best_tau, {
        "overall_acc": best_acc,
        "pos_acc": best_pos_acc,
        "neg_acc": best_neg_acc,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def compute_roc_auc(d: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC with -distance as score (smaller distance = more positive)."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, -d))


def _acc_per_bin_from_pred(
    y: np.ndarray,
    pred: np.ndarray,
    prog: np.ndarray,
    step: np.ndarray,
    prog_edges: np.ndarray,
    step_edges: np.ndarray,
    step_max: int,
) -> Tuple[List[Tuple[List[float], List[float]]], List[str], List[Tuple[List[float], List[float]]], List[str]]:
    """For given pred, return (pos_acc_pct, neg_acc_pct), bin_labels_pct, (pos_acc_step, neg_acc_step), bin_labels_step."""
    n_bins_pct = len(prog_edges) - 1
    pos_acc_pct, neg_acc_pct = [], []
    bin_labels_pct = []
    for i in range(n_bins_pct):
        lo, hi = prog_edges[i], prog_edges[i + 1]
        mask = (prog >= lo) & (prog < hi)
        if not np.any(mask):
            pos_acc_pct.append(0.0)
            neg_acc_pct.append(0.0)
            bin_labels_pct.append(f"{int(lo*100)}–{int(hi*100)}%")
            continue
        y_b, pred_b = y[mask], pred[mask]
        pos_mask = (y_b == 1)
        neg_mask = (y_b == 0)
        n_pos, n_neg = pos_mask.sum(), neg_mask.sum()
        pa = (pred_b.astype(np.int32) & pos_mask.astype(np.int32)).sum() / n_pos if n_pos else 0.0
        na = ((pred_b == 0) & neg_mask).sum() / n_neg if n_neg else 0.0
        pos_acc_pct.append(float(pa))
        neg_acc_pct.append(-float(na))
        bin_labels_pct.append(f"{int(lo*100)}–{int(hi*100)}%")

    n_bins_step = len(step_edges) - 1
    pos_acc_step, neg_acc_step = [], []
    bin_labels_step = []
    for i in range(n_bins_step):
        lo, hi = step_edges[i], step_edges[i + 1]
        mask = (step >= lo) & (step < hi)
        if not np.any(mask):
            pos_acc_step.append(0.0)
            neg_acc_step.append(0.0)
            bin_labels_step.append(f"{int(lo)}–{int(hi)}" if i < n_bins_step - 1 or step_max < 10 else f"{int(lo)}+")
            continue
        y_b, pred_b = y[mask], pred[mask]
        pos_mask = (y_b == 1)
        neg_mask = (y_b == 0)
        n_pos, n_neg = pos_mask.sum(), neg_mask.sum()
        pa = (pred_b.astype(np.int32) & pos_mask.astype(np.int32)).sum() / n_pos if n_pos else 0.0
        na = ((pred_b == 0) & neg_mask).sum() / n_neg if n_neg else 0.0
        pos_acc_step.append(float(pa))
        neg_acc_step.append(-float(na))
        bin_labels_step.append(
            f"{int(lo)}+" if (i == n_bins_step - 1 and step_max >= 10) else f"{int(lo)}–{int(hi)}"
        )
    return (pos_acc_pct, neg_acc_pct), bin_labels_pct, (pos_acc_step, neg_acc_step), bin_labels_step


def _bin_counts(
    y: np.ndarray,
    prog: np.ndarray,
    step: np.ndarray,
    prog_edges: np.ndarray,
    step_edges: np.ndarray,
    step_max: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Return (n_pos, n_neg) per bin for progress and for step."""
    counts_pct, counts_step = [], []
    for i in range(len(prog_edges) - 1):
        mask = (prog >= prog_edges[i]) & (prog < prog_edges[i + 1])
        n_pos = int((y[mask] == 1).sum())
        n_neg = int((y[mask] == 0).sum())
        counts_pct.append((n_pos, n_neg))
    for i in range(len(step_edges) - 1):
        mask = (step >= step_edges[i]) & (step < step_edges[i + 1])
        n_pos = int((y[mask] == 1).sum())
        n_neg = int((y[mask] == 0).sum())
        counts_step.append((n_pos, n_neg))
    return counts_pct, counts_step


def plot_pos_neg_acc_four_methods(
    y: np.ndarray,
    prog: np.ndarray,
    step: np.ndarray,
    pred_embed_thr: np.ndarray,
    pred_embed_lr: np.ndarray,
    pred_struct_thr: np.ndarray,
    pred_struct_lr: np.ndarray,
    out_path: str,
    title_prefix: str = "",
) -> None:
    """One figure per checkpoint: 4 methods (Embedding/Broken × Threshold/LogReg), 2 rows (by progress %, by step)."""
    n_bins_pct = 10
    prog_edges = np.linspace(0, 1, n_bins_pct + 1)
    step_max = int(step.max()) if step.size else 0
    step_bins = list(range(0, min(11, step_max + 2)))
    if step_max >= 10:
        step_bins.append(step_max + 1)
    step_edges = np.array(step_bins, dtype=np.float64)

    counts_pct, counts_step = _bin_counts(y, prog, step, prog_edges, step_edges, step_max)

    preds = [
        (pred_embed_thr, "Embed + Thr"),
        (pred_embed_lr, "Embed + LogReg"),
        (pred_struct_thr, "Broken + Thr"),
        (pred_struct_lr, "Broken + LogReg"),
    ]
    results = []
    for pred, _ in preds:
        (pa_pct, na_pct), labels_pct, (pa_step, na_step), labels_step = _acc_per_bin_from_pred(
            y, pred, prog, step, prog_edges, step_edges, step_max
        )
        results.append(((pa_pct, na_pct), labels_pct, (pa_step, na_step), labels_step))

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    w = 0.35
    fontsize_num = 6
    fontsize_count = 5
    for col, ((pred, name), ((pa_pct, na_pct), labels_pct, (pa_step, na_step), labels_step)) in enumerate(zip(preds, results)):
        # Row 0: by progress %
        ax_pct = axes[0, col]
        x = np.arange(len(labels_pct))
        ax_pct.bar(x - w / 2, pa_pct, width=w, label="pos_acc", color="C0")
        ax_pct.bar(x + w / 2, na_pct, width=w, label="neg_acc", color="C1")
        for i, (pv, nv) in enumerate(zip(pa_pct, na_pct)):
            ax_pct.text(i - w / 2, pv + 0.02, f"{pv:.2f}", ha="center", va="bottom", fontsize=fontsize_num)
            ax_pct.text(i + w / 2, nv - 0.02, f"{-nv:.2f}", ha="center", va="top", fontsize=fontsize_num)
        for i, (np_pos, np_neg) in enumerate(counts_pct):
            ax_pct.text(i, -0.95, f"+{np_pos}\n-{np_neg}", ha="center", va="top", fontsize=fontsize_count)
        ax_pct.axhline(0, color="black", linewidth=0.5)
        ax_pct.set_xticks(x)
        ax_pct.set_xticklabels(labels_pct, rotation=45, ha="right", fontsize=7)
        ax_pct.set_ylabel("Accuracy (pos ↑ / neg ↓)")
        ax_pct.set_title(f"{title_prefix}{name} (progress %)")
        ax_pct.set_ylim(-1.05, 1.05)
        # Row 1: by step
        ax_step = axes[1, col]
        x = np.arange(len(labels_step))
        ax_step.bar(x - w / 2, pa_step, width=w, label="pos_acc", color="C0")
        ax_step.bar(x + w / 2, na_step, width=w, label="neg_acc", color="C1")
        for i, (pv, nv) in enumerate(zip(pa_step, na_step)):
            ax_step.text(i - w / 2, pv + 0.02, f"{pv:.2f}", ha="center", va="bottom", fontsize=fontsize_num)
            ax_step.text(i + w / 2, nv - 0.02, f"{-nv:.2f}", ha="center", va="top", fontsize=fontsize_num)
        for i, (np_pos, np_neg) in enumerate(counts_step):
            ax_step.text(i, -0.95, f"+{np_pos}\n-{np_neg}", ha="center", va="top", fontsize=fontsize_count)
        ax_step.axhline(0, color="black", linewidth=0.5)
        ax_step.set_xticks(x)
        ax_step.set_xticklabels(labels_step, rotation=45, ha="right", fontsize=7)
        ax_step.set_ylabel("Accuracy (pos ↑ / neg ↓)")
        ax_step.set_xlabel("Iteration")
        ax_step.set_title(f"{title_prefix}{name} (step)")
        ax_step.set_ylim(-1.05, 1.05)
    axes[0, 0].legend(loc="best", fontsize=7)
    axes[1, 0].legend(loc="best", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def confusion_matrix_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (TN, FP, FN, TP). Label 1 = positive (converge), 0 = negative."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def plot_confusion_matrices(
    y_true: np.ndarray,
    preds_list: List[Tuple[np.ndarray, str]],
    out_path: str,
    title_prefix: str = "",
) -> None:
    """Plot 2x2 grid of confusion matrix heatmaps for 4 methods. Color scale: 0 to total sample count."""
    n_total = len(y_true)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes.flat
    for idx, (pred, name) in enumerate(preds_list):
        tn, fp, fn, tp = confusion_matrix_counts(y_true, pred)
        # Rows = True (Pos top, Neg bottom), Cols = Pred (Pos left, Neg right) -> [[TP, FN], [FP, TN]]
        cm = np.array([[tp, fn], [fp, tn]], dtype=np.float64)
        ax = axes_flat[idx]
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=n_total or 1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred Pos", "Pred Neg"], fontsize=14)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["True Pos", "True Neg"], fontsize=14)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=18)
        ax.set_title(f"{title_prefix}{name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices_by_step(
    y_true: np.ndarray,
    step: np.ndarray,
    preds_list: List[Tuple[np.ndarray, str]],
    out_path: str,
    title_prefix: str = "",
) -> None:
    """Plot confusion matrix per search-step bin: rows = step bins, cols = 4 methods, each cell = 2x2 CM."""
    step_max = int(step.max()) if step.size else 0
    step_bins = list(range(0, min(11, step_max + 2)))
    if step_max >= 10:
        step_bins.append(step_max + 1)
    step_edges = np.array(step_bins, dtype=np.float64)
    n_step_bins = len(step_edges) - 1
    step_labels = []
    for i in range(n_step_bins):
        lo, hi = step_edges[i], step_edges[i + 1]
        step_labels.append(
            f"{int(lo)}+" if (i == n_step_bins - 1 and step_max >= 10) else f"{int(lo)}–{int(hi)}"
        )
    method_names = [name for _, name in preds_list]
    n_total = len(y_true)

    fig, axes = plt.subplots(n_step_bins, 4, figsize=(14, 2.5 * n_step_bins))
    if n_step_bins == 1:
        axes = axes.reshape(1, -1)
    for row in range(n_step_bins):
        lo, hi = step_edges[row], step_edges[row + 1]
        mask = (step >= lo) & (step < hi)
        y_b = y_true[mask]
        for col, (pred, _) in enumerate(preds_list):
            pred_b = pred[mask]
            tn, fp, fn, tp = confusion_matrix_counts(y_b, pred_b)
            # Pos top & left: [[TP, FN], [FP, TN]]
            cm = np.array([[tp, fn], [fp, tn]], dtype=np.float64)
            ax = axes[row, col]
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=n_total or 1)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Pos", "Neg"], fontsize=12)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Pos", "Neg"], fontsize=12)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=14)
            if col == 0:
                ax.set_ylabel(f"step {step_labels[row]}\nn={int(mask.sum())}", fontsize=12)
            if row == 0:
                ax.set_title(f"{title_prefix}{method_names[col]}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _precision_recall_per_bin(
    y: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float, int, int]:
    """Precision and recall within a bin mask. Returns (precision, recall, n_pos, n_neg)."""
    y_b, pred_b = y[mask], pred[mask]
    tp = int(((pred_b == 1) & (y_b == 1)).sum())
    fp = int(((pred_b == 1) & (y_b == 0)).sum())
    fn = int(((pred_b == 0) & (y_b == 1)).sum())
    n_pos = int((y_b == 1).sum())
    n_neg = int((y_b == 0).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(precision), float(recall), n_pos, n_neg


def plot_precision_recall_four_methods(
    y: np.ndarray,
    prog: np.ndarray,
    step: np.ndarray,
    pred_embed_thr: np.ndarray,
    pred_embed_lr: np.ndarray,
    pred_struct_thr: np.ndarray,
    pred_struct_lr: np.ndarray,
    out_path: str,
    title_prefix: str = "",
) -> None:
    """Precision & recall by progress% (row 0) and by step (row 1), 4 columns = 4 methods."""
    n_bins_pct = 10
    prog_edges = np.linspace(0, 1, n_bins_pct + 1)
    step_max = int(step.max()) if step.size else 0
    step_bins = list(range(0, min(11, step_max + 2)))
    if step_max >= 10:
        step_bins.append(step_max + 1)
    step_edges = np.array(step_bins, dtype=np.float64)
    n_step_bins = len(step_edges) - 1

    pct_labels = [f"{int(prog_edges[i]*100)}–{int(prog_edges[i+1]*100)}%" for i in range(n_bins_pct)]
    step_labels = []
    for i in range(n_step_bins):
        lo, hi = step_edges[i], step_edges[i + 1]
        step_labels.append(f"{int(lo)}+" if (i == n_step_bins - 1 and step_max >= 10) else f"{int(lo)}–{int(hi)}")

    preds = [
        (pred_embed_thr, "Embed+Thr"),
        (pred_embed_lr, "Embed+LogReg"),
        (pred_struct_thr, "Broken+Thr"),
        (pred_struct_lr, "Broken+LogReg"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    w = 0.3
    fontsize_num = 6
    fontsize_count = 5

    for col, (pred, name) in enumerate(preds):
        # Row 0: by progress %
        prec_pct, rec_pct, npos_pct, nneg_pct = [], [], [], []
        for i in range(n_bins_pct):
            mask = (prog >= prog_edges[i]) & (prog < prog_edges[i + 1])
            if not np.any(mask):
                prec_pct.append(0.0); rec_pct.append(0.0); npos_pct.append(0); nneg_pct.append(0)
                continue
            p, r, np_, nn_ = _precision_recall_per_bin(y, pred, mask)
            prec_pct.append(p); rec_pct.append(r); npos_pct.append(np_); nneg_pct.append(nn_)

        ax = axes[0, col]
        x = np.arange(n_bins_pct)
        ax.bar(x - w / 2, prec_pct, width=w, label="Precision", color="C2")
        ax.bar(x + w / 2, rec_pct, width=w, label="Recall", color="C3")
        for i, (pv, rv) in enumerate(zip(prec_pct, rec_pct)):
            ax.text(i - w / 2, pv + 0.02, f"{pv:.2f}", ha="center", va="bottom", fontsize=fontsize_num)
            ax.text(i + w / 2, rv + 0.02, f"{rv:.2f}", ha="center", va="bottom", fontsize=fontsize_num)
        for i in range(n_bins_pct):
            ax.text(i, -0.08, f"+{npos_pct[i]}/-{nneg_pct[i]}", ha="center", va="top", fontsize=fontsize_count)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(pct_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Precision / Recall")
        ax.set_title(f"{title_prefix}{name} (progress %)", fontsize=9)
        ax.set_ylim(-0.15, 1.15)
        if col == 0:
            ax.legend(loc="upper left", fontsize=7)

        # Row 1: by step
        prec_step, rec_step, npos_step, nneg_step = [], [], [], []
        for i in range(n_step_bins):
            mask = (step >= step_edges[i]) & (step < step_edges[i + 1])
            if not np.any(mask):
                prec_step.append(0.0); rec_step.append(0.0); npos_step.append(0); nneg_step.append(0)
                continue
            p, r, np_, nn_ = _precision_recall_per_bin(y, pred, mask)
            prec_step.append(p); rec_step.append(r); npos_step.append(np_); nneg_step.append(nn_)

        ax = axes[1, col]
        x = np.arange(n_step_bins)
        ax.bar(x - w / 2, prec_step, width=w, label="Precision", color="C2")
        ax.bar(x + w / 2, rec_step, width=w, label="Recall", color="C3")
        for i, (pv, rv) in enumerate(zip(prec_step, rec_step)):
            ax.text(i - w / 2, pv + 0.02, f"{pv:.2f}", ha="center", va="bottom", fontsize=fontsize_num)
            ax.text(i + w / 2, rv + 0.02, f"{rv:.2f}", ha="center", va="bottom", fontsize=fontsize_num)
        for i in range(n_step_bins):
            ax.text(i, -0.08, f"+{npos_step[i]}/-{nneg_step[i]}", ha="center", va="top", fontsize=fontsize_count)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(step_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Precision / Recall")
        ax.set_xlabel("Iteration")
        ax.set_title(f"{title_prefix}{name} (step)", fontsize=9)
        ax.set_ylim(-0.15, 1.15)
        if col == 0:
            ax.legend(loc="upper left", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def load_basin_classifiers(path: str) -> Dict[str, Any]:
    """Load saved classifiers from analyze_trial_convergence. Returns dict with tau_embed, tau_struct, clf_embed, clf_struct, checkpoint_base.
    Usage:
      c = load_basin_classifiers("convergence/trial_convergence_s1_epoch50_classifiers.pkl")
      pred_embed_thr = (d_embed < c["tau_embed"]).astype(int)
      pred_struct_thr = (d_struct < c["tau_struct"]).astype(int)
      pred_embed_lr = c["clf_embed"].predict(d_embed.reshape(-1, 1))
      pred_struct_lr = c["clf_struct"].predict(d_struct.reshape(-1, 1))
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_metrics_for_threshold(d: np.ndarray, y: np.ndarray, tau: float) -> Dict[str, float]:
    """Accuracy / pos_acc / neg_acc for fixed tau."""
    y_pred = (d < tau).astype(np.int32)
    correct = (y_pred == y)
    mask_pos, mask_neg = (y == 1), (y == 0)
    n_pos, n_neg = int(mask_pos.sum()), int(mask_neg.sum())
    return {
        "overall_acc": float(correct.mean()),
        "pos_acc": (correct & mask_pos).sum() / n_pos if n_pos else 0.0,
        "neg_acc": (correct & mask_neg).sum() / n_neg if n_neg else 0.0,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def _logreg_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score, accuracy_score
    correct = (y_pred == y_true)
    mask_pos, mask_neg = (y_true == 1), (y_true == 0)
    n_pos, n_neg = mask_pos.sum(), mask_neg.sum()
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "pos_acc": (correct & mask_pos).sum() / n_pos if n_pos else 0.0,
        "neg_acc": (correct & mask_neg).sum() / n_neg if n_neg else 0.0,
        "auc": float(roc_auc_score(y_true, prob)),
    }


def _train_test_split_indices(y: np.ndarray, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y), dtype=np.int64)
    return train_test_split(idx, test_size=test_ratio, stratify=y, random_state=seed)


def logistic_regression_eval(
    d: np.ndarray,
    y: np.ndarray,
    name: str,
    test_ratio: float = 0.0,
    seed: int = 42,
    d_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> None:
    """Fit 1D logistic regression on distance; if d_test/y_test given use as test, else split by test_ratio or in-sample."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X = d.reshape(-1, 1)
    if d_test is not None and y_test is not None:
        X_train, y_train = X, y
        X_test = np.asarray(d_test, dtype=np.float64).reshape(-1, 1)
        y_test_arr = np.asarray(y_test, dtype=np.int32)
        train_only = True
    elif 0 < test_ratio < 1:
        X_train, X_test, y_train, y_test_arr = train_test_split(
            X, y, test_size=test_ratio, stratify=y, random_state=seed
        )
        train_only = True
    else:
        X_train, y_train = X, y
        X_test, y_test_arr = X, y
        train_only = False

    clf = LogisticRegression(solver="lbfgs", max_iter=1000).fit(X_train, y_train)
    prob_test = clf.predict_proba(X_test)[:, 1]
    y_pred_test = (prob_test >= 0.5).astype(np.int32)
    test_metrics = _logreg_metrics(y_test_arr, y_pred_test, prob_test)

    if train_only:
        prob_train = clf.predict_proba(X_train)[:, 1]
        train_metrics = _logreg_metrics(y_train, (prob_train >= 0.5).astype(np.int32), prob_train)
        print(
            f"  [{name}-LogReg] train: acc={train_metrics['acc']:.4f}, pos_acc={train_metrics['pos_acc']:.4f}, neg_acc={train_metrics['neg_acc']:.4f}, AUC={train_metrics['auc']:.4f}  |  "
            f"test: acc={test_metrics['acc']:.4f}, pos_acc={test_metrics['pos_acc']:.4f}, neg_acc={test_metrics['neg_acc']:.4f}, AUC={test_metrics['auc']:.4f} (n_train={len(y_train)}, n_test={len(y_test_arr)})"
        )
    else:
        print(
            f"  [{name}-LogReg] acc={test_metrics['acc']:.4f}, pos_acc={test_metrics['pos_acc']:.4f}, neg_acc={test_metrics['neg_acc']:.4f}, ROC-AUC={test_metrics['auc']:.4f} (in-sample, n={len(y)})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Basin predictor: predict if intermediate solution converges to given local optimum.")
    parser.add_argument("--checkpoint_dir", type=str, default="/home/jieyi/cuopt/out/20260205_095131_0-49_2stages", help="Dir with s1_epoch*.pt / s2_epoch*.pt.")
    parser.add_argument("--trajectory_paths", type=str, nargs="*", default=["/home/jieyi/cuopt/basin_datasets0/cvrp100_uniform.pkl#50/trajectory.jsonl", "/home/jieyi/cuopt/basin_datasets0/cvrp100_uniform.pkl#51/trajectory.jsonl", "/home/jieyi/cuopt/basin_datasets0/cvrp100_uniform.pkl#52/trajectory.jsonl", "/home/jieyi/cuopt/basin_datasets0/cvrp100_uniform.pkl#53/trajectory.jsonl", "/home/jieyi/cuopt/basin_datasets0/cvrp100_uniform.pkl#54/trajectory.jsonl"], help="trajectory.jsonl for (inter, opt) pairs.")
    parser.add_argument("--instance_pkl", type=str, default="/home/jieyi/cvrp100_uniform.pkl", help="CVRP instance pkl.")
    parser.add_argument("--problem_size", type=int, default=100)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--supplement_feature_dim", type=int, default=5)
    parser.add_argument("--use_l2_normalize", action="store_true", default=True, help="L2-normalize embeddings.")
    parser.add_argument("--min_points_per_trial", type=int, default=5, help="Min solutions per trial.")
    parser.add_argument("--max_runs_per_instance", type=int, default=10, help="Use only first N run_ids per trajectory (0 = all).")
    parser.add_argument("--max_neg_per_intermediate", type=int, default=3, help="Max negatives per intermediate.")
    parser.add_argument("--num_hard_neg_per_intermediate", type=int, default=1, help="Hard negs (nearest by broken-pairs) per inter.")
    parser.add_argument("--num_easy_neg_per_intermediate", type=int, default=2, help="Easy negs (random other optima) per inter.")
    parser.add_argument("--hard_neg_top_k", type=int, default=5, help="Top-K other optima as hard neg candidates.")
    parser.add_argument("--include_final_as_intermediate", action="store_true", help="Use final local optimum as intermediate too.")
    parser.add_argument("--classifier", type=str, choices=("threshold", "logistic", "both"), default="both", help="threshold / logistic / both.")
    parser.add_argument("--logreg_test_ratio", type=float, default=0.2, help="Test fraction (same for threshold & logreg; 0=no split).")
    parser.add_argument("--logreg_split_seed", type=int, default=42, help="Seed for train/test split.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    print(f">> Using device: {device}")

    only_strict_intermediate = not args.include_final_as_intermediate

    # 1) Build pair dataset once (from all given trajectory paths)
    pairs = build_pair_dataset(
        trajectory_paths=args.trajectory_paths,
        min_points_per_trial=args.min_points_per_trial,
        max_neg_per_intermediate=args.max_neg_per_intermediate,
        only_strict_intermediate=only_strict_intermediate,
        seed=args.seed,
        num_hard_neg_per_intermediate=args.num_hard_neg_per_intermediate,
        num_easy_neg_per_intermediate=args.num_easy_neg_per_intermediate,
        hard_neg_top_k=args.hard_neg_top_k,
        max_runs_per_instance=getattr(args, "max_runs_per_instance", 10),
    )
    if not pairs:
        print("No pairs constructed; check trajectory_paths and thresholds.")
        return
    print(f"Total pairs: {len(pairs)}", flush=True)

    indices = sorted({int(rec["instance_index"]) for rec in pairs})
    print(f"Load instances {indices} ...", flush=True)
    instance_list = load_instances_pkl(args.instance_pkl, device, indices, {})
    instance_data_by_idx = dict(zip(indices, instance_list))

    # 3) Iterate over checkpoints and evaluate (sort by stage then epoch number)
    ckpt_paths = glob.glob(os.path.join(args.checkpoint_dir, "*.pt"))

    def _ckpt_sort_key(p: str) -> Tuple[int, int]:
        base = os.path.splitext(os.path.basename(p))[0]
        m = re.match(r"s(\d+)_epoch(\d+)", base, re.IGNORECASE)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (0, 0)

    ckpts = sorted(ckpt_paths, key=_ckpt_sort_key)
    if not ckpts:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return

    print(f"Found {len(ckpts)} checkpoints.", flush=True)

    convergence_dir = os.path.join(args.checkpoint_dir, "convergence")
    os.makedirs(convergence_dir, exist_ok=True)
    log_path = os.path.join(convergence_dir, "trial_convergence_log.txt")

    for ckpt_i, ckpt_path in enumerate(ckpts):
        base = os.path.splitext(os.path.basename(ckpt_path))[0]
        log_lines: List[str] = []

        def log_print(msg: str) -> None:
            print(msg, flush=True)
            log_lines.append(msg)

        log_print(f"\n=== [{ckpt_i+1}/{len(ckpts)}] {base} ===")
        log_print(f"  Embedding {len(pairs)} pairs ...")
        d_embed, d_struct, y_arr, prog_arr, step_arr = embed_pairs_for_checkpoint(
            ckpt_path, args, device, pairs, instance_data_by_idx
        )
        if d_embed.size == 0:
            continue

        test_ratio = args.logreg_test_ratio
        split_seed = args.logreg_split_seed
        use_split = 0 < test_ratio < 1
        if use_split:
            train_idx, test_idx = _train_test_split_indices(y_arr, test_ratio, split_seed)
            d_embed_train, d_embed_test = d_embed[train_idx], d_embed[test_idx]
            d_struct_train, d_struct_test = d_struct[train_idx], d_struct[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]
            prog_test = prog_arr[test_idx]
            step_test = step_arr[test_idx]
            log_print(f"  Train/test: n_train={len(y_train)}, n_test={len(y_test)}")

        if args.classifier in ("threshold", "both"):
            if use_split:
                tau_embed, _ = choose_best_threshold(d_embed_train, y_train)
                tau_struct, _ = choose_best_threshold(d_struct_train, y_train)
                metrics_embed_test = compute_metrics_for_threshold(d_embed_test, y_test, tau_embed)
                metrics_struct_test = compute_metrics_for_threshold(d_struct_test, y_test, tau_struct)
                auc_embed_test = compute_roc_auc(d_embed_test, y_test)
                auc_struct_test = compute_roc_auc(d_struct_test, y_test)
                log_print(
                    f"  [Embedding] tau={tau_embed:.4f} (from train)  |  test: acc={metrics_embed_test['overall_acc']:.4f}, "
                    f"pos_acc={metrics_embed_test['pos_acc']:.4f}, neg_acc={metrics_embed_test['neg_acc']:.4f}, "
                    f"AUC={auc_embed_test:.4f} (n_test={len(y_test)})"
                )
                log_print(
                    f"  [Broken ] tau={tau_struct:.4f} (from train)  |  test: acc={metrics_struct_test['overall_acc']:.4f}, "
                    f"pos_acc={metrics_struct_test['pos_acc']:.4f}, neg_acc={metrics_struct_test['neg_acc']:.4f}, "
                    f"AUC={auc_struct_test:.4f} (n_test={len(y_test)})"
                )
                buckets = [
                    (0.0, 0.2, "0%–20%"),
                    (0.2, 0.6, "20%–60%"),
                    (0.6, 1.01, "60%–100%"),
                ]
                for lo, hi, name in buckets:
                    mask = (prog_test >= lo) & (prog_test < hi)
                    if not np.any(mask):
                        continue
                    m_embed_b = compute_metrics_for_threshold(d_embed_test[mask], y_test[mask], tau_embed)
                    m_struct_b = compute_metrics_for_threshold(d_struct_test[mask], y_test[mask], tau_struct)
                    n_b = int(mask.sum())
                    log_print(
                        f"    Progress {name} (test): "
                        f"embed_acc={m_embed_b['overall_acc']:.4f}, broken_acc={m_struct_b['overall_acc']:.4f} (n={n_b})"
                    )
            else:
                tau_embed, metrics_embed = choose_best_threshold(d_embed, y_arr)
                auc_embed = compute_roc_auc(d_embed, y_arr)
                tau_struct, metrics_struct = choose_best_threshold(d_struct, y_arr)
                auc_struct = compute_roc_auc(d_struct, y_arr)
                log_print(
                    f"  [Embedding] Best tau = {tau_embed:.4f} "
                    f"(overall_acc={metrics_embed['overall_acc']:.4f}, "
                    f"pos_acc={metrics_embed['pos_acc']:.4f}, neg_acc={metrics_embed['neg_acc']:.4f}, "
                    f"n_pos={metrics_embed['n_pos']}, n_neg={metrics_embed['n_neg']}, ROC-AUC={auc_embed:.4f})"
                )
                log_print(
                    f"  [Broken ] Best tau = {tau_struct:.4f} "
                    f"(overall_acc={metrics_struct['overall_acc']:.4f}, "
                    f"pos_acc={metrics_struct['pos_acc']:.4f}, neg_acc={metrics_struct['neg_acc']:.4f}, "
                    f"n_pos={metrics_struct['n_pos']}, n_neg={metrics_struct['n_neg']}, ROC-AUC={auc_struct:.4f})"
                )
                buckets = [
                    (0.0, 0.2, "0%–20%"),
                    (0.2, 0.6, "20%–60%"),
                    (0.6, 1.01, "60%–100%"),
                ]
                for lo, hi, name in buckets:
                    mask = (prog_arr >= lo) & (prog_arr < hi)
                    if not np.any(mask):
                        continue
                    m_embed_b = compute_metrics_for_threshold(d_embed[mask], y_arr[mask], tau_embed)
                    m_struct_b = compute_metrics_for_threshold(d_struct[mask], y_arr[mask], tau_struct)
                    n_b = m_embed_b["n_pos"] + m_embed_b["n_neg"]
                    log_print(
                        f"    Progress {name}: "
                        f"embed_acc={m_embed_b['overall_acc']:.4f}, broken_acc={m_struct_b['overall_acc']:.4f} (n={n_b})"
                    )
            log_print("  neg_acc = fraction of non-ground-truth optima correctly predicted as NO.")

        if args.classifier in ("logistic", "both"):
            if use_split:
                logistic_regression_eval(
                    d_embed_train, y_train, "Embedding",
                    d_test=d_embed_test, y_test=y_test,
                )
                logistic_regression_eval(
                    d_struct_train, y_train, "BrokenPairs",
                    d_test=d_struct_test, y_test=y_test,
                )
            else:
                logistic_regression_eval(
                    d_embed, y_arr, "Embedding",
                    test_ratio=test_ratio, seed=split_seed,
                )
                logistic_regression_eval(
                    d_struct, y_arr, "BrokenPairs",
                    test_ratio=test_ratio, seed=split_seed,
                )

        tau_embed, _ = choose_best_threshold(d_embed_train, y_train) if use_split else choose_best_threshold(d_embed, y_arr)
        tau_struct, _ = choose_best_threshold(d_struct_train, y_train) if use_split else choose_best_threshold(d_struct, y_arr)
        d_plot = d_embed_test if use_split else d_embed
        d_struct_plot = d_struct_test if use_split else d_struct
        y_plot = y_test if use_split else y_arr
        prog_plot = prog_test if use_split else prog_arr
        step_plot = step_test if use_split else step_arr
        pred_embed_thr = (d_plot < tau_embed).astype(np.int32)
        pred_struct_thr = (d_struct_plot < tau_struct).astype(np.int32)
        from sklearn.linear_model import LogisticRegression
        X_embed_train = (d_embed_train if use_split else d_embed).reshape(-1, 1)
        X_embed_plot = d_plot.reshape(-1, 1)
        y_train_fit = y_train if use_split else y_arr
        clf_embed = LogisticRegression(solver="lbfgs", max_iter=1000).fit(X_embed_train, y_train_fit)
        pred_embed_lr = (clf_embed.predict_proba(X_embed_plot)[:, 1] >= 0.5).astype(np.int32)
        X_struct_train = (d_struct_train if use_split else d_struct).reshape(-1, 1)
        X_struct_plot = d_struct_plot.reshape(-1, 1)
        clf_struct = LogisticRegression(solver="lbfgs", max_iter=1000).fit(X_struct_train, y_train_fit)
        pred_struct_lr = (clf_struct.predict_proba(X_struct_plot)[:, 1] >= 0.5).astype(np.int32)
        classifier_path = os.path.join(convergence_dir, f"trial_convergence_{base}_classifiers.pkl")
        with open(classifier_path, "wb") as f:
            pickle.dump({
                "tau_embed": float(tau_embed),
                "tau_struct": float(tau_struct),
                "clf_embed": clf_embed,
                "clf_struct": clf_struct,
                "checkpoint_base": base,
            }, f)
        log_print(f"  Classifiers saved: {classifier_path}")
        plot_path = os.path.join(convergence_dir, f"trial_convergence_{base}.png")
        plot_pos_neg_acc_four_methods(
            y_plot, prog_plot, step_plot,
            pred_embed_thr, pred_embed_lr, pred_struct_thr, pred_struct_lr,
            plot_path, title_prefix=f"{base} ",
        )
        log_print(f"  Plot saved: {plot_path}")
        pr_path = os.path.join(convergence_dir, f"trial_convergence_{base}_precision_recall.png")
        plot_precision_recall_four_methods(
            y_plot, prog_plot, step_plot,
            pred_embed_thr, pred_embed_lr, pred_struct_thr, pred_struct_lr,
            pr_path, title_prefix=f"{base} ",
        )
        log_print(f"  Precision/Recall plot saved: {pr_path}")

        preds_for_cm = [
            (pred_embed_thr, "Embed+Thr"),
            (pred_embed_lr, "Embed+LogReg"),
            (pred_struct_thr, "Broken+Thr"),
            (pred_struct_lr, "Broken+LogReg"),
        ]
        cm_path = os.path.join(convergence_dir, f"trial_convergence_{base}_confusion.png")
        plot_confusion_matrices(y_plot, preds_for_cm, cm_path, title_prefix=f"{base} ")
        log_print(f"  Confusion matrices saved: {cm_path}")
        cm_by_step_path = os.path.join(convergence_dir, f"trial_convergence_{base}_confusion_by_step.png")
        plot_confusion_matrices_by_step(y_plot, step_plot, preds_for_cm, cm_by_step_path, title_prefix=f"{base} ")
        log_print(f"  Confusion by step saved: {cm_by_step_path}")
        log_print("  Confusion (TN, FP, FN, TP) per method:")
        for pred, name in preds_for_cm:
            tn, fp, fn, tp = confusion_matrix_counts(y_plot, pred)
            log_print(f"    {name}: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()

