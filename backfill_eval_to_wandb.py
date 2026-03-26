#!/usr/bin/env python3
"""Backfill stage validation metrics/plots to an existing W&B run.

This script replays fixed validation for historical checkpoints (typically every 5 epochs)
and logs results to the original W&B run, useful when plots were missing in old runs.

By default, it logs to original training keys for cross-run comparability.
"""

import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import wandb
from scipy.stats import wasserstein_distance

from CVRPEnv import CVRPEnv
from dataloader import load_val_data_1a1n10d, load_val_data_1p1n
from helper import load_instances_pkl, plot_distance_histogram, plot_embedding_2d, seed_everything
from net import SolutionEmbedder
from train_basin_contrastive import (
    _extract_wandb_run_id_from_checkpoint,
    _infer_wandb_run_id_from_local,
    embed_multi_instance,
    embed_solutions,
)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _extract_epoch(path: str, prefix: str) -> Optional[int]:
    m = re.search(rf"{re.escape(prefix)}_epoch(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else None


def _build_embedder(args: argparse.Namespace) -> SolutionEmbedder:
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
def eval_s1(
    embedder: SolutionEmbedder,
    env: CVRPEnv,
    records: List[Tuple[int, dict]],
    inst_by_idx: Dict[int, dict],
) -> Tuple[float, float, float, List[float], List[float]]:
    d_ap_all: List[float] = []
    d_an_all: List[float] = []
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
            d_an_all.extend(d_ad.cpu().tolist())
    mean_ap = sum(d_ap_all) / len(d_ap_all) if d_ap_all else 0.0
    mean_an = sum(d_an_all) / len(d_an_all) if d_an_all else 0.0
    ratio = mean_an / (mean_ap + 1e-8) if d_ap_all and d_an_all else 0.0
    return mean_ap, mean_an, ratio, d_ap_all, d_an_all


@torch.no_grad()
def eval_s2(
    embedder: SolutionEmbedder,
    env: CVRPEnv,
    records: List[Tuple[int, dict]],
    inst_by_idx: Dict[int, dict],
    batch_size2: int,
) -> Tuple[float, float, float, List[float], List[float]]:
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
    mean_ap = sum(d_ap_all) / len(d_ap_all) if d_ap_all else 0.0
    mean_an = sum(d_an_all) / len(d_an_all) if d_an_all else 0.0
    ratio = mean_an / (mean_ap + 1e-8) if d_ap_all and d_an_all else 0.0
    return mean_ap, mean_an, ratio, d_ap_all, d_an_all


@torch.no_grad()
def build_s1_embedding_plot(
    embedder: SolutionEmbedder,
    env: CVRPEnv,
    records: List[Tuple[int, dict]],
    inst_by_idx: Dict[int, dict],
    save_path: str,
) -> bool:
    """Build S1 val embedding PCA plot (anchor/neighbour/distant)."""
    all_emb: List[torch.Tensor] = []
    all_inst_ids: List[int] = []
    all_groups: List[int] = []
    for idx, rec in records:
        inst = inst_by_idx[idx]
        cache: Dict[str, dict] = {}
        env.load(inst["depot_xy"], inst["node_xy_demand"], cache)
        emb_a = embed_solutions(embedder, [rec["anchor_solution"]], env, cache)
        emb_p = embed_solutions(embedder, [rec["neighbor_solution"]], env, cache)
        all_emb.extend([emb_a[0], emb_p[0]])
        all_inst_ids.extend([idx, idx])
        all_groups.extend([0, 1])  # 0=anchor, 1=neighbour
        if rec["distant_solutions"]:
            emb_d = embed_solutions(embedder, rec["distant_solutions"], env, cache)
            for j in range(emb_d.size(0)):
                all_emb.append(emb_d[j])
                all_inst_ids.append(idx)
                all_groups.append(2)  # 2=distant
    if not all_emb:
        return False
    emb_t = torch.stack(all_emb, dim=0)
    plot_embedding_2d(
        emb_t,
        instance_ids=all_inst_ids,
        group_labels=all_groups,
        method="pca",
        save_path=save_path,
    )
    return True


@torch.no_grad()
def build_s2_embedding_plot(
    embedder: SolutionEmbedder,
    env: CVRPEnv,
    records: List[Tuple[int, dict]],
    inst_by_idx: Dict[int, dict],
    save_path: str,
) -> bool:
    """Build S2 val embedding PCA plot (anchor/positive/negative)."""
    all_emb: List[torch.Tensor] = []
    all_inst_ids: List[int] = []
    all_groups: List[int] = []
    for idx, rec in records:
        inst = inst_by_idx[idx]
        cache: Dict[str, dict] = {}
        env.load(inst["depot_xy"], inst["node_xy_demand"], cache)
        emb_a = embed_solutions(embedder, [rec["anchor_solution"]], env, cache)
        emb_p = embed_solutions(embedder, [rec["positive_solution"]], env, cache)
        emb_n = embed_solutions(embedder, [rec["negative_solution"]], env, cache)
        all_emb.extend([emb_a[0], emb_p[0], emb_n[0]])
        all_inst_ids.extend([idx, idx, idx])
        all_groups.extend([0, 1, 2])  # anchor/positive/negative
    if not all_emb:
        return False
    emb_t = torch.stack(all_emb, dim=0)
    plot_embedding_2d(
        emb_t,
        instance_ids=all_inst_ids,
        group_labels=all_groups,
        method="pca",
        save_path=save_path,
    )
    return True


def _pick_wandb_run_id(args: argparse.Namespace, all_ckpts: List[str]) -> Optional[str]:
    if args.wandb_run_id:
        return args.wandb_run_id
    id_txt = os.path.join(args.run_dir, "wandb_run_id.txt")
    if os.path.isfile(id_txt):
        with open(id_txt, "r", encoding="utf-8") as f:
            rid = f.read().strip()
        if rid:
            return rid
    for p in sorted(all_ckpts):
        rid = _extract_wandb_run_id_from_checkpoint(p)
        if rid:
            return rid
    if all_ckpts:
        return _infer_wandb_run_id_from_local(all_ckpts[-1], PROJECT_ROOT)
    return None


def _get_wandb_run_path(wb_run) -> str:
    """Return 'entity/project/run_id' robustly across wandb versions."""
    path = getattr(wb_run, "path", None)
    if isinstance(path, str) and path.count("/") >= 2:
        return path
    if isinstance(path, (list, tuple)) and len(path) >= 3:
        return f"{path[0]}/{path[1]}/{path[2]}"
    entity = getattr(wb_run, "entity", None)
    project = getattr(wb_run, "project", None)
    run_id = getattr(wb_run, "id", None)
    if entity and project and run_id:
        return f"{entity}/{project}/{run_id}"
    raise RuntimeError("Cannot derive wandb run path from current run object.")


def _resolve_existing_path(path: str, alt_candidates: List[str]) -> str:
    """Resolve an existing file path with fallback candidates."""
    if os.path.isfile(path):
        return path
    for c in alt_candidates:
        if c and os.path.isfile(c):
            print(f"[backfill] Using fallback path: {c}")
            return c
    tried = [path] + [c for c in alt_candidates if c]
    raise FileNotFoundError(f"File not found. Tried: {tried}")


def _fetch_epoch_step_map(wb_run, stage_idx: int) -> Dict[int, int]:
    """Build mapping epoch -> step from existing wandb history."""
    out: Dict[int, int] = {}
    epoch_key = f"s{stage_idx}/epoch"
    try:
        api = wandb.Api()
        run_path = _get_wandb_run_path(wb_run)
        run = api.run(run_path)
        for row in run.scan_history(keys=["_step", epoch_key]):
            step = row.get("_step")
            ep = row.get(epoch_key)
            if step is None or ep is None:
                continue
            try:
                ep_i = int(ep)
                step_i = int(step)
            except (TypeError, ValueError):
                continue
            # Keep latest step for that epoch if repeated.
            prev = out.get(ep_i, -1)
            if step_i > prev:
                out[ep_i] = step_i
    except Exception as e:
        print(f"[backfill] WARN: cannot fetch epoch->step map for s{stage_idx}: {e}")
    return out


def _fetch_existing_key_steps(wb_run, keys: List[str]) -> Dict[str, set]:
    """Fetch existing step positions for specific keys in target run."""
    key_steps: Dict[str, set] = {k: set() for k in keys}
    if not keys:
        return key_steps
    try:
        api = wandb.Api()
        run_path = _get_wandb_run_path(wb_run)
        run = api.run(run_path)
        scan_keys = ["_step"] + keys
        for row in run.scan_history(keys=scan_keys):
            step = row.get("_step")
            if step is None:
                continue
            try:
                step_i = int(step)
            except (TypeError, ValueError):
                continue
            for k in keys:
                if row.get(k) is not None:
                    key_steps[k].add(step_i)
    except Exception as e:
        print(f"[backfill] WARN: cannot fetch existing key->step map: {e}")
    return key_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill val metrics/plots to existing W&B run.")
    parser.add_argument("--run_dir", type=str, required=True, help="Training output dir containing s*_epoch*.pt.")
    parser.add_argument("--project", type=str, default="landscape", help="W&B project name.")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="Target W&B run id. Auto-detected if omitted.")
    parser.add_argument("--wandb_resume_mode", type=str, choices=["allow", "must"], default="must")
    parser.add_argument("--epochs_every", type=int, default=5, help="Backfill only epochs divisible by this value.")
    parser.add_argument("--no_images", action="store_true", help="Skip uploading histogram images.")
    parser.add_argument(
        "--metric_namespace",
        type=str,
        choices=["original", "backfill", "both"],
        default="original",
        help="Where to log metrics: original keys (comparable), backfill keys (isolated), or both.",
    )
    parser.add_argument(
        "--allow_overwrite_existing",
        action="store_true",
        help="Allow writing when target key already exists at target step. Default: skip conflicting keys with warning.",
    )
    parser.add_argument("--align_to_existing_steps", action="store_true", default=True, help="Align backfill logs to existing stage epoch->step mapping in wandb.")
    parser.add_argument("--no_align_to_existing_steps", action="store_true", help="Disable step alignment and append logs at current step.")
    parser.add_argument("--include_epoch0_random", action="store_true", default=True, help="Also backfill epoch0 using random-initialized encoder.")
    parser.add_argument("--no_include_epoch0_random", action="store_true", help="Disable epoch0 random-init backfill.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--problem_size", type=int, default=100)
    parser.add_argument("--instance_pkl", type=str, default=os.path.join(PROJECT_ROOT, "cvrp100_uniform.pkl"))
    parser.add_argument("--val_data_1a1n10d", type=str, default=os.path.join(PROJECT_ROOT, "basin_datasets0_analyze", "val_data_1a1n10d.jsonl"))
    parser.add_argument("--val_data_1p1n", type=str, default=os.path.join(PROJECT_ROOT, "perturb_k1_collect", "val_data_1p1n.jsonl"))
    parser.add_argument("--batch_size2", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--supplement_feature_dim", type=int, default=5)
    parser.add_argument("--use_l2_normalize", action="store_true", default=True)
    args = parser.parse_args()
    args.instance_pkl = _resolve_existing_path(
        args.instance_pkl,
        [
            os.path.join(os.getcwd(), "cvrp100_uniform.pkl"),
            os.path.expanduser("~/cvrp100_uniform.pkl"),
        ],
    )

    seed_everything(args.seed)
    args.device = torch.device(args.device)
    env = CVRPEnv(problem_size=args.problem_size, device=args.device)
    embedder = _build_embedder(args)
    embedder.eval()

    s1_ckpts = sorted(glob.glob(os.path.join(args.run_dir, "s1_epoch*.pt")), key=lambda p: _extract_epoch(p, "s1") or 0)
    s2_ckpts = sorted(glob.glob(os.path.join(args.run_dir, "s2_epoch*.pt")), key=lambda p: _extract_epoch(p, "s2") or 0)
    all_ckpts = sorted(set(s1_ckpts + s2_ckpts))
    if not all_ckpts:
        raise RuntimeError(f"No s1/s2 checkpoints found in {args.run_dir}")

    run_id = _pick_wandb_run_id(args, all_ckpts)
    if not run_id:
        raise RuntimeError("Cannot infer W&B run id. Please pass --wandb_run_id.")

    wb = wandb.init(
        project=args.project,
        id=run_id,
        resume=args.wandb_resume_mode,
    )
    print(f"[backfill] Logging to run id={run_id}")
    current_run_step = int(getattr(wb, "step", 0) or 0)
    print(f"[backfill] Current run step={current_run_step}")
    log_original = args.metric_namespace in ("original", "both")
    log_backfill = args.metric_namespace in ("backfill", "both")
    use_step_align = args.align_to_existing_steps and (not args.no_align_to_existing_steps)
    s1_step_map: Dict[int, int] = _fetch_epoch_step_map(wb, 1) if use_step_align else {}
    s2_step_map: Dict[int, int] = _fetch_epoch_step_map(wb, 2) if use_step_align else {}
    if use_step_align:
        print(f"[backfill] Step alignment: s1 epochs={len(s1_step_map)}, s2 epochs={len(s2_step_map)}")

    # Keep a consistent historical x-axis for backfilled metrics.
    # This does not bypass wandb's monotonic step rule; it adds an explicit axis to plot against.
    try:
        wandb.define_metric("backfill/orig_step")
        for k in [
            "backfill/val_s1/d_ap", "backfill/val_s1/d_an", "backfill/val_s1/d_an_over_d_ap", "backfill/plot_s1/hist_wasserstein",
            "backfill/val_s2/d_ap", "backfill/val_s2/d_an", "backfill/val_s2/d_an_over_d_ap", "backfill/plot_s2/hist_wasserstein",
            "val_s1/d_ap", "val_s1/d_an", "val_s1/d_an_over_d_ap", "plot_s1/hist_wasserstein",
            "val_s2/d_ap", "val_s2/d_an", "val_s2/d_an_over_d_ap", "plot_s2/hist_wasserstein",
        ]:
            wandb.define_metric(k, step_metric="backfill/orig_step")
    except Exception as e:
        print(f"[backfill] WARN: define_metric failed: {e}")

    tracked_keys = [
        "val_s1/d_ap", "val_s1/d_an", "val_s1/d_an_over_d_ap", "plot_s1/distance_hist",
        "plot_s1/embedding_2d",
        "val_s2/d_ap", "val_s2/d_an", "val_s2/d_an_over_d_ap", "plot_s2/distance_hist",
        "plot_s2/embedding_2d",
        "backfill/val_s1/d_ap", "backfill/val_s1/d_an", "backfill/val_s1/d_an_over_d_ap", "backfill/plot_s1/distance_hist",
        "backfill/plot_s1/embedding_2d",
        "backfill/val_s2/d_ap", "backfill/val_s2/d_an", "backfill/val_s2/d_an_over_d_ap", "backfill/plot_s2/distance_hist",
        "backfill/plot_s2/embedding_2d",
    ]
    existing_key_steps = _fetch_existing_key_steps(wb, tracked_keys)
    n_conflict_events = 0
    n_conflict_keys = 0
    n_skipped_keys = 0

    plot_dir = os.path.join(args.run_dir, "backfill_plot")
    os.makedirs(plot_dir, exist_ok=True)

    # Stage 1 fixed val
    s1_records = load_val_data_1a1n10d(args.val_data_1a1n10d)
    s1_indices = sorted(set(idx for idx, _ in s1_records))
    s1_inst_list = load_instances_pkl(args.instance_pkl, args.device, s1_indices, {})
    s1_inst_by_idx = dict(zip(s1_indices, s1_inst_list))

    # Stage 2 fixed val
    s2_records = load_val_data_1p1n(args.val_data_1p1n)
    s2_indices = sorted(set(idx for idx, _ in s2_records))
    s2_inst_list = load_instances_pkl(args.instance_pkl, args.device, s2_indices, {})
    s2_inst_by_idx = dict(zip(s2_indices, s2_inst_list))

    include_epoch0_random = args.include_epoch0_random and (not args.no_include_epoch0_random)
    if include_epoch0_random:
        # Epoch 0: random initialization baseline (same seed/model init path as training entry).
        log0_s1 = {}
        s1_ap0, s1_an0, s1_ratio0, s1_dap0, s1_dan0 = eval_s1(embedder, env, s1_records, s1_inst_by_idx)
        if log_backfill:
            log0_s1.update({
                "backfill/val_s1/d_ap": s1_ap0,
                "backfill/val_s1/d_an": s1_an0,
                "backfill/val_s1/d_an_over_d_ap": s1_ratio0,
                "backfill/epoch": 0,
                "backfill/stage_idx": 1,
            })
        if log_original:
            log0_s1.update({
                "val_s1/d_ap": s1_ap0,
                "val_s1/d_an": s1_an0,
                "val_s1/d_an_over_d_ap": s1_ratio0,
                "s1/epoch": 0,
            })
        if (not args.no_images) and s1_dap0 and s1_dan0:
            s1_hist0 = os.path.join(plot_dir, "distance_hist_s1_epoch0.png")
            plot_distance_histogram(s1_dap0, s1_dan0, save_path=s1_hist0)
            s1_w0 = float(wasserstein_distance(s1_dap0, s1_dan0))
            s1_emb0 = os.path.join(plot_dir, "embedding_2d_s1_epoch0.png")
            s1_has_emb = build_s1_embedding_plot(embedder, env, s1_records, s1_inst_by_idx, s1_emb0)
            if log_backfill:
                log0_s1["backfill/plot_s1/distance_hist"] = wandb.Image(s1_hist0)
                log0_s1["backfill/plot_s1/hist_wasserstein"] = s1_w0
                if s1_has_emb:
                    log0_s1["backfill/plot_s1/embedding_2d"] = wandb.Image(s1_emb0)
            if log_original:
                log0_s1["plot_s1/distance_hist"] = wandb.Image(s1_hist0)
                log0_s1["plot_s1/hist_wasserstein"] = s1_w0
                if s1_has_emb:
                    log0_s1["plot_s1/embedding_2d"] = wandb.Image(s1_emb0)
        log0_s1["backfill/orig_step"] = float(s1_step_map.get(0, 0))
        wandb.log(log0_s1)
        print(f"[backfill][S1] epoch0(random) d_ap={s1_ap0:.4f} d_an={s1_an0:.4f} ratio={s1_ratio0:.4f}")

        log0_s2 = {}
        s2_ap0, s2_an0, s2_ratio0, s2_dap0, s2_dan0 = eval_s2(embedder, env, s2_records, s2_inst_by_idx, args.batch_size2)
        if log_backfill:
            log0_s2.update({
                "backfill/val_s2/d_ap": s2_ap0,
                "backfill/val_s2/d_an": s2_an0,
                "backfill/val_s2/d_an_over_d_ap": s2_ratio0,
                "backfill/epoch": 0,
                "backfill/stage_idx": 2,
            })
        if log_original:
            log0_s2.update({
                "val_s2/d_ap": s2_ap0,
                "val_s2/d_an": s2_an0,
                "val_s2/d_an_over_d_ap": s2_ratio0,
                "s2/epoch": 0,
            })
        if (not args.no_images) and s2_dap0 and s2_dan0:
            s2_hist0 = os.path.join(plot_dir, "distance_hist_s2_epoch0.png")
            plot_distance_histogram(s2_dap0, s2_dan0, save_path=s2_hist0)
            s2_w0 = float(wasserstein_distance(s2_dap0, s2_dan0))
            s2_emb0 = os.path.join(plot_dir, "embedding_2d_s2_epoch0.png")
            s2_has_emb = build_s2_embedding_plot(embedder, env, s2_records, s2_inst_by_idx, s2_emb0)
            if log_backfill:
                log0_s2["backfill/plot_s2/distance_hist"] = wandb.Image(s2_hist0)
                log0_s2["backfill/plot_s2/hist_wasserstein"] = s2_w0
                if s2_has_emb:
                    log0_s2["backfill/plot_s2/embedding_2d"] = wandb.Image(s2_emb0)
            if log_original:
                log0_s2["plot_s2/distance_hist"] = wandb.Image(s2_hist0)
                log0_s2["plot_s2/hist_wasserstein"] = s2_w0
                if s2_has_emb:
                    log0_s2["plot_s2/embedding_2d"] = wandb.Image(s2_emb0)
        log0_s2["backfill/orig_step"] = float(s2_step_map.get(0, 0))
        wandb.log(log0_s2)
        print(f"[backfill][S2] epoch0(random) d_ap={s2_ap0:.4f} d_an={s2_an0:.4f} ratio={s2_ratio0:.4f}")

    # Backfill S1
    for ckpt_path in s1_ckpts:
        epoch = _extract_epoch(ckpt_path, "s1")
        if epoch is None or epoch % args.epochs_every != 0:
            continue
        ckpt = torch.load(ckpt_path, map_location=args.device)
        embedder.load_state_dict(ckpt["embedder_state"])
        mean_ap, mean_an, ratio, d_ap, d_an = eval_s1(embedder, env, s1_records, s1_inst_by_idx)
        log = {}
        if log_backfill:
            log.update({
                "backfill/val_s1/d_ap": mean_ap,
                "backfill/val_s1/d_an": mean_an,
                "backfill/val_s1/d_an_over_d_ap": ratio,
                "backfill/epoch": epoch,
                "backfill/stage_idx": 1,
            })
        if log_original:
            log.update({
                "val_s1/d_ap": mean_ap,
                "val_s1/d_an": mean_an,
                "val_s1/d_an_over_d_ap": ratio,
                "s1/epoch": epoch,
            })
        step = s1_step_map.get(epoch) if use_step_align else None
        log["backfill/orig_step"] = float(step) if step is not None else float(epoch)
        if (not args.no_images) and d_ap and d_an:
            hist_path = os.path.join(plot_dir, f"distance_hist_s1_epoch{epoch}.png")
            plot_distance_histogram(d_ap, d_an, save_path=hist_path)
            w = float(wasserstein_distance(d_ap, d_an))
            emb_path = os.path.join(plot_dir, f"embedding_2d_s1_epoch{epoch}.png")
            has_emb = build_s1_embedding_plot(embedder, env, s1_records, s1_inst_by_idx, emb_path)
            if log_backfill:
                log["backfill/plot_s1/distance_hist"] = wandb.Image(hist_path)
                log["backfill/plot_s1/hist_wasserstein"] = w
                if has_emb:
                    log["backfill/plot_s1/embedding_2d"] = wandb.Image(emb_path)
            if log_original:
                log["plot_s1/distance_hist"] = wandb.Image(hist_path)
                log["plot_s1/hist_wasserstein"] = w
                if has_emb:
                    log["plot_s1/embedding_2d"] = wandb.Image(emb_path)
        if step is None:
            wandb.log(log)
        else:
            if step < current_run_step:
                print(
                    f"[backfill][WARN] S1 epoch{epoch}: target step={step} < current run step={current_run_step}, "
                    "fallback to append mode (no explicit step)."
                )
                wandb.log(log)
                continue
            conflict_keys = [k for k in list(log.keys()) if step in existing_key_steps.get(k, set())]
            if conflict_keys:
                n_conflict_events += 1
                n_conflict_keys += len(conflict_keys)
                print(
                    f"[backfill][WARN] S1 epoch{epoch} step={step}: existing keys detected -> {conflict_keys}"
                )
                if not args.allow_overwrite_existing:
                    for k in conflict_keys:
                        log.pop(k, None)
                        n_skipped_keys += 1
            if not log:
                print(f"[backfill][WARN] S1 epoch{epoch}: all keys conflicted and were skipped.")
                continue
            wandb.log(log, step=step)
            for k in log.keys():
                existing_key_steps.setdefault(k, set()).add(step)
        print(f"[backfill][S1] epoch{epoch} d_ap={mean_ap:.4f} d_an={mean_an:.4f} ratio={ratio:.4f}")

    # Backfill S2
    for ckpt_path in s2_ckpts:
        epoch = _extract_epoch(ckpt_path, "s2")
        if epoch is None or epoch % args.epochs_every != 0:
            continue
        ckpt = torch.load(ckpt_path, map_location=args.device)
        embedder.load_state_dict(ckpt["embedder_state"])
        mean_ap, mean_an, ratio, d_ap, d_an = eval_s2(embedder, env, s2_records, s2_inst_by_idx, args.batch_size2)
        log = {}
        if log_backfill:
            log.update({
                "backfill/val_s2/d_ap": mean_ap,
                "backfill/val_s2/d_an": mean_an,
                "backfill/val_s2/d_an_over_d_ap": ratio,
                "backfill/epoch": epoch,
                "backfill/stage_idx": 2,
            })
        if log_original:
            log.update({
                "val_s2/d_ap": mean_ap,
                "val_s2/d_an": mean_an,
                "val_s2/d_an_over_d_ap": ratio,
                "s2/epoch": epoch,
            })
        step = s2_step_map.get(epoch) if use_step_align else None
        log["backfill/orig_step"] = float(step) if step is not None else float(epoch)
        if (not args.no_images) and d_ap and d_an:
            hist_path = os.path.join(plot_dir, f"distance_hist_s2_epoch{epoch}.png")
            plot_distance_histogram(d_ap, d_an, save_path=hist_path)
            w = float(wasserstein_distance(d_ap, d_an))
            emb_path = os.path.join(plot_dir, f"embedding_2d_s2_epoch{epoch}.png")
            has_emb = build_s2_embedding_plot(embedder, env, s2_records, s2_inst_by_idx, emb_path)
            if log_backfill:
                log["backfill/plot_s2/distance_hist"] = wandb.Image(hist_path)
                log["backfill/plot_s2/hist_wasserstein"] = w
                if has_emb:
                    log["backfill/plot_s2/embedding_2d"] = wandb.Image(emb_path)
            if log_original:
                log["plot_s2/distance_hist"] = wandb.Image(hist_path)
                log["plot_s2/hist_wasserstein"] = w
                if has_emb:
                    log["plot_s2/embedding_2d"] = wandb.Image(emb_path)
        if step is None:
            wandb.log(log)
        else:
            if step < current_run_step:
                print(
                    f"[backfill][WARN] S2 epoch{epoch}: target step={step} < current run step={current_run_step}, "
                    "fallback to append mode (no explicit step)."
                )
                wandb.log(log)
                continue
            conflict_keys = [k for k in list(log.keys()) if step in existing_key_steps.get(k, set())]
            if conflict_keys:
                n_conflict_events += 1
                n_conflict_keys += len(conflict_keys)
                print(
                    f"[backfill][WARN] S2 epoch{epoch} step={step}: existing keys detected -> {conflict_keys}"
                )
                if not args.allow_overwrite_existing:
                    for k in conflict_keys:
                        log.pop(k, None)
                        n_skipped_keys += 1
            if not log:
                print(f"[backfill][WARN] S2 epoch{epoch}: all keys conflicted and were skipped.")
                continue
            wandb.log(log, step=step)
            for k in log.keys():
                existing_key_steps.setdefault(k, set()).add(step)
        print(f"[backfill][S2] epoch{epoch} d_ap={mean_ap:.4f} d_an={mean_an:.4f} ratio={ratio:.4f}")

    if n_conflict_events > 0:
        print(
            f"[backfill][SUMMARY] conflict_events={n_conflict_events}, "
            f"conflict_keys={n_conflict_keys}, skipped_keys={n_skipped_keys}, "
            f"allow_overwrite_existing={args.allow_overwrite_existing}"
        )

    wandb.finish()
    print("[backfill] Done.")


if __name__ == "__main__":
    main()

