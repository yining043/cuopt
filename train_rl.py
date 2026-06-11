"""Parallel training for the cuOpt local-search subset-selection policy.

Architecture
------------
* Master process (this script): holds the policy + optimizer, never imports
  cuOpt. Each round it (1) dumps current weights, (2) launches `batch_episodes`
  rollout workers across all GPUs, (3) aggregates their full-feedback
  transitions into policy updates, (4) periodically evaluates by averaging final
  cost over several stochastic cuOpt runs.
* Rollout workers (rl_rollout.py): one cuOpt Solve each, on a dedicated GPU.

One episode == one cuOpt Solve. Within it every inner local-search iteration is
one policy step: cuOpt proposes K candidate node/operator subsets, the policy
selects one, and cuOpt executes the selected subset if it yields an improving
move. For training, C++ additionally labels all K candidates on copied
solutions, so the update is a full-feedback contextual-bandit objective rather
than an episode-level Monte Carlo return.

Run (use all GPUs):
  conda activate cuopt_dev
  python train_rl.py --index 1 --time_limit 5 --rounds 300 \
      --batch_episodes 8 --gpus 0,1,2,3 --eval_runs 8 --runname rl_inst1
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from model import CostPredictor, load_checkpoint


def autocast_context(device, amp_dtype):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# Full-feedback policy update
# ---------------------------------------------------------------------------

def reinforce_update(model, optimizer, episodes, coords, demand, cap, device, args,
                     tw_features=None):
    """Full-feedback contextual-bandit policy gradient.

    Each step exposes labels for all K candidate subsets. We normalize the
    valid-arm labels within the state and maximize the exact expected label
    reward J = sum_i p_i * adv_i, plus an entropy bonus. `top1_lookahead_acc`
    tracks whether the policy argmax matches the best short-horizon label, and
    `top1_current_acc` tracks the same metric for one-step labels.
    """
    all_steps = []
    n_low_spread = 0
    # In-window decision points cuOpt actually collected (the warm-up part of each
    # solve runs plain cuOpt and never fires the callback, so it contributes 0 here).
    n_collected_in_window = sum(len(ep) for ep in episodes if ep)
    for ep_trans in episodes:
        if not ep_trans:
            continue
        idxs = list(range(len(ep_trans)))
        if args.max_update_steps_per_ep and len(idxs) > args.max_update_steps_per_ep:
            idxs = list(np.random.choice(idxs, args.max_update_steps_per_ep, replace=False))
        for i in idxs:
            t = ep_trans[i]
            # Spread filter: skip decision points whose valid lookahead-reward
            # range (max-min) is below threshold -- those K probes are nearly
            # indistinguishable, so they carry little ranking signal for the bandit.
            if args.min_reward_spread > 0.0:
                rv = t['rewards'][t['valid']]
                if rv.numel() < 2 or float(rv.max() - rv.min()) <= args.min_reward_spread:
                    n_low_spread += 1
                    continue
            all_steps.append(t)

    if not all_steps:
        return {"loss": 0.0, "entropy": 0.0, "grad_norm": 0.0,
                "top1_acc": 0.0, "rand_top1": 0.0, "n_update_steps": 0,
                "n_opt_steps": 0, "top1_lookahead_acc": 0.0,
                "top1_current_acc": 0.0, "n_filtered_low_spread": n_low_spread}

    coords_d = coords.to(device)
    dem_d = (demand / cap).to(device)
    tw_d = tw_features.to(device) if tw_features is not None else None

    eps = 1e-6

    def step_loss(t):
        """Full-feedback bandit loss for one decision point."""
        masks = t['masks'].to(device).long()
        valid = t['valid'].to(device)
        rewards = t['rewards'].to(device)
        immediate_rewards = t.get('immediate_rewards', t['rewards']).to(device)
        K, L = masks.shape
        sol = t['sol'].to(device).long().unsqueeze(0).expand(K, -1)
        nodes = coords_d.expand(K, -1, -1)
        demands = dem_d.expand(K, -1)
        cost_0 = torch.full((K,), t['cost_0'], dtype=torch.float32, device=device)
        tw = tw_d.expand(K, -1, -1) if tw_d is not None else None

        with autocast_context(device, args.amp_dtype):
            scores = model(nodes, demands, sol, masks, cost_0, tw_features=tw)
        scores = scores.float()
        logits = args.score_sign * scores / args.temperature
        if args.logit_clip > 0:
            logits = logits.clamp(-args.logit_clip, args.logit_clip)
        logits = logits.masked_fill(~valid, float('-inf'))
        log_probs = F.log_softmax(logits, dim=0)
        probs = log_probs.exp()

        rv = rewards[valid]
        if rv.numel() > 1 and rv.std() > eps:
            adv = (rewards - rv.mean()) / (rv.std() + eps)
        else:
            adv = rewards - (rv.mean() if rv.numel() > 0 else 0.0)
        adv = adv.masked_fill(~valid, 0.0)

        # Zero out invalid arms in log_probs BEFORE the product. The naive
        # `probs * log_probs` evaluates 0 * (-inf) = NaN at masked slots; even
        # though masked_fill fixes the forward value, the NaN still flows through
        # the backward pass and poisons gradients via log_softmax. Masking the
        # log-probs first makes those slots contribute exactly 0 in both
        # directions, so empty arms truly do not participate.
        safe_log_probs = log_probs.masked_fill(~valid, 0.0)
        J = (probs * adv).sum()
        entropy = -(probs * safe_log_probs).sum()
        loss = -J - args.entropy_coef * entropy

        pred_arm = probs.argmax()
        best_arm = rewards.masked_fill(~valid, float('-inf')).argmax()
        best_current_arm = immediate_rewards.masked_fill(~valid, float('-inf')).argmax()
        top1 = float((pred_arm == best_arm).item())
        top1_current = float((pred_arm == best_current_arm).item())
        rnd = 1.0 / float(valid.sum().item())
        return loss, float((-J).item()), float(entropy.item()), top1, top1_current, rnd

    model.train()
    n = len(all_steps)
    accum = max(1, args.update_minibatch)   # steps accumulated per optimizer.step()
    update_epochs = max(1, args.update_epochs)

    # Multiple epochs over the same collected steps. Each epoch gets a fresh
    # shuffle and pads the tail with random resamples from the same pool so every
    # optimizer step uses exactly `accum` steps (uniform gradient scale).
    epoch_orders = []
    for _ in range(update_epochs):
        order = np.random.permutation(n).tolist()
        rem = (-len(order)) % accum
        if rem:
            order += list(np.random.choice(n, rem, replace=(n < rem)))
        epoch_orders.append(order)

    n_opt_planned = sum(len(order) // accum for order in epoch_orders)
    print(f"  [update] collected_in_window={n_collected_in_window} "
          f"skipped_low_spread={n_low_spread} (spread<= {args.min_reward_spread:g}) "
          f"-> kept_for_update={n} | epochs={update_epochs} "
          f"train_step_passes={n * update_epochs}, "
          f"accumulate={accum} steps per update -> {n_opt_planned} optimizer steps",
          flush=True)

    total_loss = total_ent = total_top1 = total_top1_current = total_rand = 0.0
    n_seen = 0
    n_opt = 0
    last_grad = 0.0
    t_upd0 = time.time()

    optimizer.zero_grad()
    for epoch, order in enumerate(epoch_orders, start=1):
        for j, i in enumerate(order):
            loss, jval, ent, t1, t1_current, rnd = step_loss(all_steps[int(i)])
            (loss / accum).backward()
            total_loss += jval; total_ent += ent
            total_top1 += t1; total_top1_current += t1_current; total_rand += rnd; n_seen += 1
            if (j + 1) % accum == 0:
                last_grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
                optimizer.step()
                optimizer.zero_grad()
                n_opt += 1
                print(f"    [optimizer_step {n_opt}/{n_opt_planned}] "
                      f"epoch={epoch}/{update_epochs} "
                      f"loss={total_loss / max(n_seen,1):.4f} "
                      f"entropy={total_ent / max(n_seen,1):.3f} "
                      f"top1_lookahead={total_top1 / max(n_seen,1):.3f} "
                      f"top1_current={total_top1_current / max(n_seen,1):.3f} "
                      f"grad_norm={last_grad:.3f} "
                      f"elapsed_seconds={time.time() - t_upd0:.0f}",
                      flush=True)

    denom = max(n_seen, 1)
    return {
        "loss": total_loss / denom,
        "entropy": total_ent / denom,
        "grad_norm": last_grad,
        "top1_acc": total_top1 / denom,
        "top1_lookahead_acc": total_top1 / denom,
        "top1_current_acc": total_top1_current / denom,
        "rand_top1": total_rand / denom,
        "n_update_steps": n,
        "n_collected_in_window": n_collected_in_window,
        "n_filtered_low_spread": n_low_spread,
        "update_epochs": update_epochs,
        "n_train_step_passes": n * update_epochs,
        "n_opt_steps": n_opt,
    }


# ---------------------------------------------------------------------------
# Parallel rollout launching
# ---------------------------------------------------------------------------

def curriculum_time_limit(rnd, args):
    """Per-round solve time under the optional time curriculum.

    Grows the base --time_limit by --cl_increment seconds every --cl_every rounds,
    capped at --cl_max_time. Returns the unchanged base when the curriculum is off
    (cl_increment<=0), so behavior is identical to before unless explicitly enabled.
    """
    base = float(args.time_limit)
    if args.cl_increment <= 0 or args.cl_every <= 0:
        return base
    t = base + (rnd // args.cl_every) * args.cl_increment
    if args.cl_max_time > 0:
        t = min(t, float(args.cl_max_time))
    return float(t)


def launch_rollouts(specs, gpus, out_dir, args, weights_path, train, mode_run="policy",
                    progress=None, time_limit=None, collect_last_sec=None):
    """specs: list of (tag, seed). Returns list of result dicts (loaded .pt).

    Runs in waves of len(gpus); each worker pinned to one GPU.
    If `progress` (a label string) is set, print one flushed line per finished
    rollout so the live tmux log shows collection/eval advancing.

    `time_limit` / `collect_last_sec` override the args defaults for this call
    (used by the curriculum + collection-window scheme); None falls back to args.
    """
    tl = float(args.time_limit if time_limit is None else time_limit)
    cls = float(args.collect_last_sec if collect_last_sec is None else collect_last_sec)
    results = [None] * len(specs)
    worker_log_dir = os.path.join(out_dir, "worker_logs")
    os.makedirs(worker_log_dir, exist_ok=True)
    total = len(specs)
    done = 0

    pending = list(enumerate(specs))
    while pending:
        wave = pending[:len(gpus)]
        pending = pending[len(gpus):]
        procs = []
        for slot, (ridx, (tag, seed)) in enumerate(wave):
            gpu = gpus[slot % len(gpus)]
            out_file = os.path.join(out_dir, f"roll_{tag}.pt")
            if os.path.exists(out_file):
                os.remove(out_file)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            reward_horizon = args.reward_horizon if train else 1
            cmd = [
                sys.executable, "rl_rollout.py",
                "--mode_run", mode_run,
                "--out", out_file,
                "--data_path", args.data_path,
                "--index", str(args.index),
                *(["--data_pt", args.data_pt] if args.data_pt else []),
                "--time_limit", str(tl),
                "--scale", str(args.scale),
                "--n_vehicles", str(args.n_vehicles),
                "--k", str(args.k),
                "--reward_horizon", str(reward_horizon),
                "--collect_last_sec", str(cls),
                "--model_mode", args.mode,
                "--score_sign", str(args.score_sign),
                "--temperature", str(args.temperature),
                "--logit_clip", str(args.logit_clip),
                "--selection", args.eval_selection,
                "--amp_dtype", args.amp_dtype,
                "--train", str(1 if train else 0),
                "--seed", str(seed),
            ]
            if weights_path and mode_run == "policy":
                cmd += ["--weights", weights_path]
            logf = open(os.path.join(worker_log_dir, f"{tag}.log"), "w")
            p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
            procs.append((ridx, tag, out_file, p, logf))

        for ridx, tag, out_file, p, logf in procs:
            p.wait()
            logf.close()
            if os.path.exists(out_file):
                try:
                    results[ridx] = torch.load(out_file, map_location="cpu")
                except Exception as e:
                    print(f"[warn] failed to load {out_file}: {e}", flush=True)
                    results[ridx] = None
            else:
                print(f"[warn] rollout {tag} produced no output (see worker log)", flush=True)
                results[ridx] = None
            done += 1
            if progress:
                r = results[ridx]
                fc = r.get("final_cost") if r else None
                ns = r.get("n_steps") if r else None
                fc_s = f"{fc:.4f}" if isinstance(fc, (int, float)) else "NA"
                print(f"  [{progress} {done}/{total}] {tag} final_cost={fc_s} steps={ns}",
                      flush=True)
    return results


def eval_policy(gpus, out_dir, args, weights_path, time_limit=None):
    """Average final cost over args.eval_runs policy runs.

    Uses the per-round `time_limit` (curriculum) and the same collection window as
    training, so eval measures the policy in exactly the regime it is trained on.
    """
    pol_specs = [(f"eval_pol_{i}", 10000 + i) for i in range(args.eval_runs)]
    pol = launch_rollouts(pol_specs, gpus, out_dir, args, weights_path, train=False,
                          mode_run="policy", progress="evaluate-policy",
                          time_limit=time_limit, collect_last_sec=args.collect_last_sec)
    pol_costs = [r["final_cost"] for r in pol if r and r["final_cost"] is not None]
    return {
        "eval_policy_mean": float(np.mean(pol_costs)) if pol_costs else None,
        "eval_policy_min": float(np.min(pol_costs)) if pol_costs else None,
        "eval_policy_std": float(np.std(pol_costs)) if pol_costs else None,
        "eval_n": len(pol_costs),
    }


def eval_random_baseline(gpus, out_dir, args, time_limit=None):
    """Average final cost over args.eval_runs random-subset runs.

    The random baseline is policy-independent but DOES depend on the solve time
    (and collection window), so under the curriculum it is recomputed per distinct
    time_limit and cached by the caller; with the curriculum off it is computed once.
    Uses the same warm-up + last-N-second window as the policy eval for fairness.
    """
    rnd_specs = [(f"eval_rnd_{i}", 30000 + i) for i in range(args.eval_runs)]
    rnd = launch_rollouts(rnd_specs, gpus, out_dir, args, None, train=False,
                          mode_run="random", progress="evaluate-random",
                          time_limit=time_limit, collect_last_sec=args.collect_last_sec)
    rnd_costs = [r["final_cost"] for r in rnd if r and r["final_cost"] is not None]
    return {
        "eval_random_mean": float(np.mean(rnd_costs)) if rnd_costs else None,
        "eval_random_min": float(np.min(rnd_costs)) if rnd_costs else None,
    }


# ---------------------------------------------------------------------------
# Checkpoint retention
# ---------------------------------------------------------------------------

FIRST_N_KEEP = 10              # rounds 0..9 are kept permanently
ROLLING_WINDOW = 10            # the most recent 10 rounds are always kept
WORKER_LOG_KEEP_ROUNDS = 2     # keep collection cuOpt logs for the latest N rounds


def _safe_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def cleanup_round_artifacts(out_dir, rnd):
    """Drop transient per-rollout files once they are no longer useful.

    * roll_r{rnd}_e*.pt: transitions already consumed by this round's update
      (loaded into memory) -> delete immediately.
    * worker_logs/r{round}_e*.log: detailed cuOpt output, useful only for
      debugging recent rounds -> keep the latest WORKER_LOG_KEEP_ROUNDS rounds.
    Leaves run.log, train_rl_log.jsonl, and all model checkpoints untouched.
    """
    for f in glob.glob(os.path.join(out_dir, f"roll_r{rnd}_e*.pt")):
        _safe_remove(f)
    wl_dir = os.path.join(out_dir, "worker_logs")
    for f in glob.glob(os.path.join(wl_dir, "r*_e*.log")):
        m = re.match(r"r(\d+)_e\d+\.log$", os.path.basename(f))
        if m and int(m.group(1)) <= rnd - WORKER_LOG_KEEP_ROUNDS:
            _safe_remove(f)


def manage_checkpoints(out_dir, rnd, save_every, save_fn, keep_rounds=None):
    """Save this round's policy and prune snapshots per the retention policy.

    A snapshot for round r is KEPT if any holds:
      * r < FIRST_N_KEEP                  (first 10 rounds, permanent)
      * r % save_every == 0               (milestone rounds, permanent)
      * r in keep_rounds                  (rounds that were evaluated, permanent)
      * r > rnd - ROLLING_WINDOW          (within the latest-10 rolling window)
    Anything else is deleted as the rolling window slides forward.
    """
    keep_rounds = keep_rounds or set()
    save_fn(os.path.join(out_dir, f"policy_round{rnd}.pt"))
    for f in glob.glob(os.path.join(out_dir, "policy_round*.pt")):
        m = re.search(r"policy_round(\d+)\.pt$", os.path.basename(f))
        if not m:
            continue
        r = int(m.group(1))
        keep = ((r < FIRST_N_KEEP) or (r % save_every == 0)
                or (r in keep_rounds) or (r > rnd - ROLLING_WINDOW))
        if not keep:
            try:
                os.remove(f)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parallel REINFORCE for cuOpt node selection")
    parser.add_argument("--data_path", default="../../cuopt-examples/data/test_cvrp1000_hgs_n128_C250.txt")
    parser.add_argument("--data_pt", default=None,
                        help="Cached CVRPTW instance (.pt). When set, trains CVRPTW with TW features.")
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--time_limit", type=float, default=2)
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--batch_episodes", type=int, default=8, help="rollouts aggregated per update")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--scale", type=float, default=1e2)
    parser.add_argument("--n_vehicles", type=int, default=21)
    parser.add_argument("--k", type=int, default=32, help="K candidate subsets (CUOPT_RL_K)")
    parser.add_argument("--reward_horizon", type=int, default=2,
                        help="training only: full-feedback label rollout depth; eval uses 1")
    parser.add_argument("--collect_last_sec", type=float, default=2.0,
                        help="If >0, only the final N seconds of each solve run the probe/"
                             "callback (plain cuOpt warm-up before); passed to rollouts as "
                             "CUOPT_RL_COLLECT_LAST_SEC. Applies to collection AND eval/random.")
    parser.add_argument("--cl_increment", type=float, default=1.0,
                        help="Curriculum: seconds added to the solve time_limit at each bump. "
                             "0 disables the curriculum.")
    parser.add_argument("--cl_patience", type=int, default=3,
                        help="Adaptive curriculum: bump the time_limit once top1_lookahead "
                             "sets no new (per-level) high for this many consecutive rounds. "
                             ">0 selects the ADAPTIVE schedule (overrides --cl_every). After "
                             "each bump the per-level top1 memory is reset.")
    parser.add_argument("--cl_every", type=int, default=3,
                        help="Curriculum (fixed schedule, used only when --cl_patience<=0): "
                             "number of rounds between each +cl_increment bump.")
    parser.add_argument("--cl_max_time", type=float, default=20.0,
                        help="Curriculum: cap on the grown time_limit; <=0 means no cap.")
    parser.add_argument("--min_reward_spread", type=float, default=0.1,
                        help="Drop decision points whose valid lookahead-reward spread "
                             "(max-min) <= this threshold. 0 keeps every step.")
    parser.add_argument("--mode", type=str, default="v2",
                        choices=["ratio", "new", "v2", "v7", "flashv7"])
    parser.add_argument("--backbone_ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--amp_dtype", choices=["none", "bf16"], default="none",
                        help="Autocast dtype for policy forward passes.")
    parser.add_argument("--eval_selection", choices=["greedy", "sample"], default="greedy",
                        help="Policy arm selection for train-time eval; training always samples.")
    parser.add_argument("--score_sign", type=float, default=-1.0)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--logit_clip", type=float, default=0.0,
                        help="Optional symmetric clamp applied to policy logits after temperature; <=0 disables.")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--update_minibatch", type=int, default=16)
    parser.add_argument("--update_epochs", type=int, default=5,
                        help="epochs over the collected rollout steps per round")
    parser.add_argument("--max_update_steps_per_ep", type=int, default=256,
                        help="subsample steps per episode for the update (0=use all)")
    parser.add_argument("--eval_runs", type=int, default=8, help="cuOpt runs to average per eval")
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--master_device", type=str, default="cuda:0")
    parser.add_argument("--runname", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    if args.runname is None:
        args.runname = f"rl_inst{args.index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join("outputs", args.runname)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_rl_log.jsonl")
    weights_path = os.path.join(out_dir, "current_weights.pt")

    device = torch.device(args.master_device if torch.cuda.is_available() else "cpu")
    print(f"[setup] gpus={gpus} master={device} batch_episodes={args.batch_episodes} "
          f"K={args.k} reward_horizon_train={args.reward_horizon} "
          f"reward_horizon_eval=1 update_epochs={args.update_epochs} "
          f"eval_runs={args.eval_runs} eval_selection={args.eval_selection} "
          f"amp_dtype={args.amp_dtype} time_limit={args.time_limit}")

    # Instance data for the master update (coords/demand + optional TW).
    from run_cuopt import pairwise_euclidean_distance  # noqa: F401 (torch-only helper)
    from load_nco_data import load_raw_data, load_cvrptw_data, build_tw_features
    tw_features = None
    if args.data_pt:
        inst = load_cvrptw_data(args.data_pt)
        coords1 = inst["coords"].unsqueeze(0)
        demand1 = inst["demand"].unsqueeze(0)
        cap = inst["capacity"]
        tw_features = build_tw_features(inst)
        n_node_feat = 6
        best_known = float("nan")  # random CVRPTW has no HGS reference
        args.n_vehicles = inst["n_vehicles"]
        args.scale = inst["scale"]
        print(f"[setup] CVRPTW instance {args.data_pt}: customers={coords1.shape[1]-1} "
              f"n_vehicles={args.n_vehicles} H={inst['H']:.1f}")
    else:
        raw_nodes, raw_cap, raw_demand, raw_cost, raw_flag = load_raw_data(
            args.data_path, episode=1, begin_index=args.index)
        coords1 = raw_nodes[0:1]
        demand1 = raw_demand[0:1]
        cap = raw_cap[0].item()
        best_known = raw_cost[0].item()
        n_node_feat = 3

    model = CostPredictor(device=device, mode=args.mode, n_node_feat=n_node_feat,
                          max_vehicles=args.n_vehicles)
    if args.backbone_ckpt:
        load_checkpoint(args.backbone_ckpt, model)
        print(f"[init] warm-started from {args.backbone_ckpt}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.use_wandb:
        import wandb
        wandb.init(project="cuopt-rl", name=args.runname, config=vars(args))

    def save_weights(path):
        torch.save({"model_state_dict": model.state_dict()}, path)

    best_eval = float("inf")
    best_top1_lookahead = float("-inf")
    # Adaptive curriculum state. ADAPTIVE mode (cl_patience>0): the solve time grows
    # by cl_increment whenever top1_lookahead sets no new *per-level* high for
    # cl_patience consecutive rounds; on each bump the per-level top1 memory and the
    # patience counter are reset (so each level re-learns from scratch and gets its
    # own best-top1 checkpoint). cl_patience<=0 falls back to the fixed cl_every
    # schedule via curriculum_time_limit().
    cl_adaptive = args.cl_increment > 0 and args.cl_patience > 0
    current_tl = float(args.time_limit)        # stateful time under adaptive CL
    cl_level = 0                               # which time-level we are on
    level_best_top1 = float("-inf")            # best top1 since last switch
    rounds_since_top1_improve = 0              # patience counter (per level)
    pending_level_start = True                 # round 0 is the first level's start
    # Random-subset baseline reference(s). Without the curriculum this holds one
    # stationary entry; with the curriculum it caches one entry per distinct solve
    # time so every eval compares against random at the SAME time + window.
    random_ref_cache = {}
    # Rounds that triggered an eval: their checkpoints are kept permanently so
    # any evaluated policy can later be benchmarked / reloaded.
    evaluated_rounds = set()
    for rnd in range(args.rounds):
        save_weights(weights_path)

        round_tl = current_tl if cl_adaptive else curriculum_time_limit(rnd, args)
        is_level_start = pending_level_start
        pending_level_start = False
        cl_note = "" if round_tl == args.time_limit else (
            f" (CL level {cl_level}, time_limit={round_tl:g}s)" if cl_adaptive
            else f" (curriculum time_limit={round_tl:g})")
        print(f"[round {rnd}] collecting {args.batch_episodes} episodes on gpus={gpus}{cl_note} ...",
              flush=True)
        t0 = time.time()
        specs = [(f"r{rnd}_e{e}", rnd * 1000 + e) for e in range(args.batch_episodes)]
        rollouts = launch_rollouts(specs, gpus, out_dir, args, weights_path, train=True,
                                   mode_run="policy", progress=f"collect round {rnd}",
                                   time_limit=round_tl, collect_last_sec=args.collect_last_sec)
        roll_wall = time.time() - t0

        episodes = [r["transitions"] for r in rollouts if r and r.get("transitions")]
        final_costs = [r["final_cost"] for r in rollouts if r and r["final_cost"] is not None]
        n_steps_list = [r["n_steps"] for r in rollouts if r]
        all_rewards = [t['reward'] for ep in episodes for t in ep]

        t_upd = time.time()
        stats = reinforce_update(model, optimizer, episodes, coords1, demand1, cap, device, args,
                                 tw_features=tw_features)
        update_wall = time.time() - t_upd
        top1_lookahead = float(stats.get("top1_lookahead_acc", stats.get("top1_acc", 0.0)))
        has_steps = stats.get("n_update_steps", 0) > 0
        if cl_adaptive:
            # New-high is measured against the PER-LEVEL best (reset on each switch),
            # so the patience counter and per-level checkpoint restart every level.
            prev_best_top1_lookahead = level_best_top1
            top1_new_high = has_steps and top1_lookahead > level_best_top1
            if top1_new_high:
                level_best_top1 = top1_lookahead
                rounds_since_top1_improve = 0
            elif has_steps:
                rounds_since_top1_improve += 1
            best_top1_record = None if level_best_top1 == float("-inf") else level_best_top1
        else:
            prev_best_top1_lookahead = best_top1_lookahead
            top1_new_high = has_steps and top1_lookahead > best_top1_lookahead
            if top1_new_high:
                best_top1_lookahead = top1_lookahead
            best_top1_record = None if best_top1_lookahead == float("-inf") else best_top1_lookahead

        # Decide (but do not yet apply) an adaptive CL time bump for the NEXT round:
        # triggered when top1 has plateaued for cl_patience rounds and we are below cap.
        cl_switch = (cl_adaptive and has_steps
                     and current_tl < args.cl_max_time
                     and rounds_since_top1_improve >= args.cl_patience)

        record = {
            "round": rnd,
            "round_time_limit": round_tl,
            "collect_last_sec": args.collect_last_sec,
            "train_cost_mean": float(np.mean(final_costs)) if final_costs else None,
            "train_cost_min": float(np.min(final_costs)) if final_costs else None,
            "best_known": best_known,
            "n_episodes": len(episodes),
            "mean_steps": float(np.mean(n_steps_list)) if n_steps_list else 0,
            "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "roll_wall_s": roll_wall,
            "update_wall_s": update_wall,
            "top1_lookahead_new_high": bool(top1_new_high),
            "best_top1_lookahead": best_top1_record,
            "cl_level": cl_level,
            "cl_rounds_since_top1_improve": rounds_since_top1_improve,
            "cl_patience": args.cl_patience if cl_adaptive else None,
            **stats,
        }

        # Eval workers load current_weights.pt, so refresh it after the update.
        save_weights(weights_path)

        # Per-level best-top1 checkpoint: under adaptive CL, snapshot the policy
        # whenever it sets a new per-level top1 high. One file per time-level holds
        # that level's best-top1 weights (overwritten as it improves within a level),
        # which is what we later benchmark / compare across levels.
        if cl_adaptive and top1_new_high:
            lvl_ckpt = os.path.join(out_dir, f"best_top1_level{cl_level}_t{round_tl:g}s.pt")
            save_weights(lvl_ckpt)
            save_weights(os.path.join(out_dir, "best_top1_policy.pt"))
            print(f"[checkpoint @ round {rnd}] new per-level top1 high "
                  f"{prev_best_top1_lookahead if prev_best_top1_lookahead != float('-inf') else float('nan'):.3f}"
                  f"->{top1_lookahead:.3f} (level {cl_level}, time_limit={round_tl:g}s) "
                  f"saved -> {os.path.basename(lvl_ckpt)}", flush=True)

        eval_reasons = []
        if args.eval_every and rnd % args.eval_every == 0:
            eval_reasons.append("periodic")
        # Force an eval at BOTH ends of every CL level so start-vs-end can be compared
        # fairly at the SAME solve time: the first round of a level (just after a bump,
        # or round 0) and the round that triggers the next bump.
        if cl_adaptive:
            if is_level_start:
                eval_reasons.append("cl_level_start")
            if cl_switch:
                eval_reasons.append("cl_level_end")
        elif args.cl_increment > 0 and args.cl_every > 0:
            if rnd % args.cl_every == 0 and "periodic" not in eval_reasons:
                eval_reasons.append("cl_level_start")
            if rnd % args.cl_every == args.cl_every - 1:
                eval_reasons.append("cl_level_end")
        if rnd == args.rounds - 1 and "final" not in eval_reasons:
            eval_reasons.append("final")
        if top1_new_high:
            prev_s = "none" if prev_best_top1_lookahead == float("-inf") else f"{prev_best_top1_lookahead:.3f}"
            eval_reasons.append(f"top1_lookahead_new_high:{prev_s}->{top1_lookahead:.3f}")
        record["eval_trigger"] = "+".join(eval_reasons) if eval_reasons else None

        if eval_reasons:
            evaluated_rounds.add(rnd)
            ref_key = round(round_tl, 3)
            if ref_key not in random_ref_cache:
                print(f"[round {rnd}] computing random baseline ({args.eval_runs} runs) "
                      f"at time_limit={round_tl:g} window={args.collect_last_sec:g} ...",
                      flush=True)
                random_ref_cache[ref_key] = eval_random_baseline(
                    gpus, out_dir, args, time_limit=round_tl)
                print(f"[baseline] random_mean={random_ref_cache[ref_key]['eval_random_mean']:.4f} "
                      f"random_minimum={random_ref_cache[ref_key]['eval_random_min']:.4f} "
                      f"(reference for time_limit={round_tl:g})",
                      flush=True)
            random_ref = random_ref_cache[ref_key]
            print(f"[round {rnd}] evaluating ({args.eval_runs} policy runs; "
                  f"trigger={record['eval_trigger']}) ...", flush=True)
            ev = eval_policy(gpus, out_dir, args, weights_path, time_limit=round_tl)
            ev.update(random_ref)
            record.update(ev)
            if ev["eval_policy_mean"] is not None and ev["eval_policy_mean"] < best_eval:
                best_eval = ev["eval_policy_mean"]
                save_weights(os.path.join(out_dir, "best_policy.pt"))
            rmean = ev["eval_random_mean"]
            gap = (None if not rmean or ev["eval_policy_mean"] is None else
                   (rmean - ev["eval_policy_mean"]) / rmean)
            record["eval_gap_vs_random"] = gap
            print(f"[evaluation @ round {rnd}] "
                  f"policy_mean={ev['eval_policy_mean']:.4f} "
                  f"policy_minimum={ev['eval_policy_min']:.4f} "
                  f"random_mean={rmean:.4f} "
                  f"random_minimum={ev['eval_random_min']:.4f} "
                  f"gap_vs_random={gap} "
                  f"best_known={best_known:.4f}", flush=True)

        # Apply the adaptive CL bump now (after this round was evaluated as the level
        # end). Reset the per-level top1 memory + patience so the new level re-learns
        # and earns its own checkpoint; mark the next round as the new level's start.
        if cl_switch:
            old_tl = current_tl
            current_tl = min(current_tl + args.cl_increment, float(args.cl_max_time))
            cl_level += 1
            print(f"[CL-SWITCH @ round {rnd}] top1_lookahead plateaued: "
                  f"{rounds_since_top1_improve} consecutive rounds with no new high "
                  f"(level_best_top1={level_best_top1:.3f}). "
                  f"time_limit {old_tl:g}s -> {current_tl:g}s -> CL level {cl_level}. "
                  f"resetting per-level top1 memory + patience.", flush=True)
            record["cl_switch"] = {
                "round": rnd, "from_time_limit": old_tl, "to_time_limit": current_tl,
                "new_level": cl_level, "rounds_no_improve": rounds_since_top1_improve,
                "level_best_top1": level_best_top1,
            }
            level_best_top1 = float("-inf")
            rounds_since_top1_improve = 0
            pending_level_start = True

        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if args.use_wandb:
            import wandb
            wandb.log(record)

        print(f"[round {rnd}] "
              f"train_cost_mean={record['train_cost_mean']} "
              f"episodes={len(episodes)} "
              f"mean_steps={record['mean_steps']:.0f} "
              f"mean_reward={record['mean_reward']:.5f} "
              f"loss={stats['loss']:.4f} "
              f"entropy={stats['entropy']:.3f} "
              f"top1_lookahead={stats['top1_acc']:.3f} "
              f"top1_current={stats.get('top1_current_acc', 0.0):.3f} "
              f"top1_new_high={record['top1_lookahead_new_high']} "
              f"eval_trigger={record['eval_trigger']} "
              f"random_top1_accuracy={stats['rand_top1']:.3f} "
              f"grad_norm={stats['grad_norm']:.3f} "
              f"collected_in_window={stats.get('n_collected_in_window', stats['n_update_steps'])} "
              f"skipped_low_spread={stats.get('n_filtered_low_spread', 0)} "
              f"kept_for_update={stats['n_update_steps']} "
              f"round_time_limit={round_tl:g} "
              + (f"cl_level={record['cl_level']} "
                 f"no_improve={record['cl_rounds_since_top1_improve']}/{args.cl_patience} "
                 f"cl_switch={'YES' if cl_switch else 'no'} " if cl_adaptive else "")
              + f"update_epochs={stats.get('update_epochs', 1)} "
              f"train_step_passes={stats.get('n_train_step_passes', stats['n_update_steps'])} "
              f"optimizer_steps={stats.get('n_opt_steps', 0)} "
              f"rollout_wall_seconds={roll_wall:.0f} "
              f"update_wall_seconds={update_wall:.1f}",
              flush=True)

        manage_checkpoints(out_dir, rnd, args.save_every, save_weights,
                           keep_rounds=evaluated_rounds)
        cleanup_round_artifacts(out_dir, rnd)

    print(f"[done] best eval policy cost={best_eval}")


if __name__ == "__main__":
    main()
