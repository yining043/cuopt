"""Parallel REINFORCE training for the cuOpt node-selection policy.

Architecture
------------
* Master process (this script): holds the policy + optimizer, never imports
  cuOpt. Each round it (1) dumps current weights, (2) launches `batch_episodes`
  rollout workers across all GPUs, (3) aggregates their transitions into one
  big REINFORCE update, (4) periodically evaluates by averaging final cost over
  several stochastic cuOpt runs (cuOpt needs ~8 runs for a stable estimate).
* Rollout workers (rl_rollout.py): one cuOpt Solve each, on a dedicated GPU.

One episode == one cuOpt Solve. Within it every inner local-search iteration is
one RL step (policy scores K candidate subsets, samples one, cuOpt executes it,
cost reduction = reward). Returns are discounted within each local-search
descent (segment boundaries detected via the iteration counter resetting).

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
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from model import CostPredictor, load_checkpoint


# ---------------------------------------------------------------------------
# Returns / REINFORCE update
# ---------------------------------------------------------------------------

def reinforce_update(model, optimizer, episodes, coords, demand, cap, device, args,
                     tw_features=None):
    """Full-feedback contextual-bandit policy gradient.

    Each step exposes the probe reward of all K arms. We normalize rewards
    within the state and maximize the expected reward J = sum_i p_i * adv_i
    (exact, low-variance), plus an entropy bonus. `top1_acc` tracks how often
    the policy's argmax arm is the truly best-probing arm (>> 1/K means
    learning).
    """
    all_steps = []
    for ep_trans in episodes:
        if not ep_trans:
            continue
        idxs = list(range(len(ep_trans)))
        if args.max_update_steps_per_ep and len(idxs) > args.max_update_steps_per_ep:
            idxs = list(np.random.choice(idxs, args.max_update_steps_per_ep, replace=False))
        all_steps.extend(ep_trans[i] for i in idxs)

    if not all_steps:
        return {"loss": 0.0, "entropy": 0.0, "grad_norm": 0.0,
                "top1_acc": 0.0, "rand_top1": 0.0, "n_update_steps": 0,
                "n_opt_steps": 0}

    coords_d = coords.to(device)
    dem_d = (demand / cap).to(device)
    tw_d = tw_features.to(device) if tw_features is not None else None

    eps = 1e-6

    def step_loss(t):
        """Full-feedback bandit loss for ONE step (forward batch = K). Returns
        (loss_with_grad, -J_value, entropy_value, top1, rand_top1)."""
        masks = t['masks'].to(device).long()
        valid = t['valid'].to(device)
        rewards = t['rewards'].to(device)
        K, L = masks.shape
        sol = t['sol'].to(device).long().unsqueeze(0).expand(K, -1)
        nodes = coords_d.expand(K, -1, -1)
        demands = dem_d.expand(K, -1)
        cost_0 = torch.full((K,), t['cost_0'], dtype=torch.float32, device=device)
        tw = tw_d.expand(K, -1, -1) if tw_d is not None else None

        scores = model(nodes, demands, sol, masks, cost_0, tw_features=tw)
        logits = (args.score_sign * scores / args.temperature).masked_fill(~valid, float('-inf'))
        log_probs = F.log_softmax(logits, dim=0)
        probs = log_probs.exp()

        rv = rewards[valid]
        if rv.numel() > 1 and rv.std() > eps:
            adv = (rewards - rv.mean()) / (rv.std() + eps)
        else:
            adv = rewards - (rv.mean() if rv.numel() > 0 else 0.0)
        adv = adv.masked_fill(~valid, 0.0)

        J = (probs * adv).sum()
        entropy = -(probs * log_probs).masked_fill(~valid, 0.0).sum()
        loss = -J - args.entropy_coef * entropy

        best_arm = rewards.masked_fill(~valid, float('-inf')).argmax()
        top1 = float((probs.argmax() == best_arm).item())
        rnd = 1.0 / float(valid.sum().item())
        return loss, float((-J).item()), float(entropy.item()), top1, rnd

    model.train()
    n = len(all_steps)
    accum = max(1, args.update_minibatch)   # steps accumulated per optimizer.step()

    # One epoch over all collected steps (shuffled). Pad the tail with random
    # resamples from the same pool so EVERY optimizer step uses exactly `accum`
    # steps (uniform gradient scale).
    order = np.random.permutation(n).tolist()
    rem = (-n) % accum
    if rem:
        order += list(np.random.choice(n, rem, replace=(n < rem)))

    n_opt_planned = len(order) // accum
    print(f"  [update] collected_steps={n} -> 1 epoch, "
          f"accumulate={accum} steps per update -> {n_opt_planned} optimizer steps",
          flush=True)

    total_loss = total_ent = total_top1 = total_rand = 0.0
    n_seen = 0
    n_opt = 0
    last_grad = 0.0
    t_upd0 = time.time()

    optimizer.zero_grad()
    for j, i in enumerate(order):
        loss, jval, ent, t1, rnd = step_loss(all_steps[int(i)])
        (loss / accum).backward()
        total_loss += jval; total_ent += ent
        total_top1 += t1; total_rand += rnd; n_seen += 1
        if (j + 1) % accum == 0:
            last_grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
            optimizer.step()
            optimizer.zero_grad()
            n_opt += 1
            print(f"    [optimizer_step {n_opt}/{n_opt_planned}] "
                  f"loss={total_loss / max(n_seen,1):.4f} "
                  f"top1_accuracy={total_top1 / max(n_seen,1):.3f} "
                  f"grad_norm={last_grad:.3f} "
                  f"elapsed_seconds={time.time() - t_upd0:.0f}",
                  flush=True)

    denom = max(n_seen, 1)
    return {
        "loss": total_loss / denom,
        "entropy": total_ent / denom,
        "grad_norm": last_grad,
        "top1_acc": total_top1 / denom,
        "rand_top1": total_rand / denom,
        "n_update_steps": n,
        "n_opt_steps": n_opt,
    }


# ---------------------------------------------------------------------------
# Parallel rollout launching
# ---------------------------------------------------------------------------

def launch_rollouts(specs, gpus, out_dir, args, weights_path, train, mode_run="policy",
                    progress=None):
    """specs: list of (tag, seed). Returns list of result dicts (loaded .pt).

    Runs in waves of len(gpus); each worker pinned to one GPU.
    If `progress` (a label string) is set, print one flushed line per finished
    rollout so the live tmux log shows collection/eval advancing.
    """
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
            cmd = [
                sys.executable, "rl_rollout.py",
                "--mode_run", mode_run,
                "--out", out_file,
                "--data_path", args.data_path,
                "--index", str(args.index),
                *(["--data_pt", args.data_pt] if args.data_pt else []),
                "--time_limit", str(args.time_limit),
                "--scale", str(args.scale),
                "--n_vehicles", str(args.n_vehicles),
                "--k", str(args.k),
                "--model_mode", args.mode,
                "--score_sign", str(args.score_sign),
                "--temperature", str(args.temperature),
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


def eval_policy(gpus, out_dir, args, weights_path):
    """Average final cost over args.eval_runs stochastic policy runs."""
    pol_specs = [(f"eval_pol_{i}", 10000 + i) for i in range(args.eval_runs)]
    pol = launch_rollouts(pol_specs, gpus, out_dir, args, weights_path, train=False,
                          mode_run="policy", progress="evaluate-policy")
    pol_costs = [r["final_cost"] for r in pol if r and r["final_cost"] is not None]
    return {
        "eval_policy_mean": float(np.mean(pol_costs)) if pol_costs else None,
        "eval_policy_min": float(np.min(pol_costs)) if pol_costs else None,
        "eval_policy_std": float(np.std(pol_costs)) if pol_costs else None,
        "eval_n": len(pol_costs),
    }


def eval_random_baseline(gpus, out_dir, args):
    """Average final cost over args.eval_runs random-subset runs.

    The random baseline is stationary (independent of the policy), so this is
    computed ONCE and reused as the fixed reference for the gap in every eval.
    """
    rnd_specs = [(f"eval_rnd_{i}", 30000 + i) for i in range(args.eval_runs)]
    rnd = launch_rollouts(rnd_specs, gpus, out_dir, args, None, train=False,
                          mode_run="random", progress="evaluate-random")
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


def manage_checkpoints(out_dir, rnd, save_every, save_fn):
    """Save this round's policy and prune snapshots per the retention policy.

    A snapshot for round r is KEPT if any holds:
      * r < FIRST_N_KEEP                  (first 10 rounds, permanent)
      * r % save_every == 0               (milestone rounds, permanent)
      * r > rnd - ROLLING_WINDOW          (within the latest-10 rolling window)
    Anything else is deleted as the rolling window slides forward.
    """
    save_fn(os.path.join(out_dir, f"policy_round{rnd}.pt"))
    for f in glob.glob(os.path.join(out_dir, "policy_round*.pt")):
        m = re.search(r"policy_round(\d+)\.pt$", os.path.basename(f))
        if not m:
            continue
        r = int(m.group(1))
        keep = (r < FIRST_N_KEEP) or (r % save_every == 0) or (r > rnd - ROLLING_WINDOW)
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
    parser.add_argument("--time_limit", type=float, default=5)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--batch_episodes", type=int, default=8, help="rollouts aggregated per update")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--scale", type=float, default=1e2)
    parser.add_argument("--n_vehicles", type=int, default=21)
    parser.add_argument("--k", type=int, default=32, help="K candidate subsets (CUOPT_RL_K)")
    parser.add_argument("--mode", type=str, default="v2", choices=["ratio", "new", "v2"])
    parser.add_argument("--backbone_ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--score_sign", type=float, default=-1.0)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--update_minibatch", type=int, default=16)
    parser.add_argument("--max_update_steps_per_ep", type=int, default=256,
                        help="subsample steps per episode for the update (0=use all)")
    parser.add_argument("--eval_runs", type=int, default=8, help="cuOpt runs to average per eval")
    parser.add_argument("--eval_every", type=int, default=10)
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
          f"K={args.k} eval_runs={args.eval_runs} time_limit={args.time_limit}")

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
    random_ref = None   # fixed random-subset baseline, computed once at first eval
    for rnd in range(args.rounds):
        save_weights(weights_path)

        print(f"[round {rnd}] collecting {args.batch_episodes} episodes on gpus={gpus} ...",
              flush=True)
        t0 = time.time()
        specs = [(f"r{rnd}_e{e}", rnd * 1000 + e) for e in range(args.batch_episodes)]
        rollouts = launch_rollouts(specs, gpus, out_dir, args, weights_path, train=True,
                                   mode_run="policy", progress=f"collect round {rnd}")
        roll_wall = time.time() - t0

        episodes = [r["transitions"] for r in rollouts if r and r.get("transitions")]
        final_costs = [r["final_cost"] for r in rollouts if r and r["final_cost"] is not None]
        n_steps_list = [r["n_steps"] for r in rollouts if r]
        all_rewards = [t['reward'] for ep in episodes for t in ep]

        t_upd = time.time()
        stats = reinforce_update(model, optimizer, episodes, coords1, demand1, cap, device, args,
                                 tw_features=tw_features)
        update_wall = time.time() - t_upd

        record = {
            "round": rnd,
            "train_cost_mean": float(np.mean(final_costs)) if final_costs else None,
            "train_cost_min": float(np.min(final_costs)) if final_costs else None,
            "best_known": best_known,
            "n_episodes": len(episodes),
            "mean_steps": float(np.mean(n_steps_list)) if n_steps_list else 0,
            "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "roll_wall_s": roll_wall,
            "update_wall_s": update_wall,
            **stats,
        }

        if rnd % args.eval_every == 0:
            if random_ref is None:
                print(f"[round {rnd}] computing random baseline ONCE ({args.eval_runs} runs) ...",
                      flush=True)
                random_ref = eval_random_baseline(gpus, out_dir, args)
                print(f"[baseline] random_mean={random_ref['eval_random_mean']:.4f} "
                      f"random_minimum={random_ref['eval_random_min']:.4f} (fixed reference)",
                      flush=True)
            print(f"[round {rnd}] evaluating ({args.eval_runs} policy runs) ...", flush=True)
            ev = eval_policy(gpus, out_dir, args, weights_path)
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
              f"top1_accuracy={stats['top1_acc']:.3f} "
              f"random_top1_accuracy={stats['rand_top1']:.3f} "
              f"grad_norm={stats['grad_norm']:.3f} "
              f"collected_steps={stats['n_update_steps']} "
              f"optimizer_steps={stats.get('n_opt_steps', 0)} "
              f"rollout_wall_seconds={roll_wall:.0f} "
              f"update_wall_seconds={update_wall:.1f}",
              flush=True)

        manage_checkpoints(out_dir, rnd, args.save_every, save_weights)
        cleanup_round_artifacts(out_dir, rnd)

    print(f"[done] best eval policy cost={best_eval}")


if __name__ == "__main__":
    main()
