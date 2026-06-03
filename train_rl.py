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

def reinforce_update(model, optimizer, episodes, coords, demand, cap, device, args):
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
                "top1_acc": 0.0, "rand_top1": 0.0, "n_update_steps": 0}

    coords_d = coords.to(device)
    dem_d = (demand / cap).to(device)

    model.train()
    optimizer.zero_grad()
    n = len(all_steps)
    total_loss, total_ent, total_top1, total_rand = 0.0, 0.0, 0.0, 0.0
    eps = 1e-6

    for start in range(0, n, args.update_minibatch):
        end = min(start + args.update_minibatch, n)
        mb_loss = 0.0
        for i in range(start, end):
            t = all_steps[i]
            masks = t['masks'].to(device).long()
            valid = t['valid'].to(device)
            rewards = t['rewards'].to(device)
            K, L = masks.shape
            sol = t['sol'].to(device).long().unsqueeze(0).expand(K, -1)
            nodes = coords_d.expand(K, -1, -1)
            demands = dem_d.expand(K, -1)
            cost_0 = torch.full((K,), t['cost_0'], dtype=torch.float32, device=device)

            scores = model(nodes, demands, sol, masks, cost_0)
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
            mb_loss = mb_loss - J - args.entropy_coef * entropy

            total_loss += float((-J).item())
            total_ent += float(entropy.item())
            # metrics
            best_arm = rewards.masked_fill(~valid, float('-inf')).argmax()
            total_top1 += float((probs.argmax() == best_arm).item())
            total_rand += 1.0 / float(valid.sum().item())
        (mb_loss / n).backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    return {
        "loss": total_loss / n,
        "entropy": total_ent / n,
        "grad_norm": float(grad_norm),
        "top1_acc": total_top1 / n,
        "rand_top1": total_rand / n,
        "n_update_steps": n,
    }


# ---------------------------------------------------------------------------
# Parallel rollout launching
# ---------------------------------------------------------------------------

def launch_rollouts(specs, gpus, out_dir, args, weights_path, train, mode_run="policy"):
    """specs: list of (tag, seed). Returns list of result dicts (loaded .pt).

    Runs in waves of len(gpus); each worker pinned to one GPU.
    """
    results = [None] * len(specs)
    worker_log_dir = os.path.join(out_dir, "worker_logs")
    os.makedirs(worker_log_dir, exist_ok=True)

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
                    print(f"[warn] failed to load {out_file}: {e}")
                    results[ridx] = None
            else:
                print(f"[warn] rollout {tag} produced no output (see worker log)")
                results[ridx] = None
    return results


def evaluate(gpus, out_dir, args, weights_path):
    """Average final cost over args.eval_runs stochastic runs for policy + origin."""
    pol_specs = [(f"eval_pol_{i}", 10000 + i) for i in range(args.eval_runs)]
    ori_specs = [(f"eval_ori_{i}", 20000 + i) for i in range(args.eval_runs)]
    pol = launch_rollouts(pol_specs, gpus, out_dir, args, weights_path, train=False, mode_run="policy")
    ori = launch_rollouts(ori_specs, gpus, out_dir, args, None, train=False, mode_run="origin")
    pol_costs = [r["final_cost"] for r in pol if r and r["final_cost"] is not None]
    ori_costs = [r["final_cost"] for r in ori if r and r["final_cost"] is not None]
    return {
        "eval_policy_mean": float(np.mean(pol_costs)) if pol_costs else None,
        "eval_policy_min": float(np.min(pol_costs)) if pol_costs else None,
        "eval_policy_std": float(np.std(pol_costs)) if pol_costs else None,
        "eval_origin_mean": float(np.mean(ori_costs)) if ori_costs else None,
        "eval_origin_min": float(np.min(ori_costs)) if ori_costs else None,
        "eval_n": len(pol_costs),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parallel REINFORCE for cuOpt node selection")
    parser.add_argument("--data_path", default="../../cuopt-examples/data/test_cvrp1000_hgs_n128_C250.txt")
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

    # Instance data for the master update (coords/demand only).
    from run_cuopt import pairwise_euclidean_distance  # noqa: F401 (torch-only helper)
    from load_nco_data import load_raw_data
    raw_nodes, raw_cap, raw_demand, raw_cost, raw_flag = load_raw_data(
        args.data_path, episode=1, begin_index=args.index)
    coords1 = raw_nodes[0:1]
    demand1 = raw_demand[0:1]
    cap = raw_cap[0].item()
    best_known = raw_cost[0].item()

    model = CostPredictor(device=device, mode=args.mode)
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
    for rnd in range(args.rounds):
        save_weights(weights_path)

        t0 = time.time()
        specs = [(f"r{rnd}_e{e}", rnd * 1000 + e) for e in range(args.batch_episodes)]
        rollouts = launch_rollouts(specs, gpus, out_dir, args, weights_path, train=True, mode_run="policy")
        roll_wall = time.time() - t0

        episodes = [r["transitions"] for r in rollouts if r and r.get("transitions")]
        final_costs = [r["final_cost"] for r in rollouts if r and r["final_cost"] is not None]
        n_steps_list = [r["n_steps"] for r in rollouts if r]
        all_rewards = [t['reward'] for ep in episodes for t in ep]

        stats = reinforce_update(model, optimizer, episodes, coords1, demand1, cap, device, args)

        record = {
            "round": rnd,
            "train_cost_mean": float(np.mean(final_costs)) if final_costs else None,
            "train_cost_min": float(np.min(final_costs)) if final_costs else None,
            "best_known": best_known,
            "n_episodes": len(episodes),
            "mean_steps": float(np.mean(n_steps_list)) if n_steps_list else 0,
            "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "roll_wall_s": roll_wall,
            **stats,
        }

        if rnd % args.eval_every == 0:
            ev = evaluate(gpus, out_dir, args, weights_path)
            record.update(ev)
            if ev["eval_policy_mean"] is not None and ev["eval_policy_mean"] < best_eval:
                best_eval = ev["eval_policy_mean"]
                save_weights(os.path.join(out_dir, "best_policy.pt"))
            gap = (None if not ev["eval_origin_mean"] else
                   (ev["eval_origin_mean"] - ev["eval_policy_mean"]) / ev["eval_origin_mean"])
            record["eval_gap_vs_origin"] = gap
            print(f"[eval @r{rnd}] policy={ev['eval_policy_mean']:.4f}(min {ev['eval_policy_min']:.4f}) "
                  f"origin={ev['eval_origin_mean']:.4f}(min {ev['eval_origin_min']:.4f}) "
                  f"gap={gap} best_known={best_known:.4f}")

        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if args.use_wandb:
            import wandb
            wandb.log(record)

        print(f"[r{rnd:4d}] train_cost={record['train_cost_mean']} "
              f"eps={len(episodes)} steps~{record['mean_steps']:.0f} R={record['mean_reward']:.5f} "
              f"loss={stats['loss']:.4f} ent={stats['entropy']:.3f} "
              f"top1={stats['top1_acc']:.3f}(rand{stats['rand_top1']:.3f}) "
              f"gnorm={stats['grad_norm']:.3f} updN={stats['n_update_steps']} wall={roll_wall:.0f}s")

        if (rnd + 1) % args.save_every == 0:
            save_weights(os.path.join(out_dir, f"policy_r{rnd + 1}.pt"))

    print(f"[done] best eval policy cost={best_eval}")


if __name__ == "__main__":
    main()
