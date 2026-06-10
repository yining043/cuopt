"""Single-run benchmark for RL local-search subset selection.

Modes:
  random      - uniform random pick among K candidate subsets
  policy      - trained CostPredictor pick
  policy_eps  - epsilon-greedy trained CostPredictor pick

Policy modes default to greedy argmax. Use --selection sample to sample from
the policy softmax distribution.
"""

import argparse
import os
import random
import time

import numpy as np
import torch
from cuopt.routing import CustomizeNodesCallback

from model import CostPredictor
from rl_callback import RLPolicyCallback
from run_cuopt import (
    get_cuopt_model,
    pairwise_euclidean_distance,
    run_cuopt,
)
from load_nco_data import load_raw_data


class RandomSubsetCallback(CustomizeNodesCallback):
    """Pick one of K probe subsets uniformly at random."""

    def customize_nodes_to_search(self, solution_flat, num_routes,
                                  solution_cost, trail_masks_flat, num_trails,
                                  trail_rewards, iter):
        K = num_trails
        max_length = len(trail_masks_flat) // K
        valid = []
        for t in range(K):
            base = t * max_length
            if any(trail_masks_flat[base + i] > 0 for i in range(max_length)):
                valid.append(t)
        if not valid:
            return [0] * max_length
        idx = random.choice(valid)
        base = idx * max_length
        return [trail_masks_flat[base + i] for i in range(max_length)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["random", "policy", "policy_eps"], required=True)
    ap.add_argument("--weights", default=None,
                    help="Checkpoint path for policy modes; required unless --mode=random.")
    ap.add_argument("--data_path",
                    default="../../cuopt-examples/data/test_cvrp1000_hgs_n128_C250.txt")
    ap.add_argument("--data_pt", default=None,
                    help="Cached CVRPTW instance (.pt). When set, benchmarks CVRPTW with TW features.")
    ap.add_argument("--index", type=int, default=1)
    ap.add_argument("--time_limit", type=float, default=10)
    ap.add_argument("--scale", type=float, default=1e2)
    ap.add_argument("--n_vehicles", type=int, default=21)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_mode", default="v2")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--logit_clip", type=float, default=0.0,
                    help="Optional symmetric clamp applied to policy logits after temperature; <=0 disables.")
    ap.add_argument("--amp_dtype", choices=["none", "bf16"], default="none",
                    help="Autocast dtype for policy forward passes.")
    ap.add_argument("--epsilon", type=float, default=0.1,
                    help="epsilon-greedy for policy_eps mode (random arm with prob epsilon)")
    ap.add_argument("--score_sign", type=float, default=-1.0)
    ap.add_argument("--selection", choices=["greedy", "sample"], default="greedy",
                    help="Policy arm selection for eval: argmax greedy or softmax sampling.")
    args = ap.parse_args()
    if args.mode != "random" and not args.weights:
        ap.error("--weights is required for policy and policy_eps modes")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.environ["CUOPT_LS_MODE"] = "oracle"
    os.environ["CUOPT_RL_K"] = str(args.k)
    os.environ["CUOPT_RL_REWARD_HORIZON"] = "1"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tw_features = None
    if args.data_pt is not None:
        from load_nco_data import load_cvrptw_data, build_tw_features
        from run_cuopt import get_cuopt_model_tw
        inst = load_cvrptw_data(args.data_pt)
        coords1 = inst["coords"].unsqueeze(0)
        demand1 = inst["demand"].unsqueeze(0)
        cap = inst["capacity"]
        scale = inst["scale"]
        n_node_feat = 6
        max_vehicles = inst["n_vehicles"]
        tw_features = build_tw_features(inst)

        def build_model():
            return get_cuopt_model_tw(inst, inst["n_vehicles"], scale)
    else:
        raw_nodes, raw_cap, raw_demand, raw_cost, _ = load_raw_data(
            args.data_path, episode=1, begin_index=args.index)
        raw_dist = pairwise_euclidean_distance(raw_nodes)
        coords1 = raw_nodes[0:1]
        demand1 = raw_demand[0:1]
        cap = raw_cap[0].item()
        scale = args.scale
        n_node_feat = 3
        max_vehicles = args.n_vehicles

        def build_model():
            return get_cuopt_model(0, raw_dist, raw_demand, raw_cap, args.n_vehicles, args.scale)

    if args.mode == "random":
        callback = RandomSubsetCallback()
    else:
        model = CostPredictor(device=device, mode=args.model_mode, n_node_feat=n_node_feat,
                              max_vehicles=max_vehicles)
        sd = torch.load(args.weights, map_location=device)
        model.load_state_dict(sd.get("model_state_dict", sd))
        model.eval()
        eps = args.epsilon if args.mode == "policy_eps" else 0.0
        callback = RLPolicyCallback(
            model, coords1, demand1, cap, device,
            temperature=args.temperature, score_sign=args.score_sign,
            train=False, epsilon=eps, tw_features=tw_features,
            selection=args.selection, amp_dtype=args.amp_dtype,
            logit_clip=args.logit_clip)

    model_dm = build_model()
    wall_start = time.perf_counter()
    solution = run_cuopt(model_dm, args.time_limit, callback=callback)
    wall_time_sec = time.perf_counter() - wall_start
    if solution:
        cost = solution.get_total_objective() / scale
        print(f"[benchmark] final_cost={cost:.6f} mode={args.mode} seed={args.seed} "
              f"selection={args.selection} temperature={args.temperature} "
              f"k={args.k} model_mode={args.model_mode} amp_dtype={args.amp_dtype} "
              f"logit_clip={args.logit_clip} score_sign={args.score_sign} "
              f"wall_time_sec={wall_time_sec:.3f}")
    else:
        print(f"[benchmark] no feasible solution mode={args.mode} seed={args.seed} "
              f"selection={args.selection} temperature={args.temperature} "
              f"k={args.k} model_mode={args.model_mode} amp_dtype={args.amp_dtype} "
              f"logit_clip={args.logit_clip} score_sign={args.score_sign} "
              f"wall_time_sec={wall_time_sec:.3f}")


if __name__ == "__main__":
    main()
