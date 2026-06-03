"""Single-run benchmark for RL node-selection (stdout -> plot_avg.py).

Modes:
  random  - uniform random pick among K candidate subsets (random baseline)
  policy  - trained CostPredictor argmax pick (best_policy.pt)
"""

import argparse
import os
import random

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
    ap.add_argument("--weights", default="outputs/rl_run2/best_policy.pt")
    ap.add_argument("--data_path",
                    default="../../cuopt-examples/data/test_cvrp1000_hgs_n128_C250.txt")
    ap.add_argument("--index", type=int, default=1)
    ap.add_argument("--time_limit", type=float, default=10)
    ap.add_argument("--scale", type=float, default=1e2)
    ap.add_argument("--n_vehicles", type=int, default=21)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_mode", default="v2")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--epsilon", type=float, default=0.1,
                    help="epsilon-greedy for policy_eps mode (random arm with prob epsilon)")
    ap.add_argument("--score_sign", type=float, default=-1.0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.environ["CUOPT_LS_MODE"] = "oracle"
    os.environ["CUOPT_RL_K"] = str(args.k)

    raw_nodes, raw_cap, raw_demand, raw_cost, _ = load_raw_data(
        args.data_path, episode=1, begin_index=args.index)
    raw_dist = pairwise_euclidean_distance(raw_nodes)
    coords1 = raw_nodes[0:1]
    demand1 = raw_demand[0:1]
    cap = raw_cap[0].item()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.mode == "random":
        callback = RandomSubsetCallback()
    else:
        model = CostPredictor(device=device, mode=args.model_mode)
        sd = torch.load(args.weights, map_location=device)
        model.load_state_dict(sd.get("model_state_dict", sd))
        eps = args.epsilon if args.mode == "policy_eps" else 0.0
        callback = RLPolicyCallback(
            model, coords1, demand1, cap, device,
            temperature=args.temperature, score_sign=args.score_sign,
            train=False, epsilon=eps)

    model_dm = get_cuopt_model(0, raw_dist, raw_demand, raw_cap, args.n_vehicles, args.scale)
    solution = run_cuopt(model_dm, args.time_limit, callback=callback)
    if solution:
        cost = solution.get_total_objective() / args.scale
        print(f"[benchmark] final_cost={cost:.6f} mode={args.mode} seed={args.seed}")
    else:
        print(f"[benchmark] no feasible solution mode={args.mode} seed={args.seed}")


if __name__ == "__main__":
    main()
