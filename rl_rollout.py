"""Single-episode rollout worker for parallel RL training.

Runs exactly one cuOpt Solve on the GPU assigned via CUDA_VISIBLE_DEVICES,
using the current policy weights (inference only), and saves the collected
transitions (+ final cost) to --out. The master process aggregates many of
these into one REINFORCE update.

Running each cuOpt instance in its own process with a single visible GPU keeps
CUDA contexts clean and lets us use all GPUs in parallel.
"""

import argparse
import os
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode_run", choices=["policy", "origin", "random"], default="policy")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--data_pt", default=None,
                    help="If set, load a cached CVRPTW instance (.pt) instead of the line-format CVRP data.")
    ap.add_argument("--index", type=int, default=1)
    ap.add_argument("--time_limit", type=float, default=5)
    ap.add_argument("--scale", type=float, default=1e2)
    ap.add_argument("--n_vehicles", type=int, default=21)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--model_mode", default="v2")
    ap.add_argument("--score_sign", type=float, default=-1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--train", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from model import CostPredictor
    from rl_callback import RLPolicyCallback, RandomSubsetCallback
    from run_cuopt import (pairwise_euclidean_distance, get_cuopt_model,
                           get_cuopt_model_tw, run_cuopt)
    from load_nco_data import load_raw_data, load_cvrptw_data, build_tw_features

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cvrptw = args.data_pt is not None
    tw_features = None
    if cvrptw:
        inst = load_cvrptw_data(args.data_pt)
        coords1 = inst["coords"].unsqueeze(0)
        demand1 = inst["demand"].unsqueeze(0)
        cap = inst["capacity"]
        n_vehicles = inst["n_vehicles"]
        scale = inst["scale"]
        tw_features = build_tw_features(inst)
        n_node_feat = 6

        def build_model():
            return get_cuopt_model_tw(inst, n_vehicles, scale)
    else:
        raw_nodes, raw_cap, raw_demand, raw_cost, raw_flag = load_raw_data(
            args.data_path, episode=1, begin_index=args.index)
        raw_dist = pairwise_euclidean_distance(raw_nodes)
        coords1 = raw_nodes[0:1]
        demand1 = raw_demand[0:1]
        cap = raw_cap[0].item()
        n_vehicles = args.n_vehicles
        scale = args.scale
        n_node_feat = 3

        def build_model():
            return get_cuopt_model(0, raw_dist, raw_demand, raw_cap, n_vehicles, scale)

    if args.mode_run == "origin":
        os.environ.pop("CUOPT_LS_MODE", None)
        cuopt_model = build_model()
        t0 = time.time()
        sol = run_cuopt(cuopt_model, args.time_limit, callback=None)
        wall = time.time() - t0
        final = sol.get_total_objective() / scale if sol else None
        torch.save({"final_cost": final, "wall": wall, "n_steps": 0, "transitions": []}, args.out)
        return

    if args.mode_run == "random":
        # Same oracle K-candidate mechanism as the policy, but pick uniformly at
        # random among the valid subsets -> the apples-to-apples baseline.
        os.environ["CUOPT_LS_MODE"] = "oracle"
        os.environ["CUOPT_RL_K"] = str(args.k)
        cb = RandomSubsetCallback()
        cuopt_model = build_model()
        t0 = time.time()
        sol = run_cuopt(cuopt_model, args.time_limit, callback=cb)
        wall = time.time() - t0
        final = sol.get_total_objective() / scale if sol else None
        torch.save({"final_cost": final, "wall": wall, "n_steps": 0, "transitions": []}, args.out)
        return

    os.environ["CUOPT_LS_MODE"] = "oracle"
    os.environ["CUOPT_RL_K"] = str(args.k)

    model = CostPredictor(device=device, mode=args.model_mode, n_node_feat=n_node_feat,
                          max_vehicles=n_vehicles)
    if args.weights and os.path.exists(args.weights):
        sd = torch.load(args.weights, map_location=device)
        model.load_state_dict(sd.get("model_state_dict", sd))
    model.eval()

    cb = RLPolicyCallback(
        model, coords1, demand1, cap, device,
        temperature=args.temperature, score_sign=args.score_sign,
        train=bool(args.train), tw_features=tw_features)
    cuopt_model = build_model()
    t0 = time.time()
    sol = run_cuopt(cuopt_model, args.time_limit, callback=cb)
    wall = time.time() - t0
    final = sol.get_total_objective() / scale if sol else None

    torch.save({
        "final_cost": final,
        "wall": wall,
        "n_steps": len(cb.transitions),
        "transitions": cb.transitions if args.train else [],
    }, args.out)


if __name__ == "__main__":
    main()
