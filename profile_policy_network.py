#!/usr/bin/env python3
"""Synthetic profiler for the RL policy network only.

This does not import cuOpt and does not run local search. It builds one shared
route state plus K candidate masks, then measures CostPredictor forward or
forward+backward latency and peak CUDA memory for old v2 versus v7 mode.

The default size models the planned scale-up: 1000 customers + depot and K=100.
On machines without CUDA, pass smaller --nodes/--vehicles/--k values for a CPU
sanity check.
"""

import argparse
import contextlib
import json
import math
import time
import traceback

import torch

from model import CostPredictor


def autocast_context(device, amp_dtype):
    if device.type != "cuda" or amp_dtype == "none":
        return contextlib.nullcontext()
    if amp_dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"unsupported amp dtype: {amp_dtype}")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_route(num_nodes, max_vehicles, max_length, device):
    """Build a padded incumbent route sequence compatible with current_sol."""
    route = []
    customer = 1
    customers_per_route = max(1, math.ceil(max(num_nodes - 1, 1) / max(max_vehicles, 1)))
    for r in range(max_vehicles):
        if len(route) >= max_length:
            break
        route.append(0 if r == 0 else num_nodes + r - 1)
        for _ in range(customers_per_route):
            if customer >= num_nodes or len(route) >= max_length:
                break
            route.append(customer)
            customer += 1
        if len(route) < max_length:
            route.append(num_nodes + r)
        if customer >= num_nodes and r >= max_vehicles - 1:
            break

    sol = torch.full((max_length,), -1, dtype=torch.long, device=device)
    active_len = min(len(route), max_length)
    if active_len:
        sol[:active_len] = torch.tensor(route[:active_len], dtype=torch.long, device=device)
    return sol, active_len


def make_inputs(args, device):
    torch.manual_seed(args.seed)
    max_length = args.nodes + args.vehicles * 4
    coords = torch.rand((1, args.nodes, 2), dtype=torch.float32, device=device)
    demand = torch.rand((1, args.nodes), dtype=torch.float32, device=device)
    demand[:, 0] = 0.0
    sol_1, active_len = build_route(args.nodes, args.vehicles, max_length, device)

    masks = torch.zeros((args.k, max_length), dtype=torch.long, device=device)
    active_positions = torch.arange(active_len, dtype=torch.long, device=device)
    n_selected = max(1, min(active_len, int(round(active_len * args.selection_frac))))
    op_bits = torch.tensor([1, 2, 4, 8], dtype=torch.long, device=device)
    for arm in range(args.k):
        perm = active_positions[torch.randperm(active_len, device=device)[:n_selected]]
        bits = op_bits[torch.randint(0, len(op_bits), (n_selected,), device=device)]
        masks[arm, perm] = bits

    nodes = coords.expand(args.k, -1, -1)
    demands = demand.expand(args.k, -1)
    current_sol = sol_1.unsqueeze(0).expand(args.k, -1)
    cost_0 = torch.full((args.k,), args.cost, dtype=torch.float32, device=device)
    return nodes, demands, current_sol, masks, cost_0, active_len


def run_mode(mode, args, device):
    result = {
        "mode": mode,
        "device": str(device),
        "nodes": args.nodes,
        "customers": args.nodes - 1,
        "vehicles": args.vehicles,
        "k": args.k,
        "backward": args.backward,
        "amp_dtype": args.amp_dtype,
        "status": "ok",
    }
    try:
        nodes, demands, current_sol, masks, cost_0, active_len = make_inputs(args, device)
        result["active_route_len"] = active_len
        model = CostPredictor(
            device=device,
            mode=mode,
            N=args.nodes,
            max_vehicles=args.vehicles,
            n_node_feat=3,
        )
        model.train(args.backward)

        def one_step():
            if args.backward:
                model.zero_grad(set_to_none=True)
                with autocast_context(device, args.amp_dtype):
                    scores = model(nodes, demands, current_sol, masks, cost_0)
                    loss = scores.float().mean()
                loss.backward()
                return loss.detach()
            with torch.no_grad(), autocast_context(device, args.amp_dtype):
                return model(nodes, demands, current_sol, masks, cost_0).float()

        for _ in range(args.warmup):
            one_step()
        sync(device)

        start_alloc = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            start_alloc = torch.cuda.memory_allocated(device)

        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            out = one_step()
            sync(device)
            times.append(time.perf_counter() - t0)

        result.update({
            "mean_ms": 1000.0 * sum(times) / len(times),
            "min_ms": 1000.0 * min(times),
            "max_ms": 1000.0 * max(times),
            "output_mean": float(out.float().mean().detach().cpu().item()),
        })
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device)
            result["peak_allocated_mb"] = peak / (1024 ** 2)
            result["peak_delta_mb"] = (peak - start_alloc) / (1024 ** 2)
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            result["status"] = "oom"
            result["error"] = message.splitlines()[0]
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            result["status"] = "error"
            result["error"] = message.splitlines()[0]
            result["traceback"] = traceback.format_exc()
    return result


def main():
    parser = argparse.ArgumentParser(description="Profile v2 vs v7 policy network.")
    parser.add_argument("--modes", nargs="+", default=["v2", "v7"],
                        choices=["ratio", "new", "v2", "v7", "flashv7"])
    parser.add_argument("--nodes", type=int, default=1001,
                        help="Number of nodes including depot; 1001 = 1000 customers + depot.")
    parser.add_argument("--vehicles", type=int, default=100)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--selection_frac", type=float, default=0.02)
    parser.add_argument("--cost", type=float, default=4000.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--backward", action="store_true",
                        help="Measure forward+backward instead of inference only.")
    parser.add_argument("--amp_dtype", choices=["none", "bf16"], default="bf16")
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json", default=None, help="Optional path to write JSON results.")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but torch.cuda.is_available() is false")
    if device.type == "cpu" and args.nodes >= 1001 and args.k >= 100:
        raise SystemExit(
            "Refusing target-scale CPU run. Use CUDA, or pass smaller "
            "--nodes/--vehicles/--k for a CPU sanity check."
        )

    env = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if device.type == "cuda":
        env.update({
            "cuda_device_name": torch.cuda.get_device_name(device),
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        })

    print(json.dumps({"environment": env}, sort_keys=True))
    results = [run_mode(mode, args, device) for mode in args.modes]
    for row in results:
        if row["status"] == "ok":
            mem = row.get("peak_delta_mb")
            mem_s = "NA" if mem is None else f"{mem:.1f}MB"
            print(
                f"{row['mode']:>5} status=ok active_L={row['active_route_len']} "
                f"mean={row['mean_ms']:.2f}ms min={row['min_ms']:.2f}ms "
                f"peak_delta={mem_s}"
            )
        else:
            print(f"{row['mode']:>5} status={row['status']} error={row.get('error')}")

    payload = {"environment": env, "results": results}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
