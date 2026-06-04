"""Generate + cache a random CVRPTW instance (same scale/size as the CVRP one).

The instance mirrors the 1000-customer CVRP setup (coords in [0,1], demand 1-9,
capacity 250, 21 vehicles, scale=1e2) and adds random time windows.

cuOpt validates time windows against the *primary cost matrix* (no separate
transit-time matrix is added), whose entries are ``euclidean_dist * scale``.
So time windows / service times are generated directly in those scaled units.

Time windows are reachability-aware: customer i's window [e_i, l_i] always lies
inside [depot->i travel time, H], guaranteeing a single-customer route is
feasible. The horizon H is generous so multi-customer routes stay feasible too.

Run:
  python gen_cvrptw.py --out data/cvrptw_inst1.pt --seed 1 --validate
"""

import argparse
import os

import numpy as np
import torch


def euclidean_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def generate_cvrptw(n_customers=1000, seed=0, capacity=250, scale=1e2,
                    n_vehicles=40, demand_lo=1, demand_hi=9,
                    width_frac=(0.35, 0.7), horizon_mult=10.0,
                    service_frac=0.01):
    rng = np.random.default_rng(seed)
    N = n_customers + 1

    coords = rng.random((N, 2)).astype(np.float32)
    demand = np.zeros(N, dtype=np.int64)
    demand[1:] = rng.integers(demand_lo, demand_hi + 1, size=n_customers)

    dist_scaled = euclidean_matrix(coords) * scale
    depot_t = dist_scaled[0]  # travel time depot -> i (scaled units)

    # Generous horizon: at least a full there-and-back for the farthest customer.
    H = max(horizon_mult * scale, float(depot_t.max() * 2 + 10.0))

    s_val = int(round(service_frac * scale))
    earliest = np.zeros(N, dtype=np.float64)
    latest = np.full(N, H, dtype=np.float64)
    service = np.zeros(N, dtype=np.int64)

    for i in range(1, N):
        t0 = float(depot_t[i])                 # cannot start before arrival
        w = rng.uniform(*width_frac) * H       # window width
        lo = t0
        hi = max(lo + 1.0, H - w)
        e = rng.uniform(lo, hi)
        l = min(H, e + w)
        earliest[i] = e
        latest[i] = l
        service[i] = s_val

    return {
        "coords": torch.tensor(coords, dtype=torch.float32),       # [N,2]
        "demand": torch.tensor(demand, dtype=torch.long),          # [N]
        "capacity": int(capacity),
        "earliest": torch.tensor(earliest.round().astype(np.int64)),  # [N]
        "latest": torch.tensor(latest.round().astype(np.int64)),      # [N]
        "service": torch.tensor(service, dtype=torch.long),           # [N]
        "H": float(H),
        "n_vehicles": int(n_vehicles),
        "scale": float(scale),
        "seed": int(seed),
    }


def validate(inst, time_limit=10.0):
    """Solve once with plain cuOpt (no callback) to confirm feasibility."""
    from run_cuopt import get_cuopt_model_tw, run_cuopt

    data_model = get_cuopt_model_tw(inst, inst["n_vehicles"], inst["scale"])
    sol = run_cuopt(data_model, time_limit, callback=None)
    if sol is None:
        print("[validate] NO FEASIBLE SOLUTION (status != 0)")
        return None
    cost = sol.get_total_objective() / inst["scale"]
    nveh = sol.get_vehicle_count()
    print(f"[validate] feasible: cost={cost:.4f} vehicles={nveh} "
          f"(time_limit={time_limit}s)")
    return cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cvrptw_inst1.pt")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n_customers", type=int, default=1000)
    ap.add_argument("--capacity", type=int, default=250)
    ap.add_argument("--scale", type=float, default=1e2)
    ap.add_argument("--n_vehicles", type=int, default=40)
    ap.add_argument("--horizon_mult", type=float, default=10.0)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--validate_time", type=float, default=10.0)
    args = ap.parse_args()

    inst = generate_cvrptw(
        n_customers=args.n_customers, seed=args.seed, capacity=args.capacity,
        scale=args.scale, n_vehicles=args.n_vehicles,
        horizon_mult=args.horizon_mult)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(inst, args.out)
    print(f"[gen] saved CVRPTW instance -> {args.out}")
    print(f"  customers={args.n_customers} capacity={inst['capacity']} "
          f"n_vehicles={inst['n_vehicles']} scale={inst['scale']} H={inst['H']:.1f}")
    print(f"  demand range [{int(inst['demand'][1:].min())},{int(inst['demand'][1:].max())}] "
          f"service={int(inst['service'][1])} "
          f"window width [{int((inst['latest']-inst['earliest'])[1:].min())},"
          f"{int((inst['latest']-inst['earliest'])[1:].max())}]")

    if args.validate:
        validate(inst, time_limit=args.validate_time)


if __name__ == "__main__":
    main()
