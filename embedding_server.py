#!/usr/bin/env python3
"""Unix domain socket server that serves solution embeddings for the C++ diversity manager.

Architecture:
  C++ adapted_sol.cuh  --(route-format solution)--> this server --(embedding)--> C++

Protocol (line-delimited text over Unix stream socket):
  Request:  "0 5 3 0 7 2 0\n"        (space-separated route-format ints, 0=depot)
  Response: "0.123 -0.456 ...\n"      (space-separated 128 floats)

Usage:
  python embedding_server.py \
    --checkpoint out/.../s2_epoch50.pt \
    --instance_pkl /path/to/cvrp100_uniform.pkl \
    --instance_index 0 \
    --socket_path /tmp/cuopt_embedding.sock \
    --device cuda
"""

import argparse
import os
import signal
import socket
import sys
import threading
import time

import pickle

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net import SolutionEmbedder
from CVRPEnv import CVRPEnv


MODEL_PARAMS = {
    "problem": "CVRP",
    "embedding_dim": 128,
    "encoder_layer_num": 3,
    "supplement_feature_dim": 5,
    "depot_feature_dim": 5,
    "node_feature_dim": 6,
    "head_num": 8,
    "qkv_dim": 16,
    "hidden_dim": 128,
    "use_l2_normalize": True,
}


def _load_instance_from_pkl(instance_pkl: str, instance_index: int):
    """Load a single CVRP instance from the standard pkl format.

    Each pkl entry: (depot[[x,y]], customers[[x,y],...], demands[...], capacity).
    Returns (depot_xy, customer_xy, customer_demand_normalized, problem_size).
    """
    with open(instance_pkl, "rb") as f:
        problems = pickle.load(f)
    depot, customers, demands, capacity = problems[instance_index]
    coords = [depot[0]] + customers                               # (n+1) x [x,y]
    all_demands = [0] + list(demands)                              # (n+1)
    coords_t = torch.tensor(coords, dtype=torch.float32)          # (n+1, 2)
    demands_t = torch.tensor(all_demands, dtype=torch.float32)    # (n+1,)

    depot_xy = coords_t[0:1].unsqueeze(0)                         # (1,1,2)
    customer_xy = coords_t[1:]                                     # (problem_size, 2)
    customer_demand = demands_t[1:] / float(capacity)
    node_xy_demand = torch.cat([
        customer_xy.unsqueeze(0),
        customer_demand.unsqueeze(0).unsqueeze(-1),
    ], dim=-1)                                                     # (1, problem_size, 3)
    return depot_xy, node_xy_demand, customer_xy.shape[0]


class EmbeddingService:
    """Holds model + env; computes embedding for a single route-format solution."""

    def __init__(self, checkpoint_path: str, instance_pkl: str, instance_index: int, device: str):
        self.device = torch.device(device)

        # Load model
        self.embedder = SolutionEmbedder(MODEL_PARAMS).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if "embedder_state" in ckpt:
            self.embedder.load_state_dict(ckpt["embedder_state"])
        elif "encoder_state" in ckpt:
            self.embedder.encoder.load_state_dict(ckpt["encoder_state"])
            print("WARNING: legacy checkpoint (encoder_state only); pos_encoder randomly initialized")
        self.embedder.eval()
        print(f"Loaded embedder from {checkpoint_path} "
              f"(stage {ckpt.get('stage', '?')}, epoch {ckpt.get('epoch', '?')})")

        # Load instance
        depot_xy, node_xy_demand, self.problem_size = _load_instance_from_pkl(
            instance_pkl, instance_index
        )
        self.env = CVRPEnv(self.problem_size, self.device)
        self.env.load(depot_xy.to(self.device), node_xy_demand.to(self.device), basin_info={})
        print(f"Loaded instance {instance_index} from {instance_pkl} "
              f"(problem_size={self.problem_size})")

    @torch.no_grad()
    def embed(self, route_solution: list) -> np.ndarray:
        """Compute 128-dim embedding for a route-format solution [0, c1, c2, 0, ...]."""
        h = "_srv_tmp"
        self.env._basin_info[h] = {"solution": route_solution}
        ctx = self.env.prepare_from_hashes([h])
        emb = self.embedder(ctx, self.env)  # (1, 128)
        del self.env._basin_info[h]
        return emb.squeeze(0).cpu().numpy()


def handle_client(conn: socket.socket, service: EmbeddingService):
    """Handle one persistent connection: read lines, respond with embeddings."""
    buf = b""
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    route_sol = list(map(int, line.split()))
                    emb = service.embed(route_sol)
                    resp = " ".join(f"{v:.8f}" for v in emb) + "\n"
                    conn.sendall(resp.encode("utf-8"))
                except Exception as e:
                    conn.sendall(f"ERROR {e}\n".encode("utf-8"))
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Embedding socket server for cuOpt diversity")
    parser.add_argument("--checkpoint", required=True, help="Path to model .pt checkpoint")
    parser.add_argument("--instance_pkl", required=True, help="Path to CVRP instances pickle")
    parser.add_argument("--instance_index", type=int, default=0)
    parser.add_argument("--socket_path", default="/tmp/cuopt_embedding.sock")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    service = EmbeddingService(args.checkpoint, args.instance_pkl, args.instance_index, args.device)

    if os.path.exists(args.socket_path):
        os.unlink(args.socket_path)

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.socket_path)
    srv.listen(4)
    print(f"Embedding server listening on {args.socket_path}")

    def cleanup(signum, frame):
        print("\nShutting down...")
        srv.close()
        if os.path.exists(args.socket_path):
            os.unlink(args.socket_path)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        while True:
            conn, _ = srv.accept()
            t = threading.Thread(target=handle_client, args=(conn, service), daemon=True)
            t.start()
    except KeyboardInterrupt:
        cleanup(None, None)


if __name__ == "__main__":
    main()
