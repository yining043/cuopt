#!/usr/bin/env python3
"""
One-time conversion: ml_data_grpo.pt -> directory of .npy files for memory-mapped loading.
Run on a machine with enough RAM to load the full .pt (e.g. 120GB).
Usage: python convert_pt_to_npy.py ml_data_grpo.pt [output_dir]
       Default output_dir is ml_data_grpo_npy/
"""
import os
import sys
import numpy as np
import torch

def main():
    pt_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else pt_path.replace(".pt", "_npy")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {pt_path} (this may use a lot of RAM)...")
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    keys = [
        "nodes_tensor",
        "demands_tensor",
        "current_sol_tensor",
        "selected_tensor",
        "cost_tensor",
        "state_id_tensor",
    ]
    for key in keys:
        if key not in data:
            print(f"  Warning: {key} not in .pt, skip")
            continue
        t = data[key]
        if isinstance(t, torch.Tensor):
            arr = t.numpy()
        else:
            arr = np.array(t)
        out_path = os.path.join(out_dir, f"{key}.npy")
        np.save(out_path, arr)
        print(f"  Saved {key} -> {out_path} shape {arr.shape} dtype {arr.dtype}")
        del data[key]
    del data
    print(f"Done. Training: use --data {out_dir}")

if __name__ == "__main__":
    main()