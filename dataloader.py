import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from helper import load_instances_pkl, solution_flat_to_solution


def _load_basin_info(path: str) -> Dict[str, dict]:
    """Load basin_info.jsonl: hash -> record with precomputed route-like `solution`."""
    info: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            # Prefer legacy NeuOpt-style `solution_flat` if present; otherwise use `solution` as-is.
            if "solution_flat" in rec:
                rec["solution"] = solution_flat_to_solution(rec["solution_flat"])
                rec.pop("solution_flat", None)
            elif "solution" in rec:
                # Assume already route-like with depot=0.
                pass
            else:
                # Fallback: no solution info available.
                rec["solution"] = [0, 0]
            info[rec["hash"]] = rec
    return info


def _load_basin_pairs(path: str) -> List[dict]:
    """Load basin_pairs.jsonl: each line has anchor/neighbor hashes and a weight."""
    pairs: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pairs.append({
                "anchor_hash": rec["anchor_basin"]["hash"],
                "neighbor_hash": rec["neighbor_basin"]["hash"],
                "weight": float(rec["weight"]),
            })
    return pairs


def _load_distant_basins(path: str) -> List[dict]:
    """Load distant_basins.jsonl. Supports legacy and exp2 formats."""
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            anchor = rec["anchor_basin_hash"]
            distant_hashes = rec.get("distant_basin_hashes")
            if distant_hashes is None:
                distant_hashes = [db["basin_hash"] for db in rec["distant_basins"]]
            records.append({"anchor_basin_hash": anchor, "distant_basin_hashes": distant_hashes})
    return records


def parse_instance_indices(instance_indices: str) -> List[int]:
    """Parse '0-49' or '0,1,2' or '0-2,5' into list of ints."""
    indices: List[int] = []
    for part in instance_indices.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            indices.extend(range(int(a), int(b) + 1))
        elif part:
            indices.append(int(part))
    return indices


@dataclass
class TrainBatch:
    """One batch = one instance. hashes, pair_indices, weights; depot_xy, node_xy_demand, problem_size, dummy_size for that instance."""
    hashes: List[str]
    pair_indices: List[Tuple[int, int]]
    weights: List[float]
    instance_idx: int
    depot_xy: Optional[torch.Tensor] = None
    node_xy_demand: Optional[torch.Tensor] = None
    problem_size: int = 0
    dummy_size: int = 1
    include_mask: Optional[torch.Tensor] = None


@dataclass
class BasinData:
    """Result of load_basin_data: basin info, pairs, distant anchors, optional sign_mask for masked_in_batch."""
    indices: List[int]
    basin_info: Dict[str, dict]
    basin_pairs: List[dict]
    anchor_to_distant: Dict[str, List[str]]
    info_path_for_size: str
    data_dir: str
    sign_mask_np: Optional[np.ndarray]
    hash_to_sign_idx: Optional[Dict[str, int]]

def load_basin_data(
    instance_root: str,
    instance_indices: str,
    instance_prefix: str,
    neg_mode: str,
) -> BasinData:
    """Load basin_info, basin_pairs, anchor_to_distant, and optional sign_mask for masked_in_batch."""
    indices = parse_instance_indices(instance_indices)
    basin_info: Dict[str, dict] = {}
    basin_pairs: List[dict] = []
    anchor_to_distant: Dict[str, List[str]] = {}
    first_info_path: Optional[str] = None
    for idx in indices:
        inst_dir = os.path.join(instance_root, f"{instance_prefix}{idx}")
        info_path = os.path.join(inst_dir, "basin_info.jsonl")
        pairs_path = os.path.join(inst_dir, "basin_pairs.jsonl")
        distant_path = os.path.join(inst_dir, "distant_basins.jsonl")
        if not os.path.isfile(info_path):
            print(f"Warning: not found {info_path}, skipping index {idx}")
            continue
        if not os.path.isfile(pairs_path):
            print(f"Warning: not found {pairs_path}, skipping index {idx}")
            continue
        if first_info_path is None:
            first_info_path = info_path
        info_i = _load_basin_info(info_path)
        for h, v in info_i.items():
            v["instance_idx"] = idx
            basin_info[h] = v
        for p in _load_basin_pairs(pairs_path):
            p["instance_idx"] = idx
            basin_pairs.append(p)
        if os.path.isfile(distant_path):
            for rec in _load_distant_basins(distant_path):
                ah = rec["anchor_basin_hash"]
                anchor_to_distant.setdefault(ah, []).extend(rec["distant_basin_hashes"])
        else:
            print(f"Warning: not found {distant_path} (neg_mode=distant will have no distant for this instance)")
    if first_info_path is None:
        raise RuntimeError("No instance dir had both basin_info.jsonl and basin_pairs.jsonl; check paths.")
    data_dir = os.path.join(instance_root, f"{instance_prefix}{indices[0]}")

    sign_mask_np: Optional[np.ndarray] = None
    hash_to_sign_idx: Optional[Dict[str, int]] = None
    if neg_mode == "masked_in_batch":
        sign_mask_path = os.path.join(data_dir, "anchor_basin_sign_mask.npy")
        sign_order_path = os.path.join(data_dir, "anchor_basin_sign_order.json")
        if not os.path.isfile(sign_mask_path):
            raise FileNotFoundError(sign_mask_path)
        if not os.path.isfile(sign_order_path):
            raise FileNotFoundError(sign_order_path)
        sign_mask_np = np.load(sign_mask_path)
        with open(sign_order_path, "r", encoding="utf-8") as f:
            sign_order_list = json.load(f)
        hash_to_sign_idx = {h: i for i, h in enumerate(sign_order_list)}

    data = BasinData(
        indices=indices,
        basin_info=basin_info,
        basin_pairs=basin_pairs,
        anchor_to_distant=anchor_to_distant,
        info_path_for_size=first_info_path,
        data_dir=data_dir,
        sign_mask_np=sign_mask_np,
        hash_to_sign_idx=hash_to_sign_idx,
    )

    print(
        f"basin_info: {len(data.basin_info)} basins, basin_pairs: {len(data.basin_pairs)} pairs, "
        f"anchors_with_distant: {len(data.anchor_to_distant)}, neg_mode: {neg_mode}"
    )

    return data


class BasinPairDataset(Dataset):
    """Simple dataset over basin_pairs list."""

    def __init__(self, pairs: List[dict]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        return self.pairs[idx]


class GroupByInstanceBatchSampler(Sampler[List[int]]):
    """Yields batches of indices where all pairs have the same instance_idx (required for loss)."""

    def __init__(self, pairs: List[dict], batch_size: int, drop_last: bool = False, shuffle: bool = True):
        self.pairs = pairs
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self._groups: Dict[int, List[int]] = {}
        for i, p in enumerate(pairs):
            idx = p.get("instance_idx", 0)
            self._groups.setdefault(idx, []).append(i)

    def __iter__(self):
        batches: List[List[int]] = []
        for idx_list in self._groups.values():
            if self.shuffle:
                random.shuffle(idx_list)
            for start in range(0, len(idx_list), self.batch_size):
                batch = idx_list[start : start + self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)
                elif not self.drop_last:
                    # Pad the last chunk to full batch_size by sampling from this instance.
                    # This keeps "one batch = one instance" while avoiding smaller last batch.
                    if len(idx_list) == 0:
                        continue
                    need = self.batch_size - len(batch)
                    batch = batch + random.choices(idx_list, k=need)
                    batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        n = 0
        for idx_list in self._groups.values():
            n += len(idx_list) // self.batch_size if self.drop_last else (len(idx_list) + self.batch_size - 1) // self.batch_size
        return n


def _add_hashes(hashes: List[str], idx_map: Dict[str, int], h: str) -> None:
    if h not in idx_map:
        idx_map[h] = len(hashes)
        hashes.append(h)


def _collate_distant(
    batch: List[dict],
    anchor_to_distant: Dict[str, List[str]],
    max_negatives_per_anchor: int,
    instance_data_by_idx: Optional[Dict[int, Any]] = None,
) -> TrainBatch:
    instance_idx = batch[0].get("instance_idx", 0)
    hashes, idx_map = [], {}
    for item in batch:
        _add_hashes(hashes, idx_map, item["anchor_hash"])
        _add_hashes(hashes, idx_map, item["neighbor_hash"])
        for dh in anchor_to_distant.get(item["anchor_hash"], [])[:max_negatives_per_anchor]:
            _add_hashes(hashes, idx_map, dh)
    pair_indices = [(idx_map[it["anchor_hash"]], idx_map[it["neighbor_hash"]]) for it in batch]
    weights = [float(it["weight"]) for it in batch]
    depot_xy = node_xy_demand = None
    problem_size, dummy_size = 0, 1
    if instance_data_by_idx and instance_idx in instance_data_by_idx:
        inst = instance_data_by_idx[instance_idx]
        depot_xy, node_xy_demand = inst["depot_xy"], inst["node_xy_demand"]
        problem_size = inst["problem_size"]
    return TrainBatch(hashes=hashes, pair_indices=pair_indices, weights=weights, instance_idx=instance_idx, depot_xy=depot_xy, node_xy_demand=node_xy_demand, problem_size=problem_size, dummy_size=dummy_size, include_mask=None)


def _collate_masked_in_batch(
    batch: List[dict],
    sign_mask_np: np.ndarray,
    hash_to_sign_idx: Dict[str, int],
    instance_data_by_idx: Optional[Dict[int, Any]] = None,
) -> TrainBatch:
    instance_idx = batch[0].get("instance_idx", 0)
    hashes, idx_map = [], {}
    for item in batch:
        _add_hashes(hashes, idx_map, item["anchor_hash"])
        _add_hashes(hashes, idx_map, item["neighbor_hash"])
    n_pairs, n_emb = len(batch), len(hashes)
    include_mask = torch.ones(n_pairs, n_emb, dtype=torch.float32)
    sign_mask = torch.from_numpy(sign_mask_np)
    for i, item in enumerate(batch):
        a_idx = idx_map[item["anchor_hash"]]
        p_idx = idx_map[item["neighbor_hash"]]
        a_sidx = hash_to_sign_idx.get(item["anchor_hash"])
        for k, h_k in enumerate(hashes):
            if k == a_idx or k == p_idx:
                continue
            if a_sidx is not None and h_k in hash_to_sign_idx:
                k_sidx = hash_to_sign_idx[h_k]
                if sign_mask[a_sidx, k_sidx].item() == 1:
                    include_mask[i, k] = 0.0
    pair_indices = [(idx_map[it["anchor_hash"]], idx_map[it["neighbor_hash"]]) for it in batch]
    for i in range(n_pairs):
        include_mask[i, pair_indices[i][0]] = 1.0
        include_mask[i, pair_indices[i][1]] = 1.0
    weights = [float(it["weight"]) for it in batch]
    depot_xy = node_xy_demand = None
    problem_size, dummy_size = 0, 1
    if instance_data_by_idx and instance_idx in instance_data_by_idx:
        inst = instance_data_by_idx[instance_idx]
        depot_xy, node_xy_demand = inst["depot_xy"], inst["node_xy_demand"]
        problem_size = inst["problem_size"]
    return TrainBatch(hashes=hashes, pair_indices=pair_indices, weights=weights, instance_idx=instance_idx, depot_xy=depot_xy, node_xy_demand=node_xy_demand, problem_size=problem_size, dummy_size=dummy_size, include_mask=include_mask)


def build_basin_pair_loader(
    pairs: List[dict],
    batch_size: int,
    neg_mode: str = "distant",
    anchor_to_distant: Optional[Dict[str, List[str]]] = None,
    max_negatives_per_anchor: int = 5,
    sign_mask_np: Optional[np.ndarray] = None,
    hash_to_sign_idx: Optional[Dict[str, int]] = None,
    instance_data_by_idx: Optional[Dict[int, Any]] = None,
    drop_last: bool = False,
    **kwargs,
) -> DataLoader:
    """Batches grouped by instance (same instance per batch so loss is valid)."""
    if neg_mode == "masked_in_batch" and sign_mask_np is not None and hash_to_sign_idx is not None:
        fn = lambda b: _collate_masked_in_batch(b, sign_mask_np, hash_to_sign_idx, instance_data_by_idx)
    else:
        fn = lambda b: _collate_distant(b, anchor_to_distant or {}, max_negatives_per_anchor, instance_data_by_idx)
    batch_sampler = GroupByInstanceBatchSampler(pairs, batch_size, drop_last=drop_last, shuffle=True)
    return DataLoader(BasinPairDataset(pairs), batch_sampler=batch_sampler, collate_fn=fn, **kwargs)


def build_loader_from_args(args: Any, device: torch.device) -> Tuple[DataLoader, "BasinData"]:
    """Build basin-pair DataLoader and BasinData from trainer args. Input: args, device."""
    indices_str = args.instance_indices or "0"
    basin_data = load_basin_data(args.instance_root, indices_str, args.instance_prefix, args.neg_mode)
    indices = parse_instance_indices(indices_str)
    instance_data_list = load_instances_pkl(args.instance_pkl, device, indices, basin_data.basin_info)
    instance_data_by_idx = dict(zip(indices, instance_data_list))
    loader = build_basin_pair_loader(
        basin_data.basin_pairs,
        args.batch_size,
        neg_mode=args.neg_mode,
        anchor_to_distant=basin_data.anchor_to_distant,
        max_negatives_per_anchor=args.max_negatives,
        sign_mask_np=basin_data.sign_mask_np,
        hash_to_sign_idx=basin_data.hash_to_sign_idx,
        instance_data_by_idx=instance_data_by_idx,
    )
    print(f"Loaded {len(instance_data_list)} instances, {len(basin_data.basin_pairs)} pairs")
    return loader, basin_data
