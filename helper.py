import pickle
from typing import Dict, List, Optional
import numpy as np
import torch
import sys
import shutil
from pathlib import Path
from typing import Set

def get_solution_with_dummy_depot(solution, problem_size):
    # solution.size: (batch, solution)
    batch_size, _ = solution.size()
    dummy_size = solution.size(-1) - problem_size
    solution = solution.clone()
    solution[solution != 0] += (dummy_size - 1)
    device = solution.device
    solution[solution == 0] = torch.arange(0, dummy_size, device=device).repeat(batch_size, 1).view(-1)
    return solution

def dummify(input: torch.Tensor, dummy_size: int, dim: int = 1) -> torch.Tensor:
    """input (B, problem_size, ...) -> output (B, problem_size+dummy_size, ...). Prepends dummy_size copies of input[:, :1]."""
    dummy = input[:, :1].repeat_interleave(dummy_size, dim=dim)
    return torch.cat([dummy, input], dim=dim)

def solution_flat_to_solution(solution_flat: List[int]) -> List[int]:
    """Convert basin `solution_flat` to a route-like visit sequence with depot=0."""
    customer_count = int(solution_flat[0]) - 1
    mapped = [0 if int(x) > customer_count else int(x) for x in solution_flat]
    seq = [0] + mapped + [0]
    out: List[int] = []
    for x in seq:
        if x == 0 and out and out[-1] == 0:
            continue
        out.append(x)
    return out

def rec2sol(rec):
    # input: rec (solution in linked list format)
    # reference: Ma, Yining, Zhiguang Cao, and Yeow Meng Chee. "Learning to search feasible and infeasible regions of routing problems with flexible neural k-opt." Advances in Neural Information Processing Systems 36 (2024).
    batch_size, seq_length = rec.size()
    visited_time = torch.zeros((batch_size, seq_length)).to(rec.device)
    pre = torch.zeros((batch_size), device=rec.device).long()
    for i in range(seq_length):
        visited_time[torch.arange(batch_size), rec[torch.arange(batch_size), pre]] = (i + 1)
        pre = rec[torch.arange(batch_size), pre]

    visited_time = visited_time % seq_length
    return visited_time.argsort()

def sol2rec(solution):
    # transform solution to linked list
    # solution.size: (batch, solution)
    solution_pre = solution
    solution_post = torch.cat((solution[:, 1:], solution[:, :1]), 1)

    rec = solution.clone()
    rec.scatter_(1, solution_pre, solution_post)
    return rec

def load_instances_pkl(
    instance_pkl: str, device: torch.device, indices: list, basin_info: Dict[str, dict]
) -> list:
    """Load CVRP instances from pkl. Returns list[dict] with depot_xy and node_xy_demand only (no dummy depots in coordinates)."""
    with open(instance_pkl, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        data = [data]
    out = []
    for i in indices:
        item = data[i]
        depot, loc, demand, capacity = item[0], item[1], item[2], item[3]
        depot_xy = torch.FloatTensor(np.atleast_2d(depot)).squeeze(0).to(device)
        if depot_xy.dim() == 1:
            depot_xy = depot_xy.unsqueeze(0)
        depot_xy = depot_xy.unsqueeze(0)  # (1,1,2)
        loc = torch.FloatTensor(loc).to(device)  # (problem_size,2)
        demand = torch.FloatTensor(demand).to(device) / float(capacity)  # (problem_size,)
        problem_size = loc.size(0)
        node_xy_demand = torch.cat([loc.unsqueeze(0), demand.unsqueeze(0).unsqueeze(-1)], dim=-1)  # (1,problem_size,3)
        out.append(
            {
                "depot_xy": depot_xy,               # (1,1,2)
                "node_xy_demand": node_xy_demand,   # (1,problem_size,3)
                "problem_size": problem_size,
            }
        )
    return out


def copy_all_src(dst_root: str, subdir: str = "src", home_dir: Optional[str] = None) -> str:
    """
    Copy all imported .py source files under home_dir into dst_root/subdir.

    - Only copies files whose absolute path is under home_dir.
    - Preserves relative paths to avoid name collisions.
    - Skips non-existent files.
    - Also copies the entry script (sys.argv[0]) if it is under home_dir.

    Returns the destination directory path.
    """
    dst_root_p = Path(dst_root).resolve()
    home = Path(home_dir).resolve() if home_dir else Path.cwd().resolve()
    dst_path = dst_root_p / subdir
    dst_path.mkdir(parents=True, exist_ok=True)

    def _is_under(p: Path, root: Path) -> bool:
        try:
            p.resolve().relative_to(root)
            return True
        except Exception:
            return False

    copied: Set[Path] = set()

    entry = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
    if entry and entry.exists() and entry.suffix == ".py" and _is_under(entry, home):
        rel = entry.relative_to(home)
        target = dst_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(entry), str(target))
        copied.add(entry)

    for _, mod in list(sys.modules.items()):
        src = getattr(mod, "__file__", None)
        if not src:
            continue
        try:
            p = Path(src).resolve()
        except Exception:
            continue
        if not p.exists() or p.suffix != ".py":
            continue
        if p in copied or not _is_under(p, home):
            continue
        rel = p.relative_to(home)
        target = dst_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(p), str(target))
        copied.add(p)

    return str(dst_path)