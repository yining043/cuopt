import pickle
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import imageio.v2 as imageio
import sys
import shutil
from pathlib import Path
from typing import Set, Tuple

import random

def seed_everything(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def get_solution_with_dummy_depot(solution, problem_size):
    # solution.size: (batch, solution). Out-of-place only to avoid CUDA illegal instruction on in-place boolean indexing.
    batch_size, seq_len = solution.size()
    dummy_size = seq_len - problem_size
    device = solution.device
    dtype = solution.dtype
    mask_nonzero = solution != 0
    mask_zero = ~mask_nonzero
    # Step 1: non-zero -> value + (dummy_size - 1); zeros unchanged for now
    out = torch.where(mask_nonzero, solution + (dummy_size - 1), solution)
    # Step 2: fill zero positions with 0, 1, ..., dummy_size-1 per row (no in-place indexed assign)
    zero_fill = torch.arange(0, dummy_size, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
    # Index of k-th zero in each row: cumsum of mask_zero then -1 at zero positions
    zero_ord = (mask_zero.long().cumsum(1) - 1).clamp(min=0)
    out = torch.where(mask_zero, zero_fill.gather(1, zero_ord.clamp(max=dummy_size - 1)), out)
    return out

def dummify(input: torch.Tensor, dummy_size: int, dim: int = 1) -> torch.Tensor:
    """input (B, problem_size, ...) -> output (B, problem_size+dummy_size, ...). Prepends dummy_size copies of input[:, :1]."""
    dummy = input[:, :1].repeat_interleave(dummy_size, dim=dim)
    return torch.cat([dummy, input], dim=dim)

def solution_flat_to_solution(solution_flat: List[int]) -> List[int]:
    """Convert basin `solution_flat` to a route-like visit sequence with depot=0."""
    arr = np.asarray(solution_flat, dtype=np.int64)
    customer_count = int(arr[0]) - 1
    mapped = np.where(arr > customer_count, 0, arr)
    seq = np.concatenate([[0], mapped, [0]])
    # Drop 0 when previous element is 0 (remove consecutive duplicate zeros)
    keep = np.ones(len(seq), dtype=bool)
    keep[1:] = (seq[:-1] != 0) | (seq[1:] != 0)
    return seq[keep].tolist()


def hamming_distance(solution_a: List[int], solution_b: List[int]) -> int:
    """Hamming distance between two route-like solutions (pad with 0)."""
    la, lb = len(solution_a), len(solution_b)
    L = max(la, lb)
    a = solution_a + [0] * (L - la)
    b = solution_b + [0] * (L - lb)
    return sum(int(x != y) for x, y in zip(a, b))


def solution_to_routes(solution: List[int]) -> List[List[int]]:
    """Route-like visit sequence (depot=0) -> list of routes, each route [0, ..., 0]."""
    routes: List[List[int]] = []
    i = 0
    while i < len(solution):
        if solution[i] != 0:
            i += 1
            continue
        route = [0]
        i += 1
        while i < len(solution) and solution[i] != 0:
            route.append(int(solution[i]))
            i += 1
        if i < len(solution):
            route.append(0)
            i += 1
        if len(route) >= 2:
            routes.append(route)
    return routes


def _routes_to_adjacent_pairs(routes: List[List[int]]) -> Set[Tuple[int, int]]:
    """Extract adjacent node pairs (undirected, normalized) from routes."""
    pairs: Set[Tuple[int, int]] = set()
    for route in routes:
        for j in range(len(route) - 1):
            u, v = int(route[j]), int(route[j + 1])
            pairs.add((min(u, v), max(u, v)))
    return pairs


def solution_to_pairs(solution: List[int]) -> Set[Tuple[int, int]]:
    """Route-like solution -> set of adjacent (undirected) pairs. For caching in structure early stop."""
    return _routes_to_adjacent_pairs(solution_to_routes(solution))


def broken_pairs_ratio(solution_a: List[int], solution_b: List[int]) -> float:
    """
    Broken pairs ratio: fraction of adjacent pairs in solution_a that are not adjacent in solution_b.
    In [0, 1]; 0 = identical structure, 1 = no shared adjacent pairs.
    """
    pairs_a = solution_to_pairs(solution_a)
    pairs_b = solution_to_pairs(solution_b)
    broken = pairs_a - pairs_b
    return len(broken) / len(pairs_a) if pairs_a else 0.0


def broken_pairs_ratio_from_pairs(pairs_a: Set[Tuple[int, int]], pairs_b: Set[Tuple[int, int]]) -> float:
    """Ratio when pairs are precomputed (avoids repeated solution_to_routes)."""
    broken = pairs_a - pairs_b
    return len(broken) / len(pairs_a) if pairs_a else 0.0


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


def plot_embedding_2d(
    embeddings: Union[torch.Tensor, np.ndarray],
    instance_ids: Optional[Union[torch.Tensor, np.ndarray, List[int]]] = None,
    costs: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None,
    group_labels: Optional[Union[torch.Tensor, np.ndarray, List[int]]] = None,
    method: str = "pca",
    save_path: Optional[str] = None,
    figsize_per_plot: tuple = (5, 4),
) -> np.ndarray:
    """
    Visualize embeddings in 2D with PCA or t-SNE. Three subplots: color by instance_id, cost, group (basin/pair).

    Args:
        embeddings: (N, D) solution embeddings.
        instance_ids: (N,) int, which instance each solution belongs to.
        costs: (N,) float, cost of each solution (good vs bad).
        group_labels: (N,) int, e.g. 0=anchor 1=neighbour (stage1) or 0=anchor 1=positive (stage2).
        method: "pca" or "tsne".
        save_path: If set, save figure to this path.
        figsize_per_plot: (w, h) per subplot.

    Returns:
        coords: (N, 2) 2D coordinates used for plotting.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plot_embedding_2d")

    X = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else np.asarray(embeddings)
    N = X.shape[0]

    if method == "pca":
        X_centered = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        coords = (U[:, :2] * S[:2]).astype(np.float64)
    elif method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            raise ImportError("sklearn is required for method='tsne'. Install with: pip install scikit-learn")
        coords = TSNE(n_components=2, random_state=42, perplexity=min(30, N - 1)).fit_transform(X)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    n_plots = sum([instance_ids is not None, costs is not None, group_labels is not None])
    if n_plots == 0:
        n_plots = 1

    fig, axes = plt.subplots(1, n_plots, figsize=(figsize_per_plot[0] * n_plots, figsize_per_plot[1]))
    if n_plots == 1:
        axes = [axes]

    idx = 0

    if instance_ids is not None:
        ids = instance_ids.cpu().numpy() if isinstance(instance_ids, torch.Tensor) else np.asarray(instance_ids)
        uniq = np.unique(ids)
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(uniq), 1)))
        for i, uid in enumerate(uniq):
            mask = ids == uid
            axes[idx].scatter(coords[mask, 0], coords[mask, 1], c=[colors[i % len(colors)]], label=f"inst {uid}", s=8, alpha=0.7)
        axes[idx].set_title("By instance_id")
        axes[idx].legend(loc="best", fontsize=6)
        idx += 1

    if costs is not None:
        c = costs.cpu().numpy() if isinstance(costs, torch.Tensor) else np.asarray(costs)
        sc = axes[idx].scatter(coords[:, 0], coords[:, 1], c=c, s=8, alpha=0.7, cmap="viridis")
        plt.colorbar(sc, ax=axes[idx])
        axes[idx].set_title("By cost")
        idx += 1

    if group_labels is not None:
        g = group_labels.cpu().numpy() if isinstance(group_labels, torch.Tensor) else np.asarray(group_labels)
        uniq = np.unique(g)
        label_map = {0: "anchor", 1: "neighbour/positive", 2: "negative"}
        for i, ug in enumerate(uniq):
            mask = g == ug
            label = label_map.get(int(ug), f"group {ug}")
            axes[idx].scatter(coords[mask, 0], coords[mask, 1], label=label, s=8, alpha=0.7)
        axes[idx].set_title("By basin / pair (anchor vs neighbour/positive/negative)")
        axes[idx].legend(loc="best", fontsize=6)
        idx += 1

    if n_plots == 1 and idx == 0:
        axes[0].scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.7)
        axes[0].set_title(f"Embeddings ({method})")

    for ax in axes:
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return coords


def plot_distance_histogram(
    d_ap: Union[torch.Tensor, np.ndarray, List[float]],
    d_an: Union[torch.Tensor, np.ndarray, List[float]],
    save_path: Optional[str] = None,
    bins: int = 50,
    xlabel: str = "Euclidean distance",
    ylabel: str = "Frequency",
) -> None:
    """
    Plot distance distributions for positive pairs (A,P) and negative pairs (A,N).
    Green: d(A,P). Red: d(A,N). Well-separated peaks after training indicate good margin.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plot_distance_histogram")

    d_ap = np.asarray(d_ap).flatten()
    d_an = np.asarray(d_an).flatten()

    all_d = np.concatenate([d_ap, d_an])
    lo, hi = all_d.min(), all_d.max()
    if hi - lo < 1e-6:
        lo, hi = lo - 0.5, hi + 0.5
    bin_edges = np.linspace(lo, hi, bins + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(d_ap, bins=bin_edges, density=True, alpha=0.6, color="green", label="Positive (A,P)")
    ax.hist(d_an, bins=bin_edges, density=True, alpha=0.6, color="red", label="Negative (A,N)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title("Distance Histogram")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_distance_vs_similarity(
    embeddings: Union[torch.Tensor, np.ndarray],
    pair_indices: Union[torch.Tensor, np.ndarray, List[Tuple[int, int]]],
    similarity_values: Union[torch.Tensor, np.ndarray, List[float]],
    similarity_name: str = "True similarity",
    use_binned_mean: bool = False,
    n_bins: int = 20,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Plot embedding distance d(e_i, e_j) vs true similarity (cost diff, broken pairs ratio, same basin, etc.).
    Good embedding: small d with similar solutions (low broken pairs ratio, same basin), large d with different.
    Returns Pearson and Spearman correlation (negative corr if similarity = "higher = more similar").

    Args:
        embeddings: (N, D).
        pair_indices: (n_pairs, 2) row indices (i, j) for each pair.
        similarity_values: (n_pairs,) e.g. |cost_i - cost_j|, or edge Jaccard, or 0/1 same_basin.
        similarity_name: Y-axis label.
        use_binned_mean: if True, bin by distance and plot mean similarity per bin (smoother).
        n_bins: number of bins when use_binned_mean.
        save_path: optional figure path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plot_distance_vs_similarity")

    X = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else np.asarray(embeddings)
    pairs = np.asarray(pair_indices)
    if pairs.ndim == 1:
        pairs = pairs.reshape(-1, 2)
    sim = np.asarray(similarity_values).flatten()
    assert len(pairs) == len(sim), "pair_indices and similarity_values length must match"

    d = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)

    fig, ax = plt.subplots(figsize=(5, 4))
    if use_binned_mean:
        bins = np.percentile(d, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            bins = np.linspace(d.min(), d.max(), n_bins + 1)
        bin_ix = np.searchsorted(bins[1:-1], d)  # 0 .. n_bins-1
        bin_means_d = []
        bin_means_sim = []
        for b in range(len(bins) - 1):
            mask = bin_ix == b
            if mask.sum() > 0:
                bin_means_d.append(d[mask].mean())
                bin_means_sim.append(sim[mask].mean())
        if bin_means_d:
            ax.plot(bin_means_d, bin_means_sim, "o-", color="steelblue", linewidth=2, markersize=6)
        ax.set_xlabel("Embedding distance (bin mean)")
    else:
        ax.scatter(d, sim, s=5, alpha=0.5)
        ax.set_xlabel("Embedding distance d(e_i, e_j)")

    ax.set_ylabel(similarity_name)

    pearson = np.corrcoef(d, sim)[0, 1] if len(d) > 1 else 0.0
    try:
        from scipy.stats import spearmanr
        sp, _ = spearmanr(d, sim)
        spearman = float(sp) if not (sp != sp) else 0.0  # NaN check
    except ImportError:
        spearman = 0.0

    ax.set_title(f"Distance vs similarity (Pearson={pearson:.3f}, Spearman={spearman:.3f})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return {"pearson": float(pearson), "spearman": spearman}


def compute_recall_at_k(
    embeddings: Union[torch.Tensor, np.ndarray],
    anchor_indices: Union[torch.Tensor, np.ndarray, List[int]],
    positive_indices_per_anchor: Union[List[List[int]], List[Union[torch.Tensor, np.ndarray]]],
    k_values: Optional[List[int]] = None,
    exclude_self: bool = True,
) -> Dict[int, float]:
    """
    k-NN retrieval: for each anchor, "correct" = all its neighbours (stage1) or positive samples (stage2).
    Recall@k = mean over anchors of (|correct ∩ top-k| / |correct|). Anchors with 0 correct are skipped.

    Args:
        embeddings: (N, D) all solution embeddings.
        anchor_indices: (n_anchors,) row index in embeddings for each anchor.
        positive_indices_per_anchor: length n_anchors. positive_indices_per_anchor[i] = list of row
            indices that are correct neighbors for anchor i (can be many).
        k_values: list of k, e.g. [1, 5, 10, 20]. If None, use [1, 5, 10, 20].
        exclude_self: if True, exclude anchor itself from its k-NN.

    Returns:
        {k: recall_at_k}.
    """
    X = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else np.asarray(embeddings)
    a_ix = np.asarray(anchor_indices).flatten()
    n_anchors = len(a_ix)
    pos_sets = [
        set(np.asarray(p).flatten().tolist()) for p in positive_indices_per_anchor
    ]
    assert len(pos_sets) == n_anchors, "positive_indices_per_anchor length must match anchor_indices"
    if k_values is None:
        k_values = [1, 5, 10, 20]
    k_max = max(k_values)

    anchor_emb = X[a_ix]
    dists = np.linalg.norm(anchor_emb[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)
    if exclude_self:
        for i in range(n_anchors):
            dists[i, a_ix[i]] = np.inf

    topk = np.argsort(dists, axis=1)[:, :k_max]
    out: Dict[int, float] = {}
    for k in k_values:
        recalls = []
        for i in range(n_anchors):
            if not pos_sets[i]:
                continue
            hit = len(pos_sets[i] & set(topk[i, :k].tolist()))
            recalls.append(hit / len(pos_sets[i]))
        out[k] = sum(recalls) / len(recalls) if recalls else 0.0
    return out


def make_triplet_gif(
    coords: np.ndarray,
    roles: np.ndarray,
    triplet_ids: np.ndarray,
    role_label_map: Dict[int, str],
    role_color_map: Dict[int, str],
    title_prefix: str,
    gif_path: str,
    duration: float = 2.0,
) -> None:
    """Generic GIF helper: each frame highlights a single triplet/record in 2D.

    Args:
        coords: (N, 2) 2D coordinates from plot_embedding_2d.
        roles: (N,) int, group label per point (e.g. 0=anchor,1=pos,2=neg).
        triplet_ids: (N,) int, which triplet / record each point belongs to.
        role_label_map: mapping role -> legend label.
        role_color_map: mapping role -> matplotlib color.
        title_prefix: e.g. "S1 record" or "S2 triplet".
        gif_path: output GIF path.
        duration: seconds per frame.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[GIF] matplotlib not available, skip GIF.")
        return

    if coords.size == 0:
        return

    coords = np.asarray(coords)
    roles = np.asarray(roles)
    triplet_ids = np.asarray(triplet_ids)

    frames: List[np.ndarray] = []
    uniq_triplets = np.unique(triplet_ids)

    for t_idx in uniq_triplets:
        mask = triplet_ids == t_idx
        trip_coords = coords[mask]
        trip_roles = roles[mask]

        fig, ax = plt.subplots(figsize=(4, 4))
        for role in sorted(np.unique(trip_roles)):
            role_mask = trip_roles == role
            if not np.any(role_mask):
                continue
            color = role_color_map.get(int(role), "gray")
            label = role_label_map.get(int(role), f"group {int(role)}")
            ax.scatter(
                trip_coords[role_mask, 0],
                trip_coords[role_mask, 1],
                s=40,
                c=color,
                label=label,
            )

        ax.set_title(f"{title_prefix} {int(t_idx)}")
        ax.legend(loc="best", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf, dtype=np.uint8)[..., :3].copy()
        frames.append(frame)
        plt.close(fig)

    if frames:
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        imageio.mimsave(gif_path, frames, duration=duration)