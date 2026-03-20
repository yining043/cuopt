"""
Training script for CostPredictor (regression + concordance ranking)
Predicts cost[-1]/cost[0] ratio given nodes, demands, solution, and selected anchors.
Uses grouped sampling by state_id for within-state ranking.
Data: .npy directory from parse_dataset.py (mmap loading).

Loss modes:
  reg              - Huber regression on ratio
  concordance      - Pairwise logistic ranking within state
  reg_concordance  - reg + lambda * concordance

Metrics (12 × 3 groups all/easy/hard):
  mae, concordance, top1_hit, top3_hit, top_frac_hit, sample_m_top1,
  random_top1, random_top3, top1_lift, top3_lift, regret, model_vs_random
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import wandb
import argparse
import os
import math
import random
from datetime import datetime
from model import CostPredictor, load_checkpoint
import numpy as np


# ============================================================================
# Dataset
# ============================================================================

class MmapVRPDataset(Dataset):
    """Loads from a directory of .npy files (from parse_dataset.py); large arrays are memory-mapped."""
    def __init__(self, npy_dir):
        npy_dir = os.path.abspath(npy_dir)
        state_path = os.path.join(npy_dir, "state_id_tensor.npy")
        cost_path = os.path.join(npy_dir, "cost_tensor.npy")
        self.state_id_tensor = torch.from_numpy(np.load(state_path))
        self.previous_cost_tensor = torch.from_numpy(np.load(cost_path))
        self._len = len(self.state_id_tensor)
        self._nodes = np.load(os.path.join(npy_dir, "nodes_tensor.npy"), mmap_mode="r")
        self._demands = np.load(os.path.join(npy_dir, "demands_tensor.npy"), mmap_mode="r")
        self._current_sol = np.load(os.path.join(npy_dir, "current_sol_tensor.npy"), mmap_mode="r")
        self._selected = np.load(os.path.join(npy_dir, "selected_tensor.npy"), mmap_mode="r")
        print(f"MmapVRPDataset: {npy_dir} -> {self._len} samples (mmap)")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        prev_cost = self.previous_cost_tensor[idx]
        cost_0 = prev_cost[0]
        target_ratio = prev_cost[-1] / prev_cost[0]
        return {
            'nodes': torch.from_numpy(self._nodes[idx].copy()),
            'demands': torch.from_numpy(self._demands[idx].copy()),
            'current_sol': torch.from_numpy(self._current_sol[idx].copy()),
            'selected': torch.from_numpy(self._selected[idx].copy()),
            'cost_0': cost_0,
            'target_ratio': target_ratio,
            'cost_last': prev_cost[-1],
            'state_id': self.state_id_tensor[idx],
        }


# ============================================================================
# Batch Sampler
# ============================================================================

class GroupedBatchSampler:
    """Yields batches where each batch contains complete state groups.
    Each batch has num_groups states, each with up to max_trails trails."""
    def __init__(self, state_ids, num_groups_per_batch=16,
                 max_trails_per_group=16, num_batches=800):
        ids = np.asarray(state_ids)
        order = np.argsort(ids, kind='mergesort')
        sorted_ids = ids[order]
        breaks = np.flatnonzero(np.diff(sorted_ids)) + 1
        groups_idx = np.split(order, breaks)
        unique_ids = sorted_ids[np.concatenate(([0], breaks))]
        self.groups = {int(uid): grp.tolist() for uid, grp in zip(unique_ids, groups_idx)}
        self.valid_group_ids = [g for g, v in self.groups.items() if len(v) >= 2]
        self.num_groups_per_batch = num_groups_per_batch
        self.max_trails_per_group = max_trails_per_group
        self.num_batches = num_batches
        print(f"  GroupedBatchSampler: {len(self.valid_group_ids)} states with >=2 trails "
              f"(out of {len(self.groups)} total)")

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen = random.sample(
                self.valid_group_ids,
                min(self.num_groups_per_batch, len(self.valid_group_ids))
            )
            batch = []
            for gid in chosen:
                indices = self.groups[gid]
                if len(indices) > self.max_trails_per_group:
                    indices = random.sample(indices, self.max_trails_per_group)
                batch.extend(indices)
            yield batch

    def __len__(self):
        return self.num_batches


# ============================================================================
# Checkpoint
# ============================================================================

def save_checkpoint(model, optimizer, epoch, global_step, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))


# ============================================================================
# Loss Functions
# ============================================================================

def regression_loss(predicted, target):
    return F.huber_loss(predicted, target, delta=0.05)


def concordance_loss(predicted, target_ratio, state_ids, temperature=0.01, min_pair_spread=0.0):
    """Pairwise logistic ranking loss: directly optimizes within-state concordance."""
    unique_states = state_ids.unique()
    all_pair_losses = []

    for sid in unique_states:
        mask = (state_ids == sid)
        if mask.sum() < 2:
            continue

        group_pred = predicted[mask]
        group_target = target_ratio[mask]
        n = group_pred.size(0)

        diff_target = group_target.unsqueeze(0) - group_target.unsqueeze(1)
        diff_pred = group_pred.unsqueeze(0) - group_pred.unsqueeze(1)

        upper = torch.triu(torch.ones(n, n, device=predicted.device), diagonal=1).bool()
        dt = diff_target[upper]
        dp = diff_pred[upper]

        if min_pair_spread > 0:
            valid = dt.abs() > min_pair_spread
            if valid.sum() == 0:
                continue
            dt = dt[valid]
            dp = dp[valid]

        correct_direction = dt.sign() * dp
        pair_loss = F.softplus(-correct_direction / temperature)
        all_pair_losses.append(pair_loss.mean())

    if not all_pair_losses:
        return torch.tensor(0.0, device=predicted.device, requires_grad=True)

    return torch.stack(all_pair_losses).mean()


# ============================================================================
# Metrics
# ============================================================================

METRIC_KEYS = [
    'mae', 'r2', 'concordance', 'top1_hit', 'top3_hit', 'top_frac_hit',
    'sample_m_top1', 'random_top1', 'random_top3', 'top1_lift', 'top3_lift',
    'regret', 'model_vs_random',
]

GROUPS = ['all', 'easy', 'hard']


def compute_batch_metrics(preds, targets, state_ids, spread_threshold=0.0,
                          top_frac=0.3, sample_m=10):
    """Compute 13 metrics × 3 groups (all/easy/hard).
    Per-state metrics are averaged across states.  R² is computed globally
    per group (not per-state) so the denominator has meaningful variance.
    Returns (result_dict, r2_parts) where r2_parts holds raw ss_res/ss_tot
    for correct epoch-level accumulation.
    """
    device = preds.device
    group_data = {g: [] for g in GROUPS}
    group_preds = {g: [] for g in GROUPS}
    group_targets = {g: [] for g in GROUPS}

    for sid in state_ids.unique():
        mask = (state_ids == sid)
        p = preds[mask]
        t = targets[mask]
        n = int(p.numel())

        spread = (t.max() - t.min()).item() if n >= 2 else 0.0
        groups = ['all']
        if spread_threshold > 0:
            groups.append('easy' if spread > spread_threshold else 'hard')

        stat = {'mae': (p - t).abs().mean().item()}

        if n >= 2:
            diff_p = p.unsqueeze(0) - p.unsqueeze(1)
            diff_t = t.unsqueeze(0) - t.unsqueeze(1)
            upper = torch.triu(torch.ones(n, n, device=device), diagonal=1).bool()
            stat['concordance'] = (diff_p[upper].sign() == diff_t[upper].sign()).float().mean().item()

            true_best = t.argmin()
            pred_sorted = p.argsort()
            stat['top1_hit'] = float(true_best == pred_sorted[0])
            stat['top3_hit'] = float(true_best in pred_sorted[:min(3, n)])
            k_frac = max(1, int(math.ceil(top_frac * n)))
            stat['top_frac_hit'] = float(true_best in pred_sorted[:k_frac])

            stat['random_top1'] = 1.0 / n
            stat['random_top3'] = min(3, n) / n

            pred_best_idx = p.argmin()
            stat['regret'] = (t[pred_best_idx] - t[true_best]).item()
            stat['model_vs_random'] = (t.mean() - t[pred_best_idx]).item()

            if n >= sample_m:
                idx = random.sample(range(n), sample_m)
                idx_t = torch.tensor(idx, device=device, dtype=torch.long)
                stat['sample_m_top1'] = float(t[idx_t].argmin() == p[idx_t].argmin())

        for g in groups:
            group_data[g].append(stat)
            group_preds[g].append(p)
            group_targets[g].append(t)

    result = {}
    r2_parts = {}
    for g in GROUPS:
        prefix = f'{g}/'
        states = group_data[g]

        if not states:
            for key in METRIC_KEYS:
                result[prefix + key] = float('nan')
            continue

        def _mean(key):
            vals = [s[key] for s in states if key in s]
            return sum(vals) / len(vals) if vals else float('nan')

        result[prefix + 'mae'] = _mean('mae')

        if group_preds[g]:
            all_p = torch.cat(group_preds[g])
            all_t = torch.cat(group_targets[g])
            ss_res = ((all_p - all_t) ** 2).sum().item()
            ss_tot = ((all_t - all_t.mean()) ** 2).sum().item()
            result[prefix + 'r2'] = 1.0 - ss_res / (ss_tot + 1e-8) if ss_tot > 1e-8 else float('nan')
            r2_parts[f'{g}/ss_res'] = ss_res
            r2_parts[f'{g}/ss_tot'] = ss_tot
        else:
            result[prefix + 'r2'] = float('nan')

        result[prefix + 'concordance'] = _mean('concordance')
        result[prefix + 'top1_hit'] = _mean('top1_hit')
        result[prefix + 'top3_hit'] = _mean('top3_hit')
        result[prefix + 'top_frac_hit'] = _mean('top_frac_hit')
        result[prefix + 'sample_m_top1'] = _mean('sample_m_top1')
        result[prefix + 'random_top1'] = _mean('random_top1')
        result[prefix + 'random_top3'] = _mean('random_top3')

        t1 = result[prefix + 'top1_hit']
        r1 = result[prefix + 'random_top1']
        result[prefix + 'top1_lift'] = t1 - r1 if not (math.isnan(t1) or math.isnan(r1)) else float('nan')

        t3 = result[prefix + 'top3_hit']
        r3 = result[prefix + 'random_top3']
        result[prefix + 'top3_lift'] = t3 - r3 if not (math.isnan(t3) or math.isnan(r3)) else float('nan')

        result[prefix + 'regret'] = _mean('regret')
        result[prefix + 'model_vs_random'] = _mean('model_vs_random')

    return result, r2_parts


def _accumulate(running, counts, batch_metrics, r2_parts):
    for k, v in batch_metrics.items():
        if k.endswith('/r2'):
            continue
        if not math.isnan(v):
            running[k] = running.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
    for k, v in r2_parts.items():
        running[k] = running.get(k, 0.0) + v


def _finalize(running, counts):
    result = {k: running[k] / counts[k] for k in running
              if counts.get(k, 0) > 0 and not k.endswith('/ss_res') and not k.endswith('/ss_tot')}
    for g in GROUPS:
        ss_res = running.get(f'{g}/ss_res', 0.0)
        ss_tot = running.get(f'{g}/ss_tot', 0.0)
        result[f'{g}/r2'] = (1.0 - ss_res / ss_tot) if ss_tot > 1e-8 else float('nan')
    return result


def _fmt(v, width=6):
    if math.isnan(v):
        return "  N/A "
    return f"{v:{width}.4f}"


def _print_group(label, m, prefix):
    mae = m.get(f'{prefix}mae', float('nan'))
    r2 = m.get(f'{prefix}r2', float('nan'))
    conc = m.get(f'{prefix}concordance', float('nan'))
    t1 = m.get(f'{prefix}top1_hit', float('nan'))
    r1 = m.get(f'{prefix}random_top1', float('nan'))
    l1 = m.get(f'{prefix}top1_lift', float('nan'))
    t3 = m.get(f'{prefix}top3_hit', float('nan'))
    tf = m.get(f'{prefix}top_frac_hit', float('nan'))
    sm = m.get(f'{prefix}sample_m_top1', float('nan'))
    reg = m.get(f'{prefix}regret', float('nan'))
    mvr = m.get(f'{prefix}model_vs_random', float('nan'))
    print(f"  [{label:4s}] MAE:{_fmt(mae)} R²:{_fmt(r2)} | Conc:{_fmt(conc)} "
          f"| Top1:{_fmt(t1)} (rand{_fmt(r1)} lift{_fmt(l1, 7)}) "
          f"| Top3:{_fmt(t3)} | Frac30:{_fmt(tf)} | Sam10:{_fmt(sm)} "
          f"| Regret:{_fmt(reg)} | vsRand:{_fmt(mvr, 7)}")


# ============================================================================
# Train / Eval
# ============================================================================

def _compute_loss(predicted, target_ratio, state_ids, loss_type,
                  lambda_concordance, temperature, min_pair_spread):
    """Returns (loss, l_reg_value, l_conc_value)."""
    if loss_type == 'reg':
        loss = regression_loss(predicted, target_ratio)
        return loss, loss.item(), 0.0
    elif loss_type == 'concordance':
        loss = concordance_loss(predicted, target_ratio, state_ids, temperature, min_pair_spread)
        return loss, 0.0, loss.item()
    else:
        l_reg = regression_loss(predicted, target_ratio)
        l_conc = concordance_loss(predicted, target_ratio, state_ids, temperature, min_pair_spread)
        loss = l_reg + lambda_concordance * l_conc
        return loss, l_reg.item(), l_conc.item()


def train_one_epoch(model, train_loader, optimizer, device, epoch, global_step,
                    loss_type='reg_concordance', lambda_concordance=0.2,
                    temperature=0.01, min_pair_spread=0.0,
                    spread_threshold=0.0, top_frac=0.3, sample_m=10):
    model.train()
    total_loss = 0.0
    total_l_reg = 0.0
    total_l_conc = 0.0
    running = {}
    counts = {}
    num_batches = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        nodes = batch['nodes'].to(device)
        demands = batch['demands'].to(device)
        current_sol = batch['current_sol'].to(device)
        selected = batch['selected'].to(device)
        cost_0 = batch['cost_0'].to(device)
        target_ratio = batch['target_ratio'].to(device)
        state_ids = batch['state_id'].to(device)

        optimizer.zero_grad()
        predicted = model(nodes, demands, current_sol, selected, cost_0)
        loss, lr_val, lc_val = _compute_loss(
            predicted, target_ratio, state_ids, loss_type,
            lambda_concordance, temperature, min_pair_spread)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        global_step += 1
        total_loss += loss.item()
        total_l_reg += lr_val
        total_l_conc += lc_val
        num_batches += 1

        with torch.no_grad():
            batch_m, r2p = compute_batch_metrics(
                predicted.detach(), target_ratio, state_ids,
                spread_threshold, top_frac, sample_m)
        _accumulate(running, counts, batch_m, r2p)

        if global_step % 10 == 0:
            step_log = {
                "train/batch_loss": loss.item(),
                "train/batch_l_reg": lr_val,
                "train/batch_l_conc": lc_val,
                "global_step": global_step,
            }
            for g in GROUPS:
                for k in METRIC_KEYS:
                    key = f'{g}/{k}'
                    if key in batch_m and not math.isnan(batch_m[key]):
                        step_log[f"train/batch_{key}"] = batch_m[key]
            wandb.log(step_log)

    avg_loss = total_loss / num_batches
    avg_l_reg = total_l_reg / num_batches
    avg_l_conc = total_l_conc / num_batches
    avg_m = _finalize(running, counts)

    print(f"  Train - Loss: {avg_loss:.6f}  L_reg: {avg_l_reg:.6f} | L_conc: {avg_l_conc:.6f}")
    _print_group("ALL", avg_m, "all/")
    _print_group("EASY", avg_m, "easy/")
    _print_group("HARD", avg_m, "hard/")

    log = {
        "train/loss": avg_loss,
        "train/l_reg": avg_l_reg,
        "train/l_conc": avg_l_conc,
        "epoch": epoch + 1,
    }
    for g in GROUPS:
        for k in METRIC_KEYS:
            key = f'{g}/{k}'
            if key in avg_m:
                log[f"train/{key}"] = avg_m[key]
    wandb.log(log)

    return avg_loss, global_step, avg_m


def evaluate(model, val_loader, device, epoch,
             loss_type='reg_concordance', lambda_concordance=0.2,
             temperature=0.01, min_pair_spread=0.0,
             spread_threshold=0.0, top_frac=0.3, sample_m=10):
    model.eval()
    total_loss = 0.0
    total_l_reg = 0.0
    total_l_conc = 0.0
    running = {}
    counts = {}
    num_batches = 0

    for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
        nodes = batch['nodes'].to(device)
        demands = batch['demands'].to(device)
        current_sol = batch['current_sol'].to(device)
        selected = batch['selected'].to(device)
        cost_0 = batch['cost_0'].to(device)
        target_ratio = batch['target_ratio'].to(device)
        state_ids = batch['state_id'].to(device)

        with torch.no_grad():
            predicted = model(nodes, demands, current_sol, selected, cost_0)
            loss, lr_val, lc_val = _compute_loss(
                predicted, target_ratio, state_ids, loss_type,
                lambda_concordance, temperature, min_pair_spread)

            batch_m, r2p = compute_batch_metrics(
                predicted, target_ratio, state_ids,
                spread_threshold, top_frac, sample_m)

        total_loss += loss.item()
        total_l_reg += lr_val
        total_l_conc += lc_val
        num_batches += 1
        _accumulate(running, counts, batch_m, r2p)

    avg_loss = total_loss / num_batches
    avg_l_reg = total_l_reg / num_batches
    avg_l_conc = total_l_conc / num_batches
    avg_m = _finalize(running, counts)

    print(f"  Val   - Loss: {avg_loss:.6f}  L_reg: {avg_l_reg:.6f} | L_conc: {avg_l_conc:.6f}")
    _print_group("ALL", avg_m, "all/")
    _print_group("EASY", avg_m, "easy/")
    _print_group("HARD", avg_m, "hard/")

    log = {
        "val/loss": avg_loss,
        "val/l_reg": avg_l_reg,
        "val/l_conc": avg_l_conc,
        "epoch": epoch + 1,
    }
    for g in GROUPS:
        for k in METRIC_KEYS:
            key = f'{g}/{k}'
            if key in avg_m:
                log[f"val/{key}"] = avg_m[key]
    wandb.log(log)

    return avg_loss, avg_m


# ============================================================================
# Main
# ============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CostPredictor(device=device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    start_epoch = 0
    global_step = 0

    if args.load_weights:
        load_checkpoint(args.load_weights, model, optimizer=None)
        print(f"Loaded model weights from {args.load_weights} (fresh optimizer)")
    elif args.resume:
        start_epoch, global_step = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from {args.resume} at epoch {start_epoch + 1}")

    npy_dir = os.path.abspath(args.data)
    if not os.path.isfile(os.path.join(npy_dir, "state_id_tensor.npy")):
        print(f"Error: {npy_dir} has no state_id_tensor.npy")
        return
    dataset = MmapVRPDataset(npy_dir)

    all_state_ids = dataset.state_id_tensor.numpy()
    all_cost = dataset.previous_cost_tensor.numpy()
    ratio_all = all_cost[:, -1] / (all_cost[:, 0] + 1e-8)

    order = np.argsort(all_state_ids, kind='mergesort')
    sorted_sids = all_state_ids[order]
    breaks = np.flatnonzero(np.diff(sorted_sids)) + 1
    groups = np.split(order, breaks)
    unique_states = sorted_sids[np.concatenate(([0], breaks))]

    group_counts = np.array([len(g) for g in groups])
    group_min = np.array([ratio_all[g].min() for g in groups])
    group_max = np.array([ratio_all[g].max() for g in groups])
    group_spread = group_max - group_min

    keep = group_counts >= 2
    if args.min_spread > 0:
        keep &= group_spread >= args.min_spread
        print(f"Min-spread filter ({args.min_spread}): {keep.sum()}/{len(unique_states)} states kept")
        unique_states = unique_states[keep]
        group_spread = group_spread[keep]
    else:
        unique_states = unique_states[keep] if not keep.all() else unique_states

    valid_spreads = group_spread[keep] if args.min_spread <= 0 else group_spread
    spread_threshold = float(np.median(valid_spreads)) if len(valid_spreads) > 0 else 0.0
    print(f"Spread threshold (median): {spread_threshold:.6f}")

    rng = np.random.RandomState(42)
    rng.shuffle(unique_states)
    split = int(0.9 * len(unique_states))
    train_states = set(unique_states[:split].tolist())
    val_states = set(unique_states[split:].tolist())

    state_lookup = np.zeros(all_state_ids.max() + 1, dtype=np.int8)
    for s in train_states:
        state_lookup[s] = 1
    for s in val_states:
        state_lookup[s] = 2
    sample_labels = state_lookup[all_state_ids]
    train_indices = np.flatnonzero(sample_labels == 1).tolist()
    val_indices = np.flatnonzero(sample_labels == 2).tolist()

    print(f"State-based split: {len(train_states)} train states, {len(val_states)} val states")
    print(f"  Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_sampler = GroupedBatchSampler(
        all_state_ids[train_indices],
        num_groups_per_batch=args.groups_per_batch,
        max_trails_per_group=args.trails_per_group,
        num_batches=args.num_train_batches,
    )
    val_sampler = GroupedBatchSampler(
        all_state_ids[val_indices],
        num_groups_per_batch=args.groups_per_batch,
        max_trails_per_group=args.trails_per_group,
        num_batches=args.num_val_batches,
    )

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            num_workers=4, pin_memory=True)

    checkpoint_dir = args.output or os.path.join("outputs", args.runname)

    wandb.init(project="cuopt-reg", name=args.runname)
    wandb.config.update({
        "groups_per_batch": args.groups_per_batch,
        "trails_per_group": args.trails_per_group,
        "num_train_batches": args.num_train_batches,
        "num_val_batches": args.num_val_batches,
        "lambda_concordance": args.lambda_concordance,
        "learning_rate": args.lr,
        "num_epochs": args.epochs,
        "train_states": len(train_states),
        "val_states": len(val_states),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "model_params": num_params,
        "data_dir": npy_dir,
        "loss_type": args.loss,
        "temperature": args.temperature,
        "target": "cost[-1]/cost[0]",
        "min_spread": args.min_spread,
        "min_pair_spread": args.min_pair_spread,
        "spread_threshold": spread_threshold,
        "top_frac": args.top_frac,
        "sample_m": args.sample_m,
    })

    print(f"Device: {device}")
    print(f"Loss: {args.loss} | lambda_conc: {args.lambda_concordance} | temp: {args.temperature}")

    SAVE_CRITERIA = {
        'top1_lift':  {'key': 'all/top1_lift',  'higher_better': True},
        'concordance': {'key': 'all/concordance', 'higher_better': True},
        'regret':      {'key': 'all/regret',      'higher_better': False},
        'r2':          {'key': 'all/r2',          'higher_better': True},
    }
    best_vals = {name: (-float('inf') if c['higher_better'] else float('inf'))
                 for name, c in SAVE_CRITERIA.items()}

    shared_kwargs = dict(
        loss_type=args.loss,
        lambda_concordance=args.lambda_concordance,
        temperature=args.temperature,
        min_pair_spread=args.min_pair_spread,
        spread_threshold=spread_threshold,
        top_frac=args.top_frac,
        sample_m=args.sample_m,
    )

    for epoch in range(start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs} | LR: {current_lr:.2e}")
        print(f"{'='*60}")

        train_loss, global_step, train_m = train_one_epoch(
            model, train_loader, optimizer, device, epoch, global_step,
            **shared_kwargs)

        val_loss, val_m = evaluate(
            model, val_loader, device, epoch, **shared_kwargs)

        scheduler.step()
        wandb.log({"lr": current_lr, "epoch": epoch + 1})

        saved_reasons = []
        for name, crit in SAVE_CRITERIA.items():
            val = val_m.get(crit['key'], float('nan'))
            if math.isnan(val):
                continue
            improved = (val > best_vals[name]) if crit['higher_better'] else (val < best_vals[name])
            if improved:
                best_vals[name] = val
                saved_reasons.append(f"{name}={val:.4f}")

        if saved_reasons:
            save_checkpoint(model, optimizer, epoch + 1, global_step, checkpoint_dir)
            print(f"  ** Saved — improved: {', '.join(saved_reasons)} **")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch + 1, global_step, checkpoint_dir)

        vt1 = val_m.get('all/top1_hit', 0)
        vc = val_m.get('all/concordance', 0)
        vr = val_m.get('all/regret', 0)
        vr2 = val_m.get('all/r2', float('nan'))
        print(f"  Summary: Loss={val_loss:.6f} | Top1={vt1:.4f} | Conc={vc:.4f} | Regret={vr:.6f} | R²={vr2:.4f}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Path to .npy directory from parse_dataset.py")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--runname", type=str, default='cost_pred')
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training: load model + optimizer, continue from saved epoch")
    parser.add_argument("--load_weights", type=str, default=None,
                        help="Load only model weights; start from epoch 1 with fresh optimizer")
    parser.add_argument("--groups_per_batch", type=int, default=16)
    parser.add_argument("--trails_per_group", type=int, default=16)
    parser.add_argument("--num_train_batches", type=int, default=800)
    parser.add_argument("--num_val_batches", type=int, default=100)
    parser.add_argument("--lambda_concordance", type=float, default=0.2,
                        help="Weight for concordance loss in reg_concordance mode")
    parser.add_argument("--loss", type=str, default="reg_concordance",
                        choices=["reg", "concordance", "reg_concordance"],
                        help="Loss type: reg (Huber), concordance (pairwise ranking), reg_concordance (both)")
    parser.add_argument("--temperature", type=float, default=0.01,
                        help="Temperature for concordance loss (smaller = sharper)")
    parser.add_argument("--min_spread", type=float, default=0.0,
                        help="Min ratio spread within state to keep (0 = no filter)")
    parser.add_argument("--min_pair_spread", type=float, default=0.0,
                        help="For concordance: only use pairs with |ratio_i - ratio_j| > this")
    parser.add_argument("--top_frac", type=float, default=0.3,
                        help="Fraction for top_frac_hit metric (default 0.3 = top 30%%)")
    parser.add_argument("--sample_m", type=int, default=10,
                        help="m for sample_m_top1 metric: subsample m trails then check top1")
    args = parser.parse_args()

    args.runname = args.runname + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = args.output or os.path.join("outputs", args.runname)
    print(f"Checkpoint directory: {checkpoint_dir}")
    train(args)
