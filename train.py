"""
Training script for CostPredictor (regression + GRPO ranking)
Predicts cost[-1]/cost[0] ratio given nodes, demands, solution, and selected anchors.
Uses grouped sampling by state_id and GRPO-style within-state ranking loss.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from tqdm import tqdm
import wandb
import argparse
import os
import glob
import random
from datetime import datetime
from model import CostPredictor
import numpy as np


class VRPDataset(Dataset):
    def __init__(self, datafile):
        print(f"Loading data from {datafile}...")
        data = torch.load(datafile)
        self.nodes_tensor = data["nodes_tensor"]
        self.demands_tensor = data["demands_tensor"]
        self.current_sol_tensor = data["current_sol_tensor"]
        self.selected_tensor = data["selected_tensor"]
        self.previous_cost_tensor = data["cost_tensor"]  # [T, 4]
        self.state_id_tensor = data["state_id_tensor"]   # [T]

    def __len__(self):
        return self.nodes_tensor.shape[0]

    def __getitem__(self, idx):
        prev_cost = self.previous_cost_tensor[idx]
        cost_0 = prev_cost[0]
        target_ratio = prev_cost[-1] / prev_cost[0]
        return {
            'nodes': self.nodes_tensor[idx],
            'demands': self.demands_tensor[idx],
            'current_sol': self.current_sol_tensor[idx],
            'selected': self.selected_tensor[idx],
            'cost_0': cost_0,
            'target_ratio': target_ratio,
            'cost_last': prev_cost[-1],
            'state_id': self.state_id_tensor[idx],
        }


class MmapVRPDataset(Dataset):
    """Loads from a directory of .npy files (from convert_pt_to_npy.py); large arrays are memory-mapped."""
    def __init__(self, npy_dir):
        npy_dir = os.path.abspath(npy_dir)
        # Small tensors: load into RAM
        state_path = os.path.join(npy_dir, "state_id_tensor.npy")
        cost_path = os.path.join(npy_dir, "cost_tensor.npy")
        self.state_id_tensor = torch.from_numpy(np.load(state_path))
        self.previous_cost_tensor = torch.from_numpy(np.load(cost_path))
        self._len = len(self.state_id_tensor)
        # Large tensors: memory-mapped
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


class GroupedBatchSampler:
    """Yields batches where each batch contains complete state groups.
    Each batch has num_groups states, each with up to max_trails trails."""
    def __init__(self, state_ids, num_groups_per_batch=16,
                 max_trails_per_group=16, num_batches=800):
        self.groups = {}
        for idx, sid in enumerate(state_ids):
            sid_val = int(sid) if hasattr(sid, 'item') else sid
            self.groups.setdefault(sid_val, []).append(idx)
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


def save_checkpoint(model, optimizer, epoch, global_step, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))


def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint.get('global_step', 0)


def loss_function(predicted, target):
    return F.huber_loss(predicted, target, delta=0.05)


def grpo_loss(predicted, target_ratio, state_ids):
    """GRPO-style within-group ranking loss.
    For each state group: advantage_i = (mean_ratio - ratio_i) / (std + eps)
    Loss = mean over groups of mean(advantage * pred)
    Minimizing pushes pred down for good trials (low ratio), up for bad ones.
    """
    unique_states = state_ids.unique()
    group_losses = []

    for sid in unique_states:
        mask = (state_ids == sid)
        if mask.sum() < 2:
            continue

        group_pred = predicted[mask]
        group_target = target_ratio[mask]

        mean_r = group_target.mean()
        std_r = group_target.std().clamp(min=1e-4)
        advantage = (mean_r - group_target) / std_r

        group_losses.append((advantage * group_pred).mean())

    if not group_losses:
        return torch.tensor(0.0, device=predicted.device, requires_grad=True)

    return torch.stack(group_losses).mean()


def soft_concordance_loss(predicted, target_ratio, state_ids, temperature=0.01, min_pair_spread=0.0):
    """Pairwise logistic ranking loss: directly optimizes within-state concordance.
    For each ordered pair (i,j) where ratio_i < ratio_j (i is better),
    loss = -log sigmoid((pred_j - pred_i) / temperature).
    Minimizing pushes pred_i < pred_j, aligning with argmin at inference.
    If min_pair_spread > 0, only pairs with |target_i - target_j| > min_pair_spread are used.
    """
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

        # Only keep pairs with |dt| > min_pair_spread
        if min_pair_spread > 0:
            valid = dt.abs() > min_pair_spread
            if valid.sum() == 0:
                continue
            dt = dt[valid]
            dp = dp[valid]

        # dt > 0 means j has higher ratio (j is worse), so we want pred_j > pred_i
        # dt < 0 means j has lower ratio (j is better), so we want pred_j < pred_i
        # Unify: for each pair, the "correct direction" is sign(dt) * dp > 0
        correct_direction = dt.sign() * dp
        pair_loss = F.softplus(-correct_direction / temperature)
        all_pair_losses.append(pair_loss.mean())

    if not all_pair_losses:
        return torch.tensor(0.0, device=predicted.device, requires_grad=True)

    return torch.stack(all_pair_losses).mean()


def top1_loss(predicted, target_ratio, state_ids, tau=0.1):
    """Cross-entropy loss: maximize log-probability of the best trail per state.
    logits_k = -pred_k / tau, then CE with label = argmin(target_ratio)."""
    unique_states = state_ids.unique()
    group_losses = []

    for sid in unique_states:
        mask = (state_ids == sid)
        if mask.sum() < 2:
            continue
        group_pred = predicted[mask]
        group_target = target_ratio[mask]
        k_star = group_target.argmin()
        logits = (-group_pred / tau).unsqueeze(0)
        group_losses.append(F.cross_entropy(logits, k_star.unsqueeze(0)))

    if not group_losses:
        return torch.tensor(0.0, device=predicted.device, requires_grad=True)
    return torch.stack(group_losses).mean()


def soft_top1_loss(predicted, target_ratio, state_ids, tau=0.1, tau_label=1.0):
    """Soft top-1 CE loss with z-score advantage label smoothing.
    Soft target = softmax(advantage / tau_label) where advantage is z-scored.
    Robust to label noise in combinatorial optimization."""
    unique_states = state_ids.unique()
    group_losses = []

    for sid in unique_states:
        mask = (state_ids == sid)
        if mask.sum() < 2:
            continue
        group_pred = predicted[mask]
        group_target = target_ratio[mask]

        mean_r = group_target.mean()
        std_r = group_target.std().clamp(min=1e-6)
        advantage = (mean_r - group_target) / std_r

        soft_target = F.softmax(advantage / tau_label, dim=0).detach()
        log_pred = F.log_softmax(-group_pred / tau, dim=0)
        group_losses.append(-(soft_target * log_pred).sum())

    if not group_losses:
        return torch.tensor(0.0, device=predicted.device, requires_grad=True)
    return torch.stack(group_losses).mean()


def within_state_concordance(preds, targets, state_ids):
    """Within-state pairwise concordance: fraction of (i,j) pairs within
    the same state where sign(pred_i - pred_j) == sign(target_i - target_j)."""
    concordances = []
    for sid in state_ids.unique():
        mask = (state_ids == sid)
        if mask.sum() < 2:
            continue
        p = preds[mask]
        t = targets[mask]
        n = p.size(0)
        diff_p = p.unsqueeze(0) - p.unsqueeze(1)
        diff_t = t.unsqueeze(0) - t.unsqueeze(1)
        upper = torch.triu(torch.ones(n, n, device=p.device), diagonal=1).bool()
        agree = (diff_p[upper].sign() == diff_t[upper].sign()).float().mean()
        concordances.append(agree)
    if not concordances:
        return 0.5
    return torch.stack(concordances).mean().item()


def within_state_topk_hit(preds, targets, state_ids, k=1):
    """Fraction of states where the true best trail is in the model's top-k predictions."""
    hits = []
    for sid in state_ids.unique():
        mask = (state_ids == sid)
        if mask.sum() < 2:
            continue
        p = preds[mask]
        t = targets[mask]
        true_best = t.argmin()
        pred_topk = p.argsort()[:k]
        hits.append(float(true_best in pred_topk))
    if not hits:
        return 0.0
    return sum(hits) / len(hits)


def compute_metrics(predicted, target, cost_0):
    with torch.no_grad():
        residual = predicted - target
        mae = residual.abs().mean().item()
        mse = (residual ** 2).mean().item()
        rmse = mse ** 0.5

        ss_res = (residual ** 2).sum().item()
        ss_tot = ((target - target.mean()) ** 2).sum().item()
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        mape = (residual.abs() / target.clamp(min=0.1)).mean().item() * 100

        pred_cost = predicted * cost_0
        true_cost = target * cost_0
        cost_mae = (pred_cost - true_cost).abs().mean().item()

        pred_improvement = 1.0 - predicted
        true_improvement = 1.0 - target
        imp_corr = torch.corrcoef(torch.stack([pred_improvement, true_improvement]))[0, 1].item()

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'cost_mae': cost_mae,
        'improvement_corr': imp_corr if not np.isnan(imp_corr) else 0.0,
    }


def train_one_epoch(model, train_loader, optimizer, device, epoch, global_step,
                    loss_type='soft_top1', lambda_rank=1.0, lambda_concordance=0.2,
                    temperature=0.01, tau=0.1, tau_label=1.0, min_pair_spread=0.0):
    model.train()
    total_loss = 0
    total_components = {}
    all_preds = []
    all_targets = []
    all_cost_0 = []
    all_state_ids = []
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

        l_reg_val = None
        l_grpo_val = None
        l_conc_val = None
        if loss_type == 'reg':
            loss = loss_function(predicted, target_ratio)
            l_reg_val = loss.item()
        elif loss_type == 'grpo':
            loss = grpo_loss(predicted, target_ratio, state_ids)
            l_grpo_val = loss.item()
        elif loss_type == 'concordance':
            loss = soft_concordance_loss(predicted, target_ratio, state_ids, temperature, min_pair_spread)
        elif loss_type == 'reg_concordance':
            l_reg = loss_function(predicted, target_ratio)
            l_conc = soft_concordance_loss(predicted, target_ratio, state_ids, temperature, min_pair_spread)
            loss = l_reg + lambda_concordance * l_conc
            l_reg_val = l_reg.item()
            l_conc_val = l_conc.item()
        elif loss_type == 'top1':
            loss = top1_loss(predicted, target_ratio, state_ids, tau)
        elif loss_type == 'soft_top1':
            loss = soft_top1_loss(predicted, target_ratio, state_ids, tau, tau_label)
        else:  # reg_grpo
            l_reg = loss_function(predicted, target_ratio)
            l_grpo = grpo_loss(predicted, target_ratio, state_ids)
            loss = l_reg + lambda_rank * l_grpo
            l_reg_val = l_reg.item()
            l_grpo_val = l_grpo.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        global_step += 1
        total_loss += loss.item()
        if l_reg_val is not None:
            total_components['l_reg'] = total_components.get('l_reg', 0) + l_reg_val
        if l_grpo_val is not None:
            total_components['l_grpo'] = total_components.get('l_grpo', 0) + l_grpo_val
        if l_conc_val is not None:
            total_components['l_conc'] = total_components.get('l_conc', 0) + l_conc_val
        num_batches += 1

        all_preds.append(predicted.detach())
        all_targets.append(target_ratio.detach())
        all_cost_0.append(cost_0.detach())
        all_state_ids.append(state_ids.detach())

        if global_step % 10 == 0:
            batch_ws_conc = within_state_concordance(predicted.detach(), target_ratio, state_ids)
            batch_log = {
                "train/batch_loss": loss.item(),
                "train/batch_ws_concordance": batch_ws_conc,
                "global_step": global_step,
                "epoch": epoch + 1,
            }
            if l_reg_val is not None:
                batch_log["train/batch_l_reg"] = l_reg_val
            if l_grpo_val is not None:
                batch_log["train/batch_l_grpo"] = l_grpo_val
            if l_conc_val is not None:
                batch_log["train/batch_l_conc"] = l_conc_val
            wandb.log(batch_log)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_cost_0 = torch.cat(all_cost_0)
    all_state_ids = torch.cat(all_state_ids)
    avg_loss = total_loss / num_batches

    metrics = compute_metrics(all_preds, all_targets, all_cost_0)
    ws_conc = within_state_concordance(all_preds, all_targets, all_state_ids)
    top1_hit = within_state_topk_hit(all_preds, all_targets, all_state_ids, k=1)
    top3_hit = within_state_topk_hit(all_preds, all_targets, all_state_ids, k=3)

    with torch.no_grad():
        residual = all_preds - all_targets

    print(f"  Train - Loss: {avg_loss:.6f} [{loss_type}]")
    if 'l_reg' in total_components:
        l_grpo_str = f" | L_grpo: {total_components.get('l_grpo',0)/num_batches:.6f}" if 'l_grpo' in total_components else ""
        l_conc_str = f" | L_conc: {total_components.get('l_conc',0)/num_batches:.6f}" if 'l_conc' in total_components else ""
        print(f"          L_reg: {total_components['l_reg']/num_batches:.6f}{l_grpo_str}{l_conc_str}")
    elif 'l_grpo' in total_components:
        print(f"          L_grpo: {total_components['l_grpo']/num_batches:.6f}")
    elif 'l_conc' in total_components:
        print(f"          L_conc: {total_components['l_conc']/num_batches:.6f}")
    print(f"          MAE: {metrics['mae']:.6f} | RMSE: {metrics['rmse']:.6f} | R²: {metrics['r2']:.4f}")
    print(f"          MAPE: {metrics['mape']:.2f}% | Cost MAE: {metrics['cost_mae']:.1f} | Imp Corr: {metrics['improvement_corr']:.4f}")
    print(f"          Within-State Concordance: {ws_conc:.4f}")
    print(f"          Top-1 Hit: {top1_hit:.4f} | Top-3 Hit: {top3_hit:.4f}")
    print(f"          Pred ratio: [{all_preds.min():.4f}, {all_preds.max():.4f}] mean={all_preds.mean():.4f}")
    print(f"          True ratio: [{all_targets.min():.4f}, {all_targets.max():.4f}] mean={all_targets.mean():.4f}")
    print(f"          Residual:   [{residual.min():.6f}, {residual.max():.6f}] mean={residual.mean():.6f} std={residual.std():.6f}")

    epoch_log = {
        "train/epoch_loss": avg_loss,
        "train/mae": metrics['mae'],
        "train/rmse": metrics['rmse'],
        "train/r2": metrics['r2'],
        "train/mape": metrics['mape'],
        "train/cost_mae": metrics['cost_mae'],
        "train/improvement_corr": metrics['improvement_corr'],
        "train/ws_concordance": ws_conc,
        "train/top1_hit": top1_hit,
        "train/top3_hit": top3_hit,
        "train/pred_ratio_mean": all_preds.mean().item(),
        "train/pred_ratio_std": all_preds.std().item(),
        "train/residual_mean": residual.mean().item(),
        "train/residual_std": residual.std().item(),
        "epoch": epoch + 1,
    }
    if 'l_reg' in total_components:
        epoch_log["train/l_reg"] = total_components['l_reg'] / num_batches
    if 'l_grpo' in total_components:
        epoch_log["train/l_grpo"] = total_components['l_grpo'] / num_batches
    if 'l_conc' in total_components:
        epoch_log["train/l_conc"] = total_components['l_conc'] / num_batches
    wandb.log(epoch_log)

    return avg_loss, global_step


def evaluate(model, val_loader, device, epoch):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_cost_0 = []
    all_state_ids = []
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
            loss = loss_function(predicted, target_ratio)

        total_loss += loss.item()
        num_batches += 1
        all_preds.append(predicted)
        all_targets.append(target_ratio)
        all_cost_0.append(cost_0)
        all_state_ids.append(state_ids)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_cost_0 = torch.cat(all_cost_0)
    all_state_ids = torch.cat(all_state_ids)
    avg_loss = total_loss / num_batches

    metrics = compute_metrics(all_preds, all_targets, all_cost_0)
    ws_conc = within_state_concordance(all_preds, all_targets, all_state_ids)
    top1_hit = within_state_topk_hit(all_preds, all_targets, all_state_ids, k=1)
    top3_hit = within_state_topk_hit(all_preds, all_targets, all_state_ids, k=3)

    with torch.no_grad():
        residual = all_preds - all_targets
        pred_cost = all_preds * all_cost_0
        true_cost = all_targets * all_cost_0
        cost_residual = pred_cost - true_cost

    print(f"  Val   - Loss: {avg_loss:.6f} | MAE: {metrics['mae']:.6f} | RMSE: {metrics['rmse']:.6f} | R²: {metrics['r2']:.4f}")
    print(f"          MAPE: {metrics['mape']:.2f}% | Cost MAE: {metrics['cost_mae']:.1f} | Imp Corr: {metrics['improvement_corr']:.4f}")
    print(f"          Within-State Concordance: {ws_conc:.4f}")
    print(f"          Top-1 Hit: {top1_hit:.4f} | Top-3 Hit: {top3_hit:.4f}")
    print(f"          Pred ratio: [{all_preds.min():.4f}, {all_preds.max():.4f}] mean={all_preds.mean():.4f}")
    print(f"          True ratio: [{all_targets.min():.4f}, {all_targets.max():.4f}] mean={all_targets.mean():.4f}")
    print(f"          Residual:   [{residual.min():.6f}, {residual.max():.6f}] mean={residual.mean():.6f} std={residual.std():.6f}")
    print(f"          Cost error: [{cost_residual.min():.1f}, {cost_residual.max():.1f}] mean={cost_residual.mean():.1f}")

    wandb.log({
        "val/loss": avg_loss,
        "val/mae": metrics['mae'],
        "val/rmse": metrics['rmse'],
        "val/r2": metrics['r2'],
        "val/mape": metrics['mape'],
        "val/cost_mae": metrics['cost_mae'],
        "val/improvement_corr": metrics['improvement_corr'],
        "val/ws_concordance": ws_conc,
        "val/top1_hit": top1_hit,
        "val/top3_hit": top3_hit,
        "val/pred_ratio_mean": all_preds.mean().item(),
        "val/pred_ratio_std": all_preds.std().item(),
        "val/residual_mean": residual.mean().item(),
        "val/residual_std": residual.std().item(),
        "val/cost_error_mean": cost_residual.mean().item(),
        "val/cost_error_std": cost_residual.std().item(),
        "epoch": epoch + 1,
    })

    return avg_loss, metrics, ws_conc, top1_hit


def train(data_pattern, checkpoint_dir, num_epochs=1000, lr=5e-5,
          resume_checkpoint=None, load_weights=None, runname=None,
          groups_per_batch=16, trails_per_group=16,
          num_train_batches=800, num_val_batches=100,
          lambda_rank=1.0, lambda_concordance=0.2, loss_type='soft_top1', temperature=0.01,
          tau=0.1, tau_label=1.0, min_spread=0.0, min_pair_spread=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CostPredictor(device=device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.1)
    start_epoch = 0
    global_step = 0

    if load_weights:
        load_checkpoint(load_weights, model, optimizer=None)
        print(f"Loaded model weights from {load_weights} (training from epoch 1, fresh optimizer)")
    elif resume_checkpoint:
        start_epoch, global_step = load_checkpoint(resume_checkpoint, model, optimizer)
        print(f"Resumed from {resume_checkpoint} at epoch {start_epoch + 1}")

    # Support both: 1) directory of .npy (e.g. data_train from convert_pt_to_npy.py)  2) .pt file(s)
    if os.path.isdir(data_pattern):
        npy_dir = os.path.abspath(data_pattern)
        if os.path.isfile(os.path.join(npy_dir, "state_id_tensor.npy")):
            dataset = MmapVRPDataset(npy_dir)
            datasets = [dataset]
            files = [npy_dir]
        else:
            print(f"Directory {data_pattern} has no state_id_tensor.npy. Exiting.")
            return
    else:
        files = sorted(glob.glob(data_pattern))
        if not files:
            print(f"No files matching '{data_pattern}'. Exiting.")
            return
        print(f"Loading {len(files)} data file(s): {files}")
        datasets = [VRPDataset(f) for f in files]
        dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    # --- State-based train/val split (with optional min-spread filter) ---
    if isinstance(dataset, ConcatDataset):
        all_state_ids = np.concatenate([d.state_id_tensor.numpy() for d in datasets])
        all_cost = np.concatenate([d.previous_cost_tensor.numpy() for d in datasets])
    else:
        all_state_ids = dataset.state_id_tensor.numpy()
        all_cost = dataset.previous_cost_tensor.numpy()

    unique_states = np.unique(all_state_ids)

    if min_spread > 0:
        ratio_all = all_cost[:, -1] / (all_cost[:, 0] + 1e-8)
        valid_states = []
        for s in unique_states:
            r = ratio_all[all_state_ids == s]
            if len(r) >= 2 and (r.max() - r.min()) >= min_spread:
                valid_states.append(s)
        valid_states = np.array(valid_states)
        print(f"Min-spread filter ({min_spread}): {len(valid_states)}/{len(unique_states)} states kept")
        unique_states = valid_states

    rng = np.random.RandomState(42)
    rng.shuffle(unique_states)
    split = int(0.9 * len(unique_states))
    train_states_set = set(unique_states[:split].tolist())
    val_states_set = set(unique_states[split:].tolist())

    train_indices = [i for i, s in enumerate(all_state_ids) if s in train_states_set]
    val_indices = [i for i, s in enumerate(all_state_ids) if s in val_states_set]

    print(f"State-based split: {len(train_states_set)} train states, {len(val_states_set)} val states")
    print(f"  Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_state_ids_arr = all_state_ids[train_indices]
    val_state_ids_arr = all_state_ids[val_indices]

    train_sampler = GroupedBatchSampler(
        train_state_ids_arr,
        num_groups_per_batch=groups_per_batch,
        max_trails_per_group=trails_per_group,
        num_batches=num_train_batches,
    )
    val_sampler = GroupedBatchSampler(
        val_state_ids_arr,
        num_groups_per_batch=groups_per_batch,
        max_trails_per_group=trails_per_group,
        num_batches=num_val_batches,
    )

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            num_workers=4, pin_memory=True)

    wandb.init(project="cuopt-regression", name=runname or os.path.basename(checkpoint_dir))
    wandb.config.update({
        "groups_per_batch": groups_per_batch,
        "trails_per_group": trails_per_group,
        "num_train_batches": num_train_batches,
        "num_val_batches": num_val_batches,
        "lambda_rank": lambda_rank,
        "lambda_concordance": lambda_concordance,
        "learning_rate": lr,
        "num_epochs": num_epochs,
        "train_states": len(train_states_set),
        "val_states": len(val_states_set),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "model_params": num_params,
        "data_files": files,
        "loss_type": loss_type,
        "temperature": temperature,
        "tau": tau,
        "tau_label": tau_label,
        "loss": f"{loss_type} (lambda={lambda_rank}, temp={temperature}, tau={tau}, tau_label={tau_label})",
        "target": "cost[-1]/cost[0]",
        "min_spread": min_spread,
        "min_pair_spread": min_pair_spread,
    })

    print(f"Device: {device}")
    print(f"Loss type: {loss_type} | Lambda rank: {lambda_rank} | Lambda concordance: {lambda_concordance} | Temperature: {temperature} | Tau: {tau} | Tau_label: {tau_label}")
    if loss_type in ('concordance', 'reg_concordance') and min_pair_spread > 0:
        print(f"  Concordance min_pair_spread: {min_pair_spread} (only pairs with |dt| > {min_pair_spread})")

    best_top1_hit = 0.0

    for epoch in range(start_epoch, num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs} | LR: {current_lr:.2e}")
        print(f"{'='*60}")

        avg_train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, device, epoch, global_step,
            loss_type=loss_type, lambda_rank=lambda_rank, lambda_concordance=lambda_concordance,
            temperature=temperature, tau=tau, tau_label=tau_label, min_pair_spread=min_pair_spread
        )

        avg_val_loss, val_metrics, val_ws_conc, val_top1_hit = evaluate(model, val_loader, device, epoch)

        scheduler.step()

        wandb.log({"lr": current_lr, "epoch": epoch + 1})

        is_best = val_top1_hit > best_top1_hit
        if is_best:
            best_top1_hit = val_top1_hit
            save_checkpoint(model, optimizer, epoch + 1, global_step, checkpoint_dir)
            print(f"  ** New best Top-1 Hit: {best_top1_hit:.4f} — checkpoint saved **")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch + 1, global_step, checkpoint_dir)

        print(f"  Summary: Train Loss={avg_train_loss:.6f} | Val Loss={avg_val_loss:.6f} | Val Top1={val_top1_hit:.4f} | Val WS-Conc={val_ws_conc:.4f}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ml_data.pt")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--runname", type=str, default='grpo')
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training: load model + optimizer, continue from saved epoch")
    parser.add_argument("--load_weights", type=str, default=None,
                        help="Load only model weights from checkpoint; start from epoch 1 with fresh optimizer")
    parser.add_argument("--groups_per_batch", type=int, default=16)
    parser.add_argument("--trails_per_group", type=int, default=16)
    parser.add_argument("--num_train_batches", type=int, default=800)
    parser.add_argument("--num_val_batches", type=int, default=100)
    parser.add_argument("--lambda_rank", type=float, default=1.0,
                        help="Weight for GRPO ranking loss (only used with reg_grpo)")
    parser.add_argument("--lambda_concordance", type=float, default=0.2,
                        help="Weight for concordance loss (only used with reg_concordance, default 0.2 for reg dominance)")
    parser.add_argument("--loss", type=str, default="soft_top1",
                        choices=["reg", "grpo", "reg_grpo", "reg_concordance", "concordance", "top1", "soft_top1"],
                        help="Loss type: soft_top1, top1, reg, grpo, reg_grpo, reg_concordance (reg+concordance, reg dominates), concordance")
    parser.add_argument("--temperature", type=float, default=0.01,
                        help="Temperature for soft concordance loss (smaller = sharper)")
    parser.add_argument("--tau", type=float, default=0.1,
                        help="Temperature for top1/soft_top1 pred softmax (smaller = sharper)")
    parser.add_argument("--tau_label", type=float, default=1.0,
                        help="Temperature for soft_top1 label softmax over z-scored advantage (smaller = more peaked)")
    parser.add_argument("--min_spread", type=float, default=0.0,
                        help="Min ratio spread within state to keep for training (0 = no filter)")
    parser.add_argument("--min_pair_spread", type=float, default=0.0,
                        help="For concordance: only use pairs with |target_i - target_j| > this (0 = use all)")
    args = parser.parse_args()

    args.runname = args.runname + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = args.output or os.path.join("outputs", args.runname)
    print(f"Checkpoint directory: {checkpoint_dir}")
    train(args.data, checkpoint_dir, args.epochs, args.lr, args.resume, args.load_weights, args.runname,
          args.groups_per_batch, args.trails_per_group,
          args.num_train_batches, args.num_val_batches,
          args.lambda_rank, args.lambda_concordance, args.loss, args.temperature, args.tau, args.tau_label, args.min_spread, args.min_pair_spread)
