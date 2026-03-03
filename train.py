"""
Training script for Transformer Policy
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from tqdm import tqdm
import wandb
import argparse
import os
import glob
from datetime import datetime
from model import Policy
import matplotlib.pyplot as plt
import numpy as np
from itertools import islice

class VRPDataset(Dataset):
    def __init__(self, datafile):
        print(f"Loading data from {datafile}...")
        data = torch.load(datafile)
        self.nodes_tensor = data["nodes_tensor"]
        self.demands_tensor = data["demands_tensor"]
        self.current_sol_tensor = data["current_sol_tensor"]
        self.candidates_tensor = data["candidates_tensor"]
        self.selected_tensor = data["selected_tensor"]
    
    def __len__(self):
        return self.nodes_tensor.shape[0]
    
    def __getitem__(self, idx):
        return {
            'nodes': self.nodes_tensor[idx],
            'demands': self.demands_tensor[idx],
            'current_sol': self.current_sol_tensor[idx],
            'candidates': self.candidates_tensor[idx],
            'selected': self.selected_tensor[idx]
        }

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

def loss_function(logits, candidates, selected, pos_weight=None):
    candidates = candidates.squeeze(1)
    selected = selected.squeeze(1)
    
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor([pos_weight], device=logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, selected.float(), 
            pos_weight=pos_weight_tensor,
            reduction='none'
        )
    else:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, selected.float(), 
            reduction='none'
        )
    
    masked_loss = bce_loss * candidates.float()
    
    num_candidates = candidates.float().sum()
    if num_candidates > 0:
        loss = masked_loss.sum() / num_candidates
    else:
        loss = masked_loss.sum()
    
    return loss

def get_batch_counts(preds, selected_flat, candidates_mask):
    """
    计算 batch 内的 TP, FP, TN, FN 数量 (用于后续计算指标)
    """
    # 确保 mask 是 float 以进行数学运算
    mask = candidates_mask.float()
    
    # TP: 预测为1, 实际为1
    tp = (preds * selected_flat * mask).sum().item()
    # FP: 预测为1, 实际为0
    fp = (preds * (1 - selected_flat) * mask).sum().item()
    # FN: 预测为0, 实际为1
    fn = ((1 - preds) * selected_flat * mask).sum().item()
    # TN: 预测为0, 实际为0
    tn = ((1 - preds) * (1 - selected_flat) * mask).sum().item()
    
    return tp, fp, tn, fn

def train_one_epoch(model, train_loader, optimizer, device, epoch, global_step, pos_weight=8.83, threshold=0.5):
    model.train()
    train_loss = 0
    
    # 用于计算整个 Epoch 的全局指标
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    steps = int(len(train_loader) * 0.1)
    for batch in tqdm(islice(train_loader, steps), total=steps, desc=f"Epoch {epoch+1} [Train]"):
        nodes = batch['nodes'].to(device)
        demands = batch['demands'].to(device)
        current_sol = batch['current_sol'].to(device)
        candidates = batch['candidates'].to(device)
        selected = batch['selected'].to(device)
        
        optimizer.zero_grad()
        logits = model(nodes, demands, current_sol, candidates)
        loss = loss_function(logits, candidates, selected, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        global_step += 1
        train_loss += loss.item()
        
        with torch.no_grad():
            candidates_mask = candidates.squeeze(1).bool()
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            selected_flat = selected.squeeze(1).float()
            
            # 计算当前 batch 的 counts
            tp, fp, tn, fn = get_batch_counts(preds, selected_flat, candidates_mask)
            
            # 累加到全局
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
            
            # 简单的 Batch 级指标 (仅用于 logging，不用于最终报告)
            batch_acc = (tp + tn) / (tp + fp + tn + fn + 1e-8)
            batch_prec = tp / (tp + fp + 1e-8)
            batch_rec = tp / (tp + fn + 1e-8)
            batch_f1 = 2 * batch_prec * batch_rec / (batch_prec + batch_rec + 1e-8)
            batch_tnr = tn / (tn + fp + 1e-8)
        
        if global_step % 10 == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/accuracy": batch_acc,
                "train/precision": batch_prec,
                "train/recall": batch_rec,
                "train/f1": batch_f1,
                "train/tnr": batch_tnr,
                "train/tp": tp,
                "train/fp": fp,
                "train/tn": tn,
                "train/fn": fn,
                "global_step": global_step,
                "epoch": epoch + 1,            
            })
    
    avg_train_loss = train_loss / len(train_loader)
    
    # 计算 Epoch 级别的全局指标
    epoch_acc = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn + 1e-8)
    epoch_precision = total_tp / (total_tp + total_fp + 1e-8)
    epoch_recall = total_tp / (total_tp + total_fn + 1e-8)
    epoch_tnr = total_tn / (total_tn + total_fp + 1e-8)
    epoch_f1 = 2 * epoch_precision * epoch_recall / (epoch_precision + epoch_recall + 1e-8)
    
    print(f"  Train - Loss: {avg_train_loss:.4f} | Acc: {epoch_acc:.4f} | TNR: {epoch_tnr:.4f}")
    print(f"         Precision: {epoch_precision:.4f} | Recall: {epoch_recall:.4f} | F1: {epoch_f1:.4f}")
    print(f"         Counts - TP: {int(total_tp)}, FP: {int(total_fp)}, TN: {int(total_tn)}, FN: {int(total_fn)}")
    
    return avg_train_loss, epoch_acc, global_step

def evaluate(model, val_loader, device, pos_weight=8.83, threshold=0.5):
    model.eval()
    val_loss = 0
    
    # 用于计算整个 Validation Set 的全局指标
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    steps = int(len(val_loader) * 0.1)
    for batch in tqdm(islice(val_loader, steps), total=steps, desc="Evaluating"):
        nodes = batch['nodes'].to(device)
        demands = batch['demands'].to(device)
        current_sol = batch['current_sol'].to(device)
        candidates = batch['candidates'].to(device)
        selected = batch['selected'].to(device)
        
        with torch.no_grad():
            logits = model(nodes, demands, current_sol, candidates)
            v_loss = loss_function(logits, candidates, selected, pos_weight=pos_weight)
            val_loss += v_loss.item()
            
            candidates_mask = candidates.squeeze(1).bool()
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            selected_flat = selected.squeeze(1).float()
            
            # 计算 counts
            tp, fp, tn, fn = get_batch_counts(preds, selected_flat, candidates_mask)
            
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
    
    avg_val_loss = val_loss / len(val_loader)
    
    # 计算全局指标
    val_acc = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn + 1e-8)
    val_precision = total_tp / (total_tp + total_fp + 1e-8)
    val_recall = total_tp / (total_tp + total_fn + 1e-8)
    val_tnr = total_tn / (total_tn + total_fp + 1e-8)
    val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall + 1e-8)
    
    print(f"  Val   - Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | TNR: {val_tnr:.4f}")
    print(f"         Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")
    print(f"         Counts - TP: {int(total_tp)}, FP: {int(total_fp)}, TN: {int(total_tn)}, FN: {int(total_fn)}")

    return avg_val_loss, val_acc, val_precision, val_recall, val_f1, val_tnr, total_tp, total_fp, total_tn, total_fn

def train(dataset, checkpoint_dir, num_epochs=1000, batch_size=64, lr=5e-5, resume_checkpoint=None, runname=None, pos_weight=8.83, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Policy(device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_epoch = 0
    global_step = 0
    
    if resume_checkpoint:
        start_epoch, global_step = load_checkpoint(resume_checkpoint, model, optimizer)
    
    train_size = int(0.9 * len(dataset))
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    wandb.init(project="cuopt", name=runname or os.path.basename(checkpoint_dir))
    wandb.config.update({
        "pos_weight": pos_weight,
        "threshold": threshold,
        "batch_size": batch_size,
        "learning_rate": lr,
    })
    
    print(f"Training on {len(dataset)} samples. Pos_weight={pos_weight:.2f}, Threshold={threshold:.2f}")
    
    for epoch in range(start_epoch, num_epochs):
        avg_train_loss, avg_train_acc, global_step = train_one_epoch(
            model, train_loader, optimizer, device, epoch, global_step, pos_weight=pos_weight, threshold=threshold
        )
        
        with torch.no_grad():
            avg_val_loss, avg_val_acc, avg_val_precision, avg_val_recall, avg_val_f1, avg_val_tnr, v_tp, v_fp, v_tn, v_fn = evaluate(model, val_loader, device, pos_weight=pos_weight, threshold=threshold)
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {avg_val_f1:.4f}")
        
        wandb.log({
            "val/loss": avg_val_loss, 
            "val/accuracy": avg_val_acc,
            "val/precision": avg_val_precision,
            "val/recall": avg_val_recall,
            "val/f1": avg_val_f1,
            "val/tnr": avg_val_tnr,
            "val/tp": v_tp,
            "val/fp": v_fp,
            "val/tn": v_tn,
            "val/fn": v_fn,
            "epoch": epoch+1,
            "global_step": global_step
        })
        
        if (epoch + 1) % 5 == 0 or epoch + 1 > num_epochs - 10:
            save_checkpoint(model, optimizer, epoch+1, global_step, checkpoint_dir)
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ml_data_*.pt", help="Path to .pt files (supports wildcards)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--runname", type=str, default='default')
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    args.runname = args.runname + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = args.output or os.path.join("outputs", args.runname)
    print(f"Checkpoint directory: {checkpoint_dir}")
    train(args.data, checkpoint_dir, args.epochs, args.batch_size, args.lr, args.resume, args.runname)