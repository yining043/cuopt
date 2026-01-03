"""
Training script for Transformer Policy
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import wandb
import argparse
import os
from datetime import datetime
from model import Policy

class VRPDataset(Dataset):
    def __init__(self, datafile):
        data = torch.load(datafile)
        self.nodes_tensor = data["nodes_tensor"]
        self.demands_tensor = data["demands_tensor"]
        self.current_sol_tensor = data["current_sol_tensor"]
        self.candidates_tensor = data["candidates_tensor"]
        self.selected_tensor = data["selected_tensor"]
        self.score_tensor = data["score_tensor"]
    
    def __len__(self):
        return self.nodes_tensor.shape[0]
    
    def __getitem__(self, idx):
        return {
            'nodes': self.nodes_tensor[idx],
            'demands': self.demands_tensor[idx],
            'current_sol': self.current_sol_tensor[idx],
            'candidates': self.candidates_tensor[idx],
            'selected': self.selected_tensor[idx],
            'score': self.score_tensor[idx]
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

def loss_function(logits, score, margin=0.1, mode='rank'):
    # 1. Z-Score MSE Loss
    # since we want to minimize the loss, we need to negate the score !!!
    target_score = - score
    s_mean, s_std = target_score.mean(dim=(1), keepdim=True), target_score.std(dim=(1), keepdim=True) + 1e-6
    target = (target_score - s_mean) / s_std
    l_mean, l_std = logits.mean(dim=(1), keepdim=True), logits.std(dim=(1), keepdim=True) + 1e-6
    preds = (logits - l_mean) / l_std
    mse_loss = F.mse_loss(preds, target)

    # 2. All-Pairs Ranking Loss
    preds_diff = preds.unsqueeze(2) - preds.unsqueeze(1)
    target_diff = target.unsqueeze(2) - target.unsqueeze(1)
    # label is 1 if target_i > target_j which is score_i < score_j
    label = torch.sign(target_diff) 
    score_weight = torch.abs(target_diff)
    mask = torch.eye(logits.size(1), device=logits.device).unsqueeze(0) == 0
    valid_pair_mask = mask & (label != 0)
    # rank loss
    rank_loss_matrix = torch.relu(margin - label * preds_diff) * score_weight
    rank_loss = rank_loss_matrix.sum() / valid_pair_mask.sum()

    if mode == 'mse': return mse_loss
    elif mode == 'rank': return rank_loss
    elif mode == 'both': return mse_loss + rank_loss
    else: raise ValueError(f"Invalid mode: {mode}")


def train_one_epoch(model, train_loader, optimizer, device, epoch, global_step):
    model.train()
    train_loss = 0
    train_acc = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        
        nodes = batch['nodes'].to(device)
        demands = batch['demands'].to(device)
        current_sol = batch['current_sol'].to(device)
        candidates = batch['candidates'].to(device)
        selected = batch['selected'].to(device)
        score = batch['score'].to(device)
        
        optimizer.zero_grad()
        logits = model(nodes, demands, current_sol, candidates, selected)
        loss = loss_function(logits, score)        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        global_step += 1
        train_loss += loss.item()
        # 计算 Top-1 准确率
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == score.argmin(dim=1)).float().mean()
            train_acc += acc.item()
        
        # Log every 10 gradient steps
        if global_step % 10 == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/acc": acc.item(),
                "global_step": global_step,
                "epoch": epoch + 1
            })
        
    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)
    return avg_train_loss, avg_train_acc, global_step

def evaluate(model, val_loader, device):
    model.eval()
    val_loss = 0
    val_acc = 0
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        nodes = batch['nodes'].to(device)
        demands = batch['demands'].to(device)
        current_sol = batch['current_sol'].to(device)
        candidates = batch['candidates'].to(device)
        selected = batch['selected'].to(device)
        score = batch['score'].to(device)
        
        logits = model(nodes, demands, current_sol, candidates, selected)
        v_loss = loss_function(logits, score)
        val_loss += v_loss.item()
        val_acc += (logits.argmax(dim=1) == score.argmin(dim=1)).float().mean().item()
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)
    return avg_val_loss, avg_val_acc

def train(data_path, checkpoint_dir, num_epochs=1000, batch_size=64, lr=5e-5, resume_checkpoint=None, runname=None):
    ###########################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Policy(device=device)
    ###########################################################################
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_epoch = 0
    global_step = 0
    
    if resume_checkpoint:
        start_epoch, global_step = load_checkpoint(resume_checkpoint, model, optimizer)
    
    dataset = VRPDataset(data_path)
    train_size = int(0.9 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    wandb.init(project="cuopt", name=runname or os.path.basename(checkpoint_dir))
    
    for epoch in range(start_epoch, num_epochs):
        # Train one epoch
        avg_train_loss, avg_train_acc, global_step = train_one_epoch(
            model, train_loader, optimizer, device, epoch, global_step
        )
        
        # Evaluate
        avg_val_loss, avg_val_acc = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}: Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")
        wandb.log({
            "val/loss": avg_val_loss, 
            "val/acc": avg_val_acc,
            "epoch": epoch+1,
            "global_step": global_step
        })
        
        # Save checkpoint every 50 epochs or in the last 10 epochs
        if (epoch + 1) % 5 == 0 or epoch + 1 > num_epochs - 10:
            save_checkpoint(model, optimizer, epoch+1, global_step, checkpoint_dir)
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ml_data_large.pt")
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