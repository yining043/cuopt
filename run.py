"""
Training script for Transformer Policy
Contains training loop, data loading, checkpointing, and evaluation
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import wandb
import argparse
import os
from datetime import datetime
import glob

from transformer_model import TransformerCandidatePolicy

def adaptive_ranking_loss_simple(logp, scores):
    """
    logp:   模型输出, 形状 [Batch, 10] (10个候选动作的对数概率)
    scores: 真实 Cost, 形状 [Batch, 10] (路径越短，值越小)
    """
    # 第一步：把 10 个数据变成 10x10 的对战表
    # logp.unsqueeze(2) 变成 [Batch, 10, 1] (列)
    # logp.unsqueeze(1) 变成 [Batch, 1, 10] (行)
    # 相减得到 diff_logp: [Batch, 10, 10]
    # diff_logp[i][j] 存储的是 logp[i] - logp[j]
    diff_logp = logp.unsqueeze(2) - logp.unsqueeze(1)

    # 第二步：计算 Cost 的差距
    # 我们希望 Cost 越小的，logp 越大
    # 所以算 scores[j] - scores[i]。如果 > 0，说明 i 比 j 好
    diff_scores = scores.unsqueeze(1) - scores.unsqueeze(2)
    diff_scores = diff_scores.to(logp.device)

    # 第三步：只看"赢了"的那些对决
    # 我们只关心那些 i 比 j 好的情况 (mask)
    mask = (diff_scores > 0).float().to(logp.device)
    
    # 第四步：自适应修正 (这是你最关心的部分)
    # 权重 = Cost 的实际差值。
    # 如果 A 比 B 只好一点点，权重就小；如果 A 比 B 好非常多，权重就大。
    weights = diff_scores * mask

    # 第五步：计算损失
    # 我们希望当 i 赢了 j 时，diff_logp (logp_i - logp_j) 越大越好
    # 这里用 sigmoid 转化：sigmoid(logp_i - logp_j) 越接近 1 越好
    # 对其取 -log 就是我们要最小化的 Loss
    loss_matrix = -F.logsigmoid(diff_logp)

    # 第六步：加权平均
    # 只对有意义的对决 (mask=1) 求加权和
    total_loss = (loss_matrix * weights).sum() / (mask.sum() + 1e-7)

    return total_loss

class VRPDataset(Dataset):
    def __init__(self, data):
        self.nodes_tensor = data["nodes_tensor"]
        self.demands_tensor = data["demands_tensor"]
        self.current_sol_tensor = data["current_sol_tensor"]
        self.candidates_tensor = data["candidates_tensor"]
        self.selected_tensor = data["selected_tensor"]
        self.score_tensor = data["score_tensor"]
        self.max_candidates_length = data["max_candidates_length"]
    
    def __len__(self):
        return self.nodes_tensor.shape[0]
    
    def __getitem__(self, idx):
        return {
            'nodes': self.nodes_tensor[idx],
            'demands': self.demands_tensor[idx],
            'current_sol': self.current_sol_tensor[idx],
            'candidates': self.candidates_tensor[idx],
            'selected': self.selected_tensor[idx],
            'score': self.score_tensor[idx],
            'max_candidates_length': self.max_candidates_length
        }

def save_checkpoint(policy, optimizer, epoch, global_step, checkpoint_dir, max_checkpoints=5):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}_step_{global_step}.pt")
    
    torch.save({
        'epoch': epoch + 1,
        'global_step': global_step,
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")), 
                        key=lambda x: (int(x.split('_')[-3]), int(x.split('_')[-1].split('.')[0])))
    
    if len(checkpoints) > max_checkpoints:
        for old_checkpoint in checkpoints[:-max_checkpoints]:
            os.remove(old_checkpoint)

def load_checkpoint(checkpoint_path, policy, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=policy.device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['global_step']

def eval_only(checkpoint_path, use_autoregressive_decoder=True, loss_type="adaptive"):
    policy = TransformerCandidatePolicy(use_autoregressive_decoder=use_autoregressive_decoder)
    load_checkpoint(checkpoint_path, policy)
    
    data = torch.load("ml_data.pt")
    dataset = VRPDataset(data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    gamma = 0.5 if loss_type == "margin" else None
    
    policy.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_scores = batch['score']
            set_log_probs = policy(
                batch['nodes'], batch['demands'], batch['current_sol'], 
                batch['candidates'], batch['selected']
            )
            
            # batch_scores越小越好，所以用升序排序
            sorted_indices = torch.argsort(batch_scores, dim=1, descending=False)
            good_idx = sorted_indices[:, 0]  # score最小的（最好的）
            bad_idx = sorted_indices[:, -1]  # score最大的（最差的）
            
            batch_range = torch.arange(set_log_probs.shape[0], device=set_log_probs.device)
            sum_good = set_log_probs[batch_range, good_idx]
            sum_bad = set_log_probs[batch_range, bad_idx]
            if batch_scores[0, 0] != batch_scores[0, -1]:
                print(torch.argsort(set_log_probs, dim=1, descending=True)[0].cpu())
                print(set_log_probs[0], 'vs\n', batch_scores[0], 'at\n', batch_scores[0].gather(0, torch.argsort(set_log_probs, dim=1, descending=True)[0].cpu()))
                print(sum_good[0], 'vs\n', sum_bad[0])
            # Calculate loss based on loss_type
            if loss_type == "adaptive":
                loss = adaptive_ranking_loss_simple(set_log_probs, batch_scores)
            else:  # margin ranking loss
                loss = torch.clamp(gamma - (sum_good - sum_bad), min=0.0).mean()
            
            total_loss += loss.item()
            num_batches += 1
    
    print(f"Loss: {total_loss / num_batches:.4f}, margin: {sum_good.mean().item() - sum_bad.mean().item():.4f}")

def train_transformer_policy(use_autoregressive_decoder=True, runname_prefix="", resume_checkpoint=None, loss_type="adaptive"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runname = f"{runname_prefix}_{timestamp}" if runname_prefix else timestamp
    checkpoint_dir = os.path.join("outputs", runname)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    policy = TransformerCandidatePolicy(use_autoregressive_decoder=use_autoregressive_decoder)
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-5)
    
    start_epoch = 0
    global_step = 0
    
    if resume_checkpoint:
        start_epoch, global_step = load_checkpoint(resume_checkpoint, policy, optimizer)
        print(f"Resumed from checkpoint: epoch {start_epoch}, step {global_step}")
    
    data = torch.load("ml_data.pt")
    dataset = VRPDataset(data)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    num_epochs = 200
    gamma = 0.5 if loss_type == "margin" else None
    
    wandb.init(
        project="cuopt",
        name=runname,
    )
    
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    for epoch in range(start_epoch, num_epochs):
        policy.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            batch_scores = batch['score']
            
            optimizer.zero_grad()
            set_log_probs = policy(
                batch['nodes'], batch['demands'], batch['current_sol'], 
                batch['candidates'], batch['selected']
            )
            
            # Calculate margin (shared for both loss types)
            # batch_scores越小越好，所以用升序排序
            sorted_indices = torch.argsort(batch_scores, dim=1, descending=False)
            good_idx = sorted_indices[:, 0]  # score最小的（最好的）
            bad_idx = sorted_indices[:, -1]  # score最大的（最差的）
            batch_range = torch.arange(set_log_probs.shape[0], device=set_log_probs.device)
            sum_good = set_log_probs[batch_range, good_idx]
            sum_bad = set_log_probs[batch_range, bad_idx]
            margin_value = (sum_good - sum_bad).mean().item()
            
            # Calculate loss based on loss_type
            if loss_type == "adaptive":
                loss = adaptive_ranking_loss_simple(set_log_probs, batch_scores)
            else:  # margin ranking loss
                loss = torch.clamp(gamma - (sum_good - sum_bad), min=0.0).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            global_step += 1

            margin_value = (sum_good - sum_bad).mean().item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'margin': f'{margin_value:.4f}'})
            
            wandb.log({
                "train/loss": loss.item(),
                "train/avg_loss": total_loss / num_batches,
                "train/margin": margin_value,
                "train/good_score": sum_good.mean().item(),
                "train/bad_score": sum_bad.mean().item(),
                "epoch": epoch + 1,
                "global_step": global_step
            })
        
        avg_train_loss = total_loss / num_batches
        
        policy.eval()
        val_total_loss = 0
        val_num_batches = 0
        val_total_good_sum = 0
        val_total_bad_sum = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in val_pbar:
                batch_scores = batch['score']
                
                set_log_probs = policy(
                    batch['nodes'], batch['demands'], batch['current_sol'], 
                    batch['candidates'], batch['selected']
                )
                
                # Calculate margin (shared for both loss types)
                # batch_scores越小越好，所以用升序排序
                sorted_indices = torch.argsort(batch_scores, dim=1, descending=False)
                good_idx = sorted_indices[:, 0]  # score最小的（最好的）
                bad_idx = sorted_indices[:, -1]  # score最大的（最差的）
                batch_range = torch.arange(set_log_probs.shape[0], device=set_log_probs.device)
                sum_good = set_log_probs[batch_range, good_idx]
                sum_bad = set_log_probs[batch_range, bad_idx]
                
                # Calculate loss based on loss_type
                if loss_type == "adaptive":
                    loss = adaptive_ranking_loss_simple(set_log_probs, batch_scores)
                else:  # margin ranking loss
                    loss = torch.clamp(gamma - (sum_good - sum_bad), min=0.0).mean()
                
                val_total_loss += loss.item()
                val_total_good_sum += sum_good.mean().item()
                val_total_bad_sum += sum_bad.mean().item()
                val_num_batches += 1
                
                margin_value = (sum_good - sum_bad).mean().item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'margin': f'{margin_value:.4f}'})
        
        avg_val_loss = val_total_loss / val_num_batches
        avg_val_margin = val_total_good_sum / val_num_batches - val_total_bad_sum / val_num_batches
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Margin: {avg_val_margin:.4f}")
        
        wandb.log({
            "train/epoch_loss": avg_train_loss,
            "val/epoch_loss": avg_val_loss,
            "val/margin": avg_val_margin,
            "val/good_score": val_total_good_sum / val_num_batches,
            "val/bad_score": val_total_bad_sum / val_num_batches,
            "epoch": epoch + 1
        })
        
        save_checkpoint(policy, optimizer, epoch, global_step, checkpoint_dir)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=["NAR", "AR"], type=str)
    parser.add_argument("--runname", type=str, default="", help="Prefix for run folder name")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--eval", type=str, default=None, help="Checkpoint path for evaluation only mode")
    parser.add_argument("--loss_type", type=str, default="adaptive", choices=["adaptive", "margin"], help="Loss type: 'adaptive' for adaptive_ranking_loss_simple, 'margin' for margin ranking loss")
    args = parser.parse_args()

    if args.eval:
        eval_only(args.eval, use_autoregressive_decoder=(args.policy == "AR"), loss_type=args.loss_type)
    else:
        train_transformer_policy(
            use_autoregressive_decoder=(args.policy == "AR"),
            runname_prefix=args.runname,
            resume_checkpoint=args.resume,
            loss_type=args.loss_type
        )

