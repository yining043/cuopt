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
import matplotlib.pyplot as plt
import numpy as np

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

def plot_ranking_confusion_matrix(logits, score, save_path=None):
    """
    绘制10分类排名混淆矩阵（绝对位置）
    
    Args:
        logits: 模型输出的logits（作为预测值） [B, T]
        target: 归一化后的目标分数 [B, T]
        save_path: 保存路径，如果为None则不绘图，只返回矩阵
    
    Returns:
        confusion_matrix_np: 混淆矩阵的numpy数组 [T, T]
    """
    B, T = logits.shape
    
    actual_ranks = score.argsort(dim=1, descending=True)  # [B, T], actual_ranks[b, i] = 实际排名第i的trial索引
    pred_ranks = logits.argsort(dim=1)  # [B, T], pred_ranks[b, i] = 预测排名第i的trial索引
    pred_rank_positions = pred_ranks.argsort(dim=1)  # [B, T], pred_rank_positions[b, trial_idx] = 该trial的预测排名位置
    pred_positions = pred_rank_positions.gather(1, actual_ranks)  # [B, T], 这些trial的预测排名位置
    actual_positions = torch.arange(T, device=logits.device).unsqueeze(0).expand(B, -1)  # [B, T]
    flat_actual = actual_positions.flatten()  # [B*T]
    flat_pred = pred_positions.flatten()  # [B*T]
    linear_indices = flat_actual * T + flat_pred  # [B*T]
    counts = torch.bincount(linear_indices, minlength=T * T)  # [T*T]
    confusion_matrix = counts.view(T, T)  # [T, T]
    confusion_matrix_np = confusion_matrix.cpu().numpy() / B
    
    # 如果提供了save_path，则绘图
    if save_path is not None:
        accuracy = confusion_matrix_np.diagonal().sum() / T
        plt.figure(figsize=(8, 7))
        plt.imshow(confusion_matrix_np, cmap='Blues', vmin=0, vmax=1)
        plt.colorbar(label='Probability')
        plt.xlabel('Actual Rank')
        plt.ylabel('Predicted Rank')
        plt.title(f'Ranking Confusion Matrix (Accuracy: {accuracy:.2%})')
        plt.xticks(range(T))
        plt.yticks(range(T))
        # 添加数值标注
        for i in range(T):
            for j in range(T):
                if confusion_matrix_np[i, j] > 0:
                    plt.text(j, i, f'{confusion_matrix_np[i, j]:.2f}', 
                            ha='center', va='center', 
                            color='white' if confusion_matrix_np[i, j] > 0.5 else 'black')
        plt.savefig(save_path)
        plt.close()
    
    return confusion_matrix_np

def plot_accuracy_matrix(logits, score, save_path=None):
    """
    绘制相对顺序混淆矩阵
    
    Args:
        logits: 模型输出的logits（作为预测值） [B, T]
        target: 归一化后的目标分数 [B, T]
        save_path: 保存路径，如果为None则不绘图，只返回矩阵
    
    Returns:
        accuracy_matrix_np: 正确率矩阵的numpy数组 [T, T]
    """
    # 计算10x10相对顺序正确率矩阵
    # 只统计相对顺序：i>j时是否正确，不看全局rank id
    B, T = logits.shape
    actual_ranks = score.argsort(dim=1, descending=True)  # [B, T]
    pred_ranks = logits.argsort(dim=1)  # [B, T]
    pred_rank_positions = pred_ranks.argsort(dim=1)  # [B, T], pred_rank_positions[b, trial_idx] = 该trial的预测排名位置
    pred_positions = pred_rank_positions.gather(1, actual_ranks)  # [B, T], 每个实际排名位置对应的预测排名位置
    pred_i = pred_positions.unsqueeze(2)  # [B, T, 1]
    pred_j = pred_positions.unsqueeze(1)  # [B, 1, T]
    is_correct = (pred_i > pred_j).float()  # [B, T, T], 1表示正确，0表示错误
    mask = torch.tril(torch.ones(T, T, device=logits.device), diagonal=-1).bool()  # [T, T]
    mask = mask.unsqueeze(0).expand(B, -1, -1)  # [B, T, T]
    correct_matrix = (is_correct * mask.float()).sum(dim=0)  # [T, T]
    total_matrix = mask.float().sum(dim=0)  # [T, T]
    accuracy_matrix = correct_matrix / (total_matrix + 1e-8)
    accuracy_matrix_np = accuracy_matrix.cpu().numpy()
    
    # 如果提供了save_path，则绘图
    if save_path is not None:
        plt.figure(figsize=(8, 7))
        plt.imshow(accuracy_matrix_np, cmap='Blues', vmin=0, vmax=1)
        plt.colorbar(label='Accuracy')
        plt.xlabel('Actual Rank j')
        plt.ylabel('Actual Rank i')
        plt.title('Relative Order Accuracy (i>j)')
        plt.xticks(range(T))
        plt.yticks(range(T))
        # 只在下三角区域显示数值
        for i in range(T):
            for j in range(T):
                if i > j:
                    plt.text(j, i, f'{accuracy_matrix_np[i, j]:.2f}', 
                            ha='center', va='center', color='white' if accuracy_matrix_np[i, j] > 0.5 else 'black')
        plt.savefig(save_path)
        plt.close()
    
    return accuracy_matrix_np

def compute_matrix_metrics(accuracy_matrix, ranking_confusion_matrix):
    T = accuracy_matrix.shape[0]
    lower_tri_mask = np.tril(np.ones((T, T)), k=-1).astype(bool)
    accuracy_lower_tri_mean = accuracy_matrix[lower_tri_mask].mean()
    ranking_diag_mean = np.diag(ranking_confusion_matrix).mean()
    
    return accuracy_lower_tri_mean, ranking_diag_mean


def loss_function(logits, score, temperature=1.0):
    # 1. 显微镜操作：把当前 Batch 内部微小的差异放大到标准尺度
    # 这一步保证了无论搜索处于哪个阶段，梯度都是饱满的
    mean = score.mean(dim=1, keepdim=True)
    std = score.std(dim=1, keepdim=True) + 1e-8
    normalized_costs = (score - mean) / std
    
    # 2. 生成概率分布 (越小越好，取反)
    target_probs = F.softmax(-normalized_costs / temperature, dim=1)
    
    # 3. 拟合
    pred_log_probs = F.log_softmax(logits, dim=1)
    return F.kl_div(pred_log_probs, target_probs, reduction='batchmean')

def loss_function_(logits, score, mode='relative'):
    # 1. Normalized MSE Loss
    # since we want to minimize the loss, we need to negate the score !!!
    target_score = - score
    s_mean = target_score.mean(dim=(1), keepdim=True) # [B, 1]
    target = target_score - s_mean
    # l_mean, l_std = logits.mean(dim=(1), keepdim=True), logits.std(dim=(1), keepdim=True) + 1e-6
    # preds = (logits - l_mean) / l_std
    mse_loss = F.mse_loss(logits, target)

    # 3. Relative Order Accuracy Loss
    # 对于所有 i>j 的情况，计算相对顺序预测的loss（向量化实现）
    # 基于实际排名位置：如果实际排名i > 实际排名j（i更差），预测中也应该i > j
    B, T = logits.shape
    
    # 获取实际排名位置（基于target，越大越好）
    actual_ranks = target.argsort(dim=1)  # [B, T], actual_ranks[b, i] = 实际排名第i的trial索引
    # 构建实际排名位置映射：trial_idx -> rank_position（向量化）
    actual_rank_positions = actual_ranks.argsort(dim=1)  # [B, T], actual_rank_positions[b, i] = trial i的实际排名位置
    
    # 使用广播计算所有i>j的相对顺序
    # actual_rank_positions: [B, T]
    # 扩展为 [B, T, 1] 和 [B, 1, T] 进行广播
    actual_i_pos = actual_rank_positions.unsqueeze(2)  # [B, T, 1]
    actual_j_pos = actual_rank_positions.unsqueeze(1)  # [B, 1, T]
    actual_order = (actual_i_pos > actual_j_pos).float()  # [B, T, T], 1 if i排在j之后，0 otherwise
    
    # 预测相对顺序：使用preds的差值
    pred_i_val = logits.unsqueeze(2)  # [B, T, 1]
    pred_j_val = logits.unsqueeze(1)  # [B, 1, T]
    pred_order_prob = torch.sigmoid((pred_i_val - pred_j_val) * 100)  # [B, T, T]
    
    # 只统计i>j的情况（下三角矩阵，不包括对角线）
    mask = torch.tril(torch.ones(T, T, device=logits.device), diagonal=-1).bool()  # [T, T], 下三角mask
    mask = mask.unsqueeze(0).expand(B, -1, -1)  # [B, T, T]
    
    # BCE loss：预测概率应该匹配实际顺序
    bce_loss = F.binary_cross_entropy(pred_order_prob, actual_order, reduction='none')  # [B, T, T]
    relative_order_loss = (bce_loss * mask).sum() / mask.sum()
    
    # # 绘制混淆矩阵
    import pdb; pdb.set_trace()
    small_idx = target.std(1).sort()[1][:B//2]   
    large_idx = target.std(1).sort()[1][-B//2:]
    plot_accuracy_matrix(logits, score, 'confusion_matrix.png')
    plot_ranking_confusion_matrix(logits, score, 'ranking_confusion_matrix.png')
    plot_ranking_confusion_matrix(logits[small_idx], target[small_idx])
    plot_ranking_confusion_matrix(logits[large_idx], target[large_idx])

    if mode == 'mse': 
        return mse_loss
    elif mode == 'relative': 
        return relative_order_loss

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

        ### random shuffle the selected candidates
        B, T, N = selected.shape
        shuffled_idx = torch.rand(B, T, device=device).argsort(dim=1) # [B, 10]  
        score = torch.gather(score, 1, shuffled_idx)  
        shuffled_idx_expanded = shuffled_idx.unsqueeze(-1).expand(-1, -1, N)
        selected = torch.gather(selected, 1, shuffled_idx_expanded)   
        # score = score[:, 1:]
        # selected = selected[:, 1:]
        
        ## end of random shuffle
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
            # 使用当前batch计算矩阵指标
            current_logits = logits.detach().cpu()
            current_scores = score.detach().cpu()
            current_accuracy_matrix = plot_accuracy_matrix(current_logits, current_scores)
            current_ranking_confusion_matrix = plot_ranking_confusion_matrix(current_logits, current_scores)
            current_accuracy_lower_tri_mean, current_ranking_diag_mean = compute_matrix_metrics(
                current_accuracy_matrix, current_ranking_confusion_matrix
            )
            
            wandb.log({
                "train/loss": loss.item(),
                "train/top1_acc": acc.item(),
                "train/accuracy_lower_tri_mean": current_accuracy_lower_tri_mean,
                "train/ranking_diag_mean": current_ranking_diag_mean,
                "global_step": global_step,
                "train/logits_mean": logits.mean().item(),
                "train/logits_std": logits.std().item(),
                "epoch": epoch + 1,            
                })
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)
    return avg_train_loss, avg_train_acc, global_step

def evaluate(model, val_loader, device):
    model.eval()
    val_loss = 0
    val_acc = 0
    logits_list = []
    scores_list = []
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        nodes = batch['nodes'].to(device)
        demands = batch['demands'].to(device)
        current_sol = batch['current_sol'].to(device)
        candidates = batch['candidates'].to(device)
        selected = batch['selected'].to(device)
        score = batch['score'].to(device)

        ### random shuffle the selected candidates
        B, T, N = selected.shape
        shuffled_idx = torch.rand(B, T, device=device).argsort(dim=1) # [B, 10]  
        score = torch.gather(score, 1, shuffled_idx)  
        shuffled_idx_expanded = shuffled_idx.unsqueeze(-1).expand(-1, -1, N)
        selected = torch.gather(selected, 1, shuffled_idx_expanded)   
        ## end of random shuffle
        
        logits = model(nodes, demands, current_sol, candidates, selected)
        v_loss = loss_function(logits, score)
        val_loss += v_loss.item()
        val_acc += (logits.argmax(dim=1) == score.argmin(dim=1)).float().mean().item()
        
        logits_list.append(logits.detach().cpu())
        scores_list.append(score.detach().cpu())
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)

    # 计算矩阵指标
    combined_logits = torch.cat(logits_list, dim=0)
    combined_scores = torch.cat(scores_list, dim=0)
    accuracy_matrix = plot_accuracy_matrix(combined_logits, combined_scores)
    ranking_confusion_matrix = plot_ranking_confusion_matrix(combined_logits, combined_scores)
    accuracy_lower_tri_mean, ranking_diag_mean = compute_matrix_metrics(accuracy_matrix, ranking_confusion_matrix)

    return avg_val_loss, avg_val_acc, accuracy_matrix, ranking_confusion_matrix, accuracy_lower_tri_mean, ranking_diag_mean

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
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    wandb.init(project="cuopt", name=runname or os.path.basename(checkpoint_dir))
    
    for epoch in range(start_epoch, num_epochs):
        # Train one epoch
        avg_train_loss, avg_train_acc, global_step = train_one_epoch(
            model, train_loader, optimizer, device, epoch, global_step
        )
        
        # Evaluate
        with torch.no_grad():
            avg_val_loss, avg_val_acc, accuracy_matrix, ranking_confusion_matrix, val_accuracy_lower_tri_mean, val_ranking_diag_mean = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}: Avg Acc: {avg_train_acc:.4f}, Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")
        # Convert matrices to row lists for wandb.Table
        max_trails = accuracy_matrix.shape[0]
        accuracy_table_data = accuracy_matrix.tolist()
        ranking_table_data = ranking_confusion_matrix.tolist()
        columns = [f"Rank {i}" for i in range(max_trails)]
        
        wandb.log({
            "val/loss": avg_val_loss, 
            "val/top1_acc": avg_val_acc,
            "val/accuracy_lower_tri_mean": val_accuracy_lower_tri_mean,
            "val/ranking_diag_mean": val_ranking_diag_mean,
            "epoch": epoch+1,
            "global_step": global_step,
            "val/accuracy_matrix": wandb.Table(data=accuracy_table_data, columns=columns),
            "val/ranking_confusion_matrix": wandb.Table(data=ranking_table_data, columns=columns)
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