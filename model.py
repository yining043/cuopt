from json import encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import sys

# ============================================================================
# Feature Embedding
# ============================================================================

class NodeFeatureEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.embed = nn.Linear(3, d_model, bias=False)
    
    def forward(self, x):
        return self.embed(x)

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_weight = nn.Parameter(torch.ones(1,4,1,1))
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model, bias=False)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x, key_padding_mask, pos_scores):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        padding_mask_expanded = torch.zeros(batch_size, 1, 1, seq_len, device=x.device, dtype=x.dtype)
        if key_padding_mask is not None:
            padding_mask_expanded = padding_mask_expanded.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), 
                float("-inf")
            )

        total_attn_mask = self.pos_bias_weight * pos_scores + padding_mask_expanded

        attn_out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=total_attn_mask,
            dropout_p=0.0,
            is_causal=False
        )
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        attn_out = self.out_proj(attn_out)
    
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // self.num_heads
        
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads) 
            for _ in range(num_layers)
        ])
        
        self.pos_q_proj = nn.Linear(d_model, d_model, bias=False)
        self.pos_k_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, key_padding_mask, pos_embedding):
        batch_size, seq_len, _ = pos_embedding.shape
        q_pos = self.pos_q_proj(pos_embedding).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_pos = self.pos_k_proj(pos_embedding).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores_pos = torch.matmul(q_pos, k_pos.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        for layer in self.layers:
            x = layer(x, key_padding_mask, scores_pos)
        return x


# ============================================================================
# Decoder
# ============================================================================

class Decoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # 将每个节点的embedding映射到选择概率
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_model*2, d_model, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1)  # 输出每个节点的选择logit
        )

    def forward(self, node_embed):
        # node_embed: [B, max_length, d_model]
        # 输出: [B, max_length] - logits（未经过sigmoid）
        logits = self.mlp(node_embed).squeeze(-1)
        return logits

# ============================================================================
# Complete Policy Network
# ============================================================================

class Policy(nn.Module):
    def __init__(self, 
                 d_model=128,
                 num_heads=4,
                 num_encoder_layers=3,
                 max_vehicles=21,
                 N=1001,
                 device='cpu'):
        super().__init__()
        
        self.d_model = d_model
        self.N = N
        self.max_vehicles = max_vehicles
        self.max_length = N + max_vehicles * 4
        self.device = torch.device(device)
        
        self.feature_embed = NodeFeatureEmbedding(d_model)
        self.positional_encoding = self._create_positional_encoding(self.max_length, d_model).to(self.device)

        self.encoder = Encoder(d_model, num_heads, num_encoder_layers)
        self.candidate_bias = nn.Parameter(torch.zeros(2, 1, 1, d_model))
        self.decoder = Decoder(d_model)
        
        self.to(self.device)
        self.init_parameters()
        
    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
    
    def _create_positional_encoding(self, max_len, d_model):
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, nodes_tensor, demands_tensor, current_sol_tensor, candidates_tensor):
        """
        Args:
            nodes_tensor: [B, N, 2] - 节点坐标
            demands_tensor: [B, N, 1] - 节点需求
            current_sol_tensor: [B, max_length] - 当前解
            candidates_tensor: [B, 1, max_length] - 候选节点mask
        
        Returns:
            logits: [B, max_length] - 每个位置/节点的选择logit
        """
        device = self.device
        batch_size, N, _ = nodes_tensor.shape
        max_length = current_sol_tensor.size(1)
        
        nodes_tensor = nodes_tensor.to(device)
        demands_tensor = demands_tensor.to(device)
        current_sol_tensor = current_sol_tensor.to(device)
        candidates_tensor = candidates_tensor.to(device)
        
        # 特征嵌入：[B, N, d_model]
        node_embeddings = self.feature_embed(torch.cat([nodes_tensor, demands_tensor], dim=-1))
        # 扩展到 max_length（为 dummy depot 节点复制第一个节点的 embedding）
        node_embeddings = torch.cat([
            node_embeddings, 
            node_embeddings[:, :1, :].repeat(1, max_length - N, 1)
        ], dim=1)
        
        # 位置编码：基于 current_sol_tensor 中的节点顺序
        pe = self.positional_encoding[:max_length].unsqueeze(0).expand(batch_size, -1, -1)
        sol_indices = current_sol_tensor.clamp(0, node_embeddings.size(1) - 1).unsqueeze(-1).expand(-1, -1, self.d_model)
        pos_embedding_gathered = torch.zeros_like(node_embeddings).to(device)
        pos_embedding_gathered.scatter_add_(1, sol_indices.long(), pe)
        
        # 候选节点 bias：区分候选节点和非候选节点
        candidate_bias = self.candidate_bias[0].expand_as(node_embeddings)
        non_candidate_bias = self.candidate_bias[1].expand_as(node_embeddings)
        candidates_mask = candidates_tensor.squeeze(1).unsqueeze(-1).expand_as(node_embeddings).float()
        node_embeddings += candidates_mask * candidate_bias + (1 - candidates_mask) * non_candidate_bias

        # Encoder：编码节点特征
        encoder_mask = current_sol_tensor < 0  # padding mask
        encoder_mask[:, 0] = True  # mask depot
        node_embeddings = self.encoder(node_embeddings, encoder_mask, pos_embedding_gathered)
        
        # Decoder：对每个节点输出选择 logit
        logits = self.decoder(node_embeddings)  # [B, max_length]
        
        return logits





##***
def visualize(nodes, current_sol, selected, logits, candidates, percentile=50, threshold=0.5, save_path='visualization.png'):
    """
    逻辑：
    1. 接收 1085 长度的 mask。
    2. 找出 mask 为 1 的索引 (比如 [5, 100, 1005])。
    3. 把所有 > 1000 的索引强制变成 0 (Depot)。
    4. 画图。
    """
    # --- 1. 数据展平 ---
    def to_flat(x):
        return x.detach().cpu().numpy().reshape(-1) if torch.is_tensor(x) else x
    
    # Nodes: (1001, 2) -> 保持不动
    nodes_np = nodes.detach().cpu().numpy().squeeze()
    if nodes_np.ndim == 3: nodes_np = nodes_np[0]
    
    num_nodes = len(nodes_np) # 1001

    # 其他数据展平 (长度应为 1085)
    sol = to_flat(current_sol).astype(int)
    mask_sel = to_flat(selected)
    mask_cand = to_flat(candidates)
    scores = to_flat(logits)

    # --- 2. 核心：获取索引并映射 Depot ---
    
    def get_indices_with_depot_mapping(mask):
        # 1. 找出 Mask 为 1 的原始位置 (0~1084)
        raw_indices = np.where(mask > 0.5)[0]
        
        # 2. 复制一份，防止修改原始数据
        mapped_indices = raw_indices.copy()
        
        # 3. 关键逻辑：多出来的(>=1001)都是Depot(0)
        mapped_indices[mapped_indices >= num_nodes] = 0
        
        return mapped_indices

    # A. 候选集
    cand_indices = get_indices_with_depot_mapping(mask_cand)
    
    # B. 真实标签
    true_indices = get_indices_with_depot_mapping(mask_sel)
    
    # C. 预测结果
    softmax_scores = 1 / (1 + np.exp(-scores))
    # 逻辑：分数前 50% 或者 概率 > 0.5
    # pred_mask = scores > scores.mean() + percentile * scores.std()
    # pred_mask = pred_mask & (mask_cand > 0)
    pred_mask =  (softmax_scores > threshold)
    pred_indices = get_indices_with_depot_mapping(pred_mask)

    # D. 路径映射 (防止 sol 里的 1001 越界)
    sol[sol >= num_nodes] = 0

    # --- 3. 画图 ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    plot_data = [
        (cand_indices, "All Candidates", "orange"),
        (true_indices, "Ground Truth", "green"),
        (pred_indices, f"Prediction (Top {100-percentile}%)", "red")
    ]

    for ax, (indices, title, color) in zip(axes, plot_data):
        # 背景点
        ax.scatter(nodes_np[:,0], nodes_np[:,1], c='lightgray', s=10, alpha=0.5)
        # Depot
        ax.scatter(nodes_np[0,0], nodes_np[0,1], c='black', marker='s', s=60, zorder=5, label='Depot')
        
        # 路径
        if len(sol) > 0:
            ax.plot(nodes_np[sol, 0], nodes_np[sol, 1], c='blue', lw=0.5, alpha=0.3)
        
        # 选中的点 (分离 Depot 和 Customer 以便统计)
        depot_hits = indices[indices == 0]
        customer_hits = indices[indices != 0]
        
        # 画 Customer
        if len(customer_hits) > 0:
            ax.scatter(nodes_np[customer_hits, 0], nodes_np[customer_hits, 1], 
                       c=color, s=50, edgecolors='white', zorder=10, 
                       label=f'Cust ({len(customer_hits)})')
        
        # 标题显示 Depot 选中次数
        full_title = f"{title}\n(Cust: {len(customer_hits)}, Depot: {len(depot_hits)})"
        ax.set_title(full_title)
        
        # 修复 Legend 报错：只有当真的画了东西时才显示图例
        if len(customer_hits) > 0:
            ax.legend(loc='upper right')
            
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path)

# --- 调用方法 ---
# idx = 0
# visualize(nodes_tensor[idx].cpu(),current_sol_tensor[idx].cpu(), selected_tensor[idx].cpu(),logits.detach()[idx].cpu(),threshold=0.6)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint.get('global_step', 0)

def main():
    model = Policy()
    if len(sys.argv) > 1:
        load_checkpoint(sys.argv[1], model)
    data = torch.load('ml_data_full_100.pt')
    # shuffle
    indices = np.random.permutation(len(data['nodes_tensor']))
    # indices = range(len(data['nodes_tensor']))
    nodes_tensor = data['nodes_tensor'][indices[:500]]
    demands_tensor = data['demands_tensor'][indices[:500]]
    current_sol_tensor = data['current_sol_tensor'][indices[:500]]
    candidates_tensor = data['candidates_tensor'][indices[:500]]
    selected_tensor = data['selected_tensor'][indices[:500]]
    
    # 计算 label 的比例
    candidates_mask = candidates_tensor.squeeze(1).bool()  # [B, max_length]
    selected_mask = selected_tensor.squeeze(1).bool()      # [B, max_length]
    
    # 统计候选节点中被选中的节点比例
    num_candidates = candidates_mask.sum().item()
    num_selected = (selected_mask & candidates_mask).sum().item()
    positive_ratio = num_selected / num_candidates if num_candidates > 0 else 0
    
    print(f"Label Statistics:")
    print(f"  Total candidates: {num_candidates}")
    print(f"  Selected (positive): {num_selected}")
    print(f"  Not selected (negative): {num_candidates - num_selected}")
    print(f"  Positive ratio: {positive_ratio:.4f} ({positive_ratio*100:.2f}%)")
    print(f"  Negative ratio: {1-positive_ratio:.4f} ({(1-positive_ratio)*100:.2f}%)")
    print()

    # 推荐的 pos_weight
    recommended_pos_weight = (1 - positive_ratio) / positive_ratio
    print(f"Recommended pos_weight for balanced training: {recommended_pos_weight:.2f}")
    print(f"  Formula: neg_ratio / pos_ratio = {1-positive_ratio:.4f} / {positive_ratio:.4f}")
    print()
    print("Sample logits (first 10):")
    
    # 每个样本的统计
    per_sample_candidates = candidates_mask.sum(dim=1).float()
    per_sample_selected = (selected_mask & candidates_mask).sum(dim=1).float()
    per_sample_ratio = per_sample_selected / per_sample_candidates
    
    print(f"Per-sample statistics:")
    print(f"  Avg candidates per sample: {per_sample_candidates.mean().item():.2f}")
    print(f"  Avg selected per sample: {per_sample_selected.mean().item():.2f}")
    print(f"  Avg positive ratio per sample: {per_sample_ratio.mean().item():.4f}")
    print(f"  Min/Max positive ratio: {per_sample_ratio.min().item():.4f} / {per_sample_ratio.max().item():.4f}")
    print()
    
    logits = model(nodes_tensor, demands_tensor, current_sol_tensor, candidates_tensor)
    torch.set_printoptions(precision=10)
    import pdb; pdb.set_trace()
    print(logits[:10])

if __name__ == "__main__":
    main()