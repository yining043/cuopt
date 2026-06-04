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
    def __init__(self, d_model, n_feat=3):
        super().__init__()
        self.embed = nn.Linear(n_feat, d_model, bias=False)
    
    def forward(self, x):
        return self.embed(x)

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, use_sel=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_weight = nn.Parameter(torch.ones(1,4,1,1))
        if use_sel:
            self.sel_bias_weight = nn.Parameter(torch.zeros(1,num_heads,1,1))
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model, bias=False)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x, key_padding_mask, pos_scores, sel_scores=None):
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
        if sel_scores is not None and hasattr(self, 'sel_bias_weight'):
            total_attn_mask = total_attn_mask + self.sel_bias_weight * sel_scores

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
    def __init__(self, d_model, num_heads, num_layers, use_sel=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // self.num_heads
        self.use_sel = use_sel
        
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, use_sel=use_sel) 
            for _ in range(num_layers)
        ])
        
        self.pos_q_proj = nn.Linear(d_model, d_model, bias=False)
        self.pos_k_proj = nn.Linear(d_model, d_model, bias=False)

        if use_sel:
            self.sel_linear = nn.Linear(4, d_model, bias=False)
            self.sel_q_proj = nn.Linear(d_model, d_model, bias=False)
            self.sel_k_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, key_padding_mask, pos_embedding, selected_mask=None):
        batch_size, seq_len, _ = pos_embedding.shape
        q_pos = self.pos_q_proj(pos_embedding).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_pos = self.pos_k_proj(pos_embedding).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores_pos = torch.matmul(q_pos, k_pos.transpose(-2, -1)) / math.sqrt(self.head_dim)

        scores_sel = None
        if selected_mask is not None and self.use_sel:
            sel_4ch = RegressionHeadV2.decode_bitmask(selected_mask)  # [B, L, 4]
            sel_embed = self.sel_linear(sel_4ch)
            q_sel = self.sel_q_proj(sel_embed).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k_sel = self.sel_k_proj(sel_embed).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            scores_sel = torch.matmul(q_sel, k_sel.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        for layer in self.layers:
            x = layer(x, key_padding_mask, scores_pos, scores_sel)
        return x


# ============================================================================
# Regression Head
# ============================================================================

class RegressionHead(nn.Module):
    def __init__(self, d_model, use_selected_pool=False):
        super().__init__()
        self.use_selected_pool = use_selected_pool
        self.attn_pool = nn.Linear(d_model, 1)
        self.cost_embed = nn.Linear(1, d_model)
        mlp_in = d_model * 3 if use_selected_pool else d_model * 2
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, node_embed, padding_mask, log_cost_0, selected_mask=None):
        attn_scores = self.attn_pool(node_embed).squeeze(-1)
        attn_scores = attn_scores.masked_fill(padding_mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        pooled = (attn_weights.unsqueeze(-1) * node_embed).sum(dim=1)

        cost_feat = self.cost_embed(log_cost_0.unsqueeze(-1))
        if self.use_selected_pool and selected_mask is not None:
            sel = selected_mask.unsqueeze(-1).float()
            selected_pool = (node_embed * sel).sum(1) / sel.sum(1).clamp(min=1)
            combined = torch.cat([pooled, selected_pool, cost_feat], dim=-1)
        else:
            combined = torch.cat([pooled, cost_feat], dim=-1)
        return self.mlp(combined).squeeze(-1)


class RegressionHeadV2(nn.Module):
    """Enhanced regression head with attention-based selected/non-selected pooling,
    contrastive pooling (sel - nonsel), selection ratio, and type distribution features.
    selected_mask is a bitmask: bit0=sliding(1), bit1=vrp(2), bit2=recycle_vrp(4), bit3=two_opt(8)."""

    NUM_ANCHOR_TYPES = 4

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.attn_pool = nn.Linear(d_model, 1)
        self.sel_attn_pool = nn.Linear(d_model, 1)
        self.nonsel_attn_pool = nn.Linear(d_model, 1)
        self.cost_embed = nn.Linear(1, d_model)
        self.ratio_embed = nn.Linear(1, d_model // 4)
        self.type_ratio_embed = nn.Linear(self.NUM_ANCHOR_TYPES, d_model // 4)

        mlp_in = d_model * 4 + d_model // 4 + d_model // 4
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    @staticmethod
    def decode_bitmask(selected_mask):
        """Decode bitmask int to 4 float channels: [B, L, 4]."""
        sel_int = selected_mask.long()
        return torch.stack([
            (sel_int & 1).float(),
            ((sel_int >> 1) & 1).float(),
            ((sel_int >> 2) & 1).float(),
            ((sel_int >> 3) & 1).float(),
        ], dim=-1)

    def forward(self, node_embed, padding_mask, log_cost_0, selected_mask=None):
        # Global attention pool
        attn_scores = self.attn_pool(node_embed).squeeze(-1)
        attn_scores = attn_scores.masked_fill(padding_mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        pooled = (attn_weights.unsqueeze(-1) * node_embed).sum(dim=1)

        cost_feat = self.cost_embed(log_cost_0.unsqueeze(-1))

        sel = (selected_mask > 0)
        valid = ~padding_mask

        # Learned attention pool over SELECTED nodes only
        sel_scores = self.sel_attn_pool(node_embed).squeeze(-1)
        sel_scores = sel_scores.masked_fill(~sel | padding_mask, float('-inf'))
        sel_weights = F.softmax(sel_scores, dim=-1).nan_to_num(0.0)
        sel_pool = (sel_weights.unsqueeze(-1) * node_embed).sum(dim=1)

        # Learned attention pool over NON-SELECTED valid nodes
        nonsel_mask = valid & ~sel
        nonsel_scores = self.nonsel_attn_pool(node_embed).squeeze(-1)
        nonsel_scores = nonsel_scores.masked_fill(~nonsel_mask, float('-inf'))
        nonsel_weights = F.softmax(nonsel_scores, dim=-1).nan_to_num(0.0)
        nonsel_pool = (nonsel_weights.unsqueeze(-1) * node_embed).sum(dim=1)

        contrast = sel_pool - nonsel_pool

        n_sel = sel.float().sum(dim=1, keepdim=True)
        n_valid = valid.float().sum(dim=1, keepdim=True).clamp(min=1)
        ratio_feat = self.ratio_embed(n_sel / n_valid)

        # Per-type count ratios: how many anchors of each type
        sel_4ch = self.decode_bitmask(selected_mask)  # [B, L, 4]
        type_counts = sel_4ch.sum(dim=1)              # [B, 4]
        type_ratios = type_counts / n_valid.clamp(min=1)
        type_feat = self.type_ratio_embed(type_ratios) # [B, d_model//4]

        combined = torch.cat([pooled, sel_pool, contrast, cost_feat, ratio_feat, type_feat], dim=-1)
        return self.mlp(combined).squeeze(-1)


# ============================================================================
# Complete Policy Network
# ============================================================================

class CostPredictor(nn.Module):
    def __init__(self, 
                 d_model=128,
                 num_heads=4,
                 num_encoder_layers=3,
                 max_vehicles=21,
                 N=1001,
                 device='cpu',
                 mode='v2',
                 n_node_feat=3):
        super().__init__()
        
        self.d_model = d_model
        self.N = N
        self.max_vehicles = max_vehicles
        self.max_length = N + max_vehicles * 4
        self.device = torch.device(device)
        self.mode = mode
        # 3 = [x, y, demand]; 6 adds [earliest, latest, service] for CVRPTW.
        self.n_node_feat = n_node_feat
        
        self.feature_embed = NodeFeatureEmbedding(d_model, n_feat=n_node_feat)
        self.positional_encoding = self._create_positional_encoding(self.max_length, d_model).to(self.device)

        if mode == 'v2':
            self.sel_input_proj = nn.Linear(4, d_model)
            self.encoder = Encoder(d_model, num_heads, num_encoder_layers, use_sel=True)
            self.regression_head = RegressionHeadV2(d_model)
        elif mode == 'new':
            self.selected_bias = nn.Parameter(torch.zeros(2, 1, 1, d_model))
            self.encoder = Encoder(d_model, num_heads, num_encoder_layers, use_sel=True)
            self.regression_head = RegressionHead(d_model, use_selected_pool=True)
        else:  # 'ratio'
            self.selected_bias = nn.Parameter(torch.zeros(2, 1, 1, d_model))
            self.encoder = Encoder(d_model, num_heads, num_encoder_layers, use_sel=False)
            self.regression_head = RegressionHead(d_model, use_selected_pool=False)
        
        self.to(self.device)
        self.init_parameters()
        
    def init_parameters(self):
        is_v2 = (self.mode == 'v2')
        for name, param in self.named_parameters():
            if 'ln' in name or 'LayerNorm' in name:
                if 'weight' in name:
                    nn.init.ones_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
            elif 'sel_input_proj.weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.3)
            elif 'sel_input_proj.bias' in name:
                nn.init.zeros_(param)
            elif 'sel_embedding' in name or 'sel_linear' in name:
                nn.init.normal_(param, mean=0.0, std=0.2 if is_v2 else 0.02)
            elif 'sel_bias_weight' in name:
                nn.init.constant_(param, 1.0 if is_v2 else 0.1)
            else:
                stdv = 1. / math.sqrt(param.size(-1))
                param.data.uniform_(-stdv, stdv)
    
    def _create_positional_encoding(self, max_len, d_model):
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, nodes_tensor, demands_tensor, current_sol_tensor, selected_tensor, cost_0,
                tw_features=None):
        """
        Args:
            nodes_tensor:       [B, N, 2] - node coordinates
            demands_tensor:     [B, N, 1] - node demands
            current_sol_tensor: [B, max_length] - current solution
            selected_tensor:    [B, max_length] - anchor type bitmask (0-15: bit0=sliding, bit1=vrp, bit2=recycle_vrp, bit3=two_opt)
            cost_0:             [B] - initial cost (raw value)
            tw_features:        [B, N, k] - optional time-window features (e.g. earliest/latest/service, normalized). Required when n_node_feat>3.
        
        Returns:
            predicted: [B] - predicted log(cost[0] - cost[-1])
        """
        device = self.device
        batch_size, N, _ = nodes_tensor.shape
        max_length = current_sol_tensor.size(1)
        
        nodes_tensor = nodes_tensor.to(device)
        demands_tensor = demands_tensor.to(device)
        current_sol_tensor = current_sol_tensor.to(device)
        selected_tensor = selected_tensor.to(device).long()
        cost_0 = cost_0.to(device)
        
        if demands_tensor.dim() == 2:
            demands_tensor = demands_tensor.unsqueeze(-1)
        feat = [nodes_tensor, demands_tensor]
        if tw_features is not None:
            feat.append(tw_features.to(device))
        node_embeddings = self.feature_embed(torch.cat(feat, dim=-1))
        node_embeddings = torch.cat([
            node_embeddings, 
            node_embeddings[:, :1, :].repeat(1, max_length - N, 1)
        ], dim=1)
        
        pe = self.positional_encoding[:max_length].unsqueeze(0).expand(batch_size, -1, -1)
        sol_indices = current_sol_tensor.clamp(0, node_embeddings.size(1) - 1).unsqueeze(-1).expand(-1, -1, self.d_model)
        pos_embedding_gathered = torch.zeros_like(node_embeddings).to(device)
        pos_embedding_gathered.scatter_add_(1, sol_indices.long(), pe)
        
        encoder_mask = current_sol_tensor < 0
        encoder_mask[:, 0] = True

        if self.mode == 'v2':
            sel_4ch = RegressionHeadV2.decode_bitmask(selected_tensor)  # [B, L, 4]
            sel_signal = self.sel_input_proj(sel_4ch)
            node_embeddings = node_embeddings + sel_signal
            node_embeddings = self.encoder(node_embeddings, encoder_mask, pos_embedding_gathered, selected_tensor)
        elif self.mode == 'new':
            sel_bias = self.selected_bias[0].expand_as(node_embeddings)
            non_sel_bias = self.selected_bias[1].expand_as(node_embeddings)
            sel_mask = selected_tensor.unsqueeze(-1).expand_as(node_embeddings).float()
            node_embeddings = node_embeddings + sel_mask * sel_bias + (1 - sel_mask) * non_sel_bias
            node_embeddings = self.encoder(node_embeddings, encoder_mask, pos_embedding_gathered, selected_tensor)
        else:  # 'ratio'
            sel_bias = self.selected_bias[0].expand_as(node_embeddings)
            non_sel_bias = self.selected_bias[1].expand_as(node_embeddings)
            sel_mask = selected_tensor.unsqueeze(-1).expand_as(node_embeddings).float()
            node_embeddings = node_embeddings + sel_mask * sel_bias + (1 - sel_mask) * non_sel_bias
            node_embeddings = self.encoder(node_embeddings, encoder_mask, pos_embedding_gathered)

        log_cost_0 = torch.log(cost_0.float().clamp(min=1.0))
        if self.mode in ('v2', 'new'):
            predicted = self.regression_head(node_embeddings, encoder_mask, log_cost_0, selected_tensor)
        else:
            predicted = self.regression_head(node_embeddings, encoder_mask, log_cost_0)
        return predicted





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
    model = CostPredictor()
    if len(sys.argv) > 1:
        load_checkpoint(sys.argv[1], model)
    data = torch.load('ml_data_anchor.pt')
    indices = np.random.permutation(len(data['nodes_tensor']))
    n = min(500, len(indices))
    nodes_tensor = data['nodes_tensor'][indices[:n]]
    demands_tensor = data['demands_tensor'][indices[:n]]
    current_sol_tensor = data['current_sol_tensor'][indices[:n]]
    selected_tensor = data['selected_tensor'][indices[:n]]
    previous_cost = data['cost_tensor'][indices[:n]]

    cost_0 = previous_cost[:, 0]
    target_ratio = previous_cost[:, -1] / previous_cost[:, 0]

    print(f"Samples: {n}")
    print(f"cost_0: min={cost_0.min():.1f}, max={cost_0.max():.1f}, mean={cost_0.mean():.1f}")
    print(f"target_ratio: min={target_ratio.min():.4f}, max={target_ratio.max():.4f}, mean={target_ratio.mean():.4f}")

    predicted = model(nodes_tensor, demands_tensor, current_sol_tensor, selected_tensor, cost_0)
    print(f"predicted_ratio: min={predicted.min():.4f}, max={predicted.max():.4f}, mean={predicted.mean():.4f}")
    print(f"Sample predictions vs targets (first 10):")
    for i in range(min(10, n)):
        print(f"  [{i}] pred={predicted[i].item():.4f}, target={target_ratio[i].item():.4f}, cost_0={cost_0[i].item():.0f}")

if __name__ == "__main__":
    main()
