import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
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


class FlashSelfAttentionLayer(nn.Module):
    """Pre-norm Transformer block using SDPA without dense LxL bias tensors."""

    def __init__(self, d_model, num_heads, ffn_mult=4):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(ffn_mult * d_model)
        self.ffn_in = nn.Linear(d_model, 2 * hidden, bias=False)
        self.ffn_out = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x, key_padding_mask=None):
        batch_size, seq_len, _ = x.shape
        h = self.ln1(x)
        qkv = self.qkv_proj(h)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros(
                batch_size, 1, 1, seq_len, dtype=q.dtype, device=q.device)
            attn_mask = attn_mask.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf"))

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        x = x + self.out_proj(attn_out)

        gate, value = self.ffn_in(self.ln2(x)).chunk(2, dim=-1)
        x = x + self.ffn_out(F.silu(gate) * value)
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


class FlashRouteEncoder(nn.Module):
    """Route-order encoder for the incumbent solution state.

    The encoder intentionally keeps candidate selection out of self-attention.
    In the online RL callback all K arms share the same route state, so this
    block can run once per callback and the lightweight candidate head can score
    all K masks against the shared contextual route embeddings.
    """

    def __init__(self, d_model, num_heads, num_layers, n_node_feat, max_length):
        super().__init__()
        self.d_model = d_model
        self.feature_embed = NodeFeatureEmbedding(d_model, n_feat=n_node_feat)
        self.route_meta_proj = nn.Linear(5, d_model, bias=False)
        self.layers = nn.ModuleList([
            FlashSelfAttentionLayer(d_model, num_heads) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.register_buffer(
            "positional_encoding",
            self._create_positional_encoding(max_length, d_model),
            persistent=False,
        )

    @staticmethod
    def _create_positional_encoding(max_len, d_model):
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, node_features, route_index, valid_route):
        batch_size, seq_len = route_index.shape
        n_nodes = node_features.size(1)
        gather_index = route_index.clamp(min=0, max=max(n_nodes - 1, 0))
        gather_index = torch.where(
            (route_index >= 0) & (route_index < n_nodes),
            gather_index,
            torch.zeros_like(gather_index),
        )
        route_features = torch.gather(
            node_features, 1,
            gather_index.unsqueeze(-1).expand(-1, -1, node_features.size(-1)),
        )

        x = self.feature_embed(route_features)
        pos = self.positional_encoding[:seq_len].to(x.device, x.dtype)
        x = x + pos.unsqueeze(0)

        denom = max(seq_len - 1, 1)
        pos_frac = torch.arange(seq_len, device=x.device, dtype=x.dtype) / float(denom)
        is_dummy = (route_index >= n_nodes) & valid_route
        is_depot = ((route_index == 0) | is_dummy).to(x.dtype)
        dummy_offset = (route_index - n_nodes).clamp(min=0)
        dummy_route_id = torch.div(dummy_offset, 4, rounding_mode="floor")
        dummy_slot = (dummy_offset % 4).to(x.dtype) / 3.0
        dummy_slot = torch.where(is_dummy, dummy_slot, torch.zeros_like(dummy_slot))

        first_valid = valid_route & (torch.cumsum(valid_route.long(), dim=1) == 1)
        route_start = first_valid | (is_dummy & ((dummy_offset % 4) == 0))
        route_marker = torch.where(
            is_dummy,
            dummy_route_id + 1,
            torch.zeros_like(dummy_route_id),
        )
        route_id = torch.cummax(route_marker, dim=1).values - 1
        route_id = route_id.clamp(min=0).to(x.dtype)
        route_id = route_id / route_id.max(dim=1, keepdim=True).values.clamp(min=1.0)

        valid_count = torch.cumsum(valid_route.long(), dim=1)
        local_start = torch.where(route_start, valid_count, torch.zeros_like(valid_count))
        last_start = torch.cummax(local_start, dim=1).values
        local_pos = (valid_count - last_start).clamp(min=0)
        local_pos = local_pos.to(x.dtype)
        local_pos = local_pos / local_pos.max(dim=1, keepdim=True).values.clamp(min=1.0)
        route_meta = torch.stack([
            is_depot,
            pos_frac.unsqueeze(0).expand(batch_size, -1),
            route_id,
            local_pos,
            dummy_slot,
        ], dim=-1)
        x = x + self.route_meta_proj(route_meta)

        key_padding_mask = ~valid_route
        x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        x = self.final_norm(x)
        return x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)


class FlashCandidateHeadV7(nn.Module):
    """Shared-KV candidate decoder with route-boundary-aware local features.

    Standalone candidate scorer used by CostPredictor ``mode='v7'``. It keeps the
    shared route encoder out of self-attention (the route state is encoded once
    by FlashRouteEncoder), conditions candidates through query tokens and
    scalar/pooled summaries, and lets both SDPA cross-attention layers read one
    shared route K/V cache when all K candidates belong to the same incumbent
    state.
    """

    NUM_ANCHOR_TYPES = 4
    EDGE_DIM = 7
    NUM_LATENT_TOKENS = 4
    NUM_TOKENS = 1 + NUM_ANCHOR_TYPES + NUM_LATENT_TOKENS
    SCALAR_DIM = 53

    def __init__(self, d_model, num_heads):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.global_attn_pool = nn.Linear(d_model, 1)
        self.sel_attn_pool = nn.Linear(d_model, 1)
        self.nonsel_attn_pool = nn.Linear(d_model, 1)
        self.type_queries = nn.Parameter(torch.empty(self.NUM_ANCHOR_TYPES, d_model))
        self.type_token_embed = nn.Parameter(torch.empty(self.NUM_ANCHOR_TYPES, d_model))

        self.cost_embed = nn.Linear(1, d_model)
        self.count_embed = nn.Linear(1 + self.NUM_ANCHOR_TYPES, d_model // 2)
        self.edge_summary_embed = nn.Linear(self.EDGE_DIM * 3, d_model // 2)
        self.edge_memory_proj = nn.Linear(self.EDGE_DIM, d_model, bias=False)
        self.type_scalar_proj = nn.Linear(1 + self.EDGE_DIM, d_model, bias=False)

        self.candidate_scalar_embed = nn.Sequential(
            nn.LayerNorm(self.SCALAR_DIM),
            nn.Linear(self.SCALAR_DIM, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.base_dim = d_model * 12 + d_model + d_model
        self.candidate_proj = nn.Sequential(
            nn.LayerNorm(self.base_dim),
            nn.Linear(self.base_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.token_mixer = FlashSelfAttentionLayer(d_model, num_heads, ffn_mult=2)
        self.token_mixer2 = FlashSelfAttentionLayer(d_model, num_heads, ffn_mult=2)
        self.cross_q_proj = nn.Linear(d_model, d_model, bias=False)
        self.cross_kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.cross_out_proj = nn.Linear(d_model, d_model, bias=False)
        self.cross2_q_proj = nn.Linear(d_model, d_model, bias=False)
        self.cross2_kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.cross2_out_proj = nn.Linear(d_model, d_model, bias=False)
        self.candidate_self_attn = FlashSelfAttentionLayer(d_model, num_heads, ffn_mult=2)
        self.latent_token_embed = nn.Parameter(torch.empty(self.NUM_LATENT_TOKENS, d_model))
        self.latent_condition_proj = nn.Linear(d_model * 2, self.NUM_LATENT_TOKENS * d_model)

        score_dim = self.base_dim + d_model * self.NUM_TOKENS + d_model * 2
        self.score_mlp = nn.Sequential(
            nn.LayerNorm(score_dim),
            nn.Linear(score_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    @staticmethod
    def _masked_attention_pool(x, mask, scorer):
        scores = scorer(x).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1).nan_to_num(0.0)
        return (weights.unsqueeze(-1) * x).sum(dim=1)

    @staticmethod
    def _masked_mean(values, mask):
        weights = mask.to(values.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp(min=1.0)
        return (values * weights).sum(dim=1) / denom

    @staticmethod
    def _masked_max(values, mask):
        neg_inf = torch.finfo(values.dtype).min
        out = values.masked_fill(~mask.unsqueeze(-1), neg_inf).amax(dim=1)
        has_value = mask.any(dim=1, keepdim=True)
        return torch.where(has_value, out, torch.zeros_like(out))

    @staticmethod
    def _masked_min(values, mask):
        pos_inf = torch.finfo(values.dtype).max
        out = values.masked_fill(~mask.unsqueeze(-1), pos_inf).amin(dim=1)
        has_value = mask.any(dim=1, keepdim=True)
        return torch.where(has_value, out, torch.zeros_like(out))

    @staticmethod
    def _masked_scalar_mean(values, mask):
        weights = mask.to(values.dtype)
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (values * weights).sum(dim=1, keepdim=True) / denom

    @classmethod
    def _masked_scalar_std(cls, values, mask):
        mean = cls._masked_scalar_mean(values, mask)
        weights = mask.to(values.dtype)
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        var = ((values - mean).square() * weights).sum(dim=1, keepdim=True) / denom
        return torch.sqrt(var.clamp(min=0.0) + 1e-9)

    @staticmethod
    def _masked_scalar_max(values, mask):
        neg_inf = torch.finfo(values.dtype).min
        out = values.masked_fill(~mask, neg_inf).amax(dim=1, keepdim=True)
        has_value = mask.any(dim=1, keepdim=True)
        return torch.where(has_value, out, torch.zeros_like(out))

    @staticmethod
    def _masked_scalar_min(values, mask):
        pos_inf = torch.finfo(values.dtype).max
        out = values.masked_fill(~mask, pos_inf).amin(dim=1, keepdim=True)
        has_value = mask.any(dim=1, keepdim=True)
        return torch.where(has_value, out, torch.zeros_like(out))

    @staticmethod
    def _shift_left(x):
        return torch.cat([x[:, :1], x[:, :-1]], dim=1)

    @staticmethod
    def _shift_right(x):
        return torch.cat([x[:, 1:], x[:, -1:]], dim=1)

    @staticmethod
    def _gather_route_coords(node_features, route_index):
        coords = node_features[..., :2]
        n_nodes = coords.size(1)
        gather_index = route_index.clamp(min=0, max=max(n_nodes - 1, 0))
        gather_index = torch.where(
            (route_index >= 0) & (route_index < n_nodes),
            gather_index,
            torch.zeros_like(gather_index),
        )
        return torch.gather(
            coords, 1, gather_index.unsqueeze(-1).expand(-1, -1, coords.size(-1)))

    @staticmethod
    def _gather_route_values(values, route_index):
        n_nodes = values.size(1)
        gather_index = route_index.clamp(min=0, max=max(n_nodes - 1, 0))
        gather_index = torch.where(
            (route_index >= 0) & (route_index < n_nodes),
            gather_index,
            torch.zeros_like(gather_index),
        )
        return torch.gather(
            values, 1, gather_index.unsqueeze(-1).expand(-1, -1, values.size(-1)))

    def _route_descriptors(self, node_features, route_index, valid_route, batch_size, shared_route):
        coords = self._gather_route_coords(node_features, route_index)
        if node_features.size(-1) > 2:
            demand = self._gather_route_values(node_features[..., 2:3], route_index).squeeze(-1)
        else:
            demand = torch.zeros(coords.shape[:2], dtype=coords.dtype, device=coords.device)

        route_tokens = route_index[:1] if shared_route else route_index
        depot_coords = node_features[:, :1, :2]
        if shared_route:
            coords = coords[:1].expand(batch_size, -1, -1)
            demand = demand[:1].expand(batch_size, -1)
            valid = valid_route[:1].expand(batch_size, -1)
            route_tokens = route_tokens.expand(batch_size, -1)
            depot_coords = depot_coords[:1].expand(batch_size, -1, -1)
        else:
            valid = valid_route

        dtype = coords.dtype
        n_nodes = node_features.size(1)
        is_dummy = (route_tokens >= n_nodes) & valid
        dummy_offset = (route_tokens - n_nodes).clamp(min=0)
        first_valid = valid & (torch.cumsum(valid.long(), dim=1) == 1)
        route_start = first_valid | (is_dummy & ((dummy_offset % 4) == 0))

        route_marker = torch.where(is_dummy, dummy_offset // 4 + 1, torch.zeros_like(dummy_offset))
        route_id = torch.cummax(route_marker, dim=1).values - 1
        route_id = route_id.clamp(min=0).to(dtype)
        route_id = route_id / route_id.max(dim=1, keepdim=True).values.clamp(min=1.0)

        valid_count = torch.cumsum(valid.long(), dim=1)
        local_start = torch.where(route_start, valid_count, torch.zeros_like(valid_count))
        last_start = torch.cummax(local_start, dim=1).values
        local_pos = (valid_count - last_start).clamp(min=0).to(dtype)
        local_pos = local_pos / local_pos.max(dim=1, keepdim=True).values.clamp(min=1.0)

        denom = max(coords.size(1) - 1, 1)
        pos_frac = torch.arange(coords.size(1), device=coords.device, dtype=dtype) / float(denom)
        pos_frac = pos_frac.unsqueeze(0).expand(batch_size, -1)
        is_depot_like = ((route_tokens == 0) | is_dummy).to(dtype)
        return coords, demand.to(dtype), depot_coords.to(dtype), pos_frac, route_id, local_pos, is_depot_like, valid

    def _edge_features(self, node_features, route_index, valid_route, batch_size, shared_route):
        coords = self._gather_route_coords(node_features, route_index)
        route_tokens = route_index[:1] if shared_route else route_index
        if shared_route:
            coords = coords[:1].expand(batch_size, -1, -1)
            valid = valid_route[:1].expand(batch_size, -1)
            route_tokens = route_tokens.expand(batch_size, -1)
        else:
            valid = valid_route

        n_nodes = node_features.size(1)
        is_dummy = (route_tokens >= n_nodes) & valid
        dummy_offset = (route_tokens - n_nodes).clamp(min=0)
        first_valid = valid & (torch.cumsum(valid.long(), dim=1) == 1)
        route_start = first_valid | (is_dummy & ((dummy_offset % 4) == 0))
        next_route_start = self._shift_right(route_start)

        prev_coords = self._shift_left(coords)
        next_coords = self._shift_right(coords)
        prev_valid = self._shift_left(valid) & valid & ~route_start
        next_valid = self._shift_right(valid) & valid & ~next_route_start
        edge_in = torch.linalg.vector_norm(coords - prev_coords, dim=-1)
        edge_out = torch.linalg.vector_norm(next_coords - coords, dim=-1)
        chord = torch.linalg.vector_norm(next_coords - prev_coords, dim=-1)
        edge_in = edge_in * prev_valid.to(edge_in.dtype)
        edge_out = edge_out * next_valid.to(edge_out.dtype)
        chord_valid = prev_valid & next_valid
        chord = chord * chord_valid.to(chord.dtype)
        savings = torch.where(
            chord_valid,
            edge_in + edge_out - chord,
            torch.zeros_like(chord),
        )

        denom = max(coords.size(1) - 1, 1)
        pos_frac = torch.arange(coords.size(1), device=coords.device, dtype=coords.dtype) / float(denom)
        pos_frac = pos_frac.unsqueeze(0).expand(batch_size, -1)
        is_depot_like = ((route_tokens == 0) | is_dummy).to(coords.dtype)

        edge = torch.stack([
            edge_in,
            edge_out,
            chord,
            savings,
            pos_frac,
            prev_valid.to(coords.dtype),
            next_valid.to(coords.dtype),
        ], dim=-1)
        edge = edge.masked_fill(~valid.unsqueeze(-1), 0.0)
        edge[..., 4] = edge[..., 4] * (1.0 - is_depot_like)
        return edge

    def _candidate_scalar_features(
        self,
        node_features,
        route_index,
        valid_route,
        batch_size,
        shared_route,
        edge,
        sel_any,
        nonsel,
        valid_for_candidate,
        type_counts,
        n_valid,
        n_sel,
        valid_candidate,
    ):
        coords, demand, depot_coords, pos_frac, route_id, local_pos, is_depot_like, _ = (
            self._route_descriptors(node_features, route_index, valid_route, batch_size, shared_route))

        n_sel_safe = n_sel.clamp(min=1.0)
        sel_ratio = n_sel / n_valid
        log_count_ratio = torch.log1p(n_sel) / torch.log1p(n_valid)
        type_mix = type_counts / n_sel_safe
        type_entropy = -(type_mix * (type_mix + 1e-6).log()).sum(dim=1, keepdim=True) / math.log(4.0)
        type_max = type_mix.amax(dim=1, keepdim=True)

        def scalar_triplet(values):
            v_min = self._masked_scalar_min(values, sel_any)
            v_max = self._masked_scalar_max(values, sel_any)
            return torch.cat([
                self._masked_scalar_mean(values, sel_any),
                self._masked_scalar_std(values, sel_any),
                v_max - v_min,
            ], dim=-1)

        demand_sum = (demand * sel_any.to(demand.dtype)).sum(dim=1, keepdim=True)
        demand_mean = demand_sum / n_sel_safe
        demand_max = self._masked_scalar_max(demand, sel_any)
        demand_std = self._masked_scalar_std(demand, sel_any)

        coord_mean = self._masked_mean(coords, sel_any)
        coord_sq_mean = self._masked_mean(coords.square(), sel_any)
        coord_std = torch.sqrt((coord_sq_mean - coord_mean.square()).clamp(min=0.0) + 1e-9)
        coord_min = self._masked_min(coords, sel_any)
        coord_max = self._masked_max(coords, sel_any)
        coord_bbox = coord_max - coord_min
        bbox_area = coord_bbox[:, :1] * coord_bbox[:, 1:2]
        centroid_dist = torch.linalg.vector_norm(coord_mean - depot_coords.squeeze(1), dim=-1, keepdim=True)

        depot_rate = self._masked_scalar_mean(is_depot_like, sel_any)
        edge_mean = self._masked_mean(edge, sel_any)
        edge_max = self._masked_max(edge, sel_any)
        edge_nonsel = self._masked_mean(edge, nonsel)
        boundary_rate = 1.0 - 0.5 * (edge_mean[:, 5:6] + edge_mean[:, 6:7])

        scalars = torch.cat([
            valid_candidate.to(edge.dtype).unsqueeze(-1),
            sel_ratio,
            log_count_ratio,
            type_mix,
            type_entropy,
            type_max,
            scalar_triplet(pos_frac),
            scalar_triplet(local_pos),
            scalar_triplet(route_id),
            demand_sum,
            demand_mean,
            demand_max,
            demand_std,
            coord_mean,
            coord_std,
            coord_bbox,
            bbox_area,
            centroid_dist,
            depot_rate,
            boundary_rate,
            edge_mean,
            edge_max,
            edge_mean - edge_nonsel,
        ], dim=-1)
        if scalars.size(-1) != self.SCALAR_DIM:
            raise RuntimeError(
                f"FlashCandidateHeadV7 scalar dim mismatch: got {scalars.size(-1)}, expected {self.SCALAR_DIM}")
        return scalars.masked_fill(~valid_candidate.unsqueeze(-1), 0.0)

    @staticmethod
    def _route_start_mask(route_tokens, valid, n_nodes):
        is_dummy = (route_tokens >= n_nodes) & valid
        dummy_offset = (route_tokens - n_nodes).clamp(min=0)
        first_valid = valid & (torch.cumsum(valid.long(), dim=1) == 1)
        return first_valid | (is_dummy & ((dummy_offset % 4) == 0))

    def _cross_attention_tokens_shared_kv(
        self,
        tokens,
        memory,
        valid_memory,
        shared_route,
        q_proj=None,
        kv_proj=None,
        out_proj=None,
    ):
        q_proj = self.cross_q_proj if q_proj is None else q_proj
        kv_proj = self.cross_kv_proj if kv_proj is None else kv_proj
        out_proj = self.cross_out_proj if out_proj is None else out_proj

        if shared_route:
            batch_size, n_tokens, _ = tokens.shape
            seq_len = memory.size(1)
            flat_tokens = tokens.reshape(1, batch_size * n_tokens, self.d_model)
            q = q_proj(flat_tokens).view(
                1, batch_size * n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
            kv = kv_proj(memory[:1]).view(
                1, seq_len, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(dim=0)

            attn_mask = None
            if valid_memory is not None:
                attn_mask = torch.zeros(1, 1, 1, seq_len, dtype=q.dtype, device=q.device)
                attn_mask = attn_mask.masked_fill(
                    (~valid_memory[:1])[:, None, None, :], float("-inf"))

            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
            attended = attended.transpose(1, 2).contiguous().view(
                batch_size, n_tokens, self.d_model)
            return tokens + out_proj(attended)

        batch_size, n_tokens, _ = tokens.shape
        seq_len = memory.size(1)
        q = q_proj(tokens).view(
            batch_size, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        kv = kv_proj(memory).view(
            batch_size, seq_len, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(dim=0)

        attn_mask = None
        if valid_memory is not None:
            attn_mask = torch.zeros(
                batch_size, 1, 1, seq_len, dtype=q.dtype, device=q.device)
            attn_mask = attn_mask.masked_fill(
                (~valid_memory)[:, None, None, :], float("-inf"))

        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, n_tokens, self.d_model)
        return tokens + out_proj(attended)

    def forward(self, route_embed, valid_route, log_cost_0, selected_mask,
                route_index, node_features, shared_state=False):
        valid_route = valid_route.bool()
        selected_mask = selected_mask.long()
        batch_size, _ = selected_mask.shape
        shared_route = shared_state and route_embed.size(0) == 1 and batch_size > 1
        if shared_route:
            route_base_single = route_embed[:1]
            valid_single = valid_route[:1]
            route_base = route_base_single.expand(batch_size, -1, -1)
            valid_for_candidate = valid_single.expand(batch_size, -1)
            edge_base = self._edge_features(
                node_features, route_index[:1], valid_single, 1, False).to(route_embed.dtype)
        else:
            route_base_single = route_embed
            valid_single = valid_route
            route_base = route_embed
            valid_for_candidate = valid_route
            edge_base = self._edge_features(
                node_features, route_index, valid_route, batch_size, False).to(route_embed.dtype)

        edge = edge_base.expand(batch_size, -1, -1) if shared_route else edge_base
        route_memory_single = route_base_single + self.edge_memory_proj(edge_base)
        route_memory = route_memory_single.expand(batch_size, -1, -1) if shared_route else route_memory_single

        sel_any = (selected_mask > 0) & valid_for_candidate
        valid_candidate = sel_any.any(dim=1)
        nonsel = valid_for_candidate & ~sel_any
        sel_4ch = RegressionHeadV2.decode_bitmask(selected_mask).to(route_embed.dtype)
        sel_4ch = sel_4ch * valid_for_candidate.unsqueeze(-1).to(sel_4ch.dtype)

        n_nodes = node_features.size(1)
        route_tokens = route_index[:1] if shared_route else route_index
        route_tokens = route_tokens.expand(batch_size, -1) if shared_route else route_tokens
        route_start = self._route_start_mask(route_tokens, valid_for_candidate, n_nodes)
        next_route_start = self._shift_right(route_start)
        prev_valid = self._shift_left(valid_for_candidate) & valid_for_candidate & ~route_start
        next_valid = self._shift_right(valid_for_candidate) & valid_for_candidate & ~next_route_start

        global_pool = self._masked_attention_pool(route_memory, valid_for_candidate, self.global_attn_pool)
        sel_pool = self._masked_attention_pool(route_memory, sel_any, self.sel_attn_pool)
        nonsel_pool = self._masked_attention_pool(route_memory, nonsel, self.nonsel_attn_pool)
        contrast = sel_pool - nonsel_pool

        prev_memory = self._shift_left(route_memory)
        next_memory = self._shift_right(route_memory)
        prev_pool = self._masked_mean(prev_memory, sel_any & prev_valid)
        next_pool = self._masked_mean(next_memory, sel_any & next_valid)
        local_delta = sel_pool - 0.5 * (prev_pool + next_pool)

        type_scores = torch.einsum("bld,td->blt", route_memory, self.type_queries)
        type_scores = type_scores / math.sqrt(float(self.d_model))
        type_scores = type_scores.masked_fill(sel_4ch <= 0, float("-inf"))
        type_weights = F.softmax(type_scores, dim=1).nan_to_num(0.0)
        type_pool = torch.einsum("blt,bld->btd", type_weights, route_memory)

        n_valid = valid_for_candidate.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        n_sel = sel_any.float().sum(dim=1, keepdim=True)
        type_counts = sel_4ch.sum(dim=1)
        type_ratios = type_counts / n_valid
        count_feat = self.count_embed(torch.cat([n_sel / n_valid, type_ratios], dim=-1))
        cost_feat = self.cost_embed(log_cost_0.unsqueeze(-1))

        edge_mean = self._masked_mean(edge, sel_any)
        edge_max = self._masked_max(edge, sel_any)
        edge_nonsel = self._masked_mean(edge, nonsel)
        edge_feat = self.edge_summary_embed(torch.cat([edge_mean, edge_max, edge_mean - edge_nonsel], dim=-1))

        scalar_input = self._candidate_scalar_features(
            node_features, route_index, valid_route, batch_size, shared_route, edge,
            sel_any, nonsel, valid_for_candidate, type_counts, n_valid, n_sel, valid_candidate)
        scalar_feat = self.candidate_scalar_embed(scalar_input.to(route_embed.dtype))

        type_mask = sel_4ch > 0
        type_edge = torch.einsum("blf,blt->btf", edge, type_mask.to(edge.dtype))
        type_edge = type_edge / type_mask.to(edge.dtype).sum(dim=1).unsqueeze(-1).clamp(min=1.0)
        type_scalar = torch.cat([type_ratios.unsqueeze(-1), type_edge], dim=-1)
        type_tokens = type_pool + self.type_scalar_proj(type_scalar) + self.type_token_embed.unsqueeze(0)
        type_valid = type_counts > 0
        type_tokens = type_tokens.masked_fill(~type_valid.unsqueeze(-1), 0.0)

        base_features = torch.cat([
            global_pool,
            sel_pool,
            nonsel_pool,
            contrast,
            cost_feat,
            type_pool.flatten(1),
            prev_pool,
            next_pool,
            local_delta,
            count_feat,
            edge_feat,
            scalar_feat,
        ], dim=-1)
        base_token = self.candidate_proj(base_features).unsqueeze(1)

        latent_context = torch.cat([sel_pool, scalar_feat], dim=-1)
        latent_delta = self.latent_condition_proj(latent_context).view(
            batch_size, self.NUM_LATENT_TOKENS, self.d_model)
        latent_tokens = self.latent_token_embed.unsqueeze(0) + latent_delta

        tokens = torch.cat([base_token, type_tokens, latent_tokens], dim=1)
        token_padding = torch.cat([
            torch.zeros(batch_size, 1, dtype=torch.bool, device=tokens.device),
            ~type_valid,
            torch.zeros(batch_size, self.NUM_LATENT_TOKENS, dtype=torch.bool, device=tokens.device),
        ], dim=1)

        tokens = self._cross_attention_tokens_shared_kv(
            tokens, route_memory_single, valid_single, shared_route)
        tokens = self.token_mixer(tokens, token_padding)
        tokens = self._cross_attention_tokens_shared_kv(
            tokens,
            route_memory_single,
            valid_single,
            shared_route,
            q_proj=self.cross2_q_proj,
            kv_proj=self.cross2_kv_proj,
            out_proj=self.cross2_out_proj,
        )
        tokens = self.token_mixer2(tokens, token_padding)
        tokens = tokens.masked_fill(token_padding.unsqueeze(-1), 0.0)

        token_valid_weights = (~token_padding).to(tokens.dtype).unsqueeze(-1)
        token_summary = (tokens * token_valid_weights).sum(dim=1) / token_valid_weights.sum(dim=1).clamp(min=1.0)
        first_latent = 1 + self.NUM_ANCHOR_TYPES
        latent_summary = tokens[:, first_latent:].mean(dim=1)
        if shared_state and token_summary.size(0) > 1:
            candidate_context = self.candidate_self_attn(
                token_summary.unsqueeze(0), (~valid_candidate).unsqueeze(0)).squeeze(0)
        else:
            candidate_context = token_summary

        score_features = torch.cat([
            base_features,
            tokens.flatten(1),
            latent_summary,
            candidate_context,
        ], dim=-1)
        return self.score_mlp(score_features).squeeze(-1)


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
        
        if mode in ('v7', 'flashv7'):
            self.flash_encoder = FlashRouteEncoder(
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_encoder_layers,
                n_node_feat=n_node_feat,
                max_length=self.max_length,
            )
            self.flash_head = FlashCandidateHeadV7(d_model, num_heads)
        else:
            self.feature_embed = NodeFeatureEmbedding(d_model, n_feat=n_node_feat)
            self.positional_encoding = self._create_positional_encoding(
                self.max_length, d_model).to(self.device)

        if mode in ('v7', 'flashv7'):
            pass
        elif mode == 'v2':
            self.sel_input_proj = nn.Linear(4, d_model)
            self.encoder = Encoder(d_model, num_heads, num_encoder_layers, use_sel=True)
            self.regression_head = RegressionHeadV2(d_model)
        elif mode == 'new':
            self.selected_bias = nn.Parameter(torch.zeros(2, 1, 1, d_model))
            self.encoder = Encoder(d_model, num_heads, num_encoder_layers, use_sel=True)
            self.regression_head = RegressionHead(d_model, use_selected_pool=True)
        elif mode == 'ratio':
            self.selected_bias = nn.Parameter(torch.zeros(2, 1, 1, d_model))
            self.encoder = Encoder(d_model, num_heads, num_encoder_layers, use_sel=False)
            self.regression_head = RegressionHead(d_model, use_selected_pool=False)
        else:
            raise ValueError(f"unknown CostPredictor mode: {mode!r}")
        
        self.to(self.device)
        self.init_parameters()
        
    def init_parameters(self):
        is_v2 = (self.mode == 'v2')
        layer_norm_params = set()
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                    layer_norm_params.add(id(module.weight))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    layer_norm_params.add(id(module.bias))
        for name, param in self.named_parameters():
            if id(param) in layer_norm_params:
                continue
            if 'ln' in name or 'norm' in name or 'LayerNorm' in name:
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

    @staticmethod
    def _is_batch_broadcast(tensor):
        return tensor is None or tensor.size(0) <= 1 or tensor.stride(0) == 0
    
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

        if self.mode in ('v7', 'flashv7'):
            return self._forward_flash(
                nodes_tensor, demands_tensor, current_sol_tensor,
                selected_tensor, cost_0, tw_features)
        
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
        
        pe = self.positional_encoding[:max_length].to(
            device=device, dtype=node_embeddings.dtype).unsqueeze(0).expand(batch_size, -1, -1)
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

    def _forward_flash(self, nodes_tensor, demands_tensor, current_sol_tensor, selected_tensor,
                       cost_0, tw_features=None):
        if demands_tensor.dim() == 2:
            demands_tensor = demands_tensor.unsqueeze(-1)
        valid_full = current_sol_tensor >= 0
        route_index = current_sol_tensor
        valid_route = valid_full
        mask_index = route_index.clamp(min=0, max=max(selected_tensor.size(1) - 1, 0))
        selected_mask = torch.gather(selected_tensor, 1, mask_index)
        selected_mask = selected_mask.masked_fill(
            (~valid_route) | (route_index < 0) | (route_index >= selected_tensor.size(1)),
            0,
        )

        shared_state = (
            nodes_tensor.size(0) > 1
            and self._is_batch_broadcast(nodes_tensor)
            and self._is_batch_broadcast(demands_tensor)
            and self._is_batch_broadcast(current_sol_tensor)
            and self._is_batch_broadcast(tw_features)
        )
        if shared_state:
            feat = [nodes_tensor[:1], demands_tensor[:1]]
            if tw_features is not None:
                feat.append(tw_features[:1].to(nodes_tensor.device))
            node_features = torch.cat(feat, dim=-1)
            state_embed = self.flash_encoder(
                node_features,
                route_index[:1],
                valid_route[:1],
            )
            route_embed = state_embed
            valid_route_for_head = valid_route[:1]
        else:
            feat = [nodes_tensor, demands_tensor]
            if tw_features is not None:
                feat.append(tw_features.to(nodes_tensor.device))
            node_features = torch.cat(feat, dim=-1)
            route_embed = self.flash_encoder(node_features, route_index, valid_route)
            valid_route_for_head = valid_route

        log_cost_0 = torch.log(cost_0.float().clamp(min=1.0))
        route_index_for_head = route_index[:1] if shared_state else route_index
        node_features_for_head = node_features
        return self.flash_head(
            route_embed, valid_route_for_head, log_cost_0, selected_mask,
            route_index_for_head, node_features_for_head,
            shared_state=shared_state)





##***
def visualize(nodes, current_sol, selected, logits, candidates, percentile=50, threshold=0.5, save_path='visualization.png'):
    import matplotlib.pyplot as plt

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
