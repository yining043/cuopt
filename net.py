import torch
import torch.nn as nn
from typing import Any, Optional, Tuple
import numpy as np

# --- Attention reshape util ---


def reshape_by_heads(qkv: torch.Tensor, head_num: int) -> torch.Tensor:
    """Reshape (B, N, H*D) -> (B, H, N, D)."""
    B, N, _ = qkv.size()
    q_reshaped = qkv.view(B, N, head_num, -1)
    return q_reshaped.permute(0, 2, 1, 3)


# --- Residual + LayerNorm / FFN ---


class AddAndNormalizationModule(nn.Module):
    """Simple residual + LayerNorm."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Optional[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        if x is None:
            return self.norm(y)
        return self.norm(x + y)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.W1 = nn.Linear(embedding_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, embedding_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.act(self.W1(x)))


class EncoderLayer(nn.Module):
    """Multi-head self-attention + FFN block."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params["embedding_dim"]
        head_num = model_params["head_num"]
        self.head_num = head_num
        qkv_dim = model_params["qkv_dim"]

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.combine = nn.Linear(head_num * qkv_dim, embedding_dim, bias=False)

        self.add_norm1 = AddAndNormalizationModule(embedding_dim)
        self.ff = FeedForward(embedding_dim, model_params["hidden_dim"])
        self.add_norm2 = AddAndNormalizationModule(embedding_dim)

    def multi_head_attention(self, x: torch.Tensor, route_attn: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.size()
        q = reshape_by_heads(self.Wq(x), self.head_num)  # (B,H,N,D)
        k = reshape_by_heads(self.Wk(x), self.head_num)
        v = reshape_by_heads(self.Wv(x), self.head_num)

        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)  # (B,H,N,N)
        if route_attn is not None:
            # route_attn is a positional bias of shape (B,H,N,N)
            scores = scores + route_attn
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)  # (B,H,N,D)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, -1)  # (B,N,H*D)
        return self.combine(out)

    def forward(self, x: torch.Tensor, route_attn: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.multi_head_attention(x, route_attn=route_attn)
        y = self.add_norm1(x, attn_out)
        y2 = self.ff(y)
        return self.add_norm2(y, y2)


# --- Encoder (depot + node embed, stacked layers) ---

class Encoder(nn.Module):
    """
    - Embeds depot (x,y) and node (x,y,demand + supplement_feature).
    - Applies several EncoderLayer blocks.
    """

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.problem = model_params["problem"]
        embedding_dim = model_params["embedding_dim"]
        encoder_layer_num = model_params["encoder_layer_num"]
        supplement_feature_dim = model_params.get("supplement_feature_dim", 0)
        depot_in = model_params.get("depot_feature_dim", 2 + supplement_feature_dim)
        node_in = model_params.get("node_feature_dim", 3 + supplement_feature_dim)
        self.embedding_depot = nn.Linear(depot_in, embedding_dim)
        self.embedding_node = nn.Linear(node_in, embedding_dim)

        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy: torch.Tensor, node_xy_demand_tw: torch.Tensor, route_attn: Optional[torch.Tensor] = None) -> torch.Tensor:
        if depot_xy is not None:
            embedded_depot = self.embedding_depot(depot_xy)          # (B,1,E)
        embedded_node = self.embedding_node(node_xy_demand_tw)       # (B,N,E)

        if depot_xy is not None:
            out = torch.cat((embedded_depot, embedded_node), dim=1)  # (B,1+N,E)
        else:
            out = embedded_node

        for layer in self.layers:
            out = layer(out, route_attn=route_attn)

        return out


class MultiHeadPosCompat(nn.Module):
    """Maps h_pos (B, N, E) to route attention bias (B, H, N, N) via Q,K projection."""

    def __init__(self, embedding_dim: int, head_num: int, qkv_dim: int):
        super().__init__()
        self.head_num = head_num
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

    def forward(self, h_pos: torch.Tensor) -> torch.Tensor:
        q = reshape_by_heads(self.Wq(h_pos), self.head_num)
        k = reshape_by_heads(self.Wk(h_pos), self.head_num)
        route_attn = torch.matmul(q, k.transpose(-2, -1))
        return route_attn


class SolutionEmbedder(nn.Module):
    """
    Single entry: forward(context, env) -> get_dynamic_feature -> position encoding -> encoder -> mean pool.
    env is passed at forward time, not in __init__.
    """

    def __init__(self, model_params: dict):
        super().__init__()
        self.model_params = model_params
        self.encoder = Encoder(**model_params)
        self.embedding_dim = model_params["embedding_dim"]
        self.head_num = model_params["head_num"]
        qkv_dim = model_params["qkv_dim"]
        self.pos_encoder = MultiHeadPosCompat(self.embedding_dim, self.head_num, qkv_dim)

    @staticmethod
    def basesin(x: np.ndarray, Td: int, fai: float) -> np.ndarray:
        """Base sine wave with period Td and phase fai."""
        return np.sin((2 * np.pi / float(Td)) * x + fai)

    @staticmethod
    def basecos(x: np.ndarray, Td: int, fai: float) -> np.ndarray:
        """Base cosine wave with period Td and phase fai."""
        return np.cos((2 * np.pi / float(Td)) * x + fai)

    def cyclic_position_encoding_pattern(self, n_position, emb_dim, mean_pooling=True):
        Td_set = np.linspace(np.power(n_position, 1 / (emb_dim // 2)), n_position, emb_dim // 2, dtype='int')
        x = np.zeros((n_position, emb_dim))

        for i in range(emb_dim):
            Td = Td_set[i // 3 * 3 + 1] if (i // 3 * 3 + 1) < (emb_dim // 2) else Td_set[-1]
            fai = 0 if i <= (emb_dim // 2) else 2 * np.pi * ((-i + (emb_dim // 2)) / (emb_dim // 2))
            longer_pattern = np.arange(0, np.ceil((n_position) / Td) * Td, 0.01)
            if i % 2 == 1:
                x[:, i] = self.basecos(longer_pattern, Td, fai)[
                    np.linspace(0, len(longer_pattern), n_position, dtype='int', endpoint=False)]
            else:
                x[:, i] = self.basesin(longer_pattern, Td, fai)[
                    np.linspace(0, len(longer_pattern), n_position, dtype='int', endpoint=False)]
        pattern = torch.from_numpy(x).type(torch.FloatTensor)
        pattern_sum = torch.zeros_like(pattern).cpu()

        # averaging the adjacient embeddings if needed (optional, almost the same performance)
        arange = torch.arange(n_position).cpu()
        pooling = [0] if not mean_pooling else [-2, -1, 0, 1, 2]
        time = 0
        for i in pooling:
            time += 1
            index = (arange + i + n_position) % n_position
            pattern_sum += pattern.gather(0, index.view(-1, 1).expand_as(pattern))
        pattern = 1. / time * pattern_sum - pattern.mean(0)

        return pattern

    def _position_encoding(self, base: torch.Tensor, embedding_dim: int, order_vector: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = order_vector.size()
        position_enc = base.expand(batch_size, *base.size()).clone().to(order_vector.device)
        index = order_vector.unsqueeze(-1).expand(batch_size, seq_length, embedding_dim)
        return torch.gather(position_enc, 1, index)

    def forward(self, context: Tuple[torch.Tensor, ...], env: Any) -> torch.Tensor:
        visited_time, depot_feature, node_feature = env.get_dynamic_feature(context)
        _, solution_size = visited_time.size()

        # positional features encoding (cyclic position encoding)
        pattern = self.cyclic_position_encoding_pattern(solution_size, self.embedding_dim).to(visited_time.device)
        h_pos = self._position_encoding(pattern, self.embedding_dim, visited_time)
        aux_scores = self.pos_encoder(h_pos) # (B, N+dummy, E)

        # encoder
        h_em_final = self.encoder(depot_feature, node_feature, route_attn=aux_scores)

        # mean pooling
        return h_em_final.mean(dim=1) # (B, E)