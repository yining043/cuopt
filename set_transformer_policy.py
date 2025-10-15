"""
Set Transformer Policy for CuOpt Node Candidate Selection
Two-stage architecture:
  1. Graph Encoder: encodes all nodes using path structure
  2. Set Transformer: segments candidate set with per-element classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


# ============================================================================
# Set Transformer Base Modules
# ============================================================================

class MAB(nn.Module):
    """Multihead Attention Block"""
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, X, Y):
        """
        X: [batch, n, d_model] - query
        Y: [batch, m, d_model] - key/value
        """
        H, _ = self.attn(X, Y, Y, need_weights=False)
        H = self.ln1(X + H)
        out = self.ln2(H + self.fc(H))
        return out


class SAB(nn.Module):
    """Set Attention Block (self-attention within a set)"""
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.mab = MAB(d_model, num_heads, dropout)
    
    def forward(self, X):
        """
        X: [batch, n, d_model]
        """
        return self.mab(X, X)


class PMA(nn.Module):
    """Pooling by Multihead Attention"""
    def __init__(self, d_model, num_heads, num_seeds, dropout=0.0):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, d_model))
        self.mab = MAB(d_model, num_heads, dropout)
    
    def forward(self, X):
        """
        X: [batch, n, d_model]
        Output: [batch, num_seeds, d_model]
        """
        batch_size = X.size(0)
        S = self.S.expand(batch_size, -1, -1)
        return self.mab(S, X)


# ============================================================================
# Graph Encoder (Stage 1)
# ============================================================================

class GraphAttentionLayer(nn.Module):
    """Single Graph Attention Layer"""
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.head_dim = out_features // num_heads
        
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.randn(1, num_heads, 2 * self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x, edge_index):
        """
        x: [N, in_features]
        edge_index: [2, E] - source and target node indices
        Returns: [N, out_features]
        """
        N = x.size(0)
        
        # Linear transformation
        h = self.W(x)  # [N, out_features]
        h = h.view(N, self.num_heads, self.head_dim)  # [N, num_heads, head_dim]
        
        # Edge attention
        if edge_index.size(1) == 0:
            # No edges, return transformed features (no aggregation)
            return h.view(N, -1)  # [N, out_features]
        
        src, dst = edge_index[0], edge_index[1]
        h_src = h[src]  # [E, num_heads, head_dim]
        h_dst = h[dst]  # [E, num_heads, head_dim]
        
        # Attention coefficients
        h_concat = torch.cat([h_src, h_dst], dim=-1)  # [E, num_heads, 2*head_dim]
        e = self.leaky_relu((h_concat * self.a).sum(dim=-1))  # [E, num_heads]
        
        # Softmax over incoming edges for each node
        alpha = torch.zeros(N, self.num_heads, dtype=e.dtype, device=e.device)
        alpha = alpha.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.num_heads), e)
        alpha = alpha[dst]  # [E, num_heads]
        alpha = torch.exp(e - alpha)  # Numerically stable softmax
        
        # Aggregate
        out = torch.zeros(N, self.num_heads, self.head_dim, dtype=h.dtype, device=h.device)
        weighted_h = alpha.unsqueeze(-1) * h_src  # [E, num_heads, head_dim]
        out = out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, self.head_dim), weighted_h)
        
        out = self.dropout(out)
        return out.view(N, -1)  # [N, out_features]


class GraphEncoder(nn.Module):
    """Graph Attention Network for encoding all nodes"""
    def __init__(self, node_feature_dim, d_model, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(node_feature_dim, d_model)
        
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(d_model, d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
    def forward(self, node_features, edge_index):
        """
        node_features: [N, node_feature_dim]
        edge_index: [2, E]
        Returns: [N, d_model]
        """
        x = self.input_proj(node_features)
        
        for gat, ln in zip(self.gat_layers, self.layer_norms):
            x = ln(x + gat(x, edge_index))
        
        return x


# ============================================================================
# Set Transformer Segmentation (Stage 2)
# ============================================================================

class SetTransformerSegmentation(nn.Module):
    """Set Transformer for candidate segmentation"""
    def __init__(self, d_model, num_heads=4, num_sab_layers=2, dropout=0.1):
        super().__init__()
        
        # SAB layers for candidate interaction
        self.sab_layers = nn.ModuleList([
            SAB(d_model, num_heads, dropout)
            for _ in range(num_sab_layers)
        ])
        
        # Optional: PMA for global context
        self.use_global = True
        if self.use_global:
            self.pma = PMA(d_model, num_heads, num_seeds=4, dropout=dropout)
        
        # Segmentation head (per-element binary classification)
        head_input_dim = d_model + (4 * d_model if self.use_global else 0)
        self.seg_head = nn.Sequential(
            nn.Linear(head_input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)  # [not_select, select]
        )
    
    def forward(self, candidate_embeddings):
        """
        candidate_embeddings: [batch, M, d_model] or [M, d_model]
        Returns: [batch, M, 2] or [M, 2] - logits for each candidate
        """
        # Handle single batch
        if candidate_embeddings.dim() == 2:
            candidate_embeddings = candidate_embeddings.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        x = candidate_embeddings
        
        # SAB layers: candidates interact
        for sab in self.sab_layers:
            x = sab(x)
        
        # Optional global context
        if self.use_global:
            z = self.pma(x)  # [batch, num_seeds, d_model]
            z_flat = z.flatten(start_dim=1)  # [batch, num_seeds * d_model]
            z_expanded = z_flat.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, z_expanded], dim=-1)
        
        # Per-element segmentation
        logits = self.seg_head(x)  # [batch, M, 2]
        
        if squeeze_output:
            logits = logits.squeeze(0)
        
        return logits


# ============================================================================
# Complete Policy Network
# ============================================================================

class NodeCandidateSelectionPolicy(nn.Module):
    """Complete two-stage policy for node candidate selection"""
    def __init__(self, 
                 node_feature_dim=3,
                 d_model=128,
                 num_heads=4,
                 num_graph_layers=2,
                 num_sab_layers=2,
                 dropout=0.1):
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.d_model = d_model
        
        # Stage 1: Graph Encoder
        self.graph_encoder = GraphEncoder(
            node_feature_dim, d_model, num_heads, num_graph_layers, dropout
        )
        
        # Stage 2: Set Transformer Segmentation
        self.set_segmentation = SetTransformerSegmentation(
            d_model, num_heads, num_sab_layers, dropout
        )
    
    def forward(self, state, problem_data):
        """
        Forward pass: returns logits for each candidate
        
        Args:
            state: dict with keys:
                - 'routes_2d': list of routes
                - 'candidate_node_ids': list of candidate node indices
            problem_data: dict with keys:
                - 'coordinates': [N, 2] numpy array
                - 'demand': cudf.Series or numpy array
        
        Returns:
            logits: [M, 2] where M = len(candidate_node_ids)
        """
        # Extract features and edges
        node_features = self._extract_node_features(problem_data)
        edge_index = self._extract_edges(state['routes_2d'], node_features.size(0))
        
        # Stage 1: Encode all nodes
        node_embeddings = self.graph_encoder(node_features, edge_index)
        
        # Stage 2: Extract candidate embeddings
        candidate_ids = state['candidate_node_ids']
        assert len(candidate_ids) != 0, "No candidates"
        
        # extra stage: processing candidate ids
        candidate_ids = torch.tensor(candidate_ids)
        candidate_ids_mapped = torch.where(
            candidate_ids >= node_features.size(0),
            torch.zeros_like(candidate_ids),  # 映射到 depot (节点0)
            candidate_ids  # 保持原值
        )
        candidate_embeddings = node_embeddings[candidate_ids_mapped]
        
        # Stage 2: Set Transformer segmentation
        logits = self.set_segmentation(candidate_embeddings)
        
        return logits
    
    def sample(self, state, problem_data, deterministic=False):
        """
        Sample action and compute log probability
        
        Returns:
            selected_indices: list of selected candidate indices (in range [0, M-1])
            logp: log probability (scalar tensor)
        """
        # Set eval mode for deterministic sampling
        was_training = self.training
        if deterministic:
            self.eval()
        
        with torch.no_grad() if deterministic else torch.enable_grad():
            logits = self.forward(state, problem_data)
        
        if logits.size(0) == 0:
            if was_training and deterministic:
                self.train()
            return [], torch.tensor(0.0)
        
        # Create categorical distribution
        dist = Categorical(logits=logits)
        
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        
        # Compute log probability
        logp = dist.log_prob(actions).sum()
        
        # Extract selected indices (where action == 1)
        selected_indices = (actions == 1).nonzero(as_tuple=True)[0].tolist()
        
        # Restore training mode
        if was_training and deterministic:
            self.train()
        
        return selected_indices, logp
    
    def evaluate(self, state, problem_data, actions):
        """
        Re-evaluate log probability and entropy for given actions
        Used in PPO training
        
        Args:
            state: state dict
            problem_data: problem data dict
            actions: [M] tensor of actions (0 or 1 for each candidate)
        
        Returns:
            logp: log probability (scalar)
            entropy: entropy (scalar)
        """
        logits = self.forward(state, problem_data)
        
        if logits.size(0) == 0:
            return torch.tensor(0.0), torch.tensor(0.0)
        
        dist = Categorical(logits=logits)
        logp = dist.log_prob(actions).sum()
        entropy = dist.entropy().sum()
        
        return logp, entropy
    
    def _extract_node_features(self, problem_data):
        """
        Extract normalized node features
        Returns: [N, node_feature_dim] tensor
        """
        coords = problem_data['coordinates']
        
        # Handle cudf Series
        if hasattr(problem_data['demand'], 'to_numpy'):
            demand = problem_data['demand'].to_numpy()
        else:
            demand = np.array(problem_data['demand'])
        
        # Normalize
        coords_norm = coords / 100.0
        demand_norm = demand / 100
        
        # Combine features
        features = np.concatenate([
            coords_norm,
            demand_norm.reshape(-1, 1)
        ], axis=1)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_edges(self, routes_2d, num_nodes):
        """
        Extract edges from routes
        Returns: [2, E] tensor
        """
        edges = []
        
        for route in routes_2d:
            for i in range(len(route) - 1):
                src, dst = route[i], route[i + 1]
                # Add bidirectional edges
                edges.append([src, dst])
                edges.append([dst, src])
        
        if len(edges) == 0:
            # No edges, return empty tensor
            return torch.zeros(2, 0, dtype=torch.long)
        
        edge_index = torch.tensor(edges, dtype=torch.long).T
        return edge_index


# ============================================================================
# Utility Functions
# ============================================================================

def extract_state_from_collector(state_dict):
    """Convert collector state to policy input format"""
    return {
        'routes_2d': state_dict['routes_2d'],
        'candidate_node_ids': state_dict['candidate_node_ids']
    }


def extract_problem_from_collector(problem_dict):
    """Convert collector problem to policy input format"""
    return {
        'coordinates': problem_dict['coordinates'],
        'demand': problem_dict['demand']
    }

