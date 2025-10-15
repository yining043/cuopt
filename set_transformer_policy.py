"""
Set Transformer Policy for CuOpt Node Candidate Selection
Two-stage architecture:
  1. Graph Encoder: encodes all nodes using path structure
  2. Set Transformer: segments candidate set with per-element classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
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
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.0):
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
    def __init__(self, node_feature_dim, d_model, num_heads=4, num_layers=2, dropout=0.0):
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
    def __init__(self, d_model, num_heads=4, num_sab_layers=2, dropout=0.0):
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
            nn.Linear(d_model, 1)  # Single logit for binary decision
        )
    
    def forward(self, candidate_embeddings):
        """
        candidate_embeddings: [batch, M, d_model] or [M, d_model]
        Returns: [batch, M, 1] or [M, 1] - single logit for each candidate
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
                 dropout=0.0):
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
        node_embeddings = self.graph_encoder(node_features, edge_index)  # [N, d_model]
        
        # Expand embeddings to include virtual depot nodes
        n_real_nodes = node_features.size(0)
        n_routes = len(state['routes_2d'])
        n_virtual_depots = n_routes * 4  # CuOpt creates 4 virtual depots per route
        
        # Create virtual depot embeddings with route-specific information
        depot_embedding = node_embeddings[0]  # [d_model]
        virtual_embeddings = []
        
        for route_id in range(n_routes):
            # Create 4 virtual depots for this route
            for i in range(4):
                # Add route-specific positional encoding
                route_encoding = torch.zeros(self.d_model, device=depot_embedding.device)
                # Use sinusoidal encoding for route_id
                position = route_id * 4 + i
                div_term = torch.exp(torch.arange(0, self.d_model, 2, device=depot_embedding.device) * 
                                    -(np.log(10000.0) / self.d_model))
                route_encoding[0::2] = torch.sin(position * div_term)
                if self.d_model > 1:
                    route_encoding[1::2] = torch.cos(position * div_term[:len(route_encoding[1::2])])
                
                # Virtual depot = depot embedding + route encoding (small scale)
                virtual_emb = depot_embedding + 0.1 * route_encoding
                virtual_embeddings.append(virtual_emb)
        
        # Concatenate: [real nodes, virtual depots]
        if virtual_embeddings:
            virtual_embeddings = torch.stack(virtual_embeddings)  # [n_virtual, d_model]
            node_embeddings_extended = torch.cat([node_embeddings, virtual_embeddings], dim=0)
        else:
            node_embeddings_extended = node_embeddings
        
        # Stage 2: Extract candidate embeddings (directly index, no mapping needed)
        candidate_ids = state['candidate_node_ids']
        assert len(candidate_ids) != 0, "No candidates"
        
        candidate_ids_tensor = torch.tensor(candidate_ids, dtype=torch.long)
        # Clamp to valid range
        max_id = node_embeddings_extended.size(0) - 1
        candidate_ids_clamped = torch.clamp(candidate_ids_tensor, 0, max_id)
        candidate_embeddings = node_embeddings_extended[candidate_ids_clamped]
        
        # Stage 2: Set Transformer segmentation
        logits = self.set_segmentation(candidate_embeddings)
        
        return logits
    
    def sample(self, state, problem_data, sample_size=40, deterministic=False):
        """
        Sample action and compute log probability
        Select top min(M, sample_size) candidates based on probabilities
        
        Args:
            state: current state dict
            problem_data: problem data dict
            sample_size: number of nodes to select (default 40)
            deterministic: whether to use deterministic selection
        
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
        
        # Squeeze to get [M] shape and convert to probabilities
        logits = logits.squeeze(-1)  # [M, 1] -> [M]
        probs = torch.sigmoid(logits)
        
        # Determine k = min(M, sample_size)
        M = probs.size(0)
        k = min(M, sample_size)
        
        # Select top-k candidates based on probabilities
        top_k_probs, top_k_indices = torch.topk(probs, k=k)
        selected_indices = top_k_indices.tolist()
        
        # Compute log probability for the selection
        # Create action mask: selected nodes = 1, others = 0
        actions = torch.zeros(M, dtype=torch.long, device=probs.device)
        actions[top_k_indices] = 1
        
        dist = Bernoulli(probs=probs)
        logp = dist.log_prob(actions.float()).mean()
        
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
        
        # Squeeze to get [M] shape and convert to probabilities
        logits = logits.squeeze(-1)  # [M, 1] -> [M]
        probs = torch.sigmoid(logits)
        
        # Create Bernoulli distribution
        dist = Bernoulli(probs=probs)
        
        # Use mean instead of sum to normalize by number of candidates
        # This ensures logp and entropy are comparable across different problem sizes
        logp = dist.log_prob(actions.float()).mean()
        entropy = dist.entropy().mean()
        
        return logp, entropy
    
    def precompute_cache(self, state, problem_data):
        """
        Precompute data that doesn't depend on network parameters
        This caches data processing to avoid repeated computation
        
        Args:
            state: state dict
            problem_data: problem data dict
        
        Returns:
            cache: dict with preprocessed data
        """
        # Extract node features (numpy operations)
        node_features = self._extract_node_features(problem_data)
        
        # Extract edges from routes
        edge_index = self._extract_edges(state['routes_2d'], node_features.size(0))
        
        # Get candidate IDs
        candidate_ids = state['candidate_node_ids']
        candidate_ids_tensor = torch.tensor(candidate_ids, dtype=torch.long)
        
        # Metadata
        n_routes = len(state['routes_2d'])
        n_real_nodes = node_features.size(0)
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'candidate_ids': candidate_ids_tensor,
            'n_routes': n_routes,
            'n_real_nodes': n_real_nodes
        }
    
    def forward_gnn_only(self, cache):
        """
        Forward only GNN part using cached data
        Returns candidate embeddings that can be batched
        
        Args:
            cache: dict from precompute_cache()
        
        Returns:
            candidate_embeddings: [M, d_model] tensor
        """
        # GNN encoding (uses current parameters, allows gradients)
        node_embeddings = self.graph_encoder(cache['node_features'], cache['edge_index'])
        
        # Add virtual depot embeddings
        depot_embedding = node_embeddings[0]
        virtual_embeddings = []
        
        for route_id in range(cache['n_routes']):
            for i in range(4):
                route_encoding = torch.zeros(self.d_model, device=depot_embedding.device)
                position = route_id * 4 + i
                div_term = torch.exp(torch.arange(0, self.d_model, 2, device=depot_embedding.device) * 
                                    -(np.log(10000.0) / self.d_model))
                route_encoding[0::2] = torch.sin(position * div_term)
                if self.d_model > 1:
                    route_encoding[1::2] = torch.cos(position * div_term[:len(route_encoding[1::2])])
                virtual_emb = depot_embedding + 0.1 * route_encoding
                virtual_embeddings.append(virtual_emb)
        
        # Concatenate real and virtual nodes
        if virtual_embeddings:
            virtual_embeddings = torch.stack(virtual_embeddings)
            node_embeddings_extended = torch.cat([node_embeddings, virtual_embeddings], dim=0)
        else:
            node_embeddings_extended = node_embeddings
        
        # Extract candidate embeddings using cached IDs
        max_id = node_embeddings_extended.size(0) - 1
        candidate_ids_clamped = torch.clamp(cache['candidate_ids'], 0, max_id)
        candidate_embeddings = node_embeddings_extended[candidate_ids_clamped]
        
        return candidate_embeddings
    
    def evaluate_batch(self, caches, actions_list):
        """
        Batch evaluate using cached data
        GNN part is done per-sample (different graph structures)
        Set Transformer is batched (major speedup)
        
        Args:
            caches: list of cache dicts from precompute_cache()
            actions_list: list of action tensors
        
        Returns:
            logps: [batch_size] tensor of log probabilities
            entropies: [batch_size] tensor of entropies
        """
        batch_size = len(caches)
        
        # Step 1: GNN forward for each sample (loop - different graph structures)
        all_candidate_embeddings = []
        all_num_candidates = []
        
        for cache in caches:
            candidate_emb = self.forward_gnn_only(cache)  # [M_i, d_model]
            all_candidate_embeddings.append(candidate_emb)
            all_num_candidates.append(candidate_emb.size(0))
        
        # Step 2: Batch Set Transformer (major speedup!)
        max_M = max(all_num_candidates)
        batch_candidates = torch.zeros(batch_size, max_M, self.d_model, 
                                      device=all_candidate_embeddings[0].device)
        batch_masks = torch.zeros(batch_size, max_M, dtype=torch.bool,
                                  device=all_candidate_embeddings[0].device)
        
        for i, (emb, M) in enumerate(zip(all_candidate_embeddings, all_num_candidates)):
            batch_candidates[i, :M] = emb
            batch_masks[i, :M] = True
        
        # Forward Set Transformer once for entire batch
        batch_logits = self.set_segmentation(batch_candidates)  # [B, max_M, 1]
        batch_logits = batch_logits.squeeze(-1)  # [B, max_M]
        
        # Step 3: Compute logp and entropy for each sample
        batch_logps = []
        batch_entropies = []
        
        for i, (M, actions) in enumerate(zip(all_num_candidates, actions_list)):
            # Extract valid logits (not padding)
            logits = batch_logits[i, :M]
            probs = torch.sigmoid(logits)
            
            # Compute log probability and entropy
            dist = Bernoulli(probs=probs)
            logp = dist.log_prob(actions.float()).mean()
            entropy = dist.entropy().mean()
            
            batch_logps.append(logp)
            batch_entropies.append(entropy)
        
        # Stack into tensors
        logps = torch.stack(batch_logps)
        entropies = torch.stack(batch_entropies)
        
        return logps, entropies
    
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

