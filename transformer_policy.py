"""
Transformer Policy for Node Candidate Selection in VRP
Uses solution-aware encoder and autoregressive decoder for candidate selection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# Feature Embedding
# ============================================================================

class NodeFeatureEmbedding(nn.Module):
    """Embed (x, y, demand) to d_model with normalization"""
    def __init__(self, d_model, problem_scale, capacity_scale):
        super().__init__()
        self.embed = nn.Linear(3, d_model)
        self.problem_scale = problem_scale
        self.capacity_scale = capacity_scale
    
    def forward(self, x):
        """
        Args:
        x: [batch, N, 3] - node features (x, y, demand)
            problem_scale: float or [batch, 1, 1] - scale for coordinates
            capacity_scale: float or [batch, 1, 1] - scale for demand
        
        Returns:
            [batch, N, d_model] - embedded features
        """
        # Clone to avoid modifying input
        x = x.clone()
        
        # Normalize coordinates and demand separately (avoid broadcasting issues)
        x[:, :, 0:2] = x[:, :, 0:2] / self.problem_scale
        x[:, :, 2:3] = x[:, :, 2:3] / self.capacity_scale
        
        return self.embed(x)


# ============================================================================
# Solution-Aware Transformer Encoder
# ============================================================================

class SolutionTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x


class SolutionEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            SolutionTransformerLayer(d_model, num_heads) 
            for _ in range(num_layers)
        ])
    
    def forward(self, x, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return x


# ============================================================================
# Autoregressive Decoder
# ============================================================================

class AutoregressiveDecoder(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        
        # Start token for decoding
        self.start_token = nn.Parameter(torch.randn(d_model))
        
        # Cross-attention: context → candidates
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        # FFN for context processing
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    
    def forward_step(self, context, candidate_embeddings, candidate_mask):
        """
        Args:
            context: [batch, t, d_model]
            candidate_embeddings: [batch, max_nodes, d_model]
            candidate_mask: [batch, max_nodes]
        
        Returns:
            logits: [batch, M] - logits over candidates
        """
        # Cross-attention: context attends to candidates
        # Query: last position of context
        query = context.mean(1, True)  # [batch, 1, d_model]
        
        # Create attention mask for candidates
        # candidate_mask: [batch, M], need to convert for multihead attention
        # False = valid, True = invalid
        key_padding_mask = (candidate_mask == 0)  # [batch, M]
        
        attn_out, _ = self.cross_attn(
            query, 
            candidate_embeddings, 
            candidate_embeddings,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        # Residual + norm
        query = self.ln1(query + attn_out)
        # FFN
        ffn_out = self.ffn(query)
        query = self.ln2(query + ffn_out)
        
        # Project to logits over candidates
        # query: [batch, 1, d_model]
        # We compute compatibility between query and all candidates
        logits = torch.bmm(query, candidate_embeddings.transpose(1, 2))  # [batch, 1, M]
        logits = logits.squeeze(1)  # [batch, M]
        
        # Scale by sqrt(d_model)
        logits = logits / np.sqrt(self.d_model)
        
        # Mask out invalid candidates
        logits = logits.masked_fill(candidate_mask == 0, -1e20)
        return logits
    
    def forward(self, candidate_embeddings, candidate_mask, k_batch, temperature=1.0, return_entropy=False, given_sequence=None):
        """
        Args:
            candidate_embeddings: [batch, max_nodes, d_model]
            candidate_mask: [batch, max_nodes]
            k_batch: [batch] or int - number to select per batch
            temperature: sampling temperature
            return_sequence: return full logits
            given_sequence: [batch, max_k] for teacher forcing
        
        Returns:
            selected_indices: [batch, max_k]
            log_probs: [batch]
            all_logits: [batch, max_k, max_nodes] if return_sequence
        """
        
        batch_size = candidate_embeddings.size(0)
        max_nodes = candidate_embeddings.size(1)
        device = candidate_embeddings.device
        
        max_k = k_batch.max().item()
        context = self.start_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        
        selected_indices = []
        log_probs = []
        entropies = [] if return_entropy else None
        
        dynamic_mask = candidate_mask.clone()
        for t in range(max_k):
            active_mask = t < k_batch
            # Forward step
            logits = self.forward_step(context, candidate_embeddings, dynamic_mask)  # [batch, max_M]
            
            if given_sequence is not None:
                indices = given_sequence[:, t]
                
                logits_scaled = logits / temperature
                probs = F.softmax(logits_scaled, dim=-1)
                dist = torch.distributions.Categorical(probs)
                
                indices_clamped = indices.clamp(0, max_nodes - 1)
                log_prob = dist.log_prob(indices_clamped)
                log_prob = log_prob * active_mask.float()
                
                log_probs.append(log_prob)
            else:
                logits_scaled = logits / temperature
                probs = F.softmax(logits_scaled, dim=-1)
                
                dist = torch.distributions.Categorical(probs)
                indices = dist.sample()
                
                log_prob = dist.log_prob(indices)
                
                indices = torch.where(active_mask, indices, torch.tensor(-1, dtype=torch.long, device=device))
                log_prob = log_prob * active_mask.float()
                
                log_probs.append(log_prob)

            selected_indices.append(indices)
            indices_expanded = indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.d_model)
            indices_expanded_clamped = indices_expanded.clamp(0, max_nodes - 1)
            selected_emb = torch.gather(candidate_embeddings, 1, indices_expanded_clamped)
            context = torch.cat([context, selected_emb], dim=1)
            # compute entropy based on only valid logits
            if return_entropy:
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1) * active_mask.float()
                entropies.append(entropy)

            safe_idx = indices.clamp_min(0).unsqueeze(1)  # [-1] → 0
            dynamic_mask.scatter_(1, safe_idx, 0)
        
        selected_indices = torch.stack(selected_indices, dim=1)
        log_probs_stacked = torch.stack(log_probs, dim=1)
        log_probs_total = log_probs_stacked.mean(dim=1)
        if return_entropy:
            entropies_stacked = torch.stack(entropies, dim=1)
            return selected_indices, log_probs_total, entropies_stacked
        
        return selected_indices, log_probs_total


# ============================================================================
# Heatmap Decoder
# ============================================================================

class HeatmapDecoder(nn.Module):
    def __init__(self, d_model, num_heads=None):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, candidate_embeddings, candidate_mask, k_batch, temperature=1.0, return_entropy=False, given_sequence=None):
        """
        Args:
            candidate_embeddings: [batch, max_nodes, d_model]
            candidate_mask: [batch, max_nodes]
            k_batch: [batch] or int - number to select per batch
            temperature: sampling temperature
            return_sequence: return full logits
            given_sequence: [batch, max_k] for teacher forcing
        
        Returns:
            selected_indices: [batch, max_k]
            log_probs: [batch]
            all_logits: [batch, max_k, max_nodes] if return_sequence
        """
        
        batch_size, max_nodes = candidate_embeddings.shape[:2]
        device = candidate_embeddings.device
        max_k = k_batch.max().item()
        
        logits = self.score_mlp(candidate_embeddings).squeeze(-1) / temperature
        logits = logits.masked_fill(candidate_mask == 0, -1e20)
        probs = F.softmax(logits, dim=-1)
        
        selected_indices = []
        log_probs = []
        entropies = [] if return_entropy else None
        
        for b_idx in range(batch_size):
            k = k_batch[b_idx].item()
            dist = torch.distributions.Categorical(probs[b_idx])
            
            if given_sequence is not None:
                selected = given_sequence[b_idx]
            else:
                selected = torch.multinomial(probs[b_idx], k, replacement=False)
            
            log_prob = dist.log_prob(selected).sum()
            
            selected_indices.append(selected)
            log_probs.append(log_prob)
            if return_entropy:
                entropies.append(-(probs[b_idx] * torch.log(probs[b_idx] + 1e-10)).sum())
        
        selected_indices = torch.stack(selected_indices, dim=0)
        log_probs = torch.stack(log_probs, dim=0)
        
        if return_entropy:
            entropies = torch.stack(entropies, dim=0).unsqueeze(1).expand(-1, max_k)
            return selected_indices, log_probs, entropies
        
        return selected_indices, log_probs


# ============================================================================
# Complete Policy Network
# ============================================================================

class TransformerCandidatePolicy(nn.Module):
    def __init__(self, 
                 d_model=128,
                 num_heads=8,
                 num_encoder_layers=3,
                 max_vehicles=20,
                 N=100,
                 problem_scale=100.0,
                 capacity_scale=100.0,
                 device='cuda'):
        super().__init__()
        
        self.d_model = d_model
        self.max_vehicles = max_vehicles
        self.N = N
        self.max_length = N + max_vehicles * 4
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.feature_embed = NodeFeatureEmbedding(d_model, problem_scale, capacity_scale)
        self.encoder = SolutionEncoder(d_model, num_heads, num_encoder_layers)
        self.decoder = HeatmapDecoder(d_model, num_heads)
        self.positional_encoding = self._create_positional_encoding(self.max_length, d_model).to(self.device)
        
        self.to(self.device)
    
    def _create_positional_encoding(self, max_len, d_model):
        """Standard sinusoidal positional encoding"""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def extract_node_features(self, problem_data_list):
        """Extract node features (supports single and batch)"""
        if not isinstance(problem_data_list, list):
            problem_data_list = [problem_data_list]
        
        coords_list = []
        demand_list = []
        
        for pd in problem_data_list:
            coords_list.append(pd['coordinates'])  # numpy array
            demand_list.append(pd['demand'].to_numpy().reshape(-1, 1))  # cudf.Series → numpy
        
        coords_batch = np.stack(coords_list)
        demand_batch = np.stack(demand_list)
        features_batch = np.concatenate([coords_batch, demand_batch], axis=-1)
        
        return torch.tensor(features_batch, dtype=torch.float32, device=self.device)
    
    def extract_state_features(self, state):
        if not isinstance(state, list):
            state = [state]
        
        sol_list = []
        candidate_mask_list = []
        sample_size_list = []
        
        for s in state:
            sol = s['solution_flat'] + [0] * (self.max_length - len(s['solution_flat']))
            sol_list.append(sol)
            candidate_mask = s['candidate_mask'] + [False] * (self.max_length - len(s['candidate_mask']))
            candidate_mask_list.append(candidate_mask)
            sample_size_list.append(s['sample_size'])
        
        sol_batch = torch.tensor(sol_list, dtype=torch.long, device=self.device)
        candidate_mask_batch = torch.tensor(candidate_mask_list, dtype=torch.bool, device=self.device)
        num_real_nodes = (sol_batch > 0).sum(dim=1)
        sample_size_batch = torch.tensor(sample_size_list, dtype=torch.long, device=self.device)
        
        return sol_batch, candidate_mask_batch, num_real_nodes, sample_size_batch
    
    def forward(self, states, problem_data_list, k=None, temperature=1.0, given_sequence=None):
        if not isinstance(states, list):
            states = [states]
            problem_data_list = [problem_data_list]
        
        B = len(states)
        device = self.device

        node_features = self.extract_node_features(problem_data_list)
        node_embeddings = self.feature_embed(node_features)        
        node_embeddings = torch.cat([
            node_embeddings, 
            node_embeddings[:, :1, :].repeat(1, self.max_length - self.N, 1)
        ], dim=1)

        sol_batch, candidate_mask_batch, num_real_nodes, sample_size_batch = self.extract_state_features(states)
        pe = self.positional_encoding.unsqueeze(0).expand(B, -1, -1)
        indices = sol_batch.unsqueeze(-1).expand(-1, -1, self.d_model)
        node_embeddings.scatter_add_(1, indices, pe)

        padding_mask = torch.arange(self.max_length, device=device).unsqueeze(0).expand(B, -1)
        encoder_mask = padding_mask >= num_real_nodes.unsqueeze(1)
        node_embeddings = self.encoder(node_embeddings, key_padding_mask=encoder_mask)
        
        if given_sequence is None:
            selected_node_ids, log_probs = self.decoder(
                node_embeddings, candidate_mask_batch, sample_size_batch
            )
            return selected_node_ids, log_probs
        else:
            _, log_probs, entropies = self.decoder(
                node_embeddings, candidate_mask_batch, sample_size_batch,
                return_entropy=True, given_sequence=given_sequence
            )
        
        return log_probs, entropies




