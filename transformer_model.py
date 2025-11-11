"""
Transformer Policy Model for Node Candidate Selection in VRP
Contains model definitions: embedding, encoder, decoder, and complete policy network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ============================================================================
# Feature Embedding
# ============================================================================

class NodeFeatureEmbedding(nn.Module):
    """Embed (x, y, demand) to d_model with normalization"""
    def __init__(self, d_model):
        super().__init__()
        self.embed = nn.Linear(3, d_model)
    
    def forward(self, x):
        return self.embed(x)


# ============================================================================
# Solution-Aware Transformer Encoder
# ============================================================================

class SolutionTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
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
            for _ in range(2 * num_layers)
        ])
    
    def forward(self, x, key_padding_mask=None, candidates_mask=None):
        for idx, layer in enumerate(self.layers):
            if idx % 2 == 0:
                x = layer(x, key_padding_mask)
            else:   
                x = layer(x, key_padding_mask | ~candidates_mask)
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
        
    def forward_step(self, query, h, candidate_mask):
        logits = torch.bmm(query, h.transpose(1, 2)).squeeze(1)
        logits = logits / math.sqrt(self.d_model)
        logits = logits.masked_fill(~candidate_mask, -1e20)
        return logits
    
    def forward(self, h, candidate_mask, selected_flat=None):
        batch_size, max_nodes, _ = h.size()
        k_batch = selected_flat.sum(dim=1).long()
        max_k = k_batch.max().item()
        device = h.device
        
        # which order should be used???
        if selected_flat is not None:
            positions = torch.arange(max_nodes, device=device).unsqueeze(0).expand(batch_size, -1)
            masked_positions = torch.where(selected_flat, positions, torch.tensor(max_nodes, device=device))
            _, sorted_indices = torch.sort(masked_positions, dim=1)
            given_sequence = sorted_indices[:, :max_k]
            given_sequence = torch.where(given_sequence >= max_nodes, torch.tensor(-1, device=device), given_sequence)
        else:
            given_sequence = None
        
        context = self.start_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        
        selected_indices = []
        log_probs = []
        dynamic_mask = candidate_mask.clone()
        
        for t in range(max_k):
            active_mask = t < k_batch
            logits = self.forward_step(context, h, dynamic_mask)

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            if given_sequence is not None:
                indices = given_sequence[:, t].clamp(0, max_nodes - 1)
            else:
                indices = dist.sample()
                
            log_prob = dist.log_prob(indices) * active_mask.float()
            log_probs.append(log_prob)
            selected_indices.append(indices)
            
            indices_expanded = indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.d_model)
            selected_emb = torch.gather(h, 1, indices_expanded)
            context = context + selected_emb #torch.cat([context, selected_emb], dim=1)
            dynamic_mask.scatter_(1, indices.clamp_min(0).unsqueeze(1), 0)
        
        log_probs_total = torch.stack(log_probs, dim=1).sum(dim=1) / k_batch.float()
        return torch.stack(selected_indices, dim=1), log_probs_total

# ============================================================================
# Heat Map Decoder
# ============================================================================

class HeatMapDecoder(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        
        # Query and Key for heat map
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
    
    def forward(self, h, candidate_mask, selected_flat=None):
        global_context = h.mean(dim=1, keepdim=True)
        q = self.q_linear(global_context)
        k = self.k_linear(h)
        logits = torch.bmm(q, k.transpose(1, 2)).squeeze(1)
        logits = logits / math.sqrt(self.d_model) # 缩放
        logits = logits.masked_fill(~candidate_mask, -1e20)

        log_probs = F.log_softmax(logits, dim=-1)
        if selected_flat is None:
            raise NotImplementedError("Selected flat is not supported for heat map decoder")
        else:
            actions = selected_flat

        selected_mask = actions.float() # (B, 10, 1085)
        set_log_probs = (log_probs * actions).sum(dim=-1) / (selected_mask.sum(dim=-1) + 1e-8)
        return actions, set_log_probs

# ============================================================================
# Complete Policy Network
# ============================================================================

class TransformerCandidatePolicy(nn.Module):
    def __init__(self, 
                 d_model=128,
                 num_heads=8,
                 num_encoder_layers=3,
                 max_vehicles=21,
                 N=1001,
                 use_autoregressive_decoder=True,
                 device='cuda'):
        super().__init__()
        
        self.d_model = d_model
        self.N = N
        self.max_length = N + max_vehicles * 4
        self.device = torch.device(device)
        
        self.feature_embed = NodeFeatureEmbedding(d_model)
        self.encoder = SolutionEncoder(d_model, num_heads, num_encoder_layers)
        if use_autoregressive_decoder:
            self.decoder = AutoregressiveDecoder(d_model, num_heads)
        else:
            self.decoder = HeatMapDecoder(d_model, num_heads)
        self.positional_encoding = self._create_positional_encoding(self.max_length, d_model).to(self.device)
        
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
    
    def forward(self, nodes_tensor, demands_tensor, current_sol_tensor, candidates_tensor, selected_tensor):
        device = self.device
        batch_size, N, _ = nodes_tensor.shape
        
        nodes_tensor = nodes_tensor.to(device)
        demands_tensor = demands_tensor.to(device)
        current_sol_tensor = current_sol_tensor.to(device)
        candidates_tensor = candidates_tensor.to(device)
        selected_tensor = selected_tensor.to(device)
        
        # feature embedding
        h = self.feature_embed(torch.cat([nodes_tensor, demands_tensor], dim=-1))
        h = torch.cat([h, h[:, :1, :].repeat(1, self.max_length - N, 1)], dim=1)
        
        # positional encoding
        pe = self.positional_encoding[:self.max_length].unsqueeze(0).expand(batch_size, -1, -1)
        sol_indices = current_sol_tensor.clamp(0, h.size(1) - 1).unsqueeze(-1).expand(-1, -1, self.d_model)
        h.scatter_add_(1, sol_indices.long(), pe)
        
        # encoder
        num_real_nodes = (current_sol_tensor >= 0).sum(dim=1).unsqueeze(1)
        encoder_mask = torch.arange(self.max_length, device=device).unsqueeze(0).expand(batch_size, -1) >= num_real_nodes
        candidates_mask = candidates_tensor[:, 0, :].bool()  # Use first trail's candidates mask [needs to change data collection for simplification!!]
        h = self.encoder(h, key_padding_mask=encoder_mask, candidates_mask=candidates_mask)

        # decoder
        candidates_flat = candidates_tensor.view(batch_size * 10, self.max_length)
        selected_flat = selected_tensor.view(batch_size * 10, self.max_length)
        h_expanded = h.unsqueeze(1).expand(-1, 10, -1, -1).contiguous().view(batch_size * 10, self.max_length, self.d_model)
        
        _, log_probs = self.decoder(h_expanded, candidates_flat, selected_flat)
        log_probs = log_probs.view(batch_size, 10)
        
        return log_probs

