"""
Transformer Policy Model for Node Candidate Selection in VRP
Contains model definitions: embedding, encoder, decoder, and complete policy network
"""
from json import encoder
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
        self.embed = nn.Linear(3, d_model, bias=False)
    
    def forward(self, x):
        return self.embed(x)

# ============================================================================
# Transformer Encoder
# ============================================================================

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2, bias=False),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model, bias=False)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads) 
            for _ in range(num_layers)
        ])
    
    def forward(self, x, key_padding_mask=None):
        for idx, layer in enumerate(self.layers):
            x = layer(x, key_padding_mask)
        return x


# ============================================================================
# Decoder
# ============================================================================

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.ReLU(),
            nn.Linear(d_model, d_model//2, bias=False),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )

    def forward(self, embed):
        logits = self.mlp(embed)
        return logits

# ============================================================================
# Complete Policy Network
# ============================================================================

class Policy(nn.Module):
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
        self.max_vehicles = max_vehicles
        self.max_length = N + max_vehicles * 4
        self.device = torch.device(device)
        
        self.feature_embed = NodeFeatureEmbedding(d_model)
        self.positional_encoding = self._create_positional_encoding(self.max_length, d_model).to(self.device)

        self.encoder_global = Encoder(d_model, num_heads, num_encoder_layers)
        self.encoder_local = Encoder(d_model, num_heads, num_encoder_layers)
        self.decoder = Decoder(d_model, num_heads)
        self.start_token = nn.Parameter(torch.randn(1, 1, d_model))

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
        max_length = current_sol_tensor.size(1)
        
        nodes_tensor = nodes_tensor.to(device)
        demands_tensor = demands_tensor.to(device)
        current_sol_tensor = current_sol_tensor.to(device)
        candidates_tensor = candidates_tensor.to(device)
        selected_tensor = selected_tensor.to(device)
        
        # feature embedding
        embed = self.feature_embed(torch.cat([nodes_tensor, demands_tensor], dim=-1))
        # make the dummy depot nodes (every 4 ndoes for each vehicle) 
        # note that the candidates and selected tensor only care 1-N customer nodes and 4*m dummy depot nodes !!!
        embed = torch.cat([embed, embed[:, :1, :].repeat(1, max_length - N, 1)], dim=1)
        
        # positional encoding
        pe = self.positional_encoding[:max_length].unsqueeze(0).expand(batch_size, -1, -1)
        sol_indices = current_sol_tensor.clamp(0, embed.size(1) - 1).unsqueeze(-1).expand(-1, -1, self.d_model)
        embed.scatter_add_(1, sol_indices.long(), pe)

        # encoder round 1
        encoder_mask = current_sol_tensor < 0
        encoder_mask[:, 0] = True # depot should be masked out !!!
        embed = self.encoder_global(embed, key_padding_mask=encoder_mask)
        embed_expanded = embed.unsqueeze(1).expand(-1, 10, -1, -1)
        # encoder round 2: selected nodes
        max_selected_number = selected_tensor[:,0,:].sum(-1).max().item()
        pos_indices = torch.arange(selected_tensor.size(-1), device=device).unsqueeze(0).unsqueeze(0).expand_as(selected_tensor)
        pos_indices = (pos_indices * selected_tensor).long()
        pos_indices = pos_indices.sort(dim=-1, descending=True)[0]
        pos_indices = pos_indices[:, :, :max_selected_number]
        selected_embeds = embed_expanded.gather(dim=2, index=pos_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_model))
        # process selected embeds
        selected_embeds = selected_embeds.view(-1, max_selected_number, self.d_model)
        selected_embeds = torch.cat([self.start_token.expand(batch_size*10, -1, -1), selected_embeds], dim=1)
        attention_mask = (pos_indices == 0).view(-1, max_selected_number)
        attention_mask = torch.cat([torch.zeros(batch_size*10, 1, device=device).bool(), attention_mask], dim=1)
        # perform encode
        selected_embeds = self.encoder_local(selected_embeds, key_padding_mask=attention_mask)
        selected_embeds = selected_embeds.view(batch_size, 10, -1, self.d_model)[:,:,0]

        # decoder
        logits = self.decoder(selected_embeds).squeeze(-1)
        return logits

def main():
    model = Policy()
    data = torch.load('ml_data_large_temp.pt')
    nodes_tensor = data['nodes_tensor'][:128]
    demands_tensor = data['demands_tensor'][:128]
    current_sol_tensor = data['current_sol_tensor'][:128]
    candidates_tensor = data['candidates_tensor'][:128]
    selected_tensor = data['selected_tensor'][:128]
    logits = model(nodes_tensor, demands_tensor, current_sol_tensor, candidates_tensor, selected_tensor)
    torch.set_printoptions(precision=10)
    print(logits[:10])

if __name__ == "__main__":
    main()