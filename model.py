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
    def __init__(self, d_model, num_heads, input_dim):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model*2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_model*2, d_model//4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_model//4, 1)
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
                 num_heads=4,
                 num_encoder_layers=3,
                 max_vehicles=21,
                 N=1001,
                 use_autoregressive_decoder=True,
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
        self.selected_bias = nn.Parameter(torch.zeros(2, 1, 1, d_model))
        self.candidate_bias = nn.Parameter(torch.zeros(2, 1, 1, d_model))
        self.decoder = Decoder(d_model, num_heads, input_dim=2*d_model)
        # 专门处理那 40 个点之间的关系
        self.intra_set_attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.norm_intra = nn.LayerNorm(d_model)

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
        trail_size = selected_tensor.size(1)
        
        nodes_tensor = nodes_tensor.to(device)
        demands_tensor = demands_tensor.to(device)
        current_sol_tensor = current_sol_tensor.to(device)
        candidates_tensor = candidates_tensor.to(device)
        selected_tensor = selected_tensor.to(device)
        
        node_embeddings = self.feature_embed(torch.cat([nodes_tensor, demands_tensor], dim=-1))
        node_embeddings = torch.cat([node_embeddings, node_embeddings[:, :1, :].repeat(1, max_length - N, 1)], dim=1)
        
        # positional encoding
        pe = self.positional_encoding[:max_length].unsqueeze(0).expand(batch_size, -1, -1)
        sol_indices = current_sol_tensor.clamp(0, node_embeddings.size(1) - 1).unsqueeze(-1).expand(-1, -1, self.d_model)
        pos_embedding_gathered = torch.zeros_like(node_embeddings).to(device)
        pos_embedding_gathered.scatter_add_(1, sol_indices.long(), pe)
        
        # bias embedding by the candidate tensor
        candidate_bias = self.candidate_bias[0].expand_as(node_embeddings)
        unselected_bias = self.candidate_bias[1].expand_as(node_embeddings)
        bias_candidate = candidates_tensor.view(batch_size, -1, 1).expand_as(node_embeddings).float()
        node_embeddings += bias_candidate * candidate_bias + (1 - bias_candidate) * unselected_bias

        # encoder round 1
        encoder_mask = current_sol_tensor < 0
        encoder_mask[:, 0] = True # depot should be masked out !!!
        node_embeddings = self.encoder(node_embeddings, encoder_mask, pos_embedding_gathered)
        
        valid_mask = ~encoder_mask
        valid_mask_expanded = valid_mask.unsqueeze(-1).float()
        global_context = (node_embeddings * valid_mask_expanded).sum(dim=1) / valid_mask_expanded.sum(dim=1)
        
        max_selected_number = selected_tensor[:,0,:].sum(-1).max().item()
        
        pos_indices = torch.arange(selected_tensor.size(-1), device=device).unsqueeze(0).unsqueeze(0).expand_as(selected_tensor)
        pos_indices = (pos_indices * selected_tensor).long()
        pos_indices = pos_indices.sort(dim=-1, descending=True)[0]
        pos_indices = pos_indices[:, :, :max_selected_number]
        
        embed_expanded = node_embeddings.unsqueeze(1).expand(-1, trail_size, -1, -1)
        pos_indices_expanded = pos_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_model)
        selected_embeds = embed_expanded.gather(dim=2, index=pos_indices_expanded)
        selected_embeds = selected_embeds.view(-1, max_selected_number, self.d_model)

        global_context = global_context.unsqueeze(1).expand(-1, trail_size, -1)

        attention_mask = (pos_indices == 0).view(-1, max_selected_number)
        x_input = selected_embeds.view(batch_size * trail_size, max_selected_number, self.d_model)
        selected_embeds, _ = self.intra_set_attention(
            query=x_input, 
            key=x_input, 
            value=x_input, 
            key_padding_mask=attention_mask
        )

        valid_selected_mask = ~attention_mask  # [batch*trail_size, max_selected_number]
        valid_selected_mask_expanded = valid_selected_mask.unsqueeze(-1).float()
        candidate_vector = (selected_embeds * valid_selected_mask_expanded).sum(dim=1) / valid_selected_mask_expanded.sum(dim=1)  # [batch*trail_size, d_model]
        candidate_vector = candidate_vector.view(batch_size, trail_size, self.d_model)

        context_concat = torch.cat([candidate_vector, global_context], dim=-1)
        logits = self.decoder(context_concat).squeeze(-1)
        return logits

def main():
    model = Policy()
    data = torch.load('ml_data_large.pt')
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