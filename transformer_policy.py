"""
Transformer Policy for CuOpt Node Candidate Selection
Architecture:
  1. Encoder: Solution-aware transformer encodes all nodes with route structure
  2. Decoder: Autoregressive decoder selects candidates with interdependence modeling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# Feature Embedding
# ============================================================================

class NodeFeatureEmbedding(nn.Module):
    """Embed (x, y, demand) to d_model"""
    def __init__(self, d_model):
        super().__init__()
        self.embed = nn.Linear(3, d_model)
        self.ln = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        x: [batch, N, 3] - node features (x, y, demand)
        Returns: [batch, N, d_model]
        """
        return self.ln(self.embed(x))


# ============================================================================
# Solution-Aware Transformer Encoder
# ============================================================================

class SolutionTransformerLayer(nn.Module):
    """Transformer layer with optional route-aware attention bias"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x, attn_mask=None):
        """
        x: [batch, N, d_model]
        attn_mask: [batch, N, N] - attention bias (optional)
        Returns: [batch, N, d_model]
        """
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.ln1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        
        return x


class SolutionEncoder(nn.Module):
    """Encode all nodes with solution structure"""
    def __init__(self, d_model, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            SolutionTransformerLayer(d_model, num_heads) 
            for _ in range(num_layers)
        ])
    
    def forward(self, x, attn_bias=None):
        """
        x: [batch, N, d_model]
        attn_bias: [batch, N, N] - route-based attention bias (optional)
        Returns: [batch, N, d_model]
        """
        for layer in self.layers:
            x = layer(x, attn_bias)
        return x


# ============================================================================
# Autoregressive Decoder
# ============================================================================

class AutoregressiveDecoder(nn.Module):
    """Autoregressive decoder for candidate selection with interdependence"""
    def __init__(self, d_model, num_heads, max_candidates=40):
        super().__init__()
        self.d_model = d_model
        self.max_candidates = max_candidates
        
        # Start token for decoding
        self.start_token = nn.Parameter(torch.randn(d_model))
        
        # Cross-attention: context → candidates
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        # FFN for context processing
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Output projection to logits
        self.output_proj = nn.Linear(d_model, 1)
    
    def forward_step(self, context, candidate_embeddings, candidate_mask):
        """
        Single decoding step
        
        Args:
            context: [batch, t, d_model] - previously selected nodes
            candidate_embeddings: [batch, M, d_model]
            candidate_mask: [batch, M] - 1 for valid candidates, 0 for invalid/selected
        
        Returns:
            logits: [batch, M] - logits over candidates
        """
        batch_size = context.size(0)
        
        # Cross-attention: context attends to candidates
        # Query: last position of context
        query = context[:, -1:, :]  # [batch, 1, d_model]
        
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
        logits = logits.masked_fill(candidate_mask == 0, -1e9)
        
        return logits
    
    def forward(self, candidate_embeddings, candidate_mask, k, temperature=1.0, return_sequence=False, given_sequence=None):
        """
        Autoregressive selection of k candidates
        
        Args:
            candidate_embeddings: [batch, max_M, d_model]
            candidate_mask: [batch, max_M] - 1 for valid, 0 for padding
            k: number of candidates to select
            temperature: sampling temperature (higher = more random)
            return_sequence: if True, return full sequence of logits
            given_sequence: [batch, k] - if provided, use teacher forcing (for evaluate)
        
        Returns:
            selected_indices: [batch, k] - selected candidate indices
            log_probs: [batch] - log probability of sequence
            (optional) all_logits: [batch, k, max_M] if return_sequence=True
        """
        batch_size = candidate_embeddings.size(0)
        device = candidate_embeddings.device
        
        # Initialize context with start token
        context = self.start_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)  # [batch, 1, d_model]
        
        # Track selected indices and log probs
        selected_indices = []
        log_probs = []
        all_logits = [] if return_sequence else None
        
        # Create dynamic mask (copy of candidate_mask)
        dynamic_mask = candidate_mask.clone()  # [batch, max_M]
        
        # Teacher forcing mode
        use_teacher_forcing = given_sequence is not None
        
        for t in range(k):
            # Forward step
            logits = self.forward_step(context, candidate_embeddings, dynamic_mask)  # [batch, max_M]
            
            if return_sequence:
                all_logits.append(logits)
            
            # Get index for this step
            if use_teacher_forcing:
                # Teacher forcing: use given sequence
                indices = given_sequence[:, t]
                
                # Still compute log prob for these given indices
                logits_scaled = logits / temperature
                probs = F.softmax(logits_scaled, dim=-1)
                dist = torch.distributions.Categorical(probs)
                
                # Clamp indices to valid range for log_prob (handle padding -1)
                indices_clamped = indices.clamp(0, self.max_candidates - 1)
                log_prob = dist.log_prob(indices_clamped)
                
                # Zero out log_prob for padding indices (where indices == -1)
                padding_mask = (indices >= 0).float()
                log_prob = log_prob * padding_mask
                
                log_probs.append(log_prob)
            else:
                # Sampling mode
                logits_scaled = logits / temperature  # [batch, max_M]
                probs = F.softmax(logits_scaled, dim=-1)  # [batch, max_M]
                
                # Sample one index per batch
                dist = torch.distributions.Categorical(probs)
                indices = dist.sample()  # [batch]
                
                # Compute log prob
                log_prob = dist.log_prob(indices)  # [batch]
                log_probs.append(log_prob)
            
            selected_indices.append(indices)
            
            # Update context: add selected candidate embeddings
            # Clamp indices to valid range before gather (handle padding -1)
            indices_safe = indices.clamp(0, self.max_candidates - 1)
            indices_expanded = indices_safe.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.d_model)  # [batch, 1, d_model]
            selected_emb = torch.gather(candidate_embeddings, 1, indices_expanded)  # [batch, 1, d_model]
            context = torch.cat([context, selected_emb], dim=1)  # [batch, t+2, d_model]
            
            # Update dynamic mask: mask out selected candidates (fully vectorized)
            # Create a mask for positions to zero out
            valid_sample_mask = (indices >= 0).unsqueeze(1).float()  # [batch, 1]
            scatter_positions = torch.zeros_like(dynamic_mask)  # [batch, max_candidates]
            scatter_positions.scatter_(1, indices_safe.unsqueeze(1), 1.0)  # Mark positions to mask
            scatter_positions = scatter_positions * valid_sample_mask  # Only for valid samples
            dynamic_mask = dynamic_mask * (1 - scatter_positions)  # Apply mask (vectorized)
        
        # Stack results
        selected_indices = torch.stack(selected_indices, dim=1)  # [batch, k]
        log_probs_stacked = torch.stack(log_probs, dim=1)  # [batch, k]
        
        # Mean over valid (non-padding) steps
        if use_teacher_forcing:
            # Count valid steps per sample (where indices >= 0)
            valid_steps = (given_sequence >= 0).float().sum(dim=1).clamp(min=1)  # [batch]
            log_probs = log_probs_stacked.sum(dim=1) / valid_steps  # [batch]
        else:
            # No padding in sampling mode
            log_probs = log_probs_stacked.mean(dim=1)  # [batch]
        
        if return_sequence:
            all_logits = torch.stack(all_logits, dim=1)  # [batch, k, max_M]
            return selected_indices, log_probs, all_logits
        
        return selected_indices, log_probs


# ============================================================================
# Complete Policy Network
# ============================================================================

class TransformerCandidatePolicy(nn.Module):
    """Complete transformer policy with autoregressive decoder"""
    def __init__(self, 
                 d_model=128,
                 num_heads=8,
                 num_encoder_layers=3,
                 max_candidates=40,
                 device='cuda'):
        super().__init__()
        
        self.d_model = d_model
        self.max_candidates = max_candidates
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Feature embedding
        self.feature_embed = NodeFeatureEmbedding(d_model)
        
        # Solution-aware encoder
        self.encoder = SolutionEncoder(d_model, num_heads, num_encoder_layers)
        
        # Autoregressive decoder
        self.decoder = AutoregressiveDecoder(d_model, num_heads, max_candidates)
        
        # Learnable dummy depot embeddings (4 per route, max 20 routes = 80 total)
        # We'll dynamically slice based on actual num_routes
        max_dummy_depots = 80  # Assume max 20 routes
        self.dummy_depot_embeddings = nn.Parameter(torch.randn(max_dummy_depots, d_model))
        
        # Move model to device
        self.to(self.device)
    
    def extract_node_features(self, problem_data):
        """
        Extract node features from problem data
        
        Args:
            problem_data: dict with 'coordinates' [N, 2] and 'demand'
        
        Returns:
            features: [N, 3] - (x, y, demand)
        """
        coords = problem_data['coordinates']  # [N, 2] numpy array
        
        # Handle demand (could be cudf.Series or numpy array)
        demand = problem_data['demand']
        if hasattr(demand, 'to_numpy'):
            demand = demand.to_numpy()
        if not isinstance(demand, np.ndarray):
            demand = np.array(demand)
        
        demand = demand.reshape(-1, 1)  # [N, 1]
        
        # Concatenate
        features = np.concatenate([coords, demand], axis=1)  # [N, 3]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def extract_node_features_batch(self, problem_data_list):
        """
        Batch extract node features - vectorized
        
        Args:
            problem_data_list: list of B problem data dicts
        
        Returns:
            features_batch: [B, N, 3]
        """
        coords_list = [pd['coordinates'] for pd in problem_data_list]
        demand_list = []
        for pd in problem_data_list:
            demand = pd['demand']
            if hasattr(demand, 'to_numpy'):
                demand = demand.to_numpy()
            elif not isinstance(demand, np.ndarray):
                demand = np.array(demand)
            demand_list.append(demand.reshape(-1, 1))
        
        # Stack into batch
        coords_batch = np.stack(coords_list)  # [B, N, 2]
        demand_batch = np.stack(demand_list)  # [B, N, 1]
        features_batch = np.concatenate([coords_batch, demand_batch], axis=-1)  # [B, N, 3]
        
        return torch.tensor(features_batch, dtype=torch.float32, device=self.device)
    
    def create_route_attention_bias(self, solution_flat, num_routes, total_nodes):
        """
        Create attention bias based on route structure
        Nodes in the same route get higher attention weights
        
        Args:
            solution_flat: list of node_ids in route order
            num_routes: number of routes
            total_nodes: total number of nodes
        
        Returns:
            bias: [total_nodes, total_nodes] - attention bias matrix
        """
        # Create route assignment for each node
        node_to_route = torch.full((total_nodes,), -1, dtype=torch.long, device=self.device)
        
        # Parse solution_flat to identify routes
        idx = 0
        for route_id in range(num_routes):
            # Skip 4 dummy depot nodes
            idx += 4
            
            # Collect real nodes in this route
            while idx < len(solution_flat):
                node_id = solution_flat[idx]
                # Check if we hit next dummy depot
                if node_id >= total_nodes - num_routes * 4:
                    break
                node_to_route[node_id] = route_id
                idx += 1
        
        # Create bias matrix
        # Same route: 0.0 (no bias), different route: -1.0 (slight penalty)
        bias = torch.zeros(total_nodes, total_nodes, device=self.device)
        for i in range(total_nodes):
            for j in range(total_nodes):
                if node_to_route[i] != -1 and node_to_route[j] != -1:
                    if node_to_route[i] != node_to_route[j]:
                        bias[i, j] = -1.0
        
        return bias
    
    def encode(self, state, problem_data):
        """
        Encode all nodes with solution structure
        
        Args:
            state: dict with solution_flat, candidate_mask, num_routes
            problem_data: dict with coordinates, demand
        
        Returns:
            node_embeddings: [total_nodes, d_model] - includes dummy depots
        """
        # Extract features for real nodes
        node_features = self.extract_node_features(problem_data)  # [N, 3]
        N = node_features.size(0)
        
        # Embed features
        node_embeddings = self.feature_embed(node_features.unsqueeze(0))  # [1, N, d_model]
        
        # Create route attention bias
        total_nodes = len(state['candidate_mask'])
        # For now, skip the bias (can be expensive to compute)
        # attn_bias = self.create_route_attention_bias(
        #     state['solution_flat'], state['num_routes'], total_nodes
        # )
        
        # Encode with transformer
        node_embeddings = self.encoder(node_embeddings, attn_bias=None)  # [1, N, d_model]
        node_embeddings = node_embeddings.squeeze(0)  # [N, d_model]
        
        # Add dummy depot embeddings (learnable, deterministic)
        num_routes = state['num_routes']
        num_dummy_depots = num_routes * 4
        
        if total_nodes > N:
            # Use learnable dummy depot embeddings (deterministic)
            dummy_embeddings = self.dummy_depot_embeddings[:num_dummy_depots]  # [num_dummy_depots, d_model]
            
            # Concatenate real nodes + dummy depots
            node_embeddings = torch.cat([node_embeddings, dummy_embeddings], dim=0)  # [total_nodes, d_model]
        
        return node_embeddings  # [total_nodes, d_model]
    
    def encode_batch(self, states, problem_data_list):
        """
        Batch encode all states - vectorized
        
        Args:
            states: list of B state dicts
            problem_data_list: list of B problem data dicts
        
        Returns:
            node_embeddings_batch: [B, total_nodes, d_model]
        """
        B = len(states)
        
        # Vectorized feature extraction
        node_features_batch = self.extract_node_features_batch(problem_data_list)  # [B, N, 3]
        
        # Batch embed and encode
        node_embeddings_batch = self.feature_embed(node_features_batch)  # [B, N, d_model]
        node_embeddings_batch = self.encoder(node_embeddings_batch, attn_bias=None)  # [B, N, d_model]
        
        # Vectorized dummy depot creation (learnable, deterministic)
        num_routes = states[0]['num_routes']
        num_dummy_depots = num_routes * 4
        
        # Use learnable dummy depot embeddings (expand to batch)
        dummy_embeddings = self.dummy_depot_embeddings[:num_dummy_depots].unsqueeze(0)  # [1, num_dummy, d_model]
        dummy_embeddings = dummy_embeddings.expand(B, -1, -1)  # [B, num_dummy, d_model]
        
        node_embeddings_batch = torch.cat([node_embeddings_batch, dummy_embeddings], dim=1)  # [B, total_nodes, d_model]
        
        return node_embeddings_batch
    
    def prepare_candidates(self, node_embeddings, candidate_mask):
        """
        Extract and pad candidate embeddings
        
        Args:
            node_embeddings: [total_nodes, d_model] - includes dummy depots
            candidate_mask: list or tensor [total_nodes] - 1 for candidates, 0 otherwise
        
        Returns:
            candidate_embeddings: [max_candidates, d_model] - padded
            valid_mask: [max_candidates] - 1 for valid, 0 for padding
            candidate_node_ids: list of node_ids that are candidates
        """
        # Extract candidate node_ids
        if isinstance(candidate_mask, torch.Tensor):
            candidate_mask = candidate_mask.tolist()
        
        candidate_node_ids = [i for i, v in enumerate(candidate_mask) if v == 1]
        M = len(candidate_node_ids)
        
        # Extract embeddings for candidates
        candidate_embeddings = node_embeddings[candidate_node_ids]  # [M, d_model]
        
        # Pad to max_candidates
        device = node_embeddings.device
        if M < self.max_candidates:
            padding = torch.zeros(self.max_candidates - M, self.d_model, device=device)
            candidate_embeddings = torch.cat([candidate_embeddings, padding], dim=0)
            valid_mask = torch.cat([torch.ones(M, device=device), torch.zeros(self.max_candidates - M, device=device)])
        else:
            # Truncate if M > max_candidates
            candidate_embeddings = candidate_embeddings[:self.max_candidates]
            candidate_node_ids = candidate_node_ids[:self.max_candidates]
            valid_mask = torch.ones(self.max_candidates, device=device)
        
        return candidate_embeddings, valid_mask, candidate_node_ids
    
    def prepare_candidates_batch(self, node_embeddings_batch, candidate_masks):
        """
        Batch prepare candidates - uses torch.nonzero per sample
        
        Args:
            node_embeddings_batch: [B, total_nodes, d_model]
            candidate_masks: list of B masks [total_nodes]
        
        Returns:
            candidate_embeddings_batch: [B, 40, d_model]
            valid_masks_batch: [B, 40]
            candidate_node_ids_list: list of B lists
        """
        B = node_embeddings_batch.size(0)
        device = self.device
        
        # Convert masks to tensor
        candidate_masks_tensor = torch.tensor(candidate_masks, dtype=torch.bool, device=device)  # [B, total_nodes]
        
        batch_cand_emb_list = []
        batch_valid_list = []
        candidate_node_ids_list = []
        
        for i in range(B):
            mask_i = candidate_masks_tensor[i]  # [total_nodes]
            cand_ids = torch.nonzero(mask_i, as_tuple=False).squeeze(-1)  # [M]
            M = len(cand_ids)
            candidate_node_ids_list.append(cand_ids.tolist())
            
            # Gather candidate embeddings
            cand_emb = node_embeddings_batch[i, cand_ids, :]  # [M, d_model]
            
            # Pad/truncate to 40
            if M < 40:
                padding = torch.zeros(40 - M, self.d_model, device=device)
                cand_emb = torch.cat([cand_emb, padding], dim=0)
                valid = torch.cat([torch.ones(M, device=device), torch.zeros(40 - M, device=device)])
            else:
                cand_emb = cand_emb[:40]
                candidate_node_ids_list[-1] = candidate_node_ids_list[-1][:40]
                valid = torch.ones(40, device=device)
            
            batch_cand_emb_list.append(cand_emb)
            batch_valid_list.append(valid)
        
        return torch.stack(batch_cand_emb_list), torch.stack(batch_valid_list), candidate_node_ids_list
    
    def extract_and_pad_selected_indices(self, selection_masks, candidate_node_ids_list):
        """
        Extract and pad selected indices - uses torch.nonzero per sample
        
        Args:
            selection_masks: list of B masks [total_nodes]
            candidate_node_ids_list: list of B lists of candidate node_ids
        
        Returns:
            selected_indices_tensor: [B, max_k] (padded with -1)
            batch_k_values: [B] tensor of actual k values
        """
        B = len(selection_masks)
        device = self.device
        
        # Convert to tensors
        selection_masks_tensor = torch.tensor(selection_masks, dtype=torch.bool, device=device)  # [B, total_nodes]
        
        batch_selected_indices = []
        batch_k_values = []
        
        for i in range(B):
            # Extract selected node_ids (vectorized per sample)
            selected_node_ids = torch.nonzero(selection_masks_tensor[i], as_tuple=False).squeeze(-1)  # [k]
            
            # Map to candidate indices
            cand_ids_list = candidate_node_ids_list[i]
            node_id_to_idx = {nid: idx for idx, nid in enumerate(cand_ids_list)}
            sel_idx = [node_id_to_idx[nid.item()] for nid in selected_node_ids if nid.item() in node_id_to_idx]
            
            batch_selected_indices.append(sel_idx)
            batch_k_values.append(len(sel_idx))
        
        # Vectorized padding
        max_k = max(batch_k_values) if batch_k_values else 0
        if max_k == 0:
            return torch.empty(B, 0, dtype=torch.long, device=device), torch.zeros(B, dtype=torch.long, device=device)
        
        # Pad all to max_k
        selected_indices_padded = []
        for sel_idx in batch_selected_indices:
            padded = sel_idx + [-1] * (max_k - len(sel_idx))
            selected_indices_padded.append(padded)
        
        selected_indices_tensor = torch.tensor(selected_indices_padded, dtype=torch.long, device=device)  # [B, max_k]
        batch_k_values_tensor = torch.tensor(batch_k_values, dtype=torch.long, device=device)  # [B]
        
        return selected_indices_tensor, batch_k_values_tensor
    
    def sample(self, state, problem_data, sample_size=40, temperature=1.0):
        """
        Sample action and compute log probability
        
        Args:
            state: dict with solution_flat, candidate_mask, num_routes
            problem_data: dict with coordinates, demand
            sample_size: number of nodes to select
            temperature: sampling temperature (higher = more random, lower = more peaked)
        
        Returns:
            selection_mask: list [total_nodes] - 1 for selected, 0 otherwise
            logp: scalar - log probability
        """
        # Respect training/eval state (use torch.no_grad() during eval)
        # Always sample, never greedy
        with torch.no_grad() if not self.training else torch.enable_grad():
            # Encode all nodes
            node_embeddings = self.encode(state, problem_data)  # [N, d_model]
            
            # Prepare candidates
            candidate_embeddings, valid_mask, candidate_node_ids = self.prepare_candidates(
                node_embeddings, state['candidate_mask']
            )
            
            # Add batch dimension
            candidate_embeddings = candidate_embeddings.unsqueeze(0)  # [1, max_candidates, d_model]
            valid_mask = valid_mask.unsqueeze(0)  # [1, max_candidates]
            
            # Determine k
            M = len(candidate_node_ids)
            k = min(M, sample_size)
            
            if k == 0:
                # No candidates
                selection_mask = [0] * len(state['candidate_mask'])
                return selection_mask, torch.tensor(0.0)
            
            # Autoregressive decoding with temperature
            selected_indices, log_prob = self.decoder(
                candidate_embeddings, valid_mask, k, temperature=temperature, return_sequence=False
            )
            
            # Convert to selection_mask
            selected_indices = selected_indices.squeeze(0).tolist()  # [k]
            selected_node_ids = [candidate_node_ids[idx] for idx in selected_indices]
            
            selection_mask = np.zeros(len(state['candidate_mask']), dtype=np.int32)
            selection_mask[selected_node_ids] = 1
            selection_mask = selection_mask.tolist()
        
        return selection_mask, log_prob.item()
    
    def evaluate(self, states, problem_data_list, selection_masks):
        """
        Fully vectorized evaluate - all operations batched except autoregressive masking
        
        Args:
            states: list of state dicts
            problem_data_list: list of problem data dicts
            selection_masks: list of selection_masks [total_nodes]
        
        Returns:
            logps: [B] tensor - log probabilities
            entropies: [B] tensor - entropies
        """
        B = len(states)
        device = self.device
        
        # 1. Batch encode all states (fully vectorized)
        node_embeddings_batch = self.encode_batch(states, problem_data_list)  # [B, total_nodes, d_model]
        
        # 2. Batch prepare candidates (fully vectorized)
        candidate_embeddings_batch, valid_masks_batch, candidate_node_ids_list = \
            self.prepare_candidates_batch(node_embeddings_batch, [s['candidate_mask'] for s in states])
        # [B, 40, d_model], [B, 40], list[B]
        
        # 3. Extract and pad selected indices (fully vectorized)
        selected_indices_tensor, batch_k_values = \
            self.extract_and_pad_selected_indices(selection_masks, candidate_node_ids_list)
        # [B, max_k], [B]
        
        max_k = selected_indices_tensor.size(1)
        if max_k == 0:
            return torch.zeros(B, device=device), torch.zeros(B, device=device)
        
        # 4. Batch forward through decoder with TEACHER FORCING
        # Use given_sequence to ensure same context as during sampling
        # Forward to get all logits (for both logp and entropy)
        _, log_probs, all_logits = self.decoder(
            candidate_embeddings_batch,
            valid_masks_batch,
            max_k,
            temperature=1.0,
            return_sequence=True,
            given_sequence=selected_indices_tensor  # Same teacher forcing
        )  # [B, max_k, 40]
        
        # Create step mask for entropy computation
        step_mask = torch.arange(max_k, device=device).unsqueeze(0) < batch_k_values.unsqueeze(1)  # [B, max_k]
        
        # Apply masking to logits (same as decoder does internally)
        sample_valid_masks = valid_masks_batch.unsqueeze(1).expand(-1, max_k, -1).clone()  # [B, max_k, 40]
        
        # Autoregressive masking
        selected_one_hot = F.one_hot(selected_indices_tensor.clamp(0, 39), num_classes=40).float()
        cumsum_selected = torch.cumsum(selected_one_hot, dim=1)
        cumsum_selected_shifted = torch.cat([
            torch.zeros(B, 1, 40, device=device),
            cumsum_selected[:, :-1, :]
        ], dim=1)
        autoregressive_mask = (cumsum_selected_shifted == 0).float()
        sample_valid_masks = sample_valid_masks * autoregressive_mask
        
        # Compute probs with proper masking
        all_logits_masked = all_logits.masked_fill(sample_valid_masks == 0, -1e9)
        probs = F.softmax(all_logits_masked, dim=-1)  # [B, max_k, 40]
        
        # Compute entropy (vectorized)
        entropy_all = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, max_k]
        entropy_all = entropy_all * step_mask.float()
        entropies = entropy_all.sum(dim=1) / batch_k_values.float().clamp(min=1)  # [B]
        
        return log_probs, entropies

