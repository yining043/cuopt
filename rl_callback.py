"""RL policy callback for cuOpt node-selection.

Reuses the existing `pred_with_NN` path in local_search.cu:
  1. cuOpt discovers anchors (1 full search) and generates K random subsets,
     each run once (no look-ahead), producing K executed-anchor bitmasks.
  2. `customize_nodes_to_search` scores the K masks with CostPredictor, treats
     the scores as logits over a K-arm action, samples one arm (the action),
     and returns the chosen mask as the node selection.
  3. After cuOpt executes the chosen mask in the *real* search, C++ calls
     `on_search_result` with the cost delta -> the reward for that action.

The look-ahead that made the oracle slow is gone; the RL return over the real
trajectory replaces it.
"""

import random

import torch
import torch.nn.functional as F
from cuopt.routing import CustomizeNodesCallback


class RandomSubsetCallback(CustomizeNodesCallback):
    """Pick one of K probe subsets uniformly at random (random baseline)."""

    def customize_nodes_to_search(self, solution_flat, num_routes,
                                  solution_cost, trail_masks_flat, num_trails,
                                  trail_rewards, iter):
        K = num_trails
        max_length = len(trail_masks_flat) // K
        valid = []
        for t in range(K):
            base = t * max_length
            if any(trail_masks_flat[base + i] > 0 for i in range(max_length)):
                valid.append(t)
        if not valid:
            return [0] * max_length
        idx = random.choice(valid)
        base = idx * max_length
        return [trail_masks_flat[base + i] for i in range(max_length)]


class RLPolicyCallback(CustomizeNodesCallback):
    def __init__(self, model, coordinates, demand, vehicle_capacity, device,
                 temperature=1.0, score_sign=-1.0, train=True, epsilon=0.0,
                 tw_features=None):
        super().__init__()
        self.model = model
        self.device = device
        # coordinates: [1, N, 2]; demand: [1, N]; vehicle_capacity: scalar
        self.coordinates = coordinates.to(device)
        self.demand = (demand / vehicle_capacity).to(device)
        # tw_features: [1, N, k] normalized time-window features (CVRPTW) or None
        self.tw_features = tw_features.to(device) if tw_features is not None else None
        self.temperature = temperature
        # CostPredictor predicts a cost ratio (lower = better), so default
        # score_sign=-1 turns it into a sensible logit (higher = better arm).
        self.score_sign = score_sign
        self.train = train
        self.epsilon = epsilon
        self.transitions = []   # finalized (s, a, r) steps for this episode
        self._pending = None    # action awaiting its reward
        self.eps = 1e-6

    def _build_arm_inputs(self, solution_flat, solution_cost, trail_masks):
        K, max_length = trail_masks.shape
        sol_tensor = torch.tensor(
            solution_flat + [-1] * (max_length - len(solution_flat)),
            dtype=torch.long, device=self.device,
        ).unsqueeze(0).expand(K, -1)
        nodes = self.coordinates.expand(K, -1, -1)
        demands = self.demand.expand(K, -1)
        cost_0 = torch.full((K,), float(solution_cost),
                            dtype=torch.float32, device=self.device)
        tw = self.tw_features.expand(K, -1, -1) if self.tw_features is not None else None
        return sol_tensor, nodes, demands, cost_0, tw

    def customize_nodes_to_search(self, solution_flat, num_routes,
                                  solution_cost, trail_masks_flat, num_trails,
                                  trail_rewards, iter):
        K = num_trails
        max_length = len(trail_masks_flat) // K
        trail_masks = torch.tensor(
            trail_masks_flat, dtype=torch.long, device=self.device
        ).reshape(K, max_length)
        rewards = torch.tensor(trail_rewards, dtype=torch.float32, device=self.device)  # [K]

        non_empty = (trail_masks > 0).any(dim=1)
        if not non_empty.any():
            self._pending = None
            return [0] * max_length

        sol_tensor, nodes, demands, cost_0, tw = self._build_arm_inputs(
            solution_flat, solution_cost, trail_masks)

        with torch.no_grad():
            scores = self.model(nodes, demands, sol_tensor, trail_masks, cost_0, tw_features=tw)  # [K]
            logits = self.score_sign * scores / self.temperature
            logits = logits.masked_fill(~non_empty, float('-inf'))
            probs = F.softmax(logits, dim=0)
            if self.train:
                idx = int(torch.multinomial(probs, 1).item())
            elif self.epsilon > 0 and random.random() < self.epsilon:
                valid_idx = torch.where(non_empty)[0]
                idx = int(valid_idx[torch.randint(len(valid_idx), (1,))].item())
            else:
                idx = int(torch.argmax(probs).item())

        if self.train:
            # Full-feedback bandit: store all-K probe rewards as the training
            # target. Detached CPU tensors; log-probs recomputed (with grad) at
            # update time.
            self._pending = {
                'sol': sol_tensor[0].detach().to('cpu'),
                'masks': trail_masks.detach().to('cpu').to(torch.int16),
                'valid': non_empty.detach().to('cpu'),
                'rewards': rewards.detach().to('cpu'),         # [K] probe deltas
                'cost_0': float(solution_cost),
                'idx': idx,
                'iter': int(iter),
            }
        return trail_masks[idx].int().cpu().tolist()

    def on_search_result(self, cost_before, cost_after, move_found, iteration):
        if self._pending is None:
            return
        reward = max(0.0, (cost_before - cost_after) / max(abs(cost_before), self.eps))
        step = dict(self._pending)
        step['reward'] = reward
        step['cost_before'] = float(cost_before)
        step['cost_after'] = float(cost_after)
        self.transitions.append(step)
        self._pending = None
