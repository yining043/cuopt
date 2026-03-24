"""CVRP env: load instance+basin_info; prepare_from_hashes(hashes)->context; get_dynamic_feature(context)->(visited_time, depot_f, node_f)."""
from typing import Dict, List, Optional, Tuple

import torch
from helper import get_solution_with_dummy_depot, sol2rec

_SCALE = int(1e3)


def _preprocessing_core(rec: torch.Tensor, demand: torch.Tensor, dummy_size: int) -> Tuple[torch.Tensor, ...]:
    """Pure function for the recurrence loop; can be torch.compile'd to reduce Python overhead."""
    device = rec.device
    batch_size, seq_length = rec.size()
    arange = torch.arange(batch_size, device=device, dtype=torch.long)
    pre = torch.zeros(batch_size, device=device, dtype=torch.long)
    route = torch.zeros(batch_size, device=device, dtype=torch.long)
    route_plan_visited_time = torch.zeros((batch_size, seq_length), device=device, dtype=torch.long)
    cum_demand = torch.zeros((batch_size, seq_length), device=device, dtype=demand.dtype)
    partial_sum_wrt_route_plan = torch.zeros((batch_size, dummy_size), device=device, dtype=demand.dtype)

    for i in range(seq_length):
        next_ = rec[arange, pre]
        next_is_dummy = next_ < dummy_size
        route = route + next_is_dummy.long()
        val_visited = ((route % dummy_size) * _SCALE + (i + 1) % seq_length).unsqueeze(1)
        route_plan_visited_time.scatter_(1, next_.unsqueeze(1), val_visited)
        route_idx = (route % dummy_size).unsqueeze(1)
        old_partial = partial_sum_wrt_route_plan.gather(1, route_idx).squeeze(1)
        demand_at_next = demand[arange, next_]
        new_cum_demand = old_partial + demand_at_next
        partial_sum_wrt_route_plan.scatter_(1, route_idx, new_cum_demand.unsqueeze(1))
        cum_val = new_cum_demand * (~next_is_dummy).to(demand.dtype)
        cum_demand.scatter_(1, next_.unsqueeze(1), cum_val.unsqueeze(1))
        pre = next_

    route_plan_0x = route_plan_visited_time // _SCALE
    return route_plan_0x, route_plan_visited_time % _SCALE, cum_demand, partial_sum_wrt_route_plan


try:
    _preprocessing_core_compiled = torch.compile(_preprocessing_core, dynamic=True, fullgraph=False)
except Exception:
    _preprocessing_core_compiled = None

class CVRPEnv:
    """Holds dummy xy/demand; load() stores instance+basin_info; prepare_from_hashes(hashes)->context; get_dynamic_feature(context)->(vt, depot_f, node_f)."""

    def __init__(self, problem_size: int, device: torch.device):

        self.problem_size = problem_size
        self.num_nodes = 1 + problem_size
        self.device = device
        self._dummy_xy: Optional[torch.Tensor] = None
        self._dummy_demand: Optional[torch.Tensor] = None
        self._dummy_size: int = 1
        self._depot_xy: Optional[torch.Tensor] = None
        self._node_xy_demand: Optional[torch.Tensor] = None
        self._basin_info: Optional[Dict[str, dict]] = None

    def load(self, depot_xy: torch.Tensor, node_xy_demand: torch.Tensor, basin_info: Dict[str, dict]) -> None:
        """Store instance and basin_info."""
        self._depot_xy = depot_xy.to(self.device)
        self._node_xy_demand = node_xy_demand.to(self.device)
        self._basin_info = basin_info

    def prepare_from_hashes(self, hashes: List[str]) -> Tuple[torch.Tensor, ...]:
        """Build context from hashes. Uses precomputed rec['solution'] (depot=0)."""
        B = len(hashes)

        solutions_list = [self._basin_info[h]["solution"] for h in hashes]
        solutions_tensors = [torch.tensor(s, dtype=torch.long, device=self.device) for s in solutions_list]
        solutions = torch.nn.utils.rnn.pad_sequence(solutions_tensors, batch_first=True, padding_value=0)

        # Number of dummy depots = total length - problem_size (at least 1)
        dummy_size = max(1, solutions.size(1) - self.problem_size)
        self._dummy_xy = torch.cat([self._depot_xy.expand(B, dummy_size, 2), self._node_xy_demand.expand(B, -1, -1)[:, :, :2]], dim=1).to(self.device)
        self._dummy_demand = torch.cat([torch.zeros(B, dummy_size, device=self.device), self._node_xy_demand.expand(B, -1, -1)[:, :, 2]], dim=1).to(self.device)
        self._dummy_size = dummy_size

        solutions = get_solution_with_dummy_depot(solutions, self.problem_size)
        rec = sol2rec(solutions)
        context = self.preprocessing(rec)

        return context
    
    def preprocessing(self, rec: torch.Tensor) -> torch.Tensor:
        """Uses compiled core when available for lower Python overhead."""
        assert rec.size(1) < 1000
        demand = self._dummy_demand
        dummy_size = self._dummy_size
        fn = _preprocessing_core_compiled if _preprocessing_core_compiled is not None else _preprocessing_core
        return fn(rec, demand, dummy_size)

    def get_dynamic_feature(self, context):

        route_plan_0x, visited_time, cum_demand, partial_sum_wrt_route_plan = context
        demand = self._dummy_demand.unsqueeze(-1)
        cum_demand = cum_demand.unsqueeze(-1)
        route_total_demand_per_node = partial_sum_wrt_route_plan.gather(-1, route_plan_0x).unsqueeze(-1)

        infeasibility_indicator_after_visit = torch.clamp_min(cum_demand - 1.00001, 0.0) > 0
        infeasibility_indicator_before_visit = torch.clamp_min((cum_demand - demand) - 1.00001, 0.0) > 0

        to_actor = torch.cat((
            cum_demand, # demand,
            route_total_demand_per_node - cum_demand,
            (demand == 0).float(),
            infeasibility_indicator_before_visit,
            infeasibility_indicator_after_visit,
        ), -1)  # the node features

        feature = torch.cat([self._dummy_xy, self._dummy_demand.unsqueeze(-1)], dim=-1)
        feature = torch.cat((feature, to_actor), dim=-1)
        depot_feature = torch.cat((feature[:, :self._dummy_size, :3], feature[:, :self._dummy_size, 4:]), dim=2)  # rm demand dimension
        node_feature = feature[:, self._dummy_size:]

        return visited_time, depot_feature, node_feature