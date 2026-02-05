"""CVRP env: load instance+basin_info; prepare_from_hashes(hashes)->context; get_dynamic_feature(context)->(visited_time, depot_f, node_f)."""
from typing import Dict, List, Optional, Tuple

import torch
from helper import get_solution_with_dummy_depot, sol2rec

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
        demand = self._dummy_demand
        device = rec.device

        batch_size, seq_length = rec.size()
        assert seq_length < 1000
        arange = torch.arange(batch_size, device=device)

        pre = torch.zeros(batch_size, device=device).long()
        route = torch.zeros(batch_size, device=device).long()
        route_plan_visited_time = torch.zeros((batch_size, seq_length), device=device).long()
        cum_demand = torch.zeros((batch_size, seq_length), device=device)
        partial_sum_wrt_route_plan = torch.zeros((batch_size, self._dummy_size), device=device)

        for i in range(seq_length):
            next_ = rec[arange, pre]
            next_is_dummy_node = next_ < self._dummy_size
            route[next_is_dummy_node] += 1
            route_plan_visited_time[arange, next_] = (route % self._dummy_size) * int(1e3) + (i + 1) % seq_length
            new_cum_demand = partial_sum_wrt_route_plan[arange, route % self._dummy_size] + demand[arange, next_]
            partial_sum_wrt_route_plan[arange, route % self._dummy_size] = new_cum_demand.clone()
            cum_demand[arange, next_] = new_cum_demand * (~next_is_dummy_node)

            pre = next_.clone()

        route_plan_0x = (route_plan_visited_time // int(1e3))

        out = (
                route_plan_0x,  # route plan 0xxxxx, belongs to which route
               (route_plan_visited_time % int(1e3)),  # visited time
               cum_demand.clone(),  # cum_demand (inclusive)
               partial_sum_wrt_route_plan.clone()# partial_sum_wrt_route_plan
               )

        return out

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