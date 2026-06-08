/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "local_search.cuh"

#include <utilities/cuda_helpers.cuh>
#include "compute_ejections.cuh"
#include "compute_insertions.cuh"
#include "vrp/nodes_to_search.cuh"
#include "vrp/vrp_search.cuh"

#include <routing/utilities/cuopt_utils.cuh>
#include <utilities/copy_helpers.hpp>

#include <thrust/fill.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <unordered_set>

#include <vector>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <cmath>
#include <set>

#include <cuopt/routing/utilities/internals.hpp>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
local_search_t<i_t, f_t, REQUEST>::local_search_t(const solution_handle_t<i_t, f_t>* sol_handle_,
                                                  i_t n_orders,
                                                  i_t max_routes,
                                                  bool depot_included,
                                                  viables_t<i_t, f_t>& viables_)
  : cycle_finder_small(sol_handle_, depot_included, 5, 50000),
    cycle_finder_big(sol_handle_, depot_included, 5, 50000),
    move_candidates(n_orders, max_routes, sol_handle_, viables_),
    vehicle_assignment(sol_handle_),
    two_opt_cand_data_(0, sol_handle_->get_stream()),
    sampled_nodes_data_(0, sol_handle_->get_stream()),
    found_sliding_solution_data_(max_routes, sol_handle_->get_stream()),
    sampled_tsp_data_(max_routes * n_orders, sol_handle_->get_stream()),
    moved_regions_(0, sol_handle_->get_stream()),
    locks_(max_routes, sol_handle_->get_stream()),
    sliding_anchor_per_route_(0, sol_handle_->get_stream()),
    two_opt_anchor_per_route_(0, sol_handle_->get_stream())
{
  raft::common::nvtx::range fun_scope("local_search_t");
}

template <typename i_t, typename f_t, request_t REQUEST>
void local_search_t<i_t, f_t, REQUEST>::clear_executed_anchors()
{
  last_executed_anchors_.clear();
  last_executed_anchor_operator_.clear();
}

template <typename i_t, typename f_t, request_t REQUEST>
void local_search_t<i_t, f_t, REQUEST>::append_executed_anchors(const std::vector<i_t>& node_ids,
                                                                int operator_id)
{
  for (i_t n : node_ids) {
    last_executed_anchors_.push_back(n);
    last_executed_anchor_operator_.push_back(operator_id);
  }
}

// sets the search weights and excess values for the local search
template <typename i_t, typename f_t, request_t REQUEST>
void local_search_t<i_t, f_t, REQUEST>::set_active_weights(const infeasible_cost_t weights_,
                                                           bool include_objective)
{
  move_candidates.weights.copy_from(weights_);
  move_candidates.include_objective = include_objective;
}

template <typename i_t, typename f_t, request_t REQUEST>
bool local_search_t<i_t, f_t, REQUEST>::run_two_opt_search(solution_t<i_t, f_t, REQUEST>& sol)
{
  raft::common::nvtx::range fun_scope("run_two_opt_search");
  bool move_found = perform_two_opt(sol, move_candidates);
  sol.sol_handle->sync_stream();
  if (move_found) {
    sol.compute_cost();
    cuopt_func_call(sol.check_cost_coherence(move_candidates.weights));
    return true;
  }
  return false;
}

template <typename i_t, typename f_t, request_t REQUEST>
bool local_search_t<i_t, f_t, REQUEST>::run_sliding_search(solution_t<i_t, f_t, REQUEST>& sol)
{
  raft::common::nvtx::range fun_scope("run_sliding_search");
  bool move_found = sol.problem_ptr->is_tsp ? perform_sliding_tsp(sol, move_candidates)
                                            : perform_sliding_window(sol, move_candidates);
  sol.sol_handle->sync_stream();
  if (move_found) {
    sol.compute_cost();
    cuopt_func_call(sol.check_cost_coherence(move_candidates.weights));
    return true;
  }
  return false;
}

template <typename i_t, typename f_t, request_t REQUEST>
bool local_search_t<i_t, f_t, REQUEST>::run_collect_prizes(solution_t<i_t, f_t, REQUEST>& sol)
{
  raft::common::nvtx::range fun_scope("run_collect_prizes");
  // don't run prize collection if there is no prize dimension
  if (!sol.problem_ptr->dimensions_info.has_dimension(dim_t::PRIZE)) { return false; }

  bool move_found = perform_prize_collection(sol);
  sol.sol_handle->sync_stream();
  if (move_found) {
    sol.compute_cost();
    cuopt_func_call(sol.check_cost_coherence(move_candidates.weights));
    return true;
  }
  return false;
}

template <typename i_t, typename f_t, request_t REQUEST>
bool local_search_t<i_t, f_t, REQUEST>::run_cross_search(solution_t<i_t, f_t, REQUEST>& sol)
{
  raft::common::nvtx::range fun_scope("run_cross_search");
  if (sol.n_routes < 2) { return false; }
  // independent thread scheduling is not present in pascal and we use it in populate_cross_list
  // kernel
  if (sol.sol_handle->get_device_properties().major < 7) { return false; }
  move_candidates.reset(sol.sol_handle);
  calculate_route_compatibility(sol);
  [[maybe_unused]] double cost_before = 0., cost_after = 0.;
  cuopt_func_call(sol.compute_cost());
  cuopt_func_call(cost_before =
                    sol.get_cost(move_candidates.include_objective, move_candidates.weights));
  find_insertions<i_t, f_t, REQUEST>(sol, move_candidates, search_type_t::CROSS);
  // directly populate move path after a cross move
  auto success = populate_cross_moves(sol, move_candidates);
  sol.sol_handle->sync_stream();
  // if we found a move, perform moves and continue
  if (success && move_candidates.move_path.n_insertions.value(sol.sol_handle->get_stream()) != 0) {
    perform_moves(sol, move_candidates);
    cuopt_func_call(sol.check_cost_coherence(move_candidates.weights));
    cuopt_func_call(cost_after =
                      sol.get_cost(move_candidates.include_objective, move_candidates.weights));

    cuopt_assert(cost_before - cost_after > EPSILON, "Cost should improve!");
    cuopt_assert(abs((cost_before - cost_after) -
                       move_candidates.debug_delta.value(sol.sol_handle->get_stream()) <
                     EPSILON * (1 + abs(cost_before))),
                 "Cost mismatch on cross costs!");
    return true;
  }
  return false;
}

template <typename i_t, typename f_t, request_t REQUEST>
template <request_t r_t, std::enable_if_t<r_t == request_t::PDP, bool>>
bool local_search_t<i_t, f_t, REQUEST>::run_fast_search(solution_t<i_t, f_t, r_t>& sol,
                                                        bool full_set,
                                                        i_t changed_nb_size,
                                                        bool random_shuffle,
                                                        bool look_ahead)
{
  raft::common::nvtx::range fun_scope("run_fast_search");

  std::vector<fast_operators_t> fast_operators{fast_operators_t::SLIDING, fast_operators_t::CROSS};
  if (!sol.problem_ptr->fleet_info.is_homogenous_ && !sol.problem_ptr->has_non_uniform_breaks()) {
    fast_operators.push_back(fast_operators_t::REGRET);
  }

  std::shuffle(fast_operators.begin(), fast_operators.end(), rng);

  // In PDP mode we do not run the full set
  for (auto const& op : fast_operators) {
    switch (op) {
      case fast_operators_t::SLIDING: {
        if (run_sliding_search(sol)) { return true; }
        break;
      }
      case fast_operators_t::VRP: {
        break;
      }
      case fast_operators_t::REGRET: {
        if (run_vehicle_assignment<i_t, f_t, REQUEST>(sol, move_candidates, vehicle_assignment)) {
          return true;
        }
        break;
      }
      case fast_operators_t::TWO_OPT: {
        break;
      }
      case fast_operators_t::CROSS: {
        if (run_cross_search(sol)) { return true; }
        break;
      }
    }
  }
  return false;
}

template <typename i_t, typename f_t, request_t REQUEST>
template <request_t r_t, std::enable_if_t<r_t == request_t::VRP, bool>>
bool local_search_t<i_t, f_t, REQUEST>::run_fast_search(solution_t<i_t, f_t, r_t>& sol,
                                                        bool full_set,
                                                        i_t changed_nb_size,
                                                        bool random_shuffle,
                                                        bool look_ahead)
{
  raft::common::nvtx::range fun_scope("run_fast_search");

  std::vector<fast_operators_t> fast_operators{fast_operators_t::SLIDING};

  if (!sol.problem_ptr->is_tsp) {
    fast_operators.push_back(fast_operators_t::VRP);
    fast_operators.push_back(fast_operators_t::TWO_OPT);
  }

  if (!sol.problem_ptr->fleet_info.is_homogenous_ && !sol.problem_ptr->has_non_uniform_breaks()) {
    fast_operators.push_back(fast_operators_t::REGRET);
  }

  // std::shuffle(fast_operators.begin(), fast_operators.end(), rng);

  auto& nodes_to_search = move_candidates.nodes_to_search;
  // this is activated if we ever want to run with full nodes
  // if (full_set) {
  //   sol.set_routes_to_search();
  //   extract_nodes_to_search(sol, move_candidates);
  // }
  if (!nodes_to_search.sample_nodes_to_search(sol, rng, full_set, random_shuffle = random_shuffle)) { return false; }

  clear_executed_anchors();
  bool move_found = false;
  move_found = run_sliding_search(sol) || move_found;
  move_found = perform_vrp_search(sol, move_candidates, changed_nb_size,
                                  &last_executed_anchors_, &last_executed_anchor_operator_) ||
               move_found;
  move_found = run_two_opt_search(sol) || move_found;

  // for (auto const& op : fast_operators) {
  //   switch (op) {
  //     case fast_operators_t::SLIDING: {
  //       move_found = run_sliding_search(sol) || move_found;
  //       break;
  //     }
  //     case fast_operators_t::VRP: {
  //       move_found = perform_vrp_search(sol, move_candidates, changed_nb_size) || move_found;
  //       break;
  //     }
  //     case fast_operators_t::REGRET: {
  //       move_found =
  //         run_vehicle_assignment<i_t, f_t, REQUEST>(sol, move_candidates, vehicle_assignment) ||
  //         move_found;
  //       break;
  //     }
  //     case fast_operators_t::TWO_OPT: {
  //       move_found = run_two_opt_search(sol) || move_found;
  //       break;
  //     }
  //     case fast_operators_t::CROSS: {
  //       break;
  //     }
  //   }
  // }

  move_candidates.nodes_to_search.restore_found_nodes(sol);

  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
template <request_t r_t>
std::vector<i_t> 
local_search_t<i_t, f_t, REQUEST>::build_solution_flat(solution_t<i_t, f_t, r_t>& sol) const
{
  size_t n_nodes = sol.get_num_orders();
  i_t total_nodes = n_nodes + sol.n_routes * 4;
  // Copy data from GPU to CPU
  std::vector<i_t> h_route_ids(n_nodes);
  std::vector<i_t> h_intra_idx(n_nodes);
  raft::copy(h_route_ids.data(), 
            sol.route_node_map.route_id_per_node.data(), 
            n_nodes, 
            sol.sol_handle->get_stream());
  raft::copy(h_intra_idx.data(), 
            sol.route_node_map.intra_route_idx_per_node.data(), 
            n_nodes, 
            sol.sol_handle->get_stream());
  
  // Initialize solution_flat
  std::vector<i_t> solution_flat;
  solution_flat.reserve(total_nodes);
  
  // Initialize temporary route storage
  std::vector<std::vector<i_t>> routes_temp(sol.n_routes);
  std::vector<i_t> max_route_length(sol.n_routes, 0);
  for (i_t r = 0; r < sol.n_routes; ++r) {
    routes_temp[r].resize(n_nodes, -1);
  }
  
  // Rebuild route structure from node mappings
  sol.sol_handle->sync_stream();
  for (i_t node_id = 0; node_id < (i_t)n_nodes; ++node_id) {
    i_t route_id = h_route_ids[node_id];
    i_t intra_idx = h_intra_idx[node_id];
    if (route_id != -1) {
      routes_temp[route_id][intra_idx] = node_id;
      max_route_length[route_id] = std::max(max_route_length[route_id], intra_idx);
    }
  }
  
  // Build flattened solution: for each route, add 4 dummy depot nodes + real nodes
  for (i_t route_id = 0; route_id < sol.n_routes; ++route_id) {
    // Add 4 dummy depot nodes
    for (i_t batch = 0; batch < 4; ++batch) {
      i_t dummy_id = n_nodes + route_id * 4 + batch;
      solution_flat.push_back(dummy_id);
    }
    // Add real nodes (only iterate to actual max, skip position 0 depot)
    for (i_t i = 1; i <= max_route_length[route_id]; ++i) {
      i_t node_id = routes_temp[route_id][i];
      if (node_id != -1) {
        solution_flat.push_back(node_id);
      }
    }
  }
  
  return solution_flat;
}

// template <typename i_t>
// i_t get_sample_size_vrp(i_t n_of_changed_nodes)
// {
//   return n_of_changed_nodes; //!!!
//   i_t num = 40;
//   if (n_of_changed_nodes < num)
//     num = n_of_changed_nodes;
//   else if (n_of_changed_nodes < num * 2)
//     num = n_of_changed_nodes / 2;
//   return num;
// }

template <typename i_t, typename f_t, request_t REQUEST>
void local_search_t<i_t, f_t, REQUEST>::print_solution(
    solution_t<i_t, f_t, REQUEST>& sol,
    const char* prefix)
{
  sol.sol_handle->sync_stream();
  printf("%s", prefix);
  for (i_t i = 0; i < sol.get_n_routes(); ++i) {
    auto& route = sol.get_route(i);
    auto node_infos = cuopt::host_copy(route.dimensions.requests.node_info);
    i_t n_nodes = route.n_nodes.value(sol.sol_handle->get_stream());
    printf("[");
    for (i_t j = 0; j < n_nodes; ++j) {
      printf("%d", node_infos[j].node());
      if (j < n_nodes - 1) {
        printf(",");
      }
    }
    printf("] ");
  }
  printf("\n");
}

template <typename i_t, typename f_t, request_t REQUEST>
template <typename Container, typename Accessor>
void local_search_t<i_t, f_t, REQUEST>::print_collection(
    const char* prefix, 
    const Container& container, 
    size_t limit, 
    Accessor accessor)
{
  printf("%s", prefix);

  if (container.empty()) {
    printf("[]\n");
    return;
  }

  bool first = true;
  size_t count = 0;
  printf("[");
  for (const auto& item : container) {
    if (count >= limit) break;
    i_t val = accessor(item);
    if (!first) {
      printf(",");
    }
    printf("%d", val);
    first = false;
    count++;
  }
  printf("]\n");
}

template <typename i_t, typename f_t, request_t REQUEST>
void local_search_t<i_t, f_t, REQUEST>::print_collection(
    const char* prefix, const std::vector<double>& values, size_t limit)
{
  printf("%s", prefix);
  if (values.empty()) {
    printf("[]\n");
    return;
  }
  const size_t n = std::min(limit, values.size());
  printf("[");
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) printf(",");
    printf("%.9g", values[i]);
  }
  printf("]\n");
}


template <typename i_t, typename f_t, request_t REQUEST>
std::set<std::pair<i_t, i_t>> local_search_t<i_t, f_t, REQUEST>::get_undirected_edges(
    solution_t<i_t, f_t, REQUEST>& sol)
{
  std::set<std::pair<i_t, i_t>> edges;
  const i_t dummy_base_offset = sol.problem_ptr->get_num_orders();
  for (i_t i = 0; i < sol.get_n_routes(); ++i) {
    auto& route = sol.get_route(i);
    auto dummy_start = dummy_base_offset + i * 4;
    auto node_infos = cuopt::host_copy(route.dimensions.requests.node_info);
    i_t n_nodes = route.n_nodes.value(sol.sol_handle->get_stream());
    
    if (n_nodes > 1) {
      for (i_t j = 0; j < n_nodes - 1; ++j) {
        i_t u = node_infos[j].node();
        i_t v = node_infos[j + 1].node();
        if (u > v) std::swap(u, v); // 存为无向边
        if (u==0) {
          edges.insert({dummy_start, v});
          edges.insert({dummy_start + 1, v});
          edges.insert({dummy_start + 2, v});
          edges.insert({dummy_start + 3, v});
        }
        else{
          edges.insert({u, v});
        }
      }
      i_t last_node = node_infos[n_nodes - 1].node();
      if (last_node != 0) {
        edges.insert({last_node, dummy_start});
        edges.insert({last_node, dummy_start + 1});
        edges.insert({last_node, dummy_start + 2});
        edges.insert({last_node, dummy_start + 3});
      }
    }
  }
  return edges;
}

template <typename i_t, typename f_t, request_t REQUEST>
std::pair<std::set<i_t>, std::set<i_t>> local_search_t<i_t, f_t, REQUEST>::compute_impact_and_intersection(
    const std::set<std::pair<i_t, i_t>>& old_edges,
    const std::set<std::pair<i_t, i_t>>& new_edges,
    const std::vector<NodeInfo<int>>& best_nodes_to_search,
    i_t sample_size)
{
  std::set<i_t> impacted_nodes;

  // 1. added edges (New - Old)
  for (const auto& edge : new_edges) {
    if (old_edges.find(edge) == old_edges.end()) {
      impacted_nodes.insert(edge.first);
      impacted_nodes.insert(edge.second);
    }
  }

  // 2. disappeared edges (Old - New)
  for (const auto& edge : old_edges) {
    if (new_edges.find(edge) == new_edges.end()) {
      impacted_nodes.insert(edge.first);
      impacted_nodes.insert(edge.second);
    }
  }

  std::set<i_t> intersection_nodes;
  for (auto node : best_nodes_to_search) {
    i_t node_id = node.node();
    if (impacted_nodes.find(node_id) != impacted_nodes.end()) {
      intersection_nodes.insert(node_id);
    }
  }

  return {impacted_nodes, intersection_nodes};
}

template <typename i_t, typename f_t, request_t REQUEST>
std::chrono::steady_clock::duration local_search_t<i_t, f_t, REQUEST>::run_best_local_search(solution_t<i_t, f_t, REQUEST>& sol,
                                                              const bool consider_unserviced,
                                                              const bool time_limit_enabled,
                                                              const bool run_cycle_finder)
{
  // Handle a corner case when there is no single task that is feasible
  if (sol.n_routes == 0) { return std::chrono::steady_clock::duration(0); }
  
  // Reset offset at the beginning
  total_offset = std::chrono::steady_clock::duration(0);
  // for production use working weights
  move_candidates.selection_weights = move_candidates.weights;
  // for benchmarks use low random weights
  benchmark_call(move_candidates.set_random_selection_weights(rng));

  // Run regret heuristic upfront to get correct assignment of vehicles
  if (!sol.problem_ptr->fleet_info.is_homogenous_ && !sol.problem_ptr->has_non_uniform_breaks()) {
    run_vehicle_assignment(sol, move_candidates, vehicle_assignment);
  }

  i_t iter = 0;
  sol.sol_handle->sync_stream();
  sol.compute_cost();
  i_t iter_limit = max_iterations;
  const bool should_all_nodes_be_served =
    consider_unserviced && !sol.problem_ptr->has_prize_collection();
  sol.global_runtime_checks(should_all_nodes_be_served, false, "run_best_local_search_begin");
  [[maybe_unused]] double cost_before = 0., cost_after = 0.;

  //########
  using Sol = cuopt::routing::detail::solution_t<i_t, f_t, REQUEST>;
  using clock = std::chrono::steady_clock;
  const i_t N_nodes_w_dummy = sol.problem_ptr->get_num_orders() + 4 * sol.get_n_routes();
  const i_t N_nodes = sol.problem_ptr->get_num_orders() + sol.get_n_routes();
  printf("[search #%d] N_nodes_w_dummy: %d, N_nodes: %d\n", global_local_search_iter, N_nodes_w_dummy, N_nodes);
  std::vector<NodeInfo<int>> full_node_to_search(N_nodes_w_dummy), work_node_to_search(N_nodes_w_dummy), best_node_to_search(N_nodes_w_dummy);
  std::mt19937_64 rng(std::random_device{}());
  bool pred_with_NN = false;
  // Get customize nodes callback
  callbacks::customize_nodes_callback_t<i_t, f_t>* obs_callback = nullptr;
  if (sol.problem_ptr->solver_settings_ptr) {
    for (auto callback : sol.problem_ptr->solver_settings_ptr->get_routing_callbacks()) {
      if (callback->get_type() == callbacks::callback_type_t::CUSTOMIZE_NODES) {
        obs_callback = static_cast<callbacks::customize_nodes_callback_t<i_t, f_t>*>(callback);
        pred_with_NN = true;
        break;
      }
    }
  }
  auto load_to_device_both = [&](std::vector<NodeInfo<int>> nodes_to_search) {
    move_candidates.nodes_to_search.h_nodes_to_search = nodes_to_search;
    move_candidates.nodes_to_search.n_sampled_nodes = nodes_to_search.size();
  };
  while (iter < iter_limit) {
    if constexpr (REQUEST == request_t::VRP) { 
      extract_nodes_to_search(sol, move_candidates);
      full_node_to_search = move_candidates.nodes_to_search.h_nodes_to_search;
    }
    iter++;
    double last_estimated_cost = 1000000000.0;
    // fast loop, insider this sliding, fast vrp search and fast cross search happens
    while (true) { 
      if (time_limit_enabled && local_search_t<i_t, f_t, REQUEST>::check_time_limit()) { break; }
      iter++;
      auto pause_begin = clock::now();
      const char* ls_mode = std::getenv("CUOPT_LS_MODE");
      bool origin = !(ls_mode && std::strcmp(ls_mode, "oracle") == 0);
      // #########

      if (!origin && pred_with_NN == false) {
        // Run full search Oracle
        Sol temp_trail_routes(sol);
        work_node_to_search = full_node_to_search;
        std::shuffle(work_node_to_search.begin(), work_node_to_search.begin() + work_node_to_search.size(), rng);
        load_to_device_both(work_node_to_search);

        printf("[iter #%d] candidate_size: %d\n", iter - 2, (int)full_node_to_search.size());
        print_solution(sol, "[before_search] sol: ");
        print_collection("[before_search] candidates: ", 
          work_node_to_search, work_node_to_search.size(), [](const auto& x) { return x.node(); });


        auto old_edges_undirected = get_undirected_edges(temp_trail_routes);
        auto move = run_fast_search(temp_trail_routes, true, 96, false, false); //!!!
        // move = run_fast_search(temp_trail_routes, temp_trail_routes.problem_ptr->is_tsp && iter == 2, 96, ii > 0);
        auto new_edges_undirected = get_undirected_edges(temp_trail_routes);


        // Record info: the set of useful nodes
        std::set<i_t> vrp_nodes, sliding_nodes, two_opt_nodes, all_anchor, excuted_anchor;
        auto& nt = move_candidates.nodes_to_search;
        nt.h_anchor_type_flags.resize(temp_trail_routes.get_num_orders() + temp_trail_routes.n_routes);
        raft::copy(nt.h_anchor_type_flags.data(), nt.anchor_type_flags.data(),
                  nt.h_anchor_type_flags.size(), temp_trail_routes.sol_handle->get_stream());
                  temp_trail_routes.sol_handle->sync_stream();
        for (i_t i = 0; i < (i_t)nt.h_anchor_type_flags.size(); ++i) {
          auto f = nt.h_anchor_type_flags[i];
          if (f == 0) continue;
          if (i < temp_trail_routes.get_num_orders()) {
            all_anchor.insert(i);
            if (f & ANCHOR_VRP) vrp_nodes.insert(i);
            if (f & ANCHOR_SLIDING) sliding_nodes.insert(i);
            if (f & ANCHOR_TWO_OPT) two_opt_nodes.insert(i);
          } else {
            i_t route_id = i - temp_trail_routes.get_num_orders();
            i_t base = temp_trail_routes.get_num_orders() + route_id * 4;
            all_anchor.insert(base + 0);
            all_anchor.insert(base + 1);
            all_anchor.insert(base + 2);
            all_anchor.insert(base + 3);
            if (f & ANCHOR_VRP) {
              vrp_nodes.insert(base + 0);
              vrp_nodes.insert(base + 1);
              vrp_nodes.insert(base + 2);
              vrp_nodes.insert(base + 3);
            }
            if (f & ANCHOR_SLIDING) {
              sliding_nodes.insert(base + 0);
              sliding_nodes.insert(base + 1);
              sliding_nodes.insert(base + 2);
              sliding_nodes.insert(base + 3);
            }
            if (f & ANCHOR_TWO_OPT) {
              two_opt_nodes.insert(base + 0);
              two_opt_nodes.insert(base + 1);
              two_opt_nodes.insert(base + 2);
              two_opt_nodes.insert(base + 3);
            }
          }
        }
        print_collection("[anchor] types=VRP: ", vrp_nodes, vrp_nodes.size(),
                        [](i_t x) { return x; });
        print_collection("[anchor] types=SLIDING: ", sliding_nodes, sliding_nodes.size(),
                        [](i_t x) { return x; });
        print_collection("[anchor] types=TWO_OPT: ", two_opt_nodes, two_opt_nodes.size(),
                        [](i_t x) { return x; });
        //########################################################
        // move_candidates.nodes_to_search.h_active_nodes_impacted.resize(N_nodes);
        // raft::copy(move_candidates.nodes_to_search.h_active_nodes_impacted.data(),
        //           move_candidates.nodes_to_search.active_nodes_impacted.data(),
        //           N_nodes,
        //           sol.sol_handle->get_stream());
        // sol.sol_handle->sync_stream();
        // std::set<i_t> impacted_nodes;
        // for (i_t i = 0; i < N_nodes; ++i) {
        //   if (move_candidates.nodes_to_search.h_active_nodes_impacted[i] == 1) {
        //     if (i < sol.get_num_orders()) {
        //       impacted_nodes.insert(i);
        //     }
        //     else {
        //       i_t route_id = i - sol.get_num_orders();
        //       i_t base = sol.get_num_orders() + route_id * 4;
        //       impacted_nodes.insert(base + 0);
        //       impacted_nodes.insert(base + 1);
        //       impacted_nodes.insert(base + 2);
        //       impacted_nodes.insert(base + 3);
        //     }
        //   }
        // }
        // auto [sol_impacted_nodes, sol_intersection_nodes] = compute_impact_and_intersection(old_edges_undirected, new_edges_undirected, full_node_to_search, 12);
        // std::set<int> intersection_anchor;
        // for (auto node_id : all_anchor) {
        //   if (sol_impacted_nodes.find(node_id) != sol_impacted_nodes.end()) {
        //     intersection_anchor.insert(node_id);
        //   }
        // }
        // print_collection("[after_search] h_active_nodes_impacted: ", 
        //   impacted_nodes, 
        //   impacted_nodes.size(),
        //   [](i_t x) { return x; });
        // print_collection("[after_search] sol_changed_nodes: ", 
        //   sol_impacted_nodes, 
        //   sol_impacted_nodes.size(),
        //   [](i_t x) { return x; });
        //##############################################
        // const auto& anchors = get_last_executed_anchors();
        // const auto& ops    = get_last_executed_anchor_operator();
        // if (!anchors.empty()) {
        //   std::set<i_t> by_op[4];
        //   for (size_t i = 0; i < anchors.size(); ++i) {
        //     int op = (i < ops.size()) ? ops[i] : 0;
        //     if (op >= 0 && op <= 3) by_op[op].insert(anchors[i]);
        //   }
        //   print_collection("[after_search] executed_anchors (sliding): ",
        //                    by_op[0], by_op[0].size(), [](i_t x) { return x; });
        //   print_collection("[after_search] executed_anchors (vrp): ",
        //                    by_op[1], by_op[1].size(), [](i_t x) { return x; });
        //   print_collection("[after_search] executed_anchors (recycle_vrp): ",
        //                    by_op[2], by_op[2].size(), [](i_t x) { return x; });
        //   print_collection("[after_search] executed_anchors (two_opt): ",
        //                    by_op[3], by_op[3].size(), [](i_t x) { return x; });
        //   excuted_anchor = std::set<i_t>(anchors.begin(), anchors.end());
        //   print_collection("[after_search] executed_anchors (all): ",
        //     excuted_anchor, excuted_anchor.size(), [](i_t x) { return x; });
        // }
        //########################################################
        // ##########
        // std::set<i_t> twenty_anchor_nodes;
        // // randomly select half of the full_node_to_search into best_node_to_search
        // std::vector<i_t> anchor_vec(all_anchor.begin(), all_anchor.end());
        // std::shuffle(anchor_vec.begin(), anchor_vec.end(), rng);
        // twenty_anchor_nodes = std::set<i_t>(anchor_vec.begin(), 
        //           anchor_vec.begin() + std::min(anchor_vec.size(), (size_t)20));
        // ##########
        
        //perform look ahead analysis on different subsets of excuted_anchor
        const int n_trails = 20;
        const int n_look_ahead = 3;
        std::vector<double> previous_cost(n_look_ahead + 1);
        previous_cost[0] = sol.get_cost(true, move_candidates.weights);
        double best_cost = 1000000000.0;
        std::set<i_t> best_subset;
        std::vector<i_t> anchor_vec(all_anchor.begin(), all_anchor.end());
        if (anchor_vec.empty()) { best_subset = all_anchor; }
        else {
          // std::uniform_int_distribution<size_t> size_dist(int(anchor_vec.size() * 1.0), int(anchor_vec.size() * 1.0));
          bool continue_flag = true;
          for (int i = 0; ((i < n_trails) || ((i >= n_trails) && continue_flag)) && i < 200; ++i) {
            std::shuffle(anchor_vec.begin(), anchor_vec.end(), rng);
            size_t subset_size = std::min(20, (int)anchor_vec.size()); //size_dist(rng);
            std::set<i_t> work_subset(anchor_vec.begin(), anchor_vec.begin() + subset_size);
            std::vector<NodeInfo<int>> work_node_list;
            work_node_list.reserve(work_subset.size());
            for (const auto& node_info : full_node_to_search) {
              if (work_subset.count(node_info.node())) work_node_list.push_back(node_info);
            }
            std::shuffle(work_node_list.begin(), work_node_list.end(), rng);
            Sol temp_trail_routes(sol);
            load_to_device_both(work_node_list);
            run_fast_search(temp_trail_routes, true, 96, false, false); //!!!
            print_collection("[before trail execution] work_subset: ", work_subset, work_subset.size(), [](i_t x) { return x; });
            print_solution(temp_trail_routes, "[after trail execution] sol: ");
            std::set<i_t> new_excuted_anchor;
            int number_of_anchors = 0;
            bool move_found_here = true;
            // init anchors_new and ops_new as empty
            std::vector<i_t> anchors_new;
            std::vector<int> ops_new;
            for (int ii = 0; ii < n_look_ahead; ++ii) {
              if (ii == 0) {
                anchors_new = get_last_executed_anchors();
                ops_new    = get_last_executed_anchor_operator();
                new_excuted_anchor = std::set<i_t>(anchors_new.begin(), anchors_new.end());
                number_of_anchors = (int)new_excuted_anchor.size();
              }
              if (move_found_here){
                load_to_device_both(full_node_to_search);
                move_found_here = run_fast_search(temp_trail_routes, true, 96, false, false); //!!!
              }
              previous_cost[ii+1] = temp_trail_routes.get_cost(true, move_candidates.weights);
            }
            auto cost = temp_trail_routes.get_cost(true, move_candidates.weights);
            if (cost < best_cost) {
              best_cost = cost;
              best_subset = new_excuted_anchor;
            }
            if (cost <= last_estimated_cost) { 
              continue_flag = false; 
              last_estimated_cost = cost;
            }
            printf("[trail #%d] last_estimated_cost: %f, cost: %f\n", i, last_estimated_cost, cost);
            std::set<i_t> by_op[4];
            for (size_t k = 0; k < anchors_new.size(); ++k) {
              int op = (k < ops_new.size()) ? ops_new[k] : 0;
              if (op >= 0 && op <= 3) by_op[op].insert(anchors_new[k]);
            }
            print_collection("executed_anchors (sliding): ",
                            by_op[0], by_op[0].size(), [](i_t x) { return x; });
            print_collection("executed_anchors (vrp): ",
                            by_op[1], by_op[1].size(), [](i_t x) { return x; });
            print_collection("executed_anchors (recycle_vrp): ",
                            by_op[2], by_op[2].size(), [](i_t x) { return x; });
            print_collection("executed_anchors (two_opt): ",
                            by_op[3], by_op[3].size(), [](i_t x) { return x; });
            excuted_anchor = std::set<i_t>(anchors_new.begin(), anchors_new.end());
            print_collection("executed_anchors (all): ",
              excuted_anchor, excuted_anchor.size(), [](i_t x) { return x; });
            print_collection("previous_cost: ", previous_cost, n_look_ahead + 1);
            printf("[trail #%d] full size: %zu, subset_size: %zu, cost: %f, number_of_anchors: %d, previous_number_of_anchors: %zu, best_cost: %f\n", 
                         i, all_anchor.size(), work_node_list.size(), cost, number_of_anchors, excuted_anchor.size(), best_cost);
          }
          if (continue_flag) {
            printf("[iter #%d] failed to find a better solution: %f\n", iter - 2, last_estimated_cost);
            last_estimated_cost = best_cost;
          }
        }
        
        // ##########
        auto label_nodes = best_subset;
        printf("[after_search] label_nodes size: %zu\n", label_nodes.size());
        print_solution(sol, "[after_search] sol: ");
        print_collection("[after_search] label_nodes: ", 
          label_nodes, 
          label_nodes.size(),
          [](i_t x) { return x; });
        
        // 添加 aggregated_intersection_nodes 中存在的节点（保持 full_node_to_search 中的顺序）
        best_node_to_search.clear();
        for (const auto& node_info : full_node_to_search) {
          i_t node_id = node_info.node();
          if (label_nodes.find(node_id) != label_nodes.end()) {
            best_node_to_search.push_back(node_info);
          }
        }
        std::shuffle(best_node_to_search.begin(), best_node_to_search.end(), rng);
      }

      // use callback and get best_node_to_search
      if (!origin && pred_with_NN == true) {
        
        if (obs_callback) {
          // Step 1: Oracle search to discover anchors
          Sol temp_oracle(sol);
          work_node_to_search = full_node_to_search;
          std::shuffle(work_node_to_search.begin(), work_node_to_search.end(), rng);
          load_to_device_both(work_node_to_search);
          run_fast_search(temp_oracle, true, 96, false, false);

          auto& nt = move_candidates.nodes_to_search;
          nt.h_anchor_type_flags.resize(temp_oracle.get_num_orders() + temp_oracle.n_routes);
          raft::copy(nt.h_anchor_type_flags.data(), nt.anchor_type_flags.data(),
                    nt.h_anchor_type_flags.size(), temp_oracle.sol_handle->get_stream());
          temp_oracle.sol_handle->sync_stream();

          std::set<i_t> all_anchor_nn;
          for (i_t i = 0; i < (i_t)nt.h_anchor_type_flags.size(); ++i) {
            auto f = nt.h_anchor_type_flags[i];
            if (f == 0) continue;
            if (i < temp_oracle.get_num_orders()) {
              all_anchor_nn.insert(i);
            } else {
              i_t route_id = i - temp_oracle.get_num_orders();
              i_t base = temp_oracle.get_num_orders() + route_id * 4;
              all_anchor_nn.insert(base + 0);
              all_anchor_nn.insert(base + 1);
              all_anchor_nn.insert(base + 2);
              all_anchor_nn.insert(base + 3);
            }
          }

          // Step 2: Run K trails. During training, CUOPT_RL_REWARD_HORIZON can
          // make each trail label use a short rollout while the policy action
          // remains the first-step executed-anchor mask.
          // K is overridable via CUOPT_RL_K to trade off probe cost vs action diversity.
          const char* rl_k_env = std::getenv("CUOPT_RL_K");
          const int K = (rl_k_env && std::atoi(rl_k_env) > 0) ? std::atoi(rl_k_env) : 100;
          const char* rl_horizon_env = std::getenv("CUOPT_RL_REWARD_HORIZON");
          const int reward_horizon =
            (rl_horizon_env && std::atoi(rl_horizon_env) > 0) ? std::atoi(rl_horizon_env) : 1;
          std::vector<i_t> anchor_vec_nn(all_anchor_nn.begin(), all_anchor_nn.end());
          std::vector<i_t> trail_masks_flat(K * N_nodes_w_dummy, 0);
          // Full-feedback labels are packed as [immediate K] + [lookahead K].
          // cuOpt only accepts improving moves, so candidates without an
          // accepted first-step move keep zero labels.
          std::vector<f_t> trail_rewards(K * 2, (f_t)0);
          const f_t base_cost = sol.get_cost(true, move_candidates.weights);

          std::unordered_map<i_t, size_t> node_id_to_h_idx;
          node_id_to_h_idx.reserve(full_node_to_search.size());
          for (size_t i = 0; i < full_node_to_search.size(); ++i) {
            node_id_to_h_idx[full_node_to_search[i].node()] = i;
          }

          for (int t = 0; t < K; ++t) {
            std::shuffle(anchor_vec_nn.begin(), anchor_vec_nn.end(), rng);
            size_t subset_size = std::min(20, (int)anchor_vec_nn.size());
            std::set<i_t> work_subset(anchor_vec_nn.begin(), anchor_vec_nn.begin() + subset_size);

            std::vector<NodeInfo<int>> work_node_list;
            work_node_list.reserve(work_subset.size());
            for (const auto& node_info : full_node_to_search) {
              if (work_subset.count(node_info.node())) work_node_list.push_back(node_info);
            }
            std::shuffle(work_node_list.begin(), work_node_list.end(), rng);

            Sol temp_trail(sol);
            load_to_device_both(work_node_list);
            bool move_found_trail = run_fast_search(temp_trail, true, 96, false, false);

            // Store only the first-step action mask. Lookahead continuation is
            // label generation, not part of the action exposed to the policy.
            auto anchors_exec = get_last_executed_anchors();
            auto ops_exec = get_last_executed_anchor_operator();
            bool has_first_step_mask = false;
            for (size_t k = 0; k < anchors_exec.size(); ++k) {
              i_t a = anchors_exec[k];
              if (a >= N_nodes_w_dummy) continue;
              int op = (k < ops_exec.size()) ? ops_exec[k] : -1;
              i_t mask = 0;
              if (op == 0) mask = 1;   // sliding
              if (op == 1) mask = 2;   // vrp
              if (op == 2) mask = 4;   // recycle_vrp
              if (op == 3) mask = 8;   // two_opt
              trail_masks_flat[t * N_nodes_w_dummy + a] |= mask;
              if (mask > 0) { has_first_step_mask = true; }
            }

            if (has_first_step_mask && move_found_trail) {
              trail_rewards[t] = base_cost - temp_trail.get_cost(true, move_candidates.weights);
              for (int h = 1; h < reward_horizon && move_found_trail; ++h) {
                load_to_device_both(full_node_to_search);
                move_found_trail = run_fast_search(temp_trail, true, 96, false, false);
              }
              trail_rewards[K + t] = base_cost - temp_trail.get_cost(true, move_candidates.weights);
            }
          }

          // Step 3: Call callback with K trail masks
          sol.sol_handle->sync_stream();
          std::vector<i_t> solution_flat = this->build_solution_flat(sol);
          f_t objective = sol.get_cost(true, move_candidates.weights);
          std::vector<i_t> selection_mask;

          obs_callback->customize_nodes_to_search(
              &solution_flat, sol.n_routes, objective,
              &trail_masks_flat, K, &trail_rewards, &selection_mask, iter);

          if (selection_mask.size() != (size_t)N_nodes_w_dummy) {
            printf("Selection mask size mismatch: %zu != %zu\n", selection_mask.size(), (size_t)N_nodes_w_dummy);
            exit(1);
          }

          // Step 4: Build best_node_to_search from selection_mask (bitmask, >0 means selected)
          best_node_to_search.clear();
          for (i_t node_id = 0; node_id < N_nodes_w_dummy; ++node_id) {
            if (selection_mask[node_id] > 0) {
              auto it = node_id_to_h_idx.find(node_id);
              if (it != node_id_to_h_idx.end()) {
                best_node_to_search.push_back(full_node_to_search[it->second]);
              }
            }
          }
          std::shuffle(best_node_to_search.begin(), best_node_to_search.end(), rng);

        } else {
          best_node_to_search = full_node_to_search;
          printf("No callback available, fall back to full_node_to_search\n");
          exit(1);
        }
      }
      // ##########
      // // randomly select half of the full_node_to_search into best_node_to_search
      // std::shuffle(full_node_to_search.begin(), full_node_to_search.end(), rng);
      // best_node_to_search.clear();
      // for (size_t i = 0; i < full_node_to_search.size(); ++i) {
      //   if (i < 40) {
      //     best_node_to_search.push_back(full_node_to_search[i]);
      //   }
      // }
      // ##########
      // end looking ahead
      if (!origin) { load_to_device_both(best_node_to_search); }
      auto pause_end   = clock::now();
      auto offset = pause_end - pause_begin;
      printf("offset: %ld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(offset).count());
      local_search_t<i_t, f_t, REQUEST>::add_offset(offset); //!!!!
      // #########

      // Run the actual search
      printf("[iter #%d] size of base nodes to search: %d, size of h_nodes_to_search: %d\n", iter - 2, (int)full_node_to_search.size(), (int)move_candidates.nodes_to_search.h_nodes_to_search.size());
      auto cost_before = sol.get_cost(true, move_candidates.weights);

      bool move_found_here = false;
      if (origin) {
        move_found_here = run_fast_search(sol, sol.problem_ptr->is_tsp && iter == 2, 96, false);
      } else {
        move_found_here = run_fast_search(sol, true, 96, false, false); //!!!
      }
      
      auto cost_after = sol.get_cost(true, move_candidates.weights);
      if (cost_after == cost_before) {
        move_found_here = false; //!!!
      }
      // RL reward feedback: report the cost delta of the action just executed.
      // No-op unless the registered callback implements on_search_result.
      if (!origin && pred_with_NN && obs_callback) {
        obs_callback->on_search_result(
          (f_t)cost_before, (f_t)cost_after, move_found_here, iter);
      }
      if (sol.is_feasible()) {
        printf("[executed] cost before: %f, cost after: %f, move_found: %d\n\n", cost_before, cost_after, move_found_here);
      }
      if (move_found_here) { continue; }
      if (consider_unserviced && sol.problem_ptr->has_prize_collection() &&
          run_collect_prizes(sol)) {
        continue;
      }
      if (!sol.problem_ptr->special_nodes.is_empty() && perform_break_moves(sol)) { continue; }
      break;
    }

    //########################################################
    sol.global_runtime_checks(
      should_all_nodes_be_served, false, "run_best_local_search_after_fast_search");
    printf("[iter #%d] run cycle finder\n", iter);
    if (!run_cycle_finder || (sol.n_routes > 1023)) { break; }
    // cycle finder is needed even for single route in PDP cases
    if (REQUEST == request_t::VRP && sol.n_routes < 2) { break; }
    move_candidates.reset(sol.sol_handle);
    calculate_route_compatibility(sol);
    find_insertions<i_t, f_t, REQUEST>(sol, move_candidates, search_type_t::IMPROVE);

    RAFT_CHECK_CUDA(sol.sol_handle->get_stream());
    sol.sol_handle->sync_stream();
    fill_gpu_graph(sol);

    move_candidates.find_best_negative_cycles(
      sol.n_routes, cycle_finder_small, cycle_finder_big, sol.sol_handle);
    cuopt_func_call(cost_before =
                      sol.get_cost(move_candidates.include_objective, move_candidates.weights));

    populate_move_path(sol, move_candidates);

    bool improved = move_candidates.move_path.n_insertions.value(sol.sol_handle->get_stream()) != 0;

    if (improved) {
      // printf("cycle found\n");
      sol.unset_routes_to_search();
      perform_moves(sol, move_candidates);
      cuopt_func_call(sol.check_cost_coherence(move_candidates.weights));
      sol.global_runtime_checks(should_all_nodes_be_served, false, "run_best_local_search_end");
      // with very big weights 1. epsilon is not enough
      cuopt_func_call(sol.compute_cost());
      cuopt_func_call(cost_after =
                        sol.get_cost(move_candidates.include_objective, move_candidates.weights));
      cuopt_assert((cost_after - cost_before) - move_candidates.cycles.total_cycle_cost < 1.,
                   "Cost mismatch after a move");
      sol.sol_handle->sync_stream();
    }

    // If there is no improvement at all, break the local search loop
    bool time_limit_reached =
      (time_limit_enabled && local_search_t<i_t, f_t, REQUEST>::check_time_limit());
    if (time_limit_reached || !improved) {
      cuopt_func_call(sol.check_cost_coherence(move_candidates.weights));
      break;
    }
  }
  // reset it, so that next time all routes will be searched unless otherwise is specified
  sol.set_routes_to_search();
  global_local_search_iter++;
  
  // Return the accumulated offset
  return total_offset;
}

template <typename i_t, typename f_t, request_t REQUEST>
void local_search_t<i_t, f_t, REQUEST>::run_random_local_search(solution_t<i_t, f_t, REQUEST>& sol,
                                                                bool time_limit_enabled)
{
  // Handle a corner case when there is no single task that is feasible
  if (sol.n_routes == 0) { return; }
  if (sol.n_routes > 1024) { return; }
  sol.sol_handle->sync_stream();

  // The weights for best local search must be set outside before calling this function
  set_active_weights(move_candidates.weights, move_candidates.include_objective);
  bool is_originally_feasible = sol.is_feasible();
  sol.global_runtime_checks(false, is_originally_feasible, "run_random_local_search_begin");

  move_candidates.reset(sol.sol_handle);
  move_candidates.random_move_candidates.reset(sol.sol_handle);
  calculate_route_compatibility(sol);
  find_insertions<i_t, f_t, REQUEST>(sol, move_candidates, search_type_t::RANDOM);

  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());
  sol.sol_handle->sync_stream();
  populate_random_moves(sol);
  bool time_limit_reached =
    (time_limit_enabled && local_search_t<i_t, f_t, REQUEST>::check_time_limit());
  // if there is no more insertions found
  if (move_candidates.move_path.n_insertions.value(sol.sol_handle->get_stream()) == 0 ||
      time_limit_reached) {
    return;
  }
  perform_moves(sol, move_candidates);

  sol.global_runtime_checks(false, is_originally_feasible, "run_random_local_search_end");
  sol.sol_handle->sync_stream();

  set_active_weights(move_candidates.weights, move_candidates.include_objective);
}

template <typename i_t, typename f_t, request_t REQUEST>
void local_search_t<i_t, f_t, REQUEST>::perturb_solution(solution_t<i_t, f_t, REQUEST>& sol,
                                                         i_t perturb_count)
{
  if (perturb_count <= 0) {
    i_t min_count = 1;
    i_t max_count = 8;
    i_t n_routes  = std::max(1, sol.get_n_routes());
    perturb_count = std::max(min_count, std::min(100 / n_routes, max_count));
  }

  for (i_t i = 0; i < perturb_count; ++i) {
    run_random_local_search(sol, false);
  }
}

template class local_search_t<int, float, request_t::PDP>;
template class local_search_t<int, float, request_t::VRP>;

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
