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
#include <cuopt/routing/utilities/internals.hpp>

#include <thrust/fill.h>
#include <unordered_set>

#include <chrono>
#include <unordered_map>
#include <unordered_set>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
local_search_t<i_t, f_t, REQUEST>::local_search_t(const solution_handle_t<i_t, f_t>* sol_handle_,
                                                  i_t n_orders,
                                                  i_t max_routes,
                                                  bool depot_included,
                                                  const viables_t<i_t, f_t>& viables_)
  : cycle_finder_small(sol_handle_, depot_included, 5, 50000),
    cycle_finder_big(sol_handle_, depot_included, 5, 50000),
    move_candidates(n_orders, max_routes, sol_handle_, viables_),
    vehicle_assignment(sol_handle_),
    two_opt_cand_data_(0, sol_handle_->get_stream()),
    sampled_nodes_data_(0, sol_handle_->get_stream()),
    found_sliding_solution_data_(max_routes, sol_handle_->get_stream()),
    sampled_tsp_data_(max_routes * n_orders, sol_handle_->get_stream()),
    moved_regions_(0, sol_handle_->get_stream()),
    locks_(max_routes, sol_handle_->get_stream())
{
  raft::common::nvtx::range fun_scope("local_search_t");
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
                                                        int iter)
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

template <typename i_t, typename f_t, request_t REQUEST>
template <request_t r_t, std::enable_if_t<r_t == request_t::VRP, bool>>
bool local_search_t<i_t, f_t, REQUEST>::run_fast_search(solution_t<i_t, f_t, r_t>& sol,
                                                        bool full_set,
                                                        int iter)
{
  raft::common::nvtx::range fun_scope("run_fast_search");

  auto& nodes_to_search = move_candidates.nodes_to_search;
  callbacks::customize_nodes_callback_t* obs_callback = nullptr;
  if (full_set) {
    sol.set_routes_to_search();
    extract_nodes_to_search(sol, move_candidates);
  }
  else {
    // Get customize nodes callback
    for (auto callback : sol.problem_ptr->solver_settings_ptr->get_routing_callbacks()) {
      if (callback->get_type() == callbacks::callback_type_t::CUSTOMIZE_NODES) {
        obs_callback = static_cast<callbacks::customize_nodes_callback_t*>(callback);
        break;
      }
    }
  }
  auto& h_nodes = nodes_to_search.h_nodes_to_search;
  bool needs_customization = h_nodes.size() > 0;
  if (!full_set && needs_customization && obs_callback) {
    // Prepare current solution for observation
    size_t n_nodes = sol.get_num_orders();
    i_t total_nodes = n_nodes + sol.n_routes * 4;

    // Build solution_flat using extracted function
    std::vector<i_t> solution_flat = build_solution_flat(sol);
    // Calculate objective for callback
    f_t objective = sol.get_cost(true, move_candidates.weights);

    // Prepare nodes to search and candidate_mask
    std::vector<i_t> candidate_mask(total_nodes, 0);
    std::unordered_map<i_t, size_t> node_id_to_h_idx;  // Map node_id to h_nodes index
    node_id_to_h_idx.reserve(h_nodes.size());
    for (size_t i = 0; i < h_nodes.size(); ++i) {
      i_t node_id = h_nodes[i].node();
      candidate_mask[node_id] = 1;
      node_id_to_h_idx[node_id] = i;  // Store index in h_nodes
    }

    // Callback: Get selection_mask from callback
    std::vector<i_t> selection_mask;
    obs_callback->customize_nodes_to_search(
        &solution_flat,
        sol.n_routes,
        objective,
        &candidate_mask,
        &selection_mask,
        iter
    );
    if (selection_mask.size() != (size_t)total_nodes) { 
      printf("Selection mask size mismatch: %zu != %zu\n", selection_mask.size(), (size_t)total_nodes);
      exit(1); 
    }
    // Extract selected nodes from mask and build sampled lists (single pass)
    std::vector<i_t> sampled_indices;
    nodes_to_search.h_sampled_nodes.clear();
    for (i_t node_id = 0; node_id < total_nodes; ++node_id) {
      if (selection_mask[node_id] == 1) {
        auto it = node_id_to_h_idx.find(node_id);
        size_t h_idx = it->second;
        sampled_indices.push_back(h_idx);
        nodes_to_search.h_sampled_nodes.push_back(h_nodes[h_idx]);
      }
    }
    if (sampled_indices.empty()) { return false; }

    // Transfer sampled nodes to GPU
    nodes_to_search.n_sampled_nodes = sampled_indices.size();
    nodes_to_search.sample_nodes_graph.start_capture(sol.sol_handle->get_stream());
    raft::copy(nodes_to_search.sampled_nodes_to_search.data(),
                nodes_to_search.h_sampled_nodes.data(),
                nodes_to_search.n_sampled_nodes,
                sol.sol_handle->get_stream());
    nodes_to_search.reset_active_nodes(sol.sol_handle);
    nodes_to_search.sample_nodes_graph.end_capture(sol.sol_handle->get_stream());
    nodes_to_search.sample_nodes_graph.launch_graph(sol.sol_handle->get_stream());

    // Remove sampled nodes using swap-and-pop (O(k log k))
    std::sort(sampled_indices.begin(), sampled_indices.end(), std::greater<i_t>());
    for (i_t idx : sampled_indices) {
      if (idx < (i_t)h_nodes.size() - 1) {
        h_nodes[idx] = std::move(h_nodes.back());
      }
      h_nodes.pop_back();
    }
  }
  else {
    if (!nodes_to_search.sample_nodes_to_search(sol, rng, full_set)) { return false;}    
  }

  std::vector<fast_operators_t> fast_operators{fast_operators_t::SLIDING};

  if (!sol.problem_ptr->is_tsp) {
    fast_operators.push_back(fast_operators_t::VRP);
    fast_operators.push_back(fast_operators_t::TWO_OPT);
  }

  if (!sol.problem_ptr->fleet_info.is_homogenous_ && !sol.problem_ptr->has_non_uniform_breaks()) {
    fast_operators.push_back(fast_operators_t::REGRET);
  }

  std::shuffle(fast_operators.begin(), fast_operators.end(), rng);

  bool move_found = false;

  for (auto const& op : fast_operators) {
    switch (op) {
      case fast_operators_t::SLIDING: {
        move_found = run_sliding_search(sol) || move_found;
        break;
      }
      case fast_operators_t::VRP: {
        move_found = perform_vrp_search(sol, move_candidates) || move_found;
        break;
      }
      case fast_operators_t::REGRET: {
        move_found =
          run_vehicle_assignment<i_t, f_t, REQUEST>(sol, move_candidates, vehicle_assignment) ||
          move_found;
        break;
      }
      case fast_operators_t::TWO_OPT: {
        move_found = run_two_opt_search(sol) || move_found;
        break;
      }
      case fast_operators_t::CROSS: {
        break;
      }
    }
  }

  // Reward: Send feedback to reward callback after search iteration
  if (!full_set && obs_callback && needs_customization) {
    for (auto callback : sol.problem_ptr->solver_settings_ptr->get_routing_callbacks()) {
      if (callback->get_type() == callbacks::callback_type_t::REWARD) {
        auto reward_callback = static_cast<callbacks::reward_callback_t*>(callback);
        f_t solution_cost = sol.get_cost(true, move_candidates.weights);
        // Build solution_flat for reward callback
        std::vector<i_t> solution_flat = build_solution_flat(sol);
        reward_callback->receive_reward(move_found, solution_cost, iter, &solution_flat, sol.n_routes);
        break;
      }
    }
  }

  move_candidates.nodes_to_search.restore_found_nodes(sol);
  if (full_set) { return move_found; }
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
void local_search_t<i_t, f_t, REQUEST>::run_best_local_search(solution_t<i_t, f_t, REQUEST>& sol,
                                                              const bool consider_unserviced,
                                                              const bool time_limit_enabled,
                                                              const bool run_cycle_finder)
{
  if (!consider_unserviced) {
    return;
  }
  // Handle a corner case when there is no single task that is feasible
  if (sol.n_routes == 0) { return; }
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
  const i_t iter_limit = max_iterations;
  const bool should_all_nodes_be_served =
    consider_unserviced && !sol.problem_ptr->has_prize_collection();
  sol.global_runtime_checks(should_all_nodes_be_served, false, "run_best_local_search_begin");
  [[maybe_unused]] double cost_before = 0., cost_after = 0.;

  // Local search start callback
  callbacks::local_search_start_callback_t* local_search_start_callback = nullptr;
  for (auto callback : sol.problem_ptr->solver_settings_ptr->get_routing_callbacks()) {
    if (callback->get_type() == callbacks::callback_type_t::LOCAL_SEARCH_START) {
      local_search_start_callback = static_cast<callbacks::local_search_start_callback_t*>(callback);
      break;
    }
  }
  if (local_search_start_callback) {
    std::vector<i_t> solution_flat = build_solution_flat(sol);
    f_t cost = sol.get_cost(true, move_candidates.weights);
    
    // Convert weights to std::vector<double> for callback
    std::vector<double> weights_vec = move_candidates.weights.to_vec();
    std::vector<double> selection_weights_vec = move_candidates.selection_weights.to_vec();
    
    local_search_start_callback->on_local_search_start(
      &solution_flat,
      sol.n_routes,
      cost,
      &weights_vec,
      &selection_weights_vec,
      should_all_nodes_be_served
    );
  }

  while (iter < iter_limit) {
    if constexpr (REQUEST == request_t::VRP) { extract_nodes_to_search(sol, move_candidates); }
    iter++;
    // fast loop, insider this sliding, fast vrp search and fast cross search happens
    while (true) {
      if (time_limit_enabled && local_search_t<i_t, f_t, REQUEST>::check_time_limit()) { break; }
      iter++;
      bool move_found = run_fast_search(sol, sol.problem_ptr->is_tsp && iter == 2, iter);
      // printf("move_found: %d\n", move_found);
      if (move_found) { continue; }
      if (consider_unserviced && sol.problem_ptr->has_prize_collection() &&
          run_collect_prizes(sol)) {
        continue;
      }
      if (!sol.problem_ptr->special_nodes.is_empty() && perform_break_moves(sol)) { continue; }
      break;
    }

    // Before cycle finder callback
    callbacks::before_cycle_finder_callback_t* before_cycle_finder_callback = nullptr;
    for (auto callback : sol.problem_ptr->solver_settings_ptr->get_routing_callbacks()) {
      if (callback->get_type() == callbacks::callback_type_t::BEFORE_CYCLE_FINDER) {
        before_cycle_finder_callback = static_cast<callbacks::before_cycle_finder_callback_t*>(callback);
        break;
      }
    }
    if (before_cycle_finder_callback) {
      std::vector<i_t> solution_flat = build_solution_flat(sol);
      f_t cost = sol.get_cost(true, move_candidates.weights);
      
      before_cycle_finder_callback->on_before_cycle_finder(
        &solution_flat,
        sol.n_routes,
        cost,
        iter + 1
      );
    }

    sol.global_runtime_checks(
      should_all_nodes_be_served, false, "run_best_local_search_after_fast_search");

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

    // After cycle finder callback
    callbacks::after_cycle_finder_callback_t* after_cycle_finder_callback = nullptr;
    for (auto callback : sol.problem_ptr->solver_settings_ptr->get_routing_callbacks()) {
      if (callback->get_type() == callbacks::callback_type_t::AFTER_CYCLE_FINDER) {
        after_cycle_finder_callback = static_cast<callbacks::after_cycle_finder_callback_t*>(callback);
        break;
      }
    }
    if (after_cycle_finder_callback) {
      std::vector<i_t> solution_flat = build_solution_flat(sol);
      f_t cost = sol.get_cost(true, move_candidates.weights);
      
      after_cycle_finder_callback->on_after_cycle_finder(
        &solution_flat,
        sol.n_routes,
        cost,
        iter + 1,
        improved
      );
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
