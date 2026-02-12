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
#include <unordered_set>

#include <vector>
#include <algorithm>
#include <random>
#include <unordered_map>

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
                                                        i_t changed_nb_size,
                                                        bool random_shuffle)
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
                                                        bool random_shuffle)
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

  std::shuffle(fast_operators.begin(), fast_operators.end(), rng);

  auto& nodes_to_search = move_candidates.nodes_to_search;
  // this is activated if we ever want to run with full nodes
  if (full_set) {
    sol.set_routes_to_search();
    extract_nodes_to_search(sol, move_candidates);
  }
  if (!nodes_to_search.sample_nodes_to_search(sol, rng, full_set, random_shuffle = random_shuffle)) { return false; }

  bool move_found = false;

  for (auto const& op : fast_operators) {
    switch (op) {
      case fast_operators_t::SLIDING: {
        move_found = run_sliding_search(sol) || move_found;
        break;
      }
      case fast_operators_t::VRP: {
        move_found = perform_vrp_search(sol, move_candidates, changed_nb_size) || move_found;
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

  move_candidates.nodes_to_search.restore_found_nodes(sol);
  if (full_set) { return move_found; }
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

template <typename i_t>
i_t get_sample_size_vrp(i_t n_of_changed_nodes)
{
  i_t num = 40;
  if (n_of_changed_nodes < num)
    num = n_of_changed_nodes;
  else if (n_of_changed_nodes < num * 2)
    num = n_of_changed_nodes / 2;
  return num;
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
  const i_t max_p_size = 1;
  const i_t N_nodes_w_dummy = sol.problem_ptr->get_num_orders() + 4 * sol.get_n_routes();
  const i_t N_nodes = sol.problem_ptr->get_num_orders() + sol.get_n_routes();
  printf("[search #%d] N_nodes_w_dummy: %d, N_nodes: %d\n", global_local_search_iter, N_nodes_w_dummy, N_nodes);
  std::mt19937_64 rng(std::random_device{}());
  // Get customize early stop callback
  callbacks::customize_early_stop_callback_t<i_t, f_t>* early_stop_callback = nullptr;
  if (sol.problem_ptr->solver_settings_ptr) {
    for (auto callback : sol.problem_ptr->solver_settings_ptr->get_routing_callbacks()) {
      if (callback->get_type() == callbacks::callback_type_t::CUSTOMIZE_EARLY_STOP) {
        early_stop_callback = static_cast<callbacks::customize_early_stop_callback_t<i_t, f_t>*>(callback);
        break;
      }
    }
  }
  bool early_stop = false;
  while (iter < iter_limit) {
    if constexpr (REQUEST == request_t::VRP) { extract_nodes_to_search(sol, move_candidates); }
    iter++;
    // fast loop, insider this sliding, fast vrp search and fast cross search happens
    while (true) { 
      if (time_limit_enabled && local_search_t<i_t, f_t, REQUEST>::check_time_limit()) { break; }
      iter++;

      // Run the actual search
      auto cost_before = sol.get_cost(true, move_candidates.weights);
      bool move_found_here = run_fast_search(sol, sol.problem_ptr->is_tsp && iter == 2, 96, false);
      auto cost_after = sol.get_cost(true, move_candidates.weights);
      if (sol.is_feasible()) {
        printf("[iter #%d] cost before: %f, cost after: %f, move_found: %d\n", iter, cost_before, cost_after, move_found_here);
      }
      auto pause_begin = clock::now();

      if (early_stop_callback) {
        // Sync stream before building solution_flat
        sol.sol_handle->sync_stream();

        // Build solution_flat and invoke callback for early stop decision
        std::vector<i_t> solution_flat = this->build_solution_flat(sol);
        f_t objective = sol.get_cost(true, move_candidates.weights);
        early_stop_callback->customize_early_stop(
            &solution_flat,
            sol.n_routes,
            objective,
            iter,
            &early_stop
        );
      }
      auto pause_end   = clock::now();
      auto offset = pause_end - pause_begin;
      printf("[iter #%d] offset: %ld ms\n", iter, std::chrono::duration_cast<std::chrono::milliseconds>(offset).count());
      local_search_t<i_t, f_t, REQUEST>::add_offset(offset);
      // #########

      if (move_found_here && !early_stop) { continue; }
      else if (early_stop) { printf("[iter #%d] early stop signal received\n\n", iter); }
      if (consider_unserviced && sol.problem_ptr->has_prize_collection() &&
          run_collect_prizes(sol)) {
        continue;
      }
      if (!sol.problem_ptr->special_nodes.is_empty() && perform_break_moves(sol)) { continue; }
      break;
    }
    if (early_stop) { break; }
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
