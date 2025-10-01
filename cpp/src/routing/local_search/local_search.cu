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
#include "vrp/vrp_execute.cuh"

// forward declaration to use recycle-mode search without including implementation TUs
namespace cuopt {
namespace routing {
namespace detail {
template <typename i_t, typename f_t, request_t REQUEST>
bool find_vrp_moves(solution_t<i_t, f_t, REQUEST>& sol,
                    move_candidates_t<i_t, f_t>& move_candidates,
                    bool recycle,
                    i_t changed_nb_size);
template <typename i_t, typename f_t, request_t REQUEST>
bool recycle_unused_moves(solution_t<i_t, f_t, REQUEST>& sol,
                          move_candidates_t<i_t, f_t>& move_candidates,
                          i_t changed_nb_size);
template <typename i_t, typename f_t, request_t REQUEST>
bool perform_vrp_search(solution_t<i_t, f_t, REQUEST>& sol,
                          move_candidates_t<i_t, f_t>& move_candidates,
                          i_t changed_nb_size);
}  // namespace detail
}  // namespace routing
}  // namespace cuopt

#include <routing/utilities/cuopt_utils.cuh>
#include <utilities/copy_helpers.hpp>

#include <thrust/fill.h>

#include <chrono>
#include <unordered_set>

#include <vector>
#include <algorithm>
#include <random>

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
                                                        i_t changed_nb_size)
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
                                                        i_t changed_nb_size)
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
  if (!nodes_to_search.sample_nodes_to_search(sol, rng, full_set)) { return false; }

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
void local_search_t<i_t, f_t, REQUEST>::run_best_local_search(solution_t<i_t, f_t, REQUEST>& sol,
                                                              const bool consider_unserviced,
                                                              const bool time_limit_enabled,
                                                              const bool run_cycle_finder)
{
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
  i_t iter_limit = max_iterations;
  const bool should_all_nodes_be_served =
    consider_unserviced && !sol.problem_ptr->has_prize_collection();
  sol.global_runtime_checks(should_all_nodes_be_served, false, "run_best_local_search_begin");
  [[maybe_unused]] double cost_before = 0., cost_after = 0.;

  //########
  using Sol = cuopt::routing::detail::solution_t<i_t, f_t, REQUEST>;
  using clock = std::chrono::steady_clock;
  const i_t N1 = sol.problem_ptr->get_num_orders() + 4 * sol.get_n_routes();  // nodes
  const i_t N2 = sol.problem_ptr->get_num_orders() + sol.get_n_routes();  // nodes
  std::vector<NodeInfo<int>> base_node_to_search(N1), work_node_to_search(N1), save_node_to_search(N1), best_node_to_search_before(N1),best_node_to_search_after(N1);
  std::vector<int> save_h_best_id_per_node(N2), h_best_id_per_node(N2);
  std::mt19937 rng(std::random_device{}());

  // define function to load to device both tables
  auto load_nodes_to_search = [&](std::vector<NodeInfo<int>> nodes_to_search) {
    move_candidates.nodes_to_search.h_nodes_to_search = nodes_to_search;
    move_candidates.nodes_to_search.n_sampled_nodes = nodes_to_search.size();
  };

  //########
  while (iter < iter_limit) {
    printf("iter: %d\n", iter);
    if constexpr (REQUEST == request_t::VRP) { extract_nodes_to_search(sol, move_candidates); }
    iter++;
    // fast loop, insider this sliding, fast vrp search and fast cross search happens
    while (true) { 
       if (time_limit_enabled && local_search_t<i_t, f_t, REQUEST>::check_time_limit()) { break; }
      iter++;
      //########################################################
      auto pause_begin = clock::now();
      // 10 次候选
      double best_score = 1000000000.0;
      base_node_to_search = move_candidates.nodes_to_search.h_nodes_to_search;
      best_node_to_search_before = base_node_to_search;
      best_node_to_search_after = base_node_to_search;
      const size_t KK = base_node_to_search.size();
      if (KK > 1) {
          std::shuffle(base_node_to_search.begin(),
                      base_node_to_search.begin() + KK,
                      rng);
      }

      // std::vector<NodeInfo<int>> intersection;
      // intersection.clear();
      // for (const auto& node : base_node_to_search) {
      //   intersection.push_back(node);
      // }
      bool total_success = false;
      for (int t = 0; t < 1; ++t) {
        
        if (t == 0) {
          work_node_to_search = base_node_to_search;  // 不扰动
        } else {
          work_node_to_search = base_node_to_search;  // 从原始拷贝一份再轻微扰动
          const size_t KK = work_node_to_search.size();
          if (KK > 1) {
              std::shuffle(work_node_to_search.begin(),
                          work_node_to_search.begin() + KK,
                          rng);
          }
        }
        load_nodes_to_search(work_node_to_search);
        Sol trail_routes(sol);
        bool move = true;
        bool success = true;
        if constexpr (REQUEST == request_t::VRP) {
          for (int k = 0; k < 1; ++k) {
            if (move_candidates.nodes_to_search.sample_nodes_to_search(trail_routes, rng, false) && move) {
              move = perform_vrp_search(trail_routes, move_candidates, 96);
              move_candidates.nodes_to_search.restore_found_nodes(trail_routes);
              // printf("trail %d, move_found: %d, n_sampled_nodes: %d\n", t, move_found, move_candidates.nodes_to_search.n_sampled_nodes);
            
              //for the first time
              if (k==0) {
                save_node_to_search = move_candidates.nodes_to_search.h_nodes_to_search;
                raft::copy(save_h_best_id_per_node.data(),
                  move_candidates.vrp_move_candidates.best_id_per_node.data(),
                  save_h_best_id_per_node.size(),
                  sol.sol_handle->get_stream()
                );
                // Synchronize again to ensure copy is complete
                sol.sol_handle->sync_stream();
              }
            }
            else if (k==0)
            {
              success = false;
              move = false;
              save_node_to_search = move_candidates.nodes_to_search.h_nodes_to_search;
              break;
            }
            else
            {
              break;
            }
          }
        } else {
          exit(0);
        }

        auto s = trail_routes.get_cost(true, move_candidates.weights);
        if (s < best_score && success) {
          best_score = s;
          best_node_to_search_before = work_node_to_search;  // 保存最佳配置
          best_node_to_search_after = save_node_to_search;  // 保存最佳配置
          h_best_id_per_node = save_h_best_id_per_node;
        }
        total_success = total_success || success;

        // 试探后：move_candidates.nodes_to_search.h_nodes_to_search 是剩余的节点列表
        // {
        //   const auto& remaining_nodes = move_candidates.nodes_to_search.h_nodes_to_search;
        //   std::set<int> remaining_node_ids;
        //   for (const auto& node : remaining_nodes) {
        //     remaining_node_ids.insert(node.node());
        //   }
          
        //   // 更新 intersection 为交集（移除不在剩余节点中的而且host_best_ids[nid]==-1的节点）
        //   intersection.erase(
        //     std::remove_if(intersection.begin(), intersection.end(),
        //       [&remaining_node_ids](const NodeInfo<int>& node) {
        //         return remaining_node_ids.find(node.node()) == remaining_node_ids.end();
        //       }),
        //     intersection.end()
        //   );
        // }



        // auto s = trail_routes.get_cost(true, move_candidates.weights);
        // if (s < best_score) {
        //   best_score = s;
        //   best_node_to_search = work_node_to_search;  // 保存最佳配置
        // }
      }
      
      if constexpr (REQUEST == request_t::VRP) {
        // int count_non_null = 0;
        // printf("trail %d, count_non_null: %d\n", t, count_non_null);

        // 用 aggregated_pairs 替换 move_candidates.vrp_move_candidates.best_id_per_node
        // auto& aggregated_pairs = move_candidates.nodes_to_search.h_recycled_node_pairs;
        // bool nodes_remained = !aggregated_pairs.empty();
        // printf("fast-search pairs aggregated before:  %zu\n", aggregated_pairs.size());
        // if (nodes_remained) {

        //   // 去重配对
        //   auto pair_less = [](const int2& a, const int2& b) {
        //     if (a.x != b.x) { return a.x < b.x; }
        //     return a.y < b.y;
        //   };
        //   std::sort(aggregated_pairs.begin(), aggregated_pairs.end(), pair_less);
        //   aggregated_pairs.erase(std::unique(aggregated_pairs.begin(),
        //                                     aggregated_pairs.end(),
        //                                     [](const int2& a, const int2& b) {
        //                                       return a.x == b.x && a.y == b.y;
        //                                     }),
        //                         aggregated_pairs.end());
        //   raft::copy(move_candidates.nodes_to_search.recycled_node_pairs.data(),
        //               aggregated_pairs.data(),
        //               aggregated_pairs.size(),
        //               sol.sol_handle->get_stream());
        // }
        
        // printf("fast-search pairs aggregated after: %zu\n", aggregated_pairs.size());
        
        // 10次试探后，intersection就是剩余未搜索的节点
        // printf("remaining nodes after 10 trials: %zu\n", intersection.size());

      } else {
        exit(0);
      }
      // 使用剩余节点集合进行最后的搜索
      load_nodes_to_search(best_node_to_search_after);
      if (total_success){
        // Ensure all device operations are complete before using host data
        move_candidates.nodes_to_search.h_recycled_node_pairs.clear();
        for (i_t nid = 0; nid < (int)h_best_id_per_node.size(); ++nid) {
          const i_t best = h_best_id_per_node[nid];
          if (best != -1) {
            move_candidates.nodes_to_search.h_recycled_node_pairs.push_back(int2{nid, best});
          }
        }
        raft::copy(move_candidates.nodes_to_search.recycled_node_pairs.data(),
                  move_candidates.nodes_to_search.h_recycled_node_pairs.data(),
                  move_candidates.nodes_to_search.h_recycled_node_pairs.size(),
                  sol.sol_handle->get_stream());
        sol.sol_handle->sync_stream();
        move_candidates.nodes_to_search.n_sampled_nodes = move_candidates.nodes_to_search.h_recycled_node_pairs.size(); //!!!! not sure if bug!!!!
      }
      printf("number of nodes to search: %zu, size of base_nodes_to_search: %zu\n", move_candidates.nodes_to_search.h_nodes_to_search.size(), base_node_to_search.size());
      printf("number of recycled pairs: %zu\n", move_candidates.nodes_to_search.h_recycled_node_pairs.size());
      //!!!! no delete!!!!
      
      auto pause_end   = clock::now();
      auto offset = pause_end - pause_begin;
      local_search_t<i_t, f_t, REQUEST>::add_offset(offset);
      //########################################################
      // raft::print_device_vector("viable_to_pickups: ",
      //                   move_candidates.viables.viable_to_pickups.data() + 1000,
      //                   100,
      //                   std::cout);
      // raft::print_device_vector("viable_from_pickups: ",
      //                       move_candidates.viables.viable_from_pickups.data() + 1000,
      //                       100,
      //                       std::cout);
      //########################################################
      //########################################################
      // try trail with sizes [16, 64, 256]
      // for (i_t size : {4, 8, 16, 32, 64, 128, 256}) {
      //   Sol trail_routes(sol);
      //   printf("trail with size %d\n", size);
      //   run_fast_search(trail_routes, trail_routes.problem_ptr->is_tsp && iter == 2, size);
      // }
      // //real search size
      // ###########################
      //########################################################
      //########################################################
      f_t cost_before = sol.get_cost(true, move_candidates.weights);
      bool move_found_here = false;
      if constexpr (REQUEST == request_t::VRP) {
        // 仅 VRP：使用 recycle 的轻量搜索
        if (move_candidates.nodes_to_search.h_recycled_node_pairs.size() > 0 && total_success) {
          for (int attempt = 0; attempt < 1; ++attempt) {
              auto copy_sampled_nodes = move_candidates.nodes_to_search.h_sampled_nodes;
              move_found_here = recycle_unused_moves(sol, move_candidates, 96) || move_found_here;
              if (move_found_here) {
                break;
              }
              else{
                load_nodes_to_search(best_node_to_search_before);
                move_candidates.nodes_to_search.h_sampled_nodes = copy_sampled_nodes;
                move_candidates.nodes_to_search.n_sampled_nodes = copy_sampled_nodes.size();
                move_found_here = run_sliding_search(sol) || move_found_here;
                if (move_found_here) {
                  break;
                }
                else{
                  // load_nodes_to_search(best_node_to_search_before);
                  // move_found_here = run_vehicle_assignment<int, float, request_t::VRP>(sol, move_candidates, vehicle_assignment) || move_found_here;
                  if (move_found_here) {
                    break;
                  }
                  else{
                    // load_nodes_to_search(best_node_to_search_before);
                    move_found_here = run_two_opt_search(sol) || move_found_here;
                    if (move_found_here) {
                      break;
                    }
                  }
                }
              }
            }            
          move_candidates.nodes_to_search.restore_found_nodes(sol);
        }
        else {
          // load_nodes_to_search(base_node_to_search);
          // move_found_here = run_fast_search(sol, sol.problem_ptr->is_tsp && iter == 2, 96);
          printf("not success at : %d\n", iter);
        }
      } else {
        // 非 VRP：沿用原 run_fast_search
        exit(0);
      }
      f_t cost_after = sol.get_cost(true, move_candidates.weights);
      printf("cost before: %f, cost after: %f, move_found: %d\n", cost_before, cost_after, move_found_here);

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
    
    // disable cycle finder for now
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
