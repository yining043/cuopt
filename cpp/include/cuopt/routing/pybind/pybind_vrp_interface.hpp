/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <random>

// Include cuOpt routing headers - following cuOpt's exact pattern
#include <cuopt/routing/data_model_view.hpp>
#include <cuopt/routing/solver_settings.hpp>

// Forward declarations for other types
namespace raft { class handle_t; }
namespace rmm { template<typename T> class device_uvector; }

// Forward declarations for cuOpt internal types - simplified to avoid template issues
namespace cuopt {
namespace routing {
namespace detail {
    // Only forward declare the basic types without complex template parameters
    template <typename i_t, typename f_t> class move_candidates_t;
    template <typename i_t, typename f_t> class solution_handle_t;
    template <typename i_t, typename f_t> class viables_t;
} // namespace detail
} // namespace routing
} // namespace cuopt

namespace cuopt {
namespace routing {
namespace pybind {

// VrpLS class - cuOpt Local Search wrapper for VRP optimization, following cuOpt's exact initialization pattern
class VrpLS {
public:
    VrpLS(int num_locations, int fleet_size, int num_orders);
    ~VrpLS();

    // VRP problem setup methods - reuse cuOpt's data_model_view_t exactly
    void add_cost_matrix(const std::vector<std::vector<float>>& cost_matrix, uint8_t vehicle_type = 0);
    void add_capacity_dimension(const std::string& name, const std::vector<int>& demand, const std::vector<int>& capacity);
    // Initialization method - requires initial solution
    void initialize_search(const std::vector<std::vector<int>>& routes, 
                          const std::vector<int>& vehicle_ids = {});

    // The core method - directly call cuOpt's perform_vrp_search via local_search_t
    bool perform_vrp_search();

    // Resource management methods - following cuOpt's pattern
    void acquire_resource();
    void release_resource();
    void sync_streams();

    // Node management methods - following cuOpt's pattern  
    void extract_nodes_to_search();
    void restore_found_nodes();
    bool sample_nodes_to_search(bool full_set = false);
    
    // Move candidates management
    std::vector<std::vector<int>> get_move_candidates() const;
    void set_move_candidates(const std::vector<std::vector<int>>& modified_candidates);
    
    // Debug interface to compare data sources
    std::vector<std::vector<int>> get_sampled_nodes() const;
    
    // Weights management
    std::vector<double> get_weights() const;
    void set_weights(const std::vector<double>& weights);
    std::vector<double> get_selection_weights() const;
    void set_selection_weights(const std::vector<double>& selection_weights);
    
    // Route search management
    void set_routes_to_search();
    void unset_routes_to_search();
    
    // Move candidates management
    void reset_move_candidates();
    
    // Solution management methods
    void setup_solution(const std::vector<std::vector<int>>& routes, 
                       const std::vector<int>& vehicle_ids = {});
    
    // Solution access methods - simplified to whole solution operations only
    std::vector<std::vector<int>> get_solution_routes() const;
    std::vector<int> get_solution_vehicle_ids() const;
    double get_cost() const;
    int get_n_routes() const;
    int get_num_orders() const;

    // CUDA implementation methods - implemented in .cu file
    void initialize_search_impl(const std::vector<std::vector<int>>& routes, 
                               const std::vector<int>& vehicle_ids);
    bool perform_vrp_search_impl();
    void setup_solution_impl(const std::vector<std::vector<int>>& routes, 
                            const std::vector<int>& vehicle_ids);
    void validate_routes(const std::vector<std::vector<int>>& routes, const std::string& context);
    std::vector<int> process_vehicle_ids(const std::vector<std::vector<int>>& routes, 
                                        const std::vector<int>& vehicle_ids, 
                                        const std::string& context);
    int get_n_routes_impl() const;
    double get_cost_impl() const;
    
    std::vector<std::vector<int>> get_solution_routes_impl() const;
    std::vector<int> get_solution_vehicle_ids_impl() const;

    // Resource management implementation methods
    void acquire_resource_impl();
    void release_resource_impl();
    void sync_streams_impl();

    // Node management implementation methods
    void extract_nodes_to_search_impl();
    void restore_found_nodes_impl();
    bool sample_nodes_to_search_impl(bool full_set = false);
    
    // Move candidates management
    std::vector<std::vector<int>> get_move_candidates_impl() const;
    void set_move_candidates_impl(const std::vector<std::vector<int>>& modified_candidates);
    std::vector<std::vector<int>> get_sampled_nodes_impl() const;
    
    // Weights management
    std::vector<double> get_weights_impl() const;
    void set_weights_impl(const std::vector<double>& weights);
    std::vector<double> get_selection_weights_impl() const;
    void set_selection_weights_impl(const std::vector<double>& selection_weights);
    

private:
    // Core cuOpt components - reuse existing cuOpt classes exactly as cuOpt does
    std::unique_ptr<raft::handle_t> handle_;
    std::unique_ptr<data_model_view_t<int, float>> data_model_ptr_;
    std::unique_ptr<solver_settings_t<int, float>> solver_settings_ptr_;
    
    // cuOpt internal objects - follow cuOpt's exact initialization pattern
    // Use opaque pointer to avoid template instantiation in header
    void* problem_ptr_;
    
    // GPU stream management - simplified for single solution (no pool_allocator needed)
    // Use opaque pointer to avoid template instantiation in header
    void* stream_pool_ptr_;        // rmm::cuda_stream_pool* (simplified from pool_allocator)
    void* solution_handle_ptr_;    // solution_handle_t<int, float>* (extracted from pool_allocator)
    
    // Solution and local search - reuse cuOpt's existing classes exactly
    // Use opaque pointer to avoid template instantiation in header
    void* solution_ptr_;
    void* local_search_ptr_;
    
    // Device memory management - follow cuOpt's exact memory pattern
    std::vector<rmm::device_uvector<float>> device_cost_matrices_;
    std::vector<float*> cost_matrix_ptrs_;
    std::vector<rmm::device_uvector<int>> device_demands_;
    std::vector<rmm::device_uvector<int>> device_capacities_;
    
    // Resource management state - following cuOpt's pattern
    bool resource_acquired_ = false;
    int resource_index_ = -1;
    
    
    // Random number generator for node sampling
    std::mt19937 rng_;
    
    bool finalize_called_ = false;
};


} // namespace pybind
} // namespace routing
} // namespace cuopt