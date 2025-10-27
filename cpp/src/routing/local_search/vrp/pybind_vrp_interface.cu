/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuopt/routing/pybind/pybind_vrp_interface.hpp>
#include "../../dimensions.cuh"

// CUDA-specific includes
#include <raft/core/handle.hpp>

// cuOpt routing headers
#include "../../structures.hpp"
#include "../../problem/problem.cuh"
#include "../../solution/solution.cuh"
#include "../../solution/pool_allocator.cuh"
#include "../local_search.cuh"
#include "vrp_search.cuh"

namespace py = pybind11;

namespace cuopt {
namespace routing {
namespace pybind {

// ============================================================================
// Type aliases and helper functions
// ============================================================================

// Type aliases for cleaner code
using Solution = detail::solution_t<int, float, request_t::VRP>;
using LocalSearch = detail::local_search_t<int, float, request_t::VRP>;
using SolutionHandle = detail::solution_handle_t<int, float>;
using Problem = detail::problem_t<int, float>;

// Helper to check if initialize_search has been called
inline void check_initialized(bool initialize_called, const char* func_name) {
    if (!initialize_called) {
        throw std::runtime_error(std::string("Must call initialize_search() before ") + func_name);
    }
}

// ============================================================================
// Core functionality - initialization and search
// ============================================================================

// Destructor - clean up all resources
VrpLS::~VrpLS() {
    if (solution_ptr_) {
        delete static_cast<Solution*>(solution_ptr_);
        solution_ptr_ = nullptr;
    }
    if (local_search_ptr_) {
        delete static_cast<LocalSearch*>(local_search_ptr_);
        local_search_ptr_ = nullptr;
    }
    if (solution_handle_ptr_) {
        delete static_cast<SolutionHandle*>(solution_handle_ptr_);
        solution_handle_ptr_ = nullptr;
    }
    if (stream_pool_ptr_) {
        delete static_cast<rmm::cuda_stream_pool*>(stream_pool_ptr_);
        stream_pool_ptr_ = nullptr;
    }
    if (problem_ptr_) {
        delete static_cast<Problem*>(problem_ptr_);
        problem_ptr_ = nullptr;
    }
}

// Initialize cuOpt internal objects for local search
void VrpLS::initialize_search_impl(const std::vector<std::vector<int>>& routes, 
                                   const std::vector<int>& vehicle_ids) {
    if (finalize_called_) return;
    
    // Create problem from data model
    auto problem = std::make_unique<Problem>(*data_model_ptr_, *solver_settings_ptr_);
    problem_ptr_ = problem.release();
    auto* problem_obj = static_cast<Problem*>(problem_ptr_);
    
    // Create CUDA stream and solution handle
    auto stream_pool = std::make_unique<rmm::cuda_stream_pool>(1);
    stream_pool_ptr_ = stream_pool.release();
    
    auto stream_pool_obj = static_cast<rmm::cuda_stream_pool*>(stream_pool_ptr_);
    auto solution_handle = std::make_unique<SolutionHandle>(stream_pool_obj->get_stream(0));
    solution_handle_ptr_ = solution_handle.release();

    // Setup solution with routes and vehicle IDs
    setup_solution_impl(routes, vehicle_ids);
    
    // Create local search
    auto local_search = std::make_unique<LocalSearch>(
        static_cast<SolutionHandle*>(solution_handle_ptr_),
        problem_obj->get_num_orders(),
        problem_obj->get_fleet_size(),
        problem_obj->order_info.depot_included_,
        problem_obj->viables);
    
    // Set active weights
    detail::infeasible_cost_t weights(cuopt::routing::detail::default_weights);
    local_search->set_active_weights(weights, true);
    local_search_ptr_ = local_search.release();

    finalize_called_ = true;
}



// Perform VRP local search - find and execute a single move
bool VrpLS::perform_vrp_search_impl() {
    check_initialized(finalize_called_, "perform_vrp_search()");
    
    auto* solution_obj = static_cast<Solution*>(solution_ptr_);
    auto* local_search_obj = static_cast<LocalSearch*>(local_search_ptr_);
    
    return cuopt::routing::detail::perform_vrp_search(*solution_obj, local_search_obj->move_candidates);
}

// Perform two-opt local search - intra-route optimization
bool VrpLS::run_two_opt_search_impl() {
    check_initialized(finalize_called_, "run_two_opt_search()");
    
    auto* solution_obj = static_cast<Solution*>(solution_ptr_);
    auto* local_search_obj = static_cast<LocalSearch*>(local_search_ptr_);
    
    return local_search_obj->perform_two_opt(*solution_obj, local_search_obj->move_candidates);
}

// ============================================================================
// Resource management - simplified for single solution
// ============================================================================

void VrpLS::acquire_resource_impl() {
    resource_acquired_ = true;
}

void VrpLS::release_resource_impl() {
    resource_acquired_ = false;
}

void VrpLS::sync_streams_impl() {
    auto* stream_pool_obj = static_cast<rmm::cuda_stream_pool*>(stream_pool_ptr_);
    stream_pool_obj->get_stream(0).synchronize();
}

// ============================================================================
// Node candaiate management
// ============================================================================

void VrpLS::extract_nodes_to_search_impl() {
    auto* solution_obj = static_cast<Solution*>(solution_ptr_);
    auto* local_search_obj = static_cast<LocalSearch*>(local_search_ptr_);
    cuopt::routing::detail::extract_nodes_to_search(*solution_obj, local_search_obj->move_candidates);
}

void VrpLS::restore_found_nodes_impl() {
    auto* solution_obj = static_cast<Solution*>(solution_ptr_);
    auto* local_search_obj = static_cast<LocalSearch*>(local_search_ptr_);
    local_search_obj->move_candidates.nodes_to_search.restore_found_nodes(*solution_obj);
}

bool VrpLS::sample_nodes_to_search_impl(bool full_set) {
    auto* solution_obj = static_cast<Solution*>(solution_ptr_);
    auto* local_search_obj = static_cast<LocalSearch*>(local_search_ptr_);
    return local_search_obj->move_candidates.nodes_to_search.sample_nodes_to_search(*solution_obj, rng_, full_set);
}

std::vector<std::vector<int>> VrpLS::get_move_candidates_impl() const {
    auto* local_search_obj = static_cast<LocalSearch*>(local_search_ptr_);
    
    // Get move candidates from local search
    auto& move_candidates = local_search_obj->move_candidates;
    const auto& h_nodes_to_search = move_candidates.nodes_to_search.h_nodes_to_search;
    
    std::vector<std::vector<int>> candidates;
    
    // Convert to nested vector format for Python - just return the raw data
    for (const auto& node_info : h_nodes_to_search) {
        std::vector<int> move_data;
        move_data.push_back(node_info.node());
        move_data.push_back(node_info.location());
        move_data.push_back(static_cast<int>(node_info.node_type()));
        candidates.push_back(move_data);
    }
    
    return candidates;
}

void VrpLS::set_move_candidates_impl(const std::vector<std::vector<int>>& modified_candidates) {
    auto* local_search_obj = static_cast<LocalSearch*>(local_search_ptr_);

    // Convert Python data back to NodeInfo format
    std::vector<cuopt::routing::detail::NodeInfo<int>> nodes_to_search;
    for (const auto& candidate : modified_candidates) {
        if (candidate.size() >= 3) {
            int node = candidate[0];
            int location = candidate[1];
            auto node_type = static_cast<cuopt::routing::node_type_t>(candidate[2]);
            
            nodes_to_search.emplace_back(node, location, node_type);
        }
    }
    
    // Update nodes_to_search
    printf("Setting nodes_to_search size to %zu\n", nodes_to_search.size());
    local_search_obj->move_candidates.nodes_to_search.h_nodes_to_search = nodes_to_search;
}

std::vector<std::vector<int>> VrpLS::get_sampled_nodes_impl() const {
    auto* local_search_obj = static_cast<LocalSearch*>(local_search_ptr_);
    
    // Get sampled nodes from local search (the actual data used by perform_vrp_search)
    auto& move_candidates = local_search_obj->move_candidates;
    const auto& h_sampled_nodes = move_candidates.nodes_to_search.h_sampled_nodes;
    
    std::vector<std::vector<int>> sampled_nodes;
    
    // Convert to nested vector format for Python - same format as get_move_candidates
    for (const auto& node_info : h_sampled_nodes) {
        std::vector<int> move_data;
        move_data.push_back(node_info.node());
        move_data.push_back(node_info.location());
        move_data.push_back(static_cast<int>(node_info.node_type()));
        sampled_nodes.push_back(move_data);
    }
    
    return sampled_nodes;
}

std::vector<double> VrpLS::get_weights_impl() const {
    auto* local_search_obj = static_cast<LocalSearch*>(local_search_ptr_);
    
    // Get weights from move_candidates
    auto& weights = local_search_obj->move_candidates.weights;
    
    std::vector<double> weight_values;
    // dim_t::SIZE is the number of dimensions
    for (size_t i = 0; i < static_cast<size_t>(cuopt::routing::detail::dim_t::SIZE); ++i) {
        weight_values.push_back(weights[i]);
    }
    
    return weight_values;
}

void VrpLS::set_weights_impl(const std::vector<double>& weights) {
    auto* local_search_obj = static_cast<LocalSearch*>(local_search_ptr_);
    
    // Set weights in move_candidates
    auto& move_weights = local_search_obj->move_candidates.weights;
    
    size_t size_to_copy = std::min(weights.size(), static_cast<size_t>(cuopt::routing::detail::dim_t::SIZE));
    for (size_t i = 0; i < size_to_copy; ++i) {
        move_weights[i] = weights[i];
    }
}

std::vector<double> VrpLS::get_selection_weights_impl() const {
    auto* local_search_obj = static_cast<LocalSearch*>(local_search_ptr_);
    
    // Get selection_weights from move_candidates
    auto& selection_weights = local_search_obj->move_candidates.selection_weights;
    
    std::vector<double> weight_values;
    // dim_t::SIZE is the number of dimensions
    for (size_t i = 0; i < static_cast<size_t>(cuopt::routing::detail::dim_t::SIZE); ++i) {
        weight_values.push_back(selection_weights[i]);
    }
    
    return weight_values;
}

void VrpLS::set_selection_weights_impl(const std::vector<double>& selection_weights) {
    auto* local_search_obj = static_cast<LocalSearch*>(local_search_ptr_);
    
    // Set selection_weights in move_candidates
    auto& move_selection_weights = local_search_obj->move_candidates.selection_weights;
    
    size_t size_to_copy = std::min(selection_weights.size(), static_cast<size_t>(cuopt::routing::detail::dim_t::SIZE));
    for (size_t i = 0; i < size_to_copy; ++i) {
        move_selection_weights[i] = selection_weights[i];
    }
}


// ============================================================================
// Solution input/output - set initial solution and export results
// ============================================================================

// Validate routes format
void VrpLS::validate_routes(const std::vector<std::vector<int>>& routes, const std::string& context) {
    for (const auto& route : routes) {
        if (route.empty()) {
            throw std::runtime_error("Empty route found in " + context);
        }
        // Allow [0, 0] for empty routes, but other routes must start and end with depot
        if (route.size() == 2 && route[0] == 0 && route[1] == 0) {
            continue;
        }
        if (route.front() != 0 || route.back() != 0) {
            throw std::runtime_error("Routes must start and end with depot (node 0) in " + context);
        }
    }
}

// Process vehicle IDs - handle empty case and validation
std::vector<int> VrpLS::process_vehicle_ids(const std::vector<std::vector<int>>& routes, 
                                            const std::vector<int>& vehicle_ids, 
                                            const std::string& context) {
    std::vector<int> processed_vehicle_ids;
    
    if (vehicle_ids.empty()) {
        // Generate default vehicle IDs (0, 1, 2, ...)
        processed_vehicle_ids.resize(routes.size());
        std::iota(processed_vehicle_ids.begin(), processed_vehicle_ids.end(), 0);
    } else {
        // Validate and use provided vehicle IDs
        if (routes.size() != vehicle_ids.size()) {
            throw std::runtime_error("Number of routes must match number of vehicle IDs in " + context);
        }
        processed_vehicle_ids = vehicle_ids;
    }
    
    return processed_vehicle_ids;
}


// Setup solution with routes and vehicle IDs (includes validation)
void VrpLS::setup_solution_impl(const std::vector<std::vector<int>>& routes, 
                                const std::vector<int>& vehicle_ids) {
    // Validate routes and process vehicle IDs
    validate_routes(routes, "solution setup");
    std::vector<int> processed_vehicle_ids = process_vehicle_ids(routes, vehicle_ids, "solution setup");
    
    auto* problem_obj = static_cast<Problem*>(problem_ptr_);
    
    // Release old solution if exists (prevent memory leak)
    if (solution_ptr_) {
        delete static_cast<Solution*>(solution_ptr_);
        solution_ptr_ = nullptr;
    }
    
    // Create solution with processed vehicle IDs
    auto solution = std::make_unique<Solution>(*problem_obj, 0,
        static_cast<SolutionHandle*>(solution_handle_ptr_), processed_vehicle_ids);
    solution_ptr_ = solution.release();

    // Apply user-provided routes to the created solution
    auto* solution_obj = static_cast<Solution*>(solution_ptr_);
    
    // Convert user routes to cuOpt's internal format
    std::vector<std::pair<int, std::vector<detail::NodeInfo<int>>>> custom_routes;
    
    for (size_t i = 0; i < routes.size(); ++i) {
        const auto& route = routes[i];
        std::vector<detail::NodeInfo<int>> node_infos;

        // Skip depot and add only service nodes
        for (size_t j = 1; j < route.size() - 1; ++j) {
            int node_id = route[j];
            if (node_id > 0) {
                node_infos.push_back(problem_obj->get_node_info_of_node(node_id));
            }
        }

        // Always add route, even if empty (cuOpt will handle empty routes properly)
        custom_routes.emplace_back(i, node_infos);
    }

    // Remove existing routes and add custom routes
    std::vector<int> all_route_ids(solution_obj->get_n_routes());
    std::iota(all_route_ids.begin(), all_route_ids.end(), 0);
    solution_obj->remove_routes(all_route_ids);
    solution_obj->add_routes(custom_routes);
    solution_obj->compute_backward_forward();
    solution_obj->compute_cost();
}

// Get solution routes as vector of vectors
std::vector<std::vector<int>> VrpLS::get_solution_routes_impl() const {
    if (!solution_ptr_) {
        return {};
    }
    
    auto* solution_obj = static_cast<Solution*>(solution_ptr_);
    std::vector<std::vector<int>> all_routes;
    
    for (int i = 0; i < solution_obj->get_n_routes(); ++i) {
        const auto& route = solution_obj->get_route(i);
        int n_nodes = route.n_nodes.value(solution_obj->sol_handle->get_stream());
        
        if (n_nodes <= 1) {
            // Empty route - just depot to depot
            all_routes.push_back({0, 0});
            continue;
        }
        
        // Copy device data to host
        auto node_infos_temp = cuopt::host_copy(route.dimensions.requests.node_info);
        
        std::vector<int> route_nodes;
        for (int j = 0; j <= n_nodes; ++j) {
            if (!node_infos_temp[j].is_break()) {
                route_nodes.push_back(node_infos_temp[j].node());
            }
        }
        all_routes.push_back(route_nodes);
    }
    
    return all_routes;
}

// Get vehicle IDs for each route
std::vector<int> VrpLS::get_solution_vehicle_ids_impl() const {
    if (!solution_ptr_) {
        return {};
    }
    
    auto* solution_obj = static_cast<Solution*>(solution_ptr_);
    std::vector<int> vehicle_ids;
    
    for (int i = 0; i < solution_obj->get_n_routes(); ++i) {
        const auto& route = solution_obj->get_route(i);
        int vehicle_id = route.vehicle_id.value(solution_obj->sol_handle->get_stream());
        vehicle_ids.push_back(vehicle_id);
    }
    
    return vehicle_ids;
}

// ============================================================================
// Query Operations - Get route info, costs, etc.
// ============================================================================


// Get number of routes in the solution
int VrpLS::get_n_routes_impl() const {
    if (!solution_ptr_) return 0;
    
    auto* solution_obj = static_cast<Solution*>(solution_ptr_);
    return solution_obj->get_n_routes();
}

// Get total cost of the solution
double VrpLS::get_cost_impl() const {
    if (!solution_ptr_) return 0.0;
    
    auto* solution_obj = static_cast<Solution*>(solution_ptr_);
    detail::infeasible_cost_t weights(cuopt::routing::detail::default_weights);
    return solution_obj->get_total_cost(weights);
}

// Get number of orders in the problem
int VrpLS::get_num_orders() const {
    return data_model_ptr_ ? data_model_ptr_->get_num_orders() : 0;
}

// ============================================================================
// Public wrapper functions - merged with impl where appropriate
// ============================================================================

void VrpLS::initialize_search(const std::vector<std::vector<int>>& routes, 
                              const std::vector<int>& vehicle_ids) {
    initialize_search_impl(routes, vehicle_ids);
}

void VrpLS::setup_solution(const std::vector<std::vector<int>>& routes, 
                           const std::vector<int>& vehicle_ids) {
    setup_solution_impl(routes, vehicle_ids);
}

std::vector<std::vector<int>> VrpLS::get_solution_routes() const {
    return get_solution_routes_impl();
}

std::vector<int> VrpLS::get_solution_vehicle_ids() const {
    return get_solution_vehicle_ids_impl();
}

void VrpLS::acquire_resource() {
    check_initialized(finalize_called_, "acquire_resource()");
    acquire_resource_impl();
}

void VrpLS::release_resource() {
    release_resource_impl();
}

void VrpLS::sync_streams() {
    check_initialized(finalize_called_, "sync_streams()");
    sync_streams_impl();
}

void VrpLS::extract_nodes_to_search() {
    check_initialized(finalize_called_, "extract_nodes_to_search()");
    extract_nodes_to_search_impl();
}

void VrpLS::restore_found_nodes() {
    check_initialized(finalize_called_, "restore_found_nodes()");
    restore_found_nodes_impl();
}

bool VrpLS::sample_nodes_to_search(bool full_set) {
    check_initialized(finalize_called_, "sample_nodes_to_search()");
    return sample_nodes_to_search_impl(full_set);
}

std::vector<std::vector<int>> VrpLS::get_move_candidates() const {
    check_initialized(finalize_called_, "get_move_candidates()");
    return get_move_candidates_impl();
}

void VrpLS::set_move_candidates(const std::vector<std::vector<int>>& modified_candidates) {
    check_initialized(finalize_called_, "set_move_candidates()");
    set_move_candidates_impl(modified_candidates);
}

std::vector<std::vector<int>> VrpLS::get_sampled_nodes() const {
    check_initialized(finalize_called_, "get_sampled_nodes()");
    return get_sampled_nodes_impl();
}

std::vector<double> VrpLS::get_weights() const {
    check_initialized(finalize_called_, "get_weights()");
    return get_weights_impl();
}

void VrpLS::set_weights(const std::vector<double>& weights) {
    check_initialized(finalize_called_, "set_weights()");
    set_weights_impl(weights);
}

std::vector<double> VrpLS::get_selection_weights() const {
    check_initialized(finalize_called_, "get_selection_weights()");
    return get_selection_weights_impl();
}

void VrpLS::set_selection_weights(const std::vector<double>& selection_weights) {
    check_initialized(finalize_called_, "set_selection_weights()");
    set_selection_weights_impl(selection_weights);
}

void VrpLS::set_routes_to_search() {
    check_initialized(finalize_called_, "set_routes_to_search()");
    auto* solution_obj = static_cast<Solution*>(solution_ptr_);
    solution_obj->set_routes_to_search();
}

void VrpLS::unset_routes_to_search() {
    check_initialized(finalize_called_, "unset_routes_to_search()");
    auto* solution_obj = static_cast<Solution*>(solution_ptr_);
    solution_obj->unset_routes_to_search();
}

void VrpLS::reset_move_candidates() {
    check_initialized(finalize_called_, "reset_move_candidates()");
    auto* local_search_obj = static_cast<LocalSearch*>(local_search_ptr_);
    auto* solution_obj = static_cast<Solution*>(solution_ptr_);
    local_search_obj->move_candidates.reset(solution_obj->sol_handle);
}

bool VrpLS::perform_vrp_search() {
    check_initialized(finalize_called_, "perform_vrp_search()");
    return perform_vrp_search_impl();
}

bool VrpLS::run_two_opt_search() {
    check_initialized(finalize_called_, "run_two_opt_search()");
    return run_two_opt_search_impl();
}

double VrpLS::get_cost() const {
    return get_cost_impl();
}

int VrpLS::get_n_routes() const {
    return get_n_routes_impl();
}

} // namespace pybind
} // namespace routing
} // namespace cuopt

// ============================================================================
// Pybind11 module definition
// ============================================================================

PYBIND11_MODULE(cuopt_pybind, m) {
    m.doc() = "Pybind11 plugin for cuOpt VRP search interface";
    py::class_<cuopt::routing::pybind::VrpLS>(m, "VrpLS")
        .def(py::init<int, int, int>(),
            py::arg("num_locations"),
            py::arg("fleet_size"),
            py::arg("num_orders"))
        .def("add_cost_matrix", &cuopt::routing::pybind::VrpLS::add_cost_matrix,
            py::arg("cost_matrix"),
            py::arg("vehicle_type") = 0)
        .def("add_capacity_dimension", &cuopt::routing::pybind::VrpLS::add_capacity_dimension,
            py::arg("name"),
            py::arg("demand"),
            py::arg("capacity"))
        .def("initialize_search", &cuopt::routing::pybind::VrpLS::initialize_search,
            py::arg("routes"), py::arg("vehicle_ids") = std::vector<int>())
        .def("setup_solution", &cuopt::routing::pybind::VrpLS::setup_solution,
            py::arg("routes"), py::arg("vehicle_ids") = std::vector<int>())
        .def("perform_vrp_search", &cuopt::routing::pybind::VrpLS::perform_vrp_search)
        .def("run_two_opt_search", &cuopt::routing::pybind::VrpLS::run_two_opt_search)
        .def("acquire_resource", &cuopt::routing::pybind::VrpLS::acquire_resource)
        .def("release_resource", &cuopt::routing::pybind::VrpLS::release_resource)
        .def("sync_streams", &cuopt::routing::pybind::VrpLS::sync_streams)
        .def("extract_nodes_to_search", &cuopt::routing::pybind::VrpLS::extract_nodes_to_search)
        .def("restore_found_nodes", &cuopt::routing::pybind::VrpLS::restore_found_nodes)
        .def("sample_nodes_to_search", &cuopt::routing::pybind::VrpLS::sample_nodes_to_search,
            py::arg("full_set") = false)
        .def("get_move_candidates", &cuopt::routing::pybind::VrpLS::get_move_candidates)
        .def("set_move_candidates", &cuopt::routing::pybind::VrpLS::set_move_candidates,
            py::arg("modified_candidates"))
        .def("get_sampled_nodes", &cuopt::routing::pybind::VrpLS::get_sampled_nodes)
        .def("get_weights", &cuopt::routing::pybind::VrpLS::get_weights)
        .def("set_weights", &cuopt::routing::pybind::VrpLS::set_weights,
            py::arg("weights"))
        .def("get_selection_weights", &cuopt::routing::pybind::VrpLS::get_selection_weights)
        .def("set_selection_weights", &cuopt::routing::pybind::VrpLS::set_selection_weights,
            py::arg("selection_weights"))
        .def("set_routes_to_search", &cuopt::routing::pybind::VrpLS::set_routes_to_search)
        .def("unset_routes_to_search", &cuopt::routing::pybind::VrpLS::unset_routes_to_search)
        .def("reset_move_candidates", &cuopt::routing::pybind::VrpLS::reset_move_candidates)
        .def("get_solution_routes", &cuopt::routing::pybind::VrpLS::get_solution_routes)
        .def("get_solution_vehicle_ids", &cuopt::routing::pybind::VrpLS::get_solution_vehicle_ids)
        .def("get_cost", &cuopt::routing::pybind::VrpLS::get_cost)
        .def("get_n_routes", &cuopt::routing::pybind::VrpLS::get_n_routes)
        .def("get_num_orders", &cuopt::routing::pybind::VrpLS::get_num_orders);

}