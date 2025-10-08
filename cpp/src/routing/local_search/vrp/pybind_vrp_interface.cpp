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

#include <cuopt/routing/pybind/pybind_vrp_interface.hpp>

#include <cuda_runtime.h>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <cuopt/routing/solve.hpp>

namespace cuopt {
namespace routing {
namespace pybind {

// VrpLS implementation - following cuOpt's initialization pattern
VrpLS::VrpLS(int num_locations, int fleet_size, int num_orders)
    : handle_(std::make_unique<raft::handle_t>()),
      data_model_ptr_(std::make_unique<data_model_view_t<int, float>>(
          handle_.get(), num_locations, fleet_size, num_orders)),
      solver_settings_ptr_(std::make_unique<solver_settings_t<int, float>>()),
      problem_ptr_(nullptr),
      stream_pool_ptr_(nullptr),
      solution_handle_ptr_(nullptr),
      solution_ptr_(nullptr),
      local_search_ptr_(nullptr),
      rng_(std::random_device{}()),
      finalize_called_(false) {
    // Initialize CUDA device context
    int device_count;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        throw std::runtime_error("No CUDA devices found");
        exit(1);
    }
}

// Destructor is implemented in the .cu file

void VrpLS::add_cost_matrix(const std::vector<std::vector<float>>& cost_matrix, uint8_t vehicle_type) {
    if (!handle_) {
        throw std::runtime_error("Handle not initialized");
    }

    // Convert to device memory
    auto stream = handle_->get_stream();
    auto n = cost_matrix.size();
    
    rmm::device_uvector<float> device_matrix(n * n, stream);
    std::vector<float> flat_matrix(n * n);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            flat_matrix[i * n + j] = cost_matrix[i][j];
        }
    }
    
    raft::update_device(device_matrix.data(), flat_matrix.data(), n * n, stream);
    
    device_cost_matrices_.push_back(std::move(device_matrix));
    cost_matrix_ptrs_.push_back(device_cost_matrices_.back().data());
    
    // Add to data model
    data_model_ptr_->add_cost_matrix(cost_matrix_ptrs_.back(), vehicle_type);
}

void VrpLS::add_capacity_dimension(const std::string& name, const std::vector<int>& demand, const std::vector<int>& capacity) {
    if (!handle_) {
        throw std::runtime_error("Handle not initialized");
    }

    auto stream = handle_->get_stream();
    
    // Convert demands to device memory
    rmm::device_uvector<int> device_demand(demand.size(), stream);
    raft::update_device(device_demand.data(), demand.data(), demand.size(), stream);
    
    // Convert capacities to device memory
    rmm::device_uvector<int> device_capacity(capacity.size(), stream);
    raft::update_device(device_capacity.data(), capacity.data(), capacity.size(), stream);
    
    device_demands_.push_back(std::move(device_demand));
    device_capacities_.push_back(std::move(device_capacity));
    
    // Add to data model
    data_model_ptr_->add_capacity_dimension(name, device_demands_.back().data(), device_capacities_.back().data());
}


} // namespace pybind
} // namespace routing
} // namespace cuopt

// Pybind11 bindings are in the .cu file for CUDA compilation