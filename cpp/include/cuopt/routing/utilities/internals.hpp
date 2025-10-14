/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#pragma once

#include <vector>

namespace cuopt {
namespace routing {
namespace callbacks {

// Base Callback class
class Callback {
public:
    virtual ~Callback() {}
};

// Callback type enumeration
enum class callback_type_t {
    OBSERVATION
};

// Base routing callback
class base_routing_callback_t : public Callback {
public:
    virtual callback_type_t get_type() const = 0;
};

// Observation callback for routing search
// The callback receives current routing solution and node candidates,
// and returns indices of nodes to sample for local search
class observation_callback_t : public base_routing_callback_t {
public:
    // Get observation and sample nodes for search
    // @param routes: Current routing solution (2D array of node IDs)
    // @param node_ids_to_search: Available node IDs for sampling
    // @param sampled_indices_out: OUTPUT - Indices into node_ids_to_search (not node IDs!)
    // @param objective_value: Current objective cost
    // @param n_routes: Number of routes
    virtual void get_observation_and_sample(
        const std::vector<std::vector<int>>* routes,
        const std::vector<int>* node_ids_to_search,
        std::vector<int>* sampled_indices_out,
        float objective_value,
        int n_routes
    ) = 0;
    
    callback_type_t get_type() const override {
        return callback_type_t::OBSERVATION;
    }
};

}  // namespace callbacks
}  // namespace routing
}  // namespace cuopt

