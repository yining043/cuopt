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
    CUSTOMIZE_NODES,
    REWARD
};

// Base routing callback
class base_routing_callback_t : public Callback {
public:
    virtual callback_type_t get_type() const = 0;
};

// Customize nodes callback - customizes node sampling for local search
class customize_nodes_callback_t : public base_routing_callback_t {
public:
    // Customize which nodes to sample based on current search state
    // 
    // @param routes_2d              Current routing solution (2D vector, each inner vector is a route)
    // @param candidate_node_ids     Available node IDs for potential sampling
    // @param solution_cost          Current objective cost value
    // @param num_routes             Number of routes in current solution
    // @param sampled_indices_out    OUTPUT - Sampled indices into candidate_node_ids array
    //                               Note: Must return indices (0 to N-1), not actual node IDs
    virtual void customize_nodes_to_search(
        const std::vector<std::vector<int>>* routes_2d,
        const std::vector<int>* candidate_node_ids,
        float solution_cost,
        int num_routes,
        std::vector<int>* sampled_indices_out
    ) = 0;
    
    callback_type_t get_type() const override {
        return callback_type_t::CUSTOMIZE_NODES;
    }
};

// Reward callback - receives feedback after search iteration
class reward_callback_t : public base_routing_callback_t {
public:
    // Receive reward signal from search iteration
    // 
    // @param improvement_found  Whether an improving move was discovered
    // @param solution_cost      Current solution objective cost
    virtual void receive_reward(
        bool improvement_found,
        float solution_cost
    ) = 0;
    
    callback_type_t get_type() const override {
        return callback_type_t::REWARD;
    }
};

}  // namespace callbacks
}  // namespace routing
}  // namespace cuopt

