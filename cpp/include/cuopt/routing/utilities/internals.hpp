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
    // @param solution_flat          Current solution (flat array by route: [route0_dummies, route0_nodes, route1_dummies, ...])
    // @param num_routes             Number of routes in current solution
    // @param solution_cost          Current objective cost value
    // @param candidate_mask         Mask indexed by node_id (candidate_mask[node_id]=1 means node_id is candidate)
    // @param selection_mask_out     OUTPUT - Selection mask indexed by node_id (selection_mask[node_id]=1 means select node_id)
    //                               Same length as candidate_mask
    virtual void customize_nodes_to_search(
        const std::vector<int>* solution_flat,
        int num_routes,
        float solution_cost,
        const std::vector<int>* candidate_mask,
        std::vector<int>* selection_mask_out
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

