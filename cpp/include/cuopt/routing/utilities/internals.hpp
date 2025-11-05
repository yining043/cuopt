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
    REWARD,
    LOCAL_SEARCH_START,
    BEFORE_CYCLE_FINDER,
    AFTER_CYCLE_FINDER
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
        std::vector<int>* selection_mask_out,
        int iter
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
        float solution_cost,
        int iter
    ) = 0;
    
    callback_type_t get_type() const override {
        return callback_type_t::REWARD;
    }
};

// Local search start callback - observes state before local search begins
class local_search_start_callback_t : public base_routing_callback_t {
public:
    // Observe search state before local search begins
    // 
    // @param solution_flat              Current solution (flat array by route)
    // @param num_routes                 Number of routes in current solution
    // @param solution_cost              Current objective cost value
    // @param weights                    Weights used for cost computation
    // @param selection_weights          Weights used for move selection
    // @param should_all_nodes_be_served Whether all nodes should be served
    virtual void on_local_search_start(
        const std::vector<int>* solution_flat,
        int num_routes,
        double solution_cost,
        const std::vector<double>* weights,
        const std::vector<double>* selection_weights,
        bool should_all_nodes_be_served
    ) = 0;
    
    callback_type_t get_type() const override {
        return callback_type_t::LOCAL_SEARCH_START;
    }
};

// Before cycle finder callback - observes state before cycle finder execution
class before_cycle_finder_callback_t : public base_routing_callback_t {
public:
    // Observe search state before cycle finder execution
    // 
    // @param solution_flat  Current solution (flat array by route)
    // @param num_routes     Number of routes in current solution
    // @param solution_cost  Current objective cost value
    // @param iter           Current iteration number
    virtual void on_before_cycle_finder(
        const std::vector<int>* solution_flat,
        int num_routes,
        float solution_cost,
        int iter
    ) = 0;
    
    callback_type_t get_type() const override {
        return callback_type_t::BEFORE_CYCLE_FINDER;
    }
};

// After cycle finder callback - observes state after cycle finder execution
class after_cycle_finder_callback_t : public base_routing_callback_t {
public:
    // Observe search state after cycle finder execution
    // 
    // @param solution_flat  Current solution (flat array by route)
    // @param num_routes     Number of routes in current solution
    // @param solution_cost  Current objective cost value
    // @param iter           Current iteration number
    // @param improved       Whether improvement was found
    virtual void on_after_cycle_finder(
        const std::vector<int>* solution_flat,
        int num_routes,
        float solution_cost,
        int iter,
        bool improved
    ) = 0;
    
    callback_type_t get_type() const override {
        return callback_type_t::AFTER_CYCLE_FINDER;
    }
};

}  // namespace callbacks
}  // namespace routing
}  // namespace cuopt

