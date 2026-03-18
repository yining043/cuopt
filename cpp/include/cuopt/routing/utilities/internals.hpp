/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Base callback type enum
enum class callback_type_t {
  CUSTOMIZE_NODES
};

// Base routing callback class
class base_routing_callback_t {
public:
  virtual ~base_routing_callback_t() = default;
  virtual callback_type_t get_type() const = 0;
};

// Customize nodes callback - customizes node sampling for local search
template <typename i_t, typename f_t>
class customize_nodes_callback_t : public base_routing_callback_t {
public:
  virtual ~customize_nodes_callback_t() = default;
  
  // Customize which nodes to sample based on current search state
  // 
  // @param solution_flat          Current solution (flat array by route)
  // @param num_routes             Number of routes in current solution
  // @param solution_cost          Current objective cost value
  // @param trail_masks_flat       K executed_anchor masks flattened: [K * N_nodes_w_dummy]
  // @param num_trails             Number of trails K
  // @param selection_mask_out     OUTPUT - Selection mask indexed by node_id
  // @param iteration              Current iteration number
  virtual void customize_nodes_to_search(
    const std::vector<i_t>* solution_flat,
    i_t num_routes,
    f_t solution_cost,
    const std::vector<i_t>* trail_masks_flat,
    i_t num_trails,
    std::vector<i_t>* selection_mask_out,
    i_t iteration
  ) = 0;
  
  callback_type_t get_type() const override {
    return callback_type_t::CUSTOMIZE_NODES;
  }
};

}  // namespace callbacks
}  // namespace routing
}  // namespace cuopt

