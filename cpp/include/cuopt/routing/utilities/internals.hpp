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
  CUSTOMIZE_EARLY_STOP
};

// Base routing callback class
class base_routing_callback_t {
public:
  virtual ~base_routing_callback_t() = default;
  virtual callback_type_t get_type() const = 0;
};

// Customize early stop callback - decides whether to stop local search early
template <typename i_t, typename f_t>
class customize_early_stop_callback_t : public base_routing_callback_t {
public:
  virtual ~customize_early_stop_callback_t() = default;

  // Customize early stop decision based on current search state
  //
  // @param solution_flat  Current solution (flat array by route)
  // @param num_routes     Number of routes in current solution
  // @param objective      Current objective cost value
  // @param iteration      Current iteration number
  // @param early_stop_out OUTPUT - Set to true to stop local search early
  // @param phase          0 = regular LS step, 1 = cycle_finder step (default 0 for backwards compat)
  virtual void customize_early_stop(
    const std::vector<i_t>* solution_flat,
    i_t num_routes,
    f_t objective,
    i_t iteration,
    bool* early_stop_out,
    i_t phase = 0
  ) = 0;

  callback_type_t get_type() const override {
    return callback_type_t::CUSTOMIZE_EARLY_STOP;
  }
};

}  // namespace callbacks
}  // namespace routing
}  // namespace cuopt

