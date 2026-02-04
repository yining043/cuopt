/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

struct retail_params_t {
  retail_params_t& set_drop_return_trip()
  {
    drop_return_trip = true;
    return *this;
  }

  retail_params_t& set_multi_capacity()
  {
    multi_capacity = true;
    return *this;
  }

  retail_params_t& set_vehicle_tw()
  {
    vehicle_tw = true;
    return *this;
  }

  retail_params_t& set_vehicle_lower_bound(int val)
  {
    vehicle_lower_bound = val;
    return *this;
  }

  retail_params_t& set_pickup()
  {
    pickup = true;
    return *this;
  }

  retail_params_t& set_vehicle_breaks()
  {
    vehicle_breaks = true;
    return *this;
  }

  retail_params_t& set_vehicle_max_costs()
  {
    vehicle_max_costs = true;
    return *this;
  }

  retail_params_t& set_vehicle_max_times()
  {
    vehicle_max_times = true;
    return *this;
  }

  retail_params_t& set_vehicle_fixed_costs()
  {
    vehicle_fixed_costs = true;
    return *this;
  }

  bool drop_return_trip{false};
  bool multi_capacity{false};
  bool vehicle_tw{false};
  int vehicle_lower_bound{0};
  bool pickup{false};
  bool vehicle_breaks{false};
  bool vehicle_max_costs{false};
  bool vehicle_max_times{false};
  bool vehicle_fixed_costs{false};
};
