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

#include <routing/solver.hpp>

#include <vector>

namespace cuopt {
namespace routing {
namespace test {

template <typename i_t, typename f_t>
void check_route(data_model_view_t<i_t, f_t> const& data_model,
                 host_assignment_t<i_t> const& h_routing_solution);

template <typename i_t, typename f_t>
void check_route(data_model_view_t<i_t, f_t> const& data_model,
                 assignment_t<i_t> const& routing_solution);

}  // namespace test
}  // namespace routing
}  // namespace cuopt
