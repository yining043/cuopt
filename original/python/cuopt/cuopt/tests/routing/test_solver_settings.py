# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cudf

from cuopt import routing
from cuopt.routing import utils

filename = utils.RAPIDS_DATASET_ROOT_DIR + "/solomon/In/r107.txt"


def test_min_vehicles():
    min_vehicles = 10
    d = utils.create_data_model(filename, run_nodes=10)
    d.set_min_vehicles(min_vehicles)

    s = routing.SolverSettings()
    s.set_time_limit(4)
    routing_solution = routing.Solve(d, s)
    ret_vehicle_num = d.get_min_vehicles()

    assert ret_vehicle_num == min_vehicles
    assert routing_solution.get_vehicle_count() >= min_vehicles
    assert routing_solution.get_status() == 0


def test_max_distance():
    d = utils.create_data_model(filename, run_nodes=10)
    max_distance = cudf.Series([250.0] * d.get_fleet_size())
    d.set_vehicle_max_costs(max_distance)
    s = routing.SolverSettings()
    s.set_time_limit(4)
    routing_solution = routing.Solve(d, s)
    routes = routing_solution.get_route()
    trucks = routes["truck_id"].unique()
    for i in range(0, len(trucks)):
        truck_route = routes[routes["truck_id"] == trucks.iloc[i]]
        assert truck_route["arrival_stamp"].iloc[-1] < max_distance[0]


def test_dump_results():
    d = utils.create_data_model(filename, run_nodes=10)
    s = routing.SolverSettings()
    file_path = "best_results.txt"
    interval = 0.001
    s.dump_best_results(file_path, interval)
    s.set_time_limit(4)
    routing_solution = routing.Solve(d, s)
    assert routing_solution.get_status() == 0
    ret_file_path = s.get_best_results_file_path()
    ret_interval = s.get_best_results_interval()
    assert file_path == ret_file_path
    assert interval == ret_interval
