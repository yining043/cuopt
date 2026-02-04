# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
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

import pandas as pd


def gen_cross_point_benchmark(fpath):
    or_pdf = pd.read_csv(fpath)

    data = {
        "DATA": [],
        "OR_COST": [],
        "OR_VEHICLE": [],
        "OR_DURATION": [],
        "CROSS_COST": [],
        "CROSS_VEHICLE": [],
        "CROSS_DURATION": [],
        "FINAL_COST": [],
        "FINAL_VEHICLE": [],
        "FINAL_DURATION": [],
    }
    for index, row in or_pdf.iterrows():
        f = row["File Name"]
        or_cost = row["Total Distance"]
        or_vehicle = row["Vehicles"]
        or_runtime = row["SolverSettings Run Time"]
        cuopt_pdf = pd.read_csv("inter/" + f + ".csv")

        cross_cost = -1
        cross_vehicle = -1
        cross_duration = -1

        for r_index, r_row in cuopt_pdf.iterrows():
            duration = row["SolverSettings Run Time"]
            vehicle = row["Vehicles"]
            cost = row["Total Distance"]
            if vehicle == or_vehicle and cost < or_cost:
                cross_cost = cost
                cross_vehicle = vehicle
                cross_duration = duration

                break

        last_row = cuopt_pdf.iloc[-1]

        final_duration = float(last_row["SolverSettings Run Time"])
        final_vehicle = int(last_row["Vehicles"])
        final_cost = float(last_row["Total Distance"])

        data["DATA"].append(f)
        data["OR_COST"].append(or_cost)
        data["OR_VEHICLE"].append(or_vehicle)
        data["OR_DURATION"].append(or_runtime)
        data["CROSS_COST"].append(cross_cost)
        data["CROSS_VEHICLE"].append(cross_vehicle)
        data["CROSS_DURATION"].append(cross_duration)
        data["FINAL_COST"].append(final_cost)
        data["FINAL_VEHICLE"].append(final_vehicle)
        data["FINAL_DURATION"].append(final_duration)
        pdf = pd.DataFrame(data)
        pdf["SPEEDUP"] = pdf["OR_DURATION"] / pdf["FINAL_DURATION"]
        pdf.to_excel(fpath + "_bench.xlsx")


or_bench_files = ["pdp_1000_.csv"]

for fname in or_bench_files:
    gen_cross_point_benchmark(fname)
