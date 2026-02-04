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


def validate_solver_config(
    time_limit,
    objectives,
    config_file,
    verbose_mode,
    error_logging,
    updating=False,
    comparison_time_limit=None,
):
    if updating and comparison_time_limit is None:
        return (
            False,
            "No solver config to update. The set_solver_config endpoint must be used before update is available",  # noqa
        )

    if (time_limit is not None) and (time_limit <= 0):
        return (False, "SolverSettings time limit must be greater than 0")

    if config_file is not None and len(config_file) == 0:
        return (
            False,
            "File path to save configuration should be valid and not empty",
        )

    return (True, "Valid SolverSettings Config")
