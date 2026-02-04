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

import numpy as np


def validate_cost_matrix(
    cost_matrix, is_travel_time=False, updating=False, comparison_matrix=None
):
    if (updating or is_travel_time) and (len(comparison_matrix) == 0):
        return (
            False,
            "If updating a matrix, a matrix must already be set. If adding a travel time matrix, a primary matrix must already be set",  # noqa
        )
    shape = None
    for vehicle_type, matrix in cost_matrix.items():
        row_lengths = [len(x) for x in matrix]
        if not len(set(row_lengths)) == 1:
            return (
                False,
                "All rows in the cost matrix must be of the same length",
            )

        if len(matrix) != len(matrix[0]):
            return (False, "Cost matrix must be a square matrix")

        np_cost_matrix = np.array(matrix)
        min_cost_matrix_value = np_cost_matrix.min()
        if min_cost_matrix_value < 0:
            return (False, "All values in cost matrix must be >= 0")

        if shape is None:
            shape = np_cost_matrix.shape
        elif shape != np_cost_matrix.shape:
            return (
                False,
                "Matrices for all vehicle types must be the same shape",
            )

        if comparison_matrix is not None:
            if (
                vehicle_type not in comparison_matrix
                or np_cost_matrix.shape
                != comparison_matrix[vehicle_type].shape
            ):
                return (
                    False,
                    "When updating a cost matrix or setting a travel time matrix, the shape of the input must match the shape of the primary matrix",  # noqa
                )
    if comparison_matrix is not None:
        return (True, "Valid Matrix")

    return (True, "Valid Cost Matrix")
