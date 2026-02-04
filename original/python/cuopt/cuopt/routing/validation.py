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


def validate_matrix(mat, mat_name, num_locations):
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(mat_name + " is expected to be a square matrix")

    if num_locations != mat.shape[0]:
        raise ValueError(
            "Number of locations doesn't match"
            + " number of locations in matrix"
        )

    if mat.isnull().any(axis=1).any():
        raise ValueError(mat_name + " cannot have NULL values")

    validate_positive_df(mat, mat_name)


def validate_positive_df(df, df_name):
    if (df < 0).any(axis=1).any():
        raise ValueError(
            "All values in "
            + df_name
            + " must be greater than or equal to zero"
        )


def validate_positive(var, var_name):
    if hasattr(var, "__len__"):
        if (var <= 0).any():
            raise ValueError(
                "All values in " + var_name + " must be greater than zero"
            )
    elif var <= 0:
        raise ValueError(var_name + " must be greater than zero")


def validate_non_negative(var, var_name):
    if hasattr(var, "__len__"):
        if (var < 0).any():
            raise ValueError(
                "All values in "
                + var_name
                + "  must be greater than or equal to zero"
            )
    elif var < 0:
        raise ValueError(var_name + "  must be greater than or equal to zero")


def validate_size(var_1, var_1_str, var_2, var_2_str):
    if hasattr(var_1, "__len__"):
        var_1 = len(var_1)
    if hasattr(var_2, "__len__"):
        var_2 = len(var_2)
    if var_1 != var_2:
        msg = var_1_str + " size doesn't match " + var_2_str
        raise ValueError(msg)


def validate_time_windows(earliest, latest, size, size_str):
    validate_size(earliest, "earliest times", size, size_str)
    validate_non_negative(earliest, "earliest times")
    validate_size(latest, "latest times", size, size_str)
    validate_non_negative(latest, "latest times")
    if not (earliest <= latest).all():
        raise ValueError("All earliest times must be lesser than latest times")


def validate_range(var, var_name, min, max):
    if isinstance(var, cudf.Series):
        if (var < min).any():
            raise ValueError(
                "All values in "
                + var_name
                + " must be greater than or equal to "
                + str(min)
            )
        if (var > max).any():
            raise ValueError(
                "All values in "
                + var_name
                + " must be less than or equal to "
                + str(max)
            )
    else:
        if var < min:
            raise ValueError(
                var_name + " must be greater than or equal to " + str(min)
            )
        if var > max:
            raise ValueError(
                var_name + " must be less than or equal to " + str(max)
            )
