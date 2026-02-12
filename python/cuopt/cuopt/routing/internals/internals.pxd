# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libc.stdint cimport uintptr_t

cdef extern from "Python.h":
    cdef cppclass PyObject

cdef extern from "cuopt/routing/utilities/callbacks_implems.hpp" namespace "cuopt::routing::callbacks":  # noqa
    cdef cppclass default_customize_early_stop_callback_t[int, float]:
        PyObject* pyCallbackClass

cdef class CustomizeEarlyStopCallback:
    cdef default_customize_early_stop_callback_t[int, float] native_callback
