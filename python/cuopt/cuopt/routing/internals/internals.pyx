# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
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
from libcpp.vector cimport vector


cdef extern from "Python.h":
    cdef cppclass PyObject


cdef extern from "cuopt/routing/utilities/callbacks_implems.hpp" namespace "cuopt::routing::callbacks":
    cdef cppclass Callback:
        pass
    
    cdef cppclass default_get_solution_callback_t(Callback):
        void get_solution(const vector[vector[int]]* routes, float objective_value, int n_routes) except +
        PyObject* pyCallbackClass


cdef class PyCallback:
    pass


cdef class GetSolutionCallback(PyCallback):
    """
    Callback to receive solutions during routing search
    
    Examples
    --------
    >>> from cuopt.routing import GetSolutionCallback
    >>> 
    >>> class MyCallback(GetSolutionCallback):
    ...     def get_solution(self, routes_2d, objective_value, n_routes):
    ...         print(f"Cost: {objective_value:.2f}, Routes: {n_routes}")
    """
    
    cdef default_get_solution_callback_t native_callback
    
    def __init__(self):
        self.native_callback.pyCallbackClass = <PyObject*><void*>self
    
    def get_native_callback(self):
        return <uintptr_t>&(self.native_callback)
    
    def _cpp_callback_wrapper(self, unsigned long long routes_ptr, float objective_value, int n_routes):
        """Internal wrapper"""
        cdef const vector[vector[int]]* routes = <const vector[vector[int]]*>routes_ptr
        py_routes = routes[0]
        self.get_solution(py_routes, objective_value, n_routes)
    
    def get_solution(self, routes_2d, objective_value, n_routes):
        """
        Called when run_fast_search executes - override this in your subclass
        
        Parameters
        ----------
        routes_2d : list of lists
            Each element is a route (list of node IDs)
        objective_value : float
            Objective cost
        n_routes : int
            Number of routes
        """
        pass

