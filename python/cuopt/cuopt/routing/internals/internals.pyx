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
    
    cdef cppclass default_observation_callback_t(Callback):
        void get_observation_and_sample(const vector[vector[int]]* routes, const vector[int]* node_ids_to_search, vector[int]* sampled_out, float objective_value, int n_routes) except +
        PyObject* pyCallbackClass


cdef class PyCallback:
    pass


cdef class ObservationCallback(PyCallback):
    """
    Callback to receive observations and control node sampling during routing search
    
    The callback receives the current routing solution state and a list of candidate
    nodes for local search. It should return INDICES (not node IDs) of nodes to sample.
    
    Examples
    --------
    >>> from cuopt.routing import ObservationCallback
    >>> import random
    >>> 
    >>> class MyCallback(ObservationCallback):
    ...     def get_observation_and_sample(self, routes_2d, node_ids_to_search, 
    ...                                    objective_value, n_routes):
    ...         print(f"Cost: {objective_value:.2f}, Routes: {n_routes}")
    ...         # Return INDICES, not node IDs
    ...         sample_size = min(40, len(node_ids_to_search))
    ...         return random.sample(range(len(node_ids_to_search)), sample_size)
    """
    
    cdef default_observation_callback_t native_callback
    
    def __init__(self):
        self.native_callback.pyCallbackClass = <PyObject*><void*>self
    
    def get_native_callback(self):
        return <uintptr_t>&(self.native_callback)
    
    def _cpp_callback_wrapper(self, unsigned long long routes_ptr, unsigned long long nodes_ptr, 
                              float objective_value, int n_routes):
        """Internal wrapper that bridges C++ to Python"""
        cdef const vector[vector[int]]* routes = <const vector[vector[int]]*>routes_ptr
        cdef const vector[int]* nodes = <const vector[int]*>nodes_ptr
        
        py_routes = routes[0]
        py_nodes = nodes[0]
        
        sampled_indices = self.get_observation_and_sample(py_routes, py_nodes, 
                                                          objective_value, n_routes)
        
        return sampled_indices
    
    def get_observation_and_sample(self, routes_2d, node_ids_to_search, 
                                   objective_value, n_routes):
        """
        Receive observation and return indices of nodes to sample
        
        Override this method in your subclass to implement custom sampling logic.
        
        Parameters
        ----------
        routes_2d : list of list of int
            Current routing solution, each inner list is a route (node IDs)
        node_ids_to_search : list of int
            Candidate node IDs available for sampling
        objective_value : float
            Current objective cost value
        n_routes : int
            Number of routes in the solution
            
        Returns
        -------
        list of int
            Indices into node_ids_to_search array (NOT node IDs themselves!)
            Example: if you want to sample the first and third nodes from
            node_ids_to_search, return [0, 2]
        """
        import random
        
        n_available = len(node_ids_to_search)
        if n_available == 0:
            return []
        
        # Determine sample size
        if n_available < 40:
            sample_size = n_available
        elif n_available < 80:
            sample_size = n_available // 2
        else:
            sample_size = 40
        
        # Return random indices into node_ids_to_search
        return random.sample(range(n_available), sample_size)

