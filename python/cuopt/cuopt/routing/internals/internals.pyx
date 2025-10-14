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
    
    cdef cppclass default_customize_nodes_callback_t(Callback):
        void customize_nodes_to_search(const vector[vector[int]]* routes_2d, const vector[int]* candidate_node_ids, float solution_cost, int num_routes, vector[int]* sampled_indices_out) except +
        PyObject* pyCallbackClass
    
    cdef cppclass default_reward_callback_t(Callback):
        void receive_reward(int improvement_found, float solution_cost) except +
        PyObject* pyCallbackClass


cdef class PyCallback:
    pass


cdef class CustomizeNodesCallback(PyCallback):
    """
    Callback for customizing node sampling in routing search
    
    Receives current solution state and returns which nodes to sample for local search.
    This combines observation and action selection into a single cohesive callback.
    
    Examples
    --------
    >>> from cuopt.routing import CustomizeNodesCallback
    >>> import random
    >>> 
    >>> class MyCustomCallback(CustomizeNodesCallback):
    ...     def customize_nodes_to_search(self, routes_2d, candidate_node_ids, 
    ...                                   solution_cost, num_routes):
    ...         # Adaptive sampling based on problem size
    ...         num_candidates = len(candidate_node_ids)
    ...         sample_size = min(40, num_candidates)
    ...         return random.sample(range(num_candidates), sample_size)
    """
    
    cdef default_customize_nodes_callback_t native_callback
    
    def __init__(self):
        self.native_callback.pyCallbackClass = <PyObject*><void*>self
    
    def get_native_callback(self):
        return <uintptr_t>&(self.native_callback)
    
    def _cpp_customize_nodes_to_search(self, unsigned long long routes_ptr, unsigned long long nodes_ptr,
                                       float solution_cost, int num_routes):
        cdef const vector[vector[int]]* routes_2d = <const vector[vector[int]]*>routes_ptr
        cdef const vector[int]* candidate_node_ids = <const vector[int]*>nodes_ptr
        
        py_routes_2d = routes_2d[0]
        py_candidate_node_ids = candidate_node_ids[0]
        
        return self.customize_nodes_to_search(py_routes_2d, py_candidate_node_ids, 
                                              solution_cost, num_routes)
    
    def customize_nodes_to_search(self, routes_2d, candidate_node_ids, solution_cost, num_routes):
        """
        Customize which nodes to sample for local search
        
        Override this method to implement custom node selection logic based on
        the current solution state.
        
        Parameters
        ----------
        routes_2d : list of list of int
            Current routing solution (2D list where each inner list represents a route)
        candidate_node_ids : list of int
            Available candidate node IDs that can be sampled for local search
        solution_cost : float
            Current solution objective cost
        num_routes : int
            Total number of routes in current solution
            
        Returns
        -------
        list of int
            Sampled indices into candidate_node_ids array (NOT actual node IDs).
            Indices must be in range [0, N-1] where N is len(candidate_node_ids).
            Example: to sample 1st and 3rd candidates, return [0, 2]
        """
        import random
        
        num_candidates = len(candidate_node_ids)
        if num_candidates == 0:
            return []
        
        if num_candidates < 40:
            sample_size = num_candidates
        elif num_candidates < 80:
            sample_size = num_candidates // 2
        else:
            sample_size = 40
        
        return random.sample(range(num_candidates), sample_size)


cdef class RewardCallback(PyCallback):
    """
    Reward callback for routing search
    
    Receives feedback after each local search iteration. This follows the reinforcement
    learning paradigm where rewards signal the outcome of taking actions.
    
    Examples
    --------
    >>> from cuopt.routing import RewardCallback
    >>> 
    >>> class MyRewardCallback(RewardCallback):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.improvement_count = 0
    ...     
    ...     def receive_reward(self, improvement_found, solution_cost):
    ...         if improvement_found:
    ...             self.improvement_count += 1
    ...             print(f"✓ Improvement #{self.improvement_count}: cost={solution_cost:.2f}")
    """
    
    cdef default_reward_callback_t native_callback
    
    def __init__(self):
        self.native_callback.pyCallbackClass = <PyObject*><void*>self
    
    def get_native_callback(self):
        return <uintptr_t>&(self.native_callback)
    
    def _cpp_receive_reward(self, int improvement_found, float solution_cost):
        self.receive_reward(bool(improvement_found), solution_cost)
    
    def receive_reward(self, improvement_found, solution_cost):
        """
        Receive reward signal from search iteration
        
        Override this method to implement custom reward processing logic.
        
        Parameters
        ----------
        improvement_found : bool
            Whether an improving move was discovered in this iteration
        solution_cost : float
            Current solution objective cost
        """
        pass

