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
from enum import IntEnum


cdef extern from "Python.h":
    cdef cppclass PyObject


cdef extern from "cuopt/routing/utilities/callbacks_implems.hpp" namespace "cuopt::routing::callbacks":
    cdef cppclass Callback:
        pass
    
    cdef cppclass default_customize_nodes_callback_t(Callback):
        void customize_nodes_to_search(const vector[int]* solution_flat, const vector[int]* candidate_mask, float solution_cost, int num_routes, vector[int]* selection_mask_out, int iter) except +
        PyObject* pyCallbackClass
    
    cdef cppclass default_reward_callback_t(Callback):
        void receive_reward(int improvement_found, float solution_cost, int iter) except +
        PyObject* pyCallbackClass
    
    cdef cppclass default_local_search_start_callback_t(Callback):
        void on_local_search_start(const vector[int]* solution_flat, int num_routes, double solution_cost, int should_all_nodes_be_served) except +
        PyObject* pyCallbackClass
    
    cdef cppclass default_before_cycle_finder_callback_t(Callback):
        void on_before_cycle_finder(const vector[int]* solution_flat, int num_routes, float solution_cost, int iter) except +
        PyObject* pyCallbackClass
    
    cdef cppclass default_after_cycle_finder_callback_t(Callback):
        void on_after_cycle_finder(const vector[int]* solution_flat, int num_routes, float solution_cost, int iter, int improved) except +
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
    ...     def customize_nodes_to_search(self, solution_flat, candidate_mask, 
    ...                                   solution_cost, num_routes):
    ...         # Extract candidates and sample
    ...         candidates = [i for i in range(len(candidate_mask)) if candidate_mask[i]]
    ...         sampled = random.sample(candidates, min(40, len(candidates)))
    ...         # Return selection mask
    ...         selection = [0] * len(candidate_mask)
    ...         for node_id in sampled: selection[node_id] = 1
    ...         return selection
    """
    
    cdef default_customize_nodes_callback_t native_callback
    
    def __init__(self):
        self.native_callback.pyCallbackClass = <PyObject*><void*>self
    
    def get_native_callback(self):
        return <uintptr_t>&(self.native_callback)
    
    def _cpp_customize_nodes_to_search(self, unsigned long long solution_ptr, int num_routes,
                                       float solution_cost, unsigned long long mask_ptr, int iter):
        cdef const vector[int]* solution_flat = <const vector[int]*>solution_ptr
        cdef const vector[int]* candidate_mask = <const vector[int]*>mask_ptr
        
        py_solution_flat = solution_flat[0]
        py_candidate_mask = candidate_mask[0]
        
        return self.customize_nodes_to_search(py_solution_flat, num_routes,
                                              solution_cost, py_candidate_mask, iter)
    
    def customize_nodes_to_search(self, solution_flat, num_routes, solution_cost, candidate_mask, iter):
        """
        Customize which nodes to sample for local search
        
        Override this method to implement custom node selection logic based on
        the current solution state.
        
        Parameters
        ----------
        solution_flat : list of int
            Current solution as flat array organized by route:
            [route0_dummy0, route0_dummy1, ..., route0_node1, route0_node2, ..., route1_dummy0, ...]
        num_routes : int
            Total number of routes in current solution
        solution_cost : float
            Current solution objective cost
        candidate_mask : list of int
            Binary mask indexed by node_id: candidate_mask[node_id]=1 means node_id is a candidate
            Length = num_orders + num_routes * 4
        iter : int
            Current iteration number in local search
            
        Returns
        -------
        list of int
            Selection mask indexed by node_id (same length as candidate_mask).
            selection_mask[node_id]=1 means select node_id, 0 means not select.
            Example: [0, 0, 0, 0, 0, 1, 0, ..., 1, 0] with len=num_orders+num_routes*4
        """
        import random
        
        # Extract candidate node_ids from mask
        candidate_node_ids = [node_id for node_id in range(len(candidate_mask)) if candidate_mask[node_id] == 1]
        num_candidates = len(candidate_node_ids)
        
        if num_candidates == 0:
            return [0] * len(candidate_mask)
        
        if num_candidates < 40:
            sample_size = num_candidates
        elif num_candidates < 80:
            sample_size = num_candidates // 2
        else:
            sample_size = 40
        
        # Sample node IDs
        sampled_node_ids = random.sample(candidate_node_ids, sample_size)
        
        # Convert to selection mask
        selection_mask = [0] * len(candidate_mask)
        for node_id in sampled_node_ids:
            selection_mask[node_id] = 1
        
        return selection_mask


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
    
    def _cpp_receive_reward(self, int improvement_found, float solution_cost, int iter):
        self.receive_reward(bool(improvement_found), solution_cost, iter)
    
    def receive_reward(self, improvement_found, solution_cost, iter):
        """
        Receive reward signal from search iteration
        
        Override this method to implement custom reward processing logic.
        
        Parameters
        ----------
        improvement_found : bool
            Whether an improving move was discovered in this iteration
        solution_cost : float
            Current solution objective cost
        iter : int
            Current iteration number in local search
        """
        pass


cdef class LocalSearchStartCallback(PyCallback):
    """
    Callback for observing state before local search begins
    
    This callback is invoked once at the start of local search, before any
    iterations begin. It provides access to the initial solution state and
    infeasibility weights.
    
    Examples
    --------
    >>> from cuopt.routing import LocalSearchStartCallback
    >>> 
    >>> class MyStartCallback(LocalSearchStartCallback):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.initial_cost = None
    ...     
    ...     def on_local_search_start(self, solution_flat, num_routes, 
    ...                              solution_cost, weights, selection_weights, should_all_nodes_be_served):
    ...         self.initial_cost = solution_cost
    ...         print(f"Starting local search: cost={solution_cost:.2f}, routes={num_routes}")
    """
    
    cdef default_local_search_start_callback_t native_callback
    
    def __init__(self):
        self.native_callback.pyCallbackClass = <PyObject*><void*>self
    
    def get_native_callback(self):
        return <uintptr_t>&(self.native_callback)
    
    def _cpp_on_local_search_start(self, unsigned long long solution_ptr, int num_routes,
                                   double solution_cost, unsigned long long weights_ptr,
                                   unsigned long long selection_weights_ptr, int should_all_nodes_be_served):
        cdef const vector[int]* solution_flat = <const vector[int]*>solution_ptr
        cdef const vector[double]* weights = <const vector[double]*>weights_ptr
        cdef const vector[double]* selection_weights = <const vector[double]*>selection_weights_ptr
        
        py_solution_flat = solution_flat[0] if solution_flat else []
        if weights:
            py_weights = weights[0]
        else:
            py_weights = []
        if selection_weights:
            py_selection_weights = selection_weights[0]
        else:
            py_selection_weights = []
        
        self.on_local_search_start(
            py_solution_flat,
            num_routes,
            solution_cost,
            py_weights,
            py_selection_weights,
            bool(should_all_nodes_be_served)
        )
    
    def on_local_search_start(self, solution_flat, num_routes, solution_cost, 
                             weights, selection_weights, should_all_nodes_be_served):
        """
        Observe search state before local search begins
        
        Override this method to implement custom observation logic.
        
        Parameters
        ----------
        solution_flat : list of int
            Current solution as flat array organized by route:
            [route0_dummy0, route0_dummy1, ..., route0_node1, route0_node2, ..., route1_dummy0, ...]
        num_routes : int
            Number of routes in current solution
        solution_cost : float
            Current solution objective cost
        weights : list of float
            Weights used for cost computation
        selection_weights : list of float
            Weights used for move selection
        should_all_nodes_be_served : bool
            Whether all nodes should be served
        """
        pass


cdef class BeforeCycleFinderCallback(PyCallback):
    """
    Callback for observing state before cycle finder execution
    
    This callback is invoked before each cycle finder execution during
    local search iterations. It allows monitoring the solution state
    before potential improvements are attempted.
    
    Examples
    --------
    >>> from cuopt.routing import BeforeCycleFinderCallback
    >>> 
    >>> class MyBeforeCallback(BeforeCycleFinderCallback):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.costs = []
    ...     
    ...     def on_before_cycle_finder(self, solution_flat, num_routes, 
    ...                               solution_cost, iter):
    ...         self.costs.append(solution_cost)
    ...         if iter % 10 == 0:
    ...             print(f"Iter {iter}: cost={solution_cost:.2f}")
    """
    
    cdef default_before_cycle_finder_callback_t native_callback
    
    def __init__(self):
        self.native_callback.pyCallbackClass = <PyObject*><void*>self
    
    def get_native_callback(self):
        return <uintptr_t>&(self.native_callback)
    
    def _cpp_on_before_cycle_finder(self, unsigned long long solution_ptr, int num_routes,
                                    float solution_cost, int iter):
        cdef const vector[int]* solution_flat = <const vector[int]*>solution_ptr
        
        py_solution_flat = solution_flat[0]
        
        self.on_before_cycle_finder(
            py_solution_flat,
            num_routes,
            solution_cost,
            iter
        )
    
    def on_before_cycle_finder(self, solution_flat, num_routes, solution_cost, iter):
        """
        Observe search state before cycle finder execution
        
        Override this method to implement custom observation logic.
        
        Parameters
        ----------
        solution_flat : list of int
            Current solution as flat array organized by route:
            [route0_dummy0, route0_dummy1, ..., route0_node1, route0_node2, ..., route1_dummy0, ...]
        num_routes : int
            Number of routes in current solution
        solution_cost : float
            Current solution objective cost
        iter : int
            Current iteration number
        """
        pass


cdef class AfterCycleFinderCallback(PyCallback):
    """
    Callback for observing state after cycle finder execution
    
    This callback is invoked after each cycle finder execution during
    local search iterations. It provides information about whether
    an improvement was found and the resulting solution state.
    
    Examples
    --------
    >>> from cuopt.routing import AfterCycleFinderCallback
    >>> 
    >>> class MyAfterCallback(AfterCycleFinderCallback):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.improvements = 0
    ...     
    ...     def on_after_cycle_finder(self, solution_flat, num_routes, 
    ...                              solution_cost, iter, improved):
    ...         if improved:
    ...             self.improvements += 1
    ...             print(f"Iter {iter}: ✓ Improvement #{self.improvements}, cost={solution_cost:.2f}")
    """
    
    cdef default_after_cycle_finder_callback_t native_callback
    
    def __init__(self):
        self.native_callback.pyCallbackClass = <PyObject*><void*>self
    
    def get_native_callback(self):
        return <uintptr_t>&(self.native_callback)
    
    def _cpp_on_after_cycle_finder(self, unsigned long long solution_ptr, int num_routes,
                                   float solution_cost, int iter, int improved):
        cdef const vector[int]* solution_flat = <const vector[int]*>solution_ptr
        
        py_solution_flat = solution_flat[0]
        
        self.on_after_cycle_finder(
            py_solution_flat,
            num_routes,
            solution_cost,
            iter,
            bool(improved)
        )
    
    def on_after_cycle_finder(self, solution_flat, num_routes, solution_cost, iter, improved):
        """
        Observe search state after cycle finder execution
        
        Override this method to implement custom observation logic.
        
        Parameters
        ----------
        solution_flat : list of int
            Current solution as flat array organized by route:
            [route0_dummy0, route0_dummy1, ..., route0_node1, route0_node2, ..., route1_dummy0, ...]
        num_routes : int
            Number of routes in current solution
        solution_cost : float
            Current solution objective cost (after improvement if improved=True)
        iter : int
            Current iteration number
        improved : bool
            Whether an improvement was found in this iteration
        """
        pass

