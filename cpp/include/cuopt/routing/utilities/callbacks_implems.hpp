/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <Python.h>
#include <cuopt/routing/utilities/internals.hpp>

namespace cuopt {
namespace routing {
namespace callbacks {

// Default customize nodes callback implementation (bridges C++ to Python)
class default_customize_nodes_callback_t : public customize_nodes_callback_t {
public:
    void customize_nodes_to_search(
        const std::vector<int>* solution_flat,
        int num_routes,
        float solution_cost,
        const std::vector<int>* candidate_mask,
        std::vector<int>* selection_mask_out,
        int iter
    ) override {
        PyObject* result = PyObject_CallMethod(
            this->pyCallbackClass,
            "_cpp_customize_nodes_to_search",
            "(KifKi)",
            reinterpret_cast<unsigned long long>(solution_flat),
            num_routes,
            solution_cost,
            reinterpret_cast<unsigned long long>(candidate_mask),
            iter
        );
        
        if (result && PyList_Check(result)) {
            Py_ssize_t size = PyList_Size(result);
            selection_mask_out->reserve(size);
            for (Py_ssize_t i = 0; i < size; ++i) {
                PyObject* item = PyList_GetItem(result, i);
                if (PyLong_Check(item)) {
                    selection_mask_out->push_back(PyLong_AsLong(item));
                }
            }
        }
        else {
            if (PyErr_Occurred()) {
                PyErr_Print();  // 打印异常信息
                PyErr_Clear();  // 清理异常状态
                exit(1);
            }
        }
        
        if (result) Py_DECREF(result);
        else {
            if (PyErr_Occurred()) {
                PyErr_Print();  // 打印异常信息
                PyErr_Clear();  // 清理异常状态
                exit(1);
            }
        }
    }
    
    PyObject* pyCallbackClass;
};

// Default reward callback implementation (bridges C++ to Python)
class default_reward_callback_t : public reward_callback_t {
public:
    void receive_reward(
        bool improvement_found,
        float solution_cost,
        int iter,
        const std::vector<int>* solution_flat,
        int num_routes
    ) override {
        PyObject* result = PyObject_CallMethod(
            this->pyCallbackClass,
            "_cpp_receive_reward",
            "(ifiKi)",
            improvement_found ? 1 : 0,
            solution_cost,
            iter,
            reinterpret_cast<unsigned long long>(solution_flat),
            num_routes
        );
        
        if (result) Py_DECREF(result);
        else {
            if (PyErr_Occurred()) {
                PyErr_Print();  // 打印异常信息
                PyErr_Clear();  // 清理异常状态
                exit(1);
            }
        }
    }
    
    PyObject* pyCallbackClass;
};

// Default local search start callback implementation (bridges C++ to Python)
class default_local_search_start_callback_t : public local_search_start_callback_t {
public:
    void on_local_search_start(
        const std::vector<int>* solution_flat,
        int num_routes,
        double solution_cost,
        const std::vector<double>* weights,
        const std::vector<double>* selection_weights,
        bool should_all_nodes_be_served
    ) override {
        PyObject* result = PyObject_CallMethod(
            this->pyCallbackClass,
            "_cpp_on_local_search_start",
            "(KidKKi)",
            reinterpret_cast<unsigned long long>(solution_flat),
            num_routes,
            solution_cost,
            reinterpret_cast<unsigned long long>(weights),
            reinterpret_cast<unsigned long long>(selection_weights),
            should_all_nodes_be_served ? 1 : 0
        );
        
        if (result) Py_DECREF(result);
        else {
            if (PyErr_Occurred()) {
                PyErr_Print();  // 打印异常信息
                PyErr_Clear();  // 清理异常状态
                exit(1);
            }
        }
    }
    
    PyObject* pyCallbackClass;
};

// Default before cycle finder callback implementation (bridges C++ to Python)
class default_before_cycle_finder_callback_t : public before_cycle_finder_callback_t {
public:
    void on_before_cycle_finder(
        const std::vector<int>* solution_flat,
        int num_routes,
        float solution_cost,
        int iter
    ) override {
        PyObject* result = PyObject_CallMethod(
            this->pyCallbackClass,
            "_cpp_on_before_cycle_finder",
            "(Kifi)",
            reinterpret_cast<unsigned long long>(solution_flat),
            num_routes,
            solution_cost,
            iter
        );
        
        if (result) Py_DECREF(result);
        else {
            if (PyErr_Occurred()) {
                PyErr_Print();  // 打印异常信息
                PyErr_Clear();  // 清理异常状态
                exit(1);
            }
        }
    }
    
    PyObject* pyCallbackClass;
};

// Default after cycle finder callback implementation (bridges C++ to Python)
class default_after_cycle_finder_callback_t : public after_cycle_finder_callback_t {
public:
    void on_after_cycle_finder(
        const std::vector<int>* solution_flat,
        int num_routes,
        float solution_cost,
        int iter,
        bool improved
    ) override {
        PyObject* result = PyObject_CallMethod(
            this->pyCallbackClass,
            "_cpp_on_after_cycle_finder",
            "(Kifii)",
            reinterpret_cast<unsigned long long>(solution_flat),
            num_routes,
            solution_cost,
            iter,
            improved ? 1 : 0
        );
        
        if (result) Py_DECREF(result);
        else {
            if (PyErr_Occurred()) {
                PyErr_Print();  // 打印异常信息
                PyErr_Clear();  // 清理异常状态
                exit(1);
            }
        }
    }
    
    PyObject* pyCallbackClass;
};

}  // namespace callbacks
}  // namespace routing
}  // namespace cuopt

