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
        const std::vector<int>* candidate_mask,
        float solution_cost,
        int num_routes,
        std::vector<int>* selection_mask_out
    ) override {
        PyObject* result = PyObject_CallMethod(
            this->pyCallbackClass,
            "_cpp_customize_nodes_to_search",
            "(KKfi)",
            reinterpret_cast<unsigned long long>(solution_flat),
            reinterpret_cast<unsigned long long>(candidate_mask),
            solution_cost,
            num_routes
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
        
        if (result) Py_DECREF(result);
    }
    
    PyObject* pyCallbackClass;
};

// Default reward callback implementation (bridges C++ to Python)
class default_reward_callback_t : public reward_callback_t {
public:
    void receive_reward(
        bool improvement_found,
        float solution_cost
    ) override {
        PyObject* result = PyObject_CallMethod(
            this->pyCallbackClass,
            "_cpp_receive_reward",
            "(if)",
            improvement_found ? 1 : 0,
            solution_cost
        );
        
        if (result) Py_DECREF(result);
    }
    
    PyObject* pyCallbackClass;
};

}  // namespace callbacks
}  // namespace routing
}  // namespace cuopt

