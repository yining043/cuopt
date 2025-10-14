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

// Default implementation that bridges C++ to Python callback
class default_observation_callback_t : public observation_callback_t {
public:
    void get_observation_and_sample(
        const std::vector<std::vector<int>>* routes,
        const std::vector<int>* node_ids_to_search,
        std::vector<int>* sampled_indices_out,
        float objective_value,
        int n_routes
    ) override {
        // Call Python callback method
        PyObject* res = PyObject_CallMethod(
            this->pyCallbackClass, 
            "_cpp_callback_wrapper", 
            "(KKfi)", 
            reinterpret_cast<unsigned long long>(routes),
            reinterpret_cast<unsigned long long>(node_ids_to_search),
            objective_value,
            n_routes
        );
        
        // Parse Python list of indices
        if (res && PyList_Check(res)) {
            Py_ssize_t size = PyList_Size(res);
            sampled_indices_out->reserve(size);
            for (Py_ssize_t i = 0; i < size; ++i) {
                PyObject* item = PyList_GetItem(res, i);
                if (PyLong_Check(item)) {
                    sampled_indices_out->push_back(PyLong_AsLong(item));
                }
            }
        }
        
        if (res) Py_DECREF(res);
    }
    
    PyObject* pyCallbackClass;
};

}  // namespace callbacks
}  // namespace routing
}  // namespace cuopt

