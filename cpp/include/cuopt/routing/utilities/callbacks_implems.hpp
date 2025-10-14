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

class default_get_solution_callback_t : public get_solution_callback_t {
public:
    void get_solution(
        const std::vector<std::vector<int>>* routes,
        float objective_value,
        int n_routes
    ) override {
        PyObject* res = PyObject_CallMethod(
            this->pyCallbackClass, 
            "_cpp_callback_wrapper", 
            "(Kfi)", 
            reinterpret_cast<unsigned long long>(routes),
            objective_value,
            n_routes
        );
        
        if (res) Py_DECREF(res);
    }
    
    PyObject* pyCallbackClass;
};

}  // namespace callbacks
}  // namespace routing
}  // namespace cuopt

