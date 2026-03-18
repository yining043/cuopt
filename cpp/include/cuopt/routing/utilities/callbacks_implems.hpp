/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>

namespace cuopt {
namespace routing {
namespace callbacks {

template <typename i_t, typename f_t>
class default_customize_nodes_callback_t : public customize_nodes_callback_t<i_t, f_t> {
public:
  void customize_nodes_to_search(
    const std::vector<i_t>* solution_flat,
    i_t num_routes,
    f_t solution_cost,
    const std::vector<i_t>* trail_masks_flat,
    i_t num_trails,
    std::vector<i_t>* selection_mask_out,
    i_t iteration
  ) override
  {
    PyObject* pycl = (PyObject*)this->pyCallbackClass;
    
    PyObject* py_solution_flat = PyList_New(solution_flat->size());
    for (size_t i = 0; i < solution_flat->size(); ++i) {
      PyList_SetItem(py_solution_flat, i, PyLong_FromLong((*solution_flat)[i]));
    }
    
    PyObject* py_trail_masks = PyList_New(trail_masks_flat->size());
    for (size_t i = 0; i < trail_masks_flat->size(); ++i) {
      PyList_SetItem(py_trail_masks, i, PyLong_FromLong((*trail_masks_flat)[i]));
    }
    
    PyObject* result = PyObject_CallMethod(
      pycl, 
      "customize_nodes_to_search", 
      "OifOii",
      py_solution_flat,
      (int)num_routes,
      (double)solution_cost,
      py_trail_masks,
      (int)num_trails,
      (int)iteration
    );
    
    Py_DECREF(py_solution_flat);
    Py_DECREF(py_trail_masks);
    
    if (result == nullptr) {
      PyErr_Print();
      return;
    }
    
    if (PyList_Check(result)) {
      Py_ssize_t size = PyList_Size(result);
      selection_mask_out->resize(size);
      for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject* item = PyList_GetItem(result, i);
        (*selection_mask_out)[i] = (i_t)PyLong_AsLong(item);
      }
    }
    
    Py_DECREF(result);
  }
  
  PyObject* pyCallbackClass;
};

}  // namespace callbacks
}  // namespace routing
}  // namespace cuopt

