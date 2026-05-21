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
class default_customize_early_stop_callback_t : public customize_early_stop_callback_t<i_t, f_t> {
public:
  void customize_early_stop(
    const std::vector<i_t>* solution_flat,
    i_t num_routes,
    f_t objective,
    i_t iteration,
    bool* early_stop_out,
    i_t phase = 0
  ) override
  {
    PyObject* pycl = (PyObject*)this->pyCallbackClass;

    // Convert solution_flat to Python list
    PyObject* py_solution_flat = PyList_New(solution_flat->size());
    for (size_t i = 0; i < solution_flat->size(); ++i) {
      PyList_SetItem(py_solution_flat, i, PyLong_FromLong((*solution_flat)[i]));
    }

    // Call Python method: customize_early_stop(solution_flat, objective, num_routes, iteration, phase) -> bool
    PyObject* result = PyObject_CallMethod(
      pycl,
      "customize_early_stop",
      "Odiii",
      py_solution_flat,
      (double)objective,
      (int)num_routes,
      (int)iteration,
      (int)phase
    );

    Py_DECREF(py_solution_flat);

    if (result == nullptr) {
      PyErr_Print();
      return;
    }

    // Extract early_stop from result (should be bool)
    // NOTE: 4/10 era did NOT have abort_solve_flag write here. We're matching that.
    if (PyBool_Check(result)) {
      *early_stop_out = (result == Py_True);
    }

    Py_DECREF(result);
  }

  PyObject* pyCallbackClass;
};

}  // namespace callbacks
}  // namespace routing
}  // namespace cuopt
