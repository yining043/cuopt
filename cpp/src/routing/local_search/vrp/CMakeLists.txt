# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Add Pybind11 module to the main cuOpt build
if(BUILD_PYTHON_BINDINGS)
    # Find pybind11
    find_package(pybind11 REQUIRED)
    
    # Create the pybind11 module - use both .cpp and .cu following cuOpt's separation pattern
    pybind11_add_module(cuopt_pybind
        pybind_vrp_interface.cpp
        pybind_vrp_interface.cu
    )
    
    # Set target properties
    set_target_properties(cuopt_pybind PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    # Include directories
    target_include_directories(cuopt_pybind PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    
    # Link libraries - link against the main cuOpt library
    target_link_libraries(cuopt_pybind PRIVATE
        cuopt::cuopt
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
    )
    
    # Compile definitions
    target_compile_definitions(cuopt_pybind PRIVATE
        PYBIND11_USE_SMART_HOLDER_AS_DEFAULT=1
    )
    
    # Compiler flags - following cuOpt's exact flags
    target_compile_options(cuopt_pybind PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
        $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:-Werror=cross-execution-space-call>
        $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-declarations>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
        $<$<COMPILE_LANGUAGE:CUDA>:-Wno-error=non-template-friend>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xfatbin=-compress-all>
        $<$<COMPILE_LANGUAGE:CUDA>:-fopenmp>
        # Additional flags to match cuOpt's main build
        $<$<COMPILE_LANGUAGE:CUDA>:-static-global-template-stub=false>
    )
    
    # Installation
    install(TARGETS cuopt_pybind
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python3/dist-packages
    )
endif()
