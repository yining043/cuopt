#!/bin/bash

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

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -d "cpp" ] || [ ! -f "cpp/CMakeLists.txt" ]; then
    print_error "Please run this script from the cuOpt root directory"
    exit 1
fi

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    print_error "nvcc not found. Please install CUDA toolkit"
    exit 1
fi

print_status "CUDA version: $(nvcc --version | grep release | cut -d' ' -f5)"

# Check for pybind11
if ! python3 -c "import pybind11" 2>/dev/null; then
    print_error "pybind11 not found. Please install: pip install pybind11"
    exit 1
fi

print_status "pybind11 version: $(python3 -c "import pybind11; print(pybind11.__version__)")"

# Build cuOpt with Python bindings enabled
print_status "Building cuOpt with Python bindings..."

cd cpp
mkdir -p build
cd build

# Configure with CMake
print_status "Configuring cuOpt build with Python bindings..."

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CUDA_STANDARD=17 \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DCMAKE_INSTALL_PREFIX="$(pwd)/install" \
    -DPYTHON_EXECUTABLE=$(which python3) \
    ..

if [ $? -ne 0 ]; then
    print_error "CMake configuration failed"
    exit 1
fi

# Build
print_status "Building cuOpt with Python bindings..."
ninja cuopt_pybind

if [ $? -ne 0 ]; then
    print_error "Build failed"
    exit 1
fi

print_status "Build completed successfully!"

# Install
print_status "Installing Python bindings..."
ninja install

if [ $? -ne 0 ]; then
    print_error "Installation failed"
    exit 1
fi

print_status "Installation completed successfully!"

# Build completed successfully
print_status "Build and installation completed successfully!"
print_status "To test the Python module, run:"
echo "  export PYTHONPATH=\"$(pwd)/cpp/build/install/lib/python3/dist-packages:\$PYTHONPATH\""
echo "  python test_vrp_pybind.py"