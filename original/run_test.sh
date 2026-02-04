#!/bin/bash
# Script to run tests with proper library paths set

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cuopt

# Set library paths to ensure correct nvJitLink library is found
export LD_LIBRARY_PATH=/home/jieyi/.conda/envs/cuopt/targets/x86_64-linux/lib:/home/jieyi/.conda/envs/cuopt/lib:$LD_LIBRARY_PATH

# Preload the correct nvJitLink library to avoid symbol version issues
export LD_PRELOAD=/home/jieyi/.conda/envs/cuopt/targets/x86_64-linux/lib/libnvJitLink.so.12.9.86

# Run the test script
python "$@"
