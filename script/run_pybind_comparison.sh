#!/bin/bash
# Batch processing script for pybind comparison
# Automatically reads all run_ids for the given instance_index and processes them
# Supports resume from interruption

set -e  # Exit immediately on error

PKL_FILE="/home/jieyi/cvrp100_uniform.pkl"
HGS_FILE="/home/jieyi/hgs_cvrp100_uniform.pkl"
INSTANCE_IDX=${1:-0}  # Default to 0 if not provided
CUDA_DEVICE=${2:-0}  # Default to 0 if not provided
N_RUNS=${3:-30}  # Default to 30 if not provided
MAX_ITER=${4:-100}  # Default to 100 if not provided
BASIN_DIR="basin_datasets0"

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}

# Log file (fixed name for resume support)
LOG_FILE="pybind_comparison_idx${INSTANCE_IDX}.log"

echo "========================================" | tee -a ${LOG_FILE}
echo "Starting pybind comparison at $(date)" | tee -a ${LOG_FILE}
echo "Instance index: ${INSTANCE_IDX}" | tee -a ${LOG_FILE}
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}" | tee -a ${LOG_FILE}
echo "Number of runs per run_id: ${N_RUNS}" | tee -a ${LOG_FILE}
echo "Max iterations: ${MAX_ITER}" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}

# Get all run_ids from trajectory.jsonl
echo "Scanning for run_ids to process..." | tee -a ${LOG_FILE}

export INSTANCE_IDX
export N_RUNS
RUN_IDS_TO_PROCESS=$(python3 -u << 'PYEOF'
import sys
import os
sys.path.insert(0, '/home/jieyi/cuopt')
from compare_pybind_cuopt import get_all_run_ids, get_basin_paths, is_run_id_completed

instance_path = "/home/jieyi/cvrp100_uniform.pkl"
instance_index = int(os.environ.get('INSTANCE_IDX', '0'))
basin_base_dir = "basin_datasets0"
n_runs = int(os.environ.get('N_RUNS', '30'))

basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
trajectory_path = basin_paths['trajectory_path']

if not os.path.exists(trajectory_path):
    print(f"Error: trajectory.jsonl not found at {trajectory_path}", file=sys.stderr)
    sys.exit(1)

all_run_ids = get_all_run_ids(trajectory_path)
print(f"Total run_ids found: {len(all_run_ids)}", file=sys.stderr)

# Check which run_ids still need to be processed
to_process = []
for run_id in all_run_ids:
    if not is_run_id_completed(instance_path, instance_index, run_id, basin_base_dir, n_runs):
        to_process.append(run_id)
        print(f"  To process: run_id={run_id}", file=sys.stderr)
    else:
        print(f"  Skipping (already done): run_id={run_id}", file=sys.stderr)

print(f"\nRun_ids to process: {len(to_process)}", file=sys.stderr)
print(f"Already completed: {len(all_run_ids) - len(to_process)}", file=sys.stderr)

# Output run_ids to process (one per line)
for run_id in to_process:
    print(run_id)
PYEOF
)

if [ -z "$RUN_IDS_TO_PROCESS" ]; then
    echo "All run_ids already completed!" | tee -a ${LOG_FILE}
    exit 0
fi

# Statistics
TOTAL_RUNS=$(echo "$RUN_IDS_TO_PROCESS" | wc -l)
CURRENT=0

echo "" | tee -a ${LOG_FILE}
echo "Starting to process ${TOTAL_RUNS} run_ids..." | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

# Process each run_id one by one
for RUN_ID in $RUN_IDS_TO_PROCESS; do
    CURRENT=$((CURRENT + 1))
    
    echo "========================================" | tee -a ${LOG_FILE}
    echo "Progress: ${CURRENT}/${TOTAL_RUNS}" | tee -a ${LOG_FILE}
    echo "Processing run_id=${RUN_ID}" | tee -a ${LOG_FILE}
    echo "Started at: $(date)" | tee -a ${LOG_FILE}
    echo "========================================" | tee -a ${LOG_FILE}
    
    # Run comparison
    python3 -u compare_pybind_cuopt.py \
        --pkl ${PKL_FILE} \
        --idx ${INSTANCE_IDX} \
        --run_id ${RUN_ID} \
        --n_runs ${N_RUNS} \
        --max_iter ${MAX_ITER} \
        --basin_dir ${BASIN_DIR} \
        --hgs ${HGS_FILE} 2>&1 | tee -a ${LOG_FILE}
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Run_id ${RUN_ID} completed successfully" | tee -a ${LOG_FILE}
    else
        echo "✗ Run_id ${RUN_ID} failed with exit code: ${EXIT_CODE}" | tee -a ${LOG_FILE}
        echo "Continuing to next run_id..." | tee -a ${LOG_FILE}
    fi
    
    echo "Finished at: $(date)" | tee -a ${LOG_FILE}
    echo "" | tee -a ${LOG_FILE}
    
    # Wait 2 seconds to ensure resources are fully released
    sleep 2
    
    # Display GPU memory status
    echo "GPU Memory status:" | tee -a ${LOG_FILE}
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -n1 | tee -a ${LOG_FILE}
    echo "" | tee -a ${LOG_FILE}
done

echo "========================================" | tee -a ${LOG_FILE}
echo "All run_ids completed at $(date)" | tee -a ${LOG_FILE}
echo "Log file: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}
