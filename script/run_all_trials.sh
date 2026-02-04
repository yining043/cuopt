#!/bin/bash
# Batch processing script for all trials
# Each trial runs independently to completely avoid GPU memory accumulation

set -e  # Exit immediately on error

PKL_FILE="/home/jieyi/cvrp100_uniform.pkl"
HGS_FILE="/home/jieyi/hgs_cvrp100_uniform.pkl"
INSTANCE_IDX=${1:-0}  # Default to 0 if not provided
CUDA_DEVICE=${2:-0}  # Default to 0 if not provided
NUM_RUNS=100
BASIN_DIR="basin_datasets0"

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}

# Log file (fixed name for resume support)
LOG_FILE="basin_data_collection_idx${INSTANCE_IDX}.log"

echo "========================================" | tee -a ${LOG_FILE}
echo "Starting batch processing at $(date)" | tee -a ${LOG_FILE}
echo "Instance index: ${INSTANCE_IDX}" | tee -a ${LOG_FILE}
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}

# First, get all (run_id, trial_id) pairs that need to be processed
echo "Scanning for trials to process..." | tee -a ${LOG_FILE}

# Use Python to get all trial combinations
export INSTANCE_IDX
python3 -u << 'PYEOF' | tee -a ${LOG_FILE}
import sys
import os
sys.path.insert(0, '/home/jieyi/cuopt')
from analyze_basin import get_all_run_trial_pairs, get_basin_paths, is_trial_already_analyzed

instance_path = "/home/jieyi/cvrp100_uniform.pkl"
instance_index = int(os.environ.get('INSTANCE_IDX', '0'))
basin_base_dir = "basin_datasets0"

basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
all_pairs = get_all_run_trial_pairs(basin_paths['trajectory_path'])

print(f"\nTotal trials found: {len(all_pairs)}")

# Check which trials still need to be processed
to_process = []
for run_id, trial_id in all_pairs:
    if not is_trial_already_analyzed(instance_path, instance_index, run_id, trial_id, basin_base_dir):
        to_process.append((run_id, trial_id))
        print(f"  To process: run_id={run_id}, trial_id={trial_id}")
    else:
        print(f"  Skipping (already done): run_id={run_id}, trial_id={trial_id}")

print(f"\nTrials to process: {len(to_process)}")
print(f"Already completed: {len(all_pairs) - len(to_process)}")
PYEOF

# Read the list of trials to process
export INSTANCE_IDX
TRIALS_TO_PROCESS=$(python3 -u << 'PYEOF'
import sys
import os
sys.path.insert(0, '/home/jieyi/cuopt')
from analyze_basin import get_all_run_trial_pairs, get_basin_paths, is_trial_already_analyzed

instance_path = "/home/jieyi/cvrp100_uniform.pkl"
instance_index = int(os.environ.get('INSTANCE_IDX', '0'))
basin_base_dir = "basin_datasets0"

basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
all_pairs = get_all_run_trial_pairs(basin_paths['trajectory_path'])

for run_id, trial_id in all_pairs:
    if not is_trial_already_analyzed(instance_path, instance_index, run_id, trial_id, basin_base_dir):
        print(f"{run_id}:{trial_id}")
PYEOF
)

if [ -z "$TRIALS_TO_PROCESS" ]; then
    echo "All trials already completed!" | tee -a ${LOG_FILE}
    exit 0
fi

# Statistics
TOTAL_TRIALS=$(echo "$TRIALS_TO_PROCESS" | grep -c . || echo "0")
CURRENT=0

echo "" | tee -a ${LOG_FILE}
echo "Starting to process ${TOTAL_TRIALS} trials..." | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

# Process each trial one by one
# Use while read loop to handle spaces and special characters properly
while IFS=: read -r RUN_ID TRIAL_ID; do
    if [ -z "$RUN_ID" ] || [ -z "$TRIAL_ID" ]; then
        continue
    fi
    CURRENT=$((CURRENT + 1))
    
    echo "========================================" | tee -a ${LOG_FILE}
    echo "Progress: ${CURRENT}/${TOTAL_TRIALS}" | tee -a ${LOG_FILE}
    echo "Processing run_id=${RUN_ID}, trial_id=${TRIAL_ID}" | tee -a ${LOG_FILE}
    echo "Started at: $(date)" | tee -a ${LOG_FILE}
    echo "========================================" | tee -a ${LOG_FILE}
    
    # Run analysis
    python3 -u analyze_basin.py \
        --pkl ${PKL_FILE} \
        --idx ${INSTANCE_IDX} \
        --run_id ${RUN_ID} \
        --trial_id ${TRIAL_ID} \
        --num_runs ${NUM_RUNS} \
        --basin_dir ${BASIN_DIR} \
        --hgs ${HGS_FILE} 2>&1 | tee -a ${LOG_FILE}
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Trial completed successfully" | tee -a ${LOG_FILE}
    else
        echo "✗ Trial failed with exit code: ${EXIT_CODE}" | tee -a ${LOG_FILE}
        echo "Continuing to next trial..." | tee -a ${LOG_FILE}
    fi
    
    echo "Finished at: $(date)" | tee -a ${LOG_FILE}
    echo "" | tee -a ${LOG_FILE}
    
    # Wait 2 seconds to ensure resources are fully released
    sleep 2
    
    # Display GPU memory status
    echo "GPU Memory status:" | tee -a ${LOG_FILE}
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -n1 | tee -a ${LOG_FILE}
    echo "" | tee -a ${LOG_FILE}
done <<< "$TRIALS_TO_PROCESS"

echo "========================================" | tee -a ${LOG_FILE}
echo "All trials completed at $(date)" | tee -a ${LOG_FILE}
echo "Log file: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}
