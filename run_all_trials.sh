#!/bin/bash
# Batch processing script for all trials
# Each trial runs independently to completely avoid GPU memory accumulation

# Note: We don't use 'set -e' because we want to continue processing even if individual trials fail
# Instead, we check exit codes explicitly for critical operations

PKL_FILE="/home/jieyi/cvrp100_uniform.pkl"
HGS_FILE="/home/jieyi/hgs_cvrp100_uniform.pkl"
INSTANCE_IDX=${1:-0}  # Default to 0 if not provided
CUDA_DEVICE=${2:-0}  # Default to 0 if not provided
MAX_RUNS=${3:-10}  # Default to 10 if not provided (0 means process all runs)
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
if [ "${MAX_RUNS}" -gt 0 ]; then
    echo "Max runs to process: ${MAX_RUNS}" | tee -a ${LOG_FILE}
else
    echo "Processing all runs (no limit)" | tee -a ${LOG_FILE}
fi
echo "========================================" | tee -a ${LOG_FILE}

# First, get all (run_id, trial_id) pairs that need to be processed
echo "Scanning for trials to process..." | tee -a ${LOG_FILE}

# Use Python to get all trial combinations and check completion status in ONE pass
# This avoids reading large files twice
export INSTANCE_IDX
export MAX_RUNS
TRIALS_TO_PROCESS=$(python3 -u << 'PYEOF'
import sys
import os
import json
sys.path.insert(0, '/home/jieyi/cuopt')
from analyze_basin import get_all_run_trial_pairs, get_basin_paths

instance_path = "/home/jieyi/cvrp100_uniform.pkl"
instance_index = int(os.environ.get('INSTANCE_IDX', '0'))
basin_base_dir = "basin_datasets0"
max_runs = int(os.environ.get('MAX_RUNS', '10'))

basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
all_pairs = get_all_run_trial_pairs(basin_paths['trajectory_path'])

total_unique_runs = len(set(run_id for run_id, _ in all_pairs))
print(f"Total trials found: {len(all_pairs)}", file=sys.stderr)
print(f"Total unique runs found: {total_unique_runs}", file=sys.stderr)

# Get unique run_ids and limit if max_runs > 0
if max_runs > 0:
    unique_run_ids = sorted(set(run_id for run_id, _ in all_pairs))[:max_runs]
    print(f"Processing first {max_runs} runs: {unique_run_ids}", file=sys.stderr)
    # Filter pairs to only include the limited runs
    all_pairs = [(run_id, trial_id) for run_id, trial_id in all_pairs if run_id in unique_run_ids]
    print(f"Filtered trials for first {max_runs} runs: {len(all_pairs)}", file=sys.stderr)
else:
    print(f"Processing all {total_unique_runs} runs (no limit)", file=sys.stderr)

# Pre-load trajectory and training data ONCE to avoid repeated file reads
trajectory_path = basin_paths['trajectory_path']
basin_dir = os.path.join("basin_datasets0_analyze", basin_paths['instance_id'])
training_file = os.path.join(basin_dir, 'training_data.jsonl')

# Build a cache of all trial solutions from trajectory
trajectory_cache = {}
with open(trajectory_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            run_id = data.get('run_id')
            trial_id = data.get('trial_id')
            if run_id is not None and trial_id is not None:
                key = (run_id, trial_id)
                if key not in trajectory_cache:
                    trajectory_cache[key] = set()
                trajectory_cache[key].add((data.get('global_iter'), data.get('local_iter')))
        except:
            pass

# Build a cache of analyzed solutions from training_data.jsonl
analyzed_cache = {}
if os.path.exists(training_file):
    with open(training_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                run_id = data.get('run_id')
                trial_id = data.get('trial_id')
                if run_id is not None and trial_id is not None and 'initial_solution' in data:
                    key = (run_id, trial_id)
                    if key not in analyzed_cache:
                        analyzed_cache[key] = set()
                    analyzed_cache[key].add((data.get('global_iter'), data.get('local_iter')))
            except:
                pass

# Now check each trial using cached data (much faster!)
to_process = []
for run_id, trial_id in all_pairs:
    # Quick check: visualization file exists?
    viz_file = os.path.join(basin_dir, f'comprehensive_analysis_run_{run_id}_trial_{trial_id}.png')
    if not os.path.exists(viz_file):
        to_process.append((run_id, trial_id))
        print(f"  To process: run_id={run_id}, trial_id={trial_id}", file=sys.stderr)
        continue
    
    # Check if all solutions are analyzed using cached data
    key = (run_id, trial_id)
    trial_solutions = trajectory_cache.get(key, set())
    analyzed_solutions = analyzed_cache.get(key, set())
    
    if not trial_solutions or trial_solutions.issubset(analyzed_solutions):
        print(f"  Skipping (already done): run_id={run_id}, trial_id={trial_id}", file=sys.stderr)
    else:
        to_process.append((run_id, trial_id))
        print(f"  To process: run_id={run_id}, trial_id={trial_id}", file=sys.stderr)

print(f"\nTrials to process: {len(to_process)}", file=sys.stderr)
print(f"Already completed: {len(all_pairs) - len(to_process)}", file=sys.stderr)

# Output the list of trials to process
for run_id, trial_id in to_process:
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
    # Capture exit code correctly even with pipe
    python3 -u analyze_basin.py \
        --pkl ${PKL_FILE} \
        --idx ${INSTANCE_IDX} \
        --run_id ${RUN_ID} \
        --trial_id ${TRIAL_ID} \
        --num_runs ${NUM_RUNS} \
        --basin_dir ${BASIN_DIR} \
        --hgs ${HGS_FILE} 2>&1 | tee -a ${LOG_FILE}
    
    EXIT_CODE=${PIPESTATUS[0]}  # Get exit code of python3, not tee
    
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
