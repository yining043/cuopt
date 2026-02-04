#!/bin/bash
# Batch processing script for analyze_intermediate_basins.py with --collect_only mode
# Each trial runs independently to completely avoid GPU memory accumulation
# Supports resume: removes incomplete trial data and continues

set -e  # Exit immediately on error

PKL_FILE="/home/jieyi/cvrp100_uniform.pkl"
HGS_FILE="/home/jieyi/hgs_cvrp100_uniform.pkl"
INSTANCE_IDX=${1:-0}  # Default to 0 if not provided
CUDA_DEVICE=${2:-0}  # Default to 0 if not provided
RUN_ID=${3:-""}  # Optional: specific run_id to process, or empty for all
MAX_ITER=100
BASIN_DIR="basin_datasets0"

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}

# Log file (fixed name for resume support)
LOG_FILE="collect_only_analysis_idx${INSTANCE_IDX}.log"

echo "========================================" | tee -a ${LOG_FILE}
echo "Starting collect_only batch processing at $(date)" | tee -a ${LOG_FILE}
echo "Instance index: ${INSTANCE_IDX}" | tee -a ${LOG_FILE}
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}" | tee -a ${LOG_FILE}
if [ -n "${RUN_ID}" ]; then
    echo "Run ID filter: ${RUN_ID}" | tee -a ${LOG_FILE}
fi
echo "========================================" | tee -a ${LOG_FILE}

# First, get all (run_id, trial_id) pairs that need to be processed
echo "Scanning for trials to process..." | tee -a ${LOG_FILE}

# Use Python to get all trial combinations and check completion status
export INSTANCE_IDX
export RUN_ID
python3 -u << 'PYEOF' | tee -a ${LOG_FILE}
import sys
import os
import json
sys.path.insert(0, '/home/jieyi/cuopt')
from analyze_basin import get_all_run_trial_pairs
from utils import get_basin_paths

instance_path = "/home/jieyi/cvrp100_uniform.pkl"
instance_index = int(os.environ.get('INSTANCE_IDX', '0'))
basin_base_dir = "basin_datasets0"
run_id_filter = os.environ.get('RUN_ID', '')

basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
all_pairs = get_all_run_trial_pairs(basin_paths['trajectory_path'])

# Filter by run_id if specified
if run_id_filter:
    all_pairs = [(r, t) for r, t in all_pairs if r == run_id_filter]

print(f"\nTotal trials found: {len(all_pairs)}")

# Check which trials still need to be processed
# A trial is considered complete if:
# 1. The intermediate_states JSON file exists
# 2. The collect_only visualization files exist

def is_trial_complete(instance_path, instance_index, run_id, trial_id, basin_base_dir):
    """Check if a trial is completely collected."""
    basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
    output_dir = os.path.join("basin_datasets0_analyze", basin_paths['instance_id'])
    
    # Check for intermediate states file
    state_file = os.path.join(output_dir, f'intermediate_states_run_{run_id}_trial_{trial_id}.json')
    if not os.path.exists(state_file):
        return False
    
    # Check for collect_only visualization files
    instance_id = basin_paths['instance_id']
    viz_dir = os.path.join("basin_datasets0_analyze", instance_id, f"{run_id}_trial_plot")
    solution_comparison = os.path.join(viz_dir, f'run_{run_id}_collect_only_solution_comparison.png')
    cost_curve = os.path.join(viz_dir, f'run_{run_id}_collect_only_cost_curve_by_trial_pybind.png')
    
    # Both visualization files should exist (they are generated together)
    if not (os.path.exists(solution_comparison) and os.path.exists(cost_curve)):
        return False
    
    return True

to_process = []
for run_id, trial_id in all_pairs:
    if not is_trial_complete(instance_path, instance_index, run_id, trial_id, basin_base_dir):
        to_process.append((run_id, trial_id))
        print(f"  To process: run_id={run_id}, trial_id={trial_id}")
    else:
        print(f"  Skipping (already done): run_id={run_id}, trial_id={trial_id}")

print(f"\nTrials to process: {len(to_process)}")
print(f"Already completed: {len(all_pairs) - len(to_process)}")
PYEOF

# Read the list of trials to process
export INSTANCE_IDX
export RUN_ID
TRIALS_TO_PROCESS=$(python3 -u << 'PYEOF'
import sys
import os
sys.path.insert(0, '/home/jieyi/cuopt')
from analyze_basin import get_all_run_trial_pairs
from utils import get_basin_paths

instance_path = "/home/jieyi/cvrp100_uniform.pkl"
instance_index = int(os.environ.get('INSTANCE_IDX', '0'))
basin_base_dir = "basin_datasets0"
run_id_filter = os.environ.get('RUN_ID', '')

basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
all_pairs = get_all_run_trial_pairs(basin_paths['trajectory_path'])

if run_id_filter:
    all_pairs = [(r, t) for r, t in all_pairs if r == run_id_filter]

def is_trial_complete(instance_path, instance_index, run_id, trial_id, basin_base_dir):
    basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
    output_dir = os.path.join("basin_datasets0_analyze", basin_paths['instance_id'])
    state_file = os.path.join(output_dir, f'intermediate_states_run_{run_id}_trial_{trial_id}.json')
    instance_id = basin_paths['instance_id']
    viz_dir = os.path.join("basin_datasets0_analyze", instance_id, f"{run_id}_trial_plot")
    solution_comparison = os.path.join(viz_dir, f'run_{run_id}_collect_only_solution_comparison.png')
    cost_curve = os.path.join(viz_dir, f'run_{run_id}_collect_only_cost_curve_by_trial_pybind.png')
    return os.path.exists(state_file) and os.path.exists(solution_comparison) and os.path.exists(cost_curve)

# Sort by run_id first, then by trial_id to ensure trials of the same run are grouped together
for run_id, trial_id in sorted(all_pairs):
    if not is_trial_complete(instance_path, instance_index, run_id, trial_id, basin_base_dir):
        print(f"{run_id}:{trial_id}")
PYEOF
)

if [ -z "$TRIALS_TO_PROCESS" ]; then
    echo "All trials already completed!" | tee -a ${LOG_FILE}
    exit 0
fi

# Statistics
TOTAL_TRIALS=$(echo "$TRIALS_TO_PROCESS" | wc -l)
CURRENT=0

echo "" | tee -a ${LOG_FILE}
echo "Starting to process ${TOTAL_TRIALS} trials..." | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

# Process each trial one by one, grouped by run_id
PREV_RUN_ID=""
for trial_pair in $TRIALS_TO_PROCESS; do
    CURRENT=$((CURRENT + 1))
    RUN_ID=$(echo $trial_pair | cut -d: -f1)
    TRIAL_ID=$(echo $trial_pair | cut -d: -f2)
    
    # Note: In collect_only mode, visualizations are generated per trial,
    # so we don't need to generate run-level visualizations separately
    
    PREV_RUN_ID=$RUN_ID
    
    echo "========================================" | tee -a ${LOG_FILE}
    echo "Progress: ${CURRENT}/${TOTAL_TRIALS}" | tee -a ${LOG_FILE}
    echo "Processing run_id=${RUN_ID}, trial_id=${TRIAL_ID}" | tee -a ${LOG_FILE}
    echo "Started at: $(date)" | tee -a ${LOG_FILE}
    echo "========================================" | tee -a ${LOG_FILE}
    
    # Clean up incomplete data for this trial before running
    # This ensures we can resume cleanly if the previous run was interrupted
    echo "Cleaning up incomplete data for this trial..." | tee -a ${LOG_FILE}
    export INSTANCE_IDX
    export RUN_ID
    export TRIAL_ID
    python3 -u << 'PYEOF'
import sys
import os
sys.path.insert(0, '/home/jieyi/cuopt')
from utils import get_basin_paths

instance_path = "/home/jieyi/cvrp100_uniform.pkl"
instance_index = int(os.environ.get('INSTANCE_IDX', '0'))
basin_base_dir = "basin_datasets0"
run_id = os.environ.get('RUN_ID', '')
trial_id = int(os.environ.get('TRIAL_ID', '0'))

basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
output_dir = os.path.join("basin_datasets0_analyze", basin_paths['instance_id'])
instance_id = basin_paths['instance_id']
viz_dir = os.path.join("basin_datasets0_analyze", instance_id, f"{run_id}_trial_plot")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)

# Remove incomplete intermediate states file
state_file = os.path.join(output_dir, f'intermediate_states_run_{run_id}_trial_{trial_id}.json')
if os.path.exists(state_file):
    os.remove(state_file)
    print(f"  Removed: {state_file}")

# Remove incomplete visualization files
solution_comparison = os.path.join(viz_dir, f'run_{run_id}_collect_only_solution_comparison.png')
cost_curve = os.path.join(viz_dir, f'run_{run_id}_collect_only_cost_curve_by_trial_pybind.png')

for f in [solution_comparison, cost_curve]:
    if os.path.exists(f):
        os.remove(f)
        print(f"  Removed: {f}")

print(f"  Cleanup complete for run_id={run_id}, trial_id={trial_id}")
PYEOF
    
    # Run collect_only analysis
    python3 -u analyze_intermediate_basins.py \
        --pkl ${PKL_FILE} \
        --idx ${INSTANCE_IDX} \
        --run_id ${RUN_ID} \
        --trial_id ${TRIAL_ID} \
        --max_iter ${MAX_ITER} \
        --basin_dir ${BASIN_DIR} \
        --hgs ${HGS_FILE} \
        --collect_only 2>&1 | tee -a ${LOG_FILE}
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Trial completed successfully" | tee -a ${LOG_FILE}
    else
        echo "✗ Trial failed with exit code: ${EXIT_CODE}" | tee -a ${LOG_FILE}
        echo "Continuing to next trial..." | tee -a ${LOG_FILE}
        # Clean up incomplete data for failed trial
        export INSTANCE_IDX
        export RUN_ID
        export TRIAL_ID
        python3 -u << 'PYEOF'
import sys
import os
sys.path.insert(0, '/home/jieyi/cuopt')
from utils import get_basin_paths

instance_path = "/home/jieyi/cvrp100_uniform.pkl"
instance_index = int(os.environ.get('INSTANCE_IDX', '0'))
basin_base_dir = "basin_datasets0"
run_id = os.environ.get('RUN_ID', '')
trial_id = int(os.environ.get('TRIAL_ID', '0'))

basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
output_dir = os.path.join("basin_datasets0_analyze", basin_paths['instance_id'])
instance_id = basin_paths['instance_id']
viz_dir = os.path.join("basin_datasets0_analyze", instance_id, f"{run_id}_trial_plot")

# Remove incomplete files
state_file = os.path.join(output_dir, f'intermediate_states_run_{run_id}_trial_{trial_id}.json')
solution_comparison = os.path.join(viz_dir, f'run_{run_id}_collect_only_solution_comparison.png')
cost_curve = os.path.join(viz_dir, f'run_{run_id}_collect_only_cost_curve_by_trial_pybind.png')

for f in [state_file, solution_comparison, cost_curve]:
    if os.path.exists(f):
        os.remove(f)

print(f"  Cleaned up incomplete data for failed trial")
PYEOF
    fi
    
    echo "  Finished trial at: $(date)" | tee -a ${LOG_FILE}
    
    # Wait 1 second between trials to ensure resources are released
    sleep 1
done

# Display GPU memory status
echo "GPU Memory status:" | tee -a ${LOG_FILE}
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -n1 | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

echo "========================================" | tee -a ${LOG_FILE}
echo "All runs completed at $(date)" | tee -a ${LOG_FILE}
echo "Log file: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}
