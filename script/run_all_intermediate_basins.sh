#!/bin/bash
# Batch processing script for analyze_intermediate_basins.py
# Each trial runs independently to completely avoid GPU memory accumulation
# Supports resume: removes incomplete trial data from JSONL and continues

# Note: We don't use 'set -e' because we want to continue processing even if individual trials fail
# Instead, we check exit codes explicitly for critical operations

PKL_FILE="/home/jieyi/cvrp100_uniform.pkl"
HGS_FILE="/home/jieyi/hgs_cvrp100_uniform.pkl"
INSTANCE_IDX=${1:-0}  # Default to 0 if not provided
CUDA_DEVICE=${2:-0}  # Default to 0 if not provided
RUN_ID=${3:-""}  # Optional: specific run_id to process, or empty for all
NUM_RUNS=100
MAX_ITER=100
BASIN_DIR="basin_datasets0"

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}

# Log file (fixed name for resume support)
LOG_FILE="pybind_basin_analysis_idx${INSTANCE_IDX}.log"

echo "========================================" | tee -a ${LOG_FILE}
echo "Starting batch processing at $(date)" | tee -a ${LOG_FILE}
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
# 1. The comprehensive visualization PNG exists
# 2. The Excel file exists
# 3. All intermediate states in training_data_pybind.jsonl are present

def is_trial_complete(instance_path, instance_index, run_id, trial_id, basin_base_dir):
    """Check if a trial is completely analyzed."""
    basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
    output_dir = os.path.join("basin_datasets0_analyze", basin_paths['instance_id'])
    pybind_dir = os.path.join(output_dir, 'pybind_basin_analysis')
    
    # Check for visualization file
    viz_file = os.path.join(pybind_dir, f'pybind_basins_run_{run_id}_trial_{trial_id}.png')
    if not os.path.exists(viz_file):
        return False
    
    # Check for Excel file
    excel_file = os.path.join(pybind_dir, f'basin_analysis_run_{run_id}_trial_{trial_id}.xlsx')
    if not os.path.exists(excel_file):
        return False
    
    # Check training_data_pybind.jsonl
    training_file = os.path.join(pybind_dir, 'training_data_pybind.jsonl')
    if not os.path.exists(training_file):
        return False
    
    # Count how many records exist for this trial
    # We need to check if the trial has been fully processed
    # Since we don't know the exact number of intermediate states beforehand,
    # we'll consider it complete if the visualization and Excel exist
    # (they are created only after all intermediate states are processed)
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
from analyze_basin import get_all_run_trial_pairs, get_basin_paths

instance_path = "/home/jieyi/cvrp100_uniform.pkl"
instance_index = int(os.environ.get('INSTANCE_IDX', '0'))
basin_base_dir = "basin_datasets0"
run_id_filter = os.environ.get('RUN_ID', '')

basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
all_pairs = get_all_run_trial_pairs(basin_paths['trajectory_path'])

if run_id_filter:
    all_pairs = [(r, t) for r, t in all_pairs if r == run_id_filter]

def is_trial_complete(instance_path, instance_index, run_id, trial_id, basin_base_dir):
    """Check if a trial is completely analyzed - consistent with first check."""
    basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
    output_dir = os.path.join("basin_datasets0_analyze", basin_paths['instance_id'])
    pybind_dir = os.path.join(output_dir, 'pybind_basin_analysis')
    viz_file = os.path.join(pybind_dir, f'pybind_basins_run_{run_id}_trial_{trial_id}.png')
    excel_file = os.path.join(pybind_dir, f'basin_analysis_run_{run_id}_trial_{trial_id}.xlsx')
    # Check both files exist (training_data_pybind.jsonl check is optional since it's cumulative)
    return os.path.exists(viz_file) and os.path.exists(excel_file)

# Sort by run_id first, then by trial_id to ensure trials of the same run are grouped together
# Also remove duplicates to avoid processing the same trial multiple times
seen_pairs = set()
for run_id, trial_id in sorted(all_pairs):
    pair_key = (run_id, trial_id)
    if pair_key not in seen_pairs and not is_trial_complete(instance_path, instance_index, run_id, trial_id, basin_base_dir):
        seen_pairs.add(pair_key)
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
# Use process substitution to avoid subshell issues with while read
while IFS= read -r trial_pair; do
    if [ -z "$trial_pair" ]; then
        continue  # Skip empty lines
    fi
    CURRENT=$((CURRENT + 1))
    RUN_ID=$(echo "$trial_pair" | cut -d: -f1)
    TRIAL_ID=$(echo "$trial_pair" | cut -d: -f2)
    
    # If run_id changed, generate visualizations for previous run
    # Check BEFORE updating PREV_RUN_ID
    if [ -n "$PREV_RUN_ID" ] && [ "$PREV_RUN_ID" != "$RUN_ID" ]; then
        echo ">>> Run ID changed from ${PREV_RUN_ID} to ${RUN_ID}" | tee -a ${LOG_FILE}
        echo "Finished run ${PREV_RUN_ID} at: $(date)" | tee -a ${LOG_FILE}
        
        # Generate run-level visualizations from training_data_pybind.jsonl
        echo "Generating run-level visualizations for run_id=${PREV_RUN_ID}..." | tee -a ${LOG_FILE}
        export INSTANCE_IDX
        export RUN_ID=$PREV_RUN_ID
        python3 -u << 'PYEOF'
import sys
import os
sys.path.insert(0, '/home/jieyi/cuopt')
from analyze_intermediate_basins import generate_run_visualizations_from_jsonl

instance_path = "/home/jieyi/cvrp100_uniform.pkl"
instance_index = int(os.environ.get('INSTANCE_IDX', '0'))
run_id = os.environ.get('RUN_ID', '')
basin_base_dir = "basin_datasets0"
hgs_solution_path = "/home/jieyi/hgs_cvrp100_uniform.pkl"
num_vehicles = 30

# Get num_orders from instance
from test_basin_pybind import create_vrp_instance_from_pkl
vrp_instance = create_vrp_instance_from_pkl(instance_path, instance_index, num_vehicles)
num_orders = vrp_instance['num_orders']
del vrp_instance

try:
    generate_run_visualizations_from_jsonl(
        instance_path, instance_index, run_id,
        basin_base_dir, hgs_solution_path, num_orders, num_vehicles
    )
except Exception as e:
    print(f"  Error generating visualizations: {e}")
    import traceback
    traceback.print_exc()
PYEOF
        
        echo "" | tee -a ${LOG_FILE}
        sleep 2
    elif [ -z "$PREV_RUN_ID" ]; then
        echo ">>> Starting first run: ${RUN_ID}" | tee -a ${LOG_FILE}
    fi
    
    # Update PREV_RUN_ID AFTER checking for run_id change
    PREV_RUN_ID=$RUN_ID
    
    echo "========================================" | tee -a ${LOG_FILE}
    echo "Progress: ${CURRENT}/${TOTAL_TRIALS}" | tee -a ${LOG_FILE}
    echo "Processing run_id=${RUN_ID}, trial_id=${TRIAL_ID}" | tee -a ${LOG_FILE}
    echo "Started at: $(date)" | tee -a ${LOG_FILE}
    echo "========================================" | tee -a ${LOG_FILE}
    
    # Check if trial is already complete before cleaning up
    # Only clean up if trial is NOT complete (to avoid re-processing completed trials)
    echo "Checking if trial is already complete..." | tee -a ${LOG_FILE}
    export INSTANCE_IDX
    export RUN_ID
    export TRIAL_ID
    IS_COMPLETE=$(python3 -u << 'PYEOF'
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
pybind_dir = os.path.join(output_dir, 'pybind_basin_analysis')

# Check if trial is complete (both files exist)
viz_file = os.path.join(pybind_dir, f'pybind_basins_run_{run_id}_trial_{trial_id}.png')
excel_file = os.path.join(pybind_dir, f'basin_analysis_run_{run_id}_trial_{trial_id}.xlsx')

is_complete = os.path.exists(viz_file) and os.path.exists(excel_file)
if is_complete:
    print(f"  Trial is already complete, skipping cleanup", file=sys.stderr)
    print("1")
else:
    print(f"  Trial is not complete, will clean up and process", file=sys.stderr)
    print("0")
PYEOF
    )
    
    if [ "$IS_COMPLETE" = "1" ]; then
        echo "Trial ${RUN_ID}:${TRIAL_ID} is already complete, skipping..." | tee -a ${LOG_FILE}
        echo "  Finished trial at: $(date)" | tee -a ${LOG_FILE}
        continue
    fi
    
    # Clean up incomplete data for this trial before running
    # This ensures we can resume cleanly if the previous run was interrupted
    echo "Cleaning up incomplete data for this trial..." | tee -a ${LOG_FILE}
    export INSTANCE_IDX
    export RUN_ID
    export TRIAL_ID
    python3 -u << 'PYEOF'
import sys
import os
import json
sys.path.insert(0, '/home/jieyi/cuopt')
from utils import get_basin_paths

instance_path = "/home/jieyi/cvrp100_uniform.pkl"
instance_index = int(os.environ.get('INSTANCE_IDX', '0'))
basin_base_dir = "basin_datasets0"
run_id = os.environ.get('RUN_ID', '')
trial_id = int(os.environ.get('TRIAL_ID', '0'))

basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
output_dir = os.path.join("basin_datasets0_analyze", basin_paths['instance_id'])
pybind_dir = os.path.join(output_dir, 'pybind_basin_analysis')
os.makedirs(pybind_dir, exist_ok=True)

# Remove incomplete visualization and Excel files (only if they exist and trial is incomplete)
viz_file = os.path.join(pybind_dir, f'pybind_basins_run_{run_id}_trial_{trial_id}.png')
excel_file = os.path.join(pybind_dir, f'basin_analysis_run_{run_id}_trial_{trial_id}.xlsx')

for f in [viz_file, excel_file]:
    if os.path.exists(f):
        os.remove(f)
        print(f"  Removed: {f}")

# Remove incomplete records from training_data_pybind.jsonl
training_file = os.path.join(pybind_dir, 'training_data_pybind.jsonl')
if os.path.exists(training_file):
    lines_to_keep = []
    removed_count = 0
    with open(training_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if not (data.get('run_id') == run_id and data.get('trial_id') == trial_id):
                        lines_to_keep.append(line)
                    else:
                        removed_count += 1
                except:
                    lines_to_keep.append(line)  # Keep malformed lines
    
    if removed_count > 0:
        with open(training_file, 'w') as f:
            f.writelines(lines_to_keep)
        print(f"  Removed {removed_count} incomplete records from {training_file}")
    else:
        print(f"  No incomplete records found in {training_file}")
else:
    print(f"  Training file does not exist yet: {training_file}")

print(f"  Cleanup complete for run_id={run_id}, trial_id={trial_id}")
PYEOF
    
    # Run analysis
    # Capture exit code correctly even with pipe
    python3 -u analyze_intermediate_basins.py \
        --pkl ${PKL_FILE} \
        --idx ${INSTANCE_IDX} \
        --run_id ${RUN_ID} \
        --trial_id ${TRIAL_ID} \
        --n_runs ${NUM_RUNS} \
        --max_iter ${MAX_ITER} \
        --basin_dir ${BASIN_DIR} \
        --hgs ${HGS_FILE} 2>&1 | tee -a ${LOG_FILE}
    
    EXIT_CODE=${PIPESTATUS[0]}  # Get exit code of python3, not tee
    
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
import json
sys.path.insert(0, '/home/jieyi/cuopt')
from utils import get_basin_paths

instance_path = "/home/jieyi/cvrp100_uniform.pkl"
instance_index = int(os.environ.get('INSTANCE_IDX', '0'))
basin_base_dir = "basin_datasets0"
run_id = os.environ.get('RUN_ID', '')
trial_id = int(os.environ.get('TRIAL_ID', '0'))

basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
output_dir = os.path.join("basin_datasets0_analyze", basin_paths['instance_id'])
pybind_dir = os.path.join(output_dir, 'pybind_basin_analysis')

# Remove incomplete files
viz_file = os.path.join(pybind_dir, f'pybind_basins_run_{run_id}_trial_{trial_id}.png')
excel_file = os.path.join(pybind_dir, f'basin_analysis_run_{run_id}_trial_{trial_id}.xlsx')

for f in [viz_file, excel_file]:
    if os.path.exists(f):
        os.remove(f)

# Remove incomplete records from JSONL
training_file = os.path.join(pybind_dir, 'training_data_pybind.jsonl')
if os.path.exists(training_file):
    lines_to_keep = []
    with open(training_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if not (data.get('run_id') == run_id and data.get('trial_id') == trial_id):
                        lines_to_keep.append(line)
                except:
                    lines_to_keep.append(line)
    
    with open(training_file, 'w') as f:
        f.writelines(lines_to_keep)

print(f"  Cleaned up incomplete data for failed trial")
PYEOF
    fi
    
    echo "  Finished trial at: $(date)" | tee -a ${LOG_FILE}
    
    # Wait 1 second between trials to ensure resources are released
    sleep 1
done <<EOF
$TRIALS_TO_PROCESS
EOF

# Generate visualizations for the last run
if [ -n "$PREV_RUN_ID" ]; then
    echo "Finished run ${PREV_RUN_ID} at: $(date)" | tee -a ${LOG_FILE}
    
    # Generate run-level visualizations from training_data_pybind.jsonl
    echo "Generating run-level visualizations for run_id=${PREV_RUN_ID}..." | tee -a ${LOG_FILE}
    export INSTANCE_IDX
    export RUN_ID=$PREV_RUN_ID
    python3 -u << 'PYEOF'
import sys
import os
sys.path.insert(0, '/home/jieyi/cuopt')
from analyze_intermediate_basins import generate_run_visualizations_from_jsonl

instance_path = "/home/jieyi/cvrp100_uniform.pkl"
instance_index = int(os.environ.get('INSTANCE_IDX', '0'))
run_id = os.environ.get('RUN_ID', '')
basin_base_dir = "basin_datasets0"
hgs_solution_path = "/home/jieyi/hgs_cvrp100_uniform.pkl"
num_vehicles = 30

# Get num_orders from instance
from test_basin_pybind import create_vrp_instance_from_pkl
vrp_instance = create_vrp_instance_from_pkl(instance_path, instance_index, num_vehicles)
num_orders = vrp_instance['num_orders']
del vrp_instance

try:
    generate_run_visualizations_from_jsonl(
        instance_path, instance_index, run_id,
        basin_base_dir, hgs_solution_path, num_orders, num_vehicles
    )
except Exception as e:
    print(f"  Error generating visualizations: {e}")
    import traceback
    traceback.print_exc()
PYEOF
    
    echo "" | tee -a ${LOG_FILE}
fi

# Display GPU memory status
echo "GPU Memory status:" | tee -a ${LOG_FILE}
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -n1 | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

echo "========================================" | tee -a ${LOG_FILE}
echo "All runs completed at $(date)" | tee -a ${LOG_FILE}
echo "Log file: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}
