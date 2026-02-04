#!/bin/bash

# Script to run basin statistics analysis for all runs one by one
# Usage: ./run_basin_analysis_all_runs.sh [instance_index] [data_dir]

INSTANCE_INDEX=${1:-0}
DATA_DIR=${2:-/home/jieyi/cuopt/basin_datasets0_analyze}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting basin analysis for instance index: $INSTANCE_INDEX"
echo "Data directory: $DATA_DIR"
echo "Script directory: $SCRIPT_DIR"
echo ""

# Find all unique run IDs from Excel files
INSTANCE_DIR="$DATA_DIR/cvrp100_uniform.pkl#$INSTANCE_INDEX"
echo "Looking for Excel files in: $INSTANCE_DIR"

if [ ! -d "$INSTANCE_DIR" ]; then
    echo "Error: Instance directory not found: $INSTANCE_DIR"
    exit 1
fi

# Extract all run IDs and their trial counts
# Pattern: basin_analysis_run_YYYYMMDD_HHMMSS_trial_X.xlsx
echo "Scanning for complete runs..."
ALL_RUN_TRIALS=$(find "$INSTANCE_DIR" -name "basin_analysis_run_*.xlsx" -type f | \
    grep -v "pybind_basin_analysis" | \
    grep -v "_old" | \
    grep -v "_try" | \
    sed -n 's/.*basin_analysis_run_\([0-9_]*\)_trial_\([0-9]*\)\.xlsx/\1:\2/p')

# Count trials per run and find expected number of trials
declare -A RUN_TRIAL_COUNTS
declare -A RUN_TRIAL_SET
MAX_TRIAL_ID=0

while IFS=: read -r run_id trial_id; do
    if [ -n "$run_id" ] && [ -n "$trial_id" ]; then
        RUN_TRIAL_COUNTS["$run_id"]=$((${RUN_TRIAL_COUNTS["$run_id"]:-0} + 1))
        RUN_TRIAL_SET["$run_id"]="${RUN_TRIAL_SET["$run_id"]} $trial_id"
        if [ "$trial_id" -gt "$MAX_TRIAL_ID" ]; then
            MAX_TRIAL_ID=$trial_id
        fi
    fi
done <<< "$ALL_RUN_TRIALS"

# Expected number of trials (assuming trials are 0-indexed or 1-indexed, use max+1 as estimate)
# Or we can check the most common trial count
EXPECTED_TRIALS=0
if [ ${#RUN_TRIAL_COUNTS[@]} -gt 0 ]; then
    # Find the most common trial count (likely the expected number)
    EXPECTED_TRIALS=$(printf '%s\n' "${RUN_TRIAL_COUNTS[@]}" | sort -n | uniq -c | sort -rn | head -1 | awk '{print $2}')
fi

echo "Expected number of trials per run: $EXPECTED_TRIALS"
echo ""

# Filter to only complete runs (runs with expected number of trials)
COMPLETE_RUN_IDS=""
for run_id in "${!RUN_TRIAL_COUNTS[@]}"; do
    trial_count=${RUN_TRIAL_COUNTS["$run_id"]}
    if [ "$trial_count" -eq "$EXPECTED_TRIALS" ]; then
        COMPLETE_RUN_IDS="$COMPLETE_RUN_IDS $run_id"
    else
        echo "Skipping incomplete run: $run_id (has $trial_count trials, expected $EXPECTED_TRIALS)"
    fi
done

RUN_IDS=$(echo "$COMPLETE_RUN_IDS" | tr ' ' '\n' | grep -v '^$' | sort -u)

if [ -z "$RUN_IDS" ]; then
    echo "Error: No run IDs found in $INSTANCE_DIR"
    exit 1
fi

# Count total complete runs
TOTAL_RUNS=$(echo "$RUN_IDS" | wc -l)
INCOMPLETE_COUNT=$((${#RUN_TRIAL_COUNTS[@]} - TOTAL_RUNS))
echo "Found $TOTAL_RUNS complete run(s) (skipped $INCOMPLETE_COUNT incomplete runs)"
echo ""

# Run analysis for each run
COUNTER=1
for RUN_ID in $RUN_IDS; do
    echo "=========================================="
    echo "[$COUNTER/$TOTAL_RUNS] Processing run: $RUN_ID"
    echo "=========================================="
    
    cd "$SCRIPT_DIR"
    python3 analyze_basin_statistics.py \
        --data_dir "$DATA_DIR" \
        --instance_index "$INSTANCE_INDEX" \
        --run_id "$RUN_ID"
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Successfully processed run: $RUN_ID"
    else
        echo "✗ Error processing run: $RUN_ID (exit code: $EXIT_CODE)"
    fi
    echo ""
    
    COUNTER=$((COUNTER + 1))
done

echo "=========================================="
echo "All runs processed!"
echo "=========================================="
