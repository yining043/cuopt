#!/bin/bash
# Batch: generate distant_basins.jsonl and update basin_info.jsonl.
# Usage: ./run_find_distant_basins_batch.sh [instances]
#   instances: e.g. 0-29 or 0,1,2 (default: 0-29)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INSTANCES="${1:-0-29}"
MAX_RUNS=10
TOP_K=255
MIN_BASIN_TOTAL_COUNT=10
LOG_FILE="find_distant_basins_batch_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo "Batch processing instances: $INSTANCES"
echo "Parameters:"
echo "  max_runs: $MAX_RUNS"
echo "  top_k: $TOP_K"
echo "  min_basin_total_count: $MIN_BASIN_TOTAL_COUNT"
echo "  log_file: $LOG_FILE"
echo "============================================================"
echo ""
echo "Starting at: $(date)"
echo ""

# Run find_distant_basins.py
# Log both to file and stdout
python -u find_distant_basins.py \
    --instances "$INSTANCES" \
    --max_runs "$MAX_RUNS" \
    --top_k "$TOP_K" \
    --min_basin_total_count "$MIN_BASIN_TOTAL_COUNT" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]} 

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Batch processing completed successfully!"
else
    echo "Batch processing failed with exit code: $EXIT_CODE"
fi
echo "Finished at: $(date)"
echo "Log file: $LOG_FILE"
echo "============================================================"

exit $EXIT_CODE
