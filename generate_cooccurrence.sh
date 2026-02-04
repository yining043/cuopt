#!/bin/bash
#
# Generate basin co-occurrence data and visualizations for specified instances
# This script:
# 1. Generates basin pairs from training data
# 2. Visualizes top 100 weight basin networks
# 3. Creates summary statistics Excel file
#
# Usage:
#   ./generate_cooccurrence.sh                    # Process all instances (0-10)
#   ./generate_cooccurrence.sh 0                  # Process only instance 0
#   ./generate_cooccurrence.sh 0 1 2             # Process instances 0, 1, 2
#   ./generate_cooccurrence.sh 0-5                # Process instances 0 to 5
#   ./generate_cooccurrence.sh 0-10               # Process instances 0 to 10
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse command line arguments
if [ $# -eq 0 ]; then
    # No arguments: process all instances 0-10
    INSTANCE_INDICES="0-10"
else
    # Parse arguments
    INSTANCE_INDICES=""
    for arg in "$@"; do
        if [[ "$arg" =~ ^[0-9]+-[0-9]+$ ]]; then
            # Range format: start-end
            START=$(echo "$arg" | cut -d'-' -f1)
            END=$(echo "$arg" | cut -d'-' -f2)
            if [ -z "$INSTANCE_INDICES" ]; then
                INSTANCE_INDICES="$START-$END"
            else
                INSTANCE_INDICES="$INSTANCE_INDICES,$START-$END"
            fi
        elif [[ "$arg" =~ ^[0-9]+$ ]]; then
            # Single number
            if [ -z "$INSTANCE_INDICES" ]; then
                INSTANCE_INDICES="$arg"
            else
                INSTANCE_INDICES="$INSTANCE_INDICES,$arg"
            fi
        else
            echo "ERROR: Invalid argument: $arg"
            echo "Usage: $0 [instance_index...] or [start-end]"
            echo "Examples:"
            echo "  $0           # Process all instances 0-10"
            echo "  $0 0         # Process only instance 0"
            echo "  $0 0 1 2     # Process instances 0, 1, 2"
            echo "  $0 0-5       # Process instances 0 to 5"
            exit 1
        fi
    done
fi

echo "============================================================"
echo "Basin Co-occurrence Generation Pipeline"
echo "============================================================"
echo "Instance range: $INSTANCE_INDICES"
echo ""

# Step 1: Generate basin pairs for specified instances
echo "Step 1: Generating basin pairs from training data..."
echo "------------------------------------------------------------"
python generate_basin_pairs.py --instances "$INSTANCE_INDICES"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to generate basin pairs"
    exit 1
fi
echo ""

# Step 2: Visualize basin networks for specified instances
echo "Step 2: Visualizing top 100 weight basin networks..."
echo "------------------------------------------------------------"
python visualize_basin_pairs.py --instances "$INSTANCE_INDICES"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to visualize basin networks"
    exit 1
fi
echo ""

# Step 3: Generate summary statistics
echo "Step 3: Generating summary statistics..."
echo "------------------------------------------------------------"
python summarize_basin_statistics.py --instances "$INSTANCE_INDICES"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to generate summary statistics"
    exit 1
fi
echo ""

echo "============================================================"
echo "Pipeline completed successfully!"
echo "============================================================"
echo ""
echo "Generated files:"
echo "  - Basin pairs: basin_datasets0_analyze/cvrp100_uniform.pkl#*/basin_pairs.jsonl"
echo "  - Basin info: basin_datasets0_analyze/cvrp100_uniform.pkl#*/basin_info.jsonl"
echo "  - Network visualizations: basin_datasets0_analyze/cvrp100_uniform.pkl#*/basin_network_top100.png"
echo "  - Summary statistics: basin_statistics_summary.xlsx"
echo ""
