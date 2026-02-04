#!/usr/bin/env bash
set -euo pipefail

# Batch runner for perturb.py
# - Runs perturbation in batches (by deduped optima index)
# - Merges batch JSONL + Excel
# - Runs visualization once on merged Excel
#
# Usage:
#   ./run_perturb.sh [operator_type] [instance_index] [gpu_id] [batch_size]
#
# Examples:
#   ./run_perturb.sh remove_and_insert           # default instance_index=0, GPU from env or default
#   ./run_perturb.sh remove_and_insert 1         # instance_index=1
#   ./run_perturb.sh remove_and_insert 1 2       # instance_index=1, GPU 2

OPERATOR_TYPE="${1:-remove_and_insert}"
INSTANCE_INDEX="${2:-0}"
# Instance path is fixed here; not passed as CLI argument
INSTANCE_PATH="/home/jieyi/cvrp100_uniform.pkl"
GPU_ID="${3:-${CUDA_VISIBLE_DEVICES:-}}"

# Set CUDA_VISIBLE_DEVICES if GPU_ID is provided
if [[ -n "${GPU_ID}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  echo "Using GPU: ${GPU_ID} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Using GPU from environment: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
  echo "WARNING: CUDA_VISIBLE_DEVICES not set. Using default GPU (usually GPU 0)."
  echo "  To specify GPU: ./run_perturb.sh ${OPERATOR_TYPE} ${INSTANCE_INDEX} ${INSTANCE_PATH} <gpu_id>"
  echo "  Or set: export CUDA_VISIBLE_DEVICES=<gpu_id>"
fi

# Config
BATCH_SIZE="${4:-200}"
K=5
RUNS=30
OUT_BASE_DIR="/home/jieyi/cuopt/basin_datasets0_analyze"

# Optional: set to 1 to delete per-batch outputs after merging
CLEAN_BATCH_FILES=1

# CUDA debugging: set to 1 to enable CUDA_LAUNCH_BLOCKING (slower but better error reporting)
# This helps debug CUDA errors like misaligned address
CUDA_DEBUG="${CUDA_LAUNCH_BLOCKING:-0}"
if [[ "${CUDA_DEBUG}" == "1" ]]; then
  export CUDA_LAUNCH_BLOCKING=1
  echo "CUDA_LAUNCH_BLOCKING=1 enabled (slower but better error reporting)"
fi

cd /home/jieyi/cuopt

echo "== Running perturb batches =="
echo "  operator_type: ${OPERATOR_TYPE}"
echo "  instance_index:${INSTANCE_INDEX}"
echo "  instance_path: ${INSTANCE_PATH}"
echo "  batch_size:    ${BATCH_SIZE}"
echo "  k:             ${K}"
echo "  runs/step:     ${RUNS}"
echo "  out_base_dir:  ${OUT_BASE_DIR}"
echo

for ((S=0;;S+=BATCH_SIZE)); do
  BATCH_ID="S${S}_B${BATCH_SIZE}_k${K}_r${RUNS}"
  echo "---- batch start_idx=${S} (batch_id=${BATCH_ID}) ----"
  python -u perturb.py \
    --instance_path "${INSTANCE_PATH}" \
    --instance_index "${INSTANCE_INDEX}" \
    --operator_type "${OPERATOR_TYPE}" \
    --k "${K}" \
    --num_local_search_runs "${RUNS}" \
    --start_idx "${S}" --max_optima "${BATCH_SIZE}" \
    --output_base_dir "${OUT_BASE_DIR}" \
    --batch_id "${BATCH_ID}" \
    || break
  
  # Clean GPU memory between batches to prevent OOM
  echo "  Cleaning GPU memory..."
  python - <<PY
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("  GPU memory cleared")
except ImportError:
    pass
PY
done

# Infer output directory for this instance_index (based on basename(instance_path))
INSTANCE_FILE="$(basename "${INSTANCE_PATH}")"
OUT_DIR="${OUT_BASE_DIR}/${INSTANCE_FILE}#${INSTANCE_INDEX}"
cd "${OUT_DIR}"

echo
echo "== Merging batch outputs in: ${OUT_DIR} =="

JSONL_ALL="${OPERATOR_TYPE}_training_data.ALL_r${RUNS}.jsonl"
XLSX_ALL="${OPERATOR_TYPE}_results.ALL_r${RUNS}.xlsx"

echo "-- Merge (dedup) + optional cleanup"
cd /home/jieyi/cuopt
python -u merge_perturb_batches.py \
  --out_dir "${OUT_DIR}" \
  --operator_type "${OPERATOR_TYPE}" \
  --runs "${RUNS}" \
  --keep last \
  --instance_path "${INSTANCE_PATH}" \
  --instance_index "${INSTANCE_INDEX}" \
  --basin_base_dir "basin_datasets0" \
  --delete_batches
cd "${OUT_DIR}"

echo
echo "== Visualization (once) =="
cd /home/jieyi/cuopt
python - <<PY
from perturb import visualize_perturbation_results
visualize_perturbation_results(
    "${OUT_DIR}/${XLSX_ALL}",
    "${OUT_DIR}",
)
PY

echo
echo "Done."

