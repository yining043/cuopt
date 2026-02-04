#!/usr/bin/env bash
set -euo pipefail

# k=1 variance experiment: repeat k=1 perturbation multiple times on the same
# set of local optima to assess stability (return_prob, jaccard, broken_pairs).
# Outputs go to k1_variance_rep1/, k1_variance_rep2/, ... for analysis with
# analyze_k1_repeats.py (--repeats_base_dir k1_variance).
#
# Usage:
#   ./run_k1_variance.sh [instance_index] [gpu_id] [num_optima] [repeats]
#
# Defaults:
#   instance_index = 1
#   gpu_id         = (from CUDA_VISIBLE_DEVICES or default GPU)
#   num_optima     = 50
#   repeats        = 10

INSTANCE_INDEX="${1:-1}"
GPU_ID="${2:-${CUDA_VISIBLE_DEVICES:-}}"
NUM_OPTIMA="${3:-50}"
REPEATS="${4:-10}"

OPERATOR_TYPE="remove_and_insert"
INSTANCE_PATH="/home/jieyi/cvrp100_uniform.pkl"

# Base directory for repeats (each repeat: k1_variance_rep1, k1_variance_rep2, ...)
REPEATS_BASE_DIR="/home/jieyi/cuopt/k1_variance"

K=1
RUNS=30

if [[ -n "${GPU_ID}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  echo "Using GPU: ${GPU_ID} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
else
  echo "Using default GPU (CUDA_VISIBLE_DEVICES not explicitly set)"
fi

cd /home/jieyi/cuopt

# Activate conda environment
set +u
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate cuopt_dev 2>/dev/null || {
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate cuopt_dev 2>/dev/null || true
}
set -u

echo "== k=1 variance experiment =="
echo "  operator_type:   ${OPERATOR_TYPE}"
echo "  instance_index:  ${INSTANCE_INDEX}"
echo "  instance_path:   ${INSTANCE_PATH}"
echo "  repeats_base:    ${REPEATS_BASE_DIR}"
echo "  k:               ${K}"
echo "  runs/step:       ${RUNS}"
echo "  num_optima:      ${NUM_OPTIMA}"
echo "  repeats:         ${REPEATS}"
echo

for ((r = 1; r <= REPEATS; r++)); do
  OUT_BASE_DIR="${REPEATS_BASE_DIR}_rep${r}"
  BATCH_ID="rep${r}"
  SEED="${r}"
  echo "---- Repeat ${r}/${REPEATS} (out_dir=${OUT_BASE_DIR}, batch_id=${BATCH_ID}, seed=${SEED}) ----"

  python -u perturb.py \
    --instance_path "${INSTANCE_PATH}" \
    --instance_index "${INSTANCE_INDEX}" \
    --operator_type "${OPERATOR_TYPE}" \
    --k "${K}" \
    --num_local_search_runs "${RUNS}" \
    --start_idx 0 \
    --max_optima "${NUM_OPTIMA}" \
    --output_base_dir "${OUT_BASE_DIR}" \
    --batch_id "${BATCH_ID}" \
    --seed "${SEED}"

  echo "  Cleaning GPU memory..."
  python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
except: pass
try:
    import cupy
    mempool = cupy.get_default_memory_pool()
    mempool.free_all_blocks()
except: pass
" 2>/dev/null || true
  echo
done

echo "========================================"
echo "Done. Run analyze_k1_repeats.py to plot variance:"
echo "  python analyze_k1_repeats.py --repeats_base_dir ${REPEATS_BASE_DIR} --instance_index ${INSTANCE_INDEX} --repeats ${REPEATS}"
echo "========================================"
