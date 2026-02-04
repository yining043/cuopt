#!/usr/bin/env bash
set -euo pipefail

# K=1 sample collector with batch processing, merge, verify, and cleanup
#
# Usage:
#   ./run_perturb_k1_collect.sh [instance_index] [gpu_id] [batch_size] [max_first_runs]
#
# Examples:
#   ./run_perturb_k1_collect.sh 0        # instance_index=0, default GPU
#   ./run_perturb_k1_collect.sh 1 2 100  # instance_index=1, GPU 2, batch_size=100
#   ./run_perturb_k1_collect.sh 0 "" 200 10   # only basins that appear in first 10 runs

INSTANCE_INDEX="${1:-0}"
GPU_ID="${2:-${CUDA_VISIBLE_DEVICES:-}}"
BATCH_SIZE="${3:-200}"
MAX_FIRST_RUNS="${4:-}"   # optional: only run on basins from first N runs (e.g. 10)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPERATOR_TYPE="remove_and_insert"
INSTANCE_PATH="/home/jieyi/cvrp100_uniform.pkl"
RUNS=30
BASIN_BASE_DIR="${SCRIPT_DIR}/basin_datasets0"
EXISTING_DATA_DIR="${SCRIPT_DIR}/basin_datasets0_analyze"
OUT_BASE_DIR="${SCRIPT_DIR}/perturb_k1_collect"

# Derived paths
INSTANCE_FILE="$(basename "${INSTANCE_PATH}")"
K5_OUT_DIR="${EXISTING_DATA_DIR}/${INSTANCE_FILE}#${INSTANCE_INDEX}"
OUT_DIR="${OUT_BASE_DIR}/${INSTANCE_FILE}#${INSTANCE_INDEX}"
LOG_FILE="${OUT_DIR}/run_k1_collect.log"

# GPU setup
if [[ -n "${GPU_ID}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

cd "${SCRIPT_DIR}"

# Activate conda environment
set +u
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate cuopt_dev 2>/dev/null || true
set -u

# Create output directory and start logging
mkdir -p "${OUT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================"
echo "K=1 Sample Collection"
echo "========================================"
echo "  instance_index:   ${INSTANCE_INDEX}"
echo "  batch_size:       ${BATCH_SIZE}"
echo "  max_first_runs:   ${MAX_FIRST_RUNS:-all}"
echo "  k5_data_dir:      ${K5_OUT_DIR}"
echo "  output_dir:      ${OUT_DIR}"
echo "  log_file:        ${LOG_FILE}"
echo "  GPU:             ${CUDA_VISIBLE_DEVICES:-default}"
echo "========================================"
echo ""

# ============================================
# Step 1: Merge existing k=5 batch files first
# ============================================
echo "== Step 1: Merge existing k=5 batch files =="
K5_JSONL="${K5_OUT_DIR}/${OPERATOR_TYPE}_training_data.ALL_r${RUNS}.jsonl"

if [[ -d "${K5_OUT_DIR}" ]]; then
  python -u merge_perturb_batches_jsonl_only.py \
    --out_dir "${K5_OUT_DIR}" \
    --operator_type "${OPERATOR_TYPE}" \
    --runs "${RUNS}" \
    --rebuild_xlsx \
    --delete_batches
  echo "  Merged k=5 data and deleted batch files."
else
  echo "  WARNING: k=5 data directory not found: ${K5_OUT_DIR}"
fi
echo ""

# ============================================
# Step 2: Run k=1 collection in batches (to avoid GPU OOM)
# ============================================
echo "== Step 2: Run k=1 collection =="

# Get total optima count (optionally only basins from first N runs)
BASIN_DIR="${BASIN_BASE_DIR}/${INSTANCE_FILE}#${INSTANCE_INDEX}"
OPTIMA_FILE="${BASIN_DIR}/optima.jsonl"
TRAJECTORY_FILE="${BASIN_DIR}/trajectory.jsonl"
if [[ -n "${MAX_FIRST_RUNS}" ]]; then
  TOTAL_OPTIMA=$(python3 -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from perturb import load_optima_from_jsonl, load_trajectory_info
optima_file = '${OPTIMA_FILE}'
trajectory_file = '${TRAJECTORY_FILE}'
n_first = ${MAX_FIRST_RUNS}
try:
    optima = load_optima_from_jsonl(optima_file)
    traj = load_trajectory_info(trajectory_file)
    all_run_ids = set()
    for occs in traj.values():
        for occ in occs:
            r = occ.get('run_id')
            if r is not None:
                all_run_ids.add(r)
    def key(r):
        try: return (0, int(r))
        except (TypeError, ValueError): return (1, str(r))
    first_n = set(sorted(all_run_ids, key=key)[:n_first])
    filtered = [o for o in optima if any(occ.get('run_id') in first_n for occ in traj.get(o.get('edges_hash'), []))]
    print(len(filtered))
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
")
  MAX_FIRST_RUNS_ARG="--max_first_runs ${MAX_FIRST_RUNS}"
else
  TOTAL_OPTIMA=$(python3 -c "
import json
hashes = set()
optima_file = '${OPTIMA_FILE}'
try:
    with open(optima_file) as f:
        for line in f:
            rec = json.loads(line)
            hashes.add(rec.get('edges_hash'))
    print(len(hashes))
except Exception as e:
    print(f'Error: {e}', file=__import__('sys').stderr)
    print(10000)
")
  MAX_FIRST_RUNS_ARG=""
fi
echo "  Total unique optima: ${TOTAL_OPTIMA}"
if [[ -n "${MAX_FIRST_RUNS}" && "${TOTAL_OPTIMA}" -eq 0 ]]; then
  echo "  (Hint: with max_first_runs=${MAX_FIRST_RUNS}, 0 optima may mean trajectory missing or no run_id in trajectory)"
fi

for ((S=0; S<TOTAL_OPTIMA; S+=BATCH_SIZE)); do
  BATCH_ID="S${S}_B${BATCH_SIZE}"
  echo "---- batch start_idx=${S} (batch_id=${BATCH_ID}) ----"
  
  python -u perturb_k1_collect.py \
    --instance_path "${INSTANCE_PATH}" \
    --instance_index "${INSTANCE_INDEX}" \
    --basin_base_dir "${BASIN_BASE_DIR}" \
    --existing_data_dir "${EXISTING_DATA_DIR}" \
    --output_base_dir "${OUT_BASE_DIR}" \
    --operator_type "${OPERATOR_TYPE}" \
    --num_local_search_runs "${RUNS}" \
    --start_idx "${S}" \
    --max_optima "${BATCH_SIZE}" \
    --resume \
    ${MAX_FIRST_RUNS_ARG} \
    || { echo "  Batch ${BATCH_ID} finished or failed, continuing..."; }
  
  # Clean GPU memory between batches to prevent OOM
  echo "  Cleaning GPU memory..."
  python3 << 'ENDPY' 2>/dev/null || true
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
except Exception:
    pass
try:
    import cupy
    mempool = cupy.get_default_memory_pool()
    mempool.free_all_blocks()
except Exception:
    pass
ENDPY
done
echo ""

# ============================================
# Step 3: Verify and report (ALL files only)
# ============================================
echo "== Step 3: Verify results =="
python3 << PY
import json
import os
import glob

out_dir = "${OUT_DIR}"
operator = "${OPERATOR_TYPE}"

runs = ${RUNS}
summary_all = os.path.join(out_dir, f"k1_collection_summary_{operator}.ALL_r{runs}.jsonl")
xlsx_all = os.path.join(out_dir, f"k1_collection_summary_{operator}.ALL_r{runs}.xlsx")

# Recompute the target anchor set for this run, so that stats
# only reflect basins under the current max_first_runs filter.
max_first_runs = ${MAX_FIRST_RUNS:-0} if "${MAX_FIRST_RUNS:-}" != "" else 0
use_filter = max_first_runs > 0

allowed_hashes = None
if use_filter:
    basin_dir = "${BASIN_DIR}"
    optima_file = "${OPTIMA_FILE}"
    trajectory_file = "${TRAJECTORY_FILE}"

    # 1) Collect distinct run_ids from trajectory
    all_run_ids = set()
    if os.path.exists(trajectory_file):
        with open(trajectory_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                rid = rec.get("run_id")
                if rid is not None:
                    all_run_ids.add(rid)

    def run_id_key(r):
        try:
            return (0, int(r))
        except (TypeError, ValueError):
            return (1, str(r))

    first_n_run_ids = set(sorted(all_run_ids, key=run_id_key)[: max_first_runs])

    # 2) Build mapping edges_hash -> whether it appears in first N run_ids
    in_first_n = {}
    if os.path.exists(trajectory_file) and first_n_run_ids:
        with open(trajectory_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                h = rec.get("edges_hash")
                rid = rec.get("run_id")
                if not h or rid is None:
                    continue
                if h in in_first_n:
                    continue
                if rid in first_n_run_ids:
                    in_first_n[h] = True

    allowed_hashes = {h for h, ok in in_first_n.items() if ok}

# Count JSONL from ALL (optionally filtered by allowed_hashes)
jsonl_count = 0
success_count = 0
if os.path.exists(summary_all):
    with open(summary_all) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                h = rec.get("anchor_hash")
                if allowed_hashes is not None and h not in allowed_hashes:
                    continue
                jsonl_count += 1
                if rec.get("success"):
                    success_count += 1

# Count Excel rows from ALL (optionally filtered by allowed_hashes).
# Note: JSONL is the authoritative source; Excel count is only for sanity check.
xlsx_count = 0
if os.path.exists(xlsx_all):
    try:
        from openpyxl import load_workbook

        wb = load_workbook(xlsx_all, read_only=True)
        ws = wb.active

        # Try to locate the anchor_hash column from the header row.
        header = [c for c in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
        try:
            anchor_idx = header.index("anchor_hash")
        except ValueError:
            anchor_idx = None

        for row in ws.iter_rows(min_row=2, values_only=True):
            if not row:
                continue
            if allowed_hashes is not None and anchor_idx is not None:
                h = row[anchor_idx]
                if h is not None and isinstance(h, str):
                    h_str = h
                else:
                    h_str = str(h) if h is not None else None
                if h_str is None or h_str not in allowed_hashes:
                    continue
            xlsx_count += 1

        wb.close()
    except Exception:
        # If anything goes wrong, we simply leave xlsx_count as 0
        pass

print(f"  Summary JSONL:  {jsonl_count} records")
print(f"  Success count:  {success_count}")
print(f"  Excel rows:     {xlsx_count}")

if jsonl_count > 0:
    print(f"  Success rate:   {success_count}/{jsonl_count} ({100*success_count/jsonl_count:.1f}%)")
    print("  Status: OK")
else:
    print("  Status: WARNING - no data collected")
PY

# ============================================
# Step 4: Backup results.ALL jsonl (append-only safety)
# ============================================
RESULTS_ALL="${OUT_DIR}/k1_collection_results_${OPERATOR_TYPE}.ALL_r${RUNS}.jsonl"
BACKUP_DIR="${OUT_DIR}/backups"

if [[ -f "${RESULTS_ALL}" ]]; then
  mkdir -p "${BACKUP_DIR}"
  TS="$(date +%Y%m%d_%H%M%S)"
  BACKUP_PATH="${BACKUP_DIR}/k1_collection_results_${OPERATOR_TYPE}.ALL_r${RUNS}.${TS}.jsonl"
  cp "${RESULTS_ALL}" "${BACKUP_PATH}"
  echo "== Step 4: Backup results =="
  echo "  Backed up ${RESULTS_ALL} -> ${BACKUP_PATH}"
else
  echo "== Step 4: Backup results =="
  echo "  WARNING: results ALL file not found, skip backup: ${RESULTS_ALL}"
fi
echo ""

echo "========================================"
echo "Done."
echo "  Output dir: ${OUT_DIR}"
echo "  Log file:   ${LOG_FILE}"
echo "========================================"
