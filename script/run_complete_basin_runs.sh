#!/usr/bin/env bash
#
# Complete instances 0-99 in basin_datasets0 that have fewer than 100 runs.
# Uses callback/test_basin.py, writes to --global_root=basin_datasets0, round-robin across GPU 2 and 3.
#
# Usage:
#   ./run_complete_basin_runs.sh
#   DATA_DIR=basin_datasets0 INSTANCE_PATH=/path/to.pkl HGS_PATH=/path/to_hgs.pkl ./run_complete_basin_runs.sh
#
set -euo pipefail

cd /home/jieyi/cuopt

# Overridable
DATA_DIR="${DATA_DIR:-basin_datasets0}"
INSTANCE_PATH="${INSTANCE_PATH:-/home/jieyi/cvrp100_uniform.pkl}"
HGS_PATH="${HGS_PATH:-/home/jieyi/hgs_cvrp100_uniform.pkl}"
GPUS=(2 3)  # Only use GPU 2 and 3
NUM_GPUS=${#GPUS[@]}
TIME_LIMIT="${TIME_LIMIT:-2}"
TARGET_RUNS="${TARGET_RUNS:-100}"

# Hardcoded list of instances to complete: "instance_index n_runs_to_add"
INSTANCE_BASENAME="$(basename "${INSTANCE_PATH}")"
echo "Completing instances in ${DATA_DIR} (instance=${INSTANCE_BASENAME})..."

TODO="18 73
20 79
29 57
35 84
41 24
54 96
64 63
80 100
81 100
82 100
83 100
84 100
85 100
86 100
87 100
88 100
89 100
90 100
91 100
92 100
93 100
94 100
95 100
96 100
97 100
98 100
99 100"

echo "Instances to complete:"
echo "${TODO}"
echo ""

job_index=0
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  i="${line%% *}"
  n="${line##* }"
  [[ -z "$i" || -z "$n" || "$n" -le 0 ]] && continue

  gpu_idx=$((job_index % NUM_GPUS))
  gpu=${GPUS[$gpu_idx]}
  instance_id="${INSTANCE_BASENAME}#${i}"
  log_dir="${DATA_DIR}/${instance_id}"
  mkdir -p "${log_dir}"
  log_file="${log_dir}/run_complete_gpu${gpu}.log"

  echo ">>> GPU ${gpu}, instance_index ${i}, n_cuopt_runs=${n} -> ${log_file}"

  CUDA_VISIBLE_DEVICES=${gpu} python -u test_basin.py \
    --time_limit="${TIME_LIMIT}" \
    --instance_path="${INSTANCE_PATH}" \
    --hgs_solution_path="${HGS_PATH}" \
    --n_cuopt_runs="${n}" \
    --instance_index="${i}" \
    --visualize \
    >"${log_file}" 2>&1 &

  ((job_index++)) || true
  # Wait every NUM_GPUS jobs to avoid multiple processes per GPU
  if (( job_index % NUM_GPUS == 0 )); then
    wait
  fi
done <<< "${TODO}"

wait
echo "All completion jobs finished. Re-run count_basin_runs.py to verify."
