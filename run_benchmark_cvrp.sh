#!/bin/bash
# Standard CVRP RL benchmark.
#
# Runs a fresh random baseline plus one or more policy checkpoints over a
# user-specified set of seeds at a fixed per-solve time limit, then produces:
#   1. summary.txt  via summarize_rl_benchmark.py  (mean/min/std/gap vs random)
#   2. an averaged best-so-far curve via plot_avg.py (one line per series:
#      Random + each policy label; supply 1 policy for a 2-line plot, 2 for a
#      3-line plot, etc.)
#
# Output text files are named so summarize_rl_benchmark.py parses them:
#   random_seed<seed>.txt
#   policy_round<round>_<mode>_seed<seed>.txt   (mode = greedy | sample_t<temp>)
#
# Example (3-line plot: random + v2 + v7):
#   POLICIES=(
#     "Vtwo|outputs/rl_cvrp_h2_ep2_lr3e4/policy_round143.pt|v2|143|sample|0.8"
#     "Vseven|outputs/<v7_run>/policy_round9.pt|v7|9|sample|0.8"
#   ) NUM_SEEDS=12 TIME_LIMIT=30 GPUS="0 1 2 3" bash run_benchmark_cvrp.sh
set -euo pipefail

# --- General config -------------------------------------------------------
OUT_DIR=${OUT_DIR:-outputs/benchmark_cvrp_$(date +%Y%m%d_%H%M%S)}
TIME_LIMIT=${TIME_LIMIT:-30}
K=${K:-16}
AMP_DTYPE=${AMP_DTYPE:-none}
GPUS=${GPUS:-0}

# Seeds: explicit SEEDS list wins; otherwise NUM_SEEDS from BASE_SEED.
NUM_SEEDS=${NUM_SEEDS:-12}
BASE_SEED=${BASE_SEED:-5001}
if [[ -z "${SEEDS:-}" ]]; then
    SEEDS=""
    for ((i = 0; i < NUM_SEEDS; i++)); do
        SEEDS+="$((BASE_SEED + i)) "
    done
fi

# Instance selection (benchmark_rl.py defaults if unset).
DATA_PATH=${DATA_PATH:-../../cuopt-examples/data/test_cvrp1000_hgs_n128_C250.txt}
INDEX=${INDEX:-1}
N_VEHICLES=${N_VEHICLES:-21}
SCALE=${SCALE:-1e2}

# Policy checkpoints. Each entry: "label|weights|model_mode|round|selection|temperature".
# label must be letters only (used as the plot_avg group name).
if [[ -z "${POLICIES:-}" ]]; then
    POLICIES=(
        "Policy|outputs/rl_cvrp_h2_ep2_lr3e4/policy_round143.pt|v2|143|sample|0.8"
    )
fi

# Plot config.
DO_PLOT=${DO_PLOT:-1}
YMIN=${YMIN:-3400}
YMAX=${YMAX:-3800}
SEGMENTS=${SEGMENTS:-10000}

CONDA_SH=${CONDA_SH:-/opt/miniconda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-cuopt_dev}

# --- Helpers --------------------------------------------------------------
mode_token() {
    local selection=$1 temp=$2
    if [[ "${selection}" == "greedy" ]]; then
        echo "greedy"
        return
    fi
    case "${temp}" in
        0.25) echo "sample_t025" ;;
        1|1.0) echo "sample_t1" ;;
        0.8) echo "sample_t08" ;;
        *) echo "sample_t${temp//./}" ;;
    esac
}

sanitize_label() {
    # Keep letters only so plot_avg groups each policy as its own line.
    local raw=$1
    echo "${raw//[^a-zA-Z]/}"
}

# --- Validate policies ----------------------------------------------------
for spec in "${POLICIES[@]}"; do
    IFS='|' read -r label weights model_mode round selection temperature <<< "${spec}"
    if [[ -z "${weights}" || ! -f "${weights}" ]]; then
        echo "Missing checkpoint for policy '${label}': ${weights}" >&2
        exit 1
    fi
done

# --- Environment ----------------------------------------------------------
set +u
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u

GPU_CSV=$(echo "${GPUS}" | tr ' ' ',')
CUDA_VISIBLE_DEVICES="${GPU_CSV}" python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("[benchmark_cvrp] CUDA preflight failed: CUDA unavailable.", file=sys.stderr)
    sys.exit(1)
print(f"[benchmark_cvrp] CUDA preflight ok: visible_devices={torch.cuda.device_count()}")
PY

mkdir -p "${OUT_DIR}"
read -r -a GPU_ARR <<< "${GPUS}"
NUM_GPUS=${#GPU_ARR[@]}
if [[ "${NUM_GPUS}" -lt 1 ]]; then
    echo "No GPUs configured." >&2
    exit 1
fi

# --- Run functions --------------------------------------------------------
run_random() {
    local seed=$1 gpu=$2
    local out="${OUT_DIR}/random_seed${seed}.txt"
    if [[ -s "${out}" ]] && grep -Eq "final_cost=|no feasible" "${out}"; then
        echo "[skip] random seed=${seed}"
        return
    fi
    echo "[run] random seed=${seed} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" python benchmark_rl.py \
        --mode random \
        --data_path "${DATA_PATH}" --index "${INDEX}" \
        --n_vehicles "${N_VEHICLES}" --scale "${SCALE}" \
        --time_limit "${TIME_LIMIT}" --k "${K}" --seed "${seed}" \
        > "${out}"
}

run_policy() {
    local seed=$1 gpu=$2 weights=$3 model_mode=$4 round=$5 selection=$6 temperature=$7
    local mode
    mode=$(mode_token "${selection}" "${temperature}")
    local out="${OUT_DIR}/policy_round${round}_${mode}_seed${seed}.txt"
    if [[ -s "${out}" ]] && grep -Eq "final_cost=|no feasible" "${out}"; then
        echo "[skip] policy round=${round} ${mode} seed=${seed}"
        return
    fi
    echo "[run] policy round=${round} ${mode} seed=${seed} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" python benchmark_rl.py \
        --mode policy \
        --weights "${weights}" \
        --model_mode "${model_mode}" \
        --selection "${selection}" \
        --temperature "${temperature}" \
        --amp_dtype "${AMP_DTYPE}" \
        --data_path "${DATA_PATH}" --index "${INDEX}" \
        --n_vehicles "${N_VEHICLES}" --scale "${SCALE}" \
        --time_limit "${TIME_LIMIT}" --k "${K}" --seed "${seed}" \
        > "${out}"
}

# --- Build flat task list -------------------------------------------------
TASKS=()
for seed in ${SEEDS}; do
    TASKS+=("random|${seed}")
done
for spec in "${POLICIES[@]}"; do
    for seed in ${SEEDS}; do
        TASKS+=("policy|${seed}|${spec}")
    done
done

# --- Dispatch round-robin across GPUs -------------------------------------
task_idx=0
for task in "${TASKS[@]}"; do
    gpu=${GPU_ARR[$((task_idx % NUM_GPUS))]}
    kind=${task%%|*}
    rest=${task#*|}
    if [[ "${kind}" == "random" ]]; then
        run_random "${rest}" "${gpu}" &
    else
        seed=${rest%%|*}
        spec=${rest#*|}
        IFS='|' read -r label weights model_mode round selection temperature <<< "${spec}"
        run_policy "${seed}" "${gpu}" "${weights}" "${model_mode}" "${round}" "${selection}" "${temperature}" &
    fi
    task_idx=$((task_idx + 1))
    if (( task_idx % NUM_GPUS == 0 )); then
        wait
    fi
done
wait

# --- Summary --------------------------------------------------------------
python summarize_rl_benchmark.py "${OUT_DIR}" > "${OUT_DIR}/summary.txt"
cat "${OUT_DIR}/summary.txt"

# --- Plot (Random + one line per policy label) ----------------------------
if [[ "${DO_PLOT}" == "1" ]]; then
    PLOT_INPUT_DIR="${OUT_DIR}/plot_avg_inputs"
    mkdir -p "${PLOT_INPUT_DIR}"
    for seed in ${SEEDS}; do
        ln -sf "../random_seed${seed}.txt" "${PLOT_INPUT_DIR}/Random${seed}.txt"
    done
    for spec in "${POLICIES[@]}"; do
        IFS='|' read -r label weights model_mode round selection temperature <<< "${spec}"
        mode=$(mode_token "${selection}" "${temperature}")
        plabel=$(sanitize_label "${label}")
        for seed in ${SEEDS}; do
            ln -sf "../policy_round${round}_${mode}_seed${seed}.txt" \
                "${PLOT_INPUT_DIR}/${plabel}${seed}.txt"
        done
    done
    MPLCONFIGDIR=/tmp/matplotlib-codex python plot_avg.py "${PLOT_INPUT_DIR}"/*.txt \
        --out "${OUT_DIR}/plot_avg.png" \
        --ymax "${YMAX}" --ymin "${YMIN}" --abs --seg "${SEGMENTS}"
    echo "Plot written to ${OUT_DIR}/plot_avg.png"
fi

echo "Benchmark files written to ${OUT_DIR}"
