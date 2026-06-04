#!/bin/bash
# RL node-selection benchmark: random / policy / policy_eps
# Uses all GPUs in parallel (exclusive). Output logs for plot_avg.py.

set -euo pipefail

gpus=(${GPUS:-2 3})
num_gpus=${#gpus[@]}
runs_per_mode=${RUNS:-8}
time_limit=${TIME_LIMIT:-2}
data_path=${DATA_PATH:-../../cuopt-examples/data/test_cvrp1000_hgs_n128_C250.txt}
data_pt=${DATA_PT:-}
weights=${WEIGHTS:-outputs/rl_run2/best_policy.pt}
k=${K:-16}
epsilon=${EPSILON:-0.1}
out_dir=${OUT_DIR:-outputs/benchmark_rl_2s}

mkdir -p "$out_dir"
echo "Benchmark: ${runs_per_mode} x (random + policy + policy_eps)"
echo "GPUs: ${gpus[*]} | time_limit=${time_limit}s | K=${k} | epsilon=${epsilon} | weights=${weights} | data_pt=${data_pt:-none}"
echo "Output: ${out_dir}/"

task_id=0
launch() {
    local mode=$1
    local run_idx=$2
    local seed=$3
    local gpu_id=${gpus[$((task_id % num_gpus))]}
    local prefix
    case "$mode" in
        random) prefix="random_baseline" ;;
        policy) prefix="policy" ;;
        policy_eps) prefix="policy_eps" ;;
        *) echo "unknown mode: $mode"; exit 1 ;;
    esac
    local out_file="${out_dir}/${prefix}_${run_idx}.txt"
    echo "  -> ${prefix}_${run_idx} on GPU ${gpu_id} (seed=${seed})"
    local extra_args=()
    if [ "$mode" = "policy_eps" ]; then
        extra_args=(--epsilon "$epsilon")
    fi
    if [ -n "$data_pt" ]; then
        extra_args+=(--data_pt "$data_pt")
    fi
    CUDA_VISIBLE_DEVICES=$gpu_id python benchmark_rl.py \
        --mode "$mode" \
        --weights "$weights" \
        --data_path "$data_path" \
        --index 1 \
        --time_limit "$time_limit" \
        --k "$k" \
        --seed "$seed" \
        "${extra_args[@]}" \
        > "$out_file" 2>&1 &
    task_id=$((task_id + 1))
    if (( task_id % num_gpus == 0 )); then
        wait
        echo "  batch done"
    fi
}

for i in $(seq 1 "$runs_per_mode"); do
    launch random "$i" $((1000 + i))
done
for i in $(seq 1 "$runs_per_mode"); do
    launch policy "$i" $((2000 + i))
done
for i in $(seq 1 "$runs_per_mode"); do
    launch policy_eps "$i" $((3000 + i))
done

wait
echo "All benchmark runs finished -> ${out_dir}/"
