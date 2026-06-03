#!/bin/bash

# =================配置区域=================
gpus=(${GPUS:-2 3})
total_tasks=${TOTAL_TASKS:-4}
prefix=${PREFIX:-origin_again}
time_limit=${TIME_LIMIT:-30}
data_path=${DATA_PATH:-../../cuopt-examples/data/test_cvrp1000_hgs_n128_C250.txt}
# =========================================

num_gpus=${#gpus[@]}
mkdir -p outputs
echo "🚀 启动: 共 $total_tasks 个任务，$num_gpus 卡并行... 输出前缀: $prefix, time_limit: ${time_limit}s"

for ((i=1; i<=total_tasks; i++)); do
    gpu_id=${gpus[$(( (i-1) % num_gpus ))]}
    
    echo "  -> 启动任务 $i (GPU: $gpu_id)"

    # 备用参数（如需使用，请加到下方命令中，注意不要在带有 \ 的中间插入注释）：
    # --use \
    # --v v2 \
    # --policy outputs/conc_bybrid_new_value_20260326_042847/checkpoint_epoch_200.pt \

    CUDA_VISIBLE_DEVICES=$gpu_id python run_cuopt.py \
        --data_path "$data_path" \
        --index 1 \
        --time_limit "$time_limit" \
        > "./outputs/${prefix}_${i}.txt" &

    # 分批等待逻辑：当达到 GPU 数量的整数倍且不是最后一个任务时，触发 wait
    (( i % num_gpus == 0 && i < total_tasks )) && wait && echo "✅ 批次完成，开始下一批！"
done

wait
echo "🎉 所有任务已全部完成！"
