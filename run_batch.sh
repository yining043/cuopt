#!/bin/bash

# =================配置区域=================
# 1. 显卡 ID 列表 (例如：4张卡并行)
gpus=(0 1 2 3)

# 2. 总共要跑的任务数
total_tasks=16

# 3. 自定义输出文件的前缀名 (例如设置为 "exp_cuopt"，输出就会变成 exp_cuopt_1.txt, exp_cuopt_2.txt...)
output_prefix="test_new_model"
# =========================================

num_gpus=${#gpus[@]}

echo "🚀 准备开始：共 $total_tasks 个任务，每次并行 $num_gpus 个..."
echo "📂 输出文件将命名为: ${output_prefix}_1.txt, ${output_prefix}_2.txt ..."

for ((i=1; i<=total_tasks; i++)); do
    
    # 计算分配的 GPU
    gpu_idx=$(( (i - 1) % num_gpus ))
    gpu_id=${gpus[$gpu_idx]}

    echo "  -> 启动任务 index=$i (GPU: $gpu_id) -> 写入 ${output_prefix}_${i}.txt"

    # 执行命令，使用 ${output_prefix} 替换原来的写死的名字
    CUDA_VISIBLE_DEVICES=$gpu_id python run_cuopt.py \
        --index 1 \
        --time 30 \
	--use \
	--v v2 \
        --policy outputs/conc_only_new_20260324_085704/checkpoint_epoch_70.pt\
        > ../cuopt-examples/code/${output_prefix}_${i}.txt &

    # 分批等待逻辑
    if (( i % num_gpus == 0 )); then
        if (( i < total_tasks )); then
            echo "⏳ --- 第 $((i / num_gpus)) 批任务已满，等待当前 ${num_gpus} 个任务跑完... ---"
            wait
            echo "✅ 第 $((i / num_gpus)) 批结束，开始下一批！"
        fi
    fi

done

wait
echo "🎉 所有任务已全部完成！"
