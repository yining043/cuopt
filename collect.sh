#!/bin/bash

# 循环 1 到 128
for i in {0..127}
do
   # 格式化数字为两位数，例如 1 变成 01, 10 保持 10
   formatted_index=$(printf "%03d" $i)

   echo "Processing Index: $i ..."

   # 执行你的命令
   # 注意：--index 使用的是原始数字 $i，重定向文件名使用的是带 0 的 $formatted_index
   CUDA_VISIBLE_DEVICES=0 python run_cuopt.py \
      --time_limit 50 \
      --index $i > "dataset_anchor/instance_${formatted_index}.txt"

done

echo "All task done!"
