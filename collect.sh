#!/bin/bash

# 循环 0 到 127
for i in {0..127}
do
   # 格式化数字为三位数，例如 1 变成 001, 10 变成 010
   formatted_index=$(printf "%03d" $i)
   
   # 计算要使用的显卡 ID (取模 4 得到: 0, 1, 2, 3)
   gpu_id=2

   echo "Processing Index: $i on GPU $gpu_id ..."

   # 修改 time_limit 为 10，动态分配显卡，并在末尾加上 & 让其在后台运行
   CUDA_VISIBLE_DEVICES=$gpu_id python run_cuopt.py \
      --time_limit 30 \
      --index 1 > "dataset_sole/instance_${formatted_index}.txt" &

   # 每提交 4 个任务就阻塞等待它们全部完成，然后再启动下一批
   if [ $(( (i + 1) % 1 )) -eq 0 ]; then
       wait
   fi
done

# 最后的 wait 用于捕获可能未凑满 4 个的剩余任务
# （虽然 128 能被 4 整除，但保留此行是编写脚本的好习惯）
wait

echo "All tasks done!"
