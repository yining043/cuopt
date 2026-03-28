#!/bin/bash

# 循环 0 到 127
for i in {0..127}
do
   # 格式化数字为三位数
   formatted_index=$(printf "%03d" $i)
   
   # 动态计算显卡 ID (取模 3，结果分配给卡 0, 1, 2)
   gpu_id=$((i % 3))

   echo "Processing Index: $i on GPU $gpu_id ..."

   # 恢复使用 $i 作为索引参数，放入后台执行 (&)
   CUDA_VISIBLE_DEVICES=$gpu_id python run_cuopt.py \
      --time_limit 50 \
      --index $i > "dataset_bsf/instance_${formatted_index}.txt" &

   # 每提交 3 个任务就阻塞等待它们全部完成
   if [ $(( (i + 1) % 3 )) -eq 0 ]; then
       wait
   fi
done

# 捕获最后未凑满 3 个的剩余任务
# （128 不能被 3 整除，最后会余下 2 个任务，这个 wait 非常关键）
wait

echo "All tasks done!"