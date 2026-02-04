# 从Basin数据集加载Solution作为初始解

## 功能说明

`test_basin_pybind.py` 现在支持从之前运行的trials中加载solution作为初始解，用于复现或继续优化轨迹。

## 使用方法

### 方法1: 从optima.jsonl加载最终解（推荐）

```bash
python3 test_basin_pybind.py \
    --pkl /home/jieyi/cvrp100_uniform.pkl \
    --idx 0 \
    --vehicle 30 \
    --optima /home/jieyi/cuopt/basin_datasets0/cvrp100_uniform.pkl#0/optima.jsonl \
    --optimum_id 0 \
    --hgs /home/jieyi/hgs_cvrp100_uniform.pkl
```

### 方法2: 通过trials.jsonl自动查找optimum_id

```bash
python3 test_basin_pybind.py \
    --pkl /home/jieyi/cvrp100_uniform.pkl \
    --idx 0 \
    --vehicle 30 \
    --trials /home/jieyi/cuopt/basin_datasets0/cvrp100_uniform.pkl#0/trials.jsonl \
    --trial_id 0 \
    --optima /home/jieyi/cuopt/basin_datasets0/cvrp100_uniform.pkl#0/optima.jsonl \
    --hgs /home/jieyi/hgs_cvrp100_uniform.pkl
```

### 方法3: 从trajectory.jsonl加载中间解（实验性）

注意：trajectory中的solution_flat格式可能无法完全转换为routes，如果失败会自动尝试从optima加载。

```bash
python3 test_basin_pybind.py \
    --pkl /home/jieyi/cvrp100_uniform.pkl \
    --idx 0 \
    --vehicle 30 \
    --trajectory /home/jieyi/cuopt/basin_datasets0/cvrp100_uniform.pkl#0/trajectory.jsonl \
    --trial_id 0 \
    --global_iter 5 \
    --trials /home/jieyi/cuopt/basin_datasets0/cvrp100_uniform.pkl#0/trials.jsonl \
    --optima /home/jieyi/cuopt/basin_datasets0/cvrp100_uniform.pkl#0/optima.jsonl \
    --hgs /home/jieyi/hgs_cvrp100_uniform.pkl
```

## 参数说明

- `--optima`: optima.jsonl文件路径，包含最终解的edges信息
- `--optimum_id`: 要加载的optimum ID（对应trial的最终解）
- `--trials`: trials.jsonl文件路径，用于映射trial_id到optimum_id
- `--trial_id`: Trial ID
- `--trajectory`: trajectory.jsonl文件路径，包含中间解的solution_flat
- `--global_iter`: 全局迭代次数（用于trajectory）
- `--local_iter`: 局部迭代次数（用于trajectory）

## 注意事项

1. **optima.jsonl格式**: 包含edges信息，可以转换为routes，但edges格式可能不完整，某些节点可能被添加为单节点route
2. **trajectory.jsonl格式**: solution_flat格式可能无法完全转换，建议优先使用optima.jsonl
3. **Solution验证**: 加载的solution会自动验证可行性（容量约束、所有订单都被服务）

## 示例输出

```
🚀 cuOpt VRP Local Search Test
Using instance from: /home/jieyi/cvrp100_uniform.pkl (index=0)

📂 Loading initial solution via trials.jsonl -> optima.jsonl (trial_id=0)
  Found optimum_id=0 for trial_id=0
  Loaded solution with 11 routes, cost=1453.42

✓ Initial solution validated: Solution is feasible
✓ Initial solution - Cost: 1453.42
...
```



