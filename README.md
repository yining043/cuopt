# run_cuopt.py 使用说明

cuOpt CVRP 求解与收敛曲线画图脚本，支持 early stop callback。

## 数据

- **pkl**：`--problem_path`、`--solution_path`（默认见脚本内路径）
- **txt**：`--data_path`（原始格式）

其他常用：`--n_instances`、`--start_index`、`--time_limit`、`--n_vehicles`、`--scale`。

---

## 子命令

### 1. solve — 跑求解

```bash
# 只求解，不存 log、不画图
python run_cuopt.py solve --n_instances 10 --time_limit 5

# 求解 + 存 log + 自动画图（输出到 curves/<时间戳>/）
python run_cuopt.py solve --n_instances 10 --time_limit 5 --log run.log --plot
```

- 加 `--use_callback` 启用 early stop；`--early_stop_base` 可选 `random` / `embedding` / `structure`。
- 用 `--log` 或 `--plot` 时，结果写入 **curves/<时间戳>/**（含 log 与 4 张图）。

### 2. plot — 用 log 重新画图

```bash
python run_cuopt.py plot curves/20260218_055012/log.txt -o out.png
python run_cuopt.py plot a.log b.log --labels "A,B" --ymin 14 --ymax 16
```

- 不跑求解，只根据已有 log 画图；可多文件、可调 `--ymin`/`--ymax` 等。

### 3. curves — 跑 3 配置并画曲线

```bash
python run_cuopt.py curves --n_instances 5 --time_limit 5
```

- 自动跑 no ES / emb+lr / struct+lr，生成 **curves/<时间戳>/**，内含 `log.txt` 与 4 张图，该 log 可用 `plot` 重画。

### 4. compare — 跑 5 配置并画箱线图

```bash
python run_cuopt.py compare --n_instances 3 --n_repeat 10
```

- 跑 5 种 early-stop 配置，每配置重复多次，按 instance 输出 cost 箱线图、统计图、JSON；结果在 **curves/<时间戳>/**。

---

## 输出目录

| 子命令   | 有输出时目录 |
|----------|----------------|
| solve    | `curves/<时间戳>/`（需 `--log` 或 `--plot`） |
| curves   | `curves/<时间戳>/` |
| compare  | `curves/<时间戳>/` |
| plot     | 由 `-o`/`--out` 等指定，不自动建时间戳目录 |

可通过 `--out_dir` 指定父目录（默认 `.`），时间戳目录为 `out_dir/curves/<时间戳>/`。

## 查看帮助

```bash
python run_cuopt.py -h
python run_cuopt.py solve -h
python run_cuopt.py plot -h
```
