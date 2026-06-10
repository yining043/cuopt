# CVRP RL Pipeline — 交接说明 / Handoff

本文是训练（train）与评估（benchmark）流程的导览,用于工作交接。先读这里,再按
"阅读顺序"深入各文件。

> 任务范畴:cuOpt 仍然负责候选生成、可行性、局部搜索执行;学习到的 policy 只在每次
> local-search callback 从 cuOpt 给出的 `K` 个候选 node/operator subset 中选一个。
> 当前代码只保留两个模型:`v2`(旧基线)和 `v7`(当前 flash head,别名 `flashv7`);
> 中间迭代 `v3`–`v6` 已删除。

---

## 1. 一句话端到端

```text
run_train_cvrp.sh
  └─ train_rl.py (master: 权重/优化器/REINFORCE 更新/eval)
       └─ 每轮启动 N 个 rl_rollout.py (每卡一个 cuOpt Solve, 纯推理收集 transitions)
            └─ model.py CostPredictor(v2/v7) + rl_callback.py 在 callback 里选 arm
  → outputs/<RUNNAME>/policy_round<N>.pt  (+ run.log)

run_benchmark_cvrp.sh
  └─ benchmark_rl.py  (random 基线 + 一个/多个 policy checkpoint)
  → outputs/<OUT_DIR>/{random_seed*.txt, policy_round*_*.txt}
       ├─ summarize_rl_benchmark.py → summary.txt   (mean/min/std + gap vs random)
       └─ plot_avg.py               → plot_avg.png  (平均 best-so-far 曲线)
```

---

## 2. 阅读顺序(推荐)

1. **`train_rl.py` 顶部 docstring**(第 1–23 行)— 训练架构总览:master 进程持有 policy +
   优化器、从不 import cuOpt;每轮 dump 权重 → 跨 GPU 并行启动 rollout worker →
   聚合 full-feedback transitions 做更新 → 周期性 eval。
2. **`rl_rollout.py` 顶部 docstring**(第 1–10 行)— 单个 episode:一次 cuOpt Solve、
   独占一张卡、纯推理、把 transitions + final cost 存盘交给 master 聚合。
3. **`docs/rl_cvrp_h2_ep2_lr3e4_pi_report.md`** — MDP / RL formulation、baseline 结果、
   当前 v7 验证设置(顶部有 v2/v7 合并说明;v4/v5 段落为设计历史)。
4. **`docs/rl_cvrp_flash_network_improvements.md`** — 为什么从 v2 走向 flash head(v7):
   v2 的显存/吞吐瓶颈、SDPA/flash 友好的网络重排。

---

## 3. 文件职责

### 训练侧
| 文件 | 职责 |
| --- | --- |
| `run_train_cvrp.sh` | 操作入口:tmux 启动训练,默认 `v7`,全部超参 env 可覆盖 |
| `train_rl.py` | Master:policy + 优化器、REINFORCE 更新、eval、调度 rollout worker |
| `rl_rollout.py` | Rollout worker:一次 cuOpt Solve(推理),收集 transitions |
| `model.py` | `CostPredictor`(`mode='v2'` 旧基线 / `mode='v7'`=`FlashCandidateHeadV7`),给 K 个候选打分 |
| `rl_callback.py` | `RLPolicyCallback`(callback 内 softmax 选 arm)、`RandomSubsetCallback`(随机基线) |

### 评估 / 分析侧
| 文件 | 职责 |
| --- | --- |
| `run_benchmark_cvrp.sh` | 操作入口:跑 random + 一/多个 checkpoint,自动调 summarize + plot |
| `benchmark_rl.py` | 加载 checkpoint(`--model_mode` v2/v7),`--mode random\|policy`,输出每轮 `cost before/after` 轨迹 + `final_cost=` |
| `summarize_rl_benchmark.py` | 扫描 benchmark 目录算 mean/min/std + gap vs random |
| `plot_avg.py` | 画平均 best-so-far 曲线;按文件名"纯字母前缀"分组成多条线 |
| `profile_policy_network.py` | 不依赖 cuOpt,纯网络前后向延迟/峰值显存对比(v2 vs v7) |

---

## 4. 怎么跑

### 4.1 训练

默认(v7,复现当前验证 run):

```bash
bash run_train_cvrp.sh
```

训练 v2:

```bash
MODE=v2 RUNNAME=rl_cvrp_v2_run AMP_DTYPE=none LOGIT_CLIP=0 ENTROPY_COEF=0.01 \
  bash run_train_cvrp.sh
```

常用 env(默认值见脚本头部):`MODE`(v2/v7)、`GPUS`、`MASTER`、`RUNNAME`、`ROUNDS`、
`TIME_LIMIT`、`K`、`LR`、`TEMPERATURE`、`ENTROPY_COEF`、`LOGIT_CLIP`、`AMP_DTYPE`、
`EVAL_SELECTION`。

产物:`outputs/<RUNNAME>/policy_round<N>.pt`、`outputs/<RUNNAME>/run.log`。
监控:`tmux attach -t <SESSION>`(默认 `rl_cvrp`),或 `tail -f outputs/<RUNNAME>/run.log`。

### 4.2 Benchmark + 统计 + 画图

每个 policy 一条 `POLICIES` 项:`"label|weights|model_mode|round|selection|temperature"`。
`label` 必须是纯字母(作为 plot 分组名 → 一条线)。Random 每次重新跑。

两线(random + 单 policy):

```bash
POLICIES=(
  "Policy|outputs/rl_cvrp_h2_ep2_lr3e4/policy_round143.pt|v2|143|sample|0.8"
) NUM_SEEDS=12 TIME_LIMIT=30 GPUS="0 1 2 3" bash run_benchmark_cvrp.sh
```

三线(random + v2 + v7):

```bash
POLICIES=(
  "Vtwo|outputs/rl_cvrp_h2_ep2_lr3e4/policy_round143.pt|v2|143|sample|0.8"
  "Vseven|outputs/rl_cvrp_flashv7_k16_h2_ep2_t08_e04_clip20_r10/policy_round9.pt|v7|9|sample|0.8"
) NUM_SEEDS=12 TIME_LIMIT=30 GPUS="0 1 2 3" bash run_benchmark_cvrp.sh
```

产物:`outputs/<OUT_DIR>/summary.txt`、`outputs/<OUT_DIR>/plot_avg.png`、
原始 `*.txt`(random / policy 每个 seed 一个)。

常用 env:`SEEDS` 或 `NUM_SEEDS`+`BASE_SEED`、`TIME_LIMIT`、`K`、`GPUS`、`AMP_DTYPE`、
`OUT_DIR`、`YMIN`/`YMAX`/`SEGMENTS`(plot 范围)、`DO_PLOT=0`(跳过画图)。

### 4.3 网络 profiling(可选,不需 cuOpt)

```bash
python profile_policy_network.py --modes v2 v7
```

---

## 5. 文件命名约定(为什么 summarize / plot 能直接吃)

- `summarize_rl_benchmark.py` 解析:`random_seed<seed>.txt`、
  `policy_round<round>_<mode>_seed<seed>.txt`(`mode` = `greedy` | `sample_t<temp>`)。
  按 `(round, mode)` 分组统计;旧命名 `random_<seed>.txt` / `policy_r<round>_<seed>.txt`
  仍兼容。
- `plot_avg.py` 按文件名开头的"纯字母前缀"分组(数字截断)。`run_benchmark_cvrp.sh`
  会在 `plot_avg_inputs/` 建软链:`Random<seed>.txt` + 每个 policy `<Label><seed>.txt`。
  注意:`v2`/`v7` 这类带数字的前缀都会被归成 `V`,所以 label 用 `Vtwo`/`Vseven`
  这种纯字母名才能分开成不同线。

---

## 6. 环境

脚本默认 `conda activate cuopt_dev`(`CONDA_SH`/`CONDA_ENV` 可覆盖)。`outputs/` 被
`.gitignore` 忽略,不进版本库。
