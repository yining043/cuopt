# merged_perturb_k1_stats.xlsx — 表里记了什么、怎么算的

## 文件结构

- **Summary** 表：1 张总表，**一行一个 instance（0–49）**，列很多是因为每个指标都拆成两列（All runs / First 10 runs）。
- **0, 1, …, 49** 表：每个 instance 一张小表，**行是指标、列只有 3 列**（Metric | All runs | First 10 runs），所以“长”的主要是 Summary 的**列多**，不是行多。

---

## 每个指标是什么意思、怎么算

数据来源：

- **trajectory.jsonl**：每个 (run_id, trial_id) 的“最后一条解”的 `edges_hash` → 视为一个 local optimum；按 run_id 首次出现顺序得到“前 10 个 runs”。
- **k1_collection_summary_*.jsonl**：每个 local optimum 的 k=1 perturb 是否成功（success/failure），以及对应的 run_id 等。

在此基础上，对「全部 runs」和「前 10 个 runs」分别算下面 8 个指标。

| 指标名 | 含义 | 计算方式 |
|--------|------|----------|
| **Number of runs** | 有多少个 run | 全部：trajectory 里出现过的不同 run_id 个数。前 10：前 10 个 run_id。 |
| **Local optima with perturb data** | 有多少个 local optimum 有 perturb 记录 | 在 summary 里、且 run_id 属于当前范围（全部或前 10）的 **不同 anchor_hash** 个数。 |
| **Runs completed (all local optima done)** | 有多少个 run 把“该 run 的所有 local optima”都跑完 | 对每个 run_id，用 trajectory 得到该 run 的 local optima 集合；若 summary 里该 run 的 anchor 数 ≥ 这个集合大小，算 completed。 |
| **Runs incomplete** | 未跑完的 run 数 | 同上，completed 以外的 run 数。 |
| **Among completed: runs all success** | 在已完成的 run 里，全部 perturb 都成功的 run 数 | completed 的 run 里，summary 行中没有任何一条 success=false 的。 |
| **Among completed: runs with at least one fail** | 在已完成的 run 里，至少有一次 perturb 失败的 run 数 | completed 的 run 里，至少有一条 success=false。 |
| **Summary rows (success)** | 当前范围内、成功条数 | summary 里 run_id 在范围内且 success=true 的**行数**。 |
| **Summary rows (failure)** | 当前范围内、失败条数 | summary 里 run_id 在范围内且 success=false 的**行数**。 |

“全部”= 该 instance 在 trajectory 里出现的所有 run；“前 10”= trajectory 里按首次出现顺序的前 10 个 run_id。

---

## 为什么 Summary 表看起来“很长”

- **行**：只有 50 行（每行一个 instance）。
- **列多**：每个指标拆成 2 列（`*_all`、`*_first10`），所以一共约 **1（instance）+ 8×2 = 17 列**，列名又比较长（来自上面的英文指标名），所以表会显得“很长/很宽”。

要看某个 instance 的明细，直接点下面以数字命名的工作表（0–49），那里是 8 行 × 3 列，更短、更易读。
