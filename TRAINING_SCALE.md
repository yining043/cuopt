# 一个 Training Round 到底发生了什么（从零讲清楚）

这份文档解释 `train_rl.py` 跑一个 round 时的所有量级：收集多少数据、多少
step、多少 episode、多少 local search、batch size 怎么算、网络输入输出维度是
什么。不假设你知道任何背景，所有名词第一次出现都会定义。

下面的"实测数字"来自已经跑完的 CVRP 训练 `outputs/rl_run2/`（150 个 round 的
日志）。CVRPTW 用同样的超参，差异在最后单独列出。

---

## 0. 先定义 4 个层级的名词

从大到小，4 个层级，每个都是上一个的组成部分：

1. **Round（轮）**
   = 训练主循环的一次迭代。一个 round = "收集一批数据 + 更新一次网络"。
   我们计划跑 `--rounds 180` 个 round。

2. **Episode（一局）**
   = 一次完整的 cuOpt 求解（一次 `routing.Solve(...)`），预算 `time_limit=2` 秒。
   每个 round 我们**并行**跑 `--batch_episodes 8` 个 episode（8 个独立子进程，
   分布在多张 GPU 上，各自解同一个问题实例但随机种子不同）。
   - 所以：**1 round = 8 episodes**。

3. **Step（一步）**
   = 在一个 episode（一次求解）内部，cuOpt 的局部搜索会反复迭代。**每一次
   迭代**，我们的策略网络都要做一个决策（从 16 个候选子集里挑一个）。
   这"一次决策"就是一个 RL **step**。
   - 一个 episode 内有多少 step？取决于 2 秒内 cuOpt 迭代了多少次。
     实测 CVRP：**平均每个 episode ≈ 624 个 step**（范围 587–666）。

4. **Local Search（局部搜索）**
   = 在每个 step 内部，cuOpt 实际执行的搜索动作。一个 step 内部包含：
   - 1 次 anchor 发现搜索（找出哪些点可以动）
   - **K = 16 次 probe 搜索**（对 16 个候选子集各试一次，得到每个子集能降多少
     cost，这 16 个数值就是 `trail_rewards`，是 RL 的奖励信号）
   - 1 次真实搜索（把策略选中的那个子集真正应用到解上）

---

## 1. "8 × 624 ≈ 5000 steps" 是怎么来的

这就是上面层级的简单相乘：

```
每个 round 有   8 个 episode          (batch_episodes = 8)
每个 episode 有 ≈ 624 个 step         (实测平均值)
─────────────────────────────────────────────
每个 round 收集 8 × 624 ≈ 4990 ≈ 约 5000 个 step
```

换句话说：一个 round 里，策略网络一共做了 **约 5000 次"从 16 个候选里挑 1 个"
的决策**，这 5000 次决策（连同它们的奖励）就是这一轮收集到的训练数据。

> 为什么是约等于？因为每个 episode 的 step 数不完全一样（587~666 之间波动），
> 8 个加起来在 4700~5300 之间，取整说"约 5000"。

---

## 2. 每个 step 收集了哪些数据

每个 step 存一条记录（叫一条 transition）。因为我们用"全反馈"方式（不只记录
被选中的那个子集，而是记录全部 16 个候选的奖励），所以一条记录里包含 16 个
arm（候选子集）的信息：

| 字段 | 形状 | 含义 |
|------|------|------|
| `masks` | [16, max_length] | 16 个候选子集，每个是一个长 mask |
| `rewards` | [16] | 16 个候选各自的 probe 奖励（cost 降幅）|
| `valid` | [16] | 16 个候选里哪些非空有效 |
| `sol` | [max_length] | 当前解的序列 |
| `cost_0` | 标量 | 当前解的 cost |
| `idx` / `reward` / `cost_before` / `cost_after` / `iter` | 标量 | 实际选中项及其结果 |

其中 `max_length = 节点数 N + 车辆数 × 4`：
- CVRP：1001 + 21×4 = **1085**
- CVRPTW：1001 + 40×4 = **1161**（C++ 端实际 mask 长度 ~1109）

**数据体积估算**（主要是 `masks`，16 × 1085 的 int16）：
- 每个 step ≈ 34 KB
- 每个 episode ≈ 624 step → **≈ 21 MB**（实际存盘 `roll_*.pt` 文件 23–29 MB，吻合）
- 每个 round 8 个 episode → **≈ 170 MB** 原始数据

---

## 3. Batch size：更新网络时一次用多少数据

这是最容易混淆的地方，分三步讲清楚。

### 第 1 步：子采样（不是 5000 全用）

收集到的 ~5000 个 step 不会全部拿去训练，而是**每个 episode 随机抽 256 个**
（参数 `max_update_steps_per_ep = 256`）：

```
8 个 episode × 每个抽 256 个 step = 2048 个 step 用于更新
```

这 **2048** 就是日志里 `n_update_steps` 恒为 2048 的原因。

### 第 2 步：一个 round 只更新一次网络

整个 round 的 2048 个 step **合并成一次梯度下降**（一次 `optimizer.step()`）。
也就是说：

```
一个 round = 收集 ~5000 step → 抽 2048 step → 算一次平均梯度 → 更新一次网络
```

所以从"优化器"的角度看，**这一轮的 batch size = 2048 个 step**。

### 第 3 步：为什么还有个"16"（update_minibatch）

2048 个 step 不能一次性全塞进显存，所以代码按**每 16 个 step 一小组**轮流算、
把梯度累加起来（参数 `update_minibatch = 16`），全部累加完才更新一次。这个 16
**只是为了省显存的分块大小，不是独立的更新**。

### 每轮实际的前向计算量

每个 step 要把 16 个候选子集一起喂进网络（batch 维 = 16）算一次前向：

```
2048 个 step × 每个 step 16 个候选 = 32,768 次候选评估 / round
```

### 一句话总结 batch

| 说法 | 数值 |
|------|------|
| 每轮收集的 step | ~5000 |
| 每轮真正用于更新的 step（子采样后） | **2048** |
| 每轮的梯度下降次数 | **1 次** |
| 显存分块大小 (minibatch) | 16 个 step / 块 |
| 每轮网络前向的候选评估总数 | 2048 × 16 = 32,768 |

---

## 4. 网络的输入 / 输出维度

策略网络是 `CostPredictor`（见 `model.py`）。它一次给一个 step 的 **16 个候选
子集**打分，所以所有输入张量的第一维（batch 维）都是 **16**。

记 `N = 1001`（节点数，含 depot），`max_length`（见上，CVRP 1085 / CVRPTW 1161）。

### 输入

| 张量 | 形状 | 含义 |
|------|------|------|
| `nodes_tensor` | [16, 1001, 2] | 每个节点坐标 (x, y) |
| `demands_tensor` | [16, 1001, 1] | 每个节点需求 ÷ 车容量 |
| `tw_features`（仅 CVRPTW）| [16, 1001, 3] | (earliest, latest, service) ÷ 时间范围 H |
| `current_sol_tensor` | [16, max_length] | 当前解的访问序列 |
| `selected_tensor` | [16, max_length] | 该候选选中的 anchor 类型 bitmask (0–15) |
| `cost_0` | [16] | 当前解 cost |

**每个节点的特征拼接后的输入维度**：
- CVRP：2(坐标) + 1(需求) = **3 维**
- CVRPTW：2 + 1 + 3(时间窗) = **6 维**

这个维度进入第一层 `NodeFeatureEmbedding(输入维度 → d_model=128)`。

### 主干网络

- Transformer encoder：`d_model = 128`，`num_heads = 4`，`num_encoder_layers = 3`
- 带位置偏置注意力 + selection-aware 注意力（让网络知道哪些点被选中）
- 回归头 `RegressionHeadV2`：全局/选中/非选中三种注意力池化 + 对比特征 +
  选择比例/类型分布特征

### 输出

| 张量 | 形状 | 含义 |
|------|------|------|
| 网络输出 | [16] | 16 个候选各自的预测分数（预测 cost ratio，越低越好）|

之后：`logits = -分数 / temperature` → `softmax` → 训练时按概率采样一个候选，
评测时取 argmax（或 ε-greedy）。

---

## 5. 一个 Round 的完整时间线（实测 CVRP）

```
Round 开始
 │
 ├─ 存当前网络权重到磁盘
 │
 ├─ 并行启动 8 个 episode 子进程（分到多张 GPU）
 │    每个 episode = 1 次 2 秒预算的 cuOpt 求解
 │    每个 episode 内 ≈ 624 个 step
 │    每个 step = 1 次 anchor 搜索 + 16 次 probe + 1 次真实搜索
 │    （probe 时间被排除在 2 秒预算外，故每 episode 墙钟 ~15 秒）
 │    → 8 个并行，整批 rollout 墙钟 ≈ 124 秒（2 卡）
 │
 ├─ 汇总 8 个 episode 的数据：~5000 个 step
 │    每个 episode 抽 256 个 → 共 2048 个 step
 │
 ├─ 用这 2048 个 step 算一次平均策略梯度，更新一次网络
 │
 └─ 每 10 个 round 额外做一次评测：
      8 次 policy 求解 + 8 次 origin（无策略）求解，比较平均 cost

Round 结束
```

---

## 6. 关键超参一览（计划中的 CVRPTW 训练）

| 参数 | 值 | 含义 |
|------|----|----|
| `--rounds` | 180 | 总轮数 |
| `--batch_episodes` | 8 | 每轮并行 episode 数 |
| `--time_limit` | 2 | 每个 episode 的 cuOpt 预算（秒）|
| `--k` | 16 | 每个 step 的候选子集数（arm）|
| `--eval_runs` | 8 | 每次评测的求解次数 |
| `--eval_every` | 10 | 每 10 轮评测一次 |
| `--lr` | 1e-3 | Adam 学习率 |
| `--temperature` | 0.5 | softmax 温度 |
| `--entropy_coef` | 0.005 | 熵正则系数 |
| `max_update_steps_per_ep` | 256（默认）| 每个 episode 抽多少 step 进更新 |
| `update_minibatch` | 16（默认）| 显存分块大小 |

### CVRP（实测）与 CVRPTW（计划）的差异

| 项 | CVRP (`rl_run2`) | CVRPTW |
|----|------------------|--------|
| 节点特征输入维度 | 3 | **6**（加时间窗）|
| `max_length` | 1085 | **1161** |
| 车辆数 | 21 | **40** |
| steps/episode | 实测 ~624 | 预计 ~400–700（跑起来日志会给确切值）|
| 其余（K, batch_episodes, 2048 update steps 等）| 相同 | 相同 |
