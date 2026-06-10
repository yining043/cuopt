# RL CVRP Flash 网络架构改进说明

本文只记录这次已经完成的网络架构层面改动。Task Definition 没有改：

- cuOpt 仍然在每次 callback 产生 `K` 个候选 node/operator subset。
- policy 仍然只对这 `K` 个候选打分，并返回一个候选 mask 给 cuOpt 执行。
- reward、candidate 生成、action 语义、训练 label 和 callback 接口保持不变。
- `CostPredictor.forward(...)` 的输入输出仍然兼容原路径，输出仍是每个候选 arm 的标量 score。

> 代码库更新说明：中间迭代 `v3`–`v6`（含 `v4legacy` 等）已从代码中删除，现在只保留
> `v2`（旧基线）和 `v7`（当前 flash head，别名 `flashv7`）。下文涉及 `v4`/`v5`/`v6`
> 的内容作为设计演进历史保留，`v7` 是这条 flash 线路的最终版本；可运行脚本为
> `run_train_cvrp.sh`（默认 `v7`）与 `run_benchmark_cvrp.sh`。

## 1. 原 v2 网络的核心瓶颈

原 `v2` 网络把每个候选 arm 当成一个完整样本送进 encoder。在线 callback 中，`K` 个候选共享同一个 incumbent route state，但 v2 会把这个共享 state 复制成 batch size `K`，然后每个候选都跑一次完整的 route/node self-attention。

这带来三个主要问题：

1. 共享 state 被重复编码  
   对同一个 solution，`K` 个候选只有 `selected_mask` 不同，route sequence、坐标、需求、cost 都相同。v2 的 encoder 仍然对每个 arm 重复做 `L x L` attention，计算量约为 `O(K * L^2)`。

2. dense attention bias 阻碍 Flash Attention  
   v2 在 `Encoder` 中显式构造 position bias 和 selected-mask bias，形状接近 `[K, H, L, L]`。这些 dense bias tensor 会造成巨大显存占用，也让 PyTorch SDPA 更难走高效 flash/mem-efficient kernel。

3. N=1000, K=100 时不可扩展  
   未来目标是 `N=1000`、`K=100`。设 active route length 为 `L`，v2 的主要 attention 显存和计算都随 `K * L^2` 增长。这个结构在大 N、大 K 下会把系统 overhead 和 memory footprint 放大到不可接受。

因此这次的核心改造不是简单加深/加宽网络，而是把“共享 route state 编码”和“候选 arm 打分”解耦。

## 2. 当前迭代: flash v7, sampling T=0.8

根据 30s/12-seed 曲线对比，`v5` 的系统方向正确，但最终 round9 只比旧
`v2 policy_round143` 略好，优势不够显著：

```text
random mean     = 34.783540
old v2 round143 = 34.740387  gap = +0.1241%
v5 round9       = 34.730340  gap = +0.1529%
```

问题不是 task/reward/action，而是 candidate decoder 设计还不够好：v5 用
token-specific `[K,T,L]` memory mask 恢复 selected interaction，但这会增加
mask 构造和显存压力；同时 selected 几何、route span、edge/savings 这些强
ranking 特征没有被足够直接地送入 score head。v6 继续尝试 scalar-rich
Flash-friendly decoder，但 10-round 结果不稳：round2 短暂优于 random，final
round9 又略低于 random。复核后发现 v6 的 local prev/next pool 存在 route
boundary 串线风险，并且 cross-attention 仍会为每个 candidate 重算/物化
route K/V。

因此当前迭代改为 v7：保留 shared route encoder，修正 route-boundary-aware
local features，并把候选 cross-attention 改成 shared route K/V + flattened
candidate query tokens。selected/type/edge 信息进入 query token、scalar
summary 和 score features，不再作为 candidate-specific route memory 进入 K/V。

```text
model mode: v7
train selection: sampling
eval/test selection: sampling
temperature: 0.8
K: 16
train GPUs: physical GPU 3,2
master device: physical GPU 3 (CUDA_VISIBLE_DEVICES starts with 3, master=cuda:0)
entropy_coef: 0.04
logit_clip: 2.0
10-round smoke/test run: outputs/rl_cvrp_flashv7_k16_h2_ep2_t08_e04_clip20_r10
```

`T=0.8` 的原因是旧 `v2 policy_round143` 在该温度下的 curve 明显更稳。
新 v7 继续用同一 sampling temperature 做可比训练，避免 `T=0.25` 在 logits
尚未校准时过早放大弱 ranking；train/test 都使用 sampling。

上一轮 `outputs/rl_cvrp_flashv4_k16_h2_ep2_t08_sample_r10` 暴露出两个问题：

- train-time eval 读取的是 update 前的 `current_weights.pt`，后续已修成 update 后先刷新权重再 eval。
- v4 edge features 没把 dummy depot 当 depot，且 route-start dummy token 会继承上一条 route 的 incoming/chord 特征；已修到当前 `mode=v4`。旧 checkpoint replay 使用 `mode=v4legacy`，避免用新特征评旧权重。

旧 checkpoint 的 corrected 10s/8seed replay 结果：

```text
random mean=35.497607
policy_round0 mean=35.414474 gap=+0.2342%
policy_round1 mean=35.415153 gap=+0.2323%
policy_round9 mean=35.354416 gap=+0.4034%
```

## 3. 新增 flash/v3 架构总览

新结构在 `model.py` 中通过 `CostPredictor(mode="flash")` 或 `mode="v3"` 启用，主要由三部分组成：

- `FlashRouteEncoder`
- `FlashCandidateHead`
- `FlashSelfAttentionLayer`

整体数据流：

```text
node features + current solution
        |
        v
FlashRouteEncoder
  只编码一次共享 route state
        |
        v
contextual route embeddings
        |
        v
FlashCandidateHead
  对 K 个候选 mask 做 pooling / cross-attention / K-token self-attention
        |
        v
K 个候选 score
```

在线 callback 中，`K` 个候选共享同一个 route state。新模型会检测 batch 是否是 broadcast 形式，如果 `nodes_tensor`、`demands_tensor`、`current_sol_tensor`、`tw_features` 都是共享的，就只用第一个样本跑 `FlashRouteEncoder`，然后让 `FlashCandidateHead` 对所有候选 mask 打分。

## 4. FlashSelfAttentionLayer

`FlashSelfAttentionLayer` 是新的 pre-norm Transformer block，设计目标是让注意力主路径尽量保持 SDPA/Flash Attention 友好。

关键点：

- 使用 fused `qkv_proj`，一次线性层得到 Q/K/V。
- 使用 `torch.nn.functional.scaled_dot_product_attention(...)`。
- 不再构造 `[B, H, L, L]` 的 dense position/selection bias。
- padding mask 只在确实存在 padding 时构造，且形状是 broadcast 友好的 `[B, 1, 1, L]`。
- FFN 改成 gated SiLU 形式：`silu(gate) * value`，提升表达力但保持结构简单。
- 每层结束后把 padding token 清零，避免无效位置污染后续 pooling/head。

这部分的主要意义是：让 route encoder 的 self-attention 更接近标准 SDPA 形态，从而充分利用 PyTorch 的 flash/mem-efficient attention backend。

## 5. FlashRouteEncoder: 共享 route state 只编码一次

`FlashRouteEncoder` 负责把当前 incumbent solution 编成 route-order contextual embeddings。

输入仍来自原始 task：

- node coordinate
- demand
- 可选 time-window features
- current solution route sequence

新增的 route metadata 是网络特征处理层面的改进，不改变 task：

```text
[is_depot, global_position, route_id, local_position, dummy_slot]
```

这些特征解决了原模型对 route structure 表达不足的问题：

- `is_depot`: 区分 depot/dummy depot 与普通 customer。
- `global_position`: token 在整条 flattened route sequence 中的相对位置。
- `route_id`: token 所属 route 的归一化编号。
- `local_position`: token 在当前 route 内的相对位置。
- `dummy_slot`: cuOpt route dummy depot slot 信息。当前 C++ 路径每条 route 有 4 个 dummy depot slot，新模型显式解码这个结构。

为什么这很重要：

原 v2 更偏 node-level 表达，selected mask 也被混入 self-attention。对于 VRP local search，route 内位置、route 边界、depot/dummy depot 的语义非常关键。如果不显式告诉模型这些结构，模型需要从 sequence position 和 mask 中间接学习，样本效率差，也容易在大 N 时退化。

## 6. FlashCandidateHead: 轻量候选打分器

`FlashCandidateHead` 把 `K` 个候选 mask 映射成候选 token，并在共享 route embedding 上打分。

它包含几类特征：

1. 全局 route pooling  
   表示当前 incumbent solution 的整体状态。

2. selected / non-selected pooling  
   分别对候选 mask 选中的 token 和未选中的 token 做 attention pooling。

3. contrast feature  
   使用 `selected_pool - nonsel_pool` 表达候选集合相对全局剩余部分的差异。

4. operator type pooling  
   `selected_mask` 是 4-bit anchor type bitmask：
   - bit0: sliding
   - bit1: vrp
   - bit2: recycle_vrp
   - bit3: two_opt

   新 head 为每种 operator type 学一个 query，对对应类型的 selected token 做 type-specific pooling。

5. count/type ratio feature  
   显式加入 selected ratio 和 4 类 operator 的比例特征，帮助模型区分“大范围扰动”和“小范围精修”。

6. cost embedding  
   保留 `log(cost_0)` 作为当前 search state 的尺度信息。

候选 token 生成后，再做两级 attention：

- candidate-to-route cross-attention: 每个候选 token attend 到共享 route embeddings，复杂度约 `O(K * L)`。
- candidate self-attention: 在 `K` 个候选 token 之间做 self-attention，复杂度约 `O(K^2)`，用于建模候选之间的相对竞争关系。

最终每个候选输出一个 scalar score。

## 7. FlashCandidateHeadV4/V5/V6: 当前改进

v4 保留 shared route encoder，但把 candidate decoder 从“单一候选摘要”改成
更强的 candidate-conditioned decoder：

- route encoder 仍然只跑一次，不回到 `K * L^2`。
- 每个候选构造 candidate-specific route memory：
  `route_embedding + selected_operator_embedding + local_edge_embedding`。
- selected 信息不再只通过 pooling 进入 head，而是进入 cross-attention 的
  key/value memory，使 query 能直接 attend 到“被当前候选选中”的局部 token。
- 每个候选不再只有一个 token，而是 `1 + 4` 个 token：
  一个 global candidate token 加四个 operator-type token。
- type token 带 operator ratio 和 selected edge summary。
- 加入 selected token 的 prev/next neighbor pooling 和 local edge delta。
- 保留 candidate self-attention，用于 K 个候选之间的相对比较。

v4 的目标是解决 v3 曲线里的主要问题：前 500-2000 iteration 几乎没有稳定
优势。旧 v2 的 selected signal 是在 token self-attention 前注入的；v4 用
candidate-specific memory 在 `O(K * L)` decoder 成本下恢复这部分信号。

v4fix 的 30s/12-seed 独立 benchmark 结果是：

```text
random mean = 34.783540
round0 mean = 34.772700 gap = +0.0312%
round2 mean = 34.807966 gap = -0.0702%
round6 mean = 34.782165 gap = +0.0040%
round9 mean = 34.748859 gap = +0.0997%
```

这个结果说明 v4 的方向可行但候选压缩过强，`1+4` 个 token 不足以稳定表达
candidate-local route interaction。v5 因此改成 Perceiver-style latent
decoder：

- route encoder 仍然只跑一次。
- 每个候选构造 candidate-specific route memory：
  `route_embedding + selected_operator_embedding + local_edge_embedding`。
- 每个候选使用 `1 + 4 + 8` 个 token：global token、4 个 operator-type
  token、8 个 latent slots。
- token-specific SDPA cross-attention 让 type token 只读对应 operator 的
  selected memory，部分 latent slot 只读 selected memory，其他 latent slot
  读全 route memory。
- cross-attention 和 token self-attention 做两轮，仍然是 `O(K * T * L)`
  decoder cost，`T=13` 是常数，不回到 `O(K * L^2)`。
- 最后保留 K-candidate self-attention，用于候选之间的相对竞争。

v5 的 30s/12-seed 独立 benchmark 结果说明它略优于旧 v2，但幅度太小，
不满足“显著有效”的目标。因此 v6 改成更 Flash-friendly 的 scalar-rich
decoder：

- route encoder 仍然只跑一次。
- candidate memory 仍为 `route_embedding + selected_operator_embedding + local_edge_embedding`。
- cross-attention 不再使用 token-specific `[K,T,L]` mask，而只使用 route-valid
  mask `[K,1,1,L]`；selected/type 信息通过 candidate memory、pooling 和 scalar
  embedding 表达。
- token 数从 v5 的 `13` 降到 `9`：global token、4 个 operator-type token、
  4 个 latent slots。
- 显式加入 selected scalar features：selected ratio、operator entropy/max、
  global/local/route span/std/mean、demand sum/mean/max/std、selected centroid/std/bbox、
  depot/boundary rate、edge in/out/chord/savings statistics。
- v6 的 edge features 修正了 next-edge 跨 route-start dummy 的问题；旧 v4/v5
  checkpoint replay 仍保持原 feature 语义，v6 从新特征开始训练。
- 修正 `nn.Sequential` 内 `LayerNorm` 的初始化识别，确保 LayerNorm weight=1、
  bias=0，而不是被普通 uniform 初始化。

## 8. Mask 对齐和 dummy route 修正

这次架构改造里也修正了两个会影响学习信号的特征处理问题。

第一，候选 mask 对齐方式修正。

C++ 传入的 `trail_masks_flat` 是按 node/dummy id 编码的，而 `FlashRouteEncoder` 使用 route-order sequence。新模型在 `_forward_flash` 中通过 `route_index` gather `selected_tensor`，把 node/dummy-id indexed mask 对齐到 route-order token 上。这样 selected / non-selected pooling 才和 route embedding 对齐。

第二，dummy depot metadata 修正。

当前 C++ 路径不是每条 route 一个 dummy，而是每条 route 有 4 个 dummy depot slot。新模型用：

```text
dummy_route_id = (dummy_id - n_nodes) // 4
dummy_slot     = (dummy_id - n_nodes) % 4
```

来恢复 route id 和 dummy slot 信息。否则 route boundary 和 depot slot 会被错误表达。

## 9. 复杂度变化

设：

- `L`: active route sequence length
- `K`: 每次 callback 的候选数
- `D`: hidden size
- `H`: attention heads

原 v2 主路径近似为：

```text
encoder self-attention: O(K * L^2 * D)
dense attention bias memory: O(K * H * L^2)
route embedding batch memory: O(K * L * D)
```

新 flash/v3 主路径近似为：

```text
shared route encoder: O(L^2 * D)
candidate pooling/cross-attention: O(K * L * D)
candidate self-attention: O(K^2 * D)
route embedding memory: O(L * D)
candidate token memory: O(K * D)
```

新 v4 decoder 增加的是 candidate-conditioned cross-attention memory：

```text
shared route encoder: O(L^2 * D)
candidate-conditioned memory: O(K * L * D)
multi-token candidate cross-attention: O(K * T * L * D), T=5
candidate self-attention: O(K^2 * D)
```

v5 把 `T` 从 5 增加到 13，并做两轮 token-to-route cross-attention：

```text
shared route encoder: O(L^2 * D)
candidate-conditioned memory: O(K * L * D)
latent candidate decoder: O(K * T * L * D), T=13
candidate self-attention: O(K^2 * D)
```

这仍然避免了旧 v2 的 `O(K * L^2)` dense selected attention bias。对目标
`N=1000,K=100`，主要额外成本是 `K * L * D`，而不是 `K * L^2 * D`。

v6 把 token 数降到 `T=9`，并去掉 token-specific attention mask：

```text
shared route encoder: O(L^2 * D)
candidate-conditioned memory: O(K * L * D)
full-route latent decoder: O(K * T * L * D), T=9
candidate self-attention: O(K^2 * D)
attention mask memory: O(K * L), not O(K * T * L)
```

当 `N=1000`、`K=100` 时，`K * L^2` 是主要瓶颈。新结构把这部分变成一次 `L^2`，剩下是 `K * L` 和 `K^2`。对于 `K=100`，`K^2` 很小，主要成本变成 shared route encoding 和 cross-attention，显存 footprint 明显下降。

## 10. 为什么更能利用 Flash Attention

新结构对 Flash Attention 友好的原因：

- self-attention 主路径没有 dense pairwise positional bias。
- selected mask 不再进入 route self-attention，而是在 candidate head 里通过 pooling/cross-attention 表达。
- route state 只编码一次，不再把同一个 `L x L` attention 重复 K 次。
- attention 调用使用 PyTorch SDPA 接口，CUDA 可用时可以走 flash/mem-efficient backend。
- bf16 autocast 已接入训练、callback、rollout 和 benchmark，进一步降低 attention/MLP 激活显存。

这不是把旧网络“换个 API 调 SDPA”，而是把网络结构重排成更适合 SDPA 的形状。

## 11. 接口兼容性

`CostPredictor` 现在支持：

```text
mode in ["ratio", "new", "v2", "v7", "flashv7"]
```

（中间迭代 `v3`–`v6` 已从代码库移除，只保留 `v2` 与 `v7`；`flashv7` 是 `v7` 的别名。）

`v7` 没有改变外部调用方式：

```python
scores = model(
    nodes_tensor,
    demands_tensor,
    current_sol_tensor,
    selected_tensor,
    cost_0,
    tw_features=tw,
)
```

返回仍是 `[K]` 或 `[B]` score。`RLPolicyCallback` 仍然用这些 score 做 softmax，然后按配置 greedy 或 sampling 选一个 arm。

额外新增的运行参数只是 policy forward 的执行方式：

- `--mode v7` 或 `--model_mode v7`
- `--amp_dtype bf16`
- `--selection sample`
- `--temperature 0.8`

这些不改变 task definition，只改变网络和选择策略配置。

## 12. 已验证证据

### 12.1 Policy-only N=1000, K=100 profiling

profiling 脚本：

```text
profile_policy_network.py
```

它不 import cuOpt，只构造 synthetic route state 和 `K=100` candidate masks，测 policy network 本身。

测试设置：

- customers: 1000
- nodes including depot: 1001
- vehicles: 100
- K: 100
- active route length: 1200
- dtype: bf16 autocast
- GPU: NVIDIA RTX 6000 Ada Generation
- PyTorch: 2.7.1+cu126

结果：

| Mode | Pass | Mean latency | Peak delta memory |
| --- | ---: | ---: | ---: |
| `v2` | forward | `150.34 ms` | `12432.7 MB` |
| `flash` | forward | `3.90 ms` | `13.7 MB` |
| `v2` | forward+backward | `428.94 ms` | `17800.8 MB` |
| `flash` | forward+backward | `15.32 ms` | `148.2 MB` |

artifact:

```text
outputs/policy_network_profile_forward_n1000_k100.json
outputs/policy_network_profile_backward_n1000_k100.json
```

这个结果直接说明：新结构在目标 scale `N=1000,K=100` 下显著降低了 attention overhead 和显存占用。

### 12.2 历史 v3 训练与 benchmark 信号

历史 v3 sampling 设置：

- train sampling
- test/eval sampling
- temperature `0.25`
- current K `16`

真实训练在用户要求下停止在 round 7，保存了 round 0 到 round 7 的 checkpoint：

```text
outputs/rl_cvrp_flash_k16_h2_ep2_t025_sample_r10/policy_round0.pt
...
outputs/rl_cvrp_flash_k16_h2_ep2_t025_sample_r10/policy_round7.pt
```

10s、8 runs benchmark 中，round 7 的 sampling `T=0.25` 是当前最好结果：

```text
random mean              = 35.468982
round7 sample T=0.25 mean = 35.258147
gap vs random             = +0.5944%
```

plot artifact:

```text
outputs/benchmark_flash_ckpts_0_1_7_10s_8runs/plot_avg_round0.png
outputs/benchmark_flash_ckpts_0_1_7_10s_8runs/plot_avg_round1.png
outputs/benchmark_flash_ckpts_0_1_7_10s_8runs/plot_avg_round7.png
```

这说明早期 flash/v3 网络已经解决了目标 scale 的 overhead/memory 问题，并在短训后的真实 cuOpt benchmark 上表现出有效 ranking signal；当前 v7 继续沿这条路线修正 candidate decoder 的表达力和边界特征。

### 12.3 v7 训练计划

当前 v7 训练命令（`run_train_cvrp.sh` 默认即为 v7）：

```bash
bash run_train_cvrp.sh
```

该脚本默认使用：

```text
CUDA_VISIBLE_DEVICES=3,2      # cuda:0 maps to physical GPU 3
master_device=cuda:0          # physical GPU 3
mode=v7
rounds=10
K=16
train/eval selection=sample
temperature=0.8
entropy_coef=0.04
logit_clip=2.0
amp_dtype=bf16
```

10-round 结果用于判断是否继续跑；如果 curve 或 final-cost gap 不达标，就继续
迭代 head 设计和 bug 检查。

## 13. 当前结论

这次网络改进的核心是：

1. 把 route state encoder 从 per-arm 编码改成 shared 编码。
2. 把 selected-mask 信息从 self-attention bias 中移出，放到候选 head 中显式建模。
3. 用 SDPA-friendly 的 Transformer block 替代 dense `L x L` bias attention。
4. 用 candidate-to-route cross-attention 和 K-token self-attention 建模候选质量与候选间竞争。
5. 补足 route id、route-local position、depot/dummy slot、operator type pooling 等 VRP 结构特征。

因此新架构更适合 Flash Attention，更省显存，也更适合未来直接 scale 到 `N=1000,K=100`。
