# CVRP Local-Search RL: Baseline Result and Flash-Friendly Architecture

Date: 2026-06-08

This work does not replace cuOpt with an RL route generator. cuOpt still owns
candidate generation, feasibility checks, local-search execution, and the rule
that only improving moves are accepted. The learned policy only chooses one
candidate node/operator subset from the `K` subsets that cuOpt already proposes
at a local-search callback.

> Note (codebase update): the intermediate flash iterations `v3`–`v6` have been
> removed; only `v2` (old baseline) and `v7` (current flash head) remain runnable.
> The runnable scripts are now `run_train_cvrp.sh` (defaults to `v7`) and
> `run_benchmark_cvrp.sh`. Sections below that discuss `v4`/`v5` are kept as design
> history; `v7` is the surviving descendant of that line.

## Fixed MDP and RL Formulation

One episode is one cuOpt CVRP solve. One RL step is one local-search callback.
The state contains the fixed CVRP instance features, the current incumbent
solution, the current objective value, and `K` candidate masks proposed by
cuOpt. Each mask marks route positions and operator anchor type:

| Mask bit | Operator |
| --- | --- |
| 1 | sliding |
| 2 | vrp |
| 4 | recycle_vrp |
| 8 | two_opt |

The action is `a_t in {0, ..., K-1}`: choose exactly one candidate subset for
cuOpt to execute on the real incumbent. The transition is cuOpt local search
after that choice; if no improving move is found, the incumbent cost does not
decrease.

Training is full-feedback contextual-bandit RL. For each state, C++ evaluates
all `K` candidates on copied solutions and returns one-step labels plus optional
short-lookahead labels:

```text
trail_rewards = [immediate_reward_0 ... immediate_reward_K-1,
                 lookahead_reward_0 ... lookahead_reward_K-1]
```

The current task definition stays fixed: candidate generation, action space,
reward labels, `reward_horizon`, cuOpt execution, and accepted-move semantics
are not changed. The neural policy scores the same `K` arms:

```text
logits_i = score_sign * score_i / temperature
pi_i = softmax(logits)_i
adv_i = normalized lookahead label for arm i
J = sum_i pi_i * adv_i
loss = -J - entropy_coef * entropy(pi)
```

For the current architecture iteration, both training and testing sample from
the policy with temperature `0.8`. This is different from the old training-time
eval, which was mostly greedy.

## Baseline Result: `outputs/rl_cvrp_h2_ep2_lr3e4`

The completed baseline used CVRP index 1 from
`../../cuopt-examples/data/test_cvrp1000_hgs_n128_C250.txt`, 21 vehicles,
`K=16`, 2-second training solves, 4 episodes per round, `reward_horizon=2`,
Adam with `lr=3e-4`, entropy coefficient `0.01`, update minibatch `256`, and
2 update epochs. It ran on physical GPUs 2 and 3 with master `cuda:1`, which
maps to physical GPU 3 when `CUDA_VISIBLE_DEVICES=2,3`.

The baseline reached logged rounds 0 through 143. Its best candidate-ranking
metric was `top1_lookahead=0.2242` at round 131, versus random `1/16=0.0625`.
The best 2-second training-time final-cost eval was `best_policy.pt` selected
at round 82: policy mean `35.9191` versus random mean `36.2127`, a `+0.81%`
gap. In 20-second benchmarks, softmax sampling was stronger than greedy: the
best observed mean was `34.8202` for `policy_round143.pt`, `K=40`, sampling,
versus random `34.9367` (`+0.33%`). With `K=16`, `policy_round2.pt` sampling
gave mean `34.8246` versus random `34.9556` (`+0.38%`).

Interpretation: the policy learned a real ranking signal, but the end-to-end
solver gain is still small. Greedy selection is not the best way to use the
learned distribution; sampling is the more promising test-time setting.

## Network Bottleneck and New Architecture

The old `v2` model already called PyTorch scaled dot-product attention, but it
was not Flash-Attention friendly. For every callback it expanded the same state
to `K` rows and ran a full sequence Transformer for every arm. It also
materialized dense `[K, heads, L, L]` positional and selection attention biases.
At future scale `N=1000`, `K=100`, this makes the dominant cost approximately
`O(K * L^2)` and creates large dense bias tensors that defeat the main memory
benefit of Flash Attention.

The new `flash` mode keeps the same `CostPredictor.forward(...) -> [K]`
interface but changes only the neural architecture:

```text
1. Shared route encoder
   Encode the incumbent route sequence once per callback with SDPA attention.
   Global route position, local within-route position, route id, depot
   indicators, demand, coordinates, and optional TW features are token features,
   not dense pairwise attention biases.

2. Candidate feature builder
   For each cuOpt candidate mask, compute selected-node, non-selected-node,
   contrast, operator-type, count-ratio, and cost-context summaries.

3. Candidate token ranker
   Build one token per candidate arm. Each token attends to the shared route
   encoding with SDPA cross-attention, then the K candidate tokens interact with
   a small SDPA self-attention block before the final score head. Empty/invalid
   candidate masks are masked out inside this arm-to-arm attention.
```

The intended complexity is now `O(L^2) + O(K * L) + O(K^2)` instead of
`O(K * L^2)`. The action, labels, reward horizon, and cuOpt transition are
unchanged; this is purely a policy-network redesign. The validation scripts use
`bf16` autocast for policy forward passes so PyTorch SDPA can use the
Flash/memory-efficient attention kernels on supported GPUs.

## Current Flash v7 Iteration

The v4fix architecture did not produce a significant independent benchmark
gain. In the 30s/12-seed benchmark, round9 was only `+0.0997%` versus random
and the earlier checkpoints were effectively tied. The current head therefore
uses v7 (`FlashCandidateHeadV7`): the shared Flash route encoder still encodes
the incumbent route once, candidates are conditioned through `1+4+4` query
tokens (one global token, four anchor-type tokens, four latent tokens) plus
scalar/pooled summaries, and two SDPA cross-attention layers read one shared
route K/V cache when all `K` candidates share the same incumbent state.

```text
mode=v7
K=16
train selection=sample
eval/test selection=sample
temperature=0.8
CUDA_VISIBLE_DEVICES=3,2
master_device=cuda:0          # physical GPU 3
entropy_coef=0.04
logit_clip=2.0
runname=rl_cvrp_flashv7_k16_h2_ep2_t08_e04_clip20_r10
```

The v7 network keeps the same feature fixes from the v4 line: candidate masks
are gathered from node/dummy id space into route order, dummy depot slots are
depot-like, and route-start dummy tokens do not inherit previous-route edge
features.

## 10-Round Validation Setup

The validation run is configured as:

| Item | Value |
| --- | --- |
| Script | `bash run_train_cvrp.sh` (defaults to `v7`) |
| Output | `outputs/rl_cvrp_flashv7_k16_h2_ep2_t08_e04_clip20_r10` |
| Model mode | `v7` |
| Rounds | 10 |
| Train/test action selection | softmax sampling |
| Temperature | `0.8` |
| Entropy coefficient | `0.04` |
| Logit clip | `2.0` |
| Candidate subsets | `K=16` |
| Training label horizon | `reward_horizon=2` |
| Eval label horizon | `reward_horizon=1` |
| Time limit | 2 seconds per training solve |
| Batch episodes | 4 per round |
| Optimizer | Adam, `lr=3e-4` |
| GPU contract | `CUDA_VISIBLE_DEVICES=3,2`; master `cuda:0` = physical GPU 3 |
| Update minibatch / epochs | `256` / `2` |
| Policy forward dtype | `bf16` autocast |

After training, run:

```bash
RUN=rl_cvrp_flashv7_k16_h2_ep2_t08_e04_clip20_r10
POLICIES=(
  "Roundnine|outputs/${RUN}/policy_round9.pt|v7|9|sample|0.8"
) \
OUT_DIR=outputs/benchmark_flashv7_round9_30s_12seed_k16_t08_sample \
GPUS="3 2 0 1" \
TEMPERATURE=0.8 \
AMP_DTYPE=bf16 \
TIME_LIMIT=30 \
K=16 \
NUM_SEEDS=12 \
bash run_benchmark_cvrp.sh
```

This runs a fresh random baseline plus the chosen `v7` checkpoint(s) with
30-second solves, `K=16`, sampling, and temperature `0.8`. Add more entries to
`POLICIES` (e.g. several rounds, or `v2` vs `v7`) to compare additional lines in
the same plot. The benchmark writes summary statistics via
`summarize_rl_benchmark.py` and an averaged best-so-far curve via `plot_avg.py`.

Decision criteria are final-cost mean/min/std versus random, paired gap versus
old `v2` at the same seeds and temperature, `top1_lookahead`, `top1_current`,
entropy, update wall time per optimizer step, and peak GPU memory. A successful
first validation should keep the learned ranking signal while reducing update
overhead enough to make `N=1000,K=100` realistic.

## Flash Profiling Evidence at N=1000, K=100

The policy-only profiler `profile_policy_network.py` does not import cuOpt. It
builds one shared incumbent route state and `K=100` synthetic candidate masks,
then measures the neural network at `1000` customers plus depot and `100`
vehicles. On an RTX 6000 Ada with PyTorch `2.7.1+cu126`, bf16 autocast, and
SDPA backends enabled:

| Mode | Pass | Mean latency | Peak delta memory |
| --- | ---: | ---: | ---: |
| `v2` | forward | `150.34 ms` | `12432.7 MB` |
| `flash` | forward | `3.90 ms` | `13.7 MB` |
| `v2` | forward+backward | `428.94 ms` | `17800.8 MB` |
| `flash` | forward+backward | `15.32 ms` | `148.2 MB` |

Artifacts:

```text
outputs/policy_network_profile_forward_n1000_k100.json
outputs/policy_network_profile_backward_n1000_k100.json
```

This directly validates the architecture target: the route state is encoded
once and candidate scoring scales as shared-route attention plus small K-token
attention, instead of materializing the old per-arm dense sequence attention
biases.

## Flash Training Run and 10s Benchmarks

The historical flash/v3 validation run was launched with the older sampling setup
(`K=16`, train/eval temperature `0.25`, physical GPUs `2,3`). Per user request,
that run was stopped after enough progress had been observed; the saved
checkpoints are rounds `0` through `7`:

```text
outputs/rl_cvrp_flash_k16_h2_ep2_t025_sample_r10/policy_round0.pt
...
outputs/rl_cvrp_flash_k16_h2_ep2_t025_sample_r10/policy_round7.pt
```

Round-0 evaluation already showed a real ranking signal:
`top1_lookahead=0.2143` versus random `0.0625`, with policy mean `36.2159`
versus random mean `36.2239` in 2-second train-time eval. The best observed
training-rollout mean among rounds 0 through 7 was round 7 at `36.0574`.

The requested post-training benchmark compared checkpoint round `0`, round `1`,
and latest round `7`. Each used 10-second solves, 8 seeds, `K=16`, physical GPUs
`2,3`, and four curves: random, sampling with `T=0.25`, sampling with `T=1.0`,
and greedy. The plots were generated with `plot_avg.py` and fixed
`--ymin 3400 --ymax 3800 --abs --seg 10000`.

| Checkpoint | Random mean | Sample T=0.25 | Sample T=1.0 | Greedy |
| --- | ---: | ---: | ---: | ---: |
| round 0 | `35.4690` | `35.4896` | `35.3530` | `35.3492` |
| round 1 | `35.4690` | `35.3161` | `35.3785` | `35.3000` |
| round 7 | `35.4690` | `35.2581` | `35.4777` | `35.3928` |

The strongest setting in this historical benchmark was latest round `7` with
sampling temperature `0.25`: mean `35.2581`, about `0.59%` better than the
random baseline mean `35.4690`. That result is now treated as the v3 baseline;
the active v7 iteration uses train/test sampling temperature `0.8`, `K=16`,
and physical GPUs `3,2` with physical GPU 3 as `cuda:0` master.

Plot artifacts:

```text
outputs/benchmark_flash_ckpts_0_1_7_10s_8runs/plot_avg_round0.png
outputs/benchmark_flash_ckpts_0_1_7_10s_8runs/plot_avg_round1.png
outputs/benchmark_flash_ckpts_0_1_7_10s_8runs/plot_avg_round7.png
```
