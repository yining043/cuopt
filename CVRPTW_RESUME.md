# CVRPTW RL Node-Selection — Resume Plan

Goal: reproduce the CVRP RL experiment (train policy -> 16x3 benchmark -> 3-line
plot of random vs policy vs policy_eps) on a **random-generated 1000-customer
CVRPTW instance**, with **time windows fed into the model**.

All code is written and smoke-tested. What remains is pure compute:
**(1) train, (2) benchmark 16x3, (3) plot.** This doc is the resume checklist
for running on a fresh GPU server.

---

## 0. State at pause

- Code complete + smoke-tested end-to-end (CVRPTW trains, evals, checkpoints).
- Instance regenerated with loosened time windows so cuOpt behaves normally:
  - `data/cvrptw_inst1.pt`: 1000 customers, 40 vehicles, capacity 250, scale 100,
    horizon H=1000, window width 350-700.
  - Sanity: 4s random run -> `final_cost=41.25`, **89% of LS steps find an
    improving move**, cost drops 14558 -> 4124 (healthy, unlike the earlier tight
    instance that was stuck at `move_found: 0`).
- No training/benchmark run yet (this server ran out of GPUs).

## Files added / changed (all committed in the repo)

- NEW `gen_cvrptw.py` — generate + cache + validate the CVRPTW instance.
- `load_nco_data.py` — `load_cvrptw_data()`, `build_tw_features()` (TW feats /H).
- `run_cuopt.py` — `get_cuopt_model_tw()` (order TW + service + vehicle TW).
- `model.py` — `CostPredictor(n_node_feat=...)` + `forward(..., tw_features=)`.
  CVRP path (n_node_feat=3, tw=None) unchanged.
- `rl_callback.py` — `RLPolicyCallback(tw_features=...)`.
- `rl_rollout.py`, `train_rl.py`, `benchmark_rl.py` — `--data_pt` switch:
  set => CVRPTW (6 feats + TW model + `max_vehicles=40`); unset => original CVRP.
- `run_rl_benchmark.sh` — `DATA_PT` env passthrough; runs random+policy+policy_eps;
  default `GPUS="2 3"`.

---

## 1. Environment setup (fresh server)

```bash
conda activate cuopt_dev          # or recreate the env used for cuOpt dev
# Build cuOpt WITH the oracle K-probe path (C++ changes are in cpp/src/.../local_search.cu):
./build.sh --cache-tool=ccache cuopt libcuopt
```

The RL path requires the custom C++ build (`CUOPT_LS_MODE=oracle` +
`CustomizeNodesCallback` with `trail_rewards` / `on_search_result`). The scripts
set `CUOPT_LS_MODE` and `CUOPT_RL_K` themselves.

CVRP data path (only needed if you also run the CVRP baseline; not needed for
CVRPTW): `/home/yining/cuopt-examples/data/test_cvrp1000_hgs_n128_C250.txt`.

## 2. Generate the instance (deterministic; skip if data/cvrptw_inst1.pt present)

```bash
CUDA_VISIBLE_DEVICES=0 python gen_cvrptw.py \
  --out data/cvrptw_inst1.pt --seed 1 --validate --validate_time 10
```
Expect: `feasible: cost=~41.7 vehicles=20`, window width [350,700], H=1000.
(seed=1 is fully reproducible, so a re-gen on another server gives the same file.)

## 3. Train (set GPUs to whatever is free, e.g. 0,1,2,3 or 2,3)

```bash
python train_rl.py --data_pt data/cvrptw_inst1.pt \
  --time_limit 2 --rounds 180 --batch_episodes 8 \
  --gpus 2,3 --k 16 --lr 1e-3 --temperature 0.5 --entropy_coef 0.005 \
  --eval_runs 8 --eval_every 10 --runname rl_cvrptw1
```
Output: `outputs/rl_cvrptw1/best_policy.pt` (+ `train_rl_log.jsonl`).
Watch: `top1_acc` rising well above `rand_top1` (~1/16=0.062); eval policy mean
< origin mean. best_known is NaN (random instance has no HGS reference) — ignore
the gap-vs-best-known print; the meaningful comparison is policy vs random-subset
in the benchmark.

Wall time note: probe time is `add_offset`'d out of the cuOpt budget, so wall
time per episode is ~7-8x `time_limit`. 8 episodes/round x 180 rounds + evals is
multi-hour on 2 GPUs (faster with 4).

## 4. Benchmark 16 x 3 (10s)

```bash
RUNS=16 TIME_LIMIT=10 \
  OUT_DIR=outputs/benchmark_cvrptw_10s \
  WEIGHTS=outputs/rl_cvrptw1/best_policy.pt \
  DATA_PT=data/cvrptw_inst1.pt \
  GPUS="2 3" bash run_rl_benchmark.sh
```
Produces in `outputs/benchmark_cvrptw_10s/`:
`random_baseline_{1..16}.txt`, `policy_{1..16}.txt`, `policy_eps_{1..16}.txt`.

Quick stats afterward:
```bash
for g in random_baseline policy_ policy_eps; do
  echo "== $g ==";
  grep -h final_cost outputs/benchmark_cvrptw_10s/${g}*.txt \
    | awk -F= '{print $2}' | awk '{s+=$1;n++} END{printf "mean=%.4f n=%d\n",s/n,n}';
done
# note: "policy_" glob also matches policy_eps; use policy_[1-9]*.txt to isolate pure policy
```

## 5. Plot (3 lines)

Y-axis is RAW C++ cost (~4100 for this CVRPTW instance, NOT the scaled ~41).
Pick ymin/ymax after a first look (e.g. ~3900-15000, or zoom 3900-5000).

```bash
python plot_avg.py \
  outputs/benchmark_cvrptw_10s/random_baseline_{1..16}.txt \
  outputs/benchmark_cvrptw_10s/policy_{1..16}.txt \
  outputs/benchmark_cvrptw_10s/policy_eps_{1..16}.txt \
  --out outputs/benchmark_cvrptw_10s/rl_random_vs_policy_eps.png \
  --abs --segments 10000 --ymin 3900 --ymax 5000
```
Groups auto-detected: `Random_baseline_`, `Policy_`, `Policy_eps_`.

---

## Key params / gotchas

- `max_vehicles=40` MUST match the instance (model `max_length` sizing). Threaded
  automatically via `--data_pt`; do not hand-build a model with default 21.
- Train uses softmax sampling; benchmark `policy` uses argmax (plateaus late),
  `policy_eps` uses epsilon=0.1 (best late-game in the CVRP runs).
- If reward signal is too sparse during training (loss/top1 not moving), loosen
  TW further: regen with `--horizon_mult 12` or widen windows in `gen_cvrptw.py`
  (`width_frac`). Current settings already give 89% improving-move rate.
- Reference CVRP result for comparison: `outputs/benchmark_rl_10s/` (mean ~35.3,
  policy_eps best). CVRPTW costs are higher (scaled ~41).

## One-shot resume (copy-paste)

```bash
conda activate cuopt_dev
cd <repo>
# (build once if not built) ./build.sh --cache-tool=ccache cuopt libcuopt
ls data/cvrptw_inst1.pt || CUDA_VISIBLE_DEVICES=0 python gen_cvrptw.py --out data/cvrptw_inst1.pt --seed 1 --validate
python train_rl.py --data_pt data/cvrptw_inst1.pt --time_limit 2 --rounds 180 --batch_episodes 8 --gpus 2,3 --k 16 --lr 1e-3 --temperature 0.5 --entropy_coef 0.005 --eval_runs 8 --eval_every 10 --runname rl_cvrptw1
RUNS=16 TIME_LIMIT=10 OUT_DIR=outputs/benchmark_cvrptw_10s WEIGHTS=outputs/rl_cvrptw1/best_policy.pt DATA_PT=data/cvrptw_inst1.pt GPUS="2 3" bash run_rl_benchmark.sh
python plot_avg.py outputs/benchmark_cvrptw_10s/random_baseline_{1..16}.txt outputs/benchmark_cvrptw_10s/policy_{1..16}.txt outputs/benchmark_cvrptw_10s/policy_eps_{1..16}.txt --out outputs/benchmark_cvrptw_10s/rl_random_vs_policy_eps.png --abs --segments 10000 --ymin 3900 --ymax 5000
```
