# Training & Analysis Workflow

Contrastive embedding model (`SolutionEmbedder`): solution + instance → 128-dim L2-normalized embedding. Trained with InfoNCE (S1) + Triplet (S2) + optional regression (S3). Analysis: checkpoint quality, convergence prediction, return probability correlation.

---

## 1. Model Architecture

```
Input:  solution_flat + (depot_xy, node_xy_demand)
  → CVRPEnv.preprocessing (visited_time, cum_demand, ...)
  → cyclic_position_encoding → pos_encoder (MultiHeadPosCompat) → route_attn bias
  → Encoder (3-layer Transformer, 8 heads, dim=128) with route_attn
  → mean pooling → L2 normalize
Output: 128-dim embedding
```

Distance: `d = ||emb_a - emb_b||_2` (L2 between normalized vectors, range [0, 2]).

Key files: `net.py` (SolutionEmbedder, Encoder, MultiHeadPosCompat), `CVRPEnv.py` (preprocessing + feature extraction), `helper.py` (sol2rec, broken_pairs, etc).

---

## 2. Training

### Stages

| Stage | Loss | Data | Purpose |
|-------|------|------|---------|
| S1 | Weighted InfoNCE | `basin_pairs.jsonl` + `distant_basins.jsonl` | Separate basin neighbours vs distant |
| S2 | Triplet Margin | `perturb_data.jsonl` (k=1 perturbation) | Fine-grained: same-basin close, diff-basin far |
| S3 | InfoNCE + P_max regression + advantage ordinal | `training_data.jsonl` (trajectory) | Optional: predict return probability, gap bins |

### Quick Start

```bash
# S1 + S2 (recommended default)
python train_basin_contrastive.py \
  --instance_indices 0-49 \
  --stage both \
  --epochs1 100 --epochs2 50 \
  --batch_size 512 --batch_size2 128 \
  --lr1 5e-4 --lr2 5e-5 \
  --save out \
  --note "2stages" \
  --gpu_id 0

# S2 only, loading S1 checkpoint
python train_basin_contrastive.py \
  --instance_indices 0-49 \
  --stage 2 \
  --load_checkpoint out/<run_name>/s1_epoch100.pt \
  --epochs2 50 \
  --save out \
  --note "s2only"

# All 3 stages
python train_basin_contrastive.py \
  --instance_indices 0-49 \
  --stage all \
  --epochs1 100 --epochs2 50 --epochs3 50 \
  --use_regression --regression_weight 0.5 \
  --quality_reg_weight 0.01 \
  --save out \
  --note "all3stages"
```

### Checkpoint Format

Saved as `out/<run_name>/s{1,2,3}_epoch{N}.pt`:

```python
{
    "embedder_state": embedder.state_dict(),  # encoder + pos_encoder (full model)
    "epoch": N,
    "stage": 1 | 2 | 3,
    # S3 only:
    "pmax_head_state": ...,       # optional
    "advantage_head_state": ...,  # optional
}
```

> **Note**: Checkpoints before 2026-03-18 used `"encoder_state"` (encoder only, **pos_encoder missing**). Loading code is backward-compatible but prints WARNING. Results from old checkpoints are unreliable — retrain recommended.

### Key Hyperparameters

| Param | Default | Notes |
|-------|---------|-------|
| `--embedding_dim` | 128 | Embedding dimension |
| `--n_layers` | 3 | Transformer encoder layers |
| `--n_heads` | 8 | Attention heads |
| `--temperature` | 0.07 | InfoNCE temperature (S1) |
| `--margin` | 0.1 | Triplet margin (S2) |
| `--neg_mode` | `masked_in_batch` | S1 negative sampling: `distant` or `masked_in_batch` |
| `--use_l2_normalize` | True | L2-normalize output embeddings |

### Output Structure

```
out/<run_name>/
  s1_epoch{5,10,...,100}.pt   # S1 checkpoints
  s2_epoch{5,10,...,50}.pt    # S2 checkpoints
  plot/                       # Distance histograms, 2D embedding plots per epoch
  src/                        # Snapshot of training source code
```

Wandb logging enabled by default (project: `landscape`). Disable with `--disable_wandb`.

---

## 3. Analysis

### 3a. Checkpoint Quality (`analyze_checkpoints.py`)

Evaluate embedding quality on held-out validation data (instances 50-54).

```bash
# S1 + S2 validation curves
python analyze_checkpoints.py \
  --checkpoint_dir out/<run_name> \
  --stage both

# Trajectory embedding visualization (PCA/t-SNE of search trajectories)
python analyze_checkpoints.py \
  --checkpoint_dir out/<run_name> \
  --stage trajectory \
  --trajectory_embed_viz \
  --trajectory_embed_n_trials 10

# Multi-run diversity analysis
python analyze_checkpoints.py \
  --checkpoint_dir out/<run_name> \
  --stage trajectory \
  --multi_run_viz \
  --multi_run_n 10

# Single-run trajectory visualization
python analyze_checkpoints.py \
  --checkpoint_dir out/<run_name> \
  --stage trajectory \
  --run_trajectory_path basin_datasets0/cvrp100_uniform.pkl#50/trajectory.jsonl
```

Output: `<checkpoint_dir>/analysis_*/` (PNGs, GIFs, HTML).

**Metrics reported**: mean d(anchor, positive), mean d(anchor, negative/distant), ratio d_an/d_ap.

### 3b. Convergence Prediction (`analyze_trial_convergence.py`)

Train a classifier (logistic regression / threshold) to predict whether an intermediate solution converges to a given local optimum, using embedding distance as feature.

```bash
python analyze_trial_convergence.py \
  --checkpoint_dir out/<run_name> \
  --trajectory_paths basin_datasets0/cvrp100_uniform.pkl#50/trajectory.jsonl \
                     basin_datasets0/cvrp100_uniform.pkl#51/trajectory.jsonl
```

Output: `<checkpoint_dir>/convergence/` — per-checkpoint accuracy, ROC AUC, classifier `.pkl` (used by solver early stop).

### 3c. Return Probability Correlation (`analyze_return_probability.py`)

Measure correlation between embedding distance and basin return probability (P_return from k=1 perturbation data).

```bash
python analyze_return_probability.py \
  --checkpoint_dir out/<run_name> \
  --return_prob_data perturb_k1_collect/return_prob_data.jsonl
```

Output: `<checkpoint_dir>/analysis_return_prob/` — Spearman/Pearson correlation, boundary classification accuracy.

---

## 4. Inference (Solver Early Stop)

Use trained embeddings to predict convergence during cuOpt solving, triggering early stop when the current solution is predicted to reach the same basin as a previously seen optimum.

```bash
cd basin_callback

# Solve with embedding-based early stop
python run_cuopt.py solve \
  --n_instances 10 --start_index 60 \
  --time_limit 5 --n_runs 1 \
  --use_callback \
  --early_stop_base embedding \
  --checkpoint /path/to/s1_epoch100.pt \
  --classifier_pkl /path/to/convergence/trial_convergence_s1_epoch100_classifiers.pkl \
  --classifier_type lr \
  --plot

# Solve with structure-based early stop (broken-pairs, no embedding model)
python run_cuopt.py solve \
  --n_instances 10 --start_index 60 \
  --time_limit 5 \
  --use_callback \
  --early_stop_base structure \
  --classifier_pkl /path/to/classifiers.pkl

# Solve without early stop (baseline)
python run_cuopt.py solve \
  --n_instances 10 --start_index 60 \
  --time_limit 5

# Plot from saved log
python run_cuopt.py plot run1.log run2.log --ymin 1200 --ymax 1800
```

Early stop strategies: `random` (baseline, random skip), `embedding` (model + classifier), `structure` (broken-pairs + classifier).

---

## 5. Typical Experiment Flow

```
1. Train S1+S2          →  out/<run>/s{1,2}_epoch*.pt
2. Analyze checkpoints   →  validation curves, pick best epoch
3. Convergence analysis  →  classifiers.pkl (tau_embed, LogReg)
4. Return prob analysis  →  correlation sanity check
5. Solver experiment     →  run_cuopt.py solve --early_stop_base embedding
6. Compare with baseline →  run_cuopt.py plot
```

### Historical Runs

| Run | Config | Notes |
|-----|--------|-------|
| `20260210_204422_0-49_2stages_stage1e100_stage2e50` | S1 100ep + S2 50ep | **Legacy ckpt (pos_encoder missing)** |
| `20260225_*_triplet_*` | Various S3 triplet experiments | S3 with regression |
| `20260226_*_infonce_*` | S3 InfoNCE variants | S3 with/without regression |

---

## 6. Data Dependencies

| File | Source | Used By |
|------|--------|---------|
| `basin_pairs.jsonl` | Step 5 of EXPERIMENT_WORKFLOW | S1 training |
| `basin_info.jsonl` | Step 5 | S1 training (solution lookup) |
| `distant_basins.jsonl` | Step 6 | S1 negative sampling (`--neg_mode distant`) |
| `perturb_data.jsonl` | Step 4 (k=1 collect) | S2 training |
| `training_data.jsonl` | Step 1 | S3 training |
| `val_data_1a1n10d.jsonl` | `script/build_val_1a1n10d.py` | S1 validation |
| `val_data_1p1n.jsonl` | `script/sample_val_1p1n.py` | S2 validation |

Default paths: `basin_datasets0_analyze/cvrp100_uniform.pkl#<idx>/` for basin data, `perturb_k1_collect/cvrp100_uniform.pkl#<idx>/` for perturbation data. Instances 0-49 for training, 50-54 for validation.
