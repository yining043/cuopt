# Experiment and Data Processing Workflow

Pipeline: raw basin data → basin analysis → k=5/k=1 perturbation → co-occurrence → distant basins → contrastive learning (embedding model). Env: conda `cuopt_dev`, GPU.

---

## 1. Directories

| Path | Content |
|------|---------|
| `basin_datasets0/` | Raw: `optima.jsonl`, `trajectory.jsonl` per instance |
| `basin_datasets0_analyze/` | Outputs: `training_data.jsonl`, `basin_pairs.jsonl`, `basin_info.jsonl`, `distant_basins.jsonl`, Excel/PNG |
| `perturb_k1_collect/` | k=1 collection JSONL/Excel per instance |
| `out/` | Model checkpoints (`.pt`) |

---

## 2. Pipeline (7 steps)

```
basin_datasets0
    → 1. Per-trial basin analysis     (run_all_trials.sh + analyze_basin.py)     → training_data.jsonl, PNGs, Excel
    → 2. Run-level statistics         (run_basin_analysis_all_runs.sh + analyze_basin_statistics.py) → basin_statistics_analysis_*.xlsx
    → 3. k=5 perturbation            (run_perturb.sh + perturb.py)               → merged k=5 JSONL/Excel
    → 4. k=1 sample collection       (run_perturb_k1_collect.sh + perturb_k1_collect.py) → perturb_k1_collect/*.jsonl
    → 5. Co-occurrence & viz         (generate_cooccurrence.sh)                  → basin_pairs.jsonl, basin_info.jsonl, network PNG, summary Excel
    → 6. Distant basins              (run_find_distant_basins_batch.sh + find_distant_basins.py) → distant_basins.jsonl
    → 7. Contrastive training        (train_basin_contrastive.py)                → out/*.pt
```

---

## 3. Commands (single instance, index 0)

| Step | Command | In → Out |
|------|---------|----------|
| 1 | `./run_all_trials.sh [idx] [gpu] [max_runs]` | trajectory → training_data.jsonl, PNGs |
| 2 | `./run_basin_analysis_all_runs.sh [idx] [data_dir]` | Excel/training_data → basin_statistics_analysis_*.xlsx |
| 3 | `./run_perturb.sh remove_and_insert [idx] [gpu] [batch_size]` | optima.jsonl → remove_and_insert_training_data.ALL_r30.jsonl |
| 4 | `./run_perturb_k1_collect.sh [idx] [gpu] [batch_size] [max_first_runs]` | k=5 merged + optima → k1_collection_*.jsonl |
| 5 | `./generate_cooccurrence.sh [idx or 0-5]` | training_data → basin_pairs.jsonl, basin_info.jsonl, network PNG, summary Excel |
| 6 | `./run_find_distant_basins_batch.sh [instances]` (default 0-29) or `find_distant_basins.py --instances 0` | basin_pairs + basin_info → distant_basins.jsonl |
| 7 | `python train_basin_contrastive.py --basin_pairs ... --basin_info ... --distant_basins ... --save out/model.pt` | JSONL → .pt |

**Step 5 extra**:  
- `summarize_basin_pair_signs.py` — counts pos/neg/zero co-occurrence pairs per basin → Excel (append mode). If "By Instance" rows exceed Excel limit (1,048,576), detail is written to `{output_stem}_by_instance.csv` and only the summary sheet to the .xlsx. Run after Step 5.  
  - Example: `python summarize_basin_pair_signs.py --instances 0-5 --output basin_pair_signs_summary.xlsx` (default instances: 0-10; optional: `--min_count`, `--max_runs`, `--max_records`, `--max_basins_per_record`).  
- `label_basin_pairs_by_sign.py` — for each `basin_pairs.jsonl`, add `sign` (`positive`/`negative`/`zero`) and `coocc_value` fields based on the co-occurrence matrix computed from `training_data.jsonl`, and (optionally) build an `n×n` sign mask (`{1, -1, 0}`) plus basin index order.
  - Single instance example (instance 0, `cvrp100_uniform.pkl#0`):

    ```bash
    python label_basin_pairs_by_sign.py \
      --basin_pairs "basin_datasets0_analyze/cvrp100_uniform.pkl#0/basin_pairs.jsonl" \
      --training_data "basin_datasets0_analyze/cvrp100_uniform.pkl#0/training_data.jsonl" \
      --min_count 10 \
      --max_runs 10 \
      --output_mask "basin_datasets0_analyze/cvrp100_uniform.pkl#0/anchor_basin_sign_mask.npy" \
      --output_basin_list "basin_datasets0_analyze/cvrp100_uniform.pkl#0/anchor_basin_sign_order.json"
    ```

  - Multi-instance batch example (instances 0–29, same naming pattern `cvrp100_uniform.pkl#<idx>`):

    ```bash
    for idx in $(seq 0 29); do
      base="basin_datasets0_analyze/cvrp100_uniform.pkl#${idx}"
      python label_basin_pairs_by_sign.py \
        --basin_pairs "${base}/basin_pairs.jsonl" \
        --training_data "${base}/training_data.jsonl" \
        --min_count 10 \
        --max_runs 10 \
        --output_mask "${base}/anchor_basin_sign_mask.npy" \
        --output_basin_list "${base}/anchor_basin_sign_order.json"
    done
    ```

---

## 4. Merge & helpers

| Script | Purpose |
|--------|---------|
| `merge_perturb_batches_jsonl_only.py` | Merge k=5 batches; dedup |
| `merge_k1_collect_batches.py` | Merge k=1 batches; e.g. `python -u merge_k1_collect_batches.py --operator_type remove_and_insert --runs 30 --delete_batches --out_dir perturb_k1_collect/cvrp100_uniform.pkl#29` |
| `merge_basin_statistics.py` | Merge basin_statistics_analysis_*.xlsx |
| `merge_perturb_batches.py` | Full merge (alt.) |
| `reset_basin_info.py` | Reset basin_info from basin_pairs; regenerate distant_basins |

**Pipeline scripts**: `run_all_trials.sh`, `run_basin_analysis_all_runs.sh`, `run_perturb.sh`, `run_perturb_k1_collect.sh`, `generate_cooccurrence.sh`, `run_find_distant_basins_batch.sh`; `analyze_basin.py`, `analyze_basin_statistics.py`, `perturb.py`, `perturb_k1_collect.py`, `generate_basin_pairs.py`, `visualize_basin_pairs.py`, `summarize_basin_statistics.py`, `find_distant_basins.py`, `train_basin_contrastive.py`. **Dependencies**: `utils.py`, `visualize_basin_cooccurrence_matrix.py`, `test_basin_pybind.py`, `test_load_data.py`.

**k=1 variance** (optional): `./run_k1_variance.sh [idx] [gpu] [num_optima] [repeats]` → then `python analyze_k1_repeats.py --repeats_base_dir .../k1_variance --instance_index N --repeats M` → boxplots in `k1_variance_rep1/<instance_id>/`.

**Other**: `summarize_basin_pair_signs.py` (Step 5 analysis); `generate_exp2_basin_data.py` (exp2 basin_pairs/basin_info/distant_basins); `plot_stn.py` (STN from trajectory.jsonl → static figure + GIF).

---

## 5. Recommended order (one instance)

1. `./run_all_trials.sh 0`  
2. `./run_basin_analysis_all_runs.sh 0`  
3. `./run_perturb.sh remove_and_insert 0`  
4. `./run_perturb_k1_collect.sh 0`  
5. `./generate_cooccurrence.sh 0`  
6. Single: `python find_distant_basins.py --instances 0`. Batch: `./run_find_distant_basins_batch.sh [instances]` (default 0-29, e.g. `0-9` or `0,1,2`).  
7. `train_basin_contrastive.py` with generated paths and `--save out/model.pt`

Multi-instance: loop 1–4 per instance; run 5–6 in batch; 7 per instance or combined.

---

## 6. CLI summary

`analyze_basin.py`: `--pkl`, `--idx`, `--run_id`, `--trial_id`, `--num_runs`, `--basin_dir`, `--hgs`  
`analyze_basin_statistics.py`: `--data_dir`, `--instance_index`, `--run_id`  
`perturb.py`: `--instance_path`, `--instance_index`, `--operator_type`, `--k`, `--num_local_search_runs`, `--start_idx`, `--max_optima`, `--output_base_dir`, `--batch_id`  
`perturb_k1_collect.py`: `--instance_path`, `--instance_index`, `--basin_base_dir`, `--existing_data_dir`, `--output_base_dir`, `--resume`  
`find_distant_basins.py`: `--instances`, `--max_runs`, `--top_k`, `--min_basin_total_count`  
`train_basin_contrastive.py`: `--basin_pairs`, `--basin_info`, `--distant_basins`, `--save`  
`analyze_k1_repeats.py`: `--repeats_base_dir`, `--instance_index`, `--repeats`  
`label_basin_pairs_by_sign.py`: `--basin_pairs`, `--training_data`, optional `--output_jsonl`, `--output_mask`, `--output_basin_list`, `--min_count`, `--max_runs`, `--max_records`, `--max_basins_per_record`

Run `script --help` for full options.

---

## 7. Scripts not in main doc (review)

**Root .sh**: `build.sh`, `build_pybind.sh`, `print_env.sh`  
**Root .py**: `analyze_intermediate_basins.py`, `all_data_stats_excel.py`, `test_vrp_pybind.py`, `tmp.py`  
**script/*.sh**: `run_all_intermediate_basins.sh`, `run_all_trials.sh`, `run_basin_collapsing.sh`, `run_collect_only.sh`, `run_complete_basin_runs.sh`, `run_pybind_comparison.sh`  
**script/*.py**: `analyze_basin_from_optima.py`, `analyze_basin_gap_distribution.py`, `analyze_basin_heatmap.py`, `analyze_intermediate_basins.py`, `basin_collapsing_experiment.py`, `compare_pybind_cuopt.py`, `compute_critical_gap_from_training_data.py`, `copy_trial_plots.py`, `count_basin_runs.py`, `generate_missing_visualizations.py`, `visualize_basin_collapsing.py`, `visualize_commitment_points.py`
