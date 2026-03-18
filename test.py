import torch
import numpy as np
from collections import Counter

# ------------------------- 配置 -------------------------
PATH = "ml_data_grpo.pt"
N_SAMPLE_STATES = 2000   # 参与分析的 state 数量，可按内存/速度调整（如 5000 / 10000）
RANDOM_SEED = 42

# ------------------------- 加载并采样 -------------------------
data = torch.load(PATH, weights_only=False)
sid = data["state_id_tensor"]
cost = data["cost_tensor"]
ratios = cost[:, -1] / cost[:, 0]
sel = data["selected_tensor"].float()

unique_states = sid.unique()
n_states = len(unique_states)
n_sample = min(N_SAMPLE_STATES, n_states)
np.random.seed(RANDOM_SEED)
sampled_states = torch.from_numpy(
    np.random.choice(unique_states.numpy(), size=n_sample, replace=False)
)
mask_trails = torch.isin(sid, sampled_states)

sid_s = sid[mask_trails]
ratios_s = ratios[mask_trails]
cost_s = cost[mask_trails]
sel_s = sel[mask_trails]

print(f"总 states: {n_states}, 采样: {n_sample}; 总 trails: {mask_trails.sum().item()}")

# ================== 实验 1：每组条数 + ratio 跨度 ==================
count_vals = []
spreads = []
for s in sampled_states:
    m = (sid_s == s)
    k = m.sum().item()
    if k < 1:
        continue
    count_vals.append(k)
    if k >= 2:
        g = ratios_s[m]
        spreads.append((g.max() - g.min()).item())

count_vals = np.array(count_vals)
print("\n【实验 1】组大小 & ratio 跨度（采样）")
print(f"每组 trail 数: mean={count_vals.mean():.1f}, median={np.median(count_vals):.0f}, "
      f"min={count_vals.min()}, max={count_vals.max()}")
print(f"只有 1 条: { (count_vals == 1).sum() }, 只有 2 条: {(count_vals == 2).sum()}, >=5 条: {(count_vals >= 5).sum()}")

if spreads:
    spreads = np.array(spreads)
    print(f"组内 ratio 跨度: mean={spreads.mean():.6f}, median={np.median(spreads):.6f}")
    print(f"  p10={np.percentile(spreads, 10):.6f}, p90={np.percentile(spreads, 90):.6f}")
    print(f"  跨度<0.005: {(spreads < 0.005).mean():.2%}, 跨度<0.01: {(spreads < 0.01).mean():.2%}")
else:
    print("(无至少 2 条的组，跳过跨度)")

# ================== 实验 2：组内 selected 的 Jaccard（采样） ==================
jaccard_sims = []
for s in sampled_states:
    m = (sid_s == s)
    if m.sum() < 2:
        continue
    group = sel_s[m]
    n = group.size(0)
    for i in range(n):
        for j in range(i + 1, n):
            inter = (group[i] * group[j]).sum().item()
            union = ((group[i] + group[j]) > 0).float().sum().item()
            if union > 0:
                jaccard_sims.append(inter / union)

if jaccard_sims:
    jaccard_sims = np.array(jaccard_sims)
    print("\n【实验 2】组内 selected Jaccard（采样）")
    print(f"  mean={jaccard_sims.mean():.4f}, median={np.median(jaccard_sims):.4f}")
    print(f"  p10={np.percentile(jaccard_sims, 10):.4f}, p90={np.percentile(jaccard_sims, 90):.4f}")
    print(f"  >0.8: {(jaccard_sims > 0.8).mean():.2%}, >0.9: {(jaccard_sims > 0.9).mean():.2%}")
else:
    print("\n【实验 2】无至少 2 条的组，跳过 Jaccard")

# ================== 实验 3：同 mask 不同 ratio 的冲突（采样） ==================
conflict_groups = 0
total_groups = 0
for s in sampled_states:
    m = (sid_s == s)
    if m.sum() < 2:
        continue
    total_groups += 1
    group_sel = sel_s[m]
    group_ratio = ratios_s[m]
    seen = {}
    has_conflict = False
    for i in range(group_sel.size(0)):
        key = tuple(group_sel[i].bool().tolist())
        r = group_ratio[i].item()
        if key in seen and abs(seen[key] - r) > 0.001:
            has_conflict = True
        seen[key] = r
    if has_conflict:
        conflict_groups += 1

print("\n【实验 3】同 mask 不同 ratio 的组（采样）")
print(f"  有冲突的组: {conflict_groups}/{total_groups} ({conflict_groups/total_groups:.2%})" if total_groups else "  无至少 2 条的组")

# ================== 实验 4：1-step vs 3-step 组内跨度（采样） ==================
ratio_1step = cost_s[:, 1] / cost_s[:, 0]
ratio_3step = cost_s[:, -1] / cost_s[:, 0]
spreads_1, spreads_3 = [], []
for s in sampled_states:
    m = (sid_s == s)
    if m.sum() < 2:
        continue
    g1 = ratio_1step[m]
    g3 = ratio_3step[m]
    spreads_1.append((g1.max() - g1.min()).item())
    spreads_3.append((g3.max() - g3.min()).item())

if spreads_1 and spreads_3:
    spreads_1 = np.array(spreads_1)
    spreads_3 = np.array(spreads_3)
    print("\n【实验 4】1-step vs 3-step 组内跨度（采样）")
    print(f"  1-step: mean={spreads_1.mean():.6f}, median={np.median(spreads_1):.6f}")
    print(f"  3-step: mean={spreads_3.mean():.6f}, median={np.median(spreads_3):.6f}")
    print(f"  跨度比 (3step/1step): {(spreads_3.mean()/spreads_1.mean()):.2f}")