#!/bin/bash
# CVRP RL training launcher (supports v2 / v7 and arbitrary hyperparameters),
# run inside a tmux session with unbuffered output for live monitoring.
#   start:   bash run_train_cvrp.sh
#   watch:   tmux attach -t "$SESSION"        (detach: Ctrl-b d)
#   log:     outputs/<runname>/run.log
#
# All settings are env-overridable. Defaults reproduce the v7 t08 K=32 r100 run.
# With the default GPUS=3,2, cuda:0 maps to physical GPU 3 (the update/master GPU).
#
# Train v2 instead:
#   MODE=v2 RUNNAME=rl_cvrp_v2_run AMP_DTYPE=none LOGIT_CLIP=0 ENTROPY_COEF=0.01 \
#     bash run_train_cvrp.sh
set -euo pipefail

SESSION=${SESSION:-rl_cvrp}
RUNNAME=${RUNNAME:-rl_cvrp_flashv7_k32_h2_ep2_t08_e04_clip20_r100}
GPUS=${GPUS:-3,2}
MASTER=${MASTER:-cuda:0}
MODE=${MODE:-v7}
ROUNDS=${ROUNDS:-100}
TIME_LIMIT=${TIME_LIMIT:-3}
BATCH_EPISODES=${BATCH_EPISODES:-4}
K=${K:-32}
REWARD_HORIZON=${REWARD_HORIZON:-2}
LR=${LR:-3e-4}
TEMPERATURE=${TEMPERATURE:-0.8}
ENTROPY_COEF=${ENTROPY_COEF:-0.04}
LOGIT_CLIP=${LOGIT_CLIP:-2.0}
UPDATE_MINIBATCH=${UPDATE_MINIBATCH:-512}
UPDATE_EPOCHS=${UPDATE_EPOCHS:-2}
EVAL_RUNS=${EVAL_RUNS:-10}
EVAL_EVERY=${EVAL_EVERY:-5}
EVAL_SELECTION=${EVAL_SELECTION:-sample}
AMP_DTYPE=${AMP_DTYPE:-none}
CONDA_SH=${CONDA_SH:-/opt/miniconda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-cuopt_dev}

set +u
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u
CUDA_VISIBLE_DEVICES=${GPUS} python - <<'PY'
import os
import sys
import torch

visible = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]
if not torch.cuda.is_available():
    print("[run_train_cvrp] CUDA preflight failed: CUDA unavailable.", file=sys.stderr)
    sys.exit(1)
if torch.cuda.device_count() != len(visible):
    print("[run_train_cvrp] CUDA preflight failed: visible GPU count mismatch.", file=sys.stderr)
    print(f"  expected={len(visible)} (GPUS) got={torch.cuda.device_count()} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", file=sys.stderr)
    sys.exit(1)
print(f"[run_train_cvrp] CUDA preflight ok: visible devices={torch.cuda.device_count()} "
      f"(cuda:0 -> physical GPU {visible[0]})")
PY

mkdir -p "outputs/${RUNNAME}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
    exit 1
fi

tmux new-session -d -s "$SESSION" "
  set +u; source ${CONDA_SH}; conda activate ${CONDA_ENV}; set -u;
  export CUDA_VISIBLE_DEVICES=${GPUS};
  PYTHONUNBUFFERED=1 python -u train_rl.py --time_limit ${TIME_LIMIT} --rounds ${ROUNDS} \
    --batch_episodes ${BATCH_EPISODES} --gpus ${GPUS} --master_device ${MASTER} --k ${K} \
    --reward_horizon ${REWARD_HORIZON} --mode ${MODE} \
    --lr ${LR} --temperature ${TEMPERATURE} --entropy_coef ${ENTROPY_COEF} \
    --logit_clip ${LOGIT_CLIP} \
    --amp_dtype ${AMP_DTYPE} \
    --max_update_steps_per_ep 0 --update_minibatch ${UPDATE_MINIBATCH} --update_epochs ${UPDATE_EPOCHS} \
    --eval_runs ${EVAL_RUNS} --eval_every ${EVAL_EVERY} --eval_selection ${EVAL_SELECTION} \
    --runname ${RUNNAME} 2>&1 | tee outputs/${RUNNAME}/run.log;
  echo '[run_train_cvrp] training process exited. Press enter to close.'; read _
"

echo "Started tmux session: ${SESSION} (runname=${RUNNAME})"
echo "Attach: tmux attach -t ${SESSION}      detach: Ctrl-b d"
echo "Tail:   tail -f outputs/${RUNNAME}/run.log"
echo "Config:"
echo "  CUDA_VISIBLE_DEVICES=${GPUS}; master=${MASTER}"
echo "  mode=${MODE} K=${K} temperature=${TEMPERATURE} eval_selection=${EVAL_SELECTION} amp_dtype=${AMP_DTYPE}"
echo "  entropy_coef=${ENTROPY_COEF} logit_clip=${LOGIT_CLIP} rounds=${ROUNDS} time_limit=${TIME_LIMIT}"
