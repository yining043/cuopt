#!/bin/bash
# CVRP RL training with full-feedback candidate-subset labels, launched inside
# a tmux session with unbuffered output for live monitoring.
#   start:   bash run_train_cvrp.sh
#   watch:   tmux attach -t rl_cvrp        (detach: Ctrl-b d)
#   log:     outputs/<runname>/run.log
#
# Defaults use physical GPUs 2,3 for rollouts. With CUDA_VISIBLE_DEVICES=2,3,
# MASTER=cuda:1 maps the master/update process to physical GPU 3.
set -euo pipefail

SESSION=${SESSION:-rl_cvrp}
RUNNAME=${RUNNAME:-rl_cvrp_h2_ep2_lr3e4}
GPUS=${GPUS:-2,3}
MASTER=${MASTER:-cuda:1}
ROUNDS=${ROUNDS:-180}
TIME_LIMIT=${TIME_LIMIT:-2}
BATCH_EPISODES=${BATCH_EPISODES:-4}
K=${K:-16}
REWARD_HORIZON=${REWARD_HORIZON:-2}
LR=${LR:-3e-4}
TEMPERATURE=${TEMPERATURE:-0.8}
ENTROPY_COEF=${ENTROPY_COEF:-0.01}
UPDATE_MINIBATCH=${UPDATE_MINIBATCH:-256}
UPDATE_EPOCHS=${UPDATE_EPOCHS:-2}
EVAL_RUNS=${EVAL_RUNS:-8}
EVAL_EVERY=${EVAL_EVERY:-10}
CONDA_SH=${CONDA_SH:-/opt/miniconda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-cuopt_dev}

mkdir -p "outputs/${RUNNAME}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
    exit 1
fi

tmux new-session -d -s "$SESSION" "
  source ${CONDA_SH}; conda activate ${CONDA_ENV};
  export CUDA_VISIBLE_DEVICES=${GPUS};
  PYTHONUNBUFFERED=1 python -u train_rl.py --time_limit ${TIME_LIMIT} --rounds ${ROUNDS} \
    --batch_episodes ${BATCH_EPISODES} --gpus ${GPUS} --master_device ${MASTER} --k ${K} \
    --reward_horizon ${REWARD_HORIZON} \
    --lr ${LR} --temperature ${TEMPERATURE} --entropy_coef ${ENTROPY_COEF} \
    --max_update_steps_per_ep 0 --update_minibatch ${UPDATE_MINIBATCH} --update_epochs ${UPDATE_EPOCHS} \
    --eval_runs ${EVAL_RUNS} --eval_every ${EVAL_EVERY} --runname ${RUNNAME} 2>&1 | tee outputs/${RUNNAME}/run.log;
  echo '[run_train_cvrp] training process exited. Press enter to close.'; read _
"

echo "Started tmux session '$SESSION' (runname=${RUNNAME}, gpus=${GPUS}, master=${MASTER}) -- CVRP, unbuffered."
echo "  CUDA_VISIBLE_DEVICES=${GPUS}; master=${MASTER}"
echo "  reward_horizon_train=${REWARD_HORIZON} reward_horizon_eval=1 update_epochs=${UPDATE_EPOCHS} time_limit=${TIME_LIMIT}"
echo "  attach: tmux attach -t $SESSION      detach: Ctrl-b d"
echo "  log:    outputs/${RUNNAME}/run.log"
