#!/bin/bash
# Launch CVRPTW RL training inside a tmux session for easy monitoring.
#   start:   bash run_train_cvrptw.sh
#   watch:   tmux attach -t rl_cvrptw     (detach: Ctrl-b then d)
#   log:     outputs/<runname>/run.log
#
# GPUs 2,3 for rollouts; master (update) device = GPU 3.
set -euo pipefail

SESSION=${SESSION:-rl_cvrptw}
RUNNAME=${RUNNAME:-rl_cvrptw1}
GPUS=${GPUS:-2,3}
MASTER=${MASTER:-cuda:3}
ROUNDS=${ROUNDS:-180}

mkdir -p "outputs/${RUNNAME}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
    exit 1
fi

tmux new-session -d -s "$SESSION" "
  source ~/.conda/etc/profile.d/conda.sh; conda activate cuopt_dev;
  python train_rl.py --data_pt data/cvrptw_inst1.pt --time_limit 2 --rounds ${ROUNDS} \
    --batch_episodes 8 --gpus ${GPUS} --master_device ${MASTER} --k 16 \
    --lr 1e-3 --temperature 0.5 --entropy_coef 0.005 \
    --max_update_steps_per_ep 0 --update_minibatch 256 \
    --eval_runs 8 --eval_every 10 --runname ${RUNNAME} 2>&1 | tee outputs/${RUNNAME}/run.log;
  echo '[run_train_cvrptw] training process exited. Press enter to close.'; read _
"

echo "Started tmux session '$SESSION' (runname=${RUNNAME}, gpus=${GPUS}, master=${MASTER})."
echo "  attach: tmux attach -t $SESSION      detach: Ctrl-b d"
echo "  log:    outputs/${RUNNAME}/run.log"
