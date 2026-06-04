#!/bin/bash
# CVRP RL training with the multi-update scheme (use all steps + minibatch SGD),
# launched inside a tmux session with UNBUFFERED output for live monitoring.
# Same instance/setup as rl_run2 (CVRP, index 1, 21 vehicles) so results are
# directly comparable to the existing baseline.
#   start:   bash run_train_cvrp.sh
#   watch:   tmux attach -t rl_cvrp        (detach: Ctrl-b d)
#   log:     outputs/<runname>/run.log
#
# GPUs 2,3 for rollouts; master (update) device = GPU 3.
set -euo pipefail

SESSION=${SESSION:-rl_cvrp}
RUNNAME=${RUNNAME:-rl_cvrp_multiupd}
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
  PYTHONUNBUFFERED=1 python -u train_rl.py --time_limit 2 --rounds ${ROUNDS} \
    --batch_episodes 4 --gpus ${GPUS} --master_device ${MASTER} --k 16 \
    --lr 1e-3 --temperature 0.5 --entropy_coef 0.005 \
    --max_update_steps_per_ep 0 --update_minibatch 256 \
    --eval_runs 8 --eval_every 10 --runname ${RUNNAME} 2>&1 | tee outputs/${RUNNAME}/run.log;
  echo '[run_train_cvrp] training process exited. Press enter to close.'; read _
"

echo "Started tmux session '$SESSION' (runname=${RUNNAME}, gpus=${GPUS}, master=${MASTER}) -- CVRP, unbuffered."
echo "  attach: tmux attach -t $SESSION      detach: Ctrl-b d"
echo "  log:    outputs/${RUNNAME}/run.log"
