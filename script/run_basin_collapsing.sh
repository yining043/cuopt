#!/bin/bash
# Example script to run Basin Collapsing Experiment

# Step 1: Setup modified build (only needed once, or when you want to rebuild)
# Uncomment the line below if you need to rebuild
# ./setup_no_cycle_finder_build.sh

# Step 2: Run the experiment
python3 basin_collapsing_experiment.py \
    --pkl /home/jieyi/cvrp100_uniform.pkl \
    --idx 0 \
    --num_runs 100 \
    --vehicle 30 \
    --basin_dir basin_datasets0 \
    --convergence_threshold 1.0
