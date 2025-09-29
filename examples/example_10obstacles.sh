#!/usr/bin/env bash
export ACTIVE_CAMERA_NAVIGATION_DIRECTORY=$HOME/workspaces/active-perception-RL-navigation/
python3 -m rl_training.enjoy_aerialgym_10obstacles --train_dir=$ACTIVE_CAMERA_NAVIGATION_DIRECTORY/examples/pre-trained_network --experiment=default_network --env=navigation_active_camera_task  --load_checkpoint_kind=best