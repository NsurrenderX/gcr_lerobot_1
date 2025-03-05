#!/bin/bash

# 多机配置
# MASTER_ADDR=${MASTER_NODE_IP}  # 主节点IP
NUM_NODES=2
NUM_PROCESSES_PER_NODE=8

accelerate launch \
    --config_file ddp_config.yaml \
    --num_processes $((NUM_NODES * NUM_PROCESSES_PER_NODE)) \
    --num_machines $NUM_NODES \
    lerobot/scripts/ddp_train.py \
    --policy.type="pi0" \
    --deepspeed="./ds_zero2.json" \
    --dataset.root="/mnt/wangxiaofa/robot_dataset/lerobot-format/bridge_orig_lerobot/" \
    --dataset.repo_id="whatever" \
    --output_dir="/mnt/wangxiaofa/pi_0_ckpts/0306_first" \
    --batch_size=4 \
    --wandb.enable=true \
    --wandb.project="pi0first" \
    --job_name="pi0_0306_first" \
    --save_freq=10000 \
    --log_dir="/mnt/wangxiaofa/logs"