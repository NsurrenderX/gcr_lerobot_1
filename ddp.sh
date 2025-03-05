#!/bin/bash

# 多机配置
# MASTER_ADDR=${MASTER_NODE_IP}  # 主节点IP
NUM_NODES=1
NUM_PROCESSES_PER_NODE=2

accelerate launch \
    --config_file ddp_config.yaml \
    --num_processes $((NUM_NODES * NUM_PROCESSES_PER_NODE)) \
    --num_machines $NUM_NODES \
    lerobot/scripts/ddp_train.py \
    --policy.type="pi0" \
    --deepspeed="/home/v-wenhuitan/pi_0_open/lerobot/ds_zero2.json" \
    --dataset.root="/data_16T/lerobot_openx/fmb_dataset_lerobot/" \
    --dataset.repo_id="whatever" \
    --output_dir="/data_16T/deepseek/pi_1" \
    --batch_size=4 \
    --wandb.enable=true \
    --wandb.project="pi0first" \
    --job_name="pi0_ddp_fractal" \
    --save_freq=10 \
    --log_dir="./logs"