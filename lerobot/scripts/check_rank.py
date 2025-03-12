#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
import os
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import deepspeed
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DistributedDataParallelKwargs

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler, DistEpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy

class AccelerateLogger:
    def __init__(self, accelerator: Accelerator, cfg):
        self.rank = cfg.local_rank
        self.accelerator = accelerator
        log_path = os.path.join(cfg.log_dir, f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        log_dir = os.path.dirname(log_path)
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = Path(log_path)
        self.log_file.parent.mkdir(exist_ok=True)
        
        # 主进程初始化日志文件
        if self.accelerator.is_main_process:
            with open(self.log_file, 'w') as f:
                f.write(f"Training Log - Start at {datetime.now()}\n")

    def log(self, message: str, level: str = "INFO"):
        """核心日志方法"""
        formatted = f"[{os.getpid()}]-[rank: {self.rank}]-[{datetime.now().strftime('%H:%M:%S')}]-[{level}] - {message}"
        
        # 主进程输出到控制台和文件
        if self.accelerator.is_main_process:
            print(formatted)
            with open(self.log_file, 'a') as f:
                f.write(formatted + "\n")
        else:
            # 其他进程仅打印
            self.accelerator.print(formatted)

    def critical(self, message: str):
        self.log(message, "CRITICAL")
        self.accelerator.fatal_error()

    def error(self, message: str):
        self.log(message, "ERROR")

    def warning(self, message: str):
        self.log(message, "WARNING")

    def info(self, message: str):
        self.log(message, "INFO")

    def debug(self, message: str):
        self.log(message, "DEBUG")
        
def load_training_state(checkpoint_path, optimizer, lr_scheduler, accelerator):
    # 加载accelerate状态
    accelerator.load_state(checkpoint_path)
    
    # 加载额外元数据
    metadata = torch.load(checkpoint_path / "metadata.pt")
    step = metadata["step"]
    
    # 恢复优化器和学习率调度器状态
    if optimizer is not None:
        optimizer.load_state_dict(accelerator.get_optimizer_state(optimizer))
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(accelerator.get_scheduler_state(lr_scheduler))
    
    return step, optimizer, lr_scheduler

def update_policy(
    accelerator: Accelerator,
    policy: PreTrainedPolicy,
    batch: Any,
    grad_clip_norm: float,
) -> tuple[MetricsTracker, dict]:
    
    policy.train()
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)

    accelerator.backward(loss)
    
    grad_norm = None
    
    if accelerator.sync_gradients:
        grad_norm = accelerator.clip_grad_norm_(
            policy.parameters(),
            grad_clip_norm,
        )
    return loss, output_dict, grad_norm


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    
    # Initialize Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
        static_graph=False
    )
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=cfg.deepspeed)
    accelerator = Accelerator(
            deepspeed_plugin=DeepSpeedPlugin(
                    hf_ds_config=cfg.deepspeed,
            ),
        )
    
    logger = AccelerateLogger(accelerator, cfg)
    
    if accelerator.is_main_process:
        logger.info(pformat(cfg.to_dict()))
        if cfg.wandb.enable and cfg.wandb.project:
            wandb_logger = WandBLogger(cfg)
        else:
            wandb_logger = None
            logger.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    else:
        wandb_logger = None

    if cfg.seed is not None:
        set_seed(cfg.seed + accelerator.process_index)  # Add process index for deterministic seeding

    # Dataset and policy setup
    dataset = make_dataset(cfg)
    logger.info(f"Dataset: {dataset}")

    # Policy setup
    logger.info("Creating policy...")
    if hasattr(cfg.policy, "tokenizer_max_length"):
        logger.info("Setiing model's tokenizer_max_length to 60")
        cfg.policy.tokenizer_max_length=65
    policy = make_policy(
        cfg=cfg.policy,
        device=accelerator.device,
        ds_meta=dataset.meta,
    )

    # Environment setup (only in main process)
    eval_env = None
    if accelerator.is_main_process and cfg.eval_freq > 0 and cfg.env is not None:
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)

    # Optimizer and scheduler
    logger.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Resume training state
    step = 0
    if cfg.resume:
        accelerator.load_state(cfg.checkpoint_path)
        if accelerator.is_main_process:
            metadata = torch.load(cfg.checkpoint_path / "metadata.pt")
            step = int(metadata["step"])
            # 广播step到所有进程
            step_tensor = torch.tensor([step], device=accelerator.device)
            torch.distributed.broadcast(step_tensor, src=0)
            step = step_tensor.item()
        else:
            # 非主进程接收step值
            step_tensor = torch.tensor([0], device=accelerator.device)
            torch.distributed.broadcast(step_tensor, src=0)
            step = step_tensor.item()
        

    # Logging setup (main process only)
    if accelerator.is_main_process:
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())
        logger.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logger.info(f"{cfg.env.task=}")
        logger.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logger.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logger.info(f"{dataset.num_episodes=}")
        logger.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logger.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Dataloader setup
    if hasattr(cfg.policy, "drop_n_last_frames"):
        sampler = DistEpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index
        )
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=cfg.local_rank,
            shuffle=True,
            seed=cfg.seed
        )
        shuffle = False

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    # Prepare components with Accelerator
    policy, optimizer, lr_scheduler, dataloader = accelerator.prepare(
        policy, optimizer, lr_scheduler, dataloader
    )
    dl_iter = cycle(dataloader)

    # Metrics setup
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "optim_s": AverageMeter("optim_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps,  # Total batch size across all processes
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step
    )

    # Main training loop
    accelerator.wait_for_everyone()
    logger.info(f"Start training on {accelerator.num_processes} devices")
    total_steps = cfg.steps * cfg.gradient_accumulation_steps
    completed_steps = step * cfg.gradient_accumulation_steps
    for _ in range(completed_steps, total_steps):
        dataloading_s = 0
        start_time = time.perf_counter()
        batch = next(dl_iter)
        dataloading_time = time.perf_counter() - start_time
        dataloading_s += dataloading_time

        print(os.environ.get('RANK'))
        print(os.environ.get('LOCAL_RANK'))
        print(os.environ.get('WORLD_SIZE'))
        print(os.environ.get('MASTER_ADDR'))
        
        break
        


if __name__ == "__main__":
    train()