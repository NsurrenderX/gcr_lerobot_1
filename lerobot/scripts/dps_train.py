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
from deepspeed.utils import get_local_rank, get_global_rank
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed import get_accelerator

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data.distributed import DistributedSampler

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

def init_logger(cfg):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    if cfg.local_rank == 0:
        formatter = logging.Formatter(
            f'[%(asctime)s] [rank: {cfg.local_rank}] [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # # 控制台Handler
        # console_handler = logging.StreamHandler()
        # console_handler.setFormatter(formatter)
        # logger.addHandler(console_handler)
        
        # 文件Handler
        log_path = Path(cfg.log_dir) / f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def update_policy(
    model_engine,
    batch: Any,
) -> tuple[MetricsTracker, dict]:
    
    batch = {k: v.to(model_engine.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    loss, output_dict = model_engine(batch)

    model_engine.backward(loss)
    model_engine.step()
    return loss, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    
    # Initialize DeepSpeed
    deepspeed.init_distributed()
    logger = init_logger(cfg)
    
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(pformat(cfg.to_dict()))
        if cfg.wandb.enable and cfg.wandb.project:
            wandb_logger = WandBLogger(cfg)
        else:
            wandb_logger = None
            logger.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    else:
        wandb_logger = None

    if cfg.seed is not None:
        set_seed(cfg.seed + int(os.environ.get('RANK', 0)))

    # Dataset setup
    dataset = make_dataset(cfg)
    logger.info(f"Dataset: {dataset}")

    # Policy setup
    logger.info("Creating policy...")
    if hasattr(cfg.policy, "tokenizer_max_length"):
        logger.info("Setting model's tokenizer_max_length to 65")
        cfg.policy.tokenizer_max_length=65
    policy = make_policy(
        cfg=cfg.policy,
        device=torch.cuda.current_device(),
        ds_meta=dataset.meta,
    )

    # Environment setup (only in main process)
    eval_env = None
    if int(os.environ.get('RANK', 0)) == 0 and cfg.eval_freq > 0 and cfg.env is not None:
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)

    # Optimizer and scheduler
    logger.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Resume training state
    step = 0

    # Logging setup (main process only)
    if int(os.environ.get('RANK', 0)) == 0:
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
            num_replicas=int(os.environ.get('WORLD_SIZE', 1)),
            rank=int(os.environ.get('RANK', 0))
        )
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=int(os.environ.get('WORLD_SIZE', 1)),
            rank=int(os.environ.get('RANK', 0)),
            shuffle=True,
            seed=cfg.seed
        )

    # DeepSpeed initialization
    model_engine, optimizer, dataloader, lr_scheduler = deepspeed.initialize(
        model=policy,
        optimizer=optimizer,
        training_data=dataset,
        lr_scheduler=lr_scheduler,
        config=cfg.deepspeed,
        model_parameters=policy.parameters(),
    )
    if cfg.resume:
        load_path, client_state = model_engine.load_checkpoint(
            cfg.checkpoint_path,
            load_optimizer_states=True,
            load_lr_scheduler_states=True
        )
        if load_path is not None:
            step = client_state['step']
            logger.info(f"Resumed training from step {step}")
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
        cfg.batch_size * int(os.environ.get('WORLD_SIZE', 1)) * cfg.gradient_accumulation_steps,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step
    )

    # Main training loop
    logger.info(f"Start training on {int(os.environ.get('WORLD_SIZE', 1))} devices")
    total_steps = cfg.steps * cfg.gradient_accumulation_steps
    completed_steps = step * cfg.gradient_accumulation_steps
    
    while completed_steps < total_steps:
        dataloading_s = 0
        start_time = time.perf_counter()
        batch = next(dl_iter)
        dataloading_time = time.perf_counter() - start_time
        dataloading_s += dataloading_time
        
        fwd_bwd_start = time.perf_counter()
        loss, output_dict, grad_norm = update_policy(
            model_engine,
            model_engine.module,
            batch,
            cfg.optimizer.grad_clip_norm,
        )
        fwd_bwd_time = time.perf_counter() - fwd_bwd_start
        
        if model_engine.is_gradient_accumulation_boundary():
            train_tracker.dataloading_s = dataloading_s
            train_tracker.update_s = fwd_bwd_time
            
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            step += 1

            loss_value = loss.detach().mean().item()
            grad_norm_value = grad_norm.item() if grad_norm is not None else 0.0
            
            train_tracker.loss = loss_value
            train_tracker.grad_norm = grad_norm_value
            train_tracker.lr = optimizer.param_groups[0]["lr"]
            train_tracker.step()
            
        completed_steps += 1
        
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0
        
        if cfg.save_checkpoint and is_saving_step and model_engine.local_rank == 0:
            logger.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            client_state = {
                "step": step,
                "config": cfg.to_dict(),
                "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None
            }
            model_engine.save_checkpoint(
                save_dir=checkpoint_dir,
                client_state=client_state,
                tag=f"step_{step}"
            )
            torch.save(client_state, os.path.join(checkpoint_dir, "metadata.pt"))
            update_last_checkpoint(checkpoint_dir)
        
        if int(os.environ.get('RANK', 0)) == 0:
            if is_log_step:
                logger.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    wandb_logger.log_dict(wandb_log_dict, step)
                train_tracker.reset_averages()

            if cfg.env and is_eval_step and model_engine.local_rank == 0:
                step_id = get_step_identifier(step, cfg.steps)
                logger.info(f"Eval policy at step {step}")
                with torch.no_grad():
                    eval_info = eval_policy(
                        eval_env,
                        model_engine.module,
                        cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed + int(os.environ.get('RANK', 0)),
                    )

                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size * int(os.environ.get('WORLD_SIZE', 1)),
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step
                )
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                logger.info(eval_tracker)
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    # Cleanup
    if int(os.environ.get('RANK', 0)) == 0 and eval_env:
        eval_env.close()
    logger.info("Training finished")


if __name__ == "__main__":
    train()