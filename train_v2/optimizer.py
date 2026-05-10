"""
Optimizer and LR Scheduler Setup

严格按照 TODO_list_v3.md 3.1 规范

关键约束:
- Adam, NO weight decay
- betas=(0.95, 0.999), eps=1e-8
- FP32 state for optimizer
- LR: 6e-5 (8x expansion)
- Warmup: 4000 steps linear + 4000 steps cosine
- Min LR: 1e-5
- Scheduler step ONLY after weight update (NOT every accumulation step)

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from train_v2.config import TrainingConfig


# ============================================================================
# LR Schedule 实现
# ============================================================================

def create_lr_lambda(config: TrainingConfig):
    """
    创建学习率调度函数

    Warmup + Cosine decay per TODO 3.1

    流程:
    1. step < warmup_steps: 线性增长到 lr
    2. step >= warmup_steps: Cosine 衰减到 min_lr
    """
    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            # Linear warmup: 0 -> 1
            return step / config.warmup_steps
        else:
            # Cosine decay
            progress = (step - config.warmup_steps) / (config.total_steps - config.warmup_steps)
            # progress: 0 -> 1
            # cosine_factor: 1 -> 0
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            # 返回 min_lr 相对于 lr 的比例
            return (config.min_lr / config.lr) + (1 - config.min_lr / config.lr) * cosine_factor

    return lr_lambda


# ============================================================================
# Optimizer 创建
# ============================================================================

def create_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> Adam:
    """
    创建 Adam 优化器

    严格按照 TODO 3.1 规范:
    - NO weight decay (强制为 0)
    - betas=(0.95, 0.999)
    - eps=1e-8

    注意:
    - FP32 state 由 PyTorch 自动处理
    - 不使用 weight_decay (ENFORCED)
    """
    # 强制验证 weight_decay
    assert config.weight_decay == 0.0, \
        f"weight_decay 必须为 0.0 per TODO, got {config.weight_decay}"

    optimizer = Adam(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=0.0,  # ENFORCED: NO weight decay
        foreach=False,     # 更安全的梯度更新
    )

    return optimizer


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: TrainingConfig,
) -> Tuple[Adam, LambdaLR]:
    """
    创建优化器和学习率调度器

    返回:
        (optimizer, scheduler)

    注意:
    - scheduler.step() 仅在 weight update 后调用
    - 梯度累积期间不调用 scheduler.step()
    """
    optimizer = create_optimizer(model, config)

    lr_lambda = create_lr_lambda(config)
    scheduler = LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


# ============================================================================
# 学习率监控
# ============================================================================

def get_current_lr(optimizer: Adam) -> float:
    """获取当前学习率"""
    return optimizer.param_groups[0]["lr"]


def get_lr_stats(optimizer: Adam) -> dict:
    """获取学习率统计信息"""
    lrs = [pg["lr"] for pg in optimizer.param_groups]
    return {
        "lr": lrs[0] if lrs else 0.0,
        "num_param_groups": len(optimizer.param_groups),
    }


# ============================================================================
# 优化器状态管理
# ============================================================================

def save_optimizer_state(optimizer: Adam, path: str) -> None:
    """保存优化器状态"""
    torch.save(optimizer.state_dict(), path)


def load_optimizer_state(optimizer: Adam, path: str, device: str = "cpu") -> None:
    """加载优化器状态"""
    state_dict = torch.load(path, map_location=device)
    optimizer.load_state_dict(state_dict)


# ============================================================================
# 学习率预热验证
# ============================================================================

def validate_warmup_schedule(config: TrainingConfig) -> dict:
    """
    验证预热调度配置

    返回关键检查点信息
    """
    lr_lambda = create_lr_lambda(config)

    checkpoints = [
        0,
        config.warmup_steps // 2,
        config.warmup_steps,
        (config.warmup_steps + config.total_steps) // 2,
        config.total_steps,
    ]

    results = {}
    for step in checkpoints:
        lr_mult = lr_lambda(step)
        actual_lr = config.lr * lr_mult
        results[f"step_{step}"] = {
            "lr_multiplier": lr_mult,
            "actual_lr": actual_lr,
        }

    return results


# ============================================================================
# 调试工具
# ============================================================================

def print_lr_schedule(config: TrainingConfig, n_points: int = 10) -> None:
    """打印学习率调度"""
    lr_lambda = create_lr_lambda(config)

    print(f"\nLR Schedule (lr={config.lr}, warmup={config.warmup_steps}, total={config.total_steps}):")
    print("-" * 60)

    steps = [i * config.total_steps // n_points for i in range(n_points + 1)]
    steps.append(config.warmup_steps)  # 确保包含 warmup 结束点
    steps = sorted(set(steps))

    for step in steps:
        lr_mult = lr_lambda(step)
        actual_lr = config.lr * lr_mult
        phase = "warmup" if step < config.warmup_steps else "cosine"
        print(f"  step {step:5d}: lr={actual_lr:.2e} (mult={lr_mult:.4f}) [{phase}]")

    print("-" * 60)
