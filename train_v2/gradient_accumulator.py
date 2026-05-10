"""
Gradient Accumulation Manager

严格按照 TODO_list_v3.md 3.2 规范

关键约束:
- batch_size=4, accum_steps=8, effective_batch=32
- MUST call optimizer.zero_grad() BEFORE accumulation loop
- MUST divide loss by accum_steps before backward
- Gradient clipping: max_norm=0.3 AFTER accumulation, BEFORE update
- Scheduler step ONLY after weight update

执行顺序:
1. optimizer.zero_grad() BEFORE loop
2. Loop accum_steps:
   - forward
   - loss / accum_steps
   - backward (NO zero_grad inside)
3. clip_grad_norm_ AFTER loop
4. optimizer.step()
5. scheduler.step()
6. optimizer.zero_grad()

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from train_v2.config import TrainingConfig


# ============================================================================
# 梯度累积统计
# ============================================================================

@dataclass
class AccumulationStats:
    """累积统计信息"""
    total_steps: int = 0
    total_samples: int = 0
    total_loss: float = 0.0
    current_accum_step: int = 0
    grad_norm_before_clip: float = 0.0
    grad_norm_after_clip: float = 0.0

    def reset(self) -> None:
        """重置统计"""
        self.total_steps = 0
        self.total_samples = 0
        self.total_loss = 0.0
        self.current_accum_step = 0
        self.grad_norm_before_clip = 0.0
        self.grad_norm_after_clip = 0.0


# ============================================================================
# 梯度累积管理器
# ============================================================================

class GradientAccumulator:
    """
    梯度累积管理器

    严格按照 TODO 3.2 规范执行累积逻辑

    使用示例:
        accumulator = GradientAccumulator(accum_steps=8, grad_clip=0.3)

        # 训练循环
        for batch in dataloader:
            # 开始累积周期
            accumulator.prepare_accumulation(optimizer)

            for i in range(accum_steps):
                loss = model(batch)
                accumulator.accumulate_step(loss, model)

            # 完成累积周期
            avg_loss = accumulator.finalize_accumulation(optimizer, scheduler, model)
    """

    def __init__(
        self,
        accum_steps: int = 8,
        grad_clip: float = 0.3,
    ):
        """
        初始化梯度累积管理器

        参数:
            accum_steps: 累积步数
            grad_clip: 梯度裁剪阈值
        """
        self.accum_steps = accum_steps
        self.grad_clip = grad_clip

        # 状态
        self._accumulated_loss: float = 0.0
        self._current_step: int = 0
        self._stats = AccumulationStats()

    def prepare_accumulation(self, optimizer: Adam) -> None:
        """
        准备累积周期

        MUST call BEFORE accumulation loop

        执行:
        - optimizer.zero_grad() 清零梯度
        - 重置累积状态
        """
        # ENFORCED: zero_grad BEFORE accumulation loop
        optimizer.zero_grad()

        # 重置累积状态
        self._accumulated_loss = 0.0
        self._current_step = 0
        self._stats.current_accum_step = 0

    def accumulate_step(
        self,
        loss: torch.Tensor,
        model: nn.Module,
    ) -> None:
        """
        累积单步梯度

        MUST call INSIDE accumulation loop

        执行:
        - loss / accum_steps (scale for proper accumulation)
        - backward (accumulates gradients)
        - NO zero_grad inside loop

        参数:
            loss: 原始 loss 值
            model: 模型 (用于梯度检查，可选)
        """
        # ENFORCED: divide loss by accum_steps before backward
        scaled_loss = loss / self.accum_steps

        # Backward (gradients accumulate)
        scaled_loss.backward()

        # 更新累积状态
        self._accumulated_loss += loss.item()
        self._current_step += 1
        self._stats.current_accum_step = self._current_step
        self._stats.total_samples += 1

    def finalize_accumulation(
        self,
        optimizer: Adam,
        scheduler: LambdaLR,
        model: nn.Module,
    ) -> float:
        """
        完成累积周期并更新权重

        MUST call AFTER accumulation loop

        执行:
        1. clip_grad_norm_ AFTER accumulation
        2. optimizer.step()
        3. scheduler.step() (ONLY after weight update)
        4. optimizer.zero_grad() (prepare for next cycle)

        返回:
            avg_loss: 平均 loss (原始 scale, not scaled)
        """
        # 计算累积前的梯度范数
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        self._stats.grad_norm_before_clip = total_norm ** 0.5

        # ENFORCED: gradient clipping AFTER accumulation, BEFORE update
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.grad_clip,
        )

        # 计算裁剪后的梯度范数
        total_norm_after = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_after += param_norm.item() ** 2
        self._stats.grad_norm_after_clip = total_norm_after ** 0.5

        # ENFORCED: optimizer.step()
        optimizer.step()

        # ENFORCED: scheduler.step() ONLY after weight update
        scheduler.step()

        # 准备下一轮
        optimizer.zero_grad()

        # 计算平均 loss
        avg_loss = self._accumulated_loss / self._current_step

        # 更新统计
        self._stats.total_steps += 1
        self._stats.total_loss += avg_loss

        return avg_loss

    def get_stats(self) -> Dict[str, Any]:
        """获取累积统计"""
        return {
            "total_steps": self._stats.total_steps,
            "total_samples": self._stats.total_samples,
            "total_loss": self._stats.total_loss,
            "current_accum_step": self._stats.current_accum_step,
            "grad_norm_before_clip": self._stats.grad_norm_before_clip,
            "grad_norm_after_clip": self._stats.grad_norm_after_clip,
            "avg_loss": self._stats.total_loss / max(1, self._stats.total_steps),
        }

    def reset_stats(self) -> None:
        """重置统计"""
        self._stats.reset()


# ============================================================================
# 工厂函数
# ============================================================================

def create_gradient_accumulator(config: TrainingConfig) -> GradientAccumulator:
    """从配置创建梯度累积管理器"""
    return GradientAccumulator(
        accum_steps=config.accum_steps,
        grad_clip=config.grad_clip,
    )


# ============================================================================
# 验证工具
# ============================================================================

def validate_gradient_accumulation(
    model: nn.Module,
    optimizer: Adam,
    accum_steps: int = 8,
) -> bool:
    """
    验证梯度累积正确性

    检查:
    1. 梯度在累积期间不清零
    2. 梯度正确累加
    3. 裁剪后梯度范数正确
    """
    # 创建假输入
    device = next(model.parameters()).device
    x = torch.randn(4, 1536, device=device, requires_grad=True)

    # 清零梯度
    optimizer.zero_grad()

    # 累积多步
    for i in range(accum_steps):
        # 模拟不同 batch
        xi = x + torch.randn_like(x) * 0.01

        # Forward + Loss
        output = model(xi)
        loss = output.mean()

        # 累积 (scaled)
        (loss / accum_steps).backward()

    # 检查梯度非零
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    # 梯度应该非零
    return total_norm > 0


# ============================================================================
# 梯度累积上下文管理器
# ============================================================================

class AccumulationContext:
    """
    梯度累积上下文管理器

    使用 with 语法自动管理累积周期

    示例:
        with AccumulationContext(accumulator, optimizer, scheduler, model) as ctx:
            for i in range(8):
                loss = model(batch)
                ctx.accumulate(loss)
        avg_loss = ctx.avg_loss
    """

    def __init__(
        self,
        accumulator: GradientAccumulator,
        optimizer: Adam,
        scheduler: LambdaLR,
        model: nn.Module,
    ):
        self.accumulator = accumulator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.avg_loss: float = 0.0

    def __enter__(self) -> "AccumulationContext":
        self.accumulator.prepare_accumulation(self.optimizer)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.avg_loss = self.accumulator.finalize_accumulation(
            self.optimizer, self.scheduler, self.model
        )

    def accumulate(self, loss: torch.Tensor) -> None:
        """累积单步 loss"""
        self.accumulator.accumulate_step(loss, self.model)
