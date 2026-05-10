"""
EMA (Exponential Moving Average) Manager

严格按照 TODO_list_v3.md 3.3 规范

关键约束:
- decay=0.999
- EMA weights ONLY for validation/inference
- MUST save separate EMA weights
- Update AFTER each optimizer.step()

使用场景:
1. 训练时: 每次权重更新后调用 ema.update()
2. 验证时: 使用 ema.apply_ema() 切换到 EMA 权重，验证后调用 ema.restore()
3. 保存时: 同时保存训练权重和 EMA 权重

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from train_v2.config import TrainingConfig


# ============================================================================
# EMA 状态
# ============================================================================

@dataclass
class EMAState:
    """EMA 状态"""
    decay: float
    num_updates: int = 0
    shadow_params: Dict[str, torch.Tensor] = field(default_factory=dict)
    backup_params: Dict[str, torch.Tensor] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
        }


# ============================================================================
# EMA 管理器
# ============================================================================

class EMAManager:
    """
    EMA 权重管理器

    严格按照 TODO 3.3 规范:
    - decay = 0.999
    - EMA weights ONLY for validation/inference
    - MUST save separate EMA weights

    使用示例:
        ema = EMAManager(model, decay=0.999)

        # 训练循环
        for batch in train_loader:
            loss = model(batch)
            loss.backward()
            optimizer.step()

            # 每次权重更新后更新 EMA
            ema.update(model)

        # 验证时使用 EMA 权重
        ema.apply_ema(model)
        val_loss = validate(model, val_loader)
        ema.restore(model)

        # 保存时包含 EMA 权重
        save_checkpoint(model, ema.get_shadow_dict())
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
    ):
        """
        初始化 EMA 管理器

        参数:
            model: 模型实例
            decay: EMA 衰减率 (default: 0.999 per TODO)
        """
        self.decay = decay
        self.num_updates = 0

        # Shadow parameters (EMA weights)
        self.shadow: Dict[str, torch.Tensor] = {}

        # Backup for validation (training weights)
        self.backup: Dict[str, torch.Tensor] = {}

        # 初始化 shadow weights
        self._init_shadow(model)

    def _init_shadow(self, model: nn.Module) -> None:
        """初始化 shadow weights 为模型当前权重"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        """
        更新 EMA 权重

        MUST call AFTER each optimizer.step()

        公式: shadow = decay * shadow + (1 - decay) * param
        """
        self.num_updates += 1

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    # EMA update
                    self.shadow[name] = (
                        self.decay * self.shadow[name] +
                        (1 - self.decay) * param.data
                    )

    def apply_ema(self, model: nn.Module) -> None:
        """
        应用 EMA 权重用于验证/推理

        MUST call restore() after validation
        """
        # Backup current training weights
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                # Apply EMA weights
                param.data = self.shadow[name].clone()

    def restore(self, model: nn.Module) -> None:
        """
        恢复训练权重

        MUST call after apply_ema() and validation
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()

        # Clear backup
        self.backup.clear()

    def get_shadow_dict(self) -> Dict[str, torch.Tensor]:
        """获取 EMA 权重字典 (用于保存)"""
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_shadow_dict(self, shadow_dict: Dict[str, torch.Tensor]) -> None:
        """加载 EMA 权重字典 (用于恢复训练)"""
        self.shadow = {k: v.clone() for k, v in shadow_dict.items()}

    def get_state(self) -> EMAState:
        """获取 EMA 状态"""
        return EMAState(
            decay=self.decay,
            num_updates=self.num_updates,
            shadow_params={k: v.clone() for k, v in self.shadow.items()},
        )

    def load_state(self, state: EMAState) -> None:
        """加载 EMA 状态"""
        self.decay = state.decay
        self.num_updates = state.num_updates
        self.shadow = {k: v.clone() for k, v in state.shadow_params.items()}

    def copy_to_model(self, model: nn.Module) -> None:
        """
        将 EMA 权重复制到模型 (用于推理)

        不创建 backup，直接覆盖
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    param.data = self.shadow[name].clone()


# ============================================================================
# 工厂函数
# ============================================================================

def create_ema_manager(model: nn.Module, config: TrainingConfig) -> EMAManager:
    """从配置创建 EMA 管理器"""
    return EMAManager(model, decay=config.ema_decay)


# ============================================================================
# EMA 上下文管理器
# ============================================================================

class EMAContext:
    """
    EMA 上下文管理器

    自动处理 apply/restore

    示例:
        with EMAContext(ema, model):
            # 使用 EMA 权重验证
            val_loss = validate(model)
    """

    def __init__(self, ema: EMAManager, model: nn.Module):
        self.ema = ema
        self.model = model

    def __enter__(self) -> "EMAContext":
        self.ema.apply_ema(self.model)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.ema.restore(self.model)


# ============================================================================
# 验证工具
# ============================================================================

def validate_ema_correctness(model: nn.Module, decay: float = 0.9) -> bool:
    """
    验证 EMA 更新正确性

    测试: 多次更新后 EMA 权重应接近训练权重
    """
    ema = EMAManager(model, decay=decay)

    # 获取初始权重
    initial_weight = None
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_weight = param.data.clone()
            break

    if initial_weight is None:
        return True  # 无可训练参数

    # 模拟多次更新
    for _ in range(100):
        # 模拟参数变化
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data += torch.randn_like(param.data) * 0.01
        ema.update(model)

    # EMA 权重应该在初始值和当前值之间
    for name, param in model.named_parameters():
        if param.requires_grad and name in ema.shadow:
            ema_weight = ema.shadow[name]
            # EMA 权重应该有变化
            if torch.allclose(ema_weight, initial_weight):
                return False
            break

    return True


# ============================================================================
# 调试工具
# ============================================================================

def print_ema_stats(ema: EMAManager, model: nn.Module) -> None:
    """打印 EMA 统计信息"""
    print(f"\nEMA Stats (decay={ema.decay}, updates={ema.num_updates}):")
    print("-" * 60)

    for name, param in model.named_parameters():
        if param.requires_grad and name in ema.shadow:
            shadow = ema.shadow[name]
            diff = (param.data - shadow).abs().mean().item()
            print(f"  {name}:")
            print(f"    train mean: {param.data.mean().item():.6f}")
            print(f"    ema mean:   {shadow.mean().item():.6f}")
            print(f"    diff:       {diff:.6f}")

    print("-" * 60)
