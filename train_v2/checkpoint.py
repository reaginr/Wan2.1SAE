"""
Checkpoint Manager

严格按照 TODO_list_v3.md 3.6 规范

保存内容:
- SAE 权重 (encoder/decoder/bias)
- EMA 权重 (shadow parameters)
- Optimizer 状态 (FP32)
- Scheduler 状态
- 训练步数
- 层索引
- 配置元数据

目录结构:
run_dir/
├── train_state.json           # 全局训练状态
├── block_out.layer14/         # 层目录
│   ├── sae_config.json        # SAE 配置
│   ├── sae_latest.pt          # 最新权重
│   └── sae_step{N}.pt         # 历史版本
├── block_out.layer19/
│   └── ...
└── ...

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from train_v2.config import TrainingConfig
from train_v2.ema import EMAManager
from train_v2.sae_engine import SAEEngine

# 延迟导入
try:
    from wan.modules.sae_new import SparseAutoEncoder
    from wan.sae.sae_run_naming import SAERunLocator, save_json, load_json
    WAN_AVAILABLE = True
except ImportError:
    SparseAutoEncoder = None
    SAERunLocator = None
    save_json = None
    load_json = None
    WAN_AVAILABLE = False

logger = logging.getLogger(__name__)

# Checkpoint 版本
CHECKPOINT_VERSION = "3.0"


# ============================================================================
# Checkpoint 内容
# ============================================================================

@dataclass
class CheckpointData:
    """Checkpoint 数据结构"""
    # 权重
    state_dict: Dict[str, torch.Tensor]

    # 训练状态
    step: int
    layer_idx: int
    hook_mode: str

    # 配置
    sae_config: Dict[str, Any]
    training_config: Dict[str, Any]

    # EMA
    ema_shadow: Optional[Dict[str, torch.Tensor]] = None

    # 元数据
    timestamp: float = 0.0
    version: str = CHECKPOINT_VERSION

    # 额外信息
    extra_info: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.extra_info is None:
            self.extra_info = {}


# ============================================================================
# Checkpoint 管理器
# ============================================================================

class CheckpointManager:
    """
    Checkpoint 管理器

    严格按照 TODO 3.6 规范保存/加载 checkpoint

    使用示例:
        manager = CheckpointManager(config)

        # 保存
        manager.save(
            run_dir="sae_runs/exp1",
            layer_idx=14,
            step=1000,
            engine=engine,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        # 加载
        data = manager.load("sae_runs/exp1/block_out.layer14/sae_latest.pt")
        manager.restore(data, engine, ema, optimizer, scheduler)
    """

    def __init__(self, config: TrainingConfig):
        """
        初始化 Checkpoint 管理器

        参数:
            config: 训练配置
        """
        self.config = config

    def save(
        self,
        run_dir: str,
        layer_idx: int,
        step: int,
        engine: SAEEngine,
        ema: EMAManager,
        optimizer: Adam,
        scheduler: LambdaLR,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        保存 checkpoint

        参数:
            run_dir: 实验目录
            layer_idx: 层索引
            step: 训练步数
            engine: SAE Engine
            ema: EMA 管理器
            optimizer: 优化器
            scheduler: 学习率调度器
            extra_info: 额外信息

        返回:
            checkpoint 路径
        """
        # 创建 locator
        loc = SAERunLocator(
            run_dir=run_dir,
            hook_mode=self.config.hook_mode,
            layer_idx=layer_idx,
        )

        # 确保目录存在
        loc.artifact_dir().mkdir(parents=True, exist_ok=True)

        # 构建 checkpoint 数据
        ckpt_dict = {
            # SAE 权重
            "state_dict": engine.sae.state_dict(),

            # 训练状态
            "step": step,
            "layer_idx": layer_idx,
            "hook_mode": self.config.hook_mode,

            # 配置
            "sae_config": engine.sae.config.to_dict(),
            "training_config": self.config.to_dict(),

            # EMA 权重
            "ema_shadow": ema.get_shadow_dict(),
            "ema_num_updates": ema.num_updates,

            # Optimizer 状态
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),

            # 元数据
            "timestamp": time.time(),
            "version": CHECKPOINT_VERSION,

            # 额外信息
            "extra_info": extra_info or {},
        }

        # 保存 .pt 文件
        ckpt_path = loc.latest_ckpt_path()
        torch.save(ckpt_dict, ckpt_path)

        # 同时保存带步数的版本
        if step > 0:
            step_path = loc.ckpt_path(step)
            torch.save(ckpt_dict, step_path)

        # 保存 .json 配置文件 (便于查看)
        save_json(
            loc.config_path(),
            {
                "sae": engine.sae.config.to_dict(),
                "hook": {
                    "hook_mode": self.config.hook_mode,
                    "layer_idx": layer_idx,
                },
                "step": step,
                "version": CHECKPOINT_VERSION,
            },
        )

        # 更新全局训练状态
        self._update_train_state(run_dir, layer_idx, step)

        logger.info(f"Saved checkpoint: {ckpt_path} (step={step})")

        return ckpt_path

    def load(
        self,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> CheckpointData:
        """
        加载 checkpoint

        参数:
            checkpoint_path: checkpoint 文件路径
            device: 目标设备

        返回:
            CheckpointData
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # 加载 .pt 文件
        ckpt_dict = torch.load(path, map_location=device)

        # 构建 CheckpointData
        data = CheckpointData(
            state_dict=ckpt_dict["state_dict"],
            step=ckpt_dict["step"],
            layer_idx=ckpt_dict["layer_idx"],
            hook_mode=ckpt_dict["hook_mode"],
            sae_config=ckpt_dict["sae_config"],
            training_config=ckpt_dict["training_config"],
            ema_shadow=ckpt_dict.get("ema_shadow"),
            timestamp=ckpt_dict.get("timestamp", 0.0),
            version=ckpt_dict.get("version", "1.0"),
            extra_info=ckpt_dict.get("extra_info", {}),
        )

        # 保存原始字典用于恢复 optimizer/scheduler
        data._raw_dict = ckpt_dict

        return data

    def restore(
        self,
        data: CheckpointData,
        engine: SAEEngine,
        ema: Optional[EMAManager] = None,
        optimizer: Optional[Adam] = None,
        scheduler: Optional[LambdaLR] = None,
        strict: bool = True,
    ) -> None:
        """
        从 checkpoint 恢复状态

        参数:
            data: CheckpointData
            engine: SAE Engine
            ema: EMA 管理器
            optimizer: 优化器
            scheduler: 学习率调度器
            strict: 是否严格匹配权重
        """
        # 恢复 SAE 权重
        engine.sae.load_state_dict(data.state_dict, strict=strict)

        # 恢复 EMA
        if ema is not None and data.ema_shadow is not None:
            ema.load_shadow_dict(data.ema_shadow)
            if hasattr(data, "_raw_dict"):
                ema.num_updates = data._raw_dict.get("ema_num_updates", 0)

        # 恢复 Optimizer
        if optimizer is not None and hasattr(data, "_raw_dict"):
            optimizer.load_state_dict(data._raw_dict["optimizer_state"])

        # 恢复 Scheduler
        if scheduler is not None and hasattr(data, "_raw_dict"):
            scheduler.load_state_dict(data._raw_dict["scheduler_state"])

        logger.info(f"Restored checkpoint: step={data.step}, layer={data.layer_idx}")

    def _update_train_state(self, run_dir: str, layer_idx: int, step: int) -> None:
        """更新全局训练状态文件"""
        state_path = Path(run_dir) / "train_state.json"

        # 加载现有状态
        state = {}
        if state_path.exists():
            try:
                state = load_json(state_path)
            except Exception:
                pass

        # 更新状态
        state["current_step"] = step
        state["current_layer"] = layer_idx
        state["last_update"] = time.time()
        state["version"] = CHECKPOINT_VERSION

        # 保存
        save_json(state_path, state)

    def get_latest_checkpoint(
        self,
        run_dir: str,
        layer_idx: int,
    ) -> Optional[Path]:
        """
        获取最新的 checkpoint 路径

        参数:
            run_dir: 实验目录
            layer_idx: 层索引

        返回:
            checkpoint 路径或 None
        """
        loc = SAERunLocator(
            run_dir=run_dir,
            hook_mode=self.config.hook_mode,
            layer_idx=layer_idx,
        )

        ckpt_path = loc.latest_ckpt_path()
        if ckpt_path.exists():
            return ckpt_path

        return None

    def list_checkpoints(
        self,
        run_dir: str,
        layer_idx: int,
    ) -> list:
        """
        列出所有 checkpoint

        参数:
            run_dir: 实验目录
            layer_idx: 层索引

        返回:
            checkpoint 路径列表
        """
        loc = SAERunLocator(
            run_dir=run_dir,
            hook_mode=self.config.hook_mode,
            layer_idx=layer_idx,
        )

        artifact_dir = loc.artifact_dir()
        if not artifact_dir.exists():
            return []

        # 查找所有 .pt 文件
        checkpoints = sorted(artifact_dir.glob("sae_step*.pt"))

        # 添加 latest
        latest = loc.latest_ckpt_path()
        if latest.exists():
            checkpoints.insert(0, latest)

        return checkpoints


# ============================================================================
# 工厂函数
# ============================================================================

def create_checkpoint_manager(config: TrainingConfig) -> CheckpointManager:
    """创建 Checkpoint 管理器"""
    return CheckpointManager(config)


# ============================================================================
# 便捷函数
# ============================================================================

def save_checkpoint(
    run_dir: str,
    layer_idx: int,
    step: int,
    engine: SAEEngine,
    ema: EMAManager,
    optimizer: Adam,
    scheduler: LambdaLR,
    config: TrainingConfig,
) -> Path:
    """便捷函数: 保存 checkpoint"""
    manager = CheckpointManager(config)
    return manager.save(
        run_dir=run_dir,
        layer_idx=layer_idx,
        step=step,
        engine=engine,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def load_checkpoint(
    checkpoint_path: str,
    engine: SAEEngine,
    ema: Optional[EMAManager] = None,
    optimizer: Optional[Adam] = None,
    scheduler: Optional[LambdaLR] = None,
    config: Optional[TrainingConfig] = None,
    device: str = "cpu",
) -> int:
    """
    便捷函数: 加载 checkpoint

    返回:
        step: 训练步数
    """
    if config is None:
        config = TrainingConfig()

    manager = CheckpointManager(config)
    data = manager.load(checkpoint_path, device=device)
    manager.restore(data, engine, ema, optimizer, scheduler)

    return data.step
