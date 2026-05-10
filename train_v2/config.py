"""
SAE Training Configuration

训练阶段完整配置，严格按照 TODO_list_v3.md 规范

核心约束：
- 8x expansion (d_hidden=12288)
- NO weight decay
- Adam with betas=(0.95, 0.999)
- Warmup + Cosine LR schedule
- Layer-wise sequential training

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# 强约束常量
# ============================================================================

# 模型架构 (固定，不允许修改)
D_MODEL = 1536          # DiT 隐藏维度
D_HIDDEN = 12288        # 8x expansion
TOP_K = 128             # topk 稀疏度

# Hook 层 (必须顺序训练)
HOOK_LAYERS = [14, 19, 24, 29]

# Timestep 有效区间
MIN_TIMESTEP = 150
MAX_TIMESTEP = 800


# ============================================================================
# 训练配置
# ============================================================================

@dataclass
class TrainingConfig:
    """
    SAE 训练配置

    严格按照 TODO_list_v3.md 第三阶段规范

    关键约束:
    - weight_decay 必须为 0.0 (NO weight decay)
    - d_hidden 必须为 12288 (8x expansion)
    - 必须顺序训练 14->19->24->29
    """

    # ===== 架构配置 (固定) =====
    d_model: int = D_MODEL
    d_hidden: int = D_HIDDEN       # 8x expansion, ENFORCED
    top_k: int = TOP_K             # ~1% sparsity

    # ===== 优化器配置 =====
    lr: float = 6e-5               # 8x expansion LR
    betas: Tuple[float, float] = (0.95, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0      # ENFORCED: NO weight decay

    # ===== 学习率调度 =====
    warmup_steps: int = 4000       # 线性 warmup
    total_steps: int = 8000        # 总步数
    min_lr: float = 1e-5           # 最小学习率

    # ===== 梯度累积 =====
    batch_size: int = 4            # 每步 batch size
    accum_steps: int = 8           # 累积步数
    effective_batch_size: int = 32 # batch_size * accum_steps
    grad_clip: float = 0.3         # 梯度裁剪

    # ===== EMA =====
    ema_decay: float = 0.999       # EMA 衰减率

    # ===== 验证与保存 =====
    val_interval: int = 160        # 验证间隔 (20 update cycles)
    checkpoint_interval: int = 400 # 保存间隔

    # ===== 早停 =====
    early_stop_patience: int = 5   # 连续验证次数无改进则停止
    early_stop_min_delta: float = 0.001  # MSE 最小改进阈值
    dead_neuron_stop_threshold: float = 0.20  # 死神经元比率上限

    # ===== 收敛标准 =====
    convergence_mse: float = 0.1
    convergence_dead_ratio: float = 0.10
    convergence_fm_increase: float = 0.05

    # ===== Hook 配置 =====
    hook_mode: str = "block_out"
    hook_layers: List[int] = field(default_factory=lambda: HOOK_LAYERS.copy())

    # ===== 路径配置 =====
    checkpoint_dir: str = "./Wan2.1-T2V-1.3B"
    prompt_dir: str = "./nsfw_prompts"
    run_dir: str = "sae_runs/train_v2_default"

    # ===== 采样配置 =====
    max_tokens_per_batch: int = 4096
    seed: int = 42

    # ===== 设备 =====
    device: str = "cuda"

    # ===== Loss 权重 =====
    lambda_aux: float = 0.1        # AuxK loss 权重
    lambda_orth: float = 1e-5      # 正交损失权重

    def __post_init__(self):
        """后处理：计算派生值"""
        self.effective_batch_size = self.batch_size * self.accum_steps

    def validate(self) -> None:
        """
        验证配置符合 TODO_list_v3.md 规范

        Raises:
            AssertionError: 如果违反约束
        """
        # 强约束检查
        assert self.weight_decay == 0.0, \
            f"weight_decay 必须为 0.0 (NO weight decay per TODO), got {self.weight_decay}"
        assert self.d_hidden == D_HIDDEN, \
            f"d_hidden 必须为 {D_HIDDEN} (8x expansion per TODO), got {self.d_hidden}"
        assert self.d_model == D_MODEL, \
            f"d_model 必须为 {D_MODEL}, got {self.d_model}"

        # 学习率范围检查
        assert 1e-6 <= self.lr <= 1e-3, \
            f"lr 应在 [1e-6, 1e-3] 范围内, got {self.lr}"
        assert self.min_lr < self.lr, \
            f"min_lr ({self.min_lr}) 应小于 lr ({self.lr})"

        # 步数检查
        assert self.warmup_steps < self.total_steps, \
            f"warmup_steps ({self.warmup_steps}) 应小于 total_steps ({self.total_steps})"

        # 梯度累积检查
        assert self.accum_steps >= 1, \
            f"accum_steps 必须大于等于 1, got {self.accum_steps}"

        # Hook 层检查
        for layer in self.hook_layers:
            assert layer in HOOK_LAYERS, \
                f"hook_layers 必须是 {HOOK_LAYERS} 中的层, got {layer}"

        # EMA 检查
        assert 0.9 <= self.ema_decay < 1.0, \
            f"ema_decay 应在 [0.9, 1.0) 范围内, got {self.ema_decay}"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "d_model": self.d_model,
            "d_hidden": self.d_hidden,
            "top_k": self.top_k,
            "lr": self.lr,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
            "batch_size": self.batch_size,
            "accum_steps": self.accum_steps,
            "effective_batch_size": self.effective_batch_size,
            "grad_clip": self.grad_clip,
            "ema_decay": self.ema_decay,
            "val_interval": self.val_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "early_stop_patience": self.early_stop_patience,
            "early_stop_min_delta": self.early_stop_min_delta,
            "dead_neuron_stop_threshold": self.dead_neuron_stop_threshold,
            "convergence_mse": self.convergence_mse,
            "convergence_dead_ratio": self.convergence_dead_ratio,
            "convergence_fm_increase": self.convergence_fm_increase,
            "hook_mode": self.hook_mode,
            "hook_layers": self.hook_layers,
            "checkpoint_dir": self.checkpoint_dir,
            "prompt_dir": self.prompt_dir,
            "run_dir": self.run_dir,
            "max_tokens_per_batch": self.max_tokens_per_batch,
            "seed": self.seed,
            "device": self.device,
            "lambda_aux": self.lambda_aux,
            "lambda_orth": self.lambda_orth,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        """从字典创建"""
        return cls(**d)

    def get_lr_at_step(self, step: int) -> float:
        """
        计算指定步数的学习率

        Warmup + Cosine schedule per TODO 3.1
        """
        if step < self.warmup_steps:
            # Linear warmup
            return self.lr * (step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.lr - self.min_lr) * cosine_factor


# ============================================================================
# 便捷函数
# ============================================================================

def get_default_training_config() -> TrainingConfig:
    """获取默认训练配置"""
    return TrainingConfig()


def get_layer_training_config(layer_idx: int, run_dir: str, **kwargs) -> TrainingConfig:
    """
    获取单层训练配置

    参数:
        layer_idx: 层索引
        run_dir: 实验目录
        **kwargs: 覆盖默认配置

    返回:
        TrainingConfig
    """
    defaults = {
        "hook_layers": [layer_idx],
        "run_dir": run_dir,
    }
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


# ============================================================================
# 验证配置
# ============================================================================

@dataclass
class ValidationConfig:
    """验证阶段配置"""

    # 验证样本数
    n_val_samples: int = 2000

    # 死神经元检测窗口
    dead_neuron_window: int = 2000

    # 指标计算配置
    compute_gini: bool = True
    compute_mutual_coherence: bool = True
    compute_fm_loss: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_val_samples": self.n_val_samples,
            "dead_neuron_window": self.dead_neuron_window,
            "compute_gini": self.compute_gini,
            "compute_mutual_coherence": self.compute_mutual_coherence,
            "compute_fm_loss": self.compute_fm_loss,
        }


# ============================================================================
# 配置验证工具
# ============================================================================

def validate_layer_order(layers: List[int]) -> bool:
    """
    验证层顺序符合规范

    必须按 14 -> 19 -> 24 -> 29 顺序训练
    """
    expected_order = [14, 19, 24, 29]
    for layer in layers:
        if layer not in expected_order:
            return False
    # 检查顺序
    indices = [expected_order.index(layer) for layer in layers]
    return indices == sorted(indices)


def assert_single_layer(config: TrainingConfig) -> None:
    """
    断言配置只包含单层

    用于强制执行 layer-wise 训练
    """
    assert len(config.hook_layers) == 1, \
        f"必须单层训练 (per TODO), got layers={config.hook_layers}"
