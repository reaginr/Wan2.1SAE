"""
训练可视化与统计模块

功能:
1. Loss 变化曲线 (训练/验证)
2. 稀疏度趋势统计
3. 死神经元检测与统计
4. MSE 验证曲线
5. 学习率变化曲线
6. 激活分布统计

作者：Claude
日期：2026-05-17
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ============================================================================
# 统计数据结构
# ============================================================================

@dataclass
class TrainingMetrics:
    """单步训练指标"""
    step: int
    loss: float
    mse: float
    lr: float
    sparsity: float  # 实际稀疏度 (非零比例)
    dead_neuron_count: int
    dead_neuron_ratio: float
    timestamp: float = 0.0


@dataclass
class ValidationMetrics:
    """验证指标"""
    step: int
    mse: float
    sparsity: float
    dead_neuron_count: int
    dead_neuron_ratio: float
    reconstruction_error: float


@dataclass
class LayerStatistics:
    """层统计信息"""
    layer_idx: int
    total_steps: int = 0

    # 训练指标历史
    train_losses: List[float] = field(default_factory=list)
    train_mses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    sparsities: List[float] = field(default_factory=list)
    dead_neuron_counts: List[int] = field(default_factory=list)
    dead_neuron_ratios: List[float] = field(default_factory=list)

    # 验证指标历史
    val_mses: List[float] = field(default_factory=list)
    val_steps: List[int] = field(default_factory=list)

    # 时间戳
    timestamps: List[float] = field(default_factory=list)

    # 最佳值
    best_mse: float = float('inf')
    best_step: int = 0

    # 死神经元追踪 (滑动窗口)
    neuron_activation_history: Optional[torch.Tensor] = None  # [window_size, d_hidden]
    window_size: int = 100

    def update_train(self, metrics: TrainingMetrics):
        """更新训练指标"""
        self.total_steps = metrics.step
        self.train_losses.append(metrics.loss)
        self.train_mses.append(metrics.mse)
        self.learning_rates.append(metrics.lr)
        self.sparsities.append(metrics.sparsity)
        self.dead_neuron_counts.append(metrics.dead_neuron_count)
        self.dead_neuron_ratios.append(metrics.dead_neuron_ratio)
        self.timestamps.append(metrics.timestamp)

        if metrics.mse < self.best_mse:
            self.best_mse = metrics.mse
            self.best_step = metrics.step

    def update_val(self, metrics: ValidationMetrics):
        """更新验证指标"""
        self.val_mses.append(metrics.mse)
        self.val_steps.append(metrics.step)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "layer_idx": self.layer_idx,
            "total_steps": self.total_steps,
            "best_mse": self.best_mse,
            "best_step": self.best_step,
            "final_mse": self.train_mses[-1] if self.train_mses else 0,
            "final_sparsity": self.sparsities[-1] if self.sparsities else 0,
            "final_dead_ratio": self.dead_neuron_ratios[-1] if self.dead_neuron_ratios else 0,
            "mean_sparsity": float(np.mean(self.sparsities)) if self.sparsities else 0,
            "mean_dead_ratio": float(np.mean(self.dead_neuron_ratios)) if self.dead_neuron_ratios else 0,
        }


# ============================================================================
# 死神经元检测器
# ============================================================================

class DeadNeuronTracker:
    """
    死神经元追踪器

    统计方法:
    1. 滑动窗口检测：维护最近 N 步的神经元激活历史
    2. 死神经元定义：在窗口内从未被激活 (激活值 = 0) 的神经元
    3. 激活阈值：激活值 > threshold 视为激活

    学术意义:
    - 死神经元过多 → SAE 容量浪费
    - 死神经元过少 → 可能过拟合
    - 理想范围：5%-15%
    """

    def __init__(
        self,
        d_hidden: int,
        window_size: int = 100,
        activation_threshold: float = 1e-6,
    ):
        self.d_hidden = d_hidden
        self.window_size = window_size
        self.activation_threshold = activation_threshold

        # 激活历史：记录每个神经元是否被激活
        # 使用布尔矩阵节省内存
        self.activation_history = torch.zeros(window_size, d_hidden, dtype=torch.bool)
        self.current_idx = 0
        self.is_filled = False

        # 累计激活次数 (用于长期统计)
        self.total_activations = torch.zeros(d_hidden, dtype=torch.long)
        self.total_steps = 0

    def update(self, z_sparse: torch.Tensor):
        """
        更新激活历史

        参数:
            z_sparse: [batch_size, d_hidden] 稀疏激活
        """
        # 计算哪些神经元被激活
        activated = (z_sparse.abs() > self.activation_threshold).any(dim=0)  # [d_hidden]

        # 更新滑动窗口
        self.activation_history[self.current_idx] = activated
        self.current_idx = (self.current_idx + 1) % self.window_size
        if self.current_idx == 0:
            self.is_filled = True

        # 更新累计统计
        self.total_activations += activated.long()
        self.total_steps += 1

    def get_dead_count(self) -> Tuple[int, float]:
        """
        获取死神经元数量和比例

        返回:
            (死神经元数量, 死神经元比例)
        """
        # 在窗口内从未被激活的神经元
        if self.is_filled:
            window = self.activation_history
        else:
            window = self.activation_history[:self.current_idx]

        if window.shape[0] == 0:
            return 0, 0.0

        # 死神经元：窗口内所有步都未激活
        never_activated = ~window.any(dim=0)
        dead_count = never_activated.sum().item()
        dead_ratio = dead_count / self.d_hidden

        return dead_count, dead_ratio

    def get_activation_frequency(self) -> torch.Tensor:
        """
        获取每个神经元的激活频率

        返回:
            [d_hidden] 激活频率 (0-1)
        """
        if self.total_steps == 0:
            return torch.zeros(self.d_hidden)

        return self.total_activations.float() / self.total_steps

    def get_statistics(self) -> Dict[str, Any]:
        """获取详细统计"""
        dead_count, dead_ratio = self.get_dead_count()
        freq = self.get_activation_frequency()

        # 激活频率分布
        active_freq = freq[freq > 0]

        return {
            "dead_neuron_count": dead_count,
            "dead_neuron_ratio": dead_ratio,
            "mean_activation_freq": float(freq.mean()),
            "median_activation_freq": float(freq.median()),
            "std_activation_freq": float(freq.std()),
            "n_always_active": int((freq == 1.0).sum()),
            "n_never_active": int((freq == 0.0).sum()),
            "n_sometimes_active": int(((freq > 0) & (freq < 1)).sum()),
        }


# ============================================================================
# 稀疏度计算器
# ============================================================================

class SparsityCalculator:
    """
    稀疏度计算器

    统计方法:
    1. L0 稀疏度：非零元素比例
    2. 实际稀疏度：TopK 之后的非零比例
    3. 有效稀疏度：大于阈值的元素比例

    学术意义:
    - 稀疏度越高 → 特征越解耦
    - 稀疏度过低 → 特征纠缠
    - 理想稀疏度：95%-99% (即 1%-5% 非零)
    """

    def __init__(
        self,
        d_hidden: int,
        top_k: int,
        threshold: float = 1e-6,
    ):
        self.d_hidden = d_hidden
        self.top_k = top_k
        self.threshold = threshold

        # 理论稀疏度 (TopK 导致的稀疏度)
        self.theoretical_sparsity = 1.0 - (top_k / d_hidden)

    def compute(self, z_sparse: torch.Tensor) -> Dict[str, float]:
        """
        计算稀疏度

        参数:
            z_sparse: [batch_size, d_hidden] 稀疏激活

        返回:
            Dict: 各种稀疏度指标
        """
        # L0 稀疏度 (非零比例)
        non_zero = (z_sparse.abs() > self.threshold).float()
        l0_sparsity = 1.0 - non_zero.mean().item()

        # 每个样本的稀疏度
        sample_sparsity = 1.0 - non_zero.sum(dim=1) / self.d_hidden
        mean_sample_sparsity = sample_sparsity.mean().item()
        std_sample_sparsity = sample_sparsity.std().item()

        # 激活值统计
        active_values = z_sparse[z_sparse.abs() > self.threshold]

        if len(active_values) > 0:
            mean_activation = float(active_values.mean())
            std_activation = float(active_values.std())
            max_activation = float(active_values.abs().max())
        else:
            mean_activation = 0.0
            std_activation = 0.0
            max_activation = 0.0

        return {
            "l0_sparsity": l0_sparsity,
            "mean_sample_sparsity": mean_sample_sparsity,
            "std_sample_sparsity": std_sample_sparsity,
            "theoretical_sparsity": self.theoretical_sparsity,
            "mean_activation": mean_activation,
            "std_activation": std_activation,
            "max_activation": max_activation,
            "n_active_features": int(non_zero.sum()),
        }


# ============================================================================
# 可视化器
# ============================================================================

class TrainingVisualizer:
    """
    训练可视化器

    生成图表:
    1. Loss 曲线 (训练 + 验证)
    2. 稀疏度趋势
    3. 死神经元统计
    4. MSE 变化
    5. 学习率曲线
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 层统计
        self.layer_stats: Dict[int, LayerStatistics] = {}

    def register_layer(self, layer_idx: int, d_hidden: int):
        """注册层"""
        self.layer_stats[layer_idx] = LayerStatistics(layer_idx=layer_idx)

    def log_train_step(
        self,
        layer_idx: int,
        step: int,
        loss: float,
        mse: float,
        lr: float,
        z_sparse: torch.Tensor,
        dead_tracker: DeadNeuronTracker,
    ):
        """记录训练步"""
        # 计算稀疏度
        sparsity = 1.0 - (z_sparse.abs() > 1e-6).float().mean().item()

        # 获取死神经元统计
        dead_count, dead_ratio = dead_tracker.get_dead_count()

        # 创建指标
        metrics = TrainingMetrics(
            step=step,
            loss=loss,
            mse=mse,
            lr=lr,
            sparsity=sparsity,
            dead_neuron_count=dead_count,
            dead_neuron_ratio=dead_ratio,
            timestamp=datetime.now().timestamp(),
        )

        self.layer_stats[layer_idx].update_train(metrics)

    def log_validation(
        self,
        layer_idx: int,
        step: int,
        mse: float,
        z_sparse: torch.Tensor,
        dead_tracker: DeadNeuronTracker,
    ):
        """记录验证结果"""
        sparsity = 1.0 - (z_sparse.abs() > 1e-6).float().mean().item()
        dead_count, dead_ratio = dead_tracker.get_dead_count()

        metrics = ValidationMetrics(
            step=step,
            mse=mse,
            sparsity=sparsity,
            dead_neuron_count=dead_count,
            dead_neuron_ratio=dead_ratio,
            reconstruction_error=mse,
        )

        self.layer_stats[layer_idx].update_val(metrics)

    def generate_plots(self, layer_idx: int):
        """生成图表"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            stats = self.layer_stats.get(layer_idx)
            if not stats or not stats.train_losses:
                logger.warning(f"No data for layer {layer_idx}")
                return

            layer_dir = self.output_dir / f"layer{layer_idx}"
            layer_dir.mkdir(parents=True, exist_ok=True)

            steps = list(range(1, len(stats.train_losses) + 1))

            # ========== 图1: Loss 曲线 ==========
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(steps, stats.train_losses, 'b-', label='Train Loss', alpha=0.7)

            if stats.val_mses:
                ax.plot(stats.val_steps, stats.val_mses, 'r-', label='Val MSE', alpha=0.7)

            ax.set_xlabel('Step')
            ax.set_ylabel('Loss / MSE')
            ax.set_title(f'Layer {layer_idx} - Training Loss Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            plt.tight_layout()
            plt.savefig(layer_dir / 'loss_curve.png', dpi=150)
            plt.close()

            # ========== 图2: 稀疏度趋势 ==========
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(steps, stats.sparsities, 'g-', label='Actual Sparsity')
            ax.axhline(y=0.99, color='r', linestyle='--', label='Target (99%)', alpha=0.5)
            ax.axhline(y=0.95, color='orange', linestyle='--', label='Min (95%)', alpha=0.5)

            ax.set_xlabel('Step')
            ax.set_ylabel('Sparsity')
            ax.set_title(f'Layer {layer_idx} - Sparsity Trend')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0.9, 1.0])

            plt.tight_layout()
            plt.savefig(layer_dir / 'sparsity_trend.png', dpi=150)
            plt.close()

            # ========== 图3: 死神经元统计 ==========
            fig, ax1 = plt.subplots(figsize=(10, 6))

            color = 'tab:red'
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Dead Neuron Count', color=color)
            ax1.plot(steps, stats.dead_neuron_counts, 'r-', label='Dead Count')
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Dead Neuron Ratio (%)', color=color)
            ax2.plot(steps, [r * 100 for r in stats.dead_neuron_ratios], 'b-', label='Dead Ratio')
            ax2.tick_params(axis='y', labelcolor=color)

            # 理想范围
            ax2.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='Ideal min (5%)')
            ax2.axhline(y=15, color='orange', linestyle='--', alpha=0.5, label='Ideal max (15%)')

            ax1.set_title(f'Layer {layer_idx} - Dead Neuron Statistics')
            fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))

            plt.tight_layout()
            plt.savefig(layer_dir / 'dead_neurons.png', dpi=150)
            plt.close()

            # ========== 图4: 学习率曲线 ==========
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(steps, stats.learning_rates, 'purple', label='Learning Rate')

            ax.set_xlabel('Step')
            ax.set_ylabel('Learning Rate')
            ax.set_title(f'Layer {layer_idx} - Learning Rate Schedule')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            plt.tight_layout()
            plt.savefig(layer_dir / 'learning_rate.png', dpi=150)
            plt.close()

            # ========== 图5: 综合仪表板 ==========
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Loss
            axes[0, 0].plot(steps, stats.train_losses, 'b-', alpha=0.7)
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_yscale('log')

            # Sparsity
            axes[0, 1].plot(steps, stats.sparsities, 'g-', alpha=0.7)
            axes[0, 1].axhline(y=0.99, color='r', linestyle='--', alpha=0.5)
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Sparsity')
            axes[0, 1].set_title('Sparsity Trend')
            axes[0, 1].grid(True, alpha=0.3)

            # Dead Neurons
            axes[1, 0].plot(steps, [r * 100 for r in stats.dead_neuron_ratios], 'r-', alpha=0.7)
            axes[1, 0].axhline(y=5, color='green', linestyle='--', alpha=0.5)
            axes[1, 0].axhline(y=15, color='orange', linestyle='--', alpha=0.5)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Dead Ratio (%)')
            axes[1, 0].set_title('Dead Neuron Ratio')
            axes[1, 0].grid(True, alpha=0.3)

            # Learning Rate
            axes[1, 1].plot(steps, stats.learning_rates, 'purple', alpha=0.7)
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')

            fig.suptitle(f'Layer {layer_idx} Training Dashboard', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(layer_dir / 'dashboard.png', dpi=150)
            plt.close()

            logger.info(f"Generated plots for layer {layer_idx} in {layer_dir}")

        except ImportError:
            logger.warning("matplotlib not available, skipping plots")

    def generate_all_plots(self):
        """为所有层生成图表"""
        for layer_idx in self.layer_stats:
            self.generate_plots(layer_idx)

    def save_statistics(self, layer_idx: int):
        """保存统计数据"""
        stats = self.layer_stats.get(layer_idx)
        if not stats:
            return

        layer_dir = self.output_dir / f"layer{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        # 详细历史
        history = {
            "steps": list(range(1, len(stats.train_losses) + 1)),
            "train_losses": stats.train_losses,
            "train_mses": stats.train_mses,
            "learning_rates": stats.learning_rates,
            "sparsities": stats.sparsities,
            "dead_neuron_counts": stats.dead_neuron_counts,
            "dead_neuron_ratios": stats.dead_neuron_ratios,
            "val_mses": stats.val_mses,
            "val_steps": stats.val_steps,
        }

        with open(layer_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # 汇总统计
        summary = stats.to_dict()
        with open(layer_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved statistics for layer {layer_idx}")

    def save_all_statistics(self):
        """保存所有层的统计数据"""
        for layer_idx in self.layer_stats:
            self.save_statistics(layer_idx)


# ============================================================================
# 便捷函数
# ============================================================================

def create_monitoring_suite(
    d_hidden: int,
    top_k: int,
    output_dir: str,
) -> Tuple[DeadNeuronTracker, SparsityCalculator, TrainingVisualizer]:
    """
    创建监控套件

    返回:
        (死神经元追踪器, 稀疏度计算器, 可视化器)
    """
    dead_tracker = DeadNeuronTracker(d_hidden=d_hidden)
    sparsity_calc = SparsityCalculator(d_hidden=d_hidden, top_k=top_k)
    visualizer = TrainingVisualizer(output_dir)

    return dead_tracker, sparsity_calc, visualizer


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    "TrainingMetrics",
    "ValidationMetrics",
    "LayerStatistics",
    "DeadNeuronTracker",
    "SparsityCalculator",
    "TrainingVisualizer",
    "create_monitoring_suite",
]
