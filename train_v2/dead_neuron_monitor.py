"""
Dead Neuron Monitor

严格按照 TODO_list_v3.md 3.4 规范

死神经元定义: 在 window (默认 2000 样本) 内无激活的神经元

关键功能:
1. 滑动窗口追踪神经元激活历史
2. 计算死神经元比率
3. 返回死神经元索引 (用于 AuxK 恢复)
4. 早停检测 (dead ratio > 20%)

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import torch

from train_v2.config import TrainingConfig


# ============================================================================
# 死神经元统计
# ============================================================================

@dataclass
class DeadNeuronStats:
    """死神经元统计"""
    n_features: int
    window: int

    # 当前状态
    dead_count: int = 0
    dead_ratio: float = 0.0

    # 历史追踪
    dead_ratio_history: List[float] = field(default_factory=list)

    # 触发早停
    should_stop: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_features": self.n_features,
            "window": self.window,
            "dead_count": self.dead_count,
            "dead_ratio": self.dead_ratio,
            "should_stop": self.should_stop,
        }


# ============================================================================
# 死神经元监控器
# ============================================================================

class DeadNeuronMonitor:
    """
    死神经元监控器

    严格按照 TODO 3.4 规范:
    - 死神经元 = window (默认 2000 样本) 内无激活
    - 触发早停条件: dead ratio > 20%

    使用示例:
        monitor = DeadNeuronMonitor(n_features=12288, window=2000)

        # 训练循环中
        for batch in train_loader:
            z_sparse, topk_idx, topk_val = sae.encode(x)
            monitor.update(topk_idx)

            # 检查死神经元比率
            dead_ratio = monitor.get_dead_ratio()
            if dead_ratio > 0.20:
                print("Warning: High dead neuron ratio!")
    """

    def __init__(
        self,
        n_features: int,
        window: int = 2000,
        early_stop_threshold: float = 0.20,
    ):
        """
        初始化死神经元监控器

        参数:
            n_features: SAE 隐藏维度 (特征数)
            window: 滑动窗口大小 (样本数)
            early_stop_threshold: 早停阈值 (死神经元比率)
        """
        self.n_features = n_features
        self.window = window
        self.early_stop_threshold = early_stop_threshold

        # 激活历史 (循环缓冲)
        # firing_history[i, j] = 特征 i 在窗口位置 j 是否激活
        self.firing_history: Optional[torch.Tensor] = None

        # 当前窗口位置
        self.current_idx: int = 0

        # 已处理的样本数
        self.total_samples: int = 0

        # 死神经元集合 (缓存)
        self._dead_set: Set[int] = set()

    def _init_history(self, device: torch.device) -> None:
        """初始化激活历史"""
        if self.firing_history is None:
            self.firing_history = torch.zeros(
                self.n_features,
                self.window,
                dtype=torch.bool,
                device=device,
            )

    def update(self, topk_indices: torch.Tensor) -> None:
        """
        更新激活历史

        参数:
            topk_indices: [N, k] TopK 索引
        """
        device = topk_indices.device
        self._init_history(device)

        # 扁平化并获取唯一激活特征
        flat_indices = topk_indices.flatten()
        unique_features = torch.unique(flat_indices)

        # 清除当前位置的激活记录
        self.firing_history[:, self.current_idx] = False

        # 记录当前 batch 的激活特征
        self.firing_history[unique_features, self.current_idx] = True

        # 更新统计
        batch_size = topk_indices.shape[0]
        self.total_samples += batch_size

        # 移动窗口位置
        self.current_idx = (self.current_idx + 1) % self.window

        # 清除死神经元缓存
        self._dead_set.clear()

    def get_dead_ratio(self) -> float:
        """
        获取死神经元比率

        返回:
            dead_ratio: 死神经元比率 [0, 1]
        """
        if self.firing_history is None:
            return 0.0

        # 检查在窗口内从未激活的特征
        ever_fired = self.firing_history.any(dim=1)
        dead_count = (~ever_fired).sum().item()

        return dead_count / self.n_features

    def get_dead_count(self) -> int:
        """获取死神经元数量"""
        if self.firing_history is None:
            return 0

        ever_fired = self.firing_history.any(dim=1)
        return (~ever_fired).sum().item()

    def get_dead_indices(self) -> torch.Tensor:
        """
        获取死神经元索引

        返回:
            dead_indices: [M] 死神经元索引张量
        """
        if self.firing_history is None:
            return torch.tensor([], dtype=torch.long)

        ever_fired = self.firing_history.any(dim=1)
        dead_indices = torch.where(~ever_fired)[0]

        return dead_indices

    def get_alive_indices(self) -> torch.Tensor:
        """
        获取活跃神经元索引

        返回:
            alive_indices: [M] 活跃神经元索引张量
        """
        if self.firing_history is None:
            return torch.arange(self.n_features)

        ever_fired = self.firing_history.any(dim=1)
        alive_indices = torch.where(ever_fired)[0]

        return alive_indices

    def check_early_stop(self) -> bool:
        """
        检查是否应该早停

        返回:
            should_stop: True 如果死神经元比率超过阈值
        """
        dead_ratio = self.get_dead_ratio()
        return dead_ratio > self.early_stop_threshold

    def get_stats(self) -> DeadNeuronStats:
        """获取统计信息"""
        dead_ratio = self.get_dead_ratio()
        dead_count = self.get_dead_count()
        should_stop = self.check_early_stop()

        return DeadNeuronStats(
            n_features=self.n_features,
            window=self.window,
            dead_count=dead_count,
            dead_ratio=dead_ratio,
            should_stop=should_stop,
        )

    def reset(self) -> None:
        """重置监控器"""
        if self.firing_history is not None:
            self.firing_history.zero_()
        self.current_idx = 0
        self.total_samples = 0
        self._dead_set.clear()

    def get_activation_frequency(self) -> torch.Tensor:
        """
        获取每个特征的激活频率

        返回:
            freq: [n_features] 激活频率
        """
        if self.firing_history is None:
            return torch.ones(self.n_features)

        # 计算每个特征在窗口内激活的次数
        activation_count = self.firing_history.sum(dim=1).float()

        # 有效窗口大小 (处理初始阶段)
        valid_window = min(self.total_samples, self.window)

        return activation_count / valid_window


# ============================================================================
# 工厂函数
# ============================================================================

def create_dead_neuron_monitor(config: TrainingConfig) -> DeadNeuronMonitor:
    """从配置创建死神经元监控器"""
    return DeadNeuronMonitor(
        n_features=config.d_hidden,
        window=2000,  # per TODO
        early_stop_threshold=config.dead_neuron_stop_threshold,
    )


# ============================================================================
# 特征使用统计
# ============================================================================

class FeatureUsageTracker:
    """
    特征使用追踪器

    用于分析特征使用分布、Gini 系数等
    """

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.activation_counts = torch.zeros(n_features)

    def update(self, topk_indices: torch.Tensor) -> None:
        """更新激活计数"""
        flat_indices = topk_indices.flatten().cpu()
        for idx in flat_indices:
            self.activation_counts[idx] += 1

    def get_gini_coefficient(self) -> float:
        """
        计算 Gini 系数

        用于检测特征垄断 (高 Gini 表示少数特征占主导)
        """
        counts = self.activation_counts.numpy()
        if counts.sum() == 0:
            return 0.0

        # 排序
        sorted_counts = sorted(counts)
        n = len(sorted_counts)

        # 计算 Gini
        cumsum = 0.0
        for i, c in enumerate(sorted_counts):
            cumsum += (2 * (i + 1) - n - 1) * c

        gini = cumsum / (n * sum(sorted_counts))
        return gini

    def reset(self) -> None:
        """重置"""
        self.activation_counts.zero_()


# ============================================================================
# 调试工具
# ============================================================================

def print_dead_neuron_report(monitor: DeadNeuronMonitor) -> None:
    """打印死神经元报告"""
    stats = monitor.get_stats()
    freq = monitor.get_activation_frequency()

    print("\n" + "=" * 50)
    print("Dead Neuron Report")
    print("=" * 50)
    print(f"  Total Features:  {stats.n_features}")
    print(f"  Window Size:     {stats.window}")
    print(f"  Dead Count:      {stats.dead_count}")
    print(f"  Dead Ratio:      {stats.dead_ratio:.4f} ({stats.dead_ratio * 100:.2f}%)")
    print(f"  Should Stop:     {stats.should_stop}")
    print("-" * 50)

    # 激活频率分布
    freq_cpu = freq.cpu()
    print(f"  Activation Freq:")
    print(f"    Min:  {freq_cpu.min().item():.6f}")
    print(f"    Max:  {freq_cpu.max().item():.6f}")
    print(f"    Mean: {freq_cpu.mean().item():.6f}")
    print(f"    Std:  {freq_cpu.std().item():.6f}")
    print("=" * 50)
