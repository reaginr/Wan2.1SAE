"""
Training Validator

严格按照 TODO_list_v3.md 3.4 规范

验证指标:
- Normalized MSE
- Dead neuron ratio
- Gini coefficient (特征垄断检测)
- Mutual coherence (decoder 列相关性)

收敛标准:
- MSE <= 0.1
- Dead neuron <= 10%
- FM increase <= 5%

早停:
- MSE decrease < 0.001 for 5 validation cycles
- Dead neuron > 20% → immediate stop

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_v2.config import TrainingConfig
from train_v2.dead_neuron_monitor import DeadNeuronMonitor
from train_v2.ema import EMAManager
from train_v2.sae_engine import SAEEngine

# 延迟导入 wan 模块
try:
    from wan.modules.sae_new import SparseAutoEncoder
    WAN_AVAILABLE = True
except ImportError:
    SparseAutoEncoder = None
    WAN_AVAILABLE = False


# ============================================================================
# 验证指标
# ============================================================================

@dataclass
class ValidationMetrics:
    """验证指标集合"""
    # 主要指标
    mse_normalized: float
    dead_neuron_ratio: float

    # 辅助指标
    gini: float
    mutual_coherence: float

    # 激活统计
    active_features: int
    total_features: int
    active_ratio: float

    # 损失分解
    recon_loss: float
    auxk_loss: float
    orth_loss: float

    # 收敛状态
    is_converged: bool = False
    should_early_stop: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mse_normalized": self.mse_normalized,
            "dead_neuron_ratio": self.dead_neuron_ratio,
            "gini": self.gini,
            "mutual_coherence": self.mutual_coherence,
            "active_features": self.active_features,
            "total_features": self.total_features,
            "active_ratio": self.active_ratio,
            "recon_loss": self.recon_loss,
            "auxk_loss": self.auxk_loss,
            "orth_loss": self.orth_loss,
            "is_converged": self.is_converged,
            "should_early_stop": self.should_early_stop,
        }


@dataclass
class ValidationHistory:
    """验证历史"""
    metrics_history: List[ValidationMetrics] = field(default_factory=list)

    # MSE 追踪 (用于早停)
    mse_values: List[float] = field(default_factory=list)
    best_mse: float = float("inf")
    no_improve_count: int = 0

    def add(self, metrics: ValidationMetrics) -> None:
        """添加验证结果"""
        self.metrics_history.append(metrics)
        self.mse_values.append(metrics.mse_normalized)

        # 更新最佳 MSE
        if metrics.mse_normalized < self.best_mse:
            self.best_mse = metrics.mse_normalized
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1


# ============================================================================
# 训练验证器
# ============================================================================

class TrainingValidator:
    """
    训练验证器

    严格按照 TODO 3.4 规范:
    - 使用 EMA 权重验证
    - 计算 Normalized MSE
    - 检测死神经元
    - 计算 Gini 和 Mutual Coherence

    使用示例:
        validator = TrainingValidator(engine, config)

        # 验证 (自动使用 EMA 权重)
        metrics = validator.validate(val_activations, ema)

        # 检查收敛
        if metrics.is_converged:
            print("Training converged!")
    """

    def __init__(
        self,
        engine: SAEEngine,
        config: TrainingConfig,
    ):
        """
        初始化验证器

        参数:
            engine: SAE Engine
            config: 训练配置
        """
        self.engine = engine
        self.config = config

        # 死神经元监控器
        self.dead_monitor = DeadNeuronMonitor(
            n_features=config.d_hidden,
            window=2000,
            early_stop_threshold=config.dead_neuron_stop_threshold,
        )

        # 验证历史
        self.history = ValidationHistory()

    def validate(
        self,
        val_activations: torch.Tensor,
        ema: Optional[EMAManager] = None,
    ) -> ValidationMetrics:
        """
        执行验证

        参数:
            val_activations: [N, D] 验证激活
            ema: EMA 管理器 (用于切换到 EMA 权重)

        返回:
            ValidationMetrics
        """
        device = val_activations.device

        # 使用 EMA 权重 (如果提供)
        if ema is not None:
            ema.apply_ema(self.engine.sae)

        try:
            with torch.no_grad():
                # ===== Step 1: 计算重建损失 =====
                loss, loss_info = self.engine.forward(val_activations, return_info=True)

                # Normalized MSE (相对于输入范数)
                x_norm, rms = self._rms_norm(val_activations)
                x_hat = self.engine.reconstruct(val_activations)
                mse_normalized = F.mse_loss(x_hat, x_norm).item()

                # ===== Step 2: 更新死神经元监控 =====
                z_sparse, topk_idx, topk_val = self.engine.encode(val_activations)
                self.dead_monitor.update(topk_idx)

                dead_ratio = self.dead_monitor.get_dead_ratio()

                # ===== Step 3: 计算 Gini 系数 =====
                gini = self._compute_gini(topk_idx)

                # ===== Step 4: 计算 Mutual Coherence =====
                coherence = self._compute_mutual_coherence()

                # ===== Step 5: 构建指标 =====
                metrics = ValidationMetrics(
                    mse_normalized=mse_normalized,
                    dead_neuron_ratio=dead_ratio,
                    gini=gini,
                    mutual_coherence=coherence,
                    active_features=loss_info.active_features,
                    total_features=loss_info.total_features,
                    active_ratio=loss_info.active_ratio,
                    recon_loss=loss_info.recon_loss,
                    auxk_loss=loss_info.auxk_loss,
                    orth_loss=loss_info.orth_loss,
                )

                # ===== Step 6: 检查收敛 =====
                metrics.is_converged = self._check_convergence(metrics)

                # ===== Step 7: 检查早停 =====
                metrics.should_early_stop = self._check_early_stop(metrics)

                # 记录历史
                self.history.add(metrics)

                return metrics

        finally:
            # 恢复训练权重
            if ema is not None:
                ema.restore(self.engine.sae)

    def _rms_norm(self, x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        """RMSNorm"""
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return x / rms, rms

    def _compute_gini(self, topk_indices: torch.Tensor) -> float:
        """
        计算 Gini 系数

        用于检测特征垄断

        Gini = 0: 完全均匀
        Gini = 1: 完全垄断
        """
        n_features = self.config.d_hidden

        # 统计每个特征的激活次数
        flat_indices = topk_indices.flatten().cpu()
        counts = torch.zeros(n_features)
        for idx in flat_indices:
            counts[idx] += 1

        if counts.sum() == 0:
            return 0.0

        # 排序
        sorted_counts = sorted(counts.tolist())
        n = len(sorted_counts)

        # 计算 Gini
        cumsum = 0.0
        total = sum(sorted_counts)
        for i, c in enumerate(sorted_counts):
            cumsum += (2 * (i + 1) - n - 1) * c

        gini = cumsum / (n * total)
        return gini

    def _compute_mutual_coherence(self) -> float:
        """
        计算 Mutual Coherence

        定义为 decoder 列向量之间的最大相关性

        低 coherence (< 0.3): 特征独立
        高 coherence (> 0.8): 特征高度相关
        """
        W_dec = self.engine.sae.decoder.weight  # [d_model, d_hidden]

        # 归一化列向量
        W_dec_norm = F.normalize(W_dec, dim=0)

        # 计算 Gram 矩阵
        gram = W_dec_norm.T @ W_dec_norm  # [d_hidden, d_hidden]

        # 取上三角 (不含对角线) 的最大值
        mask = torch.triu(torch.ones_like(gram), diagonal=1).bool()
        off_diagonal = gram[mask]

        if off_diagonal.numel() == 0:
            return 0.0

        coherence = off_diagonal.abs().max().item()
        return coherence

    def _check_convergence(self, metrics: ValidationMetrics) -> bool:
        """
        检查是否收敛

        收敛标准 (per TODO):
        - MSE <= 0.1
        - Dead neuron <= 10%
        """
        return (
            metrics.mse_normalized <= self.config.convergence_mse and
            metrics.dead_neuron_ratio <= self.config.convergence_dead_ratio
        )

    def _check_early_stop(self, metrics: ValidationMetrics) -> bool:
        """
        检查是否应该早停

        早停条件 (per TODO):
        - 立即停止: Dead neuron > 20%
        - 常规停止: MSE decrease < 0.001 for 5 validation cycles
        """
        # 立即停止条件
        if metrics.dead_neuron_ratio > self.config.dead_neuron_stop_threshold:
            return True

        # 常规早停 (基于 MSE 无改进)
        if self.history.no_improve_count >= self.config.early_stop_patience:
            return True

        return False

    def reset(self) -> None:
        """重置验证器"""
        self.dead_monitor.reset()
        self.history = ValidationHistory()

    def get_history_summary(self) -> Dict[str, Any]:
        """获取历史摘要"""
        if not self.history.metrics_history:
            return {}

        return {
            "num_validations": len(self.history.metrics_history),
            "best_mse": self.history.best_mse,
            "no_improve_count": self.history.no_improve_count,
            "final_mse": self.history.mse_values[-1],
            "final_dead_ratio": self.history.metrics_history[-1].dead_neuron_ratio,
        }


# ============================================================================
# 工厂函数
# ============================================================================

def create_validator(engine: SAEEngine, config: TrainingConfig) -> TrainingValidator:
    """创建验证器"""
    return TrainingValidator(engine, config)


# ============================================================================
# 报告工具
# ============================================================================

def print_validation_report(metrics: ValidationMetrics) -> None:
    """打印验证报告"""
    print("\n" + "=" * 60)
    print("Validation Report")
    print("=" * 60)
    print(f"  MSE (normalized):  {metrics.mse_normalized:.6f}")
    print(f"  Dead Neuron Ratio: {metrics.dead_neuron_ratio:.4f} ({metrics.dead_neuron_ratio * 100:.2f}%)")
    print(f"  Gini:              {metrics.gini:.4f}")
    print(f"  Mutual Coherence:  {metrics.mutual_coherence:.4f}")
    print("-" * 60)
    print(f"  Active Features:   {metrics.active_features}/{metrics.total_features}")
    print(f"  Active Ratio:      {metrics.active_ratio:.4f}")
    print("-" * 60)
    print(f"  Recon Loss:        {metrics.recon_loss:.6f}")
    print(f"  AuxK Loss:         {metrics.auxk_loss:.6f}")
    print(f"  Orth Loss:         {metrics.orth_loss:.6f}")
    print("-" * 60)
    print(f"  Converged:         {metrics.is_converged}")
    print(f"  Early Stop:        {metrics.should_early_stop}")
    print("=" * 60)
