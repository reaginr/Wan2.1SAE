"""
Training Sampling Pipeline - 整合模块

统一训练阶段的采样流程

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from training.sampling_config import (
    TrainingSamplingConfig,
    get_default_training_config,
    MIN_TIMESTEP,
    MAX_TIMESTEP,
)
from training.timestep_sampler import (
    LayerAwareTimestepSampler,
    create_layer_aware_sampler,
)
from training.token_sampler import (
    TrainingTokenSampler,
    create_training_token_sampler,
)
from training.sampling_monitor import (
    SamplingStatisticsMonitor,
    create_monitor,
)


@dataclass
class SamplingResult:
    """采样结果"""
    activations: torch.Tensor
    timesteps: torch.Tensor
    layer_idx: int
    metadata: Dict[str, Any]


class TrainingSamplingPipeline:
    """
    训练阶段采样流水线

    整合：
    - Layer-aware timestep sampling
    - Token sampling (RMSNorm + spatial)
    - Statistics monitoring

    使用方法:
        pipeline = TrainingSamplingPipeline(config)

        # 训练循环中
        for batch in dataloader:
            result = pipeline.sample(
                activations=activations,
                layer_idx=14,
                grid_size=(11, 30, 52),
            )

            # 使用 result.activations 训练 SAE
    """

    def __init__(
        self,
        config: Optional[TrainingSamplingConfig] = None,
        device: str = "cuda",
    ):
        self.config = config or get_default_training_config()
        self.device = device

        # 初始化组件
        self.timestep_sampler = create_layer_aware_sampler(
            device=device,
            seed=self.config.seed,
        )
        self.token_sampler = TrainingTokenSampler(self.config)
        self.monitor = create_monitor(
            min_timestep=MIN_TIMESTEP,
            max_timestep=MAX_TIMESTEP,
        )

    def sample(
        self,
        activations: torch.Tensor,
        layer_idx: int,
        grid_size: Optional[Tuple[int, int, int]] = None,
        batch_size: Optional[int] = None,
        use_low_t_penalty: bool = True,
    ) -> SamplingResult:
        """
        执行采样

        参数:
            activations: [B, L, C] 原始激活
            layer_idx: 层索引
            grid_size: (F, H, W) 网格尺寸
            batch_size: batch 大小 (用于 timestep 采样)
            use_low_t_penalty: 是否使用低 timestep 惩罚

        返回:
            SamplingResult: 包含采样后的激活和元数据
        """
        if batch_size is None:
            if activations.dim() == 3:
                batch_size = activations.shape[0]
            else:
                batch_size = 1

        # Step 1: 采样 timestep (Layer-aware Truncated Gaussian)
        timesteps = self.timestep_sampler.sample_timestep(
            layer_id=layer_idx,
            batch_size=batch_size,
            use_low_t_penalty=use_low_t_penalty,
        )

        # 取平均 timestep (用于 token 采样)
        avg_timestep = int(timesteps.float().mean().item())

        # Step 2: Token 采样 (RMSNorm + spatial)
        sampled_activations, token_metadata = self.token_sampler.sample(
            activations=activations,
            timestep=avg_timestep,
            layer_idx=layer_idx,
            grid_size=grid_size,
        )

        # Step 3: 记录到监控器
        self.monitor.record_sample(
            timestep=avg_timestep,
            layer_idx=layer_idx,
            n_tokens=len(sampled_activations),
        )

        # 合并元数据
        metadata = {
            **token_metadata,
            "timesteps_sampled": timesteps.tolist(),
            "avg_timestep": avg_timestep,
        }

        return SamplingResult(
            activations=sampled_activations,
            timesteps=timesteps,
            layer_idx=layer_idx,
            metadata=metadata,
        )

    def sample_all_layers(
        self,
        activations_by_layer: Dict[int, torch.Tensor],
        grid_size: Optional[Tuple[int, int, int]] = None,
        batch_size: int = 1,
    ) -> Dict[int, SamplingResult]:
        """
        为所有层执行采样

        参数:
            activations_by_layer: {layer_idx: [B, L, C]}
            grid_size: (F, H, W)
            batch_size: batch 大小

        返回:
            {layer_idx: SamplingResult}
        """
        results = {}

        for layer_idx, activations in activations_by_layer.items():
            results[layer_idx] = self.sample(
                activations=activations,
                layer_idx=layer_idx,
                grid_size=grid_size,
                batch_size=batch_size,
            )

        return results

    def get_monitor(self) -> SamplingStatisticsMonitor:
        """获取监控器"""
        return self.monitor

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.monitor.get_statistics()

    def print_report(self):
        """打印统计报告"""
        self.monitor.print_report()

    def reset(self):
        """重置所有状态"""
        self.timestep_sampler.reset_history()
        self.token_sampler.reset_generator()
        self.monitor.reset()


# ============================================================================
# 便捷函数
# ============================================================================

def create_training_pipeline(
    max_tokens_per_batch: int = 4096,
    seed: int = 42,
    device: str = "cuda",
) -> TrainingSamplingPipeline:
    """创建训练采样流水线"""
    config = TrainingSamplingConfig(
        max_tokens_per_batch=max_tokens_per_batch,
        seed=seed,
    )
    return TrainingSamplingPipeline(config, device)


# ============================================================================
# Pipeline Diagram
# ============================================================================

PIPELINE_DIAGRAM = """
+---------------------------------------------------------------------+
|                    Training Sampling Pipeline                        |
+---------------------------------------------------------------------+
|                                                                     |
|  Input: Raw Activations [B, L, C=1536]                             |
|         Layer Index                                                  |
|         Grid Size (F, H, W)                                         |
|                                                                     |
+---------------------------------------------------------------------+
|                                                                     |
|  +---------------------------------------------------------------+  |
|  | Step 1: Layer-Aware Timestep Sampling                        |  |
|  |                                                               |  |
|  |   Layer 14 -> TruncatedGaussian(mu=650, sigma=120) [400,800] |  |
|  |   Layer 19 -> TruncatedGaussian(mu=550, sigma=110) [300,700] |  |
|  |   Layer 24 -> TruncatedGaussian(mu=420, sigma=100) [200,600] |  |
|  |   Layer 29 -> TruncatedGaussian(mu=300, sigma=80)  [150,450] |  |
|  |                                                               |  |
|  |   [X] NO uniform sampling                                     |  |
|  |   [X] NO t < 150 (collapse region)                            |  |
|  |   [X] NO t > 800 (noise region)                               |  |
|  +---------------------------------------------------------------+  |
|                           |                                         |
|                           v                                         |
|  +---------------------------------------------------------------+  |
|  | Step 2: Token Sampling (Distribution Preserving)              |  |
|  |                                                               |  |
|  |   [OK] RMSNorm (mandatory, consistent with initialization)    |  |
|  |   [OK] Mild Spatial Stride (Layer 14/19: 2, Layer 24/29: 1)   |  |
|  |   [OK] Soft Norm Bias (weak, strength=0.15)                   |  |
|  |                                                               |  |
|  |   [X] NO decorrelation filter                                  |  |
|  |   [X] NO oversampling                                          |  |
|  |   [X] NO similarity rejection                                  |  |
|  |   [X] NO hard norm bucket                                      |  |
|  +---------------------------------------------------------------+  |
|                           |                                         |
|                           v                                         |
|  +---------------------------------------------------------------+  |
|  | Step 3: Statistics Monitoring                                 |  |
|  |                                                               |  |
|  |   - Timestep histogram                                         |  |
|  |   - Per-layer entropy                                          |  |
|  |   - Distribution drift detection                               |  |
|  |   - Coherence correlation                                      |  |
|  +---------------------------------------------------------------+  |
|                                                                     |
+---------------------------------------------------------------------+
|  Output: SamplingResult                                            |
|          - activations: [M, C]                                     |
|          - timesteps: [B]                                          |
|          - metadata: Dict                                          |
+---------------------------------------------------------------------+
"""
