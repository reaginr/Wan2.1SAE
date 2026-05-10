"""
SAE Training Sampling Configuration

训练阶段采样配置，与初始化阶段完全分离

核心原则：
1. 保持真实 activation distribution
2. Layer-aware timestep distribution
3. Truncated Gaussian sampling (禁止 uniform)
4. 禁止: hard bucket, decorrelation, oversampling

作者：Claude
日期：2026-05-10
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================================
# 强约束常量
# ============================================================================

# Timestep 有效区间（强约束）
MIN_TIMESTEP = 150  # 低于此值 -> representation collapse
MAX_TIMESTEP = 800  # 高于此值 -> 纯噪声，无语义结构

# SAE 配置
D_MODEL = 1536
D_HIDDEN = 12288
HOOK_LAYERS = [14, 19, 24, 29]


# ============================================================================
# Layer-wise Timestep Distribution 配置
# ============================================================================

@dataclass
class LayerTimestepConfig:
    """
    单层的 Timestep 采样配置

    基于 diffusion semantic hierarchy 设计：
    - 浅层 (14): 需要更多结构信息 → 偏向高 timestep
    - 深层 (29): 需要更多语义信息 → 偏向低 timestep
    """
    layer_idx: int
    mu: float           # Gaussian 中心
    sigma: float        # Gaussian 标准差
    min_t: int = MIN_TIMESTEP
    max_t: int = MAX_TIMESTEP

    # 主采样区间 (μ ± 2σ 的实际范围)
    primary_range: Tuple[int, int] = (0, 0)

    # 低 timestep 衰减因子 (asymmetric penalty)
    low_t_decay_alpha: float = -0.15  # 负偏置，降低 low-t 权重

    def __post_init__(self):
        if self.primary_range == (0, 0):
            self.primary_range = (
                max(self.min_t, int(self.mu - 2 * self.sigma)),
                min(self.max_t, int(self.mu + 2 * self.sigma))
            )

    def validate(self):
        """验证配置有效性"""
        assert self.min_t >= MIN_TIMESTEP, f"min_t={self.min_t} < {MIN_TIMESTEP}"
        assert self.max_t <= MAX_TIMESTEP, f"max_t={self.max_t} > {MAX_TIMESTEP}"
        assert self.mu >= self.min_t and self.mu <= self.max_t


# 预定义的 Layer-wise 配置
# 设计原则：
# - Layer 14 (中层): 结构特征，需要较高噪声级别 → μ=650
# - Layer 19 (中深): 语义特征，中等噪声级别 → μ=550
# - Layer 24 (深层): 高级语义，较低噪声级别 → μ=420
# - Layer 29 (最深): 最终表征，最低噪声级别 → μ=300
LAYER_TIMESTEP_CONFIGS: Dict[int, LayerTimestepConfig] = {
    14: LayerTimestepConfig(
        layer_idx=14,
        mu=650,
        sigma=120,
        primary_range=(400, 800),
        low_t_decay_alpha=-0.15,
    ),
    19: LayerTimestepConfig(
        layer_idx=19,
        mu=550,
        sigma=110,
        primary_range=(300, 700),
        low_t_decay_alpha=-0.15,
    ),
    24: LayerTimestepConfig(
        layer_idx=24,
        mu=420,
        sigma=100,
        primary_range=(200, 600),
        low_t_decay_alpha=-0.12,
    ),
    29: LayerTimestepConfig(
        layer_idx=29,
        mu=300,
        sigma=80,
        primary_range=(150, 450),
        low_t_decay_alpha=-0.10,
    ),
}


# ============================================================================
# 空间采样配置
# ============================================================================

@dataclass
class SpatialSamplingConfig:
    """
    空间采样配置 (训练阶段)

    仅用于显存控制，不改变分布
    """
    # Layer-wise spatial stride
    # 浅层可以更大 stride (特征冗余度高)
    # 深层保持原始分辨率 (特征更稀疏)
    layer_strides: Dict[int, int] = field(default_factory=lambda: {
        14: 2,  # 浅层，特征冗余，可以 stride
        19: 2,
        24: 1,  # 深层，特征稀疏，保持原始
        29: 1,
    })

    def get_stride(self, layer_idx: int) -> int:
        return self.layer_strides.get(layer_idx, 1)


# ============================================================================
# Norm Bias 配置
# ============================================================================

@dataclass
class NormBiasConfig:
    """
    Soft Norm Bias 配置 (训练阶段)

    仅允许 weak bias，保持真实分布
    """
    enabled: bool = True
    # Soft bias: prob ∝ exp(bias_strength * normalized_norm)
    # bias_strength 应该很小 (0.1 ~ 0.2)
    bias_strength: float = 0.15

    # 注意：训练阶段禁止 hard bucket sampling!


# ============================================================================
# 完整训练采样配置
# ============================================================================

@dataclass
class TrainingSamplingConfig:
    """
    训练阶段完整采样配置
    """
    # 基础配置
    d_model: int = D_MODEL
    d_hidden: int = D_HIDDEN
    hook_layers: List[int] = field(default_factory=lambda: HOOK_LAYERS)

    # Timestep 配置
    min_timestep: int = MIN_TIMESTEP
    max_timestep: int = MAX_TIMESTEP
    layer_timestep_configs: Dict[int, LayerTimestepConfig] = field(
        default_factory=lambda: LAYER_TIMESTEP_CONFIGS
    )

    # 空间配置
    spatial: SpatialSamplingConfig = field(default_factory=SpatialSamplingConfig)

    # Norm bias
    norm_bias: NormBiasConfig = field(default_factory=NormBiasConfig)

    # Batch 配置
    max_tokens_per_batch: int = 4096
    batch_size: int = 4

    # 随机种子
    seed: int = 42

    # ===== 禁止项 =====
    # 这些设置强制禁用，不允许修改
    decorrelation_enabled: bool = False  # 禁止
    oversample_ratio: float = 1.0        # 禁止 > 1
    hard_bucket_enabled: bool = False    # 禁止

    def __post_init__(self):
        # 强制验证
        assert self.decorrelation_enabled == False, "训练阶段禁止 decorrelation"
        assert self.oversample_ratio == 1.0, "训练阶段禁止 oversampling"
        assert self.hard_bucket_enabled == False, "训练阶段禁止 hard bucket sampling"

    def get_layer_config(self, layer_idx: int) -> LayerTimestepConfig:
        """获取指定层的 timestep 配置"""
        if layer_idx in self.layer_timestep_configs:
            return self.layer_timestep_configs[layer_idx]
        # 默认配置 (中间值)
        return LayerTimestepConfig(
            layer_idx=layer_idx,
            mu=500,
            sigma=100,
        )


# ============================================================================
# 便捷函数
# ============================================================================

def get_default_training_config() -> TrainingSamplingConfig:
    """获取默认训练配置"""
    return TrainingSamplingConfig()


def validate_timestep(t: int, min_t: int = MIN_TIMESTEP, max_t: int = MAX_TIMESTEP) -> bool:
    """验证 timestep 是否在有效区间"""
    return min_t <= t <= max_t


def clamp_timestep(t: int, min_t: int = MIN_TIMESTEP, max_t: int = MAX_TIMESTEP) -> int:
    """Clamp timestep 到有效区间"""
    return max(min_t, min(max_t, t))
