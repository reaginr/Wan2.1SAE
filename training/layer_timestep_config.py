"""
Layer-wise Timestep Configuration

定义各层的 timestep 采样参数

设计原则：
1. 基于 diffusion semantic hierarchy
2. 浅层需要更多噪声保留结构
3. 深层需要更少噪声保留语义

作者：Claude
日期：2026-05-10
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class LayerTimestepParams:
    """
    单层的 Timestep 参数

    参数:
        mu: Gaussian 中心
        sigma: Gaussian 标准差
        primary_range: 主采样区间 (μ ± 2σ)
        semantic_description: 语义描述
    """
    mu: float
    sigma: float
    primary_range: Tuple[int, int]
    semantic_description: str = ""


# ============================================================================
# Layer-wise Timestep 分布设计
# ============================================================================

# 设计原理：
#
# Diffusion 过程中，不同 timestep 对应不同程度的噪声：
# - t ∈ [800, 1000]: 纯噪声，无语义结构
# - t ∈ [400, 800]: 结构信息主导
# - t ∈ [150, 400]: 语义信息主导
# - t ∈ [0, 150]: 已收敛，representation collapse
#
# 不同层的特征：
# - Layer 14 (浅层): 捕获空间结构 → 需要较高噪声级别
# - Layer 19 (中深): 开始捕获语义 → 中等噪声级别
# - Layer 24 (深层): 高级语义 → 较低噪声级别
# - Layer 29 (最深): 最终表征 → 最低噪声级别
#
# 因此设计为：
# - 浅层 μ 较高，采样区间偏右
# - 深层 μ 较低，采样区间偏左

LAYER_TIMESTEP_PARAMS: Dict[int, LayerTimestepParams] = {
    14: LayerTimestepParams(
        mu=650,
        sigma=120,
        primary_range=(400, 800),
        semantic_description=(
            "Layer 14 是浅层，主要捕获空间结构特征。\n"
            "需要较高的噪声级别 (t=400~800) 保留结构信息。\n"
            "高噪声区域的特征更有助于学习空间布局和纹理。"
        ),
    ),
    19: LayerTimestepParams(
        mu=550,
        sigma=110,
        primary_range=(300, 700),
        semantic_description=(
            "Layer 19 是中深层，开始捕获语义特征。\n"
            "需要中等噪声级别 (t=300~700) 平衡结构与语义。\n"
            "是空间特征和语义特征的过渡层。"
        ),
    ),
    24: LayerTimestepParams(
        mu=420,
        sigma=100,
        primary_range=(200, 600),
        semantic_description=(
            "Layer 24 是深层，捕获高级语义特征。\n"
            "需要较低噪声级别 (t=200~600) 突出语义信息。\n"
            "低噪声区域的特征更清晰，语义更明确。"
        ),
    ),
    29: LayerTimestepParams(
        mu=300,
        sigma=80,
        primary_range=(150, 450),
        semantic_description=(
            "Layer 29 是最深层，捕获最终表征。\n"
            "需要最低噪声级别 (t=150~450) 保留细节。\n"
            "最深层特征最稀疏，低噪声能保留最多语义细节。"
        ),
    ),
}


# ============================================================================
# 可视化参数
# ============================================================================

def get_layer_distribution_visualization_data() -> Dict:
    """
    获取用于可视化的数据
    """
    import numpy as np

    data = {}
    t_range = np.arange(0, 1001, 1)

    for layer_idx, params in LAYER_TIMESTEP_PARAMS.items():
        # 计算 Gaussian PDF
        pdf = np.exp(-((t_range - params.mu) ** 2) / (2 * params.sigma ** 2))
        pdf = pdf / (params.sigma * np.sqrt(2 * np.pi))

        # 截断到有效区间
        pdf[t_range < 150] = 0
        pdf[t_range > 800] = 0

        data[layer_idx] = {
            "t_range": t_range.tolist(),
            "pdf": pdf.tolist(),
            "mu": params.mu,
            "sigma": params.sigma,
            "primary_range": params.primary_range,
        }

    return data


def print_layer_config_table():
    """打印层配置表格"""
    print("\n" + "=" * 70)
    print("Layer-wise Timestep Distribution Configuration")
    print("=" * 70)

    print(f"\n{'Layer':<8} {'μ':<8} {'σ':<8} {'Primary Range':<15} {'Description'}")
    print("-" * 70)

    for layer_idx, params in LAYER_TIMESTEP_PARAMS.items():
        range_str = f"{params.primary_range[0]}-{params.primary_range[1]}"
        desc = params.semantic_description.split('\n')[0][:30]
        print(f"{layer_idx:<8} {params.mu:<8} {params.sigma:<8} {range_str:<15} {desc}")

    print("\n" + "=" * 70)

    print("\n[为什么需要 Truncated Gaussian]")
    print("-" * 70)
    print("""
为什么必须 clamp 到 [150, 800]:

1. t < 150 (Collapse Region):
   - Diffusion 已接近收敛
   - Representation 已经定型，缺乏多样性
   - SAE 学习到的特征会 collapse 到少数模式

2. t > 800 (Noise Region):
   - 纯噪声 latent
   - 无 semantic structure
   - SAE 学习到的是噪声特征，无意义

3. t ∈ [150, 800] (Semantic Region):
   - 包含真实的语义结构
   - 噪声与语义的平衡
   - SAE 学习到的是有意义的特征

为什么使用 Layer-aware Distribution:

1. 浅层 (Layer 14):
   - 捕获空间结构 (边缘、纹理、布局)
   - 结构信息在高噪声时更明显
   - 因此 μ=650，偏向高 timestep

2. 深层 (Layer 29):
   - 捕获语义信息 (概念、属性、关系)
   - 语义信息在低噪声时更清晰
   - 因此 μ=300，偏向低 timestep

训练目标:
- 不是 reconstruction 最优
- 而是 stable sparse feature decomposition
- 需要正确的 timestep 分布
    """)


if __name__ == "__main__":
    print_layer_config_table()
