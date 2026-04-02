"""
SAE干预生成模块

功能：
1. steering_generator.py - 通过概念向量干预视频生成

核心功能：
- 加载已训练的SAE和概念向量
- 在DiT生成过程中对特定层进行干预
- 支持多个概念向量的组合干预
- 持久化干预配置和生成结果

使用示例：
    # 使用配置文件
    python -m wan.sae.steering.steering_generator \
        --config steering_config.json

    # 或使用命令行参数
    python -m wan.sae.steering.steering_generator \
        --checkpoint_dir ./Wan2.1-T2V-1.3B \
        --run_dir sae_runs/exp1 \
        --concept_dir concept_vectors \
        --prompt "A peaceful scene" \
        --output_dir steering_outputs

配置文件格式 (steering_config.json)：
{
    "prompt": "生成提示词",
    "interventions": [
        {
            "concept_name": "violence",
            "layer_key": "block_out.layer15",
            "strength": -0.5,  // 负值表示抑制
            "method": "additive",
            "timestep_range": [0, 30]
        }
    ],
    "size_w": 832,
    "size_h": 480,
    "frame_num": 81,
    "seed": 42
}

干预方法：
- "additive": z_new = z + strength * concept_vector
- "multiplicative": z_new = z * (1 + strength * concept_vector)
- "projection": 沿concept_vector方向调整
- "clamp": 限制concept_vector方向的激活值范围
"""

from .steering_generator import (
    SteeringSession,
    InterventionConfig,
    ConceptVectorManager,
    SAEIntervener,
    generate_with_intervention,
)

__all__ = [
    "SteeringSession",
    "InterventionConfig",
    "ConceptVectorManager",
    "SAEIntervener",
    "generate_with_intervention",
]
