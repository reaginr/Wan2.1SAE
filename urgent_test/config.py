"""
统一配置文件 - TODO_list_v4 (紧急版)

所有参数集中管理，方便统一修改和追踪

使用方法:
    from config import PATH_PARAMS, LAYER_TIMESTEP_PARAMS, SAE_PARAMS, TRAINING_PARAMS

作者：Claude
日期：2026-05-16
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ============================================================================
# 核心约束声明 (TODO_list_v4 强制要求)
# ============================================================================

"""
TODO_list_v4 训练阶段核心约束:

1. Timestep 范围: t ∈ [150, 800]
   - 禁止 t < 150 (低噪声区域，特征过于确定)
   - 禁止 t > 800 (高噪声区域，特征过于随机)

2. Truncated Gaussian Sampling:
   - 不是均匀采样
   - 从 N(μ, σ²) 采样，截断到 [150, 800]

3. 各层采样参数 (学术意义):
   - Layer 14 (浅层): μ=650, σ=80 → 高噪声 → 关注整体结构
   - Layer 19 (中浅层): μ=550, σ=80 → 中高噪声 → 结构过渡
   - Layer 24 (中深层): μ=420, σ=70 → 中低噪声 → 语义特征
   - Layer 29 (深层): μ=300, σ=60 → 低噪声 → 细节语义

4. 禁止事项:
   ❌ 禁止 oversample
   ❌ 禁止 decorrelation
   ❌ 禁止 weight decay (必须为 0.0)
   ❌ 禁止 timestep 超出 [150, 800]

5. SAE 结构约束:
   - d_model: 1536 (DiT 维度, 1.3B 模型固定)
   - d_hidden: 12288 (8x expansion)
   - top_k: 128 (~1% sparsity)
   - 使用 TopK 激活函数
   - 使用 RMSNorm 预处理
"""


# ============================================================================
# 路径配置
# ============================================================================

@dataclass
class PathConfig:
    """路径配置"""

    # Wan 模型路径
    model_path: str = "../Wan/Wan2.1-T2V-1.3B"

    # SAE 初始化权重目录
    sae_init_dir: str = "./sae_init"

    # 提示词文件
    prompt_file: str = "./初始化/final_clean_prompts.txt"

    # 提示词目录 (用于 latent 提取)
    prompt_dir: str = "./final_cleaned"

    # 输出目录
    output_dir: str = "./outputs"

    # 训练运行目录
    run_dir: str = "sae_runs/layer_specific_train"

    # 日志文件
    log_file: str = "./logs/train.log"


PATH_PARAMS = PathConfig()


# ============================================================================
# Timestep 配置 (核心约束)
# ============================================================================

@dataclass
class TimestepConfig:
    """Timestep 配置 (TODO_list_v4 强制约束)"""

    # 全局约束
    min_timestep: int = 150      # 最小 timestep (禁止 < 150)
    max_timestep: int = 800      # 最大 timestep (禁止 > 800)

    # 各层采样参数 (学术意义: 不同深度关注不同噪声水平的特征)
    # Layer 14: 浅层, 高噪声, 关注整体结构
    # Layer 19: 中浅层, 中高噪声
    # Layer 24: 中深层, 中低噪声, 语义特征
    # Layer 29: 深层, 低噪声, 细节语义

    layer_params: Dict[int, Dict[str, float]] = field(default_factory=lambda: {
        14: {"mu": 650, "sigma": 80, "min_t": 150, "max_t": 800},
        19: {"mu": 550, "sigma": 80, "min_t": 150, "max_t": 800},
        24: {"mu": 420, "sigma": 70, "min_t": 150, "max_t": 800},
        29: {"mu": 300, "sigma": 60, "min_t": 150, "max_t": 800},
    })

    def get_layer_params(self, layer_idx: int) -> Dict[str, float]:
        """获取指定层的采样参数"""
        return self.layer_params.get(layer_idx, {
            "mu": 300,
            "sigma": 80,
            "min_t": self.min_timestep,
            "max_t": self.max_timestep,
        })

    def validate(self):
        """验证约束"""
        for layer, params in self.layer_params.items():
            assert params["min_t"] >= self.min_timestep, \
                f"Layer {layer} min_t {params['min_t']} < {self.min_timestep}"
            assert params["max_t"] <= self.max_timestep, \
                f"Layer {layer} max_t {params['max_t']} > {self.max_timestep}"
            assert params["min_t"] <= params["mu"] <= params["max_t"], \
                f"Layer {layer} mu {params['mu']} not in [{params['min_t']}, {params['max_t']}]"


TIMESTEP_PARAMS = TimestepConfig()


# ============================================================================
# SAE 结构配置
# ============================================================================

@dataclass
class SAEConfig:
    """SAE 结构配置"""

    # DiT 维度 (1.3B 模型固定)
    d_model: int = 1536

    # SAE 扩展维度 (8x expansion)
    d_hidden: int = 12288

    # TopK 稀疏度 (~1%)
    top_k: int = 128

    # Hook 模式
    hook_mode: str = "block_out"

    # Hook 层 (必须顺序训练)
    hook_layers: List[int] = field(default_factory=lambda: [14, 19, 24, 29])

    # RMSNorm epsilon
    rms_norm_eps: float = 1e-6

    def validate(self):
        """验证约束"""
        assert self.d_hidden == 8 * self.d_model, \
            f"d_hidden {self.d_hidden} != 8 * d_model {self.d_model}"
        assert self.top_k >= 64, f"top_k {self.top_k} < 64"
        assert self.hook_mode in ["block_out", "self_attn", "cross_attn"], \
            f"Invalid hook_mode: {self.hook_mode}"


SAE_PARAMS = SAEConfig()


# ============================================================================
# 训练超参数
# ============================================================================

@dataclass
class TrainingConfig:
    """训练超参数"""

    # 训练步数
    steps: int = 2000

    # 批大小
    batch_size: int = 4

    # 梯度累积步数
    accum_steps: int = 8

    # 有效批大小
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accum_steps

    # 学习率
    lr: float = 6e-5

    # 最小学习率
    min_lr: float = 1e-5

    # 预热步数
    warmup_steps: int = 400

    # 权重衰减 (必须为 0.0)
    weight_decay: float = 0.0

    # 梯度裁剪
    grad_clip: float = 0.3

    # EMA 衰减
    ema_decay: float = 0.999

    # Adam betas
    betas: tuple = (0.95, 0.999)

    def validate(self):
        """验证约束"""
        assert self.weight_decay == 0.0, \
            f"weight_decay must be 0.0, got {self.weight_decay}"
        assert self.batch_size * self.accum_steps >= 16, \
            f"Effective batch size too small: {self.effective_batch_size}"


TRAINING_PARAMS = TrainingConfig()


# ============================================================================
# 采样配置
# ============================================================================

@dataclass
class SamplingConfig:
    """采样配置"""

    # 每个 prompt 采样的 timestep 数
    num_timesteps_per_prompt: int = 5

    # 每 timestep 保留的 token 数
    tokens_per_timestep: int = 1536

    # DiT 扩散步数
    sampling_steps: int = 30

    # 空间 stride (必须为 1)
    spatial_stride: int = 1

    # 是否禁止 oversample
    no_oversample: bool = True

    # 是否禁止 decorrelation
    no_decorrelation: bool = True

    def validate(self):
        """验证约束"""
        assert self.spatial_stride == 1, \
            f"spatial_stride must be 1, got {self.spatial_stride}"
        assert self.no_oversample, "oversample is forbidden"
        assert self.no_decorrelation, "decorrelation is forbidden"


SAMPLING_PARAMS = SamplingConfig()


# ============================================================================
# 参数测试阶段采样配置 (新增)
# ============================================================================

@dataclass
class ParamTestSamplingConfig:
    """
    参数测试阶段采样配置

    设计目标：在保持时空局部性的前提下，提供适度的特征多样性

    与训练阶段的区别：
    1. Temporal Chunk: 采样连续帧，保持时间局部性
    2. Spatial Block: 在局部 patch 内采样，而非全局随机
    3. Soft Norm Bias: 轻度偏向高 norm token
    4. Mild Decorrelation: 轻度去相关，不破坏局部结构
    """

    # ========== Timestep 采样 ==========
    # 每个 prompt 采样的 timestep 数 (可配置 5~8)
    num_timesteps_per_prompt: int = 5

    # ========== Temporal Chunk 采样 ==========
    # 每个 timestep 采样的连续帧数 (可配置 2~3)
    temporal_chunk_size: int = 3

    # 总帧数 (latent T', 对于 81 帧视频约为 11)
    temporal_frames_total: int = 11

    # ========== Spatial Block 采样 ==========
    # Spatial block 尺寸 (8×8 patch)
    spatial_block_size: int = 8

    # Latent grid 尺寸 (H, W)
    spatial_grid_h: int = 30  # 480 / 8 / 2
    spatial_grid_w: int = 52  # 832 / 8 / 2

    # 采样多少个 spatial block (每个帧)
    num_spatial_blocks: int = 24

    # ========== Norm Bias 采样 ==========
    # 是否启用 soft norm bias
    norm_bias_enabled: bool = True

    # Norm bias 强度 (0=无偏置, 1=强偏置)
    # 采样概率 p ∝ norm^(norm_bias_strength)
    norm_bias_strength: float = 0.3

    # ========== Mild Decorrelation ==========
    # 是否启用轻度去相关
    decorrelation_enabled: bool = True

    # 去相关阈值 (比 init 的 0.3 宽松)
    decorrelation_threshold: float = 0.7

    # ========== 目标 token 数 ==========
    # 每 timestep 目标 token 数 (可配置 1536~2048)
    tokens_per_timestep: int = 1536

    # ========== 采样模式 ==========
    # 采样模式: 'train' | 'param_test'
    # - 'train': 训练阶段采样 (全局随机，无 bias)
    # - 'param_test': 参数测试阶段采样 (时空局部性 + soft bias)
    sampling_mode: str = "param_test"

    def validate(self):
        """验证配置"""
        assert self.temporal_chunk_size in [2, 3], \
            f"temporal_chunk_size should be 2 or 3, got {self.temporal_chunk_size}"
        assert 0.0 <= self.norm_bias_strength <= 1.0, \
            f"norm_bias_strength should be in [0, 1], got {self.norm_bias_strength}"
        assert 0.0 < self.decorrelation_threshold <= 1.0, \
            f"decorrelation_threshold should be in (0, 1], got {self.decorrelation_threshold}"
        assert self.sampling_mode in ['train', 'param_test'], \
            f"sampling_mode should be 'train' or 'param_test', got {self.sampling_mode}"


PARAM_TEST_SAMPLING_PARAMS = ParamTestSamplingConfig()


# ============================================================================
# 验证配置
# ============================================================================

@dataclass
class ValidationConfig:
    """验证配置"""

    # 验证间隔
    val_interval: int = 200

    # Checkpoint 保存间隔
    checkpoint_interval: int = 500

    # 筛选条件
    min_cohen_d: float = 1.0      # |d| > 1.0
    min_activation_freq: float = 0.01  # 激活频率 > 1%

    # Top-K features
    top_k_features: int = 50


VALIDATION_PARAMS = ValidationConfig()


# ============================================================================
# 视频生成配置
# ============================================================================

@dataclass
class VideoGenConfig:
    """视频生成配置"""

    # 帧数
    frame_num: int = 81

    # 分辨率
    size: tuple = (832, 480)

    # 采样步数
    sampling_steps: int = 30

    # 干预强度
    gamma_values: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.5, 0.8, 1.0])

    # 每概念最大干预对数
    max_pairs_per_concept: int = 5


VIDEO_PARAMS = VideoGenConfig()


# ============================================================================
# 评估配置
# ============================================================================

@dataclass
class EvaluationConfig:
    """评估配置"""

    # NSFW 阈值
    nsfw_threshold: float = 0.5

    # Violence 阈值
    violence_threshold: float = 0.5

    # 目标 AUC
    target_auc: float = 0.85

    # MUSIQ 设备
    musiq_device: str = "cuda"


EVAL_PARAMS = EvaluationConfig()


# ============================================================================
# 数据集配置
# ============================================================================

@dataclass
class DatasetConfig:
    """数据集配置"""

    # 正负样本文件映射
    file_mapping: Dict[str, tuple] = field(default_factory=lambda: {
        "pos_prompt_1": ("sex", "positive"),
        "neg_prompt_1": ("sex", "negative"),
        "pos_prompt_3": ("violence", "positive"),
        "neg_prompt_3": ("violence", "negative"),
    })

    # 概念定义
    concepts: Dict[str, List[str]] = field(default_factory=lambda: {
        "sex": ["sex_positive", "sex_negative"],
        "violence": ["violence_positive", "violence_negative"],
    })

    # 各类别数量建议
    category_counts: Dict[str, int] = field(default_factory=lambda: {
        "sex_positive": 50,
        "sex_negative": 50,
        "violence_positive": 50,
        "violence_negative": 50,
        "clean_prompts": 100,
    })


DATASET_PARAMS = DatasetConfig()


# ============================================================================
# 全局验证
# ============================================================================

def validate_all_configs():
    """验证所有配置是否符合 TODO_list_v4 约束"""
    errors = []

    try:
        TIMESTEP_PARAMS.validate()
    except AssertionError as e:
        errors.append(f"TIMESTEP_PARAMS: {e}")

    try:
        SAE_PARAMS.validate()
    except AssertionError as e:
        errors.append(f"SAE_PARAMS: {e}")

    try:
        TRAINING_PARAMS.validate()
    except AssertionError as e:
        errors.append(f"TRAINING_PARAMS: {e}")

    try:
        SAMPLING_PARAMS.validate()
    except AssertionError as e:
        errors.append(f"SAMPLING_PARAMS: {e}")

    try:
        PARAM_TEST_SAMPLING_PARAMS.validate()
    except AssertionError as e:
        errors.append(f"PARAM_TEST_SAMPLING_PARAMS: {e}")

    if errors:
        print("[CONFIG VALIDATION ERRORS]")
        for err in errors:
            print(f"  - {err}")
        return False

    print("[OK] All configurations validated successfully")
    return True


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # 配置对象
    "PATH_PARAMS",
    "TIMESTEP_PARAMS",
    "SAE_PARAMS",
    "TRAINING_PARAMS",
    "SAMPLING_PARAMS",
    "PARAM_TEST_SAMPLING_PARAMS",
    "VALIDATION_PARAMS",
    "VIDEO_PARAMS",
    "EVAL_PARAMS",
    "DATASET_PARAMS",
    # 验证函数
    "validate_all_configs",
    # 配置类
    "PathConfig",
    "TimestepConfig",
    "SAEConfig",
    "TrainingConfig",
    "SamplingConfig",
    "ParamTestSamplingConfig",
    "ValidationConfig",
    "VideoGenConfig",
    "EvaluationConfig",
    "DatasetConfig",
]


if __name__ == "__main__":
    # 打印所有配置
    print("=" * 70)
    print("TODO_list_v4 Configuration Summary")
    print("=" * 70)

    print("\n[PATH_PARAMS]")
    print(f"  model_path: {PATH_PARAMS.model_path}")
    print(f"  sae_init_dir: {PATH_PARAMS.sae_init_dir}")
    print(f"  prompt_file: {PATH_PARAMS.prompt_file}")

    print("\n[TIMESTEP_PARAMS] (Core Constraints)")
    print(f"  Global range: [{TIMESTEP_PARAMS.min_timestep}, {TIMESTEP_PARAMS.max_timestep}]")
    for layer, params in TIMESTEP_PARAMS.layer_params.items():
        print(f"  Layer {layer}: μ={params['mu']}, σ={params['sigma']}")

    print("\n[SAE_PARAMS]")
    print(f"  d_model: {SAE_PARAMS.d_model}")
    print(f"  d_hidden: {SAE_PARAMS.d_hidden} ({SAE_PARAMS.d_hidden // SAE_PARAMS.d_model}x)")
    print(f"  top_k: {SAE_PARAMS.top_k}")
    print(f"  hook_layers: {SAE_PARAMS.hook_layers}")

    print("\n[TRAINING_PARAMS]")
    print(f"  steps: {TRAINING_PARAMS.steps}")
    print(f"  lr: {TRAINING_PARAMS.lr}")
    print(f"  weight_decay: {TRAINING_PARAMS.weight_decay} (MUST BE 0.0)")
    print(f"  grad_clip: {TRAINING_PARAMS.grad_clip}")

    print("\n[SAMPLING_PARAMS] (Training Stage - Forbidden)")
    print(f"  oversample: NOT {SAMPLING_PARAMS.no_oversample}")
    print(f"  decorrelation: NOT {SAMPLING_PARAMS.no_decorrelation}")
    print(f"  spatial_stride: {SAMPLING_PARAMS.spatial_stride} (MUST BE 1)")

    print("\n[PARAM_TEST_SAMPLING_PARAMS] (Parameter Test Stage)")
    print(f"  sampling_mode: {PARAM_TEST_SAMPLING_PARAMS.sampling_mode}")
    print(f"  num_timesteps_per_prompt: {PARAM_TEST_SAMPLING_PARAMS.num_timesteps_per_prompt}")
    print(f"  temporal_chunk_size: {PARAM_TEST_SAMPLING_PARAMS.temporal_chunk_size}")
    print(f"  spatial_block_size: {PARAM_TEST_SAMPLING_PARAMS.spatial_block_size}x{PARAM_TEST_SAMPLING_PARAMS.spatial_block_size}")
    print(f"  num_spatial_blocks: {PARAM_TEST_SAMPLING_PARAMS.num_spatial_blocks}")
    print(f"  norm_bias_enabled: {PARAM_TEST_SAMPLING_PARAMS.norm_bias_enabled}")
    print(f"  norm_bias_strength: {PARAM_TEST_SAMPLING_PARAMS.norm_bias_strength}")
    print(f"  decorrelation_enabled: {PARAM_TEST_SAMPLING_PARAMS.decorrelation_enabled}")
    print(f"  decorrelation_threshold: {PARAM_TEST_SAMPLING_PARAMS.decorrelation_threshold}")
    print(f"  tokens_per_timestep: {PARAM_TEST_SAMPLING_PARAMS.tokens_per_timestep}")

    print("\n" + "=" * 70)
    validate_all_configs()
    print("=" * 70)
