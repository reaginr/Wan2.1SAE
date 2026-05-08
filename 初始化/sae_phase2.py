"""
SAE 第二阶段核心模块 - 统一入口

包含:
1. NormDenormManager - Per-Token RMSNorm归一化管理器
2. TopKSAE, TopKSAEConfig - 工业TopK SAE实现
3. SAEInitializer, SAEInitConfig - 工业初始化器
4. SAETrainingLogger - 训练日志系统
5. WanTokenMapper - 时空Token映射工具
6. ActivationCollector - 激活采集器

核心流程:
    Step1: 激活采集 (单次forward多层hook)
    Step2: 初始化SAE (PCA + Tied绑定)
    Step3: 训练SAE

使用方法:
    from 初始化.sae_phase2 import (
        TopKSAE, TopKSAEConfig,
        NormDenormManager,
        SAEInitializer, SAEInitConfig,
        ActivationCollector, ActivationCollectorConfig,
        SAETrainingLogger, get_training_logger,
        WanTokenMapper,
    )

    # Step1: 采集激活
    collector_config = ActivationCollectorConfig(
        checkpoint_dir="F:/Wan2.1-T2V-1.3B",
        prompt_file="./prompts.txt",
        cache_dir="./cache",
    )
    collector = ActivationCollector(collector_config)
    collector.collect_activations(num_samples=500)

    # Step2: 初始化SAE
    sae_config = TopKSAEConfig(d_model=1536, d_hidden=12288, top_k=128)
    sae = TopKSAE(sae_config)

    init_config = SAEInitConfig(cache_dir="./cache")
    initializer = SAEInitializer(init_config, sae)
    initializer.initialize_from_cache(layer_idx=14)

    # Step3: 训练
    logger = get_training_logger("./logs", "exp1")
    # ... training loop ...
"""

from 初始化.sae_phase2_norm import (
    NormDenormManager,
    PerTokenRMSNorm,
)

from 初始化.sae_phase2_core import (
    TopKSAE,
    TopKSAEConfig,
    SAELossComputer,
)

from 初始化.sae_phase2_init import (
    SAEInitializer,
    SAEInitConfig,
    weiszfeld_geometric_median,
)

from 初始化.sae_phase2_logger import (
    SAETrainingLogger,
    SAETrainingLoggerConfig,
    SAEValidationLogger,
    get_training_logger,
    get_validation_logger,
)

from 初始化.sae_activation_collector import (
    ActivationCollector,
    ActivationCollectorConfig,
)

from 初始化.token_mapper import (
    WanTokenMapper,
)

__all__ = [
    # 核心SAE
    "TopKSAE",
    "TopKSAEConfig",
    "SAELossComputer",

    # 归一化
    "NormDenormManager",
    "PerTokenRMSNorm",

    # 初始化
    "SAEInitializer",
    "SAEInitConfig",
    "weiszfeld_geometric_median",

    # 激活采集
    "ActivationCollector",
    "ActivationCollectorConfig",

    # 日志
    "SAETrainingLogger",
    "SAETrainingLoggerConfig",
    "SAEValidationLogger",
    "get_training_logger",
    "get_validation_logger",

    # Token映射
    "WanTokenMapper",
]

# 版本信息
__version__ = "2.0.0"
__phase__ = "SAE核心架构与初始化"
