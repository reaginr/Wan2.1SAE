"""
初始化模块

包含 SAE 训练第二阶段所需的所有核心组件:
- Token 时空映射工具
- Per-Token RMSNorm 归一化管理器
- TopK SAE 核心架构
- 工业初始化器
- 激活采集器
- 日志系统

核心流程:
    Step1: 激活采集 (单次forward多层hook)
    Step2: 初始化SAE (PCA + Tied绑定)
    Step3: 训练SAE
"""

from 初始化.token_mapper import WanTokenMapper
from 初始化.sae_phase2 import (
    TopKSAE,
    TopKSAEConfig,
    NormDenormManager,
    SAEInitializer,
    SAEInitConfig,
    ActivationCollector,
    ActivationCollectorConfig,
    SAETrainingLogger,
    get_training_logger,
)

__all__ = [
    # Token映射
    "WanTokenMapper",

    # SAE核心
    "TopKSAE",
    "TopKSAEConfig",

    # 归一化
    "NormDenormManager",

    # 初始化
    "SAEInitializer",
    "SAEInitConfig",

    # 激活采集
    "ActivationCollector",
    "ActivationCollectorConfig",

    # 日志
    "SAETrainingLogger",
    "get_training_logger",
]
