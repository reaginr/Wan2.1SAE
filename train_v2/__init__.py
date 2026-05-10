"""
SAE Training Pipeline V2

严格按照 TODO_list_v3.md 第三阶段规范实现

核心特性:
- Layer-wise 顺序训练 (14 -> 19 -> 24 -> 29)
- 梯度累积 (batch=4, accum=8, effective=32)
- EMA 权重 (decay=0.999)
- 死神经元监控 (window=2000)
- 早停检测 (dead > 20% 或 MSE 无改进)
- Warmup + Cosine LR 调度

模块结构:
- config: 训练配置
- optimizer: 优化器和 LR 调度器
- gradient_accumulator: 梯度累积
- ema: EMA 权重管理
- sae_engine: SAE 前向 + 损失计算
- dead_neuron_monitor: 死神经元监控
- validator: 验证指标
- checkpoint: Checkpoint 管理
- training_loop: 主训练循环
- layer_trainer: Layer-wise 训练协调

使用示例:
    from train_v2 import (
        TrainingConfig,
        LayerTrainer,
        create_training_pipeline,
    )

    # 配置
    config = TrainingConfig()

    # 训练
    trainer = LayerTrainer(config, device="cuda")
    results = trainer.train_all_layers(train_loaders, val_loaders)

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

# ============================================================================
# 核心配置
# ============================================================================

from train_v2.config import (
    TrainingConfig,
    ValidationConfig,
    get_default_training_config,
    get_layer_training_config,
    validate_layer_order,
    assert_single_layer,
    D_MODEL,
    D_HIDDEN,
    TOP_K,
    HOOK_LAYERS,
    MIN_TIMESTEP,
    MAX_TIMESTEP,
)

# ============================================================================
# 优化器
# ============================================================================

from train_v2.optimizer import (
    create_optimizer,
    create_optimizer_and_scheduler,
    get_current_lr,
    get_lr_stats,
    save_optimizer_state,
    load_optimizer_state,
    validate_warmup_schedule,
    print_lr_schedule,
)

# ============================================================================
# 梯度累积
# ============================================================================

from train_v2.gradient_accumulator import (
    GradientAccumulator,
    AccumulationStats,
    AccumulationContext,
    create_gradient_accumulator,
    validate_gradient_accumulation,
)

# ============================================================================
# EMA
# ============================================================================

from train_v2.ema import (
    EMAManager,
    EMAState,
    EMAContext,
    create_ema_manager,
    validate_ema_correctness,
    print_ema_stats,
)

# ============================================================================
# SAE Engine
# ============================================================================

from train_v2.sae_engine import (
    SAEEngine,
    LossInfo,
    per_token_rms_norm,
    create_sae_engine,
    create_sae_engine_from_sae,
    create_sae_engine_from_init,
    validate_loss_computation,
    print_loss_info,
)

# ============================================================================
# 死神经元监控
# ============================================================================

from train_v2.dead_neuron_monitor import (
    DeadNeuronMonitor,
    DeadNeuronStats,
    FeatureUsageTracker,
    create_dead_neuron_monitor,
    print_dead_neuron_report,
)

# ============================================================================
# 验证器
# ============================================================================

from train_v2.validator import (
    TrainingValidator,
    ValidationMetrics,
    ValidationHistory,
    create_validator,
    print_validation_report,
)

# ============================================================================
# Checkpoint
# ============================================================================

from train_v2.checkpoint import (
    CheckpointManager,
    CheckpointData,
    create_checkpoint_manager,
    save_checkpoint,
    load_checkpoint,
)

# ============================================================================
# 训练循环
# ============================================================================

from train_v2.training_loop import (
    TrainingLoop,
    TrainingState,
    create_training_loop,
)

# ============================================================================
# Layer Trainer
# ============================================================================

from train_v2.layer_trainer import (
    LayerTrainer,
    LayerTrainingResult,
    MultiLayerResult,
    create_layer_trainer,
    train_layer_14,
    train_layer_19,
    train_layer_24,
    train_layer_29,
)


# ============================================================================
# 版本信息
# ============================================================================

__version__ = "2.0.0"
__author__ = "Claude"

__all__ = [
    # 配置
    "TrainingConfig",
    "ValidationConfig",
    "get_default_training_config",
    "get_layer_training_config",
    "validate_layer_order",
    "assert_single_layer",
    "D_MODEL",
    "D_HIDDEN",
    "TOP_K",
    "HOOK_LAYERS",
    "MIN_TIMESTEP",
    "MAX_TIMESTEP",

    # 优化器
    "create_optimizer",
    "create_optimizer_and_scheduler",
    "get_current_lr",
    "get_lr_stats",
    "save_optimizer_state",
    "load_optimizer_state",
    "validate_warmup_schedule",
    "print_lr_schedule",

    # 梯度累积
    "GradientAccumulator",
    "AccumulationStats",
    "AccumulationContext",
    "create_gradient_accumulator",
    "validate_gradient_accumulation",

    # EMA
    "EMAManager",
    "EMAState",
    "EMAContext",
    "create_ema_manager",
    "validate_ema_correctness",
    "print_ema_stats",

    # SAE Engine
    "SAEEngine",
    "LossInfo",
    "per_token_rms_norm",
    "create_sae_engine",
    "create_sae_engine_from_sae",
    "create_sae_engine_from_init",
    "validate_loss_computation",
    "print_loss_info",

    # 死神经元监控
    "DeadNeuronMonitor",
    "DeadNeuronStats",
    "FeatureUsageTracker",
    "create_dead_neuron_monitor",
    "print_dead_neuron_report",

    # 验证器
    "TrainingValidator",
    "ValidationMetrics",
    "ValidationHistory",
    "create_validator",
    "print_validation_report",

    # Checkpoint
    "CheckpointManager",
    "CheckpointData",
    "create_checkpoint_manager",
    "save_checkpoint",
    "load_checkpoint",

    # 训练循环
    "TrainingLoop",
    "TrainingState",
    "create_training_loop",

    # Layer Trainer
    "LayerTrainer",
    "LayerTrainingResult",
    "MultiLayerResult",
    "create_layer_trainer",
    "train_layer_14",
    "train_layer_19",
    "train_layer_24",
    "train_layer_29",
]
