"""
Layer-wise Training Orchestration

严格按照 TODO_list_v3.md 第三阶段规范

核心原则:
- 必须按 14 -> 19 -> 24 -> 29 顺序训练
- 一次只训练一层 (代码断言强制)
- 每层独立 checkpoint

执行流程:
1. 初始化 Layer 14 SAE
2. 训练 Layer 14 (8000 steps)
3. 保存 Layer 14 checkpoint
4. 初始化 Layer 19 SAE
5. 训练 Layer 19 (8000 steps)
6. ... (重复直到 Layer 29)

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from train_v2.config import TrainingConfig, HOOK_LAYERS, assert_single_layer
from train_v2.training_loop import TrainingLoop
from train_v2.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


# ============================================================================
# 层训练结果
# ============================================================================

@dataclass
class LayerTrainingResult:
    """单层训练结果"""
    layer_idx: int
    final_step: int
    is_converged: bool
    should_stop: bool
    stop_reason: str
    best_mse: float
    elapsed_seconds: float
    checkpoint_path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "final_step": self.final_step,
            "is_converged": self.is_converged,
            "should_stop": self.should_stop,
            "stop_reason": self.stop_reason,
            "best_mse": self.best_mse,
            "elapsed_seconds": self.elapsed_seconds,
            "checkpoint_path": self.checkpoint_path,
        }


@dataclass
class MultiLayerResult:
    """多层训练结果"""
    results: List[LayerTrainingResult] = field(default_factory=list)
    total_elapsed_seconds: float = 0.0
    all_converged: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "all_converged": self.all_converged,
        }


# ============================================================================
# Layer Trainer
# ============================================================================

class LayerTrainer:
    """
    Layer-wise 训练协调器

    严格按照 TODO 规范:
    - 顺序训练: 14 -> 19 -> 24 -> 29
    - 单层训练: 断言只包含一层

    使用示例:
        config = TrainingConfig()
        trainer = LayerTrainer(config, device="cuda")

        # 方式 1: 训练所有层
        results = trainer.train_all_layers(
            train_loaders=train_loaders_by_layer,
            val_loaders=val_loaders_by_layer,
        )

        # 方式 2: 训练单层
        result = trainer.train_single_layer(
            layer_idx=14,
            train_loader=train_loader_14,
            val_loader=val_loader_14,
        )
    """

    def __init__(
        self,
        config: TrainingConfig,
        device: str = "cuda",
    ):
        """
        初始化 Layer Trainer

        参数:
            config: 训练配置
            device: 设备
        """
        self.config = config
        self.device = device

        # 已训练层
        self._trained_layers: set = set()

        # 训练结果
        self._results: List[LayerTrainingResult] = []

    def train_all_layers(
        self,
        train_loaders: Dict[int, DataLoader],
        val_loaders: Optional[Dict[int, DataLoader]] = None,
        resume_from: Optional[Dict[int, str]] = None,
    ) -> MultiLayerResult:
        """
        训练所有层 (按顺序)

        参数:
            train_loaders: {layer_idx: DataLoader}
            val_loaders: {layer_idx: DataLoader}
            resume_from: {layer_idx: checkpoint_path}

        返回:
            MultiLayerResult
        """
        import time

        start_time = time.time()
        results = []

        # 验证配置
        layers_to_train = self.config.hook_layers.copy()
        self._validate_layer_order(layers_to_train)

        logger.info(f"Starting layer-wise training: {layers_to_train}")

        for layer_idx in layers_to_train:
            # 断言: 未训练过
            assert layer_idx not in self._trained_layers, \
                f"Layer {layer_idx} already trained"

            # 获取数据加载器
            train_loader = train_loaders.get(layer_idx)
            if train_loader is None:
                raise ValueError(f"No train_loader for layer {layer_idx}")

            val_loader = val_loaders.get(layer_idx) if val_loaders else None
            resume_path = resume_from.get(layer_idx) if resume_from else None

            # 训练单层
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Training Layer {layer_idx}")
            logger.info(f"{'=' * 60}")

            result = self.train_single_layer(
                layer_idx=layer_idx,
                train_loader=train_loader,
                val_loader=val_loader,
                resume_from=resume_path,
            )

            results.append(result)
            self._trained_layers.add(layer_idx)

            # 检查是否继续
            if result.should_stop and result.stop_reason == "early_stop_triggered":
                logger.warning(f"Early stop at layer {layer_idx}, but continuing to next layer")
                # 不中断，继续训练下一层

        elapsed = time.time() - start_time

        multi_result = MultiLayerResult(
            results=results,
            total_elapsed_seconds=elapsed,
            all_converged=all(r.is_converged for r in results),
        )

        logger.info(f"\n{'=' * 60}")
        logger.info("All layers completed!")
        logger.info(f"Total time: {elapsed / 3600:.2f} hours")
        logger.info(f"All converged: {multi_result.all_converged}")
        logger.info(f"{'=' * 60}")

        return multi_result

    def train_single_layer(
        self,
        layer_idx: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None,
    ) -> LayerTrainingResult:
        """
        训练单层

        参数:
            layer_idx: 层索引
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            resume_from: 恢复训练的 checkpoint 路径

        返回:
            LayerTrainingResult
        """
        import time

        # 创建单层配置
        layer_config = self._create_layer_config(layer_idx)

        # 断言: 只包含一层
        assert_single_layer(layer_config)

        # 创建训练循环
        loop = TrainingLoop(layer_config, device=self.device)

        # 运行训练
        result_dict = loop.run(
            layer_idx=layer_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            resume_from=resume_from,
        )

        # 获取 checkpoint 路径
        checkpoint_manager = CheckpointManager(layer_config)
        checkpoint_path = checkpoint_manager.get_latest_checkpoint(
            run_dir=layer_config.run_dir,
            layer_idx=layer_idx,
        )

        result = LayerTrainingResult(
            layer_idx=layer_idx,
            final_step=result_dict["final_step"],
            is_converged=result_dict["is_converged"],
            should_stop=result_dict["should_stop"],
            stop_reason=result_dict["stop_reason"],
            best_mse=result_dict["best_mse"],
            elapsed_seconds=result_dict["elapsed_seconds"],
            checkpoint_path=str(checkpoint_path) if checkpoint_path else "",
        )

        return result

    def _create_layer_config(self, layer_idx: int) -> TrainingConfig:
        """创建单层配置"""
        return TrainingConfig(
            d_model=self.config.d_model,
            d_hidden=self.config.d_hidden,
            top_k=self.config.top_k,
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            total_steps=self.config.total_steps,
            min_lr=self.config.min_lr,
            batch_size=self.config.batch_size,
            accum_steps=self.config.accum_steps,
            grad_clip=self.config.grad_clip,
            ema_decay=self.config.ema_decay,
            val_interval=self.config.val_interval,
            checkpoint_interval=self.config.checkpoint_interval,
            early_stop_patience=self.config.early_stop_patience,
            early_stop_min_delta=self.config.early_stop_min_delta,
            dead_neuron_stop_threshold=self.config.dead_neuron_stop_threshold,
            convergence_mse=self.config.convergence_mse,
            convergence_dead_ratio=self.config.convergence_dead_ratio,
            convergence_fm_increase=self.config.convergence_fm_increase,
            hook_mode=self.config.hook_mode,
            hook_layers=[layer_idx],  # 单层
            checkpoint_dir=self.config.checkpoint_dir,
            prompt_dir=self.config.prompt_dir,
            run_dir=self.config.run_dir,
            max_tokens_per_batch=self.config.max_tokens_per_batch,
            seed=self.config.seed,
            device=self.config.device,
            lambda_aux=self.config.lambda_aux,
            lambda_orth=self.config.lambda_orth,
        )

    def _validate_layer_order(self, layers: List[int]) -> None:
        """验证层顺序"""
        from train_v2.config import validate_layer_order

        if not validate_layer_order(layers):
            raise ValueError(
                f"Layer order must be 14 -> 19 -> 24 -> 29, got {layers}"
            )


# ============================================================================
# 工厂函数
# ============================================================================

def create_layer_trainer(config: TrainingConfig, device: str = "cuda") -> LayerTrainer:
    """创建 Layer Trainer"""
    return LayerTrainer(config, device)


# ============================================================================
# 便捷函数
# ============================================================================

def train_layer_14(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    run_dir: str = "sae_runs/train_v2",
    device: str = "cuda",
) -> LayerTrainingResult:
    """训练 Layer 14"""
    config = TrainingConfig(
        hook_layers=[14],
        run_dir=run_dir,
    )
    trainer = LayerTrainer(config, device)
    return trainer.train_single_layer(14, train_loader, val_loader)


def train_layer_19(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    run_dir: str = "sae_runs/train_v2",
    device: str = "cuda",
) -> LayerTrainingResult:
    """训练 Layer 19"""
    config = TrainingConfig(
        hook_layers=[19],
        run_dir=run_dir,
    )
    trainer = LayerTrainer(config, device)
    return trainer.train_single_layer(19, train_loader, val_loader)


def train_layer_24(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    run_dir: str = "sae_runs/train_v2",
    device: str = "cuda",
) -> LayerTrainingResult:
    """训练 Layer 24"""
    config = TrainingConfig(
        hook_layers=[24],
        run_dir=run_dir,
    )
    trainer = LayerTrainer(config, device)
    return trainer.train_single_layer(24, train_loader, val_loader)


def train_layer_29(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    run_dir: str = "sae_runs/train_v2",
    device: str = "cuda",
) -> LayerTrainingResult:
    """训练 Layer 29"""
    config = TrainingConfig(
        hook_layers=[29],
        run_dir=run_dir,
    )
    trainer = LayerTrainer(config, device)
    return trainer.train_single_layer(29, train_loader, val_loader)
