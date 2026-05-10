"""
Training Loop

严格按照 TODO_list_v3.md 3.2, 3.4 规范

执行顺序:
1. optimizer.zero_grad() before accumulation
2. Loop 8 steps: accumulate
3. clip_grad_norm_ after accumulation
4. optimizer.step()
5. scheduler.step()
6. ema.update()
7. decoder weight normalization

验证:
- 每 160 步 (20 update cycles)
- 使用 EMA 权重

早停:
- MSE decrease < 0.001 for 5 validation cycles
- Dead neuron > 20% → immediate stop

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from train_v2.config import TrainingConfig
from train_v2.checkpoint import CheckpointManager
from train_v2.dead_neuron_monitor import DeadNeuronMonitor
from train_v2.ema import EMAManager
from train_v2.gradient_accumulator import GradientAccumulator
from train_v2.optimizer import create_optimizer_and_scheduler
from train_v2.sae_engine import SAEEngine
from train_v2.validator import TrainingValidator, ValidationMetrics

# 延迟导入 wan 模块
try:
    from wan.modules.sae_new import SparseAutoEncoder, SAEConfig
    WAN_AVAILABLE = True
except ImportError:
    SparseAutoEncoder = None
    SAEConfig = None
    WAN_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# 训练状态
# ============================================================================

@dataclass
class TrainingState:
    """训练状态"""
    step: int = 0
    epoch: int = 0
    best_mse: float = float("inf")
    no_improve_count: int = 0
    is_converged: bool = False
    should_stop: bool = False
    stop_reason: str = ""

    # 时间统计
    start_time: float = 0.0
    last_checkpoint_time: float = 0.0

    # Loss 历史
    loss_history: List[float] = field(default_factory=list)


# ============================================================================
# 训练循环
# ============================================================================

class TrainingLoop:
    """
    主训练循环

    严格按照 TODO 3.2, 3.4 规范:
    - 梯度累积顺序
    - 验证间隔
    - 早停检测

    使用示例:
        config = TrainingConfig()
        loop = TrainingLoop(config, device="cuda")

        # 单层训练
        result = loop.run(
            layer_idx=14,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
        )
    """

    def __init__(
        self,
        config: TrainingConfig,
        device: str = "cuda",
    ):
        """
        初始化训练循环

        参数:
            config: 训练配置
            device: 设备
        """
        self.config = config
        self.device = device

        # 组件 (延迟初始化)
        self.engine: Optional[SAEEngine] = None
        self.optimizer: Optional[Adam] = None
        self.scheduler: Optional[LambdaLR] = None
        self.ema: Optional[EMAManager] = None
        self.accumulator: Optional[GradientAccumulator] = None
        self.validator: Optional[TrainingValidator] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None

        # 状态
        self.state = TrainingState()

    def run(
        self,
        layer_idx: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None,
        init_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        运行训练循环

        参数:
            layer_idx: 层索引 (单层)
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            resume_from: 恢复训练的 checkpoint 路径
            init_checkpoint: 初始化阶段的 SAE 权重路径 (可选)

        返回:
            训练结果字典
        """
        # ===== 初始化 =====
        self._init_components(layer_idx, init_checkpoint=init_checkpoint)

        # 恢复训练
        if resume_from is not None:
            self._resume_from_checkpoint(resume_from)

        # 记录开始时间
        self.state.start_time = time.time()

        logger.info(f"Starting training for layer {layer_idx}")
        logger.info(f"Config: steps={self.config.total_steps}, batch={self.config.batch_size}, accum={self.config.accum_steps}")

        # ===== 训练循环 =====
        while self.state.step < self.config.total_steps and not self.state.should_stop:
            # 执行一个累积周期
            avg_loss = self._train_accumulation_cycle(train_loader)

            # 更新步数
            self.state.step += 1
            self.state.loss_history.append(avg_loss)

            # ===== 验证 =====
            if self.state.step % self.config.val_interval == 0:
                if val_loader is not None:
                    metrics = self._validate(val_loader)
                    self._check_early_stop(metrics)

                    # 日志
                    self._log_validation(metrics)

            # ===== 保存 Checkpoint =====
            if self.state.step % self.config.checkpoint_interval == 0:
                self._save_checkpoint(layer_idx)

            # ===== 进度日志 =====
            if self.state.step % 10 == 0:
                self._log_progress(avg_loss)

        # ===== 训练结束 =====
        self.state.stop_reason = self.state.stop_reason or "max_steps_reached"

        # 最终保存
        self._save_checkpoint(layer_idx, is_final=True)

        return self._get_result()

    def _init_components(self, layer_idx: int, init_checkpoint: Optional[str] = None) -> None:
        """初始化所有组件"""
        # 创建 SAE
        sae_config = SAEConfig(
            d_model=self.config.d_model,
            d_hidden=self.config.d_hidden,
            activation="relu",
            sparsity="topk",
            top_k=self.config.top_k,
        )
        sae = SparseAutoEncoder(sae_config).to(self.device)

        # 创建 Engine
        self.engine = SAEEngine(sae, self.config)

        # 加载初始化权重 (如果提供)
        if init_checkpoint is not None:
            self._load_init_checkpoint(init_checkpoint)

        # 创建 Optimizer 和 Scheduler
        self.optimizer, self.scheduler = create_optimizer_and_scheduler(
            self.engine.sae, self.config
        )

        # 创建 EMA
        self.ema = EMAManager(self.engine.sae, decay=self.config.ema_decay)

        # 创建梯度累积器
        self.accumulator = GradientAccumulator(
            accum_steps=self.config.accum_steps,
            grad_clip=self.config.grad_clip,
        )

        # 创建验证器
        self.validator = TrainingValidator(self.engine, self.config)

        # 创建 Checkpoint 管理器
        self.checkpoint_manager = CheckpointManager(self.config)

        # 重置状态
        self.state = TrainingState()

        logger.debug(f"Initialized components for layer {layer_idx}")

    def _train_accumulation_cycle(self, train_loader: DataLoader) -> float:
        """
        执行一个梯度累积周期

        执行顺序:
        1. zero_grad() BEFORE loop
        2. Loop accum_steps: forward, loss/accum, backward
        3. clip_grad_norm_ AFTER loop
        4. optimizer.step()
        5. scheduler.step()
        6. ema.update()
        7. decoder normalization

        返回:
            平均 loss
        """
        # Step 1: 准备累积 (zero_grad BEFORE loop)
        self.accumulator.prepare_accumulation(self.optimizer)

        # Step 2: 累积循环
        for accum_step in range(self.config.accum_steps):
            # 获取 batch
            try:
                batch = next(self._train_iter)
            except (StopIteration, AttributeError):
                self._train_iter = iter(train_loader)
                batch = next(self._train_iter)

            # 移动到设备
            x = batch.to(self.device) if isinstance(batch, torch.Tensor) else batch["activations"].to(self.device)

            # Forward + Loss
            loss, info = self.engine.forward(x, return_info=True)

            # 累积 (loss / accum_steps)
            self.accumulator.accumulate_step(loss, self.engine.sae)

        # Step 3-6: 完成累积
        avg_loss = self.accumulator.finalize_accumulation(
            self.optimizer, self.scheduler, self.engine.sae
        )

        # Step 7: EMA 更新
        self.ema.update(self.engine.sae)

        # Step 8: Decoder 列归一化
        self.engine.normalize_decoder_columns()

        return avg_loss

    def _validate(self, val_loader: DataLoader) -> ValidationMetrics:
        """执行验证"""
        # 收集验证数据
        val_activations = []
        for batch in val_loader:
            x = batch.to(self.device) if isinstance(batch, torch.Tensor) else batch["activations"].to(self.device)
            val_activations.append(x)
            if len(val_activations) * x.shape[0] >= 2000:  # 足够样本
                break

        val_activations = torch.cat(val_activations, dim=0)

        # 验证 (使用 EMA 权重)
        metrics = self.validator.validate(val_activations, ema=self.ema)

        return metrics

    def _check_early_stop(self, metrics: ValidationMetrics) -> None:
        """检查早停条件"""
        if metrics.should_early_stop:
            self.state.should_stop = True
            self.state.stop_reason = "early_stop_triggered"
            logger.warning(f"Early stop triggered: dead_ratio={metrics.dead_neuron_ratio:.4f}")

        if metrics.is_converged:
            self.state.is_converged = True
            logger.info("Training converged!")

    def _save_checkpoint(self, layer_idx: int, is_final: bool = False) -> None:
        """保存 checkpoint"""
        extra_info = {
            "is_final": is_final,
            "is_converged": self.state.is_converged,
            "loss_history": self.state.loss_history[-100:],  # 最近 100 个
        }

        self.checkpoint_manager.save(
            run_dir=self.config.run_dir,
            layer_idx=layer_idx,
            step=self.state.step,
            engine=self.engine,
            ema=self.ema,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            extra_info=extra_info,
        )

    def _load_init_checkpoint(self, init_checkpoint: str) -> None:
        """
        加载初始化阶段的 SAE 权重

        支持格式:
        - sae_mixed_init.py 输出: {"Wenc": ..., "Wdec": ..., "bpre": ...}
        - TopKSAE.save_pretrained 输出: {"state_dict": ..., "config": ...}
        - 标准 state_dict
        """
        import os

        if not os.path.exists(init_checkpoint):
            logger.warning(f"Init checkpoint not found: {init_checkpoint}, using random init")
            return

        ckpt = torch.load(init_checkpoint, map_location=self.device)

        # 判断格式并加载
        if "Wenc" in ckpt and "Wdec" in ckpt:
            # sae_mixed_init.py 格式
            Wenc = ckpt["Wenc"].to(self.device).float()
            Wdec = ckpt["Wdec"].to(self.device).float()

            self.engine.sae.encoder.weight.data = Wenc
            self.engine.sae.decoder.weight.data = Wdec

            logger.info(f"Loaded init weights (mixed_init format): {init_checkpoint}")

        elif "state_dict" in ckpt:
            # TopKSAE.save_pretrained 格式
            state_dict = ckpt["state_dict"]
            new_state_dict = {}

            if "encoder.weight" in state_dict:
                new_state_dict["encoder.weight"] = state_dict["encoder.weight"].float()
            if "decoder.weight" in state_dict:
                new_state_dict["decoder.weight"] = state_dict["decoder.weight"].float()

            self.engine.sae.load_state_dict(new_state_dict, strict=False)
            logger.info(f"Loaded init weights (TopKSAE format): {init_checkpoint}")

        else:
            # 尝试直接加载为 state_dict
            try:
                self.engine.sae.load_state_dict(ckpt, strict=False)
                logger.info(f"Loaded init weights (standard format): {init_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to load init checkpoint: {e}, using random init")

    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """从 checkpoint 恢复"""
        from train_v2.checkpoint import CheckpointData

        data = self.checkpoint_manager.load(checkpoint_path, device=self.device)
        self.checkpoint_manager.restore(
            data, self.engine, self.ema, self.optimizer, self.scheduler
        )

        self.state.step = data.step

        logger.info(f"Resumed from step {data.step}")

    def _log_progress(self, avg_loss: float) -> None:
        """记录进度"""
        elapsed = time.time() - self.state.start_time
        steps_per_sec = self.state.step / elapsed if elapsed > 0 else 0
        eta = (self.config.total_steps - self.state.step) / steps_per_sec if steps_per_sec > 0 else 0

        lr = self.optimizer.param_groups[0]["lr"]

        logger.info(
            f"[{self.state.step}/{self.config.total_steps}] "
            f"loss={avg_loss:.6f} lr={lr:.2e} "
            f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m"
        )

    def _log_validation(self, metrics: ValidationMetrics) -> None:
        """记录验证结果"""
        logger.info(
            f"Validation @ step {self.state.step}: "
            f"mse={metrics.mse_normalized:.6f} "
            f"dead={metrics.dead_neuron_ratio:.4f} "
            f"gini={metrics.gini:.4f} "
            f"coherence={metrics.mutual_coherence:.4f}"
        )

    def _get_result(self) -> Dict[str, Any]:
        """获取训练结果"""
        elapsed = time.time() - self.state.start_time

        return {
            "final_step": self.state.step,
            "is_converged": self.state.is_converged,
            "should_stop": self.state.should_stop,
            "stop_reason": self.state.stop_reason,
            "best_mse": self.state.best_mse,
            "elapsed_seconds": elapsed,
            "final_loss": self.state.loss_history[-1] if self.state.loss_history else None,
        }


# ============================================================================
# 工厂函数
# ============================================================================

def create_training_loop(config: TrainingConfig, device: str = "cuda") -> TrainingLoop:
    """创建训练循环"""
    return TrainingLoop(config, device)
