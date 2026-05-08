"""
SAE 第二阶段 - 统一日志系统

根据 TODO list_v2 第六阶段要求实现：
- 基于Python logging模块
- 分INFO/DEBUG级别
- 按启动时间命名保存到 ./logs/
- 同时控制台输出
- 实时输出训练指标
- DEBUG模式详细输出

实时输出要求:
- 每10步: 当前步数、损失分解、归一化状态
- 每累积步: 平均损失、梯度范数、当前学习率
- 每验证周期: 全量验证指标、收敛状态
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import csv


@dataclass
class SAETrainingLoggerConfig:
    """训练日志配置"""

    # 日志目录
    log_dir: str = "./logs"

    # 日志级别
    console_level: str = "INFO"
    file_level: str = "DEBUG"

    # 输出控制
    log_to_console: bool = True
    log_to_file: bool = True

    # 验证周期
    validation_interval: int = 160  # 步
    log_every_n_steps: int = 10

    # DEBUG模式
    debug_mode: bool = False


class SAETrainingLogger:
    """
    SAE训练专用日志器

    功能:
    1. 分级日志 (INFO/DEBUG)
    2. 按时间自动命名日志文件
    3. 训练指标实时输出
    4. DEBUG模式详细统计
    5. 损失分解输出
    6. 归一化状态监控
    """

    def __init__(self, config: SAETrainingLoggerConfig, run_name: str = ""):
        self.config = config
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建日志目录
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 初始化logger
        self.logger = self._init_logger()

        # 训练状态
        self._step = 0
        self._epoch = 0
        self._accum_step = 0
        self._best_mse = float("inf")

        # 损失历史
        self._loss_history: List[Dict] = []
        self._validation_history: List[Dict] = []

        # CSV文件
        self._csv_file = None
        self._csv_writer = None

        self._init_csv()

        self.logger.info(f"训练日志器初始化: {self.log_dir}")

    def _init_logger(self) -> logging.Logger:
        """初始化Python logger"""
        logger_name = f"sae_train_{self.run_name}"
        logger = logging.getLogger(logger_name)

        # 清除已有handlers
        logger.handlers.clear()

        # 设置级别
        logger.setLevel(logging.DEBUG)

        # 格式
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 控制台handler
        if self.config.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.config.console_level))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # 文件handler
        if self.config.log_to_file:
            log_file = self.log_dir / f"{self.run_name}.log"
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(getattr(logging, self.config.file_level))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _init_csv(self):
        """初始化CSV文件"""
        csv_path = self.log_dir / f"{self.run_name}_metrics.csv"
        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)

        # 写入表头
        header = [
            "step", "timestamp",
            "loss_recon", "loss_auxk", "loss_orth", "loss_total",
            "grad_norm", "lr", "sparsity", "dead_ratio",
        ]
        self._csv_writer.writerow(header)
        self._csv_file.flush()

    def log_config(self, config: Dict[str, Any]):
        """记录配置"""
        self.logger.info("=" * 60)
        self.logger.info("训练配置:")
        for k, v in config.items():
            if isinstance(v, dict):
                self.logger.info(f"  {k}:")
                for k2, v2 in v.items():
                    self.logger.info(f"    {k2}: {v2}")
            else:
                self.logger.info(f"  {k}: {v}")
        self.logger.info("=" * 60)

        # 保存配置JSON
        config_file = self.log_dir / f"{self.run_name}_config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def log_step(
        self,
        step: int,
        loss_dict: Dict[str, float],
        grad_norm: Optional[float] = None,
        lr: Optional[float] = None,
        sparsity: Optional[float] = None,
        dead_ratio: Optional[float] = None,
        norm_status: Optional[Dict] = None,
    ):
        """
        记录训练步骤

        每10步输出: 当前步数、损失分解、归一化状态
        每累积步输出: 平均损失、梯度范数、当前学习率
        """
        self._step = step
        self._loss_history.append({
            "step": step,
            **loss_dict,
            "grad_norm": grad_norm,
            "lr": lr,
            "sparsity": sparsity,
            "dead_ratio": dead_ratio,
            "timestamp": time.time(),
        })

        # 每10步输出
        if step % self.config.log_every_n_steps == 0:
            self._log_step_summary(step, loss_dict, grad_norm, lr, sparsity, dead_ratio)

        # DEBUG模式: 详细输出
        if self.config.debug_mode and norm_status:
            self._log_norm_status(norm_status)

        # 写入CSV
        if self._csv_writer:
            row = [
                step, datetime.now().isoformat(),
                loss_dict.get("loss_recon", 0),
                loss_dict.get("loss_auxk", 0),
                loss_dict.get("loss_orth", 0),
                loss_dict.get("loss_total", 0),
                grad_norm or 0, lr or 0, sparsity or 0, dead_ratio or 0,
            ]
            self._csv_writer.writerow(row)
            self._csv_file.flush()

    def _log_step_summary(
        self,
        step: int,
        loss_dict: Dict[str, float],
        grad_norm: Optional[float],
        lr: Optional[float],
        sparsity: Optional[float],
        dead_ratio: Optional[float],
    ):
        """输出步骤摘要"""
        msg_parts = [f"Step {step}"]

        # 损失分解
        if "loss_recon" in loss_dict:
            msg_parts.append(f"recon={loss_dict['loss_recon']:.6f}")
        if "loss_auxk" in loss_dict:
            msg_parts.append(f"auxk={loss_dict['auxk']:.6f}")
        if "loss_orth" in loss_dict:
            msg_parts.append(f"orth={loss_dict['loss_orth']:.2e}")
        msg_parts.append(f"total={loss_dict.get('loss_total', 0):.6f}")

        # 其他指标
        if grad_norm is not None:
            msg_parts.append(f"grad={grad_norm:.4f}")
        if lr is not None:
            msg_parts.append(f"lr={lr:.2e}")
        if sparsity is not None:
            msg_parts.append(f"sparse={sparsity:.4f}")
        if dead_ratio is not None:
            msg_parts.append(f"dead={dead_ratio:.2%}")

        self.logger.info(" | ".join(msg_parts))

    def _log_norm_status(self, norm_status: Dict):
        """输出归一化状态 (DEBUG模式)"""
        self.logger.debug("  [Norm Status]")
        for k, v in norm_status.items():
            if isinstance(v, float):
                self.logger.debug(f"    {k}: {v:.6f}")
            elif isinstance(v, torch.Tensor):
                self.logger.debug(f"    {k}: mean={v.mean():.6f}, std={v.std():.6f}")
            else:
                self.logger.debug(f"    {k}: {v}")

    def log_accum_step(
        self,
        accum_step: int,
        avg_loss: float,
        grad_norm: float,
        lr: float,
    ):
        """记录累积步 (梯度累积完成后)"""
        self._accum_step = accum_step
        self.logger.info(
            f"[Accum {accum_step}] avg_loss={avg_loss:.6f} | "
            f"grad_norm={grad_norm:.4f} | lr={lr:.2e}"
        )

    def log_validation(
        self,
        step: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ):
        """
        记录验证结果

        每验证周期输出: 全量验证指标、收敛状态
        """
        self._validation_history.append({
            "step": step,
            **metrics,
            "timestamp": time.time(),
        })

        # 更新最佳MSE
        if "val_mse" in metrics:
            if metrics["val_mse"] < self._best_mse:
                self._best_mse = metrics["val_mse"]
                is_best = True

        self.logger.info("=" * 60)
        self.logger.info(f"Validation @ Step {step}")
        for k, v in metrics.items():
            if isinstance(v, float):
                self.logger.info(f"  {k}: {v:.6f}")
            else:
                self.logger.info(f"  {k}: {v}")
        if is_best:
            self.logger.info("  ★ New Best MSE!")
        self.logger.info("=" * 60)

        # 保存验证历史
        val_file = self.log_dir / f"{self.run_name}_validation.json"
        with open(val_file, "w", encoding="utf-8") as f:
            json.dump(self._validation_history, f, ensure_ascii=False, indent=2)

    def log_convergence_status(
        self,
        is_converged: bool,
        patience: int,
        improvement: float,
    ):
        """记录收敛状态"""
        status = "CONVERGED" if is_converged else "TRAINING"
        self.logger.info(f"[Convergence] status={status} | patience={patience} | improvement={improvement:.6f}")

    def log_event(self, event_type: str, message: str, extra: Optional[Dict] = None):
        """记录事件"""
        msg = f"[{event_type}] {message}"
        if extra:
            msg += f" | {extra}"
        self.logger.info(msg)

    def log_debug(self, message: str):
        """DEBUG级别日志"""
        self.logger.debug(message)

    def log_warning(self, message: str):
        """警告日志"""
        self.logger.warning(message)

    def log_error(self, message: str, exception: Optional[Exception] = None):
        """错误日志"""
        if exception:
            self.logger.error(f"{message}: {exception}", exc_info=True)
        else:
            self.logger.error(message)

    def log_metric_anomaly(
        self,
        metric_name: str,
        value: float,
        threshold: float,
        suggestion: str,
    ):
        """记录指标异常告警"""
        self.logger.warning(
            f"⚠️ [{metric_name}] ANOMALY: {value:.6f} > {threshold:.6f}\n"
            f"   建议: {suggestion}"
        )

    def close(self, summary: Optional[Dict] = None):
        """关闭日志器"""
        if summary:
            self.logger.info("=" * 60)
            self.logger.info("训练总结:")
            for k, v in summary.items():
                self.logger.info(f"  {k}: {v}")
            self.logger.info("=" * 60)

            # 保存总结
            summary_file = self.log_dir / f"{self.run_name}_summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        if self._csv_file:
            self._csv_file.close()


class SAEValidationLogger:
    """
    SAE验证专用日志器

    用于验证阶段的详细记录
    """

    def __init__(self, log_dir: str, run_name: str = ""):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

        self.logger = self._init_logger()

    def _init_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"sae_val_{self.run_name}")
        logger.setLevel(logging.DEBUG)

        if logger.handlers:
            return logger

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

        file = logging.FileHandler(
            self.log_dir / f"{self.run_name}_validation.log",
            mode="a",
            encoding="utf-8"
        )
        file.setLevel(logging.DEBUG)
        file.setFormatter(formatter)
        logger.addHandler(file)

        return logger

    def log_validation_step(
        self,
        step: int,
        batch_idx: int,
        metrics: Dict[str, float],
    ):
        """记录验证步骤"""
        self.logger.debug(
            f"Val Step {step} | Batch {batch_idx} | " +
            " | ".join(f"{k}={v:.6f}" for k, v in metrics.items())
        )

    def log_final_metrics(self, metrics: Dict[str, float]):
        """记录最终验证指标"""
        self.logger.info("=" * 60)
        self.logger.info("最终验证指标:")
        for k, v in metrics.items():
            self.logger.info(f"  {k}: {v:.6f}")
        self.logger.info("=" * 60)


# ============================================================================
# 便捷函数
# ============================================================================

def get_training_logger(
    log_dir: str = "./logs",
    run_name: str = "",
    debug: bool = False,
) -> SAETrainingLogger:
    """获取训练日志器"""
    config = SAETrainingLoggerConfig(
        log_dir=log_dir,
        debug_mode=debug,
    )
    return SAETrainingLogger(config, run_name)


def get_validation_logger(
    log_dir: str = "./logs",
    run_name: str = "",
) -> SAEValidationLogger:
    """获取验证日志器"""
    return SAEValidationLogger(log_dir, run_name)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SAETrainingLogger 测试")
    print("=" * 60)

    # 创建日志器
    logger = get_training_logger(
        log_dir="./test_logs",
        run_name="test_run",
        debug=True,
    )

    # 记录配置
    logger.log_config({
        "d_model": 1536,
        "d_hidden": 12288,
        "top_k": 128,
        "lr": 6e-5,
    })

    # 模拟训练步骤
    print("\n模拟训练...")
    for step in range(1, 101):
        loss = 0.5 - step * 0.001
        logger.log_step(
            step=step,
            loss_dict={
                "loss_recon": loss,
                "loss_auxk": loss * 0.1,
                "loss_orth": 1e-5,
                "loss_total": loss * 1.1,
            },
            grad_norm=0.1 + step * 0.001,
            lr=6e-5,
            sparsity=0.01,
            dead_ratio=0.05,
        )

        # 每40步验证
        if step % 40 == 0:
            logger.log_validation(
                step=step,
                metrics={
                    "val_mse": loss * 0.9,
                    "val_dead_ratio": 0.03,
                },
                is_best=(step == 40),
            )

    # 关闭
    logger.close(summary={
        "total_steps": 100,
        "best_mse": 0.35,
        "final_loss": 0.4,
    })

    print("\n测试完成! 检查 ./test_logs/ 目录")
    print("=" * 60)
