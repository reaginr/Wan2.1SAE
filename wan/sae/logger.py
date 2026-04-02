"""
统一的SAE日志管理系统

设计原则：
1. 统一接口：所有模块通过SAELogger写入日志
2. 分类存储：不同模块、不同层、不同实验的日志分开存放
3. 多种格式：支持JSONL（结构化）、CSV（表格化）、TXT（可读）
4. 自动管理：按时间、按step自动分割文件
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import csv


class SAELogManager:
    """
    SAE统一日志管理器

    使用示例:
        # 训练模块
        logger = SAELogManager("sae_runs/exp1", "train", hook_mode="block_out", layer_idx=15)
        logger.log_metric({"loss": 0.5, "sparsity": 0.12}, step=100)

        # 测试模块
        logger = SAELogManager("sae_runs/exp1", "test", hook_mode="block_out", layer_idx=15)
        logger.log_result({"prompt": "...", "z_mean": [...], "loss": 0.3})
    """

    def __init__(
        self,
        run_dir: str,
        module_type: str,  # "train", "test", "offline_train", "offline_test", "steering", "analysis"
        hook_mode: Optional[str] = None,
        layer_idx: Optional[int] = None,
        log_to_console: bool = True,
        log_to_file: bool = True,
    ):
        """
        初始化日志管理器

        参数:
            run_dir: 实验根目录
            module_type: 模块类型，决定日志存放位置
            hook_mode: hook模式（如 block_out）
            layer_idx: 层索引
            log_to_console: 是否输出到控制台
            log_to_file: 是否保存到文件
        """
        self.run_dir = Path(run_dir)
        self.module_type = module_type
        self.hook_mode = hook_mode
        self.layer_idx = layer_idx
        self.log_to_file = log_to_file

        # 创建日志目录
        self.log_dir = self._get_log_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 初始化Python logger
        self.logger = self._init_logger(log_to_console)

        # 文件句柄缓存
        self._jsonl_files: Dict[str, Any] = {}
        self._csv_writers: Dict[str, Any] = {}
        self._csv_files: Dict[str, Any] = {}

        self.logger.info(f"日志管理器初始化: {self.log_dir}")

    def _get_log_dir(self) -> Path:
        """根据模块类型确定日志目录"""
        if self.module_type == "train":
            # 训练日志: sae_runs/exp1/logs/training/
            return self.run_dir / "logs" / "training"
        elif self.module_type == "test":
            # 测试日志: sae_runs/exp1/logs/testing/
            return self.run_dir / "logs" / "testing"
        elif self.module_type == "offline_train":
            # 离线训练: sae_runs/exp1/logs/offline_training/
            return self.run_dir / "logs" / "offline_training"
        elif self.module_type == "offline_test":
            # 离线测试: sae_runs/exp1/logs/offline_testing/
            return self.run_dir / "logs" / "offline_testing"
        elif self.module_type == "steering":
            # 干预生成: sae_runs/exp1/logs/steering/
            return self.run_dir / "logs" / "steering"
        elif self.module_type == "analysis":
            # 解释性分析: sae_runs/exp1/logs/analysis/
            return self.run_dir / "logs" / "analysis"
        else:
            return self.run_dir / "logs" / self.module_type

    def _init_logger(self, log_to_console: bool) -> logging.Logger:
        """初始化Python logger"""
        # 创建唯一的logger名称
        logger_name = f"sae_{self.module_type}"
        if self.hook_mode and self.layer_idx is not None:
            logger_name += f"_{self.hook_mode}_layer{self.layer_idx}"

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        # 避免重复添加handler
        if logger.handlers:
            return logger

        # 格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 控制台输出
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # 文件输出
        if self.log_to_file:
            log_file = self.log_dir / "run.log"
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def log_metric(
        self,
        metrics: Dict[str, Any],
        step: int,
        layer_key: Optional[str] = None,
    ):
        """
        记录训练/测试指标

        参数:
            metrics: 指标字典，如 {"loss": 0.5, "sparsity": 0.12}
            step: 训练/测试步数
            layer_key: 层标识，如 "block_out.layer15"
        """
        # 添加元信息
        record = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "step": step,
            "layer_key": layer_key or f"{self.hook_mode}.layer{self.layer_idx}",
            **metrics,
        }

        # 输出到控制台
        self.logger.info(f"Step {step} | " + " | ".join([f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]))

        # 保存到JSONL
        if self.log_to_file:
            self._append_jsonl("metrics.jsonl", record)

            # 同时保存到CSV（扁平化格式）
            self._append_csv("metrics.csv", record)

    def log_result(
        self,
        result: Dict[str, Any],
        result_id: Optional[str] = None,
    ):
        """
        记录测试结果或分析结果

        参数:
            result: 结果字典，包含prompt、z_mean、loss等
            result_id: 结果标识（如prompt索引）
        """
        # 添加元信息
        record = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "result_id": result_id,
            "module_type": self.module_type,
            **result,
        }

        # 输出到控制台（简要信息）
        prompt = result.get("prompt", "")[:50] if "prompt" in result else ""
        loss = result.get("loss", "N/A")
        self.logger.info(f"Result {result_id} | loss={loss} | prompt={prompt}...")

        # 保存到JSONL
        if self.log_to_file:
            self._append_jsonl("results.jsonl", record)

            # 保存到CSV（如果结果是扁平化的）
            try:
                self._append_csv("results.csv", record)
            except Exception:
                # 如果结果包含嵌套结构，只保存JSONL
                pass

    def log_config(self, config: Dict[str, Any]):
        """记录配置信息"""
        config_record = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "config": config,
        }

        self.logger.info(f"Config saved: {list(config.keys())}")

        if self.log_to_file:
            config_file = self.log_dir / "config.json"
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_record, f, ensure_ascii=False, indent=2)

    def log_event(self, event_type: str, message: str, extra: Optional[Dict] = None):
        """记录事件（如开始训练、恢复训练等）"""
        record = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "extra": extra or {},
        }

        self.logger.info(f"[{event_type}] {message}")

        if self.log_to_file:
            self._append_jsonl("events.jsonl", record)

    def save_summary(self, summary: Dict[str, Any]):
        """保存总结报告（训练结束或测试结束时）"""
        summary_file = self.log_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Summary saved: {summary_file}")

    def _append_jsonl(self, filename: str, record: Dict):
        """追加记录到JSONL文件"""
        filepath = self.log_dir / filename
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _append_csv(self, filename: str, record: Dict):
        """追加记录到CSV文件"""
        filepath = self.log_dir / filename

        # 展平嵌套字典
        flat_record = self._flatten_dict(record)

        # 检查文件是否存在
        file_exists = filepath.exists()

        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=flat_record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(flat_record)

    def _flatten_dict(self, d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
        """展平嵌套字典"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def get_log_path(self, filename: str) -> Path:
        """获取日志文件完整路径"""
        return self.log_dir / filename

    def close(self):
        """关闭所有文件句柄"""
        for f in self._jsonl_files.values():
            f.close()
        for f in self._csv_files.values():
            f.close()


# 便捷函数
def get_train_logger(run_dir: str, hook_mode: str, layer_idx: int) -> SAELogManager:
    """获取训练日志管理器"""
    return SAELogManager(run_dir, "train", hook_mode=hook_mode, layer_idx=layer_idx)


def get_test_logger(run_dir: str, hook_mode: Optional[str] = None, layer_idx: Optional[int] = None) -> SAELogManager:
    """获取测试日志管理器"""
    return SAELogManager(run_dir, "test", hook_mode=hook_mode, layer_idx=layer_idx)


def get_offline_train_logger(run_dir: str, hook_mode: str, layer_idx: int) -> SAELogManager:
    """获取离线训练日志管理器"""
    return SAELogManager(run_dir, "offline_train", hook_mode=hook_mode, layer_idx=layer_idx)


def get_offline_test_logger(run_dir: str) -> SAELogManager:
    """获取离线测试日志管理器"""
    return SAELogManager(run_dir, "offline_test")


def get_steering_logger(run_dir: str) -> SAELogManager:
    """获取干预生成日志管理器"""
    return SAELogManager(run_dir, "steering")
