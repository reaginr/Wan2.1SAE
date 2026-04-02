"""
离线SAE训练脚本

从采集的激活值文件中训练SAE，无需运行DiT模型。

特点：
1. 支持从JSON/numpy格式的激活值文件训练
2. 支持多文件批量加载
3. 支持自定义SAE参数
4. 支持训练恢复
5. 支持多轮训练（同一数据多次遍历）

使用方法：
    python train_offline.py --data_dir offline_data/activations_run1 --run_dir sae_runs/offline_exp1

或者使用配置文件：
    python train_offline.py --config train_config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 修复导入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wan.modules.sae_new import SAEConfig, SparseAutoEncoder
from wan.sae.checkpoint_io import SAECheckpointIO
from wan.sae.logger import SAELogManager, get_offline_train_logger
from wan.sae.sae_run_naming import SAERunLocator, save_json

logger = logging.getLogger(__name__)


##########################################################################################
# 训练参数配置区域
##########################################################################################

# --------------------------- 数据配置 ---------------------------
data_params = {
    "data_dir": "offline_data/activations_run1",  # 激活值数据目录
    "manifest_file": "manifest.jsonl",  # 数据索引文件
    "layers": "15,29",  # 要训练的层（留空则自动检测）
}

# --------------------------- SAE架构配置 ---------------------------
sae_params = {
    "d_model": 1536,
    "d_hidden": 6144,
    "activation": "relu",
    "sparsity": "topk",
    "top_k": 64,
    "l1_lambda": 1e-3,
}

# --------------------------- 训练配置 ---------------------------
training_params = {
    "epochs": 10,  # 训练轮数
    "batch_size": 4096,  # 每批token数（不是prompt数）
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "lr_scheduler": "constant",  # "constant" | "cosine" | "step"
    "warmup_steps": 1000,
    "steps_per_epoch": 0,  # 0表示遍历全部数据

    # 梯度累积
    "gradient_accumulation_steps": 1,

    # 验证
    "validation_split": 0.05,  # 验证集比例
    "validate_every": 500,  # 每多少步验证一次
}

# --------------------------- Checkpoint配置 ---------------------------
checkpoint_params = {
    "run_dir": "sae_runs/offline_exp1",
    "save_every": 1000,  # 每多少步保存
    "save_best": True,  # 保存最佳模型
    "resume": False,  # 是否恢复训练
}

# --------------------------- 系统配置 ---------------------------
system_params = {
    "device_id": 0,
    "seed": 0,
    "num_workers": 4,  # 数据加载线程数
    "pin_memory": True,
}

# --------------------------- 日志配置 ---------------------------
log_params = {
    "log_interval": 100,
    "log_to_file": True,
}


##########################################################################################
# 核心代码区域
##########################################################################################

class ActivationDataset(Dataset):
    """
    激活值数据集

    从采集的数据目录加载激活值，支持按需加载
    """

    def __init__(
        self,
        data_dir: str,
        layer_key: str,
        manifest_file: str = "manifest.jsonl",
        validation: bool = False,
        validation_split: float = 0.0,
    ):
        self.data_dir = Path(data_dir)
        self.layer_key = layer_key
        self.validation = validation
        self.validation_split = validation_split

        # 加载manifest
        manifest_path = self.data_dir / manifest_file
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest文件不存在: {manifest_path}")

        self.records = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        # 划分训练/验证集
        total = len(self.records)
        val_size = int(total * validation_split)

        if validation:
            self.records = self.records[-val_size:] if val_size > 0 else []
        else:
            self.records = self.records[:-val_size] if val_size > 0 else self.records

        logger.info(f"数据集 [{layer_key}] {'验证' if validation else '训练'}集: {len(self.records)} 条记录")

        # 预计算总token数（可选，用于显示进度）
        self._total_tokens = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        """
        获取第idx条记录的所有激活值

        返回: (features, prompt)
            features: [num_timesteps, L, C] numpy array
        """
        record_meta = self.records[idx]
        record_path = self.data_dir / record_meta["record_path"]

        with open(record_path, "r", encoding="utf-8") as f:
            record = json.load(f)

        # 获取指定层的激活值
        if self.layer_key not in record["activations"]:
            raise KeyError(f"记录 {record_meta['id']} 缺少层 {self.layer_key}")

        act_info = record["activations"][self.layer_key]

        if act_info.get("format") in ["npy", "npz_compressed"]:
            # 从numpy文件加载
            npy_path = self.data_dir / act_info["file"]
            if act_info["format"] == "npz_compressed":
                data = np.load(npy_path)
                features = data["data"]
            else:
                features = np.load(npy_path)
        else:
            # 从JSON加载
            features = np.array(act_info["data"])

        return features, record["prompt"]

    def get_total_tokens(self) -> int:
        """计算总token数（用于显示进度）"""
        if self._total_tokens is not None:
            return self._total_tokens

        total = 0
        for record_meta in self.records:
            record_path = self.data_dir / record_meta["record_path"]
            with open(record_path, "r", encoding="utf-8") as f:
                record = json.load(f)

            if self.layer_key in record["activations"]:
                act_info = record["activations"][self.layer_key]
                if "shape" in act_info:
                    # [T, L, C]
                    shape = act_info["shape"]
                    total += shape[0] * shape[1]  # T * L tokens

        self._total_tokens = total
        return total


class TokenBatchSampler:
    """
    Token批次采样器

    从数据集中采样固定数量的token（而不是固定数量的记录）
    """

    def __init__(
        self,
        dataset: ActivationDataset,
        batch_size: int,  # token数
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __iter__(self) -> Iterator[List[int]]:
        """生成批次索引"""
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            self.rng.shuffle(indices)

        current_batch = []
        current_tokens = 0

        for idx in indices:
            # 获取该记录的大小（粗略估计）
            features, _ = self.dataset[idx]
            num_tokens = features.shape[0] * features.shape[1]  # T * L

            if current_tokens + num_tokens > self.batch_size and current_batch:
                yield current_batch
                current_batch = [idx]
                current_tokens = num_tokens
            else:
                current_batch.append(idx)
                current_tokens += num_tokens

        if current_batch:
            yield current_batch

    def __len__(self) -> int:
        """估计批次数量"""
        total_tokens = self.dataset.get_total_tokens()
        return max(1, total_tokens // self.batch_size)


def collate_tokens(batch: List[Tuple[np.ndarray, str]]) -> torch.Tensor:
    """
    合并批次数据为token张量

    输入: [(features, prompt), ...]
        features: [T, L, C]
    输出: [N, C] 张量，其中N是所有token的总数
    """
    all_tokens = []

    for features, _ in batch:
        # features: [T, L, C] -> [T*L, C]
        T, L, C = features.shape
        tokens = features.reshape(T * L, C)
        all_tokens.append(tokens)

    # 合并并转换为tensor
    stacked = np.concatenate(all_tokens, axis=0)
    return torch.from_numpy(stacked).float()


def train_sae_for_layer(
    layer_key: str,
    data_dir: str,
    run_dir: str,
    sae_config: SAEConfig,
    training_config: Dict[str, Any],
    device: torch.device,
    resume: bool = False,
) -> Dict[str, Any]:
    """
    为单个层训练SAE

    返回: 训练历史统计
    """
    # 解析层信息
    hook_mode, layer_str = layer_key.split(".")
    layer_idx = int(layer_str.replace("layer", ""))

    # 初始化统一日志管理器
    log_mgr = get_offline_train_logger(run_dir, hook_mode, layer_idx)
    log_mgr.log_event("train_start", f"开始训练层: {layer_key}", {
        "sae_config": sae_config.to_dict(),
        "training_config": training_config,
        "resume": resume,
    })
    log_mgr.log_config(sae_config.to_dict())

    logger.info(f"=" * 60)
    logger.info(f"开始训练层: {layer_key}")
    logger.info(f"SAE配置: d_model={sae_config.d_model}, d_hidden={sae_config.d_hidden}")
    logger.info(f"=" * 60)

    # 创建run定位器
    loc = SAERunLocator(run_dir=run_dir, hook_mode=hook_mode, layer_idx=layer_idx)
    loc.artifact_dir().mkdir(parents=True, exist_ok=True)

    # 保存SAE配置
    save_json(
        loc.config_path(),
        {
            "sae": sae_config.to_dict(),
            "hook": {"hook_mode": hook_mode, "layer_idx": layer_idx},
        },
    )

    # 创建数据集
    train_dataset = ActivationDataset(
        data_dir=data_dir,
        layer_key=layer_key,
        validation=False,
        validation_split=training_config["validation_split"],
    )

    val_dataset = None
    if training_config["validation_split"] > 0:
        val_dataset = ActivationDataset(
            data_dir=data_dir,
            layer_key=layer_key,
            validation=True,
            validation_split=training_config["validation_split"],
        )

    # 创建SAE
    sae = SparseAutoEncoder(sae_config).to(device)
    start_step = 0
    best_val_loss = float("inf")

    # 尝试恢复（使用新的统一 IO 接口，自动兼容新旧格式）
    if resume and loc.latest_ckpt_path().exists():
        logger.info(f"从checkpoint恢复: {loc.latest_ckpt_path()}")
        try:
            io = SAECheckpointIO.load(loc, device=device, strict=True, allow_legacy=True)
            sae = io.sae
            start_step = io.step
            best_val_loss = io.extra_info.get("best_val_loss", float("inf"))
            logger.info(f"恢复自 step={start_step}")
            if io._config_source == "json_fallback":
                logger.warning("  注意：从旧格式 .json 加载配置 [建议迁移]")
        except Exception as e:
            logger.error(f"恢复失败: {e}")
            raise

    # 创建优化器
    optimizer = torch.optim.AdamW(
        sae.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    # 学习率调度器
    if training_config["lr_scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config["epochs"] * 1000,  # 粗略估计
        )
    elif training_config["lr_scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_config["epochs"] // 3,
            gamma=0.5,
        )
    else:
        scheduler = None

    # 训练循环
    history = {
        "layer_key": layer_key,
        "train_losses": [],
        "val_losses": [],
        "steps": [],
    }

    global_step = start_step

    for epoch in range(training_config["epochs"]):
        logger.info(f"Epoch {epoch + 1}/{training_config['epochs']}")

        # 创建采样器
        sampler = TokenBatchSampler(
            train_dataset,
            batch_size=training_config["batch_size"],
            shuffle=True,
            seed=training_config.get("seed", 0) + epoch,
        )

        sae.train()
        epoch_losses = []
        accumulated_loss = 0

        for batch_idx, batch_indices in enumerate(sampler):
            # 加载数据
            batch_data = [train_dataset[i] for i in batch_indices]
            tokens = collate_tokens(batch_data).to(device)

            # 前向
            _, _, loss = sae(tokens, return_loss=True)

            # 梯度累积
            loss = loss / training_config["gradient_accumulation_steps"]
            loss.backward()

            accumulated_loss += loss.item()

            if (batch_idx + 1) % training_config["gradient_accumulation_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if scheduler:
                    scheduler.step()

                # 记录
                epoch_losses.append(accumulated_loss)
                history["train_losses"].append(accumulated_loss)
                history["steps"].append(global_step)

                # 日志
                if global_step % log_params["log_interval"] == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"  Step {global_step}: loss={accumulated_loss:.6f}, lr={lr:.6f}"
                    )

                # 使用统一日志管理器记录详细指标
                if global_step % log_params["loss_log_interval"] == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    # 计算稀疏度和激活数
                    with torch.no_grad():
                        # 用当前batch数据计算统计信息
                        z, _, _ = sae.encode(tokens)
                        sparsity = (z.abs() > 1e-6).float().mean().item()
                        num_activations = (z.abs() > 1e-6).sum(dim=-1).float().mean().item()

                    log_mgr.log_metric({
                        "loss": accumulated_loss,
                        "sparsity": sparsity,
                        "num_activations": num_activations,
                        "lr": lr,
                        "epoch": epoch,
                    }, step=global_step, layer_key=layer_key)

                # 验证
                if val_dataset and global_step % training_config["validate_every"] == 0:
                    val_loss = validate_sae(sae, val_dataset, device, training_config["batch_size"])
                    history["val_losses"].append(val_loss)
                    logger.info(f"  Validation loss: {val_loss:.6f}")
                    # 记录验证指标
                    log_mgr.log_metric({
                        "val_loss": val_loss,
                        "epoch": epoch,
                    }, step=global_step, layer_key=layer_key)

                    # 保存最佳模型
                    if training_config.get("save_best", True) and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        io = SAECheckpointIO(
                            sae=sae,
                            step=global_step,
                            hook_mode=hook_mode,
                            layer_idx=layer_idx,
                            extra_info={"val_loss": val_loss, "best_val_loss": best_val_loss},
                        )
                        io.save(loc, save_legacy_json=True)
                        logger.info(f"  保存最佳模型，val_loss={val_loss:.6f}")

                # 保存checkpoint（使用新格式，配置内置）
                if global_step % checkpoint_params["save_every"] == 0:
                    io = SAECheckpointIO(
                        sae=sae,
                        step=global_step,
                        hook_mode=hook_mode,
                        layer_idx=layer_idx,
                        extra_info={"epoch": epoch, "best_val_loss": best_val_loss},
                    )
                    io.save(loc, save_legacy_json=True)

                accumulated_loss = 0

        # Epoch结束
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        logger.info(f"Epoch {epoch + 1} 完成，平均loss: {avg_loss:.6f}")

    # 保存最终模型（使用新格式）
    io = SAECheckpointIO(
        sae=sae,
        step=global_step,
        hook_mode=hook_mode,
        layer_idx=layer_idx,
        extra_info={"best_val_loss": best_val_loss, "is_final": True},
    )
    io.save(loc, save_legacy_json=True)

    logger.info(f"层 {layer_key} 训练完成！")

    # 保存训练总结
    summary = {
        "layer_key": layer_key,
        "total_steps": global_step,
        "start_step": start_step,
        "epochs": training_config["epochs"],
        "final_train_loss": history["train_losses"][-1] if history["train_losses"] else None,
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
        "avg_train_loss": sum(history["train_losses"]) / len(history["train_losses"]) if history["train_losses"] else 0,
        "sae_config": sae_config.to_dict(),
    }
    log_mgr.save_summary(summary)
    log_mgr.log_event("train_complete", f"训练完成: {layer_key}", summary)

    return history


def validate_sae(
    sae: SparseAutoEncoder,
    val_dataset: ActivationDataset,
    device: torch.device,
    batch_size: int,
) -> float:
    """验证SAE"""
    sae.eval()
    total_loss = 0
    total_batches = 0

    sampler = TokenBatchSampler(val_dataset, batch_size=batch_size, shuffle=False, seed=0)

    with torch.no_grad():
        for batch_indices in sampler:
            batch_data = [val_dataset[i] for i in batch_indices]
            tokens = collate_tokens(batch_data).to(device)

            _, _, loss = sae(tokens, return_loss=True)
            total_loss += loss.item()
            total_batches += 1

    sae.train()
    return total_loss / total_batches if total_batches > 0 else float("inf")


def detect_layers(data_dir: str, manifest_file: str = "manifest.jsonl") -> List[str]:
    """
    自动检测数据集中包含哪些层
    """
    manifest_path = Path(data_dir) / manifest_file
    if not manifest_path.exists():
        return []

    # 读取第一条记录检测层
    with open(manifest_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line:
            return []

        record_meta = json.loads(first_line)
        record_path = Path(data_dir) / record_meta["record_path"]

        with open(record_path, "r", encoding="utf-8") as rf:
            record = json.load(rf)
            layers = list(record.get("activations", {}).keys())
            return layers

    return []


def main():
    parser = argparse.ArgumentParser(description="Train SAE offline from collected activations")
    parser.add_argument("--config", type=str, default="", help="JSON配置文件路径")
    parser.add_argument("--data_dir", type=str, default=data_params["data_dir"])
    parser.add_argument("--layers", type=str, default=data_params["layers"], help="逗号分隔的层索引，如'15,29'")
    parser.add_argument("--run_dir", type=str, default=checkpoint_params["run_dir"])
    parser.add_argument("--epochs", type=int, default=training_params["epochs"])
    parser.add_argument("--batch_size", type=int, default=training_params["batch_size"])
    parser.add_argument("--lr", type=float, default=training_params["learning_rate"])
    parser.add_argument("--d_hidden", type=int, default=sae_params["d_hidden"])
    parser.add_argument("--resume", action="store_true", default=checkpoint_params["resume"])
    parser.add_argument("--device_id", type=int, default=system_params["device_id"])
    parser.add_argument("--seed", type=int, default=system_params["seed"])

    args = parser.parse_args()

    # 加载配置文件（如果提供）
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        # 合并配置...

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # 设置设备
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 创建输出目录
    run_path = Path(args.run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    # 保存训练配置
    training_config = {
        "data_dir": args.data_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": training_params["weight_decay"],
        "lr_scheduler": training_params["lr_scheduler"],
        "validation_split": training_params["validation_split"],
        "gradient_accumulation_steps": training_params["gradient_accumulation_steps"],
        "seed": args.seed,
    }
    with open(run_path / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(training_config, f, ensure_ascii=False, indent=2)

    # 确定要训练的层
    if args.layers:
        layer_indices = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    else:
        # 自动检测
        detected = detect_layers(args.data_dir)
        layer_indices = []
        for key in detected:
            if ".layer" in key:
                idx = int(key.split(".layer")[1])
                layer_indices.append(idx)
        logger.info(f"自动检测到层: {layer_indices}")

    hook_mode = "block_out"  # 默认模式

    # SAE配置
    sae_config = SAEConfig(
        d_model=sae_params["d_model"],
        d_hidden=args.d_hidden,
        activation=sae_params["activation"],
        sparsity=sae_params["sparsity"],
        top_k=sae_params["top_k"],
        l1_lambda=sae_params["l1_lambda"],
    )

    # 为每个层训练SAE
    all_histories = {}

    for layer_idx in layer_indices:
        layer_key = f"{hook_mode}.layer{layer_idx}"

        history = train_sae_for_layer(
            layer_key=layer_key,
            data_dir=args.data_dir,
            run_dir=args.run_dir,
            sae_config=sae_config,
            training_config=training_config,
            device=device,
            resume=args.resume,
        )

        all_histories[layer_key] = history

    # 保存整体训练历史
    with open(run_path / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(all_histories, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("所有层训练完成！")
    logger.info(f"输出目录: {args.run_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
