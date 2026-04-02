"""
统一的 SAE Checkpoint 读写接口

设计目标：
1. 配置内置到 .pt 文件中，确保权重和配置永不分离
2. 保留 .json 文件便于快速查看（但不依赖它）
3. 向后兼容旧版本（分开存储的 .pt + .json）
4. 提供统一的读写接口，所有模块通过此类操作 checkpoint

新格式 .pt 文件内容：
{
    "state_dict": {...},           # SAE 权重（必需）
    "step": int,                   # 训练步数
    "sae_config": {...},           # SAE 架构配置（必需，新版本）
    "hook_info": {...},            # hook 信息（hook_mode, layer_idx）
    "timestamp": float,            # 保存时间戳
    "version": "2.0",              # 格式版本
}

旧格式 .pt 文件内容（兼容读取）：
{
    "state_dict": {...},
    "step": int,
}
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from wan.modules.sae_new import SAEConfig, SparseAutoEncoder
from wan.sae.sae_run_naming import SAERunLocator, load_json, save_json

logger = logging.getLogger(__name__)

# Checkpoint 格式版本
CHECKPOINT_VERSION = "2.0"


class SAECheckpointIO:
    """
    统一的 SAE Checkpoint 读写接口

    使用示例：
        # 保存
        io = SAECheckpointIO(sae, step=500)
        io.save(loc)  # loc 是 SAERunLocator

        # 加载
        io = SAECheckpointIO.load(loc, device="cuda:0")
        sae = io.sae  # 获取加载好的 SAE
        step = io.step  # 获取训练步数
    """

    def __init__(
        self,
        sae: SparseAutoEncoder,
        step: int = 0,
        hook_mode: Optional[str] = None,
        layer_idx: Optional[int] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ):
        """
        创建 Checkpoint IO 对象

        参数:
            sae: SAE 模型实例
            step: 训练步数
            hook_mode: hook 模式（如 "block_out"）
            layer_idx: 层索引
            extra_info: 额外信息（如 loss、学习率等）
        """
        self.sae = sae
        self.step = step
        self.sae_config = sae.config
        self.hook_mode = hook_mode
        self.layer_idx = layer_idx
        self.extra_info = extra_info or {}
        self.timestamp = time.time()
        self.version = CHECKPOINT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """转换为保存字典（新格式）"""
        return {
            "state_dict": self.sae.state_dict(),
            "step": self.step,
            "sae_config": self.sae_config.to_dict(),
            "hook_info": {
                "hook_mode": self.hook_mode,
                "layer_idx": self.layer_idx,
            },
            "timestamp": self.timestamp,
            "version": self.version,
            "extra_info": self.extra_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: str = "cpu") -> "SAECheckpointIO":
        """从字典加载（支持新旧格式）"""
        version = data.get("version", "1.0")
        step = data.get("step", 0)
        state_dict = data["state_dict"]

        # 获取配置（新格式内置，旧格式需要外部提供）
        if "sae_config" in data:
            # 新格式：配置内置
            sae_config = SAEConfig(**data["sae_config"])
            config_source = "checkpoint"
        else:
            # 旧格式：配置不在 .pt 中，需要稍后设置
            # 先创建默认配置，然后尝试从 .json 加载
            sae_config = None  # 标记为未设置
            config_source = "unknown"

        # 获取 hook 信息
        hook_info = data.get("hook_info", {})
        hook_mode = hook_info.get("hook_mode")
        layer_idx = hook_info.get("layer_idx")

        # 创建 SAE（如果配置已知）
        if sae_config is not None:
            sae = SparseAutoEncoder(sae_config).to(device)
            sae.load_state_dict(state_dict)
            sae.eval()
        else:
            # 旧格式：先不创建 SAE，稍后从 .json 加载配置后再创建
            sae = None

        io = cls(
            sae=sae or SparseAutoEncoder(SAEConfig(d_model=1536, d_hidden=6144)),  # 临时占位
            step=step,
            hook_mode=hook_mode,
            layer_idx=layer_idx,
            extra_info=data.get("extra_info", {}),
        )
        io._state_dict = state_dict  # 保存权重供后续加载
        io._config_source = config_source
        io._version = version
        io._needs_config_from_json = sae_config is None

        return io

    def save(self, loc: SAERunLocator, save_legacy_json: bool = True) -> None:
        """
        保存 checkpoint

        参数:
            loc: SAERunLocator 定位器
            save_legacy_json: 是否同时保存 .json 文件（便于查看）
        """
        # 确保目录存在
        loc.artifact_dir().mkdir(parents=True, exist_ok=True)

        # 保存 .pt 文件（新格式，包含配置）
        ckpt_dict = self.to_dict()
        torch.save(ckpt_dict, loc.latest_ckpt_path())

        # 同时保存带步数的版本（历史记录）
        if self.step > 0:
            torch.save(ckpt_dict, loc.ckpt_path(self.step))

        # 可选：保存 .json 文件（便于快速查看）
        if save_legacy_json:
            save_json(
                loc.config_path(),
                {
                    "sae": self.sae_config.to_dict(),
                    "hook": {"hook_mode": self.hook_mode, "layer_idx": self.layer_idx},
                    "step": self.step,
                    "timestamp": self.timestamp,
                    "version": self.version,
                },
            )

        logger.debug("保存 checkpoint 到 %s (step=%d)", loc.latest_ckpt_path(), self.step)

    @classmethod
    def load(
        cls,
        loc: SAERunLocator,
        device: str = "cpu",
        strict: bool = True,
        allow_legacy: bool = True,
    ) -> "SAECheckpointIO":
        """
        加载 checkpoint（自动兼容新旧格式）

        参数:
            loc: SAERunLocator 定位器
            device: 目标设备
            strict: 是否严格匹配权重形状
            allow_legacy: 是否允许从旧格式（.json）加载配置

        返回:
            SAECheckpointIO 对象

        异常:
            FileNotFoundError: checkpoint 不存在
            RuntimeError: 配置不匹配且 strict=True
        """
        if not loc.latest_ckpt_path().exists():
            raise FileNotFoundError(f"Checkpoint 不存在: {loc.latest_ckpt_path()}")

        # 加载 .pt 文件
        ckpt_dict = torch.load(loc.latest_ckpt_path(), map_location=device)

        # 创建 IO 对象
        io = cls.from_dict(ckpt_dict, device=device)

        # 处理旧格式：从 .json 加载配置
        if io._needs_config_from_json and allow_legacy:
            if loc.config_path().exists():
                try:
                    json_data = load_json(loc.config_path())
                    saved_cfg = json_data.get("sae")
                    if saved_cfg:
                        sae_config = SAEConfig(**saved_cfg)
                        # 用正确配置重新创建 SAE
                        io.sae = SparseAutoEncoder(sae_config).to(device)
                        io.sae.load_state_dict(io._state_dict, strict=strict)
                        io.sae.eval()
                        io.sae_config = sae_config
                        io._config_source = "json_fallback"
                        logger.info(
                            "从旧格式 .json 加载配置: %s (d_hidden=%d, top_k=%d)",
                            loc.key(), sae_config.d_hidden, sae_config.top_k
                        )
                    else:
                        raise ValueError(f"{loc.config_path()} 中没有 'sae' 配置")
                except Exception as e:
                    raise RuntimeError(
                        f"无法从旧格式恢复 {loc.key()}：.pt 中没有配置且 .json 加载失败: {e}\n"
                        f"建议：手动创建匹配的 model_params 或使用转换脚本迁移 checkpoint"
                    ) from e
            else:
                raise RuntimeError(
                    f"无法恢复 {loc.key()}：.pt 是旧格式（无配置）且 .json 不存在\n"
                    f"原始错误：找不到配置文件 {loc.config_path()}\n"
                    f"建议：使用转换脚本迁移 checkpoint，或手动提供配置"
                )

        # 验证 hook 信息一致性
        if io.hook_mode is None:
            io.hook_mode = loc.hook_mode
        if io.layer_idx is None:
            io.layer_idx = loc.layer_idx

        return io

    def get_info_str(self) -> str:
        """获取 checkpoint 信息字符串"""
        lines = [
            f"Checkpoint Info:",
            f"  Version: {self.version}",
            f"  Step: {self.step}",
            f"  Hook: {self.hook_mode}.layer{self.layer_idx}",
            f"  SAE Config:",
            f"    d_model: {self.sae_config.d_model}",
            f"    d_hidden: {self.sae_config.d_hidden}",
            f"    sparsity: {self.sae_config.sparsity}",
            f"    top_k: {getattr(self.sae_config, 'top_k', 'N/A')}",
            f"  Timestamp: {self.timestamp}",
        ]
        return "\n".join(lines)


class CheckpointMigrator:
    """Checkpoint 迁移工具：将旧格式转换为新格式"""

    @staticmethod
    def migrate_file(pt_path: Path, json_path: Optional[Path] = None, backup: bool = True) -> bool:
        """
        迁移单个 checkpoint 文件

        参数:
            pt_path: .pt 文件路径
            json_path: 对应的 .json 文件路径（可选，默认同名）
            backup: 是否备份原文件

        返回:
            是否成功
        """
        if json_path is None:
            json_path = pt_path.parent / "sae_config.json"

        if not pt_path.exists():
            logger.error("PT 文件不存在: %s", pt_path)
            return False

        if not json_path.exists():
            logger.error("JSON 文件不存在: %s", json_path)
            logger.error("无法迁移：缺少配置信息")
            return False

        try:
            # 加载旧格式数据
            ckpt_old = torch.load(pt_path, map_location="cpu")
            json_data = load_json(json_path)

            # 提取信息
            state_dict = ckpt_old["state_dict"]
            step = ckpt_old.get("step", 0)
            sae_config = SAEConfig(**json_data["sae"])
            hook_info = json_data.get("hook", {})

            # 创建新格式数据
            ckpt_new = {
                "state_dict": state_dict,
                "step": step,
                "sae_config": sae_config.to_dict(),
                "hook_info": hook_info,
                "timestamp": time.time(),
                "version": CHECKPOINT_VERSION,
                "extra_info": {},
            }

            # 备份原文件
            if backup:
                backup_path = pt_path.with_suffix(".pt.v1.backup")
                pt_path.rename(backup_path)
                logger.info("备份原文件: %s", backup_path)

            # 保存新格式
            torch.save(ckpt_new, pt_path)
            logger.info("迁移成功: %s (step=%d, d_hidden=%d)", pt_path, step, sae_config.d_hidden)

            return True

        except Exception as e:
            logger.error("迁移失败: %s", e)
            return False

    @staticmethod
    def migrate_directory(run_dir: str, dry_run: bool = False) -> Tuple[int, int]:
        """
        迁移整个实验目录下的所有 checkpoint

        参数:
            run_dir: 实验目录路径
            dry_run: 是否仅预览，不实际执行

        返回:
            (成功数, 失败数)
        """
        run_path = Path(run_dir)
        if not run_path.exists():
            logger.error("目录不存在: %s", run_dir)
            return 0, 0

        success_count = 0
        fail_count = 0

        # 查找所有 layer 子目录
        for layer_dir in run_path.iterdir():
            if not layer_dir.is_dir():
                continue

            pt_path = layer_dir / "sae_latest.pt"
            json_path = layer_dir / "sae_config.json"

            if not pt_path.exists():
                continue

            # 检查是否已经是新格式
            try:
                ckpt = torch.load(pt_path, map_location="cpu")
                if "sae_config" in ckpt:
                    logger.info("已为新格式，跳过: %s", pt_path)
                    continue
            except Exception:
                pass

            if dry_run:
                logger.info("[预览] 将迁移: %s", pt_path)
                success_count += 1
                continue

            # 执行迁移
            if CheckpointMigrator.migrate_file(pt_path, json_path, backup=True):
                success_count += 1
            else:
                fail_count += 1

        return success_count, fail_count


# 便捷函数
def save_checkpoint(
    sae: SparseAutoEncoder,
    loc: SAERunLocator,
    step: int = 0,
    save_legacy_json: bool = True,
) -> None:
    """便捷函数：保存 checkpoint"""
    io = SAECheckpointIO(
        sae=sae,
        step=step,
        hook_mode=loc.hook_mode,
        layer_idx=loc.layer_idx,
    )
    io.save(loc, save_legacy_json=save_legacy_json)


def load_checkpoint(
    loc: SAERunLocator,
    device: str = "cpu",
    strict: bool = True,
) -> Tuple[SparseAutoEncoder, int, SAEConfig]:
    """
    便捷函数：加载 checkpoint

    返回:
        (sae, step, config)
    """
    io = SAECheckpointIO.load(loc, device=device, strict=strict)
    return io.sae, io.step, io.sae_config
