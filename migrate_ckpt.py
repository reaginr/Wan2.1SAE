#!/usr/bin/env python3
"""
SAE Checkpoint 迁移脚本（独立运行，无需 GPU）

功能：将旧格式 checkpoint（.pt + .json 分开存储）转换为新格式（配置内置到 .pt）

旧格式：
    sae_latest.pt: {state_dict, step}
    sae_config.json: {sae: {...}, hook: {...}}

新格式：
    sae_latest.pt: {state_dict, step, sae_config, hook_info, version, timestamp}

用法：
    # 迁移单个文件
    python migrate_ckpt.py --pt_path sae_runs/exp1/block_out.layer15/sae_latest.pt

    # 迁移整个目录（自动查找所有 layer）
    python migrate_ckpt.py --run_dir sae_runs/exp1

    # 仅预览，不实际执行
    python migrate_ckpt.py --run_dir sae_runs/exp1 --dry_run

    # 不备份原文件
    python migrate_ckpt.py --run_dir sae_runs/exp1 --no_backup
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# 只依赖 torch，不依赖 wan 模块
try:
    import torch
except ImportError:
    print("错误：需要安装 torch")
    print("pip install torch")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = "2.0"


def load_json(path: Path) -> Dict[str, Any]:
    """加载 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """保存 JSON 文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def expand_path(path_str: str) -> Path:
    """展开 ~ 和环境变量"""
    import os
    path_str = os.path.expandvars(path_str)
    path_str = os.path.expanduser(path_str)
    return Path(path_str).resolve()


def is_new_format(ckpt: Dict[str, Any]) -> bool:
    """检查 checkpoint 是否已经是新格式"""
    return "sae_config" in ckpt and "version" in ckpt


def migrate_file(
    pt_path: Path,
    json_path: Optional[Path] = None,
    backup: bool = True,
    dry_run: bool = False,
) -> Tuple[bool, str]:
    """
    迁移单个 checkpoint 文件

    返回: (是否成功, 消息)
    """
    if json_path is None:
        json_path = pt_path.parent / "sae_config.json"

    # 检查文件存在性
    if not pt_path.exists():
        return False, f"PT 文件不存在: {pt_path}"

    if not json_path.exists():
        return False, f"JSON 文件不存在: {json_path}"

    try:
        # 加载旧格式数据
        logger.info("加载: %s", pt_path)
        ckpt_old = torch.load(pt_path, map_location="cpu")

        # 检查是否已经是新格式
        if is_new_format(ckpt_old):
            return True, "已经是新格式，无需迁移"

        logger.info("检测到旧格式，加载配置: %s", json_path)
        json_data = load_json(json_path)

        # 提取信息
        if "state_dict" not in ckpt_old:
            return False, "PT 文件中缺少 state_dict"

        state_dict = ckpt_old["state_dict"]
        step = ckpt_old.get("step", 0)

        # 从 JSON 提取配置
        if "sae" not in json_data:
            return False, "JSON 文件中缺少 'sae' 配置"

        sae_config = json_data["sae"]
        hook_info = json_data.get("hook", {})

        # 创建新格式数据
        ckpt_new = {
            "state_dict": state_dict,
            "step": step,
            "sae_config": sae_config,
            "hook_info": hook_info,
            "timestamp": time.time(),
            "version": CHECKPOINT_VERSION,
            "extra_info": ckpt_old.get("extra_info", {}),
        }

        if dry_run:
            return True, f"[预览] 将迁移: {pt_path} (step={step}, d_hidden={sae_config.get('d_hidden', 'N/A')})"

        # 备份原文件
        if backup:
            backup_path = pt_path.with_suffix(".pt.v1.backup")
            pt_path.rename(backup_path)
            logger.info("备份原文件: %s", backup_path)

        # 保存新格式
        torch.save(ckpt_new, pt_path)

        return True, f"迁移成功: {pt_path.name} (step={step}, d_hidden={sae_config.get('d_hidden', 'N/A')})"

    except Exception as e:
        return False, f"迁移失败: {e}"


def migrate_directory(
    run_dir: Path,
    backup: bool = True,
    dry_run: bool = False,
) -> Tuple[int, int, int, int]:
    """
    迁移整个实验目录下的所有 checkpoint

    返回: (成功数, 失败数, 跳过数(已新格式), 跳过数(无pt文件))
    """
    if not run_dir.exists():
        logger.error("目录不存在: %s", run_dir)
        return 0, 0, 0, 0

    success_count = 0
    fail_count = 0
    skip_new_format = 0
    skip_no_pt = 0

    # 查找所有 layer 子目录
    for layer_dir in sorted(run_dir.iterdir()):
        if not layer_dir.is_dir():
            continue

        pt_path = layer_dir / "sae_latest.pt"
        json_path = layer_dir / "sae_config.json"

        if not pt_path.exists():
            skip_no_pt += 1
            continue

        success, msg = migrate_file(pt_path, json_path, backup=backup, dry_run=dry_run)

        if "已经是新格式" in msg:
            skip_new_format += 1
            logger.info("[跳过] %s: %s", layer_dir.name, msg)
        elif success:
            success_count += 1
            logger.info("[成功] %s: %s", layer_dir.name, msg)
        else:
            fail_count += 1
            logger.error("[失败] %s: %s", layer_dir.name, msg)

    return success_count, fail_count, skip_new_format, skip_no_pt


def main():
    parser = argparse.ArgumentParser(
        description="迁移 SAE Checkpoint 从旧格式到新格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 迁移单个文件
  python migrate_ckpt.py --pt_path sae_runs/exp1/block_out.layer15/sae_latest.pt

  # 迁移整个目录
  python migrate_ckpt.py --run_dir sae_runs/exp1

  # 预览模式（不实际修改）
  python migrate_ckpt.py --run_dir sae_runs/exp1 --dry_run

  # 不备份原文件
  python migrate_ckpt.py --run_dir sae_runs/exp1 --no_backup
        """,
    )

    parser.add_argument(
        "--pt_path",
        type=str,
        help="单个 .pt 文件路径",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        help="对应的 .json 文件路径（可选，默认找 sae_config.json）",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        help="实验目录路径（自动查找所有 layer 子目录）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅预览，不实际执行迁移",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="不备份原文件",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="详细输出",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 验证参数
    if not args.pt_path and not args.run_dir:
        parser.error("必须指定 --pt_path 或 --run_dir")

    if args.pt_path and args.run_dir:
        parser.error("不能同时指定 --pt_path 和 --run_dir")

    backup = not args.no_backup
    dry_run = args.dry_run

    if dry_run:
        logger.info("=" * 60)
        logger.info("【预览模式】不会实际修改任何文件")
        logger.info("=" * 60)

    # 执行迁移
    if args.pt_path:
        # 单个文件模式
        pt_path = expand_path(args.pt_path)
        json_path = expand_path(args.json_path) if args.json_path else None

        success, msg = migrate_file(pt_path, json_path, backup=backup, dry_run=dry_run)

        if success:
            logger.info(msg)
            return 0
        else:
            logger.error(msg)
            return 1

    else:
        # 目录模式
        run_dir = expand_path(args.run_dir)
        logger.info("扫描目录: %s", run_dir)

        success, fail, skip_new, skip_no_pt = migrate_directory(
            run_dir, backup=backup, dry_run=dry_run
        )

        # 打印汇总
        logger.info("=" * 60)
        logger.info("迁移完成:")
        logger.info("  成功: %d", success)
        logger.info("  失败: %d", fail)
        logger.info("  跳过(已新格式): %d", skip_new)
        logger.info("  跳过(无PT文件): %d", skip_no_pt)
        logger.info("=" * 60)

        if fail > 0:
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
