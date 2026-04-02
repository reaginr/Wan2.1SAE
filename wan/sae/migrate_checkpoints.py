#!/usr/bin/env python3
"""
SAE Checkpoint 迁移脚本

将旧格式（配置在 .json，权重在 .pt）迁移到新格式（配置内置在 .pt）

用法:
    # 迁移单个 checkpoint
    python migrate_checkpoints.py --pt sae_runs/exp/block_out.layer15/sae_latest.pt

    # 迁移整个实验目录
    python migrate_checkpoints.py --dir sae_runs/exp

    # 预览（不实际执行）
    python migrate_checkpoints.py --dir sae_runs/exp --dry-run

    # 不备份（谨慎使用）
    python migrate_checkpoints.py --dir sae_runs/exp --no-backup

迁移后文件结构:
    旧格式:                    新格式:
    ├── sae_config.json   →    ├── sae_config.json (保留用于查看)
    ├── sae_latest.pt     →    ├── sae_latest.pt (配置内置)
    └── sae_step500.pt    →    └── sae_step500.pt (配置内置)

备份文件:
    原 .pt 文件会被重命名为 .pt.v1.backup
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wan.sae.checkpoint_io import CheckpointMigrator

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="迁移 SAE Checkpoint 从旧格式到新格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 迁移单个文件
  python migrate_checkpoints.py --pt sae_runs/exp/block_out.layer15/sae_latest.pt

  # 迁移整个目录
  python migrate_checkpoints.py --dir sae_runs/exp

  # 预览将要迁移的文件
  python migrate_checkpoints.py --dir sae_runs/exp --dry-run
        """,
    )

    parser.add_argument(
        "--pt",
        type=str,
        help="单个 .pt 文件路径",
    )
    parser.add_argument(
        "--json",
        type=str,
        help="对应的 .json 文件路径（默认自动查找）",
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="实验目录路径（迁移所有子目录）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览，不实际执行",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不备份原文件（谨慎使用）",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="详细输出",
    )

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not args.pt and not args.dir:
        parser.error("必须指定 --pt 或 --dir")

    backup = not args.no_backup

    if args.pt:
        # 单个文件迁移
        pt_path = Path(args.pt)
        json_path = Path(args.json) if args.json else None

        if args.dry_run:
            logger.info("[预览] 将迁移: %s", pt_path)
            if json_path:
                logger.info("[预览] 使用配置: %s", json_path)
            return

        success = CheckpointMigrator.migrate_file(pt_path, json_path, backup=backup)
        sys.exit(0 if success else 1)

    else:
        # 目录迁移
        logger.info("开始迁移目录: %s", args.dir)
        if args.dry_run:
            logger.info("【预览模式】不会实际修改文件")

        success, fail = CheckpointMigrator.migrate_directory(args.dir, dry_run=args.dry_run)

        logger.info("=" * 60)
        if args.dry_run:
            logger.info("预览完成: %d 个文件将被迁移", success)
        else:
            logger.info("迁移完成: 成功 %d 个, 失败 %d 个", success, fail)
        logger.info("=" * 60)

        if fail > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
