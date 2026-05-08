#!/usr/bin/env python3
"""
SAE 初始化脚本 - 命令行版本

功能:
1. 从缓存的激活初始化 SAE
2. 支持 4 层独立初始化 (14, 19, 24, 29)
3. 支持 PCA + Tied 绑定初始化
4. 初始化质量验证

使用方法:
    # 初始化单层
    python -m 初始化.sae_init_cli --cache_dir ./cache --layer 14 --output_dir ./sae_weights

    # 初始化所有层
    python -m 初始化.sae_init_cli --cache_dir ./cache --layer all --output_dir ./sae_weights

    # 自定义参数
    python -m 初始化.sae_init_cli --cache_dir ./cache --layer 14 \
        --d_hidden 12288 --top_k 128 --output_dir ./sae_weights
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from 初始化.sae_phase2_core import TopKSAE, TopKSAEConfig
from 初始化.sae_phase2_init import SAEInitializer, SAEInitConfig


def setup_logging(verbose: bool = True):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("SAEInit")


def initialize_single_layer(
    cache_dir: str,
    layer_idx: int,
    output_dir: str,
    d_hidden: int,
    top_k: int,
    expansion_factor: int,
    verbose: bool = True,
) -> dict:
    """
    初始化单个层的 SAE

    返回:
        stats: 初始化统计信息
    """
    logger = logging.getLogger("SAEInit")

    logger.info("=" * 60)
    logger.info(f"初始化 Layer {layer_idx} SAE")
    logger.info("=" * 60)
    logger.info(f"  d_hidden: {d_hidden}")
    logger.info(f"  top_k: {top_k}")
    logger.info(f"  expansion_factor: {expansion_factor}")

    start_time = time.time()

    # 创建 SAE 配置
    sae_config = TopKSAEConfig(
        d_model=1536,
        d_hidden=d_hidden,
        top_k=top_k,
    )

    # 创建 SAE
    sae = TopKSAE(sae_config)

    # 创建初始化配置
    init_config = SAEInitConfig(
        expansion_factor=expansion_factor,
    )

    # 创建初始化器
    initializer = SAEInitializer(init_config, sae, verbose=verbose)

    # 从缓存初始化
    try:
        sae = initializer.initialize_from_cache(cache_dir, layer_idx, device="cpu")

        # 获取统计信息
        stats = initializer.get_stats()

        # 保存 SAE
        layer_output_dir = Path(output_dir) / f"layer{layer_idx}"
        sae.save_pretrained(str(layer_output_dir))

        elapsed = time.time() - start_time
        stats["elapsed_time"] = elapsed
        stats["output_dir"] = str(layer_output_dir)

        logger.info(f"Layer {layer_idx} 初始化完成!")
        logger.info(f"  耗时: {elapsed:.1f}秒")
        logger.info(f"  保存到: {layer_output_dir}")

        return stats

    except Exception as e:
        logger.error(f"Layer {layer_idx} 初始化失败: {e}")
        raise


def initialize_all_layers(
    cache_dir: str,
    output_dir: str,
    d_hidden: int,
    top_k: int,
    expansion_factor: int,
    layers: list,
    verbose: bool = True,
) -> dict:
    """初始化所有指定层的 SAE"""

    logger = logging.getLogger("SAEInit")

    logger.info("=" * 60)
    logger.info("批量初始化 SAE")
    logger.info("=" * 60)
    logger.info(f"  目标层: {layers}")
    logger.info(f"  缓存目录: {cache_dir}")
    logger.info(f"  输出目录: {output_dir}")

    all_stats = {}
    total_start = time.time()

    for layer_idx in layers:
        logger.info(f"\n{'='*40}")
        logger.info(f"处理 Layer {layer_idx}")
        logger.info(f"{'='*40}")

        stats = initialize_single_layer(
            cache_dir=cache_dir,
            layer_idx=layer_idx,
            output_dir=output_dir,
            d_hidden=d_hidden,
            top_k=top_k,
            expansion_factor=expansion_factor,
            verbose=verbose,
        )

        all_stats[f"layer{layer_idx}"] = stats

    total_elapsed = time.time() - total_start

    # 保存总体统计
    summary = {
        "layers": layers,
        "total_time": total_elapsed,
        "per_layer_stats": all_stats,
        "config": {
            "d_hidden": d_hidden,
            "top_k": top_k,
            "expansion_factor": expansion_factor,
        }
    }

    summary_path = Path(output_dir) / "initialization_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("批量初始化完成!")
    logger.info(f"  总耗时: {total_elapsed:.1f}秒")
    logger.info(f"  摘要保存到: {summary_path}")
    logger.info("=" * 60)

    return summary


def validate_cache(cache_dir: str, layers: list) -> bool:
    """验证缓存文件是否存在"""
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        print(f"错误: 缓存目录不存在: {cache_dir}")
        return False

    missing = []
    for layer in layers:
        layer_file = cache_path / f"layer{layer}.pt"
        if not layer_file.exists():
            missing.append(f"layer{layer}.pt")

    if missing:
        print(f"错误: 缺少缓存文件: {missing}")
        print(f"请先运行激活采集: python -m 初始化.sae_activation_collector ...")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="SAE 初始化脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 初始化 layer 14
    python -m 初始化.sae_init_cli --cache_dir ./cache --layer 14

    # 初始化所有层
    python -m 初始化.sae_init_cli --cache_dir ./cache --layer all

    # 自定义参数
    python -m 初始化.sae_init_cli --cache_dir ./cache --layer 14 --d_hidden 24576 --top_k 128
        """
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="激活缓存目录 (默认: ./cache)"
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="14",
        help="要初始化的层，如 '14' 或 'all' (默认: 14)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sae_weights",
        help="输出目录 (默认: ./sae_weights)"
    )
    parser.add_argument(
        "--d_hidden",
        type=int,
        default=12288,
        help="SAE 隐藏维度，8x=12288, 16x=24576 (默认: 12288)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=128,
        help="TopK 稀疏度 (默认: 128)"
    )
    parser.add_argument(
        "--expansion_factor",
        type=int,
        default=8,
        choices=[8, 16],
        help="扩展倍数 (默认: 8)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少输出"
    )

    args = parser.parse_args()

    # 设置日志
    logger = setup_logging(verbose=not args.quiet)

    # 确定要初始化的层
    if args.layer.lower() == "all":
        layers = [14, 19, 24, 29]
    else:
        try:
            layers = [int(x.strip()) for x in args.layer.split(",")]
        except ValueError:
            logger.error(f"无效的层参数: {args.layer}")
            sys.exit(1)

    # 验证层号
    valid_layers = {14, 19, 24, 29}
    for layer in layers:
        if layer not in valid_layers:
            logger.warning(f"层 {layer} 不是推荐的层 (14, 19, 24, 29)")

    # 验证缓存
    if not validate_cache(args.cache_dir, layers):
        sys.exit(1)

    # 验证参数
    d_hidden = args.d_hidden
    top_k = args.top_k

    ratio = top_k / d_hidden * 100
    if not (0.1 <= ratio <= 1.0):
        logger.error(f"top_k/d_hidden 比例必须在 0.1%%~1%%，当前: {ratio:.2f}%%")
        sys.exit(1)

    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 执行初始化
    if len(layers) == 1:
        initialize_single_layer(
            cache_dir=args.cache_dir,
            layer_idx=layers[0],
            output_dir=args.output_dir,
            d_hidden=d_hidden,
            top_k=top_k,
            expansion_factor=args.expansion_factor,
            verbose=not args.quiet,
        )
    else:
        initialize_all_layers(
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            d_hidden=d_hidden,
            top_k=top_k,
            expansion_factor=args.expansion_factor,
            layers=layers,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
