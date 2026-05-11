"""
SAE Training V2 - 启动脚本

严格按照 TODO_list_v3.md 规范实现

使用方法:
    # 快速验证 (200 steps)
    python run_sae_train_v2.py --mode quick_test \
        --layers 14,19,24,29 \
        --model_path /path/to/Wan2.1-T2V-1.3B \
        --prompt_file ./prompts.txt \
        --init_dir ./sae_init

    # 完整训练 (8000 steps)
    python run_sae_train_v2.py --mode train \
        --layers 14,19,24,29 \
        --model_path /path/to/Wan2.1-T2V-1.3B \
        --prompt_dir ./nsfw_prompts

    # 从 checkpoint 恢复
    python run_sae_train_v2.py --mode resume \
        --layer 14 \
        --checkpoint sae_runs/exp1/block_out.layer14/sae_latest.pt

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent))

# train_v2 模块
from train_v2 import (
    TrainingConfig,
    LayerTrainer,
    TrainingLoop,
    CheckpointManager,
    HOOK_LAYERS,
    D_MODEL,
    D_HIDDEN,
)

# Wan 模块
from wan.configs.wan_t2v_1_3B import t2v_1_3B
from wan.modules.sae_new import SAEConfig, SparseAutoEncoder
from wan.sae.hooking import register_dit_hooks, remove_hooks
from wan.text2video import WanT2V


# ============================================================================
# 日志配置
# ============================================================================

def setup_logging(log_file: str, verbose: bool = True):
    """
    配置日志系统

    输出到:
    1. 控制台 (INFO级别)
    2. 文件 (DEBUG级别)
    """
    # 创建日志目录
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # 清除已有的 handlers
    root_logger.handlers.clear()

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO if not verbose else logging.DEBUG)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # 文件 handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


# ============================================================================
# 提示词加载
# ============================================================================

def load_prompts(
    prompt_source: str,
    max_prompts: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
) -> List[str]:
    """
    加载提示词

    参数:
        prompt_source: 提示词来源
            - 文件: ./prompts.txt (单个文件)
            - 目录: ./prompts_dir/ (多个 .txt 文件)
        max_prompts: 最大加载数量
        shuffle: 是否随机打乱
        seed: 随机种子

    返回:
        提示词列表
    """
    path = Path(prompt_source)
    prompts = []

    if path.is_file():
        # 单个文件
        logger.info(f"Loading prompts from file: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and len(line) >= 8:  # 过滤过短
                    prompts.append(line)

    elif path.is_dir():
        # 目录
        logger.info(f"Loading prompts from directory: {path}")
        for txt_file in sorted(path.glob("*.txt")):
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and len(line) >= 8:
                        prompts.append(line)
    else:
        raise FileNotFoundError(f"Prompt source not found: {path}")

    # 去重
    original_count = len(prompts)
    prompts = list(dict.fromkeys(prompts))  # 保持顺序的去重

    # 打乱
    if shuffle:
        random.seed(seed)
        random.shuffle(prompts)
        logger.info(f"Shuffled prompts with seed={seed}")

    # 限制数量
    if max_prompts is not None and len(prompts) > max_prompts:
        prompts = prompts[:max_prompts]

    logger.info(f"Loaded {len(prompts)} unique prompts (from {original_count} raw)")

    return prompts


# ============================================================================
# 激活数据集
# ============================================================================

class ActivationDataset(Dataset):
    """从 DiT 提取的激活数据集"""

    def __init__(self, activations: torch.Tensor):
        self.activations = activations

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx]


# ============================================================================
# 激活提取器
# ============================================================================

class ActivationExtractor:
    """
    从 Wan DiT 提取隐藏状态激活

    优化: 只运行 DiT forward，不运行 VAE 解码
    这可以将每个 prompt 的处理时间从 ~15 分钟降到 ~4 分钟
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        t5_cpu: bool = True,
    ):
        self.model_path = model_path
        self.device = device
        self.t5_cpu = t5_cpu

        logger.info(f"Loading Wan model from {model_path}")
        self.t2v = WanT2V(
            config=t2v_1_3B,
            checkpoint_dir=model_path,
            device_id=0,
            t5_cpu=t5_cpu,
        )
        logger.info("Wan model loaded successfully")

    def _run_dit_sampling(
        self,
        prompt: str,
        size: tuple = (832, 480),
        frame_num: int = 81,
        sampling_steps: int = 30,
        shift: float = 5.0,
        guide_scale: float = 5.0,
        seed: int = -1,
    ):
        """
        只运行 DiT 扩散采样，不运行 VAE 解码

        这比完整 generate() 快 3-4 倍，因为我们不需要生成视频
        """
        import math
        import random
        import sys
        from contextlib import contextmanager
        from tqdm import tqdm
        import torch.cuda.amp as amp

        config = self.t2v.config
        device = self.t2v.device

        # 计算 latent shape
        F = frame_num
        target_shape = (
            self.t2v.vae.model.z_dim,
            (F - 1) // self.t2v.vae_stride[0] + 1,
            size[1] // self.t2v.vae_stride[1],
            size[0] // self.t2v.vae_stride[2],
        )

        seq_len = math.ceil(
            (target_shape[2] * target_shape[3]) /
            (self.t2v.patch_size[1] * self.t2v.patch_size[2]) *
            target_shape[1] / self.t2v.sp_size
        ) * self.t2v.sp_size

        # 随机种子
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)

        # 文本编码
        if not self.t2v.t5_cpu:
            self.t2v.text_encoder.model.to(device)
            context = self.t2v.text_encoder([prompt], device)
            context_null = self.t2v.text_encoder([""], device)
        else:
            context = self.t2v.text_encoder([prompt], torch.device('cpu'))
            context_null = self.t2v.text_encoder([""], device)
            context = [t.to(device) for t in context]
            context_null = [t.to(device) for t in context_null]

        # 初始噪声
        noise = [
            torch.randn(
                target_shape[0], target_shape[1], target_shape[2], target_shape[3],
                dtype=torch.float32,
                device=device,
                generator=seed_g
            )
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.t2v.model, 'no_sync', noop_no_sync)

        # 使用 Euler 调度器
        from diffusers import FlowMatchEulerDiscreteScheduler
        sample_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=config.num_train_timesteps,
            shift=shift,
        )
        sample_scheduler.set_timesteps(sampling_steps, device=device)
        timesteps = sample_scheduler.timesteps

        # 扩散采样 (只运行 DiT，不运行 VAE)
        latents = noise
        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}

        with amp.autocast(dtype=self.t2v.param_dtype), torch.no_grad(), no_sync():
            for _, t in enumerate(tqdm(timesteps, desc="DiT sampling", leave=False)):
                latent_model_input = latents
                timestep = torch.stack([t])

                self.t2v.model.to(device)
                noise_pred_cond = self.t2v.model(latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.t2v.model(latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0),
                    return_dict=False, generator=seed_g
                )[0]
                latents = [temp_x0.squeeze(0)]

        # 不运行 VAE 解码，直接返回
        return latents

    def extract(
        self,
        prompts: List[str],
        layer_indices: List[int],
        hook_mode: str = "block_out",
        sampling_steps: int = 30,
        max_tokens_per_prompt: int = 1024,
        seed: int = 42,
    ) -> Dict[int, torch.Tensor]:
        """
        提取多层激活

        参数:
            prompts: 提示词列表
            layer_indices: 层索引列表
            hook_mode: hook 模式
            sampling_steps: 扩散步数
            max_tokens_per_prompt: 每个提示词最大 token 数
            seed: 随机种子

        返回:
            {layer_idx: activations [N, D]}
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 存储每层的激活
        layer_activations = {idx: [] for idx in layer_indices}

        # 注册 hooks
        hook_handlers = []
        hook_outputs = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                hook_outputs[layer_idx] = output.detach()
            return hook

        for layer_idx in layer_indices:
            target_layer = self.t2v.model.blocks[layer_idx]
            hook_handlers.append(
                target_layer.register_forward_hook(make_hook(layer_idx))
            )

        logger.info(f"Registered hooks for layers: {layer_indices}")

        try:
            for i, prompt in enumerate(prompts):
                if (i + 1) % 5 == 0 or i == 0:
                    logger.info(f"Processing prompt {i+1}/{len(prompts)}")

                # 只运行 DiT 扩散采样，不运行 VAE 解码
                # 这比完整 generate() 快 3-4 倍
                try:
                    _ = self._run_dit_sampling(
                        prompt=prompt,
                        size=(832, 480),
                        frame_num=81,
                        sampling_steps=sampling_steps,
                        seed=seed + i,
                    )
                except Exception as e:
                    logger.warning(f"Sampling failed for prompt {i}: {e}")
                    continue

                # 收集各层激活
                for layer_idx in layer_indices:
                    if layer_idx in hook_outputs:
                        act = hook_outputs[layer_idx]  # [B, L, D]

                        # RMSNorm
                        rms = torch.sqrt(act.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
                        act_norm = act / rms

                        # 展平并采样
                        act_flat = act_norm.reshape(-1, act.size(-1))

                        if len(act_flat) > max_tokens_per_prompt:
                            indices = torch.randperm(len(act_flat))[:max_tokens_per_prompt]
                            act_flat = act_flat[indices]

                        layer_activations[layer_idx].append(act_flat.cpu())

                # 清空 hook_outputs 节省内存
                hook_outputs.clear()

        finally:
            for h in hook_handlers:
                h.remove()

        # 合并每层的激活
        result = {}
        for layer_idx in layer_indices:
            if layer_activations[layer_idx]:
                result[layer_idx] = torch.cat(layer_activations[layer_idx], dim=0)
                logger.info(
                    f"Layer {layer_idx}: extracted {len(result[layer_idx])} tokens "
                    f"(shape: {result[layer_idx].shape})"
                )

        return result


# ============================================================================
# 训练流程
# ============================================================================

def train_layers(
    config: TrainingConfig,
    layer_indices: List[int],
    model_path: str,
    prompts: List[str],
    init_dir: Optional[str] = None,
    checkpoints: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """
    训练多层 SAE (顺序执行)

    参数:
        config: 训练配置
        layer_indices: 层索引列表
        model_path: 模型路径
        prompts: 提示词列表
        init_dir: 初始化权重目录
        checkpoints: 恢复训练的 checkpoint

    返回:
        训练结果
    """
    results = {}
    start_time = time.time()

    # 提取激活 (一次性提取所有层)
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Extracting Activations")
    logger.info("=" * 70)

    extractor = ActivationExtractor(model_path=model_path, device=config.device)

    layer_activations = extractor.extract(
        prompts=prompts,
        layer_indices=layer_indices,
        hook_mode=config.hook_mode,
        max_tokens_per_prompt=config.max_tokens_per_batch // len(prompts),
    )

    # 逐层训练
    for layer_idx in layer_indices:
        logger.info("\n" + "=" * 70)
        logger.info(f"STEP 2: Training Layer {layer_idx}")
        logger.info("=" * 70)

        if layer_idx not in layer_activations:
            logger.error(f"No activations for layer {layer_idx}, skipping")
            continue

        activations = layer_activations[layer_idx]

        # 分割训练/验证
        n_total = len(activations)
        n_train = int(n_total * 0.9)

        # 打乱索引
        perm = torch.randperm(n_total)
        train_indices = perm[:n_train]
        val_indices = perm[n_train:]

        train_act = activations[train_indices]
        val_act = activations[val_indices]

        logger.info(f"  Train tokens: {len(train_act)}")
        logger.info(f"  Val tokens: {len(val_act)}")

        # 创建数据加载器
        train_loader = DataLoader(
            ActivationDataset(train_act),
            batch_size=config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            ActivationDataset(val_act),
            batch_size=config.batch_size,
        )

        # 查找初始化权重
        init_checkpoint = None
        if init_dir:
            init_checkpoint = find_init_checkpoint(init_dir, layer_idx)

        # 创建训练循环
        loop = TrainingLoop(config, device=config.device)

        # 运行训练
        result = loop.run(
            layer_idx=layer_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            resume_from=checkpoints.get(layer_idx) if checkpoints else None,
            init_checkpoint=init_checkpoint,
        )

        results[layer_idx] = result

        # 打印层训练结果
        logger.info("\n" + "-" * 70)
        logger.info(f"Layer {layer_idx} Training Summary")
        logger.info("-" * 70)
        logger.info(f"  Final step: {result['final_step']}")
        logger.info(f"  Converged: {result['is_converged']}")
        logger.info(f"  Best MSE: {result['best_mse']:.6f}")
        logger.info(f"  Elapsed: {result['elapsed_seconds']:.1f}s")

    total_elapsed = time.time() - start_time

    # 打印总结果
    logger.info("\n" + "=" * 70)
    logger.info("ALL LAYERS COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_elapsed / 3600:.2f} hours")
    logger.info(f"Layers trained: {list(results.keys())}")

    all_converged = all(r['is_converged'] for r in results.values())
    logger.info(f"All converged: {all_converged}")

    # 保存结果摘要
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_hours": total_elapsed / 3600,
        "all_converged": all_converged,
        "layers": {
            idx: {
                "final_step": r['final_step'],
                "is_converged": r['is_converged'],
                "best_mse": r['best_mse'],
            }
            for idx, r in results.items()
        },
        "config": config.to_dict(),
    }

    return summary


def find_init_checkpoint(init_dir: str, layer_idx: int) -> Optional[str]:
    """查找初始化权重文件"""
    init_path = Path(init_dir)

    patterns = [
        f"sae_init_layer{layer_idx}.pt",
        f"layer{layer_idx}/sae.pt",
        f"block_out.layer{layer_idx}/sae.pt",
    ]

    for pattern in patterns:
        path = init_path / pattern
        if path.exists():
            logger.info(f"  Found init checkpoint: {path}")
            return str(path)

    logger.warning(f"  Init checkpoint not found for layer {layer_idx}")
    return None


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SAE Training V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 快速验证 (200 steps, 所有层)
    python run_sae_train_v2.py --mode quick_test \\
        --layers 14,19,24,29 \\
        --model_path /root/Wan/Wan2.1-T2V-1.3B \\
        --prompt_file ./prompts.txt \\
        --init_dir ./sae_init

    # 完整训练 (8000 steps)
    python run_sae_train_v2.py --mode train \\
        --layers 14,19,24,29 \\
        --model_path /root/Wan/Wan2.1-T2V-1.3B \\
        --prompt_dir ./nsfw_prompts \\
        --init_dir ./sae_init

    # nohup 后台运行
    nohup python run_sae_train_v2.py --mode train \\
        --layers 14,19,24,29 \\
        --model_path /root/Wan/Wan2.1-T2V-1.3B \\
        --prompt_dir ./nsfw_prompts \\
        --log_file ./logs/train.log \\
        --init_dir ./sae_init \\
        > /dev/null 2>&1 &
        """
    )

    # 运行模式
    parser.add_argument("--mode", type=str, default="train",
                        choices=["quick_test", "train", "resume"],
                        help="运行模式: quick_test (200步验证), train (完整训练), resume (恢复)")

    # 层配置
    parser.add_argument("--layers", type=str, default="14,19,24,29",
                        help="训练层 (逗号分隔, 如 14,19,24,29)")
    parser.add_argument("--layer", type=int, default=None,
                        help="训练单层 (覆盖 --layers)")

    # 路径配置
    parser.add_argument("--model_path", type=str, required=True,
                        help="Wan2.1-T2V-1.3B 模型路径")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="提示词文件 (单个 .txt)")
    parser.add_argument("--prompt_dir", type=str, default=None,
                        help="提示词目录 (多个 .txt)")
    parser.add_argument("--run_dir", type=str, default="sae_runs/exp_default",
                        help="输出目录")
    parser.add_argument("--log_file", type=str, default=None,
                        help="日志文件路径 (默认: {run_dir}/train.log)")
    parser.add_argument("--init_dir", type=str, default=None,
                        help="初始化权重目录")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="恢复训练的 checkpoint (单层)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="恢复训练的 checkpoint 目录 (多层)")

    # 训练参数
    parser.add_argument("--steps", type=int, default=None,
                        help="总训练步数 (默认: quick_test=200, train=8000)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批大小 (default: 4)")
    parser.add_argument("--lr", type=float, default=6e-5,
                        help="学习率 (default: 6e-5)")
    parser.add_argument("--d_hidden", type=int, default=12288,
                        help="SAE 隐藏维度 (default: 12288)")
    parser.add_argument("--top_k", type=int, default=128,
                        help="TopK 稀疏度 (default: 128)")

    # 提示词参数
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="最大提示词数量")
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="打乱提示词顺序 (default: True)")
    parser.add_argument("--no_shuffle", action="store_false", dest="shuffle",
                        help="不打乱提示词顺序")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    # 其他
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (default: cuda)")
    parser.add_argument("--verbose", action="store_true",
                        help="详细日志输出")

    args = parser.parse_args()

    # 解析层配置
    if args.layer is not None:
        layer_indices = [args.layer]
    else:
        layer_indices = [int(x.strip()) for x in args.layers.split(",")]

    # 验证层配置
    for layer in layer_indices:
        if layer not in HOOK_LAYERS:
            parser.error(f"Layer {layer} 不在允许列表 {HOOK_LAYERS} 中")

    # 设置默认步数
    if args.steps is None:
        args.steps = 200 if args.mode == "quick_test" else 8000

    # 设置默认日志文件
    if args.log_file is None:
        args.log_file = f"{args.run_dir}/train.log"

    # 初始化日志
    logger = setup_logging(args.log_file, verbose=args.verbose)

    # 创建配置
    config = TrainingConfig(
        d_hidden=args.d_hidden,
        top_k=args.top_k,
        lr=args.lr,
        total_steps=args.steps,
        batch_size=args.batch_size,
        run_dir=args.run_dir,
        device=args.device,
        seed=args.seed,
        hook_layers=layer_indices,
    )

    # 验证配置
    config.validate()

    # 打印配置
    logger.info("\n" + "=" * 70)
    logger.info("SAE Training V2 - Configuration")
    logger.info("=" * 70)
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Layers: {layer_indices}")
    logger.info("-" * 70)
    logger.info(f"  Model path: {args.model_path}")
    logger.info(f"  Run dir: {args.run_dir}")
    logger.info(f"  Log file: {args.log_file}")
    logger.info("-" * 70)
    logger.info(f"  d_hidden: {config.d_hidden} ({config.d_hidden // D_MODEL}x expansion)")
    logger.info(f"  top_k: {config.top_k}")
    logger.info(f"  lr: {config.lr}")
    logger.info(f"  batch_size: {config.batch_size}")
    logger.info(f"  accum_steps: {config.accum_steps}")
    logger.info(f"  effective_batch: {config.effective_batch_size}")
    logger.info(f"  total_steps: {config.total_steps}")
    logger.info(f"  warmup_steps: {config.warmup_steps}")
    logger.info(f"  val_interval: {config.val_interval}")
    logger.info(f"  checkpoint_interval: {config.checkpoint_interval}")
    logger.info("-" * 70)
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Seed: {args.seed}")
    if args.init_dir:
        logger.info(f"  Init dir: {args.init_dir}")
    logger.info("=" * 70)

    # 加载提示词
    if args.prompt_file:
        prompt_source = args.prompt_file
    elif args.prompt_dir:
        prompt_source = args.prompt_dir
    else:
        parser.error("必须指定 --prompt_file 或 --prompt_dir")

    prompts = load_prompts(
        prompt_source,
        max_prompts=args.max_prompts,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    if not prompts:
        parser.error("没有加载到任何提示词")

    # 处理恢复训练
    checkpoints = {}
    if args.checkpoint:
        checkpoints[layer_indices[0]] = args.checkpoint
    elif args.checkpoint_dir:
        for layer_idx in layer_indices:
            ckpt_path = find_init_checkpoint(args.checkpoint_dir, layer_idx)
            if ckpt_path:
                checkpoints[layer_idx] = ckpt_path

    # 运行训练
    start_time = time.time()

    summary = train_layers(
        config=config,
        layer_indices=layer_indices,
        model_path=args.model_path,
        prompts=prompts,
        init_dir=args.init_dir,
        checkpoints=checkpoints if checkpoints else None,
    )

    # 保存结果摘要
    summary_path = Path(args.run_dir) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\nSummary saved to: {summary_path}")

    # 打印最终结果
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Total time: {summary['total_elapsed_hours']:.2f} hours")
    logger.info(f"All converged: {summary['all_converged']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
