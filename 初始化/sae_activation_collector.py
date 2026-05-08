"""
SAE 第二阶段 - 激活采集系统

根据工程要求:
1. 复用 wan/sae/hooking.py 的 hook 系统
2. 单次 forward 多层并行 hook (layer 14, 19, 24, 29)
3. Hook 位置: Transformer Block 最终 residual 输出
4. Activation 必须落盘缓存 (BF16, CPU tensor)
5. T5 embedding 必须缓存，禁止重复编码
6. 支持 prompt batch 处理

核心流程:
    Step1: 单次 forward 采样 → hook 4 layers → save cache
    Step2: CPU PCA 并行
    Step3: 串行 SAE 训练

使用方法:
    from 初始化.sae_activation_collector import (
        ActivationCollector, ActivationCollectorConfig
    )

    config = ActivationCollectorConfig(
        checkpoint_dir="F:/Wan2.1-T2V-1.3B",
        prompt_file="./prompts.txt",
        cache_dir="./cache",
        hook_layers=[14, 19, 24, 29],
    )

    collector = ActivationCollector(config)
    collector.collect_activations(num_samples=500)
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

# 导入 Wan 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from wan.sae.hooking import register_dit_hooks, remove_hooks, HookMode


@dataclass
class ActivationCollectorConfig:
    """
    激活采集配置
    """

    # 路径配置
    checkpoint_dir: str = ""  # Wan 模型路径 (必填)
    prompt_file: str = ""  # 提示词文件路径 (必填)
    cache_dir: str = "./cache"  # 激活缓存目录

    # Hook 配置
    hook_layers: List[int] = field(default_factory=lambda: [14, 19, 24, 29])
    hook_mode: str = "block_out"  # 只允许 block_out

    # 采样配置
    num_samples: int = 500  # 采样样本数
    tokens_per_sample: int = 512  # 每个样本采样多少 token

    # Timestep 采样 (根据工程要求)
    timestep_min: float = 0.35
    timestep_max: float = 0.75
    timestep_mid_ratio: float = 0.70  # 70% 在 [0.45, 0.65], 30% 在边界

    # 视频尺寸 (决定 latent shape)
    size_w: int = 832
    size_h: int = 480
    frame_num: int = 81

    # 设备与精度
    device_id: int = 0
    dtype: str = "bf16"  # bf16 or fp16

    # 缓存配置
    save_dtype: str = "bf16"  # 保存精度
    compress: bool = False  # 是否压缩

    # 随机种子
    seed: int = 42

    def __post_init__(self):
        """校验配置"""
        # Hook 层校验 (必须包含14, 19, 24, 29)
        required_layers = {14, 19, 24, 29}
        if not required_layers.issubset(set(self.hook_layers)):
            raise ValueError(f"hook_layers 必须包含 {required_layers}")

        # Hook 模式校验 (只允许 block_out)
        if self.hook_mode != "block_out":
            raise ValueError(f"hook_mode 必须是 'block_out'，当前: {self.hook_mode}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ActivationCollector:
    """
    激活采集器

    核心功能:
    1. 加载最小模型集合 (DiT + T5)
    2. 单次 forward 多层 hook
    3. 时空 token 采样
    4. 激活缓存到磁盘

    设计原则:
    - 最大化 GPU 利用率
    - 最小化 DiT forward 次数
    - 禁止重复 T5 编码
    """

    def __init__(self, config: ActivationCollectorConfig):
        self.config = config
        self.device = torch.device(f"cuda:{config.device_id}")
        self.dtype = torch.bfloat16 if config.dtype == "bf16" else torch.float16

        # 模型组件
        self.model = None  # DiT
        self.text_encoder = None  # T5
        self.vae = None  # VAE (可选加载)

        # 缓存
        self._prompt_cache: Dict[str, torch.Tensor] = {}
        # 使用 hook 返回的 key 格式: "block_out.layer{idx}"
        self._activation_cache: Dict[str, List[torch.Tensor]] = {
            f"block_out.layer{layer}": [] for layer in config.hook_layers
        }

        # 统计
        self.stats = {
            "num_forwards": 0,
            "total_tokens_collected": 0,
            "timestep_distribution": [],
        }

        # 日志
        self._setup_logging()

    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger = logging.getLogger("ActivationCollector")

    def load_minimal_model(self):
        """
        加载最小模型集合

        只加载:
        - DiT backbone (WanModel)
        - Text encoder (T5)

        禁止加载:
        - VAE decode
        - 完整推理 pipeline
        """
        from wan.configs import WAN_CONFIGS
        from wan.modules.model import WanModel
        from wan.modules.t5 import T5EncoderModel

        self.logger.info("=" * 60)
        self.logger.info("加载最小模型集合...")
        self.logger.info("=" * 60)

        cfg = WAN_CONFIGS["t2v-1.3B"]

        # 加载 T5 文本编码器
        self.logger.info("加载 T5 文本编码器...")
        self.text_encoder = T5EncoderModel(
            text_len=cfg.text_len,
            dtype=cfg.t5_dtype,
            device=torch.device("cpu"),  # 先放 CPU
            checkpoint_path=os.path.join(self.config.checkpoint_dir, cfg.t5_checkpoint),
            tokenizer_path=os.path.join(self.config.checkpoint_dir, cfg.t5_tokenizer),
        )

        # 加载 DiT 模型
        self.logger.info("加载 DiT 模型...")
        self.model = WanModel.from_pretrained(self.config.checkpoint_dir)
        self.model.eval().requires_grad_(False)
        self.model.to(self.device)

        # 存储 VAE stride 等配置
        self.vae_stride = cfg.vae_stride
        self.patch_size = cfg.patch_size
        self.num_train_timesteps = cfg.num_train_timesteps

        self.logger.info("模型加载完成!")
        self.logger.info(f"  DiT: {sum(p.numel() for p in self.model.parameters()):,} 参数")
        self.logger.info(f"  Device: {self.device}")

    def load_prompts(self) -> List[str]:
        """加载提示词"""
        with open(self.config.prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

        self.logger.info(f"加载 {len(prompts)} 条提示词")
        return prompts

    def encode_prompts(self, prompts: List[str]) -> List[torch.Tensor]:
        """
        批量编码提示词 (缓存 T5 embedding)

        禁止重复 T5 编码
        """
        context_list = []

        # 检查 T5 模型是否在 GPU 上，如果不是则移动
        # 使用参数的设备来判断
        try:
            t5_device = next(self.text_encoder.model.parameters()).device
            if t5_device.type != "cuda":
                self.text_encoder.model.to(self.device)
        except (StopIteration, AttributeError):
            # 如果模型没有参数，直接尝试移动
            self.text_encoder.model.to(self.device)

        # 批量编码
        self.logger.info(f"批量编码 {len(prompts)} 条提示词...")

        with torch.no_grad():
            for prompt in tqdm(prompts, desc="T5 编码"):
                if prompt in self._prompt_cache:
                    context_list.append(self._prompt_cache[prompt])
                else:
                    context = self.text_encoder([prompt], self.device)
                    # context 是 list of tensors
                    context_tensor = context[0] if isinstance(context, list) else context
                    self._prompt_cache[prompt] = context_tensor
                    context_list.append(context_tensor)

        # T5 移回 CPU 节省显存
        self.text_encoder.model.cpu()
        torch.cuda.empty_cache()

        return context_list

    def sample_timestep(self) -> float:
        """
        采样 timestep

        规则:
        - 70%: t ∈ [0.45, 0.65]
        - 30%: 边界采样 [0.35, 0.45] ∪ [0.65, 0.75]
        """
        if random.random() < self.config.timestep_mid_ratio:
            # 中间区域
            t = random.uniform(0.45, 0.65)
        else:
            # 边界区域
            if random.random() < 0.5:
                t = random.uniform(0.35, 0.45)
            else:
                t = random.uniform(0.65, 0.75)

        return t

    def create_noisy_latent(self, batch_size: int = 1) -> torch.Tensor:
        """
        创建噪声 latent

        形状计算:
        - T = (frame_num - 1) // 4 + 1 = 21
        - H = size_h // 8 = 60
        - W = size_w // 8 = 104
        - latent shape: [B, 16, T, H, W]
        """
        cfg = self.config
        T = (cfg.frame_num - 1) // self.vae_stride[0] + 1
        H = cfg.size_h // self.vae_stride[1]
        W = cfg.size_w // self.vae_stride[2]

        latent = torch.randn(
            batch_size, 16, T, H, W,
            dtype=torch.float32,
            device=self.device
        )

        return latent

    def single_forward_multi_hook(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        timestep: float,
    ) -> Dict[str, torch.Tensor]:
        """
        单次 forward 多层 hook

        核心优化: 一次 DiT forward 同时捕获 4 层激活

        参数:
            latent: 噪声 latent [B, 16, T, H, W]
            context: 文本 embedding [L, D]
            timestep: 扩散时间步 (0-1 范围)

        返回:
            activations: {layer_key: tensor [B, L, D]}
        """
        # 准备收集激活的字典
        collected: Dict[str, torch.Tensor] = {}

        def on_tensor(key: str, tensor: torch.Tensor):
            """Hook 回调函数"""
            # 立即 detach 并移到 CPU
            collected[key] = tensor.detach().cpu()

        # 注册 hooks
        handles = register_dit_hooks(
            model=self.model,
            hook_layers=self.config.hook_layers,
            hook_mode="block_out",
            on_tensor=on_tensor,
        )

        # 准备 DiT 输入
        # WanModel.forward 期望:
        #   x: List[Tensor], 每个 tensor 形状 [C_in, F, H, W]
        #   context: List[Tensor], 每个 tensor 形状 [L, D]
        #   t: Tensor shape [B]

        # latent: [B, 16, T, H, W] -> list of [16, T, H, W]
        if latent.dim() == 5:
            latent_list = [latent[0]]  # 取 batch 中的第一个
        else:
            latent_list = [latent]

        # context: [L, D] -> list of [L, D]
        if isinstance(context, torch.Tensor):
            context_list = [context]
        else:
            context_list = context

        # seq_len 计算
        # T, H, W from latent shape
        _, C, T, H, W = latent.shape
        seq_len = math.ceil(
            (H * W) / (self.patch_size[1] * self.patch_size[2]) * T
        )

        # Timestep 转换: float 0-1 -> int 0-999
        t_int = int(timestep * (self.num_train_timesteps - 1))
        t_tensor = torch.tensor([t_int], device=self.device, dtype=torch.long)

        # 执行单次 forward
        try:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.dtype):
                _ = self.model(
                    latent_list,
                    t=t_tensor,
                    context=context_list,
                    seq_len=seq_len,
                )
        except Exception as e:
            self.logger.error(f"Forward 失败: {e}")
            raise
        finally:
            # 移除 hooks (无论成功失败都要移除)
            remove_hooks(handles)

        self.stats["num_forwards"] += 1

        return collected

    def spatial_token_sampling(
        self,
        activation: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """
        时空 token 采样

        规则:
        - 30% High-Norm tokens
        - 50% Mid-Norm tokens
        - 20% Low-Norm/Random tokens
        - Bucket-aware spatial sampling

        输入: activation [B, L, D]
        输出: sampled_tokens [num_tokens, D]
        """
        B, L, D = activation.shape

        # 展平
        tokens = activation.view(-1, D)  # [B*L, D]

        if tokens.shape[0] <= num_tokens:
            return tokens

        # 计算每个 token 的 norm
        norms = tokens.norm(dim=-1)  # [B*L]

        # 按比例采样
        n_high = int(num_tokens * 0.30)
        n_mid = int(num_tokens * 0.50)
        n_low = num_tokens - n_high - n_mid

        # High-Norm: top-k
        _, high_idx = torch.topk(norms, n_high)
        high_tokens = tokens[high_idx]

        # Mid-Norm: 中间区域随机采样
        sorted_norms, sorted_idx = torch.sort(norms)
        mid_start = len(sorted_norms) // 4
        mid_end = 3 * len(sorted_norms) // 4
        mid_pool = sorted_idx[mid_start:mid_end]
        mid_select_idx = torch.randperm(len(mid_pool))[:n_mid]
        mid_tokens = tokens[mid_pool[mid_select_idx]]

        # Low-Norm/Random: 随机采样
        low_idx = torch.randperm(len(tokens))[:n_low]
        low_tokens = tokens[low_idx]

        # 合并
        sampled = torch.cat([high_tokens, mid_tokens, low_tokens], dim=0)

        return sampled

    def collect_activations(
        self,
        num_samples: Optional[int] = None,
        prompts: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        采集激活

        核心流程:
        1. 加载模型
        2. 编码 prompts (缓存)
        3. 循环采样:
           - 采样 timestep
           - 创建噪声 latent
           - 单次 forward + 多层 hook
           - 空间 token 采样
           - 存入缓存
        4. 保存到磁盘

        返回:
            activations: {layer_key: tensor [N, D]}
        """
        if num_samples is None:
            num_samples = self.config.num_samples

        # 加载模型
        if self.model is None:
            self.load_minimal_model()

        # 加载 prompts
        if prompts is None:
            prompts = self.load_prompts()

        # 编码 prompts
        context_list = self.encode_prompts(prompts[:num_samples])

        # 初始化缓存
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 清空激活缓存
        for layer_key in self._activation_cache:
            self._activation_cache[layer_key] = []

        self.logger.info("=" * 60)
        self.logger.info(f"开始采集激活: {num_samples} 样本")
        self.logger.info(f"Hook 层: {self.config.hook_layers}")
        self.logger.info("=" * 60)

        # 采样循环
        for sample_idx in tqdm(range(num_samples), desc="采集激活"):
            # 采样 timestep
            timestep = self.sample_timestep()
            self.stats["timestep_distribution"].append(timestep)

            # 获取对应的 context (确保在 GPU 上)
            context = context_list[sample_idx % len(context_list)]
            if isinstance(context, list):
                context = context[0]
            # 确保 context 在正确设备上
            if context.device.type != "cuda":
                context = context.to(self.device)

            # 创建噪声 latent
            latent = self.create_noisy_latent(batch_size=1)

            # 单次 forward 多层 hook
            activations = self.single_forward_multi_hook(latent, context, timestep)

            # 空间 token 采样并存储
            for layer_key, activation in activations.items():
                # activation 已经是 [B, L, D] 形状 (B=1)
                # 如果需要，添加 batch 维度
                if activation.dim() == 2:
                    activation = activation.unsqueeze(0)

                sampled = self.spatial_token_sampling(
                    activation,  # [1, L, D]
                    self.config.tokens_per_sample
                )
                self._activation_cache[layer_key].append(sampled)
                self.stats["total_tokens_collected"] += sampled.shape[0]

            # 定期清理显存
            if (sample_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

        # 保存缓存
        self._save_cache(cache_dir)

        # 打印统计
        self._print_stats()

        # 返回合并后的激活
        result = {}
        for layer_key, tensors in self._activation_cache.items():
            result[layer_key] = torch.cat(tensors, dim=0)

        return result

    def _save_cache(self, cache_dir: Path):
        """
        保存激活缓存到磁盘

        格式: BF16/FP16, CPU tensor
        文件名: layer{idx}.pt (简洁格式)
        """
        self.logger.info(f"保存激活缓存到: {cache_dir}")

        save_dtype = torch.bfloat16 if self.config.save_dtype == "bf16" else torch.float16

        for layer_key, tensors in self._activation_cache.items():
            # 合并所有 token
            all_tokens = torch.cat(tensors, dim=0)

            # 转换精度并移到 CPU
            all_tokens = all_tokens.to(save_dtype).cpu()

            # 从 "block_out.layer14" 提取 "layer14" 作为文件名
            simple_key = layer_key.split(".")[-1] if "." in layer_key else layer_key
            save_path = cache_dir / f"{simple_key}.pt"
            torch.save(all_tokens, save_path)

            self.logger.info(f"  {simple_key}: {all_tokens.shape}, {all_tokens.dtype}")

        # 保存配置
        config_path = cache_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)

        # 保存统计
        stats_path = cache_dir / "stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)

    def _print_stats(self):
        """打印统计信息"""
        self.logger.info("=" * 60)
        self.logger.info("采集统计:")
        self.logger.info(f"  总 forward 次数: {self.stats['num_forwards']}")
        self.logger.info(f"  总 token 数: {self.stats['total_tokens_collected']}")

        # Timestep 分布
        ts = self.stats["timestep_distribution"]
        if ts:
            import numpy as np
            ts_arr = np.array(ts)
            self.logger.info(f"  Timestep 分布:")
            self.logger.info(f"    min: {ts_arr.min():.3f}")
            self.logger.info(f"    max: {ts_arr.max():.3f}")
            self.logger.info(f"    mean: {ts_arr.mean():.3f}")
            self.logger.info(f"    std: {ts_arr.std():.3f}")

        self.logger.info("=" * 60)

    def load_cache(self, cache_dir: str) -> Dict[str, torch.Tensor]:
        """加载缓存的激活"""
        cache_path = Path(cache_dir)
        activations = {}

        for layer in self.config.hook_layers:
            layer_key = f"layer{layer}"
            file_path = cache_path / f"{layer_key}.pt"
            if file_path.exists():
                activations[layer_key] = torch.load(file_path, map_location="cpu")
                self.logger.info(f"加载 {layer_key}: {activations[layer_key].shape}")

        return activations

    def cleanup(self):
        """清理资源"""
        del self.model
        del self.text_encoder
        self.model = None
        self.text_encoder = None
        self._prompt_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="SAE 激活采集")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Wan 模型路径")
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="提示词文件路径")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="激活缓存目录")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="采样样本数")
    parser.add_argument("--tokens_per_sample", type=int, default=512,
                        help="每样本 token 数")
    parser.add_argument("--device_id", type=int, default=0,
                        help="GPU 设备 ID")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()

    # 创建配置
    config = ActivationCollectorConfig(
        checkpoint_dir=args.checkpoint_dir,
        prompt_file=args.prompt_file,
        cache_dir=args.cache_dir,
        num_samples=args.num_samples,
        tokens_per_sample=args.tokens_per_sample,
        device_id=args.device_id,
        seed=args.seed,
    )

    # 创建采集器并执行
    collector = ActivationCollector(config)
    activations = collector.collect_activations()

    # 清理
    collector.cleanup()

    print("\n采集完成!")
    for layer_key, tensor in activations.items():
        print(f"  {layer_key}: {tensor.shape}")


if __name__ == "__main__":
    main()
