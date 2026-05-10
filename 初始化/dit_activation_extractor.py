"""
DiT Activation Extractor - 统一的激活提取框架

支持：
1. 初始化阶段：多 timestep 采样，early stop 支持
2. 训练阶段：单 timestep/batch 采样
3. 与 TokenSamplingManager 集成

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

# 导入现有的 hook 模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from wan.sae.hooking import HookMode, HookSpec, register_dit_hooks, remove_hooks, pack_hook_batch
from 初始化.token_sampling_manager import (
    TokenSamplingManager,
    TokenSamplingConfig,
    SamplingMode,
    ActivationStatisticsAnalyzer,
    create_init_sampler,
    create_train_sampler,
)


@dataclass
class DiTActivationConfig:
    """DiT Activation 提取配置"""

    # Wan2.1 1.3B 模型参数
    d_model: int = 1536
    num_layers: int = 30

    # Hook 配置
    hook_mode: HookMode = "block_out"
    hook_layers: List[int] = field(default_factory=lambda: [14, 19, 24, 29])

    # 时间步配置
    num_train_timesteps: int = 1000

    # 空间配置 (832x480, 81 frames)
    # VAE stride: (4, 8, 8) -> latent: (16, 11, 60, 104)
    # Patch size: (1, 2, 2) -> tokens: 11 * 30 * 52 = 17160
    vae_stride: Tuple[int, int, int] = (4, 8, 8)
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    default_resolution: Tuple[int, int, int] = (81, 480, 832)  # (F, H, W)

    # 采样配置
    max_tokens_per_hook: int = 100000  # 每个 hook 最大保存 token 数

    # Early Stop 支持
    support_early_stop: bool = True


class DiTActivationExtractor:
    """
    DiT Activation 提取器

    统一初始化和训练阶段的激活提取逻辑
    """

    def __init__(
        self,
        model,
        config: Optional[DiTActivationConfig] = None,
        sampling_config: Optional[TokenSamplingConfig] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.config = config or DiTActivationConfig()
        self.device = device

        # Hook 相关
        self._hook_handles: List = []
        self._current_activations: Dict[str, torch.Tensor] = {}
        self._current_timestep: Optional[int] = None
        self._early_stop_flag: bool = False

        # 采样器
        self.sampling_config = sampling_config
        self._sampler: Optional[TokenSamplingManager] = None

        # 累积激活 (用于初始化阶段)
        self._accumulated_activations: Dict[int, Dict[str, torch.Tensor]] = {}

    @property
    def sampler(self) -> TokenSamplingManager:
        """获取采样器 (懒加载)"""
        if self._sampler is None:
            if self.sampling_config is not None:
                self._sampler = TokenSamplingManager(self.sampling_config)
            else:
                # 默认初始化模式
                self._sampler = create_init_sampler()
        return self._sampler

    def compute_latent_shape(
        self,
        resolution: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[int, int, int, int]:
        """
        计算 latent shape

        返回: (C, F, H, W)
        """
        if resolution is None:
            F, H, W = self.config.default_resolution
        else:
            F, H, W = resolution

        vae_t, vae_h, vae_w = self.config.vae_stride

        lat_F = (F - 1) // vae_t + 1
        lat_H = H // vae_h
        lat_W = W // vae_w

        return (16, lat_F, lat_H, lat_W)  # VAE z_dim = 16

    def compute_num_tokens(
        self,
        latent_shape: Optional[Tuple[int, int, int, int]] = None,
    ) -> int:
        """计算 token 数量"""
        if latent_shape is None:
            latent_shape = self.compute_latent_shape()

        _, F, H, W = latent_shape
        p_t, p_h, p_w = self.config.patch_size

        # Patch embedding 后的 token 数
        n_tokens = (F // p_t) * (H // p_h) * (W // p_w)
        return n_tokens

    def register_hooks(self):
        """注册 forward hooks"""
        def on_tensor(key: str, tensor: torch.Tensor):
            self._current_activations[key] = tensor.detach()

        self._hook_handles = register_dit_hooks(
            model=self.model,
            hook_layers=self.config.hook_layers,
            hook_mode=self.config.hook_mode,
            on_tensor=on_tensor,
        )

    def remove_hooks(self):
        """移除 hooks"""
        remove_hooks(self._hook_handles)
        self._hook_handles = []

    def set_early_stop(self, flag: bool = True):
        """设置 early stop 标志"""
        self._early_stop_flag = flag

    def extract_at_timestep(
        self,
        latents: List[torch.Tensor],
        timestep: int,
        context: torch.Tensor,
        seq_len: int,
        context_lens: Optional[torch.Tensor] = None,
        clip_fea: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        在指定 timestep 提取激活

        参数:
            latents: 每个 sample 的 latent [C, F, H, W]
            timestep: 当前时间步
            context: 文本 embedding
            seq_len: 序列长度
            context_lens: context 长度
            clip_fea: CLIP 特征 (I2V 模式)

        返回:
            {layer_key: [N, C]} 每个 layer 的激活
        """
        self._current_activations = {}
        self._current_timestep = timestep

        # 构造 timestep tensor
        B = len(latents)
        t_tensor = torch.tensor([timestep] * B, device=self.device, dtype=torch.long)

        # Forward pass
        with torch.no_grad():
            try:
                self.model(
                    latents,
                    t=t_tensor,
                    context=context,
                    seq_len=seq_len,
                    clip_fea=clip_fea,
                )
            except Exception as e:
                if self._early_stop_flag and "early_stop" in str(e).lower():
                    pass  # 正常 early stop
                else:
                    raise

        # 打包激活
        packed = {}
        for key, tensor in self._current_activations.items():
            # tensor: [B, L, C] -> [B*L, C]
            if tensor.dim() == 3:
                B, L, C = tensor.shape
                packed[key] = tensor.reshape(B * L, C).cpu()
            else:
                packed[key] = tensor.cpu()

        return packed

    def extract_for_initialization(
        self,
        latents_list: List[List[torch.Tensor]],
        timesteps: List[int],
        context_list: List[torch.Tensor],
        seq_len: int,
        batch_size: int = 4,
        context_lens_list: Optional[List[torch.Tensor]] = None,
        clip_fea_list: Optional[List[torch.Tensor]] = None,
        sample_between_steps: bool = True,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        初始化阶段的多 timestep 激活提取

        参数:
            latents_list: 每个 batch 的 latent 列表
            timesteps: 要采样的时间步列表
            context_list: 每个 batch 的 context
            seq_len: 序列长度
            batch_size: 每个 batch 的 sample 数
            sample_between_steps: 是否在扩散步骤之间采样

        返回:
            {timestep: {layer_key: [N, C]}}
        """
        # 注册 hooks
        self.register_hooks()

        # 设置为初始化模式
        if self._sampler is None:
            self.sampling_config = TokenSamplingConfig(mode=SamplingMode.INIT)
        else:
            self.sampling_config.mode = SamplingMode.INIT

        results = {t: {} for t in timesteps}

        try:
            # 遍历 batch
            for batch_idx, latents in enumerate(latents_list):
                context = context_list[batch_idx]
                context_lens = context_lens_list[batch_idx] if context_lens_list else None
                clip_fea = clip_fea_list[batch_idx] if clip_fea_list else None

                # 遍历 timestep
                for t in timesteps:
                    activations = self.extract_at_timestep(
                        latents=latents,
                        timestep=t,
                        context=context,
                        seq_len=seq_len,
                        context_lens=context_lens,
                        clip_fea=clip_fea,
                    )

                    # 累积
                    for key, tensor in activations.items():
                        if key not in results[t]:
                            results[t][key] = []
                        results[t][key].append(tensor)

        finally:
            self.remove_hooks()

        # 合并
        for t in timesteps:
            for key in results[t]:
                results[t][key] = torch.cat(results[t][key], dim=0)

        return results

    def extract_for_training(
        self,
        latents: List[torch.Tensor],
        timestep: int,
        context: torch.Tensor,
        seq_len: int,
        context_lens: Optional[torch.Tensor] = None,
        clip_fea: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        训练阶段的单 batch 激活提取

        参数:
            latents: 当前 batch 的 latent
            timestep: 当前时间步
            context: 文本 embedding
            seq_len: 序列长度
            context_lens: context 长度
            clip_fea: CLIP 特征

        返回:
            {layer_key: [N, C]}
        """
        # 确保 hooks 已注册
        if not self._hook_handles:
            self.register_hooks()

        return self.extract_at_timestep(
            latents=latents,
            timestep=timestep,
            context=context,
            seq_len=seq_len,
            context_lens=context_lens,
            clip_fea=clip_fea,
        )

    def extract_with_sampling(
        self,
        latents: List[torch.Tensor],
        timestep: int,
        context: torch.Tensor,
        seq_len: int,
        grid_sizes: Optional[torch.Tensor] = None,
        target_tokens: int = 4096,
    ) -> Dict[str, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        提取激活并采样

        返回:
            {layer_key: (sampled_tokens, metadata)}
        """
        # 提取原始激活
        raw_activations = self.extract_for_training(
            latents=latents,
            timestep=timestep,
            context=context,
            seq_len=seq_len,
        )

        # 采样
        sampled = {}
        for key, act in raw_activations.items():
            if self.sampling_config.mode == SamplingMode.INIT:
                tokens, meta = self.sampler.sample(
                    activations=act,
                    timesteps=torch.tensor([timestep]),
                    grid_sizes=grid_sizes,
                )
            else:
                tokens, meta = self.sampler.sample_for_training(
                    activations=act,
                    timestep=timestep,
                )
            sampled[key] = (tokens, meta)

        return sampled


class EarlyStopException(Exception):
    """用于 early stop 的异常"""
    pass


class EarlyStopHook:
    """
    Early Stop Hook

    在初始化阶段，可以提前终止 forward pass
    """

    def __init__(
        self,
        extractor: DiTActivationExtractor,
        stop_at_layer: Optional[int] = None,
        stop_at_timestep: Optional[int] = None,
    ):
        self.extractor = extractor
        self.stop_at_layer = stop_at_layer
        self.stop_at_timestep = stop_at_timestep
        self._current_layer = 0

    def should_stop(self, layer_idx: int) -> bool:
        """判断是否应该停止"""
        if self.stop_at_layer is not None and layer_idx >= self.stop_at_layer:
            return True
        return False

    def __call__(self, layer_idx: int):
        """检查是否应该 early stop"""
        if self.should_stop(layer_idx):
            raise EarlyStopException(f"Early stop at layer {layer_idx}")


# ============================================================================
# 便捷函数
# ============================================================================

def compute_grid_sizes(
    latent_shape: Tuple[int, int, int, int],
    patch_size: Tuple[int, int, int] = (1, 2, 2),
) -> torch.Tensor:
    """
    计算网格尺寸

    返回: [F, H, W] 的 token 网格尺寸
    """
    _, F, H, W = latent_shape
    p_t, p_h, p_w = patch_size

    grid_F = F // p_t
    grid_H = H // p_h
    grid_W = W // p_w

    return torch.tensor([grid_F, grid_H, grid_W])


def analyze_activation_statistics(
    activations_by_layer: Dict[str, torch.Tensor],
    activations_by_timestep: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    grid_size: Optional[Tuple[int, int, int]] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    分析激活统计特性

    便捷函数，一次性运行所有分析
    """
    analyzer = ActivationStatisticsAnalyzer(device=device)

    # 1. 各层 norm 分布
    layer_results = {}
    for key, act in activations_by_layer.items():
        result = analyzer.analyze_token_norm_distribution(act, name=key)
        layer_results[key] = result

    # 2. 各层有效秩
    rank_results = {}
    for key, act in activations_by_layer.items():
        result = analyzer.analyze_effective_rank(act)
        rank_results[key] = result

    # 3. PCA 频谱
    spectrum_results = {}
    for key, act in activations_by_layer.items():
        result = analyzer.analyze_pca_spectrum(act, n_components=100)
        spectrum_results[key] = result

    # 4. 时间冗余 (如果提供)
    temporal_result = None
    if activations_by_timestep is not None:
        # 取第一个 layer
        first_layer = list(activations_by_layer.keys())[0]
        temporal_acts = {t: data[first_layer] for t, data in activations_by_timestep.items()}
        temporal_result = analyzer.analyze_temporal_redundancy(temporal_acts)

    # 5. 空间局部性 (如果提供 grid_size)
    spatial_result = None
    if grid_size is not None:
        first_layer = list(activations_by_layer.keys())[0]
        spatial_result = analyzer.analyze_spatial_locality(
            activations_by_layer[first_layer],
            grid_size=grid_size,
        )

    if verbose:
        analyzer.print_summary()

    return {
        "layer_norm_distribution": layer_results,
        "effective_rank": rank_results,
        "pca_spectrum": spectrum_results,
        "temporal_redundancy": temporal_result,
        "spatial_locality": spatial_result,
        "summary": analyzer.get_summary(),
    }


def extract_and_analyze_layer_activations(
    model,
    latents: List[torch.Tensor],
    timesteps: List[int],
    context: torch.Tensor,
    seq_len: int,
    hook_layers: List[int] = [14, 19, 24, 29],
    device: str = "cuda",
    verbose: bool = True,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[str, Any]]:
    """
    提取并分析各层激活

    返回:
        activations: {timestep: {layer_key: tensor}}
        analysis: 分析结果
    """
    config = DiTActivationConfig(hook_layers=hook_layers)
    extractor = DiTActivationExtractor(model, config=config, device=device)

    activations = extractor.extract_for_initialization(
        latents_list=[latents],
        timesteps=timesteps,
        context_list=[context],
        seq_len=seq_len,
    )

    # 分析
    analysis = analyze_activation_statistics(
        activations_by_layer=activations.get(timesteps[0], {}),
        activations_by_timestep=activations,
        device=device,
        verbose=verbose,
    )

    return activations, analysis
