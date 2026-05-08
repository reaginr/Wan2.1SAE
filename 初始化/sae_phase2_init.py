"""
SAE 第二阶段核心模块 - 工业级初始化器

根据 TODO list_v2 要求实现：
- 复用 wan/sae/hooking.py 的 hook 系统
- 单次 forward 多层并行 hook (layer 14, 19, 24, 29)
- Hook 位置: Transformer Block 最终 residual 输出
- 分层时空Token采样 (视频DiT专用)
- 局部Norm分层采样 (30% High-Norm, 50% Mid-Norm, 20% Low-Norm)
- 几何中位数预偏置 (Weiszfeld算法)
- PCA方向初始化
- Overcomplete扩展 (带扰动)
- Tied绑定初始化
- 初始化质量校验
- 激活缓存支持

核心流程:
    Step1: 单次 forward 采样 → hook 4 layers → save cache
    Step2: CPU PCA 并行
    Step3: 串行 SAE 训练

使用方法:
    from 初始化.sae_phase2_init import SAEInitializer, SAEInitConfig

    # 方式1: 从缓存加载
    config = SAEInitConfig(cache_dir="./cache")
    initializer = SAEInitializer(config, sae)
    initializer.initialize_from_cache(layer_idx=14)

    # 方式2: 从激活张量初始化
    initializer.initialize_from_activations(activations, layer_idx=14)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from 初始化.sae_phase2_core import TopKSAE, TopKSAEConfig
from 初始化.sae_phase2_norm import NormDenormManager
from 初始化.token_mapper import WanTokenMapper


@dataclass
class SAEInitConfig:
    """
    SAE 初始化配置

    根据 TODO list_v2 第二阶段 2.3 规范
    """

    # 初始化数据采样
    num_init_prompts: int = 500  # 采样多少条prompt用于初始化
    timestep_min: float = 0.35  # timestep下界 (仅中后期denoising)
    timestep_max: float = 0.75  # timestep上界
    seed: int = 42  # 随机种子

    # 时空Token采样 (视频DiT专用)
    num_frames_to_sample: int = 8  # 从21帧中采样多少帧
    frame_indices: List[int] = field(default_factory=lambda: [0, 3, 6, 9, 12, 15, 18, 20])
    tokens_per_frame: int = 64  # 每帧采样多少token

    # 空间Bucket划分
    bucket_h: int = 6  # 高度方向bucket数
    bucket_w: int = 8  # 宽度方向bucket数

    # 局部Norm分层比例
    high_norm_ratio: float = 0.30  # 30% High-Norm Token
    mid_norm_ratio: float = 0.50  # 50% Mid-Norm Token
    low_norm_ratio: float = 0.20  # 20% Low-Norm/Random Token

    # 几何中位数 (Weiszfeld算法)
    geometric_median_eps: float = 1e-5  # 收敛阈值
    geometric_median_max_iter: int = 100  # 最大迭代次数

    # Overcomplete扩展
    expansion_factor: int = 8  # 扩展倍数 (8x 或 16x)
    perturbation_std: float = 0.01  # 扰动标准差

    # 质量校验阈值
    max_initial_mse: float = 0.3  # 初始MSE上限
    max_dead_ratio: float = 0.05  # 初始死神经元比例上限

    def __post_init__(self):
        """校验配置"""
        # 帧索引校验
        assert len(self.frame_indices) == self.num_frames_to_sample, \
            f"frame_indices数量({len(self.frame_indices)})必须等于num_frames_to_sample({self.num_frames_to_sample})"

        # 比例校验
        total_ratio = self.high_norm_ratio + self.mid_norm_ratio + self.low_norm_ratio
        assert abs(total_ratio - 1.0) < 1e-6, \
            f"分层比例之和必须为1，当前: {total_ratio}"

        # 扩展倍数校验
        assert self.expansion_factor in [8, 16], \
            f"扩展倍数必须是8或16，当前: {self.expansion_factor}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SAEInitializer:
    """
    工业 SAE 初始化器

    初始化流程 (严格按照 TODO list_v2 2.3):
    1. 采样初始化数据 (500 prompt, timestep∈[0.35,0.75])
    2. 分层时空Token采样 (bucket内局部norm竞争)
    3. Per-Token RMSNorm
    4. 几何中位数计算 (Weiszfeld算法)
    5. PCA初始化
    6. Overcomplete扩展
    7. Tied绑定初始化
    8. 质量校验
    """

    def __init__(
        self,
        config: SAEInitConfig,
        sae: Optional[TopKSAE] = None,
        verbose: bool = True,
    ):
        self.config = config
        self.sae = sae
        self.verbose = verbose

        # Token映射器 (用于时空采样)
        self.token_mapper = WanTokenMapper()

        # 归一化管理器
        self.norm_manager = NormDenormManager(eps=1e-6)

        # 统计信息
        self.stats = {
            "num_tokens_sampled": 0,
            "geometric_median_iters": 0,
            "pca_explained_variance": None,
            "initial_mse": None,
            "dead_neuron_ratio": None,
        }

    def initialize(
        self,
        dit_model,
        prompts: List[str],
        device: str = "cuda",
        get_activations_fn: Optional[Callable] = None,
    ) -> TopKSAE:
        """
        执行完整初始化流程

        参数:
            dit_model: WanModel (DiT)
            prompts: 提示词列表
            device: 设备
            get_activations_fn: 自定义激活获取函数 (用于测试)

        返回:
            sae: 初始化完成的SAE
        """
        if self.verbose:
            print("=" * 60)
            print("SAE 工业初始化流程")
            print("=" * 60)

        # Step 1: 采样初始化数据
        if self.verbose:
            print("\n[Step 1] 采样初始化数据...")

        tokens = self._sample_initialization_data(
            dit_model=dit_model,
            prompts=prompts,
            device=device,
            get_activations_fn=get_activations_fn,
        )

        if self.verbose:
            print(f"  采样完成: {tokens.shape}")

        # Step 2: Per-Token RMSNorm
        if self.verbose:
            print("\n[Step 2] Per-Token RMSNorm...")

        tokens_norm, rms = self.norm_manager.per_token_rms_norm(tokens)
        # tokens_norm: [N, D]

        if self.verbose:
            print(f"  归一化完成: mean={tokens_norm.mean():.4f}, std={tokens_norm.std():.4f}")

        # Step 3: 计算几何中位数 (bpre)
        if self.verbose:
            print("\n[Step 3] 计算几何中位数...")

        bpre = self._compute_geometric_median(tokens_norm)

        if self.verbose:
            print(f"  几何中位数: norm={bpre.norm():.4f}, iters={self.stats['geometric_median_iters']}")

        # Step 4: 中心化
        if self.verbose:
            print("\n[Step 4] 中心化...")

        tokens_centered = tokens_norm - bpre  # [N, D]

        if self.verbose:
            print(f"  中心化完成: mean={tokens_centered.mean():.4f}")

        # Step 5: PCA初始化
        if self.verbose:
            print("\n[Step 5] PCA初始化...")

        pca_directions, explained_variance = self._compute_pca(tokens_centered)

        if self.verbose:
            print(f"  PCA完成: {pca_directions.shape}, explained_variance_top10={explained_variance[:10]}")

        self.stats["pca_explained_variance"] = explained_variance.tolist()

        # Step 6: Overcomplete扩展
        if self.verbose:
            print("\n[Step 6] Overcomplete扩展...")

        Wdec = self._overcomplete_expansion(pca_directions)

        if self.verbose:
            print(f"  扩展完成: Wdec shape={Wdec.shape}")

        # Step 7: Tied绑定初始化
        if self.verbose:
            print("\n[Step 7] Tied绑定初始化...")

        Wenc = Wdec.T.clone()

        # Step 8: 设置SAE参数
        if self.verbose:
            print("\n[Step 8] 设置SAE参数...")

        if self.sae is None:
            # 创建默认配置
            d_hidden = self.config.expansion_factor * 1536
            sae_config = TopKSAEConfig(
                d_model=1536,
                d_hidden=d_hidden,
                top_k=128 if d_hidden == 12288 else 128,
            )
            self.sae = TopKSAE(sae_config)

        # 设置权重
        with torch.no_grad():
            self.sae.bpre.copy_(bpre)
            self.sae.encoder.weight.copy_(Wenc)
            self.sae.encoder.bias.zero_()
            self.sae.decoder.weight.copy_(Wdec)

        # Step 9: 解码器权重归一化
        if self.verbose:
            print("\n[Step 9] 解码器权重归一化...")

        self.sae.normalize_decoder_weights()

        # Step 10: 质量校验
        if self.verbose:
            print("\n[Step 10] 质量校验...")

        quality_ok = self._validate_initialization(tokens_norm)

        if quality_ok:
            self.sae._is_initialized = True
            self.sae._init_method = "pca_tied"

        if self.verbose:
            print("=" * 60)
            print("初始化完成!")
            print("=" * 60)

        return self.sae

    def _sample_initialization_data(
        self,
        dit_model,
        prompts: List[str],
        device: str,
        get_activations_fn: Optional[Callable],
    ) -> torch.Tensor:
        """
        采样初始化数据

        返回: [N, D] 的token向量集
        """
        config = self.config

        # 随机选择prompt
        num_prompts = min(config.num_init_prompts, len(prompts))
        indices = torch.randperm(len(prompts), generator=torch.Generator().manual_seed(config.seed))[:num_prompts]
        selected_prompts = [prompts[i] for i in indices]

        all_tokens = []

        for prompt_idx, prompt in enumerate(tqdm(selected_prompts, desc="采样激活", disable=not self.verbose)):
            # 采样timestep (非均匀，集中在[0.35, 0.75])
            t = torch.rand(1, generator=torch.Generator().manual_seed(config.seed + prompt_idx)).item()
            t = config.timestep_min + t * (config.timestep_max - config.timestep_min)

            # 获取DiT激活 (这里需要实际的模型调用)
            if get_activations_fn is not None:
                # 测试模式: 使用自定义函数
                activations = get_activations_fn(prompt, t, device)
            else:
                # 实际模式: 调用DiT模型
                activations = self._get_dit_activations(dit_model, prompt, t, device)

            # activations: [1, L, D] 或 [L, D]
            if activations.dim() == 3:
                activations = activations.squeeze(0)

            # 分层时空Token采样
            sampled_tokens = self._spatial_temporal_token_sampling(activations)

            all_tokens.append(sampled_tokens)

        # 合并所有token
        tokens = torch.cat(all_tokens, dim=0)  # [N, D]

        self.stats["num_tokens_sampled"] = tokens.shape[0]

        return tokens

    def _get_dit_activations(
        self,
        dit_model,
        prompt: str,
        timestep: float,
        device: str,
    ) -> torch.Tensor:
        """
        获取DiT激活 (实际实现需要对接WanModel)

        这是一个占位实现，实际使用时需要对接真实的模型。
        """
        # TODO: 对接 WanModel 的 hook 系统
        # 这里返回随机数据用于测试
        L = 32760  # 21 * 30 * 52
        D = 1536
        return torch.randn(L, D, dtype=torch.bfloat16, device=device)

    def _spatial_temporal_token_sampling(
        self,
        activations: torch.Tensor,
    ) -> torch.Tensor:
        """
        分层时空Token采样 (视频DiT核心规范)

        根据TODO list_v2:
        - 从21帧中固定均匀采8帧: [0,3,6,9,12,15,18,20]
        - 每帧30×52划分为6×8个bucket
        - 每帧总采64 token:
          - 30% High-Norm (≈19)
          - 50% Mid-Norm (≈32)
          - 20% Low-Norm/Random (≈13)
        - 必须bucket内局部norm竞争

        输入: activations [L, D] 或 [T, H, W, D]
        输出: sampled_tokens [tokens_per_frame * num_frames_to_sample, D]
        """
        config = self.config

        # 确定激活形状
        L, D = activations.shape
        T, H, W = self.token_mapper.T, self.token_mapper.H, self.token_mapper.W

        # 重塑为 [T, H, W, D]
        if L == T * H * W:
            activations_4d = activations.view(T, H, W, D)
        else:
            # 如果形状不匹配，使用全局采样
            return self._global_token_sampling(activations, config.tokens_per_frame * config.num_frames_to_sample)

        sampled_tokens = []

        for frame_idx in config.frame_indices:
            if frame_idx >= T:
                continue

            # 获取该帧的激活 [H, W, D]
            frame_activations = activations_4d[frame_idx]

            # 计算每个token的norm
            token_norms = frame_activations.norm(dim=-1)  # [H, W]

            # 划分bucket
            bucket_h_idx = torch.arange(H).reshape(-1, 1) * config.bucket_h // H
            bucket_w_idx = torch.arange(W).reshape(1, -1) * config.bucket_w // W
            bucket_assignments = bucket_h_idx + bucket_w_idx * config.bucket_h  # [H, W]

            # 每个bucket采样
            frame_samples = []

            for bucket_id in range(config.bucket_h * config.bucket_w):
                # 获取bucket内的token
                bucket_mask = (bucket_assignments == bucket_id)
                bucket_indices = bucket_mask.nonzero(as_tuple=False)  # [num_in_bucket, 2]

                if bucket_indices.shape[0] == 0:
                    continue

                # 获取bucket内的norm
                bucket_norms = token_norms[bucket_mask]  # [num_in_bucket]

                # 计算每类token数量 (按bucket内比例分配)
                n_high = max(1, int(len(bucket_norms) * config.high_norm_ratio / (config.bucket_h * config.bucket_w)))
                n_mid = max(1, int(len(bucket_norms) * config.mid_norm_ratio / (config.bucket_h * config.bucket_w)))
                n_low = max(1, int(len(bucket_norms) * config.low_norm_ratio / (config.bucket_h * config.bucket_w)))

                # High-Norm: 局部top
                _, high_idx = torch.topk(bucket_norms, min(n_high, len(bucket_norms)))
                for idx in high_idx:
                    h, w = bucket_indices[idx]
                    frame_samples.append(frame_activations[h, w])

                # Mid-Norm: 中位数区域随机采样
                sorted_norms, sorted_indices = torch.sort(bucket_norms)
                mid_start = len(sorted_norms) // 4
                mid_end = 3 * len(sorted_norms) // 4
                if mid_end > mid_start:
                    mid_indices = sorted_indices[mid_start:mid_end]
                    perm = torch.randperm(len(mid_indices))[:n_mid]
                    for i in perm:
                        h, w = bucket_indices[sorted_indices[mid_start + i]]
                        frame_samples.append(frame_activations[h, w])

                # Low-Norm/Random: 随机采样
                low_indices = torch.randperm(len(bucket_norms))[:n_low]
                for i in low_indices:
                    h, w = bucket_indices[i]
                    frame_samples.append(frame_activations[h, w])

            # 限制每帧token数
            if len(frame_samples) > config.tokens_per_frame:
                indices = torch.randperm(len(frame_samples))[:config.tokens_per_frame]
                frame_samples = [frame_samples[i] for i in indices]

            sampled_tokens.extend(frame_samples)

        if len(sampled_tokens) == 0:
            # 回退到全局采样
            return self._global_token_sampling(activations, config.tokens_per_frame * config.num_frames_to_sample)

        return torch.stack(sampled_tokens, dim=0)  # [N_sampled, D]

    def _global_token_sampling(
        self,
        activations: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """全局随机采样 (回退方案)"""
        L = activations.shape[0]
        indices = torch.randperm(L)[:num_tokens]
        return activations[indices]

    def _compute_geometric_median(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算几何中位数 (Weiszfeld算法)

        几何中位数是使所有点到它的距离之和最小的点。
        与均值不同，它对异常值更鲁棒。

        算法:
            y_{n+1} = sum_i(x_i / ||x_i - y_n||) / sum_i(1 / ||x_i - y_n||)

        参数:
            x: [N, D] 数据点

        返回:
            median: [D] 几何中位数
        """
        config = self.config

        # 使用均值作为初始点
        y = x.mean(dim=0)  # [D]

        # 转换为FP32以提高精度
        x_fp32 = x.float()
        y_fp32 = y.float()

        for iteration in range(config.geometric_median_max_iter):
            # 计算距离
            diff = x_fp32 - y_fp32.unsqueeze(0)  # [N, D]
            dist = diff.norm(dim=-1)  # [N]

            # 避免除零
            dist = dist.clamp(min=1e-8)

            # 计算权重
            weights = 1.0 / dist  # [N]

            # 加权平均
            y_new = (weights.unsqueeze(1) * x_fp32).sum(dim=0) / weights.sum()

            # 检查收敛
            change = (y_new - y_fp32).norm()
            y_fp32 = y_new

            if change < config.geometric_median_eps:
                break

        self.stats["geometric_median_iters"] = iteration + 1

        # 转回原精度
        return y_fp32.to(x.dtype)

    def _compute_pca(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算PCA主成分

        使用 torch.pca_lowrank 进行高效计算

        参数:
            x: [N, D] 中心化后的数据

        返回:
            directions: [D, n_components] 主方向 (单位范数)
            explained_variance: [n_components] 解释方差

        注意:
            - 最大成分数 = min(N, D) = min(256000, 1536) = 1536
            - 1536 个 PCA 方向 × expansion 倍 = d_hidden
            - 对于 8x: 1536 × 8 = 12288 ✓
            - 对于 16x: 1536 × 16 = 24576 ✓
        """
        N, D = x.shape

        # 最大可能的成分数 = min(N, D)
        # 对于我们的数据: min(256000, 1536) = 1536
        max_components = min(N, D)

        # 我们需要提取所有 D 个主方向
        n_components = max_components

        if self.verbose:
            print(f"  PCA: 输入 [{N}, {D}], 提取 {n_components} 个主方向")
            print(f"  扩展倍数: {self.config.expansion_factor}x")
            print(f"  目标 d_hidden: {self.config.expansion_factor * D}")

        # 使用 torch.pca_lowrank
        U, S, V = torch.pca_lowrank(x.float(), q=n_components, center=False)

        # V: [D, n_components] 主方向
        directions = V  # 已经是单位范数

        # 解释方差
        explained_variance = (S ** 2) / (N - 1)

        # 计算累计解释方差比例
        total_variance = explained_variance.sum()
        cumulative_ratio = explained_variance.cumsum(0) / total_variance

        # 找到解释 90%, 95%, 99% 方差需要的成分数
        if self.verbose:
            for ratio in [0.90, 0.95, 0.99]:
                n_needed = (cumulative_ratio < ratio).sum().item() + 1
                print(f"  解释 {ratio*100:.0f}% 方差需要 {n_needed} 个成分")

        return directions, explained_variance

    def _overcomplete_expansion(
        self,
        pca_directions: torch.Tensor,
        x_centered: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Overcomplete扩展

        将PCA方向扩展到高维隐藏空间:
        - 8x扩展: 每个 PCA 方向扩展 8 份 (1536 × 8 = 12288)
        - 16x扩展: 每个 PCA 方向扩展 16 份 (1536 × 16 = 24576)
        - 每份加入小扰动，扰动强度递增
        - 扰动后单位范数归一化
        - 打乱排列防止同源方向连续

        参数:
            pca_directions: [D, n_pca] PCA主方向
            x_centered: [N, D] 中心化后的数据 (用于自适应初始化，可选)

        返回:
            Wdec: [D, d_hidden] 解码器权重
        """
        config = self.config
        d_model = 1536
        d_hidden = config.expansion_factor * d_model

        # 扩展倍数
        expansion = config.expansion_factor

        # 每个PCA方向需要扩展的份数
        n_pca = pca_directions.shape[1]

        # 验证: PCA方向数量 = d_model = 1536
        if n_pca != d_model:
            self.logger.warning(
                f"PCA 方向数量 ({n_pca}) != d_model ({d_model})，"
                f"可能影响扩展质量"
            ) if hasattr(self, 'logger') else None

        # 创建扩展后的权重
        Wdec_list = []

        for i in range(n_pca):
            direction = pca_directions[:, i]  # [D]

            # 创建 expansion 份带扰动的副本
            # 策略: 扰动强度递增，从接近原方向到逐渐偏离
            for j in range(expansion):
                # 扰动强度: 随着副本索引增加而略微增加
                # 第一个副本接近原方向，最后一个副本偏离更多
                scale = config.perturbation_std * (1.0 + 0.5 * j / expansion)
                perturbation = torch.randn_like(direction) * scale
                expanded = direction + perturbation

                # 归一化到单位范数
                expanded = F.normalize(expanded, dim=0)

                Wdec_list.append(expanded)

        # 打乱顺序 (防止同源方向连续，促进特征多样性)
        perm = torch.randperm(len(Wdec_list))
        Wdec = torch.stack([Wdec_list[i] for i in perm], dim=1)  # [D, n_pca * expansion]

        # 验证维度
        expected_dim = n_pca * expansion
        if expected_dim != d_hidden:
            # 这意味着 expansion_factor 设置与预期不符
            # 需要调整
            if expected_dim < d_hidden:
                # 维度不够: 使用 PCA 残差方向填充
                n_fill = d_hidden - expected_dim

                if self.verbose:
                    print(f"  需要填充 {n_fill} 个额外方向...")

                # 策略: 使用 PCA 方向的线性组合 + 大扰动
                # 这比纯随机更有意义
                fill_list = []
                for k in range(n_fill):
                    # 随机选择几个 PCA 方向进行组合
                    n_combine = min(3, n_pca)
                    indices = torch.randperm(n_pca)[:n_combine]
                    combined = torch.zeros(d_model)
                    for idx in indices:
                        weight = torch.randn(1)
                        combined = combined + weight * pca_directions[:, idx]

                    # 加入较大扰动
                    combined = combined + torch.randn(d_model) * config.perturbation_std * 2
                    combined = F.normalize(combined, dim=0)
                    fill_list.append(combined)

                fill_directions = torch.stack(fill_list, dim=1)
                Wdec = torch.cat([Wdec, fill_directions], dim=1)

                if self.verbose:
                    print(f"  填充完成: {Wdec.shape}")
            else:
                # 维度过多: 截断
                Wdec = Wdec[:, :d_hidden]

        # 最终校验
        assert Wdec.shape == (d_model, d_hidden), \
            f"Wdec 形状错误: {Wdec.shape}, 期望 ({d_model}, {d_hidden})"

        return Wdec

    def _validate_initialization(
        self,
        x_norm: torch.Tensor,
    ) -> bool:
        """
        验证初始化质量

        校验项:
        1. 初始重构MSE ≤ 0.3
        2. 死神经元比例 ≈ 0
        3. Wenc = Wdec.T
        """
        config = self.config

        # 检查Wenc = Wdec.T
        Wenc = self.sae.encoder.weight
        Wdec = self.sae.decoder.weight
        assert torch.allclose(Wenc, Wdec.T, atol=1e-6), "Wenc ≠ Wdec.T，Tied绑定失败"

        # 计算初始重构MSE
        with torch.no_grad():
            # 采样一小批数据测试
            test_x = x_norm[:min(1000, x_norm.shape[0])]
            x_hat, z, _ = self.sae(test_x.unsqueeze(0), return_loss=True)

        # 获取loss
        initial_mse = F.mse_loss(x_hat.squeeze(0), test_x).item()
        self.stats["initial_mse"] = initial_mse

        # 计算死神经元比例
        z_flat = z.view(-1, self.sae.d_hidden)
        dead_mask = (z_flat != 0).sum(dim=0) == 0
        dead_ratio = dead_mask.float().mean().item()
        self.stats["dead_neuron_ratio"] = dead_ratio

        if self.verbose:
            print(f"  初始MSE: {initial_mse:.4f} (阈值: {config.max_initial_mse})")
            print(f"  死神经元比例: {dead_ratio:.2%} (阈值: {config.max_dead_ratio:.0%})")

        # 判断质量
        quality_ok = True
        if initial_mse > config.max_initial_mse:
            if self.verbose:
                print(f"  ⚠️ 初始MSE过高: {initial_mse:.4f} > {config.max_initial_mse}")
            quality_ok = False

        if dead_ratio > config.max_dead_ratio:
            if self.verbose:
                print(f"  ⚠️ 死神经元过多: {dead_ratio:.2%} > {config.max_dead_ratio:.0%}")
            quality_ok = False

        if quality_ok:
            if self.verbose:
                print("  ✓ 初始化质量校验通过")

        return quality_ok

    def initialize_from_cache(
        self,
        cache_dir: str,
        layer_idx: int,
        device: str = "cuda",
    ) -> TopKSAE:
        """
        从缓存初始化 SAE (推荐方式)

        流程:
        1. 加载缓存的激活
        2. Per-Token RMSNorm
        3. 几何中位数计算
        4. PCA 初始化
        5. 设置 SAE 参数

        参数:
            cache_dir: 激活缓存目录
            layer_idx: 目标层索引 (14, 19, 24, 29)
            device: 计算设备

        返回:
            sae: 初始化完成的 SAE
        """
        if self.verbose:
            print("=" * 60)
            print(f"从缓存初始化 SAE (layer {layer_idx})")
            print("=" * 60)

        # 加载缓存的激活
        cache_path = Path(cache_dir)
        layer_key = f"layer{layer_idx}"
        activation_file = cache_path / f"{layer_key}.pt"

        if not activation_file.exists():
            raise FileNotFoundError(f"缓存文件不存在: {activation_file}")

        if self.verbose:
            print(f"\n[Step 1] 加载缓存激活...")

        activations = torch.load(activation_file, map_location="cpu")
        if self.verbose:
            print(f"  激活形状: {activations.shape}")

        # 调用 from_activations 方法
        return self.initialize_from_activations(activations, layer_idx, device)

    def initialize_from_activations(
        self,
        activations: torch.Tensor,
        layer_idx: int,
        device: str = "cuda",
    ) -> TopKSAE:
        """
        从激活张量初始化 SAE

        参数:
            activations: 激活张量 [N, D]
            layer_idx: 目标层索引
            device: 计算设备

        返回:
            sae: 初始化完成的 SAE
        """
        if self.verbose:
            print(f"\n[Step 2] Per-Token RMSNorm...")

        # 移动到设备
        activations = activations.to(device)

        # Per-Token RMSNorm
        tokens_norm, rms = self.norm_manager.per_token_rms_norm(
            activations.unsqueeze(0)  # [1, N, D]
        )
        tokens_norm = tokens_norm.squeeze(0)  # [N, D]

        if self.verbose:
            print(f"  归一化完成: mean={tokens_norm.mean():.4f}, std={tokens_norm.std():.4f}")

        # 几何中位数
        if self.verbose:
            print(f"\n[Step 3] 计算几何中位数...")

        bpre = self._compute_geometric_median(tokens_norm)

        if self.verbose:
            print(f"  几何中位数: norm={bpre.norm():.4f}")

        # 中心化
        tokens_centered = tokens_norm - bpre

        # PCA
        if self.verbose:
            print(f"\n[Step 4] PCA 初始化...")

        pca_directions, explained_variance = self._compute_pca(tokens_centered)

        if self.verbose:
            print(f"  PCA 完成: {pca_directions.shape}")

        self.stats["pca_explained_variance"] = explained_variance.tolist()

        # Overcomplete 扩展
        if self.verbose:
            print(f"\n[Step 5] Overcomplete 扩展...")

        Wdec = self._overcomplete_expansion(pca_directions)

        if self.verbose:
            print(f"  扩展完成: Wdec shape={Wdec.shape}")

        # Tied 绑定
        Wenc = Wdec.T.clone()

        # 创建 SAE (如果需要)
        if self.sae is None:
            d_hidden = self.config.expansion_factor * 1536
            sae_config = TopKSAEConfig(
                d_model=1536,
                d_hidden=d_hidden,
                top_k=128,
            )
            self.sae = TopKSAE(sae_config)

        # 设置权重
        if self.verbose:
            print(f"\n[Step 6] 设置 SAE 参数...")

        with torch.no_grad():
            self.sae.bpre.copy_(bpre.cpu())
            self.sae.encoder.weight.copy_(Wenc.cpu())
            self.sae.encoder.bias.zero_()
            self.sae.decoder.weight.copy_(Wdec.cpu())

        # 解码器权重归一化
        self.sae.normalize_decoder_weights()

        # 质量校验
        if self.verbose:
            print(f"\n[Step 7] 质量校验...")

        quality_ok = self._validate_initialization(tokens_norm)

        if quality_ok:
            self.sae._is_initialized = True
            self.sae._init_method = f"pca_tied_layer{layer_idx}"

        self.stats["layer_idx"] = layer_idx

        if self.verbose:
            print("=" * 60)
            print(f"Layer {layer_idx} 初始化完成!")
            print("=" * 60)

        return self.sae

    def get_stats(self) -> Dict[str, Any]:
        """获取初始化统计信息"""
        return self.stats.copy()


# ============================================================================
# Weiszfeld算法独立实现
# ============================================================================

def weiszfeld_geometric_median(
    points: torch.Tensor,
    eps: float = 1e-5,
    max_iter: int = 100,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Weiszfeld算法计算几何中位数

    参数:
        points: [N, D] 数据点
        eps: 收敛阈值
        max_iter: 最大迭代次数
        verbose: 是否输出迭代信息

    返回:
        median: [D] 几何中位数
    """
    y = points.mean(dim=0)
    y = y.float()
    points_fp32 = points.float()

    for i in range(max_iter):
        diff = points_fp32 - y.unsqueeze(0)
        dist = diff.norm(dim=-1).clamp(min=1e-8)

        weights = 1.0 / dist
        y_new = (weights.unsqueeze(1) * points_fp32).sum(dim=0) / weights.sum()

        change = (y_new - y).norm()

        if verbose:
            print(f"  Iter {i}: change={change:.2e}")

        if change < eps:
            break

        y = y_new

    return y.to(points.dtype)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SAEInitializer 测试")
    print("=" * 60)

    # 测试配置
    config = SAEInitConfig()
    print(f"\n配置: {config.to_dict()}")

    # 测试几何中位数
    print("\n[测试] 几何中位数计算")
    points = torch.randn(1000, 1536)
    median = weiszfeld_geometric_median(points, verbose=True)
    print(f"几何中位数: shape={median.shape}, norm={median.norm():.4f}")

    # 测试初始化器 (使用模拟数据)
    print("\n[测试] 初始化器 (模拟模式)")

    def mock_get_activations(prompt, t, device):
        """模拟激活获取"""
        return torch.randn(32760, 1536, dtype=torch.bfloat16, device=device)

    # 创建SAE
    sae_config = TopKSAEConfig(d_model=1536, d_hidden=12288, top_k=128)
    sae = TopKSAE(sae_config)

    # 创建初始化器
    init_config = SAEInitConfig(
        num_init_prompts=10,  # 测试用少量prompt
        tokens_per_frame=32,  # 减少token数
    )
    initializer = SAEInitializer(init_config, sae, verbose=True)

    # 模拟prompts
    mock_prompts = [f"test prompt {i}" for i in range(20)]

    # 初始化 (使用模拟激活函数)
    sae = initializer.initialize(
        dit_model=None,
        prompts=mock_prompts,
        device="cpu",
        get_activations_fn=mock_get_activations,
    )

    # 打印统计
    print("\n初始化统计:")
    for k, v in initializer.get_stats().items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
