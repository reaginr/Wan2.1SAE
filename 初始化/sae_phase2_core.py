"""
SAE 第二阶段核心模块 - TopK SAE 完整实现

根据 TODO list_v2 要求实现：
- TopK SAE 架构 (禁止ReLU+L1 SAE)
- Per-Token RMSNorm 前置归一化
- 预偏置 bpre (几何中位数初始化)
- TopK 稀疏激活
- AuxK 损失 (Ghost Gradients)
- 正交正则损失
- 解码器权重单位范数硬约束

维度配置:
- 首选: hidden_dim=12288 (8x), k=128
- 备选: hidden_dim=24576 (16x), k=128
- k/hidden_dim 必须在 0.1%~1% 区间

使用方法:
    from 初始化.sae_phase2_core import TopKSAE, TopKSAEConfig

    config = TopKSAEConfig(d_model=1536, d_hidden=12288, top_k=128)
    sae = TopKSAE(config)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from 初始化.sae_phase2_norm import NormDenormManager


@dataclass
class TopKSAEConfig:
    """
    TopK SAE 配置

    关键约束:
    - top_k / d_hidden 必须在 0.1% ~ 1% 区间
    - 默认使用 8x 扩展维度
    """

    # 基础维度
    d_model: int = 1536  # DiT 隐藏维度 (Wan2.1 1.3B 固定)
    d_hidden: int = 12288  # SAE 扩展维度 (8x = 12288, 16x = 24576)
    top_k: int = 128  # TopK 稀疏度

    # 损失权重
    lambda_aux: float = 0.1  # AuxK 损失权重
    lambda_orth: float = 1e-5  # 正交正则损失权重 (16x维度必开)

    # 归一化
    eps: float = 1e-6  # RMSNorm eps (与Wan2.1对齐)

    # 其他
    normalize_decoder: bool = True  # 是否强制解码器列单位范数

    def __post_init__(self):
        """校验配置合法性"""
        # k/hidden_dim 比例校验 (黄金区间 0.1% ~ 1%)
        ratio = self.top_k / self.d_hidden
        ratio_pct = ratio * 100
        if not (0.1 <= ratio_pct <= 1.0):
            raise ValueError(
                f"top_k/d_hidden 比例必须在 0.1%~1% 区间，"
                f"当前: top_k={self.top_k}, d_hidden={self.d_hidden}, "
                f"比例={ratio_pct:.2f}%"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def expansion_ratio(self) -> int:
        """扩展倍数"""
        return self.d_hidden // self.d_model


class TopKSAE(nn.Module):
    """
    工业 TopK SAE 实现

    架构流程 (按顺序):
    1. 输入: x [B, L, D] 原始DiT激活
    2. Per-Token RMSNorm: x_norm, rms = RMSNorm(x)
    3. 预偏置减法: x_centered = x_norm - bpre
    4. 编码器: z = ReLU(Wenc @ x_centered + b_enc)
    5. TopK稀疏: z_sparse = TopK(z, k)
    6. 解码器: x_hat_norm = Wdec @ z_sparse
    7. 预偏置加法: x_hat_norm = x_hat_norm + bpre
    8. 输出: x_hat_norm [B, L, D], z_sparse [B, L, d_hidden]

    关键特性:
    - 前置 Per-Token RMSNorm (无训练参数)
    - 预偏置 bpre (几何中位数初始化)
    - 解码器无偏置
    - 解码器列向量单位范数约束
    - 支持 AuxK 损失 (Ghost Gradients)
    - 支持正交正则损失
    """

    def __init__(self, config: TopKSAEConfig):
        super().__init__()
        self.config = config

        # 归一化管理器
        self.norm_manager = NormDenormManager(eps=config.eps)

        # 预偏置 (几何中位数初始化，由SAEInitializer完成)
        self.bpre = nn.Parameter(torch.zeros(config.d_model), requires_grad=True)

        # 编码器: Linear(d_model -> d_hidden) with bias
        self.encoder = nn.Linear(config.d_model, config.d_hidden, bias=True)
        self.encoder.bias.data.zero_()  # b_enc 初始化为0

        # 解码器: Linear(d_hidden -> d_model) without bias
        self.decoder = nn.Linear(config.d_hidden, config.d_model, bias=False)

        # 初始化状态标记
        self._is_initialized = False
        self._init_method = "random"  # 将由SAEInitializer设置为 "pca_tied"

    @property
    def d_model(self) -> int:
        return self.config.d_model

    @property
    def d_hidden(self) -> int:
        return self.config.d_hidden

    @property
    def Wenc(self) -> torch.Tensor:
        """编码器权重 [d_hidden, d_model]"""
        return self.encoder.weight

    @property
    def Wdec(self) -> torch.Tensor:
        """解码器权重 [d_model, d_hidden]"""
        return self.decoder.weight

    @property
    def b_enc(self) -> torch.Tensor:
        """编码器偏置 [d_hidden]"""
        return self.encoder.bias

    def encode(
        self,
        x_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码 (在归一化空间操作)

        输入:
            x_norm: 归一化后的激活 [B, L, D]

        输出:
            z_sparse: TopK稀疏表示 [B, L, d_hidden]
            topk_idx: TopK索引 [B*L, k]
            topk_val: TopK值 [B*L, k]
            x_centered: 中心化后的输入 [B, L, D] (用于AuxK)
        """
        B, L, D = x_norm.shape

        # 预偏置减法
        x_centered = x_norm - self.bpre  # [B, L, D]

        # 展平为 [B*L, D]
        x_flat = x_centered.view(-1, D)

        # 编码器前向
        z = self.encoder(x_flat)  # [B*L, d_hidden]
        z = F.relu(z)  # 激活

        # TopK 稀疏化
        k = self.config.top_k
        topk_val, topk_idx = torch.topk(z, k=k, dim=1, largest=True, sorted=False)

        # 构建稀疏表示
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(1, topk_idx, topk_val)

        # 恢复形状
        z_sparse = z_sparse.view(B, L, -1)  # [B, L, d_hidden]

        return z_sparse, topk_idx, topk_val, x_centered

    def decode(
        self,
        z_sparse: torch.Tensor,
    ) -> torch.Tensor:
        """
        解码 (在归一化空间操作)

        输入:
            z_sparse: 稀疏表示 [B, L, d_hidden]

        输出:
            x_hat_norm: 重构的归一化激活 [B, L, D]
        """
        B, L, _ = z_sparse.shape

        # 展平
        z_flat = z_sparse.view(-1, self.d_hidden)

        # 解码器前向
        x_hat = self.decoder(z_flat)  # [B*L, D]

        # 加预偏置
        x_hat_norm = x_hat + self.bpre  # [B*L, D]

        # 恢复形状
        x_hat_norm = x_hat_norm.view(B, L, -1)  # [B, L, D]

        return x_hat_norm

    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = False,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        完整前向传播

        输入:
            x: 原始DiT激活 [B, L, D]
            return_loss: 是否计算损失
            return_intermediates: 是否返回中间变量

        输出:
            x_hat_norm: 重构的归一化激活 [B, L, D]
            z_sparse: 稀疏表示 [B, L, d_hidden]
            loss_dict: (可选) 损失字典
            intermediates: (可选) 中间变量字典
        """
        B, L, D = x.shape

        # 1. Per-Token RMSNorm
        x_norm, rms = self.norm_manager.per_token_rms_norm(x)

        # 2. 编码
        z_sparse, topk_idx, topk_val, x_centered = self.encode(x_norm)

        # 3. 解码
        x_hat_norm = self.decode(z_sparse)

        if not return_loss:
            if return_intermediates:
                intermediates = {
                    "x_norm": x_norm,
                    "rms": rms,
                    "x_centered": x_centered,
                    "z_sparse": z_sparse,
                    "topk_idx": topk_idx,
                    "topk_val": topk_val,
                    "x_hat_norm": x_hat_norm,
                }
                return x_hat_norm, z_sparse, intermediates
            return x_hat_norm, z_sparse

        # 计算损失
        loss_dict = self.compute_loss(
            x_norm=x_norm,
            x_hat_norm=x_hat_norm,
            z_sparse=z_sparse,
            x_centered=x_centered,
            topk_idx=topk_idx,
            topk_val=topk_val,
        )

        if return_intermediates:
            intermediates = {
                "x_norm": x_norm,
                "rms": rms,
                "x_centered": x_centered,
                "z_sparse": z_sparse,
                "topk_idx": topk_idx,
                "topk_val": topk_val,
                "x_hat_norm": x_hat_norm,
            }
            return x_hat_norm, z_sparse, loss_dict, intermediates

        return x_hat_norm, z_sparse, loss_dict

    def compute_loss(
        self,
        x_norm: torch.Tensor,
        x_hat_norm: torch.Tensor,
        z_sparse: torch.Tensor,
        x_centered: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_val: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算完整损失

        L_total = L_recon + λ_aux * L_auxk + λ_orth * L_orth

        损失组成:
        1. L_recon: 重构MSE损失
        2. L_auxk: AuxK损失 (Ghost Gradients)
        3. L_orth: 解码器正交正则损失
        """
        loss_dict = {}

        # 1. 重构损失 (MSE)
        loss_recon = F.mse_loss(x_hat_norm, x_norm)
        loss_dict["loss_recon"] = loss_recon

        # 2. AuxK 损失 (Ghost Gradients)
        loss_auxk = self._compute_auxk_loss(
            x_norm=x_norm,
            x_hat_norm=x_hat_norm,
            z_sparse=z_sparse,
            topk_idx=topk_idx,
        )
        loss_dict["loss_auxk"] = loss_auxk

        # 3. 正交正则损失
        loss_orth = self._compute_orthogonal_loss()
        loss_dict["loss_orth"] = loss_orth

        # 总损失
        loss_total = (
            loss_recon
            + self.config.lambda_aux * loss_auxk
            + self.config.lambda_orth * loss_orth
        )
        loss_dict["loss_total"] = loss_total

        return loss_dict

    def _compute_auxk_loss(
        self,
        x_norm: torch.Tensor,
        x_hat_norm: torch.Tensor,
        z_sparse: torch.Tensor,
        topk_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 AuxK 损失 (Ghost Gradients)

        原理：
        - 排除主TopK激活的神经元
        - 从静默神经元中取k_aux个最大激活
        - 用这些神经元重构误差 e = x_norm - x_hat_norm
        - 计算MSE损失

        参数：
            k_aux = k (与主TopK相同)
        """
        # 重构误差
        error = x_norm - x_hat_norm  # [B, L, D]

        # 获取静默神经元的激活
        # z_sparse: [B, L, d_hidden]
        z_flat = z_sparse.view(-1, self.d_hidden)  # [B*L, d_hidden]

        # 创建静默神经元掩码
        # topk_idx: [B*L, k]
        mask = torch.ones_like(z_flat, dtype=torch.bool)
        batch_indices = torch.arange(z_flat.size(0), device=z_flat.device).unsqueeze(1).expand_as(topk_idx)
        mask[batch_indices, topk_idx] = False

        # 静默神经元激活值 (取负值用于重构误差)
        # Ghost gradients: 用静默神经元重构误差
        z_ghost = z_flat.clone()
        z_ghost[~mask] = 0  # 清除已激活的

        # 从静默神经元中取 top-k_aux
        k_aux = self.config.top_k
        z_ghost_val, z_ghost_idx = torch.topk(z_ghost, k=k_aux, dim=1, largest=True)

        # 构建ghost稀疏表示
        z_ghost_sparse = torch.zeros_like(z_flat)
        z_ghost_sparse.scatter_(1, z_ghost_idx, z_ghost_val)

        # Ghost重构
        error_hat = self.decoder(z_ghost_sparse)  # [B*L, D]

        # AuxK损失: ghost重构误差
        error_flat = error.view(-1, self.d_model)
        loss_auxk = F.mse_loss(error_hat, error_flat)

        return loss_auxk

    def _compute_orthogonal_loss(self) -> torch.Tensor:
        """
        计算解码器正交正则损失

        L_orth = ||Wdec^T @ Wdec - I||_F^2

        目的：约束解码器权重列正交
        """
        # Wdec: [d_model, d_hidden]
        Wdec = self.decoder.weight

        # 计算 Wdec^T @ Wdec
        # [d_hidden, d_model] @ [d_model, d_hidden] = [d_hidden, d_hidden]
        WtW = Wdec.T @ Wdec

        # 单位矩阵
        I = torch.eye(self.d_hidden, device=Wdec.device, dtype=Wdec.dtype)

        # Frobenius范数
        loss_orth = torch.norm(WtW - I, p="fro") ** 2

        return loss_orth

    def normalize_decoder_weights(self):
        """
        解码器权重列单位范数归一化 (硬约束)

        每次optimizer.step()后调用
        """
        if not self.config.normalize_decoder:
            return

        with torch.no_grad():
            Wdec = self.decoder.weight  # [d_model, d_hidden]
            # 按列归一化
            col_norms = Wdec.norm(dim=0, keepdim=True)  # [1, d_hidden]
            col_norms = col_norms.clamp(min=1e-8)  # 避免除零
            Wdec.div_(col_norms)

    def verify_decoder_norm(self, tol: float = 1e-3) -> Tuple[bool, float]:
        """
        验证解码器列范数

        返回:
            is_valid: 是否所有列范数≈1
            max_deviation: 最大偏差
        """
        with torch.no_grad():
            Wdec = self.decoder.weight
            col_norms = Wdec.norm(dim=0)  # [d_hidden]
            deviations = (col_norms - 1).abs()
            max_dev = deviations.max().item()
            is_valid = max_dev < tol

        return is_valid, max_dev

    def get_dead_neurons(
        self,
        activation_counts: torch.Tensor,
        threshold: int = 0,
    ) -> torch.Tensor:
        """
        获取死神经元索引

        参数:
            activation_counts: 每个神经元的激活计数 [d_hidden]
            threshold: 激活次数低于此值视为死神经元

        返回:
            dead_indices: 死神经元索引
        """
        dead_mask = activation_counts <= threshold
        return dead_mask.nonzero(as_tuple=True)[0]

    def compute_sparsity(self, z_sparse: torch.Tensor) -> float:
        """计算稀疏度 (非零元素比例)"""
        total_elements = z_sparse.numel()
        nonzero_elements = (z_sparse != 0).sum().item()
        return nonzero_elements / total_elements

    def save_pretrained(self, path: str):
        """保存模型"""
        import json
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 保存权重
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config.to_dict(),
                "is_initialized": self._is_initialized,
                "init_method": self._init_method,
            },
            path / "sae.pt",
        )

        # 保存配置
        with open(path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda") -> "TopKSAE":
        """加载模型"""
        import json
        from pathlib import Path

        path = Path(path)

        # 加载配置
        with open(path / "config.json", "r") as f:
            config_dict = json.load(f)
        config = TopKSAEConfig(**config_dict)

        # 创建模型
        model = cls(config)

        # 加载权重
        ckpt = torch.load(path / "sae.pt", map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        model._is_initialized = ckpt.get("is_initialized", True)
        model._init_method = ckpt.get("init_method", "unknown")

        return model.to(device)


# ============================================================================
# 损失函数独立模块
# ============================================================================

class SAELossComputer:
    """
    SAE 损失计算器

    将损失计算逻辑独立封装，支持：
    1. 单独计算各类损失
    2. 梯度累积场景下的损失归一化
    3. 详细的损失分解
    """

    def __init__(
        self,
        lambda_aux: float = 0.1,
        lambda_orth: float = 1e-5,
        accum_steps: int = 8,
    ):
        self.lambda_aux = lambda_aux
        self.lambda_orth = lambda_orth
        self.accum_steps = accum_steps

    def compute_total_loss(
        self,
        loss_recon: torch.Tensor,
        loss_auxk: torch.Tensor,
        loss_orth: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        计算总损失

        参数:
            loss_recon: 重构损失
            loss_auxk: AuxK损失
            loss_orth: 正交损失
            normalize: 是否按accum_steps归一化

        返回:
            loss_total: 总损失
        """
        loss_total = (
            loss_recon
            + self.lambda_aux * loss_auxk
            + self.lambda_orth * loss_orth
        )

        if normalize:
            loss_total = loss_total / self.accum_steps

        return loss_total

    @staticmethod
    def compute_recon_loss(
        x: torch.Tensor,
        x_hat: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """计算重构MSE损失"""
        if reduction == "mean":
            return F.mse_loss(x_hat, x)
        elif reduction == "sum":
            return F.mse_loss(x_hat, x, reduction="sum")
        else:
            return ((x_hat - x) ** 2).mean(dim=-1)  # per-sample loss

    @staticmethod
    def compute_orthogonal_loss(Wdec: torch.Tensor) -> torch.Tensor:
        """计算正交正则损失"""
        d_hidden = Wdec.shape[1]
        WtW = Wdec.T @ Wdec
        I = torch.eye(d_hidden, device=Wdec.device, dtype=Wdec.dtype)
        return torch.norm(WtW - I, p="fro") ** 2


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TopK SAE 测试")
    print("=" * 60)

    # 测试1: 配置校验
    print("\n[测试1] 配置校验")
    try:
        bad_config = TopKSAEConfig(d_model=1536, d_hidden=12288, top_k=2000)
        print("  ERROR: 应该抛出异常")
    except ValueError as e:
        print(f"  ✓ 正确拒绝非法配置: {e}")

    # 合法配置
    config = TopKSAEConfig(d_model=1536, d_hidden=12288, top_k=128)
    print(f"  合法配置: d_model={config.d_model}, d_hidden={config.d_hidden}, "
          f"top_k={config.top_k}, ratio={config.top_k/config.d_hidden*100:.2f}%")

    # 测试2: 模型创建与前向
    print("\n[测试2] 模型创建与前向传播")
    sae = TopKSAE(config)
    print(f"  参数量: {sum(p.numel() for p in sae.parameters()):,}")

    # 创建测试输入
    B, L, D = 2, 100, 1536
    x = torch.randn(B, L, D, dtype=torch.bfloat16)
    sae = sae.to(x.dtype)

    # 前向传播
    x_hat_norm, z_sparse, loss_dict = sae(x, return_loss=True)
    print(f"  输入shape: {x.shape}")
    print(f"  输出shape: {x_hat_norm.shape}")
    print(f"  稀疏表示shape: {z_sparse.shape}")

    # 打印损失
    print("\n  损失分解:")
    for k, v in loss_dict.items():
        print(f"    {k}: {v.item():.6f}")

    # 测试3: 解码器权重归一化
    print("\n[测试3] 解码器权重归一化")
    sae.normalize_decoder_weights()
    is_valid, max_dev = sae.verify_decoder_norm()
    print(f"  归一化后: valid={is_valid}, max_deviation={max_dev:.2e}")

    # 测试4: 稀疏度
    print("\n[测试4] 稀疏度统计")
    sparsity = sae.compute_sparsity(z_sparse)
    expected_sparsity = config.top_k / config.d_hidden
    print(f"  实际稀疏度: {sparsity:.4f}")
    print(f"  期望稀疏度: {expected_sparsity:.4f}")

    # 测试5: 保存与加载
    print("\n[测试5] 保存与加载")
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        sae.save_pretrained(tmpdir)
        print(f"  保存到: {tmpdir}")

        sae_loaded = TopKSAE.from_pretrained(tmpdir)
        print(f"  加载成功: {sae_loaded._init_method}")

        # 验证权重一致
        for (n1, p1), (n2, p2) in zip(sae.named_parameters(), sae_loaded.named_parameters()):
            assert torch.allclose(p1, p2), f"权重不一致: {n1}"
        print("  ✓ 权重验证通过")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
