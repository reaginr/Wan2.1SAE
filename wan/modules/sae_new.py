from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    d_model: int
    d_hidden: int
    activation: str = "relu"  # 预留扩展: relu / gelu / silu ...
    sparsity: str = "topk"  # "topk" | "l1"
    top_k: int = 64  # sparsity=topk 时生效：每个样本保留多少个激活
    l1_lambda: float = 1e-3  # sparsity=l1 时生效

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _apply_activation(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu":
        return F.relu(x)
    if kind == "gelu":
        return F.gelu(x)
    if kind == "silu":
        return F.silu(x)
    raise ValueError(f"Unsupported activation: {kind}")


def topk_sparsify(z: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对每个样本（行）做 top-k 稀疏化。

    输入:
        z: [N, D]
    输出:
        z_sparse: [N, D] (非 top-k 位置为 0)
        topk_idx: [N, k]
        topk_val: [N, k]
    """
    if k <= 0:
        raise ValueError("top_k must be > 0")
    if z.dim() != 2:
        raise ValueError(f"Expected z dim=2, got {z.dim()}")
    k = min(k, z.size(1))
    topk_val, topk_idx = torch.topk(z, k=k, dim=1, largest=True, sorted=False)
    z_sparse = torch.zeros_like(z)
    z_sparse.scatter_(1, topk_idx, topk_val)
    return z_sparse, topk_idx, topk_val


class SparseAutoEncoder(nn.Module):
    """
    可扩展 SAE：
    - 支持两种稀疏方式：
      1) top-k（默认）：每个样本保留 k 个最大激活
      2) l1：使用 L1 正则约束稀疏
    - 预留 activation / 结构替换空间：训练脚本通过工厂函数可替换成 transformer 等。
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.d_model, config.d_hidden, bias=False)
        self.decoder = nn.Linear(config.d_hidden, config.d_model, bias=False)

    @property
    def d_model(self) -> int:
        return self.config.d_model

    @property
    def d_hidden(self) -> int:
        return self.config.d_hidden

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        输入:
            x: [N, d_model]
        输出:
            z_sparse: [N, d_hidden]
            topk_idx/topk_val: 仅在 sparsity=topk 时返回，用于分析与存储
        """
        z = _apply_activation(self.encoder(x), self.config.activation)  # [N, d_hidden]
        if self.config.sparsity == "topk":
            z_sparse, topk_idx, topk_val = topk_sparsify(z, self.config.top_k)
            return z_sparse, topk_idx, topk_val
        if self.config.sparsity == "l1":
            return z, None, None
        raise ValueError(f"Unsupported sparsity: {self.config.sparsity}")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)  # [N, d_model]

    def forward(
        self, x: torch.Tensor, return_loss: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [N, d_model]
        返回:
            x_hat: [N, d_model]
            z: [N, d_hidden]（稀疏后的）
            loss: 标量（可选）
        """
        z, _, _ = self.encode(x)
        x_hat = self.decode(z)
        if not return_loss:
            return x_hat, z

        recon_loss = F.mse_loss(x_hat, x)
        if self.config.sparsity == "topk":
            sparsity_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        else:
            sparsity_loss = self.config.l1_lambda * z.abs().mean()
        loss = recon_loss + sparsity_loss
        return x_hat, z, loss
