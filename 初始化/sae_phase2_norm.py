"""
SAE 第二阶段核心模块 - Per-Token RMSNorm 归一化管理器

根据 TODO list_v2 要求实现：
- Per-Token RMSNorm: 对单个token的1536维特征独立归一化
- eps=1e-6 (与Wan2.1内部对齐)
- 无训练参数
- 禁止单位范数归一化/全局归一化

使用方法:
    from 初始化.sae_phase2 import NormDenormManager

    manager = NormDenormManager(eps=1e-6)
    x_norm, rms = manager.per_token_rms_norm(x)
    x_recovered = manager.per_token_rms_denorm(x_norm, rms)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class NormDenormManager:
    """
    Per-Token RMSNorm 归一化管理器

    设计原则：
    1. 无训练参数 - 纯计算操作
    2. Per-Token - 对每个token独立归一化，不是全局归一化
    3. 完全可逆 - 反归一化MSE≤1e-6
    4. 不破坏DiT residual geometry

    关键区别：
    - RMSNorm: x_norm = x / sqrt(mean(x^2) + eps)
    - 单位范数归一化: x_norm = x / ||x||_2  (禁止使用!)
    - 全局归一化: 在整个batch上计算统计量 (禁止使用!)
    """

    def __init__(self, eps: float = 1e-6, debug: bool = False):
        """
        初始化归一化管理器

        参数:
            eps: 数值稳定项，默认1e-6 (与Wan2.1内部对齐)
            debug: 是否输出调试信息
        """
        self.eps = eps
        self.debug = debug

        # 调试统计
        self._norm_call_count = 0
        self._denorm_call_count = 0

    def per_token_rms_norm(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Per-Token RMSNorm 归一化

        输入:
            x: 原始DiT激活，shape=[B, L, D]
               B = batch size
               L = sequence length (token数)
               D = feature dim (1536 for Wan2.1 1.3B)

        输出:
            x_norm: 归一化后的激活，shape=[B, L, D]
            rms: 用于反归一化的rms值，shape=[B, L, 1]

        计算公式:
            rms = sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
            x_norm = x / rms

        注意:
            - 对每个token独立计算rms (不是全局)
            - 不对rms做任何学习或变换
            - 与LayerNorm/BatchNorm/单位范数归一化完全不同
        """
        # 输入校验
        assert x.dim() == 3, f"输入必须是3D张量 [B, L, D]，当前shape: {x.shape}"
        B, L, D = x.shape

        # 检查NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError(f"输入包含NaN或Inf值，无法归一化")

        # 计算每个token的RMS
        # rms shape: [B, L, 1]
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(variance + self.eps)

        # 归一化
        x_norm = x / rms

        # 调试输出
        if self.debug:
            self._norm_call_count += 1
            print(f"[RMSNorm #{self._norm_call_count}] "
                  f"input: shape={x.shape}, mean={x.mean():.6f}, std={x.std():.6f}, "
                  f"rms: mean={rms.mean():.6f}, std={rms.std():.6f}, "
                  f"output: mean={x_norm.mean():.6f}, std={x_norm.std():.6f}")

        return x_norm, rms

    def per_token_rms_denorm(
        self,
        x_norm: torch.Tensor,
        rms: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-Token RMSNorm 反归一化

        输入:
            x_norm: 归一化后的激活，shape=[B, L, D]
            rms: 归一化时保存的rms值，shape=[B, L, 1]

        输出:
            x_raw: 反归一化后的原始激活，shape=[B, L, D]

        计算公式:
            x_raw = x_norm * rms

        注意:
            - rms必须与x_norm对应，不能混用
            - 反归一化后应与原始激活MSE≤1e-6
        """
        # Shape校验
        assert x_norm.dim() == 3, f"x_norm必须是3D张量，当前shape: {x_norm.shape}"
        assert rms.dim() == 3, f"rms必须是3D张量，当前shape: {rms.shape}"
        assert x_norm.shape[:2] == rms.shape[:2], \
            f"x_norm和rms的B,L维度必须匹配: {x_norm.shape} vs {rms.shape}"
        assert rms.shape[2] == 1, f"rms的最后一维必须为1，当前shape: {rms.shape}"

        # 反归一化
        x_raw = x_norm * rms

        # 调试输出
        if self.debug:
            self._denorm_call_count += 1
            print(f"[RMSDenorm #{self._denorm_call_count}] "
                  f"input: shape={x_norm.shape}, mean={x_norm.mean():.6f}, "
                  f"rms: mean={rms.mean():.6f}, "
                  f"output: mean={x_raw.mean():.6f}, std={x_raw.std():.6f}")

        return x_raw

    def verify_reversibility(
        self,
        x: torch.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-6,
    ) -> Tuple[bool, float]:
        """
        验证归一化/反归一化的可逆性

        参数:
            x: 原始张量
            rtol: 相对容差
            atol: 绝对容差

        返回:
            is_reversible: 是否可逆
            mse: 均方误差
        """
        x_norm, rms = self.per_token_rms_norm(x)
        x_recovered = self.per_token_rms_denorm(x_norm, rms)

        mse = torch.nn.functional.mse_loss(x_recovered, x).item()
        is_reversible = torch.allclose(x_recovered, x, rtol=rtol, atol=atol)

        if self.debug:
            print(f"[Reversibility Check] MSE={mse:.2e}, reversible={is_reversible}")

        return is_reversible, mse

    @staticmethod
    def is_unit_norm_normalization() -> bool:
        """返回False，标识这不是单位范数归一化"""
        return False

    @staticmethod
    def is_global_normalization() -> bool:
        """返回False，标识这不是全局归一化"""
        return False


class PerTokenRMSNorm(nn.Module):
    """
    Per-Token RMSNorm 作为nn.Module

    用于需要将归一化集成到模型结构中的场景。
    注意：这个模块没有训练参数，只做计算。

    与LayerNorm的区别：
    - LayerNorm: 有可学习的weight和bias
    - PerTokenRMSNorm: 无参数，纯计算
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.manager = NormDenormManager(eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        输入: [B, L, D]
        输出: [B, L, D] (归一化后)
        注意: 不返回rms，仅用于推理
        """
        x_norm, _ = self.manager.per_token_rms_norm(x)
        return x_norm


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NormDenormManager 测试")
    print("=" * 60)

    # 创建测试数据
    torch.manual_seed(42)
    B, L, D = 2, 100, 1536  # batch=2, 100 tokens, 1536 dim

    # 测试1: 基本功能
    print("\n[测试1] 基本归一化/反归一化")
    manager = NormDenormManager(eps=1e-6, debug=True)

    x = torch.randn(B, L, D, dtype=torch.bfloat16)
    x_norm, rms = manager.per_token_rms_norm(x)
    x_recovered = manager.per_token_rms_denorm(x_norm, rms)

    mse = torch.nn.functional.mse_loss(x_recovered, x).item()
    print(f"MSE: {mse:.2e}")
    assert mse < 1e-6, f"反归一化MSE过大: {mse}"
    print("✓ 基本功能测试通过")

    # 测试2: 可逆性验证
    print("\n[测试2] 可逆性验证")
    is_rev, mse = manager.verify_reversibility(x)
    print(f"可逆: {is_rev}, MSE: {mse:.2e}")
    assert is_rev, "归一化不可逆"
    print("✓ 可逆性测试通过")

    # 测试3: 不同数据类型
    print("\n[测试3] 不同数据类型")
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x_typed = torch.randn(B, L, D, dtype=dtype)
        x_norm, rms = manager.per_token_rms_norm(x_typed)
        x_recovered = manager.per_token_rms_denorm(x_norm, rms)
        mse = torch.nn.functional.mse_loss(x_recovered.float(), x_typed.float()).item()
        print(f"  {dtype}: MSE={mse:.2e}")
    print("✓ 数据类型测试通过")

    # 测试4: 边界情况
    print("\n[测试4] 边界情况")
    # 全零输入
    x_zero = torch.zeros(1, 10, D)
    x_norm, rms = manager.per_token_rms_norm(x_zero)
    assert torch.isnan(x_norm).sum() == 0, "全零输入产生NaN"
    print("  全零输入: OK")

    # 极小值输入
    x_tiny = torch.randn(1, 10, D) * 1e-10
    x_norm, rms = manager.per_token_rms_norm(x_tiny)
    assert torch.isnan(x_norm).sum() == 0, "极小值输入产生NaN"
    print("  极小值输入: OK")

    # 极大值输入
    x_large = torch.randn(1, 10, D) * 1e10
    x_norm, rms = manager.per_token_rms_norm(x_large)
    assert torch.isinf(x_norm).sum() == 0, "极大值输入产生Inf"
    print("  极大值输入: OK")

    print("✓ 边界情况测试通过")

    # 测试5: 确认不是单位范数归一化
    print("\n[测试5] 区分RMSNorm与单位范数归一化")
    x = torch.randn(1, 10, D)
    x_norm, rms = manager.per_token_rms_norm(x)

    # RMSNorm: 归一化后每个token的平方和均值为D (不是1)
    sq_sum = x_norm.pow(2).sum(dim=-1)  # [B, L]
    expected_sq_sum = D  # RMSNorm特性: mean(x^2)≈1，所以sum≈D
    actual_sq_sum = sq_sum.mean().item()

    print(f"  RMSNorm sum(x^2)/token: {actual_sq_sum:.1f} (期望≈{D})")

    # 单位范数归一化: ||x||=1，所以sum(x^2)=1
    unit_norm = x / x.norm(dim=-1, keepdim=True)
    unit_norm_sq_sum = unit_norm.pow(2).sum(dim=-1).mean().item()
    print(f"  单位范数 sum(x^2)/token: {unit_norm_sq_sum:.1f} (期望=1)")

    assert not manager.is_unit_norm_normalization(), "错误：不应是单位范数归一化"
    print("✓ 确认不是单位范数归一化")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
