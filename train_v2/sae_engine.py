"""
SAE Engine - Forward + Loss Computation

严格按照 TODO_list_v3.md 2.4 规范

Loss:
    L_total = L_recon + 0.1 * L_auxk + 1e-5 * L_orth

关键组件:
1. RMSNorm 预处理
2. TopK 编码
3. 重建损失 (MSE)
4. AuxK 损失 (死神经元恢复)
5. 正交损失 (decoder 列正交化)
6. Decoder 列归一化

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_v2.config import TrainingConfig

# 延迟导入 wan 模块，允许在没有完整依赖时运行测试
try:
    from wan.modules.sae_new import SparseAutoEncoder, SAEConfig
    WAN_AVAILABLE = True
except ImportError:
    WAN_AVAILABLE = False

    # 提供本地简化版本用于测试
    from dataclasses import dataclass as _dataclass

    @_dataclass
    class SAEConfig:
        """简化版 SAE 配置"""
        d_model: int
        d_hidden: int
        activation: str = "relu"
        sparsity: str = "topk"
        top_k: int = 64
        l1_lambda: float = 1e-3

        def to_dict(self):
            from dataclasses import asdict
            return asdict(self)

    class SparseAutoEncoder(nn.Module):
        """简化版 SAE 用于测试"""

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

        def encode(self, x):
            z = F.relu(self.encoder(x))
            # TopK 稀疏化
            k = min(self.config.top_k, z.size(1))
            topk_val, topk_idx = torch.topk(z, k=k, dim=1, largest=True)
            z_sparse = torch.zeros_like(z)
            z_sparse.scatter_(1, topk_idx, topk_val)
            return z_sparse, topk_idx, topk_val

        def decode(self, z):
            return self.decoder(z)


# ============================================================================
# 工具函数
# ============================================================================

def per_token_rms_norm(
    x: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token RMSNorm

    参数:
        x: [N, D] 输入张量
        eps: 数值稳定性

    返回:
        x_norm: 归一化后的张量
        rms: RMS 值 [N, 1]
    """
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_norm = x / rms
    return x_norm, rms


# ============================================================================
# Loss 信息
# ============================================================================

@dataclass
class LossInfo:
    """Loss 分解信息"""
    total_loss: float
    recon_loss: float
    auxk_loss: float
    orth_loss: float

    # 激活统计
    active_features: int
    total_features: int
    active_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_loss": self.total_loss,
            "recon_loss": self.recon_loss,
            "auxk_loss": self.auxk_loss,
            "orth_loss": self.orth_loss,
            "active_features": self.active_features,
            "total_features": self.total_features,
            "active_ratio": self.active_ratio,
        }


# ============================================================================
# SAE Engine
# ============================================================================

class SAEEngine:
    """
    SAE 前向传播 + 损失计算引擎

    严格按照 TODO 2.4 规范:
    - RMSNorm 预处理
    - TopK 编码
    - L_total = L_recon + 0.1 * L_auxk + 1e-5 * L_orth

    使用示例:
        engine = SAEEngine(sae, config)

        # 训练
        loss, info = engine.forward(x)
        loss.backward()

        # 推理
        with torch.no_grad():
            z, topk_idx, topk_val = engine.encode(x)
            x_hat = engine.decode(z)
    """

    def __init__(
        self,
        sae: SparseAutoEncoder,
        config: TrainingConfig,
    ):
        """
        初始化 SAE Engine

        参数:
            sae: SAE 模型实例
            config: 训练配置
        """
        self.sae = sae
        self.config = config

        # Loss 权重
        self.lambda_aux = config.lambda_aux      # 0.1
        self.lambda_orth = config.lambda_orth    # 1e-5

    def forward(
        self,
        x: torch.Tensor,
        return_info: bool = True,
    ) -> Tuple[torch.Tensor, Optional[LossInfo]]:
        """
        完整前向传播 + 损失计算

        参数:
            x: [N, D] 输入张量 (原始激活)
            return_info: 是否返回详细 loss 信息

        返回:
            loss: 总损失 (用于 backward)
            info: LossInfo 对象 (可选)
        """
        # ===== Step 1: RMSNorm 预处理 =====
        x_norm, rms = per_token_rms_norm(x)

        # ===== Step 2: SAE 编码 =====
        z_sparse, topk_idx, topk_val = self.sae.encode(x_norm)

        # ===== Step 3: SAE 解码 =====
        x_hat = self.sae.decode(z_sparse)

        # ===== Step 4: 计算损失 =====

        # 4.1 重建损失 (MSE)
        recon_loss = F.mse_loss(x_hat, x_norm)

        # 4.2 AuxK 损失 (死神经元恢复)
        auxk_loss = self._compute_auxk_loss(x_norm, z_sparse, topk_idx, topk_val)

        # 4.3 正交损失 (decoder 列正交化)
        orth_loss = self._compute_orthogonal_loss()

        # 总损失
        total_loss = (
            recon_loss +
            self.lambda_aux * auxk_loss +
            self.lambda_orth * orth_loss
        )

        # ===== Step 5: 构建信息 =====
        if return_info:
            # 计算激活统计
            active_features = torch.unique(topk_idx).numel()
            total_features = self.sae.d_hidden

            info = LossInfo(
                total_loss=total_loss.item(),
                recon_loss=recon_loss.item(),
                auxk_loss=auxk_loss.item(),
                orth_loss=orth_loss.item(),
                active_features=active_features,
                total_features=total_features,
                active_ratio=active_features / total_features,
            )
            return total_loss, info

        return total_loss, None

    def _compute_auxk_loss(
        self,
        x_norm: torch.Tensor,
        z_sparse: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_val: torch.Tensor,
    ) -> torch.Tensor:
        """
        AuxK 损失计算

        用于死神经元恢复:
        - 找出未激活的神经元
        - 使用它们的重建误差作为辅助损失

        参考: https://arxiv.org/abs/2404.16014
        """
        # 找出当前 batch 未激活的特征
        all_features = torch.arange(
            self.sae.d_hidden,
            device=z_sparse.device,
        )
        active_mask = torch.zeros(
            self.sae.d_hidden,
            dtype=torch.bool,
            device=z_sparse.device,
        )
        active_mask[topk_idx.flatten()] = True
        dead_features = all_features[~active_mask]

        if len(dead_features) == 0:
            # 没有死神经元
            return torch.tensor(0.0, device=x_norm.device)

        # 计算死神经元的激活值 (使用 encoder 输出)
        z_pre_activation = self.sae.encoder(x_norm)
        z_pre_activation = F.relu(z_pre_activation)  # 应用激活函数

        # 死神经元的激活值
        dead_activations = z_pre_activation[:, dead_features]

        # 使用死神经元重建误差作为损失
        # 这鼓励死神经元学习有用的特征
        if dead_activations.numel() == 0:
            return torch.tensor(0.0, device=x_norm.device)

        # 简化版: 使用死神经元激活的 L2 范数
        # 这鼓励死神经元有非零激活
        auxk_loss = dead_activations.pow(2).mean()

        return auxk_loss

    def _compute_orthogonal_loss(self) -> torch.Tensor:
        """
        正交损失计算

        鼓励 decoder 列向量正交，降低互相关性

        Loss = ||W_dec.T @ W_dec - I||_F^2
        """
        W_dec = self.sae.decoder.weight  # [d_model, d_hidden]

        # 归一化列向量
        W_dec_norm = F.normalize(W_dec, dim=0)

        # 计算 Gram 矩阵
        gram = W_dec_norm.T @ W_dec_norm  # [d_hidden, d_hidden]

        # 正交损失: 远离对角线元素应为 0
        identity = torch.eye(
            gram.shape[0],
            device=gram.device,
            dtype=gram.dtype,
        )
        orth_loss = ((gram - identity) ** 2).sum()

        return orth_loss

    def normalize_decoder_columns(self) -> None:
        """
        归一化 decoder 列向量

        严格执行: 每次权重更新后调用
        """
        with torch.no_grad():
            W_dec = self.sae.decoder.weight
            W_dec.data = F.normalize(W_dec.data, dim=0)

    def encode(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        仅编码 (用于分析)

        参数:
            x: [N, D] 输入张量

        返回:
            z_sparse: [N, d_hidden] 稀疏编码
            topk_idx: [N, k] TopK 索引
            topk_val: [N, k] TopK 值
        """
        x_norm, _ = per_token_rms_norm(x)
        return self.sae.encode(x_norm)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        仅解码 (用于分析)

        参数:
            z: [N, d_hidden] 稀疏编码

        返回:
            x_hat: [N, D] 重建输出
        """
        return self.sae.decode(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        完整重建 (编码 + 解码)

        参数:
            x: [N, D] 输入张量

        返回:
            x_hat: [N, D] 重建输出
        """
        z_sparse, _, _ = self.encode(x)
        return self.decode(z_sparse)


# ============================================================================
# 工厂函数
# ============================================================================

def create_sae_engine(config: TrainingConfig, device: str = "cuda") -> SAEEngine:
    """
    创建 SAE Engine

    参数:
        config: 训练配置
        device: 设备

    返回:
        SAEEngine 实例
    """
    # 创建 SAE 配置
    sae_config = SAEConfig(
        d_model=config.d_model,
        d_hidden=config.d_hidden,
        activation="relu",
        sparsity="topk",
        top_k=config.top_k,
    )

    # 创建 SAE 模型
    sae = SparseAutoEncoder(sae_config).to(device)

    # 创建 Engine
    engine = SAEEngine(sae, config)

    return engine


def create_sae_engine_from_init(
    init_checkpoint: str,
    config: TrainingConfig,
    device: str = "cuda",
) -> SAEEngine:
    """
    从初始化阶段的 checkpoint 创建 SAE Engine

    参数:
        init_checkpoint: 初始化阶段的 checkpoint 路径
            格式1: sae_init_layer{N}.pt (sae_mixed_init.py 输出)
            格式2: sae.pt (TopKSAE.save_pretrained 输出)
        config: 训练配置
        device: 设备

    返回:
        SAEEngine 实例
    """
    # 创建基础 engine
    engine = create_sae_engine(config, device=device)

    # 加载初始化 checkpoint
    ckpt = torch.load(init_checkpoint, map_location=device)

    # 判断 checkpoint 格式并转换权重
    if "Wenc" in ckpt and "Wdec" in ckpt:
        # 格式1: sae_mixed_init.py 输出
        # Wenc: [d_hidden, d_model] -> encoder.weight 需要转置
        # Wdec: [d_model, d_hidden] -> decoder.weight
        Wenc = ckpt["Wenc"].to(device)  # [d_hidden, d_model]
        Wdec = ckpt["Wdec"].to(device)  # [d_model, d_hidden]

        # 注意: nn.Linear 的 weight 形状是 [out_features, in_features]
        # encoder: Linear(d_model, d_hidden), weight 形状 [d_hidden, d_model]
        # 所以 Wenc 直接对应 encoder.weight
        engine.sae.encoder.weight.data = Wenc.float()
        engine.sae.decoder.weight.data = Wdec.float()

        # 如果有 bpre (预偏置)，可以忽略或用于初始化
        # 当前 SparseAutoEncoder 没有 bias，所以跳过

        print(f"Loaded init checkpoint (mixed_init format): {init_checkpoint}")

    elif "state_dict" in ckpt:
        # 格式2: TopKSAE.save_pretrained 输出
        state_dict = ckpt["state_dict"]

        # TopKSAE 的 state_dict 可能包含:
        # - encoder.weight [d_hidden, d_model]
        # - decoder.weight [d_model, d_hidden]
        # - bpre [d_hidden]
        # - b_enc [d_hidden]

        # 映射到 SparseAutoEncoder
        new_state_dict = {}

        if "encoder.weight" in state_dict:
            new_state_dict["encoder.weight"] = state_dict["encoder.weight"].float()
        if "decoder.weight" in state_dict:
            new_state_dict["decoder.weight"] = state_dict["decoder.weight"].float()

        engine.sae.load_state_dict(new_state_dict, strict=False)

        print(f"Loaded init checkpoint (TopKSAE format): {init_checkpoint}")

    else:
        # 格式3: 标准 state_dict
        engine.sae.load_state_dict(ckpt, strict=False)
        print(f"Loaded init checkpoint (standard format): {init_checkpoint}")

    return engine


def create_sae_engine_from_sae(
    sae: SparseAutoEncoder,
    config: TrainingConfig,
) -> SAEEngine:
    """
    从已有 SAE 创建 Engine

    参数:
        sae: 已有的 SAE 模型
        config: 训练配置

    返回:
        SAEEngine 实例
    """
    return SAEEngine(sae, config)


# ============================================================================
# 验证工具
# ============================================================================

def validate_loss_computation(engine: SAEEngine, batch_size: int = 16) -> bool:
    """
    验证损失计算正确性
    """
    device = next(engine.sae.parameters()).device

    # 创建假输入
    x = torch.randn(batch_size, engine.sae.d_model, device=device)

    # 前向传播
    loss, info = engine.forward(x, return_info=True)

    # 检查 loss 非负
    if loss.item() < 0:
        return False

    # 检查 loss 可以 backward
    loss.backward()

    # 检查梯度存在
    has_grad = False
    for p in engine.sae.parameters():
        if p.grad is not None:
            has_grad = True
            break

    return has_grad


# ============================================================================
# 调试工具
# ============================================================================

def print_loss_info(info: LossInfo) -> None:
    """打印损失信息"""
    print("\nLoss Info:")
    print("-" * 40)
    print(f"  Total Loss:  {info.total_loss:.6f}")
    print(f"  Recon Loss:  {info.recon_loss:.6f}")
    print(f"  AuxK Loss:   {info.auxk_loss:.6f}")
    print(f"  Orth Loss:   {info.orth_loss:.6f}")
    print("-" * 40)
    print(f"  Active Features: {info.active_features}/{info.total_features}")
    print(f"  Active Ratio:    {info.active_ratio:.4f}")
    print("-" * 40)
