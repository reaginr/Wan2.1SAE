"""
在从 Wan 1.3B T2V 模型采集到隐藏状态后，训练 SAE 的简单骨架。

此脚本用于离线训练：先通过 sae_collect.py 采集特征，然后用此脚本训练 SAE。
相比 sae_train_t2v_1_3b.py（在线训练），此脚本：
- 优点：训练更快（无需运行 DiT），可多轮训练调参
- 缺点：无法利用多时间步的完整扩散轨迹

示例:
    python train_sae.py \
        --features_path features_layer16.pt \
        --d_model 1536 \
        --d_hidden 6144 \
        --epochs 5 \
        --batch_size 4096 \
        --lr 1e-3 \
        --save_path sae_layer16.pt
"""

import argparse
import logging
import time
from typing import List

import torch
from torch.utils.data import DataLoader, TensorDataset

from wan.modules.sae_new import SAEConfig, SparseAutoEncoder


logger = logging.getLogger(__name__)


##########################################################################################
# 训练参数配置区域 - 可直接修改此区域的默认值
# 学术意义与建议值详见每个参数的注释
##########################################################################################

# --------------------------- 数据配置 ---------------------------
data_params = {
    # features_path: 采集的隐藏状态文件路径
    # 学术意义: 从 DiT 某层采集的激活样本，用于训练 SAE
    # 实际用法: 通常是 sae_collect.py 的输出文件
    # 建议值: 与 layer_idx 对应，如 "features_layer29.pt"
    "features_path": "features_layer29.pt",

    # save_path: 训练好的 SAE 保存路径
    # 实际用法: 保存为 .pt 文件，包含 state_dict
    # 建议值: "sae_layer{layer_idx}.pt"
    "save_path": "sae_layer29.pt",
}

# --------------------------- 模型架构配置 ---------------------------
model_params = {
    # d_model: SAE 输入维度
    # 学术意义: 必须与采集特征的维度一致，即 DiT 的 dim
    # 实际用法: Wan 1.3B 固定为 1536
    # 建议值: 1536（1.3B）或 5120（14B）
    "d_model": 1536,

    # d_hidden: SAE 隐空间维度
    # 学术意义: 扩展比率决定 SAE 能学习的特征数量
    # 学术参考: d_hidden/d_model = 4~8 是较好的稀疏性-容量权衡
    # 建议值: 6144（4x）或 12288（8x）
    "d_hidden": 6144,

    # activation: 编码器激活函数
    # 可选值: "relu" | "gelu" | "silu"
    # 建议值: "relu"（经典 SAE）或 "gelu"（更稳定）
    "activation": "relu",

    # sparsity: 稀疏化策略
    # 可选值: "topk" | "l1"
    # 建议值: "topk"（可解释性更强）
    "sparsity": "topk",

    # top_k: topk 策略下的保留数量
    # 学术意义: 控制每个样本的稀疏度
    # 建议值: 64（d_hidden=6144 时约 1%）
    "top_k": 64,

    # l1_lambda: L1 正则权重（sparsity="l1" 时生效）
    # 建议值: 1e-3
    "l1_lambda": 1e-3,
}

# --------------------------- 训练流程配置 ---------------------------
training_params = {
    # epochs: 训练轮数
    # 学术意义: 决定 SAE 收敛程度
    # 建议值: 5~10，观察 loss 曲线调整
    "epochs": 5,

    # batch_size: 批次大小
    # 学术意义: 影响梯度估计的方差和训练稳定性
    # 实际用法: 越大训练越快，但内存占用增加
    # 建议值: 4096~16384，根据 GPU 内存调整
    "batch_size": 4096,

    # lr: 学习率
    # 学术意义: 参数更新步长，影响收敛速度和稳定性
    # 建议值: 1e-3 ~ 1e-4
    "lr": 1e-3,

    # weight_decay: 权重衰减
    # 学术意义: L2 正则化，防止过拟合
    # 建议值: 0（SAE 通常不需要）或 1e-6
    "weight_decay": 0,
}

# --------------------------- 系统配置 ---------------------------
system_params = {
    # device_id: GPU 设备 ID
    # 建议值: 0
    "device_id": 0,

    # num_workers: 数据加载器工作进程数
    # 实际用法: 0 表示主进程加载（适合小数据集）
    # 建议值: 0 或 4
    "num_workers": 0,
}

# --------------------------- 日志配置 ---------------------------
log_params = {
    # log_interval: 日志打印间隔（批次数）
    # 建议值: 10
    "log_interval": 10,
}


##########################################################################################
# 核心代码区域
##########################################################################################


def format_time(seconds: float) -> str:
    """将秒数格式化为人类可读的时间字符串。"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def _print_config():
    """打印所有配置参数。"""
    logger.info("=" * 60)
    logger.info("SAE 离线训练配置")
    logger.info("=" * 60)
    for name, params in [
        ("数据配置", data_params),
        ("模型架构", model_params),
        ("训练流程", training_params),
        ("系统配置", system_params),
    ]:
        logger.info(f"\n【{name}】")
        for k, v in params.items():
            logger.info(f"  {k}: {v}")
    logger.info("=" * 60)


def train_sae(
    features_path: str,
    d_model: int,
    d_hidden: int,
    l1_lambda: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    save_path: str,
    device_id: int = 0,
    num_workers: int = 0,
):
    """
    在预采集的特征上训练 SAE。

    参数:
        features_path: 特征文件路径 (.pt)
        d_model: 输入维度
        d_hidden: 隐空间维度
        l1_lambda: L1 正则权重
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        weight_decay: 权重衰减
        save_path: 保存路径
        device_id: GPU ID
        num_workers: 数据加载工作进程数
    """
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    # 1. 加载采集好的隐藏状态特征
    logger.info("加载特征: %s", features_path)
    feats = torch.load(features_path)  # [N, d_model]
    assert feats.dim() == 2 and feats.size(1) == d_model, (
        f"features 维度不匹配，期待 [N, {d_model}]，实际 {tuple(feats.shape)}"
    )
    logger.info("特征形状: %s", feats.shape)

    dataset = TensorDataset(feats)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 2. 构建 SAE
    sae_cfg = SAEConfig(
        d_model=d_model,
        d_hidden=d_hidden,
        activation=model_params["activation"],
        sparsity=model_params["sparsity"],
        top_k=model_params["top_k"],
        l1_lambda=l1_lambda,
    )
    sae = SparseAutoEncoder(sae_cfg)
    sae.to(device)
    logger.info("SAE 已创建: d_model=%d, d_hidden=%d", d_model, d_hidden)

    optimizer = torch.optim.AdamW(
        sae.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # 3. 训练循环
    sae.train()
    train_start_time = time.time()
    step_times: List[float] = []

    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0.0
        recon_loss_sum = 0.0
        sparse_loss_sum = 0.0
        num_batches = 0

        for batch_idx, (batch,) in enumerate(loader):
            step_start = time.time()
            batch = batch.to(device)

            _, _, loss = sae(batch, return_loss=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            with torch.no_grad():
                x_hat, z = sae(batch, return_loss=False)
                recon_loss = torch.nn.functional.mse_loss(x_hat, batch).item()
                if model_params["sparsity"] == "l1":
                    sparse_loss = (l1_lambda * z.abs().mean()).item()
                else:
                    sparse_loss = 0.0

            total_loss += loss.item()
            recon_loss_sum += recon_loss
            sparse_loss_sum += sparse_loss
            num_batches += 1

            step_end = time.time()
            step_times.append(step_end - step_start)
            if len(step_times) > 100:
                step_times.pop(0)

            # 打印进度
            if (batch_idx + 1) % log_params["log_interval"] == 0:
                avg_step_time = sum(step_times) / len(step_times)
                remaining_batches = len(loader) - batch_idx - 1
                eta_epoch = avg_step_time * remaining_batches

                logger.info(
                    "Epoch %d [%d/%d] loss=%.6f recon=%.6f sparse=%.6f step_time=%.3fs ETA=%s",
                    epoch + 1, batch_idx + 1, len(loader),
                    loss.item(), recon_loss, sparse_loss,
                    avg_step_time, format_time(eta_epoch)
                )

        # Epoch 结束统计
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(num_batches, 1)
        avg_recon = recon_loss_sum / max(num_batches, 1)
        avg_sparse = sparse_loss_sum / max(num_batches, 1)

        logger.info(
            "Epoch %d/%d 完成 | avg_loss=%.6f avg_recon=%.6f avg_sparse=%.6f time=%s",
            epoch + 1, epochs, avg_loss, avg_recon, avg_sparse, format_time(epoch_time)
        )

    # 4. 保存权重
    total_time = time.time() - train_start_time
    torch.save(sae.state_dict(), save_path)
    logger.info("SAE 已保存到: %s", save_path)
    logger.info("总训练时间: %s", format_time(total_time))


def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # 参数解析
    parser = argparse.ArgumentParser(description="Train SAE on Wan hidden states.")
    parser.add_argument("--features_path", type=str, default=data_params["features_path"],
                        help="采集到的隐藏状态 .pt 文件")
    parser.add_argument("--d_model", type=int, default=model_params["d_model"],
                        help="隐藏状态维度（等于 Wan dim，例如 1536）")
    parser.add_argument("--d_hidden", type=int, default=model_params["d_hidden"],
                        help="SAE 隐空间维度（通常是 d_model 的数倍）")
    parser.add_argument("--l1_lambda", type=float, default=model_params["l1_lambda"],
                        help="L1 稀疏正则权重")
    parser.add_argument("--epochs", type=int, default=training_params["epochs"],
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=training_params["batch_size"],
                        help="batch 大小")
    parser.add_argument("--lr", type=float, default=training_params["lr"],
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=training_params["weight_decay"],
                        help="权重衰减")
    parser.add_argument("--save_path", type=str, default=data_params["save_path"],
                        help="训练好的 SAE 权重保存路径 (.pt)")
    parser.add_argument("--device_id", type=int, default=system_params["device_id"],
                        help="GPU ID")
    parser.add_argument("--num_workers", type=int, default=system_params["num_workers"],
                        help="数据加载工作进程数")

    args = parser.parse_args()

    # 打印配置
    _print_config()

    # 执行训练
    train_sae(
        features_path=args.features_path,
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        l1_lambda=args.l1_lambda,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_path=args.save_path,
        device_id=args.device_id,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
