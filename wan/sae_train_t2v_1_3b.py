"""
Wan 1.3B 文生视频（T2V）SAE 训练脚本（不做 VAE 解码）。

满足需求：
1) 从文件夹批量读取/清洗 prompt（多 txt，逐行 prompt）
2) 按 hook 模式/层列表，hook DiT 中的 self_attn / cross_attn / block_out
3) 每次取 n 个 prompt 做一次 forward，拿到 hook 特征后训练 SAE
4) 每 K step 保存 checkpoint（并保证能通过 hook_mode+layer 唯一定位 SAE 权重与配置）

注意：
 - 这里只针对 1.3B T2V：dim=1536, num_layers=30, patch_size=(1,2,2)
 - 默认按真实采样流程跑完整 timesteps（30-50 步），但不调用 VAE.decode。
 - 为降低显存：每个 time step 只保留少量 token 特征，训练后立即释放并可选清理缓存。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Dict, List

# 修复模块导入路径：将项目根目录添加到 sys.path
# 这样无论从哪运行此脚本，都能正确找到 wan 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.cuda.amp as amp

from wan.configs.wan_t2v_1_3B import t2v_1_3B
from wan.modules.sae_new import SAEConfig, SparseAutoEncoder
from wan.sae.checkpoint_io import SAECheckpointIO, load_checkpoint, save_checkpoint
from wan.sae.configs import TrainConfig, load_train_config, save_config
from wan.sae.hooking import HookMode, pack_hook_batch, register_dit_hooks, remove_hooks
from wan.sae.logger import SAELogManager, get_train_logger
from wan.sae.prompt_io import PromptCleanConfig, batch_iter, load_prompts_from_dir
from wan.sae.sae_run_naming import SAERunLocator, load_json, save_json, train_state_path
from wan.text2video import WanT2V
# 使用 diffusers 的 Euler 调度器（更简单可靠，避免多步调度器的边界问题）


logger = logging.getLogger(__name__)


##########################################################################################
# 训练参数配置区域 - 可直接修改此区域的默认值
# 学术意义与建议值详见每个参数的注释
##########################################################################################

# --------------------------- 路径配置 ---------------------------
path_params = {
    # model_path: Wan 2.1 DiT 模型权重目录路径
    # 学术意义: DiT (Diffusion Transformer) 预训练权重，作为 SAE 可解释性分析的基础模型
    # 实际用法: 指向包含 Wan2.1-T2V-1.3B 权重的目录（注意：不是 SAE checkpoint 目录）
    # 建议值: "./Wan2.1-T2V-1.3B" 或绝对路径
    "model_path": "/root/Wan/Wan2.1-T2V-1.3B",

    # prompt_dir: 提示词文件夹路径
    # 学术意义: 用于激活 DiT 特定神经元的输入文本集合，NSFW 内容有助于激活特定概念神经元
    # 实际用法: 文件夹内包含多个 .txt 文件，每行一个提示词
    # 建议值: 准备至少 1000+ 条多样化提示词以获得更好的激活覆盖
    "prompt_dir": "final_cleaned",

    # run_dir: SAE 训练输出目录
    # 学术意义: 实验追踪与可复现性，每个实验应有独立的输出目录
    # 实际用法: 会自动在其下创建 {hook_mode}.layer{idx}/ 子目录保存各层 SAE
    # 建议值: "sae_runs/exp_{日期}" 如 "sae_runs/exp_20250319_blockout"
    "run_dir": "sae_runs/exp__20250324",
}

# --------------------------- 模型架构配置 ---------------------------
model_params = {
    # d_model: DiT 隐藏层维度，也是 SAE 输入维度
    # 学术意义: 与 DiT 模型维度一致，确保 SAE 能够重建完整的激活分布
    # 实际用法: 1.3B 模型固定为 1536，14B 模型为 5120
    # 默认值: 1536 (Wan 1.3B T2V)
    # 建议值: 保持默认值，与预训练模型一致
    "d_model": 1536,

    # d_hidden: SAE 扩展维度（隐空间维度）
    # 学术意义: 决定 SAE 能学习的特征数量，越大可学习越细粒度的概念，但计算成本增加
    # 学术参考: Anthropic 研究表明 d_hidden/d_model = 4~8 是较好的稀疏性-容量权衡
    # 实际用法: 每个输入 token 被编码为 d_hidden 维稀疏向量
    # 建议值: 6144 (4x) 或 12288 (8x)，显存充足可选 8x
    "d_hidden": 6144,

    # activation: 编码器激活函数
    # 学术意义: 影响稀疏性和梯度流动，ReLU 产生严格稀疏激活，GELU/SiLU 更平滑
    # 实际用法: 决定 encode 后的非线性变换
    # 可选值: "relu" | "gelu" | "silu"
    # 建议值: "relu"（经典 SAE 配置）或 "gelu"（训练更稳定）
    "activation": "relu",

    # sparsity: 稀疏化策略
    # 学术意义:
    #   - "topk": 强制每个样本只有 k 个非零激活，可解释性更强，便于找出"最活跃"特征
    #   - "l1": L1 正则化鼓励稀疏，但非零数量不固定，整体重建质量可能更好
    # 实际用法: 决定如何约束 SAE 隐空间的稀疏性
    # 可选值: "topk" | "l1"
    # 建议值: "topk"（更适合特征可视化与解释）
    "sparsity": "topk",

    # top_k: topk 稀疏策略下的保留数量
    # 学术意义: 控制每个 token 最多同时激活的特征数，直接影响可解释性
    # 学术参考: 通常设为 d_hidden 的 1%~10%，如 d_hidden=6144 时 top_k=64 约为 1%
    # 实际用法: 只保留激活值最大的 k 个维度，其余置零
    # 建议值: 64（d_hidden=6144 时约 1% 稀疏度），可尝试 32 或 128
    "top_k": 64,

    # l1_lambda: L1 稀疏正则权重（仅在 sparsity="l1" 时生效）
    # 学术意义: 控制 L1 惩罚强度，越大稀疏性越强，但可能牺牲重建质量
    # 实际用法: 损失函数中添加 l1_lambda * |z|.mean()
    # 建议值: 1e-3 ~ 1e-4，需在稀疏性和重建质量间权衡
    "l1_lambda": 1e-3,
}

# --------------------------- 训练流程配置 ---------------------------
training_params = {
    # steps: 总训练步数
    # 学术意义: 决定 SAE 收敛程度，太少会欠拟合，太多可能过拟合训练分布
    # 实际用法: 每一步处理 batch_prompts 个提示词，跑完整 sampling_steps 个 diffusion timestep
    # 建议值: 2000~5000，可观察 loss 曲线决定提前停止
    "steps": 500,

    # batch_prompts: 每步使用的提示词数量
    # 学术意义: 决定每步的样本多样性，影响梯度估计的方差
    # 实际用法: 每个提示词独立生成 latent，并行处理
    # 建议值: 4~8（显存允许可增大，但每个 timestep 显存占用随 batch 线性增长）
    "batch_prompts": 4,

    # max_prompts: 最大加载提示词数量
    # 学术意义: 限制数据集大小，避免内存溢出，也用于快速验证实验
    # 实际用法: 从 prompt_dir 中最多加载这么多条有效提示词
    # 建议值: 2000~10000，完整实验建议用全部数据
    "max_prompts": 1000,

    # sampling_steps: 每个 batch 的 diffusion 采样步数
    # 学术意义: 决定 DiT 前向传播深度，覆盖扩散过程的不同噪声水平
    # 实际用法: 每个 step 内会遍历这么多 timesteps，每步都 hook 并训练 SAE
    # 建议值: 30~50，越多覆盖噪声范围越完整，但训练时间线性增加
    "sampling_steps": 30,

    # sample_solver: (已弃用) 现在固定使用 Euler 调度器
    # 说明: 多步调度器（UniPC/DPM++）在训练模式下有复杂的边界问题
    # 修复: 改用 diffusers.FlowMatchEulerDiscreteScheduler，简单可靠
    "sample_solver": "euler",

    # shift: 噪声日程 shift 参数
    # 学术意义: 控制扩散过程的时间偏移，影响运动流畅度和生成稳定性
    # 实际用法: 越大视频帧过渡越平滑，但过大可能模糊
    # 建议值: 5.0（480P）或 3.0（720P），与 Wan2.1 官方推荐一致
    "shift": 5.0,

    # use_cfg: 是否使用 Classifier-Free Guidance
    # 学术意义: CFG 是文本条件视频生成的关键技术，使用 CFG 可使 SAE 学习到更准确的文本-视觉对齐特征
    # 实际用法: True 时每 timestep 跑两次 DiT（条件+无条件），计算和显存翻倍
    # 建议值: False（训练更快），若追求与真实生成分布一致可设为 True
    "use_cfg": False,

    # guide_scale: CFG 引导尺度
    # 学术意义: 控制条件信号强度，越大文本控制越强，但可能降低多样性
    # 实际用法: 仅在 use_cfg=True 时生效
    # 建议值: 5.0（Wan2.1 默认值）或 6.0（1.3B 模型推荐）
    "guide_scale": 5.0,

    # lr: SAE 优化器学习率
    # 学术意义: 控制 SAE 参数更新步长，影响收敛速度和稳定性
    # 实际用法: AdamW 优化器的学习率
    # 建议值: 1e-3 ~ 1e-4，可使用学习率调度器进一步优化
    "lr": 1e-3,

    # save_every: 检查点保存间隔
    # 学术意义: 实验容错与中间结果分析，便于从中断恢复或选择最优 checkpoint
    # 实际用法: 每这么多 step 保存一次 sae_step{step}.pt
    # 建议值: 200~500，太频繁会增加 IO 开销
    "save_every": 50,
}

# --------------------------- Hook 配置 ---------------------------
hook_params = {
    # hook_mode: Hook 模式，决定捕获 DiT 的哪些激活
    # 学术意义:
    #   - "self_attn": 自注意力输出，捕获空间-时间自相关模式（如物体形状、运动模式）
    #   - "cross_attn": 交叉注意力输出，捕获文本-视觉对齐特征（如特定概念、属性）
    #   - "self_and_cross": 同时捕获两种，更全面但计算和存储翻倍
    #   - "block_out": Transformer Block 完整输出，包含自注意+交叉注意+FFN，推荐用于残差分析
    # 实际用法: 决定 register_dit_hooks 的行为
    # 可选值: "self_attn" | "cross_attn" | "self_and_cross" | "block_out"
    # 建议值: "block_out"（最全面的残差表征）或 "cross_attn"（概念可解释性最强）
    "hook_mode": "block_out",

    # hook_layers: 要 hook 的层索引列表
    # 学术意义: 不同层编码不同抽象级别特征，浅层编码低级视觉特征，深层编码语义概念
    # 学术参考: LLM 可解释性研究表明中层（1/2~2/3 深度）通常最具可解释性
    # 实际用法: 用逗号分隔的层索引，如 "0,15,29" 表示第 1、16、30 层（共 30 层）
    # 建议值: "15,29"（中层+深层）或 "0,15,29"（浅中深三层对比）
    "hook_layers": "15",

    # max_tokens_per_key: 每个 hook key 每步最大 token 数
    # 学术意义: 限制每步参与 SAE 训练的 token 数量，平衡计算成本和统计显著性
    # 实际用法: 从 [B, L, C] 中最多采样这么多 token 展平为 [N, C]
    # 建议值: 65536（约 4k tokens * 16 batch），显存紧张可减小
    "max_tokens_per_key": 65536,
}

# --------------------------- 生成尺寸配置 ---------------------------
generation_params = {
    # size_w, size_h: 生成视频的宽高
    # 学术意义: 影响 latent 空间尺寸和 seq_len，进而影响 DiT 计算量和特征分布
    # 实际用法: 决定 latent shape [z_dim, t_lat, h_lat, w_lat]
    # 建议值: 480P (832x480) 训练更快，720P (1280x720) 质量更高
    "size_w": 832,
    "size_h": 480,

    # frame_num: 生成帧数
    # 学术意义: 决定时间维度长度，影响时序特征的学习
    # 实际用法: 必须是 4n+1（VAE 下采样要求），如 81 = 4*20 + 1
    # 建议值: 81（约 5 秒视频，16fps），更长视频增加显存占用
    "frame_num": 81,
}

# --------------------------- 内存优化配置 ---------------------------
memory_params = {
    # offload_text_encoder: 是否将 T5 文本编码器卸载到 CPU
    # 学术意义: 训练 SAE 时不需要更新文本编码器，可卸载以节省显存给 DiT 和 SAE
    # 实际用法: 每步文本编码后执行 .cpu()，需要时再 .to(device)
    # 建议值: True（8GB 以下显存强烈推荐）或 False（显存充足时减少数据传输）
    "offload_text_encoder": True,

    # empty_cache_every: 显存清理频率
    # 学术意义: 防止 PyTorch 显存碎片累积导致的 OOM
    # 实际用法: 每这么多 step 调用一次 torch.cuda.empty_cache()
    # 建议值: 0（不清理，默认）或 50~100（显存紧张时）
    "empty_cache_every": 0,
}

# --------------------------- 系统配置 ---------------------------
system_params = {
    # device_id: GPU 设备 ID
    # 实际用法: 指定使用哪张 GPU
    # 建议值: 0（单卡）或根据 nvidia-smi 选择空闲卡
    "device_id": 0,

    # seed: 随机种子
    # 学术意义: 保证实验可复现，相同种子产生相同的 prompt 采样和噪声初始化
    # 实际用法: 影响 torch、numpy、random 的随机状态
    # 建议值: 固定值如 42 或 0，对比实验时保持一致
    "seed": 0,
}

# --------------------------- 提示词清洗配置 ---------------------------
prompt_clean_params = {
    # min_len: 提示词最小长度
    # 实际用法: 过滤过短的提示词（可能是噪声或无效数据）
    # 建议值: 8
    "min_len": 8,

    # max_len: 提示词最大长度
    # 实际用法: 过滤过长的提示词（T5 最大 512 tokens，过长被截断）
    # 建议值: 400
    "max_len": 400,
}

# --------------------------- 日志配置 ---------------------------
log_params = {
    # log_interval: 日志打印间隔（步）
    # 实际用法: 每这么多 step 打印一次训练状态
    # 建议值: 10（频繁查看）或 50（减少日志量）
    "log_interval": 10,

    # eta_window: ETA 计算窗口大小
    # 实际用法: 用最近多少步的平均时间来估计剩余时间
    # 建议值: 50（平滑短期波动）
    "eta_window": 50,

    # log_to_file: 是否将日志持久化到文件
    # 实际用法: 自动保存控制台输出到 run_dir/training.log
    # 建议值: True（推荐，方便后续查看）
    "log_to_file": True,

    # loss_log_interval: 详细 loss 记录间隔（步）
    # 实际用法: 每这么多 step 记录每个 SAE 的 loss 到 JSONL 文件，用于后续可视化
    # 建议值: 1（每步都记录，精确但文件大）或 10（节省空间）
    "loss_log_interval": 1,
}

# --------------------------- 恢复训练配置 ---------------------------
resume_params = {
    # enabled: 是否启用恢复训练模式
    # 学术意义: 支持长实验中断后继续，不丢失训练进度；也支持迁移学习和增量训练
    # 实际用法: 当设置为 True 时，将尝试从 sae_checkpoint 路径或 run_dir 加载已有权重
    # 建议值: False（新实验）或 True（恢复/迁移学习）
    "enabled": False,

    # sae_checkpoint: 指定要恢复的 SAE checkpoint 路径
    # 学术意义: 明确指定权重来源，支持跨实验加载和迁移学习
    # 可选值:
    #   - 空字符串 "": 从 run_dir 下自动查找最新的 checkpoint
    #   - 具体路径: 如 "sae_runs/exp1/block_out.layer15/sae_latest.pt"
    #   - 目录路径: 如 "sae_runs/exp1"（自动加载该目录下所有层的 checkpoint）
    # 建议值: 留空（自动检测）或指定具体实验目录
    "sae_checkpoint": "",

    # additional_layers: 在恢复基础上新增的层
    # 学术意义: 支持增量训练，在已训练层基础上添加新层进行联合训练
    # 实际用法: 如果之前训练了层 15，现在想同时训练层 15 和 29，设置为 [29]
    # 格式: 层索引列表，如 [20, 25]
    # 建议值: []（无新增）或指定要新增的层索引
    "additional_layers": [],

    # frozen_layers: 要冻结的层（不参与训练）
    # 学术意义: 固定已训练好的层，只训练新增层，防止已学习特征被破坏
    # 实际用法: 格式为 ["block_out.layer15", "block_out.layer20"]
    # 建议值: []（全部训练）或冻结已有层只训练新增层
    "frozen_layers": [],

    # reset_optimizer: 是否重置优化器状态
    # 学术意义: 当改变学习率、冻结部分层或进行迁移学习时，重置优化器有助于稳定训练
    # 实际用法: 设置为 True 会丢弃已保存的优化器状态，使用新的优化器
    # 建议值: False（保持状态）或 True（改变训练配置时）
    "reset_optimizer": False,

    # reset_step_count: 是否重置步数计数器
    # 学术意义: 设置为 True 表示从 step=0 开始计数（但加载已有权重）
    # 实际用法: 用于迁移学习场景，新任务从新步数开始，但保留预训练权重
    # 建议值: False（继续计数）或 True（新任务/迁移学习）
    "reset_step_count": False,
}


##########################################################################################
# 核心代码区域 - 一般无需修改
##########################################################################################


def compute_latent_shape(cfg, size_wh, frame_num: int, vae_z_dim: int) -> List[int]:
    """计算 latent 形状，与 WanT2V.generate 逻辑一致。"""
    w, h = size_wh
    F = frame_num
    t_lat = (F - 1) // cfg.vae_stride[0] + 1
    h_lat = h // cfg.vae_stride[1]
    w_lat = w // cfg.vae_stride[2]
    return [vae_z_dim, t_lat, h_lat, w_lat]


def compute_seq_len(cfg, latent_shape, sp_size: int) -> int:
    """计算序列长度，与 WanT2V.generate 逻辑一致。"""
    _, t_lat, h_lat, w_lat = latent_shape
    seq_len = math.ceil((h_lat * w_lat) / (cfg.patch_size[1] * cfg.patch_size[2]) * t_lat / sp_size) * sp_size
    return int(seq_len)


def build_sae_config_from_params() -> SAEConfig:
    """从 model_params 构建 SAEConfig。"""
    return SAEConfig(
        d_model=model_params["d_model"],
        d_hidden=model_params["d_hidden"],
        activation=model_params["activation"],
        sparsity=model_params["sparsity"],
        top_k=model_params["top_k"],
        l1_lambda=model_params["l1_lambda"],
    )


def parse_layers(s: str) -> List[int]:
    """解析层索引字符串，如 "0,5,10,29" -> [0, 5, 10, 29]。"""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


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
    logger.info("SAE 训练配置")
    logger.info("=" * 60)
    for name, params in [
        ("路径配置", path_params),
        ("模型架构", model_params),
        ("训练流程", training_params),
        ("Hook 配置", hook_params),
        ("生成尺寸", generation_params),
        ("内存优化", memory_params),
        ("系统配置", system_params),
        ("提示词清洗", prompt_clean_params),
        ("日志配置", log_params),
        ("恢复训练", resume_params),
    ]:
        logger.info(f"\n【{name}】")
        for k, v in params.items():
            logger.info(f"  {k}: {v}")
    logger.info("=" * 60)


def parse_layers(s: str) -> List[int]:
    """解析层索引字符串，如 "0,5,10,29" -> [0, 5, 10, 29]。"""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def main():
    # 支持命令行参数覆盖默认配置
    parser = argparse.ArgumentParser(description="Train SAE with Wan 1.3B T2V DiT hooks (multi-timestep, no decode).")
    parser.add_argument("--config", type=str, default="", help="JSON 配置文件路径（优先级最高）")
    parser.add_argument("--dump_default_config", type=str, default="", help="导出默认配置到 JSON 文件并退出")

    # 允许命令行覆盖关键参数
    parser.add_argument("--model_path", type=str, default=path_params["model_path"],
                        help="Wan 2.1 DiT 模型权重目录路径（旧名称 --checkpoint_dir 仍兼容但已弃用）")
    parser.add_argument("--checkpoint_dir", type=str, default="",
                        help="[已弃用] 请使用 --model_path")
    parser.add_argument("--prompt_dir", type=str, default=path_params["prompt_dir"])
    parser.add_argument("--run_dir", type=str, default=path_params["run_dir"])
    parser.add_argument("--steps", type=int, default=training_params["steps"])
    parser.add_argument("--batch_prompts", type=int, default=training_params["batch_prompts"])
    parser.add_argument("--hook_layers", type=str, default=hook_params["hook_layers"])
    parser.add_argument("--device_id", type=int, default=system_params["device_id"])

    # 恢复训练参数（基于 resume_params）
    parser.add_argument("--resume", action="store_true", default=resume_params["enabled"],
                        help="启用恢复训练模式（相当于 --resume_enabled）")
    parser.add_argument("--resume_enabled", action="store_true", default=resume_params["enabled"],
                        help="是否启用恢复训练模式")
    parser.add_argument("--sae_checkpoint", type=str, default=resume_params["sae_checkpoint"],
                        help="指定要恢复的 SAE checkpoint 路径，空字符串表示自动检测")
    parser.add_argument("--additional_layers", type=str, default="",
                        help="新增层，如'20,25'（覆盖配置文件）")
    parser.add_argument("--frozen_layers", type=str, default="",
                        help="冻结层，如'block_out.layer15,block_out.layer20'（覆盖配置文件）")
    parser.add_argument("--reset_optimizer", action="store_true",
                        help="重置优化器状态（覆盖配置文件）")
    parser.add_argument("--reset_step_count", action="store_true",
                        help="重置步数计数器（覆盖配置文件）")

    args = parser.parse_args()

    # 解析 run_dir 用于日志文件路径（在 resolve_path 之前使用原始路径）
    run_dir_raw = args.run_dir or path_params["run_dir"]

    # 设置日志
    log_handlers = [logging.StreamHandler(sys.stdout)]

    # 添加文件日志处理器
    if log_params["log_to_file"]:
        # 确保日志目录存在
        log_dir = os.path.join(run_dir_raw, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, "training.log")
        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        log_handlers.append(file_handler)
        print(f"[INFO] 日志将同时保存到: {log_file_path}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=log_handlers,
        force=True,  # 允许重新配置（如果之前已设置）
    )

    # 导出默认配置
    if args.dump_default_config:
        default_config = TrainConfig()
        save_config(args.dump_default_config, default_config)
        logger.info("默认配置已导出到: %s", args.dump_default_config)
        return

    # 加载外部 JSON 配置（如果提供）
    if args.config:
        cfg_run = load_train_config(args.config)
        logger.info("已从 %s 加载配置", args.config)
    else:
        # 处理已弃用的 --checkpoint_dir 参数
        model_path = args.model_path
        if args.checkpoint_dir:
            logger.warning("--checkpoint_dir 已弃用，请使用 --model_path。当前仍兼容处理。")
            if not model_path:
                model_path = args.checkpoint_dir

        # 处理恢复参数（命令行覆盖配置文件）
        resume_enabled = args.resume or args.resume_enabled or resume_params["enabled"]
        sae_checkpoint = args.sae_checkpoint if args.sae_checkpoint else resume_params["sae_checkpoint"]
        reset_optimizer = args.reset_optimizer or resume_params["reset_optimizer"]
        reset_step_count = args.reset_step_count or resume_params["reset_step_count"]

        # 解析 additional_layers（命令行覆盖）
        additional_layers = resume_params["additional_layers"][:]
        if args.additional_layers:
            additional_layers = parse_layers(args.additional_layers)

        # 解析 frozen_layers（命令行覆盖）
        frozen_layers = resume_params["frozen_layers"][:]
        if args.frozen_layers:
            frozen_layers = [x.strip() for x in args.frozen_layers.split(",") if x.strip()]

        # 使用代码中的默认参数构建配置
        cfg_run = TrainConfig(
            checkpoint_dir=model_path or path_params["model_path"],
            prompt_dir=args.prompt_dir or path_params["prompt_dir"],
            device_id=args.device_id,
            seed=system_params["seed"],
            max_prompts=training_params["max_prompts"],
            batch_prompts=args.batch_prompts,
            steps=args.steps,
            sampling_steps=training_params["sampling_steps"],
            sample_solver=training_params["sample_solver"],
            shift=training_params["shift"],
            use_cfg=training_params["use_cfg"],
            guide_scale=training_params["guide_scale"],
            negative_prompt="",
        )
        # 填充嵌套配置
        cfg_run.ckpt.run_dir = args.run_dir or path_params["run_dir"]
        cfg_run.ckpt.save_every = training_params["save_every"]
        cfg_run.ckpt.resume = resume_enabled
        cfg_run.memory.offload_text_encoder = memory_params["offload_text_encoder"]
        cfg_run.memory.empty_cache_every = memory_params["empty_cache_every"]
        cfg_run.hook.hook_mode = hook_params["hook_mode"]
        cfg_run.hook.hook_layers = parse_layers(args.hook_layers)
        cfg_run.hook.max_tokens_per_key = hook_params["max_tokens_per_key"]
        cfg_run.size_w = generation_params["size_w"]
        cfg_run.size_h = generation_params["size_h"]
        cfg_run.frame_num = generation_params["frame_num"]
        cfg_run.sae = build_sae_config_from_params()
        cfg_run.prompt_clean = PromptCleanConfig(
            min_len=prompt_clean_params["min_len"],
            max_len=prompt_clean_params["max_len"],
        )

        # 保存恢复参数到配置对象供后续使用
        cfg_run.resume_config = {
            "enabled": resume_enabled,
            "sae_checkpoint": sae_checkpoint,
            "additional_layers": additional_layers,
            "frozen_layers": frozen_layers,
            "reset_optimizer": reset_optimizer,
            "reset_step_count": reset_step_count,
        }

    # 自动将相对路径转换为相对于脚本位置的绝对路径
    # 这样无论从哪里运行脚本，路径都能正确解析
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # wan/ 的上级是项目根目录

    def resolve_path(path: str) -> str:
        """将相对路径转换为绝对路径（相对于脚本位置），并展开 ~ 为家目录"""
        if not path:
            return path
        # 首先展开 ~ 为家目录
        path = os.path.expanduser(path)
        # 如果是绝对路径，直接返回
        if os.path.isabs(path):
            return path
        # 相对路径：解释为相对于项目根目录
        return os.path.join(project_root, path)

    cfg_run.checkpoint_dir = resolve_path(cfg_run.checkpoint_dir)  # 内部仍使用 checkpoint_dir 名称
    cfg_run.prompt_dir = resolve_path(cfg_run.prompt_dir)
    cfg_run.ckpt.run_dir = resolve_path(cfg_run.ckpt.run_dir)

    # 从恢复配置中提取参数（用于后续逻辑）
    resume_cfg = getattr(cfg_run, 'resume_config', {
        "enabled": resume_params["enabled"],
        "sae_checkpoint": resume_params["sae_checkpoint"],
        "additional_layers": resume_params["additional_layers"],
        "frozen_layers": resume_params["frozen_layers"],
        "reset_optimizer": resume_params["reset_optimizer"],
        "reset_step_count": resume_params["reset_step_count"],
    })

    # 打印配置
    _print_config()
    logger.info("Effective TrainConfig loaded")

    # 设置随机种子和设备
    torch.manual_seed(cfg_run.seed)
    device = torch.device(f"cuda:{cfg_run.device_id}" if torch.cuda.is_available() else "cpu")

    # 验证必要参数
    if not cfg_run.checkpoint_dir:
        raise ValueError("model_path 不能为空，请通过 --model_path 或修改 path_params['model_path'] 指定")
    if not cfg_run.prompt_dir:
        raise ValueError("prompt_dir 不能为空，请通过 --prompt_dir 或修改 path_params 指定")

    # 1) 载入 prompt 并清洗
    clean_cfg = cfg_run.prompt_clean
    prompts = load_prompts_from_dir(cfg_run.prompt_dir, clean_cfg=clean_cfg, limit=cfg_run.max_prompts)
    if not prompts:
        raise RuntimeError("没有加载到任何有效 prompt，请检查 prompt_dir 与清洗规则。")
    logger.info("从 %s 加载了 %d 条清洗后的 prompts", cfg_run.prompt_dir, len(prompts))

    # 显存监控
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        init_mem = torch.cuda.memory_allocated() / 1024**3
        logger.info("初始 GPU 显存占用: %.2f GB", init_mem)

    # 2) 构建 WanT2V（只用 text_encoder + model；训练时不需要 decode）
    cfg = t2v_1_3B
    logger.info("开始加载 WanT2V，checkpoint_dir=%s", cfg_run.checkpoint_dir)

    # 检查 checkpoint 目录是否存在
    if not os.path.exists(os.path.expanduser(cfg_run.checkpoint_dir)):
        raise FileNotFoundError(f"checkpoint_dir 不存在: {cfg_run.checkpoint_dir}")
    logger.info("checkpoint_dir 存在，开始初始化 WanT2V（这可能需要几分钟）...")

    wrapper = WanT2V(
        config=cfg,
        checkpoint_dir=cfg_run.checkpoint_dir,
        device_id=cfg_run.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )
    logger.info("WanT2V 初始化完成，开始加载模型到设备 %s", device)
    model = wrapper.model
    model.eval().requires_grad_(False)
    model.to(device)
    logger.info("WanT2V 和 WanModel 已在 %s 上初始化", device)

    # latent 形状与 seq_len
    vae_z_dim = wrapper.vae.model.z_dim
    latent_shape = compute_latent_shape(cfg, (cfg_run.size_w, cfg_run.size_h), cfg_run.frame_num, vae_z_dim)
    seq_len = compute_seq_len(cfg, latent_shape, wrapper.sp_size)
    logger.info("Latent shape=%s, seq_len=%d, sp_size=%d", latent_shape, seq_len, wrapper.sp_size)

    # 3) 为每个 hook key 创建 SAE + optimizer
    hook_layers = cfg_run.hook.hook_layers
    hook_mode: HookMode = cfg_run.hook.hook_mode  # type: ignore

    sae_cfg = cfg_run.sae
    saes: Dict[str, SparseAutoEncoder] = {}
    opts: Dict[str, torch.optim.Optimizer] = {}

    # 计算可能的 keys
    possible_keys: List[str] = []
    for layer in hook_layers:
        if hook_mode == "self_and_cross":
            possible_keys.extend([f"self_attn.layer{layer}", f"cross_attn.layer{layer}"])
        else:
            possible_keys.append(f"{hook_mode}.layer{layer}")

    # 处理增强恢复参数（优先使用 resume_cfg，支持命令行覆盖）
    resume_from = resume_cfg["sae_checkpoint"] if resume_cfg["enabled"] else ""
    additional_layers = resume_cfg["additional_layers"]
    frozen_layers = resume_cfg["frozen_layers"]
    reset_optimizer = resume_cfg["reset_optimizer"]
    reset_step_count = resume_cfg["reset_step_count"]

    # 构建层加载信息
    layer_load_info: Dict[str, Dict[str, Any]] = {}
    for key in possible_keys:
        mode, layer_str = key.split(".")
        layer_idx = int(layer_str.replace("layer", ""))

        # 确定源目录
        source_run_dir = resume_from

        # 检查是否是新增层
        is_new_layer = layer_idx in additional_layers

        # 检查是否冻结
        is_frozen = key in frozen_layers

        layer_load_info[key] = {
            "source_run_dir": source_run_dir,
            "is_new_layer": is_new_layer,
            "is_frozen": is_frozen,
            "layer_idx": layer_idx,
            "mode": mode,
        }

    # 初始化或恢复 SAE（默认存储在 CPU，训练时动态加载到 GPU）
    # 显存优化策略：多个层同时训练时，避免所有 SAE 同时占用 GPU 显存
    for key in possible_keys:
        info = layer_load_info[key]
        mode = info["mode"]
        layer_idx = info["layer_idx"]
        is_new_layer = info["is_new_layer"]
        is_frozen = info["is_frozen"]

        # 目标位置（新实验目录）
        loc_target = SAERunLocator(run_dir=cfg_run.ckpt.run_dir, hook_mode=mode, layer_idx=layer_idx)

        # 源位置（恢复目录）
        source_run_dir = info["source_run_dir"]
        if source_run_dir and source_run_dir != cfg_run.ckpt.run_dir:
            loc_source = SAERunLocator(run_dir=source_run_dir, hook_mode=mode, layer_idx=layer_idx)
        else:
            loc_source = loc_target

        # 判断是否需要从 checkpoint 恢复
        should_load_ckpt = (not is_new_layer or loc_source.latest_ckpt_path().exists())

        # 使用新的统一 IO 接口加载 checkpoint（自动兼容新旧格式）
        sae_cfg_to_use = sae_cfg  # 默认使用代码配置
        sae = None
        ckpt_loaded = False

        if should_load_ckpt and loc_source.latest_ckpt_path().exists():
            try:
                # 使用 SAECheckpointIO 加载（自动处理新/旧格式）
                io = SAECheckpointIO.load(
                    loc_source,
                    device="cpu",
                    strict=True,
                    allow_legacy=True,  # 允许从旧格式（.json）回退加载
                )
                sae = io.sae
                sae_cfg_to_use = io.sae_config
                ckpt_loaded = True

                # 记录配置来源
                if io._config_source == "checkpoint":
                    logger.info("从 checkpoint 内置配置恢复 SAE: %s (d_hidden=%d, top_k=%d)",
                               key, sae_cfg_to_use.d_hidden, sae_cfg_to_use.top_k)
                elif io._config_source == "json_fallback":
                    logger.info("从旧格式 .json 恢复 SAE: %s (d_hidden=%d, top_k=%d) [建议迁移]",
                               key, sae_cfg_to_use.d_hidden, sae_cfg_to_use.top_k)
                else:
                    logger.info("从 checkpoint 恢复 SAE: %s", key)

                # 如果配置与代码不同，提示用户
                if sae_cfg_to_use.to_dict() != sae_cfg.to_dict():
                    logger.info("  注意: checkpoint 配置与代码配置不同，使用 checkpoint 配置")

            except Exception as e:
                if is_new_layer:
                    # 新增层加载失败，使用代码配置初始化
                    logger.warning("加载 checkpoint 失败，使用代码配置初始化新层 %s: %s", key, str(e))
                    sae_cfg_to_use = sae_cfg
                else:
                    # 恢复层加载失败，报错
                    raise RuntimeError(f"无法恢复 SAE {key}: {e}") from e

        # 如果未加载成功，使用代码配置初始化
        if sae is None:
            logger.info("初始化新 SAE: %s", key)
            sae = SparseAutoEncoder(sae_cfg_to_use)

        # 冻结处理
        if is_frozen:
            logger.info("冻结层: %s（不参与训练）", key)
            for param in sae.parameters():
                param.requires_grad = False

        # SAE 保留在 CPU
        saes[key] = sae

        # 优化器（冻结层不创建优化器）
        if not is_frozen:
            opt = torch.optim.AdamW(sae.parameters(), lr=training_params["lr"])
            opts[key] = opt
        else:
            opts[key] = None  # type: ignore

        # 保存配置到目标目录
        save_json(
            loc_target.config_path(),
            {
                "sae": sae_cfg_to_use.to_dict(),
                "hook": {"hook_mode": mode, "layer_idx": layer_idx},
            },
        )

    # 统计信息
    num_total = len(possible_keys)
    num_new = sum(1 for k in possible_keys if layer_load_info[k]["is_new_layer"])
    num_frozen = sum(1 for k in possible_keys if layer_load_info[k]["is_frozen"])
    logger.info("SAE 初始化完成: 总共=%d, 新增=%d, 冻结=%d, 可训练=%d",
                num_total, num_new, num_frozen, num_total - num_frozen - num_new)

    # 4) 训练循环
    prompt_batches = list(batch_iter(prompts, batch_size=cfg_run.batch_prompts, shuffle=True, seed=cfg_run.seed))
    if not prompt_batches:
        raise RuntimeError("prompt_batches 为空（可能是 batch_prompts 太大或 prompts 为空）。")
    logger.info("总 batch 数=%d, batch_size=%d", len(prompt_batches), cfg_run.batch_prompts)

    # 读取/初始化训练状态
    state_path = train_state_path(cfg_run.ckpt.run_dir)

    # 确定是否恢复训练状态
    should_resume_state = cfg_run.ckpt.resume and state_path.exists() and not reset_step_count

    # 如果有指定 sae_checkpoint 但不同目录，尝试从源目录恢复状态
    if resume_from and resume_from != cfg_run.ckpt.run_dir and os.path.exists(resume_from):
        source_state_path = train_state_path(resume_from)
        if source_state_path.exists() and not reset_step_count:
            state = load_json(source_state_path)
            start_step = int(state.get("step", 0))
            logger.info("从源目录 %s 恢复训练状态，step=%d", resume_from, start_step)
        else:
            start_step = 0
            logger.info("无法从源目录恢复状态，从头开始训练 (step=0)")
    elif should_resume_state:
        state = load_json(state_path)
        start_step = int(state.get("step", 0))
        logger.info("从 step=%d 恢复训练 (总步数=%d)", start_step, cfg_run.steps)
    else:
        start_step = 0
        if reset_step_count:
            logger.info("重置步数计数器，从头开始训练 (step=0)")
        else:
            logger.info("从头开始训练 (step=0)")

    step = start_step

    # 初始化时间统计
    step_times: List[float] = []
    last_log_time = time.time()
    train_start_time = time.time()

    # 初始化 loss 记录（用于后续可视化）
    loss_history: List[Dict] = []
    loss_log_path = os.path.join(cfg_run.ckpt.run_dir, "logs", "loss_history.jsonl")
    loss_csv_path = os.path.join(cfg_run.ckpt.run_dir, "logs", "loss_history.csv")
    if log_params["log_to_file"]:
        os.makedirs(os.path.dirname(loss_log_path), exist_ok=True)
        # 写入 CSV 头（扁平化格式，每行一个 SAE 的指标）
        if not os.path.exists(loss_csv_path):
            with open(loss_csv_path, "w", encoding="utf-8") as f:
                f.write("step,timestamp,sae_key,loss,recon_mse,l2_norm,sparsity,num_activations\n")
        logger.info("Loss 历史将保存到: %s 和 %s", loss_log_path, loss_csv_path)

    try:
        for it in range(start_step, cfg_run.steps):
            step_start = time.time()
            batch_prompts = prompt_batches[it % len(prompt_batches)]
            B = len(batch_prompts)

            # 4.2 文本编码
            wrapper.text_encoder.model.to(device)
            context = wrapper.text_encoder(batch_prompts, device)
            if cfg_run.memory.offload_text_encoder:
                wrapper.text_encoder.model.cpu()

            # 4.3 构造初始噪声 latent
            seed_g = torch.Generator(device=device)
            seed_g.manual_seed(cfg_run.seed + it)
            latents = [
                torch.randn(
                    latent_shape[0],
                    latent_shape[1],
                    latent_shape[2],
                    latent_shape[3],
                    dtype=torch.float32,
                    device=device,
                    generator=seed_g,
                )
                for _ in range(B)
            ]

            # 4.4 构造时间步序列
            # 手动构造时间步，避免使用调度器（防止边界问题）
            # 使用简单的线性时间步
            timesteps = torch.linspace(
                cfg.num_train_timesteps - 1, 0, cfg_run.sampling_steps,
                device=device, dtype=torch.long
            )

            # CFG 准备
            if cfg_run.negative_prompt == "":
                n_prompt = wrapper.sample_neg_prompt
            else:
                n_prompt = cfg_run.negative_prompt
            if cfg_run.use_cfg:
                wrapper.text_encoder.model.to(device)
                context_null = wrapper.text_encoder([n_prompt] * B, device)
                if cfg_run.memory.offload_text_encoder:
                    wrapper.text_encoder.model.cpu()
            else:
                context_null = None

            # 4.5 注册 hooks 并训练
            raw: Dict[str, torch.Tensor] = {}

            def on_tensor(k: str, v: torch.Tensor):
                raw[k] = v  # [B, L, C]

            handles = register_dit_hooks(model, hook_layers=hook_layers, hook_mode=hook_mode, on_tensor=on_tensor)
            try:
                for t in timesteps:
                    raw.clear()
                    timestep = torch.stack([t]).repeat(B)  # [B]

                    # DiT 推理（无梯度）
                    with torch.no_grad():
                        with torch.amp.autocast(device_type="cuda", dtype=cfg.param_dtype):
                            noise_pred_cond = model(latents, t=timestep, context=context, seq_len=seq_len)
                            if cfg_run.use_cfg:
                                noise_pred_uncond = model(latents, t=timestep, context=context_null, seq_len=seq_len)
                                pred = [
                                    u + cfg_run.guide_scale * (c - u)
                                    for c, u in zip(noise_pred_cond, noise_pred_uncond)
                                ]
                                del noise_pred_uncond
                            else:
                                pred = noise_pred_cond

                            # 收集特征
                            hook_batch = pack_hook_batch(raw, max_tokens_per_key=cfg_run.hook.max_tokens_per_key)

                            # 更新 latent
                            # 注意：调度器 step() 会递增内部计数器，每个 timestep 只能调用一次
                            # 手动实现 Euler 更新以避免多步调度器的状态管理问题
                            # Euler 更新公式: x_{t-1} = x_t - v_t * (sigma_t - sigma_{t-1})
                            dt = 1.0 / cfg_run.sampling_steps  # 简单的均匀步长
                            new_latents: List[torch.Tensor] = []
                            for p, z in zip(pred, latents):
                                # 手动 Euler 更新: z_next = z - pred * dt
                                z_next = z - p * dt
                                new_latents.append(z_next)
                            latents = new_latents

                            del pred, noise_pred_cond

                    # 训练 SAE（有梯度）- 在 torch.no_grad() 外部
                    step_metrics: Dict[str, Dict] = {}  # 记录每步各 SAE 的详细指标
                    for key, feats in hook_batch.items():
                        sae = saes[key]
                        opt = opts[key]
                        # 将 SAE 加载到 GPU 进行训练
                        sae.to(device)
                        sae.train()
                        feats = feats.to(device)
                        z, x_recon, loss = sae(feats, return_loss=True)

                        # 计算详细指标用于可视化
                        with torch.no_grad():
                            # Reconstruction MSE
                            recon_mse = ((x_recon - feats) ** 2).mean().item()
                            # L2 范数（权重衰减效果）
                            l2_norm = sum(p.pow(2).sum().item() for p in sae.parameters())
                            # 稀疏度（非零激活比例）
                            sparsity = (z.abs() > 1e-6).float().mean().item()
                            # 激活数量（TopK 实际值）
                            num_activations = (z.abs() > 1e-6).sum().item() / z.shape[0]

                        step_metrics[key] = {
                            "loss": loss.item(),
                            "recon_mse": recon_mse,
                            "l2_norm": l2_norm,
                            "sparsity": sparsity,
                            "num_activations": num_activations,
                        }

                        # 只更新非冻结层
                        opt = opts[key]
                        if opt is not None:
                            opt.zero_grad()
                            loss.backward()
                            opt.step()
                        # 训练完成后移回 CPU，释放 GPU 显存给下一个 SAE
                        sae.to("cpu")
                        del feats, loss, z, x_recon

                    del hook_batch
            finally:
                remove_hooks(handles)
                try:
                    del timesteps
                except Exception:
                    pass
                try:
                    del latents
                except Exception:
                    pass

            step += 1

            # 更新时间统计
            step_end = time.time()
            step_time = step_end - step_start
            step_times.append(step_time)
            if len(step_times) > log_params["eta_window"]:
                step_times.pop(0)

            # 记录每步各 SAE 的详细指标（用于后续可视化）
            if step_metrics and (step % log_params["loss_log_interval"] == 0 or step == 1):
                loss_record = {
                    "step": step,
                    "timestamp": time.time(),
                    "elapsed": step_end - train_start_time,
                    "step_time": step_time,
                    "metrics": step_metrics,
                }
                loss_history.append(loss_record)

                # 实时追加到 JSONL 文件
                if log_params["log_to_file"]:
                    with open(loss_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(loss_record, ensure_ascii=False) + "\n")

                    # 追加到 CSV 文件（扁平化格式，方便 pandas 读取）
                    with open(loss_csv_path, "a", encoding="utf-8") as f:
                        for key, metrics in step_metrics.items():
                            row = f"{step},{time.time():.3f},{key},{metrics['loss']:.6f},{metrics['recon_mse']:.6f},{metrics['l2_norm']:.4f},{metrics['sparsity']:.6f},{metrics['num_activations']:.2f}\n"
                            f.write(row)

            # 保存训练状态
            save_json(
                state_path,
                {
                    "step": step,
                    "max_steps": cfg_run.steps,
                    "hook_mode": hook_mode,
                    "hook_layers": hook_layers,
                    "sae_config": sae_cfg.to_dict(),
                    "sampling_steps": cfg_run.sampling_steps,
                    "sample_solver": cfg_run.sample_solver,
                    "seed": cfg_run.seed,
                },
            )

            # 打印日志（带 ETA 估计和显存监控）
            if step % log_params["log_interval"] == 0 or step == 1:
                avg_step_time = sum(step_times) / len(step_times)
                remaining_steps = cfg_run.steps - step
                eta_seconds = avg_step_time * remaining_steps
                elapsed = step_end - train_start_time

                mem_info = ""
                if torch.cuda.is_available():
                    cur_mem = torch.cuda.memory_allocated() / 1024**3
                    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
                    mem_info = f" cur_mem={cur_mem:.2f}GB peak_mem={peak_mem:.2f}GB"

                # 构建详细的指标字符串
                metrics_strs = []
                for key, metrics in step_metrics.items():
                    m = metrics
                    metrics_strs.append(
                        f"{key}: loss={m['loss']:.4f} mse={m['recon_mse']:.4f} "
                        f"spar={m['sparsity']:.3f} acts={m['num_activations']:.1f}"
                    )
                metrics_line = " | ".join(metrics_strs)

                logger.info(
                    "[%d/%d] batch=%d step_time=%.2fs elapsed=%s ETA=%s%s",
                    step, cfg_run.steps, B,
                    avg_step_time, format_time(elapsed), format_time(eta_seconds), mem_info
                )
                logger.info("  %s", metrics_line)

            # 保存 checkpoint（使用新格式，配置内置）
            if step % cfg_run.ckpt.save_every == 0 or step == cfg_run.steps:
                logger.info("保存 checkpoint 于 step=%d", step)
                for key, sae in saes.items():
                    mode, layer_str = key.split(".")
                    layer_idx = int(layer_str.replace("layer", ""))
                    loc = SAERunLocator(run_dir=cfg_run.ckpt.run_dir, hook_mode=mode, layer_idx=layer_idx)
                    # 使用新的统一 IO 接口保存（配置自动内置到 .pt）
                    io = SAECheckpointIO(
                        sae=sae,
                        step=step,
                        hook_mode=mode,
                        layer_idx=layer_idx,
                        extra_info={
                            "run_dir": cfg_run.ckpt.run_dir,
                            "timestamp": time.time(),
                        },
                    )
                    io.save(loc, save_legacy_json=True)  # 同时保留 .json 便于查看

            # 显存清理
            if cfg_run.memory.empty_cache_every and (step % cfg_run.memory.empty_cache_every == 0) and torch.cuda.is_available():
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logger.warning("训练被用户中断于 step=%d，保存状态...", step)
        save_json(
            state_path,
            {
                "step": step,
                "max_steps": cfg_run.steps,
                "hook_mode": hook_mode,
                "hook_layers": hook_layers,
                "sae_config": sae_cfg.to_dict(),
                "sampling_steps": cfg_run.sampling_steps,
                "sample_solver": cfg_run.sample_solver,
                "seed": cfg_run.seed,
            },
        )
        for key, sae in saes.items():
            mode, layer_str = key.split(".")
            layer_idx = int(layer_str.replace("layer", ""))
            loc = SAERunLocator(run_dir=cfg_run.ckpt.run_dir, hook_mode=mode, layer_idx=layer_idx)
            # 使用新的统一 IO 接口保存
            io = SAECheckpointIO(
                sae=sae,
                step=step,
                hook_mode=mode,
                layer_idx=layer_idx,
                extra_info={"interrupted": True},
            )
            io.save(loc, save_legacy_json=True)
        logger.info("状态和 SAE 权重已保存，可用 --resume 恢复训练。")

    except Exception as e:
        logger.exception("训练过程中发生错误: %s", e)
        save_json(
            state_path,
            {
                "step": step,
                "max_steps": cfg_run.steps,
                "hook_mode": hook_mode,
                "hook_layers": hook_layers,
                "sae_config": sae_cfg.to_dict(),
                "sampling_steps": cfg_run.sampling_steps,
                "sample_solver": cfg_run.sample_solver,
                "seed": cfg_run.seed,
                "error": str(e),
            },
        )
        for key, sae in saes.items():
            mode, layer_str = key.split(".")
            layer_idx = int(layer_str.replace("layer", ""))
            loc = SAERunLocator(run_dir=cfg_run.ckpt.run_dir, hook_mode=mode, layer_idx=layer_idx)
            # 使用新的统一 IO 接口保存
            io = SAECheckpointIO(
                sae=sae,
                step=step,
                hook_mode=mode,
                layer_idx=layer_idx,
                extra_info={"error": str(e)},
            )
            io.save(loc, save_legacy_json=True)
        raise

    total_time = time.time() - train_start_time
    logger.info("训练完成于 step=%d，总耗时 %s", step, format_time(total_time))


if __name__ == "__main__":
    main()
