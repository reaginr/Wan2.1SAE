"""
Wan 1.3B 文生视频（T2V）SAE 测试脚本：
- 批量读取/清洗 prompt
- 按 hook 模式/层列表 hook DiT
- 加载对应的 SAE（通过 hook_mode + layer_idx 唯一定位 run_dir 下的 sae_latest.pt + sae_config.json）
- 保存（prompt, hook类型, hook层数, SAE 中间层 z 的数值）

注意：保存完整 token 级 z 可能非常大。这里默认保存"按 token 维均值"的 z_mean:
    z_mean: [B, d_hidden]（每个 prompt 一条向量）
并额外提供 top-k 稀疏信息（如果 SAE 用 topk）。

同时预留批量可视化伪代码入口（后续迭代）。
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.cuda.amp as amp

from wan.configs.wan_t2v_1_3B import t2v_1_3B
from wan.modules.sae_new import SAEConfig, SparseAutoEncoder
from wan.sae.hooking import HookMode, register_dit_hooks, remove_hooks
from wan.sae.prompt_io import PromptCleanConfig, batch_iter, load_prompts_from_dir
from wan.sae.sae_run_naming import SAERunLocator, load_json
from wan.text2video import WanT2V


logger = logging.getLogger(__name__)


##########################################################################################
# 测试参数配置区域 - 可直接修改此区域的默认值
# 学术意义与建议值详见每个参数的注释
##########################################################################################

# --------------------------- 路径配置 ---------------------------
path_params = {
    # checkpoint_dir: Wan 1.3B 模型权重目录路径
    # 学术意义: DiT 预训练权重，用于生成激活 SAE 的隐藏状态
    # 实际用法: 指向包含 Wan2.1-T2V-1.3B 权重的目录
    # 建议值: "./Wan2.1-T2V-1.3B"
    "checkpoint_dir": "",

    # prompt_dir: 测试用提示词文件夹路径
    # 学术意义: 用于分析 SAE 在特定概念/风格上的激活模式的输入集合
    # 实际用法: 与训练时使用相同格式，可不同内容用于验证泛化性
    # 建议值: 准备与训练集分布不同但相关的提示词，评估泛化能力
    "prompt_dir": "",

    # run_dir: SAE 训练输出目录（训练脚本的 run_dir）
    # 学术意义: 从中加载训练好的 SAE 权重和配置
    # 实际用法: 与 sae_train_t2v_1_3b.py 中指定的 run_dir 相同
    # 建议值: 与训练时一致，如 "sae_runs/exp_20250319_blockout"
    "run_dir": "",

    # output_path: 测试结果保存路径
    # 学术意义: 保存 SAE 编码结果的文件，用于后续分析和可视化
    # 实际用法: 保存为 .pt 文件，包含每个 prompt 的 z_mean 和元信息
    # 建议值: 放在 run_dir 下，如 "{run_dir}/test_results.pt"
    "output_path": "sae_test_out.pt",
}

# --------------------------- Hook 配置 ---------------------------
hook_params = {
    # hook_mode: Hook 模式
    # 学术意义: 必须与训练时使用的 hook_mode 一致，否则加载错误的 SAE
    # 可选值: "self_attn" | "cross_attn" | "self_and_cross" | "block_out"
    # 建议值: 与训练配置完全一致
    "hook_mode": "block_out",

    # hook_layers: 要 hook 的层索引列表
    # 学术意义: 必须与训练时 hook 的层一致
    # 实际用法: 用逗号分隔的层索引，如 "0,15,29"
    # 建议值: 与训练配置完全一致
    "hook_layers": "29",
}

# --------------------------- 批处理配置 ---------------------------
batch_params = {
    # batch_prompts: 每批处理的提示词数量
    # 学术意义: 影响显存占用和并行效率
    # 实际用法: 比训练时可更大，因为不需要反向传播
    # 建议值: 8~16（显存允许可更大）
    "batch_prompts": 4,

    # max_prompts: 最大测试提示词数量
    # 学术意义: 限制测试集大小，用于快速验证或完整分析
    # 实际用法: 从 prompt_dir 中最多加载这么多条
    # 建议值: 500~2000，完整分析用全部数据
    "max_prompts": 500,
}

# --------------------------- 生成尺寸配置 ---------------------------
generation_params = {
    # size_w, size_h: 生成视频的宽高
    # 学术意义: 影响 DiT 的 seq_len 和特征分布
    # 重要: 必须与训练时使用相同的尺寸，否则特征分布可能不同
    # 建议值: 与训练配置一致（832x480 或 1280x720）
    "size_w": 832,
    "size_h": 480,

    # frame_num: 生成帧数
    # 学术意义: 影响时间维度的特征提取
    # 重要: 建议与训练时一致以保证特征分布一致
    # 建议值: 81（与训练配置一致）
    "frame_num": 81,
}

# --------------------------- 提示词清洗配置 ---------------------------
prompt_clean_params = {
    # min_len: 提示词最小长度
    # 实际用法: 过滤过短的提示词
    # 建议值: 8（与训练一致）
    "min_len": 8,

    # max_len: 提示词最大长度
    # 实际用法: 过滤过长的提示词
    # 建议值: 400（与训练一致）
    "max_len": 400,
}

# --------------------------- 系统配置 ---------------------------
system_params = {
    # device_id: GPU 设备 ID
    # 建议值: 0（单卡）或选择空闲卡
    "device_id": 0,

    # seed: 随机种子
    # 学术意义: 保证测试可复现，相同种子产生相同的噪声初始化
    # 建议值: 固定值如 0 或 42
    "seed": 0,
}

# --------------------------- 日志配置 ---------------------------
log_params = {
    # log_interval: 日志打印间隔（批次数）
    # 实际用法: 每处理这么多 batches 打印一次进度
    # 建议值: 10
    "log_interval": 10,
}


##########################################################################################
# 核心代码区域 - 一般无需修改
##########################################################################################


def compute_latent_shape(cfg, size_wh, frame_num: int, vae_z_dim: int) -> List[int]:
    """计算 latent 形状。"""
    w, h = size_wh
    F = frame_num
    t_lat = (F - 1) // cfg.vae_stride[0] + 1
    h_lat = h // cfg.vae_stride[1]
    w_lat = w // cfg.vae_stride[2]
    return [vae_z_dim, t_lat, h_lat, w_lat]


def compute_seq_len(cfg, latent_shape, sp_size: int) -> int:
    """计算序列长度。"""
    _, t_lat, h_lat, w_lat = latent_shape
    seq_len = math.ceil((h_lat * w_lat) / (cfg.patch_size[1] * cfg.patch_size[2]) * t_lat / sp_size) * sp_size
    return int(seq_len)


def parse_layers(s: str) -> List[int]:
    """解析层索引字符串。"""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def load_sae_for_key(run_dir: str, hook_mode: str, layer_idx: int, device) -> SparseAutoEncoder:
    """
    加载指定 hook_mode 和 layer_idx 的 SAE。

    参数:
        run_dir: SAE 训练输出目录
        hook_mode: hook 模式（如 "block_out"）
        layer_idx: 层索引
        device: 目标设备

    返回:
        加载并设为 eval 模式的 SparseAutoEncoder
    """
    loc = SAERunLocator(run_dir=run_dir, hook_mode=hook_mode, layer_idx=layer_idx)
    cfg = load_json(loc.config_path()).get("sae")
    if not cfg:
        raise FileNotFoundError(f"缺少 {loc.key()} 的 sae_config.json: {loc.config_path()}")
    sae_cfg = SAEConfig(**cfg)
    sae = SparseAutoEncoder(sae_cfg).to(device)
    ckpt = torch.load(loc.latest_ckpt_path(), map_location=device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae


def _print_config():
    """打印所有配置参数。"""
    logger.info("=" * 60)
    logger.info("SAE 测试配置")
    logger.info("=" * 60)
    for name, params in [
        ("路径配置", path_params),
        ("Hook 配置", hook_params),
        ("批处理", batch_params),
        ("生成尺寸", generation_params),
        ("提示词清洗", prompt_clean_params),
        ("系统配置", system_params),
        ("日志配置", log_params),
    ]:
        logger.info(f"\n【{name}】")
        for k, v in params.items():
            logger.info(f"  {k}: {v}")
    logger.info("=" * 60)


def main():
    # 参数解析（允许命令行覆盖默认配置）
    parser = argparse.ArgumentParser(description="Test SAE with Wan 1.3B T2V DiT hooks (single-step forward).")
    parser.add_argument("--checkpoint_dir", type=str, default=path_params["checkpoint_dir"])
    parser.add_argument("--prompt_dir", type=str, default=path_params["prompt_dir"])
    parser.add_argument("--run_dir", type=str, default=path_params["run_dir"])
    parser.add_argument("--output_path", type=str, default=path_params["output_path"])
    parser.add_argument("--hook_mode", type=str, default=hook_params["hook_mode"],
                        choices=["self_attn", "cross_attn", "self_and_cross", "block_out"])
    parser.add_argument("--hook_layers", type=str, default=hook_params["hook_layers"])
    parser.add_argument("--batch_prompts", type=int, default=batch_params["batch_prompts"])
    parser.add_argument("--max_prompts", type=int, default=batch_params["max_prompts"])
    parser.add_argument("--size_w", type=int, default=generation_params["size_w"])
    parser.add_argument("--size_h", type=int, default=generation_params["size_h"])
    parser.add_argument("--frame_num", type=int, default=generation_params["frame_num"])
    parser.add_argument("--device_id", type=int, default=system_params["device_id"])
    parser.add_argument("--seed", type=int, default=system_params["seed"])
    parser.add_argument("--min_len", type=int, default=prompt_clean_params["min_len"])
    parser.add_argument("--max_len", type=int, default=prompt_clean_params["max_len"])

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # 打印配置
    _print_config()

    # 验证必要参数
    if not args.checkpoint_dir:
        raise ValueError("checkpoint_dir 不能为空")
    if not args.prompt_dir:
        raise ValueError("prompt_dir 不能为空")
    if not args.run_dir:
        raise ValueError("run_dir 不能为空（SAE 训练输出目录）")

    # 设置随机种子和设备
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    # 加载并清洗提示词
    clean_cfg = PromptCleanConfig(min_len=args.min_len, max_len=args.max_len)
    prompts = load_prompts_from_dir(args.prompt_dir, clean_cfg=clean_cfg, limit=args.max_prompts)
    if not prompts:
        raise RuntimeError("没有加载到任何有效 prompt。")
    logger.info("加载了 %d 条提示词", len(prompts))

    # 构建 WanT2V
    cfg = t2v_1_3B
    wrapper = WanT2V(
        config=cfg,
        checkpoint_dir=args.checkpoint_dir,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )
    model = wrapper.model
    model.eval().requires_grad_(False).to(device)
    logger.info("WanT2V 已加载到 %s", device)

    # 计算 latent 形状
    vae_z_dim = wrapper.vae.model.z_dim
    latent_shape = compute_latent_shape(cfg, (args.size_w, args.size_h), args.frame_num, vae_z_dim)
    seq_len = compute_seq_len(cfg, latent_shape, wrapper.sp_size)
    logger.info("Latent shape=%s, seq_len=%d", latent_shape, seq_len)

    # 解析 hook 层
    hook_layers = parse_layers(args.hook_layers)
    hook_mode: HookMode = args.hook_mode  # type: ignore

    # 加载所有需要的 SAE
    saes: Dict[str, SparseAutoEncoder] = {}
    keys: List[str] = []
    for layer in hook_layers:
        if hook_mode == "self_and_cross":
            keys.extend([f"self_attn.layer{layer}", f"cross_attn.layer{layer}"])
        else:
            keys.append(f"{hook_mode}.layer{layer}")

    logger.info("加载 %d 个 SAE...", len(keys))
    for key in keys:
        mode, layer_str = key.split(".")
        layer_idx = int(layer_str.replace("layer", ""))
        saes[key] = load_sae_for_key(args.run_dir, hook_mode=mode, layer_idx=layer_idx, device=device)
        logger.info("  已加载: %s", key)

    # 测试结果收集
    results: List[Dict[str, Any]] = []
    batch_count = 0
    total_batches = (len(prompts) + args.batch_prompts - 1) // args.batch_prompts

    # 批次处理
    for batch in batch_iter(prompts, batch_size=args.batch_prompts, shuffle=False):
        B = len(batch)
        batch_count += 1

        # 文本编码
        wrapper.text_encoder.model.to(device)
        context = wrapper.text_encoder(batch, device)

        # 随机时间步和噪声 latent（单步前向）
        t = torch.randint(low=0, high=cfg.num_train_timesteps, size=(B,), device=device, dtype=torch.long)
        x_list = [torch.randn(*latent_shape, device=device, dtype=torch.float32) for _ in range(B)]

        # Hook 收集激活
        raw: Dict[str, torch.Tensor] = {}

        def on_tensor(k: str, v: torch.Tensor):
            raw[k] = v  # [B, L, C]

        handles = register_dit_hooks(model, hook_layers=hook_layers, hook_mode=hook_mode, on_tensor=on_tensor)
        try:
            with torch.no_grad(), amp.autocast(dtype=cfg.param_dtype):
                _ = model(x_list, t=t, context=context, seq_len=seq_len)
        finally:
            remove_hooks(handles)

        # 对每个 key 计算 SAE 编码
        for key, act in raw.items():
            sae = saes.get(key)
            if sae is None:
                continue
            # act: [B, L, C] -> [B*L, C]
            b, l, c = act.shape
            x_flat = act.reshape(b * l, c).to(device)
            with torch.no_grad():
                z, topk_idx, topk_val = sae.encode(x_flat)  # z: [B*L, d_hidden]
            z = z.view(b, l, -1)  # [B, L, d_hidden]
            z_mean = z.mean(dim=1).cpu()  # [B, d_hidden]

            mode, layer_str = key.split(".")
            layer_idx = int(layer_str.replace("layer", ""))

            for i in range(B):
                item: Dict[str, Any] = {
                    "prompt": batch[i],
                    "hook_type": mode,
                    "layer_idx": layer_idx,
                    "z_mean": z_mean[i],  # Tensor[d_hidden]
                }
                # 若是 topk SAE，额外保存前若干 token 的 topk idx/val
                if topk_idx is not None and topk_val is not None:
                    item["topk_idx_token0"] = topk_idx.view(b, l, -1)[i, 0].cpu()
                    item["topk_val_token0"] = topk_val.view(b, l, -1)[i, 0].cpu()
                results.append(item)

        # 打印进度
        if batch_count % log_params["log_interval"] == 0 or batch_count == total_batches:
            logger.info("处理进度: [%d/%d] batches, 已收集 %d 条结果",
                       batch_count, total_batches, len(results))

    # 保存结果
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"results": results}, out)
    logger.info("测试结果已保存到: %s", out)
    logger.info("共 %d 条记录", len(results))

    # ------------------------- 批量可视化伪代码（后续迭代） -------------------------
    logger.info("\n可视化建议:")
    logger.info("1) 将 results 按 (hook_type, layer_idx) 分组")
    logger.info("2) 取每条的 z_mean: [d_hidden]")
    logger.info("3) 进行 PCA / UMAP 降维和聚类分析")
    logger.info("4) 找出 top 激活特征并进行可视化")


if __name__ == "__main__":
    main()
