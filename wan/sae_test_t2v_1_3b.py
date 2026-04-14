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
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.cuda.amp as amp

from wan.configs.wan_t2v_1_3B import t2v_1_3B
from wan.modules.sae_new import SAEConfig, SparseAutoEncoder
from wan.sae.checkpoint_io import SAECheckpointIO, load_checkpoint
from wan.sae.hooking import HookMode, register_dit_hooks, remove_hooks
from wan.sae.logger import SAELogManager, get_test_logger
from wan.sae.path_utils import resolve_path, resolve_dir
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
    # model_path: Wan 2.1 DiT 模型权重目录路径
    # 学术意义: DiT 预训练权重，用于生成激活 SAE 的隐藏状态
    # 实际用法: 指向包含 Wan2.1-T2V-1.3B 权重的目录（注意：不是 SAE checkpoint 目录）
    # 建议值: "./Wan2.1-T2V-1.3B"
    "model_path": "/root/Wan/Wan2.1-T2V-1.3B",

    # prompt_dir: 测试用提示词文件夹路径
    # 学术意义: 用于分析 SAE 在特定概念/风格上的激活模式的输入集合
    # 实际用法: 与训练时使用相同格式，可不同内容用于验证泛化性
    # 建议值: 准备与训练集分布不同但相关的提示词，评估泛化能力
    "prompt_dir": "final_cleaned",

    # run_dir: SAE 训练输出目录（训练脚本的 run_dir）
    # 学术意义: 从中加载训练好的 SAE 权重和配置
    # 实际用法: 与 sae_train_t2v_1_3b.py 中指定的 run_dir 相同
    # 建议值: 与训练时一致，如 "sae_runs/exp_20250319_blockout"
    "run_dir": "sae_runs/exp__20250324",

    # output_path: 测试结果保存路径
    # 学术意义: 保存 SAE 编码结果的文件，用于后续分析和可视化
    # 实际用法: 保存为 .pt 文件，包含每个 prompt 的 z_mean 和元信息
    # 建议值: 放在 run_dir 下，如 "{run_dir}/test_results.pt"
    "output_path": "sae_test_out.json",
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
    "hook_layers": "15",
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
    "max_prompts": 20,
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

# --------------------------- Checkpoint 加载配置 ---------------------------
checkpoint_params = {
    # sae_checkpoint: SAE checkpoint 加载配置
    # 学术意义: 支持灵活的 checkpoint 来源，包括跨实验加载和多层独立源
    # 支持格式:
    #   1. 空字符串 "": 从 run_dir 加载所有层（默认行为）
    #   2. 单一路径: "sae_runs/exp1"（加载该目录下所有层）
    #   3. 具体文件: "sae_runs/exp1/block_out.layer15/sae_latest.pt"
    #   4. 多层指定（命令行/JSON）: {"block_out.layer15": "sae_runs/exp_A", "block_out.layer29": "sae_runs/exp_B"}
    # 建议值: 留空（从 run_dir 自动加载）或根据需求指定
    "sae_checkpoint": "sae_runs/exp__20250324",

    # layer_sources: 多层源配置（用于从不同实验加载不同层）
    # 学术意义: 支持对比分析不同训练配置下的同一层特征
    # 实际用法: 字典格式，key 为 "hook_mode.layer_idx"，value 为 run_dir 路径
    # 示例: {"block_out.layer15": "sae_runs/exp_base", "block_out.layer29": "sae_runs/exp_finetune"}
    # 建议值: {}（统一源）或按需指定
    "layer_sources": {},

    # allow_partial_load: 是否允许加载不匹配的层配置
    # 学术意义: 当测试层与训练层不完全一致时的容错处理
    # 实际用法: 设置为 True 时，允许部分层加载失败而不中断整个测试
    # 建议值: False（严格匹配）或 True（容错模式）
    "allow_partial_load": False,

    # strict_loading: 是否严格匹配权重形状
    # 实际用法: 当 SAE 架构有微小差异时使用非严格加载
    # 建议值: True（默认）或 False（兼容模式）
    "strict_loading": True,
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


def load_sae_for_key(
    run_dir: str,
    hook_mode: str,
    layer_idx: int,
    device,
    allow_partial: bool = False,
    strict: bool = True,
) -> SparseAutoEncoder:
    """
    加载指定 hook_mode 和 layer_idx 的 SAE。

    参数:
        run_dir: SAE 训练输出目录
        hook_mode: hook 模式（如 "block_out"）
        layer_idx: 层索引
        device: 目标设备
        allow_partial: 是否允许部分加载（文件不存在时返回 None 而不是报错）
        strict: 是否严格匹配权重形状

    返回:
        加载并设为 eval 模式的 SparseAutoEncoder，如果 allow_partial=True 且文件不存在则返回 None
    """
    loc = SAERunLocator(run_dir=run_dir, hook_mode=hook_mode, layer_idx=layer_idx)

    # 检查 checkpoint 是否存在
    if not loc.latest_ckpt_path().exists():
        if allow_partial:
            logger.warning("缺少 %s 的 checkpoint: %s", loc.key(), loc.latest_ckpt_path())
            return None
        raise FileNotFoundError(f"缺少 {loc.key()} 的 checkpoint: {loc.latest_ckpt_path()}")

    try:
        # 使用新的统一 IO 接口加载（自动兼容新旧格式）
        io = SAECheckpointIO.load(
            loc,
            device=device,
            strict=strict,
            allow_legacy=True,  # 允许从旧格式回退
        )

        # 记录配置来源
        if io._config_source == "json_fallback":
            logger.debug("%s 从旧格式 .json 加载配置 [建议迁移]", loc.key())

        return io.sae

    except Exception as e:
        if allow_partial:
            logger.warning("加载 %s 失败: %s", loc.key(), str(e))
            return None
        raise RuntimeError(f"加载 SAE {loc.key()} 失败: {e}") from e


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
        ("Checkpoint 加载", checkpoint_params),
    ]:
        logger.info(f"\n【{name}】")
        for k, v in params.items():
            logger.info(f"  {k}: {v}")
    logger.info("=" * 60)


def main():
    # 参数解析（允许命令行覆盖默认配置）
    parser = argparse.ArgumentParser(description="Test SAE with Wan 1.3B T2V DiT hooks (single-step forward).")
    parser.add_argument("--model_path", type=str, default=path_params["model_path"],
                        help="Wan 2.1 DiT 模型权重目录路径（旧名称 --checkpoint_dir 仍兼容但已弃用）")
    parser.add_argument("--checkpoint_dir", type=str, default="",
                        help="[已弃用] 请使用 --model_path")
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

    # Checkpoint 加载参数（基于 checkpoint_params）
    parser.add_argument("--sae_checkpoint", type=str, default=checkpoint_params["sae_checkpoint"],
                        help="SAE checkpoint 路径或目录（覆盖 run_dir）")
    parser.add_argument("--allow_partial_load", action="store_true",
                        default=checkpoint_params["allow_partial_load"],
                        help="允许部分层加载失败")
    parser.add_argument("--strict_loading", action="store_true",
                        default=checkpoint_params["strict_loading"],
                        help="严格匹配权重形状")

    # 多层源配置（允许从不同 run_dir 加载不同层）
    parser.add_argument("--layer_sources", type=str, default="",
                        help="层源配置，格式: 'layer15:run1,layer29:run2' 或 JSON 文件路径")

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # 处理已弃用的 --checkpoint_dir 参数
    model_path = args.model_path
    if args.checkpoint_dir:
        logger.warning("--checkpoint_dir 已弃用，请使用 --model_path。当前仍兼容处理。")
        if not model_path:
            model_path = args.checkpoint_dir

    # 打印配置
    _print_config()

    # 验证必要参数
    if not model_path:
        raise ValueError("model_path 不能为空，请通过 --model_path 指定 Wan 2.1 模型路径")
    if not args.prompt_dir:
        raise ValueError("prompt_dir 不能为空")
    if not args.run_dir:
        raise ValueError("run_dir 不能为空（SAE 训练输出目录）")

    # 解析所有路径（支持 ~ 展开、环境变量、相对路径）
    model_path = str(resolve_path(model_path))
    prompt_dir = str(resolve_path(args.prompt_dir))
    run_dir = str(resolve_path(args.run_dir))
    output_path = str(resolve_path(args.output_path))

    logger.info("解析后的路径:")
    logger.info("  model_path: %s", model_path)
    logger.info("  prompt_dir: %s", prompt_dir)
    logger.info("  run_dir: %s", run_dir)
    logger.info("  output_path: %s", output_path)

    # 设置随机种子和设备
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    # 加载并清洗提示词
    clean_cfg = PromptCleanConfig(min_len=args.min_len, max_len=args.max_len)
    prompts = load_prompts_from_dir(prompt_dir, clean_cfg=clean_cfg, limit=args.max_prompts)
    if not prompts:
        raise RuntimeError("没有加载到任何有效 prompt。")
    logger.info("加载了 %d 条提示词", len(prompts))

    # 构建 WanT2V
    cfg = t2v_1_3B
    wrapper = WanT2V(
        config=cfg,
        checkpoint_dir=model_path,
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

    # 解析多层源配置（命令行覆盖配置文件）
    layer_sources: Dict[str, str] = checkpoint_params["layer_sources"].copy()
    if args.layer_sources:
        if args.layer_sources.endswith('.json'):
            # 从 JSON 文件加载
            import json
            with open(args.layer_sources, 'r') as f:
                layer_sources = json.load(f)
        else:
            # 解析命令行格式: "layer15:run1,layer29:run2"
            for part in args.layer_sources.split(','):
                if ':' in part:
                    layer_part, run_part = part.split(':', 1)
                    layer_key = f"{hook_mode}.layer{layer_part.strip()}"
                    layer_sources[layer_key] = run_part.strip()

    # 处理 sae_checkpoint 参数（覆盖默认 run_dir）
    sae_checkpoint = args.sae_checkpoint if args.sae_checkpoint else checkpoint_params["sae_checkpoint"]
    allow_partial_load = args.allow_partial_load or checkpoint_params["allow_partial_load"]

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

        # 确定源目录（优先级: layer_sources > sae_checkpoint > run_dir）
        if key in layer_sources:
            source_run_dir = layer_sources[key]
            logger.info("  %s 从自定义源加载: %s", key, source_run_dir)
        elif sae_checkpoint:
            # sae_checkpoint 可以是目录或具体文件
            if os.path.isdir(sae_checkpoint):
                source_run_dir = sae_checkpoint
            else:
                # 如果是具体文件路径，取其所在目录
                source_run_dir = os.path.dirname(os.path.dirname(sae_checkpoint))
            logger.info("  %s 从 sae_checkpoint 加载: %s", key, source_run_dir)
        else:
            source_run_dir = run_dir

        try:
            sae = load_sae_for_key(
                source_run_dir,
                hook_mode=mode,
                layer_idx=layer_idx,
                device=device,
                allow_partial=allow_partial_load,
                strict=args.strict_loading,
            )
            if sae is not None:
                saes[key] = sae
                logger.info("  已加载: %s", key)
            elif not allow_partial_load:
                raise RuntimeError(f"无法加载 SAE: {key}")
        except Exception as e:
            if allow_partial_load:
                logger.warning("  加载 %s 失败（已跳过）: %s", key, str(e))
            else:
                raise RuntimeError(f"加载 SAE {key} 失败: {e}") from e

    if not saes:
        raise RuntimeError("没有成功加载任何 SAE，请检查 checkpoint 路径和配置")

    # 初始化统一日志管理器（每个层一个logger）
    log_managers: Dict[str, SAELogManager] = {}
    for key in saes.keys():
        mode, layer_str = key.split(".")
        layer_idx = int(layer_str.replace("layer", ""))
        log_managers[key] = get_test_logger(run_dir, hook_mode=mode, layer_idx=layer_idx)
        log_managers[key].log_event("test_start", f"开始测试 {key}", {"num_prompts": len(prompts)})

    # 测试结果收集
    results: List[Dict[str, Any]] = []
    batch_count = 0
    total_batches = (len(prompts) + args.batch_prompts - 1) // args.batch_prompts

    # 批次处理
    for batch in batch_iter(prompts, batch_size=args.batch_prompts, shuffle=False):
        B = len(batch)
        batch_count += 1

        logger.info("处理批次 [%d/%d], 大小=%d", batch_count, total_batches, B)

        try:
            # 文本编码
            logger.debug("  文本编码...")
            wrapper.text_encoder.model.to(device)
            context = wrapper.text_encoder(batch, device)
            logger.debug("  文本编码完成, context.shape=%s", context.shape if hasattr(context, 'shape') else 'N/A')
        except Exception as e:
            logger.error("批次 [%d/%d] 文本编码失败: %s", batch_count, total_batches, e)
            raise RuntimeError(f"批次 {batch_count} 文本编码失败: {e}") from e

        # 随机时间步和噪声 latent（单步前向）
        t = torch.randint(low=0, high=cfg.num_train_timesteps, size=(B,), device=device, dtype=torch.long)
        x_list = [torch.randn(*latent_shape, device=device, dtype=torch.float32) for _ in range(B)]

        # 随机时间步和噪声 latent（单步前向）
        try:
            logger.debug("  准备噪声 latent...")
            t = torch.randint(low=0, high=cfg.num_train_timesteps, size=(B,), device=device, dtype=torch.long)
            x_list = [torch.randn(*latent_shape, device=device, dtype=torch.float32) for _ in range(B)]
            logger.debug("  噪声 latent 准备完成, len=%d, shape=%s", len(x_list), latent_shape)
        except Exception as e:
            logger.error("批次 [%d/%d] 准备噪声 latent 失败: %s", batch_count, total_batches, e)
            raise RuntimeError(f"批次 {batch_count} 准备噪声 latent 失败: {e}") from e

        # Hook 收集激活
        raw: Dict[str, torch.Tensor] = {}

        def on_tensor(k: str, v: torch.Tensor):
            raw[k] = v  # [B, L, C]

        logger.debug("  注册 hooks...")
        handles = register_dit_hooks(model, hook_layers=hook_layers, hook_mode=hook_mode, on_tensor=on_tensor)
        logger.debug("  Hooks 注册完成: %d 个 handles", len(handles))

        try:
            logger.debug("  开始 DiT forward...")
            with torch.no_grad(), amp.autocast(dtype=cfg.param_dtype):
                _ = model(x_list, t=t, context=context, seq_len=seq_len)
            logger.debug("  DiT forward 完成, 收集到 %d 个激活", len(raw))
        except Exception as e:
            logger.error("批次 [%d/%d] DiT forward 失败: %s", batch_count, total_batches, e)
            raise RuntimeError(f"批次 {batch_count} DiT forward 失败: {e}") from e
        finally:
            remove_hooks(handles)
            logger.debug("  Hooks 已移除")

        # 对每个 key 计算 SAE 编码和 loss
        logger.debug("  处理 %d 个 key 的 SAE 编码...", len(raw))
        for key, act in raw.items():
            sae = saes.get(key)
            if sae is None:
                logger.warning("  Key %s 没有对应的 SAE，跳过", key)
                continue

            try:
                # act: [B, L, C] -> [B*L, C]
                b, l, c = act.shape
                logger.debug("    %s: act.shape=%s", key, act.shape)
                x_flat = act.reshape(b * l, c).to(device)

                with torch.no_grad():
                    # SAE编码
                    z, topk_idx, topk_val = sae.encode(x_flat)  # z: [B*L, d_hidden]
                    logger.debug("    %s: 编码完成, z.shape=%s", key, z.shape)

                    # 解码计算重构误差（loss）
                    x_recon = sae.decode(z)
                    loss = ((x_recon - x_flat) ** 2).mean().item()  # MSE loss
                    recon_mse_per_token = ((x_recon - x_flat) ** 2).mean(dim=-1).cpu()  # [B*L]

                    # 计算稀疏度
                    sparsity = (z.abs() > 1e-6).float().mean().item()
                    num_activations = (z.abs() > 1e-6).sum(dim=-1).float().mean().item()

                z = z.view(b, l, -1)  # [B, L, d_hidden]
                z_mean = z.mean(dim=1).cpu()  # [B, d_hidden]
                z_std = z.std(dim=1).cpu()  # [B, d_hidden]

                # 计算每个prompt的平均loss（按token平均）
                recon_mse_per_prompt = recon_mse_per_token.view(b, l).mean(dim=1)  # [B]

                mode, layer_str = key.split(".")
                layer_idx = int(layer_str.replace("layer", ""))

                for i in range(B):
                    result_id = f"batch{batch_count}_idx{i}_{key}"
                    item: Dict[str, Any] = {
                    "prompt": batch[i],
                    "hook_type": mode,
                    "layer_idx": layer_idx,
                    "batch_idx": batch_count - 1,
                    "prompt_idx": (batch_count - 1) * args.batch_prompts + i,
                    "z_mean": z_mean[i],  # Tensor[d_hidden]
                    "z_std": z_std[i],  # Tensor[d_hidden]
                    "loss": loss,  # 全局平均loss
                    "recon_mse": recon_mse_per_prompt[i].item(),  # 该prompt的平均重构误差
                    "sparsity": sparsity,  # 稀疏度
                    "num_activations": num_activations,  # 平均激活数
                }

                # 若是 topk SAE，额外保存 topk 信息
                if topk_idx is not None and topk_val is not None:
                    topk_idx_view = topk_idx.view(b, l, -1)
                    topk_val_view = topk_val.view(b, l, -1)
                    item["topk_idx_token0"] = topk_idx_view[i, 0].cpu()
                    item["topk_val_token0"] = topk_val_view[i, 0].cpu()
                    item["topk_idx_all_tokens"] = topk_idx_view[i].cpu()  # [L, top_k]
                    item["topk_val_all_tokens"] = topk_val_view[i].cpu()  # [L, top_k]

                results.append(item)

                # 使用统一日志管理器记录详细结果
                log_record = {
                    "prompt": batch[i],
                    "loss": item["loss"],
                    "recon_mse": item["recon_mse"],
                    "sparsity": item["sparsity"],
                    "num_activations": item["num_activations"],
                    "z_mean": z_mean[i].tolist(),
                    "z_std": z_std[i].tolist(),
                }
                if "topk_idx_token0" in item:
                    log_record["topk_idx_token0"] = item["topk_idx_token0"].tolist()
                    log_record["topk_val_token0"] = item["topk_val_token0"].tolist()

                log_managers[key].log_result(log_record, result_id=result_id)

            except Exception as e:
                logger.error("批次 [%d/%d] Key %s 的 SAE 处理失败: %s", batch_count, total_batches, key, e)
                logger.error("错误详情: act.shape=%s, sae=%s", act.shape if 'act' in locals() else 'N/A', key)
                raise RuntimeError(f"批次 {batch_count} Key {key} 的 SAE 处理失败: {e}") from e

        # 打印进度
        if batch_count % log_params["log_interval"] == 0 or batch_count == total_batches:
            logger.info("处理进度: [%d/%d] batches, 已收集 %d 条结果",
                       batch_count, total_batches, len(results))

    # 保存汇总结果（torch格式，便于后续加载）
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {"results": results}
    if avg_z_mean_by_key:
        save_dict["avg_z_mean_by_key"] = avg_z_mean_by_key
    torch.save(save_dict, out)
    logger.info("测试结果已保存到: %s", out)
    logger.info("共 %d 条记录", len(results))

    # 保存JSON格式结果（便于查看和分析）
    json_out = out.with_suffix(".json")
    json_results = []
    for r in results:
        json_item = {
            "prompt": r["prompt"],
            "hook_type": r["hook_type"],
            "layer_idx": r["layer_idx"],
            "loss": r["loss"],
            "recon_mse": r["recon_mse"],
            "sparsity": r["sparsity"],
            "num_activations": r["num_activations"],
            "z_mean": r["z_mean"].tolist() if hasattr(r["z_mean"], "tolist") else r["z_mean"],
        }
        if "topk_idx_token0" in r:
            json_item["topk_idx_token0"] = r["topk_idx_token0"].tolist() if hasattr(r["topk_idx_token0"], "tolist") else r["topk_idx_token0"]
            json_item["topk_val_token0"] = r["topk_val_token0"].tolist() if hasattr(r["topk_val_token0"], "tolist") else r["topk_val_token0"]
        json_results.append(json_item)

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    logger.info("JSON格式结果已保存到: %s", json_out)

    # 记录测试完成事件
    for key, log_mgr in log_managers.items():
        log_mgr.log_event("test_complete", f"测试完成 {key}", {"total_results": len(results)})
        # 保存summary
        key_results = [r for r in results if f"{r['hook_type']}.layer{r['layer_idx']}" == key]
        if key_results:
            avg_loss = sum(r["loss"] for r in key_results) / len(key_results)
            avg_sparsity = sum(r["sparsity"] for r in key_results) / len(key_results)
            log_mgr.save_summary({
                "total_prompts": len(key_results),
                "avg_loss": avg_loss,
                "avg_recon_mse": sum(r["recon_mse"] for r in key_results) / len(key_results),
                "avg_sparsity": avg_sparsity,
                "avg_num_activations": sum(r["num_activations"] for r in key_results) / len(key_results),
            })

    # ------------------------- 批量可视化伪代码（后续迭代） -------------------------
    logger.info("\n可视化建议:")
    logger.info("1) 将 results 按 (hook_type, layer_idx) 分组")
    logger.info("2) 取每条的 z_mean: [d_hidden]")
    logger.info("3) 进行 PCA / UMAP 降维和聚类分析")
    logger.info("4) 找出 top 激活特征并进行可视化")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error("=" * 60)
        logger.error("运行失败: %s", e)
        logger.error("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
