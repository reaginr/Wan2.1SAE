"""
简单采集脚本：从 WanT2V.generate 触发完整采样过程并采集 block 输入。

你现在的需求更适合使用：
    - `wan/sae_train_t2v_1_3b.py`（只跑 DiT 单步前向训练 SAE）
    - `wan/sae_test_t2v_1_3b.py`（批量测试并导出 z 特征）

本文件可作为对照/调试用途。
"""

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

import torch

from wan.configs.wan_t2v_1_3B import t2v_1_3B
from wan.text2video import WanT2V


logger = logging.getLogger(__name__)


##########################################################################################
# 采集参数配置区域 - 可直接修改此区域的默认值
# 学术意义与建议值详见每个参数的注释
##########################################################################################

# --------------------------- 路径配置 ---------------------------
path_params = {
    # checkpoint_dir: Wan 1.3B 模型权重目录路径
    # 学术意义: DiT 预训练权重，用于生成视频并采集隐藏状态
    # 建议值: "./Wan2.1-T2V-1.3B"
    "checkpoint_dir": "",

    # output_path: 采集的特征保存路径
    # 学术意义: 保存的隐藏状态可用于离线训练 SAE
    # 建议值: "./features_layer{layer_idx}.pt"
    "output_path": "./features_layer29.pt",
}

# --------------------------- 采集配置 ---------------------------
collect_params = {
    # layer_idx: 要采集的 block 层索引
    # 学术意义: 不同层编码不同抽象级别特征
    # 实际用法: 0-based 索引，-1 表示最后一层
    # 建议值: 29（最后一层）或 15（中层）
    "layer_idx": -1,

    # max_tokens: 最大采集 token 数量
    # 学术意义: 限制内存占用，采集足够的统计样本
    # 实际用法: 达到此数量后停止采集
    # 建议值: 1000000（约 100 万 tokens）
    "max_tokens": 1_000_000,

    # sampling_steps: 生成时的扩散步数
    # 学术意义: 步数越多生成质量越高，但采集时间线性增加
    # 实际用法: 生成视频时的去噪步数
    # 建议值: 20（采集可减小以节省时间）或 30（标准质量）
    "sampling_steps": 20,

    # size_w, size_h: 生成视频尺寸
    # 建议值: 1280x720（720P）或 832x480（480P）
    "size_w": 1280,
    "size_h": 720,

    # frame_num: 生成帧数
    # 建议值: 81（约 5 秒，16fps）
    "frame_num": 81,
}

# --------------------------- 提示词配置 ---------------------------
prompt_params = {
    # 默认提示词列表
    # 学术意义: 用于激活特定神经元或概念的输入集合
    # 实际用法: 可自行修改为从文件读取
    # 建议值: 根据研究目标准备多样化的提示词
    "demo_prompts": [
        "一只猫追逐蝴蝶的高清写实视频，阳光明媚的下午，电影感镜头",
        "夜晚城市街头下雨，霓虹灯反射在地面上，慢镜头",
        "海边日落，金色阳光洒在沙滩上，波浪轻轻拍打着岸边",
        "未来主义城市，飞行汽车穿梭在高楼之间，赛博朋克风格",
    ],
}

# --------------------------- 系统配置 ---------------------------
system_params = {
    # device_id: GPU 设备 ID
    # 建议值: 0
    "device_id": 0,

    # offload_model: 是否卸载模型到 CPU
    # 学术意义: 节省显存，但速度较慢
    # 建议值: True（采集过程显存占用大）
    "offload_model": True,
}


##########################################################################################
# 核心代码区域
##########################################################################################


def iter_prompts() -> Iterable[str]:
    """
    迭代提示词生成器。

    实际使用时，可改写为从文件读取：
        with open("prompts.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
    """
    for p in prompt_params["demo_prompts"]:
        yield p


def collect_hidden_states(
    checkpoint_dir: str,
    output_path: str,
    layer_idx: int = -1,
    max_tokens: int = 1_000_000,
    device_id: int = 0,
) -> None:
    """
    运行 WanT2V 推理，挂 hook 采集指定层的隐藏状态。

    参数:
        checkpoint_dir: Wan 模型权重目录
        output_path: 保存特征的 .pt 路径
        layer_idx: 要 hook 的 block 下标，-1 为最后一层
        max_tokens: 最多采集多少 token（B*L，总行数）
        device_id: 使用的 GPU ID
    """
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    # 构建 WanT2V（1.3B 配置）
    cfg = t2v_1_3B
    model_wrapper = WanT2V(
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )
    model = model_wrapper.model  # 这是 WanModel
    logger.info("WanT2V 已加载")

    # 选择要 hook 的层
    blocks = list(model.blocks)
    if layer_idx < 0:
        layer_idx = len(blocks) - 1
    assert 0 <= layer_idx < len(blocks), f"layer_idx 超出范围: [0, {len(blocks)-1}]"
    target_block = blocks[layer_idx]
    logger.info("目标层: layer_idx=%d (共 %d 层)", layer_idx, len(blocks))

    hidden_buffer: List[torch.Tensor] = []
    collected_tokens = 0

    def hook_fn(module, input, output):
        nonlocal collected_tokens
        x = input[0]  # [B, L, C]
        b, l, c = x.shape
        tokens = b * l
        if collected_tokens >= max_tokens:
            return

        # 如果超过上限，只取一部分
        if collected_tokens + tokens > max_tokens:
            remain = max_tokens - collected_tokens
            x = x.reshape(-1, c)[:remain]
        else:
            x = x.reshape(-1, c)

        hidden_buffer.append(x.detach().cpu())
        collected_tokens += x.shape[0]

    hook_handle = target_block.register_forward_hook(hook_fn)
    logger.info("Hook 已注册，开始采集...")

    try:
        # 通过 generate 触发前向，采集隐藏状态
        for i, prompt in enumerate(iter_prompts()):
            if collected_tokens >= max_tokens:
                logger.info("已达到最大 token 数量 %d，停止采集", max_tokens)
                break

            logger.info("[%d] 处理: %s...", i+1, prompt[:30])
            _ = model_wrapper.generate(
                input_prompt=prompt,
                size=(collect_params["size_w"], collect_params["size_h"]),
                frame_num=collect_params["frame_num"],
                shift=5.0,
                sample_solver="unipc",
                sampling_steps=collect_params["sampling_steps"],
                guide_scale=5.0,
                n_prompt="",
                seed=-1,
                offload_model=system_params["offload_model"],
            )
            logger.info("    已采集 %d tokens", collected_tokens)

    finally:
        hook_handle.remove()
        logger.info("Hook 已移除")

    if not hidden_buffer:
        raise RuntimeError("没有采集到任何隐藏状态，请检查 iter_prompts / hook 等逻辑。")

    features = torch.cat(hidden_buffer, dim=0)  # [N, C]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, output_path)
    logger.info("隐藏状态已保存: shape=%s -> %s", features.shape, output_path)


def _print_config():
    """打印配置。"""
    logger.info("=" * 60)
    logger.info("SAE 采集配置")
    logger.info("=" * 60)
    for name, params in [
        ("路径配置", path_params),
        ("采集配置", collect_params),
        ("提示词", prompt_params),
        ("系统配置", system_params),
    ]:
        logger.info(f"\n【{name}】")
        for k, v in params.items():
            if k == "demo_prompts":
                logger.info(f"  {k}: [{len(v)} 条提示词]")
            else:
                logger.info(f"  {k}: {v}")
    logger.info("=" * 60)


def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # 参数解析
    parser = argparse.ArgumentParser(description="Collect hidden states for SAE training (Wan 1.3B T2V).")
    parser.add_argument("--checkpoint_dir", type=str, default=path_params["checkpoint_dir"],
                        help="Wan checkpoint 目录")
    parser.add_argument("--output_path", type=str, default=path_params["output_path"],
                        help="保存特征的 .pt 路径")
    parser.add_argument("--layer_idx", type=int, default=collect_params["layer_idx"],
                        help="要 hook 的 block 下标，-1 为最后一层")
    parser.add_argument("--max_tokens", type=int, default=collect_params["max_tokens"],
                        help="最多采集多少 token")
    parser.add_argument("--device_id", type=int, default=system_params["device_id"],
                        help="使用的 GPU ID")

    args = parser.parse_args()

    # 验证必要参数
    if not args.checkpoint_dir:
        raise ValueError("checkpoint_dir 不能为空，请通过 --checkpoint_dir 或修改 path_params 指定")

    # 打印配置
    _print_config()

    # 执行采集
    collect_hidden_states(
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output_path,
        layer_idx=args.layer_idx,
        max_tokens=args.max_tokens,
        device_id=args.device_id,
    )


if __name__ == "__main__":
    main()
