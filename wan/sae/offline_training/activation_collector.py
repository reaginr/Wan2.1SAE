"""
激活值采集器 - 在线采集并保存提示词的激活值用于离线训练

输出格式（JSON Lines，每行一个样本）：
{
    "prompt": "提示词文本",
    "metadata": {
        "prompt_idx": 0,
        "batch_idx": 0,
        "timestamp": "2025-04-01T12:00:00"
    },
    "activations": {
        "block_out.layer15": {
            "timesteps": [0, 5, 10, 15, ...],  # 时间步列表
            "features": [  # 每个时间步的激活值
                {"t": 0, "shape": [L, C], "data": [[...], [...]]},  # 扁平化存储
                ...
            ]
        },
        "block_out.layer29": { ... }
    }
}

存储优化：
- 使用JSON Lines格式，方便流式读取
- 大数组使用numpy保存为单独的.npy文件，JSON中存储引用
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.cuda.amp as amp

# 修复导入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wan.configs.wan_t2v_1_3B import t2v_1_3B
from wan.sae.hooking import HookMode, pack_hook_batch, register_dit_hooks, remove_hooks
from wan.sae.prompt_io import PromptCleanConfig, batch_iter, load_prompts_from_dir
from wan.text2video import WanT2V

logger = logging.getLogger(__name__)


##########################################################################################
# 采集参数配置区域
##########################################################################################

# --------------------------- 路径配置 ---------------------------
path_params = {
    "checkpoint_dir": "~/Wan/Wan2.1-T2V-1.3B",
    "prompt_dir": "final_cleaned",
    "output_dir": "offline_data/activations_run1",  # 激活值保存目录
}

# --------------------------- Hook 配置 ---------------------------
hook_params = {
    "hook_mode": "block_out",
    "hook_layers": "15,29",  # 要采集的层
    "max_tokens_per_key": 65536,
}

# --------------------------- 采样配置 ---------------------------
sampling_params = {
    "sampling_steps": 30,  # 每个prompt的扩散步数
    "shift": 5.0,
    "use_cfg": False,
    "guide_scale": 5.0,
}

# --------------------------- 批处理配置 ---------------------------
batch_params = {
    "batch_prompts": 4,
    "max_prompts": 1000,  # 最多采集多少条prompt
}

# --------------------------- 生成尺寸配置 ---------------------------
generation_params = {
    "size_w": 832,
    "size_h": 480,
    "frame_num": 81,
}

# --------------------------- 存储配置 ---------------------------
storage_params = {
    # 激活值存储格式: "json" | "numpy" | "hybrid"
    # - json: 全部存为JSON（体积大，兼容性最好）
    # - numpy: 大数组存为.npy文件（推荐，体积小）
    # - hybrid: 小数组JSON，大数组numpy（默认）
    "format": "numpy",

    # 是否压缩numpy文件
    "compress_numpy": True,

    # 每个文件最多存储多少条prompt（0表示不限制）
    "max_prompts_per_file": 100,

    # 是否保存时间步信息
    "save_timesteps": True,

    # 是否保存token位置信息（可能很大）
    "save_token_positions": False,
}

# --------------------------- 提示词清洗配置 ---------------------------
prompt_clean_params = {
    "min_len": 8,
    "max_len": 400,
}

# --------------------------- 系统配置 ---------------------------
system_params = {
    "device_id": 0,
    "seed": 0,
    "offload_text_encoder": True,
}

# --------------------------- 日志配置 ---------------------------
log_params = {
    "log_interval": 10,
}


##########################################################################################
# 核心代码区域
##########################################################################################

class ActivationCollector:
    """
    激活值采集器

    负责：
    1. 运行DiT前向传播
    2. 在指定时间步采集指定层的激活值
    3. 将激活值保存为结构化格式
    """

    def __init__(
        self,
        model,
        wrapper,
        cfg,
        device,
        hook_layers: List[int],
        hook_mode: HookMode,
        sampling_steps: int,
        save_timesteps: bool = True,
        max_tokens_per_key: int = 65536,
    ):
        self.model = model
        self.wrapper = wrapper
        self.cfg = cfg
        self.device = device
        self.hook_layers = hook_layers
        self.hook_mode = hook_mode
        self.sampling_steps = sampling_steps
        self.save_timesteps = save_timesteps
        self.max_tokens_per_key = max_tokens_per_key

        # 计算时间步序列
        self.timesteps = torch.linspace(
            cfg.num_train_timesteps - 1, 0, sampling_steps,
            device=device, dtype=torch.long
        )

    def collect_for_prompts(
        self,
        prompts: List[str],
        batch_idx: int,
        latent_shape: List[int],
        seq_len: int,
        use_cfg: bool = False,
        guide_scale: float = 5.0,
        negative_prompt: str = "",
    ) -> List[Dict[str, Any]]:
        """
        为一批prompt采集激活值

        返回: 每个prompt的激活值记录列表
        """
        B = len(prompts)
        results = []

        # 文本编码
        self.wrapper.text_encoder.model.to(self.device)
        context = self.wrapper.text_encoder(prompts, self.device)

        # CFG准备
        if use_cfg:
            if negative_prompt == "":
                n_prompt = self.wrapper.sample_neg_prompt
            else:
                n_prompt = negative_prompt
            context_null = self.wrapper.text_encoder([n_prompt] * B, self.device)
        else:
            context_null = None

        # 初始化噪声
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(system_params["seed"] + batch_idx)
        latents = [
            torch.randn(
                latent_shape[0], latent_shape[1], latent_shape[2], latent_shape[3],
                dtype=torch.float32, device=self.device, generator=seed_g,
            )
            for _ in range(B)
        ]

        # 为每个prompt初始化激活记录
        for i in range(B):
            results.append({
                "prompt": prompts[i],
                "metadata": {
                    "prompt_idx": batch_idx * B + i,
                    "batch_idx": batch_idx,
                    "timestamp": datetime.now().isoformat(),
                },
                "activations": {},
            })

        # 注册hooks
        handles = register_dit_hooks(
            self.model,
            hook_layers=self.hook_layers,
            hook_mode=self.hook_mode,
            on_tensor=self._make_callback(),
        )

        try:
            # 按时间步迭代
            for t_idx, t in enumerate(self.timesteps):
                self.current_raw = {}  # 清空当前采集
                timestep = torch.stack([t]).repeat(B)  # [B]

                # DiT前向（无梯度）
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=self.cfg.param_dtype):
                        noise_pred_cond = self.model(latents, t=timestep, context=context, seq_len=seq_len)

                        if use_cfg and context_null is not None:
                            noise_pred_uncond = self.model(latents, t=timestep, context=context_null, seq_len=seq_len)
                            pred = [
                                u + guide_scale * (c - u)
                                for c, u in zip(noise_pred_cond, noise_pred_uncond)
                            ]
                            del noise_pred_uncond
                        else:
                            pred = noise_pred_cond

                # 处理采集到的激活值
                if self.current_raw:
                    hook_batch = pack_hook_batch(
                        self.current_raw,
                        max_tokens_per_key=self.max_tokens_per_key
                    )

                    # 按key存储到对应prompt
                    for key, feats in hook_batch.items():
                        # feats: [N, C]，其中N = B * L（可能经过截断）
                        N, C = feats.shape
                        L = N // B  # 每个prompt的token数

                        for i in range(B):
                            if key not in results[i]["activations"]:
                                results[i]["activations"][key] = {
                                    "timesteps": [],
                                    "features": [],
                                }

                            # 提取第i个prompt的特征
                            feat_i = feats[i*L:(i+1)*L].cpu().numpy()

                            results[i]["activations"][key]["timesteps"].append(int(t.item()))
                            results[i]["activations"][key]["features"].append(feat_i)

                # 更新latent
                dt = 1.0 / self.sampling_steps
                new_latents = []
                for p, z in zip(pred, latents):
                    z_next = z - p * dt
                    new_latents.append(z_next)
                latents = new_latents

                del pred, noise_pred_cond

        finally:
            remove_hooks(handles)

        return results

    def _make_callback(self):
        """创建hook回调函数"""
        def on_tensor(k: str, v: torch.Tensor):
            self.current_raw[k] = v
        return on_tensor


def save_activations_batch(
    records: List[Dict[str, Any]],
    output_dir: str,
    file_idx: int,
    storage_format: str = "numpy",
    compress: bool = True,
) -> str:
    """
    保存一批激活值记录

    存储结构：
    output_dir/
    ├── manifest.jsonl          # 索引文件，每行一条记录元信息
    ├── data/
    │   ├── batch_000/
    │   │   ├── record_000.json      # prompt和元信息
    │   │   ├── record_000_layer15.npy  # 激活值
    │   │   └── record_000_layer29.npy
    │   └── batch_001/
    │       └── ...
    └── metadata.json           # 整体元信息

    返回: manifest文件路径
    """
    output_path = Path(output_dir)
    data_dir = output_path / "data" / f"batch_{file_idx:03d}"
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []

    for rec_idx, record in enumerate(records):
        record_id = f"{file_idx:03d}_{rec_idx:03d}"
        record_path = data_dir / f"record_{rec_idx:03d}.json"

        # 分离prompt元信息和激活值
        record_meta = {
            "id": record_id,
            "prompt": record["prompt"],
            "metadata": record["metadata"],
        }

        # 保存激活值
        activation_files = {}
        for key, act_data in record["activations"].items():
            # key格式: "block_out.layer15"
            layer_name = key.replace(".", "_")

            if storage_format in ["numpy", "hybrid"]:
                # 保存为numpy文件
                npy_path = data_dir / f"record_{rec_idx:03d}_{layer_name}.npy"

                # 组织数据: [num_timesteps, L, C]
                features = act_data["features"]  # list of [L, C] arrays
                stacked = np.stack(features, axis=0)  # [T, L, C]

                if compress:
                    np.savez_compressed(npy_path, data=stacked, timesteps=act_data["timesteps"])
                else:
                    np.save(npy_path, stacked)

                activation_files[key] = {
                    "file": str(npy_path.relative_to(output_path)),
                    "format": "npz_compressed" if compress else "npy",
                    "shape": list(stacked.shape),
                    "timesteps": act_data["timesteps"],
                }
            else:
                # JSON格式（不推荐用于大数据）
                activation_files[key] = {
                    "data": [f.tolist() for f in act_data["features"]],
                    "timesteps": act_data["timesteps"],
                    "format": "json",
                }

        record_meta["activations"] = activation_files

        # 保存记录
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(record_meta, f, ensure_ascii=False, indent=2)

        manifest_entries.append({
            "id": record_id,
            "prompt": record["prompt"][:100],  # 截断提示词
            "record_path": str(record_path.relative_to(output_path)),
        })

    # 追加到manifest
    manifest_path = output_path / "manifest.jsonl"
    with open(manifest_path, "a", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return str(manifest_path)


def compute_latent_shape(cfg, size_wh, frame_num: int, vae_z_dim: int) -> List[int]:
    """计算latent形状"""
    w, h = size_wh
    F = frame_num
    t_lat = (F - 1) // cfg.vae_stride[0] + 1
    h_lat = h // cfg.vae_stride[1]
    w_lat = w // cfg.vae_stride[2]
    return [vae_z_dim, t_lat, h_lat, w_lat]


def compute_seq_len(cfg, latent_shape, sp_size: int) -> int:
    """计算序列长度"""
    _, t_lat, h_lat, w_lat = latent_shape
    seq_len = math.ceil((h_lat * w_lat) / (cfg.patch_size[1] * cfg.patch_size[2]) * t_lat / sp_size) * sp_size
    return int(seq_len)


def parse_layers(s: str) -> List[int]:
    """解析层索引字符串"""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(description="Collect activations from Wan 1.3B T2V for offline SAE training")
    parser.add_argument("--checkpoint_dir", type=str, default=path_params["checkpoint_dir"])
    parser.add_argument("--prompt_dir", type=str, default=path_params["prompt_dir"])
    parser.add_argument("--output_dir", type=str, default=path_params["output_dir"])
    parser.add_argument("--hook_mode", type=str, default=hook_params["hook_mode"])
    parser.add_argument("--hook_layers", type=str, default=hook_params["hook_layers"])
    parser.add_argument("--batch_prompts", type=int, default=batch_params["batch_prompts"])
    parser.add_argument("--max_prompts", type=int, default=batch_params["max_prompts"])
    parser.add_argument("--sampling_steps", type=int, default=sampling_params["sampling_steps"])
    parser.add_argument("--storage_format", type=str, default=storage_params["format"],
                       choices=["json", "numpy", "hybrid"])
    parser.add_argument("--max_prompts_per_file", type=int, default=storage_params["max_prompts_per_file"])
    parser.add_argument("--device_id", type=int, default=system_params["device_id"])
    parser.add_argument("--seed", type=int, default=system_params["seed"])

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存采集配置
    collection_config = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint_dir": args.checkpoint_dir,
        "prompt_dir": args.prompt_dir,
        "hook_mode": args.hook_mode,
        "hook_layers": parse_layers(args.hook_layers),
        "sampling_steps": args.sampling_steps,
        "storage_format": args.storage_format,
        "max_prompts_per_file": args.max_prompts_per_file,
        "seed": args.seed,
    }
    with open(output_path / "collection_config.json", "w", encoding="utf-8") as f:
        json.dump(collection_config, f, ensure_ascii=False, indent=2)

    # 设置设备
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    # 加载prompts
    clean_cfg = PromptCleanConfig(min_len=prompt_clean_params["min_len"], max_len=prompt_clean_params["max_len"])
    prompts = load_prompts_from_dir(args.prompt_dir, clean_cfg=clean_cfg, limit=args.max_prompts)
    if not prompts:
        raise RuntimeError("没有加载到任何有效prompt")
    logger.info("加载了 %d 条提示词", len(prompts))

    # 构建WanT2V
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
    logger.info("WanT2V 已加载")

    # 计算latent形状
    vae_z_dim = wrapper.vae.model.z_dim
    latent_shape = compute_latent_shape(cfg, (generation_params["size_w"], generation_params["size_h"]),
                                       generation_params["frame_num"], vae_z_dim)
    seq_len = compute_seq_len(cfg, latent_shape, wrapper.sp_size)
    logger.info("Latent shape=%s, seq_len=%d", latent_shape, seq_len)

    # 解析hook层
    hook_layers = parse_layers(args.hook_layers)
    hook_mode: HookMode = args.hook_mode  # type: ignore

    # 创建采集器
    collector = ActivationCollector(
        model=model,
        wrapper=wrapper,
        cfg=cfg,
        device=device,
        hook_layers=hook_layers,
        hook_mode=hook_mode,
        sampling_steps=args.sampling_steps,
        save_timesteps=storage_params["save_timesteps"],
        max_tokens_per_key=hook_params["max_tokens_per_key"],
    )

    # 开始采集
    logger.info("开始采集激活值...")
    batch_count = 0
    total_batches = (len(prompts) + args.batch_prompts - 1) // args.batch_prompts
    file_idx = 0
    all_records = []

    for batch in batch_iter(prompts, batch_size=args.batch_prompts, shuffle=False):
        batch_count += 1
        logger.info("处理批次 [%d/%d]", batch_count, total_batches)

        # 采集
        records = collector.collect_for_prompts(
            prompts=batch,
            batch_idx=batch_count - 1,
            latent_shape=latent_shape,
            seq_len=seq_len,
            use_cfg=sampling_params["use_cfg"],
            guide_scale=sampling_params["guide_scale"],
        )

        all_records.extend(records)

        # 定期保存
        if args.max_prompts_per_file > 0 and len(all_records) >= args.max_prompts_per_file:
            save_activations_batch(
                all_records[:args.max_prompts_per_file],
                args.output_dir,
                file_idx,
                storage_format=args.storage_format,
                compress=storage_params["compress_numpy"],
            )
            all_records = all_records[args.max_prompts_per_file:]
            file_idx += 1

        if batch_count % log_params["log_interval"] == 0:
            logger.info("已处理 %d/%d 批次，已保存 %d 条记录", batch_count, total_batches,
                       file_idx * args.max_prompts_per_file + len(all_records))

    # 保存剩余记录
    if all_records:
        save_activations_batch(
            all_records,
            args.output_dir,
            file_idx,
            storage_format=args.storage_format,
            compress=storage_params["compress_numpy"],
        )

    # 保存最终元数据
    metadata = {
        "total_prompts": len(prompts),
        "total_batches": batch_count,
        "total_files": file_idx + 1 if all_records else file_idx,
        "layers": hook_layers,
        "hook_mode": hook_mode,
        "sampling_steps": args.sampling_steps,
        "collection_time": datetime.now().isoformat(),
    }
    with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("采集完成！共 %d 条提示词，保存到 %s", len(prompts), args.output_dir)


if __name__ == "__main__":
    main()
