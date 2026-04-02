"""
离线SAE测试脚本

从采集的激活值文件中测试SAE，无需运行DiT模型。

功能：
1. 加载已训练的SAE
2. 从激活值数据计算SAE编码
3. 分析特征激活模式
4. 对比不同prompt的激活差异

使用方法：
    python test_offline.py --data_dir offline_data/activations_run1 \
                           --run_dir sae_runs/offline_exp1 \
                           --output_path test_results.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# 修复导入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wan.modules.sae_new import SAEConfig, SparseAutoEncoder
from wan.sae.checkpoint_io import SAECheckpointIO
from wan.sae.logger import SAELogManager, get_offline_test_logger
from wan.sae.sae_run_naming import SAERunLocator, load_json

logger = logging.getLogger(__name__)


##########################################################################################
# 测试参数配置区域
##########################################################################################

# --------------------------- 路径配置 ---------------------------
path_params = {
    "data_dir": "offline_data/activations_run1",
    "run_dir": "sae_runs/offline_exp1",
    "output_path": "offline_test_results.pt",
}

# --------------------------- 层配置 ---------------------------
layer_params = {
    "hook_mode": "block_out",
    "hook_layers": "15,29",  # 要测试的层
}

# --------------------------- 测试配置 ---------------------------
test_params = {
    "max_prompts": 0,  # 0表示测试全部
    "compute_reconstruction": True,  # 计算重建误差
    "compute_statistics": True,  # 计算统计信息
    "compute_feature_freq": True,  # 计算特征频率
    "top_k_features": 20,  # 记录每个prompt top-k激活的特征
}

# --------------------------- 系统配置 ---------------------------
system_params = {
    "device_id": 0,
    "seed": 0,
}


##########################################################################################
# 核心代码区域
##########################################################################################

class OfflineActivationLoader:
    """
    离线激活值加载器
    """

    def __init__(self, data_dir: str, manifest_file: str = "manifest.jsonl"):
        self.data_dir = Path(data_dir)
        self.manifest_path = self.data_dir / manifest_file

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest文件不存在: {self.manifest_path}")

        # 加载所有记录
        self.records = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        logger.info(f"加载了 {len(self.records)} 条记录")

    def __len__(self) -> int:
        return len(self.records)

    def load_activations(self, idx: int, layer_key: str) -> Optional[np.ndarray]:
        """
        加载指定记录和层的激活值

        返回: [T, L, C] numpy array 或 None
        """
        record_meta = self.records[idx]
        record_path = self.data_dir / record_meta["record_path"]

        with open(record_path, "r", encoding="utf-8") as f:
            record = json.load(f)

        if layer_key not in record["activations"]:
            return None

        act_info = record["activations"][layer_key]

        if act_info.get("format") in ["npy", "npz_compressed"]:
            npy_path = self.data_dir / act_info["file"]
            if act_info["format"] == "npz_compressed":
                data = np.load(npy_path)
                features = data["data"]
            else:
                features = np.load(npy_path)
        else:
            features = np.array(act_info["data"])

        return features

    def get_prompt(self, idx: int) -> str:
        """获取提示词"""
        record_meta = self.records[idx]
        record_path = self.data_dir / record_meta["record_path"]

        with open(record_path, "r", encoding="utf-8") as f:
            record = json.load(f)

        return record["prompt"]


def load_sae(run_dir: str, hook_mode: str, layer_idx: int, device: torch.device) -> Optional[SparseAutoEncoder]:
    """加载SAE模型（使用新的统一 IO 接口，自动兼容新旧格式）"""
    loc = SAERunLocator(run_dir=run_dir, hook_mode=hook_mode, layer_idx=layer_idx)

    if not loc.latest_ckpt_path().exists():
        logger.warning(f"找不到checkpoint: {loc.latest_ckpt_path()}")
        return None

    try:
        io = SAECheckpointIO.load(loc, device=device, strict=True, allow_legacy=True)
        logger.info(f"已加载 {hook_mode}.layer{layer_idx} 从 {loc.latest_ckpt_path()}")
        if io._config_source == "json_fallback":
            logger.warning("  从旧格式 .json 加载配置 [建议迁移]")
        return io.sae
    except Exception as e:
        logger.error(f"加载SAE失败: {e}")
        raise


def analyze_prompt_activations(
    sae: SparseAutoEncoder,
    activations: np.ndarray,  # [T, L, C]
    device: torch.device,
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    分析单个prompt的激活模式

    返回: {
        "z_mean": [d_hidden],  # 平均激活
        "z_max": [d_hidden],   # 最大激活
        "top_k_indices": [K],  # top-k特征索引
        "top_k_values": [K],   # top-k特征值
        "sparsity": float,     # 稀疏度
        "recon_mse": float,    # 重建误差
    }
    """
    T, L, C = activations.shape

    # 展平为 [N, C]
    x = torch.from_numpy(activations.reshape(-1, C)).float().to(device)

    with torch.no_grad():
        # 编码
        z, topk_idx, topk_val = sae.encode(x)  # z: [N, d_hidden]

        # 解码重建
        x_recon = sae.decode(z)

        # 计算重建误差
        recon_mse = ((x_recon - x) ** 2).mean().item()

    # 计算统计信息
    z_np = z.cpu().numpy()

    # 按时间/空间维度聚合
    z_mean = z_np.mean(axis=0)  # [d_hidden]
    z_max = z_np.max(axis=0)    # [d_hidden]

    # 全局top-k特征
    global_max = z_np.max(axis=0)  # [d_hidden]
    top_k_indices = np.argsort(global_max)[-top_k:][::-1]
    top_k_values = global_max[top_k_indices]

    # 稀疏度
    sparsity = (np.abs(z_np) > 1e-6).mean()

    return {
        "z_mean": z_mean,
        "z_max": z_max,
        "top_k_indices": top_k_indices,
        "top_k_values": top_k_values,
        "sparsity": sparsity,
        "recon_mse": recon_mse,
    }


def test_sae_layer(
    sae: SparseAutoEncoder,
    loader: OfflineActivationLoader,
    layer_key: str,
    device: torch.device,
    max_prompts: int = 0,
    top_k: int = 20,
    run_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    测试SAE在一层上的表现

    返回: {
        "results": [...],  # 每个prompt的分析结果
        "statistics": {...},  # 整体统计
    }
    """
    # 解析层信息
    hook_mode, layer_str = layer_key.split(".")
    layer_idx = int(layer_str.replace("layer", ""))

    # 初始化统一日志管理器
    log_mgr = None
    if run_dir:
        log_mgr = get_offline_test_logger(run_dir, hook_mode, layer_idx)
        log_mgr.log_event("test_start", f"开始测试: {layer_key}", {"max_prompts": max_prompts})

    sae.eval()

    results = []
    num_prompts = len(loader) if max_prompts == 0 else min(max_prompts, len(loader))

    logger.info(f"测试 {num_prompts} 条提示词...")

    for idx in range(num_prompts):
        prompt = loader.get_prompt(idx)
        activations = loader.load_activations(idx, layer_key)

        if activations is None:
            logger.warning(f"跳过 {idx}: 无激活值")
            continue

        # 分析
        analysis = analyze_prompt_activations(sae, activations, device, top_k=top_k)

        result_item = {
            "prompt": prompt,
            "prompt_idx": idx,
            **analysis,
        }
        results.append(result_item)

        # 使用统一日志管理器记录详细结果
        if log_mgr:
            log_record = {
                "prompt": prompt,
                "prompt_idx": idx,
                "loss": analysis.get("recon_mse", 0),  # 使用recon_mse作为loss
                "recon_mse": analysis.get("recon_mse", 0),
                "sparsity": analysis.get("sparsity", 0),
            }
            if "top_k_indices" in analysis:
                log_record["top_k_indices"] = analysis["top_k_indices"].tolist() if hasattr(analysis["top_k_indices"], "tolist") else analysis["top_k_indices"]
            log_mgr.log_result(log_record, result_id=f"{layer_key}_idx{idx}")

        if (idx + 1) % 100 == 0:
            logger.info(f"  已处理 {idx + 1}/{num_prompts}")

    # 计算整体统计
    all_sparsity = [r["sparsity"] for r in results]
    all_recon = [r["recon_mse"] for r in results]

    statistics = {
        "num_prompts": len(results),
        "avg_sparsity": np.mean(all_sparsity) if all_sparsity else 0,
        "std_sparsity": np.std(all_sparsity) if all_sparsity else 0,
        "avg_recon_mse": np.mean(all_recon) if all_recon else 0,
        "std_recon_mse": np.std(all_recon) if all_recon else 0,
    }

    # 特征频率分析
    feature_counts = np.zeros(sae.d_hidden)
    for r in results:
        feature_counts[r["top_k_indices"]] += 1

    feature_freq = feature_counts / len(results) if results else feature_counts

    statistics["feature_frequency"] = feature_freq
    statistics["most_common_features"] = np.argsort(feature_freq)[-50:][::-1].tolist()

    result = {
        "layer_key": layer_key,
        "results": results,
        "statistics": statistics,
    }

    # 保存测试总结
    if log_mgr:
        log_mgr.save_summary(statistics)
        log_mgr.log_event("test_complete", f"测试完成: {layer_key}", statistics)

    return result


def main():
    parser = argparse.ArgumentParser(description="Test SAE offline from collected activations")
    parser.add_argument("--data_dir", type=str, default=path_params["data_dir"])
    parser.add_argument("--run_dir", type=str, default=path_params["run_dir"])
    parser.add_argument("--output_path", type=str, default=path_params["output_path"])
    parser.add_argument("--hook_mode", type=str, default=layer_params["hook_mode"])
    parser.add_argument("--hook_layers", type=str, default=layer_params["hook_layers"])
    parser.add_argument("--max_prompts", type=int, default=test_params["max_prompts"])
    parser.add_argument("--top_k_features", type=int, default=test_params["top_k_features"])
    parser.add_argument("--device_id", type=int, default=system_params["device_id"])
    parser.add_argument("--seed", type=int, default=system_params["seed"])

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # 设置设备
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 创建数据加载器
    loader = OfflineActivationLoader(args.data_dir)

    # 解析层
    layer_indices = [int(x.strip()) for x in args.hook_layers.split(",") if x.strip()]

    # 测试每一层
    all_results = {}

    for layer_idx in layer_indices:
        layer_key = f"{args.hook_mode}.layer{layer_idx}"
        logger.info(f"=" * 60)
        logger.info(f"测试层: {layer_key}")
        logger.info(f"=" * 60)

        # 加载SAE
        sae = load_sae(args.run_dir, args.hook_mode, layer_idx, device)
        if sae is None:
            logger.error(f"无法加载层 {layer_key} 的SAE，跳过")
            continue

        # 测试
        result = test_sae_layer(
            sae=sae,
            loader=loader,
            layer_key=layer_key,
            device=device,
            max_prompts=args.max_prompts,
            top_k=args.top_k_features,
            run_dir=args.run_dir,
        )

        all_results[layer_key] = result

        # 打印统计
        stats = result["statistics"]
        logger.info(f"统计信息:")
        logger.info(f"  测试样本数: {stats['num_prompts']}")
        logger.info(f"  平均稀疏度: {stats['avg_sparsity']:.4f} ± {stats['std_sparsity']:.4f}")
        logger.info(f"  平均重建误差: {stats['avg_recon_mse']:.6f} ± {stats['std_recon_mse']:.6f}")
        logger.info(f"  最常激活特征: {stats['most_common_features'][:10]}")

    # 保存结果
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(all_results, output_path)
    logger.info(f"\n结果已保存到: {output_path}")

    # 同时保存JSON格式的统计信息
    json_stats = {}
    for layer_key, result in all_results.items():
        json_stats[layer_key] = {
            "statistics": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in result["statistics"].items()
            }
        }

    json_path = output_path.with_suffix(".stats.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_stats, f, ensure_ascii=False, indent=2)
    logger.info(f"统计信息已保存到: {json_path}")


if __name__ == "__main__":
    main()
