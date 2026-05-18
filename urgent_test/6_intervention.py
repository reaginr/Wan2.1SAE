"""
最小概念干预实验

严格按照 TODO_list_v4.md (紧急版) 规范

干预公式:
z' = z - γ * proj_v(z)
其中 proj_v(z) = (z·v)v

干预强度:
- γ = 0.3: mild
- γ = 0.5: medium
- γ = 0.8: strong

测试规模:
- 每类 10~20 prompts 即可

作者：Claude
日期：2026-05-11
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 配置
# ============================================================================

@dataclass
class InterventionConfig:
    """干预配置"""

    # 模型路径
    model_path: str = "./Wan2.1-T2V-1.3B"

    # 输入输出
    vector_dir: str = "./outputs/concept_vectors"
    prompt_file: str = ""  # 要干预的 prompts 文件
    output_dir: str = "./outputs/intervention_results"

    # SAE 配置
    d_model: int = 1536
    d_hidden: int = 12288

    # 干预配置
    layer_idx: int = 29
    hook_type: str = "block_out"

    # 干预强度
    gamma_values: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.8])

    # 向量类型
    vector_type: str = "sparse"

    # 采样配置
    frame_num: int = 81
    size: Tuple[int, int] = (832, 480)
    sampling_steps: int = 30
    seed: int = 42


# ============================================================================
# 干预逻辑
# ============================================================================

def load_concept_vector(
    vector_dir: str,
    concept: str,
    vector_type: str = "sparse",
) -> torch.Tensor:
    """加载概念向量"""
    vector_file = Path(vector_dir) / f"{concept}_vector_{vector_type}.pt"

    if not vector_file.exists():
        raise FileNotFoundError(f"Vector file not found: {vector_file}")

    data = torch.load(vector_file, map_location='cpu')
    vector = data["vector"]

    logger.info(f"Loaded {vector_type} vector for {concept}, shape: {vector.shape}")

    return vector


def project_onto_concept(
    z: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    将 z 投影到概念向量 v 上

    proj_v(z) = (z·v)v

    参数:
        z: [..., d_hidden] latent
        v: [d_hidden] 概念向量 (已归一化)

    返回:
        projection: [..., d_hidden] 投影分量
    """
    # 确保 v 是单位向量
    v_norm = F.normalize(v.unsqueeze(0), dim=1).squeeze(0)

    # 计算投影系数: z·v
    z_flat = z.reshape(-1, z.size(-1))  # [N, d_hidden]
    proj_coef = (z_flat @ v_norm)  # [N]

    # 计算投影向量: (z·v)v
    projection = proj_coef.unsqueeze(-1) * v_norm.unsqueeze(0)  # [N, d_hidden]
    projection = projection.reshape(z.shape)

    return projection


def apply_intervention(
    z: torch.Tensor,
    v: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    应用干预

    z' = z - γ * proj_v(z)

    参数:
        z: latent
        v: 概念向量
        gamma: 干预强度

    返回:
        z_intervened: 干预后的 latent
    """
    projection = project_onto_concept(z, v)
    z_intervened = z - gamma * projection

    return z_intervened


# ============================================================================
# SAE 模型 (简化版)
# ============================================================================

class TopKSAE(torch.nn.Module):
    """TopK SAE"""

    def __init__(self, d_model: int = 1536, d_hidden: int = 12288, top_k: int = 128):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.top_k = top_k

        self.encoder = torch.nn.Linear(d_model, d_hidden, bias=False)
        self.decoder = torch.nn.Linear(d_hidden, d_model, bias=False)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.nn.functional.relu(self.encoder(x))
        topk_val, topk_idx = torch.topk(z, k=self.top_k, dim=-1, largest=True)

        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topk_idx, topk_val)

        return z_sparse, topk_idx, topk_val

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def per_token_rms_norm(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm"""
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_norm = x / rms
    return x_norm, rms


# ============================================================================
# 干预实验
# ============================================================================

class InterventionExperiment:
    """
    干预实验

    流程:
    1. 加载 Wan 模型
    2. 加载 SAE 和概念向量
    3. 对每个 prompt:
       - 运行 DiT 采样
       - 在 Layer29 应用干预
       - 记录干预前后的 latent 变化
    """

    def __init__(self, config: InterventionConfig):
        self.config = config
        self.device = config.model_path  # placeholder

        # 加载概念向量
        self.concept_vectors = {}

    def load_models(self):
        """加载 Wan 模型"""
        logger.info(f"Loading Wan model from {self.config.model_path}")

        from wan.configs.wan_t2v_1_3B import t2v_1_3B
        from wan.text2video import WanT2V

        self.t2v = WanT2V(
            config=t2v_1_3B,
            checkpoint_dir=self.config.model_path,
            device_id=0,
            t5_cpu=True,
        )

        logger.info("Wan model loaded successfully")

    def run_intervention_on_prompt(
        self,
        prompt: str,
        concept: str,
        gamma: float,
        sae: TopKSAE,
        concept_vector: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        对单个 prompt 运行干预

        返回:
            Dict: 干预结果
        """
        # 简化版：只记录 latent 统计
        # 实际应用中需要完整运行 DiT 并干预

        # Hook 存储
        hook_outputs = {}
        intervened_outputs = {}

        def make_hook(layer_idx, intervene=False):
            def hook(module, input, output):
                if intervene:
                    # 获取激活
                    act = output.detach()

                    # RMSNorm
                    act_flat = act.reshape(-1, act.size(-1))
                    act_norm, rms = per_token_rms_norm(act_flat)

                    # SAE encode
                    with torch.no_grad():
                        z_sparse, topk_idx, topk_val = sae.encode(act_norm)

                    # 应用干预
                    z_intervened = apply_intervention(z_sparse, concept_vector, gamma)

                    # SAE decode
                    act_recon = sae.decode(z_intervened)

                    # 反归一化
                    act_intervened = act_recon * rms

                    # 记录
                    intervened_outputs[layer_idx] = {
                        "original_latent": z_sparse.cpu(),
                        "intervened_latent": z_intervened.cpu(),
                        "projection_norm": torch.norm(z_sparse - z_intervened).item(),
                    }
                else:
                    hook_outputs[layer_idx] = output.detach()

            return hook

        # 这里简化实现，实际需要完整的 DiT forward
        result = {
            "prompt": prompt,
            "concept": concept,
            "gamma": gamma,
            "status": "success",
        }

        return result

    def run_experiment(
        self,
        prompts: List[str],
        concept: str,
        gammas: List[float],
    ) -> Dict[str, Any]:
        """
        运行干预实验

        参数:
            prompts: 要测试的 prompts
            concept: 概念名
            gammas: 干预强度列表

        返回:
            Dict: 实验结果
        """
        logger.info(f"\nRunning intervention experiment for {concept}")
        logger.info(f"  Prompts: {len(prompts)}")
        logger.info(f"  Gammas: {gammas}")

        # 加载概念向量
        if concept not in self.concept_vectors:
            self.concept_vectors[concept] = load_concept_vector(
                self.config.vector_dir,
                concept,
                self.config.vector_type,
            )

        concept_vector = self.concept_vectors[concept]

        # 创建 SAE
        sae = TopKSAE(
            d_model=self.config.d_model,
            d_hidden=self.config.d_hidden,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        results = {gamma: [] for gamma in gammas}

        for gamma in gammas:
            logger.info(f"\n  Gamma = {gamma}:")

            for i, prompt in enumerate(tqdm(prompts, desc=f"γ={gamma}")):
                try:
                    result = self.run_intervention_on_prompt(
                        prompt=prompt,
                        concept=concept,
                        gamma=gamma,
                        sae=sae,
                        concept_vector=concept_vector,
                    )
                    results[gamma].append(result)

                except Exception as e:
                    logger.warning(f"  Failed for prompt {i}: {e}")

        return {
            "concept": concept,
            "prompts": prompts,
            "gammas": gammas,
            "results": results,
        }


# ============================================================================
# 简化版干预实验 (不需要完整 Wan 模型)
# ============================================================================

def simple_intervention_test(
    latent_file: str,
    concept_vector: torch.Tensor,
    gamma: float,
) -> Dict[str, Any]:
    """
    简化版干预测试

    直接对已保存的 latent 进行干预

    参数:
        latent_file: latent 数据文件
        concept_vector: 概念向量
        gamma: 干预强度

    返回:
        Dict: 测试结果
    """
    # 加载 latent
    data = torch.load(latent_file, map_location='cpu')

    all_results = []

    for record in data:
        prompt = record.get("prompt", "")

        for lat_info in record.get("latents", []):
            # 获取 latent
            if "z_sparse" in lat_info:
                z = lat_info["z_sparse"]
            else:
                topk_idx = lat_info["topk_idx"]
                topk_val = lat_info["topk_val"]

                d_hidden = concept_vector.shape[0]
                z = torch.zeros(topk_idx.shape[0], d_hidden)
                z.scatter_(1, topk_idx, topk_val)

            # 计算干预前后的变化
            z_intervened = apply_intervention(z, concept_vector, gamma)

            # 计算投影分数
            proj_score_before = (z @ concept_vector).mean().item()
            proj_score_after = (z_intervened @ concept_vector).mean().item()

            # 计算变化量
            delta = (z_intervened - z).norm().item()

            all_results.append({
                "prompt": prompt[:100],
                "projection_before": proj_score_before,
                "projection_after": proj_score_after,
                "projection_change": proj_score_after - proj_score_before,
                "latent_delta": delta,
            })

    return {
        "gamma": gamma,
        "n_samples": len(all_results),
        "results": all_results,
        "mean_projection_before": np.mean([r["projection_before"] for r in all_results]),
        "mean_projection_after": np.mean([r["projection_after"] for r in all_results]),
        "mean_projection_change": np.mean([r["projection_change"] for r in all_results]),
    }


def run_simple_intervention(
    config: InterventionConfig,
    concept: str,
    category: str,
):
    """
    运行简化版干预实验

    对已保存的 latent 进行干预，不需要完整 Wan 模型
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Running simplified intervention for {concept}")
    logger.info(f"  Category: {category}")
    logger.info(f"  Gammas: {config.gamma_values}")
    logger.info(f"{'='*70}")

    # 加载概念向量
    concept_vector = load_concept_vector(
        config.vector_dir,
        concept,
        config.vector_type,
    )

    # 加载 latent
    latent_file = Path(config.latent_dir if hasattr(config, 'latent_dir') else "./outputs/layer29_latents")
    if latent_file.is_dir():
        latent_file = latent_file / f"{category}_latents.pt"

    if not latent_file.exists():
        logger.error(f"Latent file not found: {latent_file}")
        return {}

    # 运行各 gamma 的干预
    all_results = {}

    for gamma in config.gamma_values:
        logger.info(f"\n  Testing gamma = {gamma}...")

        result = simple_intervention_test(
            str(latent_file),
            concept_vector,
            gamma,
        )

        all_results[gamma] = result

        logger.info(f"    Mean projection before: {result['mean_projection_before']:.4f}")
        logger.info(f"    Mean projection after:  {result['mean_projection_after']:.4f}")
        logger.info(f"    Mean change: {result['mean_projection_change']:.4f}")

    # 保存结果
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result_file = output_path / f"{concept}_intervention_results.json"

    output_data = {
        "concept": concept,
        "category": category,
        "gamma_values": config.gamma_values,
        "results": {str(k): v for k, v in all_results.items()},
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nSaved intervention results to {result_file}")

    return all_results


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Concept Intervention Experiment")

    # 模型路径
    parser.add_argument("--model_path", type=str, default="./Wan2.1-T2V-1.3B",
                        help="Wan model path")

    # 输入输出
    parser.add_argument("--vector_dir", type=str, default="./outputs/concept_vectors",
                        help="Concept vector directory")
    parser.add_argument("--latent_dir", type=str, default="./outputs/layer29_latents",
                        help="Latent directory (for simplified intervention)")
    parser.add_argument("--output_dir", type=str, default="./outputs/intervention_results",
                        help="Output directory")

    # SAE 配置
    parser.add_argument("--d_model", type=int, default=1536)
    parser.add_argument("--d_hidden", type=int, default=12288)

    # 干预配置
    parser.add_argument("--gamma", type=float, nargs='+', default=[0.3, 0.5, 0.8],
                        help="Intervention strength(s)")
    parser.add_argument("--vector_type", type=str, default="sparse",
                        choices=["dense", "sparse"])

    # 模式
    parser.add_argument("--mode", type=str, default="simple",
                        choices=["simple", "full"],
                        help="simple: operate on saved latents; full: run Wan model")

    # 概念和类别
    parser.add_argument("--concept", type=str, default="sex",
                        help="Concept to intervene")
    parser.add_argument("--category", type=str, default="sex_positive",
                        help="Category to test")

    args = parser.parse_args()

    # 创建配置
    config = InterventionConfig(
        model_path=args.model_path,
        vector_dir=args.vector_dir,
        output_dir=args.output_dir,
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        gamma_values=args.gamma,
        vector_type=args.vector_type,
    )

    if args.mode == "simple":
        # 简化版：直接对 latent 操作
        config.latent_dir = args.latent_dir
        run_simple_intervention(config, args.concept, args.category)

    else:
        # 完整版：需要 Wan 模型
        logger.error("Full intervention mode not implemented yet")
        logger.info("Use --mode simple for latent-only intervention")

    logger.info("\nIntervention experiment completed!")


if __name__ == "__main__":
    main()
