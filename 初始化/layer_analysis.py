"""
Layer-wise Activation 分析工具

专门分析：
1. Layer29 active ratio 显著高于其他层的原因
2. 不同层的特征解缠程度
3. 初始化 vs 层内在特性的影响

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class LayerAnalysisResult:
    """单层分析结果"""
    layer_idx: int
    active_ratio: float
    competition_entropy: float
    gini: float
    participation_ratio: float
    norm_statistics: Dict[str, float]
    feature_diversity: float
    intrinsic_dimension: float


class LayerWiseAnalyzer:
    """
    层级分析器

    分析不同层的激活特性，回答：
    Layer29 active ratio 高是初始化问题还是层内在特性？
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def analyze_single_layer(
        self,
        activations: torch.Tensor,
        layer_idx: int,
        top_k: int = 128,
        sample_size: int = 5000,
    ) -> LayerAnalysisResult:
        """
        分析单层激活

        参数:
            activations: [N, C] 激活数据
            layer_idx: 层索引
            top_k: TopK 数量
            sample_size: 采样大小
        """
        if activations.dim() == 3:
            activations = activations.reshape(-1, activations.shape[-1])

        # 移动到设备
        activations = activations.to(self.device).float()

        # 采样
        n = min(sample_size, len(activations))
        if n < len(activations):
            perm = torch.randperm(len(activations), device=self.device)[:n]
            activations = activations[perm]

        # RMSNorm
        rms = torch.sqrt(activations.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        activations_normed = activations / rms

        # 1. Norm 统计
        norms = activations.norm(dim=-1)
        norm_stats = {
            "mean": norms.mean().item(),
            "std": norms.std().item(),
            "min": norms.min().item(),
            "max": norms.max().item(),
            "cv": (norms.std() / norms.mean()).item(),  # 变异系数
        }

        # 2. Active Ratio 计算
        # 使用随机投影模拟 SAE 编码
        n_features = activations_normed.shape[1]
        random_weights = torch.randn(n_features, n_features * 8, device=self.device)
        random_weights = F.normalize(random_weights, dim=0)

        encoded = activations_normed @ random_weights  # [N, K]

        # TopK 激活
        _, topk_indices = torch.topk(encoded.abs(), top_k, dim=-1)

        # Firing counts
        firing_counts = torch.zeros(n_features * 8, device=self.device)
        firing_counts.scatter_add_(0, topk_indices.flatten(),
                                   torch.ones(topk_indices.numel(), device=self.device))

        active_ratio = (firing_counts > 0).sum().item() / (n_features * 8)

        # 3. Competition Entropy
        prob = firing_counts / (firing_counts.sum() + 1e-10)
        prob = prob.clamp(min=1e-10)
        entropy = -(prob * torch.log(prob)).sum().item()
        max_entropy = torch.log(torch.tensor(n_features * 8, dtype=torch.float32)).item()
        competition_entropy = entropy / max_entropy

        # 4. Gini
        values = firing_counts.float().sort()[0]
        n_vals = len(values)
        index = torch.arange(1, n_vals + 1, dtype=torch.float32, device=self.device)
        gini = (2 * (index * values).sum()) / (n_vals * values.sum() + 1e-10) - (n_vals + 1) / n_vals

        # 5. 参与比 (Participation Ratio) - 估计内在维度
        # 使用协方差矩阵的特征值
        if len(activations_normed) > 1000:
            act_subsample = activations_normed[:1000]
        else:
            act_subsample = activations_normed

        # 计算相关性矩阵
        act_centered = act_subsample - act_subsample.mean(dim=0)
        cov = act_centered.T @ act_centered / len(act_centered)

        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        if len(eigenvalues) > 0:
            participation_ratio = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
            participation_ratio = participation_ratio.item()
        else:
            participation_ratio = 0.0

        # 6. 特征多样性 (基于 token 间余弦相似度)
        # 采样计算
        n_pairs = min(10000, len(activations_normed) * (len(activations_normed) - 1) // 2)
        idx_i = torch.randint(0, len(activations_normed), (n_pairs,), device=self.device)
        idx_j = torch.randint(0, len(activations_normed), (n_pairs,), device=self.device)
        mask = idx_i != idx_j

        cos_sim = (activations_normed[idx_i[mask]] * activations_normed[idx_j[mask]]).sum(dim=-1).abs()
        feature_diversity = 1.0 - cos_sim.mean().item()  # 多样性 = 1 - 平均相似度

        return LayerAnalysisResult(
            layer_idx=layer_idx,
            active_ratio=active_ratio,
            competition_entropy=competition_entropy,
            gini=gini.item(),
            participation_ratio=participation_ratio,
            norm_statistics=norm_stats,
            feature_diversity=feature_diversity,
            intrinsic_dimension=participation_ratio,
        )

    def compare_layers(
        self,
        activations_by_layer: Dict[int, torch.Tensor],
        top_k: int = 128,
        sample_size: int = 5000,
    ) -> Dict[str, Any]:
        """
        比较不同层的特性

        回答：Layer29 active ratio 高是初始化问题还是层内在特性？
        """
        results = {}

        for layer_idx, activations in activations_by_layer.items():
            results[layer_idx] = self.analyze_single_layer(
                activations, layer_idx, top_k, sample_size
            )

        # 分析 Layer29 vs 其他层
        if 29 in results:
            layer29 = results[29]
            other_layers = [l for l in results if l != 29]

            # 计算其他层的平均
            avg_other = {
                "active_ratio": sum(results[l].active_ratio for l in other_layers) / len(other_layers),
                "competition_entropy": sum(results[l].competition_entropy for l in other_layers) / len(other_layers),
                "gini": sum(results[l].gini for l in other_layers) / len(other_layers),
                "participation_ratio": sum(results[l].participation_ratio for l in other_layers) / len(other_layers),
                "feature_diversity": sum(results[l].feature_diversity for l in other_layers) / len(other_layers),
            }

            # 分析结论
            conclusions = []

            # 1. Active ratio 差异分析
            active_ratio_ratio = layer29.active_ratio / avg_other["active_ratio"]
            if active_ratio_ratio > 1.5:
                conclusions.append(
                    f"Layer29 active ratio 是其他层的 {active_ratio_ratio:.2f} 倍，差异显著"
                )

            # 2. 参与比分析 (内在维度)
            if layer29.participation_ratio > avg_other["participation_ratio"]:
                conclusions.append(
                    f"Layer29 参与比 ({layer29.participation_ratio:.2f}) 高于其他层平均 ({avg_other['participation_ratio']:.2f})"
                )
                conclusions.append(
                    "→ Layer29 特征更解缠 (intrinsic semantic disentanglement)"
                )
            else:
                conclusions.append(
                    f"Layer29 参与比 ({layer29.participation_ratio:.2f}) 未显著高于其他层"
                )
                conclusions.append(
                    "→ 高 active ratio 可能来自初始化更适合该层"
                )

            # 3. 特征多样性分析
            if layer29.feature_diversity > avg_other["feature_diversity"]:
                conclusions.append(
                    f"Layer29 特征多样性 ({layer29.feature_diversity:.4f}) 更高"
                )
            else:
                conclusions.append(
                    f"Layer29 特征多样性 ({layer29.feature_diversity:.4f}) 较低"
                )

            # 4. Norm 分布分析
            norm_cv_ratio = layer29.norm_statistics["cv"] / avg_other.get("norm_cv", 1)
            if norm_cv_ratio < 0.8:
                conclusions.append(
                    f"Layer29 norm 变异系数 ({layer29.norm_statistics['cv']:.4f}) 较低，特征更均匀"
                )

            # 综合判断
            if layer29.participation_ratio > avg_other["participation_ratio"] * 1.1:
                primary_cause = "intrinsic_layer_property"
                explanation = (
                    "Layer29 active ratio 高主要来自该层的内在语义解缠特性。\n"
                    "深层特征更稀疏，天然具有更高的 active ratio。"
                )
            else:
                primary_cause = "initialization_bias"
                explanation = (
                    "Layer29 active ratio 高可能来自初始化更适合该层。\n"
                    "其他层可能需要调整初始化策略。"
                )

            results["comparison"] = {
                "layer29": {
                    "active_ratio": layer29.active_ratio,
                    "participation_ratio": layer29.participation_ratio,
                    "feature_diversity": layer29.feature_diversity,
                    "norm_cv": layer29.norm_statistics["cv"],
                },
                "avg_other_layers": avg_other,
                "active_ratio_ratio": active_ratio_ratio,
                "conclusions": conclusions,
                "primary_cause": primary_cause,
                "explanation": explanation,
            }

        return results

    def print_comparison(self, results: Dict[str, Any]):
        """打印比较结果"""
        print("\n" + "=" * 70)
        print("Layer-wise Activation Analysis")
        print("=" * 70)

        # 打印各层统计
        print("\n[各层统计]")
        print(f"{'Layer':<8} {'Active%':<10} {'Entropy':<10} {'Gini':<10} {'PR':<10} {'Diversity':<10}")
        print("-" * 66)

        for layer_idx in sorted([k for k in results if isinstance(k, int)]):
            r = results[layer_idx]
            print(f"{layer_idx:<8} {r.active_ratio*100:<10.2f} {r.competition_entropy:<10.4f} "
                  f"{r.gini:<10.4f} {r.participation_ratio:<10.2f} {r.feature_diversity:<10.4f}")

        # 打印比较分析
        if "comparison" in results:
            comp = results["comparison"]
            print("\n" + "=" * 70)
            print("[Layer29 vs 其他层]")
            print("=" * 70)

            print(f"\nLayer29:")
            for k, v in comp["layer29"].items():
                print(f"  {k}: {v}")

            print(f"\n其他层平均:")
            for k, v in comp["avg_other_layers"].items():
                print(f"  {k}: {v}")

            print(f"\n结论:")
            for c in comp["conclusions"]:
                print(f"  - {c}")

            print(f"\n主要原因: {comp['primary_cause']}")
            print(f"\n解释:")
            print(f"  {comp['explanation']}")

        print("\n" + "=" * 70)


def analyze_layer29_phenomenon(
    activations_by_layer: Dict[int, torch.Tensor],
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    分析 Layer29 现象的便捷函数

    返回完整的分析报告
    """
    analyzer = LayerWiseAnalyzer(device=device)
    results = analyzer.compare_layers(activations_by_layer)
    analyzer.print_comparison(results)
    return results
