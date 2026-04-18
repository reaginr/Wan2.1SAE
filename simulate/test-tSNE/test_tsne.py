"""
t-SNE可视化测试脚本

测试流程：
1. 生成模拟数据（4种分离度类型）
2. 对每个数据集运行t-SNE可视化
3. 验证Silhouette Score是否符合预期
4. 生成对比报告

使用：
    python simulate/test-tSNE/test_tsne.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wan.sae.interpretability import ActivationIO
from wan.sae.interpretability.visualize_tsne import TSNEVisualizer, ClusteringMetrics


def test_tsne_on_category(
    activation_root: str,
    category: str,
    layer_key: str = "sae_layer15",
    output_dir: str = "simulate/test-tSNE/results",
) -> Dict:
    """对单个类别运行t-SNE测试"""
    print(f"\n{'='*60}")
    print(f"测试: {category}")
    print(f"{'='*60}")

    # 解析层key
    layer_type, layer_idx = layer_key.split("_layer")
    layer_idx = int(layer_idx)

    # 初始化
    io = ActivationIO(activation_root)

    visualizer = TSNEVisualizer(
        io=io,
        category=category,
        layer_type=layer_type,
        layer_idx=layer_idx,
        perplexity=30,
        n_iter=1000,
        learning_rate=200.0,
        random_state=42,
        n_components=2,
        pca_components=50,  # 降维到50维加速t-SNE
        batch_size=32,
    )

    # 运行可视化
    result = visualizer.run(output_dir)

    return result


def evaluate_result(category: str, result: Dict) -> bool:
    """评估结果是否符合预期"""
    silhouette = result["metrics"]["silhouette_score"]

    expected_ranges = {
        "well_separated": (0.6, 1.0),
        "partially_separated": (0.25, 0.6),
        "overlapping": (-0.1, 0.25),
        "sparse_separated": (0.5, 1.0),
    }

    min_val, max_val = expected_ranges.get(category, (0.0, 1.0))
    passed = min_val <= silhouette <= max_val

    print(f"\n  评估结果:")
    print(f"    Silhouette Score: {silhouette:.4f}")
    print(f"    预期范围: [{min_val:.2f}, {max_val:.2f}]")
    print(f"    状态: {'✓ PASS' if passed else '✗ FAIL'}")

    return passed


def generate_comparison_report(
    results: Dict[str, Dict],
    output_path: str = "simulate/test-tSNE/comparison_report.json",
):
    """生成对比报告"""
    summary = {
        "test_summary": {},
        "ranking_by_separation": [],
        "visualization_files": {},
    }

    for category, result in results.items():
        summary["test_summary"][category] = {
            "silhouette_score": result["metrics"]["silhouette_score"],
            "calinski_harabasz": result["metrics"]["calinski_harabasz_score"],
            "center_distance": result["metrics"]["center_distance"],
            "num_samples": result["num_pos"] + result["num_neg"],
            "separation_quality": result["interpretation"]["separation_quality"],
        }

    # 按分离度排序
    sorted_cats = sorted(
        results.items(),
        key=lambda x: x[1]["metrics"]["silhouette_score"],
        reverse=True,
    )
    summary["ranking_by_separation"] = [
        {"rank": i+1, "category": cat, "silhouette": result["metrics"]["silhouette_score"]}
        for i, (cat, result) in enumerate(sorted_cats)
    ]

    # 可视化文件路径
    for category in results.keys():
        summary["visualization_files"][category] = {
            "png": f"{category}_sae_layer15_tsne.png",
            "metrics": f"{category}_sae_layer15_metrics.json",
            "data": f"{category}_sae_layer15_tsne.npz",
        }

    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n报告已保存: {output_path}")
    return summary


def print_final_summary(results: Dict[str, Dict], all_passed: bool):
    """打印最终摘要"""
    print("\n" + "="*60)
    print("t-SNE 测试最终摘要")
    print("="*60)

    print("\n各数据集分离度排名:")
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["metrics"]["silhouette_score"],
        reverse=True,
    )

    for i, (category, result) in enumerate(sorted_results, 1):
        silhouette = result["metrics"]["silhouette_score"]
        quality = result["interpretation"]["separation_quality"]
        print(f"  {i}. {category:20s} Silhouette={silhouette:.4f} ({quality})")

    print(f"\n整体测试: {'✓ 全部通过' if all_passed else '✗ 部分失败'}")
    print("="*60)


def main():
    # 配置
    activation_root = "simulate/test-tSNE/mock_activations"
    output_dir = "simulate/test-tSNE/results"
    categories = [
        "well_separated",
        "partially_separated",
        "overlapping",
        "sparse_separated",
    ]

    print("="*60)
    print("t-SNE 可视化测试")
    print("="*60)
    print(f"数据目录: {activation_root}")
    print(f"输出目录: {output_dir}")
    print(f"测试类别: {categories}")

    # 检查数据是否存在，不存在则生成
    if not Path(activation_root).exists():
        print("\n模拟数据不存在，正在生成...")
        from generate_mock_data import generate_all_mock_datasets
        generate_all_mock_datasets(activation_root)

    # 运行测试
    results = {}
    all_passed = True

    for category in categories:
        try:
            result = test_tsne_on_category(
                activation_root=activation_root,
                category=category,
                output_dir=output_dir,
            )
            results[category] = result

            # 评估结果
            passed = evaluate_result(category, result)
            if not passed:
                all_passed = False

        except Exception as e:
            print(f"  错误: {e}")
            all_passed = False

    # 生成对比报告
    summary = generate_comparison_report(results)

    # 打印最终摘要
    print_final_summary(results, all_passed)

    # 返回退出码
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
