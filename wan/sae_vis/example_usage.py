"""
SAE 可视化工具使用示例

本脚本演示如何使用可视化工具分析训练结果。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wan.sae_vis import load_loss_data, plot_loss_curve, plot_multi_metrics


def example_basic_usage():
    """示例 1: 基础用法 - 加载并查看数据"""
    print("=" * 60)
    print("示例 1: 基础数据加载")
    print("=" * 60)

    # 加载数据
    loss_file = "sae_runs/exp__20250324/logs/loss_history.jsonl"

    if not os.path.exists(loss_file):
        print(f"文件不存在: {loss_file}")
        print("请先完成训练生成 loss 数据")
        return

    df = load_loss_data(loss_file)

    # 查看基本信息
    print(f"\n数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print(f"\n前 5 行:")
    print(df.head())

    # 统计信息
    print(f"\n统计信息:")
    print(f"  步数范围: {df['step'].min()} - {df['step'].max()}")
    print(f"  SAE Keys: {df['sae_key'].unique().tolist()}")
    print(f"  最终 Loss: {df['loss'].iloc[-1]:.6f}")
    print(f"  最小 Loss: {df['loss'].min():.6f}")
    print(f"  平均稀疏度: {df['sparsity'].mean():.4f}")


def example_plot_single():
    """示例 2: 绘制单指标图表"""
    print("\n" + "=" * 60)
    print("示例 2: 绘制 Loss 曲线")
    print("=" * 60)

    import matplotlib.pyplot as plt

    loss_file = "sae_runs/exp__20250324/logs/loss_history.jsonl"
    if not os.path.exists(loss_file):
        print("数据文件不存在，跳过")
        return

    df = load_loss_data(loss_file)

    # 自定义参数
    params = {
        "figure_size": (10, 6),
        "dpi": 150,
        "line_width": 2.0,
        "smoothing_window": 10,
        "log_scale_y": False,
        "grid": True,
        "title": "SAE Training Loss (Layer 15)",
        "xlabel": "Training Step",
        "ylabel": "Loss",
        "alpha": 0.8,
        "marker_size": 0,
        "show_plot": False,
    }

    # 绘制
    fig = plot_loss_curve(df, params)

    # 保存
    output_path = "sae_runs/exp__20250324/loss_single_example.png"
    fig.savefig(output_path, dpi=params["dpi"], bbox_inches="tight")
    print(f"图表已保存: {output_path}")

    plt.close(fig)


def example_plot_multi():
    """示例 3: 绘制多指标组合图"""
    print("\n" + "=" * 60)
    print("示例 3: 绘制多指标组合图")
    print("=" * 60)

    import matplotlib.pyplot as plt

    loss_file = "sae_runs/exp__20250324/logs/loss_history.jsonl"
    if not os.path.exists(loss_file):
        print("数据文件不存在，跳过")
        return

    df = load_loss_data(loss_file)

    # 自定义参数
    params = {
        "figure_size": (16, 10),
        "dpi": 200,
        "line_width": 1.5,
        "smoothing_window": 10,
        "log_scale_y": False,
        "grid": True,
        "title": "SAE Training Analysis",
        "metrics": ["loss", "recon_mse", "sparsity", "num_activations"],
        "sae_keys": [],  # 自动检测
        "alpha": 0.8,
        "marker_size": 0,
        "show_plot": False,
    }

    # 绘制
    fig = plot_multi_metrics(df, params)

    # 保存
    output_path = "sae_runs/exp__20250324/loss_multi_example.png"
    fig.savefig(output_path, dpi=params["dpi"], bbox_inches="tight")
    print(f"图表已保存: {output_path}")

    plt.close(fig)


def example_custom_analysis():
    """示例 4: 自定义分析 - 计算收敛速度"""
    print("\n" + "=" * 60)
    print("示例 4: 自定义分析 - 收敛速度")
    print("=" * 60)

    import numpy as np

    loss_file = "sae_runs/exp__20250324/logs/loss_history.jsonl"
    if not os.path.exists(loss_file):
        print("数据文件不存在，跳过")
        return

    df = load_loss_data(loss_file)

    # 筛选特定 SAE
    df_layer = df[df["sae_key"] == "block_out.layer15"]

    if len(df_layer) < 10:
        print("数据点太少，无法分析")
        return

    losses = df_layer["loss"].values
    steps = df_layer["step"].values

    # 计算收敛速度（最近 10 步的平均下降）
    window = min(10, len(losses) // 2)
    if window > 1:
        recent_avg = losses[-window:].mean()
        earlier_avg = losses[-window*2:-window].mean()
        improvement = (earlier_avg - recent_avg) / earlier_avg * 100
        print(f"最近 {window} 步平均改进: {improvement:.2f}%")

    # 计算半衰期（loss 下降到一半需要的步数）
    initial_loss = losses[0]
    half_loss = initial_loss / 2
    half_idx = np.where(losses <= half_loss)[0]
    if len(half_idx) > 0:
        half_step = steps[half_idx[0]]
        print(f"Loss 减半步数: {half_step} (从 {initial_loss:.4f} 到 {half_loss:.4f})")
    else:
        print(f"Loss 尚未减半 (当前 {losses[-1]:.4f}, 初始 {initial_loss:.4f})")

    # 稀疏度稳定性
    sparsity_std = df_layer["sparsity"].std()
    print(f"稀疏度标准差: {sparsity_std:.6f} (越小越稳定)")


def example_compare_experiments():
    """示例 5: 对比多个实验"""
    print("\n" + "=" * 60)
    print("示例 5: 多实验对比")
    print("=" * 60)

    # 假设有两个实验
    exp_dirs = ["exp__20250324", "exp__20250325"]

    from wan.sae_vis.plot_loss import plot_comparison
    import matplotlib.pyplot as plt

    dfs = []
    labels = []

    for exp_dir in exp_dirs:
        loss_file = f"sae_runs/{exp_dir}/logs/loss_history.jsonl"
        if os.path.exists(loss_file):
            df = load_loss_data(loss_file)
            dfs.append(df)
            labels.append(exp_dir)
            print(f"加载: {exp_dir} ({len(df)} 条记录)")

    if len(dfs) < 2:
        print("需要至少 2 个实验进行对比")
        return

    # 对比参数
    params = {
        "figure_size": (12, 6),
        "dpi": 200,
        "line_width": 2.0,
        "smoothing_window": 10,
        "log_scale_y": False,
        "grid": True,
        "title": "Experiment Comparison",
        "compare_metric": "loss",
        "xlabel": "Training Step",
        "ylabel": "Loss",
        "alpha": 0.8,
    }

    fig = plot_comparison(dfs, labels, params)

    output_path = "sae_runs/loss_comparison_example.png"
    fig.savefig(output_path, dpi=params["dpi"], bbox_inches="tight")
    print(f"对比图已保存: {output_path}")

    plt.close(fig)


def main():
    """运行所有示例"""
    print("\n" + "=" * 70)
    print("SAE 可视化工具使用示例")
    print("=" * 70)

    # 检查数据是否存在
    test_file = "sae_runs/exp__20250324/logs/loss_history.jsonl"
    if not os.path.exists(test_file):
        print(f"\n注意: 未找到示例数据文件: {test_file}")
        print("请先运行训练脚本生成 loss 数据，或修改示例中的路径。\n")

    # 运行示例
    try:
        example_basic_usage()
    except Exception as e:
        print(f"示例 1 失败: {e}")

    try:
        example_plot_single()
    except Exception as e:
        print(f"示例 2 失败: {e}")

    try:
        example_plot_multi()
    except Exception as e:
        print(f"示例 3 失败: {e}")

    try:
        example_custom_analysis()
    except Exception as e:
        print(f"示例 4 失败: {e}")

    try:
        example_compare_experiments()
    except Exception as e:
        print(f"示例 5 失败: {e}")

    print("\n" + "=" * 70)
    print("示例运行完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
