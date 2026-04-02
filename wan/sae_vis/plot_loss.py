"""
SAE 训练 Loss 可视化工具

功能：
1) 从 JSONL 或 CSV 文件读取持久化的 loss 数据
2) 绘制多种类型的 loss 曲线（线性/对数、平滑/原始、单图/多图）
3) 支持多实验对比可视化
4) 输出高清图片（支持 PNG、PDF、SVG 格式）

用法：
    python wan/sae_vis/plot_loss.py
    python wan/sae_vis/plot_loss.py --loss_file "path/to/loss_history.jsonl"
    python wan/sae_vis/plot_loss.py --compare "exp1,exp2,exp3"

参数传递方式：
    1. 直接修改本文件顶部的参数字典（推荐）
    2. 命令行参数覆盖（如 --loss_file xxx）
    3. 通过 --config 加载 JSON 配置文件
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

# 修复模块导入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


##########################################################################################
# 可视化参数配置区域 - 可直接修改此区域的默认值
# 学术意义与建议值详见每个参数的注释
##########################################################################################

# --------------------------- 数据路径配置 ---------------------------
data_params = {
    # loss_file: 主要 loss 数据文件路径
    # 学术意义: 训练过程中记录的 metrics 数据，用于分析 SAE 收敛性和稀疏性
    # 实际用法: 指向 sae_runs/{exp}/logs/ 目录下的 loss_history.jsonl 或 loss_history.csv
    # 建议值: "sae_runs/exp__20250324/logs/loss_history.jsonl"
    "loss_file": "sae_runs/exp__20250324/logs/loss_history.jsonl",

    # compare_dirs: 多实验对比目录列表
    # 学术意义: 对比不同超参数配置（如 top_k、学习率、层）的训练动态
    # 实际用法: 逗号分隔的实验目录名，如 "exp1,exp2,exp3"
    # 建议值: 空字符串（单实验）或 "exp_topk32,exp_topk64,exp_topk128"
    "compare_dirs": "",

    # run_dir_base: 实验根目录基础路径
    # 实际用法: 所有实验目录的父目录
    # 建议值: "sae_runs" 或绝对路径
    "run_dir_base": "sae_runs",
}

# --------------------------- 图表类型配置 ---------------------------
plot_params = {
    # plot_type: 图表类型
    # 学术意义: 不同图表适合分析不同方面的训练动态
    #   - "loss": 总损失曲线（最常用，观察整体收敛）
    #   - "recon": 重建误差（评估重建质量）
    #   - "sparsity": 稀疏度曲线（验证稀疏性约束效果）
    #   - "multi": 多指标组合图（全面分析）
    #   - "compare": 多实验对比（超参数调优）
    # 建议值: "loss" 用于监控，"multi" 用于论文，"compare" 用于调参
    "plot_type": "multi",

    # metrics: 要绘制的指标列表
    # 实际用法: 从 ["loss", "recon_mse", "l2_norm", "sparsity", "num_activations"] 中选择
    # 建议值: ["loss", "recon_mse", "sparsity"] 或 ["loss"]
    "metrics": ["loss", "recon_mse", "sparsity"],

    # sae_keys: 要绘制的 SAE 键列表
    # 实际用法: 如 ["block_out.layer15"] 或 ["self_attn.layer15", "cross_attn.layer15"]
    # 建议值: 空列表（自动检测所有）或指定特定层
    "sae_keys": [],
}

# --------------------------- 样式配置 ---------------------------
style_params = {
    # figure_size: 图表尺寸（英寸）
    # 实际用法: (宽, 高)，影响输出图片的分辨率
    # 建议值: (12, 6) 单图，(16, 10) 多子图，(10, 6) 论文用
    "figure_size": (14, 8),

    # dpi: 输出图片分辨率
    # 实际用法: 每英寸点数，越高越清晰但文件越大
    # 建议值: 150（屏幕查看），300（论文印刷），600（高质量）
    "dpi": 300,

    # line_width: 曲线线宽
    # 建议值: 1.5（默认），2.0（强调），1.0（细线）
    "line_width": 1.5,

    # marker_size: 数据点标记大小
    # 实际用法: 0 表示不显示标记，>0 显示标记（数据点多时建议 0）
    # 建议值: 0（连续曲线），3（稀疏标记），5（明显标记）
    "marker_size": 0,

    # alpha: 曲线透明度
    # 实际用法: 0-1 之间，多曲线重叠时降低透明度
    # 建议值: 0.8（默认），0.5（多曲线对比）
    "alpha": 0.8,

    # grid: 是否显示网格
    # 建议值: True（便于读数），False（简洁）
    "grid": True,

    # log_scale_y: Y 轴是否使用对数坐标
    # 学术意义: loss 通常指数下降，对数坐标更易观察早期变化
    # 建议值: True（推荐），False（线性坐标）
    "log_scale_y": False,

    # smoothing_window: 平滑窗口大小
    # 学术意义: 消除随机波动，展示趋势；窗口越大越平滑但可能丢失细节
    # 实际用法: 0 表示不平滑，N 表示 N 点移动平均
    # 建议值: 0（原始数据），10（轻度平滑），50（强平滑）
    "smoothing_window": 10,
}

# --------------------------- 输出配置 ---------------------------
output_params = {
    # output_file: 输出文件路径
    # 实际用法: 支持 .png、.pdf、.svg 格式
    # 建议值: 空字符串（自动生成）或指定路径如 "figures/loss_curve.png"
    "output_file": "",

    # output_format: 输出格式
    # 可选值: "png" | "pdf" | "svg" | "all"（输出所有格式）
    # 建议值: "png"（通用），"pdf"（论文矢量图），"svg"（网页/编辑）
    "output_format": "png",

    # show_plot: 是否显示图表窗口
    # 实际用法: 无 GUI 环境设为 False，本地开发设为 True
    # 建议值: False（服务器），True（本地）
    "show_plot": False,

    # title: 图表标题
    # 实际用法: 空字符串自动生成，或自定义如 "SAE Training Loss (Layer 15)"
    # 建议值: ""（自动）或自定义描述
    "title": "",

    # xlabel: X 轴标签
    # 建议值: "Training Step" 或 "Training Step (batch)"
    "xlabel": "Training Step",

    # ylabel: Y 轴标签
    # 建议值: "Loss" 或 "Metric Value"
    "ylabel": "Metric Value",
}

# --------------------------- 多实验对比配置 ---------------------------
compare_params = {
    # compare_labels: 实验标签列表
    # 实际用法: 逗号分隔的图例标签，如 "TopK=32,TopK=64,TopK=128"
    # 建议值: 空字符串（自动使用目录名）或自定义标签
    "compare_labels": "",

    # compare_colors: 实验颜色列表
    # 实际用法: matplotlib 支持的颜色名或十六进制，如 "#FF6B6B,#4ECDC4,#45B7D1"
    # 建议值: 空字符串（自动配色）或自定义配色
    "compare_colors": "",

    # compare_linestyles: 实验线型列表
    # 实际用法: matplotlib 线型，如 "-,--,-.,:"
    # 建议值: 空字符串（实线）或区分不同实验
    "compare_linestyles": "",
}

# --------------------------- 高级配置 ---------------------------
advanced_params = {
    # step_range: 显示的步数范围
    # 实际用法: [start, end]，空列表表示全部
    # 建议值: []（全部）或 [0, 1000]（前1000步）
    "step_range": [],

    # highlight_steps: 高亮显示的特定步数
    # 实际用法: 用于标记 checkpoint 保存点，如 [100, 200, 500]
    # 建议值: []（无高亮）或 checkpoint 保存间隔
    "highlight_steps": [],

    # annotation: 是否添加注释（如最小 loss 点）
    # 建议值: True（自动标注关键信息），False（简洁）
    "annotation": True,

    # tight_layout: 是否使用紧凑布局
    # 建议值: True（自动调整边距），False（手动控制）
    "tight_layout": True,
}


##########################################################################################
# 核心代码区域 - 一般无需修改
##########################################################################################


def parse_step_range(s: str) -> List[int]:
    """解析步数范围字符串，如 '0,1000' -> [0, 1000]"""
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def parse_list_str(s: str, delimiter: str = ",") -> List[str]:
    """解析逗号分隔的字符串列表"""
    if not s:
        return []
    return [p.strip() for p in s.split(delimiter) if p.strip()]


def smooth_data(data: np.ndarray, window: int) -> np.ndarray:
    """应用移动平均平滑"""
    if window <= 1 or len(data) < window:
        return data
    # 使用卷积实现移动平均
    kernel = np.ones(window) / window
    # 边缘处理：使用 'same' 模式保持长度
    smoothed = np.convolve(data, kernel, mode='same')
    return smoothed


def load_loss_data_jsonl(file_path: str) -> pd.DataFrame:
    """从 JSONL 文件加载 loss 数据"""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            step = record["step"]
            timestamp = record.get("timestamp", 0)
            elapsed = record.get("elapsed", 0)
            step_time = record.get("step_time", 0)

            # 展开每个 SAE 的 metrics
            for sae_key, metrics in record["metrics"].items():
                row = {
                    "step": step,
                    "timestamp": timestamp,
                    "elapsed": elapsed,
                    "step_time": step_time,
                    "sae_key": sae_key,
                }
                row.update(metrics)
                records.append(row)

    return pd.DataFrame(records)


def load_loss_data_csv(file_path: str) -> pd.DataFrame:
    """从 CSV 文件加载 loss 数据"""
    return pd.read_csv(file_path)


def load_loss_data(file_path: str) -> pd.DataFrame:
    """自动检测格式并加载 loss 数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Loss 文件不存在: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".jsonl":
        return load_loss_data_jsonl(file_path)
    elif ext == ".csv":
        return load_loss_data_csv(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，请使用 .jsonl 或 .csv")


def create_figure(figsize: Tuple[int, int]) -> Tuple[Figure, plt.Axes]:
    """创建图表"""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def plot_single_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    sae_key: Optional[str] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    linestyle: str = "-",
    linewidth: float = 1.5,
    marker_size: int = 0,
    alpha: float = 0.8,
    smoothing_window: int = 0,
) -> None:
    """绘制单个指标"""
    # 筛选数据
    if sae_key and "sae_key" in df.columns:
        df_plot = df[df["sae_key"] == sae_key].copy()
    else:
        df_plot = df.copy()

    if df_plot.empty:
        return

    # 获取数据
    steps = df_plot["step"].values
    values = df_plot[metric].values

    # 应用平滑
    if smoothing_window > 1:
        values = smooth_data(values, smoothing_window)

    # 构建标签
    if label is None:
        label = f"{sae_key} - {metric}" if sae_key else metric

    # 绘制
    if marker_size > 0:
        ax.plot(
            steps, values,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker="o",
            markersize=marker_size,
            alpha=alpha,
            markevery=max(1, len(steps) // 20),  # 最多显示 20 个标记
        )
    else:
        ax.plot(
            steps, values,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )


def plot_loss_curve(df: pd.DataFrame, params: Dict) -> Figure:
    """绘制 loss 曲线"""
    figsize = tuple(params.get("figure_size", style_params["figure_size"]))
    fig, ax = create_figure(figsize)

    # 获取唯一的 SAE keys
    sae_keys = params.get("sae_keys", plot_params["sae_keys"])
    if not sae_keys and "sae_key" in df.columns:
        sae_keys = df["sae_key"].unique().tolist()
    if not sae_keys:
        sae_keys = [None]

    # 绘制每个 SAE 的 loss
    for i, sae_key in enumerate(sae_keys):
        plot_single_metric(
            ax, df, "loss", sae_key=sae_key,
            linewidth=params.get("line_width", style_params["line_width"]),
            marker_size=params.get("marker_size", style_params["marker_size"]),
            alpha=params.get("alpha", style_params["alpha"]),
            smoothing_window=params.get("smoothing_window", style_params["smoothing_window"]),
        )

    # 设置样式
    ax.set_xlabel(params.get("xlabel", output_params["xlabel"]))
    ax.set_ylabel(params.get("ylabel", output_params["ylabel"]))
    ax.set_title(params.get("title", "SAE Training Loss"))

    if params.get("grid", style_params["grid"]):
        ax.grid(True, alpha=0.3)

    if params.get("log_scale_y", style_params["log_scale_y"]):
        ax.set_yscale("log")

    ax.legend()

    return fig


def plot_multi_metrics(df: pd.DataFrame, params: Dict) -> Figure:
    """绘制多指标组合图"""
    metrics = params.get("metrics", plot_params["metrics"])
    figsize = tuple(params.get("figure_size", style_params["figure_size"]))

    # 动态计算子图布局
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes

    # 获取唯一的 SAE keys
    sae_keys = params.get("sae_keys", plot_params["sae_keys"])
    if not sae_keys and "sae_key" in df.columns:
        sae_keys = df["sae_key"].unique().tolist()
    if not sae_keys:
        sae_keys = [None]

    colors = plt.cm.tab10(np.linspace(0, 1, len(sae_keys)))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        for i, sae_key in enumerate(sae_keys):
            plot_single_metric(
                ax, df, metric, sae_key=sae_key,
                color=colors[i],
                linewidth=params.get("line_width", style_params["line_width"]),
                marker_size=params.get("marker_size", style_params["marker_size"]),
                alpha=params.get("alpha", style_params["alpha"]),
                smoothing_window=params.get("smoothing_window", style_params["smoothing_window"]),
            )

        ax.set_xlabel("Training Step")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()}")

        if params.get("grid", style_params["grid"]):
            ax.grid(True, alpha=0.3)

        if len(sae_keys) > 1:
            ax.legend()

    # 隐藏多余的子图
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig


def plot_comparison(dfs: List[pd.DataFrame], labels: List[str], params: Dict) -> Figure:
    """绘制多实验对比图"""
    figsize = tuple(params.get("figure_size", style_params["figure_size"]))
    metric = params.get("compare_metric", "loss")

    fig, ax = create_figure(figsize)

    # 解析颜色、线型
    colors_str = params.get("compare_colors", compare_params["compare_colors"])
    colors = parse_list_str(colors_str) if colors_str else None

    linestyles_str = params.get("compare_linestyles", compare_params["compare_linestyles"])
    linestyles = parse_list_str(linestyles_str) if linestyles_str else None

    for i, (df, label) in enumerate(zip(dfs, labels)):
        color = colors[i] if colors and i < len(colors) else None
        linestyle = linestyles[i] if linestyles and i < len(linestyles) else "-"

        # 获取第一个 SAE key 的数据
        if "sae_key" in df.columns:
            sae_key = df["sae_key"].unique()[0]
            df_plot = df[df["sae_key"] == sae_key]
        else:
            df_plot = df

        steps = df_plot["step"].values
        values = df_plot[metric].values

        # 应用平滑
        smoothing_window = params.get("smoothing_window", style_params["smoothing_window"])
        if smoothing_window > 1:
            values = smooth_data(values, smoothing_window)

        ax.plot(
            steps, values,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=params.get("line_width", style_params["line_width"]),
            alpha=params.get("alpha", style_params["alpha"]),
        )

    ax.set_xlabel(params.get("xlabel", output_params["xlabel"]))
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(params.get("title", f"{metric.replace('_', ' ').title()} Comparison"))

    if params.get("grid", style_params["grid"]):
        ax.grid(True, alpha=0.3)

    if params.get("log_scale_y", style_params["log_scale_y"]):
        ax.set_yscale("log")

    ax.legend()

    return fig


def save_figure(fig: Figure, output_file: str, output_format: str, dpi: int) -> None:
    """保存图表到文件"""
    if output_format == "all":
        formats = ["png", "pdf", "svg"]
    else:
        formats = [output_format]

    base_path = os.path.splitext(output_file)[0]

    for fmt in formats:
        if fmt == "png":
            fig.savefig(f"{base_path}.png", dpi=dpi, bbox_inches="tight")
            print(f"  已保存: {base_path}.png")
        elif fmt == "pdf":
            fig.savefig(f"{base_path}.pdf", bbox_inches="tight")
            print(f"  已保存: {base_path}.pdf")
        elif fmt == "svg":
            fig.savefig(f"{base_path}.svg", bbox_inches="tight")
            print(f"  已保存: {base_path}.svg")


def main():
    # 支持命令行参数覆盖默认配置
    parser = argparse.ArgumentParser(description="Visualize SAE training loss curves.")
    parser.add_argument("--config", type=str, default="", help="JSON 配置文件路径")
    parser.add_argument("--loss_file", type=str, default=data_params["loss_file"])
    parser.add_argument("--compare_dirs", type=str, default=data_params["compare_dirs"])
    parser.add_argument("--run_dir_base", type=str, default=data_params["run_dir_base"])
    parser.add_argument("--plot_type", type=str, default=plot_params["plot_type"])
    parser.add_argument("--output_file", type=str, default=output_params["output_file"])
    parser.add_argument("--output_format", type=str, default=output_params["output_format"])
    parser.add_argument("--show_plot", action="store_true", default=output_params["show_plot"])
    parser.add_argument("--title", type=str, default=output_params["title"])
    parser.add_argument("--dpi", type=int, default=style_params["dpi"])
    parser.add_argument("--smoothing_window", type=int, default=style_params["smoothing_window"])
    parser.add_argument("--log_scale_y", action="store_true", default=style_params["log_scale_y"])

    args = parser.parse_args()

    # 收集参数
    params = {
        "loss_file": args.loss_file,
        "compare_dirs": args.compare_dirs,
        "run_dir_base": args.run_dir_base,
        "plot_type": args.plot_type,
        "output_file": args.output_file,
        "output_format": args.output_format,
        "show_plot": args.show_plot,
        "title": args.title,
        "dpi": args.dpi,
        "smoothing_window": args.smoothing_window,
        "log_scale_y": args.log_scale_y,
        # 保持默认的其他参数
        "figure_size": style_params["figure_size"],
        "line_width": style_params["line_width"],
        "marker_size": style_params["marker_size"],
        "alpha": style_params["alpha"],
        "grid": style_params["grid"],
        "metrics": plot_params["metrics"],
        "sae_keys": plot_params["sae_keys"],
        "xlabel": output_params["xlabel"],
        "ylabel": output_params["ylabel"],
    }

    # 自动推断输出文件名
    if not params["output_file"]:
        if params["plot_type"] == "compare" and params["compare_dirs"]:
            params["output_file"] = f"loss_comparison.{params['output_format']}"
        else:
            loss_file = params["loss_file"]
            base_dir = os.path.dirname(os.path.dirname(loss_file))  # sae_runs/exp_xx/
            params["output_file"] = os.path.join(
                base_dir, f"loss_{params['plot_type']}.{params['output_format']}"
            )

    print("=" * 60)
    print("SAE Loss 可视化工具")
    print("=" * 60)
    print(f"图表类型: {params['plot_type']}")
    print(f"输出文件: {params['output_file']}")
    print("-" * 60)

    # 执行绘图
    if params["plot_type"] == "compare" and params["compare_dirs"]:
        # 多实验对比模式
        exp_dirs = parse_list_str(params["compare_dirs"])
        labels = parse_list_str(compare_params["compare_labels"]) or exp_dirs

        dfs = []
        for exp_dir in exp_dirs:
            loss_file = os.path.join(
                params["run_dir_base"], exp_dir, "logs", "loss_history.jsonl"
            )
            if not os.path.exists(loss_file):
                # 尝试 CSV 格式
                loss_file = loss_file.replace(".jsonl", ".csv")
            df = load_loss_data(loss_file)
            dfs.append(df)

        fig = plot_comparison(dfs, labels, params)

    else:
        # 单实验模式
        df = load_loss_data(params["loss_file"])
        print(f"加载数据: {len(df)} 条记录")
        print(f"SAE Keys: {df['sae_key'].unique().tolist() if 'sae_key' in df.columns else 'N/A'}")
        print(f"步数范围: {df['step'].min()} - {df['step'].max()}")

        if params["plot_type"] == "loss":
            fig = plot_loss_curve(df, params)
        elif params["plot_type"] == "multi":
            fig = plot_multi_metrics(df, params)
        else:
            fig = plot_loss_curve(df, params)

    # 保存
    save_figure(fig, params["output_file"], params["output_format"], params["dpi"])

    # 显示
    if params["show_plot"]:
        plt.show()

    plt.close(fig)
    print("=" * 60)
    print("可视化完成!")


if __name__ == "__main__":
    main()
