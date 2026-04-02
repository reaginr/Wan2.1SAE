"""
SAE Loss 快速可视化脚本

一键绘制常用图表，无需修改代码参数。

用法：
    # 基础用法 - 自动查找当前实验的 loss 文件
    python wan/sae_vis/quick_plot.py

    # 指定实验目录
    python wan/sae_vis/quick_plot.py exp__20250324

    # 指定具体文件
    python wan/sae_vis/quick_plot.py --file sae_runs/exp__20250324/logs/loss_history.jsonl

    # 对比多个实验
    python wan/sae_vis/quick_plot.py --compare exp__20250324,exp__20250325

    # 输出论文质量图片
    python wan/sae_vis/quick_plot.py --paper

    # 查看原始数据（无平滑）
    python wan/sae_vis/quick_plot.py --no-smooth
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wan.sae_vis.plot_loss import main as plot_main, parse_list_str


def find_latest_experiment(base_dir: str = "sae_runs") -> str:
    """自动查找最新的实验目录"""
    if not os.path.exists(base_dir):
        return None

    exp_dirs = []
    for d in os.listdir(base_dir):
        exp_path = os.path.join(base_dir, d)
        if os.path.isdir(exp_path) and d.startswith("exp_"):
            # 检查是否有 loss 文件
            loss_file = os.path.join(exp_path, "logs", "loss_history.jsonl")
            if os.path.exists(loss_file):
                exp_dirs.append((d, os.path.getmtime(loss_file)))

    if not exp_dirs:
        return None

    # 按修改时间排序，返回最新的
    exp_dirs.sort(key=lambda x: x[1], reverse=True)
    return exp_dirs[0][0]


def main():
    parser = argparse.ArgumentParser(description="快速绘制 SAE loss 曲线")
    parser.add_argument("exp_dir", nargs="?", default=None, help="实验目录名（如 exp__20250324）")
    parser.add_argument("--file", type=str, default=None, help="指定 loss 文件路径")
    parser.add_argument("--compare", type=str, default=None, help="对比多个实验，逗号分隔")
    parser.add_argument("--paper", action="store_true", help="论文质量输出（300 DPI，无平滑）")
    parser.add_argument("--no-smooth", action="store_true", help="禁用平滑")
    parser.add_argument("--log", action="store_true", help="使用对数 Y 轴")
    parser.add_argument("--show", action="store_true", help="显示图表窗口")
    parser.add_argument("--type", type=str, default="multi", choices=["loss", "multi", "compare"],
                        help="图表类型")

    args = parser.parse_args()

    # 构建命令行参数传递给 plot_loss.py
    plot_args = [
        "--plot_type", args.type,
        "--show_plot" if args.show else "",
        "--log_scale_y" if args.log else "",
    ]

    # 处理输出质量
    if args.paper:
        plot_args.extend(["--dpi", "300", "--smoothing_window", "0"])
    elif args.no_smooth:
        plot_args.extend(["--smoothing_window", "0"])
    else:
        plot_args.extend(["--smoothing_window", "10"])

    # 处理文件路径
    if args.file:
        plot_args.extend(["--loss_file", args.file])
    elif args.compare:
        plot_args.extend(["--compare_dirs", args.compare, "--plot_type", "compare"])
    elif args.exp_dir:
        loss_file = f"sae_runs/{args.exp_dir}/logs/loss_history.jsonl"
        plot_args.extend(["--loss_file", loss_file])
    else:
        # 自动查找最新实验
        latest = find_latest_experiment()
        if latest:
            print(f"自动使用最新实验: {latest}")
            loss_file = f"sae_runs/{latest}/logs/loss_history.jsonl"
            plot_args.extend(["--loss_file", loss_file])
        else:
            print("错误: 未找到实验目录，请指定 exp_dir 或 --file")
            return

    # 移除空字符串
    plot_args = [a for a in plot_args if a]

    # 调用主绘图函数
    sys.argv = ["plot_loss.py"] + plot_args
    plot_main()


if __name__ == "__main__":
    main()
