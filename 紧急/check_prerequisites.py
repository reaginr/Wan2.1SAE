"""
快速启动脚本 - 检查环境并生成运行命令

作者：Claude
日期：2026-05-11
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def check_file(path: str, description: str) -> bool:
    """检查文件是否存在"""
    if Path(path).exists():
        print(f"[OK] {description}: {path}")
        return True
    else:
        print(f"[MISSING] {description}: {path}")
        return False


def check_directory(path: str, description: str) -> bool:
    """检查目录是否存在"""
    if Path(path).is_dir():
        files = list(Path(path).glob("*.txt"))
        print(f"[OK] {description}: {path} ({len(files)} files)")
        return True
    else:
        print(f"[MISSING] {description}: {path}")
        return False


def main():
    print("="*70)
    print("Layer29 风险概念提取 Pipeline - 环境检查")
    print("="*70)

    print("\n[1] 检查必要文件...")

    checks = []

    # 检查 Wan 模型
    model_path = input("\n输入 Wan 模型路径 (默认: ./Wan2.1-T2V-1.3B): ").strip()
    if not model_path:
        model_path = "./Wan2.1-T2V-1.3B"
    checks.append(check_directory(model_path, "Wan 模型"))

    # 检查 SAE checkpoint
    sae_path = input("输入 Layer29 SAE checkpoint 路径: ").strip()
    if sae_path:
        checks.append(check_file(sae_path, "SAE checkpoint"))

    # 检查数据集
    prompt_dir = input("输入 prompt 数据集目录 (默认: ./datasets): ").strip()
    if not prompt_dir:
        prompt_dir = "./datasets"

    print(f"\n[2] 检查数据集目录: {prompt_dir}")

    required_files = [
        "sex_positive.txt",
        "sex_negative.txt",
        "violence_positive.txt",
        "violence_negative.txt",
    ]

    for f in required_files:
        checks.append(check_file(f"{prompt_dir}/{f}", f"数据集"))

    # 检查可选文件
    optional_files = [
        "clean_prompts.txt",
    ]

    print("\n[3] 检查可选文件...")
    for f in optional_files:
        check_file(f"{prompt_dir}/{f}", f"可选数据")

    # 输出目录
    output_dir = input("\n输入输出目录 (默认: ./outputs): ").strip()
    if not output_dir:
        output_dir = "./outputs"

    # 生成命令
    print("\n" + "="*70)
    print("生成的运行命令")
    print("="*70)

    if all(checks):
        print("\n[所有必要文件已就绪]\n")

        cmd = f"""# 完整 Pipeline 运行命令
python run_pipeline.py \\
    --model_path {model_path} \\
    --sae_checkpoint {sae_path} \\
    --prompt_dir {prompt_dir} \\
    --output_dir {output_dir} \\
    --concepts sex,violence

# 或者使用 nohup 后台运行
nohup python run_pipeline.py \\
    --model_path {model_path} \\
    --sae_checkpoint {sae_path} \\
    --prompt_dir {prompt_dir} \\
    --output_dir {output_dir} \\
    > pipeline.log 2>&1 &
"""
        print(cmd)
    else:
        print("\n[部分必要文件缺失，请先准备数据]\n")

        print("""
# 数据准备建议

1. 准备 prompt 数据集：
   - sex_positive.txt: 50~100 个性相关风险 prompt
   - sex_negative.txt: 50~100 个对照组
   - violence_positive.txt: 50~100 个暴力相关 prompt
   - violence_negative.txt: 50~100 个对照组

2. 格式要求：
   - 每行一个 prompt
   - UTF-8 编码
   - 至少 8 个字符

3. 数据要求：
   - 正负样本语义接近
   - 只改变风险属性
   - 长度和风格一致
""")

    print("="*70)


if __name__ == "__main__":
    main()
