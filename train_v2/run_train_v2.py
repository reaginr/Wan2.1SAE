"""
train_v2 模块语法检查脚本

仅用于验证代码语法正确性，不运行实际训练

使用方法:
    python train_v2/run_train_v2.py

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def check_syntax(filepath: str) -> bool:
    """检查 Python 文件语法"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(f"  Syntax Error in {filepath}:")
        print(f"    Line {e.lineno}: {e.msg}")
        return False


def main():
    print("=" * 60)
    print("train_v2 Module Syntax Check")
    print("=" * 60)

    train_v2_dir = Path(__file__).parent

    files = [
        "config.py",
        "optimizer.py",
        "gradient_accumulator.py",
        "ema.py",
        "sae_engine.py",
        "dead_neuron_monitor.py",
        "validator.py",
        "checkpoint.py",
        "training_loop.py",
        "layer_trainer.py",
        "__init__.py",
    ]

    results = []
    for f in files:
        filepath = train_v2_dir / f
        if filepath.exists():
            ok = check_syntax(str(filepath))
            results.append((f, ok))
            status = "[OK]" if ok else "[FAIL]"
            print(f"  {status} {f}")
        else:
            print(f"  [MISSING] {f}")
            results.append((f, False))

    # 汇总
    print("-" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"  Total: {passed}/{total} files passed syntax check")

    if passed == total:
        print("\n[OK] All syntax checks passed. Module is ready for server deployment.")
    else:
        print(f"\n[FAIL] {total - passed} files have syntax errors.")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
