"""
统一路径解析工具
支持：相对路径、绝对路径、~ 展开、环境变量展开、跨平台兼容
"""
from __future__ import annotations

import os
from pathlib import Path, PurePath
from typing import Union


PathLike = Union[str, Path, os.PathLike]


def resolve_path(path: PathLike, base_dir: PathLike | None = None, must_exist: bool = False) -> Path:
    """
    解析路径为绝对 Path 对象。

    支持：
    - ~ / ~user 展开为家目录
    - $ENV / ${ENV} 环境变量展开
    - 相对路径（基于 base_dir 或当前工作目录）
    - 跨平台路径分隔符自动处理

    参数:
        path: 输入路径（str 或 Path）
        base_dir: 相对路径的基准目录（默认当前工作目录）
        must_exist: 是否要求路径必须存在

    返回:
        解析后的绝对 Path 对象

    示例:
        >>> resolve_path("~/data")
        PosixPath('/home/user/data')

        >>> resolve_path("./runs/exp1", base_dir="/project")
        PosixPath('/project/runs/exp1')

        >>> resolve_path("$HOME/models")
        PosixPath('/home/user/models')
    """
    if path is None:
        raise ValueError("path cannot be None")

    # 转为字符串处理
    path_str = str(path)

    # 1. 展开环境变量 $VAR 和 ${VAR}
    path_str = os.path.expandvars(path_str)

    # 2. 展开 ~ 为用户家目录
    path_str = os.path.expanduser(path_str)

    # 3. 转为 Path 对象
    p = Path(path_str)

    # 4. 如果是相对路径，基于 base_dir 解析
    if not p.is_absolute():
        if base_dir is not None:
            base = resolve_path(base_dir, must_exist=False)
            p = base / p
        else:
            p = p.resolve()  # 基于当前工作目录

    # 5. 规范化路径（去除 . 和 ..）
    p = p.resolve()

    # 6. 检查存在性（可选）
    if must_exist and not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    return p


def resolve_dir(path: PathLike, base_dir: PathLike | None = None, create: bool = False, must_exist: bool = False) -> Path:
    """
    解析目录路径，可选自动创建。

    参数:
        path: 输入路径
        base_dir: 相对路径的基准目录
        create: 是否自动创建目录（如果不存在）
        must_exist: 是否要求必须已存在（与 create 互斥）

    返回:
        解析后的绝对 Path 对象
    """
    p = resolve_path(path, base_dir, must_exist=False)

    if create and not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    elif must_exist and not p.exists():
        raise FileNotFoundError(f"Directory does not exist: {p}")

    return p


def resolve_file(path: PathLike, base_dir: PathLike | None = None, must_exist: bool = False) -> Path:
    """
    解析文件路径。

    参数:
        path: 输入路径
        base_dir: 相对路径的基准目录
        must_exist: 是否要求文件必须存在

    返回:
        解析后的绝对 Path 对象
    """
    p = resolve_path(path, base_dir, must_exist=False)

    if must_exist and not p.is_file():
        raise FileNotFoundError(f"File does not exist: {p}")

    return p


def normalize_path_for_display(path: PathLike) -> str:
    """
    将路径标准化为可显示的字符串形式。
    优先使用相对路径（如果基于当前目录），否则使用绝对路径。
    """
    p = resolve_path(path)
    cwd = Path.cwd()

    try:
        # 尝试返回相对于当前目录的路径
        rel = p.relative_to(cwd)
        return f"./{rel}" if str(rel) != "." else "."
    except ValueError:
        # 不在当前目录下，返回绝对路径
        return str(p)


def get_project_root(marker_files: tuple = (".git", "pyproject.toml", "setup.py", "CLAUDE.md")) -> Path:
    """
    自动查找项目根目录。
    向上搜索包含标记文件的目录。

    参数:
        marker_files: 标记项目根的文件/目录名

    返回:
        项目根目录的绝对路径
    """
    current = Path.cwd().resolve()

    for path in [current] + list(current.parents):
        for marker in marker_files:
            if (path / marker).exists():
                return path

    # 如果没找到，返回当前目录
    return current


def make_relative_if_possible(path: PathLike, base: PathLike | None = None) -> str:
    """
    尝试将路径转为相对于 base 的相对路径。
    如果无法相对化，返回原路径的字符串。
    """
    p = resolve_path(path)
    base_path = resolve_path(base) if base else Path.cwd()

    try:
        rel = p.relative_to(base_path)
        return str(rel)
    except ValueError:
        return str(p)


# 向后兼容的快捷函数
def expand_path(path: PathLike) -> Path:
    """展开 ~ 和环境变量的快捷函数。"""
    return resolve_path(path)


def ensure_dir(path: PathLike) -> Path:
    """确保目录存在，如果不存在则创建。"""
    return resolve_dir(path, create=True)
