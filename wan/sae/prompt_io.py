from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from wan.sae.path_utils import resolve_path


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_WHITESPACE = re.compile(r"\s+")
_ALLOWED_BASIC = re.compile(r"^[\x20-\x7E]+$")  # 可打印 ASCII
_HAS_LETTER = re.compile(r"[A-Za-z]")


@dataclass
class PromptCleanConfig:
    # 允许后续扩展：更复杂的规则 / 语言模型过滤等
    min_len: int = 8
    max_len: int = 400
    require_ascii_printable: bool = True
    require_english_letters: bool = True


def clean_prompt_line(line: str, cfg: PromptCleanConfig) -> Optional[str]:
    """
    清洗一行 prompt。
    需求背景：NSFW txt 中可能混入中文乱码、异常特殊字符、编码错误片段。

    策略（可扩展）：
    - 去掉控制字符
    - 规范空白
    - 可选：只保留可打印 ASCII
    - 可选：必须包含英文字符
    - 长度过滤
    """
    if line is None:
        return None
    s = line.strip()
    if not s:
        return None
    s = _CONTROL_CHARS.sub("", s)
    s = _WHITESPACE.sub(" ", s).strip()
    if not s:
        return None
    if len(s) < cfg.min_len or len(s) > cfg.max_len:
        return None
    if cfg.require_ascii_printable and _ALLOWED_BASIC.match(s) is None:
        return None
    if cfg.require_english_letters and _HAS_LETTER.search(s) is None:
        return None
    return s


def iter_prompt_files(prompt_dir: str) -> Iterator[Path]:
    p = resolve_path(prompt_dir)
    if not p.exists():
        raise FileNotFoundError(f"prompt_dir not found: {p}")
    for f in sorted(p.rglob("*.txt")):
        if f.is_file():
            yield f


def load_prompts_from_dir(
    prompt_dir: str,
    clean_cfg: PromptCleanConfig,
    limit: Optional[int] = None,
) -> List[str]:
    prompts: List[str] = []
    for f in iter_prompt_files(prompt_dir):
        try:
            # 用 errors='ignore' 兜底处理编码错误行
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            # 最兜底：按 bytes 读再用 latin1 还原（尽量不崩）
            text = f.read_bytes().decode("latin1", errors="ignore")

        for raw in text.splitlines():
            s = clean_prompt_line(raw, clean_cfg)
            if s is None:
                continue
            prompts.append(s)
            if limit is not None and len(prompts) >= limit:
                return prompts
    return prompts


def batch_iter(items: List[str], batch_size: int, shuffle: bool = False, seed: int = 0) -> Iterable[List[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    idx = list(range(len(items)))
    if shuffle:
        import random

        rng = random.Random(seed)
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        batch = [items[j] for j in idx[i : i + batch_size]]
        if batch:
            yield batch

