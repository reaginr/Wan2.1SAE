from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from wan.sae.path_utils import resolve_path, resolve_dir


@dataclass(frozen=True)
class SAERunLocator:
    run_dir: str
    hook_mode: str
    layer_idx: int

    def key(self) -> str:
        return f"{self.hook_mode}.layer{self.layer_idx}"

    def artifact_dir(self) -> Path:
        return resolve_path(self.run_dir) / self.key()

    def config_path(self) -> Path:
        return self.artifact_dir() / "sae_config.json"

    def latest_ckpt_path(self) -> Path:
        return self.artifact_dir() / "sae_latest.pt"

    def ckpt_path(self, step: int) -> Path:
        return self.artifact_dir() / f"sae_step{step}.pt"


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def train_state_path(run_dir: str) -> Path:
    """
    全局训练状态（step、参数快照等）保存位置。
    """
    return resolve_path(run_dir) / "train_state.json"

