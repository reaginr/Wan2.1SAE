from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Tuple

import torch


HookMode = Literal["self_attn", "cross_attn", "self_and_cross", "block_out"]


@dataclass(frozen=True)
class HookSpec:
    layer_idx: int
    hook_mode: HookMode

    def key(self) -> str:
        return f"{self.hook_mode}.layer{self.layer_idx}"


def _ensure_2d_tokens(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, L, C] -> [B*L, C]
    """
    if x.dim() != 3:
        raise ValueError(f"Expected [B,L,C], got shape={tuple(x.shape)}")
    b, l, c = x.shape
    return x.reshape(b * l, c)


def register_dit_hooks(
    model,
    hook_layers: List[int],
    hook_mode: HookMode,
    on_tensor: Callable[[str, torch.Tensor], None],
) -> List:
    """
    只 hook DiT（WanModel.blocks）内部模块：
    - self_attn 输出
    - cross_attn 输出
    - self+cross 输出（分别回调）
    - block 输出（整个 transformer block 的输出）

    on_tensor(key, tensor):
        key 形如:
            self_attn.layer0
            cross_attn.layer12
            block_out.layer29
        tensor: [B, L, C]
    """
    handles = []
    blocks = list(model.blocks)
    max_i = len(blocks) - 1
    for i in hook_layers:
        if i < 0 or i > max_i:
            raise ValueError(f"layer_idx out of range: {i}, valid=[0,{max_i}]")
        block = blocks[i]

        if hook_mode in ("self_attn", "self_and_cross"):
            def _make_self_cb(layer_idx: int):
                def _cb(module, inp, out):
                    on_tensor(f"self_attn.layer{layer_idx}", out)
                return _cb
            handles.append(block.self_attn.register_forward_hook(_make_self_cb(i)))

        if hook_mode in ("cross_attn", "self_and_cross"):
            def _make_cross_cb(layer_idx: int):
                def _cb(module, inp, out):
                    on_tensor(f"cross_attn.layer{layer_idx}", out)
                return _cb
            handles.append(block.cross_attn.register_forward_hook(_make_cross_cb(i)))

        if hook_mode == "block_out":
            def _make_block_cb(layer_idx: int):
                def _cb(module, inp, out):
                    on_tensor(f"block_out.layer{layer_idx}", out)
                return _cb
            handles.append(block.register_forward_hook(_make_block_cb(i)))

    return handles


def remove_hooks(handles: List) -> None:
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


def pack_hook_batch(
    raw: Dict[str, torch.Tensor],
    max_tokens_per_key: int | None = None,
) -> Dict[str, torch.Tensor]:
    """
    把 hook 收集到的 [B,L,C] 统一变成 [N,C]，并可选截断 token 数。
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        t = _ensure_2d_tokens(v).detach()
        if max_tokens_per_key is not None and t.size(0) > max_tokens_per_key:
            t = t[:max_tokens_per_key]
        out[k] = t
    return out

