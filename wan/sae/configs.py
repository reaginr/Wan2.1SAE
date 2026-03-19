from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from wan.modules.sae_new import SAEConfig
from wan.sae.prompt_io import PromptCleanConfig


HookMode = Literal["self_attn", "cross_attn", "self_and_cross", "block_out"]
Solver = Literal["unipc", "dpm++"]


@dataclass
class WanT2V13BConfig:
    """
    Wan2.1 文生视频（T2V）1.3B 的默认参数集合。

    说明：
    - 这些参数在仓库 `wan/configs/wan_t2v_1_3B.py` 中已经存在；
      这里把它们“显式写入配置”，用于统一管理与记录实验配置。
    - 训练/测试脚本运行时仍会实例化 `WanT2V(config=t2v_1_3B, ...)`；
      本配置更多用于：默认值、可复现实验记录、文档化、以及未来扩展为可替换配置源。
    """

    # -------- 模型识别信息 --------
    model_name: str = "wan2.1_t2v_1.3B"

    # -------- VAE（latent 时空下采样）--------
    # VAE checkpoint 文件名（通常位于 checkpoint_dir 下）
    vae_checkpoint: str = "Wan2.1_VAE.pth"
    # VAE stride: (time, height, width)
    vae_stride_t: int = 4
    vae_stride_h: int = 8
    vae_stride_w: int = 8

    # -------- T5 文本编码器 --------
    # T5 encoder checkpoint 文件名
    t5_checkpoint: str = "models_t5_umt5-xxl-enc-bf16.pth"
    # tokenizer 路径（相对 checkpoint_dir 或 HF id，取决于仓库加载逻辑）
    t5_tokenizer: str = "google/umt5-xxl"
    # 文本 token 上限长度
    text_len: int = 512

    # -------- DiT 主干（WanModel）--------
    # patch_size: (t_patch, h_patch, w_patch)
    patch_size_t: int = 1
    patch_size_h: int = 2
    patch_size_w: int = 2
    # transformer dim（SAE 的 d_model 应与此一致）
    dim: int = 1536
    # FFN 隐层维度
    ffn_dim: int = 8960
    # time embedding 频率维度
    freq_dim: int = 256
    # 注意力头数与层数
    num_heads: int = 12
    num_layers: int = 30
    # window size（-1 表示全局）
    window_size_h: int = -1
    window_size_w: int = -1
    # qk_norm / cross_attn_norm
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6

    # -------- diffusion / inference 默认 --------
    num_train_timesteps: int = 1000
    sample_fps: int = 16
    # 默认负提示词（来自 shared_config.py），此处不重复粘贴超长文本；
    # 训练脚本仍会从 wrapper.sample_neg_prompt 拿到完整字符串。
    sample_neg_prompt_from_repo: bool = True


@dataclass
class HookConfig:
    """
    DiT hook 配置：
    - 只对 `WanModel.blocks` 内部进行 hook
    - layer_idx 采用 0-based：layer 0 即第 0 个 block 的输出
    """

    hook_mode: HookMode = "block_out"
    # 例如 [0, 15, 29]
    hook_layers: List[int] = field(default_factory=lambda: [29])
    # 每个 hook key（hook_type+layer）最多使用多少 token 参与训练/分析
    max_tokens_per_key: int = 65536


@dataclass
class CheckpointConfig:
    """
    checkpoint 与恢复相关配置。
    """

    # 输出目录：会在其下按 {hook_type}.layer{idx}/ 保存 SAE 配置与权重
    run_dir: str = "sae_runs/run1"
    # 每多少 step 保存一次 sae_step{step}.pt
    save_every: int = 200
    # 是否从已有状态恢复
    resume: bool = False
    # 若为 True，则从 run_dir 下的 train_state.json 读取 step，并加载每个 key 的 sae_latest.pt
    # 恢复后会从下一步继续训练（跳过已完成的 prompt batch）


@dataclass
class MemoryConfig:
    """
    显存/性能相关配置。
    """

    # 是否在每个 step 后把 T5 encoder 放回 CPU，节省显存
    offload_text_encoder: bool = False
    # 每多少 step 调一次 torch.cuda.empty_cache()：
    # 0 表示不调用；建议仅在显存碎片或长跑训练时开启
    empty_cache_every: int = 0


@dataclass
class TrainConfig:
    """
    SAE 训练配置（Wan 1.3B T2V）。

    训练要点：
    - 对每个 prompt batch，跑完整 diffusion timesteps（sampling_steps），但不做 VAE.decode
    - 每个 time step 立即训练 SAE，并释放临时张量
    """

    # -------- 路径与设备 --------
    checkpoint_dir: str = ""  # Wan 1.3B 权重目录（必填）
    prompt_dir: str = ""  # prompt txt 目录（必填）
    device_id: int = 0
    seed: int = 0

    # -------- Wan2.1 相关默认参数（显式）--------
    wan: WanT2V13BConfig = field(default_factory=WanT2V13BConfig)

    # -------- prompt 清洗 --------
    prompt_clean: PromptCleanConfig = field(default_factory=PromptCleanConfig)
    # 最多加载多少条 prompt（上限）
    max_prompts: int = 2000
    # 每步用多少条 prompt（n）
    batch_prompts: int = 4

    # -------- hook --------
    hook: HookConfig = field(default_factory=HookConfig)

    # -------- SAE --------
    sae: SAEConfig = field(default_factory=lambda: SAEConfig(d_model=1536, d_hidden=6144))

    # -------- 采样/训练循环 --------
    # 外层训练步数（一个 step 对应一批 prompt 完整采样）
    steps: int = 2000
    # 每个 step 内的 diffusion 时间步数（建议 30-50）
    sampling_steps: int = 30
    sample_solver: Solver = "unipc"
    shift: float = 5.0

    # 是否启用 CFG（更贴近真实生成，但每步 DiT forward 翻倍）
    use_cfg: bool = False
    guide_scale: float = 5.0
    # 负提示词；空表示使用 repo 默认 sample_neg_prompt
    negative_prompt: str = ""

    # -------- checkpoint / resume / memory --------
    ckpt: CheckpointConfig = field(default_factory=CheckpointConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # -------- 生成设置（用于确定 latent shape/seq_len；训练不 decode）--------
    size_w: int = 1280
    size_h: int = 720
    frame_num: int = 81


@dataclass
class TestConfig:
    """
    SAE 测试/推理配置（Wan 1.3B T2V）。

    用途：
    - 批量 prompt 输入
    - hook 指定层
    - 自动按 hook_type + layer_idx 定位 run_dir 下的 SAE 配置与权重
    - 导出 SAE 中间特征（如 z_mean、topk idx/val）
    """

    checkpoint_dir: str = ""  # Wan 1.3B 权重目录（必填）
    prompt_dir: str = ""  # prompt txt 目录（必填）
    run_dir: str = ""  # 训练输出目录（必填）
    output_path: str = "sae_test_out.pt"

    device_id: int = 0
    seed: int = 0

    wan: WanT2V13BConfig = field(default_factory=WanT2V13BConfig)
    prompt_clean: PromptCleanConfig = field(default_factory=PromptCleanConfig)

    # 最多测试多少条 prompt
    max_prompts: int = 500
    batch_prompts: int = 4

    hook: HookConfig = field(default_factory=HookConfig)

    # 生成设置（用于确定 latent shape/seq_len；测试默认单次 forward）
    size_w: int = 1280
    size_h: int = 720
    frame_num: int = 81


def _as_plain(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _as_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_as_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _as_plain(v) for k, v in obj.items()}
    return obj


def save_config(path: str, cfg: TrainConfig | TestConfig) -> None:
    """
    保存为 JSON（带缩进，便于人工编辑）。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_as_plain(cfg), ensure_ascii=False, indent=2), encoding="utf-8")


def load_train_config(path: str) -> TrainConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrainConfig(
        checkpoint_dir=data.get("checkpoint_dir", ""),
        prompt_dir=data.get("prompt_dir", ""),
        device_id=int(data.get("device_id", 0)),
        seed=int(data.get("seed", 0)),
        wan=WanT2V13BConfig(**data.get("wan", {})),
        prompt_clean=PromptCleanConfig(**data.get("prompt_clean", {})),
        max_prompts=int(data.get("max_prompts", 2000)),
        batch_prompts=int(data.get("batch_prompts", 4)),
        hook=HookConfig(**data.get("hook", {})),
        sae=SAEConfig(**data.get("sae", {})),
        steps=int(data.get("steps", 2000)),
        sampling_steps=int(data.get("sampling_steps", 30)),
        sample_solver=data.get("sample_solver", "unipc"),
        shift=float(data.get("shift", 5.0)),
        use_cfg=bool(data.get("use_cfg", False)),
        guide_scale=float(data.get("guide_scale", 5.0)),
        negative_prompt=data.get("negative_prompt", ""),
        ckpt=CheckpointConfig(**data.get("ckpt", {})),
        memory=MemoryConfig(**data.get("memory", {})),
        size_w=int(data.get("size_w", 1280)),
        size_h=int(data.get("size_h", 720)),
        frame_num=int(data.get("frame_num", 81)),
    )


def load_test_config(path: str) -> TestConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TestConfig(
        checkpoint_dir=data.get("checkpoint_dir", ""),
        prompt_dir=data.get("prompt_dir", ""),
        run_dir=data.get("run_dir", ""),
        output_path=data.get("output_path", "sae_test_out.pt"),
        device_id=int(data.get("device_id", 0)),
        seed=int(data.get("seed", 0)),
        wan=WanT2V13BConfig(**data.get("wan", {})),
        prompt_clean=PromptCleanConfig(**data.get("prompt_clean", {})),
        max_prompts=int(data.get("max_prompts", 500)),
        batch_prompts=int(data.get("batch_prompts", 4)),
        hook=HookConfig(**data.get("hook", {})),
        size_w=int(data.get("size_w", 1280)),
        size_h=int(data.get("size_h", 720)),
        frame_num=int(data.get("frame_num", 81)),
    )

