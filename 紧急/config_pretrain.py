"""
预训练阶段配置

目的: 快速收敛到合理状态，验证超参数有效性
特点: 中等步数、中等数据、轻量预热、启用 EMA

使用:
    python run_train_layer_specific.py --config 紧急/config_pretrain.py

作者: Claude
日期: 2026-05-17
"""

from config import (
    PATH_PARAMS,
    TIMESTEP_PARAMS,
    SAE_PARAMS,
    TRAINING_PARAMS,
    SAMPLING_PARAMS,
    VALIDATION_PARAMS,
)

# ============================================================================
# 预训练阶段配置 (覆盖默认值)
# ============================================================================

# ========== 路径配置 ==========
PRETRAIN_PATH_PARAMS = {
    "model_path": PATH_PARAMS.model_path,
    "sae_init_dir": PATH_PARAMS.sae_init_dir,
    "prompt_file": "./初始化/final_clean_prompts.txt",
    "run_dir": "sae_runs/pretrain",
    "log_file": "./logs/pretrain.log",
}

# ========== 训练配置 ==========
PRETRAIN_TRAINING_PARAMS = {
    # 训练步数: 中等
    "steps": 2000,

    # 预热步数: 轻量 (5%)
    "warmup_steps": 100,

    # 批量配置
    "batch_size": 4,
    "accum_steps": 8,
    "effective_batch": 32,  # 4 × 8

    # 学习率
    "lr": 6e-5,
    "min_lr": 1e-5,

    # 正则化
    "weight_decay": 0.0,  # 必须为 0
    "grad_clip": 0.3,

    # EMA: 启用
    "use_ema": True,
    "ema_decay": 0.999,

    # Adam
    "betas": (0.95, 0.999),
}

# ========== 采样配置 ==========
PRETRAIN_SAMPLING_PARAMS = {
    # 每个 prompt 采样的 timestep 数
    "num_timesteps_per_prompt": 5,

    # 每 timestep 保留的 token 数
    "tokens_per_timestep": 1536,

    # DiT 扩散步数
    "sampling_steps": 30,

    # stride 固定为 1
    "spatial_stride": 1,
}

# ========== 验证配置 ==========
PRETRAIN_VALIDATION_PARAMS = {
    # 验证间隔
    "val_interval": 200,

    # Checkpoint 保存间隔
    "checkpoint_interval": 500,

    # 筛选条件
    "min_cohen_d": 1.0,
    "min_activation_freq": 0.01,
}

# ========== 数据配置 ==========
PRETRAIN_DATA_PARAMS = {
    # 提示词数量: 中等
    "max_prompts": 50,

    # 随机种子
    "seed": 42,
}

# ========== 日志配置 ==========
PRETRAIN_LOG_PARAMS = {
    # 日志间隔
    "log_interval": 10,

    # 进度条
    "show_progress": True,

    # 详细输出
    "verbose": False,
}

# ============================================================================
# 合并配置
# ============================================================================

def get_pretrain_config():
    """获取预训练阶段的完整配置"""
    return {
        # 路径
        "model_path": PRETRAIN_PATH_PARAMS["model_path"],
        "sae_init_dir": PRETRAIN_PATH_PARAMS["sae_init_dir"],
        "prompt_file": PRETRAIN_PATH_PARAMS["prompt_file"],
        "run_dir": PRETRAIN_PATH_PARAMS["run_dir"],
        "log_file": PRETRAIN_PATH_PARAMS["log_file"],

        # 训练
        "steps": PRETRAIN_TRAINING_PARAMS["steps"],
        "warmup_steps": PRETRAIN_TRAINING_PARAMS["warmup_steps"],
        "batch_size": PRETRAIN_TRAINING_PARAMS["batch_size"],
        "accum_steps": PRETRAIN_TRAINING_PARAMS["accum_steps"],
        "lr": PRETRAIN_TRAINING_PARAMS["lr"],
        "min_lr": PRETRAIN_TRAINING_PARAMS["min_lr"],
        "weight_decay": PRETRAIN_TRAINING_PARAMS["weight_decay"],
        "grad_clip": PRETRAIN_TRAINING_PARAMS["grad_clip"],
        "use_ema": PRETRAIN_TRAINING_PARAMS["use_ema"],
        "ema_decay": PRETRAIN_TRAINING_PARAMS["ema_decay"],
        "betas": PRETRAIN_TRAINING_PARAMS["betas"],

        # 采样
        "num_timesteps_per_prompt": PRETRAIN_SAMPLING_PARAMS["num_timesteps_per_prompt"],
        "tokens_per_timestep": PRETRAIN_SAMPLING_PARAMS["tokens_per_timestep"],
        "sampling_steps": PRETRAIN_SAMPLING_PARAMS["sampling_steps"],

        # 验证
        "val_interval": PRETRAIN_VALIDATION_PARAMS["val_interval"],
        "checkpoint_interval": PRETRAIN_VALIDATION_PARAMS["checkpoint_interval"],

        # 数据
        "max_prompts": PRETRAIN_DATA_PARAMS["max_prompts"],
        "seed": PRETRAIN_DATA_PARAMS["seed"],

        # 日志
        "log_interval": PRETRAIN_LOG_PARAMS["log_interval"],
        "verbose": PRETRAIN_LOG_PARAMS["verbose"],

        # SAE 结构 (继承)
        "d_model": SAE_PARAMS.d_model,
        "d_hidden": SAE_PARAMS.d_hidden,
        "top_k": SAE_PARAMS.top_k,
        "hook_layers": SAE_PARAMS.hook_layers,
    }


# ============================================================================
# 打印配置
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("预训练阶段配置")
    print("=" * 70)

    print("\n[训练配置]")
    print(f"  steps: {PRETRAIN_TRAINING_PARAMS['steps']}")
    print(f"  warmup_steps: {PRETRAIN_TRAINING_PARAMS['warmup_steps']} ({PRETRAIN_TRAINING_PARAMS['warmup_steps']/PRETRAIN_TRAINING_PARAMS['steps']*100:.0f}%)")
    print(f"  batch_size: {PRETRAIN_TRAINING_PARAMS['batch_size']}")
    print(f"  accum_steps: {PRETRAIN_TRAINING_PARAMS['accum_steps']}")
    print(f"  effective_batch: {PRETRAIN_TRAINING_PARAMS['effective_batch']}")
    print(f"  lr: {PRETRAIN_TRAINING_PARAMS['lr']}")
    print(f"  use_ema: {PRETRAIN_TRAINING_PARAMS['use_ema']}")

    print("\n[采样配置]")
    print(f"  num_timesteps_per_prompt: {PRETRAIN_SAMPLING_PARAMS['num_timesteps_per_prompt']}")
    print(f"  tokens_per_timestep: {PRETRAIN_SAMPLING_PARAMS['tokens_per_timestep']}")

    print("\n[数据配置]")
    print(f"  max_prompts: {PRETRAIN_DATA_PARAMS['max_prompts']}")

    print("\n[验证配置]")
    print(f"  val_interval: {PRETRAIN_VALIDATION_PARAMS['val_interval']}")
    print(f"  checkpoint_interval: {PRETRAIN_VALIDATION_PARAMS['checkpoint_interval']}")

    # 预估时间
    n_prompts = PRETRAIN_DATA_PARAMS['max_prompts']
    n_timesteps = PRETRAIN_SAMPLING_PARAMS['num_timesteps_per_prompt']
    estimated_activation_time = n_prompts * 3  # 约3分钟/prompt
    estimated_train_time = PRETRAIN_TRAINING_PARAMS['steps'] * 0.5 / 60  # 约0.5秒/step

    print("\n[预估时间]")
    print(f"  激活提取: ~{estimated_activation_time:.0f} 分钟")
    print(f"  训练: ~{estimated_train_time:.0f} 分钟")
    print(f"  总计: ~{estimated_activation_time + estimated_train_time:.0f} 分钟")

    print("=" * 70)
