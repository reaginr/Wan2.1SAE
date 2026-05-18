"""
参数测试阶段配置

目的: 验证代码正确性、采样策略有效性、训练收敛趋势
特点: 中等数据量、中等步数、频繁验证、无预热

使用:
    python run_train_layer_specific.py --config 紧急/config_test.py

作者: Claude
日期: 2026-05-18
"""

from config import (
    PATH_PARAMS,
    TIMESTEP_PARAMS,
    SAE_PARAMS,
    TRAINING_PARAMS,
    SAMPLING_PARAMS,
    VALIDATION_PARAMS,
    PARAM_TEST_SAMPLING_PARAMS,
)

# ============================================================================
# 参数测试阶段配置 (覆盖默认值)
# ============================================================================

# ========== 路径配置 ==========
# 提示词文件 (统一来源)
TEST_PATH_PARAMS = {
    "prompt_file": "./初始化/final_clean_prompts.txt",
}

# ========== 训练配置 ==========
TEST_TRAINING_PARAMS = {
    # 训练步数: 每层 100 步
    "steps": 100,

    # 预热步数: 不需要 (步数太少)
    "warmup_steps": 0,

    # 批量配置
    "batch_size": 4,
    "accum_steps": 4,  # 减少累积，加快反馈
    "effective_batch": 16,  # 4 × 4

    # 学习率: 与正式训练一致
    "lr": 6e-5,
    "min_lr": 1e-5,

    # 正则化
    "weight_decay": 0.0,
    "grad_clip": 0.3,

    # EMA: 不需要 (步数太少)
    "use_ema": False,
    "ema_decay": 0.999,

    # Adam
    "betas": (0.95, 0.999),
}

# ========== 采样配置 (基础) ==========
TEST_SAMPLING_PARAMS = {
    # 每个 prompt 采样的 timestep 数
    "num_timesteps_per_prompt": 5,

    # 每 timestep 保留的 token 数
    "tokens_per_timestep": 1024,

    # DiT 扩散步数
    "sampling_steps": 30,

    # stride 固定为 1
    "spatial_stride": 1,
}

# ========== 参数测试阶段采样配置 (新增) ==========
# 使用 'param_test' 模式的采样策略
# 特点: 时空局部性 + soft norm bias + mild decorrelation
TEST_PARAM_TEST_SAMPLING_PARAMS = {
    # 采样模式
    "sampling_mode": "param_test",

    # Timestep 采样
    "num_timesteps_per_prompt": 5,

    # Temporal Chunk: 采样连续 2 帧
    "temporal_chunk_size": 2,
    "temporal_frames_total": 11,

    # Spatial Block: 8×8 patch
    "spatial_block_size": 8,
    "spatial_grid_h": 30,
    "spatial_grid_w": 52,
    "num_spatial_blocks": 24,

    # Norm Bias: 轻度偏向高 norm token
    "norm_bias_enabled": True,
    "norm_bias_strength": 0.3,

    # Mild Decorrelation
    "decorrelation_enabled": True,
    "decorrelation_threshold": 0.7,

    # 目标 token 数
    "tokens_per_timestep": 1024,
}

# ========== 验证配置 ==========
TEST_VALIDATION_PARAMS = {
    # 频繁验证，观察趋势
    "val_interval": 20,

    # 频繁保存
    "checkpoint_interval": 50,

    # 筛选条件
    "min_cohen_d": 1.0,
    "min_activation_freq": 0.01,
}

# ========== 数据配置 ==========
TEST_DATA_PARAMS = {
    # 提示词数量: 100 条
    "max_prompts": 100,

    # 随机种子
    "seed": 42,
}

# ========== 日志配置 ==========
TEST_LOG_PARAMS = {
    # 每 5 步打印一次
    "log_interval": 5,

    # 进度条
    "show_progress": True,

    # 详细输出
    "verbose": True,
}

# ========== 输出目录 ==========
TEST_OUTPUT_PARAMS = {
    "run_dir": "sae_runs/test_params",
    "log_file": "./logs/test_params.log",
}

# ============================================================================
# 合并配置
# ============================================================================

def get_test_config():
    """获取参数测试阶段的完整配置"""
    return {
        # 路径
        "model_path": PATH_PARAMS.model_path,
        "sae_init_dir": PATH_PARAMS.sae_init_dir,
        "prompt_file": TEST_PATH_PARAMS["prompt_file"],
        "run_dir": TEST_OUTPUT_PARAMS["run_dir"],
        "log_file": TEST_OUTPUT_PARAMS["log_file"],

        # 训练
        "steps": TEST_TRAINING_PARAMS["steps"],
        "warmup_steps": TEST_TRAINING_PARAMS["warmup_steps"],
        "batch_size": TEST_TRAINING_PARAMS["batch_size"],
        "accum_steps": TEST_TRAINING_PARAMS["accum_steps"],
        "lr": TEST_TRAINING_PARAMS["lr"],
        "min_lr": TEST_TRAINING_PARAMS["min_lr"],
        "weight_decay": TEST_TRAINING_PARAMS["weight_decay"],
        "grad_clip": TEST_TRAINING_PARAMS["grad_clip"],
        "use_ema": TEST_TRAINING_PARAMS["use_ema"],
        "ema_decay": TEST_TRAINING_PARAMS["ema_decay"],
        "betas": TEST_TRAINING_PARAMS["betas"],

        # 采样 (基础)
        "num_timesteps_per_prompt": TEST_SAMPLING_PARAMS["num_timesteps_per_prompt"],
        "tokens_per_timestep": TEST_SAMPLING_PARAMS["tokens_per_timestep"],
        "sampling_steps": TEST_SAMPLING_PARAMS["sampling_steps"],

        # 参数测试阶段采样 (新增)
        "sampling_mode": TEST_PARAM_TEST_SAMPLING_PARAMS["sampling_mode"],
        "temporal_chunk_size": TEST_PARAM_TEST_SAMPLING_PARAMS["temporal_chunk_size"],
        "spatial_block_size": TEST_PARAM_TEST_SAMPLING_PARAMS["spatial_block_size"],
        "num_spatial_blocks": TEST_PARAM_TEST_SAMPLING_PARAMS["num_spatial_blocks"],
        "norm_bias_enabled": TEST_PARAM_TEST_SAMPLING_PARAMS["norm_bias_enabled"],
        "norm_bias_strength": TEST_PARAM_TEST_SAMPLING_PARAMS["norm_bias_strength"],
        "decorrelation_enabled": TEST_PARAM_TEST_SAMPLING_PARAMS["decorrelation_enabled"],
        "decorrelation_threshold": TEST_PARAM_TEST_SAMPLING_PARAMS["decorrelation_threshold"],

        # 验证
        "val_interval": TEST_VALIDATION_PARAMS["val_interval"],
        "checkpoint_interval": TEST_VALIDATION_PARAMS["checkpoint_interval"],

        # 数据
        "max_prompts": TEST_DATA_PARAMS["max_prompts"],
        "seed": TEST_DATA_PARAMS["seed"],

        # 日志
        "log_interval": TEST_LOG_PARAMS["log_interval"],
        "verbose": TEST_LOG_PARAMS["verbose"],

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
    print("参数测试阶段配置")
    print("=" * 70)

    print("\n[路径配置]")
    print(f"  prompt_file: {TEST_PATH_PARAMS['prompt_file']}")

    print("\n[训练配置]")
    print(f"  steps: {TEST_TRAINING_PARAMS['steps']} (每层)")
    print(f"  warmup_steps: {TEST_TRAINING_PARAMS['warmup_steps']}")
    print(f"  batch_size: {TEST_TRAINING_PARAMS['batch_size']}")
    print(f"  accum_steps: {TEST_TRAINING_PARAMS['accum_steps']}")
    print(f"  lr: {TEST_TRAINING_PARAMS['lr']}")
    print(f"  use_ema: {TEST_TRAINING_PARAMS['use_ema']}")

    print("\n[采样配置 (基础)]")
    print(f"  num_timesteps_per_prompt: {TEST_SAMPLING_PARAMS['num_timesteps_per_prompt']}")
    print(f"  tokens_per_timestep: {TEST_SAMPLING_PARAMS['tokens_per_timestep']}")

    print("\n[参数测试阶段采样配置 (新增)]")
    print(f"  sampling_mode: {TEST_PARAM_TEST_SAMPLING_PARAMS['sampling_mode']}")
    print(f"  temporal_chunk_size: {TEST_PARAM_TEST_SAMPLING_PARAMS['temporal_chunk_size']}")
    print(f"  spatial_block_size: {TEST_PARAM_TEST_SAMPLING_PARAMS['spatial_block_size']}x{TEST_PARAM_TEST_SAMPLING_PARAMS['spatial_block_size']}")
    print(f"  num_spatial_blocks: {TEST_PARAM_TEST_SAMPLING_PARAMS['num_spatial_blocks']}")
    print(f"  norm_bias_enabled: {TEST_PARAM_TEST_SAMPLING_PARAMS['norm_bias_enabled']}")
    print(f"  norm_bias_strength: {TEST_PARAM_TEST_SAMPLING_PARAMS['norm_bias_strength']}")
    print(f"  decorrelation_enabled: {TEST_PARAM_TEST_SAMPLING_PARAMS['decorrelation_enabled']}")
    print(f"  decorrelation_threshold: {TEST_PARAM_TEST_SAMPLING_PARAMS['decorrelation_threshold']}")

    print("\n[数据配置]")
    print(f"  max_prompts: {TEST_DATA_PARAMS['max_prompts']}")
    print(f"  训练层: 14, 19, 24, 29 (全部 4 层)")

    print("\n[验证配置]")
    print(f"  val_interval: {TEST_VALIDATION_PARAMS['val_interval']}")
    print(f"  checkpoint_interval: {TEST_VALIDATION_PARAMS['checkpoint_interval']}")

    # 预估
    print("\n[预估资源]")
    total_records = TEST_DATA_PARAMS['max_prompts'] * 4 * TEST_SAMPLING_PARAMS['num_timesteps_per_prompt']
    print(f"  激活记录数: {total_records} 条")
    print(f"  激活提取时间: ~{TEST_DATA_PARAMS['max_prompts'] * 3} 分钟")
    print(f"  训练时间 (4层×100步): ~{4 * 100 * 0.5 / 60:.0f} 分钟")
    print(f"  总时间: ~{(TEST_DATA_PARAMS['max_prompts'] * 3 + 4 * 100 * 0.5 / 60):.0f} 分钟")

    print("=" * 70)
