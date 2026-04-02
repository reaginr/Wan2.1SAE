"""
SAE离线训练模块

功能：
1. activation_collector.py - 在线采集并保存激活值
2. train_offline.py - 从激活值文件离线训练SAE
3. test_offline.py - 离线测试SAE性能

使用流程：
1. 先用activation_collector.py采集激活值
2. 再用train_offline.py训练SAE
3. 最后用test_offline.py测试SAE

示例：
    # 1. 采集激活值
    python -m wan.sae.offline_training.activation_collector \
        --checkpoint_dir ./Wan2.1-T2V-1.3B \
        --prompt_dir ./prompts \
        --output_dir offline_data/run1 \
        --hook_layers "15,29"

    # 2. 离线训练
    python -m wan.sae.offline_training.train_offline \
        --data_dir offline_data/run1 \
        --run_dir sae_runs/offline_exp1 \
        --epochs 10 \
        --batch_size 4096

    # 3. 离线测试
    python -m wan.sae.offline_training.test_offline \
        --data_dir offline_data/run1 \
        --run_dir sae_runs/offline_exp1 \
        --hook_layers "15,29"
"""

from .activation_collector import ActivationCollector, save_activations_batch
from .train_offline import train_sae_for_layer, ActivationDataset
from .test_offline import test_sae_layer, OfflineActivationLoader

__all__ = [
    "ActivationCollector",
    "save_activations_batch",
    "train_sae_for_layer",
    "ActivationDataset",
    "test_sae_layer",
    "OfflineActivationLoader",
]
