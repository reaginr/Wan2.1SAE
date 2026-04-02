# SAE Checkpoint 管理指南

## 1. Checkpoint 文件结构

### 1.1 存储位置

根据你的配置 (`run_dir: "sae_runs/exp__20250324"`)，checkpoint 保存在：

```
sae_runs/exp__20250324/                    # 实验根目录
├── train_state.json                       # 全局训练状态（恢复训练必需）
├── logs/                                  # 日志目录
│   ├── training.log                       # 完整训练日志
│   ├── loss_history.jsonl                 # 每步 loss 记录（JSONL 格式）
│   └── loss_history.csv                   # 每步 loss 记录（CSV 格式）
└── block_out.layer15/                     # 每个 hook 层的独立目录
    ├── sae_config.json                    # SAE 架构配置
    ├── sae_latest.pt                      # 最新权重（软链接/复制）
    ├── sae_step50.pt                      # 第 50 步的历史版本
    ├── sae_step100.pt                     # 第 100 步的历史版本（保存时）
    └── ...
```

### 1.2 关键文件说明

| 文件 | 大小 | 用途 | 恢复训练必需 |
|------|------|------|-------------|
| `train_state.json` | ~1KB | 记录当前 step、配置参数等 | ✅ 必须 |
| `sae_latest.pt` | ~100MB | SAE 最新权重 | ✅ 必须 |
| `sae_config.json` | ~1KB | SAE 架构配置 | ✅ 必须 |
| `sae_step{step}.pt` | ~100MB | 历史版本（可选） | ❌ 可选 |
| `training.log` | 持续增长 | 完整控制台输出 | ❌ 参考 |
| `loss_history.jsonl` | 持续增长 | 每步 loss 数据 | ❌ 可视化用 |

---

## 2. 从服务器下载 Checkpoint 到本机

### 2.1 使用 scp（推荐）

假设服务器地址为 `user@server-ip`，本地路径为 `D:\Wan2.1SAE\`：

**下载整个实验目录（推荐）：**

```bash
# Windows PowerShell
scp -r user@server-ip:/root/Wan2.1SAE/sae_runs/exp__20250324 \
  "D:\Wan2.1SAE\sae_runs\"

# 或者使用 rsync（断点续传）
rsync -avz --progress user@server-ip:/root/Wan2.1SAE/sae_runs/exp__20250324 \
  "D:\Wan2.1SAE\sae_runs\"
```

**只下载关键文件（最小化）：**

```bash
# 创建本地目录
mkdir -p "D:\Wan2.1SAE\sae_runs\exp__20250324\block_out.layer15"

# 下载训练状态和配置
scp user@server-ip:/root/Wan2.1SAE/sae_runs/exp__20250324/train_state.json \
  "D:\Wan2.1SAE\sae_runs\exp__20250324\"

scp user@server-ip:/root/Wan2.1SAE/sae_runs/exp__20250324/block_out.layer15/sae_config.json \
  "D:\Wan2.1SAE\sae_runs\exp__20250324\block_out.layer15\"

# 下载最新权重
scp user@server-ip:/root/Wan2.1SAE/sae_runs/exp__20250324/block_out.layer15/sae_latest.pt \
  "D:\Wan2.1SAE\sae_runs\exp__20250324\block_out.layer15\"

# 下载 loss 历史（用于可视化）
scp user@server-ip:/root/Wan2.1SAE/sae_runs/exp__20250324/logs/loss_history.jsonl \
  "D:\Wan2.1SAE\sae_runs\exp__20250324\logs\"
```

### 2.2 使用 SFTP 工具（图形界面）

- **FileZilla** / **WinSCP** / **MobaXterm**
- 连接服务器后，导航到 `/root/Wan2.1SAE/sae_runs/exp__20250324/`
- 拖拽下载到本地目录

### 2.3 使用 Python（自动化脚本）

```python
# download_checkpoint.py
import paramiko
import os

# 配置
SERVER_HOST = "your-server-ip"
SERVER_USER = "root"
SERVER_PATH = "/root/Wan2.1SAE/sae_runs/exp__20250324"
LOCAL_PATH = r"D:\Wan2.1SAE\sae_runs\exp__20250324"

# 连接 SSH
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(SERVER_HOST, username=SERVER_USER)

# 使用 SFTP 下载
sftp = ssh.open_sftp()
os.makedirs(LOCAL_PATH, exist_ok=True)

files_to_download = [
    "train_state.json",
    "block_out.layer15/sae_config.json",
    "block_out.layer15/sae_latest.pt",
    "logs/loss_history.jsonl",
    "logs/loss_history.csv",
]

for remote_file in files_to_download:
    remote_path = f"{SERVER_PATH}/{remote_file}"
    local_file = os.path.join(LOCAL_PATH, remote_file.replace("/", os.sep))
    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    print(f"Downloading: {remote_file}...")
    sftp.get(remote_path, local_file)

sftp.close()
ssh.close()
print("Download complete!")
```

---

## 3. 恢复训练（Resume Training）

### 3.1 服务器上继续训练（推荐）

如果 checkpoint 还在服务器上，直接加 `--resume` 参数：

```bash
cd /root/Wan2.1SAE
python wan/sae_train_t2v_1_3b.py \
  --resume \
  --run_dir "sae_runs/exp__20250324" \
  --steps 500
```

**恢复训练会自动：**
- 从 `train_state.json` 读取当前 step（50）
- 从 `sae_latest.pt` 加载 SAE 权重
- 恢复优化器状态（如有）
- 继续从第 51 步训练到第 500 步

### 3.2 本机恢复训练

如果已将 checkpoint 下载到本机：

```bash
# Windows
python wan\sae_train_t2v_1_3b.py \
  --resume \
  --run_dir "sae_runs\exp__20250324" \
  --checkpoint_dir "Wan2.1-T2V-1.3B" \
  --steps 500
```

### 3.3 修改训练参数继续训练

```bash
# 增加训练步数（从 500 改为 1000）
python wan/sae_train_t2v_1_3b.py \
  --resume \
  --run_dir "sae_runs/exp__20250324" \
  --steps 1000

# 更改学习率（覆盖原配置）
python wan/sae_train_t2v_1_3b.py \
  --resume \
  --run_dir "sae_runs/exp__20250324" \
  --steps 1000
# 然后编辑代码中的 training_params["lr"] = 5e-4
```

---

## 4. Loss 可视化

### 4.1 使用 loss_history.csv

下载的 `loss_history.csv` 格式如下：

```csv
step,timestamp,block_out.layer15
1,1711366727.123,0.523456
2,1711367318.234,0.512345
...
50,1711388715.654,0.423456
```

**Python 可视化：**

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("sae_runs/exp__20250324/logs/loss_history.csv")

# 绘制 loss 曲线
plt.figure(figsize=(12, 6))
plt.plot(df["step"], df["block_out.layer15"], label="block_out.layer15")
plt.xlabel("Training Step")
plt.ylabel("SAE Loss")
plt.title("SAE Training Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png", dpi=300)
plt.show()
```

**TensorBoard 格式：**

```python
from torch.utils.tensorboard import SummaryWriter
import json

writer = SummaryWriter("runs/sae_training")

with open("sae_runs/exp__20250324/logs/loss_history.jsonl", "r") as f:
    for line in f:
        record = json.loads(line)
        step = record["step"]
        for key, loss in record["losses"].items():
            writer.add_scalar(f"Loss/{key}", loss, step)

writer.close()
```

### 4.2 实时监控 Loss

在训练过程中查看最新 loss：

```bash
# 查看最后 10 行 loss 记录
tail -n 10 sae_runs/exp__20250324/logs/loss_history.csv

# 实时追踪 loss 变化
watch -n 10 "tail -n 5 sae_runs/exp__20250324/logs/loss_history.csv"
```

---

## 5. 多实验管理

### 5.1 目录结构建议

```
sae_runs/
├── exp__20250324_block15_topk64/      # 原始实验
├── exp__20250325_block15_topk128/     # 改 top_k 参数
├── exp__20250326_block29_topk64/      # 换层
└── exp__20250327_multi_layer/         # 多层同时训练
```

### 5.2 快速切换实验

```bash
# 使用环境变量或脚本
export EXP_NAME="exp__20250324_block15_topk64"

python wan/sae_train_t2v_1_3b.py \
  --resume \
  --run_dir "sae_runs/${EXP_NAME}" \
  --steps 500
```

---

## 6. 常见问题

### Q1: 恢复训练报错 "train_state.json not found"

检查 run_dir 路径是否正确：

```bash
ls -la sae_runs/exp__20250324/train_state.json
```

### Q2: 恢复训练后 step 不对

`train_state.json` 中的 `step` 字段应该是 50，检查内容：

```bash
cat sae_runs/exp__20250324/train_state.json
```

### Q3: 想从更早的 checkpoint 恢复（如 step 30）

```python
# 手动修改 train_state.json 中的 step 为 30
# 然后复制对应的历史版本为 latest
cp sae_runs/exp__20250324/block_out.layer15/sae_step30.pt \
   sae_runs/exp__20250324/block_out.layer15/sae_latest.pt

# 再运行恢复训练
python wan/sae_train_t2v_1_3b.py --resume --run_dir "sae_runs/exp__20250324"
```

### Q4: 磁盘空间不足

删除历史版本只保留最新：

```bash
# 删除所有 step 版本，只保留 latest
rm sae_runs/exp__20250324/block_out.layer15/sae_step*.pt

# 或压缩旧版本
tar czf old_checkpoints.tar.gz sae_runs/exp__20250324/block_out.layer15/sae_step*.pt
rm sae_runs/exp__20250324/block_out.layer15/sae_step*.pt
```

---

## 7. 一键备份脚本

创建 `backup_checkpoint.sh`：

```bash
#!/bin/bash

EXP_NAME="exp__20250324"
REMOTE_USER="root"
REMOTE_HOST="your-server-ip"
REMOTE_PATH="/root/Wan2.1SAE/sae_runs/${EXP_NAME}"
LOCAL_PATH="/mnt/d/Wan2.1SAE/sae_runs/${EXP_NAME}"  # WSL 路径格式

echo "=== Backing up checkpoint from server ==="
echo "Remote: ${REMOTE_HOST}:${REMOTE_PATH}"
echo "Local: ${LOCAL_PATH}"

mkdir -p "${LOCAL_PATH}/block_out.layer15"
mkdir -p "${LOCAL_PATH}/logs"

# 下载关键文件
scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/train_state.json" "${LOCAL_PATH}/"
scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/block_out.layer15/sae_config.json" "${LOCAL_PATH}/block_out.layer15/"
scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/block_out.layer15/sae_latest.pt" "${LOCAL_PATH}/block_out.layer15/"
scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/logs/loss_history.jsonl" "${LOCAL_PATH}/logs/"

echo "=== Backup complete ==="
ls -lh "${LOCAL_PATH}/block_out.layer15/"
```

运行：

```bash
chmod +x backup_checkpoint.sh
./backup_checkpoint.sh
```
