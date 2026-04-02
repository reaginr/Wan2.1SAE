
Put all your models (Wan2.1-T2V-1.3B, Wan2.1-T2V-14B, Wan2.1-I2V-14B-480P, Wan2.1-I2V-14B-720P) in a folder and specify the max GPU number you want to use.

```bash
bash ./test.sh <local model dir> <gpu number>
```

学校服务器
```
bash ./test.sh /home/hhb/Wan2.1-T2V-1.3B
```

英博云
```
cd ~/Wan2.1SAE
bash ~/Wan2.1-main/Wan2.1-main/tests/test.sh ~/Wan 1
python ~/Wan2.1-main/Wan2.1-main/generate.py \
--task t2v-1.3B \
--size 480*832 \
--ckpt_dir /root/Wan/Wan2.1-T2V-1.3B
```
**「后台挂机执行 / 断开SSH依然继续运行」**
# 一、用 screen（最稳、最简单，推荐）
## 1. 先安装（如果没装）
```bash
apt install screen -y
```

## 2. 创建一个后台会话（名字叫 wan）
```bash
screen -S wan
```

## 3. 在里面正常运行你的脚本
```bash
cd /root/Wan2.1-main/Wan2.1-main
./test.sh   # 你原来的测试命令
```

## 4. **断开SSH，让它后台跑**
按快捷键：
```
Ctrl + A   然后按  D
```
✅ 就会**脱离会话**，程序**继续后台跑**。

---

# 二、下次登录，如何回到会话？
```bash
screen -r wan
```

---

# 三、查看所有后台会话
```bash
screen -ls
```
---
# 四、关闭会话（不想跑了）
```bash
screen -S wan -X quit
screen -S [PID] -X quit
```
---
# ✅ 超级总结（你只需要记 3 条）
1. **创建后台**：`screen -S wan`
2. **后台运行**：`Ctrl + A + D`
3. **回到后台**：`screen -r wan`
---
## 为什么要用这个？
- **关闭终端、断网、退出SSH → 程序继续跑**
- 不会中断
- 能随时回去看日志
- 适合跑大模型、生成视频
---

你现在直接运行：
```bash
screen -L -Logfile run.log -S wan
```
然后跑你的测试脚本，再按 `Ctrl+A D`
**这样推出会话即可**

进入 screen 后，想翻历史日志：
先按：Ctrl + A → 然后松开，再按 [
你会进入 screen 的 copy 模式
然后就可以用：
↑ ↓ 上下键 翻行
PgUp / PgDn 翻页
想退出翻页模式，按 ESC

vim ~/Wan2.1-main/Wan2.1-main/wan/sae_train_t2v_1_3b.py
python ~/Wan2.1SAE/wan/sae_train_t2v_1_3b.py

查看 日志 
cat /root/Wan2.1-main/Wan2.1-main/run.log


  ## 完整修复总结

  ### 核心改动

  **1. 完全移除调度器使用** (`wan/sae_train_t2v_1_3b.py`)
  ```python
  # 之前：使用 diffusers 调度器（有边界问题）
  from diffusers import FlowMatchEulerDiscreteScheduler
  sample_scheduler = FlowMatchEulerDiscreteScheduler(...)
  sample_scheduler.set_timesteps(...)

  # 现在：手动构造时间步
  timesteps = torch.linspace(
      cfg.num_train_timesteps - 1, 0, cfg_run.sampling_steps,
      device=device, dtype=torch.long
  )

  2. 手动实现 Euler 更新 (wan/sae_train_t2v_1_3b.py)
  # 手动 Euler 更新，不依赖任何调度器
  dt = 1.0 / cfg_run.sampling_steps
  for p, z in zip(pred, latents):
      z_next = z - p * dt  # 简单的 Euler 积分
      new_latents.append(z_next)

  3. 修复 DPM 调度器边界问题 (wan/utils/fm_solvers.py)
  - 所有访问 sigmas[step_index] 的地方添加边界检查
  - 使用 step_idx = min(step_index, len(sigmas) - 1)

  4. 修复 UniPC 调度器边界问题 (wan/utils/fm_solvers_unipc.py)
  - 同样添加边界检查

  ---
  运行前必须执行

  # 清除 Python 字节码缓存（关键！）
  find /root/Wan2.1SAE -name "*.pyc" -delete
  find /root/Wan2.1SAE -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

  # 运行
  python wan/sae_train_t2v_1_3b.py

  强制进入旧有会话
  screen -D -r wan