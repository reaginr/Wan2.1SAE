
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
cd ~/Wan2.1-main/Wan2.1-main
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
screen -S wan
```
然后跑你的测试脚本，再按 `Ctrl+A D`
**这样推出会话即可**

vim ~/Wan2.1-main/Wan2.1-main/wan/sae_train_t2v_1_3b.py
python ~/Wan2.1-main/Wan2.1-main/wan/sae_train_t2v_1_3b.py