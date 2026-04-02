# Wan2.1 T2V 1.3B 模型完整架构解析（整合版 v3.0）

**版本**: v3.0 (整合第一版与修正版，完整回答所有问题)
**针对模型**: Wan2.1-T2V-1.3B
**硬件环境**: 24GB VRAM
**日期**: 2026-03-29

---

## 目录

1. [关键概念澄清（重要修正）](#第一部分关键概念澄清重要修正)
2. [T2V 1.3B 完整流程文字示意图](#第二部分t2v-13b-完整流程文字示意图)
3. [DiT 层输出的意义（新问题解答）](#第三部分dit-层输出的意义新问题解答)
4. [所有问题详细解答（Q&A）](#第四部分所有问题详细解答qa)
5. [SAE 创新应用（24GB显存优化）](#第五部分sae-创新应用24gb显存优化)

---

# 第一部分：关键概念澄清（重要修正）

## 1.1 关于"编码视频"的错误（第一版文档已修正）

**您的指正完全正确**。

| 阶段 | 输入 | 是否有视频编码 | 说明 |
|------|------|---------------|------|
| **训练阶段** | 真实视频 | ✅ 是 | 训练数据是真实视频，需用 VAE Encoder 压缩成 latent |
| **推理阶段(T2V)** | **仅文本** | ❌ **否** | 输入只有文本，**直接生成随机噪声**作为起点 |

**正确的 Wan T2V 推理流程**:
```
文本提示 → T5文本编码器 → 随机噪声初始化 → DiT去噪(50时间步) → VAE解码 → 81帧视频
```

**训练阶段才需要的视频编码**:
```
真实视频 → VAE Encoder → Latent z₀ → 加噪 → DiT学习预测噪声 → 重建视频
```

---

# 第二部分：T2V 1.3B 完整流程文字示意图

## 2.1 完整流程图

```
═══════════════════════════════════════════════════════════════════════════════
                              【阶段一：文本编码】
═══════════════════════════════════════════════════════════════════════════════

用户输入: "一只猫在草地上奔跑"
              ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │ T5-XXL 文本编码器 (UMT5-XXL-enc-bf16.pth)                          │
    │ - 多语言 T5 模型，46层 Transformer                                   │
    │ - 参数: 约 4.9B (但推理时冻结，只跑前向)                              │
    └─────────────────────────────────────────────────────────────────────┘
              ↓
【输出A】文本特征: [1, 512, 4096]
        - Dim 0 (1): Batch size = 1
        - Dim 1 (512): 文本 token 数量（固定长度，不足填充，超出截断）
        - Dim 2 (4096): 每个 token 的特征维度

        【每一维代表什么？】
        → 单维度没有明确语义！这是分布式表征（Distributed Representation）
        → 类似 word2vec：单个维度无意义，但4096维向量的整体编码了语义
        → "猫"的向量与"动物"、"毛茸茸"等概念向量接近
              ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 文本嵌入投影层 (Sequential)                                          │
    │ Linear(4096 → 1536) → GELU → Linear(1536 → 1536)                   │
    │ 目的: 将 T5 输出维度对齐到 DiT 维度                                   │
    └─────────────────────────────────────────────────────────────────────┘
              ↓
【输出B】对齐后的文本特征: [1, 512, 1536]
        （这个张量将在后续每个 DiT Block 的 Cross-Attention 中使用）

═══════════════════════════════════════════════════════════════════════════════
                              【阶段二：噪声初始化】
═══════════════════════════════════════════════════════════════════════════════

随机数生成器 ( seeded )
              ↓
【输出C】初始噪声 latent: [1, 16, 21, 60, 104] (float32, 标准正态分布)
        - Dim 0 (1): Batch size
        - Dim 1 (16): VAE latent 通道数 (z_dim)
        - Dim 2 (21): 时间维度 (对应81帧，VAE时间下采样4倍: (81-1)/4+1=21)
        - Dim 3 (60): 高度维度 (480/8=60，VAE空间下采样8倍)
        - Dim 4 (104): 宽度维度 (832/8=104，VAE空间下采样8倍)

        【物理意义】:
        这是一个"压缩的时空立方体"，每个位置包含16个通道的特征。
        此时的内容是纯随机噪声，没有任何可识别的结构。

═══════════════════════════════════════════════════════════════════════════════
                              【阶段三：DiT 去噪】
                              【核心：50个时间步 × 30层】
═══════════════════════════════════════════════════════════════════════════════

FOR timestep = 50, 49, 48, ..., 2, 1, 0:

    ┌─────────────────────────────────────────────────────────────────────┐
    │ 【步骤1】时间步嵌入 (Timestep Embedding)                            │
    │                                                                     │
    │ 输入: timestep (标量，如 50)                                        │
    │       ↓                                                            │
    │ 正弦位置编码 → [256维]                                              │
    │       ↓                                                            │
    │ MLP: Linear(256→1536) → SiLU → Linear(1536→1536)                   │
    │       ↓                                                            │
    │ 时间特征: [1, 1536]                                                 │
    │       ↓                                                            │
    │ 投影: Linear(1536 → 6×1536)                                         │
    │       ↓                                                            │
    │ 【输出D】时间步调制参数: [1, 6, 1536]                                │
    │         这6组参数将用于每个Block的 adaLN-Zero 调制                   │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │ 【步骤2】Patch Embedding (3D卷积)                                   │
    │                                                                     │
    │ 输入: 当前噪声状态 [1, 16, 21, 60, 104]                             │
    │       ↓                                                            │
    │ Conv3d(kernel=(1,2,2), stride=(1,2,2))                             │
    │ - 时间维度: kernel=1, stride=1 (不压缩时间)                          │
    │ - 空间维度: kernel=2, stride=2 (宽高各减半)                          │
    │       ↓                                                            │
    │ 【输出E】Patch化特征: [1, 1536, 21, 30, 52]                          │
    │       ↓                                                            │
    │ 展平: reshape → [1, 32760, 1536]                                    │
    │         32760 = 21 × 30 × 52 (时间帧 × 高patch数 × 宽patch数)        │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │ 【步骤3】DiT Blocks 处理 (30层循环)                                  │
    │                                                                     │
    │ FOR layer_idx = 0 TO 29:                                            │
    │                                                                     │
    │   【当前层输入】: [1, 32760, 1536]                                   │
    │                                                                     │
    │   ┌─────────────────────────────────────────────────────────────┐   │
    │   │ 3.1 Self-Attention (自注意力 - 空间-时间全局)                 │   │
    │   │                                                             │   │
    │   │ 输入: x [1, 32760, 1536], 但先经过 adaLN-Zero 调制:          │   │
    │   │       x_norm = LayerNorm(x) × (1 + scale) + shift            │   │
    │   │                                                             │   │
    │   │ Q, K, V 生成:                                               │   │
    │   │   Q = Linear(x_norm) → reshape [1, 32760, 12, 128]          │   │
    │   │   K = Linear(x_norm) → reshape [1, 32760, 12, 128]          │   │
    │   │   V = Linear(x_norm) → reshape [1, 32760, 12, 128]          │   │
    │   │   (12头注意力，每头128维)                                    │   │
    │   │                                                             │   │
    │   │ 【关键】3D RoPE 位置编码:                                     │   │
    │   │   为 Q, K 添加时空位置信息                                    │   │
    │   │   - 时间位置编码: 区分不同帧                                  │   │
    │   │   - 空间位置编码: 区分同一帧的不同位置                         │   │
    │   │                                                             │   │
    │   │ Flash Attention 计算:                                       │   │
    │   │   Attention(Q, K, V) = softmax(Q×K^T/√d) × V                │   │
    │   │   注意力矩阵: [32760, 32760]                                 │   │
    │   │   【关键特性】:                                              │   │
    │   │   - 帧1的某个patch可以直接看到帧81的任何patch!                │   │
    │   │   - 这是全局注意力，不是局部窗口                              │   │
    │   │                                                             │   │
    │   │ 输出投影: Linear(1536 → 1536)                               │   │
    │   │                                                             │   │
    │   │ 【残差连接】: x = x + gate × attn_output                     │   │
    │   └─────────────────────────────────────────────────────────────┘   │
    │                              ↓                                      │
    │   ┌─────────────────────────────────────────────────────────────┐   │
    │   │ 3.2 Cross-Attention (交叉注意力 - 文本条件)                   │   │
    │   │                                                             │   │
    │   │ 【这是文本影响生成的唯一位置!】                               │   │
    │   │                                                             │   │
    │   │ Q (Query): [1, 32760, 1536] ← 来自视觉特征                    │   │
    │   │   "视觉问：我需要什么信息？"                                   │   │
    │   │                                                             │   │
    │   │ K (Key):   [1, 512, 1536]   ← 来自文本特征 (输出B)           │   │
    │   │   "文本答：我提供这些信息..."                                  │   │
    │   │                                                             │   │
    │   │ V (Value): [1, 512, 1536]   ← 来自文本特征 (输出B)           │   │
    │   │   "文本内容的具体值"                                          │   │
    │   │                                                             │   │
    │   │ 注意力计算:                                                  │   │
    │   │   权重 = softmax(Q × K^T / √d)  → [32760, 512] 矩阵          │   │
    │   │   每个视觉位置(行)对512个文本token(列)计算注意力权重          │   │
    │   │                                                             │   │
    │   │ 【物理意义举例】:                                            │   │
    │   │   - 如果视觉位置要生成"猫"，"猫"token的权重会很高             │   │
    │   │   - 如果视觉位置要生成"草地"，相关纹理描述token权重高          │   │
    │   │                                                             │   │
    │   │ 输出: [1, 32760, 1536]                                       │   │
    │   │                                                             │   │
    │   │ 【残差连接】: x = x + cross_attn_output                      │   │
    │   └─────────────────────────────────────────────────────────────┘   │
    │                              ↓                                      │
    │   ┌─────────────────────────────────────────────────────────────┐   │
    │   │ 3.3 FFN (Feed-Forward Network，前馈网络)                      │   │
    │   │                                                             │   │
    │   │ 结构:                                                       │   │
    │   │   Linear(1536 → 8960)  # 扩展5.8倍                           │   │
    │   │       ↓                                                    │   │
    │   │   GELU(approximate='tanh')  # 非线性激活                     │   │
    │   │       ↓                                                    │   │
    │   │   Linear(8960 → 1536)  # 压缩回原维度                         │   │
    │   │                                                             │   │
    │   │ 【FFN的物理意义】:                                           │   │
    │   │   - 升维(8960): 在高维空间进行复杂的特征变换                    │   │
    │   │   - 非线性: 引入表达能力，学习复杂的映射关系                    │   │
    │   │   - 降维(1536): 提取关键信息，回到标准维度                      │   │
    │   │   - 类比：像"特征精炼器"                                     │   │
    │   │                                                             │   │
    │   │ 输出: [1, 32760, 1536]                                       │   │
    │   │                                                             │   │
    │   │ 【残差连接 + 门控】: x = x + gate × ffn_output               │   │
    │   └─────────────────────────────────────────────────────────────┘   │
    │                                                                     │
    │ 【当前层输出】: [1, 32760, 1536]                                    │
    │                                                                     │
    │ 【每层输出的意义】: 见第三部分详细解释                               │
    │                                                                     │
    END FOR (完成30层)                                                   │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │ 【步骤4】Head 输出层                                               │
    │                                                                     │
    │ 输入: [1, 32760, 1536]                                              │
    │       ↓                                                            │
    │ LayerNorm + Linear(1536 → 16×1×2×2=64)                             │
    │       ↓                                                            │
    │ 【输出F】预测的噪声: [1, 16, 21, 60, 104]                            │
    │                                                                     │
    │ 【重要】: 这是预测的噪声 v_pred，不是去噪后的图像！                  │
    │                                                                     │
    │ DiT 的作用: 告诉调度器"应该往哪个方向去噪"                          │
    │ 调度器的作用: 实际执行"走一步"                                      │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │ 【步骤5】分类器自由引导 (Classifier-Free Guidance, CFG)             │
    │                                                                     │
    │ 上述步骤重复两次:                                                   │
    │   - 第一次: 使用真实文本条件 → v_cond                               │
    │   - 第二次: 使用空文本 "" → v_uncond                                │
    │                                                                     │
    │ 最终预测:                                                           │
    │   v_final = v_uncond + guide_scale × (v_cond - v_uncond)            │
    │            (guide_scale=5.0 默认)                                   │
    │                                                                     │
    │ 【物理意义】:                                                       │
    │   - v_uncond: 无文本指导时的"自然"去噪方向                          │
    │   - (v_cond - v_uncond): 文本指导的"额外"方向                      │
    │   - ×5.0: 放大文本影响，使生成更贴合提示词                          │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │ 【步骤6】调度器步进 (Scheduler Step)                                │
    │                                                                     │
    │ 更新公式 (Flow Matching / Euler):                                   │
    │   latent_{t-1} = latent_t - step_size × v_final                     │
    │                                                                     │
    │ 【物理意义】:                                                       │
    │   - 从当前噪声状态 latent_t 出发                                    │
    │   - 沿着预测的噪声反方向 (-v_final) 走一步                          │
    │   - 步长 step_size 由调度器根据当前时间步计算                        │
    │                                                                     │
    │ 【关键】: 这里没有"另一个模块"去除噪声！                             │
    │          调度器只是执行简单的减法更新。                               │
    │          DiT预测方向，调度器执行移动。                                │
    └─────────────────────────────────────────────────────────────────────┘

    【下一个时间步的输入】= latent_{t-1}

    【循环继续】直到 timestep = 0

END FOR (完成50个时间步)

═══════════════════════════════════════════════════════════════════════════════
                              【阶段四：VAE解码】
═══════════════════════════════════════════════════════════════════════════════

【输出G】最终去噪后的 latent: [1, 16, 21, 60, 104]
        （这是一个完整的时空立方体，包含所有81帧的压缩表示）
              ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │ WanVAE Decoder (3D Causal Convolution)                             │
    │                                                                     │
    │ 【Causal特性】: 解码第t帧时，只能看到latent的第0到t帧                │
    │                这保证了时序因果性，防止"未来信息泄露"                │
    │                                                                     │
    │ 逐时间位置解码:                                                     │
    │   FOR t in range(21):                                               │
    │       latent_slice = latent[:, t:t+1, :, :]  # [1, 16, 1, 60, 104] │
    │       frame = CausalConv3D_Decoder(latent_slice, cache)            │
    │       output_frames.append(frame)                                   │
    │                                                                     │
    │   # 时间上采样 4 倍                                                 │
    │   video = temporal_upsample(output_frames, factor=4)               │
    └─────────────────────────────────────────────────────────────────────┘
              ↓
【输出H】最终视频: [3, 81, 480, 832]
        - Dim 0 (3): RGB 通道
        - Dim 1 (81): 帧数
        - Dim 2 (480): 高度
        - Dim 3 (832): 宽度

═══════════════════════════════════════════════════════════════════════════════
                                  【流程结束】
═══════════════════════════════════════════════════════════════════════════════
```

---

# 第三部分：DiT 层输出的意义（新问题解答）

## Q: 既然每个时间步都需要循环经过30个层，那么每个层的输出又有什么意义？

### 3.1 逐层输出的意义

```
【形象比喻】: 30层 DiT = 30位专家，每位专家负责不同的"精修"任务

输入层 (Block 0):
  输入: 原始噪声 + 时间步信息
  输出: 初步的"草图" - 大尺度的空间结构和运动趋势

中间层 (Block 14-15):
  输入: 前层的草图
  输出: "线稿" - 物体的形状、轮廓、相对位置

深层 (Block 27-29):
  输入: 线稿
  输出: "上色稿" - 细节纹理、颜色、光照
```

### 3.2 从特征可视化角度的理解

| 层范围 | 输出特征类型 | 可视化近似 |
|--------|-------------|-----------|
| Block 0-5 | 低级视觉特征 | 边缘检测、颜色块 |
| Block 6-15 | 中级语义特征 | 物体部件、形状轮廓 |
| Block 16-25 | 高级语义特征 | 完整物体、场景布局 |
| Block 26-29 | 细节精修特征 | 纹理、光照、细微结构 |

### 3.3 数学角度的理解

```
第0层输出: f_0(x) = x + Attention(LN(x)) + FFN(LN(x))
           ↓
第1层输出: f_1(f_0(x)) = f_0(x) + Attention(LN(f_0(x))) + ...
           ↓
...
第29层输出: f_29(...f_0(x)...) = 最终的噪声预测

每一层都是对特征的逐步"提炼"和"精修"，就像:
x → 粗加工 → 半精加工 → 精加工 → 抛光
```

### 3.4 为什么需要30层？

```
【表达能力需求】:
- 视频生成的复杂度远高于图像
- 需要建模: 空间结构 + 时序动态 + 文本语义对齐
- 浅层: 捕获局部空间相关性
- 中层: 构建物体级别的表征
- 深层: 整合全局语义和时序一致性

【残差连接的作用】:
- 如果没有残差，深层梯度会消失
- 残差连接允许信息直接传递:x = x + f(x)
- 模型可以"选择"使用哪些层的变换
```

---

## Q: 对于多个时间步来说，这些层的意义又是什么？

### 3.5 时间步 vs 层的二维结构

```
                    层维度 (Layer)
              0    1    2    ...   29
           ┌────┬────┬────┬────┬────┐
    第50步 │ B0 │ B1 │ B2 │... │ B29│ → 输出噪声预测50
           ├────┼────┼────┼────┼────┤
    第49步 │ B0 │ B1 │ B2 │... │ B29│ → 输出噪声预测49
时  ...    ├────┼────┼────┼────┼────┤
间 ...     │    │    │    │    │    │
步 ...     ├────┼────┼────┼────┼────┤
    第1步  │ B0 │ B1 │ B2 │... │ B29│ → 输出噪声预测1
           ├────┼────┼────┼────┼────┤
    第0步  │ B0 │ B1 │ B2 │... │ B29│ → 输出噪声预测0
           └────┴────┴────┴────┴────┘

【注意】: 所有时间步使用同一组 Block 参数 (B0-B29 的权重相同)

【意义解释】:
- 横向(层): 从粗糙到精细的特征提取流水线
- 纵向(时间步): 从噪声到清晰图像的渐进式生成

比喻:
- 横向 = 工厂的30道工序（每次加工都要走一遍）
- 纵向 = 重复加工50次，逐渐精修
```

### 3.6 不同时间步的层行为差异

```
虽然同一层在不同时间步的参数相同，但行为不同:

Block 29 在第50步(高噪声):
  输入: 几乎纯噪声
  任务: 预测"大致的噪声方向"，不需要精细细节
  输出: 粗略的噪声估计

Block 29 在第0步(低噪声):
  输入: 接近清晰的latent
  任务: 精修微小的噪声残差，添加细节
  输出: 精细的噪声估计

【关键】: 时间步信息通过 adaLN-Zero 注入每层
         让同一层能根据"当前噪声水平"调整行为
```

---

# 第四部分：所有问题详细解答（Q&A）

## Q: 模型真的是同时对81帧进行生成吗？

### A: 是的，**并行生成所有81帧**，不是逐帧生成。

```
【证据1】: 初始噪声的维度
  latent_0: [1, 16, 21, 60, 104]
            └── 21个时间位置（对应81帧）

  这21个位置在初始时就全部存在，不是逐步生成的。

【证据2】: Self-Attention 的全局性
  注意力矩阵: [32760, 32760]
  其中包含帧间注意力:
    - 帧1的patch可以 attending to 帧81的patch
    - 这证明所有帧同时存在于计算中

【证据3】: 不是自回归
  自回归生成: Frame N 作为 Frame N+1 的输入
  Wan2.1: 所有帧共享同一个输入（文本+噪声），并行生成
```

### 文生图 vs 文生视频的区别

| 特性 | 文生图 (DiT) | 文生视频 (Wan T2V) |
|------|-------------|-------------------|
| 输入噪声 | [16, H/8, W/8] | [16, T/4, H/8, W/8] |
| 序列长度 | H/8 × W/8 | T/4 × H/8 × W/8 |
| 帧生成 | 1帧 | 81帧并行 |
| 时序建模 | 无 | 3D RoPE + 全局注意力 |
| 一致性 | 天然单帧 | 需额外机制保证帧间一致 |

### 关于"1000步采样但只生成1帧"

```
理论上可行，但效果≠文生图模型:

1. 架构差异:
   - Wan T2V 的 DiT 期望处理时空数据
   - 单帧时，时间维度的 RoPE 和 attention 参数未充分利用

2. VAE 差异:
   - Wan 使用 3D Causal VAE
   - 单帧时 Causal 机制可能导致边界伪影

3. 训练分布:
   - Wan T2V 在视频数据上训练
   - 单帧生成不在训练分布内

结论: 不建议这样做。如果需要单帧图像，应使用专门的文生图模型。
```

---

## Q: 能否从代码中提取生成的视频帧参数对隐视频各维度的影响？

### A: 可以，以下是完整的维度映射关系：

```python
# 生成参数
frame_num = 81      # 生成帧数
height = 480        # 视频高度
width = 832         # 视频宽度

# VAE 下采样参数 (来自 config)
vae_stride = (4, 8, 8)  # (时间, 高度, 宽度)
patch_size = (1, 2, 2)  # DiT的patch大小

# ═══════════════════════════════════════════════════════════════
# 维度计算流程
# ═══════════════════════════════════════════════════════════════

# 1. 视频 → VAE编码后的 Latent 维度
latent_t = (frame_num - 1) // vae_stride[0] + 1  # (81-1)//4 + 1 = 21
latent_h = height // vae_stride[1]                # 480//8 = 60
latent_w = width // vae_stride[2]                 # 832//8 = 104

# Latent形状: [batch, 16, 21, 60, 104]

# 2. Latent → Patch Embedding 后的维度
patch_t = latent_t // patch_size[0]   # 21//1 = 21
patch_h = latent_h // patch_size[1]   # 60//2 = 30
patch_w = latent_w // patch_size[2]   # 104//2 = 52

# Patch序列长度: 21 × 30 × 52 = 32,760

# 3. DiT 处理的序列维度
seq_len = patch_t × patch_h × patch_w  # 32,760

# DiT输入: [batch, seq_len, dim] = [1, 32760, 1536]

# ═══════════════════════════════════════════════════════════════
# 各参数变化的影响
# ═══════════════════════════════════════════════════════════════

"""
frame_num (帧数):
  - 影响 latent_t = (frame_num-1)//4 + 1
  - 影响 seq_len (线性增长)
  - 影响计算量 (seq_len^2 的 attention 开销)

height/width (分辨率):
  - 影响 latent_h, latent_w
  - 影响 seq_len (平方增长)
  - 高分辨率 = 显存爆炸性增长

vae_stride (VAE下采样率):
  - 固定为 (4,8,8)，不可调
  - 决定压缩率

patch_size (DiT patch大小):
  - 固定为 (1,2,2)，不可调
  - 影响 tokens 数量
"""

# 显存估算公式
def estimate_vram(frame_num, height, width, batch_size=1):
    latent_t = (frame_num - 1) // 4 + 1
    latent_h = height // 8
    latent_w = width // 8

    patch_t = latent_t  # patch_size[0]=1
    patch_h = latent_h // 2
    patch_w = latent_w // 2

    seq_len = patch_t * patch_h * patch_w

    # Attention 激活: [batch, heads, seq_len, seq_len]
    attn_memory = batch_size * 12 * seq_len * seq_len * 4 / (1024**3)  # GB

    # 模型参数: ~1.3B = 2.6GB (bf16)
    model_memory = 2.6

    # 其他开销
    other_memory = 2.0

    total = attn_memory + model_memory + other_memory
    return total, seq_len

# 例子
# 81帧, 480x832: seq_len=32760, VRAM≈50GB (需要gradient checkpointing)
# 这就是为什么需要 offload 和 tiling 策略
```

---

## Q: 第一帧如何影响后续帧？

### A: 通过 **Self-Attention** 和 **3D RoPE** 隐式影响，不是显式传递。

```
【错误理解】: 第一帧输出 → 作为第二帧输入 (自回归)

【正确机制】:
  所有帧的latent同时存在
  Block 0 的 Self-Attention 允许:
    - 帧1的每个patch看到帧2的每个patch
    - 帧1看到帧81

  通过3D RoPE，相邻帧的位置编码相似
  这鼓励模型生成相似但偏移的内容（运动）

【具体实现】:
  Position(帧1, h, w) → RoPE编码 A
  Position(帧2, h, w) → RoPE编码 B (与A相似，时间维度+1)

  在Attention中: A和B的点积值较高
  → 这两个位置的特征会互相"拉近距离"
  → 内容相似但有微小差异（运动）
```

### 与文生图的核心区别

```
文生图: 输入 → DiT → 输出 (结束)

文生视频:
  输入(噪声) → DiT处理所有帧 → 输出(所有帧)
              ↓
         在DiT内部:
         - 自注意力连接所有帧
         - 文本通过Cross-Attention影响所有帧
         - 3D RoPE提供时空位置感知
```

---

## Q: FFN 是什么？

### A: Feed-Forward Network（前馈网络）

```
结构:
  Linear(1536 → 8960)
      ↓
  GELU(激活函数)
      ↓
  Linear(8960 → 1536)

作用:
  1. 升维: 在高维空间进行复杂的非线性变换
  2. 非线性: GELU引入非线性，增强表达能力
  3. 降维: 提取关键信息，回到原始维度

类比:
  像"信息搅拌机"：
  - 先把信息"展开"(8960维)
  - 充分"搅拌"(GELU激活)
  - 再"压缩"回精华(1536维)

为什么需要:
  Attention是"信息交换"机制
  FFN是"信息变换"机制
  两者缺一不可
```

---

## Q: 81帧是如何解码的？分别还原还是聚合还原？

### A: **聚合后一起还原**，通过 Causal 机制保持连续性。

```
【不是】: 分别解码81帧 → 拼接
【是】:   整个时空立方体 → Causal VAE → 81帧

具体过程:
  latent: [16, 21, 60, 104] (时空立方体)
      ↓
  FOR t in range(21):  # 21个时间切片
      slice = latent[:, t:t+1, :, :]  # [16, 1, 60, 104]

      IF t == 0:
          frame = decoder(slice, cache=None)
      ELSE:
          # Causal: 可以看到之前的切片
          frame = decoder(slice, cache=accumulated)

      accumulated.append(frame)

  # 时间上采样: 21 → 81 (factor=4)
  video = temporal_upsample(accumulated)

【为什么不是分别解码】:
  分别解码无法保证帧间连续性
  Causal机制强制时序依赖
```

---

## Q: SAE 是每个时间步都生效吗？特定的层和时间步？

### A: **取决于您的配置**，代码支持灵活配置。

```python
# 当前代码的默认配置 (sae_train_t2v_1_3b.py)
hook_params = {
    "hook_mode": "block_out",      # Hook block的输出
    "hook_layers": "15",            # 只Hook第15层
    "hook_points": "after_block",   # 在Block之后
}

training_params = {
    "sampling_steps": 30,           # 跑30个时间步
}

# ═══════════════════════════════════════════════════════════════
# 实际Hook发生的时机
# ═══════════════════════════════════════════════════════════════

FOR t in [50, 49, ..., 1, 0]:  # 每个时间步
    FOR layer in [0, 1, ..., 29]:  # 每层

        # 前向传播
        output = block_layer.forward(input)

        # Hook检查
        IF layer == 15 AND hook_mode == "block_out":
            # 捕获特征
            features = output  # [1, 32760, 1536]

            # SAE训练
            z, recon, loss = sae(features)
            loss.backward()
            optimizer.step()

# ═══════════════════════════════════════════════════════════════
# 不同配置的效果
# ═══════════════════════════════════════════════════════════════

【配置1】: hook_layers="15", 所有时间步都Hook
  → 每个时间步的第15层输出都会被SAE处理
  → 共 50个时间步 × 1层 = 50次SAE前向/反向

【配置2】: 只Hook特定时间步 (需要修改代码)
  IF t in [0, 25, 49] AND layer == 15:
      sae_train(features)
  → 只有3个时间步会触发SAE训练

【配置3】: Hook多层
  hook_layers = "5,15,25"
  → 每时间步有3个层被Hook
  → 共 50 × 3 = 150次SAE处理
```

### 重要澄清：层 vs Block vs 时间步

```
【30层 DiT】= 30个不同的 Block 实例
  每个 Block 有自己的参数 (Q,K,V投影, FFN权重等)
  Block 0 ≠ Block 15 ≠ Block 29

【50个时间步】= 同组 Block 被复用50次
  第1步的 Block 0 和第2步的 Block 0 是同一个实例！
  参数共享，只是输入不同。

【关系总结】:
  模型架构: 30层 (固定)
  采样过程: 50步 (可调)
  总计算量: 50步 × 30层 = 1500次 Block 前向
```

---

# 第五部分：SAE 创新应用（24GB显存优化）

## 5.1 显存优化的核心策略

```
问题: 24GB显存，现有配置接近极限

分析:
  Wan 1.3B模型: ~2.6GB
  激活值(50步×30层×32760×1536): ~15-20GB
  总计: ~24GB (无SAE训练空间)

解决方案:
```

### 策略1: 时间步稀疏采样（推荐）

```python
def select_key_timesteps(total=50, mode="boundary"):
    if mode == "boundary":
        return [0, 25, 49]    # 只选首尾和中间
    elif mode == "uniform":
        return [0, 10, 20, 30, 40, 49]
    elif mode == "early_focus":
        return [0, 5, 10, 15, 20, 30, 40, 49]

# 修改训练循环
key_steps = select_key_timesteps(50, mode="boundary")

for t in timesteps:
    # DiT前向 (必须做)
    noise_pred = dit_forward(latents, t)

    # 只在关键时间步训练SAE
    if t in key_steps:
        features = hook_activations()
        loss = sae_train_step(features)

    # 调度器步进
    latents = scheduler.step(noise_pred, t, latents)

# 显存节省: 50步 → 3步，节省 ~94%
```

### 策略2: 层间稀疏采样

```python
# 只训练最关键的层
selected_layers = {15}  # 中层通常最丰富

# 显存节省: 30层 → 1层，节省 ~97%
```

### 策略3: 离线特征采集 + 离线训练（最推荐）

```python
# ═══════════════════════════════════════════════════════════════
# 阶段1: 特征采集 (推理模式，无梯度)
# ═══════════════════════════════════════════════════════════════

feature_buffer = []

for prompt in prompts:
    for t in timesteps:
        with torch.no_grad():
            # 只跑DiT前向
            _ = dit_forward(latents, t)

            # 采集特征到CPU
            feat = hook_features().cpu()
            feature_buffer.append({
                'layer15': feat,
                'timestep': t,
                'prompt': prompt
            })

        latents = scheduler.step(noise_pred, t, latents)

    # 定期保存到磁盘
    if len(feature_buffer) > 1000:
        torch.save(feature_buffer, f'features_batch_{i}.pt')
        feature_buffer = []

# ═══════════════════════════════════════════════════════════════
# 阶段2: 离线训练SAE (纯SAE，无DiT)
# ═══════════════════════════════════════════════════════════════

features = load_features_from_disk()  # 从磁盘加载

sae = SparseAutoEncoder(
    d_model=1536,
    d_hidden=3072,      # 降维，原6144
    top_k=32            # 减少，原64
)

optimizer = torch.optim.AdamW(sae.parameters(), lr=1e-3)

for epoch in range(10):
    for feat in features:
        feat = feat.cuda()
        z, recon, loss = sae(feat, return_loss=True)
        loss.backward()
        optimizer.step()

# 显存占用: 只有SAE本身 ~100MB
# 可以训练多个不同配置的SAE
```

## 5.2 推荐的24GB配置

```python
optimal_config = {
    # SAE结构
    "d_model": 1536,
    "d_hidden": 3072,       # 2倍扩展 (原为4倍)
    "top_k": 32,            # 减少k值 (原为64)
    "sparsity": "topk",

    # 训练策略
    "strategy": "offline",   # 离线训练
    "hook_layers": [15],     # 只训练中层
    "key_timesteps": [0, 25, 49],  # 关键时间步

    # 优化
    "batch_size": 4096,      # 离线可用大batch
    "precision": "bf16",
    "gradient_checkpointing": True,
}

# 预期显存:
# - DiT推理: ~3GB
# - SAE训练: ~2GB
# - 特征缓存: ~10GB (可调节)
# 总计: ~15GB，留有9GB余量
```

## 5.3 创新方向（详细伪代码）

### 创新1: 时步条件SAE

```python
class TimestepConditionedSAE(nn.Module):
    """
    SAE编码考虑当前去噪时间步
    假设: 不同时间步的特征分布不同
    """
    def __init__(self, d_model=1536, d_hidden=3072):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)

        # 时间步嵌入网络
        self.timestep_embed = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, d_hidden)
        )

    def encode(self, x, timestep):
        """
        x: [N, d_model]
        timestep: [N] 或标量，范围 [0, 1]
        """
        # 基础编码
        z = F.relu(self.encoder(x))  # [N, d_hidden]

        # 时间步调制
        t_embed = self.timestep_embed(timestep.view(-1, 1))  # [N, d_hidden]
        z = z * (1 + t_embed)  # 自适应缩放

        # Top-K稀疏
        z_sparse, indices, values = topk_sparsify(z, k=32)
        return z_sparse, indices, values

    def forward(self, x, timestep, return_loss=True):
        z, _, _ = self.encode(x, timestep)
        x_hat = self.decoder(z)

        if return_loss:
            loss = F.mse_loss(x_hat, x)
            return x_hat, z, loss
        return x_hat, z

# 训练
for feat_batch, t_batch in dataloader:
    # t_batch: 时间步归一化到[0,1]
    # t≈1: 高噪声(早期), t≈0: 低噪声(晚期)

    x_hat, z, loss = tc_sae(feat_batch, t_batch, return_loss=True)
    loss.backward()
    optimizer.step()

# 预期发现:
# t≈1 (早期): 结构特征、大尺度模式
# t≈0 (晚期): 纹理特征、细节refinement
```

### 创新2: 跨时间步一致性SAE

```python
class ConsistencyRegularizedSAE(nn.Module):
    """
    鼓励相同概念在不同时间步有相似的SAE表示
    """
    def __init__(self, d_model=1536, d_hidden=3072):
        super().__init__()
        self.sae = SparseAutoEncoder(d_model, d_hidden)

    def forward(self, x_t1, x_t2, return_loss=True):
        """
        x_t1: 时间步t1的特征 [N, d_model]
        x_t2: 时间步t2的特征 (相同概念) [N, d_model]
        """
        # 分别编码
        z_t1, _, recon_loss_1 = self.sae(x_t1, return_loss=True)
        z_t2, _, recon_loss_2 = self.sae(x_t2, return_loss=True)

        # 重建损失
        recon_loss = recon_loss_1 + recon_loss_2

        # 一致性损失1: 激活位置的重叠
        sparsity_1 = (z_t1.abs() > 1e-6).float()
        sparsity_2 = (z_t2.abs() > 1e-6).float()
        overlap = (sparsity_1 * sparsity_2).sum() / (sparsity_1.sum() + 1e-8)
        consistency_loss = 1 - overlap

        # 一致性损失2: 激活值的相似性
        value_sim = F.cosine_similarity(z_t1, z_t2, dim=-1).mean()

        # 总损失
        total_loss = recon_loss + 0.1 * consistency_loss - 0.1 * value_sim

        return total_loss, z_t1, z_t2

# 应用场景:
# 同一prompt在不同时间步的特征应该对应相同的概念
# SAE应该学习到"概念一致性"
```

### 创新3: 层级对比SAE

```python
class HierarchicalSAE(nn.Module):
    """
    多层SAE联合训练，学习层级表示
    """
    def __init__(self, layers=[5, 15, 25], d_model=1536):
        super().__init__()
        self.layers = layers

        # 为每层创建独立的SAE
        self.saes = nn.ModuleDict({
            f'layer_{l}': SparseAutoEncoder(d_model, 2048)  # 较小hidden
            for l in layers
        })

    def forward(self, features_dict):
        """
        features_dict: {layer_idx: features}
        """
        losses = {}
        representations = {}

        for layer_idx in self.layers:
            feat = features_dict[layer_idx]
            sae = self.saes[f'layer_{layer_idx}']

            z, _, loss = sae(feat, return_loss=True)
            losses[layer_idx] = loss
            representations[layer_idx] = z

        # 可选: 添加跨层一致性损失
        # 浅层和深层的关系

        return representations, losses

# 训练数据组织:
# 同时采集第5, 15, 25层的特征
# 期望:
#   第5层: 低级特征(边缘、颜色)
#   第15层: 中级特征(形状、部件)
#   第25层: 高级特征(语义、运动)
```

---

**文档结束**

*本文档整合了第一版和修正版，详细回答了用户提出的所有问题，包括新问题和之前的问题。*
