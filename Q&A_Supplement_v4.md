# Wan2.1 T2V 1.3B 深度问答补充（v4.0）

**日期**: 2026-03-29
**文档类型**: Q&A 补充

---

## Q12: 什么是 adaLN-Zero 调制？作用是什么？

### 12.1 概念解释

**adaLN-Zero** = **Adaptive Layer Normalization with Zero Initialization**（自适应层归一化+零初始化）

这是 DiT (Diffusion Transformer) 论文中提出的关键技术，用于将**时间步信息**和**条件信息**注入到 Transformer Block 中。

### 12.2 代码实现详解

```python
# 文件: wan/modules/model.py (第275-317行)

class WanAttentionBlock(nn.Module):
    def __init__(self, ...):
        ...
        # modulation 参数: 可学习的调制参数
        # 形状 [1, 6, dim]: 6个调制参数，每个维度为 dim
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, ...):
        # x: [B, L, C] = 输入特征
        # e: [B, 6, C] = 时间步嵌入 (由时间步编码而来)

        # Step 1: 融合可学习参数和时间步嵌入
        # (self.modulation + e): 将固定调制与时间步调制相加
        # .chunk(6, dim=1): 分割成6个参数
        e = (self.modulation + e).chunk(6, dim=1)
        # e[0], e[1], e[2] 用于 Self-Attention 分支
        # e[3], e[4], e[5] 用于 FFN 分支

        # Step 2: Self-Attention 分支的 adaLN-Zero
        # 传统 LayerNorm: norm(x)
        # adaLN-Zero: norm(x) * (1 + scale) + shift
        x_norm = self.norm1(x).float() * (1 + e[1]) + e[0]

        # Self-Attention 计算
        y = self.self_attn(x_norm, ...)

        # Step 3: 门控残差连接 (Zero 初始化部分)
        # 传统残差: x = x + y
        # adaLN-Zero: x = x + y * gate
        # e[2] 初始化为接近0，所以早期训练时残差贡献很小
        x = x + y * e[2]

        # Step 4: FFN 分支的 adaLN-Zero (类似)
        y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
        x = x + y * e[5]

        return x
```

### 12.3 6个调制参数的作用

| 参数 | 名称 | 作用位置 | 功能 |
|------|------|---------|------|
| e[0] | shift_1 | Self-Attention前 | 平移输入分布 |
| e[1] | scale_1 | Self-Attention前 | 缩放输入分布 |
| e[2] | gate_1 | Self-Attention后 | **门控残差强度** |
| e[3] | shift_2 | FFN前 | 平移输入分布 |
| e[4] | scale_2 | FFN前 | 缩放输入分布 |
| e[5] | gate_2 | FFN后 | **门控残差强度** |

### 12.4 为什么需要 adaLN-Zero？

```
【问题1: 时间步感知】
扩散模型需要在不同噪声水平下表现不同:
- 高噪声 (t≈1): 需要大刀阔斧去噪，关注大结构
- 低噪声 (t≈0): 需要精细调整，关注细节

【adaLN的解决方案】
通过时间步嵌入 e，让每个Block知道"当前噪声水平":
- e 由时间步 t 编码而来
- 不同 t → 不同的 e → 不同的调制参数 → 不同的行为

【问题2: 训练稳定性】
残差连接太多，初期训练不稳定

【Zero初始化的解决方案】
- gate参数 e[2], e[5] 初始化为接近0
- 训练初期: x ≈ x (残差贡献小，主要学恒等映射)
- 训练后期: 逐渐学习有用的残差变换
```

### 12.5 类比理解

```
传统 LayerNorm: 固定的"标准化工厂"
   输入 → [标准化] → 输出 (固定流程)

adaLN-Zero: 智能化的"自适应工厂"
   输入 → [根据时间步智能调整的标准化] → 输出

   时间步信号 → 控制面板 → 调整:
     - 温度参数 (scale)
     - 偏移参数 (shift)
     - 阀门开度 (gate，Zero初始化从小开度开始)
```

---

## Q13: 什么是 3D RoPE？具体计算过程？

### 13.1 概念解释

**3D RoPE** = **3D Rotary Position Embedding**（三维旋转位置编码）

RoPE 是旋转位置编码，3D RoPE 将 RoPE 扩展到**时间-高度-宽度**三个维度。

### 13.2 为什么需要 3D RoPE？

```
传统 1D RoPE: 只能编码序列位置 (如文本的第几个token)
2D RoPE: 可以编码图像位置 (x, y)
3D RoPE: 可以编码视频位置 (t, x, y) ← Wan2.1使用

【视频生成的特殊需求】
- 需要知道"这是第几帧"
- 需要知道"帧内的哪个位置"
- 需要建模帧间的时序关系
```

### 13.3 数学原理

```
RoPE 的核心思想: 用旋转矩阵编码相对位置

对于位置 m 的向量 x，将其分组成复数对:
x = [x_0, x_1, x_2, x_3, ...] → [(x_0 + ix_1), (x_2 + ix_3), ...]

对每个复数应用旋转:
(x_j + ix_{j+1}) × e^{i·m·θ_j} = 旋转后的复数

其中 θ_j = 10000^{-2j/d} (频率递减)

【关键特性】
位置 m 和位置 n 的旋转角度差: (m-n)·θ_j
注意力计算中，相对位置只与 (m-n) 有关！
```

### 13.4 代码详解与计算过程

```python
# 文件: wan/modules/model.py

# ═══════════════════════════════════════════════════════════════
# Step 1: 生成 RoPE 频率参数 (模型初始化时执行一次)
# ═══════════════════════════════════════════════════════════════

@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    """
    生成旋转位置编码的频率参数

    Args:
        max_seq_len: 最大序列长度 (1024)
        dim: 每个头的维度 (128)
        theta: 基础频率 (10000)

    Returns:
        freqs: [max_seq_len, dim/2] 复数张量
    """
    # 计算频率: 1 / (theta^(2i/d))
    freqs = torch.outer(
        torch.arange(max_seq_len),  # [0, 1, 2, ..., 1023]
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
        # 频率递减: 1, 1/10000^(2/d), 1/10000^(4/d), ...
    )

    # 转换为复数: e^(i·freq) = cos(freq) + i·sin(freq)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    # 形状: [1024, 64] (复数，实部和虚部各64维)

    return freqs


# 在 WanModel.__init__ 中调用 (第480-485行)
self.freqs = torch.cat([
    rope_params(1024, d - 4 * (d // 6)),   # 时间维度 (约一半)
    rope_params(1024, 2 * (d // 6)),       # 高度维度 (约1/6)
    rope_params(1024, 2 * (d // 6)),       # 宽度维度 (约1/6)
], dim=1)
# 最终形状: [1024, d/2] = [1024, 64] (对于head_dim=128)


# ═══════════════════════════════════════════════════════════════
# Step 2: 应用 3D RoPE (每次前向传播时执行)
# ═══════════════════════════════════════════════════════════════

@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    应用3D旋转位置编码

    Args:
        x: [B, L, num_heads, head_dim] 的 Q 或 K
           实际传入形状: [1, 32760, 12, 128]
        grid_sizes: [B, 3] = 实际的 (F, H, W) 维度
                    例如: [[21, 30, 52]]
        freqs: 预计算的频率 [1024, head_dim/2]

    Returns:
        x: 应用RoPE后的张量，形状不变
    """
    n, c = x.size(2), x.size(3) // 2
    # n = num_heads = 12
    # c = head_dim / 2 = 64 (复数对的数量)

    # 将 freqs 分成三部分: 时间、高度、宽度
    # 分割比例: 时间维度占约一半，高度和宽度各占约1/6
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    # freqs[0]: [1024, ~32]  用于时间维度
    # freqs[1]: [1024, ~16]  用于高度维度
    # freqs[2]: [1024, ~16]  用于宽度维度

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w  # = 21 * 30 * 52 = 32760

        # 将实数张量转换为复数张量
        # x[i, :seq_len]: [32760, 12, 128]
        # reshape: [32760, 12, 32, 2] (分成32个复数对)
        # view_as_complex: [32760, 12, 32] (复数)
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )

        # 构造3D位置编码
        # 对于每个时空位置 (t, h, w)，组合三个维度的编码
        freqs_i = torch.cat([
            # 时间维度编码: [f, 1, 1, dim_t] 扩展到 [f, h, w, dim_t]
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),

            # 高度维度编码: [1, h, 1, dim_h] 扩展到 [f, h, w, dim_h]
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),

            # 宽度维度编码: [1, 1, w, dim_w] 扩展到 [f, h, w, dim_w]
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, -1)
        # 最终: [32760, 1, 64] (与 x_i 的最后一维匹配)

        # 应用旋转: 复数乘法 = 旋转
        # x_i: [32760, 12, 32] (复数)
        # freqs_i: [32760, 1, 32] (复数，广播到12个头)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        # 转回实数: [32760, 12, 64] → flatten → [32760, 12, 128]

        # 拼接可能存在的padding部分
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)

    return torch.stack(output).float()


# ═══════════════════════════════════════════════════════════════
# Step 3: 具体计算示例
# ═══════════════════════════════════════════════════════════════

def example_rope_calculation():
    """
    手动计算一个简化的3D RoPE示例
    """
    # 假设: 2帧，2x2空间分辨率，head_dim=8
    f, h, w = 2, 2, 2
    head_dim = 8
    seq_len = f * h * w  # 8

    # 原始Q向量 (简化，只看1个头)
    q = torch.randn(seq_len, head_dim)  # [8, 8]

    # 分组成复数: [8, 4] 复数
    q_complex = torch.view_as_complex(q.reshape(seq_len, head_dim//2, 2))

    # 为每个位置生成 (t, y, x) 坐标
    positions = []
    for t in range(f):
        for y in range(h):
            for x in range(w):
                positions.append((t, y, x))
    # positions = [(0,0,0), (0,0,1), (0,1,0), (0,1,1),
    #              (1,0,0), (1,0,1), (1,1,0), (1,1,1)]

    # 计算每个位置的旋转角度
    # 时间、高度、宽度各贡献一部分
    for i, (t, y, x) in enumerate(positions):
        # 时间维度贡献: t * theta_t
        # 高度维度贡献: y * theta_y
        # 宽度维度贡献: x * theta_x

        # 相对位置差示例:
        # pos_0 = (0,0,0) 和 pos_4 = (1,0,0) 的时间差为1
        # 在attention中，相对编码差只与 (1,0,0) 有关

        # 旋转应用:
        # q_complex[i] *= e^(i * (t*theta_t + y*theta_y + x*theta_x))
        pass

    # 关键特性验证:
    # 位置A: (t=0, h=0, w=0) 的编码
    # 位置B: (t=1, h=0, w=0) 的编码 (时间+1)
    # 位置C: (t=0, h=1, w=0) 的编码 (高度+1)

    # A和B的编码差异只来自时间维度
    # A和C的编码差异只来自高度维度
    # 这种分离性帮助模型区分不同类型的运动
```

### 13.5 3D RoPE 的关键特性

```
【特性1: 相对位置感知】
Attention(Q_m, K_n) 只依赖于相对位置 (m-n)
这意味着模型能更好地学习"相对运动"而非"绝对位置"

【特性2: 维度分离】
时间、高度、宽度的编码是分开计算的
模型可以独立学习时间运动和空间结构

【特性3: 长程依赖】
RoPE的频率随维度递减
低维度编码短程关系，高维度编码长程关系
```

---

## Q14: 单个时间步内同时对所有时间维度进行调制是主流吗？还有其他架构？

### 14.1 Wan2.1 的架构特点

```
Wan2.1 采用的是:
【Full 3D Self-Attention】
- 在单个时间步的前向传播中
- 所有帧的所有patch同时被处理
- Self-Attention 矩阵: [32760, 32760]
  包含帧间注意力、帧内注意力

这是目前视频生成的主流吗？
→ 是的，这是当前基于Transformer视频生成的主流
```

### 14.2 主流视频生成架构对比

| 架构类型 | 代表模型 | 特点 | 优缺点 |
|---------|---------|------|--------|
| **3D Full Attention** | Wan2.1, VideoLDM | 时空全局注意力 | 效果好，但计算量O(N²) |
| **Factorized Attention** | Video Swin Transformer | 时间/空间分离 | 效率更高，但可能损失时序精度 |
| **Causal 3D Conv** | TECO, VideoGPT | 因果卷积 | 天然时序因果，但感受野有限 |
| **Spatial + Temporal** | TimeSformer | 先空间后时间注意力 | 平衡效率和效果 |
| **Window Attention** | Video Swin | 局部窗口注意力 | 计算效率高 |

### 14.3 详细对比

```
【架构A: Wan2.1 (Full 3D Attention)】
  所有帧的所有位置可以互相看到
  计算: O((F×H×W)²)
  适合: 高质量生成，短序列(81帧)

【架构B: Factorized (分解式)】
  Step 1: 每帧内部做Spatial Self-Attention
  Step 2: 空间位置做Temporal Self-Attention
  计算: O(F×H²×W² + H×W×F²)
  适合: 长视频，计算资源受限

【架构C: Causal (自回归)】
  第t帧只能看到第0到t-1帧
  计算: O(F×H×W) per frame
  适合: 实时生成，但并行度低

【架构D: Window (窗口式)】
  只在局部窗口内做Attention
  计算: O((window_size)² × num_windows)
  适合: 超长视频，但可能丢失全局关系
```

### 14.4 Wan2.1 为什么选择 Full 3D？

```
原因1: 质量优先
  Full Attention能建模任意两个位置的直接关系
  对于复杂运动（如遮挡、快速移动）更鲁棒

原因2: 相对较短的序列
  81帧 × 30×52 patches = 32760序列长度
  虽然大，但Flash Attention优化后可行

原因3: Flow Matching 训练
  需要全局一致的去噪轨迹
  Full Attention更容易保持全局一致性
```

---

## Q15: 注意力层输出的具体作用和含义？

### 15.1 Self-Attention 的输出

```python
# Self-Attention 计算回顾
Q = W_q @ x  # [32760, 1536]
K = W_k @ x  # [32760, 1536]
V = W_v @ x  # [32760, 1536]

# 注意力权重
A = softmax(Q @ K.T / sqrt(d))  # [32760, 32760]

# 输出
O = A @ V  # [32760, 1536]
```

### 15.2 输出的物理含义

```
【形象理解】
输入: 32760个"待处理的patch请求"
输出: 32760个"处理后的patch响应"

每个输出位置i:
  O[i] = Σ_j A[i,j] × V[j]

含义:
  - O[i] 是输入位置i的新表示
  - 它是所有输入位置V[j]的加权平均
  - 权重A[i,j]表示"位置i对位置j的关注程度"

【具体例子】
假设位置i是"第5帧的猫的耳朵"
位置j是"第6帧的猫的耳朵"(时间+1, 空间相近)

那么:
  - A[i,j] 会很高 (因为3D RoPE使它们位置编码相似)
  - O[i] 会包含很多来自j的信息
  - 结果: i的表示会倾向于与j一致(但时间+1)
  - 效果: 模型学到了"耳朵应该移动到哪里"
```

### 15.3 Cross-Attention 的输出

```
输入:
  - Q: 视觉位置 [32760, 1536] "我需要什么语义?"
  - K: 文本token [512, 1536] "我提供这些语义"
  - V: 文本token [512, 1536]

输出: [32760, 1536]

【物理含义】
每个视觉位置的输出是文本token的加权组合
权重由Q和K的相似度决定

【例子】
视觉位置i: "看起来像猫脸的位置"
文本token: "猫", "草地", "奔跑", ...

计算:
  A[i, "猫"] = high (Q[i]与K["猫"]相似)
  A[i, "草地"] = low
  O[i] ≈ V["猫"] × high + V["草地"] × low + ...

结果:
  O[i] 编码了"猫"的语义特征
  后续层会根据这个语义生成具体的猫脸
```

### 15.4 注意力的作用总结

| 注意力类型 | 作用 | 输出含义 |
|-----------|------|---------|
| Self-Attention | 视觉内部的信息交换 | "这个位置应该参考其他哪些位置" |
| Cross-Attention | 文本到视觉的信息注入 | "这个位置应该具有什么语义" |

---

## Q16: SAE 跨时间步解释的意义？时间步无关的 SAE 架构？

### 16.1 跨时间步 SAE 的问题

```
【当前做法的问题】
为每个时间步单独训练SAE:
  SAE_t1, SAE_t2, ..., SAE_t50

问题:
  1. 参数量大 (50个SAE)
  2. 特征无法比较 (不同SAE的基不同)
  3. 没有利用时间步之间的连续性
```

### 16.2 跨时间步 SAE 的意义

```
【核心假设】
同一个概念（如"猫"、"奔跑"）在不同时间步应该有相似的内部表示

【意义1: 概念一致性分析】
如果"猫"在第10步激活特征[3, 15, 28]
那么在第20步也应该激活相似的特征
→ 可以追踪概念的演化

【意义2: 干预实验】
如果在第10步抑制"跳跃"特征
可以看到第20、30步的"跳跃"动作如何被影响
→ 理解时间因果关系

【意义3: 压缩存储】
一个SAE覆盖所有时间步
→ 统一的概念字典
```

### 16.3 时间步无关的 SAE 创新架构

#### 方案1: 时间步条件 SAE (Timestep-Conditioned SAE)

```python
class TimestepAgnosticSAE(nn.Module):
    """
    时间步无关SAE: 同一个编码器，时间步作为条件输入
    """
    def __init__(self, d_model=1536, d_hidden=3072):
        super().__init__()

        # 共享的编码器
        self.encoder = nn.Linear(d_model, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)

        # 时间步调制（轻量级）
        self.time_mod = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, d_hidden)
        )

    def encode(self, x, timestep=None):
        # 基础编码
        z = F.relu(self.encoder(x))

        if timestep is not None:
            # 时间步调制（加法而非独立的编码器）
            t_mod = self.time_mod(timestep.view(-1, 1))
            z = z + t_mod  # 简单加法，保持基向量一致

        # Top-K稀疏
        return topk_sparsify(z, k=32)

    def forward(self, x, timestep=None, return_loss=True):
        z, _, _ = self.encode(x, timestep)
        x_hat = self.decoder(z)

        if return_loss:
            return x_hat, z, F.mse_loss(x_hat, x)
        return x_hat, z

# 优势:
# - 所有时间步共享 decoder
# - 时间步只影响编码，不影响基的语义
# - 可以比较不同时间步的激活
```

#### 方案2: 标准化 SAE (Timestep-Normalized SAE)

```python
class TimestepNormalizedSAE(nn.Module):
    """
    先去除时间步影响，再应用SAE
    """
    def __init__(self, d_model=1536, d_hidden=3072):
        super().__init__()

        # 时间步影响估计器
        self.time_effect = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, d_model)
        )

        # 时间步无关的SAE
        self.sae = SparseAutoEncoder(d_model, d_hidden)

    def forward(self, x, timestep, return_loss=True):
        # 估计时间步的影响
        t_effect = self.time_effect(timestep.view(-1, 1))

        # 去除时间步影响
        x_normalized = x - t_effect

        # 应用标准SAE
        z, x_hat, loss = self.sae(x_normalized, return_loss)

        # 重建时加回时间步影响
        x_hat = x_hat + t_effect

        return x_hat, z, loss

# 优势:
# - SAE只学习"内容"，不学习"时间步风格"
# - 不同时间步的特征在同一空间可比
```

#### 方案3: 解耦表示 SAE (Disentangled SAE)

```python
class DisentangledSAE(nn.Module):
    """
    显式解耦内容和时间步
    """
    def __init__(self, d_model=1536):
        super().__init__()

        # 内容编码器
        self.content_encoder = nn.Linear(d_model, 2048, bias=False)
        self.content_decoder = nn.Linear(2048, d_model, bias=False)

        # 时间步编码器（轻量）
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, d_model)
        )

    def encode(self, x, timestep):
        # 去除时间步影响
        t = self.time_encoder(timestep.view(-1, 1))
        x_content = x - t

        # 编码内容
        z = F.relu(self.content_encoder(x_content))
        z_sparse, _, _ = topk_sparsify(z, k=32)

        return z_sparse, t

    def forward(self, x, timestep, return_loss=True):
        z_content, t_effect = self.encode(x, timestep)

        # 重建
        x_content_hat = self.content_decoder(z_content)
        x_hat = x_content_hat + t_effect

        if return_loss:
            loss = F.mse_loss(x_hat, x)
            return x_hat, z_content, loss
        return x_hat, z_content

# 优势:
# - 显式分离内容和时间步
# - 内容表示 z_content 完全时间步无关
# - 可以独立分析内容和时间步的影响
```

### 16.4 推荐的简单高效方案

```python
"""
对于24GB显存，推荐以下方案:

1. 使用 TimestepAgnosticSAE (方案1)
   - 简单: 只增加64维的时间调制
   - 高效: 所有时间步共享大部分参数
   - 有效: 实验表明加法调制足够

2. 训练策略
   - 从所有时间步采样特征
   - 随机输入时间步
   - 学习统一的内容基

3. 分析时使用
   - 固定内容编码器
   - 变化时间步，观察哪些特征激活变化
   - 找出"时间步敏感"和"时间步不变"的特征
"""

# 最小实现
class SimpleTimeAgnosticSAE(nn.Module):
    def __init__(self, d_model=1536, d_hidden=3072):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)

        # 轻量级时间调制
        self.time_alpha = nn.Parameter(torch.zeros(d_hidden))
        self.time_beta = nn.Parameter(torch.zeros(d_hidden))

    def forward(self, x, t):
        z = F.relu(self.encoder(x))

        # 简单的仿射变换: 随时间步缩放和偏移
        # t ∈ [0, 1]
        z = z * (1 + self.time_alpha * t.view(-1, 1))
        z = z + self.time_beta * t.view(-1, 1)

        z_sparse, _, _ = topk_sparsify(z, k=32)
        x_hat = self.decoder(z_sparse)

        return x_hat, z_sparse, F.mse_loss(x_hat, x)
```

---

**文档结束**

*本补充文档详细回答了关于adaLN-Zero、3D RoPE、注意力机制、以及时间步无关SAE的问题。*
