SAFREE 是一项创新性的、无需训练（training-free）且自适应（adaptive）的文本到图像（T2I）和文本到视频（T2V）生成安全防护方法。它旨在解决现有安全生成方法（如“去学习”或模型编辑）在即时概念移除、数据依赖性以及可能导致模型权重改变进而影响生成质量的问题。SAFREE 的核心思想是在不修改预训练模型权重的情况下，通过同时在文本嵌入空间和视觉潜在空间进行不安全内容过滤，来确保生成内容的安全性、保真度和高质量。

该方法的核心技术细节如下：

1.  **基于毒性概念子空间的自适应 Token 选择 (Adaptive Token Selection Based on Toxic Concept Subspace Proximity)**:
    SAFREE 首先定义一个毒性概念子空间 $C \in \mathbb{R}^{D \times K}$，该子空间由一系列用户定义的毒性关键词的嵌入向量 $c_0, c_1, ..., c_{K-1}$ 的列向量拼接而成。为了识别输入提示中与毒性概念相关的 Token，该方法通过掩盖（masking out）提示中的每个 Token 来计算其余 Token 嵌入的平均值，得到 pooled input embedding $p_{\setminus i}$。接着，计算 $p_{\setminus i}$ 在 $C$ 上的正交投影，其残差向量 $d_{\setminus i}$ 表示 $p_{\setminus i}$ 与子空间 $C$ 的正交分量，定义为：
    $d_{\setminus i} = p_{\setminus i} - Cz = (I - P_C) p_{\setminus i}$
    其中 $P_C = C (C^T C)^{-1} C^T$ 是向子空间 $C$ 投影的矩阵。较长的残差向量 $d_{\setminus i}$ 表明被移除的 Token 与毒性概念更强相关。通过比较每个 Token 掩盖后的嵌入与 $C$ 的距离，以及该组距离的平均值，确定需要过滤的 Token。具体地，如果某个 Token 对应的 $||d_{\setminus i}||_2 > (1 + \alpha) \cdot \text{mean}(D(p|C).\text{delete}(i))$，则将其标记为需要处理的 Token。

2.  **通过概念正交 Token 投影实现安全生成 (Safe Generation via Concept Orthogonal Token Projection)**:
    为了避免直接移除或替换不安全 Token 导致的语义不连贯和生成质量下降，SAFREE 提出将检测到的不安全 Token 嵌入投影到一个既与毒性概念子空间 $C$ 正交，又保留在原始输入空间 $I$ 内的“安全”空间。投影矩阵 $P_I = I (I^T I)^{-1} I^T$ 将 Token 嵌入投影到输入空间。最终的安全投影嵌入 $p_{safe}$ 的计算公式为：
    $p_{proj} = P_I (I - P_C) p$
    $p_{safe} = m \odot p_{proj} + (1 - m) \odot p$
    其中 $m$ 是一个掩码向量，指示哪些 Token 被检测为毒性（$m_i=1$），$\odot$ 表示元素级乘法。这样，只有被标记的毒性 Token 嵌入才会被投影，而其他安全 Token 则保持不变，从而最大程度地保留了原始提示的完整性和语义。

3.  **通过自验证过滤机制自适应控制安全防护强度 (Adaptive Control of Safeguard Strengths with Self-Validating Filtering)**:
    为了平衡过滤毒性与保留原始生成能力之间的权衡，SAFREE 引入了一个自验证过滤机制。它根据原始输入嵌入 $p$ 和投影后的嵌入 $p_{proj}$ 之间的余弦相似度，动态调整应用过滤嵌入的去噪步数 $t'$：
    $t' = \gamma \cdot \text{sigmoid}(1 - \text{cos}(p, p_{proj}))$
    最终，在去噪过程中，如果当前步数 $t \le \text{round}(t')$，则使用安全嵌入 $p_{safe}$；否则，使用原始嵌入 $p$。这种机制确保了只在必要时才加强过滤，避免了过度过滤对无害内容质量的影响。

4.  **傅里叶域中的自适应潜在空间重注意力 (Adaptive Latent Re-attention in Fourier Domain)**:
    考虑到不安全内容通常在像素层面呈现区域性特征，SAFREE 进一步在扩散模型的潜在空间中引入视觉过滤策略。它利用傅里叶变换将潜在特征分解为频率分量。由于低频分量通常捕获图像的全局结构和风格，SAFREE 会衰减来自过滤后的提示嵌入 $p_{safree}$ 的低频特征，同时保留与原始提示 $p$ 更一致的视觉区域。具体地，对于潜在特征 $h(\cdot)$，傅里叶变换后得到 $F(p)$ 和 $F(p_{safree})$。
    $F(p) = b \odot \text{FFT}(h(p))$
    $F(p_{safree}) = b \odot \text{FFT}(h(p_{safree}))$
    其中 $b$ 是对应于低频分量的二元掩码。然后，对 $F(p_{safree})$ 的分量进行调整：
    $F'_i = \begin{cases} s \cdot F(p_{safree})_i & \text{if } F(p_{safree})_i > F(p)_i \\ F(p_{safree})_i & \text{otherwise} \end{cases}$
    其中 $s < 1$。最后，通过逆傅里叶变换得到精炼后的特征 $h' = \text{IFFT}(F')$。此过程有效抑制了潜在空间中与不安全概念相关的低频信息，同时避免了过度平滑，确保了生成图像的质量和细节。

SAFREE 适用于多种 T2I 骨干模型（如 SDXL、SD-v3）和 T2V 模型（如 ZeroScopeT2V、CogVideoX），无需额外训练或模型修改，展现了其强大的通用性和灵活性。实验结果表明，SAFREE 在对抗性攻击下的攻击成功率（ASR）显著低于其他无需训练的方法，并且与需要训练的方法相比也具有竞争力，同时保持了高质量的生成输出。

---

## 在Wan2.1 T2V模型上的应用与实现细节

### 与现有SAE概念提取方法的对比

| 特性 | SAFREE | SAE概念提取（当前实现） |
|------|--------|------------------------|
| **干预空间** | 文本嵌入空间 + 视觉潜在空间 | SAE隐空间（DiT中间层） |
| **是否需要训练** | 无需训练 | SAE需预训练，概念提取无需训练 |
| **概念表示** | 毒性关键词的文本嵌入拼接 | SAE学习到的稀疏特征 |
| **干预方式** | 投影到正交子空间 | 概念向量减法抑制 |
| **自适应调整** | 根据余弦相似度动态调整 | 固定阈值或人工调整 |
| **视觉过滤** | 傅里叶域低频衰减 | 基于patch的SAE特征干预 |

### 融合方案设计

```
现有SAE概念提取流程：
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  阶段一：    │───→│  阶段二：    │───→│  阶段三：    │
│ 采集SAE激活  │    │ 提取概念向量 │    │ 实时干预生成 │
└─────────────┘    └─────────────┘    └─────────────┘

融合SAFREE后的增强流程：
┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐
│  阶段一：    │───→│  阶段二：    │───→│      阶段三：增强干预    │
│ 采集SAE激活  │    │ 提取概念向量 │    │  ┌───────┐  ┌────────┐ │
└─────────────┘    └─────────────┘    │  │SAE干预│  │SAFREE  │ │
                                       │  │(中层) │  │(文本+ │ │
                                       │  └───────┘  │ 视觉)  │ │
                                       │       ↓     └────────┘ │
                                       │  ┌──────────────────┐  │
                                       │  │   融合决策模块    │  │
                                       │  │ (加权/顺序/条件)  │  │
                                       │  └──────────────────┘  │
                                       └─────────────────────────┘
```

### 消融实验设计建议

对于消融实验，**推荐采用顺序递进式融合**：

#### 方案A：独立对比（推荐用于论文）

```python
# Baseline 1: 无防护
model.generate(prompt)  # 原始生成

# Baseline 2: 仅SAFREE文本过滤
safree_embed = safree_text_filter(prompt, toxic_concepts)
model.generate(prompt, context=safree_embed)

# Baseline 3: 仅SAE干预（你的现有方法）
concept_vec = load("sex_sae_layer15.npy")
model.generate_with_sae_intervention(prompt, concept_vec, strength=0.5)

# Ours: SAE + SAFREE 串联
safree_embed = safree_text_filter(prompt, toxic_concepts)  # 第一层
model.generate_with_dual_intervention(
    prompt,
    text_embed=safree_embed,      # SAFREE处理的文本
    sae_concept=concept_vec,       # SAE概念向量
    fusion_strategy="sequential"   # 顺序：先SAFREE文本，再SAE视觉
)

# Ours+: SAE + SAFREE 并联加权
model.generate_with_fusion(
    prompt,
    interventions={
        "safree_text": weight_t,
        "safree_visual": weight_v,
        "sae_layer15": weight_sae
    }
)
```

#### 方案B：模块化集成（推荐用于实际部署）

```python
# 在现有阶段三代码中增加SAFREE模块

class UnifiedSafetyIntervention:
    def __init__(self):
        self.sae_concepts = load_sae_concepts()      # 你的SAE概念向量
        self.safree_config = load_safree_config()    # SAFREE毒性关键词

    def intervene(self, dit_hidden, prompt_embed, step):
        # 1. SAFREE文本空间干预（如果启用）
        if self.use_safree_text:
            prompt_embed = self.safree_text_filter(prompt_embed)

        # 2. SAE视觉空间干预（你的现有方法）
        if self.use_sae:
            z = self.sae.encode(dit_hidden)
            risk = (z * self.sae_concepts).sum()
            if risk > threshold:
                z = z - strength * self.sae_concepts
                dit_hidden = self.sae.decode(z)

        # 3. SAFREE视觉空间干预（傅里叶域）
        if self.use_safree_visual:
            dit_hidden = self.safree_fourier_filter(dit_hidden)

        return dit_hidden, prompt_embed
```

### 关于毒性概念关键词提取

**不需要单独新增代码提取毒性概念关键词**，原因如下：

1. **概念重叠性**：你的SAE概念提取已经通过正负样本对比学习到了"sex"概念向量，这本身就包含了毒性语义信息

2. **双重表示**：
   ```python
   # SAFREE需要的毒性关键词 → 直接使用你的正负提示词差异
   toxic_keywords = extract_from_pos_neg_prompts(pos_file, neg_file)
   # 例如：从pos_prompt_1.txt中提取与neg差异大的词汇

   # 或者更简单的：直接使用你的类别名称
   safree_concepts = ["sex", "naked", "nude", ...]  # 根据类别定义
   ```

3. **融合表示优势**：
   ```
   SAFREE概念: 文本嵌入空间中的关键词向量 [K, 4096]（T5输出）
   SAE概念:    视觉潜在空间中的稀疏特征 [6144]（SAE d_hidden）

   两者互补：
   - SAFREE在输入端过滤文本 → 阻止概念进入DiT
   - SAE在传播中干预视觉 → 修正已传播的不良特征
   ```

**推荐做法**：
```python
# 复用现有的pos/neg提示词文件
# 从pos_prompt_1.txt中提取高频差异词作为SAFREE的C子空间

from collections import Counter
import jieba

def extract_safree_concepts(pos_file, neg_file, top_k=20):
    """从正负提示词差异中提取SAFREE概念关键词"""
    pos_words = set(jieba.cut(open(pos_file).read()))
    neg_words = set(jieba.cut(open(neg_file).read()))

    # 只在正样本中出现的词
    toxic_candidates = pos_words - neg_words

    # 统计频率
    pos_counter = Counter(jieba.cut(open(pos_file).read()))
    toxic_concepts = [w for w, _ in pos_counter.most_common(top_k)
                      if w in toxic_candidates]

    return toxic_concepts  # 这些就是SAFREE的C子空间关键词

# 提取的concepts直接用于构建SAFREE的C矩阵
C = [T5_embed(word) for word in extract_safree_concepts(pos_file, neg_file)]
C = torch.stack(C, dim=1)  # [4096, K] 毒性概念子空间
```

---

## 第四步详解：傅里叶域中的自适应潜在空间重注意力

### 核心思想

不安全内容在图像/视频中往往具有**区域性低频特征**（如大面积肤色、特定纹理）。傅里叶变换可以将潜在特征分解为不同频率分量，其中：
- **低频分量**：全局结构、颜色分布、大尺度模式
- **高频分量**：细节、边缘、纹理

SAFREE通过**衰减与毒性概念相关的低频分量**，同时保留高频细节，实现精准过滤。

### 数学原理

```python
# 输入：DiT的潜在特征 h [B, C, H, W]（视频是[B, C, T, H, W]）

# Step 1: 2D傅里叶变换
H = torch.fft.fft2(h)  # 复数张量 [B, C, H, W]
H_shift = torch.fft.fftshift(H)  # 低频移到中心

# Step 2: 创建低频掩码（中心区域）
# 低频对应傅里叶谱的中心区域
H, W = h.shape[-2:]
center_h, center_w = H // 2, W // 2
radius = min(H, W) // 8  # 低频半径，可调

Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))
dist = torch.sqrt((Y - center_h)**2 + (X - center_w)**2)
low_freq_mask = (dist < radius).float()  # [H, W]，中心为1，边缘为0

# Step 3: 分别处理两个版本的潜在特征
# h(p): 原始提示的潜在特征
# h(p_safree): 经过SAFREE文本过滤后的潜在特征

H_orig = torch.fft.fftshift(torch.fft.fft2(h_original))      # F(p)
H_safe = torch.fft.fftshift(torch.fft.fft2(h_safree))        # F(p_safree)

# Step 4: 自适应低频调整
H_refined = H_safe.clone()

# 对于低频区域
low_freq_region = low_freq_mask.bool()

# 条件衰减：如果安全版本的低频能量 > 原始版本，则衰减
condition = torch.abs(H_safe[low_freq_region]) > torch.abs(H_orig[low_freq_region])
# 衰减系数 s < 1
s = 0.7
H_refined[low_freq_region][condition] *= s

# Step 5: 逆傅里叶变换
H_refined = torch.fft.ifftshift(H_refined)
h_refined = torch.fft.ifft2(H_refined).real  # 取实部
```

### 直观理解

```
原始潜在特征 h(p):
┌─────────────────────┐
│  ▓▓▓ 低频（肤色）    │  ← 全局颜色/结构（不安全）
│  ▓▓▓  大面积区域     │
│  ░░░ 高频（细节）    │  ← 边缘纹理（中性）
│  ░░░  保留          │
└─────────────────────┘

SAFREE处理后 h(p_safree):
┌─────────────────────┐
│  ▒▒▒ 低频衰减       │  ← 衰减为原来70%（抑制不安全颜色）
│  ▒▒▒  (s=0.7)       │
│  ░░░ 高频保留       │  ← 细节不受影响
│  ░░░                │
└─────────────────────┘
              ↓ IFFT
生成图像：
- 不再有大面积不安全肤色（低频被抑制）
- 但人物轮廓、表情细节仍然清晰（高频保留）
- 避免过度模糊，保持生成质量
```

### 与SAE方法的协同

```
┌─────────────────────────────────────────────────────┐
│                 Wan2.1 T2V 生成过程                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  文本输入 ──→ T5编码 ──→ 文本嵌入 [L, 4096]          │
│                            │                        │
│                    ┌───────┴───────┐               │
│                    ▼               ▼               │
│              SAFREE文本        SAE概念检测         │
│              投影过滤          (检测风险token)      │
│                    │               │               │
│                    └───────┬───────┘               │
│                            ▼                       │
│                    DiT去噪过程                      │
│                            │                       │
│                    ┌───────┴───────┐              │
│                    ▼               ▼              │
│              DiT输出层15       DiT输出层29         │
│            [B,1536,H,W]      [B,1536,H,W]         │
│                    │               │              │
│              SAE干预        SAFREE傅里叶过滤       │
│              (中层概念)      (视觉低频抑制)        │
│                    │               │              │
│                    └───────┬───────┘              │
│                            ▼                      │
│                    VAE解码 → 视频输出              │
│                                                     │
└─────────────────────────────────────────────────────┘

协同效果：
1. SAFREE文本层：阻止毒性概念进入DiT（预防）
2. SAE中层：在特征传播中检测并修正风险（监控）
3. SAFREE视觉层：在输出前衰减不安全的低频模式（兜底）
```

### 实现建议

在你的代码中，第四步可以在阶段三的干预模块中增加：

```python
# wan/sae/interpretability/intervention.py 新增

def fourier_domain_filter(latent, original_latent, scale=0.7):
    """
    SAFREE傅里叶域低频衰减

    Args:
        latent: 当前潜在特征（已干预）[B, C, H, W]
        original_latent: 原始潜在特征（未干预）[B, C, H, W]
        scale: 衰减系数 s < 1

    Returns:
        refined_latent: 精炼后的潜在特征
    """
    # FFT
    F_latent = torch.fft.fft2(latent)
    F_orig = torch.fft.fft2(original_latent)

    # 创建低频掩码（中心圆形区域）
    *_, H, W = latent.shape
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )
    radius = torch.sqrt(x**2 + y**2)
    low_freq_mask = (radius < 0.25).to(latent.device)  # 中心25%区域

    # 条件衰减
    F_refined = F_latent.clone()
    magnitude_latent = torch.abs(F_latent)
    magnitude_orig = torch.abs(F_orig)

    # 在低频区域，如果干预后的能量 > 原始，则衰减
    mask = low_freq_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    condition = (magnitude_latent > magnitude_orig) * mask

    F_refined = torch.where(
        condition.bool(),
        F_latent * scale,
        F_latent
    )

    # IFFT
    refined = torch.fft.ifft2(F_refined).real
    return refined
```

---

## 总结

| 问题 | 答案 |
|------|------|
| 是否需要单独提取毒性关键词？ | **不需要**，复用现有pos/neg提示词文件，或直接用类别名称定义关键词 |
| SAE和SAFREE能否同时进行？ | **可以**，推荐顺序融合：SAFREE文本 → SAE中层 → SAFREE视觉 |
| 消融实验设计 | Baseline: 无防护/仅SAFREE/仅SAE；Ours: SAE+SAFREE串联；Ours+: 并联加权 |
| 第四步作用 | 在视觉潜在空间的傅里叶域中，衰减与毒性概念相关的低频分量，保留高频细节 |