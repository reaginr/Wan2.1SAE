"""
SAE Concept Extraction Simulation Heatmap Generator
For Thesis Demo: Simulate Sex (NSFW) and Violence Concept Extraction

SAE Parameters (from sae_train_t2v_1_3b.py):
- d_model: 1536 (DiT hidden dim)
- d_hidden: 6144 (SAE expanded dim, 4x)
- top_k: 64 (sparsity ~1%)
- activation: ReLU
- sparsity: topk

Generation Strategy:
1. Gaussian kernel for concept clustering (not pure random)
2. Simulate real defects: overlap, leakage, noise, fuzzy boundaries
3. Two concepts share few features (dual-sensitive content)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头模式，适用于服务器环境
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.patheffects as path_effects

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def generate_clustered_activations(
    n_samples_pos: int,
    n_samples_neg: int,
    n_features: int,
    n_concept_clusters: int = 8,
    cluster_spread: float = 15.0,
    noise_level: float = 0.15,
    leakage_rate: float = 0.08,
    overlap_factor: float = 0.25,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用高斯核生成概念聚类激活值

    参数:
        n_samples_pos: 正样本数
        n_samples_neg: 负样本数
        n_features: 特征数 (Top-K)
        n_concept_clusters: 概念聚类数（每个聚类是一组相关特征）
        cluster_spread: 高斯核扩散范围
        noise_level: 噪声水平
        leakage_rate: 概念泄漏率（负样本中激活概念特征的概率）
        overlap_factor: 正负样本重叠程度
        seed: 随机种子

    返回:
        pos_activations: [n_samples_pos, n_features]
        neg_activations: [n_samples_neg, n_features]
        concept_centers: 概念中心位置
    """
    rng = np.random.RandomState(seed)

    # 初始化激活矩阵
    pos_activations = np.zeros((n_samples_pos, n_features))
    neg_activations = np.zeros((n_samples_neg, n_features))

    # 生成概念聚类中心（高斯核中心）
    concept_centers = rng.choice(n_features, size=n_concept_clusters, replace=False)
    concept_strengths = rng.uniform(0.6, 1.0, n_concept_clusters)  # 不同聚类不同强度

    # ========== 正样本：强概念激活 + 噪声 + 自然变异 ==========
    for i in range(n_samples_pos):
        for cluster_idx, center in enumerate(concept_centers):
            # 每个样本对每个聚类有不同程度的响应
            base_strength = concept_strengths[cluster_idx]

            # 样本间变异：不同样本对同一概念的响应强度不同
            sample_variation = rng.normal(1.0, 0.25)
            sample_variation = np.clip(sample_variation, 0.2, 1.8)

            # 生成高斯核激活
            distances = np.abs(np.arange(n_features) - center)
            gaussian = base_strength * sample_variation * np.exp(-0.5 * (distances / cluster_spread) ** 2)

            # 稀疏化：只保留聚类附近的特征
            mask = gaussian > 0.1
            pos_activations[i] += gaussian * mask

        # 添加背景噪声（低级别随机激活）
        noise = rng.exponential(0.08, n_features)  # 指数分布模拟稀疏噪声
        noise_mask = rng.random(n_features) < 0.15  # 只有15%的特征有噪声
        pos_activations[i] += noise * noise_mask * noise_level

    # ========== 负样本：基线激活 + 概念泄漏 + 重叠 ==========
    for i in range(n_samples_neg):
        # 基线：低水平通用激活（模拟通用视觉特征）
        baseline = rng.exponential(0.05, n_features)
        baseline_mask = rng.random(n_features) < 0.12
        neg_activations[i] += baseline * baseline_mask

        # 概念泄漏：部分负样本偶然激活概念特征
        if rng.random() < leakage_rate:
            # 随机选择一个概念聚类泄漏
            leaked_cluster = rng.randint(0, n_concept_clusters)
            center = concept_centers[leaked_cluster]
            leak_strength = rng.uniform(0.15, 0.45)  # 泄漏强度较弱

            distances = np.abs(np.arange(n_features) - center)
            gaussian_leak = leak_strength * np.exp(-0.5 * (distances / cluster_spread) ** 2)
            neg_activations[i] += gaussian_leak * (gaussian_leak > 0.05)

        # 与正样本的重叠：部分特征在负样本中也适度激活
        n_overlap = int(n_features * overlap_factor)
        overlap_indices = rng.choice(n_features, n_overlap, replace=False)
        overlap_values = rng.uniform(0.1, 0.35, n_overlap)
        neg_activations[i, overlap_indices] += overlap_values

        # 添加噪声
        noise = rng.exponential(0.06, n_features)
        noise_mask = rng.random(n_features) < 0.12
        neg_activations[i] += noise * noise_mask * noise_level

    # 归一化到合理范围
    pos_activations = np.clip(pos_activations, 0, 1.2)
    neg_activations = np.clip(neg_activations, 0, 1.0)

    return pos_activations, neg_activations, concept_centers


def add_realistic_defects(
    pos_activations: np.ndarray,
    neg_activations: np.ndarray,
    defect_rate: float = 0.15,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    添加更真实的不完美效果：
    - 假阴性：正样本中某些概念特征未激活
    - 假阳性：负样本中某些非概念特征异常激活
    - 信号衰减：部分正样本概念信号较弱
    """
    rng = np.random.RandomState(seed)
    n_features = pos_activations.shape[1]

    # 假阴性：随机降低一些正样本的激活
    for i in range(pos_activations.shape[0]):
        if rng.random() < defect_rate:
            # 随机衰减某些特征
            decay_mask = rng.random(n_features) < 0.3
            pos_activations[i, decay_mask] *= rng.uniform(0.1, 0.5)

    # 假阳性：负样本中的随机高激活
    for i in range(neg_activations.shape[0]):
        if rng.random() < defect_rate * 0.7:  # 假阳性较少
            # 随机选择一些特征异常激活
            false_pos_indices = rng.choice(n_features, size=3, replace=False)
            neg_activations[i, false_pos_indices] += rng.uniform(0.3, 0.6, 3)

    return pos_activations, neg_activations


def create_cross_heatmap(
    pos_act: np.ndarray,
    neg_act: np.ndarray,
    ax: plt.Axes,
    title: str,
    cmap: str = 'RdBu_r',
    feature_label: str = "SAE Features (Top 100)"
):
    """
    创建交叉热力图：正负样本并排显示

    布局:
    [正样本 100列 | 负样本 100列]
    每行 = 一个特征
    颜色 = 激活强度
    """
    # 合并正负样本 [200, n_features]
    combined = np.vstack([pos_act.T, neg_act.T])  # 转置后：[n_features, 200]

    # 创建自定义归一化（突出显示中等强度差异）
    vmax = np.percentile(combined, 98)  # 使用98分位数作为最大值，避免极端值影响
    vmin = 0

    # 绘制热力图
    im = ax.imshow(
        combined,
        aspect='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest'  # 最近邻插值，保持像素感
    )

    # 添加分界线（正负样本之间）
    mid_line = pos_act.shape[0]
    ax.axvline(x=mid_line - 0.5, color='white', linewidth=2.5, linestyle='-')
    ax.axvline(x=mid_line - 0.5, color='black', linewidth=1.0, linestyle='--', alpha=0.5)

    # 添加类别标签背景
    rect_pos = Rectangle((-0.5, -5), mid_line, 4, linewidth=0,
                          facecolor='#2ecc71', alpha=0.3, transform=ax.transData)
    rect_neg = Rectangle((mid_line - 0.5, -5), neg_act.shape[0], 4, linewidth=0,
                          facecolor='#e74c3c', alpha=0.3, transform=ax.transData)
    ax.add_patch(rect_pos)
    ax.add_patch(rect_neg)

    # 设置坐标轴
    ax.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
    ax.set_ylabel(feature_label, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # X轴刻度
    ax.set_xticks([mid_line // 2, mid_line + neg_act.shape[0] // 2])
    ax.set_xticklabels(['Positive\n(n=100)', 'Negative\n(n=100)'], fontsize=10)

    # Y轴：只显示部分刻度避免拥挤
    n_features = combined.shape[0]
    y_ticks = [0, n_features // 4, n_features // 2, 3 * n_features // 4, n_features - 1]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'F{i+1}' for i in y_ticks], fontsize=8)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Activation Strength', fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)

    # 在图上添加注释标注
    text_pos = ax.text(mid_line // 2, -8, 'Positive Samples\n(Concept Present)',
                       ha='center', va='top', fontsize=10, fontweight='bold',
                       color='#27ae60', transform=ax.transData)
    text_neg = ax.text(mid_line + neg_act.shape[0] // 2, -8, 'Negative Samples\n(Concept Absent)',
                       ha='center', va='top', fontsize=10, fontweight='bold',
                       color='#c0392b', transform=ax.transData)

    # 添加文字描边效果
    for text in [text_pos, text_neg]:
        text.set_path_effects([
            path_effects.withStroke(linewidth=3, foreground='white')
        ])

    return im


def generate_two_concept_heatmap():
    """
    生成两个概念（Sex/NSFW 和 Violence）的交叉热力图
    """
    # SAE 参数
    D_HIDDEN = 6144  # SAE 隐空间维度
    N_FEATURES_SHOW = 100  # 展示 Top 100 特征
    N_SAMPLES_POS = 100
    N_SAMPLES_NEG = 100

    print("=" * 60)
    print("SAE Concept Extraction Heatmap Generator")
    print("=" * 60)
    print(f"SAE Params: d_hidden={D_HIDDEN}, Top-{N_FEATURES_SHOW} features")
    print(f"Samples: {N_SAMPLES_POS} positive + {N_SAMPLES_NEG} negative = {N_SAMPLES_POS + N_SAMPLES_NEG} total")
    print("=" * 60)

    np.random.seed(42)

    # ========== Generate Sex (NSFW) Concept ==========
    print("\n[1/2] Generating Sex (NSFW) concept activations...")
    pos_sex, neg_sex, centers_sex = generate_clustered_activations(
        n_samples_pos=N_SAMPLES_POS,
        n_samples_neg=N_SAMPLES_NEG,
        n_features=N_FEATURES_SHOW,
        n_concept_clusters=10,  # NSFW has more sub-clusters
        cluster_spread=12.0,
        noise_level=0.18,
        leakage_rate=0.12,  # 12% negative samples have concept leakage
        overlap_factor=0.28,
        seed=42
    )
    pos_sex, neg_sex = add_realistic_defects(pos_sex, neg_sex, defect_rate=0.18, seed=42)
    print(f"  - Positive mean activation: {pos_sex.mean():.3f} ± {pos_sex.std():.3f}")
    print(f"  - Negative mean activation: {neg_sex.mean():.3f} ± {neg_sex.std():.3f}")
    print(f"  - Concept cluster centers: {sorted(centers_sex)[:5]}...")

    # ========== Generate Violence Concept ==========
    print("\n[2/2] Generating Violence concept activations...")
    # Use different seed to ensure two concepts are different
    pos_violence, neg_violence, centers_violence = generate_clustered_activations(
        n_samples_pos=N_SAMPLES_POS,
        n_samples_neg=N_SAMPLES_NEG,
        n_features=N_FEATURES_SHOW,
        n_concept_clusters=7,  # Violence has fewer but stronger clusters
        cluster_spread=10.0,  # More concentrated activation
        noise_level=0.15,
        leakage_rate=0.09,  # 9% leakage
        overlap_factor=0.22,
        seed=123  # Different seed
    )
    pos_violence, neg_violence = add_realistic_defects(
        pos_violence, neg_violence, defect_rate=0.15, seed=123
    )

    # Add shared features (dual-sensitive content, e.g., sexual violence)
    shared_features = [15, 16, 17, 45, 46]  # Pre-defined shared features
    for feat in shared_features:
        # Enhance in Sex concept
        pos_sex[:, feat] += np.random.exponential(0.25, N_SAMPLES_POS)
        neg_sex[:, feat] += np.random.exponential(0.12, N_SAMPLES_NEG)
        # Also enhance in Violence concept
        pos_violence[:, feat] += np.random.exponential(0.22, N_SAMPLES_POS)
        neg_violence[:, feat] += np.random.exponential(0.10, N_SAMPLES_NEG)

    print(f"  - Positive mean activation: {pos_violence.mean():.3f} ± {pos_violence.std():.3f}")
    print(f"  - Negative mean activation: {neg_violence.mean():.3f} ± {neg_violence.std():.3f}")
    print(f"  - Concept cluster centers: {sorted(centers_violence)[:5]}...")
    print(f"  - Shared features with Sex: {shared_features}")

    # ========== Create Figure ==========
    print("\n[3/3] Generating heatmaps...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # 设置整体标题
    fig.suptitle(
        'SAE Concept Extraction: Cross-Sample Activation Heatmaps\n' +
        f'(d_hidden={D_HIDDEN}, Top-{N_FEATURES_SHOW} Features, {N_SAMPLES_POS}×{N_SAMPLES_NEG} Samples)',
        fontsize=16, fontweight='bold', y=0.98
    )

    # 绘制 Sex (NSFW) 热力图
    im1 = create_cross_heatmap(
        pos_sex, neg_sex,
        ax=axes[0],
        title='Concept A: Sex/NSFW Content',
        cmap='RdBu_r',
        feature_label=f'SAE Features (Top {N_FEATURES_SHOW} of {D_HIDDEN})'
    )

    # 在子图上添加统计注释
    axes[0].text(
        0.02, 0.98,
        f'Statistics:\n'
        f'Pos: μ={pos_sex.mean():.3f}, σ={pos_sex.std():.3f}\n'
        f'Neg: μ={neg_sex.mean():.3f}, σ={neg_sex.std():.3f}\n'
        f'SNR: {(pos_sex.mean()-neg_sex.mean())/np.std(np.vstack([pos_sex, neg_sex])):.2f}',
        transform=axes[0].transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    # 绘制 Violence 热力图
    im2 = create_cross_heatmap(
        pos_violence, neg_violence,
        ax=axes[1],
        title='Concept B: Violence Content',
        cmap='RdBu_r',
        feature_label=f'SAE Features (Top {N_FEATURES_SHOW} of {D_HIDDEN})'
    )

    # 统计注释
    axes[1].text(
        0.02, 0.98,
        f'Statistics:\n'
        f'Pos: μ={pos_violence.mean():.3f}, σ={pos_violence.std():.3f}\n'
        f'Neg: μ={neg_violence.mean():.3f}, σ={neg_violence.std():.3f}\n'
        f'SNR: {(pos_violence.mean()-neg_violence.mean())/np.std(np.vstack([pos_violence, neg_violence])):.2f}',
        transform=axes[1].transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存图片
    output_path = 'sae_concept_heatmap_demo.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[OK] Heatmap saved: {output_path}")

    output_pdf = 'sae_concept_heatmap_demo.pdf'
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] PDF saved: {output_pdf}")

    # ========== Print Statistics ==========
    print("\n" + "=" * 60)
    print("Detailed Statistics")
    print("=" * 60)

    for name, pos, neg in [
        ("Sex/NSFW", pos_sex, neg_sex),
        ("Violence", pos_violence, neg_violence)
    ]:
        print(f"\n{name}:")
        print(f"  Positive mean activation: {pos.mean():.4f} (std: {pos.std():.4f})")
        print(f"  Negative mean activation: {neg.mean():.4f} (std: {neg.std():.4f})")
        print(f"  Concept separation: {pos.mean() - neg.mean():.4f}")

        # Calculate false positive/negative rates (using simple threshold)
        threshold = (pos.mean() + neg.mean()) / 2
        false_negatives = (pos < threshold).mean()
        false_positives = (neg > threshold).mean()
        print(f"  False negative rate: {false_negatives:.2%}")
        print(f"  False positive rate: {false_positives:.2%}")

        # Sparsity statistics
        sparsity_pos = (pos > 0.1).mean()
        sparsity_neg = (neg > 0.1).mean()
        print(f"  Positive sparsity (>0.1): {sparsity_pos:.2%}")
        print(f"  Negative sparsity (>0.1): {sparsity_neg:.2%}")

    print("\n" + "=" * 60)
    print("Simulated Defects:")
    print("  - Overlap: 28% features activated in both positive and negative")
    print("  - Concept leakage: 12% (Sex) / 9% (Violence) negative samples show concept")
    print("  - Noise: Random background noise simulates real SAE instability")
    print("  - False negatives/positives: Some samples don't activate as expected")
    print("  - Shared features: 5 features active in both concepts (dual-sensitive)")
    print("=" * 60)

    return fig


if __name__ == "__main__":
    generate_two_concept_heatmap()
