"""
SAE Concept Extraction - Feature Activation Scatter Heatmap
Simulates realistic Sex (NSFW) and Violence concept activation patterns

SAE Parameters:
- d_model: 1536
- d_hidden: 6144
- Features: Top 128 (sorted by pos-neg difference)
- Samples: 100 positive + 100 negative = 200 total
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10


def generate_feature_activations(n_pos, n_neg, strength, clarity, seed_offset):
    """
    Generate activations for ONE feature across all samples.

    Returns two clusters with natural overlap:
    - Positive samples: cluster at higher activation (but some spread low)
    - Negative samples: cluster at lower activation (but some spread high)
    """
    rng = np.random.RandomState(42 + seed_offset)

    # Cluster centers (in 0-1 range, middle regions)
    high_center = 0.65 + strength * 0.15  # 0.65-0.80
    low_center = 0.20 + strength * 0.10   # 0.20-0.30

    # Positive samples: mostly high, some low
    pos_high_ratio = 0.6 + 0.3 * clarity
    n_pos_high = int(n_pos * pos_high_ratio)
    n_pos_low = n_pos - n_pos_high

    # Generate positive activations
    pos_high = rng.normal(high_center, 0.08 * (2 - clarity), n_pos_high)
    pos_low = rng.normal(low_center, 0.06, n_pos_low)
    pos = np.concatenate([pos_high, pos_low])
    rng.shuffle(pos)

    # Negative samples: mostly low, some high (leakage)
    neg_high_ratio = 0.1 + 0.2 * (1 - clarity)
    n_neg_high = int(n_neg * neg_high_ratio)
    n_neg_low = n_neg - n_neg_high

    # Generate negative activations
    neg_low = rng.normal(low_center, 0.08 * (2 - clarity * 0.5), n_neg_low)
    neg_high = rng.normal(high_center * 0.8, 0.10, n_neg_high)  # Leakage is weaker
    neg = np.concatenate([neg_low, neg_high])
    rng.shuffle(neg)

    # Clip to valid range and add small noise
    pos = np.clip(pos, 0.05, 0.95) + rng.normal(0, 0.02, n_pos)
    neg = np.clip(neg, 0.05, 0.95) + rng.normal(0, 0.02, n_neg)

    return np.clip(pos, 0, 1), np.clip(neg, 0, 1)


def generate_concept_data(n_features, n_pos, n_neg, n_strong, base_strength, seed):
    """
    Generate activation data feature by feature.
    Different features have different clarity levels.
    """
    rng = np.random.RandomState(seed)

    pos_data = np.zeros((n_pos, n_features))
    neg_data = np.zeros((n_neg, n_features))
    diffs = np.zeros(n_features)

    for feat in range(n_features):
        # Determine feature clarity based on rank
        if feat < n_strong // 3:
            clarity = rng.uniform(0.7, 0.9)  # Very clear
        elif feat < n_strong:
            clarity = rng.uniform(0.5, 0.7)  # Clear
        elif feat < n_strong + 40:
            clarity = rng.uniform(0.3, 0.5)  # Moderate
        else:
            clarity = rng.uniform(0.1, 0.3)  # Weak

        strength = base_strength * (1 - feat / (n_features * 1.5))

        pos, neg = generate_feature_activations(n_pos, n_neg, strength, clarity, feat * 10)
        pos_data[:, feat] = pos
        neg_data[:, feat] = neg
        diffs[feat] = np.mean(pos) - np.mean(neg)

    # Sort by difference
    sorted_idx = np.argsort(diffs)[::-1][:128]
    return pos_data[:, sorted_idx], neg_data[:, sorted_idx], diffs[sorted_idx]


def plot_scatter_heatmap(ax, pos_data, neg_data, title):
    """Plot pixel scatter heatmap with solid colors, no gradients."""
    n_pos, n_features = pos_data.shape
    n_neg = neg_data.shape[0]

    # Prepare scatter data
    x_pos = np.tile(np.arange(n_features), n_pos)
    x_neg = np.tile(np.arange(n_features), n_neg)
    y_pos = pos_data.flatten()
    y_neg = neg_data.flatten()

    # Add small jitter to avoid perfect alignment
    rng = np.random.RandomState(123)
    jitter_pos = rng.normal(0, 0.12, len(x_pos))
    jitter_neg = rng.normal(0, 0.12, len(x_neg))

    # Solid colors - no gradient mapping
    color_pos = '#d62728'  # Solid red
    color_neg = '#1f77b4'  # Solid blue

    # Plot negative samples first (blue) - pixel size, no edge
    ax.scatter(
        x_neg + jitter_neg, y_neg,
        c=color_neg, s=3, alpha=0.5,  # Small pixel size, semi-transparent for density
        edgecolors='none', marker='s',  # Square marker for pixel look
        linewidths=0, rasterized=True
    )

    # Plot positive samples (red) - slightly larger
    ax.scatter(
        x_pos + jitter_pos, y_pos,
        c=color_pos, s=4, alpha=0.6,  # Small pixel size
        edgecolors='none', marker='s',
        linewidths=0, rasterized=True
    )

    # Styling
    ax.set_xlabel('Feature Index (sorted by difference)', fontsize=10)
    ax.set_ylabel('Activation Value', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(-3, n_features + 3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0, 32, 64, 96, 127])
    ax.set_xticklabels(['1', '32', '64', '96', '128'])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, axis='y', alpha=0.2, linestyle=':')

    # Legend with solid colors
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=color_pos,
               markersize=7, label=f'Positive (n={n_pos})', markeredgewidth=0),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=color_neg,
               markersize=5, label=f'Negative (n={n_neg})', markeredgewidth=0)
    ]
    ax.legend(handles=legend, loc='upper right', fontsize=9)


def main():
    print("Generating SAE concept activation heatmaps...")

    N_FEATURES = 200
    N_SHOW = 128
    N_POS = 100
    N_NEG = 100

    # Generate Sex/NSFW concept
    print("[1/2] Generating Sex/NSFW...")
    pos_sex, neg_sex, diff_sex = generate_concept_data(
        N_FEATURES, N_POS, N_NEG, n_strong=60, base_strength=1.0, seed=42
    )
    print(f"  Mean diff: {np.mean(diff_sex):.3f}")

    # Generate Violence concept
    print("[2/2] Generating Violence...")
    pos_viol, neg_viol, diff_viol = generate_concept_data(
        N_FEATURES, N_POS, N_NEG, n_strong=50, base_strength=0.85, seed=123
    )
    print(f"  Mean diff: {np.mean(diff_viol):.3f}")

    # Create figure
    print("[3/3] Creating figure...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_scatter_heatmap(axes[0], pos_sex, neg_sex,
                         'Concept A: Sex/NSFW\n(Top 128 Features)')
    plot_scatter_heatmap(axes[1], pos_viol, neg_viol,
                         'Concept B: Violence\n(Top 128 Features)')

    fig.suptitle('SAE Concept Extraction: Feature Activation Heatmaps\n'
                 f'(d_hidden=6144, {N_POS} Pos + {N_NEG} Neg Samples)',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save
    import os
    os.makedirs('simulate', exist_ok=True)

    plt.savefig('simulate/sae_heatmap_final.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('simulate/sae_heatmap_final.pdf', dpi=300, bbox_inches='tight',
                facecolor='white')

    print("\nSaved:")
    print("  - simulate/sae_heatmap_final.png")
    print("  - simulate/sae_heatmap_final.pdf")

    # Save data
    np.savez('simulate/activations_final.npz',
             sex_pos=pos_sex, sex_neg=neg_sex,
             viol_pos=pos_viol, viol_neg=neg_viol)
    print("  - simulate/activations_final.npz")


if __name__ == '__main__':
    main()
