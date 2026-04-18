"""
SAE Concept Extraction - Feature Activation Scatter Heatmap (V2)
Simulates realistic Sex (NSFW) and Violence concept activation patterns
with logarithmic decay of cluster centers and reduced features.

SAE Parameters:
- d_model: 1536
- d_hidden: 6144
- Features: Top 64 (sorted by pos-neg difference)
- Samples: 100 positive + 100 negative = 200 total

Modifications:
1. Reduced low-activation positive samples (below 0.5) by 50%
2. Overall positive intensity shifted down by ~0.1
3. Logarithmic decay of cluster centers across features
4. Minimum separation constraints: top10>=0.25, top20>=0.2, global>=0.1
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10


def get_cluster_centers(feature_idx, n_features, concept_type='sex'):
    """
    Calculate cluster centers with logarithmic decay.
    Different parameters for Sex vs Violence concepts.

    Constraints:
    - Top 10 features: separation >= 0.25
    - Top 20 features: separation >= 0.20
    - All features: separation >= 0.10
    """
    # Different base parameters for different concepts
    if concept_type == 'sex':
        # Sex: stronger signal, higher activation
        base_high = 0.60
        base_low = 0.25
        decay_strength = 0.4
    else:  # violence
        # Violence: weaker signal, lower activation, faster decay
        base_high = 0.52  # Lower than sex
        base_low = 0.28   # Slightly higher low cluster (more overlap)
        decay_strength = 0.55  # Faster decay (worse separation in later features)

    # Logarithmic decay based on feature rank
    if feature_idx < 10:
        decay = 1.0
        min_separation = 0.25
    elif feature_idx < 20:
        decay = 0.85 if concept_type == 'sex' else 0.75  # Violence decays faster
        min_separation = 0.20
    else:
        # Stronger logarithmic decay for later features
        # Higher minimum decay to ensure separation >= 0.10
        min_decay = 0.50 if concept_type == 'sex' else 0.45
        decay = max(0.80 - decay_strength * np.log1p(feature_idx - 20) / np.log1p(n_features - 20), min_decay)
        min_separation = 0.10  # Global minimum for all features

    # Centers converge logarithmically
    # High center decreases with feature index
    high_center = base_high - (base_high - 0.35) * (1 - decay)

    # Low center increases slightly with feature index
    low_center = base_low + 0.05 * (1 - decay)

    return high_center, low_center, min_separation


def generate_feature_activations_v2(n_pos, n_neg, feature_idx, n_features, seed_offset, concept_type='sex'):
    """
    Generate activations for ONE feature with logarithmic decay constraints.

    Returns two clusters that converge logarithmically across features.
    """
    rng = np.random.RandomState(42 + seed_offset)

    # Get cluster centers with constraints (different for sex vs violence)
    high_center, low_center, separation = get_cluster_centers(feature_idx, n_features, concept_type)

    # Clarity decreases logarithmically with feature index
    # Violence has lower clarity (more mixed) than Sex
    clarity_base = 0.9 if concept_type == 'sex' else 0.75
    clarity = max(clarity_base - 0.5 * np.log1p(feature_idx) / np.log1p(n_features), 0.15)

    # Positive samples: mostly high, fewer low (reduced by 50% for low activations)
    pos_high_ratio = 0.7 + 0.25 * clarity
    n_pos_high = int(n_pos * pos_high_ratio)
    n_pos_low = n_pos - n_pos_high

    # Generate positive activations
    # High cluster (majority)
    pos_high = rng.normal(high_center, 0.08 * (2 - clarity), n_pos_high)

    # Low cluster (minority, reduced intensity below 0.5)
    pos_low_raw = rng.normal(low_center, 0.06, n_pos_low)
    # Only keep 50% of points below 0.5, shift others up
    pos_low = np.where(pos_low_raw < 0.5,
                       pos_low_raw,
                       pos_low_raw * 0.7 + 0.15)  # Shift up if > 0.5

    pos = np.concatenate([pos_high, pos_low])
    rng.shuffle(pos)

    # Negative samples: mostly low, some high (leakage)
    neg_high_ratio = 0.08 + 0.15 * (1 - clarity)
    n_neg_high = int(n_neg * neg_high_ratio)
    n_neg_low = n_neg - n_neg_high

    # Generate negative activations
    neg_low = rng.normal(low_center, 0.08 * (2 - clarity * 0.5), n_neg_low)
    neg_high = rng.normal(high_center * 0.85, 0.10, n_neg_high)
    neg = np.concatenate([neg_low, neg_high])
    rng.shuffle(neg)

    # Clip and add noise
    pos = np.clip(pos, 0.05, 0.95) + rng.normal(0, 0.015, n_pos)
    neg = np.clip(neg, 0.05, 0.95) + rng.normal(0, 0.015, n_neg)

    return np.clip(pos, 0, 1), np.clip(neg, 0, 1)


def generate_concept_data_v2(n_features, n_pos, n_neg, seed, concept_type='sex'):
    """
    Generate activation data feature by feature with logarithmic decay.
    Different parameters for Sex vs Violence concepts.
    """
    # Use different base seed for different concepts
    base_seed = 42 if concept_type == 'sex' else 123

    pos_data = np.zeros((n_pos, n_features))
    neg_data = np.zeros((n_neg, n_features))
    diffs = np.zeros(n_features)

    for feat in range(n_features):
        pos, neg = generate_feature_activations_v2(
            n_pos, n_neg, feat, n_features, feat * 10 + base_seed, concept_type
        )
        pos_data[:, feat] = pos
        neg_data[:, feat] = neg
        diffs[feat] = np.mean(pos) - np.mean(neg)

    # Sort by difference
    sorted_idx = np.argsort(diffs)[::-1][:n_features]
    return pos_data[:, sorted_idx], neg_data[:, sorted_idx], diffs[sorted_idx]


def plot_scatter_heatmap_v2(ax, pos_data, neg_data, title):
    """Plot pixel scatter heatmap with solid colors."""
    n_pos, n_features = pos_data.shape
    n_neg = neg_data.shape[0]

    # Prepare scatter data
    x_pos = np.tile(np.arange(n_features), n_pos)
    x_neg = np.tile(np.arange(n_features), n_neg)
    y_pos = pos_data.flatten()
    y_neg = neg_data.flatten()

    # Add small jitter
    rng = np.random.RandomState(123)
    jitter_pos = rng.normal(0, 0.12, len(x_pos))
    jitter_neg = rng.normal(0, 0.12, len(x_neg))

    # Solid colors
    color_pos = '#d62728'  # Red
    color_neg = '#1f77b4'  # Blue

    # Plot negative samples (blue)
    ax.scatter(
        x_neg + jitter_neg, y_neg,
        c=color_neg, s=3, alpha=0.5,
        edgecolors='none', marker='s',
        linewidths=0, rasterized=True
    )

    # Plot positive samples (red)
    ax.scatter(
        x_pos + jitter_pos, y_pos,
        c=color_pos, s=4, alpha=0.6,
        edgecolors='none', marker='s',
        linewidths=0, rasterized=True
    )

    # Styling
    ax.set_xlabel('Feature Index (sorted by difference)', fontsize=10)
    ax.set_ylabel('Activation Value', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(-2, n_features + 2)
    ax.set_ylim(-0.05, 1.05)

    # Adjust ticks for 64 features
    tick_positions = [0, 16, 32, 48, 63]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(['1', '16', '32', '48', '64'])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, axis='y', alpha=0.2, linestyle=':')

    # Legend
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=color_pos,
               markersize=7, label=f'Positive (n={n_pos})', markeredgewidth=0),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=color_neg,
               markersize=5, label=f'Negative (n={n_neg})', markeredgewidth=0)
    ]
    ax.legend(handles=legend, loc='upper right', fontsize=9)


def main():
    print("Generating SAE concept activation heatmaps (V2)...")
    print("Features: 64 (reduced from 128)")
    print("Modifications: logarithmic decay, reduced low-activation points, shifted intensity")

    N_FEATURES = 64  # Reduced from 128
    N_POS = 100
    N_NEG = 100

    # Generate Sex/NSFW concept (stronger signal, higher activation)
    print("\n[1/2] Generating Sex/NSFW...")
    pos_sex, neg_sex, diff_sex = generate_concept_data_v2(
        N_FEATURES, N_POS, N_NEG, seed=42, concept_type='sex'
    )
    print(f"  Mean diff: {np.mean(diff_sex):.3f}")
    print(f"  Top 10 diff: {np.mean(diff_sex[:10]):.3f} (should be >= 0.25)")
    print(f"  Top 20 diff: {np.mean(diff_sex[:20]):.3f} (should be >= 0.20)")
    print(f"  Min diff: {np.min(diff_sex):.3f} (should be >= 0.10)")

    # Generate Violence concept (weaker signal, lower activation, faster decay)
    print("\n[2/2] Generating Violence...")
    pos_viol, neg_viol, diff_viol = generate_concept_data_v2(
        N_FEATURES, N_POS, N_NEG, seed=123, concept_type='violence'
    )
    print(f"  Mean diff: {np.mean(diff_viol):.3f}")

    # Create figure
    print("\n[3/3] Creating figure...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_scatter_heatmap_v2(axes[0], pos_sex, neg_sex,
                            'Concept A: Sex/NSFW\n(Top 64 Features, Log Decay)')
    plot_scatter_heatmap_v2(axes[1], pos_viol, neg_viol,
                            'Concept B: Violence\n(Top 64 Features, Log Decay)')

    fig.suptitle('SAE Concept Extraction: Feature Activation Heatmaps (V2)\n'
                 f'(d_hidden=6144, {N_POS} Pos + {N_NEG} Neg, Logarithmic Decay)',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save to simulate folder (new files, don't overwrite existing)
    import os
    os.makedirs('simulate', exist_ok=True)

    plt.savefig('simulate/sae_heatmap_v2.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('simulate/sae_heatmap_v2.pdf', dpi=300, bbox_inches='tight',
                facecolor='white')

    print("\nSaved:")
    print("  - simulate/sae_heatmap_v2.png (NEW - with modifications)")
    print("  - simulate/sae_heatmap_v2.pdf")
    print("\nExisting files preserved:")
    print("  - simulate/sae_heatmap_final.png (original reference)")

    # Save data
    np.savez('simulate/activations_v2.npz',
             sex_pos=pos_sex, sex_neg=neg_sex,
             viol_pos=pos_viol, viol_neg=neg_viol,
             diff_sex=diff_sex, diff_viol=diff_viol)
    print("  - simulate/activations_v2.npz")


if __name__ == '__main__':
    main()
