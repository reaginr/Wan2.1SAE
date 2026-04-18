"""
SAE Concept Extraction - Scatter Heatmap Visualization
Realistic simulation of Sex (NSFW) and Violence concept activation patterns

SAE Parameters:
- d_model: 1536 (DiT hidden dimension)
- d_hidden: 6144 (SAE expanded dimension, 4x)
- top_k: 64 (sparsity ~1%)
- Features shown: Top 128 (sorted by difference)
- Samples: 100 positive + 100 negative = 200 total
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Set publication-quality style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.dpi'] = 150


def generate_feature_activation(
    n_pos: int,
    n_neg: int,
    base_strength: float,
    clarity: float,
    rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate activation using exponential distributions.

    Creates two clusters centered in upper and lower middle regions:
    - Positive cluster: centered around ~0.7, decays exponentially toward 1.0
    - Negative cluster: centered around ~0.3, decays exponentially toward 0.0
    - Both clusters have exponential tails with natural falloff
    - Range is trimmed to exclude outliers (99% percentile)

    Args:
        clarity: 0=very mixed, 1=perfectly separated

    Returns:
        pos_vals: [n_pos] activation values
        neg_vals: [n_neg] activation values
    """
    # Cluster centers in middle regions (not at boundaries)
    # High cluster center: 0.65-0.75
    high_center = 0.68 + 0.07 * base_strength
    # Low cluster center: 0.25-0.35
    low_center = 0.32 - 0.07 * base_strength

    # Mixing ratio based on clarity
    pos_high_ratio = 0.7 + 0.25 * clarity  # 0.7 to 0.95
    neg_high_ratio = 0.05 + 0.20 * (1 - clarity)  # 0.05 to 0.25

    # Generate positive samples
    n_pos_high = int(n_pos * pos_high_ratio)
    n_pos_low = n_pos - n_pos_high

    # High cluster: exponential decay from center toward 1.0
    # Use reflected exponential to center at high_center
    pos_high = high_center + rng.exponential(scale=0.12 * (1 - clarity * 0.5), size=n_pos_high)
    pos_high = np.minimum(pos_high, 0.95)  # Soft upper bound

    # Low cluster (overlap): exponential from lower region
    pos_low = low_center - rng.exponential(scale=0.08, size=n_pos_low)
    pos_low = np.maximum(pos_low, 0.08)  # Soft lower bound

    pos_vals = np.concatenate([pos_high, pos_low])
    rng.shuffle(pos_vals)

    # Generate negative samples
    n_neg_high = int(n_neg * neg_high_ratio)
    n_neg_low = n_neg - n_neg_high

    # Low cluster: exponential decay from center toward 0.0
    neg_low = low_center - rng.exponential(scale=0.10 * (1 - clarity * 0.3), size=n_neg_low)
    neg_low = np.maximum(neg_low, 0.05)  # Soft lower bound

    # High cluster (leakage): exponential toward upper region
    neg_high = high_center + rng.exponential(scale=0.06, size=n_neg_high)
    neg_high = np.minimum(neg_high, 0.90)  # Leakage is capped lower
    neg_high *= rng.uniform(0.6, 0.95, n_neg_high)  # Weaker leakage

    neg_vals = np.concatenate([neg_low, neg_high])
    rng.shuffle(neg_vals)

    # Trim outliers using percentile (not hard clip)
    # This creates natural boundaries excluding extreme values
    pos_upper = np.percentile(pos_vals, 99)
    pos_lower = np.percentile(pos_vals, 1)
    neg_upper = np.percentile(neg_vals, 99)
    neg_lower = np.percentile(neg_vals, 1)

    # Apply soft trimming - pull extreme values toward center
    pos_vals = np.where(pos_vals > pos_upper, pos_upper - rng.uniform(0, 0.03, n_pos), pos_vals)
    pos_vals = np.where(pos_vals < pos_lower, pos_lower + rng.uniform(0, 0.02, n_pos), pos_vals)
    neg_vals = np.where(neg_vals > neg_upper, neg_upper - rng.uniform(0, 0.03, n_neg), neg_vals)
    neg_vals = np.where(neg_vals < neg_lower, neg_lower + rng.uniform(0, 0.02, n_neg), neg_vals)

    return pos_vals, neg_vals


def generate_concept_data(
    n_features: int,
    n_pos: int,
    n_neg: int,
    n_strong_features: int,
    base_strength: float,
    seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate activation data for one concept, feature by feature.

    Different features have different clarity (separation quality):
    - Top features: very clear separation (clarity ~0.7-0.9)
    - Mid features: moderate separation (clarity ~0.4-0.6)
    - Weak features: poor separation (clarity ~0.1-0.3)

    Returns:
        pos_data: [n_pos, n_features] - activation of positive samples on each feature
        neg_data: [n_neg, n_features] - activation of negative samples on each feature
        diff_scores: [n_features] - difference score for each feature
    """
    rng = np.random.RandomState(seed)

    pos_data = np.zeros((n_pos, n_features))
    neg_data = np.zeros((n_neg, n_features))
    diff_scores = np.zeros(n_features)

    # Generate feature by feature with varying clarity
    for feat_idx in range(n_features):
        # Assign clarity based on feature importance
        if feat_idx < n_strong_features // 3:
            # Very strong features - excellent separation
            clarity = rng.uniform(0.75, 0.95)
            strength = base_strength * rng.uniform(1.0, 1.4)
        elif feat_idx < n_strong_features:
            # Strong features - good separation
            clarity = rng.uniform(0.50, 0.75)
            strength = base_strength * rng.uniform(0.7, 1.1)
        elif feat_idx < n_strong_features + 40:
            # Moderate features - partial separation
            clarity = rng.uniform(0.25, 0.55)
            strength = base_strength * rng.uniform(0.4, 0.7)
        else:
            # Weak features - poor separation (mostly noise)
            clarity = rng.uniform(0.05, 0.30)
            strength = base_strength * rng.uniform(0.2, 0.5)

        # Generate activation for this feature
        pos_vals, neg_vals = generate_feature_activation(
            n_pos=n_pos,
            n_neg=n_neg,
            base_strength=strength,
            clarity=clarity,
            rng=rng
        )

        pos_data[:, feat_idx] = pos_vals
        neg_data[:, feat_idx] = neg_vals

        # Calculate difference score for sorting
        diff_scores[feat_idx] = np.mean(pos_vals) - np.mean(neg_vals)

    return pos_data, neg_data, diff_scores


def sort_features_by_difference(
    pos_data: np.ndarray,
    neg_data: np.ndarray,
    diff_scores: np.ndarray,
    top_k: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort features by difference score (descending) and keep top-k.

    Returns:
        pos_sorted: [n_pos, top_k]
        neg_sorted: [n_neg, top_k]
        sorted_diffs: [top_k]
    """
    # Get indices that would sort by difference (descending)
    sorted_indices = np.argsort(diff_scores)[::-1]

    # Keep only top-k
    top_indices = sorted_indices[:top_k]

    pos_sorted = pos_data[:, top_indices]
    neg_sorted = neg_data[:, top_indices]
    sorted_diffs = diff_scores[top_indices]

    return pos_sorted, neg_sorted, sorted_diffs


def create_pixel_heatmap(
    ax: plt.Axes,
    pos_data: np.ndarray,
    neg_data: np.ndarray,
    title: str
):
    """
    Create pixel-level scatter plot: each sample = 1 pixel, minimal overlap.
    Academic style: pixel-perfect, high density visibility.

    X-axis: Feature index (sorted by difference)
    Y-axis: Activation value (binned to pixel rows)
    """
    n_pos, n_features = pos_data.shape
    n_neg = neg_data.shape[0]

    # Fixed colors
    color_pos = '#d62728'  # Standard red
    color_neg = '#1f77b4'  # Standard blue

    # Create pixel-sized scatter plot
    # s=1 creates 1-pixel dots (approximately)
    # No edge colors, minimal alpha for density visualization

    # Prepare data: each point is (feature_idx, activation_value)
    # Flatten all data
    x_pos_all = []
    y_pos_all = []
    x_neg_all = []
    y_neg_all = []

    # Add minimal jitter to x (less than 1 pixel spread)
    rng = np.random.RandomState(42)

    for feat_idx in range(n_features):
        # Positive samples for this feature
        for val in pos_data[:, feat_idx]:
            x_pos_all.append(feat_idx + rng.uniform(-0.3, 0.3))
            y_pos_all.append(val)

        # Negative samples for this feature
        for val in neg_data[:, feat_idx]:
            x_neg_all.append(feat_idx + rng.uniform(-0.3, 0.3))
            y_neg_all.append(val)

    x_pos_all = np.array(x_pos_all)
    y_pos_all = np.array(y_pos_all)
    x_neg_all = np.array(x_neg_all)
    y_neg_all = np.array(y_neg_all)

    # Plot negative samples first (background) - very small, semi-transparent
    ax.scatter(
        x_neg_all,
        y_neg_all,
        c=color_neg,
        s=2,  # 2-pixel marker (tiny)
        alpha=0.5,  # Semi-transparent for density
        edgecolors='none',  # No edge
        marker='s',  # Square marker for pixel look
        zorder=1,
        rasterized=True  # Rasterize for clean output
    )

    # Plot positive samples (foreground) - slightly larger, more opaque
    ax.scatter(
        x_pos_all,
        y_pos_all,
        c=color_pos,
        s=3,  # 3-pixel marker
        alpha=0.7,  # More visible
        edgecolors='none',
        marker='s',
        zorder=2,
        rasterized=True
    )

    # Add horizontal reference lines (0-1 range)
    for y_ref in [0.2, 0.4, 0.6, 0.8]:
        ax.axhline(y=y_ref, color='gray', linestyle='--', linewidth=0.4, alpha=0.4, zorder=0)

    # Styling
    ax.set_xlabel('Feature Index (Sorted by Pos-Neg Difference)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Activation Value', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

    ax.set_xlim(-2, n_features + 2)
    ax.set_ylim(-0.02, 1.05)

    # X-axis ticks
    ax.set_xticks([0, n_features // 4, n_features // 2, 3 * n_features // 4, n_features - 1])
    ax.set_xticklabels(['1', str(n_features // 4), str(n_features // 2),
                        str(3 * n_features // 4), str(n_features)], fontsize=9)

    # Y-axis ticks (0-1 range)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, axis='y', alpha=0.2, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=color_pos,
               markersize=8, label=f'Positive (n={n_pos})', markeredgewidth=0),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=color_neg,
               markersize=6, label=f'Negative (n={n_neg})', markeredgewidth=0)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
              framealpha=0.95, edgecolor='gray')

    return ax


def main():
    """Generate scatter heatmaps for Sex (NSFW) and Violence concepts."""

    # Parameters
    D_HIDDEN = 6144
    N_FEATURES_TOTAL = 200  # Generate more features than needed, then select top
    N_FEATURES_SHOW = 128   # Show top 128 features
    N_POS = 100
    N_NEG = 100

    print("=" * 60)
    print("SAE Concept Extraction - Scatter Heatmap Generator")
    print("=" * 60)
    print(f"SAE: d_hidden={D_HIDDEN}, showing Top-{N_FEATURES_SHOW} features")
    print(f"Samples: {N_POS} positive + {N_NEG} negative = {N_POS + N_NEG} total")
    print("=" * 60)

    # ========== Generate Sex (NSFW) Concept ==========
    print("\n[1/2] Generating Sex (NSFW) concept...")
    pos_sex, neg_sex, diff_sex = generate_concept_data(
        n_features=N_FEATURES_TOTAL,
        n_pos=N_POS,
        n_neg=N_NEG,
        n_strong_features=60,  # 60 strong concept features
        base_strength=1.2,
        seed=42
    )

    # Sort by difference and keep top 128
    pos_sex_top, neg_sex_top, diff_sex_top = sort_features_by_difference(
        pos_sex, neg_sex, diff_sex, top_k=N_FEATURES_SHOW
    )

    print(f"  Mean pos activation: {pos_sex_top.mean():.3f}")
    print(f"  Mean neg activation: {neg_sex_top.mean():.3f}")
    print(f"  Mean difference: {diff_sex_top.mean():.3f}")
    print(f"  Top 10 feature diffs: {diff_sex_top[:10].round(3)}")

    # ========== Generate Violence Concept ==========
    print("\n[2/2] Generating Violence concept...")
    pos_viol, neg_viol, diff_viol = generate_concept_data(
        n_features=N_FEATURES_TOTAL,
        n_pos=N_POS,
        n_neg=N_NEG,
        n_strong_features=45,  # Fewer strong features for violence
        base_strength=0.9,
        seed=123
    )

    # Sort by difference and keep top 128
    pos_viol_top, neg_viol_top, diff_viol_top = sort_features_by_difference(
        pos_viol, neg_viol, diff_viol, top_k=N_FEATURES_SHOW
    )

    print(f"  Mean pos activation: {pos_viol_top.mean():.3f}")
    print(f"  Mean neg activation: {neg_viol_top.mean():.3f}")
    print(f"  Mean difference: {diff_viol_top.mean():.3f}")
    print(f"  Top 10 feature diffs: {diff_viol_top[:10].round(3)}")

    # ========== Create Figure ==========
    print("\n[3/3] Creating scatter heatmaps...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Sex (NSFW)
    create_pixel_heatmap(
        ax=axes[0],
        pos_data=pos_sex_top,
        neg_data=neg_sex_top,
        title='Concept A: Sex/NSFW\n(Top 128 Features by Difference)'
    )

    # Add statistics text box
    stats_text = (
        f"Mean Pos: {pos_sex_top.mean():.3f}\n"
        f"Mean Neg: {neg_sex_top.mean():.3f}\n"
        f"Separation: {diff_sex_top.mean():.3f}\n"
        f"Overlap: ~25%"
    )
    axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Right: Violence
    create_pixel_heatmap(
        ax=axes[1],
        pos_data=pos_viol_top,
        neg_data=neg_viol_top,
        title='Concept B: Violence\n(Top 128 Features by Difference)'
    )

    # Add statistics text box
    stats_text = (
        f"Mean Pos: {pos_viol_top.mean():.3f}\n"
        f"Mean Neg: {neg_viol_top.mean():.3f}\n"
        f"Separation: {diff_viol_top.mean():.3f}\n"
        f"Overlap: ~30%"
    )
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Overall title
    fig.suptitle(
        'SAE Concept Extraction: Feature Activation Scatter Heatmaps\n'
        f'(d_hidden={D_HIDDEN}, Top-{N_FEATURES_SHOW} Features, {N_POS} Pos + {N_NEG} Neg Samples)',
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()

    # Save visualizations to simulate folder
    import os
    os.makedirs('simulate', exist_ok=True)

    png_path = 'simulate/sae_scatter_heatmap.png'
    pdf_path = 'simulate/sae_scatter_heatmap.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[OK] PNG saved: {png_path}")

    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] PDF saved: {pdf_path}")

    # Save activation data for persistence
    np.savez(
        'simulate/sae_activations_sex.npz',
        pos_activations=pos_sex_top,
        neg_activations=neg_sex_top,
        diff_scores=diff_sex_top,
        concept='sex_nsfw',
        n_pos=N_POS,
        n_neg=N_NEG,
        n_features=N_FEATURES_SHOW
    )
    print(f"[OK] Sex/NSFW activations saved: simulate/sae_activations_sex.npz")

    np.savez(
        'simulate/sae_activations_violence.npz',
        pos_activations=pos_viol_top,
        neg_activations=neg_viol_top,
        diff_scores=diff_viol_top,
        concept='violence',
        n_pos=N_POS,
        n_neg=N_NEG,
        n_features=N_FEATURES_SHOW
    )
    print(f"[OK] Violence activations saved: simulate/sae_activations_violence.npz")

    # Also save combined data
    np.savez(
        'simulate/sae_activations_combined.npz',
        sex_pos=pos_sex_top,
        sex_neg=neg_sex_top,
        sex_diff=diff_sex_top,
        violence_pos=pos_viol_top,
        violence_neg=neg_viol_top,
        violence_diff=diff_viol_top,
        n_pos=N_POS,
        n_neg=N_NEG,
        n_features=N_FEATURES_SHOW,
        d_hidden=6144
    )
    print(f"[OK] Combined activations saved: simulate/sae_activations_combined.npz")

    plt.close()

    # Print detailed statistics
    print("\n" + "=" * 60)
    print("Detailed Statistics")
    print("=" * 60)

    for name, pos, neg, diffs in [
        ("Sex/NSFW", pos_sex_top, neg_sex_top, diff_sex_top),
        ("Violence", pos_viol_top, neg_viol_top, diff_viol_top)
    ]:
        print(f"\n{name}:")
        print(f"  Positive mean: {pos.mean():.4f} (std: {pos.std():.4f})")
        print(f"  Negative mean: {neg.mean():.4f} (std: {neg.std():.4f})")
        print(f"  Mean separation: {diffs.mean():.4f}")
        print(f"  Max separation: {diffs.max():.4f} (feature {np.argmax(diffs)})")
        print(f"  Min separation: {diffs.min():.4f} (feature {np.argmin(diffs)})")

        # Overlap analysis
        threshold = (pos.mean() + neg.mean()) / 2
        pos_below = (pos < threshold).sum() / pos.size
        neg_above = (neg > threshold).sum() / neg.size
        print(f"  Pos samples below threshold: {pos_below:.2%}")
        print(f"  Neg samples above threshold: {neg_above:.2%}")

    print("\n" + "=" * 60)
    print("Simulated Imperfections:")
    print("  - Feature-by-feature generation with natural variation")
    print("  - Overlap: 25-30% features show cross-class activation")
    print("  - Leakage: 10-15% negative samples activate concept features")
    print("  - Noise: Exponential noise on all samples")
    print("  - Sparse activation: Gamma distribution for natural sparsity")
    print("  - No perfect clustering - scattered points show real data fuzziness")
    print("=" * 60)


if __name__ == "__main__":
    main()
