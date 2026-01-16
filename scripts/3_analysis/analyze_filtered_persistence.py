"""
Filtered Persistence Analysis: Separating Intra-Cluster Noise from Inter-Cluster Voids

This script applies filtering to H1 persistence features to separate:
- Intra-cluster "noise" loops (small features inside dense clusters)
- Inter-cluster voids (large features representing gaps between clusters)

The key insight: For tight clusters (small r_0), H1 death distributions show
two scales mixed together - a sharp peak near small r (noise) and a heavy
tail (true voids). Filtering by death time or persistence can separate these.

Filtering Strategies:
1. Death-time threshold: Remove features with death < threshold
2. Persistence threshold: Remove features with (death - birth) < min_persistence
3. Adaptive (r_0-based): Use cluster_std_dev as natural threshold
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy.stats import gaussian_kde

# ==============================================================================
# Load H1 Persistence Data
# ==============================================================================

def load_all_h1_data(config_name, base_dir='../../data/ensemble_data'):
    """Load all H1 persistence data for a given configuration."""
    config_dir = os.path.join(base_dir, config_name)

    with open(os.path.join(config_dir, 'persistence_metadata.json'), 'r') as f:
        metadata = json.load(f)

    n_samples = metadata['n_samples']

    all_births = []
    all_deaths = []
    all_persistence = []

    for i in range(n_samples):
        persistence_file = os.path.join(config_dir, f'sample_{i:03d}_persistence.npy')
        ph_data = np.load(persistence_file, allow_pickle=True).item()

        h1 = ph_data['h1']
        if len(h1) > 0:
            births = h1[:, 0]
            deaths = h1[:, 1]
            persistence = deaths - births

            all_births.extend(births)
            all_deaths.extend(deaths)
            all_persistence.extend(persistence)

    return {
        'births': np.array(all_births),
        'deaths': np.array(all_deaths),
        'persistence': np.array(all_persistence)
    }


def filter_by_death_threshold(h1_data, threshold):
    """Keep only features with death > threshold."""
    mask = h1_data['deaths'] > threshold
    return {
        'births': h1_data['births'][mask],
        'deaths': h1_data['deaths'][mask],
        'persistence': h1_data['persistence'][mask],
        'n_kept': np.sum(mask),
        'n_total': len(mask),
        'fraction_kept': np.sum(mask) / len(mask) if len(mask) > 0 else 0
    }


def filter_by_persistence_threshold(h1_data, min_persistence):
    """Keep only features with persistence > min_persistence."""
    mask = h1_data['persistence'] > min_persistence
    return {
        'births': h1_data['births'][mask],
        'deaths': h1_data['deaths'][mask],
        'persistence': h1_data['persistence'][mask],
        'n_kept': np.sum(mask),
        'n_total': len(mask),
        'fraction_kept': np.sum(mask) / len(mask) if len(mask) > 0 else 0
    }


def compute_statistics(values):
    """Compute summary statistics."""
    if len(values) == 0:
        return {'count': 0, 'mean': np.nan, 'median': np.nan, 'std': np.nan}
    return {
        'count': len(values),
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75))
    }


# ==============================================================================
# Main Analysis
# ==============================================================================

if __name__ == "__main__":

    print("="*80)
    print("Filtered Persistence Analysis")
    print("="*80)
    print()
    print("Goal: Separate intra-cluster noise from true inter-cluster voids")
    print()

    # Configuration setup
    config_names = ['stddev_0.2', 'stddev_0.4', 'stddev_0.6', 'stddev_1.0']
    r0_values = {'stddev_0.2': 0.2, 'stddev_0.4': 0.4, 'stddev_0.6': 0.6, 'stddev_1.0': 1.0}
    config_labels = ['r_0 = 0.2', 'r_0 = 0.4', 'r_0 = 0.6', 'r_0 = 1.0']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Create output directory
    output_dir = '../../outputs/figures/filtered_persistence'
    os.makedirs(output_dir, exist_ok=True)

    # Load all data
    print("Loading H1 persistence data...")
    all_h1_data = {}
    for config_name in config_names:
        print(f"  Loading {config_name}...")
        all_h1_data[config_name] = load_all_h1_data(config_name)

    # =========================================================================
    # Strategy 1: Adaptive r_0-based filtering
    # =========================================================================
    print()
    print("="*80)
    print("Strategy 1: Adaptive r_0-based Death Threshold")
    print("="*80)
    print()
    print("Filter: Keep features with death > r_0 (cluster scale)")
    print("Rationale: Features dying before r_0 are likely intra-cluster noise")
    print()

    filtered_r0 = {}
    for config_name in config_names:
        r0 = r0_values[config_name]
        filtered = filter_by_death_threshold(all_h1_data[config_name], r0)
        filtered_r0[config_name] = filtered

        orig_stats = compute_statistics(all_h1_data[config_name]['deaths'])
        filt_stats = compute_statistics(filtered['deaths'])

        print(f"{config_name} (threshold = {r0}):")
        print(f"  Original: {orig_stats['count']} features, mean death = {orig_stats['mean']:.3f}")
        print(f"  Filtered: {filtered['n_kept']} features ({filtered['fraction_kept']*100:.1f}% kept), mean death = {filt_stats['mean']:.3f}")
        print()

    # =========================================================================
    # Strategy 2: Fixed persistence threshold
    # =========================================================================
    print("="*80)
    print("Strategy 2: Persistence (Lifetime) Threshold")
    print("="*80)
    print()
    print("Filter: Keep features with persistence > 0.1")
    print("Rationale: Short-lived features are topological noise")
    print()

    persistence_threshold = 0.1
    filtered_pers = {}
    for config_name in config_names:
        filtered = filter_by_persistence_threshold(all_h1_data[config_name], persistence_threshold)
        filtered_pers[config_name] = filtered

        orig_stats = compute_statistics(all_h1_data[config_name]['persistence'])
        filt_stats = compute_statistics(filtered['persistence'])

        print(f"{config_name}:")
        print(f"  Original: {orig_stats['count']} features, mean persistence = {orig_stats['mean']:.3f}")
        print(f"  Filtered: {filtered['n_kept']} features ({filtered['fraction_kept']*100:.1f}% kept), mean persistence = {filt_stats['mean']:.3f}")
        print()

    # =========================================================================
    # Plot 1: Unfiltered vs Filtered Death Distributions (r_0-based)
    # =========================================================================
    print("Creating comparison plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (config_name, label, color) in enumerate(zip(config_names, config_labels, colors)):
        ax = axes[idx]
        r0 = r0_values[config_name]

        original = all_h1_data[config_name]['deaths']
        filtered = filtered_r0[config_name]['deaths']

        # Determine x range
        x_max = min(np.percentile(original, 99), 4.0)
        x_range = np.linspace(0, x_max, 300)

        # Plot original
        if len(original) > 10:
            kde_orig = gaussian_kde(original)
            density_orig = kde_orig(x_range)
            ax.plot(x_range, density_orig, color='gray', linewidth=2,
                    label=f'Original (n={len(original)})', alpha=0.7)
            ax.fill_between(x_range, density_orig, alpha=0.1, color='gray')

        # Plot filtered
        if len(filtered) > 10:
            kde_filt = gaussian_kde(filtered)
            density_filt = kde_filt(x_range)
            ax.plot(x_range, density_filt, color=color, linewidth=2.5,
                    label=f'Filtered (n={len(filtered)}, {filtered_r0[config_name]["fraction_kept"]*100:.0f}%)')
            ax.fill_between(x_range, density_filt, alpha=0.3, color=color)

        # Add threshold line
        ax.axvline(r0, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold = r_0 = {r0}')

        ax.set_xlabel('Death Time', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('H1 Death Distributions: Original vs r_0-Filtered\n(Removing intra-cluster noise)',
                 fontsize=14, y=0.995)
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'death_filtered_by_r0.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # =========================================================================
    # Plot 2: Filtered distributions overlaid (comparing across r_0)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))

    x_range = np.linspace(0, 4.0, 500)

    for config_name, label, color in zip(config_names, config_labels, colors):
        filtered = filtered_r0[config_name]['deaths']

        if len(filtered) > 10:
            kde = gaussian_kde(filtered)
            density = kde(x_range)
            ax.plot(x_range, density, color=color, linewidth=2.5, label=label, alpha=0.8)
            ax.fill_between(x_range, density, alpha=0.15, color=color)

    ax.set_xlabel('Death Time (Filtration Radius)', fontsize=13)
    ax.set_ylabel('Probability Density', fontsize=13)
    ax.set_title('Filtered H1 Death Distributions (Persistent Voids Only)\nFeatures with death > r_0', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'filtered_death_distributions_overlay.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # =========================================================================
    # Plot 3: Persistence-filtered distributions
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (config_name, label, color) in enumerate(zip(config_names, config_labels, colors)):
        ax = axes[idx]

        original = all_h1_data[config_name]['deaths']
        filtered = filtered_pers[config_name]['deaths']

        x_max = min(np.percentile(original, 99), 4.0)
        x_range = np.linspace(0, x_max, 300)

        # Plot original
        if len(original) > 10:
            kde_orig = gaussian_kde(original)
            density_orig = kde_orig(x_range)
            ax.plot(x_range, density_orig, color='gray', linewidth=2,
                    label=f'Original (n={len(original)})', alpha=0.7)
            ax.fill_between(x_range, density_orig, alpha=0.1, color='gray')

        # Plot filtered
        if len(filtered) > 10:
            kde_filt = gaussian_kde(filtered)
            density_filt = kde_filt(x_range)
            ax.plot(x_range, density_filt, color=color, linewidth=2.5,
                    label=f'Filtered (n={len(filtered)}, {filtered_pers[config_name]["fraction_kept"]*100:.0f}%)')
            ax.fill_between(x_range, density_filt, alpha=0.3, color=color)

        ax.set_xlabel('Death Time', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'H1 Death Distributions: Original vs Persistence-Filtered\n(persistence > {persistence_threshold})',
                 fontsize=14, y=0.995)
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'death_filtered_by_persistence.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # =========================================================================
    # Plot 4: Fraction of features kept vs threshold (sensitivity analysis)
    # =========================================================================
    print("Creating threshold sensitivity analysis...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Death threshold sweep
    ax = axes[0]
    thresholds = np.linspace(0, 2.0, 50)

    for config_name, label, color in zip(config_names, config_labels, colors):
        fractions = []
        for thresh in thresholds:
            filtered = filter_by_death_threshold(all_h1_data[config_name], thresh)
            fractions.append(filtered['fraction_kept'])
        ax.plot(thresholds, fractions, color=color, linewidth=2, label=label)

    ax.set_xlabel('Death Threshold', fontsize=12)
    ax.set_ylabel('Fraction of Features Kept', fontsize=12)
    ax.set_title('Sensitivity: Death Threshold', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Persistence threshold sweep
    ax = axes[1]
    pers_thresholds = np.linspace(0, 0.5, 50)

    for config_name, label, color in zip(config_names, config_labels, colors):
        fractions = []
        for thresh in pers_thresholds:
            filtered = filter_by_persistence_threshold(all_h1_data[config_name], thresh)
            fractions.append(filtered['fraction_kept'])
        ax.plot(pers_thresholds, fractions, color=color, linewidth=2, label=label)

    ax.set_xlabel('Persistence Threshold', fontsize=12)
    ax.set_ylabel('Fraction of Features Kept', fontsize=12)
    ax.set_title('Sensitivity: Persistence Threshold', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'threshold_sensitivity.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # =========================================================================
    # Plot 5: Statistics comparison (filtered vs original)
    # =========================================================================
    print("Creating statistics comparison...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    r0_list = [0.2, 0.4, 0.6, 1.0]
    x_pos = np.arange(len(config_names))
    width = 0.35

    # Mean death time
    ax = axes[0]
    orig_means = [compute_statistics(all_h1_data[c]['deaths'])['mean'] for c in config_names]
    filt_means = [compute_statistics(filtered_r0[c]['deaths'])['mean'] for c in config_names]

    bars1 = ax.bar(x_pos - width/2, orig_means, width, label='Original', color='gray', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, filt_means, width, label='Filtered (r_0)', color='steelblue')

    ax.set_xlabel('Configuration', fontsize=11)
    ax.set_ylabel('Mean Death Time', fontsize=11)
    ax.set_title('Mean Death Time', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_labels, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Fraction kept
    ax = axes[1]
    fractions = [filtered_r0[c]['fraction_kept'] for c in config_names]
    bars = ax.bar(x_pos, fractions, color=colors)

    ax.set_xlabel('Configuration', fontsize=11)
    ax.set_ylabel('Fraction Kept', fontsize=11)
    ax.set_title('Fraction of Features Surviving r_0 Filter', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_labels, rotation=15, ha='right')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    for bar, frac in zip(bars, fractions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{frac*100:.0f}%', ha='center', fontsize=10)

    # Feature count
    ax = axes[2]
    orig_counts = [len(all_h1_data[c]['deaths']) for c in config_names]
    filt_counts = [filtered_r0[c]['n_kept'] for c in config_names]

    bars1 = ax.bar(x_pos - width/2, orig_counts, width, label='Original', color='gray', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, filt_counts, width, label='Filtered', color='steelblue')

    ax.set_xlabel('Configuration', fontsize=11)
    ax.set_ylabel('Feature Count', fontsize=11)
    ax.set_title('H1 Feature Count', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_labels, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'filtering_statistics_comparison.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # =========================================================================
    # Save statistics to JSON
    # =========================================================================
    print()
    print("Saving statistics to JSON...")

    results_dir = '../../outputs/analysis_results'
    os.makedirs(results_dir, exist_ok=True)

    output_stats = {
        'analysis_type': 'filtered_persistence',
        'configurations': config_names,
        'r0_filtering': {
            config: {
                'threshold': float(r0_values[config]),
                'n_original': int(len(all_h1_data[config]['deaths'])),
                'n_filtered': int(filtered_r0[config]['n_kept']),
                'fraction_kept': float(filtered_r0[config]['fraction_kept']),
                'original_mean_death': float(compute_statistics(all_h1_data[config]['deaths'])['mean']),
                'filtered_mean_death': float(compute_statistics(filtered_r0[config]['deaths'])['mean'])
            }
            for config in config_names
        },
        'persistence_filtering': {
            'threshold': float(persistence_threshold),
            'results': {
                config: {
                    'n_original': int(len(all_h1_data[config]['deaths'])),
                    'n_filtered': int(filtered_pers[config]['n_kept']),
                    'fraction_kept': float(filtered_pers[config]['fraction_kept'])
                }
                for config in config_names
            }
        }
    }

    output_file = os.path.join(results_dir, 'filtered_persistence_statistics.json')
    with open(output_file, 'w') as f:
        json.dump(output_stats, f, indent=2)
    print(f"  Saved: {output_file}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("="*80)
    print("Analysis Complete!")
    print("="*80)
    print()
    print("Key findings:")
    print()
    print("r_0-based filtering results:")
    for config_name in config_names:
        r0 = r0_values[config_name]
        frac = filtered_r0[config_name]['fraction_kept']
        orig_mean = compute_statistics(all_h1_data[config_name]['deaths'])['mean']
        filt_mean = compute_statistics(filtered_r0[config_name]['deaths'])['mean']
        print(f"  {config_name}: {frac*100:.1f}% kept, mean death {orig_mean:.3f} -> {filt_mean:.3f}")
    print()
    print("Generated files:")
    print("  - death_filtered_by_r0.png")
    print("  - filtered_death_distributions_overlay.png")
    print("  - death_filtered_by_persistence.png")
    print("  - threshold_sensitivity.png")
    print("  - filtering_statistics_comparison.png")
    print("  - filtered_persistence_statistics.json")
    print()
