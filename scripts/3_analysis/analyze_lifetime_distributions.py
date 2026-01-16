"""
Analyze H0 and H1 Lifetime (Persistence) Distributions Across Configurations

This script computes and visualizes the distribution of persistence lifetimes
(death - birth) for all cluster configurations using the ensemble data.

Lifetime/persistence measures how "significant" a topological feature is:
- Short-lived features (small persistence) are often topological noise
- Long-lived features (large persistence) represent robust structural features

For H1, persistence indicates how "large" or "robust" a void/loop is.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy.stats import gaussian_kde, ks_2samp

# ==============================================================================
# Load Persistence Data from Ensemble Data
# ==============================================================================

def load_all_persistence_data(config_name, homology_dim='h1', base_dir='../../data/ensemble_data'):
    """
    Load all persistence data for a given configuration and homology dimension.

    Parameters:
    -----------
    config_name : str
        Configuration name (e.g., 'stddev_0.2')
    homology_dim : str
        'h0' or 'h1'
    base_dir : str
        Base directory containing the ensemble data

    Returns:
    --------
    dict : Dictionary with 'births', 'deaths', 'lifetimes', 'point_counts'
    """
    config_dir = os.path.join(base_dir, config_name)

    # Load metadata to know how many samples
    with open(os.path.join(config_dir, 'persistence_metadata.json'), 'r') as f:
        metadata = json.load(f)

    n_samples = metadata['n_samples']

    all_births = []
    all_deaths = []
    all_lifetimes = []
    all_point_counts = []

    for i in range(n_samples):
        persistence_file = os.path.join(config_dir, f'sample_{i:03d}_persistence.npy')
        ph_data = np.load(persistence_file, allow_pickle=True).item()

        h_data = ph_data[homology_dim]
        if len(h_data) > 0:
            births = h_data[:, 0]
            deaths = h_data[:, 1]
            lifetimes = deaths - births
            point_counts = h_data[:, 2]

            all_births.extend(births)
            all_deaths.extend(deaths)
            all_lifetimes.extend(lifetimes)
            all_point_counts.extend(point_counts)

    return {
        'births': np.array(all_births),
        'deaths': np.array(all_deaths),
        'lifetimes': np.array(all_lifetimes),
        'point_counts': np.array(all_point_counts)
    }


def compute_statistics(values):
    """Compute summary statistics for a distribution."""
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
        'q75': float(np.percentile(values, 75)),
        'q90': float(np.percentile(values, 90)),
        'q95': float(np.percentile(values, 95))
    }


# ==============================================================================
# Main Analysis
# ==============================================================================

if __name__ == "__main__":

    print("="*80)
    print("Lifetime (Persistence) Distribution Analysis")
    print("="*80)
    print()
    print("Lifetime = Death - Birth")
    print("Measures how 'significant' or 'robust' a topological feature is.")
    print()

    # Define configurations
    config_names = ['stddev_0.2', 'stddev_0.4', 'stddev_0.6', 'stddev_1.0']
    config_labels = ['r_0 = 0.2 (Very Tight)', 'r_0 = 0.4 (Tight)',
                     'r_0 = 0.6 (Baseline)', 'r_0 = 1.0 (Loose)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    r0_values = [0.2, 0.4, 0.6, 1.0]

    # Create output directory
    output_dir = '../../outputs/figures/lifetime_distributions'
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # H0 Lifetime Analysis
    # =========================================================================
    print("="*80)
    print("H0 Lifetime Analysis")
    print("="*80)
    print()

    all_h0_data = {}
    all_h0_stats = {}

    print("Loading H0 persistence data...")
    for config_name in config_names:
        print(f"  Loading {config_name}...")
        h0_data = load_all_persistence_data(config_name, 'h0')
        all_h0_data[config_name] = h0_data
        all_h0_stats[config_name] = compute_statistics(h0_data['lifetimes'])

    print()
    print("H0 Lifetime Statistics:")
    print(f"{'Config':<15} {'Count':<10} {'Mean':<10} {'Median':<10} {'Std':<10} {'Q95':<10}")
    print("-"*80)
    for config_name, stats in all_h0_stats.items():
        print(f"{config_name:<15} {stats['count']:<10} "
              f"{stats['mean']:<10.4f} {stats['median']:<10.4f} "
              f"{stats['std']:<10.4f} {stats['q95']:<10.4f}")

    # =========================================================================
    # H1 Lifetime Analysis
    # =========================================================================
    print()
    print("="*80)
    print("H1 Lifetime Analysis")
    print("="*80)
    print()

    all_h1_data = {}
    all_h1_stats = {}

    print("Loading H1 persistence data...")
    for config_name in config_names:
        print(f"  Loading {config_name}...")
        h1_data = load_all_persistence_data(config_name, 'h1')
        all_h1_data[config_name] = h1_data
        all_h1_stats[config_name] = compute_statistics(h1_data['lifetimes'])

    print()
    print("H1 Lifetime Statistics:")
    print(f"{'Config':<15} {'Count':<10} {'Mean':<10} {'Median':<10} {'Std':<10} {'Q95':<10}")
    print("-"*80)
    for config_name, stats in all_h1_stats.items():
        print(f"{config_name:<15} {stats['count']:<10} "
              f"{stats['mean']:<10.4f} {stats['median']:<10.4f} "
              f"{stats['std']:<10.4f} {stats['q95']:<10.4f}")

    # =========================================================================
    # Plot 1: H0 Lifetime KDE
    # =========================================================================
    print()
    print("Creating H0 lifetime KDE plot...")

    fig, ax = plt.subplots(figsize=(12, 7))

    all_h0_lifetimes = np.concatenate([all_h0_data[c]['lifetimes'] for c in config_names])
    x_max = min(np.percentile(all_h0_lifetimes, 99), 2.0)
    x_range = np.linspace(0, x_max, 500)

    for config_name, label, color in zip(config_names, config_labels, colors):
        lifetimes = all_h0_data[config_name]['lifetimes']

        kde = gaussian_kde(lifetimes)
        density = kde(x_range)

        ax.plot(x_range, density, label=label, color=color, linewidth=2.5, alpha=0.8)
        ax.fill_between(x_range, density, alpha=0.2, color=color)

    ax.set_xlabel('Lifetime (Death - Birth)', fontsize=13)
    ax.set_ylabel('Probability Density', fontsize=13)
    ax.set_title('H0 Lifetime Distributions - Connected Component Persistence', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'h0_lifetime_distribution_kde.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # =========================================================================
    # Plot 2: H1 Lifetime KDE
    # =========================================================================
    print("Creating H1 lifetime KDE plot...")

    fig, ax = plt.subplots(figsize=(12, 7))

    all_h1_lifetimes = np.concatenate([all_h1_data[c]['lifetimes'] for c in config_names])
    x_max = min(np.percentile(all_h1_lifetimes, 99), 2.0)
    x_range = np.linspace(0, x_max, 500)

    for config_name, label, color in zip(config_names, config_labels, colors):
        lifetimes = all_h1_data[config_name]['lifetimes']

        kde = gaussian_kde(lifetimes)
        density = kde(x_range)

        ax.plot(x_range, density, label=label, color=color, linewidth=2.5, alpha=0.8)
        ax.fill_between(x_range, density, alpha=0.2, color=color)

    ax.set_xlabel('Lifetime (Death - Birth)', fontsize=13)
    ax.set_ylabel('Probability Density', fontsize=13)
    ax.set_title('H1 Lifetime Distributions - Loop/Void Persistence', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'h1_lifetime_distribution_kde.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # =========================================================================
    # Plot 3: H0 and H1 Lifetime Subplots
    # =========================================================================
    print("Creating H0 lifetime subplot histograms...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (config_name, label, color) in enumerate(zip(config_names, config_labels, colors)):
        lifetimes = all_h0_data[config_name]['lifetimes']
        stats = all_h0_stats[config_name]

        ax = axes[idx]

        n, bins_out, patches = ax.hist(lifetimes, bins=50, alpha=0.7,
                                        color=color, edgecolor='black', linewidth=0.8)

        ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {stats['mean']:.4f}")
        ax.axvline(stats['median'], color='darkred', linestyle=':', linewidth=2,
                   label=f"Median: {stats['median']:.4f}")

        ax.set_xlabel('Lifetime (Death - Birth)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f"{label}\nN={stats['count']:,} H0 features", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        textstr = f"Std: {stats['std']:.4f}\nQ95: {stats['q95']:.4f}"
        props = dict(boxstyle='round', facecolor=color, alpha=0.15)
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)

    fig.suptitle('H0 Lifetime Distributions - Detailed View', fontsize=16, y=0.995)
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'h0_lifetime_distribution_subplots.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # H1 subplots
    print("Creating H1 lifetime subplot histograms...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (config_name, label, color) in enumerate(zip(config_names, config_labels, colors)):
        lifetimes = all_h1_data[config_name]['lifetimes']
        stats = all_h1_stats[config_name]

        ax = axes[idx]

        n, bins_out, patches = ax.hist(lifetimes, bins=50, alpha=0.7,
                                        color=color, edgecolor='black', linewidth=0.8)

        ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {stats['mean']:.4f}")
        ax.axvline(stats['median'], color='darkred', linestyle=':', linewidth=2,
                   label=f"Median: {stats['median']:.4f}")

        ax.set_xlabel('Lifetime (Death - Birth)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f"{label}\nN={stats['count']:,} H1 features", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        textstr = f"Std: {stats['std']:.4f}\nQ95: {stats['q95']:.4f}"
        props = dict(boxstyle='round', facecolor=color, alpha=0.15)
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)

    fig.suptitle('H1 Lifetime Distributions - Detailed View', fontsize=16, y=0.995)
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'h1_lifetime_distribution_subplots.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # =========================================================================
    # Plot 4: CDFs
    # =========================================================================
    print("Creating lifetime CDF plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # H0 CDF
    ax = axes[0]
    for config_name, label, color in zip(config_names, config_labels, colors):
        lifetimes = all_h0_data[config_name]['lifetimes']
        sorted_times = np.sort(lifetimes)
        cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        ax.plot(sorted_times, cdf, label=label, color=color, linewidth=2.5, alpha=0.8)

    ax.set_xlabel('Lifetime', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('H0 Lifetime CDFs', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.0])

    # H1 CDF
    ax = axes[1]
    for config_name, label, color in zip(config_names, config_labels, colors):
        lifetimes = all_h1_data[config_name]['lifetimes']
        sorted_times = np.sort(lifetimes)
        cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        ax.plot(sorted_times, cdf, label=label, color=color, linewidth=2.5, alpha=0.8)

    ax.set_xlabel('Lifetime', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('H1 Lifetime CDFs', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.0])

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'lifetime_distribution_cdfs.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # =========================================================================
    # Plot 5: Box Plots
    # =========================================================================
    print("Creating lifetime box plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # H0 box plot
    ax = axes[0]
    data_h0 = [all_h0_data[c]['lifetimes'] for c in config_names]
    bp = ax.boxplot(data_h0, tick_labels=config_labels, patch_artist=True,
                     showmeans=True, meanline=True,
                     medianprops=dict(color='darkred', linewidth=2),
                     meanprops=dict(color='red', linewidth=2, linestyle='--'))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Lifetime', fontsize=12)
    ax.set_title('H0 Lifetime Box Plots', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

    # H1 box plot
    ax = axes[1]
    data_h1 = [all_h1_data[c]['lifetimes'] for c in config_names]
    bp = ax.boxplot(data_h1, tick_labels=config_labels, patch_artist=True,
                     showmeans=True, meanline=True,
                     medianprops=dict(color='darkred', linewidth=2),
                     meanprops=dict(color='red', linewidth=2, linestyle='--'))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Lifetime', fontsize=12)
    ax.set_title('H1 Lifetime Box Plots', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'lifetime_distribution_boxplots.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # =========================================================================
    # Plot 6: Statistics vs r_0
    # =========================================================================
    print("Creating statistics vs r_0 plot...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # H0 Mean lifetime vs r_0
    ax = axes[0, 0]
    h0_means = [all_h0_stats[c]['mean'] for c in config_names]
    ax.plot(r0_values, h0_means, 'bo-', linewidth=2, markersize=10)
    ax.set_xlabel('r_0 (cluster std dev)', fontsize=11)
    ax.set_ylabel('Mean Lifetime', fontsize=11)
    ax.set_title('H0 Mean Lifetime vs r_0', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # H1 Mean lifetime vs r_0
    ax = axes[0, 1]
    h1_means = [all_h1_stats[c]['mean'] for c in config_names]
    ax.plot(r0_values, h1_means, 'go-', linewidth=2, markersize=10)
    ax.set_xlabel('r_0 (cluster std dev)', fontsize=11)
    ax.set_ylabel('Mean Lifetime', fontsize=11)
    ax.set_title('H1 Mean Lifetime vs r_0', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # H0 vs H1 comparison
    ax = axes[0, 2]
    ax.plot(r0_values, h0_means, 'bo-', linewidth=2, markersize=10, label='H0')
    ax.plot(r0_values, h1_means, 'go-', linewidth=2, markersize=10, label='H1')
    ax.set_xlabel('r_0 (cluster std dev)', fontsize=11)
    ax.set_ylabel('Mean Lifetime', fontsize=11)
    ax.set_title('H0 vs H1 Mean Lifetime', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # H0 Std vs r_0
    ax = axes[1, 0]
    h0_stds = [all_h0_stats[c]['std'] for c in config_names]
    ax.plot(r0_values, h0_stds, 'bo-', linewidth=2, markersize=10)
    ax.set_xlabel('r_0 (cluster std dev)', fontsize=11)
    ax.set_ylabel('Std Lifetime', fontsize=11)
    ax.set_title('H0 Lifetime Std vs r_0', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # H1 Std vs r_0
    ax = axes[1, 1]
    h1_stds = [all_h1_stats[c]['std'] for c in config_names]
    ax.plot(r0_values, h1_stds, 'go-', linewidth=2, markersize=10)
    ax.set_xlabel('r_0 (cluster std dev)', fontsize=11)
    ax.set_ylabel('Std Lifetime', fontsize=11)
    ax.set_title('H1 Lifetime Std vs r_0', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Q95 comparison
    ax = axes[1, 2]
    h0_q95 = [all_h0_stats[c]['q95'] for c in config_names]
    h1_q95 = [all_h1_stats[c]['q95'] for c in config_names]
    ax.plot(r0_values, h0_q95, 'bo-', linewidth=2, markersize=10, label='H0')
    ax.plot(r0_values, h1_q95, 'go-', linewidth=2, markersize=10, label='H1')
    ax.set_xlabel('r_0 (cluster std dev)', fontsize=11)
    ax.set_ylabel('Q95 Lifetime', fontsize=11)
    ax.set_title('95th Percentile Lifetime vs r_0', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Lifetime Statistics Scaling with Cluster Standard Deviation', fontsize=14, y=0.995)
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'lifetime_statistics_vs_r0.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # =========================================================================
    # Statistical Tests
    # =========================================================================
    print()
    print("="*80)
    print("Statistical Tests (Kolmogorov-Smirnov)")
    print("="*80)
    print()
    print("H0 Lifetime Distributions:")
    for i in range(len(config_names)):
        for j in range(i+1, len(config_names)):
            config1, config2 = config_names[i], config_names[j]
            ks_stat, p_value = ks_2samp(all_h0_data[config1]['lifetimes'],
                                         all_h0_data[config2]['lifetimes'])
            sig = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else ""))
            print(f"  {config1} vs {config2}: KS={ks_stat:.4f}, p={p_value:.2e} {sig}")

    print()
    print("H1 Lifetime Distributions:")
    for i in range(len(config_names)):
        for j in range(i+1, len(config_names)):
            config1, config2 = config_names[i], config_names[j]
            ks_stat, p_value = ks_2samp(all_h1_data[config1]['lifetimes'],
                                         all_h1_data[config2]['lifetimes'])
            sig = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else ""))
            print(f"  {config1} vs {config2}: KS={ks_stat:.4f}, p={p_value:.2e} {sig}")

    # =========================================================================
    # Save Statistics to JSON
    # =========================================================================
    print()
    print("Saving statistics to JSON...")

    results_dir = '../../outputs/analysis_results'
    os.makedirs(results_dir, exist_ok=True)

    output_stats = {
        'analysis_type': 'lifetime_distributions',
        'configurations': config_names,
        'h0_statistics': all_h0_stats,
        'h1_statistics': all_h1_stats,
        'scaling_with_r0': {
            'r0_values': r0_values,
            'h0_means': [all_h0_stats[c]['mean'] for c in config_names],
            'h1_means': [all_h1_stats[c]['mean'] for c in config_names],
            'h0_medians': [all_h0_stats[c]['median'] for c in config_names],
            'h1_medians': [all_h1_stats[c]['median'] for c in config_names],
            'h0_stds': [all_h0_stats[c]['std'] for c in config_names],
            'h1_stds': [all_h1_stats[c]['std'] for c in config_names]
        }
    }

    output_file = os.path.join(results_dir, 'lifetime_statistics.json')
    with open(output_file, 'w') as f:
        json.dump(output_stats, f, indent=2)
    print(f"  Saved: {output_file}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("="*80)
    print("Analysis complete!")
    print("="*80)
    print()
    print("Generated files:")
    print("  Figures:")
    print("  - h0_lifetime_distribution_kde.png")
    print("  - h1_lifetime_distribution_kde.png")
    print("  - h0_lifetime_distribution_subplots.png")
    print("  - h1_lifetime_distribution_subplots.png")
    print("  - lifetime_distribution_cdfs.png")
    print("  - lifetime_distribution_boxplots.png")
    print("  - lifetime_statistics_vs_r0.png")
    print()
    print("  Statistics:")
    print("  - lifetime_statistics.json")
    print()
    print("Key observations:")
    print(f"  H0 mean lifetime: {all_h0_stats['stddev_0.2']['mean']:.4f} -> {all_h0_stats['stddev_1.0']['mean']:.4f}")
    print(f"  H1 mean lifetime: {all_h1_stats['stddev_0.2']['mean']:.4f} -> {all_h1_stats['stddev_1.0']['mean']:.4f}")
    print(f"  H1/H0 ratio (r_0=0.2): {all_h1_stats['stddev_0.2']['mean']/all_h0_stats['stddev_0.2']['mean']:.2f}")
    print(f"  H1/H0 ratio (r_0=1.0): {all_h1_stats['stddev_1.0']['mean']/all_h0_stats['stddev_1.0']['mean']:.2f}")
    print()
