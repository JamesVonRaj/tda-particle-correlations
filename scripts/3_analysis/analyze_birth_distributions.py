"""
Analyze H1 Birth Time Distributions Across Configurations

This script computes and visualizes the distribution of H1 birth times
(when loops form) for all cluster configurations using the ensemble data.

H1 birth times represent when topological loops first appear in the Rips
filtration, which corresponds to the scale at which clusters start to
connect and form void boundaries.

Hypothesis: H1 birth times correlate with inter-cluster distance, which
is a quantity that H_P (particle nearest neighbor) misses entirely.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy.stats import gaussian_kde, ks_2samp

# ==============================================================================
# Load H1 Persistence Data from Ensemble Data
# ==============================================================================

def load_all_h1_data(config_name, base_dir='../../data/ensemble_data'):
    """
    Load all H1 persistence data for a given configuration.

    Parameters:
    -----------
    config_name : str
        Configuration name (e.g., 'stddev_0.2')
    base_dir : str
        Base directory containing the ensemble data

    Returns:
    --------
    dict : Dictionary with 'births', 'deaths', 'persistence', 'point_counts'
    """
    config_dir = os.path.join(base_dir, config_name)

    # Load metadata to know how many samples
    with open(os.path.join(config_dir, 'persistence_metadata.json'), 'r') as f:
        metadata = json.load(f)

    n_samples = metadata['n_samples']

    all_births = []
    all_deaths = []
    all_persistence = []
    all_point_counts = []

    for i in range(n_samples):
        persistence_file = os.path.join(config_dir, f'sample_{i:03d}_persistence.npy')
        ph_data = np.load(persistence_file, allow_pickle=True).item()

        h1 = ph_data['h1']
        if len(h1) > 0:
            # Extract birth times (column 0), death times (column 1), point counts (column 2)
            births = h1[:, 0]
            deaths = h1[:, 1]
            persistence = deaths - births  # lifetime
            point_counts = h1[:, 2]

            all_births.extend(births)
            all_deaths.extend(deaths)
            all_persistence.extend(persistence)
            all_point_counts.extend(point_counts)

    return {
        'births': np.array(all_births),
        'deaths': np.array(all_deaths),
        'persistence': np.array(all_persistence),
        'point_counts': np.array(all_point_counts)
    }


def compute_statistics(values, name='values'):
    """
    Compute summary statistics for a distribution.

    Parameters:
    -----------
    values : np.ndarray
        Array of values
    name : str
        Name for the statistics

    Returns:
    --------
    dict : Dictionary of statistics
    """
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
    print("H1 Birth Time Distribution Analysis")
    print("="*80)
    print()
    print("Hypothesis: H1 birth times capture inter-cluster connectivity,")
    print("which should correlate with inter-cluster distance.")
    print()

    # Define configurations (same 4 as in H0 death analysis for comparison)
    config_names = ['stddev_0.2', 'stddev_0.4', 'stddev_0.6', 'stddev_1.0']
    config_labels = ['r_0 = 0.2 (Very Tight)', 'r_0 = 0.4 (Tight)',
                     'r_0 = 0.6 (Baseline)', 'r_0 = 1.0 (Loose)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Create output directory
    output_dir = '../../outputs/figures/birth_distributions'
    os.makedirs(output_dir, exist_ok=True)

    # Load all H1 data
    all_h1_data = {}
    all_birth_stats = {}
    all_death_stats = {}
    all_persistence_stats = {}

    print("Loading H1 persistence data from ensemble...")
    for config_name in config_names:
        print(f"  Loading {config_name}...")
        h1_data = load_all_h1_data(config_name)
        all_h1_data[config_name] = h1_data
        all_birth_stats[config_name] = compute_statistics(h1_data['births'], 'births')
        all_death_stats[config_name] = compute_statistics(h1_data['deaths'], 'deaths')
        all_persistence_stats[config_name] = compute_statistics(h1_data['persistence'], 'persistence')

    # -------------------------------------------------------------------------
    # Print Summary Statistics
    # -------------------------------------------------------------------------
    print()
    print("="*80)
    print("H1 BIRTH Time Statistics")
    print("="*80)
    print()
    print(f"{'Config':<15} {'Count':<10} {'Mean':<10} {'Median':<10} {'Std':<10} {'Max':<10}")
    print("-"*80)
    for config_name, stats in all_birth_stats.items():
        print(f"{config_name:<15} {stats['count']:<10} "
              f"{stats['mean']:<10.4f} {stats['median']:<10.4f} "
              f"{stats['std']:<10.4f} {stats['max']:<10.4f}")

    print()
    print("="*80)
    print("H1 DEATH Time Statistics (for comparison)")
    print("="*80)
    print()
    print(f"{'Config':<15} {'Count':<10} {'Mean':<10} {'Median':<10} {'Std':<10} {'Max':<10}")
    print("-"*80)
    for config_name, stats in all_death_stats.items():
        print(f"{config_name:<15} {stats['count']:<10} "
              f"{stats['mean']:<10.4f} {stats['median']:<10.4f} "
              f"{stats['std']:<10.4f} {stats['max']:<10.4f}")

    print()
    print("="*80)
    print("H1 PERSISTENCE (death - birth) Statistics")
    print("="*80)
    print()
    print(f"{'Config':<15} {'Count':<10} {'Mean':<10} {'Median':<10} {'Std':<10} {'Max':<10}")
    print("-"*80)
    for config_name, stats in all_persistence_stats.items():
        print(f"{config_name:<15} {stats['count']:<10} "
              f"{stats['mean']:<10.4f} {stats['median']:<10.4f} "
              f"{stats['std']:<10.4f} {stats['max']:<10.4f}")

    # -------------------------------------------------------------------------
    # Plot 1: H1 Birth Time - Overlapping KDE
    # -------------------------------------------------------------------------
    print()
    print("Creating H1 birth time KDE plot...")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Determine x range from data
    all_births_flat = np.concatenate([all_h1_data[c]['births'] for c in config_names])
    x_max = min(np.percentile(all_births_flat, 99), 3.0)
    x_range = np.linspace(0, x_max, 500)

    for config_name, label, color in zip(config_names, config_labels, colors):
        births = all_h1_data[config_name]['births']

        # Compute KDE
        kde = gaussian_kde(births)
        density = kde(x_range)

        ax.plot(x_range, density, label=label, color=color, linewidth=2.5, alpha=0.8)
        ax.fill_between(x_range, density, alpha=0.2, color=color)

    ax.set_xlabel('Birth Time (Filtration Radius)', fontsize=13)
    ax.set_ylabel('Probability Density', fontsize=13)
    ax.set_title('H1 Birth Time Distributions - When Loops Form', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'h1_birth_distribution_kde.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 2: H1 Birth Time - Separate Subplots with Histograms
    # -------------------------------------------------------------------------
    print("Creating H1 birth time subplot histograms...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (config_name, label, color) in enumerate(zip(config_names, config_labels, colors)):
        births = all_h1_data[config_name]['births']
        stats = all_birth_stats[config_name]

        ax = axes[idx]

        # Histogram
        n, bins_out, patches = ax.hist(births, bins=50, alpha=0.7,
                                        color=color, edgecolor='black', linewidth=0.8)

        # Add vertical lines for mean and median
        ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {stats['mean']:.3f}")
        ax.axvline(stats['median'], color='darkred', linestyle=':', linewidth=2,
                   label=f"Median: {stats['median']:.3f}")

        ax.set_xlabel('Birth Time (Filtration Radius)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f"{label}\nN={stats['count']:,} H1 features", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Add text box with statistics
        textstr = f"Std: {stats['std']:.3f}\nMax: {stats['max']:.3f}"
        props = dict(boxstyle='round', facecolor=color, alpha=0.15)
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)

    fig.suptitle('H1 Birth Time Distributions - Detailed View', fontsize=16, y=0.995)
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'h1_birth_distribution_subplots.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 3: H1 Birth Time - CDFs
    # -------------------------------------------------------------------------
    print("Creating H1 birth time CDFs...")

    fig, ax = plt.subplots(figsize=(12, 7))

    for config_name, label, color in zip(config_names, config_labels, colors):
        births = all_h1_data[config_name]['births']

        # Sort for CDF
        sorted_times = np.sort(births)
        cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)

        ax.plot(sorted_times, cdf, label=label, color=color, linewidth=2.5, alpha=0.8)

    ax.set_xlabel('Birth Time (Filtration Radius)', fontsize=13)
    ax.set_ylabel('Cumulative Probability', fontsize=13)
    ax.set_title('Cumulative Distribution Functions of H1 Birth Times', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, x_max])

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'h1_birth_distribution_cdf.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 4: COMPARISON - Birth vs Death Distributions (Key Analysis!)
    # -------------------------------------------------------------------------
    print("Creating birth vs death comparison plot...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (config_name, label, color) in enumerate(zip(config_names, config_labels, colors)):
        births = all_h1_data[config_name]['births']
        deaths = all_h1_data[config_name]['deaths']

        ax = axes[idx]

        # Compute KDEs
        x_max_local = max(np.percentile(deaths, 99), np.percentile(births, 99))
        x_range_local = np.linspace(0, min(x_max_local, 4.0), 300)

        kde_birth = gaussian_kde(births)
        kde_death = gaussian_kde(deaths)

        # Plot birth distribution
        density_birth = kde_birth(x_range_local)
        ax.plot(x_range_local, density_birth, color='blue', linewidth=2.5,
                label=f"Birth (mean={np.mean(births):.3f})")
        ax.fill_between(x_range_local, density_birth, alpha=0.2, color='blue')

        # Plot death distribution
        density_death = kde_death(x_range_local)
        ax.plot(x_range_local, density_death, color='green', linewidth=2.5,
                label=f"Death (mean={np.mean(deaths):.3f})")
        ax.fill_between(x_range_local, density_death, alpha=0.2, color='green')

        ax.set_xlabel('Filtration Radius', fontsize=11)
        ax.set_ylabel('Probability Density', fontsize=11)
        ax.set_title(f"{label}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('H1 Birth vs Death Distributions\n(Birth = when loops form, Death = when loops fill)',
                 fontsize=14, y=0.995)
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'h1_birth_vs_death_comparison.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 5: Birth-Death Scatter (2D persistence diagram style)
    # -------------------------------------------------------------------------
    print("Creating birth-death scatter plot...")

    fig, ax = plt.subplots(figsize=(10, 10))

    for config_name, label, color in zip(config_names, config_labels, colors):
        births = all_h1_data[config_name]['births']
        deaths = all_h1_data[config_name]['deaths']

        # Subsample for visibility if too many points
        n_points = len(births)
        if n_points > 2000:
            idx_sample = np.random.choice(n_points, 2000, replace=False)
            births_plot = births[idx_sample]
            deaths_plot = deaths[idx_sample]
        else:
            births_plot = births
            deaths_plot = deaths

        ax.scatter(births_plot, deaths_plot, alpha=0.3, s=20, label=label, color=color)

    # Add diagonal line (birth = death)
    max_val = 4.0
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5, label='birth = death')

    ax.set_xlabel('Birth Time', fontsize=13)
    ax.set_ylabel('Death Time', fontsize=13)
    ax.set_title('H1 Features: Birth vs Death Times', fontsize=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_val])
    ax.set_ylim([0, max_val])
    ax.set_aspect('equal')

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'h1_birth_death_scatter.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 6: Persistence (lifetime) Distribution
    # -------------------------------------------------------------------------
    print("Creating persistence (lifetime) distribution plot...")

    fig, ax = plt.subplots(figsize=(12, 7))

    all_pers_flat = np.concatenate([all_h1_data[c]['persistence'] for c in config_names])
    x_max_pers = min(np.percentile(all_pers_flat, 99), 3.0)
    x_range_pers = np.linspace(0, x_max_pers, 500)

    for config_name, label, color in zip(config_names, config_labels, colors):
        persistence = all_h1_data[config_name]['persistence']

        # Compute KDE
        kde = gaussian_kde(persistence)
        density = kde(x_range_pers)

        ax.plot(x_range_pers, density, label=label, color=color, linewidth=2.5, alpha=0.8)
        ax.fill_between(x_range_pers, density, alpha=0.2, color=color)

    ax.set_xlabel('Persistence (Death - Birth)', fontsize=13)
    ax.set_ylabel('Probability Density', fontsize=13)
    ax.set_title('H1 Persistence (Lifetime) Distributions', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'h1_persistence_distribution_kde.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 7: Statistics vs r_0 (scaling analysis)
    # -------------------------------------------------------------------------
    print("Creating statistics vs r_0 plot...")

    r0_values = [0.2, 0.4, 0.6, 1.0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Mean birth vs r_0
    ax = axes[0, 0]
    birth_means = [all_birth_stats[c]['mean'] for c in config_names]
    death_means = [all_death_stats[c]['mean'] for c in config_names]
    ax.plot(r0_values, birth_means, 'bo-', linewidth=2, markersize=10, label='Birth')
    ax.plot(r0_values, death_means, 'go-', linewidth=2, markersize=10, label='Death')
    ax.set_xlabel('r_0 (cluster std dev)', fontsize=11)
    ax.set_ylabel('Mean', fontsize=11)
    ax.set_title('Mean Birth/Death Times vs r_0', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Std vs r_0
    ax = axes[0, 1]
    birth_stds = [all_birth_stats[c]['std'] for c in config_names]
    death_stds = [all_death_stats[c]['std'] for c in config_names]
    ax.plot(r0_values, birth_stds, 'bo-', linewidth=2, markersize=10, label='Birth')
    ax.plot(r0_values, death_stds, 'go-', linewidth=2, markersize=10, label='Death')
    ax.set_xlabel('r_0 (cluster std dev)', fontsize=11)
    ax.set_ylabel('Standard Deviation', fontsize=11)
    ax.set_title('Std Dev of Birth/Death Times vs r_0', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mean persistence vs r_0
    ax = axes[1, 0]
    pers_means = [all_persistence_stats[c]['mean'] for c in config_names]
    ax.plot(r0_values, pers_means, 'mo-', linewidth=2, markersize=10)
    ax.set_xlabel('r_0 (cluster std dev)', fontsize=11)
    ax.set_ylabel('Mean Persistence', fontsize=11)
    ax.set_title('Mean H1 Persistence (Lifetime) vs r_0', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Death/Birth ratio vs r_0
    ax = axes[1, 1]
    ratio = [death_means[i] / birth_means[i] for i in range(len(r0_values))]
    ax.plot(r0_values, ratio, 'ro-', linewidth=2, markersize=10)
    ax.set_xlabel('r_0 (cluster std dev)', fontsize=11)
    ax.set_ylabel('Death/Birth Ratio', fontsize=11)
    ax.set_title('Death/Birth Ratio vs r_0', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    fig.suptitle('H1 Statistics Scaling with Cluster Standard Deviation', fontsize=14, y=0.995)
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'h1_statistics_vs_r0.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # -------------------------------------------------------------------------
    # Statistical Tests
    # -------------------------------------------------------------------------
    print()
    print("="*80)
    print("Statistical Tests (Kolmogorov-Smirnov) for H1 Birth Times")
    print("="*80)
    print()
    print("Testing if birth time distributions are significantly different:")
    print()

    # Compare each pair
    for i in range(len(config_names)):
        for j in range(i+1, len(config_names)):
            config1 = config_names[i]
            config2 = config_names[j]

            ks_stat, p_value = ks_2samp(all_h1_data[config1]['births'],
                                         all_h1_data[config2]['births'])

            significant = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else ""))

            print(f"{config1} vs {config2}:")
            print(f"  KS statistic: {ks_stat:.4f}, p-value: {p_value:.2e} {significant}")

    # -------------------------------------------------------------------------
    # Save Statistics to JSON
    # -------------------------------------------------------------------------
    print()
    print("Saving statistics to JSON...")

    results_dir = '../../outputs/analysis_results'
    os.makedirs(results_dir, exist_ok=True)

    output_stats = {
        'analysis_type': 'H1_birth_time_distributions',
        'configurations': config_names,
        'birth_statistics': all_birth_stats,
        'death_statistics': all_death_stats,
        'persistence_statistics': all_persistence_stats,
        'scaling_with_r0': {
            'r0_values': r0_values,
            'birth_means': [all_birth_stats[c]['mean'] for c in config_names],
            'death_means': [all_death_stats[c]['mean'] for c in config_names],
            'persistence_means': [all_persistence_stats[c]['mean'] for c in config_names]
        }
    }

    output_file = os.path.join(results_dir, 'h1_birth_statistics.json')
    with open(output_file, 'w') as f:
        json.dump(output_stats, f, indent=2)
    print(f"  Saved: {output_file}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print()
    print("="*80)
    print("Analysis complete!")
    print("="*80)
    print()
    print("Generated files:")
    print("  Figures:")
    print("  - h1_birth_distribution_kde.png")
    print("  - h1_birth_distribution_subplots.png")
    print("  - h1_birth_distribution_cdf.png")
    print("  - h1_birth_vs_death_comparison.png")
    print("  - h1_birth_death_scatter.png")
    print("  - h1_persistence_distribution_kde.png")
    print("  - h1_statistics_vs_r0.png")
    print()
    print("  Statistics:")
    print("  - h1_birth_statistics.json")
    print()
    print("Key findings to examine:")
    print("  1. How does mean(birth) scale with r_0 compared to mean(death)?")
    print("  2. Is the birth-death scatter showing distinct clusters?")
    print("  3. Does persistence (lifetime) vary systematically with r_0?")
    print()
