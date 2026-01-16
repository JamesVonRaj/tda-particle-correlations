"""
Analyze Weighted H0 and H1 Lifetime Distributions Across Configurations

This script computes and visualizes the distribution of persistence lifetimes
(death - birth) WEIGHTED by num_points - the number of points involved in
each topological feature.

Weighting by num_points gives more importance to features involving more points,
which may represent more structurally significant topological events.

For H0: num_points is the total number of points in the merged component
For H1: num_points is the number of vertices in the loop boundary
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy.stats import gaussian_kde

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


def compute_weighted_statistics(values, weights):
    """
    Compute weighted summary statistics for a distribution.

    Parameters:
    -----------
    values : array-like
        The values (e.g., lifetimes)
    weights : array-like
        The weights (e.g., num_points)

    Returns:
    --------
    dict : Dictionary with weighted statistics
    """
    if len(values) == 0:
        return {'count': 0, 'total_weight': 0, 'weighted_mean': np.nan,
                'weighted_median': np.nan, 'weighted_std': np.nan}

    values = np.array(values)
    weights = np.array(weights)

    # Normalize weights
    total_weight = np.sum(weights)
    norm_weights = weights / total_weight

    # Weighted mean
    weighted_mean = np.sum(values * norm_weights)

    # Weighted variance and std
    weighted_var = np.sum(norm_weights * (values - weighted_mean) ** 2)
    weighted_std = np.sqrt(weighted_var)

    # Weighted median (using weighted percentile)
    weighted_median = weighted_percentile(values, weights, 50)

    # Weighted percentiles
    q25 = weighted_percentile(values, weights, 25)
    q75 = weighted_percentile(values, weights, 75)
    q90 = weighted_percentile(values, weights, 90)
    q95 = weighted_percentile(values, weights, 95)

    return {
        'count': len(values),
        'total_weight': float(total_weight),
        'weighted_mean': float(weighted_mean),
        'weighted_median': float(weighted_median),
        'weighted_std': float(weighted_std),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'weighted_q25': float(q25),
        'weighted_q75': float(q75),
        'weighted_q90': float(q90),
        'weighted_q95': float(q95)
    }


def weighted_percentile(values, weights, percentile):
    """
    Compute weighted percentile.

    Parameters:
    -----------
    values : array-like
        The values
    weights : array-like
        The weights
    percentile : float
        Percentile to compute (0-100)

    Returns:
    --------
    float : The weighted percentile value
    """
    values = np.array(values)
    weights = np.array(weights)

    # Sort by values
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Cumulative weights
    cumsum = np.cumsum(sorted_weights)
    total = cumsum[-1]

    # Find the value at the percentile
    target = percentile / 100.0 * total
    idx = np.searchsorted(cumsum, target)

    if idx >= len(sorted_values):
        return sorted_values[-1]
    return sorted_values[idx]


def weighted_kde(values, weights, x_range, bandwidth=None):
    """
    Compute a weighted kernel density estimate using histogram approximation.

    For true weighted KDE, we repeat each value proportionally to its weight.
    """
    values = np.array(values)
    weights = np.array(weights)

    # Normalize weights to reasonable repetition counts
    # Scale so max weight gives ~100 repetitions for smooth KDE
    max_weight = np.max(weights)
    scale_factor = 100 / max_weight if max_weight > 0 else 1
    rep_counts = np.round(weights * scale_factor).astype(int)
    rep_counts = np.maximum(rep_counts, 1)  # At least 1 repetition

    # Create expanded array
    expanded_values = np.repeat(values, rep_counts)

    if len(expanded_values) < 2:
        return np.zeros_like(x_range)

    # Compute KDE on expanded data
    kde = gaussian_kde(expanded_values)
    return kde(x_range)


# ==============================================================================
# Main Analysis
# ==============================================================================

if __name__ == "__main__":

    print("="*80)
    print("Weighted Lifetime (Persistence) Distribution Analysis")
    print("="*80)
    print()
    print("Lifetime = Death - Birth, WEIGHTED by num_points")
    print("Weighting gives more importance to features involving more points.")
    print()

    # Define configurations
    config_names = ['stddev_0.2', 'stddev_0.4', 'stddev_0.6', 'stddev_1.0']
    config_labels = ['r_0 = 0.2 (Very Tight)', 'r_0 = 0.4 (Tight)',
                     'r_0 = 0.6 (Baseline)', 'r_0 = 1.0 (Loose)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    r0_values = [0.2, 0.4, 0.6, 1.0]

    # Create output directory
    output_dir = '../../outputs/figures/weighted_lifetime_distributions'
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # H0 Weighted Lifetime Analysis
    # =========================================================================
    print("="*80)
    print("H0 Weighted Lifetime Analysis")
    print("="*80)
    print()

    all_h0_data = {}
    all_h0_stats = {}

    print("Loading H0 persistence data...")
    for config_name in config_names:
        print(f"  Loading {config_name}...")
        h0_data = load_all_persistence_data(config_name, 'h0')
        all_h0_data[config_name] = h0_data
        all_h0_stats[config_name] = compute_weighted_statistics(
            h0_data['lifetimes'], h0_data['point_counts']
        )

    print()
    print("H0 Weighted Lifetime Statistics:")
    print(f"{'Config':<15} {'Count':<10} {'W.Mean':<12} {'W.Median':<12} {'W.Std':<12} {'W.Q95':<12}")
    print("-"*80)
    for config_name, stats in all_h0_stats.items():
        print(f"{config_name:<15} {stats['count']:<10} "
              f"{stats['weighted_mean']:<12.4f} {stats['weighted_median']:<12.4f} "
              f"{stats['weighted_std']:<12.4f} {stats['weighted_q95']:<12.4f}")

    # =========================================================================
    # H1 Weighted Lifetime Analysis
    # =========================================================================
    print()
    print("="*80)
    print("H1 Weighted Lifetime Analysis")
    print("="*80)
    print()

    all_h1_data = {}
    all_h1_stats = {}

    print("Loading H1 persistence data...")
    for config_name in config_names:
        print(f"  Loading {config_name}...")
        h1_data = load_all_persistence_data(config_name, 'h1')
        all_h1_data[config_name] = h1_data
        all_h1_stats[config_name] = compute_weighted_statistics(
            h1_data['lifetimes'], h1_data['point_counts']
        )

    print()
    print("H1 Weighted Lifetime Statistics:")
    print(f"{'Config':<15} {'Count':<10} {'W.Mean':<12} {'W.Median':<12} {'W.Std':<12} {'W.Q95':<12}")
    print("-"*80)
    for config_name, stats in all_h1_stats.items():
        print(f"{config_name:<15} {stats['count']:<10} "
              f"{stats['weighted_mean']:<12.4f} {stats['weighted_median']:<12.4f} "
              f"{stats['weighted_std']:<12.4f} {stats['weighted_q95']:<12.4f}")

    # =========================================================================
    # Plot 1: H0 Weighted Lifetime KDE
    # =========================================================================
    print()
    print("Creating H0 weighted lifetime KDE plot...")

    fig, ax = plt.subplots(figsize=(12, 7))

    all_h0_lifetimes = np.concatenate([all_h0_data[c]['lifetimes'] for c in config_names])
    x_max = min(np.percentile(all_h0_lifetimes, 99), 2.0)
    x_range = np.linspace(0, x_max, 500)

    for config_name, label, color in zip(config_names, config_labels, colors):
        lifetimes = all_h0_data[config_name]['lifetimes']
        weights = all_h0_data[config_name]['point_counts']

        density = weighted_kde(lifetimes, weights, x_range)

        ax.plot(x_range, density, label=label, color=color, linewidth=2.5, alpha=0.8)
        ax.fill_between(x_range, density, alpha=0.2, color=color)

    ax.set_xlabel('Lifetime (Death - Birth)', fontsize=13)
    ax.set_ylabel('Weighted Probability Density', fontsize=13)
    ax.set_title('H0 Weighted Lifetime Distributions - Connected Component Persistence\n(Weighted by num_points)', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'h0_weighted_lifetime_kde.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # =========================================================================
    # Plot 2: H1 Weighted Lifetime KDE
    # =========================================================================
    print("Creating H1 weighted lifetime KDE plot...")

    fig, ax = plt.subplots(figsize=(12, 7))

    all_h1_lifetimes = np.concatenate([all_h1_data[c]['lifetimes'] for c in config_names])
    x_max = min(np.percentile(all_h1_lifetimes, 99), 2.0)
    x_range = np.linspace(0, x_max, 500)

    for config_name, label, color in zip(config_names, config_labels, colors):
        lifetimes = all_h1_data[config_name]['lifetimes']
        weights = all_h1_data[config_name]['point_counts']

        density = weighted_kde(lifetimes, weights, x_range)

        ax.plot(x_range, density, label=label, color=color, linewidth=2.5, alpha=0.8)
        ax.fill_between(x_range, density, alpha=0.2, color=color)

    ax.set_xlabel('Lifetime (Death - Birth)', fontsize=13)
    ax.set_ylabel('Weighted Probability Density', fontsize=13)
    ax.set_title('H1 Weighted Lifetime Distributions - Loop/Void Persistence\n(Weighted by num_points)', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'h1_weighted_lifetime_kde.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # =========================================================================
    # Plot 3: H0 Weighted Histogram Subplots
    # =========================================================================
    print("Creating H0 weighted lifetime subplot histograms...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (config_name, label, color) in enumerate(zip(config_names, config_labels, colors)):
        lifetimes = all_h0_data[config_name]['lifetimes']
        weights = all_h0_data[config_name]['point_counts']
        stats = all_h0_stats[config_name]

        ax = axes[idx]

        # Weighted histogram
        n, bins_out, patches = ax.hist(lifetimes, bins=50, weights=weights, alpha=0.7,
                                        color=color, edgecolor='black', linewidth=0.8)

        ax.axvline(stats['weighted_mean'], color='red', linestyle='--', linewidth=2,
                   label=f"W.Mean: {stats['weighted_mean']:.4f}")
        ax.axvline(stats['weighted_median'], color='darkred', linestyle=':', linewidth=2,
                   label=f"W.Median: {stats['weighted_median']:.4f}")

        ax.set_xlabel('Lifetime (Death - Birth)', fontsize=11)
        ax.set_ylabel('Weighted Count', fontsize=11)
        ax.set_title(f"{label}\nN={stats['count']:,} features, Total weight={stats['total_weight']:,.0f}",
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        textstr = f"W.Std: {stats['weighted_std']:.4f}\nW.Q95: {stats['weighted_q95']:.4f}"
        props = dict(boxstyle='round', facecolor=color, alpha=0.15)
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)

    fig.suptitle('H0 Weighted Lifetime Distributions - Detailed View\n(Weighted by num_points)', fontsize=16, y=0.995)
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'h0_weighted_lifetime_subplots.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # =========================================================================
    # Plot 4: H1 Weighted Histogram Subplots
    # =========================================================================
    print("Creating H1 weighted lifetime subplot histograms...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (config_name, label, color) in enumerate(zip(config_names, config_labels, colors)):
        lifetimes = all_h1_data[config_name]['lifetimes']
        weights = all_h1_data[config_name]['point_counts']
        stats = all_h1_stats[config_name]

        ax = axes[idx]

        # Weighted histogram
        n, bins_out, patches = ax.hist(lifetimes, bins=50, weights=weights, alpha=0.7,
                                        color=color, edgecolor='black', linewidth=0.8)

        ax.axvline(stats['weighted_mean'], color='red', linestyle='--', linewidth=2,
                   label=f"W.Mean: {stats['weighted_mean']:.4f}")
        ax.axvline(stats['weighted_median'], color='darkred', linestyle=':', linewidth=2,
                   label=f"W.Median: {stats['weighted_median']:.4f}")

        ax.set_xlabel('Lifetime (Death - Birth)', fontsize=11)
        ax.set_ylabel('Weighted Count', fontsize=11)
        ax.set_title(f"{label}\nN={stats['count']:,} features, Total weight={stats['total_weight']:,.0f}",
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        textstr = f"W.Std: {stats['weighted_std']:.4f}\nW.Q95: {stats['weighted_q95']:.4f}"
        props = dict(boxstyle='round', facecolor=color, alpha=0.15)
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)

    fig.suptitle('H1 Weighted Lifetime Distributions - Detailed View\n(Weighted by num_points)', fontsize=16, y=0.995)
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'h1_weighted_lifetime_subplots.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # =========================================================================
    # Plot 5: Weighted CDFs
    # =========================================================================
    print("Creating weighted lifetime CDF plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # H0 Weighted CDF
    ax = axes[0]
    for config_name, label, color in zip(config_names, config_labels, colors):
        lifetimes = all_h0_data[config_name]['lifetimes']
        weights = all_h0_data[config_name]['point_counts']

        # Sort by lifetime
        sorted_indices = np.argsort(lifetimes)
        sorted_lifetimes = lifetimes[sorted_indices]
        sorted_weights = weights[sorted_indices]

        # Weighted CDF
        cumsum = np.cumsum(sorted_weights)
        cdf = cumsum / cumsum[-1]

        ax.plot(sorted_lifetimes, cdf, label=label, color=color, linewidth=2.5, alpha=0.8)

    ax.set_xlabel('Lifetime', fontsize=12)
    ax.set_ylabel('Weighted Cumulative Probability', fontsize=12)
    ax.set_title('H0 Weighted Lifetime CDFs', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.0])

    # H1 Weighted CDF
    ax = axes[1]
    for config_name, label, color in zip(config_names, config_labels, colors):
        lifetimes = all_h1_data[config_name]['lifetimes']
        weights = all_h1_data[config_name]['point_counts']

        # Sort by lifetime
        sorted_indices = np.argsort(lifetimes)
        sorted_lifetimes = lifetimes[sorted_indices]
        sorted_weights = weights[sorted_indices]

        # Weighted CDF
        cumsum = np.cumsum(sorted_weights)
        cdf = cumsum / cumsum[-1]

        ax.plot(sorted_lifetimes, cdf, label=label, color=color, linewidth=2.5, alpha=0.8)

    ax.set_xlabel('Lifetime', fontsize=12)
    ax.set_ylabel('Weighted Cumulative Probability', fontsize=12)
    ax.set_title('H1 Weighted Lifetime CDFs', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.0])

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'weighted_lifetime_cdfs.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # =========================================================================
    # Plot 6: Weighted Statistics vs r_0
    # =========================================================================
    print("Creating weighted statistics vs r_0 plot...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # H0 Weighted Mean lifetime vs r_0
    ax = axes[0, 0]
    h0_means = [all_h0_stats[c]['weighted_mean'] for c in config_names]
    ax.plot(r0_values, h0_means, 'bo-', linewidth=2, markersize=10)
    ax.set_xlabel('r_0 (cluster std dev)', fontsize=11)
    ax.set_ylabel('Weighted Mean Lifetime', fontsize=11)
    ax.set_title('H0 Weighted Mean Lifetime vs r_0', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # H1 Weighted Mean lifetime vs r_0
    ax = axes[0, 1]
    h1_means = [all_h1_stats[c]['weighted_mean'] for c in config_names]
    ax.plot(r0_values, h1_means, 'go-', linewidth=2, markersize=10)
    ax.set_xlabel('r_0 (cluster std dev)', fontsize=11)
    ax.set_ylabel('Weighted Mean Lifetime', fontsize=11)
    ax.set_title('H1 Weighted Mean Lifetime vs r_0', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # H0 vs H1 comparison
    ax = axes[0, 2]
    ax.plot(r0_values, h0_means, 'bo-', linewidth=2, markersize=10, label='H0')
    ax.plot(r0_values, h1_means, 'go-', linewidth=2, markersize=10, label='H1')
    ax.set_xlabel('r_0 (cluster std dev)', fontsize=11)
    ax.set_ylabel('Weighted Mean Lifetime', fontsize=11)
    ax.set_title('H0 vs H1 Weighted Mean Lifetime', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # H0 Weighted Std vs r_0
    ax = axes[1, 0]
    h0_stds = [all_h0_stats[c]['weighted_std'] for c in config_names]
    ax.plot(r0_values, h0_stds, 'bo-', linewidth=2, markersize=10)
    ax.set_xlabel('r_0 (cluster std dev)', fontsize=11)
    ax.set_ylabel('Weighted Std Lifetime', fontsize=11)
    ax.set_title('H0 Weighted Lifetime Std vs r_0', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # H1 Weighted Std vs r_0
    ax = axes[1, 1]
    h1_stds = [all_h1_stats[c]['weighted_std'] for c in config_names]
    ax.plot(r0_values, h1_stds, 'go-', linewidth=2, markersize=10)
    ax.set_xlabel('r_0 (cluster std dev)', fontsize=11)
    ax.set_ylabel('Weighted Std Lifetime', fontsize=11)
    ax.set_title('H1 Weighted Lifetime Std vs r_0', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Q95 comparison
    ax = axes[1, 2]
    h0_q95 = [all_h0_stats[c]['weighted_q95'] for c in config_names]
    h1_q95 = [all_h1_stats[c]['weighted_q95'] for c in config_names]
    ax.plot(r0_values, h0_q95, 'bo-', linewidth=2, markersize=10, label='H0')
    ax.plot(r0_values, h1_q95, 'go-', linewidth=2, markersize=10, label='H1')
    ax.set_xlabel('r_0 (cluster std dev)', fontsize=11)
    ax.set_ylabel('Weighted Q95 Lifetime', fontsize=11)
    ax.set_title('Weighted 95th Percentile Lifetime vs r_0', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Weighted Lifetime Statistics Scaling with Cluster Standard Deviation', fontsize=14, y=0.995)
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'weighted_lifetime_statistics_vs_r0.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # =========================================================================
    # Plot 7: Comparison - Weighted vs Unweighted
    # =========================================================================
    print("Creating weighted vs unweighted comparison plot...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Compute unweighted stats for comparison
    from scipy.stats import describe

    # H0 Mean comparison
    ax = axes[0, 0]
    h0_unweighted_means = [np.mean(all_h0_data[c]['lifetimes']) for c in config_names]
    ax.plot(r0_values, h0_unweighted_means, 'b--', linewidth=2, markersize=8, marker='s', label='Unweighted')
    ax.plot(r0_values, h0_means, 'b-', linewidth=2, markersize=10, marker='o', label='Weighted')
    ax.set_xlabel('r_0 (cluster std dev)', fontsize=11)
    ax.set_ylabel('Mean Lifetime', fontsize=11)
    ax.set_title('H0 Mean Lifetime: Weighted vs Unweighted', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # H1 Mean comparison
    ax = axes[0, 1]
    h1_unweighted_means = [np.mean(all_h1_data[c]['lifetimes']) for c in config_names]
    ax.plot(r0_values, h1_unweighted_means, 'g--', linewidth=2, markersize=8, marker='s', label='Unweighted')
    ax.plot(r0_values, h1_means, 'g-', linewidth=2, markersize=10, marker='o', label='Weighted')
    ax.set_xlabel('r_0 (cluster std dev)', fontsize=11)
    ax.set_ylabel('Mean Lifetime', fontsize=11)
    ax.set_title('H1 Mean Lifetime: Weighted vs Unweighted', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # H0 Q95 comparison
    ax = axes[1, 0]
    h0_unweighted_q95 = [np.percentile(all_h0_data[c]['lifetimes'], 95) for c in config_names]
    ax.plot(r0_values, h0_unweighted_q95, 'b--', linewidth=2, markersize=8, marker='s', label='Unweighted')
    ax.plot(r0_values, h0_q95, 'b-', linewidth=2, markersize=10, marker='o', label='Weighted')
    ax.set_xlabel('r_0 (cluster std dev)', fontsize=11)
    ax.set_ylabel('Q95 Lifetime', fontsize=11)
    ax.set_title('H0 Q95 Lifetime: Weighted vs Unweighted', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # H1 Q95 comparison
    ax = axes[1, 1]
    h1_unweighted_q95 = [np.percentile(all_h1_data[c]['lifetimes'], 95) for c in config_names]
    ax.plot(r0_values, h1_unweighted_q95, 'g--', linewidth=2, markersize=8, marker='s', label='Unweighted')
    ax.plot(r0_values, h1_q95, 'g-', linewidth=2, markersize=10, marker='o', label='Weighted')
    ax.set_xlabel('r_0 (cluster std dev)', fontsize=11)
    ax.set_ylabel('Q95 Lifetime', fontsize=11)
    ax.set_title('H1 Q95 Lifetime: Weighted vs Unweighted', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Effect of num_points Weighting on Lifetime Statistics', fontsize=14, y=0.995)
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'weighted_vs_unweighted_comparison.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # =========================================================================
    # Save Statistics to JSON
    # =========================================================================
    print()
    print("Saving weighted statistics to JSON...")

    results_dir = '../../outputs/analysis_results'
    os.makedirs(results_dir, exist_ok=True)

    output_stats = {
        'analysis_type': 'weighted_lifetime_distributions',
        'weighting': 'num_points',
        'configurations': config_names,
        'h0_weighted_statistics': all_h0_stats,
        'h1_weighted_statistics': all_h1_stats,
        'scaling_with_r0': {
            'r0_values': r0_values,
            'h0_weighted_means': [all_h0_stats[c]['weighted_mean'] for c in config_names],
            'h1_weighted_means': [all_h1_stats[c]['weighted_mean'] for c in config_names],
            'h0_weighted_medians': [all_h0_stats[c]['weighted_median'] for c in config_names],
            'h1_weighted_medians': [all_h1_stats[c]['weighted_median'] for c in config_names],
            'h0_weighted_stds': [all_h0_stats[c]['weighted_std'] for c in config_names],
            'h1_weighted_stds': [all_h1_stats[c]['weighted_std'] for c in config_names]
        }
    }

    output_file = os.path.join(results_dir, 'weighted_lifetime_statistics.json')
    with open(output_file, 'w') as f:
        json.dump(output_stats, f, indent=2)
    print(f"  Saved: {output_file}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("="*80)
    print("Weighted Analysis complete!")
    print("="*80)
    print()
    print("Generated files:")
    print("  Figures:")
    print("  - h0_weighted_lifetime_kde.png")
    print("  - h1_weighted_lifetime_kde.png")
    print("  - h0_weighted_lifetime_subplots.png")
    print("  - h1_weighted_lifetime_subplots.png")
    print("  - weighted_lifetime_cdfs.png")
    print("  - weighted_lifetime_statistics_vs_r0.png")
    print("  - weighted_vs_unweighted_comparison.png")
    print()
    print("  Statistics:")
    print("  - weighted_lifetime_statistics.json")
    print()
    print("Key observations (weighted):")
    print(f"  H0 weighted mean lifetime: {all_h0_stats['stddev_0.2']['weighted_mean']:.4f} -> {all_h0_stats['stddev_1.0']['weighted_mean']:.4f}")
    print(f"  H1 weighted mean lifetime: {all_h1_stats['stddev_0.2']['weighted_mean']:.4f} -> {all_h1_stats['stddev_1.0']['weighted_mean']:.4f}")
    print()
