"""
Analyze H0 Death Time Distributions Across Configurations

This script computes and visualizes the distribution of H0 death times
for all four cluster configurations using the ensemble data.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy.stats import gaussian_kde, ks_2samp

# ==============================================================================
# Load H0 Death Times from Ensemble Data
# ==============================================================================

def load_all_h0_death_times(config_name, base_dir='../../data/ensemble_data'):
    """
    Load all H0 death times for a given configuration.
    
    Parameters:
    -----------
    config_name : str
        Configuration name (e.g., 'stddev_0.2')
    base_dir : str
        Base directory containing the ensemble data
        
    Returns:
    --------
    np.ndarray : Array of all H0 death times across all samples
    """
    config_dir = os.path.join(base_dir, config_name)
    
    # Load metadata to know how many samples
    with open(os.path.join(config_dir, 'persistence_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    n_samples = metadata['n_samples']
    
    all_death_times = []
    
    for i in range(n_samples):
        persistence_file = os.path.join(config_dir, f'sample_{i:03d}_persistence.npy')
        ph_data = np.load(persistence_file, allow_pickle=True).item()
        
        h0 = ph_data['h0']
        if len(h0) > 0:
            # Extract death times (column 1)
            death_times = h0[:, 1]
            all_death_times.extend(death_times)
    
    return np.array(all_death_times)


def compute_statistics(death_times):
    """
    Compute summary statistics for death times.
    
    Parameters:
    -----------
    death_times : np.ndarray
        Array of death times
        
    Returns:
    --------
    dict : Dictionary of statistics
    """
    return {
        'count': len(death_times),
        'mean': np.mean(death_times),
        'median': np.median(death_times),
        'std': np.std(death_times),
        'min': np.min(death_times),
        'max': np.max(death_times),
        'q25': np.percentile(death_times, 25),
        'q75': np.percentile(death_times, 75)
    }


# ==============================================================================
# Main Analysis
# ==============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("H0 Death Time Distribution Analysis")
    print("="*80)
    print()
    
    # Define configurations
    config_names = ['stddev_0.2', 'stddev_0.4', 'stddev_0.6', 'stddev_1.0']
    config_labels = ['r₀ = 0.2 (Very Tight)', 'r₀ = 0.4 (Tight)', 
                     'r₀ = 0.6 (Baseline)', 'r₀ = 1.0 (Loose)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Load all death times
    all_death_times = {}
    all_statistics = {}
    
    print("Loading H0 death times from ensemble data...")
    for config_name in config_names:
        print(f"  Loading {config_name}...")
        death_times = load_all_h0_death_times(config_name)
        all_death_times[config_name] = death_times
        all_statistics[config_name] = compute_statistics(death_times)
    
    print()
    print("="*80)
    print("Summary Statistics")
    print("="*80)
    print()
    print(f"{'Config':<15} {'Count':<10} {'Mean':<10} {'Median':<10} {'Std':<10} {'Max':<10}")
    print("-"*80)
    for config_name, stats in all_statistics.items():
        print(f"{config_name:<15} {stats['count']:<10} "
              f"{stats['mean']:<10.4f} {stats['median']:<10.4f} "
              f"{stats['std']:<10.4f} {stats['max']:<10.4f}")
    print()
    
    # -------------------------------------------------------------------------
    # Plot 1: Overlapping Histograms
    # -------------------------------------------------------------------------
    print("Creating overlapping histograms...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Choose bin edges that work for all distributions
    max_death = max(all_statistics[c]['max'] for c in config_names)
    bins = np.linspace(0, min(max_death, 2.0), 60)  # Focus on 0-2.0 range
    
    for config_name, label, color in zip(config_names, config_labels, colors):
        death_times = all_death_times[config_name]
        ax.hist(death_times, bins=bins, alpha=0.5, label=label, 
                color=color, density=True, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Death Time (Filtration Radius)', fontsize=13)
    ax.set_ylabel('Probability Density', fontsize=13)
    ax.set_title('H0 Death Time Distributions Across Cluster Configurations', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = '../../outputs/figures/death_distributions/h0_death_distribution_overlapping.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    
    # -------------------------------------------------------------------------
    # Plot 2: Separate Subplots
    # -------------------------------------------------------------------------
    print("Creating separate subplot histograms...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (config_name, label, color) in enumerate(zip(config_names, config_labels, colors)):
        death_times = all_death_times[config_name]
        stats = all_statistics[config_name]
        
        ax = axes[idx]
        
        # Histogram
        n, bins_out, patches = ax.hist(death_times, bins=50, alpha=0.7, 
                                        color=color, edgecolor='black', linewidth=0.8)
        
        # Add vertical lines for mean and median
        ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, 
                   label=f"Mean: {stats['mean']:.3f}")
        ax.axvline(stats['median'], color='darkred', linestyle=':', linewidth=2,
                   label=f"Median: {stats['median']:.3f}")
        
        ax.set_xlabel('Death Time (Filtration Radius)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f"{label}\nN={stats['count']:,} features", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add text box with statistics
        textstr = f"Std: {stats['std']:.3f}\nMax: {stats['max']:.3f}"
        props = dict(boxstyle='round', facecolor=color, alpha=0.15)
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)
    
    fig.suptitle('H0 Death Time Distributions - Detailed View', fontsize=16, y=0.995)
    plt.tight_layout()
    output_file = '../../outputs/figures/death_distributions/h0_death_distribution_subplots.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    
    # -------------------------------------------------------------------------
    # Plot 3: Cumulative Distribution Functions (CDFs)
    # -------------------------------------------------------------------------
    print("Creating cumulative distribution functions...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for config_name, label, color in zip(config_names, config_labels, colors):
        death_times = all_death_times[config_name]
        
        # Sort for CDF
        sorted_times = np.sort(death_times)
        cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        
        ax.plot(sorted_times, cdf, label=label, color=color, linewidth=2.5, alpha=0.8)
    
    ax.set_xlabel('Death Time (Filtration Radius)', fontsize=13)
    ax.set_ylabel('Cumulative Probability', fontsize=13)
    ax.set_title('Cumulative Distribution Functions of H0 Death Times', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 2.0])  # Focus on relevant range
    
    plt.tight_layout()
    output_file = '../../outputs/figures/death_distributions/h0_death_distribution_cdf.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    
    # -------------------------------------------------------------------------
    # Plot 4: Box Plots
    # -------------------------------------------------------------------------
    print("Creating box plots...")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Prepare data for box plot
    data_for_boxplot = [all_death_times[c] for c in config_names]
    
    bp = ax.boxplot(data_for_boxplot, tick_labels=config_labels, patch_artist=True,
                     showmeans=True, meanline=True,
                     medianprops=dict(color='darkred', linewidth=2),
                     meanprops=dict(color='red', linewidth=2, linestyle='--'),
                     boxprops=dict(alpha=0.7),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Death Time (Filtration Radius)', fontsize=13)
    ax.set_title('H0 Death Time Distributions - Box Plot Comparison', fontsize=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x labels for better readability
    plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    output_file = '../../outputs/figures/death_distributions/h0_death_distribution_boxplot.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    
    # -------------------------------------------------------------------------
    # Plot 5: Kernel Density Estimation (KDE)
    # -------------------------------------------------------------------------
    print("Creating kernel density estimation plot...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x_range = np.linspace(0, 2.0, 500)
    
    for config_name, label, color in zip(config_names, config_labels, colors):
        death_times = all_death_times[config_name]
        
        # Compute KDE
        kde = gaussian_kde(death_times)
        density = kde(x_range)
        
        ax.plot(x_range, density, label=label, color=color, linewidth=2.5, alpha=0.8)
        ax.fill_between(x_range, density, alpha=0.2, color=color)
    
    ax.set_xlabel('Death Time (Filtration Radius)', fontsize=13)
    ax.set_ylabel('Probability Density', fontsize=13)
    ax.set_title('H0 Death Time Distributions - Kernel Density Estimation', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 2.0])
    
    plt.tight_layout()
    output_file = '../../outputs/figures/death_distributions/h0_death_distribution_kde.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    
    # -------------------------------------------------------------------------
    # Statistical Tests
    # -------------------------------------------------------------------------
    print()
    print("="*80)
    print("Statistical Tests (Kolmogorov-Smirnov)")
    print("="*80)
    print()
    print("Testing if distributions are significantly different:")
    print()
    
    # Compare each pair
    for i in range(len(config_names)):
        for j in range(i+1, len(config_names)):
            config1 = config_names[i]
            config2 = config_names[j]
            
            ks_stat, p_value = ks_2samp(all_death_times[config1], 
                                         all_death_times[config2])
            
            significant = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else ""))
            
            print(f"{config1} vs {config2}:")
            print(f"  KS statistic: {ks_stat:.4f}, p-value: {p_value:.2e} {significant}")
    
    print()
    print("="*80)
    print("✓ Analysis complete!")
    print("="*80)
    print()
    print("Generated files:")
    print("  - h0_death_distribution_overlapping.png")
    print("  - h0_death_distribution_subplots.png")
    print("  - h0_death_distribution_cdf.png")
    print("  - h0_death_distribution_boxplot.png")
    print("  - h0_death_distribution_kde.png")
    print()

