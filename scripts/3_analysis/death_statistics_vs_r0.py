"""
Death Time Statistics vs Cluster Standard Deviation (r₀)

This script computes various statistics from the H0 death time distributions
across different cluster configurations and plots them as a function of r₀.

This allows direct comparison between the clustering parameter and the 
persistent homology features.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy import stats

# ==============================================================================
# Configuration
# ==============================================================================

BASE_DIR = '../../data/ensemble_data'
OUTPUT_DIR = '../../outputs/figures/statistics_vs_r0'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Automatically detect all configurations
def detect_configurations(base_dir=BASE_DIR):
    """
    Automatically detect all configurations by scanning the directory.
    Returns a dictionary mapping config_name to r₀ value.
    """
    configs = {}
    if not os.path.exists(base_dir):
        return configs
    
    for item in os.listdir(base_dir):
        if item.startswith('stddev_') and os.path.isdir(os.path.join(base_dir, item)):
            # Extract r₀ value from directory name
            try:
                r0_str = item.replace('stddev_', '')
                r0 = float(r0_str)
                configs[item] = r0
            except ValueError:
                print(f"Warning: Could not parse r₀ from directory name '{item}'")
    
    return configs

CONFIGURATIONS = detect_configurations()

# ==============================================================================
# Data Loading
# ==============================================================================

def load_all_h0_death_times(config_name, base_dir=BASE_DIR):
    """
    Load all H0 death times for a given configuration.
    
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
    Compute comprehensive statistics for death times.
    
    Parameters:
    -----------
    death_times : np.ndarray
        Array of death times
        
    Returns:
    --------
    dict : Dictionary of statistics
    """
    if len(death_times) == 0:
        return None
    
    # Compute mode using KDE (location of maximum density/max height)
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(death_times)
        # Sample the KDE to find the maximum
        x_range = np.linspace(np.min(death_times), np.max(death_times), 1000)
        kde_values = kde(x_range)
        mode = x_range[np.argmax(kde_values)]
    except:
        # If KDE fails, use histogram-based mode
        hist, bin_edges = np.histogram(death_times, bins=50)
        mode = (bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist) + 1]) / 2
    
    return {
        'count': len(death_times),
        'mean': np.mean(death_times),
        'median': np.median(death_times),
        'mode': mode,  # Location of maximum height of the distribution
        'std': np.std(death_times),
        'var': np.var(death_times),
        'min': np.min(death_times),
        'max': np.max(death_times),
        'range': np.max(death_times) - np.min(death_times),
        'q25': np.percentile(death_times, 25),
        'q75': np.percentile(death_times, 75),
        'iqr': np.percentile(death_times, 75) - np.percentile(death_times, 25),
        'q10': np.percentile(death_times, 10),
        'q90': np.percentile(death_times, 90),
        'skewness': stats.skew(death_times),
        'kurtosis': stats.kurtosis(death_times),
        'cv': np.std(death_times) / np.mean(death_times) if np.mean(death_times) > 0 else 0  # Coefficient of variation
    }


def compute_per_sample_statistics(config_name, base_dir=BASE_DIR):
    """
    Compute statistics for each individual sample (for error bars).
    
    Returns:
    --------
    dict : Dictionary with arrays of statistics across samples
    """
    config_dir = os.path.join(base_dir, config_name)
    
    with open(os.path.join(config_dir, 'persistence_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    n_samples = metadata['n_samples']
    
    sample_stats = {
        'mean': [],
        'median': [],
        'mode': [],
        'std': [],
        'count': [],
        'max': [],
        'q75': [],
        'skewness': [],
        'kurtosis': []
    }
    
    for i in range(n_samples):
        persistence_file = os.path.join(config_dir, f'sample_{i:03d}_persistence.npy')
        ph_data = np.load(persistence_file, allow_pickle=True).item()
        
        h0 = ph_data['h0']
        if len(h0) > 0:
            death_times = h0[:, 1]
            sample_stats['mean'].append(np.mean(death_times))
            sample_stats['median'].append(np.median(death_times))
            sample_stats['std'].append(np.std(death_times))
            sample_stats['count'].append(len(death_times))
            sample_stats['max'].append(np.max(death_times))
            sample_stats['q75'].append(np.percentile(death_times, 75))
            sample_stats['skewness'].append(stats.skew(death_times))
            sample_stats['kurtosis'].append(stats.kurtosis(death_times))
            
            # Compute mode (location of max height) for this sample
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(death_times)
                x_range = np.linspace(np.min(death_times), np.max(death_times), 500)
                kde_values = kde(x_range)
                mode = x_range[np.argmax(kde_values)]
                sample_stats['mode'].append(mode)
            except:
                # Fallback to histogram mode
                hist, bin_edges = np.histogram(death_times, bins=30)
                mode = (bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist) + 1]) / 2
                sample_stats['mode'].append(mode)
    
    # Convert to arrays and compute mean and std across samples
    return {
        key: {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': np.array(values)
        }
        for key, values in sample_stats.items()
    }


# ==============================================================================
# Main Analysis
# ==============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("Computing Death Time Statistics vs r₀")
    print("="*80)
    print()
    
    # Collect data for all configurations
    r0_values = []
    all_statistics = {}
    per_sample_statistics = {}
    
    for config_name in sorted(CONFIGURATIONS.keys()):
        r0 = CONFIGURATIONS[config_name]
        r0_values.append(r0)
        
        print(f"Processing {config_name} (r₀ = {r0})...")
        
        # Load all death times
        death_times = load_all_h0_death_times(config_name)
        
        # Compute overall statistics
        stats_dict = compute_statistics(death_times)
        all_statistics[r0] = stats_dict
        
        # Compute per-sample statistics (for error bars)
        sample_stats = compute_per_sample_statistics(config_name)
        per_sample_statistics[r0] = sample_stats
        
        print(f"  Total death times: {stats_dict['count']}")
        print(f"  Mean: {stats_dict['mean']:.4f} ± {sample_stats['mean']['std']:.4f}")
        print(f"  Median: {stats_dict['median']:.4f}")
        print()
    
    r0_values = np.array(r0_values)
    
    # -------------------------------------------------------------------------
    # Create comprehensive plots
    # -------------------------------------------------------------------------
    
    print("Creating plots...")
    
    # Define colors
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 6))
    
    # -------------------------------------------------------------------------
    # Plot 1a: Mean Death Time (Separate Figure)
    # -------------------------------------------------------------------------
    # Extract values
    means = [all_statistics[r0]['mean'] for r0 in r0_values]
    mean_stds = [per_sample_statistics[r0]['mean']['std'] for r0 in r0_values]
    medians = [all_statistics[r0]['median'] for r0 in r0_values]
    median_stds = [per_sample_statistics[r0]['median']['std'] for r0 in r0_values]
    modes = [all_statistics[r0]['mode'] for r0 in r0_values]
    mode_stds = [per_sample_statistics[r0]['mode']['std'] for r0 in r0_values]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(r0_values, means, yerr=mean_stds, 
                marker='o', markersize=12, linewidth=3, capsize=6, capthick=2.5,
                color='#3498db', label='Mean death time', alpha=0.8)
    ax.set_xlabel('Cluster Standard Deviation (r₀)', fontsize=14)
    ax.set_ylabel('Mean Death Time', fontsize=14)
    ax.set_title('Mean H0 Death Time vs r₀', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12)
    
    # Add note about error bars
    ax.text(0.02, 0.98, 'Error bars: ± 1 std across samples', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'mean_vs_r0.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Plot 1b: Median Death Time (Separate Figure)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(r0_values, medians, yerr=median_stds,
                marker='s', markersize=12, linewidth=3, capsize=6, capthick=2.5,
                color='#2ecc71', label='Median death time', alpha=0.8)
    ax.set_xlabel('Cluster Standard Deviation (r₀)', fontsize=14)
    ax.set_ylabel('Median Death Time', fontsize=14)
    ax.set_title('Median H0 Death Time vs r₀', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12)
    
    # Add note about error bars
    ax.text(0.02, 0.98, 'Error bars: ± 1 std across samples', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'median_vs_r0.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Plot 1c: Location of Max Height (Mode) (Separate Figure)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(r0_values, modes, yerr=mode_stds,
                marker='D', markersize=12, linewidth=3, capsize=6, capthick=2.5,
                color='#e74c3c', label='Location of max height', alpha=0.8)
    ax.set_xlabel('Cluster Standard Deviation (r₀)', fontsize=14)
    ax.set_ylabel('Location of Max Height', fontsize=14)
    ax.set_title('Location of Max Height in Death Time Distribution vs r₀', 
                fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12)
    
    # Add note about error bars and what this represents
    note_text = 'Error bars: ± 1 std across samples\nThis is the peak location of the distribution'
    ax.text(0.02, 0.98, note_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'max_height_location_vs_r0.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Plot 2: Dispersion Measures (Std, IQR, Range)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Standard deviation
    stds = [all_statistics[r0]['std'] for r0 in r0_values]
    std_stds = [per_sample_statistics[r0]['std']['std'] for r0 in r0_values]
    axes[0].errorbar(r0_values, stds, yerr=std_stds,
                     marker='D', markersize=10, linewidth=2.5, capsize=5, capthick=2,
                     color=colors[2], label='Standard deviation', alpha=0.8)
    axes[0].set_xlabel('r₀', fontsize=13)
    axes[0].set_ylabel('Standard Deviation', fontsize=13)
    axes[0].set_title('Std of Death Times vs r₀', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(fontsize=11)
    
    # IQR
    iqrs = [all_statistics[r0]['iqr'] for r0 in r0_values]
    axes[1].plot(r0_values, iqrs, marker='v', markersize=10, linewidth=2.5,
                 color=colors[3], label='Interquartile range (IQR)', alpha=0.8)
    axes[1].set_xlabel('r₀', fontsize=13)
    axes[1].set_ylabel('IQR (Q75 - Q25)', fontsize=13)
    axes[1].set_title('IQR of Death Times vs r₀', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(fontsize=11)
    
    # Range
    ranges = [all_statistics[r0]['range'] for r0 in r0_values]
    axes[2].plot(r0_values, ranges, marker='^', markersize=10, linewidth=2.5,
                 color=colors[4], label='Range (max - min)', alpha=0.8)
    axes[2].set_xlabel('r₀', fontsize=13)
    axes[2].set_ylabel('Range', fontsize=13)
    axes[2].set_title('Range of Death Times vs r₀', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].legend(fontsize=11)
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'dispersion_vs_r0.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    
    # -------------------------------------------------------------------------
    # Plot 3: Shape Measures (Skewness, Kurtosis)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Skewness
    skewnesses = [all_statistics[r0]['skewness'] for r0 in r0_values]
    skew_stds = [per_sample_statistics[r0]['skewness']['std'] for r0 in r0_values]
    ax1.errorbar(r0_values, skewnesses, yerr=skew_stds,
                 marker='p', markersize=10, linewidth=2.5, capsize=5, capthick=2,
                 color=colors[5], label='Skewness', alpha=0.8)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1.5, label='Symmetric')
    ax1.set_xlabel('Cluster Standard Deviation (r₀)', fontsize=13)
    ax1.set_ylabel('Skewness', fontsize=13)
    ax1.set_title('Skewness of Death Time Distribution vs r₀', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11)
    
    # Kurtosis
    kurtoses = [all_statistics[r0]['kurtosis'] for r0 in r0_values]
    kurt_stds = [per_sample_statistics[r0]['kurtosis']['std'] for r0 in r0_values]
    ax2.errorbar(r0_values, kurtoses, yerr=kurt_stds,
                 marker='h', markersize=10, linewidth=2.5, capsize=5, capthick=2,
                 color='#e74c3c', label='Excess kurtosis', alpha=0.8)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1.5, label='Normal distribution')
    ax2.set_xlabel('Cluster Standard Deviation (r₀)', fontsize=13)
    ax2.set_ylabel('Excess Kurtosis', fontsize=13)
    ax2.set_title('Kurtosis of Death Time Distribution vs r₀', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'shape_vs_r0.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    
    # -------------------------------------------------------------------------
    # Plot 4: Quantiles (Q10, Q25, Q50, Q75, Q90)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))
    
    q10s = [all_statistics[r0]['q10'] for r0 in r0_values]
    q25s = [all_statistics[r0]['q25'] for r0 in r0_values]
    q75s = [all_statistics[r0]['q75'] for r0 in r0_values]
    q90s = [all_statistics[r0]['q90'] for r0 in r0_values]
    
    ax.plot(r0_values, medians, marker='o', markersize=12, linewidth=3, 
            color='#2c3e50', label='Median (Q50)', alpha=0.9, zorder=5)
    ax.plot(r0_values, q25s, marker='s', markersize=10, linewidth=2.5, 
            color='#3498db', label='Q25', alpha=0.8)
    ax.plot(r0_values, q75s, marker='s', markersize=10, linewidth=2.5, 
            color='#e74c3c', label='Q75', alpha=0.8)
    ax.plot(r0_values, q10s, marker='v', markersize=9, linewidth=2, 
            color='#9b59b6', label='Q10', alpha=0.7, linestyle='--')
    ax.plot(r0_values, q90s, marker='^', markersize=9, linewidth=2, 
            color='#e67e22', label='Q90', alpha=0.7, linestyle='--')
    
    # Fill between Q25 and Q75 (IQR)
    ax.fill_between(r0_values, q25s, q75s, alpha=0.2, color='gray', label='IQR (Q25-Q75)')
    
    ax.set_xlabel('Cluster Standard Deviation (r₀)', fontsize=13)
    ax.set_ylabel('Death Time (Filtration Radius)', fontsize=13)
    ax.set_title('Quantiles of Death Time Distribution vs r₀', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper left')
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'quantiles_vs_r0.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    
    # -------------------------------------------------------------------------
    # Plot 5: Maximum Death Time and Feature Count
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Maximum death time
    maxs = [all_statistics[r0]['max'] for r0 in r0_values]
    max_stds = [per_sample_statistics[r0]['max']['std'] for r0 in r0_values]
    ax1.errorbar(r0_values, maxs, yerr=max_stds,
                 marker='*', markersize=15, linewidth=2.5, capsize=5, capthick=2,
                 color='#f39c12', label='Maximum death time', alpha=0.8)
    ax1.set_xlabel('Cluster Standard Deviation (r₀)', fontsize=13)
    ax1.set_ylabel('Maximum Death Time', fontsize=13)
    ax1.set_title('Maximum H0 Death Time vs r₀', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11)
    
    # Feature count
    counts = [all_statistics[r0]['count'] for r0 in r0_values]
    count_stds = [per_sample_statistics[r0]['count']['std'] for r0 in r0_values]
    count_means = [per_sample_statistics[r0]['count']['mean'] for r0 in r0_values]
    ax2.errorbar(r0_values, count_means, yerr=count_stds,
                 marker='o', markersize=10, linewidth=2.5, capsize=5, capthick=2,
                 color='#16a085', label='Number of H0 features', alpha=0.8)
    ax2.set_xlabel('Cluster Standard Deviation (r₀)', fontsize=13)
    ax2.set_ylabel('Number of H0 Features per Sample', fontsize=13)
    ax2.set_title('H0 Feature Count vs r₀', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'max_and_count_vs_r0.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    
    # -------------------------------------------------------------------------
    # Plot 6: Comprehensive Overview (All Key Statistics)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # 1. Mean
    axes[0].errorbar(r0_values, means, yerr=mean_stds, marker='o', markersize=8, 
                     linewidth=2, capsize=4, color='#3498db', alpha=0.8)
    axes[0].set_title('Mean Death Time', fontweight='bold')
    axes[0].set_xlabel('r₀')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Median
    axes[1].errorbar(r0_values, medians, yerr=median_stds, marker='s', markersize=8,
                     linewidth=2, capsize=4, color='#2ecc71', alpha=0.8)
    axes[1].set_title('Median Death Time', fontweight='bold')
    axes[1].set_xlabel('r₀')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Location of Max Height
    axes[2].errorbar(r0_values, modes, yerr=mode_stds, marker='D', markersize=8,
                     linewidth=2, capsize=4, color='#e74c3c', alpha=0.8)
    axes[2].set_title('Location of Max Height', fontweight='bold')
    axes[2].set_xlabel('r₀')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Std
    axes[3].errorbar(r0_values, stds, yerr=std_stds, marker='v', markersize=8,
                     linewidth=2, capsize=4, color=colors[2], alpha=0.8)
    axes[3].set_title('Standard Deviation', fontweight='bold')
    axes[3].set_xlabel('r₀')
    axes[3].grid(True, alpha=0.3)
    
    # 5. Skewness
    axes[4].errorbar(r0_values, skewnesses, yerr=skew_stds, marker='p', markersize=8,
                     linewidth=2, capsize=4, color=colors[5], alpha=0.8)
    axes[4].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[4].set_title('Skewness', fontweight='bold')
    axes[4].set_xlabel('r₀')
    axes[4].grid(True, alpha=0.3)
    
    # 6. Max
    axes[5].errorbar(r0_values, maxs, yerr=max_stds, marker='*', markersize=10,
                     linewidth=2, capsize=4, color='#f39c12', alpha=0.8)
    axes[5].set_title('Maximum Death Time', fontweight='bold')
    axes[5].set_xlabel('r₀')
    axes[5].grid(True, alpha=0.3)
    
    # 7. Feature count
    axes[6].errorbar(r0_values, count_means, yerr=count_stds, marker='o', markersize=8,
                     linewidth=2, capsize=4, color='#16a085', alpha=0.8)
    axes[6].set_title('Feature Count per Sample', fontweight='bold')
    axes[6].set_xlabel('r₀')
    axes[6].grid(True, alpha=0.3)
    
    # 8. Comparison of mean, median, mode
    axes[7].plot(r0_values, means, marker='o', markersize=7, linewidth=2,
                 color='#3498db', label='Mean', alpha=0.8)
    axes[7].plot(r0_values, medians, marker='s', markersize=7, linewidth=2,
                 color='#2ecc71', label='Median', alpha=0.8)
    axes[7].plot(r0_values, modes, marker='D', markersize=7, linewidth=2,
                 color='#e74c3c', label='Max Height', alpha=0.8)
    axes[7].set_title('Central Tendency Comparison', fontweight='bold')
    axes[7].set_xlabel('r₀')
    axes[7].legend(fontsize=8)
    axes[7].grid(True, alpha=0.3)
    
    fig.suptitle('H0 Death Time Statistics vs Cluster Standard Deviation (r₀)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    output_file = os.path.join(OUTPUT_DIR, 'comprehensive_overview_vs_r0.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    
    # -------------------------------------------------------------------------
    # Save statistics to JSON
    # -------------------------------------------------------------------------
    print("\nSaving statistics to JSON...")
    
    statistics_summary = {
        'configurations': {
            f'r0_{r0:.1f}': {
                'r0': float(r0),
                'overall_statistics': {
                    key: float(value) if not isinstance(value, (list, dict)) else value
                    for key, value in all_statistics[r0].items()
                },
                'per_sample_statistics': {
                    key: {
                        'mean': float(value['mean']),
                        'std': float(value['std'])
                    }
                    for key, value in per_sample_statistics[r0].items()
                }
            }
            for r0 in r0_values
        },
        'parameters': {
            'lambda_parent': 0.1,
            'lambda_daughter': 30,
            'window_size': 15,
            'max_edge_length': 3.0,
            'n_samples_per_config': 50
        }
    }
    
    json_file = os.path.join(OUTPUT_DIR, 'statistics_vs_r0.json')
    with open(json_file, 'w') as f:
        json.dump(statistics_summary, f, indent=4)
    print(f"  ✓ Saved: {json_file}")
    
    # -------------------------------------------------------------------------
    # Print summary table
    # -------------------------------------------------------------------------
    print()
    print("="*95)
    print("Summary Table: Death Time Statistics vs r₀")
    print("="*95)
    print()
    print(f"{'r₀':<8} {'Mean':<10} {'Median':<10} {'MaxHeight':<12} {'Std':<10} {'Skew':<10} {'Max':<10} {'Count':<10}")
    print("-"*95)
    for r0 in r0_values:
        print(f"{r0:<8.1f} "
              f"{all_statistics[r0]['mean']:<10.4f} "
              f"{all_statistics[r0]['median']:<10.4f} "
              f"{all_statistics[r0]['mode']:<12.4f} "
              f"{all_statistics[r0]['std']:<10.4f} "
              f"{all_statistics[r0]['skewness']:<10.4f} "
              f"{all_statistics[r0]['max']:<10.4f} "
              f"{per_sample_statistics[r0]['count']['mean']:<10.1f}")
    print("-"*95)
    print()
    print("Note: MaxHeight = Location of maximum height in the distribution (mode)")
    
    print("="*80)
    print("✓ Analysis complete!")
    print("="*80)
    print()
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print()
    print("Generated files:")
    print("  - mean_vs_r0.png (mean death time)")
    print("  - median_vs_r0.png (median death time)")
    print("  - max_height_location_vs_r0.png (location of max height in distribution)")
    print("  - dispersion_vs_r0.png (std, IQR, range)")
    print("  - shape_vs_r0.png (skewness and kurtosis)")
    print("  - quantiles_vs_r0.png (Q10, Q25, Q50, Q75, Q90)")
    print("  - max_and_count_vs_r0.png (maximum death time and feature count)")
    print("  - comprehensive_overview_vs_r0.png (all statistics in one figure)")
    print("  - statistics_vs_r0.json (numerical data)")
    print()

