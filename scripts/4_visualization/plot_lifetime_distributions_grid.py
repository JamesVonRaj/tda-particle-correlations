#!/usr/bin/env python
"""
Plot Lifetime (Persistence) Distributions as 3x3 Grid

Creates a figure with:
    - Rows: r_0 values (cluster standard deviation): 0.1, 0.5, 1.0
    - Columns: c values (mean cluster size): 5, 10, 50
    - Each subplot: KDE/histogram of lifetime (death - birth) distribution

Lifetime measures how "significant" or "robust" a topological feature is.
This script generates separate figures for H0 and H1 lifetimes.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy.stats import gaussian_kde

# ==============================================================================
# Configuration
# ==============================================================================

ENSEMBLE_BASE_DIR = "../../data/grid_ensemble_data"
OUTPUT_DIR = "../../outputs/figures/grid_lifetime_distributions"

# Grid parameters (must match data generation)
R0_VALUES = [0.1, 0.5, 1.0]
C_VALUES = [5, 10, 50]

# Plot styling
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.dpi'] = 150

# ==============================================================================
# Data Loading Functions
# ==============================================================================

def load_lifetimes(config_name, homology_dim='h0', base_dir=ENSEMBLE_BASE_DIR):
    """
    Load all lifetimes (persistence = death - birth) for a configuration.

    Parameters:
    -----------
    config_name : str
        Configuration name (e.g., 'r0_0.1_c_5')
    homology_dim : str
        'h0' or 'h1'
    base_dir : str
        Base directory containing the ensemble data

    Returns:
    --------
    np.ndarray : Array of all lifetimes across all samples
    """
    config_dir = os.path.join(base_dir, config_name)

    with open(os.path.join(config_dir, 'persistence_metadata.json'), 'r') as f:
        metadata = json.load(f)

    n_samples = metadata['n_samples']

    all_lifetimes = []

    for i in range(n_samples):
        persistence_file = os.path.join(config_dir, f'sample_{i:03d}_persistence.npy')
        if not os.path.exists(persistence_file):
            continue

        ph_data = np.load(persistence_file, allow_pickle=True).item()

        h_data = ph_data[homology_dim]
        if len(h_data) > 0:
            births = h_data[:, 0]
            deaths = h_data[:, 1]
            lifetimes = deaths - births
            all_lifetimes.extend(lifetimes)

    return np.array(all_lifetimes)


def get_config_name(r0, c):
    """Generate configuration name from parameters."""
    return f'r0_{r0:.1f}_c_{c}'


# ==============================================================================
# Plotting Functions
# ==============================================================================

def plot_grid_lifetime_distributions(homology_dim='h0', base_dir=ENSEMBLE_BASE_DIR,
                                      output_dir=OUTPUT_DIR):
    """
    Create a 3x3 grid plot of lifetime distributions (KDE with stats).
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    homology_label = 'H_0 (Connected Components)' if homology_dim == 'h0' else 'H_1 (Loops/Holes)'
    fig.suptitle(f'Lifetime Distributions - {homology_label}\n(Lifetime = Death - Birth)',
                 fontsize=16, fontweight='bold', y=0.98)

    for j, c in enumerate(C_VALUES):
        axes[0, j].set_title(f'c = {c}', fontsize=14, fontweight='bold', pad=10)

    kde_color = '#1f77b4' if homology_dim == 'h0' else '#2ca02c'
    fill_color = '#1f77b4' if homology_dim == 'h0' else '#2ca02c'

    all_data = {}

    # First pass: load all data
    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            config_name = get_config_name(r0, c)
            try:
                lifetimes = load_lifetimes(config_name, homology_dim, base_dir)
                all_data[(i, j)] = lifetimes
            except Exception as e:
                print(f"Warning: Could not load data for {config_name}: {e}")
                all_data[(i, j)] = None

    # Determine x-axis range
    all_lifetimes_flat = []
    for data in all_data.values():
        if data is not None and len(data) > 0:
            all_lifetimes_flat.extend(data)

    if len(all_lifetimes_flat) > 0:
        x_max = np.percentile(all_lifetimes_flat, 99)
        x_max = min(x_max, 3.0)
    else:
        x_max = 3.0

    x_range = np.linspace(0, x_max, 500)

    # Second pass: plot
    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            ax = axes[i, j]
            lifetimes = all_data[(i, j)]

            if lifetimes is None or len(lifetimes) < 10:
                ax.text(0.5, 0.5, 'Insufficient Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12, color='red')
                ax.set_xlim(0, x_max)
                continue

            try:
                kde = gaussian_kde(lifetimes, bw_method='scott')
                density = kde(x_range)

                ax.plot(x_range, density, color=kde_color, linewidth=2,
                       label=f'{homology_dim.upper()}')
                ax.fill_between(x_range, density, alpha=0.3, color=fill_color)

                mean_val = np.mean(lifetimes)
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5,
                          alpha=0.7, label=f'Mean: {mean_val:.3f}')

                stats_text = f'n = {len(lifetimes):,}\nmu = {mean_val:.4f}\nsigma = {np.std(lifetimes):.4f}'
                ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                       fontsize=8, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            except Exception as e:
                ax.text(0.5, 0.5, f'KDE Error', transform=ax.transAxes,
                       ha='center', va='center', fontsize=10, color='red')

            ax.set_xlim(0, x_max)
            ax.set_xlabel('Lifetime', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

            ax.text(0.5, 1.02, f'r_0 = {r0}, c = {c}', transform=ax.transAxes,
                   ha='center', fontsize=10, style='italic')

            if i == 0 and j == 0:
                ax.legend(loc='upper right', fontsize=8)

    for i, r0 in enumerate(R0_VALUES):
        fig.text(0.02, 0.77 - i * 0.28, f'r_0 = {r0}', fontsize=14, fontweight='bold',
                rotation=90, va='center', ha='center')

    plt.tight_layout(rect=[0.04, 0.02, 1, 0.95])

    output_file = os.path.join(output_dir, f'{homology_dim}_lifetime_distribution_grid.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")

    plt.close()
    return output_file


def plot_grid_with_reference_style(homology_dim='h0', base_dir=ENSEMBLE_BASE_DIR,
                                    output_dir=OUTPUT_DIR):
    """
    Create a 3x3 grid plot matching the reference image style.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 14))

    gs = fig.add_gridspec(3, 3, left=0.1, right=0.95, top=0.88, bottom=0.08,
                          wspace=0.25, hspace=0.35)

    homology_label = 'H_0' if homology_dim == 'h0' else 'H_1'

    for j, c in enumerate(C_VALUES):
        fig.text(0.23 + j * 0.285, 0.94, f'c = {c}', fontsize=18, fontweight='bold',
                ha='center', va='center')

    for i, r0 in enumerate(R0_VALUES):
        fig.text(0.03, 0.75 - i * 0.275, f'r_0 = {r0}', fontsize=18, fontweight='bold',
                ha='center', va='center', rotation=90)

    color = '#1f77b4' if homology_dim == 'h0' else '#2ca02c'

    all_data = {}
    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            config_name = get_config_name(r0, c)
            try:
                lifetimes = load_lifetimes(config_name, homology_dim, base_dir)
                all_data[(i, j)] = lifetimes
            except Exception as e:
                print(f"Warning: Could not load data for {config_name}: {e}")
                all_data[(i, j)] = None

    all_lifetimes = []
    for data in all_data.values():
        if data is not None and len(data) > 0:
            all_lifetimes.extend(data)

    x_max = min(np.percentile(all_lifetimes, 99) if all_lifetimes else 3.0, 3.0)
    x_range = np.linspace(0, x_max, 500)

    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            ax = fig.add_subplot(gs[i, j])
            lifetimes = all_data[(i, j)]

            ax.set_title(f'r_0 = {r0}, c = {c}', fontsize=11, pad=5)

            if lifetimes is None or len(lifetimes) < 10:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, x_max)
                ax.set_xlabel('Lifetime', fontsize=11)
                ax.set_ylabel('Density', fontsize=11)
                continue

            try:
                kde = gaussian_kde(lifetimes, bw_method='scott')
                density = kde(x_range)

                ax.plot(x_range, density, color=color, linewidth=2,
                       label=f'{homology_label}')
                ax.fill_between(x_range, density, alpha=0.25, color=color)

                ax.legend(loc='upper right', fontsize=9)

            except Exception as e:
                ax.text(0.5, 0.5, 'KDE Error', transform=ax.transAxes,
                       ha='center', va='center', fontsize=10, color='red')

            ax.set_xlim(0, x_max)
            ax.set_ylim(bottom=0)
            ax.set_xlabel('Lifetime', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.grid(True, alpha=0.3)

    output_file = os.path.join(output_dir, f'{homology_dim}_lifetime_distribution_subplots.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")

    plt.close()
    return output_file


def plot_grid_histograms(homology_dim='h0', base_dir=ENSEMBLE_BASE_DIR,
                          output_dir=OUTPUT_DIR, n_bins=50):
    """
    Create a 3x3 grid plot with histograms.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 14))

    gs = fig.add_gridspec(3, 3, left=0.1, right=0.95, top=0.88, bottom=0.08,
                          wspace=0.25, hspace=0.35)

    homology_label = 'H_0 (Connected Components)' if homology_dim == 'h0' else 'H_1 (Loops/Holes)'
    fig.suptitle(f'Lifetime Distributions (Histogram) - {homology_label}',
                 fontsize=16, fontweight='bold', y=0.98)

    for j, c in enumerate(C_VALUES):
        fig.text(0.23 + j * 0.285, 0.94, f'c = {c}', fontsize=18, fontweight='bold',
                ha='center', va='center')

    for i, r0 in enumerate(R0_VALUES):
        fig.text(0.03, 0.75 - i * 0.275, f'r_0 = {r0}', fontsize=18, fontweight='bold',
                ha='center', va='center', rotation=90)

    color = '#1f77b4' if homology_dim == 'h0' else '#2ca02c'

    all_data = {}
    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            config_name = get_config_name(r0, c)
            try:
                lifetimes = load_lifetimes(config_name, homology_dim, base_dir)
                all_data[(i, j)] = lifetimes
            except Exception as e:
                print(f"Warning: Could not load data for {config_name}: {e}")
                all_data[(i, j)] = None

    all_lifetimes = []
    for data in all_data.values():
        if data is not None and len(data) > 0:
            all_lifetimes.extend(data)

    x_max = min(np.percentile(all_lifetimes, 99) if all_lifetimes else 3.0, 3.0)
    bins = np.linspace(0, x_max, n_bins + 1)

    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            ax = fig.add_subplot(gs[i, j])
            lifetimes = all_data[(i, j)]

            ax.set_title(f'r_0 = {r0}, c = {c}', fontsize=11, pad=5)

            if lifetimes is None or len(lifetimes) < 10:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, x_max)
                ax.set_xlabel('Lifetime', fontsize=11)
                ax.set_ylabel('Count', fontsize=11)
                continue

            ax.hist(lifetimes, bins=bins, color=color, alpha=0.7,
                   edgecolor='black', linewidth=0.5)

            mean_val = np.mean(lifetimes)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      alpha=0.8, label=f'Mean: {mean_val:.3f}')

            stats_text = f'n = {len(lifetimes):,}\nmu = {mean_val:.4f}\nsigma = {np.std(lifetimes):.4f}'
            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlim(0, x_max)
            ax.set_ylim(bottom=0)
            ax.set_xlabel('Lifetime', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='upper right', fontsize=8)

    output_file = os.path.join(output_dir, f'{homology_dim}_lifetime_distribution_histogram.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")

    plt.close()
    return output_file


def plot_grid_histograms_density(homology_dim='h0', base_dir=ENSEMBLE_BASE_DIR,
                                  output_dir=OUTPUT_DIR, n_bins=50):
    """
    Create a 3x3 grid plot with density-normalized histograms.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 14))

    gs = fig.add_gridspec(3, 3, left=0.1, right=0.95, top=0.88, bottom=0.08,
                          wspace=0.25, hspace=0.35)

    homology_label = 'H_0 (Connected Components)' if homology_dim == 'h0' else 'H_1 (Loops/Holes)'
    fig.suptitle(f'Lifetime Distributions (Density Histogram) - {homology_label}',
                 fontsize=16, fontweight='bold', y=0.98)

    for j, c in enumerate(C_VALUES):
        fig.text(0.23 + j * 0.285, 0.94, f'c = {c}', fontsize=18, fontweight='bold',
                ha='center', va='center')

    for i, r0 in enumerate(R0_VALUES):
        fig.text(0.03, 0.75 - i * 0.275, f'r_0 = {r0}', fontsize=18, fontweight='bold',
                ha='center', va='center', rotation=90)

    color = '#1f77b4' if homology_dim == 'h0' else '#2ca02c'

    all_data = {}
    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            config_name = get_config_name(r0, c)
            try:
                lifetimes = load_lifetimes(config_name, homology_dim, base_dir)
                all_data[(i, j)] = lifetimes
            except Exception as e:
                print(f"Warning: Could not load data for {config_name}: {e}")
                all_data[(i, j)] = None

    all_lifetimes = []
    for data in all_data.values():
        if data is not None and len(data) > 0:
            all_lifetimes.extend(data)

    x_max = min(np.percentile(all_lifetimes, 99) if all_lifetimes else 3.0, 3.0)
    bins = np.linspace(0, x_max, n_bins + 1)

    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            ax = fig.add_subplot(gs[i, j])
            lifetimes = all_data[(i, j)]

            ax.set_title(f'r_0 = {r0}, c = {c}', fontsize=11, pad=5)

            if lifetimes is None or len(lifetimes) < 10:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, x_max)
                ax.set_xlabel('Lifetime', fontsize=11)
                ax.set_ylabel('Density', fontsize=11)
                continue

            ax.hist(lifetimes, bins=bins, color=color, alpha=0.7,
                   edgecolor='black', linewidth=0.5, density=True)

            mean_val = np.mean(lifetimes)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      alpha=0.8, label=f'Mean: {mean_val:.3f}')

            stats_text = f'n = {len(lifetimes):,}\nmu = {mean_val:.4f}\nsigma = {np.std(lifetimes):.4f}'
            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlim(0, x_max)
            ax.set_ylim(bottom=0)
            ax.set_xlabel('Lifetime', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='upper right', fontsize=8)

    output_file = os.path.join(output_dir, f'{homology_dim}_lifetime_distribution_histogram_density.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")

    plt.close()
    return output_file


def plot_h0_vs_h1_lifetime_comparison(base_dir=ENSEMBLE_BASE_DIR, output_dir=OUTPUT_DIR):
    """
    Create a 3x3 grid comparing H0 and H1 lifetime distributions on the same plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 14))

    gs = fig.add_gridspec(3, 3, left=0.1, right=0.95, top=0.88, bottom=0.08,
                          wspace=0.25, hspace=0.35)

    fig.suptitle('H0 vs H1 Lifetime Distributions\n(Blue = H0, Green = H1)',
                 fontsize=16, fontweight='bold', y=0.98)

    for j, c in enumerate(C_VALUES):
        fig.text(0.23 + j * 0.285, 0.94, f'c = {c}', fontsize=18, fontweight='bold',
                ha='center', va='center')

    for i, r0 in enumerate(R0_VALUES):
        fig.text(0.03, 0.75 - i * 0.275, f'r_0 = {r0}', fontsize=18, fontweight='bold',
                ha='center', va='center', rotation=90)

    # Load all data
    all_h0_data = {}
    all_h1_data = {}
    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            config_name = get_config_name(r0, c)
            try:
                all_h0_data[(i, j)] = load_lifetimes(config_name, 'h0', base_dir)
                all_h1_data[(i, j)] = load_lifetimes(config_name, 'h1', base_dir)
            except Exception as e:
                print(f"Warning: Could not load data for {config_name}: {e}")
                all_h0_data[(i, j)] = None
                all_h1_data[(i, j)] = None

    # Determine global x-range
    all_lifetimes = []
    for data in list(all_h0_data.values()) + list(all_h1_data.values()):
        if data is not None and len(data) > 0:
            all_lifetimes.extend(data)

    x_max = min(np.percentile(all_lifetimes, 99) if all_lifetimes else 3.0, 3.0)
    x_range = np.linspace(0, x_max, 500)

    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            ax = fig.add_subplot(gs[i, j])
            h0_lifetimes = all_h0_data[(i, j)]
            h1_lifetimes = all_h1_data[(i, j)]

            ax.set_title(f'r_0 = {r0}, c = {c}', fontsize=11, pad=5)

            # Plot H0 (blue)
            if h0_lifetimes is not None and len(h0_lifetimes) > 10:
                try:
                    kde_h0 = gaussian_kde(h0_lifetimes, bw_method='scott')
                    density_h0 = kde_h0(x_range)
                    ax.plot(x_range, density_h0, color='blue', linewidth=2,
                           label=f'H0 (mu={np.mean(h0_lifetimes):.3f})')
                    ax.fill_between(x_range, density_h0, alpha=0.2, color='blue')
                except:
                    pass

            # Plot H1 (green)
            if h1_lifetimes is not None and len(h1_lifetimes) > 10:
                try:
                    kde_h1 = gaussian_kde(h1_lifetimes, bw_method='scott')
                    density_h1 = kde_h1(x_range)
                    ax.plot(x_range, density_h1, color='green', linewidth=2,
                           label=f'H1 (mu={np.mean(h1_lifetimes):.3f})')
                    ax.fill_between(x_range, density_h1, alpha=0.2, color='green')
                except:
                    pass

            ax.set_xlim(0, x_max)
            ax.set_ylim(bottom=0)
            ax.set_xlabel('Lifetime', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

    output_file = os.path.join(output_dir, 'h0_vs_h1_lifetime_grid.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")

    plt.close()
    return output_file


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":

    print("="*80)
    print("Creating 3x3 Grid Lifetime Distribution Plots")
    print("="*80)
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(ENSEMBLE_BASE_DIR):
        print(f"Error: Data directory '{ENSEMBLE_BASE_DIR}' not found.")
        print("Please run the data generation and persistence computation scripts first.")
        exit(1)

    print(f"Data directory: {ENSEMBLE_BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Generate H0 plots
    print("Generating H0 lifetime distribution plots...")
    try:
        plot_grid_lifetime_distributions('h0', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_with_reference_style('h0', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_histograms('h0', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_histograms_density('h0', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
    except Exception as e:
        print(f"Error generating H0 plots: {e}")

    print()

    # Generate H1 plots
    print("Generating H1 lifetime distribution plots...")
    try:
        plot_grid_lifetime_distributions('h1', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_with_reference_style('h1', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_histograms('h1', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_histograms_density('h1', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
    except Exception as e:
        print(f"Error generating H1 plots: {e}")

    print()

    # Generate comparison plot
    print("Generating H0 vs H1 comparison plot...")
    try:
        plot_h0_vs_h1_lifetime_comparison(ENSEMBLE_BASE_DIR, OUTPUT_DIR)
    except Exception as e:
        print(f"Error generating comparison plot: {e}")

    print()
    print("="*80)
    print("Visualization complete!")
    print("="*80)
    print()
    print("Generated files:")
    print(f"  {OUTPUT_DIR}/")
    print("  |-- h0_lifetime_distribution_grid.png (KDE detailed)")
    print("  |-- h0_lifetime_distribution_subplots.png (KDE reference style)")
    print("  |-- h0_lifetime_distribution_histogram.png (histogram counts)")
    print("  |-- h0_lifetime_distribution_histogram_density.png (histogram density)")
    print("  |-- h1_lifetime_distribution_grid.png (KDE detailed)")
    print("  |-- h1_lifetime_distribution_subplots.png (KDE reference style)")
    print("  |-- h1_lifetime_distribution_histogram.png (histogram counts)")
    print("  |-- h1_lifetime_distribution_histogram_density.png (histogram density)")
    print("  |-- h0_vs_h1_lifetime_grid.png (comparison)")
    print()
