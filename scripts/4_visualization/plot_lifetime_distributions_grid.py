#!/usr/bin/env python
"""
Plot Lifetime (Persistence) Distributions as 3x3 Grid

Creates a figure with:
    - Rows: r_0 values (cluster standard deviation): 0.1, 0.5, 1.0
    - Columns: c values (mean cluster size): 5, 10, 50
    - Each subplot: Histogram of lifetime (death - birth) distribution

Lifetime measures how "significant" or "robust" a topological feature is.
This script generates separate figures for H0 and H1 lifetimes.
Axes are consistent within each row (same r_0) but may differ between rows.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json

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

def plot_grid_histograms(homology_dim='h0', base_dir=ENSEMBLE_BASE_DIR,
                          output_dir=OUTPUT_DIR, n_bins=50):
    """
    Create a 3x3 grid plot with histograms.
    Uses consistent X and Y axis limits within each row (same r_0).
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

    # Load all data
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

    # Compute per-row x_max, y_max, and bins
    row_params = {}
    for i, r0 in enumerate(R0_VALUES):
        row_lifetimes = []
        for j in range(len(C_VALUES)):
            lifetimes = all_data[(i, j)]
            if lifetimes is not None and len(lifetimes) > 0:
                row_lifetimes.extend(lifetimes)

        if row_lifetimes:
            x_max = np.percentile(row_lifetimes, 99.5)
        else:
            x_max = 3.0

        bins = np.linspace(0, x_max, n_bins + 1)

        y_max = 0
        for j in range(len(C_VALUES)):
            lifetimes = all_data[(i, j)]
            if lifetimes is not None and len(lifetimes) >= 10:
                counts, _ = np.histogram(lifetimes, bins=bins)
                y_max = max(y_max, np.max(counts))

        y_max = y_max * 1.05

        row_params[i] = {'x_max': x_max, 'y_max': y_max, 'bins': bins}

    # Plot with row-specific axes
    for i, r0 in enumerate(R0_VALUES):
        x_max = row_params[i]['x_max']
        y_max = row_params[i]['y_max']
        bins = row_params[i]['bins']

        for j, c in enumerate(C_VALUES):
            ax = fig.add_subplot(gs[i, j])
            lifetimes = all_data[(i, j)]

            ax.set_title(f'r_0 = {r0}, c = {c}', fontsize=11, pad=5)

            if lifetimes is None or len(lifetimes) < 10:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, x_max)
                ax.set_ylim(0, y_max)
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
            ax.set_ylim(0, y_max)
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
    Uses consistent X and Y axis limits within each row (same r_0).
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

    # Load all data
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

    # Compute per-row x_max, y_max, and bins
    row_params = {}
    for i, r0 in enumerate(R0_VALUES):
        row_lifetimes = []
        for j in range(len(C_VALUES)):
            lifetimes = all_data[(i, j)]
            if lifetimes is not None and len(lifetimes) > 0:
                row_lifetimes.extend(lifetimes)

        if row_lifetimes:
            x_max = np.percentile(row_lifetimes, 99.5)
        else:
            x_max = 3.0

        bins = np.linspace(0, x_max, n_bins + 1)

        y_max = 0
        for j in range(len(C_VALUES)):
            lifetimes = all_data[(i, j)]
            if lifetimes is not None and len(lifetimes) >= 10:
                counts, _ = np.histogram(lifetimes, bins=bins, density=True)
                y_max = max(y_max, np.max(counts))

        y_max = y_max * 1.05

        row_params[i] = {'x_max': x_max, 'y_max': y_max, 'bins': bins}

    # Plot with row-specific axes
    for i, r0 in enumerate(R0_VALUES):
        x_max = row_params[i]['x_max']
        y_max = row_params[i]['y_max']
        bins = row_params[i]['bins']

        for j, c in enumerate(C_VALUES):
            ax = fig.add_subplot(gs[i, j])
            lifetimes = all_data[(i, j)]

            ax.set_title(f'r_0 = {r0}, c = {c}', fontsize=11, pad=5)

            if lifetimes is None or len(lifetimes) < 10:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, x_max)
                ax.set_ylim(0, y_max)
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
            ax.set_ylim(0, y_max)
            ax.set_xlabel('Lifetime', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='upper right', fontsize=8)

    output_file = os.path.join(output_dir, f'{homology_dim}_lifetime_distribution_histogram_density.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")

    plt.close()
    return output_file


def plot_h0_vs_h1_lifetime_histogram_comparison(base_dir=ENSEMBLE_BASE_DIR, output_dir=OUTPUT_DIR, n_bins=50):
    """
    Create a 3x3 grid comparing H0 and H1 lifetime distributions using histograms.
    Uses consistent X and Y axis limits within each row (same r_0).
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

    # Compute per-row x_max, y_max, and bins
    row_params = {}
    for i, r0 in enumerate(R0_VALUES):
        row_lifetimes = []
        for j in range(len(C_VALUES)):
            h0_lifetimes = all_h0_data[(i, j)]
            h1_lifetimes = all_h1_data[(i, j)]
            if h0_lifetimes is not None and len(h0_lifetimes) > 0:
                row_lifetimes.extend(h0_lifetimes)
            if h1_lifetimes is not None and len(h1_lifetimes) > 0:
                row_lifetimes.extend(h1_lifetimes)

        if row_lifetimes:
            x_max = np.percentile(row_lifetimes, 99.5)
        else:
            x_max = 3.0

        bins = np.linspace(0, x_max, n_bins + 1)

        y_max = 0
        for j in range(len(C_VALUES)):
            h0_lifetimes = all_h0_data[(i, j)]
            h1_lifetimes = all_h1_data[(i, j)]

            if h0_lifetimes is not None and len(h0_lifetimes) >= 10:
                counts, _ = np.histogram(h0_lifetimes, bins=bins, density=True)
                y_max = max(y_max, np.max(counts))

            if h1_lifetimes is not None and len(h1_lifetimes) >= 10:
                counts, _ = np.histogram(h1_lifetimes, bins=bins, density=True)
                y_max = max(y_max, np.max(counts))

        y_max = y_max * 1.05

        row_params[i] = {'x_max': x_max, 'y_max': y_max, 'bins': bins}

    # Plot with row-specific axes
    for i, r0 in enumerate(R0_VALUES):
        x_max = row_params[i]['x_max']
        y_max = row_params[i]['y_max']
        bins = row_params[i]['bins']

        for j, c in enumerate(C_VALUES):
            ax = fig.add_subplot(gs[i, j])
            h0_lifetimes = all_h0_data[(i, j)]
            h1_lifetimes = all_h1_data[(i, j)]

            ax.set_title(f'r_0 = {r0}, c = {c}', fontsize=11, pad=5)

            # Plot H0 (blue)
            if h0_lifetimes is not None and len(h0_lifetimes) > 10:
                ax.hist(h0_lifetimes, bins=bins, color='blue', alpha=0.5,
                       edgecolor='blue', linewidth=0.5, density=True,
                       label=f'H0 (mu={np.mean(h0_lifetimes):.3f})')

            # Plot H1 (green)
            if h1_lifetimes is not None and len(h1_lifetimes) > 10:
                ax.hist(h1_lifetimes, bins=bins, color='green', alpha=0.5,
                       edgecolor='green', linewidth=0.5, density=True,
                       label=f'H1 (mu={np.mean(h1_lifetimes):.3f})')

            ax.set_xlim(0, x_max)
            ax.set_ylim(0, y_max)
            ax.set_xlabel('Lifetime', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

    output_file = os.path.join(output_dir, 'h0_vs_h1_lifetime_histogram.png')
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
        plot_grid_histograms('h0', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_histograms_density('h0', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
    except Exception as e:
        print(f"Error generating H0 plots: {e}")

    print()

    # Generate H1 plots
    print("Generating H1 lifetime distribution plots...")
    try:
        plot_grid_histograms('h1', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_histograms_density('h1', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
    except Exception as e:
        print(f"Error generating H1 plots: {e}")

    print()

    # Generate comparison plot
    print("Generating H0 vs H1 comparison plot...")
    try:
        plot_h0_vs_h1_lifetime_histogram_comparison(ENSEMBLE_BASE_DIR, OUTPUT_DIR)
    except Exception as e:
        print(f"Error generating comparison plot: {e}")

    print()
    print("="*80)
    print("Visualization complete!")
    print("="*80)
    print()
    print("Generated files:")
    print(f"  {OUTPUT_DIR}/")
    print("  |-- h0_lifetime_distribution_histogram.png (histogram counts)")
    print("  |-- h0_lifetime_distribution_histogram_density.png (histogram density)")
    print("  |-- h1_lifetime_distribution_histogram.png (histogram counts)")
    print("  |-- h1_lifetime_distribution_histogram_density.png (histogram density)")
    print("  |-- h0_vs_h1_lifetime_histogram.png (comparison)")
    print()
