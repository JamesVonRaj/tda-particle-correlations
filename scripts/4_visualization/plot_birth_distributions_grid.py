#!/usr/bin/env python
"""
Plot Birth Time Distributions as 3x3 Grid

Creates a figure with:
    - Rows: r_0 values (cluster standard deviation): 0.1, 0.5, 1.0
    - Columns: c values (mean cluster size): 5, 10, 50
    - Each subplot: Histogram of birth time distribution

This script generates separate figures for H0 and H1 birth times.
Axes are consistent within each row (same r_0) but may differ between rows.

H1 birth times are particularly interesting as they represent when loops
form, which corresponds to the scale at which clusters start connecting.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json

# ==============================================================================
# Configuration
# ==============================================================================

ENSEMBLE_BASE_DIR = "../../data/grid_ensemble_data"
OUTPUT_DIR = "../../outputs/figures/grid_birth_distributions"

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

def load_birth_times(config_name, homology_dim='h0', base_dir=ENSEMBLE_BASE_DIR):
    """
    Load all birth times for a given configuration and homology dimension.
    """
    config_dir = os.path.join(base_dir, config_name)

    with open(os.path.join(config_dir, 'persistence_metadata.json'), 'r') as f:
        metadata = json.load(f)

    n_samples = metadata['n_samples']

    all_birth_times = []

    for i in range(n_samples):
        persistence_file = os.path.join(config_dir, f'sample_{i:03d}_persistence.npy')
        if not os.path.exists(persistence_file):
            continue

        ph_data = np.load(persistence_file, allow_pickle=True).item()

        h_data = ph_data[homology_dim]
        if len(h_data) > 0:
            birth_times = h_data[:, 0]
            all_birth_times.extend(birth_times)

    return np.array(all_birth_times)


def load_persistence_data(config_name, homology_dim='h0', base_dir=ENSEMBLE_BASE_DIR):
    """
    Load all persistence data (birth, death, persistence) for a configuration.
    """
    config_dir = os.path.join(base_dir, config_name)

    with open(os.path.join(config_dir, 'persistence_metadata.json'), 'r') as f:
        metadata = json.load(f)

    n_samples = metadata['n_samples']

    all_births = []
    all_deaths = []
    all_persistence = []

    for i in range(n_samples):
        persistence_file = os.path.join(config_dir, f'sample_{i:03d}_persistence.npy')
        if not os.path.exists(persistence_file):
            continue

        ph_data = np.load(persistence_file, allow_pickle=True).item()

        h_data = ph_data[homology_dim]
        if len(h_data) > 0:
            births = h_data[:, 0]
            deaths = h_data[:, 1]
            persistence = deaths - births

            all_births.extend(births)
            all_deaths.extend(deaths)
            all_persistence.extend(persistence)

    return {
        'births': np.array(all_births),
        'deaths': np.array(all_deaths),
        'persistence': np.array(all_persistence)
    }


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
    fig.suptitle(f'Birth Time Distributions (Histogram) - {homology_label}',
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
                birth_times = load_birth_times(config_name, homology_dim, base_dir)
                all_data[(i, j)] = birth_times
            except Exception as e:
                print(f"Warning: Could not load data for {config_name}: {e}")
                all_data[(i, j)] = None

    # Compute per-row x_max, y_max, and bins
    row_params = {}
    for i, r0 in enumerate(R0_VALUES):
        row_times = []
        for j in range(len(C_VALUES)):
            birth_times = all_data[(i, j)]
            if birth_times is not None and len(birth_times) > 0:
                row_times.extend(birth_times)

        if row_times:
            x_max = np.percentile(row_times, 99.5)
        else:
            x_max = 5.0

        bins = np.linspace(0, x_max, n_bins + 1)

        y_max = 0
        for j in range(len(C_VALUES)):
            birth_times = all_data[(i, j)]
            if birth_times is not None and len(birth_times) >= 10:
                counts, _ = np.histogram(birth_times, bins=bins)
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
            birth_times = all_data[(i, j)]

            ax.set_title(f'r_0 = {r0}, c = {c}', fontsize=11, pad=5)

            if birth_times is None or len(birth_times) < 10:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, x_max)
                ax.set_ylim(0, y_max)
                ax.set_xlabel('Birth Time', fontsize=11)
                ax.set_ylabel('Count', fontsize=11)
                continue

            ax.hist(birth_times, bins=bins, color=color, alpha=0.7,
                   edgecolor='black', linewidth=0.5)

            mean_val = np.mean(birth_times)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      alpha=0.8, label=f'Mean: {mean_val:.2f}')

            stats_text = f'n = {len(birth_times):,}\nmu = {mean_val:.3f}\nsigma = {np.std(birth_times):.3f}'
            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlim(0, x_max)
            ax.set_ylim(0, y_max)
            ax.set_xlabel('Birth Time', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='upper right', fontsize=8)

    output_file = os.path.join(output_dir, f'{homology_dim}_birth_distribution_histogram.png')
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
    fig.suptitle(f'Birth Time Distributions (Density Histogram) - {homology_label}',
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
                birth_times = load_birth_times(config_name, homology_dim, base_dir)
                all_data[(i, j)] = birth_times
            except Exception as e:
                print(f"Warning: Could not load data for {config_name}: {e}")
                all_data[(i, j)] = None

    # Compute per-row x_max, y_max, and bins
    row_params = {}
    for i, r0 in enumerate(R0_VALUES):
        row_times = []
        for j in range(len(C_VALUES)):
            birth_times = all_data[(i, j)]
            if birth_times is not None and len(birth_times) > 0:
                row_times.extend(birth_times)

        if row_times:
            x_max = np.percentile(row_times, 99.5)
        else:
            x_max = 5.0

        bins = np.linspace(0, x_max, n_bins + 1)

        y_max = 0
        for j in range(len(C_VALUES)):
            birth_times = all_data[(i, j)]
            if birth_times is not None and len(birth_times) >= 10:
                counts, _ = np.histogram(birth_times, bins=bins, density=True)
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
            birth_times = all_data[(i, j)]

            ax.set_title(f'r_0 = {r0}, c = {c}', fontsize=11, pad=5)

            if birth_times is None or len(birth_times) < 10:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, x_max)
                ax.set_ylim(0, y_max)
                ax.set_xlabel('Birth Time', fontsize=11)
                ax.set_ylabel('Density', fontsize=11)
                continue

            ax.hist(birth_times, bins=bins, color=color, alpha=0.7,
                   edgecolor='black', linewidth=0.5, density=True)

            mean_val = np.mean(birth_times)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      alpha=0.8, label=f'Mean: {mean_val:.2f}')

            stats_text = f'n = {len(birth_times):,}\nmu = {mean_val:.3f}\nsigma = {np.std(birth_times):.3f}'
            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlim(0, x_max)
            ax.set_ylim(0, y_max)
            ax.set_xlabel('Birth Time', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='upper right', fontsize=8)

    output_file = os.path.join(output_dir, f'{homology_dim}_birth_distribution_histogram_density.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")

    plt.close()
    return output_file


def plot_birth_vs_death_histogram_comparison(homology_dim='h1', base_dir=ENSEMBLE_BASE_DIR,
                                              output_dir=OUTPUT_DIR, n_bins=50):
    """
    Create a 3x3 grid comparing birth and death distributions using histograms.
    Uses consistent X and Y axis limits within each row (same r_0).
    """
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 14))

    gs = fig.add_gridspec(3, 3, left=0.1, right=0.95, top=0.88, bottom=0.08,
                          wspace=0.25, hspace=0.35)

    homology_label = 'H_0' if homology_dim == 'h0' else 'H_1'
    fig.suptitle(f'{homology_label} Birth vs Death Time Distributions\n(Blue = Birth, Green = Death)',
                 fontsize=16, fontweight='bold', y=0.98)

    for j, c in enumerate(C_VALUES):
        fig.text(0.23 + j * 0.285, 0.94, f'c = {c}', fontsize=18, fontweight='bold',
                ha='center', va='center')

    for i, r0 in enumerate(R0_VALUES):
        fig.text(0.03, 0.75 - i * 0.275, f'r_0 = {r0}', fontsize=18, fontweight='bold',
                ha='center', va='center', rotation=90)

    # Load all data first
    all_data = {}
    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            config_name = get_config_name(r0, c)
            try:
                data = load_persistence_data(config_name, homology_dim, base_dir)
                all_data[(i, j)] = data
            except Exception as e:
                print(f"Warning: Could not load data for {config_name}: {e}")
                all_data[(i, j)] = None

    # Compute per-row x_max, y_max, and bins
    row_params = {}
    for i, r0 in enumerate(R0_VALUES):
        row_times = []
        for j in range(len(C_VALUES)):
            data = all_data[(i, j)]
            if data is not None:
                row_times.extend(data['births'])
                row_times.extend(data['deaths'])

        if row_times:
            x_max = np.percentile(row_times, 99.5)
        else:
            x_max = 5.0

        bins = np.linspace(0, x_max, n_bins + 1)

        y_max = 0
        for j in range(len(C_VALUES)):
            data = all_data[(i, j)]
            if data is not None and len(data['births']) >= 10:
                counts, _ = np.histogram(data['births'], bins=bins, density=True)
                y_max = max(y_max, np.max(counts))
                counts, _ = np.histogram(data['deaths'], bins=bins, density=True)
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
            data = all_data[(i, j)]

            ax.set_title(f'r_0 = {r0}, c = {c}', fontsize=11, pad=5)

            if data is None or len(data['births']) < 10:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, x_max)
                ax.set_ylim(0, y_max)
                continue

            # Plot birth distribution (blue)
            ax.hist(data['births'], bins=bins, color='blue', alpha=0.5,
                   edgecolor='blue', linewidth=0.5, density=True,
                   label=f'Birth (mu={np.mean(data["births"]):.2f})')

            # Plot death distribution (green)
            ax.hist(data['deaths'], bins=bins, color='green', alpha=0.5,
                   edgecolor='green', linewidth=0.5, density=True,
                   label=f'Death (mu={np.mean(data["deaths"]):.2f})')

            ax.set_xlim(0, x_max)
            ax.set_ylim(0, y_max)
            ax.set_xlabel('Time', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

    output_file = os.path.join(output_dir, f'{homology_dim}_birth_vs_death_histogram.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")

    plt.close()
    return output_file


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":

    print("="*80)
    print("Creating 3x3 Grid Birth Distribution Plots")
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
    print("Generating H0 birth distribution plots...")
    try:
        plot_grid_histograms('h0', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_histograms_density('h0', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
    except Exception as e:
        print(f"Error generating H0 plots: {e}")

    print()

    # Generate H1 plots
    print("Generating H1 birth distribution plots...")
    try:
        plot_grid_histograms('h1', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_histograms_density('h1', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        # Also create birth vs death comparison for H1
        plot_birth_vs_death_histogram_comparison('h1', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
    except Exception as e:
        print(f"Error generating H1 plots: {e}")

    print()
    print("="*80)
    print("Visualization complete!")
    print("="*80)
    print()
    print("Generated files:")
    print(f"  {OUTPUT_DIR}/")
    print("  |-- h0_birth_distribution_histogram.png (histogram counts)")
    print("  |-- h0_birth_distribution_histogram_density.png (histogram density)")
    print("  |-- h1_birth_distribution_histogram.png (histogram counts)")
    print("  |-- h1_birth_distribution_histogram_density.png (histogram density)")
    print("  |-- h1_birth_vs_death_histogram.png (comparison)")
    print()
