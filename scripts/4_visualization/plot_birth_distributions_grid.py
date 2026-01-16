#!/usr/bin/env python
"""
Plot Birth Time Distributions as 3x3 Grid

Creates a figure similar to the death distribution grid with:
    - Rows: r_0 values (cluster standard deviation): 0.1, 0.5, 1.0
    - Columns: c values (mean cluster size): 5, 10, 50
    - Each subplot: KDE of birth time distribution

This script generates separate figures for H0 and H1 birth times.

H1 birth times are particularly interesting as they represent when loops
form, which corresponds to the scale at which clusters start connecting.
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
    np.ndarray : Array of all birth times across all samples
    """
    config_dir = os.path.join(base_dir, config_name)

    # Load metadata to know how many samples
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
            # Extract birth times (column 0)
            birth_times = h_data[:, 0]
            all_birth_times.extend(birth_times)

    return np.array(all_birth_times)


def load_persistence_data(config_name, homology_dim='h0', base_dir=ENSEMBLE_BASE_DIR):
    """
    Load all persistence data (birth, death, persistence) for a configuration.

    Returns:
    --------
    dict : Dictionary with 'births', 'deaths', 'persistence' arrays
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

def plot_grid_birth_distributions(homology_dim='h0', base_dir=ENSEMBLE_BASE_DIR,
                                   output_dir=OUTPUT_DIR):
    """
    Create a 3x3 grid plot of birth time distributions.

    Parameters:
    -----------
    homology_dim : str
        'h0' or 'h1'
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    # Title for the whole figure
    homology_label = 'H_0 (Connected Components)' if homology_dim == 'h0' else 'H_1 (Loops/Holes)'
    fig.suptitle(f'Birth Time Distributions - {homology_label}\n(Normalized to Unit Intensity)',
                 fontsize=16, fontweight='bold', y=0.98)

    # Add row and column labels
    # Column labels (c values)
    for j, c in enumerate(C_VALUES):
        axes[0, j].set_title(f'c = {c}', fontsize=14, fontweight='bold', pad=10)

    # Color for KDE
    kde_color = '#1f77b4' if homology_dim == 'h0' else '#2ca02c'
    fill_color = '#1f77b4' if homology_dim == 'h0' else '#2ca02c'

    # Track global max for consistent y-axis (optional)
    all_max_densities = []
    all_data = {}

    # First pass: load all data and compute KDEs
    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            config_name = get_config_name(r0, c)
            try:
                birth_times = load_birth_times(config_name, homology_dim, base_dir)
                all_data[(i, j)] = birth_times
            except Exception as e:
                print(f"Warning: Could not load data for {config_name}: {e}")
                all_data[(i, j)] = None

    # Determine x-axis range from all data
    all_birth_times_flat = []
    for data in all_data.values():
        if data is not None and len(data) > 0:
            all_birth_times_flat.extend(data)

    if len(all_birth_times_flat) > 0:
        x_max = np.percentile(all_birth_times_flat, 99)  # Use 99th percentile to avoid outliers
        x_max = min(x_max, 5.0)  # Cap at 5.0 for normalized data
    else:
        x_max = 5.0

    x_range = np.linspace(0, x_max, 500)

    # Second pass: plot
    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            ax = axes[i, j]
            birth_times = all_data[(i, j)]

            if birth_times is None or len(birth_times) < 10:
                ax.text(0.5, 0.5, 'Insufficient Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12, color='red')
                ax.set_xlim(0, x_max)
                continue

            # Compute KDE
            try:
                kde = gaussian_kde(birth_times, bw_method='scott')
                density = kde(x_range)

                # Plot KDE
                ax.plot(x_range, density, color=kde_color, linewidth=2,
                       label=f'{homology_dim.upper()}(r)')
                ax.fill_between(x_range, density, alpha=0.3, color=fill_color)

                # Add vertical line for mean
                mean_val = np.mean(birth_times)
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5,
                          alpha=0.7, label=f'Mean: {mean_val:.2f}')

                # Statistics annotation
                stats_text = f'n = {len(birth_times):,}\nmu = {mean_val:.3f}\nsigma = {np.std(birth_times):.3f}'
                ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                       fontsize=8, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                all_max_densities.append(np.max(density))

            except Exception as e:
                ax.text(0.5, 0.5, f'KDE Error:\n{str(e)[:30]}', transform=ax.transAxes,
                       ha='center', va='center', fontsize=10, color='red')

            # Formatting
            ax.set_xlim(0, x_max)
            ax.set_xlabel('r', fontsize=11)
            ax.set_ylabel('Correlation Function', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

            # Add subplot title with parameters
            ax.text(0.5, 1.02, f'r_0 = {r0}, c = {c}', transform=ax.transAxes,
                   ha='center', fontsize=10, style='italic')

            # Legend in first subplot only
            if i == 0 and j == 0:
                ax.legend(loc='upper right', fontsize=8)

    # Add row labels on the left
    for i, r0 in enumerate(R0_VALUES):
        fig.text(0.02, 0.77 - i * 0.28, f'r_0 = {r0}', fontsize=14, fontweight='bold',
                rotation=90, va='center', ha='center')

    # Adjust layout
    plt.tight_layout(rect=[0.04, 0.02, 1, 0.95])

    # Save figure
    output_file = os.path.join(output_dir, f'{homology_dim}_birth_distribution_grid.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")

    plt.close()

    return output_file


def plot_grid_with_reference_style(homology_dim='h0', base_dir=ENSEMBLE_BASE_DIR,
                                    output_dir=OUTPUT_DIR):
    """
    Create a 3x3 grid plot matching the reference image style more closely.

    The reference shows:
    - Rows indexed by r_0 (0.1, 0.5, 1.0)
    - Columns indexed by c (5, 10, 50)
    - Each plot has the same basic KDE style
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with subplots - larger figure
    fig = plt.figure(figsize=(16, 14))

    # Create main grid of subplots with space for labels
    gs = fig.add_gridspec(3, 3, left=0.1, right=0.95, top=0.88, bottom=0.08,
                          wspace=0.25, hspace=0.35)

    # Title for the whole figure
    homology_label = 'H_0' if homology_dim == 'h0' else 'H_1'

    # Add column headers
    for j, c in enumerate(C_VALUES):
        fig.text(0.23 + j * 0.285, 0.94, f'c = {c}', fontsize=18, fontweight='bold',
                ha='center', va='center')

    # Add row headers
    for i, r0 in enumerate(R0_VALUES):
        fig.text(0.03, 0.75 - i * 0.275, f'r_0 = {r0}', fontsize=18, fontweight='bold',
                ha='center', va='center', rotation=90)

    # Colors for the distribution
    color = '#1f77b4' if homology_dim == 'h0' else '#2ca02c'

    # Load all data first
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

    # Determine global x-range
    all_times = []
    for data in all_data.values():
        if data is not None and len(data) > 0:
            all_times.extend(data)

    x_max = min(np.percentile(all_times, 99) if all_times else 5.0, 5.0)
    x_range = np.linspace(0, x_max, 500)

    # Plot each subplot
    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            ax = fig.add_subplot(gs[i, j])
            birth_times = all_data[(i, j)]

            # Subplot title
            ax.set_title(f'r_0 = {r0}, c = {c}', fontsize=11, pad=5)

            if birth_times is None or len(birth_times) < 10:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, x_max)
                ax.set_xlabel('r', fontsize=11)
                ax.set_ylabel('Correlation Function', fontsize=11)
                continue

            # Compute and plot KDE
            try:
                kde = gaussian_kde(birth_times, bw_method='scott')
                density = kde(x_range)

                ax.plot(x_range, density, color=color, linewidth=2,
                       label=f'{homology_label}(r)')
                ax.fill_between(x_range, density, alpha=0.25, color=color)

                # Add legend
                ax.legend(loc='upper right', fontsize=9)

            except Exception as e:
                ax.text(0.5, 0.5, 'KDE Error', transform=ax.transAxes,
                       ha='center', va='center', fontsize=10, color='red')

            ax.set_xlim(0, x_max)
            ax.set_ylim(bottom=0)
            ax.set_xlabel('r', fontsize=11)
            ax.set_ylabel('Correlation Function', fontsize=11)
            ax.grid(True, alpha=0.3)

    # Save figure
    output_file = os.path.join(output_dir, f'{homology_dim}_birth_distribution_subplots.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")

    plt.close()

    return output_file


def plot_grid_histograms(homology_dim='h0', base_dir=ENSEMBLE_BASE_DIR,
                          output_dir=OUTPUT_DIR, n_bins=50):
    """
    Create a 3x3 grid plot with histograms instead of KDE.

    Parameters:
    -----------
    homology_dim : str
        'h0' or 'h1'
    n_bins : int
        Number of bins for histogram
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 14))

    # Create main grid of subplots with space for labels
    gs = fig.add_gridspec(3, 3, left=0.1, right=0.95, top=0.88, bottom=0.08,
                          wspace=0.25, hspace=0.35)

    # Title for the whole figure
    homology_label = 'H_0 (Connected Components)' if homology_dim == 'h0' else 'H_1 (Loops/Holes)'
    fig.suptitle(f'Birth Time Distributions (Histogram) - {homology_label}\n(Normalized to Unit Intensity)',
                 fontsize=16, fontweight='bold', y=0.98)

    # Add column headers
    for j, c in enumerate(C_VALUES):
        fig.text(0.23 + j * 0.285, 0.94, f'c = {c}', fontsize=18, fontweight='bold',
                ha='center', va='center')

    # Add row headers
    for i, r0 in enumerate(R0_VALUES):
        fig.text(0.03, 0.75 - i * 0.275, f'r_0 = {r0}', fontsize=18, fontweight='bold',
                ha='center', va='center', rotation=90)

    # Colors for the histogram
    color = '#1f77b4' if homology_dim == 'h0' else '#2ca02c'

    # Load all data first
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

    # Determine global x-range
    all_times = []
    for data in all_data.values():
        if data is not None and len(data) > 0:
            all_times.extend(data)

    x_max = min(np.percentile(all_times, 99) if all_times else 5.0, 5.0)
    bins = np.linspace(0, x_max, n_bins + 1)

    # Plot each subplot
    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            ax = fig.add_subplot(gs[i, j])
            birth_times = all_data[(i, j)]

            # Subplot title
            ax.set_title(f'r_0 = {r0}, c = {c}', fontsize=11, pad=5)

            if birth_times is None or len(birth_times) < 10:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, x_max)
                ax.set_xlabel('r', fontsize=11)
                ax.set_ylabel('Count', fontsize=11)
                continue

            # Plot histogram
            ax.hist(birth_times, bins=bins, color=color, alpha=0.7,
                   edgecolor='black', linewidth=0.5)

            # Add vertical line for mean
            mean_val = np.mean(birth_times)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      alpha=0.8, label=f'Mean: {mean_val:.2f}')

            # Statistics annotation
            stats_text = f'n = {len(birth_times):,}\nmu = {mean_val:.3f}\nsigma = {np.std(birth_times):.3f}'
            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlim(0, x_max)
            ax.set_ylim(bottom=0)
            ax.set_xlabel('r', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='upper right', fontsize=8)

    # Save figure
    output_file = os.path.join(output_dir, f'{homology_dim}_birth_distribution_histogram.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")

    plt.close()

    return output_file


def plot_grid_histograms_density(homology_dim='h0', base_dir=ENSEMBLE_BASE_DIR,
                                  output_dir=OUTPUT_DIR, n_bins=50):
    """
    Create a 3x3 grid plot with density-normalized histograms (for comparison with KDE).

    Parameters:
    -----------
    homology_dim : str
        'h0' or 'h1'
    n_bins : int
        Number of bins for histogram
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 14))

    # Create main grid of subplots with space for labels
    gs = fig.add_gridspec(3, 3, left=0.1, right=0.95, top=0.88, bottom=0.08,
                          wspace=0.25, hspace=0.35)

    # Title for the whole figure
    homology_label = 'H_0 (Connected Components)' if homology_dim == 'h0' else 'H_1 (Loops/Holes)'
    fig.suptitle(f'Birth Time Distributions (Density Histogram) - {homology_label}\n(Normalized to Unit Intensity)',
                 fontsize=16, fontweight='bold', y=0.98)

    # Add column headers
    for j, c in enumerate(C_VALUES):
        fig.text(0.23 + j * 0.285, 0.94, f'c = {c}', fontsize=18, fontweight='bold',
                ha='center', va='center')

    # Add row headers
    for i, r0 in enumerate(R0_VALUES):
        fig.text(0.03, 0.75 - i * 0.275, f'r_0 = {r0}', fontsize=18, fontweight='bold',
                ha='center', va='center', rotation=90)

    # Colors for the histogram
    color = '#1f77b4' if homology_dim == 'h0' else '#2ca02c'

    # Load all data first
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

    # Determine global x-range
    all_times = []
    for data in all_data.values():
        if data is not None and len(data) > 0:
            all_times.extend(data)

    x_max = min(np.percentile(all_times, 99) if all_times else 5.0, 5.0)
    bins = np.linspace(0, x_max, n_bins + 1)

    # Plot each subplot
    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            ax = fig.add_subplot(gs[i, j])
            birth_times = all_data[(i, j)]

            # Subplot title
            ax.set_title(f'r_0 = {r0}, c = {c}', fontsize=11, pad=5)

            if birth_times is None or len(birth_times) < 10:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, x_max)
                ax.set_xlabel('r', fontsize=11)
                ax.set_ylabel('Density', fontsize=11)
                continue

            # Plot density histogram
            ax.hist(birth_times, bins=bins, color=color, alpha=0.7,
                   edgecolor='black', linewidth=0.5, density=True)

            # Add vertical line for mean
            mean_val = np.mean(birth_times)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      alpha=0.8, label=f'Mean: {mean_val:.2f}')

            # Statistics annotation
            stats_text = f'n = {len(birth_times):,}\nmu = {mean_val:.3f}\nsigma = {np.std(birth_times):.3f}'
            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlim(0, x_max)
            ax.set_ylim(bottom=0)
            ax.set_xlabel('r', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='upper right', fontsize=8)

    # Save figure
    output_file = os.path.join(output_dir, f'{homology_dim}_birth_distribution_histogram_density.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")

    plt.close()

    return output_file


def plot_birth_vs_death_comparison(homology_dim='h1', base_dir=ENSEMBLE_BASE_DIR,
                                    output_dir=OUTPUT_DIR):
    """
    Create a 3x3 grid comparing birth and death distributions on the same plot.

    This is particularly useful for H1 to show the relationship between
    when loops form (birth) and when they fill (death).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 14))

    # Create main grid of subplots with space for labels
    gs = fig.add_gridspec(3, 3, left=0.1, right=0.95, top=0.88, bottom=0.08,
                          wspace=0.25, hspace=0.35)

    homology_label = 'H_0' if homology_dim == 'h0' else 'H_1'
    fig.suptitle(f'{homology_label} Birth vs Death Time Distributions\n(Blue = Birth, Green = Death)',
                 fontsize=16, fontweight='bold', y=0.98)

    # Add column headers
    for j, c in enumerate(C_VALUES):
        fig.text(0.23 + j * 0.285, 0.94, f'c = {c}', fontsize=18, fontweight='bold',
                ha='center', va='center')

    # Add row headers
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

    # Determine global x-range
    all_times = []
    for data in all_data.values():
        if data is not None:
            all_times.extend(data['births'])
            all_times.extend(data['deaths'])

    x_max = min(np.percentile(all_times, 99) if all_times else 5.0, 5.0)
    x_range = np.linspace(0, x_max, 500)

    # Plot each subplot
    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            ax = fig.add_subplot(gs[i, j])
            data = all_data[(i, j)]

            ax.set_title(f'r_0 = {r0}, c = {c}', fontsize=11, pad=5)

            if data is None or len(data['births']) < 10:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, x_max)
                continue

            # Plot birth distribution (blue)
            try:
                kde_birth = gaussian_kde(data['births'], bw_method='scott')
                density_birth = kde_birth(x_range)
                ax.plot(x_range, density_birth, color='blue', linewidth=2,
                       label=f'Birth (mu={np.mean(data["births"]):.2f})')
                ax.fill_between(x_range, density_birth, alpha=0.2, color='blue')
            except:
                pass

            # Plot death distribution (green)
            try:
                kde_death = gaussian_kde(data['deaths'], bw_method='scott')
                density_death = kde_death(x_range)
                ax.plot(x_range, density_death, color='green', linewidth=2,
                       label=f'Death (mu={np.mean(data["deaths"]):.2f})')
                ax.fill_between(x_range, density_death, alpha=0.2, color='green')
            except:
                pass

            ax.set_xlim(0, x_max)
            ax.set_ylim(bottom=0)
            ax.set_xlabel('r', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

    # Save figure
    output_file = os.path.join(output_dir, f'{homology_dim}_birth_vs_death_grid.png')
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

    # Check if data exists
    if not os.path.exists(ENSEMBLE_BASE_DIR):
        print(f"Error: Data directory '{ENSEMBLE_BASE_DIR}' not found.")
        print("Please run the data generation and persistence computation scripts first:")
        print("  1. python generate_grid_ensembles.py")
        print("  2. python compute_grid_persistence.py")
        exit(1)

    print(f"Data directory: {ENSEMBLE_BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Generate H0 plots
    print("Generating H0 birth distribution plots...")
    try:
        plot_grid_birth_distributions('h0', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_with_reference_style('h0', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_histograms('h0', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_histograms_density('h0', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
    except Exception as e:
        print(f"Error generating H0 plots: {e}")

    print()

    # Generate H1 plots
    print("Generating H1 birth distribution plots...")
    try:
        plot_grid_birth_distributions('h1', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_with_reference_style('h1', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_histograms('h1', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_histograms_density('h1', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        # Also create birth vs death comparison for H1
        plot_birth_vs_death_comparison('h1', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
    except Exception as e:
        print(f"Error generating H1 plots: {e}")

    print()
    print("="*80)
    print("Visualization complete!")
    print("="*80)
    print()
    print("Generated files:")
    print(f"  {OUTPUT_DIR}/")
    print("  |-- h0_birth_distribution_grid.png (KDE detailed)")
    print("  |-- h0_birth_distribution_subplots.png (KDE reference style)")
    print("  |-- h0_birth_distribution_histogram.png (histogram counts)")
    print("  |-- h0_birth_distribution_histogram_density.png (histogram density)")
    print("  |-- h1_birth_distribution_grid.png (KDE detailed)")
    print("  |-- h1_birth_distribution_subplots.png (KDE reference style)")
    print("  |-- h1_birth_distribution_histogram.png (histogram counts)")
    print("  |-- h1_birth_distribution_histogram_density.png (histogram density)")
    print("  |-- h1_birth_vs_death_grid.png (birth vs death comparison)")
    print()
