#!/usr/bin/env python
"""
Plot Death Time Distributions as 3×3 Grid

Creates a figure similar to the reference with:
    - Rows: r₀ values (cluster standard deviation): 0.1, 0.5, 1.0
    - Columns: c values (mean cluster size): 5, 10, 50
    - Each subplot: KDE of death time distribution

This script generates separate figures for H0 and H1 death times.
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
OUTPUT_DIR = "../../outputs/figures/grid_death_distributions"

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

def load_death_times(config_name, homology_dim='h0', base_dir=ENSEMBLE_BASE_DIR):
    """
    Load all death times for a given configuration and homology dimension.
    
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
    np.ndarray : Array of all death times across all samples
    """
    config_dir = os.path.join(base_dir, config_name)
    
    # Load metadata to know how many samples
    with open(os.path.join(config_dir, 'persistence_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    n_samples = metadata['n_samples']
    
    all_death_times = []
    
    for i in range(n_samples):
        persistence_file = os.path.join(config_dir, f'sample_{i:03d}_persistence.npy')
        if not os.path.exists(persistence_file):
            continue
            
        ph_data = np.load(persistence_file, allow_pickle=True).item()
        
        h_data = ph_data[homology_dim]
        if len(h_data) > 0:
            # Extract death times (column 1)
            death_times = h_data[:, 1]
            all_death_times.extend(death_times)
    
    return np.array(all_death_times)


def get_config_name(r0, c):
    """Generate configuration name from parameters."""
    return f'r0_{r0:.1f}_c_{c}'


# ==============================================================================
# Plotting Functions
# ==============================================================================

def plot_grid_death_distributions(homology_dim='h0', base_dir=ENSEMBLE_BASE_DIR, 
                                   output_dir=OUTPUT_DIR):
    """
    Create a 3×3 grid plot of death time distributions.
    
    Parameters:
    -----------
    homology_dim : str
        'h0' or 'h1'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    
    # Title for the whole figure
    homology_label = 'H₀ (Connected Components)' if homology_dim == 'h0' else 'H₁ (Loops/Holes)'
    fig.suptitle(f'Death Time Distributions - {homology_label}\n(Normalized to Unit Intensity)', 
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
                death_times = load_death_times(config_name, homology_dim, base_dir)
                all_data[(i, j)] = death_times
            except Exception as e:
                print(f"Warning: Could not load data for {config_name}: {e}")
                all_data[(i, j)] = None
    
    # Determine x-axis range from all data
    all_death_times_flat = []
    for data in all_data.values():
        if data is not None and len(data) > 0:
            all_death_times_flat.extend(data)
    
    if len(all_death_times_flat) > 0:
        x_max = np.percentile(all_death_times_flat, 99)  # Use 99th percentile to avoid outliers
        x_max = min(x_max, 5.0)  # Cap at 5.0 for normalized data
    else:
        x_max = 5.0
    
    x_range = np.linspace(0, x_max, 500)
    
    # Second pass: plot
    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            ax = axes[i, j]
            death_times = all_data[(i, j)]
            
            if death_times is None or len(death_times) < 10:
                ax.text(0.5, 0.5, 'Insufficient Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12, color='red')
                ax.set_xlim(0, x_max)
                continue
            
            # Compute KDE
            try:
                kde = gaussian_kde(death_times, bw_method='scott')
                density = kde(x_range)
                
                # Plot KDE
                ax.plot(x_range, density, color=kde_color, linewidth=2, 
                       label=f'{homology_dim.upper()}(r)')
                ax.fill_between(x_range, density, alpha=0.3, color=fill_color)
                
                # Add vertical line for mean
                mean_val = np.mean(death_times)
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, 
                          alpha=0.7, label=f'Mean: {mean_val:.2f}')
                
                # Statistics annotation
                stats_text = f'n = {len(death_times):,}\nμ = {mean_val:.3f}\nσ = {np.std(death_times):.3f}'
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
            ax.text(0.5, 1.02, f'r₀ = {r0}, c = {c}', transform=ax.transAxes,
                   ha='center', fontsize=10, style='italic')
            
            # Legend in first subplot only
            if i == 0 and j == 0:
                ax.legend(loc='upper right', fontsize=8)
    
    # Add row labels on the left
    for i, r0 in enumerate(R0_VALUES):
        fig.text(0.02, 0.77 - i * 0.28, f'r₀ = {r0}', fontsize=14, fontweight='bold',
                rotation=90, va='center', ha='center')
    
    # Adjust layout
    plt.tight_layout(rect=[0.04, 0.02, 1, 0.95])
    
    # Save figure
    output_file = os.path.join(output_dir, f'{homology_dim}_death_distribution_grid.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    
    plt.close()
    
    return output_file


def plot_grid_with_reference_style(homology_dim='h0', base_dir=ENSEMBLE_BASE_DIR,
                                    output_dir=OUTPUT_DIR):
    """
    Create a 3×3 grid plot matching the reference image style more closely.
    
    The reference shows:
    - Rows indexed by r₀ (0.1, 0.5, 1.0)
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
    homology_label = 'H₀' if homology_dim == 'h0' else 'H₁'
    
    # Add column headers
    for j, c in enumerate(C_VALUES):
        fig.text(0.23 + j * 0.285, 0.94, f'c = {c}', fontsize=18, fontweight='bold',
                ha='center', va='center')
    
    # Add row headers
    for i, r0 in enumerate(R0_VALUES):
        fig.text(0.03, 0.75 - i * 0.275, f'r₀ = {r0}', fontsize=18, fontweight='bold',
                ha='center', va='center', rotation=90)
    
    # Colors for the distribution
    color = '#1f77b4' if homology_dim == 'h0' else '#2ca02c'
    
    # Load all data first
    all_data = {}
    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            config_name = get_config_name(r0, c)
            try:
                death_times = load_death_times(config_name, homology_dim, base_dir)
                all_data[(i, j)] = death_times
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
            death_times = all_data[(i, j)]
            
            # Subplot title
            ax.set_title(f'r₀ = {r0}, c = {c}', fontsize=11, pad=5)
            
            if death_times is None or len(death_times) < 10:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, x_max)
                ax.set_xlabel('r', fontsize=11)
                ax.set_ylabel('Correlation Function', fontsize=11)
                continue
            
            # Compute and plot KDE
            try:
                kde = gaussian_kde(death_times, bw_method='scott')
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
    output_file = os.path.join(output_dir, f'{homology_dim}_death_distribution_subplots.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    
    plt.close()
    
    return output_file


def plot_grid_histograms(homology_dim='h0', base_dir=ENSEMBLE_BASE_DIR,
                          output_dir=OUTPUT_DIR, n_bins=50):
    """
    Create a 3×3 grid plot with histograms instead of KDE.
    
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
    homology_label = 'H₀ (Connected Components)' if homology_dim == 'h0' else 'H₁ (Loops/Holes)'
    fig.suptitle(f'Death Time Distributions (Histogram) - {homology_label}\n(Normalized to Unit Intensity)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add column headers
    for j, c in enumerate(C_VALUES):
        fig.text(0.23 + j * 0.285, 0.94, f'c = {c}', fontsize=18, fontweight='bold',
                ha='center', va='center')
    
    # Add row headers
    for i, r0 in enumerate(R0_VALUES):
        fig.text(0.03, 0.75 - i * 0.275, f'r₀ = {r0}', fontsize=18, fontweight='bold',
                ha='center', va='center', rotation=90)
    
    # Colors for the histogram
    color = '#1f77b4' if homology_dim == 'h0' else '#2ca02c'
    
    # Load all data first
    all_data = {}
    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            config_name = get_config_name(r0, c)
            try:
                death_times = load_death_times(config_name, homology_dim, base_dir)
                all_data[(i, j)] = death_times
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
            death_times = all_data[(i, j)]
            
            # Subplot title
            ax.set_title(f'r₀ = {r0}, c = {c}', fontsize=11, pad=5)
            
            if death_times is None or len(death_times) < 10:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, x_max)
                ax.set_xlabel('r', fontsize=11)
                ax.set_ylabel('Count', fontsize=11)
                continue
            
            # Plot histogram
            ax.hist(death_times, bins=bins, color=color, alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
            
            # Add vertical line for mean
            mean_val = np.mean(death_times)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      alpha=0.8, label=f'Mean: {mean_val:.2f}')
            
            # Statistics annotation
            stats_text = f'n = {len(death_times):,}\nμ = {mean_val:.3f}\nσ = {np.std(death_times):.3f}'
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
    output_file = os.path.join(output_dir, f'{homology_dim}_death_distribution_histogram.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    
    plt.close()
    
    return output_file


def plot_grid_histograms_density(homology_dim='h0', base_dir=ENSEMBLE_BASE_DIR,
                                  output_dir=OUTPUT_DIR, n_bins=50):
    """
    Create a 3×3 grid plot with density-normalized histograms (for comparison with KDE).
    
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
    homology_label = 'H₀ (Connected Components)' if homology_dim == 'h0' else 'H₁ (Loops/Holes)'
    fig.suptitle(f'Death Time Distributions (Density Histogram) - {homology_label}\n(Normalized to Unit Intensity)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add column headers
    for j, c in enumerate(C_VALUES):
        fig.text(0.23 + j * 0.285, 0.94, f'c = {c}', fontsize=18, fontweight='bold',
                ha='center', va='center')
    
    # Add row headers
    for i, r0 in enumerate(R0_VALUES):
        fig.text(0.03, 0.75 - i * 0.275, f'r₀ = {r0}', fontsize=18, fontweight='bold',
                ha='center', va='center', rotation=90)
    
    # Colors for the histogram
    color = '#1f77b4' if homology_dim == 'h0' else '#2ca02c'
    
    # Load all data first
    all_data = {}
    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            config_name = get_config_name(r0, c)
            try:
                death_times = load_death_times(config_name, homology_dim, base_dir)
                all_data[(i, j)] = death_times
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
            death_times = all_data[(i, j)]
            
            # Subplot title
            ax.set_title(f'r₀ = {r0}, c = {c}', fontsize=11, pad=5)
            
            if death_times is None or len(death_times) < 10:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, x_max)
                ax.set_xlabel('r', fontsize=11)
                ax.set_ylabel('Density', fontsize=11)
                continue
            
            # Plot density histogram
            ax.hist(death_times, bins=bins, color=color, alpha=0.7, 
                   edgecolor='black', linewidth=0.5, density=True)
            
            # Add vertical line for mean
            mean_val = np.mean(death_times)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      alpha=0.8, label=f'Mean: {mean_val:.2f}')
            
            # Statistics annotation
            stats_text = f'n = {len(death_times):,}\nμ = {mean_val:.3f}\nσ = {np.std(death_times):.3f}'
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
    output_file = os.path.join(output_dir, f'{homology_dim}_death_distribution_histogram_density.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    
    plt.close()
    
    return output_file


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("Creating 3×3 Grid Death Distribution Plots")
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
    print("Generating H0 death distribution plots...")
    try:
        plot_grid_death_distributions('h0', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_with_reference_style('h0', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_histograms('h0', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_histograms_density('h0', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
    except Exception as e:
        print(f"Error generating H0 plots: {e}")
    
    print()
    
    # Generate H1 plots
    print("Generating H1 death distribution plots...")
    try:
        plot_grid_death_distributions('h1', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_with_reference_style('h1', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_histograms('h1', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        plot_grid_histograms_density('h1', ENSEMBLE_BASE_DIR, OUTPUT_DIR)
    except Exception as e:
        print(f"Error generating H1 plots: {e}")
    
    print()
    print("="*80)
    print("✓ Visualization complete!")
    print("="*80)
    print()
    print("Generated files:")
    print(f"  {OUTPUT_DIR}/")
    print("  ├── h0_death_distribution_grid.png (KDE detailed)")
    print("  ├── h0_death_distribution_subplots.png (KDE reference style)")
    print("  ├── h0_death_distribution_histogram.png (histogram counts)")
    print("  ├── h0_death_distribution_histogram_density.png (histogram density)")
    print("  ├── h1_death_distribution_grid.png (KDE detailed)")
    print("  ├── h1_death_distribution_subplots.png (KDE reference style)")
    print("  ├── h1_death_distribution_histogram.png (histogram counts)")
    print("  └── h1_death_distribution_histogram_density.png (histogram density)")
    print()

