#!/usr/bin/env python
"""
Generate All Grid Distribution Plots

Creates comprehensive 3x3 grid visualizations for:
- H0: Birth, Death, Lifetime distributions (unweighted and weighted by num_points)
- H1: Birth, Death, Lifetime distributions (unweighted and weighted by cycle_area)
- H1: Cycle area and num_vertices distributions
- H0 vs H1 comparison plots

All plots use partial data (skips configurations still computing).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import subprocess
from datetime import datetime

# ==============================================================================
# Configuration
# ==============================================================================

ENSEMBLE_BASE_DIR = "../../data/grid_ensemble_data"
OUTPUT_DIR = "../../outputs/figures_incomplete"

R0_VALUES = [0.1, 0.5, 1.0]
C_VALUES = [5, 10, 50]

# Cutoff time for "recent" data (from current computation run)
CUTOFF_TIME = "2026-01-15 23:48:00"

plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['figure.dpi'] = 150

# ==============================================================================
# Helper Functions
# ==============================================================================

def get_config_name(r0, c):
    return f'r0_{r0:.1f}_c_{c}'


def has_recent_data(config_name):
    """Check if config has recent persistence data from current run."""
    result = subprocess.run(
        f'find {ENSEMBLE_BASE_DIR}/{config_name} -name "*_persistence.npy" -newermt "{CUTOFF_TIME}" 2>/dev/null | wc -l',
        shell=True, capture_output=True, text=True
    )
    return int(result.stdout.strip()) >= 50


def load_persistence_data(config_name):
    """
    Load all persistence data for a configuration.

    Returns dict with:
        'h0_birth', 'h0_death', 'h0_lifetime', 'h0_num_points'
        'h1_birth', 'h1_death', 'h1_lifetime', 'h1_num_vertices', 'h1_cycle_area'
    """
    config_dir = os.path.join(ENSEMBLE_BASE_DIR, config_name)

    with open(os.path.join(config_dir, 'persistence_metadata.json'), 'r') as f:
        metadata = json.load(f)

    n_samples = metadata['n_samples']

    h0_data = {'birth': [], 'death': [], 'num_points': []}
    h1_data = {'birth': [], 'death': [], 'num_vertices': [], 'cycle_area': []}

    for i in range(n_samples):
        persistence_file = os.path.join(config_dir, f'sample_{i:03d}_persistence.npy')
        if not os.path.exists(persistence_file):
            continue

        ph_data = np.load(persistence_file, allow_pickle=True).item()

        # H0: [birth, death, num_points]
        if len(ph_data['h0']) > 0:
            h0_data['birth'].extend(ph_data['h0'][:, 0])
            h0_data['death'].extend(ph_data['h0'][:, 1])
            h0_data['num_points'].extend(ph_data['h0'][:, 2])

        # H1: [birth, death, num_vertices, cycle_area]
        if len(ph_data['h1']) > 0:
            h1_data['birth'].extend(ph_data['h1'][:, 0])
            h1_data['death'].extend(ph_data['h1'][:, 1])
            h1_data['num_vertices'].extend(ph_data['h1'][:, 2])
            h1_data['cycle_area'].extend(ph_data['h1'][:, 3])

    # Convert to arrays and compute lifetimes
    result = {}

    for key in h0_data:
        result[f'h0_{key}'] = np.array(h0_data[key])
    result['h0_lifetime'] = result['h0_death'] - result['h0_birth']

    for key in h1_data:
        result[f'h1_{key}'] = np.array(h1_data[key])
    result['h1_lifetime'] = result['h1_death'] - result['h1_birth']

    return result


def weighted_mean(values, weights):
    if len(values) == 0 or np.sum(weights) == 0:
        return 0
    return np.sum(values * weights) / np.sum(weights)


def weighted_std(values, weights):
    if len(values) == 0 or np.sum(weights) == 0:
        return 0
    w_mean = weighted_mean(values, weights)
    variance = np.sum(weights * (values - w_mean) ** 2) / np.sum(weights)
    return np.sqrt(variance)


# ==============================================================================
# Plotting Functions
# ==============================================================================

def plot_grid_histogram(data_key, weight_key, title, xlabel, output_file,
                        color='#1f77b4', n_bins=50, x_max_fixed=None):
    """
    Create a 3x3 grid histogram plot.

    Args:
        data_key: Key for the data to plot (e.g., 'h0_lifetime')
        weight_key: Key for weights (e.g., 'h0_num_points') or None for unweighted
        title: Plot title
        xlabel: X-axis label
        output_file: Output file path
        color: Histogram color
        n_bins: Number of bins
        x_max_fixed: If provided, use this fixed x_max for all subplots
    """
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 3, left=0.1, right=0.95, top=0.88, bottom=0.08,
                          wspace=0.25, hspace=0.35)

    weight_label = f"weighted by {weight_key.split('_')[-1]}" if weight_key else "unweighted"
    fig.suptitle(f'{title}\n({weight_label}, PARTIAL - computation in progress)',
                 fontsize=16, fontweight='bold', y=0.98)

    for j, c in enumerate(C_VALUES):
        fig.text(0.23 + j * 0.285, 0.94, f'c = {c}', fontsize=14, fontweight='bold',
                ha='center', va='center')

    for i, r0 in enumerate(R0_VALUES):
        fig.text(0.03, 0.75 - i * 0.275, f'r₀ = {r0}', fontsize=14, fontweight='bold',
                ha='center', va='center', rotation=90)

    # Load all data first to compute consistent axis limits per row
    all_data = {}
    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            config_name = get_config_name(r0, c)
            if has_recent_data(config_name):
                try:
                    all_data[(i, j)] = load_persistence_data(config_name)
                except:
                    all_data[(i, j)] = None
            else:
                all_data[(i, j)] = None

    # Compute per-row axis limits
    row_params = {}
    for i in range(len(R0_VALUES)):
        if x_max_fixed is not None:
            # Use fixed x_max for all rows
            x_max = x_max_fixed
        else:
            # Compute from data
            row_values = []
            for j in range(len(C_VALUES)):
                if all_data[(i, j)] is not None:
                    vals = all_data[(i, j)].get(data_key, np.array([]))
                    if len(vals) > 0:
                        row_values.extend(vals)

            if row_values:
                x_max = np.percentile(row_values, 99.5)
            else:
                x_max = 3.0

        row_params[i] = {'x_max': x_max, 'bins': np.linspace(0, x_max, n_bins + 1)}

    # Plot
    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            ax = fig.add_subplot(gs[i, j])
            config_name = get_config_name(r0, c)

            if all_data[(i, j)] is None:
                ax.text(0.5, 0.5, 'Computing...\n(check back later)',
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, color='gray', style='italic')
                ax.set_facecolor('#f0f0f0')
                ax.set_title(f'r₀ = {r0}, c = {c}', fontsize=11, pad=5)
                ax.set_xlim(0, row_params[i]['x_max'])
                continue

            data = all_data[(i, j)]
            values = data.get(data_key, np.array([]))
            weights = data.get(weight_key, None) if weight_key else None

            if len(values) < 10:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'r₀ = {r0}, c = {c}', fontsize=11, pad=5)
                continue

            bins = row_params[i]['bins']

            if weights is not None:
                ax.hist(values, bins=bins, weights=weights, color=color, alpha=0.7,
                       edgecolor='black', linewidth=0.5)
                w_mean = weighted_mean(values, weights)
                w_std = weighted_std(values, weights)
                stats_text = f'n = {len(values):,}\nw.μ = {w_mean:.4f}\nw.σ = {w_std:.4f}'
            else:
                ax.hist(values, bins=bins, color=color, alpha=0.7,
                       edgecolor='black', linewidth=0.5)
                w_mean = np.mean(values)
                w_std = np.std(values)
                stats_text = f'n = {len(values):,}\nμ = {w_mean:.4f}\nσ = {w_std:.4f}'

            ax.axvline(w_mean, color='red', linestyle='--', linewidth=2,
                      alpha=0.8, label=f'Mean: {w_mean:.3f}')

            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_title(f'r₀ = {r0}, c = {c}', fontsize=11, pad=5)
            ax.set_xlim(0, row_params[i]['x_max'])
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel('Weighted Count' if weights is not None else 'Count', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='upper right', fontsize=8)

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_file


def plot_h0_vs_h1_comparison(output_file, x_max_fixed=10.0):
    """Create H0 vs H1 lifetime comparison grid."""
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 3, left=0.1, right=0.95, top=0.88, bottom=0.08,
                          wspace=0.25, hspace=0.35)

    fig.suptitle('H0 vs H1 Lifetime Distributions\n(Blue=H0, Green=H1, PARTIAL)',
                 fontsize=16, fontweight='bold', y=0.98)

    for j, c in enumerate(C_VALUES):
        fig.text(0.23 + j * 0.285, 0.94, f'c = {c}', fontsize=14, fontweight='bold',
                ha='center', va='center')

    for i, r0 in enumerate(R0_VALUES):
        fig.text(0.03, 0.75 - i * 0.275, f'r₀ = {r0}', fontsize=14, fontweight='bold',
                ha='center', va='center', rotation=90)

    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            ax = fig.add_subplot(gs[i, j])
            config_name = get_config_name(r0, c)

            if not has_recent_data(config_name):
                ax.text(0.5, 0.5, 'Computing...', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12, color='gray', style='italic')
                ax.set_facecolor('#f0f0f0')
                ax.set_title(f'r₀ = {r0}, c = {c}', fontsize=11, pad=5)
                ax.set_xlim(0, x_max_fixed)
                continue

            try:
                data = load_persistence_data(config_name)
                h0_life = data['h0_lifetime']
                h1_life = data['h1_lifetime']

                x_max = x_max_fixed
                bins = np.linspace(0, x_max, 51)

                if len(h0_life) > 0:
                    ax.hist(h0_life, bins=bins, color='blue', alpha=0.5,
                           label=f'H0 (μ={np.mean(h0_life):.2f})', density=True)
                if len(h1_life) > 0:
                    ax.hist(h1_life, bins=bins, color='green', alpha=0.5,
                           label=f'H1 (μ={np.mean(h1_life):.2f})', density=True)

                ax.set_title(f'r₀ = {r0}, c = {c}', fontsize=11, pad=5)
                ax.set_xlabel('Lifetime', fontsize=11)
                ax.set_ylabel('Density', fontsize=11)
                ax.set_xlim(0, x_max)
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)

            except Exception as e:
                ax.text(0.5, 0.5, f'Error', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_file


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    import time

    def log(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}", flush=True)

    log("="*70)
    log("Generating All Grid Distribution Plots (Partial Data)")
    log("="*70)

    # Check configuration status
    log("\nConfiguration status:")
    complete = 0
    for r0 in R0_VALUES:
        for c in C_VALUES:
            config = get_config_name(r0, c)
            status = "Complete" if has_recent_data(config) else "Pending"
            log(f"  {config}: {status}")
            if has_recent_data(config):
                complete += 1
    log(f"\n{complete}/9 configurations ready\n")

    # Fixed x_max for birth/death/lifetime (matches max_edge_length in persistence computation)
    FILTRATION_MAX = 10.0

    plots = [
        # H0 plots (data_key, weight_key, title, xlabel, output_file, color, x_max_fixed)
        ("h0_birth", None, "H0 Birth Distribution", "Birth Time",
         "h0_birth/h0_birth_unweighted.png", '#1f77b4', FILTRATION_MAX),
        ("h0_birth", "h0_num_points", "H0 Birth Distribution (Weighted)", "Birth Time",
         "h0_birth/h0_birth_weighted.png", '#1f77b4', FILTRATION_MAX),
        ("h0_death", None, "H0 Death Distribution", "Death Time",
         "h0_death/h0_death_unweighted.png", '#1f77b4', FILTRATION_MAX),
        ("h0_death", "h0_num_points", "H0 Death Distribution (Weighted)", "Death Time",
         "h0_death/h0_death_weighted.png", '#1f77b4', FILTRATION_MAX),
        ("h0_lifetime", None, "H0 Lifetime Distribution", "Lifetime",
         "h0_lifetime/h0_lifetime_unweighted.png", '#1f77b4', FILTRATION_MAX),
        ("h0_lifetime", "h0_num_points", "H0 Lifetime Distribution (Weighted)", "Lifetime",
         "h0_lifetime/h0_lifetime_weighted.png", '#1f77b4', FILTRATION_MAX),

        # H1 plots
        ("h1_birth", None, "H1 Birth Distribution", "Birth Time",
         "h1_birth/h1_birth_unweighted.png", '#2ca02c', FILTRATION_MAX),
        ("h1_birth", "h1_cycle_area", "H1 Birth Distribution (Weighted by Area)", "Birth Time",
         "h1_birth/h1_birth_weighted_area.png", '#2ca02c', FILTRATION_MAX),
        ("h1_death", None, "H1 Death Distribution", "Death Time",
         "h1_death/h1_death_unweighted.png", '#2ca02c', FILTRATION_MAX),
        ("h1_death", "h1_cycle_area", "H1 Death Distribution (Weighted by Area)", "Death Time",
         "h1_death/h1_death_weighted_area.png", '#2ca02c', FILTRATION_MAX),
        ("h1_lifetime", None, "H1 Lifetime Distribution", "Lifetime",
         "h1_lifetime/h1_lifetime_unweighted.png", '#2ca02c', FILTRATION_MAX),
        ("h1_lifetime", "h1_cycle_area", "H1 Lifetime Distribution (Weighted by Area)", "Lifetime",
         "h1_lifetime/h1_lifetime_weighted_area.png", '#2ca02c', FILTRATION_MAX),

        # H1 additional (use data-driven x_max)
        ("h1_cycle_area", None, "H1 Cycle Area Distribution", "Cycle Area",
         "h1_cycle_area/h1_cycle_area_unweighted.png", '#9467bd', None),
        ("h1_num_vertices", None, "H1 Num Vertices Distribution", "Number of Vertices",
         "h1_num_vertices/h1_num_vertices_unweighted.png", '#8c564b', None),
    ]

    total_start = time.time()

    for i, (data_key, weight_key, title, xlabel, rel_path, color, x_max_fixed) in enumerate(plots):
        log(f"[{i+1}/{len(plots)+1}] {title}...")

        output_path = os.path.join(OUTPUT_DIR, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        start = time.time()
        plot_grid_histogram(data_key, weight_key, title, xlabel, output_path, color,
                           x_max_fixed=x_max_fixed)
        elapsed = time.time() - start
        log(f"    Saved: {rel_path} ({elapsed:.1f}s)")

    # H0 vs H1 comparison
    log(f"[{len(plots)+1}/{len(plots)+1}] H0 vs H1 Lifetime Comparison...")
    comparison_path = os.path.join(OUTPUT_DIR, "comparison/h0_vs_h1_lifetime.png")
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    plot_h0_vs_h1_comparison(comparison_path, x_max_fixed=FILTRATION_MAX)
    log(f"    Saved: comparison/h0_vs_h1_lifetime.png")

    total_elapsed = time.time() - total_start
    log("")
    log("="*70)
    log(f"Complete! Generated {len(plots)+1} plots in {total_elapsed:.1f}s")
    log(f"Output directory: {OUTPUT_DIR}")
    log("="*70)
