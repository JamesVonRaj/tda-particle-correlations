#!/usr/bin/env python
"""
Generate All Figures for LaTeX Report

Creates distribution plots with both x_max=5 and x_max=10 cutoffs
for comparison in the collaborator report.
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
OUTPUT_BASE_DIR = "../../outputs/report_figures"

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
    """Load all persistence data for a configuration."""
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
            if ph_data['h1'].shape[1] > 3:
                h1_data['cycle_area'].extend(ph_data['h1'][:, 3])

    result = {}
    for key in h0_data:
        result[f'h0_{key}'] = np.array(h0_data[key])
    result['h0_lifetime'] = result['h0_death'] - result['h0_birth']

    for key in h1_data:
        result[f'h1_{key}'] = np.array(h1_data[key])
    if len(result['h1_death']) > 0:
        result['h1_lifetime'] = result['h1_death'] - result['h1_birth']
    else:
        result['h1_lifetime'] = np.array([])

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
                        color='#1f77b4', n_bins=50, x_max_fixed=None, normalize=False):
    """Create a 3x3 grid histogram plot with consistent y-axis per row.

    Args:
        normalize: If True, use density=True to normalize histograms to integrate to 1.
    """
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 3, left=0.1, right=0.95, top=0.90, bottom=0.08,
                          wspace=0.25, hspace=0.35)

    weight_label = f"weighted by {weight_key.split('_')[-1]}" if weight_key else "unweighted"
    norm_label = ", normalized" if normalize else ""
    fig.suptitle(f'{title}\n({weight_label}{norm_label})',
                 fontsize=14, fontweight='bold', y=0.97)

    for j, c in enumerate(C_VALUES):
        fig.text(0.23 + j * 0.285, 0.93, f'c = {c}', fontsize=12, fontweight='bold',
                ha='center', va='center')

    for i, r0 in enumerate(R0_VALUES):
        fig.text(0.04, 0.77 - i * 0.275, f'$r_0$ = {r0}', fontsize=12, fontweight='bold',
                ha='center', va='center', rotation=90)

    # Load all data
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

    # Set x_max
    x_max = x_max_fixed if x_max_fixed else 5.0
    bins = np.linspace(0, x_max, n_bins + 1)

    # First pass: compute histogram counts to find y_max per row
    row_y_max = {i: 0 for i in range(len(R0_VALUES))}
    hist_data = {}

    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            if all_data[(i, j)] is None:
                hist_data[(i, j)] = None
                continue

            data = all_data[(i, j)]
            values = data.get(data_key, np.array([]))
            weights = data.get(weight_key, None) if weight_key else None

            if len(values) < 10:
                hist_data[(i, j)] = None
                continue

            # Compute histogram
            if weights is not None and len(weights) > 0:
                counts, _ = np.histogram(values, bins=bins, weights=weights, density=normalize)
            else:
                counts, _ = np.histogram(values, bins=bins, density=normalize)

            hist_data[(i, j)] = {'values': values, 'weights': weights, 'counts': counts}
            row_y_max[i] = max(row_y_max[i], counts.max())

    # Add 10% padding to y_max
    for i in row_y_max:
        row_y_max[i] *= 1.1

    # Second pass: plot with consistent y-axis per row
    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            ax = fig.add_subplot(gs[i, j])

            if hist_data.get((i, j)) is None:
                if all_data[(i, j)] is None:
                    ax.text(0.5, 0.5, 'Data not\navailable',
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=11, color='gray', style='italic')
                    ax.set_facecolor('#f5f5f5')
                else:
                    ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                           ha='center', va='center', fontsize=11)
                ax.set_title(f'$r_0$ = {r0}, c = {c}', fontsize=10, pad=5)
                ax.set_xlim(0, x_max)
                if row_y_max[i] > 0:
                    ax.set_ylim(0, row_y_max[i])
                continue

            hd = hist_data[(i, j)]
            values = hd['values']
            weights = hd['weights']

            if weights is not None and len(weights) > 0:
                ax.hist(values, bins=bins, weights=weights, color=color, alpha=0.7,
                       edgecolor='black', linewidth=0.5, density=normalize)
                w_mean = weighted_mean(values, weights)
                w_std = weighted_std(values, weights)
                stats_text = f'n = {len(values):,}\n$\\mu_w$ = {w_mean:.3f}\n$\\sigma_w$ = {w_std:.3f}'
            else:
                ax.hist(values, bins=bins, color=color, alpha=0.7,
                       edgecolor='black', linewidth=0.5, density=normalize)
                w_mean = np.mean(values)
                w_std = np.std(values)
                stats_text = f'n = {len(values):,}\n$\\mu$ = {w_mean:.3f}\n$\\sigma$ = {w_std:.3f}'

            ax.axvline(w_mean, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            ax.set_title(f'$r_0$ = {r0}, c = {c}', fontsize=10, pad=5)
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, row_y_max[i])
            ax.set_xlabel(xlabel, fontsize=10)
            if normalize:
                ylabel = 'Density'
            elif weights is not None:
                ylabel = 'Weighted Count'
            else:
                ylabel = 'Count'
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_file


def plot_h0_vs_h1_comparison(output_file, x_max_fixed=5.0):
    """Create H0 vs H1 lifetime comparison grid with consistent y-axis per row."""
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 3, left=0.1, right=0.95, top=0.90, bottom=0.08,
                          wspace=0.25, hspace=0.35)

    fig.suptitle('H0 vs H1 Lifetime Distributions\n(Blue=H0 components, Green=H1 loops)',
                 fontsize=14, fontweight='bold', y=0.97)

    for j, c in enumerate(C_VALUES):
        fig.text(0.23 + j * 0.285, 0.93, f'c = {c}', fontsize=12, fontweight='bold',
                ha='center', va='center')

    for i, r0 in enumerate(R0_VALUES):
        fig.text(0.04, 0.77 - i * 0.275, f'$r_0$ = {r0}', fontsize=12, fontweight='bold',
                ha='center', va='center', rotation=90)

    x_max = x_max_fixed
    bins = np.linspace(0, x_max, 51)

    # First pass: load data and compute density histograms to find y_max per row
    all_data = {}
    row_y_max = {i: 0 for i in range(len(R0_VALUES))}

    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            config_name = get_config_name(r0, c)
            if not has_recent_data(config_name):
                all_data[(i, j)] = None
                continue

            try:
                data = load_persistence_data(config_name)
                h0_life = data['h0_lifetime']
                h1_life = data['h1_lifetime']
                all_data[(i, j)] = {'h0': h0_life, 'h1': h1_life}

                # Compute density histograms to find max
                if len(h0_life) > 0:
                    counts, _ = np.histogram(h0_life, bins=bins, density=True)
                    row_y_max[i] = max(row_y_max[i], counts.max())
                if len(h1_life) > 0:
                    counts, _ = np.histogram(h1_life, bins=bins, density=True)
                    row_y_max[i] = max(row_y_max[i], counts.max())
            except:
                all_data[(i, j)] = None

    # Add 10% padding
    for i in row_y_max:
        row_y_max[i] *= 1.1

    # Second pass: plot with consistent y-axis
    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            ax = fig.add_subplot(gs[i, j])

            if all_data.get((i, j)) is None:
                ax.text(0.5, 0.5, 'Data not\navailable', transform=ax.transAxes,
                       ha='center', va='center', fontsize=11, color='gray', style='italic')
                ax.set_facecolor('#f5f5f5')
                ax.set_title(f'$r_0$ = {r0}, c = {c}', fontsize=10, pad=5)
                ax.set_xlim(0, x_max)
                if row_y_max[i] > 0:
                    ax.set_ylim(0, row_y_max[i])
                continue

            data = all_data[(i, j)]
            h0_life = data['h0']
            h1_life = data['h1']

            if len(h0_life) > 0:
                ax.hist(h0_life, bins=bins, color='blue', alpha=0.5,
                       label=f'H0 ($\\mu$={np.mean(h0_life):.2f})', density=True)
            if len(h1_life) > 0:
                ax.hist(h1_life, bins=bins, color='green', alpha=0.5,
                       label=f'H1 ($\\mu$={np.mean(h1_life):.2f})', density=True)

            ax.set_title(f'$r_0$ = {r0}, c = {c}', fontsize=10, pad=5)
            ax.set_xlabel('Lifetime', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, row_y_max[i])
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_file


def plot_weighted_comparison(data_key, weight_key, title, xlabel, output_file,
                             color='#1f77b4', x_max_fixed=5.0):
    """Create side-by-side unweighted vs weighted comparison for a single config."""
    fig, axes = plt.subplots(3, 6, figsize=(18, 12))
    fig.suptitle(f'{title}: Unweighted (left) vs Weighted (right)',
                 fontsize=14, fontweight='bold', y=0.98)

    x_max = x_max_fixed
    bins = np.linspace(0, x_max, 51)

    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            config_name = get_config_name(r0, c)
            ax_unw = axes[i, j*2]
            ax_w = axes[i, j*2 + 1]

            if not has_recent_data(config_name):
                for ax in [ax_unw, ax_w]:
                    ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                           ha='center', va='center', fontsize=10, color='gray')
                    ax.set_facecolor('#f5f5f5')
                continue

            try:
                data = load_persistence_data(config_name)
                values = data.get(data_key, np.array([]))
                weights = data.get(weight_key, None)

                if len(values) < 10:
                    continue

                # Unweighted
                ax_unw.hist(values, bins=bins, color=color, alpha=0.7, edgecolor='black', linewidth=0.3)
                ax_unw.axvline(np.mean(values), color='red', linestyle='--', linewidth=1.5)
                ax_unw.set_title(f'$r_0$={r0}, c={c}\n$\\mu$={np.mean(values):.3f}', fontsize=9)
                ax_unw.set_xlim(0, x_max)

                # Weighted
                if weights is not None and len(weights) > 0:
                    ax_w.hist(values, bins=bins, weights=weights, color=color, alpha=0.7,
                             edgecolor='black', linewidth=0.3)
                    w_mean = weighted_mean(values, weights)
                    ax_w.axvline(w_mean, color='red', linestyle='--', linewidth=1.5)
                    ax_w.set_title(f'$r_0$={r0}, c={c}\n$\\mu_w$={w_mean:.3f}', fontsize=9)
                ax_w.set_xlim(0, x_max)

            except:
                pass

    for ax in axes.flat:
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    import time

    def log(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}", flush=True)

    log("="*70)
    log("Generating Report Figures")
    log("="*70)

    # Create output directories
    for cutoff in [5, 10]:
        os.makedirs(os.path.join(OUTPUT_BASE_DIR, f'xmax_{cutoff}'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_BASE_DIR, f'xmax_{cutoff}_normalized'), exist_ok=True)

    total_start = time.time()

    # Define all plots to generate
    plots = [
        # (data_key, weight_key, title, xlabel, filename_base, color)
        ("h0_birth", None, "H0 Birth Distribution", "Birth Time", "h0_birth_unweighted", '#1f77b4'),
        ("h0_birth", "h0_num_points", "H0 Birth Distribution", "Birth Time", "h0_birth_weighted", '#1f77b4'),
        ("h0_death", None, "H0 Death Distribution", "Death Time", "h0_death_unweighted", '#1f77b4'),
        ("h0_death", "h0_num_points", "H0 Death Distribution", "Death Time", "h0_death_weighted", '#1f77b4'),
        ("h0_lifetime", None, "H0 Lifetime Distribution", "Lifetime", "h0_lifetime_unweighted", '#1f77b4'),
        ("h0_lifetime", "h0_num_points", "H0 Lifetime Distribution", "Lifetime", "h0_lifetime_weighted", '#1f77b4'),
        ("h1_birth", None, "H1 Birth Distribution", "Birth Time", "h1_birth_unweighted", '#2ca02c'),
        ("h1_birth", "h1_cycle_area", "H1 Birth Distribution", "Birth Time", "h1_birth_weighted", '#2ca02c'),
        ("h1_death", None, "H1 Death Distribution", "Death Time", "h1_death_unweighted", '#2ca02c'),
        ("h1_death", "h1_cycle_area", "H1 Death Distribution", "Death Time", "h1_death_weighted", '#2ca02c'),
        ("h1_lifetime", None, "H1 Lifetime Distribution", "Lifetime", "h1_lifetime_unweighted", '#2ca02c'),
        ("h1_lifetime", "h1_cycle_area", "H1 Lifetime Distribution", "Lifetime", "h1_lifetime_weighted", '#2ca02c'),
        ("h1_cycle_area", None, "H1 Cycle Area Distribution", "Cycle Area", "h1_cycle_area", '#9467bd'),
        ("h1_num_vertices", None, "H1 Number of Vertices", "Number of Vertices", "h1_num_vertices", '#8c564b'),
    ]

    # Generate for both x_max values and both normalized/unnormalized
    for x_max in [5, 10]:
        for normalize in [False, True]:
            norm_suffix = "_normalized" if normalize else ""
            norm_label = " (normalized)" if normalize else ""
            log(f"\n--- Generating plots with x_max = {x_max}{norm_label} ---")
            out_dir = os.path.join(OUTPUT_BASE_DIR, f'xmax_{x_max}{norm_suffix}')

            for i, (data_key, weight_key, title, xlabel, fname, color) in enumerate(plots):
                log(f"  [{i+1}/{len(plots)+1}] {fname}...")
                output_file = os.path.join(out_dir, f'{fname}.png')

                # For cycle_area and num_vertices, use data-driven x_max
                if data_key in ['h1_cycle_area', 'h1_num_vertices']:
                    plot_grid_histogram(data_key, weight_key, title, xlabel, output_file, color,
                                       x_max_fixed=None, normalize=normalize)
                else:
                    plot_grid_histogram(data_key, weight_key, title, xlabel, output_file, color,
                                       x_max_fixed=x_max, normalize=normalize)

            # H0 vs H1 comparison (always normalized/density)
            log(f"  [{len(plots)+1}/{len(plots)+1}] h0_vs_h1_comparison...")
            plot_h0_vs_h1_comparison(os.path.join(out_dir, 'h0_vs_h1_comparison.png'), x_max_fixed=x_max)

    total_elapsed = time.time() - total_start
    log("")
    log("="*70)
    log(f"Complete! Generated figures in {total_elapsed:.1f}s")
    log(f"Output directory: {OUTPUT_BASE_DIR}")
    log("="*70)
