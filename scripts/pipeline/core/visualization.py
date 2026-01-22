"""
Core visualization functions for TDA pipeline.

This module provides functions for creating various plots and figures
from persistence data.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def set_plot_style():
    """Set consistent plot styling."""
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.dpi'] = 150


def load_persistence_data(config_dir: Path, sample_id: int) -> dict:
    """Load persistence data for a specific sample."""
    persistence_file = config_dir / f'sample_{sample_id:03d}_persistence.npy'
    if not persistence_file.exists():
        raise FileNotFoundError(f"Persistence file not found: {persistence_file}")
    return np.load(persistence_file, allow_pickle=True).item()


def load_sample_data(config_dir: Path, sample_id: int) -> dict:
    """Load sample data (points, parents) for a specific sample."""
    sample_file = config_dir / f'sample_{sample_id:03d}.npy'
    if not sample_file.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_file}")
    return np.load(sample_file, allow_pickle=True).item()


def load_all_values(config_dir: Path, n_samples: int, h_dim: str, column: int) -> np.ndarray:
    """Load all values from a specific column across all samples."""
    all_values = []
    for i in range(n_samples):
        try:
            ph_data = load_persistence_data(config_dir, i)
            h_data = ph_data[h_dim]
            if len(h_data) > 0 and h_data.shape[1] > column:
                all_values.extend(h_data[:, column])
        except FileNotFoundError:
            continue
    return np.array(all_values)


def plot_persistence_diagram(
    h0: np.ndarray,
    h1: np.ndarray,
    title: str,
    output_file: Path,
    dpi: int = 200,
    max_val: Optional[float] = None
):
    """
    Create a persistence diagram.

    Parameters
    ----------
    h0 : np.ndarray
        H0 persistence data (birth, death, ...)
    h1 : np.ndarray
        H1 persistence data (birth, death, ...)
    title : str
        Plot title
    output_file : Path
        Output file path
    dpi : int
        DPI for saved figure
    max_val : float, optional
        Maximum value for axes
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    # Determine axis limits
    all_vals = []
    if len(h0) > 0:
        all_vals.extend(h0[:, 0])  # births
        all_vals.extend(h0[:, 1])  # deaths
    if len(h1) > 0:
        all_vals.extend(h1[:, 0])
        all_vals.extend(h1[:, 1])

    if max_val is None and all_vals:
        max_val = max(all_vals) * 1.1
    elif max_val is None:
        max_val = 1.0

    # Plot diagonal
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=1)

    # Plot H0 and H1 points
    if len(h0) > 0:
        ax.scatter(h0[:, 0], h0[:, 1], c='blue', alpha=0.6, s=30, label=f'H0 ({len(h0)})')
    if len(h1) > 0:
        ax.scatter(h1[:, 0], h1[:, 1], c='red', alpha=0.6, s=30, label=f'H1 ({len(h1)})')

    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_persistence_diagrams_grid(
    data_dir: Path,
    output_dir: Path,
    mode: str,
    sample_id: int = 0,
    dpi: int = 200,
    fmt: str = 'png',
    logger=None
):
    """
    Create persistence diagrams for all configurations.

    Parameters
    ----------
    data_dir : Path
        Directory containing persistence data
    output_dir : Path
        Directory to save figures
    mode : str
        'sweep' or 'grid' mode
    sample_id : int
        Which sample to plot
    dpi : int
        DPI for saved figures
    fmt : str
        Output format
    logger : logging.Logger, optional
        Logger for output
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == 'sweep':
        config_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('stddev_')])
    else:
        config_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('r0_')])

    for config_dir in config_dirs:
        config_name = config_dir.name
        try:
            ph_data = load_persistence_data(config_dir, sample_id)
            h0 = ph_data['h0']
            h1 = ph_data['h1']

            output_file = output_dir / f'{config_name}_persistence_diagram.{fmt}'
            plot_persistence_diagram(
                h0, h1,
                title=f'Persistence Diagram: {config_name}',
                output_file=output_file,
                dpi=dpi
            )

            if logger:
                logger.debug(f"Created persistence diagram for {config_name}")

        except FileNotFoundError as e:
            if logger:
                logger.warning(f"Could not create diagram for {config_name}: {e}")


def plot_distribution_histogram(
    values: np.ndarray,
    title: str,
    xlabel: str,
    output_file: Path,
    color: str = 'blue',
    n_bins: int = 50,
    dpi: int = 200
):
    """Create a histogram of values."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    if len(values) > 0:
        ax.hist(values, bins=n_bins, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

        mean_val = np.mean(values)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')

        stats_text = f'n = {len(values):,}\nmean = {mean_val:.3f}\nstd = {np.std(values):.3f}'
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center', fontsize=14)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_distribution_grids(
    data_dir: Path,
    output_dir: Path,
    mode: str,
    dpi: int = 200,
    fmt: str = 'png',
    logger=None
):
    """
    Create grid plots of distributions.

    For sweep mode: creates a single row or arranged grid.
    For grid mode: creates a proper r0 x c grid.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == 'sweep':
        _plot_sweep_distribution_grid(data_dir, output_dir, dpi, fmt, logger)
    else:
        _plot_grid_distribution_grid(data_dir, output_dir, dpi, fmt, logger)


def _plot_sweep_distribution_grid(
    data_dir: Path,
    output_dir: Path,
    dpi: int,
    fmt: str,
    logger=None
):
    """Create distribution grid for sweep mode."""
    config_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('stddev_')])

    if not config_dirs:
        if logger:
            logger.warning("No configuration directories found")
        return

    # For each homology dimension and data type
    for h_dim in ['h0', 'h1']:
        for data_type, column, label in [
            ('death', 1, 'Death Time'),
            ('birth', 0, 'Birth Time')
        ]:
            # Collect all data
            all_data = {}
            for config_dir in config_dirs:
                metadata_file = config_dir / 'persistence_metadata.json'
                if not metadata_file.exists():
                    continue

                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                n_samples = metadata['n_samples']

                values = load_all_values(config_dir, n_samples, h_dim, column)
                all_data[config_dir.name] = values

            if not all_data:
                continue

            # Create grid figure
            n_configs = len(all_data)
            n_cols = min(5, n_configs)
            n_rows = (n_configs + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [[axes]]
            elif n_rows == 1:
                axes = [axes]
            elif n_cols == 1:
                axes = [[ax] for ax in axes]

            color = '#1f77b4' if h_dim == 'h0' else '#2ca02c'

            for idx, (config_name, values) in enumerate(all_data.items()):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row][col]

                if len(values) > 0:
                    ax.hist(values, bins=30, color=color, alpha=0.7, edgecolor='black', linewidth=0.3)
                    ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=1.5)
                else:
                    ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center')

                ax.set_title(config_name.replace('stddev_', 'r0='), fontsize=9)
                ax.set_xlabel(label, fontsize=8)
                ax.tick_params(labelsize=7)

            # Hide empty subplots
            for idx in range(n_configs, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row][col].set_visible(False)

            h_label = 'H0' if h_dim == 'h0' else 'H1'
            fig.suptitle(f'{h_label} {label} Distributions', fontsize=12, fontweight='bold')
            plt.tight_layout()

            output_file = output_dir / f'{h_dim}_{data_type}_distribution_grid.{fmt}'
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
            plt.close()

            if logger:
                logger.debug(f"Created {output_file.name}")


def _plot_grid_distribution_grid(
    data_dir: Path,
    output_dir: Path,
    dpi: int,
    fmt: str,
    logger=None
):
    """Create distribution grid for grid mode (r0 x c)."""
    # Load README.json to get grid parameters
    readme_file = data_dir / 'README.json'
    if not readme_file.exists():
        if logger:
            logger.warning("README.json not found, cannot determine grid structure")
        return

    with open(readme_file, 'r') as f:
        readme = json.load(f)

    r0_values = readme['grid_parameters']['r0_values']
    c_values = readme['grid_parameters']['c_values']

    for h_dim in ['h0', 'h1']:
        for data_type, column, label in [
            ('death', 1, 'Death Time'),
            ('birth', 0, 'Birth Time')
        ]:
            fig, axes = plt.subplots(len(r0_values), len(c_values),
                                    figsize=(5 * len(c_values), 4 * len(r0_values)))

            if len(r0_values) == 1:
                axes = [axes]
            if len(c_values) == 1:
                axes = [[ax] for ax in axes]

            color = '#1f77b4' if h_dim == 'h0' else '#2ca02c'

            # Compute common axis limits per row
            row_max_x = {}
            row_max_y = {}

            for i, r0 in enumerate(r0_values):
                all_values_row = []
                for c in c_values:
                    config_name = f'r0_{r0:.1f}_c_{c}'
                    config_dir = data_dir / config_name

                    metadata_file = config_dir / 'persistence_metadata.json'
                    if not metadata_file.exists():
                        continue

                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    n_samples = metadata['n_samples']

                    values = load_all_values(config_dir, n_samples, h_dim, column)
                    all_values_row.extend(values)

                if all_values_row:
                    row_max_x[i] = np.percentile(all_values_row, 99)

            for i, r0 in enumerate(r0_values):
                for j, c in enumerate(c_values):
                    ax = axes[i][j]
                    config_name = f'r0_{r0:.1f}_c_{c}'
                    config_dir = data_dir / config_name

                    metadata_file = config_dir / 'persistence_metadata.json'
                    if not metadata_file.exists():
                        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center')
                        ax.set_title(f'r0={r0}, c={c}', fontsize=10)
                        continue

                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    n_samples = metadata['n_samples']

                    values = load_all_values(config_dir, n_samples, h_dim, column)

                    if len(values) > 0:
                        x_max = row_max_x.get(i, np.max(values))
                        x_max = max(x_max, 0.1)  # Ensure minimum x_max to avoid singular transformation
                        bins = np.linspace(0, x_max, 40)
                        ax.hist(values, bins=bins, color=color, alpha=0.7, edgecolor='black', linewidth=0.3)
                        ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=1.5)
                        ax.set_xlim(0, x_max)

                        stats_text = f'n={len(values):,}\nmean={np.mean(values):.2f}'
                        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                               fontsize=8, va='top', ha='right',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                    else:
                        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center')

                    ax.set_title(f'r0={r0}, c={c}', fontsize=10)
                    ax.set_xlabel(label, fontsize=9)
                    ax.grid(True, alpha=0.3, axis='y')

            # Add row/column labels
            for j, c in enumerate(c_values):
                fig.text(0.18 + j * 0.27, 0.96, f'c = {c}', fontsize=12, fontweight='bold', ha='center')

            for i, r0 in enumerate(r0_values):
                fig.text(0.02, 0.83 - i * 0.3, f'r0 = {r0}', fontsize=12, fontweight='bold',
                        ha='center', va='center', rotation=90)

            h_label = 'H0 (Connected Components)' if h_dim == 'h0' else 'H1 (Loops)'
            fig.suptitle(f'{h_label} {label} Distributions', fontsize=14, fontweight='bold', y=0.99)
            plt.tight_layout(rect=[0.03, 0, 1, 0.95])

            output_file = output_dir / f'{h_dim}_{data_type}_distribution_grid.{fmt}'
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
            plt.close()

            if logger:
                logger.debug(f"Created {output_file.name}")


def plot_parameter_comparison(
    data_dir: Path,
    output_dir: Path,
    mode: str,
    dpi: int = 200,
    fmt: str = 'png',
    logger=None
):
    """
    Create parameter comparison plots (death statistics vs r0).
    """
    if mode != 'sweep':
        if logger:
            logger.info("Parameter comparison plots only available for sweep mode")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    config_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('stddev_')])

    r0_values = []
    h0_mean_death = []
    h0_std_death = []
    h1_mean_death = []
    h1_std_death = []

    for config_dir in config_dirs:
        config_name = config_dir.name
        r0 = float(config_name.replace('stddev_', ''))
        r0_values.append(r0)

        metadata_file = config_dir / 'persistence_metadata.json'
        if not metadata_file.exists():
            h0_mean_death.append(np.nan)
            h0_std_death.append(np.nan)
            h1_mean_death.append(np.nan)
            h1_std_death.append(np.nan)
            continue

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        n_samples = metadata['n_samples']

        for h_dim, mean_list, std_list in [
            ('h0', h0_mean_death, h0_std_death),
            ('h1', h1_mean_death, h1_std_death)
        ]:
            values = load_all_values(config_dir, n_samples, h_dim, 1)  # death times
            if len(values) > 0:
                mean_list.append(np.mean(values))
                std_list.append(np.std(values))
            else:
                mean_list.append(np.nan)
                std_list.append(np.nan)

    # Plot
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, h_dim, means, stds, color in [
        (axes[0], 'H0', h0_mean_death, h0_std_death, 'blue'),
        (axes[1], 'H1', h1_mean_death, h1_std_death, 'green')
    ]:
        means = np.array(means)
        stds = np.array(stds)
        r0s = np.array(r0_values)

        mask = ~np.isnan(means)
        ax.errorbar(r0s[mask], means[mask], yerr=stds[mask], fmt='o-',
                   color=color, capsize=4, capthick=1.5, linewidth=2, markersize=8)

        ax.set_xlabel('r0 (Cluster Standard Deviation)', fontsize=12)
        ax.set_ylabel('Mean Death Time', fontsize=12)
        ax.set_title(f'{h_dim} Death Time vs r0', fontsize=13)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f'death_statistics_vs_r0.{fmt}'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()

    if logger:
        logger.debug(f"Created {output_file.name}")
