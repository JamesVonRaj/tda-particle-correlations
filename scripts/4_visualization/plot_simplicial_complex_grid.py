#!/usr/bin/env python
"""
Plot Simplicial Complex at Different Filtration Values

Creates 3x3 grid visualizations showing the Vietoris-Rips complex
at various filtration values (epsilon) for each parameter configuration.

Each grid shows:
    - Rows: r_0 values (cluster standard deviation): 0.1, 0.5, 1.0
    - Columns: c values (mean cluster size): 5, 10, 50
    - Points and edges that exist at the given filtration value

Multiple grids are generated for different filtration values.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial.distance import pdist, squareform
import os
import json

# ==============================================================================
# Configuration
# ==============================================================================

ENSEMBLE_BASE_DIR = "../../data/grid_ensemble_data"
OUTPUT_DIR = "../../outputs/figures/simplicial_complex_grids"

# Grid parameters (must match data generation)
R0_VALUES = [0.1, 0.5, 1.0]
C_VALUES = [5, 10, 50]

# Filtration values to visualize
FILTRATION_VALUES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# Show full domain (no cropping)
SHOW_FULL_DOMAIN = True

# Plot styling
plt.rcParams['figure.dpi'] = 150

# ==============================================================================
# Helper Functions
# ==============================================================================

def get_config_name(r0, c):
    """Generate configuration name from parameters."""
    return f'r0_{r0:.1f}_c_{c}'


def load_sample_with_most_points(config_name, base_dir=ENSEMBLE_BASE_DIR):
    """
    Load the sample with the most points from a configuration.

    Returns:
        tuple: (points array, sample_id, n_points)
    """
    config_dir = os.path.join(base_dir, config_name)

    with open(os.path.join(config_dir, 'persistence_metadata.json'), 'r') as f:
        metadata = json.load(f)

    n_samples = metadata['n_samples']

    best_sample = None
    best_n_points = 0
    best_id = 0

    for i in range(n_samples):
        sample_file = os.path.join(config_dir, f'sample_{i:03d}.npy')
        if os.path.exists(sample_file):
            data = np.load(sample_file, allow_pickle=True).item()
            n_points = len(data['points'])
            if n_points > best_n_points:
                best_n_points = n_points
                best_sample = data['points']
                best_id = i

    return best_sample, best_id, best_n_points


def crop_to_window(points, window_size, center=None):
    """
    Crop points to a fixed-size square window.

    This maintains the true density (intensity=1) in the visualization,
    unlike subsampling which would artificially reduce density.

    Parameters:
    -----------
    points : np.ndarray
        (N, 2) array of point coordinates
    window_size : float
        Size of the square window to crop
    center : tuple or None
        (x, y) center of the window. If None, uses center of point cloud.

    Returns:
    --------
    tuple: (cropped_points, (x_min, x_max, y_min, y_max))
    """
    if center is None:
        # Use center of the point cloud
        center_x = (points[:, 0].min() + points[:, 0].max()) / 2
        center_y = (points[:, 1].min() + points[:, 1].max()) / 2
    else:
        center_x, center_y = center

    half_size = window_size / 2
    x_min, x_max = center_x - half_size, center_x + half_size
    y_min, y_max = center_y - half_size, center_y + half_size

    # Filter points within window
    mask = ((points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max))

    return points[mask], (x_min, x_max, y_min, y_max)


def get_edges_at_filtration(points, epsilon):
    """
    Get all edges (pairs of points) with distance <= epsilon.

    Returns:
        list of [(x1, y1), (x2, y2)] line segments
    """
    if len(points) < 2:
        return []

    dists = squareform(pdist(points))
    edges = []

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if dists[i, j] <= epsilon:
                edges.append([points[i], points[j]])

    return edges


def plot_simplicial_complex_grid(epsilon, base_dir=ENSEMBLE_BASE_DIR,
                                  output_dir=OUTPUT_DIR):
    """
    Create a 3x3 grid showing the simplicial complex at a given filtration value.

    Shows the full domain for each configuration.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 14))

    gs = fig.add_gridspec(3, 3, left=0.08, right=0.95, top=0.90, bottom=0.05,
                          wspace=0.20, hspace=0.25)

    fig.suptitle(f'Vietoris-Rips Complex at ε = {epsilon}\n'
                 f'(edges connect points within distance ε, full domain shown)',
                 fontsize=16, fontweight='bold', y=0.97)

    # Column headers
    for j, c in enumerate(C_VALUES):
        fig.text(0.22 + j * 0.29, 0.93, f'c = {c}', fontsize=14, fontweight='bold',
                ha='center', va='center')

    # Row headers
    for i, r0 in enumerate(R0_VALUES):
        fig.text(0.03, 0.77 - i * 0.29, f'r₀ = {r0}', fontsize=14, fontweight='bold',
                ha='center', va='center', rotation=90)

    for i, r0 in enumerate(R0_VALUES):
        for j, c in enumerate(C_VALUES):
            ax = fig.add_subplot(gs[i, j])
            config_name = get_config_name(r0, c)

            try:
                points, sample_id, n_points = load_sample_with_most_points(config_name, base_dir)

                # Use all points (full domain)
                plot_points = points
                n_plotted = len(plot_points)

                # Get edges at this filtration value
                edges = get_edges_at_filtration(plot_points, epsilon)

                # Plot edges first (so points are on top)
                if edges:
                    lc = LineCollection(edges, colors='steelblue', linewidths=0.3, alpha=0.4)
                    ax.add_collection(lc)

                # Plot points - smaller size for large point counts
                point_size = max(1, 8 - np.log10(n_plotted) * 2)
                ax.scatter(plot_points[:, 0], plot_points[:, 1], s=point_size, c='darkblue',
                          alpha=0.7, zorder=3, edgecolors='none')

                # Title with stats
                title = f'r₀={r0}, c={c}'
                title += f'\n({n_plotted} pts, {len(edges)} edges)'
                ax.set_title(title, fontsize=10)

                ax.set_aspect('equal')

                # Set axis limits to full domain
                x_min, x_max = plot_points[:, 0].min(), plot_points[:, 0].max()
                y_min, y_max = plot_points[:, 1].min(), plot_points[:, 1].max()
                pad = max(x_max - x_min, y_max - y_min) * 0.02
                ax.set_xlim(x_min - pad, x_max + pad)
                ax.set_ylim(y_min - pad, y_max + pad)

                ax.set_xticks([])
                ax.set_yticks([])

            except Exception as e:
                ax.text(0.5, 0.5, f'Error:\n{str(e)[:30]}', transform=ax.transAxes,
                       ha='center', va='center', fontsize=9)
                ax.set_title(f'r₀={r0}, c={c}', fontsize=10)

    # Save
    output_file = os.path.join(output_dir, f'simplicial_complex_eps_{epsilon:.2f}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()

    return output_file


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    import time
    from datetime import datetime

    def log(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}", flush=True)

    total_start = time.time()

    log("="*70)
    log("Creating Simplicial Complex Grid Visualizations")
    log("="*70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(ENSEMBLE_BASE_DIR):
        log(f"Error: Data directory '{ENSEMBLE_BASE_DIR}' not found.")
        exit(1)

    log(f"Data directory: {ENSEMBLE_BASE_DIR}")
    log(f"Output directory: {OUTPUT_DIR}")
    log(f"Filtration values: {FILTRATION_VALUES}")
    log(f"Mode: Full domain (no cropping)")
    log("")

    for i, epsilon in enumerate(FILTRATION_VALUES):
        log(f"[{i+1}/{len(FILTRATION_VALUES)}] Creating grid for ε = {epsilon}...")
        start = time.time()
        plot_simplicial_complex_grid(epsilon, ENSEMBLE_BASE_DIR, OUTPUT_DIR)
        elapsed = time.time() - start
        log(f"    Done in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    log("")
    log("="*70)
    log(f"Complete! Generated {len(FILTRATION_VALUES)} grid plots in {total_elapsed:.1f}s")
    log(f"Output directory: {OUTPUT_DIR}")
    log("="*70)
