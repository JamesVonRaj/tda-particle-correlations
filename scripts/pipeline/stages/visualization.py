"""
Visualization stage for TDA pipeline.

This stage creates various plots and figures from persistence data.
"""

import logging
from pathlib import Path

from ..core.visualization import (
    plot_persistence_diagrams_grid,
    plot_distribution_grids,
    plot_parameter_comparison
)


def run(config: dict, output_dir: Path, logger: logging.Logger):
    """
    Run the visualization stage.

    Parameters
    ----------
    config : dict
        Full pipeline configuration
    output_dir : Path
        Output directory for this pipeline run
    logger : logging.Logger
        Logger for output
    """
    viz_config = config['visualization']
    dg = config['data_generation']
    mode = dg['mode']

    data_dir = output_dir / 'data'
    figures_dir = output_dir / 'figures'

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    dpi = viz_config.get('dpi', 200)
    fmt = viz_config.get('format', 'png')

    logger.info(f"Creating visualizations (dpi={dpi}, format={fmt})")

    # Persistence diagrams
    pd_config = viz_config.get('persistence_diagrams', {})
    if pd_config.get('enabled', True):
        logger.info("Creating persistence diagrams...")
        sample_id = pd_config.get('sample_id', 0)
        pd_dir = figures_dir / 'persistence_diagrams'
        plot_persistence_diagrams_grid(data_dir, pd_dir, mode, sample_id, dpi, fmt, logger)

    # Distribution grids
    dg_config = viz_config.get('distribution_grids', {})
    if dg_config.get('enabled', True):
        logger.info("Creating distribution grid plots...")
        dist_dir = figures_dir / 'distributions'
        plot_distribution_grids(data_dir, dist_dir, mode, dpi, fmt, logger)

    # Parameter comparisons (sweep mode only)
    pc_config = viz_config.get('parameter_comparisons', {})
    if pc_config.get('enabled', True) and mode == 'sweep':
        logger.info("Creating parameter comparison plots...")
        comp_dir = figures_dir / 'comparisons'
        plot_parameter_comparison(data_dir, comp_dir, mode, dpi, fmt, logger)

    logger.info(f"Figures saved to {figures_dir}")
