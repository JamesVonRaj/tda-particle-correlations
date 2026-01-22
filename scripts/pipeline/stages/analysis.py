"""
Analysis stage for TDA pipeline.

This stage performs statistical analysis on persistence data,
computing various distribution statistics and comparisons.
"""

import logging
from pathlib import Path

from ..core.analysis import (
    analyze_death_distributions,
    analyze_birth_distributions,
    analyze_lifetime_distributions,
    analyze_weighted_lifetime_distributions,
    analyze_death_statistics_vs_r0,
    analyze_filtered_persistence
)


def run(config: dict, output_dir: Path, logger: logging.Logger):
    """
    Run the analysis stage.

    Parameters
    ----------
    config : dict
        Full pipeline configuration
    output_dir : Path
        Output directory for this pipeline run
    logger : logging.Logger
        Logger for output
    """
    analysis_config = config['analysis']
    dg = config['data_generation']
    mode = dg['mode']

    data_dir = output_dir / 'data'
    analysis_dir = output_dir / 'analysis'

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    logger.info(f"Running analysis in {mode} mode")

    # Death distributions
    if analysis_config.get('death_distributions', True):
        logger.info("Analyzing death distributions...")
        analyze_death_distributions(data_dir, analysis_dir, mode, logger)

    # Birth distributions
    if analysis_config.get('birth_distributions', True):
        logger.info("Analyzing birth distributions...")
        analyze_birth_distributions(data_dir, analysis_dir, mode, logger)

    # Lifetime distributions
    if analysis_config.get('lifetime_distributions', True):
        logger.info("Analyzing lifetime distributions...")
        analyze_lifetime_distributions(data_dir, analysis_dir, mode, logger)

    # Weighted lifetime distributions
    if analysis_config.get('weighted_lifetime_distributions', True):
        logger.info("Analyzing weighted lifetime distributions...")
        analyze_weighted_lifetime_distributions(data_dir, analysis_dir, mode, logger)

    # Death statistics vs r0 (sweep mode only)
    if analysis_config.get('death_statistics_vs_r0', True) and mode == 'sweep':
        logger.info("Analyzing death statistics vs r0...")
        analyze_death_statistics_vs_r0(data_dir, analysis_dir, logger)

    # Filtered persistence
    filtered_config = analysis_config.get('filtered_persistence', {})
    if filtered_config.get('enabled', False):
        threshold = filtered_config.get('persistence_threshold', 0.1)
        logger.info(f"Analyzing filtered persistence (threshold={threshold})...")
        analyze_filtered_persistence(data_dir, analysis_dir, threshold, mode, logger)

    logger.info(f"Analysis results saved to {analysis_dir}")
