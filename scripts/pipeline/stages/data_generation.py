"""
Data generation stage for TDA pipeline.

This stage generates Thomas cluster process data based on configuration,
supporting both sweep mode (varying r0) and grid mode (varying r0 and c).
"""

import logging
import numpy as np
from pathlib import Path

from ..core.data_generation import generate_sweep_ensemble, generate_grid_ensemble


def run(config: dict, output_dir: Path, logger: logging.Logger):
    """
    Run the data generation stage.

    Parameters
    ----------
    config : dict
        Full pipeline configuration
    output_dir : Path
        Output directory for this pipeline run
    logger : logging.Logger
        Logger for output
    """
    dg = config['data_generation']
    mode = dg['mode']

    logger.info(f"Data generation mode: {mode}")
    logger.info(f"Samples per configuration: {dg['n_samples_per_config']}")

    if mode == 'sweep':
        _run_sweep(config, output_dir, logger)
    elif mode == 'grid':
        _run_grid(config, output_dir, logger)
    else:
        raise ValueError(f"Unknown data generation mode: {mode}")


def _run_sweep(config: dict, output_dir: Path, logger: logging.Logger):
    """Run sweep mode data generation."""
    dg = config['data_generation']
    sweep = dg['sweep']

    # Generate r0 values
    r0_values = np.linspace(
        sweep['r0_min'],
        sweep['r0_max'],
        sweep['n_configurations']
    )

    logger.info(f"r0 range: [{sweep['r0_min']}, {sweep['r0_max']}]")
    logger.info(f"Number of configurations: {sweep['n_configurations']}")
    logger.info(f"lambda_parent: {dg['lambda_parent']}")
    logger.info(f"lambda_daughter: {sweep['lambda_daughter']}")
    logger.info(f"window_size: {dg['window_size']}")

    total_samples = dg['n_samples_per_config'] * sweep['n_configurations']
    logger.info(f"Total samples to generate: {total_samples}")

    metadata = generate_sweep_ensemble(
        output_dir=output_dir,
        r0_values=r0_values,
        n_samples=dg['n_samples_per_config'],
        lambda_parent=dg['lambda_parent'],
        lambda_daughter=sweep['lambda_daughter'],
        window_size=dg['window_size'],
        logger=logger
    )

    logger.info(f"Generated {metadata['total_samples']} samples across {metadata['n_configurations']} configurations")


def _run_grid(config: dict, output_dir: Path, logger: logging.Logger):
    """Run grid mode data generation."""
    dg = config['data_generation']
    grid = dg['grid']

    r0_values = grid['r0_values']
    c_values = grid['c_values']
    n_configs = len(r0_values) * len(c_values)

    logger.info(f"r0 values: {r0_values}")
    logger.info(f"c values: {c_values}")
    logger.info(f"Number of configurations: {n_configs}")
    logger.info(f"lambda_parent: {grid['lambda_parent']}")

    # Check if using target_n_points or initial_window_size
    target_n_points = grid.get('target_n_points')
    initial_window_size = grid.get('initial_window_size')

    if target_n_points:
        logger.info(f"target_n_points: {target_n_points} (window size computed per configuration)")
    elif initial_window_size:
        logger.info(f"initial_window_size: {initial_window_size}")

    logger.info(f"normalize_to_unit_intensity: {grid['normalize_to_unit_intensity']}")

    total_samples = dg['n_samples_per_config'] * n_configs
    logger.info(f"Total samples to generate: {total_samples}")

    metadata = generate_grid_ensemble(
        output_dir=output_dir,
        r0_values=r0_values,
        c_values=c_values,
        n_samples=dg['n_samples_per_config'],
        lambda_parent=grid['lambda_parent'],
        initial_window_size=initial_window_size,
        target_n_points=target_n_points,
        normalize=grid['normalize_to_unit_intensity'],
        logger=logger
    )

    logger.info(f"Generated {total_samples} samples across {n_configs} configurations")
