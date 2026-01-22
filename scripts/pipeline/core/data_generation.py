"""
Core data generation functions for TDA pipeline.

This module provides parameterized functions for generating Thomas cluster
process data in both sweep mode (varying r0) and grid mode (varying r0 and c).
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm


def simulate_poisson_cluster_process(
    lambda_parent: float,
    lambda_daughter: float,
    cluster_std_dev: float,
    bounds_x: tuple[float, float],
    bounds_y: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates a 2D Poisson Cluster Process (Thomas process).

    Parameters
    ----------
    lambda_parent : float
        Intensity (average density) of the parent Poisson process
    lambda_daughter : float
        Average number of daughter points per parent
    cluster_std_dev : float
        Standard deviation of the Gaussian distribution for daughter scatter
    bounds_x : tuple
        (min_x, max_x) for the observation window
    bounds_y : tuple
        (min_y, max_y) for the observation window

    Returns
    -------
    tuple
        (final_points, parent_locations) arrays
    """
    # Calculate area of the observation window
    area = (bounds_x[1] - bounds_x[0]) * (bounds_y[1] - bounds_y[0])

    # Number of parent points from a Poisson distribution
    n_parents = np.random.poisson(lambda_parent * area)

    if n_parents == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    # Locations of parent points, uniformly in the window
    parent_x = np.random.uniform(bounds_x[0], bounds_x[1], n_parents)
    parent_y = np.random.uniform(bounds_y[0], bounds_y[1], n_parents)
    parent_locations = np.stack([parent_x, parent_y], axis=1)

    # Simulate daughter process
    all_daughter_points = []

    for parent in parent_locations:
        n_daughters = np.random.poisson(lambda_daughter)

        if n_daughters == 0:
            continue

        # Generate offsets from isotropic Gaussian distribution
        offset_x = np.random.normal(loc=0.0, scale=cluster_std_dev, size=n_daughters)
        offset_y = np.random.normal(loc=0.0, scale=cluster_std_dev, size=n_daughters)

        daughter_x = parent[0] + offset_x
        daughter_y = parent[1] + offset_y

        daughters = np.stack([daughter_x, daughter_y], axis=1)
        all_daughter_points.append(daughters)

    if not all_daughter_points:
        return np.empty((0, 2)), parent_locations

    all_points_combined = np.vstack(all_daughter_points)

    # Filter points to be within the observation window
    within_bounds = (
        (all_points_combined[:, 0] >= bounds_x[0]) &
        (all_points_combined[:, 0] <= bounds_x[1]) &
        (all_points_combined[:, 1] >= bounds_y[0]) &
        (all_points_combined[:, 1] <= bounds_y[1])
    )

    final_points = all_points_combined[within_bounds]

    return final_points, parent_locations


def normalize_to_unit_intensity(
    points: np.ndarray,
    original_bounds: tuple[tuple[float, float], tuple[float, float]]
) -> tuple[np.ndarray, tuple[tuple[float, float], tuple[float, float]], float]:
    """
    Normalize point pattern to have unit intensity (1 point per unit area).

    Parameters
    ----------
    points : np.ndarray
        (N, 2) array of point coordinates
    original_bounds : tuple
        ((x_min, x_max), (y_min, y_max)) original bounds

    Returns
    -------
    tuple
        (normalized_points, normalized_bounds, scale_factor)
    """
    n_points = len(points)

    if n_points == 0:
        return points, ((0, 0), (0, 0)), 0.0

    x_min, x_max = original_bounds[0]
    y_min, y_max = original_bounds[1]
    original_size_x = x_max - x_min

    # New box size for unit intensity: L = sqrt(n_points)
    new_box_size = np.sqrt(n_points)
    scale_factor = new_box_size / original_size_x

    # Translate to origin and scale
    normalized_points = np.zeros_like(points)
    normalized_points[:, 0] = (points[:, 0] - x_min) * scale_factor
    normalized_points[:, 1] = (points[:, 1] - y_min) * scale_factor

    normalized_bounds = ((0, new_box_size), (0, new_box_size))

    return normalized_points, normalized_bounds, scale_factor


def generate_sweep_ensemble(
    output_dir: Path,
    r0_values: np.ndarray,
    n_samples: int,
    lambda_parent: float,
    lambda_daughter: float,
    window_size: float,
    logger=None
) -> dict:
    """
    Generate ensemble data for sweep mode (varying r0 only).

    Parameters
    ----------
    output_dir : Path
        Directory to save generated data
    r0_values : np.ndarray
        Array of r0 values to use
    n_samples : int
        Number of samples per configuration
    lambda_parent : float
        Parent process intensity
    lambda_daughter : float
        Mean cluster size
    window_size : float
        Size of square observation window
    logger : logging.Logger, optional
        Logger for output

    Returns
    -------
    dict
        Summary metadata
    """
    bounds_x = (0, window_size)
    bounds_y = (0, window_size)

    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    # Overall metadata
    overall_metadata = {
        'mode': 'sweep',
        'n_samples_per_config': n_samples,
        'n_configurations': len(r0_values),
        'total_samples': n_samples * len(r0_values),
        'window_size': window_size,
        'base_parameters': {
            'lambda_parent': lambda_parent,
            'lambda_daughter': lambda_daughter
        },
        'configurations': {}
    }

    for r0 in r0_values:
        config_name = f'stddev_{r0:.2f}'
        config_dir = data_dir / config_name
        config_dir.mkdir(exist_ok=True)

        if logger:
            logger.info(f"Generating {n_samples} samples for r0={r0:.2f}")

        # Configuration metadata
        config_metadata = {
            'configuration_name': config_name,
            'description': f'r0={r0:.2f} clusters',
            'label': f'r0={r0:.2f}',
            'n_samples': n_samples,
            'parameters': {
                'lambda_parent': lambda_parent,
                'lambda_daughter': lambda_daughter,
                'cluster_std_dev': r0,
                'window_size': window_size,
                'bounds_x': list(bounds_x),
                'bounds_y': list(bounds_y)
            }
        }

        n_points_list = []

        for i in tqdm(range(n_samples), desc=f"  {config_name}", leave=False):
            points, parents = simulate_poisson_cluster_process(
                lambda_parent=lambda_parent,
                lambda_daughter=lambda_daughter,
                cluster_std_dev=r0,
                bounds_x=bounds_x,
                bounds_y=bounds_y
            )

            sample_data = {
                'points': points,
                'parents': parents,
                'sample_id': i
            }

            sample_file = config_dir / f'sample_{i:03d}.npy'
            np.save(sample_file, sample_data, allow_pickle=True)

            n_points_list.append(len(points))

        # Add statistics to metadata
        config_metadata['statistics'] = {
            'n_points': {
                'mean': float(np.mean(n_points_list)),
                'std': float(np.std(n_points_list)),
                'min': int(np.min(n_points_list)),
                'max': int(np.max(n_points_list))
            }
        }

        # Save configuration metadata
        with open(config_dir / 'metadata.json', 'w') as f:
            json.dump(config_metadata, f, indent=4)

        overall_metadata['configurations'][config_name] = {
            'cluster_std_dev': r0,
            'description': config_metadata['description'],
            'label': config_metadata['label']
        }

    # Save overall metadata
    with open(data_dir / 'README.json', 'w') as f:
        json.dump(overall_metadata, f, indent=4)

    return overall_metadata


def compute_window_size_for_target(
    target_n_points: int,
    lambda_parent: float,
    c: float
) -> float:
    """
    Compute the initial window size needed to achieve a target number of points.

    The expected number of points is: E[n] = lambda_parent × window_size² × c
    So: window_size = sqrt(target_n_points / (lambda_parent × c))

    Parameters
    ----------
    target_n_points : int
        Target number of points
    lambda_parent : float
        Parent process intensity
    c : float
        Mean cluster size (lambda_daughter)

    Returns
    -------
    float
        Initial window size
    """
    return np.sqrt(target_n_points / (lambda_parent * c))


def generate_grid_ensemble(
    output_dir: Path,
    r0_values: list[float],
    c_values: list[int | float],
    n_samples: int,
    lambda_parent: float,
    initial_window_size: Optional[float] = None,
    target_n_points: Optional[int] = None,
    normalize: bool = True,
    logger=None
) -> dict:
    """
    Generate ensemble data for grid mode (varying r0 and c).

    Parameters
    ----------
    output_dir : Path
        Directory to save generated data
    r0_values : list
        List of r0 values
    c_values : list
        List of c values (mean cluster sizes)
    n_samples : int
        Number of samples per configuration
    lambda_parent : float
        Parent process intensity
    initial_window_size : float, optional
        Size of initial window before normalization (used if target_n_points not specified)
    target_n_points : int, optional
        Target number of points. If specified, initial_window_size is computed
        per configuration to achieve this target on average.
    normalize : bool
        Whether to normalize to unit intensity
    logger : logging.Logger, optional
        Logger for output

    Returns
    -------
    dict
        Summary metadata
    """
    # Determine if we're using target_n_points or fixed initial_window_size
    use_target_n_points = target_n_points is not None

    if not use_target_n_points and initial_window_size is None:
        raise ValueError("Must specify either initial_window_size or target_n_points")

    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    # Overall metadata
    overall_metadata = {
        'mode': 'grid',
        'description': f'{len(r0_values)}x{len(c_values)} grid: r0 x c parameter combinations',
        'normalization': {
            'method': 'unit_intensity' if normalize else 'none',
            'intensity': 1.0 if normalize else None,
            'description': 'Coordinates scaled so intensity = 1 point per unit area' if normalize else 'No normalization'
        },
        'grid_parameters': {
            'r0_values': r0_values,
            'c_values': c_values,
            'n_configurations': len(r0_values) * len(c_values)
        },
        'simulation_parameters': {
            'lambda_parent': lambda_parent,
            'target_n_points': target_n_points,
            'initial_window_size': initial_window_size,  # Will be None if using target_n_points
            'n_samples_per_config': n_samples
        },
        'configurations': []
    }

    for r0 in r0_values:
        for c in c_values:
            # Compute window size for this configuration
            if use_target_n_points:
                config_window_size = compute_window_size_for_target(target_n_points, lambda_parent, c)
            else:
                config_window_size = initial_window_size

            config_bounds_x = (0, config_window_size)
            config_bounds_y = (0, config_window_size)

            config_name = f'r0_{r0:.1f}_c_{c}'
            config_dir = data_dir / config_name
            config_dir.mkdir(exist_ok=True)

            if logger:
                if use_target_n_points:
                    logger.info(f"Generating {n_samples} samples for r0={r0}, c={c} (window_size={config_window_size:.2f} for target {target_n_points} points)")
                else:
                    logger.info(f"Generating {n_samples} samples for r0={r0}, c={c}")

            # Configuration metadata
            config_metadata = {
                'configuration_name': config_name,
                'description': f'r0={r0}, c={c}' + (' (normalized to unit intensity)' if normalize else ''),
                'parameters': {
                    'r0': r0,
                    'c': c,
                    'lambda_parent': lambda_parent,
                    'initial_window_size': config_window_size,
                    'target_n_points': target_n_points,
                },
                'normalization': {
                    'method': 'unit_intensity' if normalize else 'none',
                    'description': 'Coordinates scaled so intensity = 1 point per unit area' if normalize else 'No normalization'
                },
                'n_samples': n_samples
            }

            all_n_points = []
            all_normalized_box_sizes = []

            for i in tqdm(range(n_samples), desc=f"  r0={r0}, c={c}", leave=False):
                points, parents = simulate_poisson_cluster_process(
                    lambda_parent=lambda_parent,
                    lambda_daughter=c,
                    cluster_std_dev=r0,
                    bounds_x=config_bounds_x,
                    bounds_y=config_bounds_y
                )

                if normalize and len(points) > 0:
                    normalized_points, normalized_bounds, scale_factor = normalize_to_unit_intensity(
                        points, (config_bounds_x, config_bounds_y)
                    )

                    # Normalize parent locations with same scale factor
                    if len(parents) > 0:
                        normalized_parents = np.zeros_like(parents)
                        normalized_parents[:, 0] = (parents[:, 0] - config_bounds_x[0]) * scale_factor
                        normalized_parents[:, 1] = (parents[:, 1] - config_bounds_y[0]) * scale_factor
                    else:
                        normalized_parents = np.empty((0, 2))

                    sample_data = {
                        'points': normalized_points,
                        'parents': normalized_parents,
                        'original_points': points,
                        'original_parents': parents,
                        'n_points': len(points),
                        'normalized_box_size': normalized_bounds[0][1],
                        'scale_factor': scale_factor,
                        'sample_id': i
                    }
                    all_normalized_box_sizes.append(normalized_bounds[0][1])
                else:
                    sample_data = {
                        'points': points,
                        'parents': parents,
                        'n_points': len(points),
                        'sample_id': i
                    }

                all_n_points.append(len(points))

                sample_file = config_dir / f'sample_{i:03d}.npy'
                np.save(sample_file, sample_data, allow_pickle=True)

            # Add statistics to metadata
            config_metadata['statistics'] = {
                'n_points': {
                    'mean': float(np.mean(all_n_points)),
                    'std': float(np.std(all_n_points)),
                    'min': int(np.min(all_n_points)),
                    'max': int(np.max(all_n_points))
                }
            }
            if all_normalized_box_sizes:
                config_metadata['statistics']['normalized_box_size'] = {
                    'mean': float(np.mean(all_normalized_box_sizes)),
                    'std': float(np.std(all_normalized_box_sizes)),
                    'min': float(np.min(all_normalized_box_sizes)),
                    'max': float(np.max(all_normalized_box_sizes))
                }

            # Save configuration metadata
            with open(config_dir / 'metadata.json', 'w') as f:
                json.dump(config_metadata, f, indent=4)

            overall_metadata['configurations'].append({
                'r0': r0,
                'c': c,
                'directory': config_name
            })

    # Save overall metadata
    with open(data_dir / 'README.json', 'w') as f:
        json.dump(overall_metadata, f, indent=4)

    return overall_metadata
