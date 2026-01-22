"""
Core analysis functions for TDA pipeline.

This module provides functions for analyzing persistence data,
including distribution statistics and comparisons.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional
from scipy.stats import gaussian_kde, ks_2samp


def load_persistence_data(
    config_dir: Path,
    n_samples: int,
    homology_dim: str = 'h0'
) -> dict:
    """
    Load persistence data for a configuration.

    Parameters
    ----------
    config_dir : Path
        Configuration directory containing persistence files
    n_samples : int
        Number of samples to load
    homology_dim : str
        Which homology dimension to load ('h0' or 'h1')

    Returns
    -------
    dict
        Dictionary with 'birth_times', 'death_times', 'lifetimes', and other arrays
    """
    all_birth_times = []
    all_death_times = []
    all_num_points = []
    all_areas = []  # For H1

    for i in range(n_samples):
        persistence_file = config_dir / f'sample_{i:03d}_persistence.npy'
        if not persistence_file.exists():
            continue

        ph_data = np.load(persistence_file, allow_pickle=True).item()
        h_data = ph_data[homology_dim]

        if len(h_data) > 0:
            all_birth_times.extend(h_data[:, 0])
            all_death_times.extend(h_data[:, 1])

            if homology_dim == 'h0' and h_data.shape[1] >= 3:
                all_num_points.extend(h_data[:, 2])
            elif homology_dim == 'h1' and h_data.shape[1] >= 3:
                all_num_points.extend(h_data[:, 2])  # num_vertices
            if homology_dim == 'h1' and h_data.shape[1] >= 4:
                all_areas.extend(h_data[:, 3])  # cycle_area

    birth_times = np.array(all_birth_times)
    death_times = np.array(all_death_times)
    lifetimes = death_times - birth_times

    result = {
        'birth_times': birth_times,
        'death_times': death_times,
        'lifetimes': lifetimes,
        'num_points': np.array(all_num_points) if all_num_points else None
    }

    if all_areas:
        result['areas'] = np.array(all_areas)

    return result


def compute_statistics(values: np.ndarray) -> dict:
    """
    Compute summary statistics for an array of values.

    Parameters
    ----------
    values : np.ndarray
        Array of values

    Returns
    -------
    dict
        Dictionary of statistics
    """
    if len(values) == 0:
        return {
            'count': 0,
            'mean': None,
            'median': None,
            'std': None,
            'min': None,
            'max': None,
            'q25': None,
            'q75': None
        }

    return {
        'count': len(values),
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75))
    }


def analyze_death_distributions(
    data_dir: Path,
    output_dir: Path,
    mode: str = 'sweep',
    logger=None
) -> dict:
    """
    Analyze death time distributions across configurations.

    Parameters
    ----------
    data_dir : Path
        Directory containing persistence data
    output_dir : Path
        Directory to save analysis results
    mode : str
        'sweep' or 'grid' mode
    logger : logging.Logger, optional
        Logger for output

    Returns
    -------
    dict
        Analysis results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find configuration directories
    if mode == 'sweep':
        config_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('stddev_')])
    else:
        config_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('r0_')])

    results = {'h0': {}, 'h1': {}}

    for config_dir in config_dirs:
        config_name = config_dir.name

        # Load metadata to get n_samples
        metadata_file = config_dir / 'persistence_metadata.json'
        if not metadata_file.exists():
            continue

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        n_samples = metadata['n_samples']

        for h_dim in ['h0', 'h1']:
            data = load_persistence_data(config_dir, n_samples, h_dim)
            stats = compute_statistics(data['death_times'])
            results[h_dim][config_name] = stats

            if logger:
                logger.debug(f"{config_name} {h_dim}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    # Save results
    output_file = output_dir / 'death_statistics.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if logger:
        logger.info(f"Saved death distribution statistics to {output_file}")

    return results


def analyze_birth_distributions(
    data_dir: Path,
    output_dir: Path,
    mode: str = 'sweep',
    logger=None
) -> dict:
    """
    Analyze birth time distributions across configurations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == 'sweep':
        config_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('stddev_')])
    else:
        config_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('r0_')])

    results = {'h0': {}, 'h1': {}}

    for config_dir in config_dirs:
        config_name = config_dir.name

        metadata_file = config_dir / 'persistence_metadata.json'
        if not metadata_file.exists():
            continue

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        n_samples = metadata['n_samples']

        for h_dim in ['h0', 'h1']:
            data = load_persistence_data(config_dir, n_samples, h_dim)
            stats = compute_statistics(data['birth_times'])
            results[h_dim][config_name] = stats

    output_file = output_dir / 'birth_statistics.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if logger:
        logger.info(f"Saved birth distribution statistics to {output_file}")

    return results


def analyze_lifetime_distributions(
    data_dir: Path,
    output_dir: Path,
    mode: str = 'sweep',
    logger=None
) -> dict:
    """
    Analyze lifetime (persistence) distributions across configurations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == 'sweep':
        config_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('stddev_')])
    else:
        config_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('r0_')])

    results = {'h0': {}, 'h1': {}}

    for config_dir in config_dirs:
        config_name = config_dir.name

        metadata_file = config_dir / 'persistence_metadata.json'
        if not metadata_file.exists():
            continue

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        n_samples = metadata['n_samples']

        for h_dim in ['h0', 'h1']:
            data = load_persistence_data(config_dir, n_samples, h_dim)
            stats = compute_statistics(data['lifetimes'])
            results[h_dim][config_name] = stats

    output_file = output_dir / 'lifetime_statistics.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if logger:
        logger.info(f"Saved lifetime distribution statistics to {output_file}")

    return results


def analyze_weighted_lifetime_distributions(
    data_dir: Path,
    output_dir: Path,
    mode: str = 'sweep',
    logger=None
) -> dict:
    """
    Analyze weighted lifetime distributions (lifetime * area for H1).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == 'sweep':
        config_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('stddev_')])
    else:
        config_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('r0_')])

    results = {'h1': {}}

    for config_dir in config_dirs:
        config_name = config_dir.name

        metadata_file = config_dir / 'persistence_metadata.json'
        if not metadata_file.exists():
            continue

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        n_samples = metadata['n_samples']

        data = load_persistence_data(config_dir, n_samples, 'h1')

        if data.get('areas') is not None and len(data['areas']) > 0:
            weighted_lifetimes = data['lifetimes'] * data['areas']
            stats = compute_statistics(weighted_lifetimes)
        else:
            stats = compute_statistics(np.array([]))

        results['h1'][config_name] = stats

    output_file = output_dir / 'weighted_lifetime_statistics.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if logger:
        logger.info(f"Saved weighted lifetime statistics to {output_file}")

    return results


def analyze_death_statistics_vs_r0(
    data_dir: Path,
    output_dir: Path,
    logger=None
) -> dict:
    """
    Analyze how death statistics vary with r0 (sweep mode only).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('stddev_')])

    results = {
        'r0_values': [],
        'h0': {'mean_death': [], 'std_death': [], 'median_death': []},
        'h1': {'mean_death': [], 'std_death': [], 'median_death': []}
    }

    for config_dir in config_dirs:
        config_name = config_dir.name

        # Extract r0 from config name
        r0 = float(config_name.replace('stddev_', ''))
        results['r0_values'].append(r0)

        metadata_file = config_dir / 'persistence_metadata.json'
        if not metadata_file.exists():
            for h_dim in ['h0', 'h1']:
                results[h_dim]['mean_death'].append(None)
                results[h_dim]['std_death'].append(None)
                results[h_dim]['median_death'].append(None)
            continue

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        n_samples = metadata['n_samples']

        for h_dim in ['h0', 'h1']:
            data = load_persistence_data(config_dir, n_samples, h_dim)
            stats = compute_statistics(data['death_times'])

            results[h_dim]['mean_death'].append(stats['mean'])
            results[h_dim]['std_death'].append(stats['std'])
            results[h_dim]['median_death'].append(stats['median'])

    output_file = output_dir / 'death_statistics_vs_r0.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if logger:
        logger.info(f"Saved death statistics vs r0 to {output_file}")

    return results


def analyze_filtered_persistence(
    data_dir: Path,
    output_dir: Path,
    persistence_threshold: float,
    mode: str = 'sweep',
    logger=None
) -> dict:
    """
    Analyze persistence data with a persistence threshold filter.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == 'sweep':
        config_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('stddev_')])
    else:
        config_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('r0_')])

    results = {'h0': {}, 'h1': {}, 'threshold': persistence_threshold}

    for config_dir in config_dirs:
        config_name = config_dir.name

        metadata_file = config_dir / 'persistence_metadata.json'
        if not metadata_file.exists():
            continue

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        n_samples = metadata['n_samples']

        for h_dim in ['h0', 'h1']:
            data = load_persistence_data(config_dir, n_samples, h_dim)

            # Filter by persistence threshold
            mask = data['lifetimes'] >= persistence_threshold
            filtered_lifetimes = data['lifetimes'][mask]
            filtered_death_times = data['death_times'][mask]

            results[h_dim][config_name] = {
                'total_features': len(data['lifetimes']),
                'filtered_features': len(filtered_lifetimes),
                'fraction_above_threshold': len(filtered_lifetimes) / max(len(data['lifetimes']), 1),
                'lifetime_stats': compute_statistics(filtered_lifetimes),
                'death_stats': compute_statistics(filtered_death_times)
            }

    output_file = output_dir / 'filtered_persistence_statistics.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if logger:
        logger.info(f"Saved filtered persistence statistics to {output_file}")

    return results
