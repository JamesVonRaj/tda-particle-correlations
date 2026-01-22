"""
Configuration loading and validation for TDA pipeline.

This module handles loading YAML configuration files and validating
that they contain all required fields with appropriate values.
"""

import yaml
from pathlib import Path
from typing import Any


# Default configuration values
DEFAULTS = {
    'experiment': {
        'name': 'experiment',
        'description': ''
    },
    'data_generation': {
        'enabled': True,
        'mode': 'sweep',
        'n_samples_per_config': 50,
        'window_size': 15,
        'lambda_parent': 0.1,
        'sweep': {
            'r0_min': 0.1,
            'r0_max': 1.0,
            'n_configurations': 20,
            'lambda_daughter': 30
        },
        'grid': {
            'r0_values': [0.1, 0.5, 1.0],
            'c_values': [5, 10, 50],
            'lambda_parent': 0.5,
            'initial_window_size': None,      # Computed from target_n_points if not specified
            'target_n_points': 1000,          # Target number of points (used to compute initial_window_size)
            'normalize_to_unit_intensity': True
        }
    },
    'persistence': {
        'enabled': True,
        'max_edge_length': 3.0,
        'grid_max_edge_length': 10.0,
        'config_overrides': {}
    },
    'analysis': {
        'enabled': True,
        'death_distributions': True,
        'birth_distributions': True,
        'lifetime_distributions': True,
        'weighted_lifetime_distributions': True,
        'death_statistics_vs_r0': True,
        'filtered_persistence': {
            'enabled': False,
            'persistence_threshold': 0.1
        }
    },
    'visualization': {
        'enabled': True,
        'dpi': 200,
        'format': 'png',
        'persistence_diagrams': {
            'enabled': True,
            'sample_id': 0
        },
        'distribution_grids': {
            'enabled': True
        },
        'simplicial_complex_grids': {
            'enabled': False,
            'filtration_values': [0.5, 1.0, 2.0]
        },
        'parameter_comparisons': {
            'enabled': True
        }
    },
    'pipeline': {
        'random_seed': None,
        'continue_on_error': False,
        'log_level': 'INFO'
    }
}


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries, with override taking precedence.

    Parameters
    ----------
    base : dict
        Base dictionary with default values
    override : dict
        Override dictionary with user-specified values

    Returns
    -------
    dict
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_config(path: str | Path) -> dict:
    """
    Load YAML configuration file and apply defaults.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file

    Returns
    -------
    dict
        Complete configuration with defaults applied

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist
    yaml.YAMLError
        If the configuration file is not valid YAML
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, 'r') as f:
        user_config = yaml.safe_load(f) or {}

    # Merge with defaults
    config = deep_merge(DEFAULTS, user_config)

    # Store the original config path for reference
    config['_config_path'] = str(path.absolute())

    return config


def validate_config(config: dict) -> list[str]:
    """
    Validate configuration and return list of errors (empty if valid).

    Parameters
    ----------
    config : dict
        Configuration dictionary to validate

    Returns
    -------
    list[str]
        List of validation error messages (empty if valid)
    """
    errors = []

    # Validate experiment section
    if 'experiment' not in config:
        errors.append("Missing required section: 'experiment'")
    else:
        if not config['experiment'].get('name'):
            errors.append("Missing required field: 'experiment.name'")

    # Validate data_generation section
    dg = config.get('data_generation', {})
    if dg.get('enabled', True):
        mode = dg.get('mode', 'sweep')
        if mode not in ('sweep', 'grid'):
            errors.append(f"Invalid data_generation.mode: '{mode}'. Must be 'sweep' or 'grid'")

        if mode == 'sweep':
            sweep = dg.get('sweep', {})
            if sweep.get('r0_min', 0.1) >= sweep.get('r0_max', 1.0):
                errors.append("data_generation.sweep.r0_min must be less than r0_max")
            if sweep.get('n_configurations', 20) < 1:
                errors.append("data_generation.sweep.n_configurations must be at least 1")

        elif mode == 'grid':
            grid = dg.get('grid', {})
            if not grid.get('r0_values'):
                errors.append("data_generation.grid.r0_values must not be empty")
            if not grid.get('c_values'):
                errors.append("data_generation.grid.c_values must not be empty")
            # Must have either initial_window_size or target_n_points
            has_window_size = grid.get('initial_window_size') is not None
            has_target_n = grid.get('target_n_points') is not None
            if not has_window_size and not has_target_n:
                errors.append("data_generation.grid must specify either initial_window_size or target_n_points")
            if has_target_n and grid.get('target_n_points', 0) <= 0:
                errors.append("data_generation.grid.target_n_points must be positive")

        if dg.get('n_samples_per_config', 50) < 1:
            errors.append("data_generation.n_samples_per_config must be at least 1")

    # Validate persistence section
    pers = config.get('persistence', {})
    if pers.get('enabled', True):
        if pers.get('max_edge_length', 3.0) <= 0:
            errors.append("persistence.max_edge_length must be positive")
        if pers.get('grid_max_edge_length', 10.0) <= 0:
            errors.append("persistence.grid_max_edge_length must be positive")

    # Validate visualization section
    viz = config.get('visualization', {})
    if viz.get('enabled', True):
        if viz.get('dpi', 200) < 50:
            errors.append("visualization.dpi must be at least 50")
        if viz.get('format', 'png') not in ('png', 'pdf', 'svg', 'jpg'):
            errors.append("visualization.format must be one of: png, pdf, svg, jpg")

    # Validate pipeline section
    pipeline = config.get('pipeline', {})
    log_level = pipeline.get('log_level', 'INFO')
    if log_level not in ('DEBUG', 'INFO', 'WARNING', 'ERROR'):
        errors.append(f"Invalid pipeline.log_level: '{log_level}'")

    return errors


def get_config_summary(config: dict) -> str:
    """
    Generate a human-readable summary of the configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    str
        Multi-line summary string
    """
    lines = []
    lines.append(f"Experiment: {config['experiment']['name']}")

    if config['experiment'].get('description'):
        lines.append(f"Description: {config['experiment']['description']}")

    dg = config['data_generation']
    if dg['enabled']:
        mode = dg['mode']
        lines.append(f"\nData Generation: {mode} mode")
        lines.append(f"  Samples per config: {dg['n_samples_per_config']}")

        if mode == 'sweep':
            sweep = dg['sweep']
            lines.append(f"  r0 range: [{sweep['r0_min']}, {sweep['r0_max']}]")
            lines.append(f"  Configurations: {sweep['n_configurations']}")
            lines.append(f"  lambda_daughter: {sweep['lambda_daughter']}")
        else:
            grid = dg['grid']
            lines.append(f"  r0 values: {grid['r0_values']}")
            lines.append(f"  c values: {grid['c_values']}")
            lines.append(f"  Configurations: {len(grid['r0_values']) * len(grid['c_values'])}")
            if grid.get('target_n_points'):
                lines.append(f"  Target n_points: {grid['target_n_points']}")
            elif grid.get('initial_window_size'):
                lines.append(f"  Initial window size: {grid['initial_window_size']}")
    else:
        lines.append("\nData Generation: disabled")

    pers = config['persistence']
    if pers['enabled']:
        lines.append("\nPersistence: enabled")
        if dg['mode'] == 'sweep':
            lines.append(f"  max_edge_length: {pers['max_edge_length']}")
        else:
            lines.append(f"  max_edge_length: {pers['grid_max_edge_length']}")
    else:
        lines.append("\nPersistence: disabled")

    analysis = config['analysis']
    if analysis['enabled']:
        enabled_analyses = [k for k, v in analysis.items()
                          if k not in ('enabled', 'filtered_persistence') and v]
        if analysis['filtered_persistence']['enabled']:
            enabled_analyses.append('filtered_persistence')
        lines.append(f"\nAnalysis: {len(enabled_analyses)} types enabled")
    else:
        lines.append("\nAnalysis: disabled")

    viz = config['visualization']
    if viz['enabled']:
        enabled_viz = [k for k, v in viz.items()
                      if isinstance(v, dict) and v.get('enabled', False)]
        lines.append(f"\nVisualization: {len(enabled_viz)} types enabled")
        lines.append(f"  Format: {viz['format']}, DPI: {viz['dpi']}")
    else:
        lines.append("\nVisualization: disabled")

    pipeline = config['pipeline']
    if pipeline.get('random_seed'):
        lines.append(f"\nRandom seed: {pipeline['random_seed']}")

    return '\n'.join(lines)
