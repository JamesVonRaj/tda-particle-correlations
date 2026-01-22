"""
Persistence computation stage for TDA pipeline.

This stage computes persistent homology (H0 and H1) for all samples
generated in the data generation stage.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from ..core.persistence import compute_persistence_for_ensemble


def run(config: dict, output_dir: Path, logger: logging.Logger):
    """
    Run the persistence computation stage.

    Parameters
    ----------
    config : dict
        Full pipeline configuration
    output_dir : Path
        Output directory for this pipeline run
    logger : logging.Logger
        Logger for output
    """
    pers = config['persistence']
    dg = config['data_generation']
    mode = dg['mode']

    # Determine max_edge_length based on mode
    if mode == 'sweep':
        max_edge_length = pers.get('max_edge_length', 3.0)
    else:  # grid
        max_edge_length = pers.get('grid_max_edge_length', 10.0)

    logger.info(f"Computing persistence with max_edge_length = {max_edge_length}")

    data_dir = output_dir / 'data'
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_summaries = compute_persistence_for_ensemble(
        data_dir=data_dir,
        max_edge_length=max_edge_length,
        mode=mode,
        logger=logger
    )

    # Save overall summary
    overall_summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_configurations': len(all_summaries),
        'total_samples_processed': sum(s['n_samples'] for s in all_summaries.values()),
        'tda_parameters': {
            'max_edge_length': max_edge_length,
            'method': 'Union-Find for H0, GUDHI for H1'
        },
        'configurations': all_summaries
    }

    if mode == 'grid':
        overall_summary['normalization'] = 'unit_intensity (intensity = 1 point per unit area)'

    with open(data_dir / 'persistence_summary.json', 'w') as f:
        json.dump(overall_summary, f, indent=4)

    logger.info(f"Processed {overall_summary['total_samples_processed']} samples")
    logger.info(f"Saved persistence summary to {data_dir / 'persistence_summary.json'}")
