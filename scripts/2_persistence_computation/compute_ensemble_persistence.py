#!/usr/bin/env python
"""
Compute Persistent Homology for All Ensemble Data

This script processes all samples in the ensemble_data directory and computes
their persistent homology (H0 and H1) using Vietoris-Rips filtration.
The persistence diagrams are saved alongside the original data in a clear,
organized structure.

This version uses the compute_rips_persistence_with_point_counts function
from tda_utils.py for correct H0 point counting.
"""

import numpy as np
import os
import json
import sys
from tqdm import tqdm
from datetime import datetime

# Import the TDA utility function
from tda_utils import compute_rips_persistence_with_point_counts

# ==============================================================================
# Configuration
# ==============================================================================

# --- Input/Output Directories ---
ENSEMBLE_BASE_DIR = "../../data/ensemble_data"

# --- TDA Parameters ---
# This is a critical parameter! It determines the maximum scale we look at.
# For a 15x15 window, a value of 3.0 means we look at structures up to radius 1.5
MAX_EDGE_LENGTH = 3.0  

# You can also define config-specific parameters if needed
CONFIG_SPECIFIC_PARAMS = {
    'stddev_0.2': {'max_edge_length': 3.0},
    'stddev_0.4': {'max_edge_length': 3.0},
    'stddev_0.6': {'max_edge_length': 3.0},
    'stddev_1.0': {'max_edge_length': 3.0},
}

# ==============================================================================
# Helper Functions
# ==============================================================================

def get_configuration_names(base_dir):
    """Get list of all configuration directories."""
    configs = []
    for item in sorted(os.listdir(base_dir)):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith('stddev_'):
            configs.append(item)
    return configs


def compute_persistence_for_sample(sample_file, max_edge_length, verbose=False):
    """
    Load a sample and compute its persistent homology with point counts.
    
    This function now uses the corrected Union-Find based H0 computation
    from tda_utils.py.
    
    Parameters:
    -----------
    sample_file : str
        Path to the sample .npy file
    max_edge_length : float
        Maximum edge length for Rips complex
    verbose : bool
        Whether to print detailed output
        
    Returns:
    --------
    dict : Dictionary containing H0 and H1 persistence data with point counts
    """
    # Load the sample
    data = np.load(sample_file, allow_pickle=True).item()
    points = data['points']
    
    if verbose:
        print(f"  Processing {len(points)} points...")
    
    # Use the corrected function from tda_utils.py
    # This correctly tracks H0 point counts using Union-Find
    persistence_dict = compute_rips_persistence_with_point_counts(
        points=points,
        max_edge_length=max_edge_length
    )
    
    return persistence_dict


def save_persistence_data(persistence_dict, output_file):
    """Save persistence data to .npy file."""
    np.save(output_file, persistence_dict, allow_pickle=True)


def process_configuration(config_name, base_dir, max_edge_length):
    """
    Process all samples for a given configuration.
    
    Parameters:
    -----------
    config_name : str
        Name of the configuration (e.g., 'stddev_0.2')
    base_dir : str
        Base directory containing ensemble data
    max_edge_length : float
        Maximum edge length for Rips complex
        
    Returns:
    --------
    dict : Summary statistics for the configuration
    """
    config_dir = os.path.join(base_dir, config_name)
    
    # Get all sample files (excluding persistence files)
    sample_files = sorted([
        f for f in os.listdir(config_dir) 
        if f.startswith('sample_') and f.endswith('.npy') and not f.endswith('_persistence.npy')
    ])
    
    print(f"\n{'='*80}")
    print(f"Processing Configuration: {config_name}")
    print(f"{'='*80}")
    print(f"  Number of samples: {len(sample_files)}")
    print(f"  Max edge length: {max_edge_length}")
    print()
    
    # Statistics tracking
    stats = {
        'n_samples': len(sample_files),
        'n_h0_features': [],
        'n_h1_features': [],
        'n_points': [],
        'max_edge_length': max_edge_length
    }
    
    # Process each sample
    for sample_file in tqdm(sample_files, desc=f"  Computing PH for {config_name}"):
        sample_path = os.path.join(config_dir, sample_file)
        
        # Compute persistence using the corrected tda_utils function
        persistence_dict = compute_persistence_for_sample(
            sample_path, 
            max_edge_length,
            verbose=False
        )
        
        # Track statistics
        stats['n_h0_features'].append(len(persistence_dict['h0']))
        stats['n_h1_features'].append(len(persistence_dict['h1']))
        stats['n_points'].append(persistence_dict['n_points'])
        
        # Save persistence data with clear naming
        # Convert sample_000.npy -> sample_000_persistence.npy
        base_name = sample_file.replace('.npy', '')
        output_file = os.path.join(config_dir, f"{base_name}_persistence.npy")
        save_persistence_data(persistence_dict, output_file)
    
    # Compute summary statistics
    summary = {
        'config_name': config_name,
        'n_samples': stats['n_samples'],
        'max_edge_length': max_edge_length,
        'n_points': {
            'mean': float(np.mean(stats['n_points'])),
            'std': float(np.std(stats['n_points'])),
            'min': int(np.min(stats['n_points'])),
            'max': int(np.max(stats['n_points']))
        },
        'h0_features': {
            'mean': float(np.mean(stats['n_h0_features'])),
            'std': float(np.std(stats['n_h0_features'])),
            'min': int(np.min(stats['n_h0_features'])),
            'max': int(np.max(stats['n_h0_features']))
        },
        'h1_features': {
            'mean': float(np.mean(stats['n_h1_features'])),
            'std': float(np.std(stats['n_h1_features'])),
            'min': int(np.min(stats['n_h1_features'])),
            'max': int(np.max(stats['n_h1_features']))
        }
    }
    
    # Save configuration-specific persistence metadata
    persistence_metadata_file = os.path.join(config_dir, 'persistence_metadata.json')
    with open(persistence_metadata_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n  ✓ Processed {stats['n_samples']} samples")
    print(f"  ✓ H0 features: {summary['h0_features']['mean']:.1f} ± {summary['h0_features']['std']:.1f}")
    print(f"  ✓ H1 features: {summary['h1_features']['mean']:.1f} ± {summary['h1_features']['std']:.1f}")
    print(f"  ✓ Saved persistence metadata to '{persistence_metadata_file}'")
    
    return summary


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("Computing Persistent Homology for All Ensemble Data")
    print("="*80)
    print("Using corrected H0 computation from tda_utils.py")
    print(f"\nBase directory: {ENSEMBLE_BASE_DIR}/")
    print(f"Default max_edge_length: {MAX_EDGE_LENGTH}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if ensemble data exists
    if not os.path.exists(ENSEMBLE_BASE_DIR):
        print(f"Error: Ensemble data directory '{ENSEMBLE_BASE_DIR}' not found.")
        print("Please run 'generate_ensembles.py' first to create the ensemble data.")
        sys.exit(1)
    
    # Get all configurations
    config_names = get_configuration_names(ENSEMBLE_BASE_DIR)
    
    if len(config_names) == 0:
        print(f"Error: No configuration directories found in '{ENSEMBLE_BASE_DIR}'.")
        sys.exit(1)
    
    print(f"Found {len(config_names)} configurations: {', '.join(config_names)}")
    print()
    
    # Process each configuration
    all_summaries = {}
    
    for config_name in config_names:
        # Get config-specific parameters or use default
        max_edge_length = CONFIG_SPECIFIC_PARAMS.get(
            config_name, {}
        ).get('max_edge_length', MAX_EDGE_LENGTH)
        
        summary = process_configuration(config_name, ENSEMBLE_BASE_DIR, max_edge_length)
        all_summaries[config_name] = summary
    
    # Save overall summary
    overall_summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_configurations': len(config_names),
        'total_samples_processed': sum(s['n_samples'] for s in all_summaries.values()),
        'tda_parameters': {
            'max_edge_length_default': MAX_EDGE_LENGTH,
            'config_specific': CONFIG_SPECIFIC_PARAMS
        },
        'method': 'Union-Find for H0, GUDHI for H1',
        'configurations': all_summaries
    }
    
    overall_summary_file = os.path.join(ENSEMBLE_BASE_DIR, 'persistence_summary.json')
    with open(overall_summary_file, 'w') as f:
        json.dump(overall_summary, f, indent=4)
    
    print(f"\n{'='*80}")
    print("✓ Persistent Homology Computation Complete!")
    print("="*80)
    print(f"\nTotal samples processed: {overall_summary['total_samples_processed']}")
    print(f"Saved overall summary to '{overall_summary_file}'")
    print()
    
    print("Data structure:")
    print(f"  {ENSEMBLE_BASE_DIR}/")
    print(f"  ├── persistence_summary.json (overall PH summary)")
    for config_name in config_names:
        print(f"  ├── {config_name}/")
        print(f"  │   ├── metadata.json (original config metadata)")
        print(f"  │   ├── persistence_metadata.json (PH statistics)")
        print(f"  │   ├── sample_000.npy (original points)")
        print(f"  │   ├── sample_000_persistence.npy (PH data)")
        print(f"  │   ├── sample_001.npy")
        print(f"  │   ├── sample_001_persistence.npy")
        print(f"  │   └── ...")
    print()
    
    print("To load persistence data:")
    print("  ph_data = np.load('ensemble_data/stddev_0.2/sample_000_persistence.npy', allow_pickle=True).item()")
    print("  h0 = ph_data['h0']  # Shape: (n_features, 3) - [birth, death, num_points]")
    print("  h1 = ph_data['h1']  # Shape: (m_features, 3) - [birth, death, num_points]")
    print("  n_points = ph_data['n_points']  # Total number of points in original sample")
    print()
    
    print("NOTE: H0 num_points now correctly represents the number of points")
    print("      in the dying cluster using Union-Find tracking!")
    print()
    
    print("Summary by configuration:")
    print("-" * 80)
    print(f"{'Config':<15} {'H0 Features':<20} {'H1 Features':<20} {'Points':<15}")
    print("-" * 80)
    for config_name, summary in all_summaries.items():
        h0_str = f"{summary['h0_features']['mean']:.1f} ± {summary['h0_features']['std']:.1f}"
        h1_str = f"{summary['h1_features']['mean']:.1f} ± {summary['h1_features']['std']:.1f}"
        pts_str = f"{summary['n_points']['mean']:.0f} ± {summary['n_points']['std']:.0f}"
        print(f"{config_name:<15} {h0_str:<20} {h1_str:<20} {pts_str:<15}")
    print("-" * 80)
    print()
