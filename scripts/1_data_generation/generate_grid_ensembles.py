#!/usr/bin/env python
"""
Generate Ensemble Data for 3x3 Grid: r0 × c Parameter Combinations

This script generates Poisson cluster process (Thomas process) data for a 
3×3 grid of parameter combinations:
    - r0 (cluster standard deviation): 0.1, 0.5, 1.0
    - c (mean cluster size / lambda_daughter): 5, 10, 50

IMPORTANT: Box size normalization for unit intensity
=======================================================
Following the convention in statistical mechanics for point patterns, we 
normalize such that the intensity (number of points per unit area) is unity.

For each realization:
    1. Generate points in an initial window
    2. Rescale coordinates by 1/sqrt(intensity) where intensity = n_points / area
    3. This gives us a normalized box where intensity = 1

This ensures consistent length scales across different parameter combinations.
"""

import numpy as np
import sys
import os
import json
from tqdm import tqdm

# --- Import the simulation function ---
try:
    from poisson_cluster import simulate_poisson_cluster_process
except ImportError:
    print("Error: Could not find 'poisson_cluster.py'.")
    print("Make sure it is in the same directory as this script.")
    sys.exit()

# ==============================================================================
# Configuration
# ==============================================================================

# --- Number of samples per configuration ---
N_SAMPLES = 50

# --- Initial Simulation Parameters ---
# We use a larger initial window to get enough points, then normalize
INITIAL_WINDOW_SIZE = 20
INITIAL_BOUNDS_X = (0, INITIAL_WINDOW_SIZE)
INITIAL_BOUNDS_Y = (0, INITIAL_WINDOW_SIZE)

# --- Parent process intensity ---
# This controls the number of clusters per unit area
LAMBDA_PARENT = 0.5  # Higher value to ensure sufficient clusters

# --- 3×3 Grid Parameters ---
R0_VALUES = [0.1, 0.5, 1.0]  # Cluster standard deviations
C_VALUES = [5, 10, 50]       # Mean cluster sizes (lambda_daughter)

# --- Output Directory ---
OUTPUT_BASE_DIR = "../../data/grid_ensemble_data"

# ==============================================================================
# Normalization Function
# ==============================================================================

def normalize_to_unit_intensity(points, original_bounds):
    """
    Normalize point pattern to have unit intensity (1 point per unit area).
    
    The normalization rescales coordinates so that:
        n_points / new_area = 1
    
    This is achieved by:
        new_box_size = sqrt(n_points)
        scale_factor = new_box_size / original_box_size
        normalized_points = (points - origin) * scale_factor
    
    Parameters:
    -----------
    points : np.ndarray
        (N, 2) array of point coordinates
    original_bounds : tuple
        ((x_min, x_max), (y_min, y_max)) original bounds
        
    Returns:
    --------
    tuple:
        - normalized_points : np.ndarray, (N, 2) array of normalized coordinates
        - normalized_bounds : tuple, ((0, L), (0, L)) where L = sqrt(N)
        - scale_factor : float, the factor used for scaling
    """
    n_points = len(points)
    
    if n_points == 0:
        return points, ((0, 0), (0, 0)), 0.0
    
    # Original dimensions
    x_min, x_max = original_bounds[0]
    y_min, y_max = original_bounds[1]
    original_size_x = x_max - x_min
    original_size_y = y_max - y_min
    
    # New box size for unit intensity: L = sqrt(n_points)
    new_box_size = np.sqrt(n_points)
    
    # Scale factors
    scale_x = new_box_size / original_size_x
    scale_y = new_box_size / original_size_y
    
    # For isotropic scaling, use the same factor (assuming square domain)
    scale_factor = new_box_size / original_size_x  # Assuming square
    
    # Translate to origin and scale
    normalized_points = np.zeros_like(points)
    normalized_points[:, 0] = (points[:, 0] - x_min) * scale_factor
    normalized_points[:, 1] = (points[:, 1] - y_min) * scale_factor
    
    normalized_bounds = ((0, new_box_size), (0, new_box_size))
    
    return normalized_points, normalized_bounds, scale_factor


# ==============================================================================
# Main Generation Function
# ==============================================================================

def generate_grid_ensemble(r0, c, n_samples, output_dir, lambda_parent=LAMBDA_PARENT):
    """
    Generate an ensemble of normalized point configurations for a given (r0, c) pair.
    
    Parameters:
    -----------
    r0 : float
        Cluster standard deviation
    c : int or float
        Mean cluster size (lambda_daughter)
    n_samples : int
        Number of samples to generate
    output_dir : str
        Base output directory
    lambda_parent : float
        Parent process intensity
        
    Returns:
    --------
    str : Path to the configuration directory
    """
    
    # Create configuration-specific directory
    config_name = f'r0_{r0:.1f}_c_{c}'
    config_dir = os.path.join(output_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)
    
    # Save metadata
    metadata = {
        'configuration_name': config_name,
        'description': f'r0={r0}, c={c} (normalized to unit intensity)',
        'parameters': {
            'r0': r0,
            'c': c,
            'lambda_parent': lambda_parent,
            'initial_window_size': INITIAL_WINDOW_SIZE,
        },
        'normalization': {
            'method': 'unit_intensity',
            'description': 'Coordinates scaled so intensity = 1 point per unit area'
        },
        'n_samples': n_samples
    }
    
    metadata_file = os.path.join(config_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Statistics tracking
    all_n_points = []
    all_normalized_box_sizes = []
    
    # Generate samples
    for i in tqdm(range(n_samples), desc=f"  r0={r0}, c={c}", leave=False):
        # Run simulation
        points, parents = simulate_poisson_cluster_process(
            lambda_parent=lambda_parent,
            lambda_daughter=c,
            cluster_std_dev=r0,
            bounds_x=INITIAL_BOUNDS_X,
            bounds_y=INITIAL_BOUNDS_Y
        )
        
        # Normalize to unit intensity
        normalized_points, normalized_bounds, scale_factor = normalize_to_unit_intensity(
            points, (INITIAL_BOUNDS_X, INITIAL_BOUNDS_Y)
        )
        
        # Also normalize parent locations (for visualization if needed)
        if len(parents) > 0:
            normalized_parents, _, _ = normalize_to_unit_intensity(
                parents, (INITIAL_BOUNDS_X, INITIAL_BOUNDS_Y)
            )
            # Actually, parents should use the same scale_factor as points
            normalized_parents = np.zeros_like(parents)
            normalized_parents[:, 0] = (parents[:, 0] - INITIAL_BOUNDS_X[0]) * scale_factor
            normalized_parents[:, 1] = (parents[:, 1] - INITIAL_BOUNDS_Y[0]) * scale_factor
        else:
            normalized_parents = np.empty((0, 2))
        
        # Track statistics
        all_n_points.append(len(points))
        all_normalized_box_sizes.append(normalized_bounds[0][1])
        
        # Save sample data
        sample_data = {
            'points': normalized_points,              # Normalized coordinates
            'parents': normalized_parents,            # Normalized parent locations
            'original_points': points,                # Original coordinates (for reference)
            'original_parents': parents,              # Original parent locations
            'n_points': len(points),
            'normalized_box_size': normalized_bounds[0][1],
            'scale_factor': scale_factor,
            'sample_id': i
        }
        
        sample_file = os.path.join(config_dir, f'sample_{i:03d}.npy')
        np.save(sample_file, sample_data, allow_pickle=True)
    
    # Update metadata with statistics
    metadata['statistics'] = {
        'n_points': {
            'mean': float(np.mean(all_n_points)),
            'std': float(np.std(all_n_points)),
            'min': int(np.min(all_n_points)),
            'max': int(np.max(all_n_points))
        },
        'normalized_box_size': {
            'mean': float(np.mean(all_normalized_box_sizes)),
            'std': float(np.std(all_normalized_box_sizes)),
            'min': float(np.min(all_normalized_box_sizes)),
            'max': float(np.max(all_normalized_box_sizes))
        }
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"  ✓ Generated {n_samples} samples for r0={r0}, c={c}")
    print(f"    Mean n_points: {np.mean(all_n_points):.1f} ± {np.std(all_n_points):.1f}")
    print(f"    Mean box size (normalized): {np.mean(all_normalized_box_sizes):.2f}")
    
    return config_dir


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("Generating 3×3 Grid Ensemble Data with Unit Intensity Normalization")
    print("="*80)
    print()
    print(f"Grid parameters:")
    print(f"  r0 values (cluster std dev): {R0_VALUES}")
    print(f"  c values (mean cluster size): {C_VALUES}")
    print(f"  Total configurations: {len(R0_VALUES) * len(C_VALUES)}")
    print(f"  Samples per configuration: {N_SAMPLES}")
    print(f"  Total samples: {len(R0_VALUES) * len(C_VALUES) * N_SAMPLES}")
    print()
    print(f"Normalization: Intensity = 1 (one point per unit area)")
    print(f"Output directory: {OUTPUT_BASE_DIR}/")
    print()
    
    # Create base output directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # Save overall metadata
    overall_metadata = {
        'description': '3×3 Grid Ensemble: r0 × c parameter combinations',
        'normalization': {
            'method': 'unit_intensity',
            'intensity': 1.0,
            'description': 'All point patterns normalized to have intensity = 1 point per unit area'
        },
        'grid_parameters': {
            'r0_values': R0_VALUES,
            'c_values': C_VALUES,
            'n_configurations': len(R0_VALUES) * len(C_VALUES)
        },
        'simulation_parameters': {
            'lambda_parent': LAMBDA_PARENT,
            'initial_window_size': INITIAL_WINDOW_SIZE,
            'n_samples_per_config': N_SAMPLES
        },
        'configurations': []
    }
    
    # Generate ensembles for each configuration
    for r0 in R0_VALUES:
        for c in C_VALUES:
            print(f"\nProcessing: r0={r0}, c={c}")
            config_dir = generate_grid_ensemble(
                r0=r0,
                c=c,
                n_samples=N_SAMPLES,
                output_dir=OUTPUT_BASE_DIR
            )
            overall_metadata['configurations'].append({
                'r0': r0,
                'c': c,
                'directory': f'r0_{r0:.1f}_c_{c}'
            })
    
    # Save overall metadata
    overall_metadata_file = os.path.join(OUTPUT_BASE_DIR, 'README.json')
    with open(overall_metadata_file, 'w') as f:
        json.dump(overall_metadata, f, indent=4)
    
    print()
    print("="*80)
    print("✓ Grid Ensemble Generation Complete!")
    print("="*80)
    print()
    print("Data structure:")
    print(f"  {OUTPUT_BASE_DIR}/")
    print(f"  ├── README.json")
    for r0 in R0_VALUES:
        for c in C_VALUES:
            print(f"  ├── r0_{r0:.1f}_c_{c}/")
            print(f"  │   ├── metadata.json")
            print(f"  │   ├── sample_000.npy ... sample_{N_SAMPLES-1:03d}.npy")
    print()
    print("Each sample contains:")
    print("  - 'points': normalized coordinates (intensity = 1)")
    print("  - 'parents': normalized parent locations")
    print("  - 'original_points': original coordinates")
    print("  - 'n_points': number of points")
    print("  - 'normalized_box_size': size of normalized domain")
    print("  - 'scale_factor': scaling applied to coordinates")
    print()

