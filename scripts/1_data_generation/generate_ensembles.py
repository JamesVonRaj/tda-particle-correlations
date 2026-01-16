import numpy as np
import sys
import os
import json
from tqdm import tqdm

# --- Import the function from your other file ---
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

# --- Simulation Parameters ---
WINDOW_SIZE = 15
BOUNDS_X = (0, WINDOW_SIZE)
BOUNDS_Y = (0, WINDOW_SIZE)

# --- Base parameters ---
BASE_PARAMS = {
    'lambda_parent': 0.1,
    'lambda_daughter': 30,
    'bounds_x': BOUNDS_X,
    'bounds_y': BOUNDS_Y
}

# --- Define 20 configurations - VARYING ONLY cluster_std_dev ---
# Generate 20 evenly spaced r_0 values between 0.1 and 1.0
N_CONFIGURATIONS = 20
R0_VALUES = np.linspace(0.1, 1.0, N_CONFIGURATIONS)

CONFIGURATIONS = {}
for r0 in R0_VALUES:
    config_name = f'stddev_{r0:.2f}'
    CONFIGURATIONS[config_name] = {
        'cluster_std_dev': r0,
        'description': f'r0={r0:.2f} clusters',
        'label': f'r0={r0:.2f}'
    }

# --- Output Directory ---
OUTPUT_BASE_DIR = "../../data/ensemble_data"

# ==============================================================================
# Main Execution
# ==============================================================================

def generate_ensemble(config_name, config_params, n_samples, output_dir):
    """
    Generate an ensemble of point configurations for a given parameter set.
    
    Parameters:
    -----------
    config_name : str
        Name of the configuration (used for folder naming)
    config_params : dict
        Dictionary containing the cluster_std_dev and metadata
    n_samples : int
        Number of samples to generate
    output_dir : str
        Base output directory
    """
    
    # Create configuration-specific directory
    config_dir = os.path.join(output_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)
    
    # Prepare full parameters
    params = BASE_PARAMS.copy()
    params['cluster_std_dev'] = config_params['cluster_std_dev']
    
    # Save metadata
    metadata = {
        'configuration_name': config_name,
        'description': config_params['description'],
        'label': config_params['label'],
        'n_samples': n_samples,
        'parameters': {
            'lambda_parent': params['lambda_parent'],
            'lambda_daughter': params['lambda_daughter'],
            'cluster_std_dev': params['cluster_std_dev'],
            'window_size': WINDOW_SIZE,
            'bounds_x': list(BOUNDS_X),
            'bounds_y': list(BOUNDS_Y)
        }
    }
    
    metadata_file = os.path.join(config_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"  Saved metadata to '{metadata_file}'")
    
    # Generate samples
    print(f"  Generating {n_samples} samples...")
    for i in tqdm(range(n_samples), desc=f"  {config_name}", leave=False):
        # Run simulation
        points, parents = simulate_poisson_cluster_process(**params)
        
        # Save points and parents separately
        sample_data = {
            'points': points,
            'parents': parents,
            'sample_id': i
        }
        
        sample_file = os.path.join(config_dir, f'sample_{i:03d}.npy')
        np.save(sample_file, sample_data, allow_pickle=True)
    
    print(f"  ✓ Generated {n_samples} samples for {config_name}")
    return config_dir


if __name__ == "__main__":
    
    print("="*80)
    print("Generating Ensemble Data for Cluster Standard Deviation Variations")
    print("="*80)
    print(f"\nNumber of samples per configuration: {N_SAMPLES}")
    print(f"Number of configurations: {len(CONFIGURATIONS)}")
    print(f"Total samples to generate: {N_SAMPLES * len(CONFIGURATIONS)}")
    print(f"Output directory: {OUTPUT_BASE_DIR}/")
    print()
    
    # Create base output directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # Save overall metadata
    overall_metadata = {
        'n_samples_per_config': N_SAMPLES,
        'n_configurations': len(CONFIGURATIONS),
        'total_samples': N_SAMPLES * len(CONFIGURATIONS),
        'window_size': WINDOW_SIZE,
        'base_parameters': {
            'lambda_parent': BASE_PARAMS['lambda_parent'],
            'lambda_daughter': BASE_PARAMS['lambda_daughter']
        },
        'configurations': {
            name: {
                'cluster_std_dev': conf['cluster_std_dev'],
                'description': conf['description'],
                'label': conf['label']
            }
            for name, conf in CONFIGURATIONS.items()
        }
    }
    
    overall_metadata_file = os.path.join(OUTPUT_BASE_DIR, 'README.json')
    with open(overall_metadata_file, 'w') as f:
        json.dump(overall_metadata, f, indent=4)
    print(f"Saved overall metadata to '{overall_metadata_file}'")
    print()
    
    # Generate ensembles for each configuration
    for config_name, config_params in CONFIGURATIONS.items():
        print(f"Processing configuration: {config_name}")
        print(f"  Description: {config_params['description']}")
        print(f"  cluster_std_dev = {config_params['cluster_std_dev']}")
        
        config_dir = generate_ensemble(
            config_name=config_name,
            config_params=config_params,
            n_samples=N_SAMPLES,
            output_dir=OUTPUT_BASE_DIR
        )
        print()
    
    print("="*80)
    print("✓ Ensemble generation complete!")
    print("="*80)
    print(f"\nData structure:")
    print(f"  {OUTPUT_BASE_DIR}/")
    print(f"  ├── README.json (overall metadata)")
    for config_name in CONFIGURATIONS.keys():
        print(f"  ├── {config_name}/")
        print(f"  │   ├── metadata.json (configuration-specific metadata)")
        print(f"  │   ├── sample_000.npy")
        print(f"  │   ├── sample_001.npy")
        print(f"  │   ├── ...")
        print(f"  │   └── sample_{N_SAMPLES-1:03d}.npy")
    print()
    print("Each .npy file contains a dictionary with keys:")
    print("  - 'points': numpy array of daughter points (N x 2)")
    print("  - 'parents': numpy array of parent points (M x 2)")
    print("  - 'sample_id': integer sample identifier")
    print()
    print("To load a sample:")
    print("  data = np.load('ensemble_data/stddev_0.2/sample_000.npy', allow_pickle=True).item()")
    print("  points = data['points']")
    print("  parents = data['parents']")
    print()

