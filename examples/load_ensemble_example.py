"""
Example script demonstrating how to load and use the ensemble data.
This shows basic usage and can be used as a template for analysis scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json

# ==============================================================================
# Load and Display Ensemble Data
# ==============================================================================

def load_sample(config_name, sample_id, base_dir='../data/ensemble_data'):
    """
    Load a single sample from the ensemble data.
    
    Parameters:
    -----------
    config_name : str
        Configuration name (e.g., 'stddev_0.2')
    sample_id : int
        Sample ID number (0-49)
    base_dir : str
        Base directory containing the ensemble data
        
    Returns:
    --------
    data : dict
        Dictionary containing 'points', 'parents', and 'sample_id'
    """
    sample_file = os.path.join(base_dir, config_name, f'sample_{sample_id:03d}.npy')
    data = np.load(sample_file, allow_pickle=True).item()
    return data


def load_all_samples(config_name, base_dir='../data/ensemble_data'):
    """
    Load all samples for a given configuration.
    
    Parameters:
    -----------
    config_name : str
        Configuration name (e.g., 'stddev_0.2')
    base_dir : str
        Base directory containing the ensemble data
        
    Returns:
    --------
    samples : list
        List of dictionaries, each containing 'points', 'parents', and 'sample_id'
    """
    config_dir = os.path.join(base_dir, config_name)
    
    # Load metadata to know how many samples to load
    with open(os.path.join(config_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    n_samples = metadata['n_samples']
    
    samples = []
    for i in range(n_samples):
        data = load_sample(config_name, i, base_dir)
        samples.append(data)
    
    return samples, metadata


def get_configuration_names(base_dir='../data/ensemble_data'):
    """Get list of all configuration names."""
    with open(os.path.join(base_dir, 'README.json'), 'r') as f:
        readme = json.load(f)
    return list(readme['configurations'].keys())


# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("Example: Loading and Visualizing Ensemble Data")
    print("="*80)
    print()
    
    # -------------------------------------------------------------------------
    # Example 1: Load a single sample
    # -------------------------------------------------------------------------
    print("Example 1: Loading a single sample")
    print("-" * 40)
    
    config_name = 'stddev_0.2'
    sample_id = 0
    
    data = load_sample(config_name, sample_id)
    points = data['points']
    parents = data['parents']
    
    print(f"Configuration: {config_name}")
    print(f"Sample ID: {sample_id}")
    print(f"Number of daughter points: {len(points)}")
    print(f"Number of parent points: {len(parents)}")
    print()
    
    # -------------------------------------------------------------------------
    # Example 2: Load all samples for one configuration
    # -------------------------------------------------------------------------
    print("Example 2: Loading all samples for one configuration")
    print("-" * 40)
    
    config_name = 'stddev_0.6'
    samples, metadata = load_all_samples(config_name)
    
    print(f"Configuration: {config_name}")
    print(f"Description: {metadata['description']}")
    print(f"Number of samples: {len(samples)}")
    print(f"Parameters: cluster_std_dev = {metadata['parameters']['cluster_std_dev']}")
    print()
    
    # Calculate statistics across ensemble
    n_points_per_sample = [len(s['points']) for s in samples]
    print(f"Points per sample - Mean: {np.mean(n_points_per_sample):.1f}, "
          f"Std: {np.std(n_points_per_sample):.1f}")
    print()
    
    # -------------------------------------------------------------------------
    # Example 3: Load and compare all configurations
    # -------------------------------------------------------------------------
    print("Example 3: Summary of all configurations")
    print("-" * 40)
    
    config_names = get_configuration_names()
    
    for config_name in config_names:
        samples, metadata = load_all_samples(config_name)
        n_points_per_sample = [len(s['points']) for s in samples]
        
        print(f"{config_name:12} | "
              f"r0={metadata['parameters']['cluster_std_dev']:.1f} | "
              f"Mean points: {np.mean(n_points_per_sample):6.1f} ± {np.std(n_points_per_sample):5.1f}")
    print()
    
    # -------------------------------------------------------------------------
    # Example 4: Visualize one sample from each configuration
    # -------------------------------------------------------------------------
    print("Example 4: Visualizing one sample from each configuration")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, config_name in enumerate(config_names):
        # Load first sample
        data = load_sample(config_name, sample_id=0)
        points = data['points']
        parents = data['parents']
        
        # Load metadata for title
        with open(f'ensemble_data/{config_name}/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Plot
        ax = axes[idx]
        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], s=2, c='blue', alpha=0.7)
        if len(parents) > 0:
            ax.scatter(parents[:, 0], parents[:, 1], s=40, c='red', marker='x')
        
        ax.set_title(f"{metadata['label']}: {metadata['description']}\n"
                    f"Sample 0, n={len(points)} points", 
                    fontsize=10)
        ax.set_xlim(metadata['parameters']['bounds_x'])
        ax.set_ylim(metadata['parameters']['bounds_y'])
        ax.set_aspect('equal', 'box')
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.suptitle("Example Samples from Each Configuration", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_file = '../outputs/figures/ensemble_visualization_example.png'
    plt.savefig(output_file, dpi=150)
    print(f"Saved visualization to '{output_file}'")
    print()
    
    print("="*80)
    print("✓ Example complete!")
    print("="*80)

