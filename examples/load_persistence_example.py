"""
Example script demonstrating how to load and analyze persistent homology data
from the ensemble dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json

# ==============================================================================
# Helper Functions for Loading Persistence Data
# ==============================================================================

def load_sample_with_persistence(config_name, sample_id, base_dir='../data/ensemble_data'):
    """
    Load both the original point data and its persistent homology.
    
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
    dict : Dictionary containing:
        - 'points': numpy array of daughter points
        - 'parents': numpy array of parent points
        - 'sample_id': sample identifier
        - 'h0': H0 persistence diagram
        - 'h1': H1 persistence diagram
        - 'n_points': number of points
        - 'max_edge_length': max edge length used in computation
    """
    config_dir = os.path.join(base_dir, config_name)
    
    # Load original point data
    sample_file = os.path.join(config_dir, f'sample_{sample_id:03d}.npy')
    sample_data = np.load(sample_file, allow_pickle=True).item()
    
    # Load persistence data
    persistence_file = os.path.join(config_dir, f'sample_{sample_id:03d}_persistence.npy')
    persistence_data = np.load(persistence_file, allow_pickle=True).item()
    
    # Combine both
    return {
        'points': sample_data['points'],
        'parents': sample_data['parents'],
        'sample_id': sample_data['sample_id'],
        'h0': persistence_data['h0'],
        'h1': persistence_data['h1'],
        'n_points': persistence_data['n_points'],
        'max_edge_length': persistence_data['max_edge_length']
    }


def load_all_persistence_for_config(config_name, base_dir='../data/ensemble_data'):
    """
    Load all persistence diagrams for a given configuration.
    
    Parameters:
    -----------
    config_name : str
        Configuration name (e.g., 'stddev_0.2')
    base_dir : str
        Base directory containing the ensemble data
        
    Returns:
    --------
    list : List of persistence data dictionaries
    dict : Configuration metadata
    """
    config_dir = os.path.join(base_dir, config_name)
    
    # Load metadata
    with open(os.path.join(config_dir, 'persistence_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    n_samples = metadata['n_samples']
    
    # Load all persistence data
    all_persistence = []
    for i in range(n_samples):
        persistence_file = os.path.join(config_dir, f'sample_{i:03d}_persistence.npy')
        persistence_data = np.load(persistence_file, allow_pickle=True).item()
        all_persistence.append(persistence_data)
    
    return all_persistence, metadata


def plot_persistence_diagram(h0, h1, ax=None, title=None, max_scale=None, color_by_points=False):
    """
    Plot a persistence diagram.
    
    Parameters:
    -----------
    h0 : np.ndarray
        H0 features, shape (n, 3) with columns [birth, death, num_points]
    h1 : np.ndarray
        H1 features, shape (m, 3) with columns [birth, death, num_points]
    ax : matplotlib axis
        Axis to plot on (creates new if None)
    title : str
        Plot title
    max_scale : float
        Maximum scale for the plot
    color_by_points : bool
        If True, color points by their point count
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot diagonal line
    if max_scale is None:
        if len(h0) > 0 and len(h1) > 0:
            max_scale = max(np.max(h0[:, :2]), np.max(h1[:, :2]))
        elif len(h0) > 0:
            max_scale = np.max(h0[:, :2])
        elif len(h1) > 0:
            max_scale = np.max(h1[:, :2])
        else:
            max_scale = 1.0
    
    ax.plot([0, max_scale], [0, max_scale], 'k--', alpha=0.3, linewidth=1)
    
    # Plot H0 features
    if len(h0) > 0:
        if color_by_points and h0.shape[1] >= 3:
            scatter = ax.scatter(h0[:, 0], h0[:, 1], c=h0[:, 2], cmap='Blues', 
                               alpha=0.6, s=30, label='H0 (Components)', vmin=2)
        else:
            ax.scatter(h0[:, 0], h0[:, 1], c='blue', alpha=0.6, s=30, label='H0 (Components)')
    
    # Plot H1 features
    if len(h1) > 0:
        if color_by_points and h1.shape[1] >= 3:
            scatter = ax.scatter(h1[:, 0], h1[:, 1], c=h1[:, 2], cmap='Reds', 
                               alpha=0.6, s=30, marker='^', label='H1 (Loops)', vmin=3)
            if len(h1) > 1:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Number of Points', fontsize=10)
        else:
            ax.scatter(h1[:, 0], h1[:, 1], c='red', alpha=0.6, s=30, marker='^', label='H1 (Loops)')
    
    ax.set_xlabel('Birth', fontsize=12)
    ax.set_ylabel('Death', fontsize=12)
    ax.set_xlim([0, max_scale])
    ax.set_ylim([0, max_scale])
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title, fontsize=14)


def compute_persistence_lifetime(diagram):
    """
    Compute lifetimes (death - birth) for persistence features.
    
    Parameters:
    -----------
    diagram : np.ndarray
        Persistence diagram, shape (n, 2 or 3) with columns [birth, death] or [birth, death, num_points]
        
    Returns:
    --------
    np.ndarray : Array of lifetimes
    """
    if len(diagram) == 0:
        return np.array([])
    return diagram[:, 1] - diagram[:, 0]


def get_point_counts(diagram):
    """
    Extract point counts from persistence diagram.
    
    Parameters:
    -----------
    diagram : np.ndarray
        Persistence diagram, shape (n, 3) with columns [birth, death, num_points]
        
    Returns:
    --------
    np.ndarray : Array of point counts
    """
    if len(diagram) == 0:
        return np.array([])
    if diagram.shape[1] < 3:
        raise ValueError("Diagram does not contain point count information (expected 3 columns)")
    return diagram[:, 2]


# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("Example: Loading and Analyzing Persistent Homology Data")
    print("="*80)
    print()
    
    # -------------------------------------------------------------------------
    # Example 1: Load a single sample with persistence
    # -------------------------------------------------------------------------
    print("Example 1: Loading a single sample with persistence data")
    print("-" * 80)
    
    config_name = 'stddev_0.6'
    sample_id = 0
    
    data = load_sample_with_persistence(config_name, sample_id)
    
    print(f"Configuration: {config_name}")
    print(f"Sample ID: {sample_id}")
    print(f"Number of points: {data['n_points']}")
    print(f"Number of H0 features: {len(data['h0'])}")
    print(f"Number of H1 features: {len(data['h1'])}")
    print()
    
    # -------------------------------------------------------------------------
    # Example 2: Analyze persistence lifetimes and point counts
    # -------------------------------------------------------------------------
    print("Example 2: Analyzing persistence lifetimes and point counts")
    print("-" * 80)
    
    h0_lifetimes = compute_persistence_lifetime(data['h0'])
    h1_lifetimes = compute_persistence_lifetime(data['h1'])
    
    h0_point_counts = get_point_counts(data['h0'])
    h1_point_counts = get_point_counts(data['h1'])
    
    if len(h0_lifetimes) > 0:
        print(f"H0 lifetimes - Mean: {np.mean(h0_lifetimes):.4f}, "
              f"Std: {np.std(h0_lifetimes):.4f}, "
              f"Max: {np.max(h0_lifetimes):.4f}")
        print(f"H0 point counts - Mean: {np.mean(h0_point_counts):.1f}, "
              f"Std: {np.std(h0_point_counts):.1f}, "
              f"Max: {int(np.max(h0_point_counts))}")
    
    if len(h1_lifetimes) > 0:
        print(f"H1 lifetimes - Mean: {np.mean(h1_lifetimes):.4f}, "
              f"Std: {np.std(h1_lifetimes):.4f}, "
              f"Max: {np.max(h1_lifetimes):.4f}")
        print(f"H1 point counts - Mean: {np.mean(h1_point_counts):.1f}, "
              f"Std: {np.std(h1_point_counts):.1f}, "
              f"Max: {int(np.max(h1_point_counts))}")
    print()
    
    # -------------------------------------------------------------------------
    # Example 3: Load all persistence data for one configuration
    # -------------------------------------------------------------------------
    print("Example 3: Loading all persistence data for one configuration")
    print("-" * 80)
    
    all_persistence, metadata = load_all_persistence_for_config(config_name)
    
    print(f"Configuration: {config_name}")
    print(f"Number of samples: {len(all_persistence)}")
    print(f"Average H0 features: {metadata['h0_features']['mean']:.1f} ± {metadata['h0_features']['std']:.1f}")
    print(f"Average H1 features: {metadata['h1_features']['mean']:.1f} ± {metadata['h1_features']['std']:.1f}")
    print()
    
    # -------------------------------------------------------------------------
    # Example 4: Compare persistence across configurations
    # -------------------------------------------------------------------------
    print("Example 4: Comparing persistence across all configurations")
    print("-" * 80)
    
    with open('../data/ensemble_data/persistence_summary.json', 'r') as f:
        summary = json.load(f)
    
    configs = summary['configurations']
    
    print(f"{'Configuration':<15} {'H0 Features':<20} {'H1 Features':<20}")
    print("-" * 55)
    for config_name, config_data in configs.items():
        h0_str = f"{config_data['h0_features']['mean']:.1f} ± {config_data['h0_features']['std']:.1f}"
        h1_str = f"{config_data['h1_features']['mean']:.1f} ± {config_data['h1_features']['std']:.1f}"
        print(f"{config_name:<15} {h0_str:<20} {h1_str:<20}")
    print()
    
    # -------------------------------------------------------------------------
    # Example 5: Visualize persistence diagrams
    # -------------------------------------------------------------------------
    print("Example 5: Visualizing persistence diagrams")
    print("-" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    config_names = ['stddev_0.2', 'stddev_0.4', 'stddev_0.6', 'stddev_1.0']
    
    for idx, config_name in enumerate(config_names):
        # Load first sample
        data = load_sample_with_persistence(config_name, sample_id=0)
        
        # Load metadata for title
        with open(f'../data/ensemble_data/{config_name}/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Plot persistence diagram
        title = f"{metadata['label']}\n{len(data['h0'])} H0 features, {len(data['h1'])} H1 features"
        plot_persistence_diagram(data['h0'], data['h1'], ax=axes[idx], title=title, max_scale=1.5)
    
    fig.suptitle("Persistence Diagrams Across Configurations (Sample 0)", fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_file = '../outputs/figures/persistence_diagrams_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to '{output_file}'")
    print()
    
    # -------------------------------------------------------------------------
    # Example 6: Visualize points with persistence diagram side-by-side
    # -------------------------------------------------------------------------
    print("Example 6: Visualizing points with persistence diagram")
    print("-" * 80)
    
    config_name = 'stddev_0.6'
    data = load_sample_with_persistence(config_name, sample_id=5)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot point cloud
    if len(data['points']) > 0:
        ax1.scatter(data['points'][:, 0], data['points'][:, 1], s=2, c='blue', alpha=0.7)
    if len(data['parents']) > 0:
        ax1.scatter(data['parents'][:, 0], data['parents'][:, 1], s=40, c='red', marker='x')
    
    ax1.set_title(f"{config_name} - Sample {data['sample_id']}\n{data['n_points']} points", fontsize=12)
    ax1.set_xlim([0, 15])
    ax1.set_ylim([0, 15])
    ax1.set_aspect('equal', 'box')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Plot persistence diagram
    plot_persistence_diagram(data['h0'], data['h1'], ax=ax2, 
                            title=f"Persistence Diagram\n{len(data['h0'])} H0, {len(data['h1'])} H1 features",
                            max_scale=1.5)
    
    plt.tight_layout()
    
    output_file = '../outputs/figures/points_with_persistence.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to '{output_file}'")
    print()
    
    print("="*80)
    print("✓ Examples complete!")
    print("="*80)

