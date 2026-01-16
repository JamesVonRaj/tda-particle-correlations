"""
Plot Persistence Diagrams with Point-Count-Dependent Marker Sizes

This script creates persistence diagrams for one sample from each configuration,
where the marker size is proportional to the number of points involved in each
topological feature (using the 3rd column of the persistence data).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# Configuration
# ==============================================================================

# Which sample to use from each configuration
SAMPLE_ID = 0

# Configurations
CONFIG_NAMES = ['stddev_0.2', 'stddev_0.4', 'stddev_0.6', 'stddev_1.0']
CONFIG_LABELS = ['r₀ = 0.2 (Very Tight)', 'r₀ = 0.4 (Tight)', 
                 'r₀ = 0.6 (Baseline)', 'r₀ = 1.0 (Loose)']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Output directory
OUTPUT_DIR = '../../outputs/figures/persistence_diagrams'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# Plotting Functions
# ==============================================================================

def load_persistence_data(config_name, sample_id, base_dir='../../data/ensemble_data'):
    """Load persistence data for a specific sample."""
    persistence_file = os.path.join(base_dir, config_name, f'sample_{sample_id:03d}_persistence.npy')
    return np.load(persistence_file, allow_pickle=True).item()


def plot_persistence_diagram_with_sizes(ax, h0, h1, title, color, max_scale=None):
    """
    Plot a persistence diagram with marker sizes proportional to point counts.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    h0 : np.ndarray
        H0 features [birth, death, num_points]
    h1 : np.ndarray
        H1 features [birth, death, num_points]
    title : str
        Plot title
    color : str
        Base color for the plot
    max_scale : float
        Maximum scale for axes
    """
    
    # Determine scale
    if max_scale is None:
        if len(h0) > 0 and len(h1) > 0:
            max_scale = max(np.max(h0[:, :2]), np.max(h1[:, :2]))
        elif len(h0) > 0:
            max_scale = np.max(h0[:, :2])
        elif len(h1) > 0:
            max_scale = np.max(h1[:, :2])
        else:
            max_scale = 1.0
    
    # Limit to reasonable range for visualization
    max_scale = min(max_scale, 2.0)
    
    # Plot diagonal line
    ax.plot([0, max_scale], [0, max_scale], 'k--', alpha=0.3, linewidth=1.5, 
            label='Birth = Death', zorder=1)
    
    # Plot H0 features
    if len(h0) > 0:
        births_h0 = h0[:, 0]
        deaths_h0 = h0[:, 1]
        sizes_h0 = h0[:, 2]
        
        # Scale marker sizes for visibility (square root scaling for better perception)
        marker_sizes_h0 = 10 + 50 * np.sqrt(sizes_h0 - sizes_h0.min() + 1)
        
        scatter_h0 = ax.scatter(births_h0, deaths_h0, s=marker_sizes_h0, 
                               c='blue', alpha=0.6, edgecolors='darkblue', 
                               linewidth=0.5, label='H0 (Components)', zorder=2)
    
    # Plot H1 features
    if len(h1) > 0:
        births_h1 = h1[:, 0]
        deaths_h1 = h1[:, 1]
        sizes_h1 = h1[:, 2]
        
        # Scale marker sizes for visibility
        marker_sizes_h1 = 10 + 50 * np.sqrt(sizes_h1 - sizes_h1.min() + 1)
        
        scatter_h1 = ax.scatter(births_h1, deaths_h1, s=marker_sizes_h1, 
                               c='red', alpha=0.6, marker='^', edgecolors='darkred',
                               linewidth=0.5, label='H1 (Loops)', zorder=3)
    
    # Formatting
    ax.set_xlabel('Birth (Filtration Radius)', fontsize=11)
    ax.set_ylabel('Death (Filtration Radius)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlim([0, max_scale])
    ax.set_ylim([0, max_scale])
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add subtle background color
    ax.set_facecolor('#f8f8f8')
    
    # Add text indicating marker size meaning
    info_text = "Marker size ∝ # points"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("Creating Persistence Diagrams with Point-Count-Dependent Marker Sizes")
    print("="*80)
    print()
    print(f"Sample ID: {SAMPLE_ID}")
    print(f"Configurations: {len(CONFIG_NAMES)}")
    print(f"Output directory: {OUTPUT_DIR}/")
    print()
    
    # -------------------------------------------------------------------------
    # Plot 1: 2x2 Grid of Persistence Diagrams
    # -------------------------------------------------------------------------
    print("Creating 2×2 grid of persistence diagrams...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    max_scale_global = 1.5  # Common scale for all plots
    
    for idx, (config_name, label, color) in enumerate(zip(CONFIG_NAMES, CONFIG_LABELS, COLORS)):
        print(f"  Processing {config_name}...")
        
        # Load persistence data
        ph_data = load_persistence_data(config_name, SAMPLE_ID)
        h0 = ph_data['h0']
        h1 = ph_data['h1']
        n_points = ph_data['n_points']
        
        # Create title with statistics
        title = f"{label}\n{n_points} points, {len(h0)} H0 features, {len(h1)} H1 features"
        
        # Plot
        plot_persistence_diagram_with_sizes(axes[idx], h0, h1, title, color, max_scale_global)
        
        print(f"    H0: {len(h0)} features, point counts range: {h0[:, 2].min():.0f}-{h0[:, 2].max():.0f}")
        print(f"    H1: {len(h1)} features, point counts range: {h1[:, 2].min():.0f}-{h1[:, 2].max():.0f}")
    
    fig.suptitle(f'Persistence Diagrams with Point-Count-Dependent Marker Sizes\n(Sample {SAMPLE_ID})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    output_file = os.path.join(OUTPUT_DIR, f'persistence_diagrams_grid_sample_{SAMPLE_ID:03d}.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_file}")
    
    # -------------------------------------------------------------------------
    # Plot 2: Individual Larger Diagrams (one per configuration)
    # -------------------------------------------------------------------------
    print("\nCreating individual large persistence diagrams...")
    
    for config_name, label, color in zip(CONFIG_NAMES, CONFIG_LABELS, COLORS):
        print(f"  Creating diagram for {config_name}...")
        
        # Load persistence data
        ph_data = load_persistence_data(config_name, SAMPLE_ID)
        h0 = ph_data['h0']
        h1 = ph_data['h1']
        n_points = ph_data['n_points']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        title = f"{label}\nSample {SAMPLE_ID}: {n_points} points, {len(h0)} H0 features, {len(h1)} H1 features"
        
        # Plot
        plot_persistence_diagram_with_sizes(ax, h0, h1, title, color, max_scale_global)
        
        # Add size legend
        # Create dummy scatter plots for size legend
        legend_sizes = [2, 3, 5, 10]  # Example point counts
        legend_markers = [10 + 50 * np.sqrt(s - 1) for s in legend_sizes]
        
        for size, marker_size in zip(legend_sizes, legend_markers):
            ax.scatter([], [], s=marker_size, c='gray', alpha=0.6, 
                      label=f'{size} points', edgecolors='black', linewidth=0.5)
        
        # Add second legend for sizes
        leg2 = ax.legend(loc='upper left', fontsize=9, title='Point Count Scale',
                        framealpha=0.9, title_fontsize=9)
        ax.add_artist(leg2)
        
        # Add original legend back
        handles, labels = ax.get_legend_handles_labels()
        # Get the first 3 handles (diagonal, H0, H1)
        ax.legend(handles[:3], labels[:3], loc='lower right', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        
        output_file = os.path.join(OUTPUT_DIR, f'persistence_diagram_{config_name}_sample_{SAMPLE_ID:03d}.png')
        plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"    Saved: {output_file}")
    
    # -------------------------------------------------------------------------
    # Plot 3: Detailed Analysis - Birth vs Death with Point Count Color
    # -------------------------------------------------------------------------
    print("\nCreating birth-death scatter plots with point count as color...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, (config_name, label, color) in enumerate(zip(CONFIG_NAMES, CONFIG_LABELS, COLORS)):
        ax = axes[idx]
        
        # Load persistence data
        ph_data = load_persistence_data(config_name, SAMPLE_ID)
        h0 = ph_data['h0']
        h1 = ph_data['h1']
        
        # Plot diagonal
        ax.plot([0, max_scale_global], [0, max_scale_global], 'k--', alpha=0.3, linewidth=1.5)
        
        # Plot H0 with color representing point count
        if len(h0) > 0:
            scatter_h0 = ax.scatter(h0[:, 0], h0[:, 1], s=30, c=h0[:, 2], 
                                   cmap='Blues', alpha=0.7, edgecolors='black', linewidth=0.3,
                                   vmin=2, label='H0 (Components)')
            cbar_h0 = plt.colorbar(scatter_h0, ax=ax, pad=0.02, fraction=0.046)
            cbar_h0.set_label('H0 Point Count', fontsize=9)
        
        # Plot H1 with color representing point count
        if len(h1) > 0:
            # Offset slightly for visibility
            scatter_h1 = ax.scatter(h1[:, 0], h1[:, 1], s=40, c=h1[:, 2],
                                   cmap='Reds', alpha=0.7, marker='^', 
                                   edgecolors='black', linewidth=0.3,
                                   vmin=3, label='H1 (Loops)')
            # Second colorbar for H1
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            axins = inset_axes(ax, width="3%", height="40%", loc='upper left',
                             bbox_to_anchor=(0.05, 0.05, 1, 1), bbox_transform=ax.transAxes)
            cbar_h1 = plt.colorbar(scatter_h1, cax=axins)
            cbar_h1.set_label('H1 Point Count', fontsize=8)
            cbar_h1.ax.tick_params(labelsize=7)
        
        ax.set_xlabel('Birth (Filtration Radius)', fontsize=11)
        ax.set_ylabel('Death (Filtration Radius)', fontsize=11)
        ax.set_title(f"{label}\nColor indicates number of points in feature", 
                    fontsize=11, fontweight='bold')
        ax.set_xlim([0, max_scale_global])
        ax.set_ylim([0, max_scale_global])
        ax.set_aspect('equal')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f8f8')
    
    fig.suptitle(f'Persistence Diagrams - Point Count as Color\n(Sample {SAMPLE_ID})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    output_file = os.path.join(OUTPUT_DIR, f'persistence_diagrams_colored_sample_{SAMPLE_ID:03d}.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_file}")
    
    print()
    print("="*80)
    print("✓ All persistence diagrams created!")
    print("="*80)
    print()
    print(f"Output files in '{OUTPUT_DIR}/':")
    print(f"  - persistence_diagrams_grid_sample_{SAMPLE_ID:03d}.png (2×2 grid)")
    print(f"  - persistence_diagram_stddev_0.2_sample_{SAMPLE_ID:03d}.png (individual)")
    print(f"  - persistence_diagram_stddev_0.4_sample_{SAMPLE_ID:03d}.png (individual)")
    print(f"  - persistence_diagram_stddev_0.6_sample_{SAMPLE_ID:03d}.png (individual)")
    print(f"  - persistence_diagram_stddev_1.0_sample_{SAMPLE_ID:03d}.png (individual)")
    print(f"  - persistence_diagrams_colored_sample_{SAMPLE_ID:03d}.png (color-coded)")
    print()

