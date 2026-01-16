"""
Alternative Persistence Diagram Visualizations

This script provides multiple alternative visualization strategies to address
the problem that H0 features with birth ≈ 0 cluster on the y-axis in standard
persistence diagrams, making it hard to see marker sizes and distinguish features.

Visualization strategies:
1. Persistence (lifetime) vs Birth plot
2. Barcode diagrams
3. Split H0/H1 with optimized scales
4. Log-scale birth axis
5. Persistence vs Death plot
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection

# ==============================================================================
# Configuration
# ==============================================================================

SAMPLE_ID = 0

CONFIG_NAMES = ['stddev_0.2', 'stddev_0.4', 'stddev_0.6', 'stddev_1.0']
CONFIG_LABELS = ['r₀ = 0.2 (Very Tight)', 'r₀ = 0.4 (Tight)', 
                 'r₀ = 0.6 (Baseline)', 'r₀ = 1.0 (Loose)']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

OUTPUT_DIR = '../../outputs/figures/persistence_diagrams_alternative'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# Data Loading
# ==============================================================================

def load_persistence_data(config_name, sample_id, base_dir='../../data/ensemble_data'):
    """Load persistence data for a specific sample."""
    persistence_file = os.path.join(base_dir, config_name, f'sample_{sample_id:03d}_persistence.npy')
    return np.load(persistence_file, allow_pickle=True).item()

# ==============================================================================
# Visualization 1: Persistence vs Birth Plot
# ==============================================================================

def plot_persistence_vs_birth(ax, h0, h1, title, max_scale=None):
    """
    Plot persistence (death - birth) vs birth time.
    This spreads out H0 features that were clustered on the y-axis.
    """
    if max_scale is None:
        max_scale = 1.5
    
    # Compute persistence (lifetime) for each feature
    if len(h0) > 0:
        births_h0 = h0[:, 0]
        persistence_h0 = h0[:, 1] - h0[:, 0]
        sizes_h0 = h0[:, 2]
        
        # Scale marker sizes
        marker_sizes_h0 = 10 + 80 * np.sqrt(sizes_h0 - sizes_h0.min() + 1)
        
        ax.scatter(births_h0, persistence_h0, s=marker_sizes_h0, 
                  c='blue', alpha=0.6, edgecolors='darkblue', 
                  linewidth=0.5, label='H0 (Components)', zorder=2)
    
    if len(h1) > 0:
        births_h1 = h1[:, 0]
        persistence_h1 = h1[:, 1] - h1[:, 0]
        sizes_h1 = h1[:, 2]
        
        marker_sizes_h1 = 10 + 80 * np.sqrt(sizes_h1 - sizes_h1.min() + 1)
        
        ax.scatter(births_h1, persistence_h1, s=marker_sizes_h1, 
                  c='red', alpha=0.6, marker='^', edgecolors='darkred',
                  linewidth=0.5, label='H1 (Loops)', zorder=3)
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1.5, 
               label='Zero Persistence', zorder=1)
    
    ax.set_xlabel('Birth Time', fontsize=11)
    ax.set_ylabel('Persistence (Death - Birth)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlim([0, max_scale])
    ax.set_ylim([0, max_scale])
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f8f8')
    
    # Add info text
    info_text = "Marker size ∝ # points\nHigher = longer-lived feature"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))

# ==============================================================================
# Visualization 2: Barcode Diagrams
# ==============================================================================

def plot_barcode(ax, h0, h1, title, max_scale=None):
    """
    Plot barcode diagram where each feature is a horizontal bar from birth to death.
    Features are colored by the number of points involved.
    """
    if max_scale is None:
        max_scale = 1.5
    
    current_y = 0
    
    # Plot H0 features
    if len(h0) > 0:
        # Sort by persistence (longest first)
        persistence_h0 = h0[:, 1] - h0[:, 0]
        sorted_indices = np.argsort(persistence_h0)[::-1]
        h0_sorted = h0[sorted_indices]
        
        for i, feature in enumerate(h0_sorted):
            birth, death, n_points = feature
            # Colormap based on point count
            color_intensity = (n_points - h0[:, 2].min()) / (h0[:, 2].max() - h0[:, 2].min() + 1e-8)
            color = plt.cm.Blues(0.3 + 0.6 * color_intensity)
            
            ax.plot([birth, death], [current_y, current_y], 
                   color=color, linewidth=3, solid_capstyle='round', alpha=0.8)
            current_y += 1
        
        current_y += 2  # Gap between H0 and H1
    
    h0_max_y = current_y
    
    # Plot H1 features
    if len(h1) > 0:
        persistence_h1 = h1[:, 1] - h1[:, 0]
        sorted_indices = np.argsort(persistence_h1)[::-1]
        h1_sorted = h1[sorted_indices]
        
        for i, feature in enumerate(h1_sorted):
            birth, death, n_points = feature
            color_intensity = (n_points - h1[:, 2].min()) / (h1[:, 2].max() - h1[:, 2].min() + 1e-8)
            color = plt.cm.Reds(0.3 + 0.6 * color_intensity)
            
            ax.plot([birth, death], [current_y, current_y], 
                   color=color, linewidth=3, solid_capstyle='round', alpha=0.8)
            current_y += 1
    
    # Add labels
    if len(h0) > 0:
        ax.text(-0.02, h0_max_y / 2, 'H0', transform=ax.get_yaxis_transform(),
                fontsize=11, fontweight='bold', ha='right', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    if len(h1) > 0:
        ax.text(-0.02, (h0_max_y + current_y) / 2, 'H1', 
                transform=ax.get_yaxis_transform(),
                fontsize=11, fontweight='bold', ha='right', va='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    ax.set_xlabel('Filtration Radius', fontsize=11)
    ax.set_ylabel('Feature Index (sorted by persistence)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlim([0, max_scale])
    ax.set_ylim([-1, current_y + 1])
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax.set_facecolor('#f8f8f8')
    
    # Remove y-tick labels (indices not meaningful)
    ax.set_yticks([])
    
    info_text = "Darker color = more points"
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))

# ==============================================================================
# Visualization 3: Split H0 and H1 with Optimized Scales
# ==============================================================================

def plot_split_h0_h1(ax_h0, ax_h1, h0, h1, title_base, max_scale=None):
    """
    Plot H0 and H1 in separate subplots with optimized scales for each.
    """
    if max_scale is None:
        max_scale = 1.5
    
    # ---- Plot H0 ----
    if len(h0) > 0:
        births_h0 = h0[:, 0]
        deaths_h0 = h0[:, 1]
        sizes_h0 = h0[:, 2]
        
        marker_sizes_h0 = 10 + 100 * np.sqrt(sizes_h0 - sizes_h0.min() + 1)
        
        # Plot diagonal
        ax_h0.plot([0, max_scale], [0, max_scale], 'k--', alpha=0.3, linewidth=1.5)
        
        scatter = ax_h0.scatter(births_h0, deaths_h0, s=marker_sizes_h0, 
                               c=sizes_h0, cmap='Blues', alpha=0.7, 
                               edgecolors='darkblue', linewidth=0.5)
        
        plt.colorbar(scatter, ax=ax_h0, label='# Points', fraction=0.046, pad=0.04)
        
        ax_h0.set_xlabel('Birth', fontsize=11)
        ax_h0.set_ylabel('Death', fontsize=11)
        ax_h0.set_title(f'{title_base} - H0 (Connected Components)', 
                       fontsize=11, fontweight='bold')
        ax_h0.set_xlim([0, max_scale])
        ax_h0.set_ylim([0, max_scale])
        ax_h0.set_aspect('equal')
        ax_h0.grid(True, alpha=0.3)
        ax_h0.set_facecolor('#f8f8f8')
        
        # Add statistics
        stats_text = f'Count: {len(h0)}\nPoint range: {sizes_h0.min():.0f}-{sizes_h0.max():.0f}'
        ax_h0.text(0.98, 0.02, stats_text, transform=ax_h0.transAxes,
                  fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    else:
        ax_h0.text(0.5, 0.5, 'No H0 Features', transform=ax_h0.transAxes,
                  fontsize=14, ha='center', va='center')
    
    # ---- Plot H1 ----
    if len(h1) > 0:
        births_h1 = h1[:, 0]
        deaths_h1 = h1[:, 1]
        sizes_h1 = h1[:, 2]
        
        marker_sizes_h1 = 10 + 100 * np.sqrt(sizes_h1 - sizes_h1.min() + 1)
        
        # Plot diagonal
        ax_h1.plot([0, max_scale], [0, max_scale], 'k--', alpha=0.3, linewidth=1.5)
        
        scatter = ax_h1.scatter(births_h1, deaths_h1, s=marker_sizes_h1, 
                               c=sizes_h1, cmap='Reds', alpha=0.7, marker='^',
                               edgecolors='darkred', linewidth=0.5)
        
        plt.colorbar(scatter, ax=ax_h1, label='# Points', fraction=0.046, pad=0.04)
        
        ax_h1.set_xlabel('Birth', fontsize=11)
        ax_h1.set_ylabel('Death', fontsize=11)
        ax_h1.set_title(f'{title_base} - H1 (Loops/Holes)', 
                       fontsize=11, fontweight='bold')
        ax_h1.set_xlim([0, max_scale])
        ax_h1.set_ylim([0, max_scale])
        ax_h1.set_aspect('equal')
        ax_h1.grid(True, alpha=0.3)
        ax_h1.set_facecolor('#f8f8f8')
        
        # Add statistics
        stats_text = f'Count: {len(h1)}\nPoint range: {sizes_h1.min():.0f}-{sizes_h1.max():.0f}'
        ax_h1.text(0.98, 0.02, stats_text, transform=ax_h1.transAxes,
                  fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    else:
        ax_h1.text(0.5, 0.5, 'No H1 Features', transform=ax_h1.transAxes,
                  fontsize=14, ha='center', va='center')

# ==============================================================================
# Visualization 4: Log-Scale Birth (for features near birth=0)
# ==============================================================================

def plot_logscale_birth(ax, h0, h1, title, max_scale=None):
    """
    Standard persistence diagram but with log-scale x-axis to spread out
    features with small birth times.
    """
    if max_scale is None:
        max_scale = 1.5
    
    # Offset for log scale (avoid log(0))
    epsilon = 1e-3
    
    # Plot diagonal (in log space)
    log_vals = np.logspace(np.log10(epsilon), np.log10(max_scale), 100)
    ax.plot(log_vals, log_vals, 'k--', alpha=0.3, linewidth=1.5, label='Birth = Death')
    
    if len(h0) > 0:
        births_h0 = h0[:, 0] + epsilon
        deaths_h0 = h0[:, 1]
        sizes_h0 = h0[:, 2]
        
        marker_sizes_h0 = 10 + 80 * np.sqrt(sizes_h0 - sizes_h0.min() + 1)
        
        ax.scatter(births_h0, deaths_h0, s=marker_sizes_h0, 
                  c='blue', alpha=0.6, edgecolors='darkblue', 
                  linewidth=0.5, label='H0 (Components)')
    
    if len(h1) > 0:
        births_h1 = h1[:, 0] + epsilon
        deaths_h1 = h1[:, 1]
        sizes_h1 = h1[:, 2]
        
        marker_sizes_h1 = 10 + 80 * np.sqrt(sizes_h1 - sizes_h1.min() + 1)
        
        ax.scatter(births_h1, deaths_h1, s=marker_sizes_h1, 
                  c='red', alpha=0.6, marker='^', edgecolors='darkred',
                  linewidth=0.5, label='H1 (Loops)')
    
    ax.set_xlabel('Birth (Log Scale)', fontsize=11)
    ax.set_ylabel('Death', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xscale('log')
    ax.set_xlim([epsilon, max_scale])
    ax.set_ylim([0, max_scale])
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.set_facecolor('#f8f8f8')
    
    info_text = "Marker size ∝ # points\nLog scale spreads H0 features"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))

# ==============================================================================
# Visualization 5: Persistence vs Death
# ==============================================================================

def plot_persistence_vs_death(ax, h0, h1, title, max_scale=None):
    """
    Plot persistence vs death time (another way to avoid y-axis clustering).
    """
    if max_scale is None:
        max_scale = 1.5
    
    if len(h0) > 0:
        deaths_h0 = h0[:, 1]
        persistence_h0 = h0[:, 1] - h0[:, 0]
        sizes_h0 = h0[:, 2]
        
        marker_sizes_h0 = 10 + 80 * np.sqrt(sizes_h0 - sizes_h0.min() + 1)
        
        ax.scatter(deaths_h0, persistence_h0, s=marker_sizes_h0, 
                  c='blue', alpha=0.6, edgecolors='darkblue', 
                  linewidth=0.5, label='H0 (Components)', zorder=2)
    
    if len(h1) > 0:
        deaths_h1 = h1[:, 1]
        persistence_h1 = h1[:, 1] - h1[:, 0]
        sizes_h1 = h1[:, 2]
        
        marker_sizes_h1 = 10 + 80 * np.sqrt(sizes_h1 - sizes_h1.min() + 1)
        
        ax.scatter(deaths_h1, persistence_h1, s=marker_sizes_h1, 
                  c='red', alpha=0.6, marker='^', edgecolors='darkred',
                  linewidth=0.5, label='H1 (Loops)', zorder=3)
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1.5, zorder=1)
    
    ax.set_xlabel('Death Time', fontsize=11)
    ax.set_ylabel('Persistence (Death - Birth)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlim([0, max_scale])
    ax.set_ylim([0, max_scale])
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f8f8')
    
    info_text = "Marker size ∝ # points\nHigher = longer-lived"
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))

# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("Creating Alternative Persistence Visualizations")
    print("="*80)
    print()
    print(f"Sample ID: {SAMPLE_ID}")
    print(f"Output directory: {OUTPUT_DIR}/")
    print()
    
    # -------------------------------------------------------------------------
    # Visualization 1: Persistence vs Birth (2x2 Grid)
    # -------------------------------------------------------------------------
    print("1. Creating Persistence vs Birth plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    max_scale_global = 1.5
    
    for idx, (config_name, label) in enumerate(zip(CONFIG_NAMES, CONFIG_LABELS)):
        ph_data = load_persistence_data(config_name, SAMPLE_ID)
        h0, h1 = ph_data['h0'], ph_data['h1']
        n_points = ph_data['n_points']
        
        title = f"{label}\n{n_points} pts | H0: {len(h0)} | H1: {len(h1)}"
        plot_persistence_vs_birth(axes[idx], h0, h1, title, max_scale_global)
    
    fig.suptitle('Persistence vs Birth Time\n(Better for distinguishing H0 features)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    output_file = os.path.join(OUTPUT_DIR, f'persistence_vs_birth_sample_{SAMPLE_ID:03d}.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Saved: {output_file}\n")
    
    # -------------------------------------------------------------------------
    # Visualization 2: Barcode Diagrams (2x2 Grid)
    # -------------------------------------------------------------------------
    print("2. Creating Barcode diagrams...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    for idx, (config_name, label) in enumerate(zip(CONFIG_NAMES, CONFIG_LABELS)):
        ph_data = load_persistence_data(config_name, SAMPLE_ID)
        h0, h1 = ph_data['h0'], ph_data['h1']
        n_points = ph_data['n_points']
        
        title = f"{label}\n{n_points} pts | H0: {len(h0)} | H1: {len(h1)}"
        plot_barcode(axes[idx], h0, h1, title, max_scale_global)
    
    fig.suptitle('Barcode Diagrams\n(Each bar shows feature lifetime)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    output_file = os.path.join(OUTPUT_DIR, f'barcodes_sample_{SAMPLE_ID:03d}.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Saved: {output_file}\n")
    
    # -------------------------------------------------------------------------
    # Visualization 3: Log-Scale Birth (2x2 Grid)
    # -------------------------------------------------------------------------
    print("3. Creating Log-scale birth diagrams...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    for idx, (config_name, label) in enumerate(zip(CONFIG_NAMES, CONFIG_LABELS)):
        ph_data = load_persistence_data(config_name, SAMPLE_ID)
        h0, h1 = ph_data['h0'], ph_data['h1']
        n_points = ph_data['n_points']
        
        title = f"{label}\n{n_points} pts | H0: {len(h0)} | H1: {len(h1)}"
        plot_logscale_birth(axes[idx], h0, h1, title, max_scale_global)
    
    fig.suptitle('Persistence Diagrams with Log-Scale Birth Axis\n(Spreads features near birth=0)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    output_file = os.path.join(OUTPUT_DIR, f'logscale_birth_sample_{SAMPLE_ID:03d}.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Saved: {output_file}\n")
    
    # -------------------------------------------------------------------------
    # Visualization 4: Persistence vs Death (2x2 Grid)
    # -------------------------------------------------------------------------
    print("4. Creating Persistence vs Death plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    for idx, (config_name, label) in enumerate(zip(CONFIG_NAMES, CONFIG_LABELS)):
        ph_data = load_persistence_data(config_name, SAMPLE_ID)
        h0, h1 = ph_data['h0'], ph_data['h1']
        n_points = ph_data['n_points']
        
        title = f"{label}\n{n_points} pts | H0: {len(h0)} | H1: {len(h1)}"
        plot_persistence_vs_death(axes[idx], h0, h1, title, max_scale_global)
    
    fig.suptitle('Persistence vs Death Time\n(Alternative view of feature lifetimes)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    output_file = os.path.join(OUTPUT_DIR, f'persistence_vs_death_sample_{SAMPLE_ID:03d}.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Saved: {output_file}\n")
    
    # -------------------------------------------------------------------------
    # Visualization 5: Split H0/H1 for one configuration (detailed example)
    # -------------------------------------------------------------------------
    print("5. Creating detailed split H0/H1 view for baseline configuration...")
    
    config_name = 'stddev_0.6'  # Baseline
    label = 'r₀ = 0.6 (Baseline)'
    
    ph_data = load_persistence_data(config_name, SAMPLE_ID)
    h0, h1 = ph_data['h0'], ph_data['h1']
    n_points = ph_data['n_points']
    
    fig, (ax_h0, ax_h1) = plt.subplots(1, 2, figsize=(16, 7))
    
    plot_split_h0_h1(ax_h0, ax_h1, h0, h1, label, max_scale_global)
    
    fig.suptitle(f'Split H0/H1 View - {label} (Sample {SAMPLE_ID})\n{n_points} points total', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    
    output_file = os.path.join(OUTPUT_DIR, f'split_h0_h1_baseline_sample_{SAMPLE_ID:03d}.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Saved: {output_file}\n")
    
    # -------------------------------------------------------------------------
    # Summary Comparison: All 5 views for one configuration
    # -------------------------------------------------------------------------
    print("6. Creating comprehensive comparison with all views for baseline...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[2, :])
    
    config_name = 'stddev_0.6'
    ph_data = load_persistence_data(config_name, SAMPLE_ID)
    h0, h1 = ph_data['h0'], ph_data['h1']
    
    # Standard persistence diagram (for comparison)
    ax1.plot([0, max_scale_global], [0, max_scale_global], 'k--', alpha=0.3, linewidth=1.5)
    if len(h0) > 0:
        sizes_h0 = h0[:, 2]
        marker_sizes_h0 = 10 + 50 * np.sqrt(sizes_h0 - sizes_h0.min() + 1)
        ax1.scatter(h0[:, 0], h0[:, 1], s=marker_sizes_h0, c='blue', alpha=0.6, 
                   edgecolors='darkblue', linewidth=0.5, label='H0')
    if len(h1) > 0:
        sizes_h1 = h1[:, 2]
        marker_sizes_h1 = 10 + 50 * np.sqrt(sizes_h1 - sizes_h1.min() + 1)
        ax1.scatter(h1[:, 0], h1[:, 1], s=marker_sizes_h1, c='red', alpha=0.6, 
                   marker='^', edgecolors='darkred', linewidth=0.5, label='H1')
    ax1.set_xlabel('Birth'); ax1.set_ylabel('Death')
    ax1.set_title('Standard\n(H0 on y-axis)', fontsize=11, fontweight='bold')
    ax1.set_xlim([0, max_scale_global]); ax1.set_ylim([0, max_scale_global])
    ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3); ax1.legend(fontsize=8)
    
    plot_persistence_vs_birth(ax2, h0, h1, 'Persistence\nvs Birth', max_scale_global)
    plot_persistence_vs_death(ax3, h0, h1, 'Persistence\nvs Death', max_scale_global)
    plot_logscale_birth(ax4, h0, h1, 'Log-Scale\nBirth', max_scale_global)
    
    # Simple H0-only view
    ax5.plot([0, max_scale_global], [0, max_scale_global], 'k--', alpha=0.3, linewidth=1.5)
    if len(h0) > 0:
        sizes_h0 = h0[:, 2]
        marker_sizes_h0 = 10 + 80 * np.sqrt(sizes_h0 - sizes_h0.min() + 1)
        scatter = ax5.scatter(h0[:, 0], h0[:, 1], s=marker_sizes_h0, c=sizes_h0, 
                             cmap='Blues', alpha=0.7, edgecolors='darkblue', linewidth=0.5)
        plt.colorbar(scatter, ax=ax5, label='# pts', fraction=0.046, pad=0.04)
    ax5.set_xlabel('Birth'); ax5.set_ylabel('Death')
    ax5.set_title('H0 Only\n(Detailed)', fontsize=11, fontweight='bold')
    ax5.set_xlim([0, max_scale_global]); ax5.set_ylim([0, max_scale_global])
    ax5.set_aspect('equal'); ax5.grid(True, alpha=0.3)
    
    # Simple H1-only view
    ax6.plot([0, max_scale_global], [0, max_scale_global], 'k--', alpha=0.3, linewidth=1.5)
    if len(h1) > 0:
        sizes_h1 = h1[:, 2]
        marker_sizes_h1 = 10 + 80 * np.sqrt(sizes_h1 - sizes_h1.min() + 1)
        scatter = ax6.scatter(h1[:, 0], h1[:, 1], s=marker_sizes_h1, c=sizes_h1, 
                             cmap='Reds', alpha=0.7, marker='^', edgecolors='darkred', linewidth=0.5)
        plt.colorbar(scatter, ax=ax6, label='# pts', fraction=0.046, pad=0.04)
    ax6.set_xlabel('Birth'); ax6.set_ylabel('Death')
    ax6.set_title('H1 Only\n(Detailed)', fontsize=11, fontweight='bold')
    ax6.set_xlim([0, max_scale_global]); ax6.set_ylim([0, max_scale_global])
    ax6.set_aspect('equal'); ax6.grid(True, alpha=0.3)
    
    # Barcode at bottom
    plot_barcode(ax7, h0, h1, 'Barcode View', max_scale_global)
    
    fig.suptitle(f'Visualization Strategy Comparison - Baseline Configuration (Sample {SAMPLE_ID})', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    output_file = os.path.join(OUTPUT_DIR, f'all_views_comparison_sample_{SAMPLE_ID:03d}.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Saved: {output_file}\n")
    
    print("="*80)
    print("✓ All alternative visualizations created!")
    print("="*80)
    print()
    print("Summary of visualization strategies:")
    print("  1. Persistence vs Birth - Spreads H0 features horizontally")
    print("  2. Barcode diagrams - Shows lifetimes as horizontal bars")
    print("  3. Log-scale birth - Spreads features near birth=0")
    print("  4. Persistence vs Death - Another perspective on lifetimes")
    print("  5. Split H0/H1 - Separate plots with optimized scales")
    print("  6. Comprehensive comparison - All views in one figure")
    print()
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print()

