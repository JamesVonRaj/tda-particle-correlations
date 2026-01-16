"""
Death Time Statistics vs r₀ with Theoretical Predictions

This script compares empirical death time statistics with theoretical predictions
from the Thomas process cluster characteristics.

Theoretical formulas for Thomas process with σ = r₀:
- Mean Parent-Daughter Distance: E[R] = σ√(π/2) ≈ 1.2533σ
- Median Parent-Daughter Distance: R_median = σ√(2ln(2)) ≈ 1.1774σ  
- RMS Distance: R_RMS = σ√2 ≈ 1.4142σ
- 95% Cluster Radius: R_95 ≈ 2.4477σ
- Mean Intra-Cluster Pair Distance: E[D_Intra] = σ√π ≈ 1.7725σ
- Mean Parent Nearest-Neighbor: E[D_PNN] = 1/(2√λ_P)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy import stats

# ==============================================================================
# Configuration
# ==============================================================================

BASE_DIR = '../../data/ensemble_data'
OUTPUT_DIR = '../../outputs/figures/statistics_vs_r0_with_theory'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thomas process parameters
LAMBDA_PARENT = 0.1  # Parent intensity
LAMBDA_DAUGHTER = 30  # Average daughters per parent

# ==============================================================================
# Theoretical Predictions
# ==============================================================================

def theoretical_mean_parent_daughter_distance(sigma):
    """Mean distance from parent to daughter: E[R] = σ√(π/2)"""
    return sigma * np.sqrt(np.pi / 2)

def theoretical_median_parent_daughter_distance(sigma):
    """Median distance from parent to daughter: R_median = σ√(2ln(2))"""
    return sigma * np.sqrt(2 * np.log(2))

def theoretical_rms_distance(sigma):
    """RMS distance from parent to daughter: R_RMS = σ√2"""
    return sigma * np.sqrt(2)

def theoretical_95_cluster_radius(sigma):
    """95% cluster radius: R_95 ≈ 2.4477σ"""
    return sigma * np.sqrt(-2 * np.log(0.05))

def theoretical_mean_intra_cluster_distance(sigma):
    """Mean intra-cluster pair distance: E[D_Intra] = σ√π"""
    return sigma * np.sqrt(np.pi)

def theoretical_mean_parent_nn_distance(lambda_p):
    """Mean parent nearest-neighbor distance: E[D_PNN] = 1/(2√λ_P)"""
    return 1.0 / (2 * np.sqrt(lambda_p))

def theoretical_quantile_radius(sigma, p):
    """p-quantile cluster radius: R_p = σ√(-2ln(1-p))"""
    return sigma * np.sqrt(-2 * np.log(1 - p))

# ==============================================================================
# Data Loading (same as original script)
# ==============================================================================

def detect_configurations(base_dir=BASE_DIR):
    """Automatically detect all configurations."""
    configs = {}
    if not os.path.exists(base_dir):
        return configs
    
    for item in os.listdir(base_dir):
        if item.startswith('stddev_') and os.path.isdir(os.path.join(base_dir, item)):
            try:
                r0_str = item.replace('stddev_', '')
                r0 = float(r0_str)
                configs[item] = r0
            except ValueError:
                print(f"Warning: Could not parse r₀ from directory name '{item}'")
    
    return configs

def load_all_h0_death_times(config_name, base_dir=BASE_DIR):
    """Load all H0 death times for a given configuration."""
    config_dir = os.path.join(base_dir, config_name)
    
    with open(os.path.join(config_dir, 'persistence_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    n_samples = metadata['n_samples']
    all_death_times = []
    
    for i in range(n_samples):
        persistence_file = os.path.join(config_dir, f'sample_{i:03d}_persistence.npy')
        ph_data = np.load(persistence_file, allow_pickle=True).item()
        
        h0 = ph_data['h0']
        if len(h0) > 0:
            death_times = h0[:, 1]
            all_death_times.extend(death_times)
    
    return np.array(all_death_times)

def compute_statistics(death_times):
    """Compute comprehensive statistics for death times."""
    if len(death_times) == 0:
        return None
    
    # Compute mode using KDE
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(death_times)
        x_range = np.linspace(np.min(death_times), np.max(death_times), 1000)
        kde_values = kde(x_range)
        mode = x_range[np.argmax(kde_values)]
    except:
        hist, bin_edges = np.histogram(death_times, bins=50)
        mode = (bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist) + 1]) / 2
    
    return {
        'count': len(death_times),
        'mean': np.mean(death_times),
        'median': np.median(death_times),
        'mode': mode,
        'std': np.std(death_times),
    }

def compute_per_sample_statistics(config_name, base_dir=BASE_DIR):
    """Compute statistics for each individual sample."""
    config_dir = os.path.join(base_dir, config_name)
    
    with open(os.path.join(config_dir, 'persistence_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    n_samples = metadata['n_samples']
    sample_stats = {'mean': [], 'median': [], 'mode': []}
    
    for i in range(n_samples):
        persistence_file = os.path.join(config_dir, f'sample_{i:03d}_persistence.npy')
        ph_data = np.load(persistence_file, allow_pickle=True).item()
        
        h0 = ph_data['h0']
        if len(h0) > 0:
            death_times = h0[:, 1]
            sample_stats['mean'].append(np.mean(death_times))
            sample_stats['median'].append(np.median(death_times))
            
            # Compute mode
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(death_times)
                x_range = np.linspace(np.min(death_times), np.max(death_times), 500)
                kde_values = kde(x_range)
                mode = x_range[np.argmax(kde_values)]
                sample_stats['mode'].append(mode)
            except:
                hist, bin_edges = np.histogram(death_times, bins=30)
                mode = (bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist) + 1]) / 2
                sample_stats['mode'].append(mode)
    
    return {
        key: {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': np.array(values)
        }
        for key, values in sample_stats.items()
    }

# ==============================================================================
# Main Analysis
# ==============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("Computing Death Time Statistics vs r₀ with Theoretical Predictions")
    print("="*80)
    print()
    
    CONFIGURATIONS = detect_configurations()
    
    # Collect data
    r0_values = []
    all_statistics = {}
    per_sample_statistics = {}
    
    for config_name in sorted(CONFIGURATIONS.keys()):
        r0 = CONFIGURATIONS[config_name]
        r0_values.append(r0)
        
        print(f"Processing {config_name} (r₀ = {r0})...")
        
        death_times = load_all_h0_death_times(config_name)
        stats_dict = compute_statistics(death_times)
        all_statistics[r0] = stats_dict
        
        sample_stats = compute_per_sample_statistics(config_name)
        per_sample_statistics[r0] = sample_stats
    
    r0_values = np.array(r0_values)
    
    # Extract empirical values
    means = np.array([all_statistics[r0]['mean'] for r0 in r0_values])
    mean_stds = np.array([per_sample_statistics[r0]['mean']['std'] for r0 in r0_values])
    medians = np.array([all_statistics[r0]['median'] for r0 in r0_values])
    median_stds = np.array([per_sample_statistics[r0]['median']['std'] for r0 in r0_values])
    modes = np.array([all_statistics[r0]['mode'] for r0 in r0_values])
    mode_stds = np.array([per_sample_statistics[r0]['mode']['std'] for r0 in r0_values])
    
    # Compute theoretical predictions
    theory_mean_pd = theoretical_mean_parent_daughter_distance(r0_values)
    theory_median_pd = theoretical_median_parent_daughter_distance(r0_values)
    theory_rms = theoretical_rms_distance(r0_values)
    theory_95_radius = theoretical_95_cluster_radius(r0_values)
    theory_intra = theoretical_mean_intra_cluster_distance(r0_values)
    theory_pnn = theoretical_mean_parent_nn_distance(LAMBDA_PARENT) * np.ones_like(r0_values)
    
    print("\nCreating plots with theoretical overlays...")
    
    # -------------------------------------------------------------------------
    # Plot 1: Mean Death Time with Theoretical Predictions
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Empirical data
    ax.errorbar(r0_values, means, yerr=mean_stds, 
                marker='o', markersize=10, linewidth=2.5, capsize=5, capthick=2,
                color='#3498db', label='Empirical Mean', alpha=0.8, zorder=5)
    
    # Theoretical predictions
    ax.plot(r0_values, theory_mean_pd, 'r--', linewidth=2, alpha=0.7,
            label=f'Theory: E[R] = σ√(π/2) ≈ 1.25σ')
    ax.plot(r0_values, theory_rms, 'g--', linewidth=2, alpha=0.7,
            label=f'Theory: R_RMS = σ√2 ≈ 1.41σ')
    ax.plot(r0_values, theory_intra, 'm--', linewidth=2, alpha=0.7,
            label=f'Theory: E[D_Intra] = σ√π ≈ 1.77σ')
    ax.axhline(y=theory_pnn[0], color='orange', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Theory: E[D_PNN] = {theory_pnn[0]:.3f} (const)')
    
    ax.set_xlabel('Cluster Standard Deviation σ (r₀)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Death Time', fontsize=14, fontweight='bold')
    ax.set_title('Mean Death Time vs σ: Empirical vs Theoretical Predictions', 
                fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper left')
    
    # Add text box
    textstr = f'λ_P = {LAMBDA_PARENT}\nλ_D = {LAMBDA_DAUGHTER}\nError bars: ± 1 std'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'mean_vs_r0_with_theory.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Plot 2: Median Death Time with Theoretical Predictions
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Empirical data
    ax.errorbar(r0_values, medians, yerr=median_stds,
                marker='s', markersize=10, linewidth=2.5, capsize=5, capthick=2,
                color='#2ecc71', label='Empirical Median', alpha=0.8, zorder=5)
    
    # Theoretical predictions
    ax.plot(r0_values, theory_median_pd, 'r--', linewidth=2, alpha=0.7,
            label=f'Theory: R_median = σ√(2ln2) ≈ 1.18σ')
    ax.plot(r0_values, theory_mean_pd, 'b--', linewidth=2, alpha=0.7,
            label=f'Theory: E[R] = σ√(π/2) ≈ 1.25σ')
    
    # Quantile radii
    theory_q50 = theoretical_quantile_radius(r0_values, 0.5)
    ax.plot(r0_values, theory_q50, 'g--', linewidth=2, alpha=0.7,
            label=f'Theory: R_0.5 (50% quantile)')
    
    ax.set_xlabel('Cluster Standard Deviation σ (r₀)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Death Time', fontsize=14, fontweight='bold')
    ax.set_title('Median Death Time vs σ: Empirical vs Theoretical Predictions', 
                fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper left')
    
    # Add text box
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'median_vs_r0_with_theory.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Plot 3: Location of Max Height with Theoretical Predictions
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Empirical data
    ax.errorbar(r0_values, modes, yerr=mode_stds,
                marker='D', markersize=10, linewidth=2.5, capsize=5, capthick=2,
                color='#e74c3c', label='Empirical Max Height Location', alpha=0.8, zorder=5)
    
    # Theoretical predictions - the mode of Rayleigh is at σ
    theory_rayleigh_mode = r0_values  # Mode of Rayleigh(σ) is σ
    ax.plot(r0_values, theory_rayleigh_mode, 'r--', linewidth=2, alpha=0.7,
            label=f'Theory: Rayleigh Mode = σ')
    ax.plot(r0_values, theory_median_pd, 'g--', linewidth=2, alpha=0.7,
            label=f'Theory: R_median ≈ 1.18σ')
    ax.plot(r0_values, theory_mean_pd, 'b--', linewidth=2, alpha=0.7,
            label=f'Theory: E[R] ≈ 1.25σ')
    
    ax.set_xlabel('Cluster Standard Deviation σ (r₀)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Death Time', fontsize=14, fontweight='bold')
    ax.set_title('Location of Max Height vs σ: Empirical vs Theoretical Predictions', 
                fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper left')
    
    # Add text box
    note_text = f'{textstr}\nThis is the peak of the distribution'
    ax.text(0.98, 0.05, note_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'max_height_location_vs_r0_with_theory.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Plot 4: All Three Together with Theory
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Empirical data
    ax.errorbar(r0_values, means, yerr=mean_stds, 
                marker='o', markersize=10, linewidth=2.5, capsize=5, capthick=2,
                color='#3498db', label='Empirical: Mean', alpha=0.8, zorder=5)
    ax.errorbar(r0_values, medians, yerr=median_stds,
                marker='s', markersize=10, linewidth=2.5, capsize=5, capthick=2,
                color='#2ecc71', label='Empirical: Median', alpha=0.8, zorder=5)
    ax.errorbar(r0_values, modes, yerr=mode_stds,
                marker='D', markersize=10, linewidth=2.5, capsize=5, capthick=2,
                color='#e74c3c', label='Empirical: Max Height', alpha=0.8, zorder=5)
    
    # Theoretical predictions
    ax.plot(r0_values, theory_rayleigh_mode, 'k--', linewidth=2.5, alpha=0.8,
            label='Theory: σ (Rayleigh mode)', zorder=3)
    ax.plot(r0_values, theory_median_pd, 'gray', linestyle='--', linewidth=2, alpha=0.7,
            label='Theory: 1.18σ (median)', zorder=2)
    ax.plot(r0_values, theory_mean_pd, 'brown', linestyle='--', linewidth=2, alpha=0.7,
            label='Theory: 1.25σ (mean)', zorder=2)
    ax.plot(r0_values, theory_intra, 'purple', linestyle='--', linewidth=2, alpha=0.7,
            label='Theory: 1.77σ (intra-cluster)', zorder=2)
    
    ax.set_xlabel('Cluster Standard Deviation σ (r₀)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Death Time', fontsize=14, fontweight='bold')
    ax.set_title('Death Time Statistics vs σ: Comprehensive Comparison', 
                fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='upper left', ncol=2)
    
    # Add text box
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'all_statistics_vs_r0_with_theory.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Plot 5: Comprehensive with All Theoretical Length Scales
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Empirical data
    ax.errorbar(r0_values, means, yerr=mean_stds, 
                marker='o', markersize=10, linewidth=3, capsize=5, capthick=2,
                color='#3498db', label='Empirical: Mean', alpha=0.9, zorder=6)
    ax.errorbar(r0_values, medians, yerr=median_stds,
                marker='s', markersize=10, linewidth=3, capsize=5, capthick=2,
                color='#2ecc71', label='Empirical: Median', alpha=0.9, zorder=6)
    ax.errorbar(r0_values, modes, yerr=mode_stds,
                marker='D', markersize=10, linewidth=3, capsize=5, capthick=2,
                color='#e74c3c', label='Empirical: Max Height', alpha=0.9, zorder=6)
    
    # All theoretical length scales
    ax.plot(r0_values, theory_rayleigh_mode, 'k-', linewidth=2.5, alpha=0.8,
            label='σ (Rayleigh mode)', zorder=4)
    ax.plot(r0_values, theory_median_pd, '--', linewidth=2, alpha=0.7, color='#8B4513',
            label='1.18σ (median parent-daughter)', zorder=3)
    ax.plot(r0_values, theory_mean_pd, '--', linewidth=2, alpha=0.7, color='#4B0082',
            label='1.25σ (mean parent-daughter)', zorder=3)
    ax.plot(r0_values, theory_rms, '--', linewidth=2, alpha=0.7, color='#006400',
            label='1.41σ (RMS distance)', zorder=3)
    ax.plot(r0_values, theory_intra, '--', linewidth=2, alpha=0.7, color='#8B008B',
            label='1.77σ (mean intra-cluster)', zorder=3)
    ax.plot(r0_values, theory_95_radius, ':', linewidth=2, alpha=0.6, color='#FF4500',
            label='2.45σ (95% cluster radius)', zorder=2)
    ax.axhline(y=theory_pnn[0], color='#FF6347', linestyle=':', linewidth=2, alpha=0.6,
               label=f'{theory_pnn[0]:.3f} (mean parent NN)', zorder=1)
    
    ax.set_xlabel('Cluster Standard Deviation σ (r₀)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Death Time / Distance', fontsize=14, fontweight='bold')
    ax.set_title('Death Time Statistics vs σ: All Theoretical Length Scales', 
                fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=9, loc='upper left', ncol=2, framealpha=0.95)
    
    # Add comprehensive text box
    theory_text = (f'Thomas Process Parameters:\n'
                   f'λ_P = {LAMBDA_PARENT} (parent intensity)\n'
                   f'λ_D = {LAMBDA_DAUGHTER} (daughters/parent)\n'
                   f'σ = r₀ (cluster std dev)\n\n'
                   f'Error bars: ± 1 std across samples')
    props2 = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black')
    ax.text(0.98, 0.02, theory_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props2)
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'comprehensive_with_all_theory.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print()
    print("="*80)
    print("✓ Analysis complete!")
    print("="*80)
    print()
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print()
    print("Generated files:")
    print("  - mean_vs_r0_with_theory.png")
    print("  - median_vs_r0_with_theory.png")
    print("  - max_height_location_vs_r0_with_theory.png")
    print("  - all_statistics_vs_r0_with_theory.png")
    print("  - comprehensive_with_all_theory.png")
    print()
    print("Theoretical length scales included:")
    print(f"  - σ (Rayleigh mode) = σ")
    print(f"  - Median parent-daughter distance = 1.177σ")
    print(f"  - Mean parent-daughter distance = 1.253σ")
    print(f"  - RMS distance = 1.414σ")
    print(f"  - Mean intra-cluster distance = 1.772σ")
    print(f"  - 95% cluster radius = 2.448σ")
    print(f"  - Mean parent nearest-neighbor = {theory_pnn[0]:.3f} (constant)")
    print()

