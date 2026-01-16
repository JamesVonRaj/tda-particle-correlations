import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- Import the function from your other file ---
try:
    # This now imports the new "Thomas process" implementation
    import sys
    sys.path.insert(0, '../1_data_generation')
    from poisson_cluster import simulate_poisson_cluster_process
except ImportError:
    print("Error: Could not find 'poisson_cluster.py'.")
    print("Make sure it is in the '../1_data_generation/' directory.")
    sys.exit()

# ==============================================================================
# Plotting Setup
# ==============================================================================

# --- Define the plotting window ---
WINDOW_SIZE = 15
BOUNDS_X = (0, WINDOW_SIZE)
BOUNDS_Y = (0, WINDOW_SIZE)

# --- Output Directory ---
OUTPUT_DIR = "../../outputs/figures/parameter_comparisons"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Baseline Parameters (adapted for Thomas process) ---
BASE_PARAMS = {
    'lambda_parent': 0.1,
    'lambda_daughter': 30,
    'cluster_std_dev': 0.6,  # This is $r_0$, the Gaussian std dev
    'bounds_x': BOUNDS_X,
    'bounds_y': BOUNDS_Y
}

# --- Define our 4 scenarios - VARYING ONLY cluster_std_dev ---
# 1. Very Tight Clusters (Very Low StdDev)
params_1 = BASE_PARAMS.copy()
params_1['cluster_std_dev'] = 0.2
title_1 = f"Very Tight Clusters:\nParent={params_1['lambda_parent']}, Daughter={params_1['lambda_daughter']}, StdDev={params_1['cluster_std_dev']}"
label_1 = "$r_0 = 0.2$ (Very Tight)"

# 2. Tight Clusters (Low StdDev)
params_2 = BASE_PARAMS.copy()
params_2['cluster_std_dev'] = 0.4
title_2 = f"Tight Clusters:\nParent={params_2['lambda_parent']}, Daughter={params_2['lambda_daughter']}, StdDev={params_2['cluster_std_dev']}"
label_2 = "$r_0 = 0.4$ (Tight)"

# 3. Baseline (Medium StdDev)
params_3 = BASE_PARAMS.copy()
params_3['cluster_std_dev'] = 0.6
title_3 = f"Baseline:\nParent={params_3['lambda_parent']}, Daughter={params_3['lambda_daughter']}, StdDev={params_3['cluster_std_dev']}"
label_3 = "$r_0 = 0.6$ (Baseline)"

# 4. Loose Clusters (High StdDev)
params_4 = BASE_PARAMS.copy()
params_4['cluster_std_dev'] = 1.0
title_4 = f"Loose Clusters:\nParent={params_4['lambda_parent']}, Daughter={params_4['lambda_daughter']}, StdDev={params_4['cluster_std_dev']}"
label_4 = "$r_0 = 1.0$ (Loose)"


# --- Helper function to run and plot ---
def plot_on_ax(ax, title, params):
    """Runs simulation and plots on a given matplotlib axis."""
    
    # Run the simulation (using the new Thomas process function)
    points, parents = simulate_poisson_cluster_process(**params)
    
    # Plot daughter points
    if points.shape[0] > 0:
        ax.scatter(points[:, 0], points[:, 1], s=2, c='blue', alpha=0.7)
    
    # Plot parent points
    if parents.shape[0] > 0:
        ax.scatter(parents[:, 0], parents[:, 1], s=40, c='red', marker='x')
        
        # Plot 2-sigma circles to visualize the cluster "size"
        std_dev = params.get('cluster_std_dev', 0.1)
        for p in parents:
            # 2*std_dev captures ~95% of points
            circle = plt.Circle((p[0], p[1]), std_dev * 2, color='red', fill=False, linestyle='--', alpha=0.4)
            ax.add_patch(circle)
    
    ax.set_title(title, fontsize=10)
    ax.set_xlim(BOUNDS_X)
    ax.set_ylim(BOUNDS_Y)
    ax.set_aspect('equal', 'box')
    ax.set_xticks([])
    ax.set_yticks([])


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    
    # --------------------------------------------------------------------------
    # Plot 1: Visual Parameter Comparison
    # --------------------------------------------------------------------------
    print("Generating cluster_std_dev variation comparison plot (Thomas Process)...")
    
    # Set a fixed random seed for reproducibility
    np.random.seed(42)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Run and plot all 4 scenarios
    plot_on_ax(axes[0, 0], title_1, params_1)
    plot_on_ax(axes[0, 1], title_2, params_2)
    plot_on_ax(axes[1, 0], title_3, params_3)
    plot_on_ax(axes[1, 1], title_4, params_4)
    
    fig.suptitle("Effect of Cluster Standard Deviation ($r_0$) on Thomas Process", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save or show the plot
    figname = os.path.join(OUTPUT_DIR, "cluster_stddev_comparison.png")
    plt.savefig(figname, dpi=200)
    print(f"Plot saved to '{figname}'")

    # ==============================================================================
    # Plot 2 & 3: Theoretical g2(r) and S(k) Plots
    # ==============================================================================
    print("Generating theoretical g2(r) and S(k) plots...")

    def calculate_g2(r_vec, params):
        """Calculates theoretical g2(r) for a 2D Thomas process."""
        lambda_p = params['lambda_parent'] # This is $\rho_P$
        r_0 = params['cluster_std_dev']
        
        # In g2(r), the overall density rho = lambda_p * lambda_daughter (c)
        # The 'c' terms cancel, leaving 1 / (rho_P * ...)
        prefactor = 1.0 / (lambda_p * 4 * np.pi * r_0**2)
        exponent = np.exp(- (r_vec**2) / (4 * r_0**2))
        
        return 1.0 + prefactor * exponent

    def calculate_sk(k_vec, params):
        """Calculates theoretical S(k) for a Thomas process."""
        c = params['lambda_daughter']
        r_0 = params['cluster_std_dev']
        
        exponent = np.exp(- (k_vec**2) * (r_0**2) )
        
        return 1.0 + c * exponent

    # --- Setup for new plots ---
    scenarios = [
        (label_1, params_1),
        (label_2, params_2),
        (label_3, params_3),
        (label_4, params_4)
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Default matplotlib colors
    
    # --- Plot g2(r) ---
    fig_g2, ax_g2 = plt.subplots(figsize=(10, 7))
    r_range = np.linspace(0, 5, 200) # Plot r from 0 to 5

    for (label, params), color in zip(scenarios, colors):
        g2_values = calculate_g2(r_range, params)
        ax_g2.plot(r_range, g2_values, label=label, color=color, lw=2)
        
    ax_g2.set_title("Theoretical Pair Correlation Function $g_2(r)$ - Varying $r_0$", fontsize=16)
    ax_g2.set_xlabel("Distance ($r$)", fontsize=12)
    ax_g2.set_ylabel("$g_2(r)$", fontsize=12)
    ax_g2.set_ylim(bottom=0) # g2(r) shouldn't go below 0
    ax_g2.legend()
    ax_g2.grid(True, linestyle='--', alpha=0.6)
    
    figname_g2 = os.path.join(OUTPUT_DIR, "cluster_g2_stddev_comparison.png")
    fig_g2.savefig(figname_g2, dpi=200)
    print(f"Plot saved to '{figname_g2}'")

    # --- Plot S(k) ---
    fig_sk, ax_sk = plt.subplots(figsize=(10, 7))
    k_range = np.linspace(0, 10, 200) # Plot k from 0 to 10

    for (label, params), color in zip(scenarios, colors):
        sk_values = calculate_sk(k_range, params)
        ax_sk.plot(k_range, sk_values, label=label, color=color, lw=2)

    ax_sk.set_title("Theoretical Structure Factor $S(k)$ - Varying $r_0$", fontsize=16)
    ax_sk.set_xlabel("Wavenumber ($k$)", fontsize=12)
    ax_sk.set_ylabel("$S(k)$", fontsize=12)
    ax_sk.set_ylim(bottom=0) # S(k) shouldn't go below 0
    ax_sk.legend()
    ax_sk.grid(True, linestyle='--', alpha=0.6)
    
    figname_sk = os.path.join(OUTPUT_DIR, "cluster_sk_stddev_comparison.png")
    fig_sk.savefig(figname_sk, dpi=200)
    print(f"Plot saved to '{figname_sk}'")
    
    # plt.show() # Uncomment to display interactively


