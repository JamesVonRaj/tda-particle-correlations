import numpy as np
import matplotlib.pyplot as plt
import gudhi
import ripser
from sklearn.neighbors import NearestNeighbors, KernelDensity
from gudhi.representations import Landscape, PersistenceImage, BettiCurve
import sys
import os
import seaborn as sns

# --- Import the cluster process function ---
try:
    from poisson_cluster import simulate_poisson_cluster_process
except ImportError:
    print("Error: Could not find 'poisson_cluster.py'.")
    print("Make sure it is in the same directory as this script.")
    sys.exit()

# ==============================================================================
# 0. Setup and Configuration
# ==============================================================================

# --- Output Directory ---
OUTPUT_DIR = "TDA_Poisson_Cluster_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Configuration Parameters ---
# Process parameters
LAMBDA_PARENT = 0.1     # Avg. parents per unit area
LAMBDA_DAUGHTER = 30    # Avg. daughters per parent
CLUSTER_RADIUS = 1.2    # Radius of clusters

# Window and simulation parameters
WINDOW_SIZE = 15
DIMENSIONS = 2 # Note: simulate_poisson_cluster_process is hardcoded for 2D
BOUNDS_X = (0, WINDOW_SIZE)
BOUNDS_Y = (0, WINDOW_SIZE)

N_REALIZATIONS = 50
MAX_HOMOLOGY_DIM = 1 # Focus on H0 and H1

# TDA parameters
K_NEIGHBORS = 15 
VECTORIZATION_RESOLUTION = 500
N_LANDSCAPES = 5
PI_RESOLUTION = 50

FILTRATIONS = ["Rips", "Alpha", "DTM", "KDE"]
HOM_DIMS = [0, 1]

PALETTE = {
    "Rips": "#1f77b4",
    "Alpha": "#ff7f0e",
    "DTM": "#2ca02c",
    "KDE": "#d62728"
}

# ==============================================================================
# 1. Helper Functions (generate_HPPP removed)
# ==============================================================================

def gudhi_to_list_format(persistence, max_dim, sqrt_filtration=False):
    """Converts GUDHI persistence tuples to a list of numpy arrays."""
    dgms = [[] for _ in range(max_dim + 1)]
    for dim, (birth, death) in persistence:
        if dim <= max_dim:
            # Exclude infinite features (essential H0 component)
            if death != np.inf and np.isfinite(death) and np.isfinite(birth):
                if sqrt_filtration:
                    birth = np.sqrt(max(0, birth)) 
                    death = np.sqrt(max(0, death))
                dgms[dim].append([birth, death])
    return [np.array(dgm) if dgm else np.empty((0, 2)) for dgm in dgms]

def build_sublevel_filtration(base_tree, vertex_values, max_dim):
    """Builds a sublevel set filtration from a base simplex tree and vertex values."""
    st = gudhi.SimplexTree()
    N_vertices = len(vertex_values)
    
    for i, value in enumerate(vertex_values):
        st.insert([i], filtration=value)

    for simplex, _ in base_tree.get_filtration():
        if len(simplex) > 1 and len(simplex) <= max_dim + 1:
            if all(i < N_vertices for i in simplex):
                max_val = max(vertex_values[i] for i in simplex)
                st.insert(simplex, filtration=max_val)
    return st

def estimate_kde_bandwidth(points):
    """Estimates KDE bandwidth using Scott's Rule of Thumb."""
    n, d = points.shape
    if n <= 1:
        return 0.5 # Default fallback
    # Scott's rule: h = n^(-1/(d+4)) * std(data)
    # We take the average standard deviation across dimensions as a robust estimate
    data_std = np.mean(np.std(points, axis=0))
    # Ensure data_std is not zero if points are collapsed
    if data_std <= 1e-9:
        return 0.5
    bandwidth = n**(-1. / (d + 4.)) * data_std
    return bandwidth

# ==============================================================================
# 2. Main Analysis Loop (Updated for Cluster Process)
# ==============================================================================
all_diagrams = {name: {h_dim: [] for h_dim in HOM_DIMS} for name in FILTRATIONS}

print(f"Starting analysis of Poisson Cluster Process over {N_REALIZATIONS} realizations...")
MAX_EDGE = WINDOW_SIZE / 2

last_realization = {}
kde_bandwidths_used = []

for realization_idx in range(N_REALIZATIONS):
    # --- MODIFICATION: Generate points from the cluster process ---
    points, parent_points = simulate_poisson_cluster_process(
        lambda_parent=LAMBDA_PARENT,
        lambda_daughter=LAMBDA_DAUGHTER,
        cluster_radius=CLUSTER_RADIUS,
        bounds_x=BOUNDS_X,
        bounds_y=BOUNDS_Y
    )
    # -------------------------------------------------------------
    
    N_POINTS = len(points)
    
    if N_POINTS <= 1:
        print(f"Skipping realization {realization_idx} (only {N_POINTS} points).")
        continue

    # A. Vietoris-Rips
    dgms_rips_raw = ripser.ripser(points, maxdim=MAX_HOMOLOGY_DIM, thresh=MAX_EDGE)['dgms']
    dgms_rips = []
    for h_dim, dgm in enumerate(dgms_rips_raw):
        dgm_finite = dgm[np.isfinite(dgm[:, 1]) & np.isfinite(dgm[:, 0])]
        if h_dim == 0:
             # H0 Handling: Remove the essential component
             if dgm_finite.size > 0:
                 dgm_sorted = dgm_finite[dgm_finite[:, 0].argsort()]
                 dgms_rips.append(dgm_sorted[1:]) # Keep all but the first (longest)
             else:
                 dgms_rips.append(dgm_finite)
        else:
            dgms_rips.append(dgm_finite)


    # B. Alpha-Shape
    alpha_complex = gudhi.AlphaComplex(points=points)
    simplex_tree_alpha = alpha_complex.create_simplex_tree()
    persistence_alpha = simplex_tree_alpha.persistence()
    dgms_alpha = gudhi_to_list_format(persistence_alpha, MAX_HOMOLOGY_DIM, sqrt_filtration=True)

    # C. Density-Sensitive Filtrations
    rips_complex_gudhi = gudhi.RipsComplex(points=points, max_edge_length=MAX_EDGE)
    simplex_tree_base = rips_complex_gudhi.create_simplex_tree(max_dimension=MAX_HOMOLOGY_DIM)

    # C1. DTM
    dtm_values = None
    k = min(K_NEIGHBORS, N_POINTS)
    if k > 1:
        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        distances, _ = nbrs.kneighbors(points)
        dtm_values = distances[:, -1]
        simplex_tree_dtm = build_sublevel_filtration(simplex_tree_base, dtm_values, MAX_HOMOLOGY_DIM)
        persistence_dtm = simplex_tree_dtm.persistence()
        dgms_dtm = gudhi_to_list_format(persistence_dtm, MAX_HOMOLOGY_DIM)
    else:
        dgms_dtm = [np.empty((0, 2))] * (MAX_HOMOLOGY_DIM + 1)

    # C2. KDE (ADAPTIVE BANDWIDTH)
    kde_values = None
    KDE_BW = estimate_kde_bandwidth(points)
    kde_bandwidths_used.append(KDE_BW)
    
    try:
        # Use the adaptively calculated bandwidth
        kde = KernelDensity(kernel='gaussian', bandwidth=KDE_BW).fit(points)
        kde_values = -kde.score_samples(points) # Negative density
        simplex_tree_kde = build_sublevel_filtration(simplex_tree_base, kde_values, MAX_HOMOLOGY_DIM)
        persistence_kde = simplex_tree_kde.persistence()
        dgms_kde = gudhi_to_list_format(persistence_kde, MAX_HOMOLOGY_DIM)
    except Exception as e:
        print(f"Warning: KDE calculation failed for realization {realization_idx}: {e}")
        dgms_kde = [np.empty((0, 2))] * (MAX_HOMOLOGY_DIM + 1)

    # Store last realization data
    if realization_idx == N_REALIZATIONS - 1:
        last_realization = {
            'points': points, 
            'parent_points': parent_points, # --- ADDED PARENTS ---
            'DTM': dtm_values, 
            'KDE': kde_values, 
            'KDE_BW': KDE_BW
        }
    
    # --- Store Diagrams ---
    for name, dgms in zip(FILTRATIONS, [dgms_rips, dgms_alpha, dgms_dtm, dgms_kde]):
        for h_dim in HOM_DIMS:
            if len(dgms) > h_dim:
                  all_diagrams[name][h_dim].append(dgms[h_dim])

print("Persistent Homology calculations complete.")
if kde_bandwidths_used:
    print(f"Average adaptive KDE bandwidth used (Scott's Rule): {np.mean(kde_bandwidths_used):.4f}")

# ==============================================================================
# 3. Vectorization and Statistical Analysis
# ==============================================================================
print("Starting vectorization and averaging...")

vectorizations = {name: {h_dim: {} for h_dim in HOM_DIMS} for name in FILTRATIONS}

for name in FILTRATIONS:
    for h_dim in HOM_DIMS:
        diagrams = all_diagrams[name][h_dim]
        non_empty_diagrams = [dgm for dgm in diagrams if dgm.size > 0]

        if not non_empty_diagrams:
            print(f"Note: No H{h_dim} features found for {name} filtration.")
            continue

        # ----------------------------------------------------------------------
        # Determine Global Range (Absolute Scales)
        # ----------------------------------------------------------------------
        all_births = np.concatenate([d[:, 0] for d in non_empty_diagrams])
        all_deaths = np.concatenate([d[:, 1] for d in non_empty_diagrams])
        all_persistences = all_deaths - all_births

        min_birth = np.min(all_births)
        max_death = np.max(all_deaths)
        max_persistence = np.max(all_persistences)

        # ROBUSTNESS CHECK
        if max_death <= min_birth or max_persistence <= 1e-9:
            print(f"Warning: Insufficient filtration range/persistence for H{h_dim} {name}. Skipping.")
            continue

        # Adjustments for specific filtrations
        if name in ["Rips", "Alpha", "DTM"]:
             min_birth = max(0, min_birth)
        if h_dim == 0 and name in ["Rips", "Alpha"]:
            min_birth = 0.0

        # 1D Range (Betti, Landscape) - Absolute Scale
        sample_range = [min_birth, max_death * 1.05]
        # 2D Range (Persistence Image) - Absolute Scale
        image_range = [min_birth, sample_range[1], 0, max_persistence * 1.05]

        # ----------------------------------------------------------------------
        # Vectorization (on Absolute Scales)
        # ----------------------------------------------------------------------
        
        # A. Betti Curves
        BC = BettiCurve(resolution=VECTORIZATION_RESOLUTION, sample_range=sample_range)
        betti_curves_data = BC.fit_transform(non_empty_diagrams)
        
        # B. Persistence Landscapes
        LS = Landscape(num_landscapes=N_LANDSCAPES, 
                       resolution=VECTORIZATION_RESOLUTION, 
                       sample_range=sample_range)
        landscapes_data = LS.fit_transform(non_empty_diagrams)

        # C. Persistence Images
        PI_BANDWIDTH = max(1e-6, (image_range[3] - image_range[2]) / 20.0)
            
        try:
            PI = PersistenceImage(bandwidth=PI_BANDWIDTH, 
                                  resolution=[PI_RESOLUTION, PI_RESOLUTION],
                                  im_range=image_range)
        except TypeError:
            # Fallback for older GUDHI versions if im_range is not recognized
            PI = PersistenceImage(bandwidth=PI_BANDWIDTH, 
                                  resolution=[PI_RESOLUTION, PI_RESOLUTION])

        images_data = PI.fit_transform(non_empty_diagrams)
        avg_image_data = np.mean(images_data, axis=0).reshape(PI_RESOLUTION, PI_RESOLUTION)


        # ----------------------------------------------------------------------
        # Statistics Calculation and Storage
        # ----------------------------------------------------------------------
        
        stats = {
            'sample_range': sample_range,
            'image_range': image_range, 
            # Grid representing the absolute scale
            'grid': np.linspace(sample_range[0], sample_range[1], VECTORIZATION_RESOLUTION),
            'all_deaths': all_deaths,
            'Betti': {
                'mean': np.mean(betti_curves_data, axis=0),
                'p2_5': np.percentile(betti_curves_data, 2.5, axis=0),
                'p97_5': np.percentile(betti_curves_data, 97.5, axis=0),
                'raw': betti_curves_data
            },
            'Landscape': {
                'mean': np.mean(landscapes_data, axis=0),
                'raw': landscapes_data
            },
            'Image': {
                'mean': avg_image_data
            }
        }
        vectorizations[name][h_dim] = stats

print("Vectorization complete.")

# ==============================================================================
# 4. Comprehensive Visualization (Plotting Functions)
# ==============================================================================
print("Generating and saving visualizations...")

# ------------------------------------------------------------------------------
# Figure 1: Input Data and Density Fields Visualization (MODIFIED)
# ------------------------------------------------------------------------------
def plot_density_visualization(last_realization):
    if not last_realization: return

    points = last_realization['points']
    parent_points = last_realization.get('parent_points') # --- GET PARENTS ---
    dtm_values = last_realization['DTM']
    kde_values = last_realization['KDE']
    kde_bw = last_realization.get('KDE_BW', 'N/A')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # --- MODIFIED TITLE ---
    fig.suptitle("Figure 1: Visualization of Filtration Parameters on Example Cluster Process", fontsize=16)

    # A. Raw Points (--- MODIFIED ---)
    ax = axes[0]
    ax.scatter(points[:, 0], points[:, 1], s=20, label='Daughter Points')
    if parent_points is not None and len(parent_points) > 0:
        ax.scatter(parent_points[:, 0], parent_points[:, 1], s=50, c='red', marker='x', label='Parent Points')
    ax.set_title("Poisson Cluster Pattern (Example)")
    ax.legend()
    # -----------------------------
    
    # B. DTM Filtration Values
    ax = axes[1]
    if dtm_values is not None:
        sc = ax.scatter(points[:, 0], points[:, 1], c=dtm_values, cmap='plasma', s=30)
        plt.colorbar(sc, ax=ax, label="Distance to k-th NN")
        ax.set_title(f"DTM (k={K_NEIGHBORS}) Values")

    # C. KDE Filtration Values
    ax = axes[2]
    if kde_values is not None:
        sc = ax.scatter(points[:, 0], points[:, 1], c=kde_values, cmap='plasma', s=30)
        plt.colorbar(sc, ax=ax, label="Negative Log Density")
        title_text = f"KDE (BW≈{kde_bw:.3f}, Adaptive)"
        ax.set_title(title_text)

    for ax in axes:
        ax.set_xlim(0, WINDOW_SIZE); ax.set_ylim(0, WINDOW_SIZE)
        ax.set_aspect('equal', 'box')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    figname = os.path.join(OUTPUT_DIR, "Fig1_Density_Visualization.png")
    plt.savefig(figname, dpi=300)
    plt.close(fig)

# ------------------------------------------------------------------------------
# Figure 2: Absolute Betti Curve Grid (Scale Comparison)
# ------------------------------------------------------------------------------
def plot_absolute_betti_grid(vectorizations):
    # A 2x2 grid showing H0 and H1 Betti curves for each filtration independently.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Figure 2: Betti Curves by Filtration Type (Absolute Scales)", fontsize=16)
    
    # Map filtration names to the axes grid
    ax_map = {
        "Rips": axes[0, 0],
        "Alpha": axes[0, 1],
        "DTM": axes[1, 0],
        "KDE": axes[1, 1]
    }

    # Define X-axis labels
    x_labels = {
        "Rips": "Distance (Radius)",
        "Alpha": "Distance (Radius)",
        "DTM": "Distance (k-th NN)",
        "KDE": "Negative Density"
    }

    for name, ax in ax_map.items():
        ax.set_title(f"{name} Filtration")
        
        plotted = False
        # Plot H0
        if 0 in vectorizations[name] and vectorizations[name][0]:
            stats_h0 = vectorizations[name][0]
            ax.plot(stats_h0['grid'], stats_h0['Betti']['mean'], label="H0 (β0)", color='blue', linestyle='--')
            plotted = True

        # Plot H1
        if 1 in vectorizations[name] and vectorizations[name][1]:
            stats_h1 = vectorizations[name][1]
            ax.plot(stats_h1['grid'], stats_h1['Betti']['mean'], label="H1 (β1)", color='red')
            plotted = True

        ax.set_xlabel(x_labels[name])
        ax.set_ylabel("Average Betti Number")
        
        if plotted:
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    figname = os.path.join(OUTPUT_DIR, "Fig2_Absolute_Betti_Grid.png")
    plt.savefig(figname, dpi=300)
    plt.close(fig)

# ------------------------------------------------------------------------------
# Figure 2B: Normalized Betti Curve Comparison (Shape Comparison)
# ------------------------------------------------------------------------------
def plot_normalized_betti_comparison(vectorizations):
    # Compare the shape of the Betti evolution by normalizing the X-axis (0 to 1).
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("Figure 2B: Normalized Betti Curve Comparison (Shape Comparison)", fontsize=16)

    # The normalized grid for the X-axis
    normalized_grid = np.linspace(0, 1, VECTORIZATION_RESOLUTION)

    for row_idx, h_dim in enumerate(HOM_DIMS):
        ax = axes[row_idx]
        ax.set_title(f"Homology Dimension H{h_dim}")
        
        plotted = False
        for name in FILTRATIONS:
            if h_dim in vectorizations[name] and vectorizations[name][h_dim]:
                stats = vectorizations[name][h_dim]
                # Plot the mean Betti curve against the normalized grid
                ax.plot(normalized_grid, stats['Betti']['mean'], label=name, color=PALETTE[name])
                plotted = True
        
        ax.set_ylabel(f"Average β{h_dim}")
        if plotted:
            ax.legend()

    # Set the X-label only on the bottom plot
    axes[-1].set_xlabel("Normalized Filtration Parameter (0=Start, 1=End)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    figname = os.path.join(OUTPUT_DIR, "Fig2B_Normalized_Betti_Comparison.png")
    plt.savefig(figname, dpi=300)
    plt.close(fig)

# ------------------------------------------------------------------------------
# Figure 3: Persistence Landscape (Layer 1) Comparison with Variance (H1)
# ------------------------------------------------------------------------------
# This plot retains the absolute scales, focusing on comparing magnitudes.
def plot_landscape_comparison_h1(vectorizations):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("Figure 3: H1 Persistence Landscape (Layer 1) Comparison (Absolute Scales)", fontsize=16)

    # Grouping for visualization: Geometric/DTM vs KDE (as their units differ)
    group1_filts = ["Rips", "Alpha", "DTM"]
    group2_filts = ["KDE"]
    h_dim = 1

    def plot_landscape_variance(ax, name, stats):
        grid = stats['grid']
        raw_data = stats['Landscape']['raw']
        # Extract Layer 1 data
        layer1_data = raw_data[:, 0:VECTORIZATION_RESOLUTION]
        
        mean_l1 = np.mean(layer1_data, axis=0)
        p2_5 = np.percentile(layer1_data, 2.5, axis=0)
        p97_5 = np.percentile(layer1_data, 97.5, axis=0)

        ax.plot(grid, mean_l1, label=name, color=PALETTE[name])
        ax.fill_between(grid, p2_5, p97_5, alpha=0.2, color=PALETTE[name])

    # Panel 0: Distance/DTM based
    ax_dist = axes[0]
    ax_dist.set_title("Distance/DTM Filtrations")
    plotted_dist = False
    for name in group1_filts:
        if h_dim in vectorizations[name] and vectorizations[name][h_dim]:
            plot_landscape_variance(ax_dist, name, vectorizations[name][h_dim])
            plotted_dist = True
    ax_dist.set_ylabel("Persistence Level (Layer 1)")
    ax_dist.set_xlabel("Filtration Parameter (Distance/DTM)")
    if plotted_dist:
        ax_dist.legend()

    # Panel 1: Density based (KDE)
    ax_dens = axes[1]
    ax_dens.set_title("KDE Filtration")
    plotted_dens = False
    for name in group2_filts:
         if h_dim in vectorizations[name] and vectorizations[name][h_dim]:
            plot_landscape_variance(ax_dens, name, vectorizations[name][h_dim])
            plotted_dens = True
    ax_dens.set_xlabel("Filtration Parameter (-Density)")
    if plotted_dens:
        ax_dens.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    figname = os.path.join(OUTPUT_DIR, "Fig3_Landscape_L1_Comparison_H1.png")
    plt.savefig(figname, dpi=300)
    plt.close(fig)

# ------------------------------------------------------------------------------
# Figure 4: Persistence Image Comparison Grid
# ------------------------------------------------------------------------------
def plot_image_comparison_grid(vectorizations):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey='row')
    fig.suptitle("Figure 4: Average Persistence Images Comparison Grid (Absolute Scales)", fontsize=16)

    for h_dim, row_axes in enumerate(axes):
        if h_dim not in HOM_DIMS: continue
        for col_idx, name in enumerate(FILTRATIONS):
            ax = row_axes[col_idx]
            ax.set_title(f"{name} H{h_dim}")

            if h_dim in vectorizations[name] and vectorizations[name][h_dim]:
                stats = vectorizations[name][h_dim]
                img_data = stats['Image']['mean']
                extent = stats['image_range']
                
                # Transpose and flip for standard visualization
                img_to_plot = np.flip(img_data.T, 0)
                
                im = ax.imshow(img_to_plot, cmap='viridis', extent=extent, aspect='auto')
                ax.set_xlabel("Birth")
                if col_idx == 0:
                    ax.set_ylabel("Persistence")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, 'No Data or Skipped', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                if col_idx == 0:
                    ax.set_ylabel("Persistence")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    figname = os.path.join(OUTPUT_DIR, "Fig4_Persistence_Image_Grid.png")
    plt.savefig(figname, dpi=300)
    plt.close(fig)

# ------------------------------------------------------------------------------
# Figure 5: H0 Death Time Distribution (Clustering Scales)
# ------------------------------------------------------------------------------
def plot_h0_death_distribution(vectorizations):
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Figure 5: H0 Death Time Distribution (Clustering Scales)", fontsize=16)

    h_dim = 0
    # Focus on distance-interpretable filtrations
    H0_FILTRATIONS = ["Rips", "Alpha", "DTM"] 

    plotted = False
    for name in H0_FILTRATIONS:
         if h_dim in vectorizations[name] and vectorizations[name][h_dim]:
            stats = vectorizations[name][h_dim]
            if len(stats['all_deaths']) > 1:
                sns.kdeplot(stats['all_deaths'], ax=ax, label=f"{name}", color=PALETTE[name], fill=True, alpha=0.1)
                plotted = True

    ax.set_xlabel("H0 Death Time (Distance Scale / DTM)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution related to Nearest Neighbor Distances and Clustering")
    if plotted:
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    figname = os.path.join(OUTPUT_DIR, "Fig5_H0_Death_Distribution.png")
    plt.savefig(figname, dpi=300)
    plt.close(fig)

# ------------------------------------------------------------------------------
# Figure 6: Rips H1 Variance Analysis (Betti and Landscape)
# ------------------------------------------------------------------------------
def plot_rips_h1_variance(vectorizations):
    name = "Rips"
    h_dim = 1
    
    if h_dim not in vectorizations[name] or not vectorizations[name][h_dim]:
        return

    stats = vectorizations[name][h_dim]
    grid = stats['grid']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Figure 6: Rips Filtration H1 Variance Analysis (95% Range)", fontsize=16)

    # A. Betti Curve Variance
    ax = axes[0]
    mean_betti = stats['Betti']['mean']
    p2_5 = stats['Betti']['p2_5']
    p97_5 = stats['Betti']['p97_5']

    ax.plot(grid, mean_betti, color=PALETTE[name], label="Average β1")
    ax.fill_between(grid, p2_5, p97_5, alpha=0.3, color=PALETTE[name], label="95% Range")
    ax.set_title("Betti Curve Variance")
    ax.set_xlabel("Distance Scale")
    ax.set_ylabel("β1")
    ax.legend()

    # B. Landscape Variance
    ax = axes[1]
    raw_landscapes = stats['Landscape']['raw']
    
    for i in range(min(3, N_LANDSCAPES)): 
        start = i * VECTORIZATION_RESOLUTION
        end = (i + 1) * VECTORIZATION_RESOLUTION
        
        layer_data = raw_landscapes[:, start:end]
        mean = np.mean(layer_data, axis=0)
        p2_5_ls = np.percentile(layer_data, 2.5, axis=0)
        p97_5_ls = np.percentile(layer_data, 97.5, axis=0)
        
        color = f'C{i}'
        ax.plot(grid, mean, label=f'Layer {i} Avg', color=color)
        ax.fill_between(grid, p2_5_ls, p97_5_ls, color=color, alpha=0.2)

    ax.set_title("Persistence Landscape Variance")
    ax.set_xlabel("Distance Scale (Midpoint)")
    ax.set_ylabel("Persistence Level")
    ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    figname = os.path.join(OUTPUT_DIR, "Fig6_Rips_H1_Variance.png")
    plt.savefig(figname, dpi=300)
    plt.close(fig)

# ==============================================================================
# 5. Execution
# ==============================================================================

plot_density_visualization(last_realization)
plot_absolute_betti_grid(vectorizations)
plot_normalized_betti_comparison(vectorizations)
plot_landscape_comparison_h1(vectorizations)
plot_image_comparison_grid(vectorizations)
plot_h0_death_distribution(vectorizations)
plot_rips_h1_variance(vectorizations)

print(f"\nAnalysis complete. All figures saved to the directory: {OUTPUT_DIR}")
