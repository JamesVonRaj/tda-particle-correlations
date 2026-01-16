import numpy as np
import matplotlib.pyplot as plt

def simulate_poisson_cluster_process(lambda_parent, lambda_daughter, cluster_std_dev, bounds_x, bounds_y):
    """
    Simulates a 2D Poisson Cluster Process (specifically, a Thomas process).

    In a Thomas process, daughter points are scattered around the parent
    using an isotropic Gaussian (Normal) distribution.

    Args:
        lambda_parent (float): The intensity (average density) of the parent
                               Poisson process. $\rho_P$ in some texts.
        lambda_daughter (float): The average number of daughter points per
                                 parent. $\c$ in some texts.
        cluster_std_dev (float): The standard deviation ($r_0$) of the
                                 Gaussian distribution used to scatter
                                 daughter points around the parent.
        bounds_x (tuple): (min_x, max_x) for the observation window.
        bounds_y (tuple): (min_y, max_y) for the observation window.

    Returns:
        tuple:
            - final_points (np.ndarray): (N, 2) array of all daughter points
                                         within the bounds.
            - parent_locations (np.ndarray): (M, 2) array of the parent
                                             cluster centers.
    """
    # 1. Simulate parent process
    # Calculate area of the observation window
    area = (bounds_x[1] - bounds_x[0]) * (bounds_y[1] - bounds_y[0])
    
    # Number of parent points from a Poisson distribution
    n_parents = np.random.poisson(lambda_parent * area)
    
    if n_parents == 0:
        return np.empty((0, 2)), np.empty((0, 2))
        
    # Locations of parent points, uniformly in the window
    parent_x = np.random.uniform(bounds_x[0], bounds_x[1], n_parents)
    parent_y = np.random.uniform(bounds_y[0], bounds_y[1], n_parents)
    parent_locations = np.stack([parent_x, parent_y], axis=1)

    # 2. Simulate daughter process
    all_daughter_points = []
    
    for parent in parent_locations:
        # Number of daughters for *this* parent
        n_daughters = np.random.poisson(lambda_daughter)
        
        if n_daughters == 0:
            continue
            
        # --- KEY CHANGE: Thomas Process ---
        # Generate offsets from an isotropic Gaussian (Normal) distribution
        # with mean 0 and standard deviation `cluster_std_dev`.
        offset_x = np.random.normal(loc=0.0, scale=cluster_std_dev, size=n_daughters)
        offset_y = np.random.normal(loc=0.0, scale=cluster_std_dev, size=n_daughters)
        
        # Add offsets to the parent location
        daughter_x = parent[0] + offset_x
        daughter_y = parent[1] + offset_y
        
        daughters = np.stack([daughter_x, daughter_y], axis=1)
        all_daughter_points.append(daughters)

    if not all_daughter_points:
        return np.empty((0, 2)), parent_locations

    # Combine all daughter points into one array
    all_points_combined = np.vstack(all_daughter_points)
    
    # 3. Filter points to be within the observation window
    within_bounds = (all_points_combined[:, 0] >= bounds_x[0]) & \
                    (all_points_combined[:, 0] <= bounds_x[1]) & \
                    (all_points_combined[:, 1] >= bounds_y[0]) & \
                    (all_points_combined[:, 1] <= bounds_y[1])
                    
    final_points = all_points_combined[within_bounds]
    
    return final_points, parent_locations

# ==============================================================================
# Example Usage (if this file is run directly)
# ==============================================================================

if __name__ == "__main__":
    
    # --- Parameters ---
    LAMBDA_P = 0.1          # Average 0.1 parents per unit area
    LAMBDA_D = 30           # Average 30 daughters per parent
    CLUSTER_STD = 0.6       # Gaussian standard deviation ($r_0$) of 0.6 units
    BOUNDS = (0, 15)
    
    # --- Run simulation ---
    points, parents = simulate_poisson_cluster_process(
        lambda_parent=LAMBDA_P,
        lambda_daughter=LAMBDA_D,
        cluster_std_dev=CLUSTER_STD,
        bounds_x=BOUNDS,
        bounds_y=BOUNDS
    )
    
    # --- Plot results ---
    print(f"Generated {len(parents)} parent clusters.")
    print(f"Generated {len(points)} total daughter points within bounds.")
    
    plt.figure(figsize=(10, 10))
    
    # Plot daughter points
    plt.scatter(points[:, 0], points[:, 1], s=5, c='blue', alpha=0.6, label='Daughter Points (Final)')
    
    # Plot parent cluster centers
    plt.scatter(parents[:, 0], parents[:, 1], s=50, c='red', marker='x', label='Parent Centers')
    
    # Draw circles representing 2 standard deviations (approx. 95% of points)
    for p in parents:
        circle = plt.Circle((p[0], p[1]), CLUSTER_STD * 2, color='red', fill=False, linestyle='--', alpha=0.4)
        plt.gca().add_patch(circle)
        
    plt.title(f"Thomas Cluster Process (r_0 = {CLUSTER_STD})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim(BOUNDS)
    plt.ylim(BOUNDS)
    plt.legend()
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.5)
    
    figname = "thomas_process_example.png"
    plt.savefig(figname, dpi=150)
    print(f"Saved example plot to {figname}")
    # plt.show()

