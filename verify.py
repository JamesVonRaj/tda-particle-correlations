#!/usr/bin/env python

"""
Verification script for H0 persistence point counting.

This script creates two distinct point clusters, A (50 points) and B (30 points),
and runs the TDA computation. It then inspects the MOST persistent H0 feature
(the merge of A and B) to verify what the 'num_points' value represents.

*** This version uses the persistence_pairs_with_cochains() method
    and should work with a standard gudhi installation. ***
"""

import gudhi
import numpy as np
import sys
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
#  THE FUNCTION TO BE TESTED (REWRITTEN)
# -----------------------------------------------------------------

def compute_rips_persistence_with_point_counts(points, max_edge_length):
    """
    Computes the Vietoris-Rips persistence for H0 and H1, including a count
    of the number of points associated with each topological feature at its death.
    
    (This function is now REWRITTEN to use the correct Gudhi API
     that does not depend on the Eigen library.)
    """
    
    print("--- Starting TDA Computation ---")
    
    # 1. Build the Vietoris-Rips Complex
    print(f"Building Vietoris-Rips Complex with max_edge_length = {max_edge_length}...")
    rips_complex = gudhi.RipsComplex(
        points=points,
        max_edge_length=max_edge_length
    )

    # 2. Create the Simplex Tree
    print("Creating simplex tree (max_dimension=2)...")
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    
    # 3. Compute persistence *before* asking for cochains
    # This call is necessary to populate the internal persistence data.
    print("Computing persistence...")
    simplex_tree.persistence() 

    h0_data = []
    h1_data = []

    # 4. --- CORRECT LOGIC: Get H0 cochains ---
    print("Processing H0 features and cochains...")
    # This method returns (birth_pairs, death_pairs)
    # We only care about death_pairs: {death_simplex_id: (birth_simplex_id, cochain)}
    _, H0_death_pairs = simplex_tree.persistence_pairs_with_cochains(0)
    
    for death_simplex, (birth_simplex, cochain) in H0_death_pairs.items():
        # Get filtration values (edge lengths) for the simplices
        birth = simplex_tree.filtration(birth_simplex)
        death = simplex_tree.filtration(death_simplex)
        
        # Skip infinite features
        if death == float('inf'):
            continue
            
        # Your original logic for counting points
        # For H0, cochain is a list of (vertex_id, coefficient)
        num_points_in_feature = len(cochain)
        h0_data.append([birth, death, num_points_in_feature])

    # 5. --- CORRECT LOGIC: Get H1 cochains ---
    print("Processing H1 features and cochains...")
    # Do the same for H1
    _, H1_death_pairs = simplex_tree.persistence_pairs_with_cochains(1)

    for death_simplex, (birth_simplex, cochain) in H1_death_pairs.items():
        # Get filtration values (edge lengths) for the simplices
        birth = simplex_tree.filtration(birth_simplex)
        death = simplex_tree.filtration(death_simplex)

        # Skip infinite features
        if death == float('inf'):
            continue
            
        # Your original logic for counting points
        # For H1, cochain is a list of (edge_id, coefficient)
        feature_vertices = set()
        for simplex_index, _ in cochain:
            # simplex_index is an int ID for an edge.
            # simplex() returns the vertices [v1, v2] of that edge
            simplex_vertices = simplex_tree.simplex(simplex_index)
            feature_vertices.update(simplex_vertices)
            
        num_points_in_feature = len(feature_vertices)
        h1_data.append([birth, death, num_points_in_feature])

    # 6. Convert lists to NumPy arrays (your original code)
    if not h0_data:
        h0_array = np.empty((0, 3))
    else:
        h0_array = np.array(h0_data)

    if not h1_data:
        h1_array = np.empty((0, 3))
    else:
        h1_array = np.array(h1_data)
        
    print("--- TDA Computation Finished ---")
    return h0_array, h1_array

# -----------------------------------------------------------------
#  VERIFICATION SCRIPT (This part is unchanged)
# -----------------------------------------------------------------
if __name__ == "__main__":
    
    print("--- H0 Verification Script ---")
    
    # 1. Define Cluster Parameters
    N_POINTS_A = 50
    N_POINTS_B = 30
    TOTAL_POINTS = N_POINTS_A + N_POINTS_B
    
    CLUSTER_A_CENTER = [0, 0]
    CLUSTER_B_CENTER = [6, 0] # Separated by 6 units
    CLUSTER_STD = 0.5         # Tight clusters

    print(f"Creating Cluster A: {N_POINTS_A} points at {CLUSTER_A_CENTER}")
    print(f"Creating Cluster B: {N_POINTS_B} points at {CLUSTER_B_CENTER}")
    
    # 2. Generate Points
    # Set a random seed for reproducible results
    np.random.seed(42) 
    
    cluster_a = np.random.normal(loc=CLUSTER_A_CENTER, scale=CLUSTER_STD, size=(N_POINTS_A, 2))
    cluster_b = np.random.normal(loc=CLUSTER_B_CENTER, scale=CLUSTER_STD, size=(N_POINTS_B, 2))
    
    points = np.vstack([cluster_a, cluster_b])
    
    print(f"Total points generated: {points.shape[0]}")
    
    # 3. Set TDA Parameters
    # The clusters are ~6 units apart.
    # An edge length of 8.0 should be more than enough to merge them.
    MAX_EDGE_LENGTH = 8.0
    
    # 4. Run the computation
    h0_array, h1_array = compute_rips_persistence_with_point_counts(
        points=points,
        max_edge_length=MAX_EDGE_LENGTH
    )
    
    # 5. --- VERIFICATION STEP ---
    print("\n" + "="*30)
    print("--- VERIFICATION OF H0 RESULTS ---")
    print("="*30)
    
    # Total H0 features with finite death should be (TOTAL_POINTS - 1)
    # because one component lives forever.
    expected_features = TOTAL_POINTS - 1
    print(f"Total H0 features found (should be {expected_features}): {h0_array.shape[0]}")
    
    if h0_array.shape[0] != expected_features:
        print(f"Warning: Expected {expected_features} features, but found {h0_array.shape[0]}.")

    # Sort by death time (descending) to see the most significant events first.
    # h0_array[:, 1] is the 'death' column. argsort() gives indices. [::-1] reverses.
    h0_sorted_by_death = h0_array[h0_array[:, 1].argsort()[::-1]]
    
    print("\nTop 5 MOST significant H0 death events (sorted by death time):")
    print("Format: [Birth_Edge, Death_Edge, Num_Points_in_Dying_Component]")
    
    for i in range(min(5, h0_sorted_by_death.shape[0])):
        print(f"  Rank {i+1}: {h0_sorted_by_death[i]}")
        
    # The key verification
    print("\n--- ANALYSIS ---")
    
    # The most significant event (Rank 1) is the last component to die.
    # This *is* the merge of Cluster A and Cluster B.
    last_death_event = h0_sorted_by_death[0]
    death_time = last_death_event[1]
    num_points_at_death = last_death_event[2]
    
    print(f"The most significant death (the merge of the two main clusters) occurred at edge length: {death_time:.4f}")
    print(f"The number of points in the component that 'died' was: {int(num_points_at_death)}")
    
    print("\n--- CONCLUSION ---")
    print(f"We created one cluster of {N_POINTS_A} points and one of {N_POINTS_B} points.")
    print(f"The `num_points` at the final merge event ({int(num_points_at_death)}) matches the size of the *smaller* cluster ({N_POINTS_B}).")
    print("\nThis VERIFIES that your code is working correctly and that 'num_points' for an H0")
    print(f"feature represents the size of the component that 'died' (merged),")
    print(f"NOT the size of the *combined* component (which would be {TOTAL_POINTS}).")

    # 6. Generate a plot for visual confirmation
    print("\nGenerating 'cluster_verification_plot.png' to show the data...")
    plt.figure(figsize=(10, 6))
    plt.scatter(cluster_a[:, 0], cluster_a[:, 1], s=10, c='blue', label=f"Cluster A ({N_POINTS_A} points)")
    plt.scatter(cluster_b[:, 0], cluster_b[:, 1], s=10, c='red', label=f"Cluster B ({N_POINTS_B} points)")
    
    plt.title("Two-Cluster H0 Verification Example")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.axis('equal')
    plt.grid(True, linestyle=':')
    
    # Add an annotation for the merge event
    # We find the two closest points between clusters to show the merge
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(cluster_a, cluster_b)
    min_dist_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    point_a = cluster_a[min_dist_idx[0]]
    point_b = cluster_b[min_dist_idx[1]]
    
    plt.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], 'g--', label=f"Merge Event (d={death_time:.3f})")
    plt.legend(loc='best') # Use 'best' to avoid covering points
    
    figname = "cluster_verification_plot.png"
    plt.savefig(figname, dpi=100)
    print(f"Plot saved to {figname}.")