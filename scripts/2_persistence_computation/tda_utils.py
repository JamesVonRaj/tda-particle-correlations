#!/usr/bin/env python

"""
TDA Utility Functions for Persistence Computations

This module provides functions for computing persistent homology 
with additional feature information, such as the number of points
constituting a topological feature.
"""

import gudhi
import numpy as np
import sys
from scipy.spatial.distance import pdist, squareform


class UnionFind:
    """Union-Find data structure for tracking connected components in H0"""
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n  # Number of points in each component
        self.birth_time = [0.0] * n  # When each component was born
    
    def find(self, x):
        """Find the root (representative) of the component containing x"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y, current_time):
        """
        Merge the components containing x and y at current_time.
        Returns info about the dying feature, or None if already connected.
        """
        root_x, root_y = self.find(x), self.find(y)
        
        # Already in the same component
        if root_x == root_y:
            return None
        
        # Ensure root_x is the older component (born earlier)
        if self.birth_time[root_x] > self.birth_time[root_y]:
            root_x, root_y = root_y, root_x
        
        # root_y is younger, so it dies
        # num_points is the sum of both component sizes (total points merged)
        death_info = {
            'birth': self.birth_time[root_y],
            'death': current_time,
            'num_points': self.size[root_x] + self.size[root_y]
        }
        
        # Merge: root_y joins root_x
        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        
        return death_info


def compute_rips_persistence_with_point_counts(points, max_edge_length):
    """
    Computes the Vietoris-Rips persistence for H0 and H1, including a count
    of the number of points associated with each topological feature at its death.

    This function is optimized to only compute simplices up to dimension 2,
    as this is all that is required for H0 (dim 0) and H1 (dim 1).

    Args:
        points (np.ndarray): 
            A NumPy array of shape (n_points, n_dims),
            e.g., (100, 2) for 100 2D points.
        
        max_edge_length (float): 
            The maximum edge length to consider for the Rips complex. 
            This is a critical parameter. The maximum filtration value
            (radius) in the persistence diagram will be (max_edge_length / 2).
            You must tune this based on the scale of your point data.

    Returns:
        dict: Dictionary containing:
            - 'h0' (np.ndarray): Array of shape (n_features, 3) for H0.
                Each row is [birth_radius, death_radius, num_points].
            - 'h1' (np.ndarray): Array of shape (m_features, 3) for H1.
                Each row is [birth_radius, death_radius, num_points].
            - 'n_points' (int): Total number of points in the input.
            - 'max_edge_length' (float): The max_edge_length parameter used.
            - 'empty' (bool): Whether the input was empty.
    """
    
    print("--- Starting TDA Computation ---")
    
    n_points = len(points)
    
    # Handle empty point sets
    if n_points == 0:
        return {
            'h0': np.empty((0, 3)),
            'h1': np.empty((0, 3)),
            'n_points': 0,
            'max_edge_length': max_edge_length,
            'empty': True
        }
    
    # ===================================================================
    # H0 COMPUTATION (Connected Components) - Using Union-Find
    # ===================================================================
    print(f"Computing H0 (connected components) for {n_points} points...")
    
    # Compute pairwise distances
    distance_matrix = squareform(pdist(points))
    
    # Get all edges (pairs of points) sorted by distance
    edges = []
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = distance_matrix[i, j]
            # Only include edges within max_edge_length
            if dist <= max_edge_length:
                edges.append((dist, i, j))
    
    edges.sort()  # Sort by distance
    
    # Initialize Union-Find
    uf = UnionFind(n_points)
    
    # Track H0 persistence information
    h0_data = []
    
    # Process edges in order of increasing distance
    for distance, i, j in edges:
        death_info = uf.union(i, j, distance)
        
        if death_info is not None:
            h0_data.append([
                death_info['birth'],
                death_info['death'],
                death_info['num_points']
            ])
    
    # Convert to numpy array
    if not h0_data:
        h0_array = np.empty((0, 3))
    else:
        h0_array = np.array(h0_data)
    
    print(f"H0: Found {len(h0_data)} finite features")
    
    # ===================================================================
    # H1 COMPUTATION (Loops) - Using GUDHI with Cochains
    # ===================================================================
    print(f"Computing H1 (loops) using GUDHI...")
    
    # Build the Vietoris-Rips Complex
    rips_complex = gudhi.RipsComplex(
        points=points,
        max_edge_length=max_edge_length
    )

    # Create the Simplex Tree (only up to dimension 2 for H1)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

    # Compute Persistence
    simplex_tree.compute_persistence()
    
    # Get persistence pairs (birth and death simplices)
    pairs = simplex_tree.persistence_pairs()

    # Process H1 features
    h1_data = []

    for birth_simplex, death_simplex in pairs:
        # The dimension of the feature is determined by the birth simplex dimension
        birth_dim = len(birth_simplex) - 1
        
        # Only process H1 features (dim=1, born from edges)
        if birth_dim != 1:
            continue
        
        # Check if it's a finite feature (has a death simplex)
        if len(death_simplex) == 0:
            continue
        
        # Get birth and death times from the filtration values
        birth_time = simplex_tree.filtration(birth_simplex)
        death_time = simplex_tree.filtration(death_simplex)
        
        # For H1 (loops), collect all unique vertices involved
        # We need to trace the cycle, but as a simple approximation,
        # we can collect vertices from birth and death simplices
        feature_vertices = set(birth_simplex)
        feature_vertices.update(death_simplex)
        
        num_points_in_feature = len(feature_vertices)
        
        h1_data.append([birth_time, death_time, num_points_in_feature])

    # Convert to numpy array
    if not h1_data:
        h1_array = np.empty((0, 3))
    else:
        h1_array = np.array(h1_data)
    
    print(f"H1: Found {len(h1_data)} finite features")
    print("--- TDA Computation Finished ---")
    
    return {
        'h0': h0_array,
        'h1': h1_array,
        'n_points': n_points,
        'max_edge_length': max_edge_length,
        'empty': False
    }

# -----------------------------------------------------------------
#  EXAMPLE USAGE (if script is run directly)
# -----------------------------------------------------------------
if __name__ == "__main__":
    """
    This block runs ONLY when you execute this script directly
    (e.g., `python tda_utils.py`).
    It serves as an example of how to use the function above and
    replicates the behavior of your previous script.
    """
    
    print("--- Running tda_utils.py as a standalone script ---")

    # 1. Define Example Parameters
    INPUT_FILENAME = 'your_points.npy' 
    OUTPUT_FILENAME_H0 = 'h0_rips_persistence.npy'
    OUTPUT_FILENAME_H1 = 'h1_rips_persistence.npy'
    
    # CRITICAL: Tune this value based on your data's scale
    # This value is just an example.
    MAX_EDGE_LENGTH = 1.0  
    
    # 2. Load Data
    try:
        points = np.load(INPUT_FILENAME)
        if points.ndim != 2 or points.shape[0] < 1:
            print(f"Error: Input array from {INPUT_FILENAME} must be 2D and non-empty.")
            sys.exit(1)
        print(f"Loaded {points.shape[0]} points from {INPUT_FILENAME}")
    except FileNotFoundError:
        print(f"Error: Test file not found at {INPUT_FILENAME}")
        print("Create a 'your_points.npy' file to run this example.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading numpy file: {e}")
        sys.exit(1)
        
    # 3. Call the main function
    result = compute_rips_persistence_with_point_counts(
        points=points,
        max_edge_length=MAX_EDGE_LENGTH
    )

    # 4. Save the results
    h0_array = result['h0']
    h1_array = result['h1']
    
    np.save(OUTPUT_FILENAME_H0, h0_array)
    print(f"\nSaved H0 data with shape {h0_array.shape} to {OUTPUT_FILENAME_H0}")
    
    np.save(OUTPUT_FILENAME_H1, h1_array)
    print(f"Saved H1 data with shape {h1_array.shape} to {OUTPUT_FILENAME_H1}")
    
    print(f"\nTotal points processed: {result['n_points']}")
    print(f"Max edge length used: {result['max_edge_length']}")
    
    print("\n--- Example script finished ---")
