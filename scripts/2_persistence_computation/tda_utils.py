#!/usr/bin/env python

"""
TDA Utility Functions for Persistence Computations

This module provides functions for computing persistent homology
with additional feature information, such as the number of points
constituting a topological feature.

Optimized version using:
- scipy.spatial.cKDTree for sparse edge finding (O(n log n + k) vs O(n²))
- Shared distance computation between H0 and H1
"""

import gudhi
import numpy as np
import sys
from collections import defaultdict, deque
from scipy.spatial import cKDTree
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


def build_edge_list(simplex_tree):
    """
    Extract all edges from simplex tree with their filtration values.
    Returns a sorted list of (filtration, vertex1, vertex2) tuples.
    """
    edges = []
    for simplex, filt in simplex_tree.get_filtration():
        if len(simplex) == 2:
            edges.append((filt, simplex[0], simplex[1]))
    edges.sort()  # Sort by filtration value
    return edges


def find_cycle_at_birth(edge_list, birth_edge, birth_filt):
    """
    Find the cycle (loop) that forms when birth_edge is added at birth_filt.

    When an H1 feature is born, it means the birth_edge completes a cycle.
    We find this cycle by doing BFS from one endpoint of the birth_edge to
    the other, using only edges that existed before the birth_edge was added.

    Uses an optimized BFS that tracks parent pointers instead of full paths,
    avoiding O(path_length) memory allocation per queue operation.

    Args:
        edge_list: Pre-sorted list of (filt, v1, v2) tuples from build_edge_list()
        birth_edge: The edge (as list of 2 vertex indices) that creates the cycle
        birth_filt: The filtration value at which the birth_edge appears

    Returns:
        list: Ordered list of vertex indices forming the cycle, or None if not found
    """
    v1, v2 = birth_edge
    birth_edge_set = frozenset(birth_edge)  # frozenset is faster for comparison

    # Build graph from pre-computed edge list
    # Only include edges with filt < birth_filt, or equal but not the birth edge
    graph = defaultdict(list)

    for filt, a, b in edge_list:
        if filt > birth_filt:
            break  # Edge list is sorted, so we can stop here
        if filt < birth_filt or frozenset((a, b)) != birth_edge_set:
            graph[a].append(b)
            graph[b].append(a)

    # BFS with parent tracking instead of path copying
    parent = {v1: None}
    queue = deque([v1])

    while queue:
        current = queue.popleft()
        if current == v2:
            # Reconstruct path from parent pointers
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1]  # Reverse to get v1 -> v2 order

        for neighbor in graph[current]:
            if neighbor not in parent:
                parent[neighbor] = current
                queue.append(neighbor)

    return None  # No path found (shouldn't happen for valid H1)


def polygon_area(vertices):
    """
    Compute the area of a polygon using the shoelace formula.

    Args:
        vertices: np.ndarray of shape (n, 2) with ordered polygon vertices

    Returns:
        float: Area of the polygon (always positive)
    """
    n = len(vertices)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]

    return abs(area) / 2.0


def compute_rips_persistence_with_point_counts(points, max_edge_length):
    """
    Computes the Vietoris-Rips persistence for H0 and H1, including additional
    geometric information about each topological feature.

    For H0 (connected components): tracks the number of points merged.
    For H1 (loops): tracks the number of vertices in the cycle and the
    area enclosed by the cycle at birth time.

    This function is optimized to only compute simplices up to dimension 2,
    as this is all that is required for H0 (dim 0) and H1 (dim 1).

    Args:
        points (np.ndarray):
            A NumPy array of shape (n_points, 2) for 2D points.
            Note: cycle area computation requires 2D points.

        max_edge_length (float):
            The maximum edge length (distance) to consider for the Rips complex.
            This is the filtration cutoff - edges between points with distance
            greater than this value will not be included. The maximum filtration
            value in the persistence diagram will be max_edge_length.
            You must tune this based on the scale of your point data.

    Returns:
        dict: Dictionary containing:
            - 'h0' (np.ndarray): Array of shape (n_features, 3) for H0.
                Each row is [birth, death, num_points].
            - 'h1' (np.ndarray): Array of shape (m_features, 4) for H1.
                Each row is [birth, death, num_vertices, cycle_area].
                num_vertices: number of points forming the cycle at birth.
                cycle_area: area enclosed by the cycle polygon at birth.
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
            'h1': np.empty((0, 4)),  # 4 columns: birth, death, num_vertices, cycle_area
            'n_points': 0,
            'max_edge_length': max_edge_length,
            'empty': True
        }
    
    # ===================================================================
    # H0 COMPUTATION (Connected Components) - Using Union-Find
    # ===================================================================
    print(f"Computing H0 (connected components) for {n_points} points...")

    # Use KDTree for efficient sparse edge finding
    # This is O(n log n + k) where k = number of edges, vs O(n²) for full matrix
    tree = cKDTree(points)

    # Find all pairs within max_edge_length using query_pairs
    # Returns set of (i, j) pairs with i < j
    pairs = tree.query_pairs(r=max_edge_length, output_type='ndarray')

    if len(pairs) == 0:
        edges = []
    else:
        # Compute distances only for the pairs we need
        # This is much faster than computing the full n×n distance matrix
        diffs = points[pairs[:, 0]] - points[pairs[:, 1]]
        distances = np.sqrt(np.sum(diffs**2, axis=1))

        # Build edge list: (distance, i, j)
        edges = [(distances[k], pairs[k, 0], pairs[k, 1]) for k in range(len(pairs))]
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
    # H1 COMPUTATION (Loops) - Using GUDHI with cycle area computation
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

    # OPTIMIZATION: Reuse the edge list from H0 computation instead of
    # rebuilding from simplex_tree.get_filtration() which is very slow.
    # The edges are the same, just need to ensure format matches.
    edge_list = edges  # Already sorted by distance (= filtration value)
    print(f"Reusing edge list with {len(edge_list)} edges")

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

        # Find the cycle that forms at birth time (using pre-built edge list)
        cycle_vertex_indices = find_cycle_at_birth(edge_list, birth_simplex, birth_time)

        if cycle_vertex_indices is not None:
            num_vertices = len(cycle_vertex_indices)
            cycle_coords = points[cycle_vertex_indices]
            cycle_area = polygon_area(cycle_coords)
        else:
            # Fallback: use vertices from birth and death simplices
            feature_vertices = set(birth_simplex)
            feature_vertices.update(death_simplex)
            num_vertices = len(feature_vertices)
            cycle_area = 0.0  # Cannot compute area without proper cycle

        h1_data.append([birth_time, death_time, num_vertices, cycle_area])

    # Convert to numpy array
    if not h1_data:
        h1_array = np.empty((0, 4))
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
