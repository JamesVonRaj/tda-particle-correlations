"""
Core persistence computation functions for TDA pipeline.

This module provides functions for computing persistent homology
on point cloud data, wrapping the TDA utilities.

Optimizations:
- Parallel processing using joblib for ensemble computation
- Optional fastcluster for H0 (single linkage hierarchical clustering)
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from collections import defaultdict, deque
import os

import gudhi
from scipy.spatial import cKDTree

# Check for optional fastcluster
try:
    import fastcluster
    from scipy.spatial.distance import pdist
    HAS_FASTCLUSTER = True
except ImportError:
    HAS_FASTCLUSTER = False

# Check for joblib
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


class UnionFind:
    """Union-Find data structure for tracking connected components in H0."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.birth_time = [0.0] * n

    def find(self, x):
        """Find the root of the component containing x."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y, current_time):
        """Merge components containing x and y at current_time."""
        root_x, root_y = self.find(x), self.find(y)

        if root_x == root_y:
            return None

        if self.birth_time[root_x] > self.birth_time[root_y]:
            root_x, root_y = root_y, root_x

        death_info = {
            'birth': self.birth_time[root_y],
            'death': current_time,
            'num_points': self.size[root_x] + self.size[root_y]
        }

        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]

        return death_info


def compute_h0_fastcluster(points: np.ndarray, max_edge_length: float) -> tuple:
    """
    Compute H0 persistence using fastcluster's single linkage.

    Single linkage hierarchical clustering produces the same dendrogram as
    Union-Find for connected components, but fastcluster is highly optimized.

    Parameters
    ----------
    points : np.ndarray
        (N, 2) array of point coordinates
    max_edge_length : float
        Maximum edge length (distances beyond this are ignored)

    Returns
    -------
    tuple
        (h0_array, edges) where:
        - h0_array: (M, 3) array with [birth, death, num_points] for each feature
        - edges: sorted list of (distance, i, j) tuples for H1 computation
    """
    n_points = len(points)

    # Compute pairwise distances (condensed form for scipy)
    distances = pdist(points)

    # Perform single linkage clustering
    # Returns array of shape (n-1, 4): [idx1, idx2, distance, cluster_size]
    linkage = fastcluster.linkage(distances, method='single')

    # Track component sizes using a simple dict
    # Initially each point is its own component of size 1
    component_size = {i: 1 for i in range(n_points)}

    # Process linkage to extract H0 persistence
    h0_data = []
    edges = []

    for step, (idx1, idx2, dist, new_size) in enumerate(linkage):
        idx1, idx2 = int(idx1), int(idx2)

        # Store edge for H1 computation (need to convert cluster indices back to points)
        # For single linkage, the merge distance is the minimum distance between clusters
        # We'll build edges separately using KDTree for H1

        if dist > max_edge_length:
            continue

        # Both components were born at time 0, so the younger one dies
        # In single linkage, we're merging two clusters
        # The merged cluster index is n_points + step
        new_cluster_id = n_points + step

        # Get sizes of merging components
        size1 = component_size.get(idx1, 1)
        size2 = component_size.get(idx2, 1)

        # H0: a component dies when it merges with another
        # Birth is always 0 for H0 (all points born at t=0)
        # Death is the merge distance
        # num_points is the total after merge
        h0_data.append([0.0, dist, size1 + size2])

        # Update component sizes
        component_size[new_cluster_id] = size1 + size2

    h0_array = np.array(h0_data) if h0_data else np.empty((0, 3))

    # Build edge list using KDTree for H1 (same as before)
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=max_edge_length, output_type='ndarray')

    if len(pairs) == 0:
        edges = []
    else:
        diffs = points[pairs[:, 0]] - points[pairs[:, 1]]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        edges = [(dists[k], pairs[k, 0], pairs[k, 1]) for k in range(len(pairs))]
        edges.sort()

    return h0_array, edges


def compute_h0_unionfind(points: np.ndarray, max_edge_length: float) -> tuple:
    """
    Compute H0 persistence using Union-Find (original implementation).

    Parameters
    ----------
    points : np.ndarray
        (N, 2) array of point coordinates
    max_edge_length : float
        Maximum edge length

    Returns
    -------
    tuple
        (h0_array, edges) where:
        - h0_array: (M, 3) array with [birth, death, num_points] for each feature
        - edges: sorted list of (distance, i, j) tuples for H1 computation
    """
    n_points = len(points)

    tree = cKDTree(points)
    pairs = tree.query_pairs(r=max_edge_length, output_type='ndarray')

    if len(pairs) == 0:
        edges = []
    else:
        diffs = points[pairs[:, 0]] - points[pairs[:, 1]]
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        edges = [(distances[k], pairs[k, 0], pairs[k, 1]) for k in range(len(pairs))]
        edges.sort()

    uf = UnionFind(n_points)
    h0_data = []

    for distance, i, j in edges:
        death_info = uf.union(i, j, distance)
        if death_info is not None:
            h0_data.append([
                death_info['birth'],
                death_info['death'],
                death_info['num_points']
            ])

    h0_array = np.array(h0_data) if h0_data else np.empty((0, 3))

    return h0_array, edges


def find_cycle_at_birth(edge_list, birth_edge, birth_filt):
    """
    Find the cycle that forms when birth_edge is added.

    Uses an optimized BFS that tracks parent pointers instead of full paths.

    Parameters
    ----------
    edge_list : list
        Pre-sorted list of (filt, v1, v2) tuples
    birth_edge : list
        The edge (as list of 2 vertex indices) that creates the cycle
    birth_filt : float
        The filtration value at which the birth_edge appears

    Returns
    -------
    list or None
        Ordered list of vertex indices forming the cycle
    """
    v1, v2 = birth_edge
    birth_edge_set = frozenset(birth_edge)  # frozenset is faster for comparison

    graph = defaultdict(list)

    for filt, a, b in edge_list:
        if filt > birth_filt:
            break
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

    return None


def polygon_area(vertices):
    """Compute the area of a polygon using the shoelace formula."""
    n = len(vertices)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]

    return abs(area) / 2.0


def compute_persistence(
    points: np.ndarray,
    max_edge_length: float,
    use_fastcluster: bool = False
) -> dict:
    """
    Compute Vietoris-Rips persistence for H0 and H1.

    Parameters
    ----------
    points : np.ndarray
        (N, 2) array of point coordinates
    max_edge_length : float
        Maximum edge length for Rips complex
    use_fastcluster : bool
        If True, use fastcluster for H0 (requires fastcluster package).
        Default is False (uses Union-Find).

    Returns
    -------
    dict
        Dictionary with 'h0', 'h1', 'n_points', 'max_edge_length', 'empty'
    """
    n_points = len(points)

    if n_points == 0:
        return {
            'h0': np.empty((0, 3)),
            'h1': np.empty((0, 4)),
            'n_points': 0,
            'max_edge_length': max_edge_length,
            'empty': True
        }

    # H0 computation
    if use_fastcluster and HAS_FASTCLUSTER:
        h0_array, edges = compute_h0_fastcluster(points, max_edge_length)
    else:
        h0_array, edges = compute_h0_unionfind(points, max_edge_length)

    # H1 computation using GUDHI
    rips_complex = gudhi.RipsComplex(points=points, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    simplex_tree.compute_persistence()

    edge_list = edges  # Reuse from H0 computation
    pairs = simplex_tree.persistence_pairs()

    h1_data = []

    for birth_simplex, death_simplex in pairs:
        birth_dim = len(birth_simplex) - 1

        if birth_dim != 1:
            continue

        if len(death_simplex) == 0:
            continue

        birth_time = simplex_tree.filtration(birth_simplex)
        death_time = simplex_tree.filtration(death_simplex)

        cycle_vertex_indices = find_cycle_at_birth(edge_list, birth_simplex, birth_time)

        if cycle_vertex_indices is not None:
            num_vertices = len(cycle_vertex_indices)
            cycle_coords = points[cycle_vertex_indices]
            cycle_area = polygon_area(cycle_coords)
        else:
            feature_vertices = set(birth_simplex)
            feature_vertices.update(death_simplex)
            num_vertices = len(feature_vertices)
            cycle_area = 0.0

        h1_data.append([birth_time, death_time, num_vertices, cycle_area])

    h1_array = np.array(h1_data) if h1_data else np.empty((0, 4))

    return {
        'h0': h0_array,
        'h1': h1_array,
        'n_points': n_points,
        'max_edge_length': max_edge_length,
        'empty': False
    }


def _process_single_sample(sample_file: Path, max_edge_length: float, use_fastcluster: bool) -> dict:
    """
    Process a single sample file (helper for parallel execution).

    Parameters
    ----------
    sample_file : Path
        Path to sample .npy file
    max_edge_length : float
        Maximum edge length for Rips complex
    use_fastcluster : bool
        Whether to use fastcluster for H0

    Returns
    -------
    dict
        Dictionary with persistence results and metadata
    """
    data = np.load(sample_file, allow_pickle=True).item()
    points = data['points']

    persistence_dict = compute_persistence(points, max_edge_length, use_fastcluster)

    # Add metadata from sample if available (grid mode)
    if 'normalized_box_size' in data:
        persistence_dict['normalized_box_size'] = data['normalized_box_size']
    if 'scale_factor' in data:
        persistence_dict['scale_factor'] = data['scale_factor']

    # Save persistence data
    output_file = sample_file.parent / f"{sample_file.stem}_persistence.npy"
    np.save(output_file, persistence_dict, allow_pickle=True)

    return {
        'n_h0_features': len(persistence_dict['h0']),
        'n_h1_features': len(persistence_dict['h1']),
        'n_points': persistence_dict['n_points']
    }


def compute_persistence_for_ensemble(
    data_dir: Path,
    max_edge_length: float,
    mode: str = 'sweep',
    n_jobs: int = -1,
    use_fastcluster: bool = False,
    logger=None
) -> dict:
    """
    Compute persistence for all samples in an ensemble.

    Parameters
    ----------
    data_dir : Path
        Directory containing ensemble data
    max_edge_length : float
        Maximum edge length for Rips complex
    mode : str
        'sweep' or 'grid' mode
    n_jobs : int
        Number of parallel jobs. -1 means use all CPUs.
        Set to 1 to disable parallelization.
    use_fastcluster : bool
        If True, use fastcluster for H0 computation.
    logger : logging.Logger, optional
        Logger for output

    Returns
    -------
    dict
        Summary statistics
    """
    # Find all configuration directories
    if mode == 'sweep':
        config_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('stddev_')])
    else:  # grid
        config_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('r0_')])

    all_summaries = {}

    # Determine whether to use parallel processing
    use_parallel = HAS_JOBLIB and n_jobs != 1

    if use_parallel and logger:
        cpu_count = os.cpu_count() or 1
        actual_jobs = cpu_count if n_jobs == -1 else min(n_jobs, cpu_count)
        logger.info(f"Using parallel processing with {actual_jobs} workers")
    elif not HAS_JOBLIB and n_jobs != 1 and logger:
        logger.warning("joblib not available, falling back to sequential processing")

    if use_fastcluster and logger:
        if HAS_FASTCLUSTER:
            logger.info("Using fastcluster for H0 computation")
        else:
            logger.warning("fastcluster not available, using Union-Find for H0")

    for config_dir in config_dirs:
        config_name = config_dir.name

        # Get sample files
        sample_files = sorted([
            f for f in config_dir.iterdir()
            if f.name.startswith('sample_') and f.suffix == '.npy' and not f.name.endswith('_persistence.npy')
        ])

        if not sample_files:
            if logger:
                logger.warning(f"No sample files found in {config_dir}")
            continue

        if logger:
            logger.info(f"Processing {config_name}: {len(sample_files)} samples")

        # Process samples (parallel or sequential)
        if use_parallel:
            # Parallel processing with joblib
            results = Parallel(n_jobs=n_jobs)(
                delayed(_process_single_sample)(sample_file, max_edge_length, use_fastcluster)
                for sample_file in tqdm(sample_files, desc=f"  {config_name}", leave=False)
            )
        else:
            # Sequential processing
            results = []
            for sample_file in tqdm(sample_files, desc=f"  {config_name}", leave=False):
                result = _process_single_sample(sample_file, max_edge_length, use_fastcluster)
                results.append(result)

        # Aggregate statistics
        stats = {
            'n_samples': len(sample_files),
            'n_h0_features': [r['n_h0_features'] for r in results],
            'n_h1_features': [r['n_h1_features'] for r in results],
            'n_points': [r['n_points'] for r in results],
            'max_edge_length': max_edge_length
        }

        # Compute summary
        summary = {
            'config_name': config_name,
            'n_samples': stats['n_samples'],
            'max_edge_length': max_edge_length,
            'n_points': {
                'mean': float(np.mean(stats['n_points'])),
                'std': float(np.std(stats['n_points'])),
                'min': int(np.min(stats['n_points'])),
                'max': int(np.max(stats['n_points']))
            },
            'h0_features': {
                'mean': float(np.mean(stats['n_h0_features'])),
                'std': float(np.std(stats['n_h0_features'])),
                'min': int(np.min(stats['n_h0_features'])),
                'max': int(np.max(stats['n_h0_features']))
            },
            'h1_features': {
                'mean': float(np.mean(stats['n_h1_features'])),
                'std': float(np.std(stats['n_h1_features'])),
                'min': int(np.min(stats['n_h1_features'])),
                'max': int(np.max(stats['n_h1_features']))
            }
        }

        # Save configuration-specific persistence metadata
        with open(config_dir / 'persistence_metadata.json', 'w') as f:
            json.dump(summary, f, indent=4)

        all_summaries[config_name] = summary

        if logger:
            logger.info(f"  H0: {summary['h0_features']['mean']:.1f} +/- {summary['h0_features']['std']:.1f}")
            logger.info(f"  H1: {summary['h1_features']['mean']:.1f} +/- {summary['h1_features']['std']:.1f}")

    return all_summaries
