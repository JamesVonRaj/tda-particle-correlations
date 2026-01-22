"""
Tests for TDA utility functions.

Tests the UnionFind class and compute_rips_persistence_with_point_counts function.
"""

import sys
import numpy as np
import pytest

# Add the module path for imports
sys.path.insert(0, 'scripts/2_persistence_computation')

from tda_utils import UnionFind, compute_rips_persistence_with_point_counts


# ===================================================================
# UnionFind Tests
# ===================================================================

class TestUnionFind:
    """Tests for the UnionFind data structure."""

    def test_union_find_init(self):
        """Verify initial state: each point is its own component with size=1."""
        uf = UnionFind(5)

        # Each point should be its own parent
        for i in range(5):
            assert uf.find(i) == i
            assert uf.size[i] == 1
            assert uf.birth_time[i] == 0.0

    def test_union_find_simple_merge(self):
        """Two points merge: num_points should be 2 (1+1)."""
        uf = UnionFind(3)

        # Merge points 0 and 1 at time 0.5
        death_info = uf.union(0, 1, 0.5)

        assert death_info is not None
        assert death_info['birth'] == 0.0
        assert death_info['death'] == 0.5
        # Key assertion: num_points is sum of both components (1+1=2)
        assert death_info['num_points'] == 2

        # After merge, they should have the same root
        assert uf.find(0) == uf.find(1)

    def test_union_find_chain_merge(self):
        """Chain merge A→B→C: verify cumulative sizes."""
        uf = UnionFind(3)

        # Merge 0 and 1 at time 0.3
        death_info_1 = uf.union(0, 1, 0.3)
        assert death_info_1['num_points'] == 2  # 1+1

        # Merge 1 and 2 at time 0.7 (this merges component {0,1} with {2})
        death_info_2 = uf.union(1, 2, 0.7)
        assert death_info_2['num_points'] == 3  # 2+1

        # All three should now be in the same component
        assert uf.find(0) == uf.find(1) == uf.find(2)

    def test_union_find_already_connected(self):
        """Union of same component returns None."""
        uf = UnionFind(3)

        # Merge 0 and 1
        uf.union(0, 1, 0.5)

        # Try to merge again - should return None
        result = uf.union(0, 1, 0.8)
        assert result is None

        # Also try the reverse order
        result = uf.union(1, 0, 0.9)
        assert result is None

    def test_union_find_merge_two_components(self):
        """Merge two multi-point components."""
        uf = UnionFind(4)

        # Create component {0, 1} at time 0.2
        uf.union(0, 1, 0.2)

        # Create component {2, 3} at time 0.3
        uf.union(2, 3, 0.3)

        # Merge both components at time 0.5
        death_info = uf.union(1, 2, 0.5)

        # num_points should be 2+2=4
        assert death_info['num_points'] == 4

        # All four should be connected
        root = uf.find(0)
        assert uf.find(1) == root
        assert uf.find(2) == root
        assert uf.find(3) == root


# ===================================================================
# Persistence Computation Tests
# ===================================================================

class TestComputeRipsPersistence:
    """Tests for compute_rips_persistence_with_point_counts function."""

    def test_empty_input(self):
        """Empty array returns empty results."""
        points = np.empty((0, 2))
        result = compute_rips_persistence_with_point_counts(points, max_edge_length=1.0)

        assert result['empty'] is True
        assert result['n_points'] == 0
        assert result['h0'].shape == (0, 3)
        assert result['h1'].shape == (0, 4)  # 4 columns: birth, death, num_vertices, cycle_area

    def test_single_point(self):
        """Single point has no finite H0 features (no merges possible)."""
        points = np.array([[0.0, 0.0]])
        result = compute_rips_persistence_with_point_counts(points, max_edge_length=1.0)

        assert result['empty'] is False
        assert result['n_points'] == 1
        # Single point = no merges = no finite H0 features
        assert result['h0'].shape == (0, 3)
        assert result['h1'].shape == (0, 4)  # 4 columns: birth, death, num_vertices, cycle_area

    def test_two_distant_points(self):
        """Two points beyond max_edge_length stay separate (no H0 features)."""
        # Points at distance 10, max_edge_length is 1
        points = np.array([[0.0, 0.0], [10.0, 0.0]])
        result = compute_rips_persistence_with_point_counts(points, max_edge_length=1.0)

        assert result['n_points'] == 2
        # No edge formed = no merges = no finite H0 features
        assert result['h0'].shape == (0, 3)

    def test_two_close_points(self):
        """Two points within range merge: num_points=2."""
        # Points at distance 0.5
        points = np.array([[0.0, 0.0], [0.5, 0.0]])
        result = compute_rips_persistence_with_point_counts(points, max_edge_length=1.0)

        assert result['n_points'] == 2
        # One merge should occur
        assert result['h0'].shape[0] == 1
        # The merged feature should have num_points=2
        assert result['h0'][0, 2] == 2  # Third column is num_points
        # Birth at 0, death at 0.5
        assert result['h0'][0, 0] == 0.0  # birth
        assert result['h0'][0, 1] == 0.5  # death

    def test_equilateral_triangle_no_h1(self):
        """Equilateral triangle: no H1 feature (triangle fills immediately)."""
        # With equilateral triangle, all edges appear at same time
        # and triangle fills immediately, so no persistent H1
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2]
        ])
        result = compute_rips_persistence_with_point_counts(points, max_edge_length=2.0)

        assert result['n_points'] == 3
        # Should have H0 features (2 merges for 3 points)
        assert result['h0'].shape[0] == 2
        # No H1 features - triangle fills immediately when edges form
        assert result['h1'].shape[0] == 0

    def test_known_configuration_linear(self):
        """Three collinear points: predictable merge order."""
        # Three points on a line: 0---1---2
        # Distance 0-1 = 1.0, distance 1-2 = 1.0, distance 0-2 = 2.0
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0]
        ])
        result = compute_rips_persistence_with_point_counts(points, max_edge_length=3.0)

        assert result['n_points'] == 3
        # Two H0 features (two merges)
        assert result['h0'].shape[0] == 2

        # Sort by death time to check merge order
        h0_sorted = result['h0'][np.argsort(result['h0'][:, 1])]

        # First merge at distance 1.0 (either 0-1 or 1-2)
        assert h0_sorted[0, 1] == 1.0  # death time
        assert h0_sorted[0, 2] == 2    # num_points = 1+1

        # Second merge at distance 1.0 (the other pair) or 2.0 (if 0-2)
        # Since 0-1 and 1-2 both have distance 1.0, they merge first
        # Then the remaining component merges, giving num_points = 3
        assert h0_sorted[1, 2] == 3    # num_points = 2+1

    def test_square_configuration(self):
        """Four points in a square: verify H0 and H1 features."""
        # Unit square
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ])
        result = compute_rips_persistence_with_point_counts(points, max_edge_length=2.0)

        assert result['n_points'] == 4
        # Three H0 features (three merges for 4 points)
        assert result['h0'].shape[0] == 3
        # Should have H1 features (the square forms a loop)
        assert result['h1'].shape[0] >= 1

        # Last merge should have num_points = 4
        h0_sorted = result['h0'][np.argsort(result['h0'][:, 1])]
        assert h0_sorted[-1, 2] == 4

    def test_h0_num_points_sum_property(self):
        """Verify that num_points represents merged component total."""
        # 5 points where we can trace merges
        points = np.array([
            [0.0, 0.0],   # 0
            [0.1, 0.0],   # 1 - very close to 0
            [1.0, 0.0],   # 2
            [1.1, 0.0],   # 3 - very close to 2
            [2.0, 0.0],   # 4
        ])
        result = compute_rips_persistence_with_point_counts(points, max_edge_length=3.0)

        # We should have 4 merges for 5 points
        assert result['h0'].shape[0] == 4

        # All num_points values should be >= 2 (minimum merge is 1+1)
        assert np.all(result['h0'][:, 2] >= 2)

        # The final merge should have num_points = 5
        h0_sorted = result['h0'][np.argsort(result['h0'][:, 1])]
        assert h0_sorted[-1, 2] == 5

    def test_result_structure(self):
        """Verify the returned dictionary has expected keys and types."""
        points = np.array([[0.0, 0.0], [1.0, 0.0]])
        result = compute_rips_persistence_with_point_counts(points, max_edge_length=2.0)

        # Check all expected keys exist
        assert 'h0' in result
        assert 'h1' in result
        assert 'n_points' in result
        assert 'max_edge_length' in result
        assert 'empty' in result

        # Check types
        assert isinstance(result['h0'], np.ndarray)
        assert isinstance(result['h1'], np.ndarray)
        assert isinstance(result['n_points'], int)
        assert isinstance(result['max_edge_length'], float)
        assert isinstance(result['empty'], bool)

        # Check array shapes: h0 has 3 columns, h1 has 4 columns
        if result['h0'].size > 0:
            assert result['h0'].shape[1] == 3  # birth, death, num_points
        if result['h1'].size > 0:
            assert result['h1'].shape[1] == 4  # birth, death, num_vertices, cycle_area


# ===================================================================
# Run tests if executed directly
# ===================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
