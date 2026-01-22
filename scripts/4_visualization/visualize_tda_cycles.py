#!/usr/bin/env python
"""
Visualize H1 Cycles in a Simplicial Complex

Shows how the shoelace formula computes cycle areas for actual H1 features
born in a Vietoris-Rips filtration of a point pattern.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict, deque
import gudhi

np.random.seed(42)  # For reproducibility


def polygon_area(vertices):
    """Compute area of a polygon using the shoelace formula."""
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    return abs(area) / 2.0


def build_edge_list(simplex_tree):
    """Extract all edges from simplex tree with their filtration values."""
    edges = []
    for simplex, filt in simplex_tree.get_filtration():
        if len(simplex) == 2:
            edges.append((filt, simplex[0], simplex[1]))
    edges.sort()
    return edges


def find_cycle_at_birth(edge_list, birth_edge, birth_filt, points):
    """
    Find the cycle that forms when birth_edge is added at birth_filt.
    Returns the ordered list of point indices forming the cycle.
    """
    v1, v2 = birth_edge
    birth_edge_set = set(birth_edge)

    # Build graph of edges that exist just before the birth edge is added
    graph = defaultdict(list)
    for filt, a, b in edge_list:
        if filt > birth_filt:
            break
        if filt < birth_filt or {a, b} != birth_edge_set:
            graph[a].append(b)
            graph[b].append(a)

    # BFS to find path from v1 to v2 (which completes the cycle)
    queue = deque([(v1, [v1])])
    visited = {v1}

    while queue:
        current, path = queue.popleft()
        if current == v2:
            return path  # This path + edge (v2, v1) forms the cycle
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None


def get_edges_at_filtration(points, epsilon):
    """Get all edges with distance <= epsilon."""
    dists = squareform(pdist(points))
    edges = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if dists[i, j] <= epsilon:
                edges.append([points[i], points[j]])
    return edges


def generate_clustered_points(n_points=70, n_clusters=5, cluster_std=0.15):
    """Generate a clustered point pattern."""
    points_per_cluster = n_points // n_clusters
    points = []

    # Generate cluster centers
    centers = np.random.uniform(0.2, 0.8, size=(n_clusters, 2))

    for center in centers:
        cluster_points = center + np.random.normal(0, cluster_std, size=(points_per_cluster, 2))
        points.extend(cluster_points)

    # Add some extra random points
    remaining = n_points - len(points)
    if remaining > 0:
        extra_points = np.random.uniform(0, 1, size=(remaining, 2))
        points.extend(extra_points)

    return np.array(points)


def main():
    # Generate point pattern
    points = generate_clustered_points(n_points=70, n_clusters=6, cluster_std=0.08)
    n_points = len(points)

    # Compute persistence using GUDHI
    max_edge = 0.5
    rips = gudhi.RipsComplex(points=points, max_edge_length=max_edge)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)

    # Compute persistence
    simplex_tree.compute_persistence()

    # Get H1 features (loops)
    h1_features = []
    for interval in simplex_tree.persistence_intervals_in_dimension(1):
        birth, death = interval
        if death < float('inf'):
            h1_features.append((birth, death))

    # Sort by birth time
    h1_features.sort(key=lambda x: x[0])

    # Build edge list for cycle finding
    edge_list = build_edge_list(simplex_tree)

    # Find cycles for H1 features
    cycles_data = []
    for birth, death in h1_features:
        # Find the edge that created this H1 feature
        for filt, a, b in edge_list:
            if abs(filt - birth) < 1e-10:
                cycle_indices = find_cycle_at_birth(edge_list, (a, b), birth, points)
                if cycle_indices is not None and len(cycle_indices) >= 3:
                    cycle_vertices = points[cycle_indices]
                    area = polygon_area(cycle_vertices)
                    cycles_data.append({
                        'birth': birth,
                        'death': death,
                        'lifetime': death - birth,
                        'indices': cycle_indices,
                        'vertices': cycle_vertices,
                        'area': area,
                        'n_vertices': len(cycle_indices)
                    })
                    break

    # Select a few interesting cycles to display
    # Sort by area to get different sizes
    cycles_data.sort(key=lambda x: x['area'], reverse=True)

    # Create figure
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Full point pattern with all edges at a medium filtration
    ax1 = fig.add_subplot(2, 2, 1)
    epsilon_show = 0.15
    edges = get_edges_at_filtration(points, epsilon_show)

    if edges:
        lc = LineCollection(edges, colors='lightgray', linewidths=0.5, alpha=0.5)
        ax1.add_collection(lc)

    ax1.scatter(points[:, 0], points[:, 1], s=30, c='darkblue', zorder=5)
    ax1.set_title(f'Point Pattern (n={n_points}) with edges at ε={epsilon_show}', fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # Plots 2-4: Individual cycles with area calculation
    for idx, cycle_idx in enumerate([0, min(1, len(cycles_data)-1), min(2, len(cycles_data)-1)]):
        if cycle_idx >= len(cycles_data):
            continue

        ax = fig.add_subplot(2, 2, idx + 2)
        cycle = cycles_data[cycle_idx]

        # Show edges at the birth filtration
        epsilon = cycle['birth'] * 1.01  # Slightly after birth to include the closing edge
        edges = get_edges_at_filtration(points, epsilon)

        if edges:
            lc = LineCollection(edges, colors='lightgray', linewidths=0.8, alpha=0.6)
            ax.add_collection(lc)

        # Plot all points
        ax.scatter(points[:, 0], points[:, 1], s=25, c='gray', alpha=0.5, zorder=3)

        # Highlight cycle vertices
        cycle_pts = cycle['vertices']
        ax.scatter(cycle_pts[:, 0], cycle_pts[:, 1], s=80, c='red', zorder=6, edgecolors='darkred', linewidths=1.5)

        # Draw cycle polygon (filled)
        polygon = MplPolygon(cycle_pts, alpha=0.3, facecolor='red', edgecolor='red', linewidth=2, zorder=4)
        ax.add_patch(polygon)

        # Draw cycle edges explicitly
        for i in range(len(cycle_pts)):
            j = (i + 1) % len(cycle_pts)
            ax.plot([cycle_pts[i, 0], cycle_pts[j, 0]],
                   [cycle_pts[i, 1], cycle_pts[j, 1]],
                   'r-', linewidth=2.5, zorder=5)

        # Label cycle vertices
        for i, (x, y) in enumerate(cycle_pts):
            ax.annotate(f'{cycle["indices"][i]}', (x, y),
                       textcoords='offset points', xytext=(5, 5),
                       fontsize=8, fontweight='bold', color='darkred')

        # Add info text
        info_text = (f'H1 Cycle #{idx+1}\n'
                    f'Birth ε = {cycle["birth"]:.4f}\n'
                    f'Death ε = {cycle["death"]:.4f}\n'
                    f'Lifetime = {cycle["lifetime"]:.4f}\n'
                    f'Vertices = {cycle["n_vertices"]}\n'
                    f'Area = {cycle["area"]:.4f}')

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        # Shoelace calculation
        n = len(cycle_pts)
        shoelace_text = "Shoelace:\n"
        terms = []
        for i in range(n):
            j = (i + 1) % n
            term = cycle_pts[i, 0] * cycle_pts[j, 1] - cycle_pts[j, 0] * cycle_pts[i, 1]
            terms.append(term)
        shoelace_text += f"Sum = {sum(terms):.3f}\n"
        shoelace_text += f"Area = |{sum(terms):.3f}|/2\n"
        shoelace_text += f"     = {abs(sum(terms))/2:.4f}"

        ax.text(0.98, 0.02, shoelace_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', horizontalalignment='right',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

        ax.set_title(f'Cycle at birth ε = {cycle["birth"]:.4f}', fontsize=11, fontweight='bold')
        ax.set_aspect('equal')

        # Zoom to cycle region with padding
        x_min, x_max = cycle_pts[:, 0].min(), cycle_pts[:, 0].max()
        y_min, y_max = cycle_pts[:, 1].min(), cycle_pts[:, 1].max()
        pad = max(x_max - x_min, y_max - y_min) * 0.5
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.suptitle('H1 Cycles in Vietoris-Rips Complex: Shoelace Area Computation\n'
                 'Red polygons show cycles (holes) at their birth filtration value',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    output_path = "../../outputs/report_figures/tda_cycles_shoelace.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    # Print summary
    print(f"\nFound {len(cycles_data)} H1 cycles")
    print("\nCycle Summary:")
    print("-" * 70)
    for i, c in enumerate(cycles_data[:5]):
        print(f"Cycle {i+1}: birth={c['birth']:.4f}, death={c['death']:.4f}, "
              f"vertices={c['n_vertices']}, area={c['area']:.4f}")

    plt.show()


if __name__ == "__main__":
    main()
