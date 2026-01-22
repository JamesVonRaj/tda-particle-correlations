#!/usr/bin/env python
"""
Visualize Multiple Valid Cycles for the Same H1 Feature

Shows that when an H1 feature is born, there can be multiple valid cycles,
and BFS finds the one with minimum edge count, not minimum area.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import LineCollection
from collections import defaultdict, deque


def polygon_area(vertices):
    """Compute area using shoelace formula."""
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    return abs(area) / 2.0


def find_all_simple_cycles(graph, v1, v2, max_length=10):
    """
    Find all simple cycles that include edge (v1, v2).
    Uses DFS to find all paths from v1 to v2 not using that edge.
    """
    cycles = []

    def dfs(current, target, path, visited):
        if len(path) > max_length:
            return
        if current == target and len(path) > 1:
            cycles.append(path.copy())
            return
        for neighbor in graph[current]:
            if neighbor not in visited or (neighbor == target and len(path) > 1):
                if neighbor == target and len(path) > 1:
                    cycles.append(path + [neighbor])
                elif neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, target, path, visited)
                    path.pop()
                    visited.remove(neighbor)

    # Remove the edge (v1, v2) from consideration
    original_v1_neighbors = graph[v1].copy()
    original_v2_neighbors = graph[v2].copy()
    if v2 in graph[v1]:
        graph[v1].remove(v2)
    if v1 in graph[v2]:
        graph[v2].remove(v1)

    # Find all paths from v1 to v2
    dfs(v1, v2, [v1], {v1})

    # Restore edges
    graph[v1] = original_v1_neighbors
    graph[v2] = original_v2_neighbors

    return cycles


def main():
    # Create a point configuration where multiple cycles are possible
    # when a particular edge closes

    # Points arranged to create an interesting structure
    points = np.array([
        [0.0, 0.0],   # 0
        [1.0, 0.0],   # 1
        [2.0, 0.0],   # 2
        [0.5, 0.8],   # 3
        [1.5, 0.8],   # 4
        [1.0, 1.5],   # 5
        [0.0, 1.0],   # 6
        [2.0, 1.0],   # 7
    ])

    # Define edges that exist BEFORE the closing edge
    # This creates a structure where edge (0,2) would close multiple possible cycles
    existing_edges = [
        (0, 1), (1, 2),  # bottom
        (0, 3), (3, 5), (5, 4), (4, 2),  # outer path through top
        (1, 3), (1, 4),  # inner connections creating multiple cycle options
        (0, 6), (6, 5),  # left side
        (2, 7), (7, 5),  # right side
    ]

    # The closing edge
    closing_edge = (0, 2)

    # Build graph
    graph = defaultdict(list)
    for a, b in existing_edges:
        graph[a].append(b)
        graph[b].append(a)

    # Find all cycles through the closing edge
    all_cycles = find_all_simple_cycles(graph, closing_edge[0], closing_edge[1], max_length=8)

    # Compute areas for each cycle
    cycle_data = []
    for cycle_indices in all_cycles:
        vertices = points[cycle_indices]
        area = polygon_area(vertices)
        cycle_data.append({
            'indices': cycle_indices,
            'vertices': vertices,
            'area': area,
            'n_vertices': len(cycle_indices)
        })

    # Sort by number of vertices (what BFS would prioritize)
    cycle_data.sort(key=lambda x: x['n_vertices'])

    print(f"Found {len(cycle_data)} distinct cycles through edge {closing_edge}")
    print("\nCycles sorted by vertex count (BFS priority):")
    print("-" * 50)
    for i, c in enumerate(cycle_data):
        print(f"Cycle {i+1}: vertices={c['n_vertices']}, area={c['area']:.4f}, path={c['indices']}")

    # Also sort by area
    by_area = sorted(cycle_data, key=lambda x: x['area'])
    print("\nCycles sorted by area (smallest first):")
    print("-" * 50)
    for i, c in enumerate(by_area):
        print(f"Cycle {i+1}: area={c['area']:.4f}, vertices={c['n_vertices']}, path={c['indices']}")

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Plot 1: Full graph structure
    ax = axes[0]
    # Draw existing edges
    edge_lines = [[points[a], points[b]] for a, b in existing_edges]
    lc = LineCollection(edge_lines, colors='gray', linewidths=2, alpha=0.7)
    ax.add_collection(lc)

    # Draw closing edge (dashed)
    ax.plot([points[closing_edge[0], 0], points[closing_edge[1], 0]],
           [points[closing_edge[0], 1], points[closing_edge[1], 1]],
           'r--', linewidth=3, label='Closing edge')

    # Draw points
    ax.scatter(points[:, 0], points[:, 1], s=150, c='darkblue', zorder=5)
    for i, (x, y) in enumerate(points):
        ax.annotate(str(i), (x, y), fontsize=12, fontweight='bold',
                   ha='center', va='center', color='white')

    ax.set_title('Graph Structure\n(red dashed = closing edge)', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(-0.3, 1.8)

    # Plot remaining cycles
    colors = plt.cm.Set1(np.linspace(0, 1, len(cycle_data)))

    for idx, (ax, cycle) in enumerate(zip(axes[1:], cycle_data[:5])):
        # Draw all edges
        edge_lines = [[points[a], points[b]] for a, b in existing_edges]
        lc = LineCollection(edge_lines, colors='lightgray', linewidths=1.5, alpha=0.5)
        ax.add_collection(lc)

        # Draw closing edge
        ax.plot([points[closing_edge[0], 0], points[closing_edge[1], 0]],
               [points[closing_edge[0], 1], points[closing_edge[1], 1]],
               'gray', linewidth=1.5, alpha=0.5)

        # Draw cycle polygon (filled)
        polygon = MplPolygon(cycle['vertices'], alpha=0.4, facecolor=colors[idx],
                            edgecolor='darkred', linewidth=2.5)
        ax.add_patch(polygon)

        # Draw cycle edges
        for i in range(len(cycle['vertices'])):
            j = (i + 1) % len(cycle['vertices'])
            ax.plot([cycle['vertices'][i, 0], cycle['vertices'][j, 0]],
                   [cycle['vertices'][i, 1], cycle['vertices'][j, 1]],
                   color='darkred', linewidth=2.5, zorder=4)

        # Draw points
        ax.scatter(points[:, 0], points[:, 1], s=100, c='gray', alpha=0.5, zorder=3)
        ax.scatter(cycle['vertices'][:, 0], cycle['vertices'][:, 1],
                  s=150, c='darkblue', zorder=5, edgecolors='red', linewidths=2)

        # Label cycle vertices
        for i, pt_idx in enumerate(cycle['indices']):
            ax.annotate(str(pt_idx), (points[pt_idx, 0], points[pt_idx, 1]),
                       fontsize=11, fontweight='bold', ha='center', va='center', color='white')

        # Determine if this is BFS choice or min area
        is_bfs = (idx == 0)  # First when sorted by vertex count
        is_min_area = (cycle == by_area[0])

        labels = []
        if is_bfs:
            labels.append("BFS choice")
        if is_min_area:
            labels.append("Min area")

        title = f"Cycle: {cycle['indices']}\n"
        title += f"Vertices: {cycle['n_vertices']}, Area: {cycle['area']:.4f}"
        if labels:
            title += f"\n({', '.join(labels)})"

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.3, 2.3)
        ax.set_ylim(-0.3, 1.8)

    plt.suptitle('Multiple Valid Cycles for Same H1 Feature\n'
                 'BFS finds minimum vertex count, not minimum area',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = "../../outputs/report_figures/cycle_choice_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
