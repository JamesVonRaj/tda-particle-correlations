#!/usr/bin/env python
"""
Visualize the Shoelace Formula for Polygon Area Computation

Creates toy examples showing how the shoelace formula computes cycle areas,
allowing visual verification of the calculation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

def polygon_area(vertices):
    """
    Compute area of a polygon using the shoelace formula.

    For vertices [(x1,y1), (x2,y2), ..., (xn,yn)]:
    Area = 0.5 * |sum_{i=0}^{n-1} (x_i * y_{i+1} - x_{i+1} * y_i)|

    Returns the area and the individual cross-product terms for visualization.
    """
    n = len(vertices)
    if n < 3:
        return 0.0, []

    terms = []
    for i in range(n):
        j = (i + 1) % n
        term = vertices[i, 0] * vertices[j, 1] - vertices[j, 0] * vertices[i, 1]
        terms.append({
            'i': i, 'j': j,
            'xi': vertices[i, 0], 'yi': vertices[i, 1],
            'xj': vertices[j, 0], 'yj': vertices[j, 1],
            'term': term
        })

    area = abs(sum(t['term'] for t in terms)) / 2.0
    return area, terms


def plot_polygon_with_area(ax, vertices, title, color='steelblue'):
    """Plot a polygon with its vertices, edges, and computed area."""
    n = len(vertices)

    # Plot filled polygon
    polygon = MplPolygon(vertices, alpha=0.3, facecolor=color, edgecolor=color, linewidth=2)
    ax.add_patch(polygon)

    # Plot vertices with labels
    for i, (x, y) in enumerate(vertices):
        ax.plot(x, y, 'o', color='darkblue', markersize=12, zorder=5)
        ax.annotate(f'$P_{i}$\n({x:.1f}, {y:.1f})', (x, y),
                   textcoords='offset points', xytext=(10, 10),
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Compute area
    area, terms = polygon_area(vertices)

    # Build formula text
    formula_lines = ["Shoelace Formula:", ""]
    formula_lines.append(r"$A = \frac{1}{2}|$" + " + ".join([f"$(x_{t['i']} y_{t['j']} - x_{t['j']} y_{t['i']})$" for t in terms]) + "$|$")
    formula_lines.append("")
    formula_lines.append(r"$A = \frac{1}{2}|$" + " + ".join([f"$({t['xi']:.1f} \\times {t['yj']:.1f} - {t['xj']:.1f} \\times {t['yi']:.1f})$" for t in terms]) + "$|$")
    formula_lines.append("")
    formula_lines.append(r"$A = \frac{1}{2}|$" + " + ".join([f"${t['term']:.2f}$" for t in terms]) + "$|$")
    formula_lines.append("")
    formula_lines.append(f"$A = \\frac{{1}}{{2}} \\times |{sum(t['term'] for t in terms):.2f}| = {area:.2f}$")

    ax.set_title(f"{title}\nArea = {area:.2f}", fontsize=12, fontweight='bold')

    # Set equal aspect and limits
    ax.set_aspect('equal')
    margin = 0.5
    ax.set_xlim(vertices[:, 0].min() - margin, vertices[:, 0].max() + margin)
    ax.set_ylim(vertices[:, 1].min() - margin, vertices[:, 1].max() + margin)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return area, terms


def main():
    # Create figure with multiple toy examples
    fig = plt.figure(figsize=(16, 14))

    # Example 1: Unit square (area should be 1)
    ax1 = fig.add_subplot(2, 3, 1)
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    area1, terms1 = plot_polygon_with_area(ax1, square, "Unit Square (expected: 1.0)", 'steelblue')

    # Example 2: Right triangle (area should be 0.5 * base * height = 0.5 * 2 * 2 = 2)
    ax2 = fig.add_subplot(2, 3, 2)
    triangle = np.array([[0, 0], [2, 0], [0, 2]], dtype=float)
    area2, terms2 = plot_polygon_with_area(ax2, triangle, "Right Triangle (expected: 2.0)", 'forestgreen')

    # Example 3: Regular pentagon (approximate)
    ax3 = fig.add_subplot(2, 3, 3)
    n_sides = 5
    angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False) - np.pi/2
    pentagon = np.column_stack([np.cos(angles), np.sin(angles)])
    # Area of regular pentagon with circumradius 1: (5/2) * sin(72°) ≈ 2.378
    area3, terms3 = plot_polygon_with_area(ax3, pentagon, f"Regular Pentagon (r=1)", 'darkorange')

    # Example 4: Irregular quadrilateral
    ax4 = fig.add_subplot(2, 3, 4)
    quad = np.array([[0, 0], [3, 0], [4, 2], [1, 3]], dtype=float)
    area4, terms4 = plot_polygon_with_area(ax4, quad, "Irregular Quadrilateral", 'crimson')

    # Example 5: L-shape (non-convex) - but shoelace works for simple polygons
    ax5 = fig.add_subplot(2, 3, 5)
    l_shape = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 3], [0, 3]], dtype=float)
    area5, terms5 = plot_polygon_with_area(ax5, l_shape, "L-Shape (non-convex)", 'purple')

    # Example 6: Small cycle from a point cloud (simulating what we compute in TDA)
    ax6 = fig.add_subplot(2, 3, 6)
    # Simulate a cycle that might form in a point cloud
    np.random.seed(42)
    center = np.array([2, 2])
    n_pts = 6
    angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False) + np.random.uniform(-0.2, 0.2, n_pts)
    radii = 1.5 + np.random.uniform(-0.3, 0.3, n_pts)
    cycle_pts = center + np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    area6, terms6 = plot_polygon_with_area(ax6, cycle_pts, "Simulated TDA Cycle", 'teal')

    plt.suptitle("Shoelace Formula Verification\n" +
                 r"$A = \frac{1}{2} \left| \sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1} y_i) \right|$",
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure
    output_path = "../../outputs/report_figures/shoelace_verification.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    # Also create a detailed breakdown figure for one example
    fig2, ax = plt.subplots(figsize=(12, 8))

    # Use the irregular quadrilateral for detailed breakdown
    vertices = np.array([[0, 0], [3, 0], [4, 2], [1, 3]], dtype=float)
    n = len(vertices)

    # Plot polygon
    polygon = MplPolygon(vertices, alpha=0.2, facecolor='crimson', edgecolor='crimson', linewidth=2)
    ax.add_patch(polygon)

    # Plot vertices
    for i, (x, y) in enumerate(vertices):
        ax.plot(x, y, 'o', color='darkblue', markersize=15, zorder=5)
        ax.annotate(f'$P_{i}$ = ({x:.0f}, {y:.0f})', (x, y),
                   textcoords='offset points', xytext=(15, 10),
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # Plot edges with arrows showing direction
    for i in range(n):
        j = (i + 1) % n
        mid_x = (vertices[i, 0] + vertices[j, 0]) / 2
        mid_y = (vertices[i, 1] + vertices[j, 1]) / 2
        dx = vertices[j, 0] - vertices[i, 0]
        dy = vertices[j, 1] - vertices[i, 1]
        ax.annotate('', xy=(vertices[j, 0], vertices[j, 1]),
                   xytext=(vertices[i, 0], vertices[i, 1]),
                   arrowprops=dict(arrowstyle='->', color='darkblue', lw=2))

    # Compute and display formula
    area, terms = polygon_area(vertices)

    # Create text box with step-by-step calculation
    calc_text = "Step-by-step Shoelace Calculation:\n\n"
    calc_text += "Formula: $A = \\frac{1}{2} |\\sum_{i} (x_i y_{i+1} - x_{i+1} y_i)|$\n\n"

    for t in terms:
        calc_text += f"Edge {t['i']}→{t['j']}: ({t['xi']:.0f} × {t['yj']:.0f}) - ({t['xj']:.0f} × {t['yi']:.0f}) = {t['xi']*t['yj']:.0f} - {t['xj']*t['yi']:.0f} = {t['term']:.0f}\n"

    calc_text += f"\nSum = {sum(t['term'] for t in terms):.0f}\n"
    calc_text += f"Area = |{sum(t['term'] for t in terms):.0f}| / 2 = {area:.1f}"

    ax.text(0.98, 0.98, calc_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
           family='monospace')

    ax.set_title("Detailed Shoelace Formula Example\nIrregular Quadrilateral", fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, 4)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)

    output_path2 = "../../outputs/report_figures/shoelace_detailed.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path2}")

    plt.show()

    # Print verification
    print("\n" + "="*60)
    print("Area Verification Summary")
    print("="*60)
    print(f"Unit Square:        {area1:.4f} (expected: 1.0)")
    print(f"Right Triangle:     {area2:.4f} (expected: 2.0)")
    print(f"Regular Pentagon:   {area3:.4f} (expected: ~2.378)")
    print(f"Irregular Quad:     {area4:.4f}")
    print(f"L-Shape:            {area5:.4f} (expected: 5.0 = 2×3 - 1×2)")
    print(f"TDA Cycle:          {area6:.4f}")


if __name__ == "__main__":
    main()
