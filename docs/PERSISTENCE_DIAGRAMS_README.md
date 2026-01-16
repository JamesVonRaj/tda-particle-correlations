# Persistence Diagrams with Point-Count-Dependent Marker Sizes

## Overview

This visualization shows persistence diagrams for one sample from each of the four cluster configurations, where the **marker size is proportional to the number of points** involved in each topological feature (from the 3rd column of the persistence data).

**Generated**: October 23, 2025  
**Sample Used**: Sample 0 from each configuration

---

## Files Generated

### Main Comparison Plots

1. **`persistence_diagrams_grid_sample_000.png`** (577 KB)
   - **2×2 grid** showing all four configurations side-by-side
   - Marker sizes scale with number of points in each feature
   - Best for: Direct visual comparison across configurations
   - Blue circles = H0 (connected components)
   - Red triangles = H1 (loops)

2. **`persistence_diagrams_colored_sample_000.png`** (494 KB)
   - **2×2 grid** with color-coded point counts
   - Color intensity indicates number of points
   - Separate colorbars for H0 (blue) and H1 (red)
   - Best for: Precise reading of point count values

### Individual Configuration Plots

3. **`persistence_diagram_stddev_0.2_sample_000.png`** (208 KB)
   - Large detailed view of very tight clusters (r₀ = 0.2)
   - 689 H0 features, 89 H1 features
   
4. **`persistence_diagram_stddev_0.4_sample_000.png`** (224 KB)
   - Large detailed view of tight clusters (r₀ = 0.4)
   - 714 H0 features, 130 H1 features

5. **`persistence_diagram_stddev_0.6_sample_000.png`** (252 KB)
   - Large detailed view of baseline clusters (r₀ = 0.6)
   - 806 H0 features, 182 H1 features

6. **`persistence_diagram_stddev_1.0_sample_000.png`** (247 KB)
   - Large detailed view of loose clusters (r₀ = 1.0)
   - 664 H0 features, 162 H1 features

---

## Interpretation

### Point Counts in Features

From the analysis of Sample 0:

**H0 Features (Connected Components)**:
- Point counts range: 2-3
- Most features involve **2 points** merging (one isolated point joining a component)
- Some features involve **3 points** (a small component merging with another)

**H1 Features (Loops)**:
- Point counts: consistently **3**
- This makes sense: the death of an H1 feature occurs when a triangle (3 points) fills the loop

### Visual Patterns

1. **Tight Clusters (r₀ = 0.2)**:
   - Fewer H1 features (89)
   - Features cluster near the birth = death diagonal
   - Small lifetimes → rapid merging

2. **Loose Clusters (r₀ = 1.0)**:
   - More H1 features (162)
   - Features spread further from diagonal
   - Larger lifetimes → more persistent structures

### Marker Size Encoding

The marker sizes follow a **square root scaling**:
```python
marker_size = 10 + 50 * sqrt(num_points - min_points + 1)
```

This ensures:
- Minimum size of 10 (for visibility)
- Progressive scaling that's perceptually balanced
- Larger features are clearly distinguishable

---

## Usage

### Generate diagrams for a different sample:

Edit the `SAMPLE_ID` variable in the script:

```python
SAMPLE_ID = 5  # Use sample 5 instead of sample 0
```

Then run:
```bash
python plot_persistence_diagrams_with_pointcounts.py
```

### Customize visualization:

Key parameters in the script:

```python
# Scale the axes
max_scale_global = 1.5  # Focus on [0, 1.5] range

# Adjust marker size scaling
marker_sizes = 10 + 50 * np.sqrt(sizes - sizes.min() + 1)
#              ^^   ^^  adjust these multipliers
#              base size  scale factor
```

---

## Key Observations

### Distribution Differences

Looking at the grid comparison:

1. **Number of H1 features increases** with cluster standard deviation:
   - r₀ = 0.2: 89 loops
   - r₀ = 1.0: 162 loops (82% increase!)

2. **Persistence (distance from diagonal)** increases with looser clusters:
   - Tight clusters: most features die quickly
   - Loose clusters: more persistent features visible

3. **Point count information** (marker size) shows:
   - Most topological features involve **2-3 points**
   - This is consistent with the local nature of Rips filtration
   - Death simplices are minimal (edges for H0, triangles for H1)

### Comparison with Death Time Distribution

These diagrams complement the death time distribution analysis:
- The **vertical spread** in the persistence diagram corresponds to the **spread in death times**
- Tighter clusters → more concentration near diagonal → narrow death time distribution
- Looser clusters → more spread → broader death time distribution

---

## Technical Notes

### Why Point Counts are Small (2-3)?

The point counts represent the vertices in the **death simplex**, not all points contributing to the feature:

- **H0 death**: When two components merge via an edge → 2 vertices
- **H1 death**: When a loop is filled by a triangle → 3 vertices

This is different from the total number of points in the entire connected component or loop cycle. It's the **local simplicial event** that causes the feature to die.

### Alternative Interpretations

For a fuller picture of feature "size", one could compute:
- Total vertices in the connected component (H0)
- Total vertices in the loop cycle (H1)

This would require tracking persistence with cochains, which we explored in `tda_utils.py`.

---

## Script

**`plot_persistence_diagrams_with_pointcounts.py`**:
- Loads persistence data for one sample from each configuration
- Creates three types of visualizations:
  1. 2×2 grid with size-scaled markers
  2. Individual large diagrams with size legend
  3. 2×2 grid with color-coded point counts
- All plots saved to `persistence_diagrams_output/` directory

---

## See Also

- `H0_DEATH_DISTRIBUTION_ANALYSIS.md` - Statistical analysis of death times
- `ENSEMBLE_DATA_README.md` - Overview of the ensemble dataset
- `PERSISTENT_HOMOLOGY_SUMMARY.md` - General persistence computation details

---

*Generated: October 23, 2025*

