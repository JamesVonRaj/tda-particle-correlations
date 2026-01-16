# Persistent Homology Analysis Summary

## Overview

This document summarizes the persistent homology computation performed on the ensemble dataset of Thomas cluster processes with varying cluster standard deviation.

**Date**: October 23, 2025  
**Total Samples Processed**: 200 (50 per configuration)  
**Computation Method**: Vietoris-Rips filtration with max_edge_length = 3.0

---

## Results Summary

### H0 Features (Connected Components)

H0 features represent connected components that merge as the filtration scale increases. These primarily capture when isolated points connect to nearby clusters.

| Configuration | Mean H0 Features | Std | Min | Max |
|---------------|------------------|-----|-----|-----|
| stddev_0.2    | 659.2            | 140.5 | 316 | 963 |
| stddev_0.4    | 620.8            | 174.1 | 283 | 1147 |
| stddev_0.6    | 631.7            | 142.1 | 348 | 973 |
| stddev_1.0    | 609.0            | 121.4 | 317 | 861 |

**Observation**: The number of H0 features is approximately equal to the number of points minus one (for each connected component that eventually merges into the main component).

### H1 Features (Loops/Holes)

H1 features represent topological loops or holes in the point cloud. These capture the void structure and clustering patterns.

| Configuration | Mean H1 Features | Std | Min | Max |
|---------------|------------------|-----|-----|-----|
| stddev_0.2    | 97.6             | 25.8 | 39  | 155 |
| stddev_0.4    | 106.8            | 37.4 | 42  | 226 |
| stddev_0.6    | 121.9            | 36.4 | 48  | 232 |
| stddev_1.0    | 137.8            | 34.1 | 62  | 213 |

**Key Finding**: As cluster standard deviation increases (looser clusters), the average number of H1 features increases by approximately 41% (from 97.6 to 137.8). This reflects the larger-scale topological structures that emerge when clusters are more spread out.

---

## Physical Interpretation

### Cluster Standard Deviation vs. Topology

1. **Tight Clusters (r₀ = 0.2)**:
   - Fewer, smaller loops
   - Loops form between tightly packed clusters
   - Shorter persistence lifetimes for H1 features

2. **Loose Clusters (r₀ = 1.0)**:
   - More, larger loops
   - Loops capture void space between overlapping clusters
   - Longer persistence lifetimes for prominent features

### Persistence Lifetimes

The lifetime of a feature (death - birth) indicates its "significance":
- **Short lifetimes**: Noise or transient features
- **Long lifetimes**: Robust topological structures

For sample stddev_0.6, sample_000:
- H0 mean lifetime: 0.229 (std: 0.152)
- H1 mean lifetime: 0.081 (std: 0.119)

This suggests H1 features are generally more ephemeral than H0 features, but the most persistent H1 features (max lifetime ~1.18) represent significant void structures in the point pattern.

---

## Data Organization

All persistence data is stored alongside the original point data:

```
ensemble_data/
├── persistence_summary.json          # Overall statistics
├── stddev_X.X/
│   ├── persistence_metadata.json     # Configuration statistics
│   ├── sample_000.npy               # Original points
│   ├── sample_000_persistence.npy   # H0 and H1 diagrams
│   └── ...
```

Each `*_persistence.npy` file contains:
- `h0`: Array of shape (K, 2) with [birth, death] pairs for H0 features
- `h1`: Array of shape (L, 2) with [birth, death] pairs for H1 features
- `n_points`: Number of points in the original sample
- `max_edge_length`: Filtration parameter (3.0)

---

## Computational Details

### Method
- **Filtration**: Vietoris-Rips complex
- **Max Edge Length**: 3.0 (explores structures up to radius 1.5)
- **Max Dimension**: 2 (computes up to 2-simplices/triangles)
- **Computed Homology**: H0 and H1 only

### Performance
- Average computation time: ~0.4 seconds per sample
- Total computation time: ~80 seconds for 200 samples
- Memory efficient: Only stores birth-death pairs, not full complexes

### Software
- GUDHI library for persistent homology computation
- NumPy for data storage and manipulation
- Python 3 for orchestration

---

## Usage for Further Analysis

### Statistical Analysis
```python
import numpy as np
from load_persistence_example import load_all_persistence_for_config

# Load all persistence for a configuration
all_ph, metadata = load_all_persistence_for_config('stddev_0.6')

# Extract all H1 lifetimes
all_h1_lifetimes = []
for ph_data in all_ph:
    h1 = ph_data['h1']
    lifetimes = h1[:, 1] - h1[:, 0]
    all_h1_lifetimes.extend(lifetimes)

# Analyze distribution
print(f"Mean H1 lifetime: {np.mean(all_h1_lifetimes):.4f}")
print(f"Median H1 lifetime: {np.median(all_h1_lifetimes):.4f}")
```

### Comparing Configurations
```python
import json

with open('ensemble_data/persistence_summary.json', 'r') as f:
    summary = json.load(f)

for config, data in summary['configurations'].items():
    print(f"{config}: {data['h1_features']['mean']:.1f} H1 features on average")
```

### Machine Learning Features
Persistence diagrams can be used as input features for ML:
- Persistence images
- Persistence landscapes
- Bottleneck/Wasserstein distances between diagrams

---

## Next Steps

Potential analyses using this data:

1. **Statistical hypothesis testing**: Test if H1 feature distributions differ significantly across configurations

2. **Persistence images**: Convert diagrams to fixed-size images for ML

3. **Bottleneck distance**: Compute pairwise distances between persistence diagrams

4. **Feature importance**: Identify which topological features best distinguish cluster configurations

5. **Theoretical comparison**: Compare empirical persistence with theoretical predictions for Thomas processes

6. **Scale analysis**: Recompute persistence with different max_edge_length values to explore multi-scale behavior

---

## References

- **Thomas Process**: A Poisson cluster process with Gaussian offspring distribution
- **Vietoris-Rips Filtration**: Standard method for computing persistence from point clouds
- **Persistent Homology**: Carlsson, G. (2009). Topology and data. *Bulletin of the American Mathematical Society*, 46(2), 255-308.

---

*Generated: October 23, 2025*  
*For questions or issues, see the main ENSEMBLE_DATA_README.md*

