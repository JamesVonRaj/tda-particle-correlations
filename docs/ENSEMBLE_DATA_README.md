# Ensemble Data for Cluster Standard Deviation Variations

## Overview

This directory contains ensemble data for analyzing the effect of cluster standard deviation (`cluster_std_dev` or `r0`) on Thomas process point patterns. The data was generated using `generate_ensembles.py`.

## Directory Structure

```
ensemble_data/
├── README.json                          # Overall metadata for all configurations
├── persistence_summary.json             # Summary of persistent homology computations
├── stddev_0.2/                          # Very tight clusters (r0 = 0.2)
│   ├── metadata.json                    # Configuration-specific metadata
│   ├── persistence_metadata.json        # Persistent homology statistics
│   ├── sample_000.npy                   # Sample 0 (points data)
│   ├── sample_000_persistence.npy       # Sample 0 (persistence data)
│   ├── sample_001.npy                   # Sample 1 (points data)
│   ├── sample_001_persistence.npy       # Sample 1 (persistence data)
│   ├── ...
│   ├── sample_049.npy                   # Sample 49 (points data)
│   └── sample_049_persistence.npy       # Sample 49 (persistence data)
├── stddev_0.4/                          # Tight clusters (r0 = 0.4)
│   ├── metadata.json
│   ├── persistence_metadata.json
│   ├── sample_000.npy
│   ├── sample_000_persistence.npy
│   ├── ...
│   └── sample_049_persistence.npy
├── stddev_0.6/                          # Baseline clusters (r0 = 0.6)
│   ├── metadata.json
│   ├── persistence_metadata.json
│   ├── sample_000.npy
│   ├── sample_000_persistence.npy
│   ├── ...
│   └── sample_049_persistence.npy
└── stddev_1.0/                          # Loose clusters (r0 = 1.0)
    ├── metadata.json
    ├── persistence_metadata.json
    ├── sample_000.npy
    ├── sample_000_persistence.npy
    ├── ...
    └── sample_049_persistence.npy
```

## Data Summary

- **Total samples**: 200 (50 per configuration)
- **Configurations**: 4 different `cluster_std_dev` values
- **Fixed parameters**: 
  - `lambda_parent` = 0.1
  - `lambda_daughter` = 30
  - `window_size` = 15 × 15
- **Variable parameter**: `cluster_std_dev` ∈ {0.2, 0.4, 0.6, 1.0}

## Configuration Details

| Folder Name  | cluster_std_dev (r0) | Description        | Mean Points | Std Points | Mean H0 | Mean H1 |
|--------------|----------------------|--------------------| -----------:|-----------:|--------:|--------:|
| stddev_0.2   | 0.2                  | Very tight clusters|       662.3 |      139.8 |   659.2 |    97.6 |
| stddev_0.4   | 0.4                  | Tight clusters     |       622.6 |      173.7 |   620.8 |   106.8 |
| stddev_0.6   | 0.6                  | Baseline clusters  |       632.9 |      141.9 |   631.7 |   121.9 |
| stddev_1.0   | 1.0                  | Loose clusters     |       610.0 |      121.4 |   609.0 |   137.8 |

**Key Observation**: As cluster standard deviation increases (looser clusters), the number of H1 features (loops) increases, reflecting the larger-scale topological structures in the point patterns.

## File Formats

### Sample Files (`.npy`)

Each `sample_XXX.npy` file contains a Python dictionary with the following keys:

- `'points'`: NumPy array of shape `(N, 2)` containing daughter point coordinates
- `'parents'`: NumPy array of shape `(M, 2)` containing parent point coordinates
- `'sample_id'`: Integer identifier for the sample (0-49)

### Persistence Files (`_persistence.npy`)

Each `sample_XXX_persistence.npy` file contains a Python dictionary with the following keys:

- `'h0'`: NumPy array of shape `(K, 3)` containing H0 (connected components) features: `[birth, death, num_points]`
- `'h1'`: NumPy array of shape `(L, 3)` containing H1 (loops) features: `[birth, death, num_points]`
- `'n_points'`: Number of points in the original sample
- `'max_edge_length'`: Maximum edge length used in Vietoris-Rips computation (default: 3.0)
- `'empty'`: Boolean flag indicating if the point set was empty

**Note**: 
- The persistence was computed using Vietoris-Rips filtration with `max_edge_length=3.0`, which means features are tracked up to radius 1.5.
- The `num_points` column indicates how many points are involved in each topological feature:
  - For H0: the number of vertices in the merging components
  - For H1: the number of vertices forming the loop structure

### Metadata Files (`.json`)

- **`README.json`**: Overall metadata describing all configurations and parameters
- **`metadata.json`**: Configuration-specific metadata (one per folder)
- **`persistence_summary.json`**: Overall summary of persistent homology computations
- **`persistence_metadata.json`**: Configuration-specific persistence statistics (one per folder)

## Usage Examples

### Load a Single Sample

```python
import numpy as np

# Load one sample
data = np.load('ensemble_data/stddev_0.2/sample_000.npy', allow_pickle=True).item()
points = data['points']      # Daughter points (N × 2 array)
parents = data['parents']    # Parent points (M × 2 array)
sample_id = data['sample_id'] # Sample identifier

print(f"Sample {sample_id} has {len(points)} daughter points and {len(parents)} parent points")
```

### Load All Samples for One Configuration

```python
import numpy as np
import glob

# Load all samples from stddev_0.6 configuration
samples = []
for sample_file in sorted(glob.glob('ensemble_data/stddev_0.6/sample_*.npy')):
    data = np.load(sample_file, allow_pickle=True).item()
    samples.append(data)

print(f"Loaded {len(samples)} samples")

# Calculate statistics
n_points = [len(s['points']) for s in samples]
print(f"Mean points per sample: {np.mean(n_points):.1f} ± {np.std(n_points):.1f}")
```

### Load Persistent Homology Data

```python
import numpy as np

# Load persistence data for a sample
ph_data = np.load('ensemble_data/stddev_0.2/sample_000_persistence.npy', allow_pickle=True).item()
h0 = ph_data['h0']  # H0 persistence diagram: [birth, death, num_points]
h1 = ph_data['h1']  # H1 persistence diagram: [birth, death, num_points]
n_points = ph_data['n_points']

print(f"Sample has {len(h0)} H0 features and {len(h1)} H1 features")

# Compute persistence lifetimes
h0_lifetimes = h0[:, 1] - h0[:, 0]  # death - birth
h1_lifetimes = h1[:, 1] - h1[:, 0]

# Extract point counts
h1_point_counts = h1[:, 2]  # number of points in each loop

print(f"Average H1 lifetime: {np.mean(h1_lifetimes):.4f}")
print(f"Average points per H1 feature: {np.mean(h1_point_counts):.1f}")
```

### Use the Helper Functions

For more convenient data loading, use the provided helper scripts:

**For point data only** (`load_ensemble_example.py`):
```python
from load_ensemble_example import load_sample, load_all_samples, get_configuration_names

# Load a single sample
data = load_sample('stddev_0.2', sample_id=0)

# Load all samples for a configuration
samples, metadata = load_all_samples('stddev_0.6')

# Get all configuration names
configs = get_configuration_names()
```

**For point data + persistence** (`load_persistence_example.py`):
```python
from load_persistence_example import load_sample_with_persistence, load_all_persistence_for_config

# Load sample with persistence data
data = load_sample_with_persistence('stddev_0.6', sample_id=0)
print(f"Points: {len(data['points'])}, H0: {len(data['h0'])}, H1: {len(data['h1'])}")

# Load all persistence for a configuration
all_persistence, metadata = load_all_persistence_for_config('stddev_0.6')
```

## Analysis Recommendations

This ensemble data is suitable for:

1. **Statistical analysis** of point pattern characteristics across different cluster sizes
2. **Topological data analysis** (TDA) using persistent homology
3. **Pair correlation function** g₂(r) estimation from empirical data
4. **Structure factor** S(k) estimation and comparison with theory
5. **Machine learning** training/testing for point pattern classification
6. **Uncertainty quantification** using ensemble statistics

## Scripts

### Data Generation
- **`generate_ensembles.py`**: Script used to generate the ensemble point process data
- **`compute_ensemble_persistence.py`**: Script used to compute persistent homology for all samples

### Visualization and Analysis
- **`plot_PCP_stddev_variation.py`**: Script for visualizing single instances with different parameters
- **`load_ensemble_example.py`**: Example demonstrating how to load and visualize the point data
- **`load_persistence_example.py`**: Example demonstrating how to load and analyze persistent homology data

### Utilities
- **`tda_utils.py`**: TDA utility functions for computing persistent homology
- **`poisson_cluster.py`**: Functions for simulating Thomas cluster processes

## Generated Files

Run the example scripts to generate visualizations:

**From `load_ensemble_example.py`**:
- `ensemble_visualization_example.png`: Example visualization showing one sample from each configuration

**From `load_persistence_example.py`**:
- `persistence_diagrams_comparison.png`: Persistence diagrams for all four configurations
- `points_with_persistence.png`: Side-by-side view of point cloud and persistence diagram

## Notes

- Each sample is an independent realization of the Thomas process with the specified parameters
- Parent points are randomly distributed according to a Poisson process with intensity `lambda_parent`
- Daughter points are distributed around each parent with Gaussian distribution (std = `cluster_std_dev`)
- All samples use the same simulation window: [0, 15] × [0, 15]
- The number of points varies between samples due to the stochastic nature of the process

## Citation

If you use this data in publications, please cite the Thomas process model and describe the generation parameters.

---

*Generated: October 24, 2025*

