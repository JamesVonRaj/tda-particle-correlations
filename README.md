# Poisson Cluster Process (PCP) Analysis

## Overview

This project analyzes Thomas Cluster Processes (a type of Poisson Cluster Process) using Topological Data Analysis (TDA) and persistent homology. The analysis focuses on how cluster standard deviation affects the topological structure of point patterns.

**Key Features:**
- Generate ensemble datasets of point processes with varying parameters
- Compute persistent homology for all samples
- Analyze H0 death time distributions
- Create comprehensive visualizations

---

## Installation & Setup

### Prerequisites
- **pyenv** for Python version management
- **Python 3.12.0** (managed via pyenv)

### Quick Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd PCP

# Run automated setup script
bash setup.sh
```

This will:
1. Check/install Python 3.12.0
2. Create a virtual environment (`pcp-analysis-env`)
3. Install all dependencies from `requirements.txt`

### Manual Setup

```bash
# Install Python 3.12.0 (if needed)
pyenv install 3.12.0

# Create and activate virtual environment
pyenv virtualenv 3.12.0 pcp-analysis-env
pyenv local pcp-analysis-env

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**For detailed setup instructions, see [SETUP.md](SETUP.md)**

---

## Directory Structure

```
PCP/
├── README.md (this file)
│
├── scripts/
│   ├── 1_data_generation/          # Generate point process data
│   │   ├── poisson_cluster.py      # Core simulation functions
│   │   └── generate_ensembles.py   # Generate 50 samples × 4 configs
│   │
│   ├── 2_persistence_computation/  # Compute persistent homology
│   │   ├── tda_utils.py           # TDA utility functions
│   │   └── compute_ensemble_persistence.py  # Compute PH for all samples
│   │
│   ├── 3_analysis/                 # Statistical analysis
│   │   └── analyze_death_distributions.py   # H0 death time analysis
│   │
│   └── 4_visualization/            # Create plots and figures
│       ├── plot_PCP.py             # Plot different parameter variations
│       ├── plot_PCP_stddev_variation.py  # Focus on stddev variations
│       └── plot_persistence_diagrams_with_pointcounts.py  # PH diagrams
│
├── examples/                       # Example usage scripts
│   ├── load_ensemble_example.py    # Load point data
│   └── load_persistence_example.py # Load persistence data
│
├── data/
│   └── ensemble_data/              # Generated ensemble data (200 samples)
│       ├── README.json             # Data metadata
│       ├── persistence_summary.json # PH computation summary
│       ├── stddev_0.2/             # Very tight clusters (50 samples)
│       │   ├── metadata.json
│       │   ├── persistence_metadata.json
│       │   ├── sample_000.npy & sample_000_persistence.npy
│       │   └── ... (50 samples total)
│       ├── stddev_0.4/             # Tight clusters (50 samples)
│       ├── stddev_0.6/             # Baseline clusters (50 samples)
│       └── stddev_1.0/             # Loose clusters (50 samples)
│
├── outputs/
│   ├── figures/
│   │   ├── death_distributions/    # H0 death time distribution plots
│   │   ├── persistence_diagrams/   # Individual persistence diagrams
│   │   ├── parameter_comparisons/  # Parameter variation comparisons
│   │   ├── ensemble_visualization_example.png
│   │   ├── persistence_diagrams_comparison.png
│   │   └── points_with_persistence.png
│   └── analysis_results/           # Computed analysis results
│
├── docs/                           # Documentation
│   ├── ENSEMBLE_DATA_README.md     # Ensemble data documentation
│   ├── PERSISTENT_HOMOLOGY_SUMMARY.md  # PH computation details
│   ├── H0_DEATH_DISTRIBUTION_ANALYSIS.md  # Death time analysis
│   └── PERSISTENCE_DIAGRAMS_README.md  # Persistence diagram docs
│
└── archive/                        # Old/deprecated files
    ├── PH.py                      # Legacy persistence script
    └── TDA_Poisson_Cluster_Analysis/  # Old analysis folder

```

---

## Quick Start

### 1. Generate Ensemble Data

```bash
cd scripts/1_data_generation
python generate_ensembles.py
```

This creates 200 samples (50 per configuration) in `data/ensemble_data/`.

### 2. Compute Persistent Homology

```bash
cd scripts/2_persistence_computation
python compute_ensemble_persistence.py
```

This computes H0 and H1 persistence for all 200 samples (~80 seconds total).

### 3. Analyze Death Time Distributions

```bash
cd scripts/3_analysis
python analyze_death_distributions.py
```

Creates 5 visualizations in `outputs/figures/death_distributions/`.

### 4. Create Persistence Diagrams

```bash
cd scripts/4_visualization
python plot_persistence_diagrams_with_pointcounts.py
```

Creates persistence diagrams with point-count-dependent marker sizes in `outputs/figures/persistence_diagrams/`.

---

## Key Results

### Ensemble Data
- **200 total samples** (50 per configuration)
- **4 configurations**: r₀ ∈ {0.2, 0.4, 0.6, 1.0}
- **Fixed parameters**: λ_parent = 0.1, λ_daughter = 30
- **Window size**: 15 × 15

### Persistent Homology
- **126,033 H0 features** analyzed across all samples
- **Point counts included** in persistence data (3rd column)
- **Format**: [birth, death, num_points]

### H0 Death Time Distributions

| Configuration | Mean Death | H1 Features |
|---------------|------------|-------------|
| stddev_0.2    | 0.130      | 97.6        |
| stddev_0.4    | 0.201      | 106.8       |
| stddev_0.6    | 0.257      | 121.9       |
| stddev_1.0    | 0.333      | 137.8       |

**Key Finding**: As cluster standard deviation increases, death times increase by 156% and H1 features increase by 41%!

---

## Usage Examples

### Load a Sample

```python
import numpy as np

# Load point data
data = np.load('data/ensemble_data/stddev_0.6/sample_000.npy', allow_pickle=True).item()
points = data['points']
parents = data['parents']

# Load persistence data
ph_data = np.load('data/ensemble_data/stddev_0.6/sample_000_persistence.npy', allow_pickle=True).item()
h0 = ph_data['h0']  # [birth, death, num_points]
h1 = ph_data['h1']  # [birth, death, num_points]
```

### Using Helper Functions

```python
import sys
sys.path.append('examples')
from load_persistence_example import load_sample_with_persistence

# Load sample with all data
data = load_sample_with_persistence('stddev_0.6', sample_id=0)
print(f"Points: {len(data['points'])}")
print(f"H0 features: {len(data['h0'])}")
print(f"H1 features: {len(data['h1'])}")
```

---

## Workflow

### Data Generation → Computation → Analysis → Visualization

1. **Generate** ensemble data (once)
2. **Compute** persistent homology (once)
3. **Analyze** distributions, statistics
4. **Visualize** results in multiple ways

All intermediate results are saved, so you can skip to any step!

---

## Documentation

Comprehensive documentation in `docs/`:

- **ENSEMBLE_DATA_README.md**: Complete guide to the ensemble dataset
- **PERSISTENT_HOMOLOGY_SUMMARY.md**: PH computation methodology and results
- **H0_DEATH_DISTRIBUTION_ANALYSIS.md**: Statistical analysis of death times
- **PERSISTENCE_DIAGRAMS_README.md**: Guide to persistence diagram visualizations

---

## Dependencies

**Python Version**: 3.12.0 (managed with pyenv)

**Required Packages** (see `requirements.txt`):
- `numpy==1.26.3` - Array operations
- `scipy==1.14.1` - Statistical tests and KDE
- `matplotlib==3.9.2` - Plotting
- `seaborn==0.13.2` - Statistical visualizations
- `gudhi==3.10.1` - Persistent homology computation
- `ripser==0.6.12` - Fast persistence computation
- `scikit-learn==1.5.2` - Machine learning utilities
- `tqdm==4.66.6` - Progress bars

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## File Naming Conventions

- **`sample_XXX.npy`**: Point cloud data (original)
- **`sample_XXX_persistence.npy`**: Persistence data for that sample
- **`metadata.json`**: Configuration-specific metadata
- **`persistence_metadata.json`**: Persistence computation metadata

---

## Citation

If you use this code or analysis, please cite:
- Thomas Process model
- GUDHI library for persistent homology
- Relevant topological data analysis papers

---

## Notes

- All scripts use **relative paths** from their directory location
- Data and outputs are stored at the project root level
- Examples can be run from the `examples/` directory
- Old/deprecated code is in `archive/` for reference

---

**Created**: October 2025  
**Author**: Data analysis pipeline for Thomas Cluster Processes  
**License**: See project license


