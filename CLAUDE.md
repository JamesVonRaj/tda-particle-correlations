# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a scientific Python project for analyzing Thomas Cluster Processes (a type of Poisson Cluster Process) using Topological Data Analysis (TDA) and Persistent Homology. The research question focuses on how cluster standard deviation (r₀) affects the topological structure of point patterns.

## Setup Commands

```bash
# Automated setup (uses pyenv)
bash setup.sh

# Manual setup
pyenv install 3.12.0
pyenv virtualenv 3.12.0 pcp-analysis-env
pyenv local pcp-analysis-env
pip install -r requirements.txt
```

## Running the Pipeline

The project follows a 4-stage pipeline that must be run in order:

```bash
# Stage 1: Generate ensemble data (creates 1000 samples: 20 configs × 50 each)
python scripts/1_data_generation/generate_ensembles.py

# Stage 2: Compute persistent homology (~80 seconds)
python scripts/2_persistence_computation/compute_ensemble_persistence.py

# Stage 3: Run statistical analysis
python scripts/3_analysis/analyze_death_distributions.py
python scripts/3_analysis/analyze_birth_distributions.py
python scripts/3_analysis/analyze_lifetime_distributions.py

# Stage 4: Generate visualizations
python scripts/4_visualization/plot_persistence_diagrams_with_pointcounts.py
python scripts/4_visualization/plot_death_distributions_grid.py
```

## Architecture

### Pipeline Stages

1. **Data Generation** (`scripts/1_data_generation/`): Simulates Thomas cluster processes with varying σ values
2. **Persistence Computation** (`scripts/2_persistence_computation/`): Computes H0 (connected components) and H1 (loops) using Vietoris-Rips filtration
3. **Analysis** (`scripts/3_analysis/`): Statistical analysis of persistence features
4. **Visualization** (`scripts/4_visualization/`): Creates figures and plots

### Core Components

- `poisson_cluster.py`: `simulate_poisson_cluster_process()` generates Thomas process point clouds
- `tda_utils.py`: `compute_rips_persistence_with_point_counts()` computes persistence with point tracking using custom Union-Find for H0 and GUDHI for H1
- Fixed parameters: λ_parent=0.1, λ_daughter=30, window_size=15×15, max_edge_length=3.0

### Data Formats

**Sample files** (`data/ensemble_data/stddev_*/sample_XXX.npy`):
```python
{'points': np.ndarray (N, 2), 'parents': np.ndarray (M, 2), 'sample_id': int}
```

**Persistence files** (`sample_XXX_persistence.npy`):
```python
{'h0': np.ndarray (K, 3), 'h1': np.ndarray (L, 3)}  # [birth, death, num_points]
```

### Path Conventions

All scripts use relative paths from their directory location:
- Data: `../../data/ensemble_data/` (from stage scripts)
- Outputs: `../../outputs/figures/` (from stage scripts)

### Example Usage

```python
# Load point data
data = np.load('data/ensemble_data/stddev_0.6/sample_000.npy', allow_pickle=True).item()
points = data['points']

# Load persistence data
ph = np.load('data/ensemble_data/stddev_0.6/sample_000_persistence.npy', allow_pickle=True).item()
h0, h1 = ph['h0'], ph['h1']  # Each row: [birth, death, num_points]
```

## Key Dependencies

- **gudhi**: Rips complex and persistence computation
- **numpy/scipy**: Array operations and statistics
- **matplotlib/seaborn**: Visualization
