# PCP Folder Reorganization Summary

## Date: October 24, 2025

## Overview

The PCP folder has been reorganized from a flat structure with all Python scripts in the root directory to a clear, hierarchical structure organized by function.

---

## Before (Cluttered)

```
PCP/
├── generate_ensembles.py
├── poisson_cluster.py
├── compute_ensemble_persistence.py
├── tda_utils.py
├── analyze_death_distributions.py
├── plot_PCP.py
├── plot_PCP_stddev_variation.py
├── plot_persistence_diagrams_with_pointcounts.py
├── load_ensemble_example.py
├── load_persistence_example.py
├── PH.py
├── ensemble_data/
├── persistence_diagrams_output/
├── cluster_stddev_variation/
├── TDA_Poisson_Cluster_Analysis/
├── *.png (scattered figures)
├── *.md (scattered documentation)
└── __pycache__/
```

**Problems:**
- ❌ All scripts mixed together in one directory
- ❌ No clear organization by purpose
- ❌ Hard to find specific functionality
- ❌ Output figures scattered everywhere
- ❌ Documentation files mixed with code

---

## After (Organized)

```
PCP/
├── README.md                       ← New main documentation
│
├── scripts/                        ← All code organized by purpose
│   ├── 1_data_generation/
│   │   ├── poisson_cluster.py
│   │   └── generate_ensembles.py
│   ├── 2_persistence_computation/
│   │   ├── tda_utils.py
│   │   └── compute_ensemble_persistence.py
│   ├── 3_analysis/
│   │   └── analyze_death_distributions.py
│   └── 4_visualization/
│       ├── plot_PCP.py
│       ├── plot_PCP_stddev_variation.py
│       └── plot_persistence_diagrams_with_pointcounts.py
│
├── examples/                       ← Example usage scripts
│   ├── load_ensemble_example.py
│   └── load_persistence_example.py
│
├── data/                           ← All data in one place
│   └── ensemble_data/
│       ├── README.json
│       ├── persistence_summary.json
│       └── stddev_*/
│
├── outputs/                        ← All outputs organized
│   ├── figures/
│   │   ├── death_distributions/
│   │   ├── persistence_diagrams/
│   │   ├── parameter_comparisons/
│   │   └── *.png (example figures)
│   └── analysis_results/
│
├── docs/                           ← All documentation together
│   ├── ENSEMBLE_DATA_README.md
│   ├── PERSISTENT_HOMOLOGY_SUMMARY.md
│   ├── H0_DEATH_DISTRIBUTION_ANALYSIS.md
│   └── PERSISTENCE_DIAGRAMS_README.md
│
└── archive/                        ← Old/deprecated code
    ├── PH.py
    └── TDA_Poisson_Cluster_Analysis/
```

**Benefits:**
- ✅ Clear workflow: 1 → 2 → 3 → 4
- ✅ Easy to find specific functionality
- ✅ All outputs in one place
- ✅ All documentation together
- ✅ Examples clearly separated
- ✅ Old code archived, not deleted

---

## File Movements

### Scripts by Function

| Old Location | New Location | Purpose |
|-------------|--------------|---------|
| `poisson_cluster.py` | `scripts/1_data_generation/` | Core simulation |
| `generate_ensembles.py` | `scripts/1_data_generation/` | Generate data |
| `tda_utils.py` | `scripts/2_persistence_computation/` | TDA utilities |
| `compute_ensemble_persistence.py` | `scripts/2_persistence_computation/` | Compute PH |
| `analyze_death_distributions.py` | `scripts/3_analysis/` | Statistical analysis |
| `plot_PCP.py` | `scripts/4_visualization/` | Visualization |
| `plot_PCP_stddev_variation.py` | `scripts/4_visualization/` | Visualization |
| `plot_persistence_diagrams_with_pointcounts.py` | `scripts/4_visualization/` | Visualization |

### Examples

| Old Location | New Location |
|-------------|--------------|
| `load_ensemble_example.py` | `examples/` |
| `load_persistence_example.py` | `examples/` |

### Data

| Old Location | New Location |
|-------------|--------------|
| `ensemble_data/` | `data/ensemble_data/` |

### Outputs

| Old Location | New Location |
|-------------|--------------|
| `h0_death_distribution_*.png` | `outputs/figures/death_distributions/` |
| `persistence_diagrams_output/*` | `outputs/figures/persistence_diagrams/` |
| `cluster_stddev_variation/*` | `outputs/figures/parameter_comparisons/` |
| `cluster_*.png` | `outputs/figures/parameter_comparisons/` |
| Other `*.png` | `outputs/figures/` |

### Documentation

| Old Location | New Location |
|-------------|--------------|
| All `*.md` files | `docs/` |
| New `README.md` | `README.md` (root) |

### Archive

| Old Location | New Location |
|-------------|--------------|
| `PH.py` | `archive/PH.py` |
| `TDA_Poisson_Cluster_Analysis/` | `archive/TDA_Poisson_Cluster_Analysis/` |

---

## Code Updates

All scripts have been updated with correct relative paths:

### Import Paths
- Scripts in `scripts/4_visualization/` import from `../1_data_generation/`
- Examples import from `../data/ensemble_data/`

### Data Paths
- All scripts now use `../../data/ensemble_data/` (from script directories)
- Examples use `../data/ensemble_data/` (from examples directory)

### Output Paths
- Analysis scripts output to `../../outputs/figures/death_distributions/`
- Visualization scripts output to `../../outputs/figures/*/`
- Examples output to `../outputs/figures/`

---

## Workflow Clarity

The numbered script folders make the workflow obvious:

```
1. scripts/1_data_generation/
   → Generate ensemble data

2. scripts/2_persistence_computation/
   → Compute persistent homology

3. scripts/3_analysis/
   → Analyze statistical properties

4. scripts/4_visualization/
   → Create publication figures
```

---

## Quick Start (After Reorganization)

```bash
# Generate data
cd scripts/1_data_generation
python generate_ensembles.py

# Compute persistence
cd ../2_persistence_computation
python compute_ensemble_persistence.py

# Analyze distributions
cd ../3_analysis
python analyze_death_distributions.py

# Create visualizations
cd ../4_visualization
python plot_persistence_diagrams_with_pointcounts.py
```

---

## Benefits

1. **Clarity**: Immediately obvious what each folder contains
2. **Workflow**: Numbered folders show the analysis pipeline
3. **Maintainability**: Easy to add new scripts to appropriate folders
4. **Documentation**: All docs in one place, easy to reference
5. **Outputs**: All figures organized by type
6. **Preservation**: Old code archived, not lost

---

## Notes

- All relative paths have been updated in all scripts
- All scripts tested to ensure they work in new locations
- No data was modified, only reorganized
- Archive contains old code for reference but is not part of active workflow

---

## Testing

To verify everything works:

```bash
# Test examples
cd examples
python load_ensemble_example.py
python load_persistence_example.py

# Test analysis (if data exists)
cd ../scripts/3_analysis
python analyze_death_distributions.py

# Test visualization
cd ../4_visualization
python plot_persistence_diagrams_with_pointcounts.py
```

All outputs will be created in `outputs/figures/` subdirectories.

---

**Summary**: The folder is now well-organized, easy to navigate, and ready for continued development!


