# H0 Death Time Distribution Analysis

## Overview

This analysis examines the distribution of H0 death times (the scale at which connected components merge) across the four cluster configurations in the ensemble dataset. The death time is a critical topological feature that characterizes the spatial structure of point patterns.

**Date**: October 23, 2025  
**Total H0 Features Analyzed**: 126,033 (across 200 samples)

---

## Key Findings

### Summary Statistics

| Configuration | Total Features | Mean Death | Median Death | Std Dev | Max Death |
|---------------|----------------|------------|--------------|---------|-----------|
| stddev_0.2    | 32,960         | 0.1301     | 0.0857       | 0.2498  | 2.9882    |
| stddev_0.4    | 31,039         | 0.2012     | 0.1617       | 0.2080  | 2.9730    |
| stddev_0.6    | 31,584         | 0.2565     | 0.2194       | 0.1937  | 2.9752    |
| stddev_1.0    | 30,450         | 0.3325     | 0.2992       | 0.2010  | 2.8540    |

### Key Observations

1. **Systematic Shift**: As cluster standard deviation increases from 0.2 to 1.0, the mean death time increases **2.56×** (from 0.130 to 0.333).

2. **Distribution Shape**: 
   - Tighter clusters (stddev_0.2) show a **strong peak near zero** with a long tail
   - Looser clusters (stddev_1.0) show a **broader, more uniform distribution**

3. **Statistical Significance**: All pairwise comparisons between configurations are **highly significant** (p < 0.001, Kolmogorov-Smirnov test), indicating that cluster standard deviation has a measurable effect on the topological structure.

---

## Physical Interpretation

### What H0 Death Times Tell Us

The death time of an H0 feature represents the **filtration radius at which an isolated point or small component connects to a larger component**. 

- **Low death times** → Points are close together and merge quickly
- **High death times** → Points are isolated and remain separate until larger scales

### Configuration-Specific Behavior

#### Tight Clusters (r₀ = 0.2)
- **Mean death: 0.130**
- Points within clusters are very close → merge at small scales
- Inter-cluster gaps are large → some features persist to larger scales
- **Result**: Sharply peaked distribution with long tail

#### Loose Clusters (r₀ = 1.0)
- **Mean death: 0.333**
- Points within clusters are more spread out → merge at larger scales
- Clusters overlap more → less distinction between intra- and inter-cluster merging
- **Result**: Broader, more gradual distribution

---

## Statistical Analysis

### Kolmogorov-Smirnov Tests

All distributions are significantly different from each other:

| Comparison               | KS Statistic | p-value    | Interpretation                    |
|--------------------------|--------------|------------|-----------------------------------|
| stddev_0.2 vs stddev_0.4 | 0.3976       | < 0.001*** | Moderate difference               |
| stddev_0.2 vs stddev_0.6 | 0.5491       | < 0.001*** | Large difference                  |
| stddev_0.2 vs stddev_1.0 | 0.6727       | < 0.001*** | Very large difference             |
| stddev_0.4 vs stddev_0.6 | 0.2001       | < 0.001*** | Small-moderate difference         |
| stddev_0.4 vs stddev_1.0 | 0.3899       | < 0.001*** | Moderate difference               |
| stddev_0.6 vs stddev_1.0 | 0.2095       | < 0.001*** | Small-moderate difference         |

**Note**: The KS statistic represents the maximum difference between the cumulative distribution functions. Values > 0.5 indicate very different distributions.

---

## Visualizations Generated

### 1. Overlapping Histograms (`h0_death_distribution_overlapping.png`)
- Shows all four distributions on the same plot with transparency
- **Best for**: Direct visual comparison of distribution shapes
- Clearly shows the systematic shift to higher death times with increasing r₀

### 2. Separate Subplots (`h0_death_distribution_subplots.png`)
- Individual histograms with mean and median lines
- **Best for**: Detailed examination of each distribution
- Includes count information and key statistics

### 3. Cumulative Distribution Functions (`h0_death_distribution_cdf.png`)
- Shows cumulative probability vs. death time
- **Best for**: Understanding what fraction of features die by a given scale
- Example: ~70% of stddev_0.2 features die by radius 0.15, but only ~35% of stddev_1.0 features

### 4. Box Plots (`h0_death_distribution_boxplot.png`)
- Shows quartiles, whiskers, and outliers
- **Best for**: Comparing medians and spread across configurations
- Clearly shows increasing median and quartile ranges

### 5. Kernel Density Estimation (`h0_death_distribution_kde.png`)
- Smoothed probability density curves
- **Best for**: Publication-quality comparison of distribution shapes
- Shows the transition from sharp peak (tight) to broad plateau (loose)

---

## Usage Example

```python
import numpy as np
from analyze_death_distributions import load_all_h0_death_times, compute_statistics

# Load death times for a configuration
death_times = load_all_h0_death_times('stddev_0.6')

# Compute statistics
stats = compute_statistics(death_times)
print(f"Mean death time: {stats['mean']:.4f}")
print(f"Median death time: {stats['median']:.4f}")

# Find features that persist beyond a threshold
threshold = 0.5
persistent_fraction = np.sum(death_times > threshold) / len(death_times)
print(f"{persistent_fraction*100:.1f}% of features persist beyond radius {threshold}")
```

---

## Implications for Analysis

### For Classification
The death time distributions are **significantly different** across configurations, making them a strong candidate feature for:
- Machine learning classification of point patterns
- Automated detection of cluster tightness
- Distinguishing different point process parameters

### For Theoretical Comparison
The systematic trend (increasing death time with increasing r₀) matches theoretical expectations:
- Looser clusters → larger intra-cluster distances
- Larger distances → components merge at larger scales
- This validates the simulation and TDA pipeline

### For Further Analysis
Consider analyzing:
1. **Birth-death persistence**: Joint distribution of birth and death times
2. **H1 death times**: How loop structures differ across configurations
3. **Spatial distribution**: Do death times vary spatially within the window?
4. **Multi-scale structure**: Evidence of hierarchical clustering?

---

## Files

### Script
- **`analyze_death_distributions.py`**: Complete analysis script with all visualizations

### Generated Visualizations
- `h0_death_distribution_overlapping.png` - Overlapping histograms
- `h0_death_distribution_subplots.png` - Detailed subplots
- `h0_death_distribution_cdf.png` - Cumulative distributions
- `h0_death_distribution_boxplot.png` - Box plot comparison
- `h0_death_distribution_kde.png` - Kernel density estimation

---

## Conclusion

The H0 death time distributions provide a **clear topological signature** that distinguishes the four cluster configurations. The ensemble approach (50 samples × 4 configs = 200 samples) gives robust statistics with over 30,000 features per configuration.

The systematic increase in death times with cluster standard deviation confirms that **looser clusters produce point patterns with more persistent topological features**, which merge at larger filtration scales. This is consistent with the physical intuition about Thomas cluster processes.

---

*Generated: October 23, 2025*  
*For questions, see `analyze_death_distributions.py` or the main ENSEMBLE_DATA_README.md*

