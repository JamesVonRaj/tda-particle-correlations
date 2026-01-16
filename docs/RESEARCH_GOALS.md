# Research Goals: TDA for Particle Correlation Analysis

## Overview

This project investigates whether **Topological Data Analysis (TDA)** can provide meaningful insights into particle clustering behavior that complement or correlate with traditional particle correlation measures used in statistical mechanics.

## Research Question

**Can persistent homology features from TDA capture the same physical information as traditional particle correlation functions and percolation characteristics?**

Specifically, we aim to:

1. Compute topological features (persistence diagrams, Betti numbers) from particle point clouds.
2. Compare these features against established particle correlation measures (, , , ).
3. Determine if the filtration parameter  in TDA is mathematically equivalent to the clustering radius  in pair-connectedness functions.

---

## Comparison with Collaborators' Data

Our collaborators measure particle correlations using quantities derived from nearest-neighbor and void probability functions. We now have concrete definitions for these measures:

### 1. Nearest Neighbor & Void Functions

These measures describe the local density and spacing of the point process.

| Measure | Definition | Physical Interpretation |
| --- | --- | --- |
| **** | Void nearest-neighbor PDF | Probability that a particle lies at a distance between  and  from an **arbitrary point** in space.

 |
| **** | Particle nearest-neighbor PDF | Probability that a particle lies at a distance between  and  from **another point** in the configuration.

 |
| **** | Void empty probability | Probability of finding a spherical cavity of radius  centered at an **arbitrary point** that is empty of particles.

 |
| **** | Particle empty probability | Probability of finding a spherical cavity of radius  centered at a **particle** that is empty of other particles.

 |

**Mathematical Relationship:**
The  and  functions are related via derivatives:




### 2. Connectedness & Percolation Functions

These measures appear to map directly to **H0 (Connected Components)** and **Vietoris-Rips filtrations**.

| Measure | Definition | TDA Correspondence |
| --- | --- | --- |
| **** | <br>**Pair-connectedness function**: The conditional probability that two particles separated by distance  are in the *same cluster*.

 | Likely correlates with **H0 persistence**, where  acts as the filtration radius. |
| **** | <br>**Direct-connectedness function**: The probability that two points are connected by a path that does not involve nodes (cannot be broken by a single cut).

 | May relate to **H1 (loops)** or specific spectral clustering properties. |
| **** | <br>**Mean Cluster Size**: A percolation metric dependent on the box size .

 | Correlates to the size of connected components in the Rips complex. |

---

## Current Status

### What We Have Implemented

1. **Data Generation**: Thomas Cluster Process (Poisson Cluster Process) with varying cluster standard deviation ().
2. **Persistence Computation**:
* H0 (connected components) with point-count tracking.
* H1 (loops/voids) using Vietoris-Rips filtration.


3. **Analysis**: Death time distribution analysis.

### Updated Hypotheses based on Collaborator Data

The collaborator's data suggests that the "clusters created by putting spheres around each point with some radius"  is the exact mechanism of a Rips filtration.

* **Hypothesis 1:** The **Pair-Connectedness function ** is directly functionally related to the **H0 barcode** at filtration value .
* 
**Hypothesis 2:** The **Percolation probability **  corresponds to the birth time of the single giant connected component (infinite death) in H0.



---

*Last updated: 2026-01-09*
