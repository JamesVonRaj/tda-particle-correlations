#!/usr/bin/env python
"""
Benchmark script for TDA persistence computation optimizations.

Compares:
1. Sequential vs parallel processing for ensemble computation
2. Union-Find vs fastcluster for H0 computation
"""

import numpy as np
import time
import sys
import os
from pathlib import Path

# Add pipeline module to path
sys.path.insert(0, str(Path(__file__).parent / 'pipeline' / 'core'))
import persistence


def generate_test_points(n_points, box_size=None, seed=None):
    """Generate random test points."""
    if seed is not None:
        np.random.seed(seed)
    if box_size is None:
        box_size = np.sqrt(n_points)  # Unit intensity
    return np.random.rand(n_points, 2) * box_size


def benchmark_h0_methods(n_points_list, n_trials=5, max_edge_factor=0.5):
    """
    Benchmark Union-Find vs fastcluster for H0 computation.

    Parameters
    ----------
    n_points_list : list
        List of point counts to test
    n_trials : int
        Number of trials per configuration
    max_edge_factor : float
        max_edge_length = box_size * factor
    """
    print("=" * 70)
    print("H0 Method Comparison: Union-Find vs Fastcluster")
    print("=" * 70)
    print()

    if not persistence.HAS_FASTCLUSTER:
        print("WARNING: fastcluster not installed, skipping comparison")
        return

    results = []

    for n_points in n_points_list:
        box_size = np.sqrt(n_points)
        max_edge = box_size * max_edge_factor

        uf_times = []
        fc_times = []

        for trial in range(n_trials):
            points = generate_test_points(n_points, box_size, seed=trial)

            # Union-Find
            start = time.perf_counter()
            h0_uf, edges_uf = persistence.compute_h0_unionfind(points, max_edge)
            uf_time = time.perf_counter() - start
            uf_times.append(uf_time)

            # Fastcluster
            start = time.perf_counter()
            h0_fc, edges_fc = persistence.compute_h0_fastcluster(points, max_edge)
            fc_time = time.perf_counter() - start
            fc_times.append(fc_time)

        uf_mean = np.mean(uf_times)
        fc_mean = np.mean(fc_times)
        speedup = uf_mean / fc_mean if fc_mean > 0 else 0

        results.append({
            'n_points': n_points,
            'uf_mean': uf_mean,
            'fc_mean': fc_mean,
            'speedup': speedup,
            'n_features': len(h0_uf)
        })

        print(f"n_points={n_points:5d}: Union-Find={uf_mean*1000:7.2f}ms, "
              f"Fastcluster={fc_mean*1000:7.2f}ms, "
              f"Speedup={speedup:5.2f}x, "
              f"H0 features={len(h0_uf)}")

    print()
    return results


def benchmark_full_persistence(n_points_list, n_trials=3, max_edge_factor=0.5):
    """
    Benchmark full persistence computation (H0 + H1).
    """
    print("=" * 70)
    print("Full Persistence Computation (H0 + H1)")
    print("=" * 70)
    print()

    results = []

    for n_points in n_points_list:
        box_size = np.sqrt(n_points)
        max_edge = box_size * max_edge_factor

        uf_times = []
        fc_times = []

        for trial in range(n_trials):
            points = generate_test_points(n_points, box_size, seed=trial)

            # Union-Find
            start = time.perf_counter()
            result_uf = persistence.compute_persistence(points, max_edge, use_fastcluster=False)
            uf_time = time.perf_counter() - start
            uf_times.append(uf_time)

            # Fastcluster (if available)
            if persistence.HAS_FASTCLUSTER:
                start = time.perf_counter()
                result_fc = persistence.compute_persistence(points, max_edge, use_fastcluster=True)
                fc_time = time.perf_counter() - start
                fc_times.append(fc_time)

        uf_mean = np.mean(uf_times)

        if persistence.HAS_FASTCLUSTER:
            fc_mean = np.mean(fc_times)
            speedup = uf_mean / fc_mean if fc_mean > 0 else 0
            print(f"n_points={n_points:5d}: Union-Find={uf_mean:7.3f}s, "
                  f"Fastcluster={fc_mean:7.3f}s, "
                  f"Speedup={speedup:5.2f}x, "
                  f"H0={len(result_uf['h0'])}, H1={len(result_uf['h1'])}")
        else:
            print(f"n_points={n_points:5d}: Time={uf_mean:7.3f}s, "
                  f"H0={len(result_uf['h0'])}, H1={len(result_uf['h1'])}")

        results.append({
            'n_points': n_points,
            'uf_mean': uf_mean,
            'fc_mean': fc_mean if persistence.HAS_FASTCLUSTER else None,
            'h0_features': len(result_uf['h0']),
            'h1_features': len(result_uf['h1'])
        })

    print()
    return results


def benchmark_parallel_processing(n_samples=20, n_points=224, n_trials=3):
    """
    Benchmark sequential vs parallel ensemble processing.

    Uses realistic parameters: n_points=224 (typical for unit intensity normalization
    with target_n_points=50000).
    """
    print("=" * 70)
    print("Parallel Processing Benchmark")
    print("=" * 70)
    print(f"Samples: {n_samples}, Points per sample: {n_points}")
    print()

    if not persistence.HAS_JOBLIB:
        print("WARNING: joblib not installed, skipping parallel benchmark")
        return

    from joblib import Parallel, delayed

    box_size = np.sqrt(n_points)
    max_edge = box_size * 0.5

    # Generate test samples
    samples = [generate_test_points(n_points, box_size, seed=i) for i in range(n_samples)]

    # Sequential processing
    seq_times = []
    for trial in range(n_trials):
        start = time.perf_counter()
        results_seq = [persistence.compute_persistence(pts, max_edge) for pts in samples]
        seq_time = time.perf_counter() - start
        seq_times.append(seq_time)

    seq_mean = np.mean(seq_times)

    # Parallel processing with different worker counts
    cpu_count = os.cpu_count() or 1
    worker_counts = [2, 4, 8, cpu_count] if cpu_count >= 8 else [2, 4, cpu_count]
    worker_counts = sorted(set([w for w in worker_counts if w <= cpu_count]))

    print(f"Sequential: {seq_mean:.3f}s ({seq_mean/n_samples*1000:.1f}ms per sample)")
    print()

    for n_workers in worker_counts:
        par_times = []
        for trial in range(n_trials):
            start = time.perf_counter()
            results_par = Parallel(n_jobs=n_workers)(
                delayed(persistence.compute_persistence)(pts, max_edge) for pts in samples
            )
            par_time = time.perf_counter() - start
            par_times.append(par_time)

        par_mean = np.mean(par_times)
        speedup = seq_mean / par_mean
        efficiency = speedup / n_workers * 100

        print(f"Parallel ({n_workers:2d} workers): {par_mean:.3f}s, "
              f"Speedup={speedup:.2f}x, "
              f"Efficiency={efficiency:.0f}%")

    print()


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print(" TDA Persistence Computation - Optimization Benchmarks")
    print("=" * 70 + "\n")

    # Suppress scipy version warning
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    # Print system info
    print(f"CPU count: {os.cpu_count()}")
    print(f"joblib available: {persistence.HAS_JOBLIB}")
    print(f"fastcluster available: {persistence.HAS_FASTCLUSTER}")
    print()

    # Benchmark H0 methods
    print("Benchmarking H0 methods (Union-Find vs Fastcluster)...")
    print("Testing with increasing point counts...\n")
    h0_results = benchmark_h0_methods(
        n_points_list=[100, 200, 500, 1000, 2000],
        n_trials=5
    )

    # Benchmark full persistence
    print("Benchmarking full persistence computation...")
    print("Testing with realistic and larger point counts...\n")
    full_results = benchmark_full_persistence(
        n_points_list=[100, 224, 500, 1000],
        n_trials=3
    )

    # Benchmark parallel processing
    print("Benchmarking parallel processing...")
    print("Simulating ensemble computation...\n")
    benchmark_parallel_processing(n_samples=20, n_points=224, n_trials=3)

    # Summary
    print("=" * 70)
    print(" Summary")
    print("=" * 70)
    print()
    print("H0 Method Comparison:")
    print("  - Fastcluster provides modest speedup for H0-only computation")
    print("  - Benefit increases with larger point counts")
    print("  - H0 is typically a small fraction of total computation time")
    print()
    print("Parallel Processing:")
    print("  - Provides significant speedup for ensemble computation")
    print("  - Near-linear scaling with worker count (embarrassingly parallel)")
    print("  - Recommended for processing multiple samples")
    print()


if __name__ == '__main__':
    main()
