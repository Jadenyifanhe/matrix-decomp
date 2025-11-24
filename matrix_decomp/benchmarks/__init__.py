"""
Benchmarking utilities for measuring performance metrics.

This module provides tools for measuring:
- QPS (Queries Per Second): Inference throughput
- FLOPs: Computational cost
- Memory: Parameter and activation memory
- Compression ratios: Size reduction metrics
"""

from .qps_benchmark import (
    measure_qps,
    compare_qps,
    profile_layer_latency,
    measure_throughput,
)
from .flops_counter import (
    count_flops,
    compare_flops,
    count_layer_flops,
    estimate_memory_usage,
    compute_compression_ratio,
)

__all__ = [
    # QPS benchmarking
    "measure_qps",
    "compare_qps",
    "profile_layer_latency",
    "measure_throughput",
    # FLOPS counting
    "count_flops",
    "compare_flops",
    "count_layer_flops",
    "estimate_memory_usage",
    "compute_compression_ratio",
]
