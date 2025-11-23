"""
QPS (Queries Per Second) benchmarking utilities.

This module helps measure the inference throughput of models,
which is critical for online serving.
"""

import time
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


def measure_qps(
    model: nn.Module,
    input_shape: Tuple,
    batch_size: int = 1,
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = "cpu",
    use_amp: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Measure QPS (queries per second) for a model.

    Args:
        model: Model to benchmark
        input_shape: Shape of input tensor (excluding batch dimension)
        batch_size: Batch size for inference
        num_warmup: Number of warmup iterations
        num_iterations: Number of measurement iterations
        device: Device to run on
        use_amp: Use automatic mixed precision (FP16)
        verbose: Print results

    Returns:
        Dictionary with QPS metrics
    """
    model = model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_shape, device=device)

    # Warmup
    if verbose:
        print(f"Warming up for {num_warmup} iterations...")

    with torch.no_grad():
        for _ in range(num_warmup):
            if use_amp:
                with torch.cuda.amp.autocast():
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)

    # Synchronize if using GPU
    if device != "cpu":
        torch.cuda.synchronize()

    # Benchmark
    if verbose:
        print(f"Benchmarking for {num_iterations} iterations...")

    latencies = []

    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.perf_counter()

            if use_amp:
                with torch.cuda.amp.autocast():
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)

            # Synchronize if using GPU
            if device != "cpu":
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            latencies.append(end_time - start_time)

    # Compute statistics
    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    median_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)

    # Compute QPS
    qps = batch_size / mean_latency
    median_qps = batch_size / median_latency

    results = {
        'mean_latency_ms': mean_latency * 1000,
        'std_latency_ms': std_latency * 1000,
        'median_latency_ms': median_latency * 1000,
        'p95_latency_ms': p95_latency * 1000,
        'p99_latency_ms': p99_latency * 1000,
        'mean_qps': qps,
        'median_qps': median_qps,
        'batch_size': batch_size,
        'device': device,
        'use_amp': use_amp,
    }

    if verbose:
        print("\nQPS Benchmark Results:")
        print(f"  Mean Latency: {results['mean_latency_ms']:.2f} ms")
        print(f"  Median Latency: {results['median_latency_ms']:.2f} ms")
        print(f"  P95 Latency: {results['p95_latency_ms']:.2f} ms")
        print(f"  P99 Latency: {results['p99_latency_ms']:.2f} ms")
        print(f"  Mean QPS: {results['mean_qps']:.2f}")
        print(f"  Median QPS: {results['median_qps']:.2f}")

    return results


def compare_qps(
    models: Dict[str, nn.Module],
    input_shape: Tuple,
    batch_size: int = 1,
    num_iterations: int = 100,
    device: str = "cpu",
    verbose: bool = True
) -> Dict:
    """
    Compare QPS across multiple models.

    Args:
        models: Dictionary mapping model names to models
        input_shape: Input shape
        batch_size: Batch size
        num_iterations: Number of iterations
        device: Device
        verbose: Print comparison

    Returns:
        Dictionary with comparison results
    """
    results = {}

    for name, model in models.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Benchmarking: {name}")
            print(f"{'='*60}")

        results[name] = measure_qps(
            model=model,
            input_shape=input_shape,
            batch_size=batch_size,
            num_iterations=num_iterations,
            device=device,
            verbose=verbose
        )

    # Compute relative performance
    if verbose and len(results) > 1:
        print(f"\n{'='*60}")
        print("QPS Comparison")
        print(f"{'='*60}")

        # Use first model as baseline
        baseline_name = list(results.keys())[0]
        baseline_qps = results[baseline_name]['mean_qps']

        print(f"{'Model':<30} {'QPS':<15} {'Latency (ms)':<15} {'Relative QPS':<15}")
        print("-" * 75)

        for name, result in results.items():
            qps = result['mean_qps']
            latency = result['mean_latency_ms']
            relative = qps / baseline_qps

            print(f"{name:<30} {qps:<15.2f} {latency:<15.2f} {relative:<15.2f}x")

    return results


def profile_layer_latency(
    model: nn.Module,
    input_shape: Tuple,
    device: str = "cpu",
    num_iterations: int = 50
) -> Dict[str, float]:
    """
    Profile latency contribution of each layer.

    Args:
        model: Model to profile
        input_shape: Input shape
        device: Device
        num_iterations: Number of iterations

    Returns:
        Dictionary mapping layer names to average latency (ms)
    """
    model = model.to(device)
    model.eval()

    layer_latencies = {}

    # Hook to measure layer latency
    def get_timing_hook(name):
        def hook(module, input, output):
            start = time.perf_counter()
            # The actual computation happened before this hook
            # So we measure by running the layer again
            with torch.no_grad():
                _ = module(input[0])
            if device != "cpu":
                torch.cuda.synchronize()
            end = time.perf_counter()

            if name not in layer_latencies:
                layer_latencies[name] = []
            layer_latencies[name].append((end - start) * 1000)

        return hook

    # Register hooks
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            handle = module.register_forward_hook(get_timing_hook(name))
            handles.append(handle)

    # Run inference
    dummy_input = torch.randn(1, *input_shape, device=device)

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Aggregate
    avg_latencies = {
        name: sum(times) / len(times)
        for name, times in layer_latencies.items()
    }

    return avg_latencies


def measure_throughput(
    model: nn.Module,
    input_shape: Tuple,
    batch_sizes: list = [1, 4, 8, 16, 32],
    device: str = "cpu",
    duration: float = 5.0
) -> Dict:
    """
    Measure throughput at different batch sizes.

    Args:
        model: Model to benchmark
        input_shape: Input shape
        batch_sizes: List of batch sizes to test
        device: Device
        duration: Duration to run each batch size (seconds)

    Returns:
        Dictionary with throughput results
    """
    model = model.to(device)
    model.eval()

    results = {}

    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, *input_shape, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        if device != "cpu":
            torch.cuda.synchronize()

        # Measure
        start_time = time.perf_counter()
        num_batches = 0

        with torch.no_grad():
            while time.perf_counter() - start_time < duration:
                _ = model(dummy_input)
                num_batches += 1

                if device != "cpu":
                    torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        throughput = (num_batches * batch_size) / elapsed

        results[batch_size] = {
            'throughput': throughput,
            'latency_ms': (elapsed / num_batches) * 1000,
            'num_batches': num_batches,
        }

    return results
