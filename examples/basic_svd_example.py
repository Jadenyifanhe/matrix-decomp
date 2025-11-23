"""
Basic SVD Decomposition Example

This example demonstrates:
1. How to decompose a weight matrix using SVD
2. How to analyze singular values
3. How to reconstruct and measure error
4. How SVD affects model size and QPS
"""

import torch
import torch.nn as nn
from matrix_decomp.decomposition import SVDDecomposer
from matrix_decomp.benchmarks import measure_qps, count_flops


def create_simple_model(input_dim=1024, hidden_dim=2048, output_dim=512):
    """Create a simple feedforward model."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )


def main():
    print("="*80)
    print("SVD Decomposition Example")
    print("="*80)

    # Create a model
    model = create_simple_model()
    print(f"\nOriginal model created")

    # Get a weight matrix to decompose
    weight = model[0].weight.data  # First layer: [2048, 1024]
    print(f"Weight shape: {weight.shape}")

    # 1. Analyze singular values
    print("\n" + "-"*80)
    print("Step 1: Analyzing Singular Values")
    print("-"*80)

    decomposer = SVDDecomposer(rank=128)
    analysis = decomposer.analyze_singular_values(weight, plot=False)

    print(f"Number of singular values: {analysis['num_singular_values']}")
    print(f"Condition number: {analysis['condition_number']:.2f}")
    print(f"\nRanks needed for different energy levels:")
    for threshold, rank in analysis['ranks_for_energy'].items():
        print(f"  {threshold}: rank {rank}")

    # 2. Decompose with different ranks
    print("\n" + "-"*80)
    print("Step 2: Decomposing with Different Ranks")
    print("-"*80)

    ranks = [32, 64, 128, 256]

    for rank in ranks:
        decomposer = SVDDecomposer(rank=rank)

        # Decompose
        U, S, Vt = decomposer.decompose(weight)

        # Reconstruct
        reconstructed = decomposer.reconstruct(U, S, Vt)

        # Compute error
        error = decomposer.compute_reconstruction_error(weight, reconstructed, metric='relative')

        # Compute compression ratio
        original_params = weight.numel()
        compressed_params = U.numel() + S.numel() + Vt.numel()
        compression_ratio = original_params / compressed_params

        print(f"\nRank {rank}:")
        print(f"  Reconstruction error (relative): {error:.6f}")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Parameters: {original_params:,} -> {compressed_params:,}")

    # 3. Create decomposed model and compare QPS
    print("\n" + "-"*80)
    print("Step 3: Comparing QPS (Original vs SVD)")
    print("-"*80)

    # Original model
    original_model = create_simple_model()

    # Create SVD decomposed version
    svd_model = create_simple_model()
    decomposer = SVDDecomposer(rank=128)

    # Decompose first layer
    U, S, Vt = decomposer.decompose(svd_model[0].weight.data)
    # Note: For vanilla SVD, we need to replace with two sequential layers
    # For simplicity, we'll reconstruct here
    svd_model[0].weight.data = decomposer.reconstruct(U, S, Vt)

    # Benchmark
    input_shape = (1024,)

    print("\nOriginal Model:")
    original_qps = measure_qps(
        original_model,
        input_shape,
        batch_size=1,
        num_iterations=50,
        verbose=False
    )
    print(f"  QPS: {original_qps['mean_qps']:.2f}")
    print(f"  Latency: {original_qps['mean_latency_ms']:.2f} ms")

    print("\nSVD Model (reconstructed):")
    svd_qps = measure_qps(
        svd_model,
        input_shape,
        batch_size=1,
        num_iterations=50,
        verbose=False
    )
    print(f"  QPS: {svd_qps['mean_qps']:.2f}")
    print(f"  Latency: {svd_qps['mean_latency_ms']:.2f} ms")

    # Note: Reconstructed SVD should have similar QPS
    # The QPS degradation happens when using U @ S @ V^T as separate operations

    print("\n" + "="*80)
    print("Key Takeaways:")
    print("="*80)
    print("1. SVD provides mathematically optimal low-rank approximation")
    print("2. Higher rank = lower reconstruction error but less compression")
    print("3. Reconstructed SVD maintains similar QPS to original")
    print("4. For actual QPS improvement, use Fused SVD or LoRA instead")
    print("="*80)


if __name__ == "__main__":
    main()
