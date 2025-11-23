"""
Basic tests to verify the framework works correctly.
"""

import torch
import torch.nn as nn
from matrix_decomp.decomposition import SVDDecomposer, LoRAAdapter, FusedSVDDecomposer


def test_svd_decomposition():
    """Test SVD decomposition and reconstruction."""
    print("Testing SVD decomposition...")

    # Create a random weight matrix
    weight = torch.randn(100, 200)

    # Decompose
    decomposer = SVDDecomposer(rank=50)
    U, S, Vt = decomposer.decompose(weight)

    # Check shapes
    assert U.shape == (100, 50), f"U shape mismatch: {U.shape}"
    assert S.shape == (50,), f"S shape mismatch: {S.shape}"
    assert Vt.shape == (50, 200), f"Vt shape mismatch: {Vt.shape}"

    # Reconstruct
    reconstructed = decomposer.reconstruct(U, S, Vt)
    assert reconstructed.shape == weight.shape, "Reconstructed shape mismatch"

    # Check error
    error = decomposer.compute_reconstruction_error(weight, reconstructed, metric='relative')
    assert error < 1.0, f"Reconstruction error too high: {error}"

    print(f"  ✓ SVD works! Error: {error:.6f}")


def test_lora_adapter():
    """Test LoRA adapter."""
    print("Testing LoRA adapter...")

    # Create weights
    large_weight = torch.randn(100, 200)
    small_weight = torch.randn(100, 200)

    # Create adapter
    adapter = LoRAAdapter(rank=32, alpha=16)

    # Decompose
    A, B = adapter.decompose(large_weight, reference_weight=small_weight)

    # Check shapes
    assert A.shape == (32, 200), f"A shape mismatch: {A.shape}"
    assert B.shape == (100, 32), f"B shape mismatch: {B.shape}"

    # Reconstruct
    delta_W = adapter.reconstruct(A, B)
    assert delta_W.shape == large_weight.shape, "Delta shape mismatch"

    print(f"  ✓ LoRA works! Compression: {large_weight.numel() / (A.numel() + B.numel()):.2f}x")


def test_fused_svd():
    """Test Fused SVD."""
    print("Testing Fused SVD...")

    weight = torch.randn(100, 200)

    decomposer = FusedSVDDecomposer(rank=50)
    U_fused, Vt_fused = decomposer.decompose(weight)

    # Check shapes
    assert U_fused.shape == (100, 50), f"U_fused shape mismatch: {U_fused.shape}"
    assert Vt_fused.shape == (50, 200), f"Vt_fused shape mismatch: {Vt_fused.shape}"

    # Reconstruct
    reconstructed = decomposer.reconstruct(U_fused, Vt_fused)
    assert reconstructed.shape == weight.shape, "Reconstructed shape mismatch"

    print("  ✓ Fused SVD works!")


def test_lora_layer():
    """Test LoRA layer module."""
    print("Testing LoRA layer module...")

    adapter = LoRAAdapter(rank=32, alpha=16)

    # Create layer
    layer = adapter.create_lora_layer(
        in_features=128,
        out_features=256,
        bias=True
    )

    # Test forward
    x = torch.randn(4, 128)
    y = layer(x)

    assert y.shape == (4, 256), f"Output shape mismatch: {y.shape}"

    # Test disable/enable
    layer.disable_adapter()
    y_disabled = layer(x)

    layer.enable_adapter()
    y_enabled = layer(x)

    # Should be different when adapter is enabled
    assert not torch.allclose(y_disabled, y_enabled), "Adapter has no effect"

    print("  ✓ LoRA layer works!")


def test_dimension_adaptation():
    """Test dimension adaptation."""
    print("Testing dimension adaptation...")

    from matrix_decomp.transfer import adapt_dimensions

    # Create matrix
    matrix = torch.randn(100, 200)

    # Adapt to smaller
    adapted = adapt_dimensions(matrix, target_rows=50, target_cols=100, method='truncate')
    assert adapted.shape == (50, 100), f"Adapted shape mismatch: {adapted.shape}"

    # Adapt to larger
    adapted = adapt_dimensions(matrix, target_rows=150, target_cols=250, method='pad')
    assert adapted.shape == (150, 250), f"Padded shape mismatch: {adapted.shape}"

    print("  ✓ Dimension adaptation works!")


def main():
    """Run all tests."""
    print("="*80)
    print("Running Basic Tests")
    print("="*80)

    tests = [
        test_svd_decomposition,
        test_lora_adapter,
        test_fused_svd,
        test_lora_layer,
        test_dimension_adaptation,
    ]

    failed = []

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ✗ {test.__name__} failed: {e}")
            failed.append(test.__name__)

    print("\n" + "="*80)
    if not failed:
        print("✓ All tests passed!")
    else:
        print(f"✗ {len(failed)} test(s) failed:")
        for name in failed:
            print(f"  - {name}")
    print("="*80)

    return len(failed) == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
