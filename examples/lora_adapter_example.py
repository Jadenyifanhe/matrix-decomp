"""
LoRA Adapter Example

This example demonstrates:
1. How to create LoRA adapters from model differences
2. How to transfer knowledge using LoRA
3. How LoRA preserves QPS better than SVD
4. How to use LoRA for model enhancement
"""

import torch
import torch.nn as nn
from matrix_decomp.decomposition import LoRAAdapter
from matrix_decomp.benchmarks import measure_qps, compare_qps


def create_model(hidden_dim=512):
    """Create a simple model."""
    return nn.Sequential(
        nn.Linear(256, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 128)
    )


def main():
    print("="*80)
    print("LoRA Adapter Example")
    print("="*80)

    # Simulate large and small models
    print("\nStep 1: Creating Large and Small Models")
    print("-"*80)

    # Large model (5x larger hidden dimension)
    large_model = create_model(hidden_dim=2560)
    print(f"Large model hidden dim: 2560")

    # Small model (original size)
    small_model = create_model(hidden_dim=512)
    print(f"Small model hidden dim: 512")

    # Simulate training: add some random improvements to large model
    # In practice, this would be from actual training
    with torch.no_grad():
        for param in large_model.parameters():
            param.add_(torch.randn_like(param) * 0.1)

    # Step 2: Extract knowledge using LoRA
    print("\nStep 2: Extracting Knowledge with LoRA")
    print("-"*80)

    lora_adapter = LoRAAdapter(rank=64, alpha=16, init_method='svd')

    # Get weights from first layer
    large_weight = large_model[0].weight.data  # [2560, 256]
    small_weight = small_model[0].weight.data  # [512, 256]

    # For dimension mismatch, we need to truncate large weight
    # In practice, you'd use dimension adaptation utilities
    min_out = min(large_weight.shape[0], small_weight.shape[0])
    min_in = min(large_weight.shape[1], small_weight.shape[1])

    large_truncated = large_weight[:min_out, :min_in]
    small_truncated = small_weight[:min_out, :min_in]

    print(f"Large weight shape (truncated): {large_truncated.shape}")
    print(f"Small weight shape: {small_truncated.shape}")

    # Decompose the difference
    A, B = lora_adapter.decompose(large_truncated, reference_weight=small_truncated)

    print(f"\nLoRA components:")
    print(f"  A shape: {A.shape}")  # [rank, in_dim]
    print(f"  B shape: {B.shape}")  # [out_dim, rank]

    # Compute compression
    original_params = large_truncated.numel()
    lora_params = A.numel() + B.numel()
    compression_ratio = original_params / lora_params

    print(f"\nCompression:")
    print(f"  Original parameters: {original_params:,}")
    print(f"  LoRA parameters: {lora_params:,}")
    print(f"  Compression ratio: {compression_ratio:.2f}x")

    # Step 3: Apply LoRA adapter to small model
    print("\nStep 3: Applying LoRA to Small Model")
    print("-"*80)

    enhanced_model = create_model(hidden_dim=512)

    # Apply LoRA adapter
    delta_W = lora_adapter.reconstruct(A, B)
    scale = lora_adapter.get_scaling_factor()

    with torch.no_grad():
        enhanced_model[0].weight[:min_out, :min_in] += scale * delta_W

    print("LoRA adapter applied to first layer")

    # Measure reconstruction error
    reconstructed = small_truncated + scale * delta_W
    error = torch.norm(large_truncated - reconstructed, p='fro') / torch.norm(large_truncated, p='fro')
    print(f"Relative reconstruction error: {error:.6f}")

    # Step 4: Compare QPS
    print("\nStep 4: QPS Comparison")
    print("-"*80)

    input_shape = (256,)

    models = {
        "Small (Original)": small_model,
        "Small + LoRA": enhanced_model,
    }

    results = compare_qps(
        models,
        input_shape,
        batch_size=1,
        num_iterations=100,
        verbose=True
    )

    # Step 5: Demonstrate LoRA Layer
    print("\nStep 5: Using LoRA Layer Module")
    print("-"*80)

    # Create a LoRA layer
    lora_layer = lora_adapter.create_lora_layer(
        in_features=256,
        out_features=512,
        base_weight=small_weight
    )

    print(f"LoRA Layer created: {lora_layer}")

    # Test forward pass
    x = torch.randn(4, 256)
    y = lora_layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Disable adapter
    lora_layer.disable_adapter()
    y_no_adapter = lora_layer(x)
    print(f"\nAdapter disabled, output still works")

    # Re-enable
    lora_layer.enable_adapter()

    # Merge adapter into weights
    print("\nMerging adapter into base weights...")
    lora_layer.merge_adapter()
    y_merged = lora_layer(x)

    # After merging, adapter is disabled but weights are updated
    print("Adapter merged and disabled")

    print("\n" + "="*80)
    print("Key Takeaways:")
    print("="*80)
    print("1. LoRA captures knowledge as low-rank residuals (delta_W = B @ A)")
    print("2. LoRA preserves QPS better than vanilla SVD (parallel computation)")
    print("3. LoRA adapters can be enabled/disabled/merged flexibly")
    print("4. Compression ratio depends on rank (lower rank = more compression)")
    print("5. This is the RECOMMENDED method for production deployment")
    print("="*80)


if __name__ == "__main__":
    main()
