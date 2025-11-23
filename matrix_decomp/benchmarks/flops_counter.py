"""
FLOPS (Floating Point Operations) counting utilities.

Estimates computational cost of models by counting FLOPs.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn


def count_layer_flops(
    module: nn.Module,
    input_shape: Tuple,
    output_shape: Optional[Tuple] = None
) -> int:
    """
    Count FLOPs for a single layer.

    Args:
        module: Layer module
        input_shape: Input shape (excluding batch)
        output_shape: Output shape (excluding batch), computed if None

    Returns:
        Number of FLOPs
    """
    if isinstance(module, nn.Linear):
        # FLOPs = 2 * in_features * out_features (multiply-add)
        # For batch: batch_size is typically 1 for per-sample FLOPs
        in_features = module.in_features
        out_features = module.out_features
        flops = 2 * in_features * out_features

        # Add bias FLOPs if present
        if module.bias is not None:
            flops += out_features

        return flops

    elif isinstance(module, nn.Conv2d):
        # FLOPs = 2 * in_channels * out_channels * kernel_h * kernel_w * out_h * out_w
        # Assuming input_shape = (C, H, W)
        in_channels = module.in_channels
        out_channels = module.out_channels
        kernel_h, kernel_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        stride_h, stride_w = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
        padding_h, padding_w = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)

        if len(input_shape) == 3:
            _, H, W = input_shape
        else:
            H, W = input_shape[0], input_shape[1]

        # Output spatial dimensions
        out_h = (H + 2 * padding_h - kernel_h) // stride_h + 1
        out_w = (W + 2 * padding_w - kernel_w) // stride_w + 1

        # FLOPs per output pixel
        flops_per_pixel = 2 * in_channels * kernel_h * kernel_w
        flops = flops_per_pixel * out_channels * out_h * out_w

        # Add bias
        if module.bias is not None:
            flops += out_channels * out_h * out_w

        return flops

    elif isinstance(module, nn.Conv1d):
        in_channels = module.in_channels
        out_channels = module.out_channels
        kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
        stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
        padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding

        if len(input_shape) == 2:
            _, L = input_shape
        else:
            L = input_shape[0]

        out_l = (L + 2 * padding - kernel_size) // stride + 1

        flops_per_position = 2 * in_channels * kernel_size
        flops = flops_per_position * out_channels * out_l

        if module.bias is not None:
            flops += out_channels * out_l

        return flops

    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
        # Normalization: mean, variance, normalize, scale, shift
        # Approximately 5 ops per element
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        return 5 * num_elements

    elif isinstance(module, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh)):
        # Activation: 1 op per element
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        return num_elements

    else:
        # Unknown layer type
        return 0


def count_flops(
    model: nn.Module,
    input_shape: Tuple,
    verbose: bool = True
) -> Dict:
    """
    Count total FLOPs for a model.

    Args:
        model: Model to analyze
        input_shape: Input shape (excluding batch dimension)
        verbose: Print per-layer FLOPs

    Returns:
        Dictionary with FLOP counts
    """
    model.eval()

    total_flops = 0
    layer_flops = {}

    # Create a forward hook to track intermediate shapes
    intermediate_shapes = {}

    def get_shape_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                # Store output shape (excluding batch)
                intermediate_shapes[name] = tuple(output.shape[1:])
        return hook

    # Register hooks
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.ReLU, nn.GELU)):
            handle = module.register_forward_hook(get_shape_hook(name))
            handles.append(handle)

    # Forward pass to get shapes
    dummy_input = torch.randn(1, *input_shape)
    with torch.no_grad():
        _ = model(dummy_input)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Count FLOPs for each layer
    current_shape = input_shape

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.ReLU, nn.GELU)):
            # Get input shape for this layer
            if name in intermediate_shapes:
                layer_input_shape = current_shape
                current_shape = intermediate_shapes[name]
            else:
                layer_input_shape = current_shape

            flops = count_layer_flops(module, layer_input_shape)
            layer_flops[name] = flops
            total_flops += flops

    # Compute GFLOPs and MFLOPs
    gflops = total_flops / 1e9
    mflops = total_flops / 1e6

    results = {
        'total_flops': total_flops,
        'gflops': gflops,
        'mflops': mflops,
        'layer_flops': layer_flops,
    }

    if verbose:
        print(f"\nFLOPs Analysis:")
        print(f"  Total FLOPs: {total_flops:,}")
        print(f"  GFLOPs: {gflops:.2f}")
        print(f"  MFLOPs: {mflops:.2f}")

        # Print top layers by FLOPs
        sorted_layers = sorted(layer_flops.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Top 10 Layers by FLOPs:")
        for i, (name, flops) in enumerate(sorted_layers[:10], 1):
            percentage = (flops / total_flops) * 100
            print(f"    {i}. {name}: {flops:,} ({percentage:.1f}%)")

    return results


def compare_flops(
    models: Dict[str, nn.Module],
    input_shape: Tuple,
    verbose: bool = True
) -> Dict:
    """
    Compare FLOPs across multiple models.

    Args:
        models: Dictionary mapping model names to models
        input_shape: Input shape
        verbose: Print comparison

    Returns:
        Dictionary with comparison results
    """
    results = {}

    for name, model in models.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Analyzing: {name}")
            print(f"{'='*60}")

        results[name] = count_flops(model, input_shape, verbose=verbose)

    # Print comparison
    if verbose and len(results) > 1:
        print(f"\n{'='*60}")
        print("FLOPs Comparison")
        print(f"{'='*60}")

        baseline_name = list(results.keys())[0]
        baseline_flops = results[baseline_name]['total_flops']

        print(f"{'Model':<30} {'GFLOPs':<15} {'Relative':<15}")
        print("-" * 60)

        for name, result in results.items():
            gflops = result['gflops']
            relative = result['total_flops'] / baseline_flops

            print(f"{name:<30} {gflops:<15.2f} {relative:<15.2f}x")

    return results


def estimate_memory_usage(
    model: nn.Module,
    input_shape: Tuple,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32
) -> Dict:
    """
    Estimate memory usage of a model.

    Args:
        model: Model to analyze
        input_shape: Input shape
        batch_size: Batch size
        dtype: Data type

    Returns:
        Dictionary with memory estimates
    """
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())

    # Bytes per parameter
    if dtype == torch.float32:
        bytes_per_param = 4
    elif dtype == torch.float16:
        bytes_per_param = 2
    elif dtype == torch.int8:
        bytes_per_param = 1
    else:
        bytes_per_param = 4

    # Parameter memory
    param_memory_bytes = num_params * bytes_per_param
    param_memory_mb = param_memory_bytes / (1024 ** 2)

    # Estimate activation memory (rough approximation)
    # This is a very rough estimate
    dummy_input = torch.randn(batch_size, *input_shape)
    activation_memory_mb = dummy_input.numel() * bytes_per_param / (1024 ** 2)

    # Estimate total
    total_memory_mb = param_memory_mb + activation_memory_mb * 2  # Factor of 2 for gradients

    return {
        'num_parameters': num_params,
        'parameter_memory_mb': param_memory_mb,
        'estimated_activation_memory_mb': activation_memory_mb,
        'estimated_total_memory_mb': total_memory_mb,
        'batch_size': batch_size,
    }


def compute_compression_ratio(
    original_model: nn.Module,
    compressed_model: nn.Module
) -> Dict:
    """
    Compute compression ratio between two models.

    Args:
        original_model: Original model
        compressed_model: Compressed model

    Returns:
        Dictionary with compression statistics
    """
    orig_params = sum(p.numel() for p in original_model.parameters())
    comp_params = sum(p.numel() for p in compressed_model.parameters())

    compression_ratio = orig_params / comp_params
    size_reduction = (orig_params - comp_params) / orig_params

    return {
        'original_parameters': orig_params,
        'compressed_parameters': comp_params,
        'compression_ratio': compression_ratio,
        'size_reduction_percentage': size_reduction * 100,
    }
