# Matrix Decomposition for Model Distillation

A comprehensive framework for knowledge transfer from large-scale models to compact models using matrix decomposition techniques. This approach improves prediction performance of small models without significantly increasing model size or degrading inference QPS.

**Version 0.2.0** - Now with QPS-optimized methods for production deployment!

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Why Vanilla SVD Hurts QPS](#why-vanilla-svd-hurts-qps-despite-lower-flops)
- [Supported Methods](#supported-methods)
- [For Ads Models (Recommended Setup)](#for-ads-models-recommended-setup)
- [API Reference](#api-reference)
- [Benchmarking](#benchmarking)
- [Advanced Topics](#advanced-topics)

## Overview

### The Problem
You have a production ads model that needs better prediction performance, but you're constrained by:
- **Model size**: Must fit in memory and storage budgets
- **Inference latency**: Must maintain similar QPS (Queries Per Second)
- **FLOPS**: Cannot significantly increase computational cost

### The Solution
1. **Scale-up Training**: Train a 5x larger version of your model to learn better representations
2. **Importance Identification**: Use internal metrics to identify which components contribute most to performance
3. **Matrix Decomposition**: Compress important weight matrices into low-rank representations
4. **Knowledge Transfer**: Inject compressed representations back into the small model
5. **Merge for Deployment**: Merge adapters into base weights for zero runtime overhead

### Key Features

- **15+ decomposition methods** with different accuracy/speed trade-offs
- **30+ importance scoring methods** for layer selection
- **QPS-optimized methods** that maintain or improve inference speed
- **Hybrid approaches** for best of both worlds
- **End-to-end pipeline** for easy experimentation
- **Comprehensive benchmarking** tools

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd matrix-decomp

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Dependencies
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- TensorLy >= 0.8.0 (optional, for Tucker/CP decomposition)

## Quick Start

### Easiest: Use the Pipeline

```python
from matrix_decomp import ModelDistillationPipeline

# Initialize pipeline
pipeline = ModelDistillationPipeline(
    large_model=large_model,     # Your 5x scaled-up model
    small_model=small_model,     # Your production model
    decomposition_method='lora', # Best for QPS (RECOMMENDED)
    rank=64,
    importance_method='taylor'   # Best for ads models
)

# Run the full pipeline
enhanced_model = pipeline.run(
    validation_data=val_loader,
    top_k_layers=10,
    fine_tune_epochs=5,
    measure_qps=True
)

# Get detailed report
pipeline.print_report()
```

### Production Deployment

```python
# After training, merge adapters for zero overhead
for module in enhanced_model.modules():
    if hasattr(module, 'merge'):
        module.merge()

# Now the model is as fast as the original!
```

### For Quick Experiments

```python
from matrix_decomp import AutoPipeline

# Automatically selects the best method based on QPS priority
pipeline = AutoPipeline(
    large_model=large_model,
    small_model=small_model,
    qps_priority=0.8  # 0=accuracy only, 1=QPS only
)
```

## Why Vanilla SVD Hurts QPS (Despite Lower FLOPs)

Many practitioners find that vanilla SVD gives prediction gains but surprisingly *degrades* QPS. Here's a detailed explanation:

### The Mathematics
- **Original**: `y = W @ x` (1 matrix multiplication)
- **SVD**: `y = U @ S @ (V^T @ x)` (2-3 sequential multiplications)

### Performance Bottlenecks

| Issue | Impact | Why It Matters |
|-------|--------|----------------|
| Sequential Operations | 2-3 kernel launches vs 1 | Each launch has ~10-50μs overhead |
| Memory Access | Poor cache locality | Reading 3 matrices is slower than 1 |
| Reduced Parallelism | Cores wait between ops | Underutilization during transitions |
| Batch Size | Worse at batch_size=1 | Common in real-time ads serving |

### Quantified Impact
- CPU inference: 20-40% slower
- GPU (batch=1): 50-100% slower
- GPU (batch=32+): 10-30% slower

### The Solution: Use LoRA-based Methods

```python
# Training: Two operations (acceptable overhead)
y = W @ x + alpha * (B @ A @ x)

# Deployment: Merge into single matrix (ZERO overhead)
W_merged = W + alpha * (B @ A)
y = W_merged @ x  # Same as original!
```

## Supported Methods

### QPS-Optimized Methods (Recommended for Production)

| Method | Class | QPS Impact | Best For |
|--------|-------|------------|----------|
| **MergedLoRA** | `MergedLoRADecomposer` | Zero overhead | Production deployment |
| **DoRA** | `DoRAAdapter` | Zero (when merged) | Maximum accuracy |
| **Standard LoRA** | `LoRAAdapter` | ~5% overhead | Fine-tuning flexibility |
| **Fused SVD** | `FusedSVDDecomposer` | ~10% overhead | When LoRA isn't applicable |

### Alternative Methods

| Method | Class | Use Case |
|--------|-------|----------|
| **RandomizedSVD** | `RandomizedSVDDecomposer` | Very large matrices |
| **QR** | `QRDecomposer` | Numerically unstable matrices |
| **Kronecker** | `KroneckerFactorization` | Block-structured matrices |
| **Tucker** | `TuckerDecomposer` | Convolutional layers (4D) |
| **CP** | `CPDecomposer` | Maximum compression |
| **Vanilla SVD** | `SVDDecomposer` | Offline analysis only |

### Hybrid Approaches

| Method | Class | Description |
|--------|-------|-------------|
| **HybridDecomposer** | `HybridDecomposer` | LoRA for important layers, SVD for others |
| **AdaptiveRank** | `AdaptiveRankDecomposer` | Auto-selects rank per layer |
| **LayerWise** | `LayerWiseDecomposer` | Different methods for layer types |
| **ImportanceWeighted** | `ImportanceWeightedDecomposer` | Higher ranks for important layers |

## For Ads Models (Recommended Setup)

### Recommended Configuration

```python
from matrix_decomp import ModelDistillationPipeline
from matrix_decomp.decomposition import DoRAAdapter

# Best configuration for ads models
pipeline = ModelDistillationPipeline(
    large_model=scaled_up_model,
    small_model=production_model,
    decomposition_method='lora',  # or use DoRA for better accuracy
    rank=64,
    importance_method='taylor',   # Most reliable for ads
    device='cuda'
)

# Layer-specific ranks (recommended)
rank_config = {
    'embedding': 256,      # User/item embeddings - high capacity
    'interaction': 128,    # Cross-feature interactions - critical
    'hidden': 64,          # MLP hidden layers - less sensitive
    'output': 64,          # Final layers - important
}
```

### Rank Selection Guidelines

| Layer Type | Recommended Rank | Rationale |
|------------|------------------|-----------|
| Embedding layers | 128-256 | High dimensional, need capacity |
| Attention Q/K/V | 64-128 | Critical for performance |
| FFN intermediate | 32-64 | Less sensitive to rank |
| Output/Classification | 64-128 | Important for predictions |

### Complete Ads Model Example

```python
import torch
from matrix_decomp import ModelDistillationPipeline
from matrix_decomp.benchmarks import measure_qps

# 1. Create your models
class AdsModel(torch.nn.Module):
    def __init__(self, scale=1):
        super().__init__()
        self.embedding = torch.nn.Embedding(100000, 64 * scale)
        self.interaction = torch.nn.Linear(64 * scale, 256 * scale)
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(256 * scale, 512 * scale),
            torch.nn.ReLU(),
            torch.nn.Linear(512 * scale, 256 * scale),
        )
        self.output = torch.nn.Linear(256 * scale, 1)

    def forward(self, x):
        x = self.embedding(x).mean(dim=1)
        x = self.interaction(x)
        x = self.hidden(x)
        return self.output(x)

# Create 5x scale-up and production models
large_model = AdsModel(scale=5)  # 5x larger
small_model = AdsModel(scale=1)  # Production size

# Train large_model... (your training code)

# 2. Transfer knowledge
pipeline = ModelDistillationPipeline(
    large_model=large_model,
    small_model=small_model,
    decomposition_method='lora',
    rank=64,
    importance_method='taylor'
)

enhanced = pipeline.run(
    validation_data=val_loader,
    top_k_layers=5,
    fine_tune_epochs=5
)

# 3. Merge for deployment
for module in enhanced.modules():
    if hasattr(module, 'merge'):
        module.merge()

# 4. Verify QPS
original_qps = measure_qps(small_model, (10,), device='cuda')
enhanced_qps = measure_qps(enhanced, (10,), device='cuda')

print(f"Original QPS: {original_qps['mean_qps']:.0f}")
print(f"Enhanced QPS: {enhanced_qps['mean_qps']:.0f}")
# Should be nearly identical!
```

## API Reference

### Main Pipeline

```python
from matrix_decomp import ModelDistillationPipeline, AutoPipeline

# ModelDistillationPipeline
pipeline = ModelDistillationPipeline(
    large_model,              # Source model
    small_model,              # Target model
    decomposition_method,     # 'lora', 'fused_svd', 'svd', etc.
    rank,                     # Low-rank dimension
    importance_method,        # 'gradient', 'taylor', 'synflow', etc.
    device='cpu'
)

# Methods
pipeline.compute_importance(data_loader, criterion)
pipeline.select_layers(top_k=10)
pipeline.transfer_knowledge(layer_names)
pipeline.fine_tune(train_loader, criterion, epochs)
pipeline.benchmark(input_shape)
pipeline.print_report()
```

### Decomposition Methods

```python
from matrix_decomp.decomposition import (
    # Standard
    SVDDecomposer,
    LoRAAdapter,
    FusedSVDDecomposer,
    TuckerDecomposer,
    CPDecomposer,

    # QPS-Optimized
    DoRAAdapter,
    MergedLoRADecomposer,
    RandomizedSVDDecomposer,
    QRDecomposer,
    KroneckerFactorization,

    # Hybrid
    HybridDecomposer,
    AdaptiveRankDecomposer,
    LayerWiseDecomposer,
    ImportanceWeightedDecomposer,
)
```

### Importance Scoring

```python
from matrix_decomp.importance import compute_importance_scores

# 30+ methods available
scores = compute_importance_scores(
    model,
    data_loader,
    method='taylor',  # See IMPORTANCE_METHODS_GUIDE.md for all options
    criterion=loss_fn
)
```

### Benchmarking

```python
from matrix_decomp.benchmarks import (
    measure_qps,
    compare_qps,
    count_flops,
    compare_flops,
    compute_compression_ratio,
)

# Measure QPS
qps = measure_qps(model, input_shape=(128,), device='cuda', batch_size=1)

# Compare multiple models
results = compare_qps(
    {'Original': model1, 'Enhanced': model2},
    input_shape=(128,),
    device='cuda'
)
```

## Benchmarking

Always benchmark before deploying:

```python
from matrix_decomp.benchmarks import measure_qps, profile_layer_latency

# Overall QPS
qps = measure_qps(
    model,
    input_shape=(128,),
    batch_size=1,        # Use production batch size
    num_iterations=1000,
    device='cuda',
    verbose=True
)

# Per-layer breakdown (find bottlenecks)
latencies = profile_layer_latency(model, input_shape=(128,))
for layer, ms in sorted(latencies.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{layer}: {ms:.2f}ms")
```

## Advanced Topics

### Handling Dimension Mismatches

When the large model has 5x dimensions but the small model is 1x:

```python
from matrix_decomp.transfer import adapt_dimensions, smart_dimension_adaptation

# Large: [1024, 2048], Small: [204, 409]
# The framework handles this automatically!
```

### Layer-Specific Configurations

```python
from matrix_decomp.decomposition import LayerWiseDecomposer

decomposer = LayerWiseDecomposer(
    default_rank=64,
    layer_configs={
        'attention': {'method': 'lora', 'rank': 128},
        'ffn': {'method': 'fused_svd', 'rank': 32},
        'embedding': {'method': 'randomized_svd', 'rank': 256},
    }
)
```

### Combining with Other Techniques

```python
# After decomposition-based transfer:

# 1. Knowledge Distillation
# Use large model as teacher during fine-tuning
teacher_outputs = large_model(x)
student_outputs = small_model(x)
kd_loss = F.kl_div(student_outputs.log(), teacher_outputs)

# 2. Quantization
# Quantize after merging adapters
quantized_model = torch.quantization.quantize_dynamic(
    enhanced_model, {torch.nn.Linear}, dtype=torch.qint8
)

# 3. Pruning
# Prune less important neurons
from torch.nn.utils import prune
prune.l1_unstructured(layer, name='weight', amount=0.3)
```

## Project Structure

```
matrix-decomp/
├── README.md                          # This file
├── METHOD_SELECTION_GUIDE.md          # Detailed method selection guide
├── IMPORTANCE_METHODS_GUIDE.md        # 30+ importance scoring methods
├── requirements.txt
├── setup.py
├── matrix_decomp/
│   ├── __init__.py
│   ├── pipeline.py                    # End-to-end pipeline
│   ├── decomposition/
│   │   ├── base.py
│   │   ├── svd_decomposition.py
│   │   ├── lora_adapter.py
│   │   ├── fused_svd.py
│   │   ├── qps_optimized.py          # DoRA, MergedLoRA, etc.
│   │   ├── hybrid.py                 # Hybrid approaches
│   │   ├── tucker_decomposition.py
│   │   └── cp_decomposition.py
│   ├── importance/                    # 30+ scoring methods
│   ├── transfer/
│   └── benchmarks/
├── examples/
│   ├── ads_model_example.py
│   ├── full_pipeline_example.py
│   └── lora_adapter_example.py
└── tests/
```

## Method Comparison

| Method | Accuracy | QPS | Memory | Complexity | Recommended For |
|--------|----------|-----|--------|------------|-----------------|
| **MergedLoRA** | High | Best | Same | Low | Production (BEST) |
| **DoRA** | Highest | Best | Same | Low | Maximum accuracy |
| **LoRA** | High | Good | +10% | Low | Fine-tuning |
| **Fused SVD** | Medium | Good | Low | Low | No adapter support |
| **RandomizedSVD** | Medium | Good | Low | Low | Large matrices |
| **Vanilla SVD** | Medium | Poor | Low | Low | Avoid for serving |
| **Tucker** | High | Variable | Medium | High | CNN layers |
| **CP** | Medium | Variable | Low | High | Extreme compression |

## Troubleshooting

### QPS Still Low After Transfer?

1. Ensure adapters are merged:
   ```python
   for m in model.modules():
       if hasattr(m, 'merge'):
           m.merge()
   ```

2. Profile to find bottleneck:
   ```python
   from matrix_decomp.benchmarks import profile_layer_latency
   latencies = profile_layer_latency(model, input_shape)
   ```

### Accuracy Dropped Too Much?

1. Increase rank for important layers
2. Use DoRA instead of standard LoRA
3. Fine-tune for more epochs
4. Use `AdaptiveRankDecomposer`

## Documentation

- [METHOD_SELECTION_GUIDE.md](METHOD_SELECTION_GUIDE.md) - Choosing the right method
- [IMPORTANCE_METHODS_GUIDE.md](IMPORTANCE_METHODS_GUIDE.md) - 30+ importance methods
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Detailed usage examples

## Citation

```bibtex
@software{matrix_decomp_distillation,
  title={Matrix Decomposition for Model Distillation},
  author={Your Team},
  year={2025},
  url={https://github.com/yourusername/matrix-decomp}
}
```

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- [Finding structure with randomness](https://arxiv.org/abs/0909.4061) (Randomized SVD)
- Tensor decomposition literature for Tucker/CP methods

## License

MIT License - see `LICENSE` file for details.
