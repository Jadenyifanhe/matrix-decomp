# Usage Guide

This guide provides detailed instructions for using the matrix decomposition framework for model distillation.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Method Selection Guide](#method-selection-guide)
4. [Common Workflows](#common-workflows)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

```bash
git clone <your-repo>
cd matrix-decomp
pip install -r requirements.txt
pip install -e .
```

### 5-Minute Example

```python
import torch
from matrix_decomp import LoRAAdapter, compute_importance_scores, KnowledgeTransfer

# Your models
large_model = YourLargeModel()  # Trained 5x scaled-up model
small_model = YourSmallModel()  # Production model

# 1. Find important layers
importance = compute_importance_scores(large_model, val_loader, method='gradient')

# 2. Transfer knowledge using LoRA (recommended)
transfer = KnowledgeTransfer(method='lora', rank=64)
transfer.transfer_model(large_model, small_model, layer_names=top_layers)

# 3. Your small model now has knowledge from large model!
```

## Core Concepts

### The Problem We're Solving

You have:
- ✅ A production model (small, fast, but limited performance)
- ✅ A scaled-up model (5x larger, better performance, but too slow/big)

You want:
- ✅ Production model performance + Large model knowledge
- ✅ WITHOUT increasing model size or degrading QPS

### How It Works

```
1. TRAIN: Large Model (5x scale-up) → Better representations learned

2. IDENTIFY: Which components improved the most? (Importance scoring)

3. DECOMPOSE: Compress important weights into low-rank form
   Original: W [1024, 2048] = 2M parameters
   Compressed: U [1024, 64] + V [64, 2048] = 196K parameters

4. TRANSFER: Inject compressed knowledge into small model
   Small Model + LoRA Adapter → Enhanced Small Model

5. DEPLOY: Same size, same QPS, better predictions!
```

### Why Matrix Decomposition?

Standard approaches:
- **Knowledge Distillation**: Requires retraining (slow, expensive)
- **Pruning**: Loses important connections
- **Quantization**: Reduces precision, may hurt accuracy

Our approach:
- **Matrix Decomposition**: Compress knowledge, transfer instantly
- **No retraining needed** (optional fine-tuning helps)
- **Preserves important patterns** from large model

## Method Selection Guide

### Decision Tree

```
START: Need to transfer knowledge from large to small model
│
├─ Q: Is QPS critical? (Online serving)
│  ├─ YES → Use LoRA (best QPS)
│  │   └─ Set rank=32-128, alpha=16
│  │
│  └─ NO → Can use SVD or Fused SVD
│      ├─ Offline inference → SVD (simpler)
│      └─ Some QPS concern → Fused SVD
│
├─ Q: Working with convolutional layers?
│  └─ YES → Consider Tucker or CP decomposition
│      ├─ Tucker: Better accuracy, more parameters
│      └─ CP: Maximum compression, less stable
│
└─ Q: Need extreme compression?
   └─ YES → Use CP with rank=16-32
```

### Method Comparison Table

| Method | QPS Impact | Accuracy | Complexity | Use Case |
|--------|-----------|----------|------------|----------|
| **LoRA** | ✅ Best | ✅ Best | Easy | **Production (Recommended)** |
| **Fused SVD** | ✅ Good | Good | Easy | Deployment |
| **Vanilla SVD** | ⚠️ Poor | Good | Easy | Research/Offline |
| **Tucker** | Good | ✅ Best | Medium | Convolutional layers |
| **CP** | ✅ Best | ⚠️ Moderate | Hard | Extreme compression |

### When to Use Each Method

#### LoRA (Recommended for Most Cases)

```python
from matrix_decomp import LoRAAdapter

adapter = LoRAAdapter(
    rank=64,        # 32-128 is typical
    alpha=16,       # Scaling factor
    init_method='svd'  # Initialize from SVD
)

# Best for:
# - Online serving (maintains QPS)
# - When you want to keep original weights
# - When you might want to enable/disable adapters
```

**Why LoRA for Production?**
- Preserves original matrix (can fall back if needed)
- Parallel computation: `W@x + B@A@x` (not sequential)
- Can merge into weights for deployment
- Battle-tested in practice (used in LLM fine-tuning)

#### Fused SVD (Good Balance)

```python
from matrix_decomp import FusedSVDDecomposer

decomposer = FusedSVDDecomposer(
    rank=64,
    fusion_mode='balanced'  # or 'left', 'right'
)

# Best for:
# - Deployment scenarios
# - When you want SVD benefits without QPS hit
# - Model compression
```

#### Vanilla SVD (For Offline/Research)

```python
from matrix_decomp import SVDDecomposer

decomposer = SVDDecomposer(
    rank=64,
    rank_selection='energy',  # Adaptive rank
    energy_threshold=0.9
)

# Best for:
# - Offline analysis
# - Research experiments
# - Understanding rank requirements
```

## Common Workflows

### Workflow 1: Basic Knowledge Transfer

```python
import torch
from matrix_decomp import KnowledgeTransfer, compute_importance_scores

# Step 1: Identify important layers
importance_scores = compute_importance_scores(
    large_model,
    validation_loader,
    method='gradient',  # or 'activation', 'fisher'
    criterion=nn.CrossEntropyLoss()
)

# Step 2: Select top-k layers
from matrix_decomp.importance import select_top_k_layers
top_layers = select_top_k_layers(importance_scores, k=10)

# Step 3: Transfer
transfer = KnowledgeTransfer(method='lora', rank=64)
stats = transfer.transfer_model(
    source_model=large_model,
    target_model=small_model,
    layer_names=[name for name, _ in top_layers],
    adapt_dims=True  # Handle dimension mismatches
)

print(f"Transferred {stats['aggregate']['num_layers']} layers")
print(f"Mean error: {stats['aggregate']['mean_relative_error']:.6f}")
```

### Workflow 2: Layer-by-Layer Control

```python
from matrix_decomp import LoRAAdapter

adapter = LoRAAdapter(rank=64, alpha=16)

# Transfer specific layers with custom ranks
layer_ranks = {
    'attention.query': 128,  # High rank for attention
    'attention.key': 128,
    'ffn.dense': 64,         # Medium for FFN
    'output': 32             # Low for output
}

for layer_name, rank in layer_ranks.items():
    adapter.rank = rank

    # Get weights
    large_weight = large_model.get_parameter(layer_name)
    small_weight = small_model.get_parameter(layer_name)

    # Transfer
    A, B = adapter.decompose(large_weight, reference_weight=small_weight)
    delta_W = adapter.reconstruct(A, B)

    # Apply
    small_weight.data += adapter.get_scaling_factor() * delta_W
```

### Workflow 3: QPS-Aware Transfer

```python
from matrix_decomp import KnowledgeTransfer
from matrix_decomp.benchmarks import measure_qps

# Define QPS requirement
MIN_QPS = 1000

# Test different methods
methods = ['lora', 'fused_svd']
best_method = None
best_qps = 0

for method in methods:
    # Create test model
    test_model = create_small_model()

    # Transfer
    transfer = KnowledgeTransfer(method=method, rank=64)
    transfer.transfer_model(large_model, test_model, layer_names=important_layers)

    # Measure QPS
    qps_result = measure_qps(test_model, input_shape=(512,), num_iterations=100)

    if qps_result['mean_qps'] >= MIN_QPS and qps_result['mean_qps'] > best_qps:
        best_method = method
        best_qps = qps_result['mean_qps']

print(f"Best method: {best_method} with QPS={best_qps}")
```

### Workflow 4: Dimension Adaptation

```python
from matrix_decomp.transfer import smart_dimension_adaptation

# Large model layer: [2048, 1024]
# Small model layer: [512, 256]

# Decompose large
decomposer = LoRAAdapter(rank=64)
A, B = decomposer.decompose(large_weight)  # A: [64, 1024], B: [2048, 64]

# Adapt to small dimensions
A_adapted, B_adapted = smart_dimension_adaptation(
    (A, B),
    target_shape=(512, 256),
    decomposition_type='lora'
)

# Now A_adapted: [64, 256], B_adapted: [512, 64]
# Ready to apply to small model!
```

## Advanced Usage

### Custom Importance Metric

```python
from matrix_decomp.importance import GradientImportance

class CustomImportance(GradientImportance):
    def compute_scores(self, model, data_loader, **kwargs):
        # Your custom importance logic
        scores = {}
        for name, module in model.named_modules():
            # Compute custom score
            scores[name] = custom_metric(module)
        return scores

# Use it
importance = CustomImportance()
scores = importance.compute_scores(model, data_loader)
```

### Layer-Specific Rank Selection

```python
from matrix_decomp import SVDDecomposer

def select_rank_by_layer(layer_name, weight_shape):
    """Custom rank selection logic."""
    if 'attention' in layer_name:
        return 128  # High rank for attention
    elif 'ffn' in layer_name:
        return 64   # Medium for FFN
    else:
        return 32   # Low for others

# Apply
for name, module in model.named_modules():
    if hasattr(module, 'weight'):
        rank = select_rank_by_layer(name, module.weight.shape)
        decomposer = SVDDecomposer(rank=rank)
        # ... decompose and transfer
```

### Incremental Transfer

```python
# Transfer layers one at a time and validate
for layer_name in important_layers:
    # Transfer
    transfer_layer(large_model, small_model, layer_name)

    # Validate
    accuracy = evaluate(small_model, val_loader)

    if accuracy < threshold:
        # Rollback
        restore_layer(small_model, layer_name)
        print(f"Skipping {layer_name} - hurts performance")
    else:
        print(f"Kept {layer_name} - accuracy: {accuracy}")
```

### Combining Methods

```python
# Use LoRA for important layers, SVD for others
for layer_name in all_layers:
    if layer_name in top_important:
        # LoRA for important layers (preserve QPS)
        method = 'lora'
        rank = 128
    else:
        # SVD for less important (more compression)
        method = 'svd'
        rank = 32

    transfer = KnowledgeTransfer(method=method, rank=rank)
    transfer.transfer_layer(large_weight, small_module, layer_name)
```

## Troubleshooting

### Issue: QPS Degraded After Transfer

**Symptoms**: Model is slower after applying decomposition

**Solutions**:
1. **Use LoRA instead of SVD**
   ```python
   # Change from:
   transfer = KnowledgeTransfer(method='svd', rank=64)
   # To:
   transfer = KnowledgeTransfer(method='lora', rank=64)
   ```

2. **Use Fused SVD**
   ```python
   transfer = KnowledgeTransfer(method='fused_svd', rank=64)
   ```

3. **Reduce number of decomposed layers**
   ```python
   # Only transfer top-3 instead of top-10
   top_layers = select_top_k_layers(importance, k=3)
   ```

4. **Merge LoRA adapters**
   ```python
   # After transfer, merge adapters into weights
   lora_layer.merge_adapter()
   ```

### Issue: Large Reconstruction Error

**Symptoms**: High error when reconstructing weights

**Solutions**:
1. **Increase rank**
   ```python
   transfer = KnowledgeTransfer(method='lora', rank=128)  # Was 64
   ```

2. **Use energy-based rank selection**
   ```python
   decomposer = SVDDecomposer(
       rank_selection='energy',
       energy_threshold=0.95  # Preserve 95% of energy
   )
   ```

3. **Analyze singular values first**
   ```python
   analysis = decomposer.analyze_singular_values(weight)
   print(analysis['ranks_for_energy'])
   # Use recommended rank
   ```

### Issue: Dimension Mismatch

**Symptoms**: Error about incompatible dimensions

**Solutions**:
1. **Enable dimension adaptation**
   ```python
   transfer.transfer_model(
       large_model,
       small_model,
       adapt_dims=True  # Important!
   )
   ```

2. **Manual adaptation**
   ```python
   from matrix_decomp.transfer import adapt_dimensions

   adapted_weight = adapt_dimensions(
       large_weight,
       target_rows=small_dim[0],
       target_cols=small_dim[1],
       method='truncate'  # or 'pad', 'interpolate'
   )
   ```

### Issue: Out of Memory

**Symptoms**: CUDA out of memory during transfer

**Solutions**:
1. **Transfer layers sequentially**
   ```python
   for layer_name in layer_names:
       transfer.transfer_layer(large_weight, small_module, layer_name)
       torch.cuda.empty_cache()
   ```

2. **Use CPU for decomposition**
   ```python
   # Move to CPU, decompose, move back
   weight_cpu = large_weight.cpu()
   components = decomposer.decompose(weight_cpu)
   components = [c.cuda() for c in components]
   ```

### Issue: No Performance Improvement

**Symptoms**: Small model performance doesn't improve after transfer

**Solutions**:
1. **Fine-tune after transfer**
   ```python
   # After transfer
   fine_tune(small_model, train_loader, epochs=5, lr=1e-4)
   ```

2. **Transfer more layers**
   ```python
   # Increase from top-5 to top-10
   top_layers = select_top_k_layers(importance, k=10)
   ```

3. **Use different importance metric**
   ```python
   # Try gradient-based instead of activation-based
   importance = compute_importance_scores(
       model, data_loader, method='gradient'
   )
   ```

4. **Check if large model actually learned**
   ```python
   # Verify large model is better than small
   large_acc = evaluate(large_model, test_loader)
   small_acc = evaluate(small_model, test_loader)
   assert large_acc > small_acc, "Large model must be better!"
   ```

## Best Practices

### 1. Always Benchmark

```python
from matrix_decomp.benchmarks import measure_qps, count_flops

# Before deployment
baseline_qps = measure_qps(original_model, input_shape)
enhanced_qps = measure_qps(enhanced_model, input_shape)

assert enhanced_qps['mean_qps'] >= baseline_qps['mean_qps'] * 0.95, \
    "QPS degraded too much!"
```

### 2. Validate on Real Data

```python
# Don't just check reconstruction error
# Check actual task performance
test_acc = evaluate(enhanced_model, test_loader)
print(f"Test accuracy: {test_acc}")
```

### 3. Start Small, Scale Up

```python
# Start with 1 layer
transfer_one_layer()
validate()

# Then 3 layers
transfer_three_layers()
validate()

# Then all important layers
transfer_all()
```

### 4. Monitor in Production

```python
# Log QPS, latency, prediction quality
metrics = {
    'qps': current_qps,
    'p99_latency': p99,
    'auc': current_auc,
    'model_version': 'enhanced_v1'
}
log_metrics(metrics)
```

## Examples

See the `examples/` directory for complete runnable examples:

- `basic_svd_example.py`: SVD basics and analysis
- `lora_adapter_example.py`: LoRA usage (recommended)
- `full_pipeline_example.py`: End-to-end workflow
- `ads_model_example.py`: Ads model specific example

Run any example:
```bash
python examples/lora_adapter_example.py
```

## Getting Help

- Check the main [README.md](README.md) for overview
- Review code comments - all modules are well-documented
- Open an issue on GitHub for bugs or questions
