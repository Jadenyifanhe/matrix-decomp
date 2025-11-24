# Method Selection Guide for Ads Model Distillation

This guide helps you choose the right decomposition method and rank configuration for your ads model distillation use case.

## Quick Decision Tree

```
Start
  │
  ├─ Is QPS critical for your deployment?
  │   │
  │   ├─ YES: Use LoRA-based methods
  │   │   │
  │   │   ├─ Need maximum accuracy? → DoRA
  │   │   ├─ Need maximum speed? → MergedLoRA
  │   │   └─ Balanced? → Standard LoRA
  │   │
  │   └─ NO: Any method works
  │       │
  │       ├─ Large matrices? → RandomizedSVD
  │       ├─ Conv layers? → Tucker
  │       └─ Default → FusedSVD
  │
  └─ Not sure? → Use AutoPipeline
```

## Why Vanilla SVD Hurts QPS (Detailed Explanation)

You mentioned that "people tried vanilla SVD... and found QPS performance is very bad." Here's the complete explanation:

### The Problem

Original linear layer: `y = W @ x`
- **1 matrix multiplication**
- Modern GPUs/TPUs are optimized for large matrix operations
- Memory: Load W once, compute once

After vanilla SVD: `y = U @ S @ (V^T @ x)`
- **2-3 sequential matrix multiplications**
- Each operation must wait for the previous one
- Memory: Load U, S, V^T from different memory locations

### Specific Performance Killers

1. **Sequential Dependencies**
   ```
   Original: [GPU: One large matrix-multiply kernel]

   SVD:      [GPU: V^T @ x] → wait → [GPU: scale by S] → wait → [GPU: U @ result]
   ```

2. **Memory Bandwidth Bottleneck**
   - Modern GPUs are often memory-bound, not compute-bound
   - Reading 3 smaller matrices from memory is slower than 1 large matrix
   - Poor cache locality between U and V^T

3. **Kernel Launch Overhead**
   - Each matrix multiply = 1 GPU kernel launch
   - Kernel launches have ~10-50μs overhead each
   - For small batch sizes, this overhead dominates

4. **Reduced Parallelism**
   - Large single matrix multiply: Can use all GPU cores simultaneously
   - Two sequential smaller multiplies: Cores are underutilized during transition

5. **Batch Size Effects**
   - Ads models often use batch_size=1 for real-time serving
   - At batch_size=1, the overhead is proportionally much worse

### Quantified Impact

Typical QPS degradation with vanilla SVD:
- CPU inference: 20-40% slower
- GPU inference (batch=1): 50-100% slower
- GPU inference (batch=32+): 10-30% slower

## Method Comparison for Ads Models

### Recommended: LoRA-based Methods

| Method | Accuracy | QPS Impact | Memory | Best For |
|--------|----------|------------|--------|----------|
| **MergedLoRA** | High | ✅ Zero overhead | Same | Production deployment |
| **DoRA** | Highest | ✅ Zero (merged) | Same | When accuracy is critical |
| **Standard LoRA** | High | ✅ ~5% overhead | +10% | Fine-tuning scenario |

### Why LoRA Preserves QPS

```python
# Training phase: Two operations (acceptable)
y = W @ x + alpha * (B @ A @ x)

# Deployment phase: Merge adapter (zero overhead)
W_merged = W + alpha * (B @ A)
y = W_merged @ x  # Single operation!
```

### Alternative Methods

| Method | Accuracy | QPS Impact | Use Case |
|--------|----------|------------|----------|
| **FusedSVD** | Good | ~10% overhead | When can't use LoRA |
| **RandomizedSVD** | Good | ~10% overhead | Very large matrices |
| **QR** | Medium | ~15% overhead | Numerically unstable matrices |
| **Tucker** | Variable | Depends | Conv layers only |
| **Vanilla SVD** | Good | ⚠️ 30-100% overhead | Avoid for online serving |

## Rank Selection Guidelines

### For Ads Models (Typical)

| Layer Type | Recommended Rank | Rationale |
|------------|-----------------|-----------|
| Embedding layers | 128-256 | High dimensional, need more capacity |
| Attention Q/K/V | 64-128 | Critical for performance |
| FFN intermediate | 32-64 | Less sensitive to rank |
| Output/Classification | 64-128 | Important for predictions |

### Based on Matrix Size

```python
def suggest_rank(weight_shape, importance_score=0.5):
    """Suggest rank based on matrix size and importance."""
    m, n = weight_shape
    max_rank = min(m, n)

    # Base rank as percentage of smallest dimension
    base_percentage = 0.05 + 0.15 * importance_score  # 5-20%
    suggested = int(max_rank * base_percentage)

    # Clamp to reasonable range
    return max(8, min(256, suggested))
```

### Based on Singular Value Distribution

```python
def rank_from_energy(weight, target_energy=0.95):
    """Find rank that preserves target_energy of information."""
    _, S, _ = torch.linalg.svd(weight)

    cumsum = torch.cumsum(S**2, dim=0)
    total = cumsum[-1]

    # Find smallest rank that captures target_energy
    for r, energy in enumerate(cumsum):
        if energy / total >= target_energy:
            return r + 1
    return len(S)
```

### Quick Reference

| Target | Energy Threshold | Typical Rank (512x512) |
|--------|-----------------|------------------------|
| Maximum accuracy | 99% | ~400 |
| High accuracy | 95% | ~100-200 |
| Balanced | 90% | ~50-100 |
| Aggressive compression | 80% | ~20-50 |
| Maximum compression | 70% | ~10-20 |

## Production Deployment Recommendations

### For Ads Models Specifically

1. **Use MergedLoRA or DoRA**
   - Train with adapter (flexibility)
   - Merge before deployment (speed)
   - Zero QPS overhead in production

2. **Rank Configuration**
   ```python
   rank_config = {
       'embedding': 256,      # High capacity for user/item embeddings
       'interaction': 128,    # Cross-features interaction layers
       'hidden': 64,          # MLP hidden layers
       'output': 64,          # Final prediction layers
   }
   ```

3. **Layer Selection**
   - Focus on feature interaction layers (highest impact)
   - Embedding layers if they're the bottleneck
   - Skip layers that are already small

4. **Fine-tuning After Transfer**
   - Always fine-tune for 3-5 epochs
   - Use lower learning rate (1e-5 to 1e-4)
   - Monitor both accuracy AND QPS

### Example Production Pipeline

```python
from matrix_decomp import ModelDistillationPipeline

# Create pipeline with production-optimized settings
pipeline = ModelDistillationPipeline(
    large_model=scaled_up_model,
    small_model=production_model,
    decomposition_method='lora',  # Best for production
    rank=64,
    importance_method='taylor',   # Most reliable for ads
)

# Run with fine-tuning
enhanced_model = pipeline.run(
    validation_data=val_loader,
    top_k_layers=10,
    fine_tune_epochs=5,
    measure_qps=True
)

# For deployment: merge adapters
for module in enhanced_model.modules():
    if hasattr(module, 'merge'):
        module.merge()  # Zero runtime overhead!

# Verify QPS
from matrix_decomp.benchmarks import measure_qps
qps = measure_qps(enhanced_model, input_shape, device='cuda')
print(f"Production QPS: {qps['mean_qps']:.0f}")
```

## Troubleshooting

### QPS Still Low After Transfer?

1. **Check if adapters are merged**
   ```python
   # Before deployment
   for name, module in model.named_modules():
       if hasattr(module, 'merged') and not module.merged:
           module.merge()
           print(f"Merged: {name}")
   ```

2. **Profile layer latencies**
   ```python
   from matrix_decomp.benchmarks import profile_layer_latency
   latencies = profile_layer_latency(model, input_shape)
   # Find the bottleneck
   ```

3. **Reduce rank for slow layers**
   - Lower rank = faster but less accurate
   - Find the right trade-off

### Accuracy Dropped Too Much?

1. **Increase rank for important layers**
2. **Use DoRA instead of standard LoRA**
3. **Fine-tune for more epochs**
4. **Use AdaptiveRankDecomposer**

### Memory Issues?

1. **Use lower ranks**
2. **Process layers sequentially**
3. **Use RandomizedSVD for large matrices**

## Conclusion

For ads model distillation with QPS constraints:

1. **Use LoRA-based methods** (MergedLoRA or DoRA)
2. **Merge adapters before deployment**
3. **Select rank based on layer importance**
4. **Always benchmark QPS before production**

The key insight: **Don't let FLOPs fool you.** Lower FLOPs doesn't mean faster inference. Sequential operations, memory access patterns, and kernel overhead matter more.
