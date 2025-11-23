# Comprehensive Guide to Importance Scoring Methods

This guide provides detailed information about all 30+ importance scoring methods implemented in the framework, helping you choose the right method for your use case.

## Table of Contents

1. [Quick Selection Guide](#quick-selection-guide)
2. [Method Categories](#method-categories)
3. [Detailed Method Descriptions](#detailed-method-descriptions)
4. [Performance Comparison](#performance-comparison)
5. [Usage Examples](#usage-examples)
6. [Method Selection Decision Tree](#method-selection-decision-tree)

## Quick Selection Guide

### For Most Users (Recommended)

| Use Case | Recommended Method | Why |
|----------|-------------------|-----|
| **General purpose** | `GradientImportance` | Fast, reliable, works for all architectures |
| **Production/Ads models** | `TaylorImportance` or `SNIPImportance` | Best balance of accuracy and speed |
| **High accuracy needed** | `LayerKnockoutImportance` | Most accurate but slowest |
| **No labels available** | `SynFlowImportance` or `EffectiveRankImportance` | Data-independent methods |
| **CNN/Vision models** | `ChannelAblationImportance` | Designed for convolutional layers |
| **After training** | `MovementPruningImportance` | Requires before/after weights |
| **Memory constrained** | `SpectralNormImportance` | No data pass needed |

### Method Characteristics Matrix

| Method | Speed | Accuracy | Data Needed | Labels Needed | GPU Friendly |
|--------|-------|----------|-------------|---------------|--------------|
| **Gradient** | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | ✓ Yes | ✓ Yes | ✓ Yes |
| **Taylor** | ⚡⚡ Medium | ⭐⭐⭐⭐ Excellent | ✓ Yes | ✓ Yes | ✓ Yes |
| **SNIP** | ⚡⚡⚡ Fast | ⭐⭐⭐⭐ Excellent | ✓ Yes | ✓ Yes | ✓ Yes |
| **SynFlow** | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | ✗ No | ✗ No | ✓ Yes |
| **Layer Knockout** | ⚡ Slow | ⭐⭐⭐⭐⭐ Best | ✓ Yes | ✓ Yes | ✓ Yes |
| **Spectral Norm** | ⚡⚡⚡⚡ Very Fast | ⭐⭐ Fair | ✗ No | ✗ No | ✓ Yes |
| **Fisher** | ⚡⚡ Medium | ⭐⭐⭐⭐ Excellent | ✓ Yes | ✓ Yes | ✓ Yes |

## Method Categories

### 1. Gradient-Based Methods

These methods use gradient information to measure importance.

#### `GradientImportance`
- **Metric**: Magnitude of gradients flowing through each layer
- **Intuition**: Layers with larger gradients have more influence on the loss
- **Pros**: Fast, simple, widely applicable
- **Cons**: Can be noisy, depends on current training state
- **Best for**: General purpose, quick analysis
- **Computational cost**: O(1 forward + 1 backward pass)

```python
from matrix_decomp.importance import GradientImportance

scorer = GradientImportance(aggregate='mean')  # or 'max', 'sum'
scores = scorer.compute_scores(model, data_loader, criterion, device='cuda')
```

#### `TaylorImportance`
- **Metric**: |weight × gradient| (Taylor expansion approximation)
- **Intuition**: Estimates change in loss if weight were removed
- **Pros**: More accurate than pure gradient, accounts for weight magnitude
- **Cons**: Slightly slower than gradient
- **Best for**: Production use, ads models
- **Computational cost**: O(1 forward + 1 backward pass)
- **Paper**: "Importance Estimation for Neural Network Pruning" (Molchanov et al., 2019)

```python
from matrix_decomp.importance import TaylorImportance

scorer = TaylorImportance()
scores = scorer.compute_scores(model, data_loader, criterion)
```

### 2. Pruning-Based Methods

Methods derived from neural network pruning research.

#### `SNIPImportance`
- **Metric**: Connection sensitivity at initialization
- **Intuition**: Measures how sensitive outputs are to each connection
- **Pros**: Works at initialization, no training needed
- **Cons**: Assumes model at initialization
- **Best for**: Analyzing model architecture, early-stage analysis
- **Computational cost**: O(1 batch × 1 backward pass)
- **Paper**: "SNIP: Single-shot Network Pruning" (Lee et al., ICLR 2019)

```python
from matrix_decomp.importance import SNIPImportance

scorer = SNIPImportance()
scores = scorer.compute_scores(model, data_loader, criterion, num_batches=1)
```

#### `GraSPImportance`
- **Metric**: Gradient signal preservation
- **Intuition**: Preserves gradient flow by keeping important connections
- **Pros**: Theoretically grounded, works at initialization
- **Cons**: More expensive than SNIP, complex computation
- **Best for**: Architecture analysis, pruning at initialization
- **Computational cost**: O(2 passes × num_batches)
- **Paper**: "Picking Winning Tickets Before Training" (Wang et al., ICLR 2020)

```python
from matrix_decomp.importance import GraSPImportance

scorer = GraSPImportance()
scores = scorer.compute_scores(model, data_loader, criterion, num_batches=1)
```

#### `SynFlowImportance`
- **Metric**: Synaptic saliency (data-independent)
- **Intuition**: Uses path products to identify important connections
- **Pros**: **No data needed!** Works at initialization
- **Cons**: Data-independent may be less accurate
- **Best for**: When you have no data, architecture search
- **Computational cost**: O(iterations × 1 forward + 1 backward)
- **Paper**: "Pruning at Initialization" (Tanaka et al., ICLR 2020)

```python
from matrix_decomp.importance import SynFlowImportance

scorer = SynFlowImportance()
# Only needs input shape, no data!
scores = scorer.compute_scores(model, input_shape=(3, 224, 224))
```

#### `MovementPruningImportance`
- **Metric**: Parameter movement during training
- **Intuition**: Parameters moving towards zero are less important
- **Pros**: Captures training dynamics
- **Cons**: Requires tracking weights before training
- **Best for**: Post-training analysis, iterative improvement
- **Computational cost**: O(weight comparison)
- **Paper**: "Movement Pruning" (Sanh et al., NeurIPS 2020)

```python
from matrix_decomp.importance import MovementPruningImportance

scorer = MovementPruningImportance()
# Before training:
scorer.save_initial_weights(model)
# ... train model ...
# After training:
scores = scorer.compute_scores(model)
```

### 3. Hessian-Based Methods

Second-order methods using Hessian (second derivative) information.

#### `HessianDiagonalImportance`
- **Metric**: Diagonal of Hessian matrix
- **Intuition**: Curvature of loss landscape
- **Pros**: Second-order information, more precise
- **Cons**: Expensive to compute, approximation
- **Best for**: When accuracy is critical, research
- **Computational cost**: O(num_samples × 2 passes)

```python
from matrix_decomp.importance import HessianDiagonalImportance

scorer = HessianDiagonalImportance()
scores = scorer.compute_scores(model, data_loader, criterion, max_batches=10)
```

#### `OptimalBrainDamageImportance`
- **Metric**: 0.5 × weight² × Hessian_diagonal
- **Intuition**: Estimated loss increase if parameter removed
- **Pros**: Classic method, theoretically sound
- **Cons**: Expensive, needs many samples
- **Best for**: Research, understanding model sensitivity
- **Computational cost**: O(num_samples × 2 passes)
- **Paper**: "Optimal Brain Damage" (LeCun et al., 1990)

```python
from matrix_decomp.importance import OptimalBrainDamageImportance

scorer = OptimalBrainDamageImportance()
scores = scorer.compute_scores(model, data_loader, criterion, max_batches=10)
```

### 4. Ablation-Based Methods

Methods that remove components and measure performance impact.

#### `LayerKnockoutImportance` ⭐ Most Accurate
- **Metric**: Performance drop when layer is zeroed
- **Intuition**: Direct measurement of layer contribution
- **Pros**: **Most accurate**, measures actual impact
- **Cons**: **Very slow** (O(num_layers × evaluation))
- **Best for**: Final validation, critical decisions
- **Computational cost**: O(num_layers × evaluation_time)

```python
from matrix_decomp.importance import LayerKnockoutImportance

scorer = LayerKnockoutImportance()
scores = scorer.compute_scores(
    model, data_loader, criterion,
    eval_metric='loss',  # or 'accuracy'
    max_batches=10
)
```

#### `WeightPerturbationImportance`
- **Metric**: Sensitivity to weight noise
- **Intuition**: Important layers are more sensitive to perturbations
- **Pros**: Measures robustness
- **Cons**: Requires multiple trials, slow
- **Best for**: Understanding model robustness
- **Computational cost**: O(num_layers × num_trials × evaluation)

```python
from matrix_decomp.importance import WeightPerturbationImportance

scorer = WeightPerturbationImportance()
scores = scorer.compute_scores(
    model, data_loader, criterion,
    perturbation_scale=0.1,
    num_trials=5
)
```

#### `ChannelAblationImportance`
- **Metric**: Performance drop when channels removed
- **Intuition**: Important channels contribute more to output
- **Pros**: Specific to CNNs, channel-level granularity
- **Cons**: Only for Conv layers, expensive
- **Best for**: CNN optimization, channel pruning
- **Computational cost**: O(num_channels_sampled × evaluation)

```python
from matrix_decomp.importance import ChannelAblationImportance

scorer = ChannelAblationImportance()
scores = scorer.compute_scores(
    model, data_loader, criterion,
    sample_ratio=0.1  # Test 10% of channels
)
```

### 5. Information-Theoretic Methods

Methods based on information theory concepts.

#### `MutualInformationImportance`
- **Metric**: I(Layer_activations; Output)
- **Intuition**: How much information about output is in each layer
- **Pros**: Theoretically grounded, captures dependencies
- **Cons**: Requires binning, approximation, slow
- **Best for**: Understanding information flow
- **Computational cost**: O(data pass + histogram computation)

```python
from matrix_decomp.importance import MutualInformationImportance

scorer = MutualInformationImportance()
scores = scorer.compute_scores(
    model, data_loader,
    num_bins=20,  # For histogram estimation
    max_batches=50
)
```

#### `EntropyBasedImportance`
- **Metric**: H(activations) = -Σ p(x) log p(x)
- **Intuition**: High entropy = more diverse information
- **Pros**: Simple, interpretable
- **Cons**: Doesn't account for output relevance
- **Best for**: Analyzing feature diversity
- **Computational cost**: O(data pass + histogram computation)

```python
from matrix_decomp.importance import EntropyBasedImportance

scorer = EntropyBasedImportance()
scores = scorer.compute_scores(model, data_loader, num_bins=50)
```

#### `ActivationSparsityImportance` (APoZ)
- **Metric**: 1 - (percentage of zero activations)
- **Intuition**: Less sparse = more active = more important
- **Pros**: Fast, simple, works for ReLU networks
- **Cons**: Biased towards ReLU, ignores magnitude
- **Best for**: ReLU networks, quick analysis
- **Computational cost**: O(1 data pass)
- **Paper**: "Network Trimming" (Hu et al., 2016)

```python
from matrix_decomp.importance import ActivationSparsityImportance

scorer = ActivationSparsityImportance()
scores = scorer.compute_scores(model, data_loader)
```

### 6. Layer Geometry Methods

Methods analyzing geometric properties of weight matrices.

#### `EffectiveRankImportance`
- **Metric**: exp(H(σ)) where H is entropy of singular values
- **Intuition**: "True" dimensionality of layer representations
- **Pros**: **No data needed**, measures layer capacity
- **Cons**: Doesn't account for task relevance
- **Best for**: Architecture analysis, understanding capacity
- **Computational cost**: O(SVD per layer)

```python
from matrix_decomp.importance import EffectiveRankImportance

scorer = EffectiveRankImportance()
scores = scorer.compute_scores(model)  # No data needed!
```

#### `SpectralNormImportance`
- **Metric**: Largest singular value σ_max
- **Intuition**: Maximum amplification of layer
- **Pros**: **Fast**, no data needed, stable
- **Cons**: Only considers largest singular value
- **Best for**: Quick analysis, Lipschitz constraints
- **Computational cost**: O(power iteration per layer)

```python
from matrix_decomp.importance import SpectralNormImportance

scorer = SpectralNormImportance()
scores = scorer.compute_scores(model)  # Very fast!
```

#### `NuclearNormImportance`
- **Metric**: Sum of all singular values Σσ_i
- **Intuition**: Total capacity of layer
- **Pros**: Considers all singular values
- **Cons**: Needs full SVD
- **Best for**: Analyzing overall layer capacity
- **Computational cost**: O(SVD per layer)

```python
from matrix_decomp.importance import NuclearNormImportance

scorer = NuclearNormImportance()
scores = scorer.compute_scores(model)
```

#### `StableRankImportance`
- **Metric**: ||W||²_F / ||W||²_2 (Frobenius / Spectral²)
- **Intuition**: Effective number of non-zero singular values
- **Pros**: More stable than rank, robust measure
- **Cons**: Still an approximation
- **Best for**: Robust capacity estimation
- **Computational cost**: O(Frobenius norm + power iteration)

```python
from matrix_decomp.importance import StableRankImportance

scorer = StableRankImportance()
scores = scorer.compute_scores(model)
```

## Performance Comparison

### Computational Cost Ranking (Fastest to Slowest)

1. **SpectralNormImportance** - No data, power iteration only
2. **EffectiveRankImportance** - No data, SVD only
3. **SynFlowImportance** - No data, few iterations
4. **GradientImportance** - 1 forward + 1 backward
5. **TaylorImportance** - 1 forward + 1 backward
6. **SNIPImportance** - 1 batch, 1 backward
7. **ActivationImportance** - 1 forward pass
8. **FisherImportance** - Multiple batches, 1 backward each
9. **GraSPImportance** - 2 passes per batch
10. **HessianDiagonalImportance** - Sample-wise gradients
11. **WeightPerturbationImportance** - Multiple trials × evaluation
12. **LayerKnockoutImportance** - Num_layers × full evaluation ⚠️ **Slowest**

### Accuracy Ranking (Best to Good)

1. **LayerKnockoutImportance** - ⭐⭐⭐⭐⭐ Ground truth
2. **TaylorImportance** - ⭐⭐⭐⭐ Excellent
3. **OptimalBrainDamageImportance** - ⭐⭐⭐⭐ Excellent
4. **SNIPImportance** - ⭐⭐⭐⭐ Excellent
5. **Fisher/GraSPImportance** - ⭐⭐⭐ Good
6. **GradientImportance** - ⭐⭐⭐ Good
7. **MutualInformationImportance** - ⭐⭐⭐ Good
8. **ActivationImportance** - ⭐⭐ Fair
9. **SpectralNormImportance** - ⭐⭐ Fair
10. **EntropyImportance** - ⭐⭐ Fair

## Method Selection Decision Tree

```
START: Need to identify important layers for decomposition
│
├─ Q: Do you have labeled data available?
│  │
│  ├─ NO → Use data-independent methods:
│  │   ├─ Fast needed? → SpectralNormImportance
│  │   ├─ Better accuracy? → SynFlowImportance
│  │   └─ Analyze capacity? → EffectiveRankImportance
│  │
│  └─ YES → Continue...
│
├─ Q: How much computational budget do you have?
│  │
│  ├─ VERY LIMITED (minutes) →
│  │   └─ GradientImportance or SpectralNormImportance
│  │
│  ├─ MODERATE (hours) →
│  │   ├─ For ads/production → TaylorImportance (RECOMMENDED)
│  │   ├─ At initialization → SNIPImportance
│  │   └─ After training → MovementPruningImportance
│  │
│  └─ HIGH (can wait) →
│      ├─ Best accuracy → LayerKnockoutImportance
│      └─ Second-order info → OptimalBrainDamageImportance
│
├─ Q: What type of model?
│  │
│  ├─ CNN/Vision →
│  │   ├─ Channel-level → ChannelAblationImportance
│  │   └─ Layer-level → TaylorImportance or GradientImportance
│  │
│  ├─ Transformer/NLP →
│  │   ├─ Attention layers → MutualInformationImportance
│  │   └─ FFN layers → TaylorImportance
│  │
│  └─ MLPs/Ads models → TaylorImportance (BEST CHOICE)
│
└─ Q: Special requirements?
    │
    ├─ Need interpretability → MutualInformationImportance or EntropyImportance
    ├─ Robustness analysis → WeightPerturbationImportance
    ├─ Training dynamics → MovementPruningImportance
    └─ Theoretical guarantees → OptimalBrainDamageImportance or FisherImportance
```

## Usage Examples

### Example 1: Quick Analysis (Recommended for most users)

```python
from matrix_decomp.importance import TaylorImportance, select_top_k_layers

# Fast and accurate for most use cases
scorer = TaylorImportance()
scores = scorer.compute_scores(model, val_loader, criterion, device='cuda')

# Select top-10 layers
top_layers = select_top_k_layers(scores, k=10)
print("Top 10 important layers:")
for name, score in top_layers:
    print(f"  {name}: {score:.4f}")
```

### Example 2: No Data Available

```python
from matrix_decomp.importance import SynFlowImportance

# No data needed!
scorer = SynFlowImportance()
scores = scorer.compute_scores(model, input_shape=(3, 224, 224))

print("Importance without any data:")
for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {name}: {score:.4f}")
```

### Example 3: Maximum Accuracy (for critical decisions)

```python
from matrix_decomp.importance import LayerKnockoutImportance

# Most accurate but slowest
scorer = LayerKnockoutImportance()
scores = scorer.compute_scores(
    model,
    val_loader,
    criterion,
    eval_metric='accuracy',  # or 'loss'
    max_batches=50,  # More batches = more accurate
    verbose=True
)
```

### Example 4: Comparing Multiple Methods

```python
from matrix_decomp.importance import (
    GradientImportance,
    TaylorImportance,
    SNIPImportance,
    EffectiveRankImportance
)

methods = {
    'Gradient': GradientImportance(),
    'Taylor': TaylorImportance(),
    'SNIP': SNIPImportance(),
    'EffectiveRank': EffectiveRankImportance(),
}

results = {}
for name, scorer in methods.items():
    print(f"\nComputing {name} importance...")

    if name == 'EffectiveRank':
        # No data needed
        scores = scorer.compute_scores(model)
    else:
        # Needs data
        scores = scorer.compute_scores(model, val_loader, criterion)

    results[name] = scores

# Compare agreement between methods
from scipy.stats import spearmanr

print("\nMethod Agreement (Spearman correlation):")
method_names = list(results.keys())
for i, name1 in enumerate(method_names):
    for name2 in method_names[i+1:]:
        # Get common layers
        common_layers = set(results[name1].keys()) & set(results[name2].keys())
        scores1 = [results[name1][layer] for layer in common_layers]
        scores2 = [results[name2][layer] for layer in common_layers]

        corr, _ = spearmanr(scores1, scores2)
        print(f"  {name1} vs {name2}: {corr:.3f}")
```

### Example 5: For Ads Models (Production Use Case)

```python
from matrix_decomp.importance import TaylorImportance, print_importance_scores

# Taylor is best for ads models: fast + accurate
scorer = TaylorImportance()

# Use a small validation set for speed
scores = scorer.compute_scores(
    ads_model,
    val_loader,
    criterion=nn.BCELoss(),  # For CTR prediction
    device='cuda',
    max_batches=100,  # Enough for good estimate
    verbose=True
)

# Print top layers
print_importance_scores(scores, top_k=10)

# Select for decomposition
top_10 = select_top_k_layers(scores, k=10)
layer_names_to_decompose = [name for name, _ in top_10]
```

## Advanced: Ensemble Methods

Combine multiple methods for more robust importance estimation:

```python
from matrix_decomp.importance import (
    GradientImportance,
    TaylorImportance,
    EffectiveRankImportance,
    select_top_k_layers
)
import numpy as np

# Compute with multiple methods
gradient_scores = GradientImportance().compute_scores(model, val_loader, criterion)
taylor_scores = TaylorImportance().compute_scores(model, val_loader, criterion)
rank_scores = EffectiveRankImportance().compute_scores(model)

# Ensemble: average normalized scores
ensemble_scores = {}
all_layers = set(gradient_scores.keys()) | set(taylor_scores.keys()) | set(rank_scores.keys())

for layer in all_layers:
    scores = []
    if layer in gradient_scores:
        scores.append(gradient_scores[layer])
    if layer in taylor_scores:
        scores.append(taylor_scores[layer])
    if layer in rank_scores:
        scores.append(rank_scores[layer])

    ensemble_scores[layer] = np.mean(scores)

# Use ensemble scores
top_layers = select_top_k_layers(ensemble_scores, k=10)
```

## Summary Recommendations

### 🥇 Top Recommendations by Use Case

1. **Ads Models / Production**: `TaylorImportance`
   - Fast, accurate, well-tested

2. **No Data Available**: `SynFlowImportance` or `EffectiveRankImportance`
   - Data-independent, still effective

3. **CNN/Vision**: `ChannelAblationImportance` or `TaylorImportance`
   - Designed for convolutional layers

4. **Maximum Accuracy**: `LayerKnockoutImportance`
   - Ground truth importance, use for validation

5. **Quick Analysis**: `GradientImportance` or `SpectralNormImportance`
   - Fastest methods, good for exploration

6. **Research / Theory**: `OptimalBrainDamageImportance` or `FisherImportance`
   - Theoretically grounded

### ❌ Methods to Avoid (Usually)

- **Don't use** `LayerKnockoutImportance` for initial exploration (too slow)
- **Don't use** `MutualInformationImportance` for large models (memory intensive)
- **Don't use** data-free methods if you have data available
- **Don't use** `MovementPruningImportance` without baseline weights

## References

Key papers for each method category:

1. **SNIP**: Lee et al., "SNIP: Single-shot Network Pruning", ICLR 2019
2. **GraSP**: Wang et al., "Picking Winning Tickets Before Training", ICLR 2020
3. **SynFlow**: Tanaka et al., "Pruning Neural Networks at Initialization", ICLR 2020
4. **OBD**: LeCun et al., "Optimal Brain Damage", NeurIPS 1990
5. **Movement Pruning**: Sanh et al., "Movement Pruning", NeurIPS 2020
6. **Taylor Pruning**: Molchanov et al., "Importance Estimation for Neural Network Pruning", CVPR 2019
7. **APoZ**: Hu et al., "Network Trimming: A Data-Driven Neuron Pruning Approach", ICLR 2016
