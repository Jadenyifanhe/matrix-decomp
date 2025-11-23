# Matrix Decomposition for Model Distillation

A comprehensive framework for knowledge transfer from large-scale models to compact models using matrix decomposition techniques. This approach improves prediction performance of small models without significantly increasing model size or degrading inference QPS.

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

## Why SVD Can Hurt QPS (Despite Lower FLOPs)

Many practitioners find that vanilla SVD gives prediction gains but surprisingly *degrades* QPS. Here's why:

### SVD Decomposition: `W ≈ U @ S @ V^T`
- **Original**: 1 matrix multiplication `y = W @ x` (shape: `[m, n] @ [n, 1]`)
- **SVD**: 2 sequential multiplications `y = U @ (S @ (V^T @ x))`
  - First: `V^T @ x` (shape: `[r, n] @ [n, 1]`)
  - Second: `(U @ S) @ result` (shape: `[m, r] @ [r, 1]`)

### Performance Bottlenecks:
1. **Sequential Operations**: Two matrix multiplications must happen in sequence, reducing parallelization
2. **Memory Access Patterns**: Two smaller matrices may have worse cache locality than one contiguous matrix
3. **Kernel Launch Overhead**: On GPUs, launching two kernels has overhead
4. **Memory Bandwidth**: Reading two matrices from memory can be slower than reading one
5. **Batch Processing**: Modern hardware is optimized for large matrix operations; smaller sequential ops are less efficient

### Better Alternatives:
- **LoRA/Adapter**: Keeps original matrix + adds low-rank residual (better parallelization)
- **Fused Operations**: Merge U and S into single matrix (`U @ S`)
- **Knowledge Distillation**: Use large model to train small model (no structural change)
- **Structured Pruning**: Remove entire neurons/channels (maintains single matrix ops)

## Supported Methods

### 1. **SVD (Singular Value Decomposition)**
- **Pros**: Mathematically optimal low-rank approximation, easy to implement
- **Cons**: Sequential operations hurt QPS, may lose performance on small ranks
- **Best for**: Offline compression when QPS is less critical
- **Implementation**: `decomposition/svd_decomposition.py`

### 2. **LoRA-Style Adapters (Low-Rank Adaptation)**
- **Pros**: Preserves original matrix, adds trainable residual, better QPS than SVD
- **Cons**: Slightly more memory (stores both original and adapter)
- **Best for**: Fine-tuning scenarios, when QPS is critical
- **Implementation**: `decomposition/lora_adapter.py`

### 3. **Tucker Decomposition**
- **Pros**: Higher compression for tensors, flexible rank selection per dimension
- **Cons**: More complex, requires careful tuning
- **Best for**: Convolutional layers, multi-dimensional weight tensors
- **Implementation**: `decomposition/tucker_decomposition.py`

### 4. **CP Decomposition (CANDECOMP/PARAFAC)**
- **Pros**: Maximum compression, separates all dimensions
- **Cons**: Less stable, may require more ranks to maintain accuracy
- **Best for**: High-dimensional tensors, extreme compression needs
- **Implementation**: `decomposition/cp_decomposition.py`

### 5. **Fused SVD (SVD with Pre-merged Matrices)**
- **Pros**: Reduces sequential operations, better QPS than vanilla SVD
- **Cons**: Still not as fast as LoRA in some cases
- **Best for**: Deployment scenarios where you want SVD benefits with better speed
- **Implementation**: `decomposition/fused_svd.py`

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

## Quick Start

### Example 1: SVD Decomposition and Transfer

```python
import torch
from matrix_decomp.decomposition.svd_decomposition import SVDDecomposer
from matrix_decomp.importance.neuron_importance import compute_importance_scores
from matrix_decomp.transfer.knowledge_transfer import KnowledgeTransfer

# Assume you have a large model and small model
large_model = YourLargeModel()
small_model = YourSmallModel()

# 1. Identify important layers
importance_scores = compute_importance_scores(
    large_model,
    validation_data,
    method='gradient'  # or 'activation', 'weight', 'fisher'
)

# 2. Select top-k important layers
top_layers = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:5]

# 3. Decompose important layers
decomposer = SVDDecomposer(rank=64)
decomposed_weights = {}
for layer_name, _ in top_layers:
    weight = large_model.get_parameter(layer_name)
    U, S, Vt = decomposer.decompose(weight)
    decomposed_weights[layer_name] = (U, S, Vt)

# 4. Transfer to small model
transfer = KnowledgeTransfer(method='svd')
transfer.inject_decomposed_weights(small_model, decomposed_weights)

# 5. Fine-tune if needed
fine_tune(small_model, train_data, epochs=5)
```

### Example 2: LoRA-Style Adapter (Recommended for QPS)

```python
from matrix_decomp.decomposition.lora_adapter import LoRAAdapter

# Create LoRA adapter from large model weights
adapter = LoRAAdapter(rank=64, alpha=16)

for layer_name in important_layers:
    large_weight = large_model.get_parameter(layer_name)
    small_weight = small_model.get_parameter(layer_name)

    # Compute residual: large_weight - small_weight
    residual = large_weight[:small_weight.shape[0], :small_weight.shape[1]] - small_weight

    # Decompose residual into low-rank adapter
    A, B = adapter.decompose_residual(residual)

    # Attach adapter to small model
    adapter.attach_to_layer(small_model, layer_name, A, B)

# The small model now has adapters that add the knowledge from large model
```

### Example 3: Complete Pipeline

```python
from matrix_decomp.pipeline import ModelDistillationPipeline

# Initialize pipeline
pipeline = ModelDistillationPipeline(
    large_model=large_model,
    small_model=small_model,
    decomposition_method='lora',  # or 'svd', 'tucker', 'cp'
    rank=64,
    importance_method='gradient'
)

# Run the full pipeline
enhanced_model = pipeline.run(
    validation_data=val_loader,
    top_k_layers=10,
    fine_tune_epochs=5,
    measure_qps=True  # Benchmark QPS before/after
)

# Get report
pipeline.print_report()
```

## Project Structure

```
matrix-decomp/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup
├── matrix_decomp/                    # Main package
│   ├── __init__.py
│   ├── decomposition/               # Decomposition methods
│   │   ├── __init__.py
│   │   ├── base.py                 # Base decomposer class
│   │   ├── svd_decomposition.py    # SVD implementation
│   │   ├── lora_adapter.py         # LoRA-style adapters
│   │   ├── tucker_decomposition.py # Tucker decomposition
│   │   ├── cp_decomposition.py     # CP decomposition
│   │   └── fused_svd.py           # Fused SVD for better QPS
│   ├── importance/                  # Importance scoring
│   │   ├── __init__.py
│   │   ├── neuron_importance.py    # Neuron-level importance
│   │   ├── gradient_based.py       # Gradient-based scoring
│   │   ├── activation_based.py     # Activation-based scoring
│   │   └── fisher_information.py   # Fisher information
│   ├── transfer/                    # Knowledge transfer
│   │   ├── __init__.py
│   │   ├── knowledge_transfer.py   # Main transfer logic
│   │   └── dimension_adapter.py    # Handle dimension mismatches
│   ├── benchmarks/                  # Performance benchmarking
│   │   ├── __init__.py
│   │   ├── qps_benchmark.py       # QPS measurement
│   │   └── flops_counter.py       # FLOPS counting
│   └── pipeline.py                 # End-to-end pipeline
├── examples/                        # Example scripts
│   ├── basic_svd_example.py
│   ├── lora_adapter_example.py
│   ├── full_pipeline_example.py
│   └── ads_model_example.py        # Specific to ads models
└── tests/                          # Unit tests
    ├── test_decomposition.py
    ├── test_importance.py
    └── test_transfer.py
```

## Method Comparison

| Method | Prediction Gain | QPS Impact | Memory | Complexity | Best Use Case |
|--------|----------------|------------|--------|------------|---------------|
| **Vanilla SVD** | Medium | ⚠️ **Negative** | Low | Low | Offline compression |
| **Fused SVD** | Medium | Neutral | Low | Low | Deployment |
| **LoRA Adapter** | High | ✅ **Positive/Neutral** | Medium | Low | Production (Recommended) |
| **Tucker** | High | Neutral | Medium | High | CNN layers |
| **CP** | Medium-High | Positive | Low | High | Extreme compression |

## Performance Optimization Tips

### For Better QPS:
1. **Use LoRA instead of vanilla SVD** - adds residual in parallel rather than sequential ops
2. **Fuse matrix operations** - merge `U @ S` into single matrix
3. **Batch normalize adapters** - fold batch norm into adapter weights
4. **Use appropriate rank** - too low hurts accuracy, too high hurts speed
5. **Profile before deploying** - always measure actual QPS, not just FLOPs

### For Better Prediction:
1. **Select important layers carefully** - use gradient-based importance
2. **Use higher ranks for critical layers** - not all layers need same rank
3. **Fine-tune after transfer** - a few epochs can recover lost performance
4. **Combine methods** - SVD for some layers, LoRA for others
5. **Regularize adapters** - prevent overfitting during fine-tuning

## Advanced Topics

### Handling Dimension Mismatches
When large model has different dimensions than small model:
```python
from matrix_decomp.transfer.dimension_adapter import adapt_dimensions

# Large model: [1024, 2048], Small model: [512, 1024]
U_large, S, Vt_large = decompose(large_weight)  # U: [1024, r], Vt: [r, 2048]

# Adapt to small dimensions
U_small = adapt_dimensions(U_large, target_rows=512, method='truncate')
Vt_small = adapt_dimensions(Vt_large, target_cols=1024, method='truncate')
```

### Layer-Specific Ranks
Different layers may need different ranks:
```python
rank_config = {
    'attention.query': 128,  # High rank for attention
    'attention.key': 128,
    'attention.value': 128,
    'ffn.dense1': 64,       # Medium rank for FFN
    'ffn.dense2': 64,
    'output': 32            # Low rank for output
}
```

### Combining with Other Techniques
- **Quantization**: Apply INT8 quantization after decomposition
- **Pruning**: Prune less important neurons before decomposition
- **Knowledge Distillation**: Use large model as teacher during fine-tuning

## Citation

If you use this framework in your research, please cite:
```bibtex
@software{matrix_decomp_distillation,
  title={Matrix Decomposition for Model Distillation},
  author={Your Team},
  year={2025},
  url={https://github.com/yourusername/matrix-decomp}
}
```

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.

## License

MIT License - see `LICENSE` file for details.

## Acknowledgments

- SVD and matrix factorization theory
- LoRA: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Tucker/CP decomposition: Tensor decomposition literature
- Neuron importance: Various pruning and neural architecture search papers
