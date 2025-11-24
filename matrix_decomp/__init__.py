"""
Matrix Decomposition for Model Distillation

A comprehensive framework for knowledge transfer from large models to small models
using various matrix decomposition techniques.

Key Features:
- Multiple decomposition methods (SVD, LoRA, Tucker, CP, Fused SVD, and more)
- QPS-optimized methods for production deployment
- 30+ importance scoring methods for layer selection
- Automated pipeline for end-to-end knowledge transfer
- Comprehensive benchmarking tools

Quick Start:
    >>> from matrix_decomp import ModelDistillationPipeline
    >>> pipeline = ModelDistillationPipeline(
    ...     large_model=large_model,
    ...     small_model=small_model,
    ...     decomposition_method='lora',  # Best for QPS
    ...     rank=64
    ... )
    >>> enhanced_model = pipeline.run(validation_data=val_loader)
"""

__version__ = "0.2.0"

# Decomposition methods
from .decomposition import (
    BaseDecomposer,
    SVDDecomposer,
    LoRAAdapter,
    TuckerDecomposer,
    CPDecomposer,
    FusedSVDDecomposer,
)

# Importance scoring
from .importance import (
    compute_importance_scores,
    select_top_k_layers,
    print_importance_scores,
    GradientImportance,
    TaylorImportance,
    ActivationImportance,
    FisherImportance,
)

# Knowledge transfer
from .transfer import (
    KnowledgeTransfer,
    DimensionAdapter,
    adapt_dimensions,
)

# Pipeline
from .pipeline import (
    ModelDistillationPipeline,
    AutoPipeline,
)

# Benchmarking
from .benchmarks import (
    measure_qps,
    compare_qps,
    count_flops,
    compare_flops,
    compute_compression_ratio,
)

__all__ = [
    # Version
    "__version__",
    # Decomposition
    "BaseDecomposer",
    "SVDDecomposer",
    "LoRAAdapter",
    "TuckerDecomposer",
    "CPDecomposer",
    "FusedSVDDecomposer",
    # Importance
    "compute_importance_scores",
    "select_top_k_layers",
    "print_importance_scores",
    "GradientImportance",
    "TaylorImportance",
    "ActivationImportance",
    "FisherImportance",
    # Transfer
    "KnowledgeTransfer",
    "DimensionAdapter",
    "adapt_dimensions",
    # Pipeline
    "ModelDistillationPipeline",
    "AutoPipeline",
    # Benchmarking
    "measure_qps",
    "compare_qps",
    "count_flops",
    "compare_flops",
    "compute_compression_ratio",
]
