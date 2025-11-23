"""
Matrix Decomposition for Model Distillation

A comprehensive framework for knowledge transfer from large models to small models
using various matrix decomposition techniques.
"""

__version__ = "0.1.0"

from .decomposition import (
    SVDDecomposer,
    LoRAAdapter,
    TuckerDecomposer,
    CPDecomposer,
    FusedSVDDecomposer,
)

from .importance import (
    compute_importance_scores,
    GradientImportance,
    ActivationImportance,
    FisherImportance,
)

from .transfer import (
    KnowledgeTransfer,
    DimensionAdapter,
)

__all__ = [
    "SVDDecomposer",
    "LoRAAdapter",
    "TuckerDecomposer",
    "CPDecomposer",
    "FusedSVDDecomposer",
    "compute_importance_scores",
    "GradientImportance",
    "ActivationImportance",
    "FisherImportance",
    "KnowledgeTransfer",
    "DimensionAdapter",
]
