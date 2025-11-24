"""
Matrix decomposition methods for model compression and knowledge transfer.

Available Methods (by QPS impact):
----------------------------------
BEST QPS (maintains or improves):
- LoRAAdapter: Low-rank adapters (RECOMMENDED for production)
- MergedLoRADecomposer: LoRA merged into weights (zero overhead)
- DoRAAdapter: Weight-decomposed LoRA (better accuracy)

GOOD QPS (slight overhead):
- FusedSVDDecomposer: SVD with merged singular values
- QRDecomposer: QR-based alternative to SVD
- RandomizedSVDDecomposer: Fast SVD for large matrices

NEUTRAL/VARIABLE QPS:
- TuckerDecomposer: For conv layers (4D tensors)
- CPDecomposer: Maximum compression
- KroneckerFactorization: For structured matrices

CAUTION - MAY HURT QPS:
- SVDDecomposer: Vanilla SVD (sequential operations)
- SemiStructuredDecomposition: Needs hardware support
"""

# Base class
from .base import BaseDecomposer

# Standard decomposition methods
from .svd_decomposition import SVDDecomposer, TruncatedSVD
from .lora_adapter import LoRAAdapter, LoRALayer
from .tucker_decomposition import TuckerDecomposer, TuckerLayer
from .cp_decomposition import CPDecomposer, CPLayer
from .fused_svd import FusedSVDDecomposer, FusedSVDLayer, AdaptiveFusedSVD

# QPS-optimized methods
from .qps_optimized import (
    DoRAAdapter,
    DoRALayer,
    MergedLoRADecomposer,
    MergedLoRALayer,
    RandomizedSVDDecomposer,
    RandomizedSVDLayer,
    QRDecomposer,
    KroneckerFactorization,
    SemiStructuredDecomposition,
)

# Hybrid approaches
from .hybrid import (
    HybridDecomposer,
    AdaptiveRankDecomposer,
    LayerWiseDecomposer,
    ImportanceWeightedDecomposer,
)

__all__ = [
    # Base
    "BaseDecomposer",

    # Standard methods
    "SVDDecomposer",
    "TruncatedSVD",
    "LoRAAdapter",
    "LoRALayer",
    "TuckerDecomposer",
    "TuckerLayer",
    "CPDecomposer",
    "CPLayer",
    "FusedSVDDecomposer",
    "FusedSVDLayer",
    "AdaptiveFusedSVD",

    # QPS-optimized
    "DoRAAdapter",
    "DoRALayer",
    "MergedLoRADecomposer",
    "MergedLoRALayer",
    "RandomizedSVDDecomposer",
    "RandomizedSVDLayer",
    "QRDecomposer",
    "KroneckerFactorization",
    "SemiStructuredDecomposition",

    # Hybrid approaches
    "HybridDecomposer",
    "AdaptiveRankDecomposer",
    "LayerWiseDecomposer",
    "ImportanceWeightedDecomposer",
]
