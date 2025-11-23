"""
Importance scoring methods for identifying critical layers to decompose.

This module provides extensive methods for measuring layer importance:

Category 1: Gradient-Based Methods
- GradientImportance: Gradient magnitude
- TaylorImportance: Taylor expansion approximation
- FisherImportance: Fisher Information Matrix
- EmpiricalFisherImportance: Empirical Fisher

Category 2: Activation-Based Methods
- ActivationImportance: Activation statistics
- SensitivityImportance: Input perturbation sensitivity

Category 3: Pruning-Based Methods
- SNIPImportance: Connection sensitivity
- GraSPImportance: Gradient signal preservation
- SynFlowImportance: Synaptic flow
- MovementPruningImportance: Parameter movement
- MagnitudePruningImportance: Weight magnitude

Category 4: Hessian-Based Methods
- HessianDiagonalImportance: Hessian diagonal
- OptimalBrainDamageImportance: OBD score
- BlockwiseHessianImportance: Block Hessian trace

Category 5: Ablation-Based Methods
- LayerKnockoutImportance: Remove layers
- WeightPerturbationImportance: Add noise
- ChannelAblationImportance: Remove channels
- DropoutBasedImportance: Monte Carlo dropout

Category 6: Information-Theoretic Methods
- MutualInformationImportance: MI with output
- EntropyBasedImportance: Activation entropy
- RepresentationSimilarityImportance: RSA
- ActivationSparsityImportance: APoZ score

Category 7: Layer Geometry Methods
- EffectiveRankImportance: Effective rank
- ConditionNumberImportance: Condition number
- SpectralNormImportance: Largest singular value
- NuclearNormImportance: Sum of singular values
- StableRankImportance: Stable rank
- LipschitzConstantImportance: Lipschitz constant
"""

# Core interface
from .neuron_importance import compute_importance_scores, select_top_k_layers, print_importance_scores

# Gradient-based
from .gradient_based import GradientImportance, TaylorImportance

# Activation-based
from .activation_based import ActivationImportance, SensitivityImportance

# Fisher Information
from .fisher_information import FisherImportance, EWCImportance

# Pruning-based
from .pruning_based import (
    SNIPImportance,
    GraSPImportance,
    SynFlowImportance,
    MovementPruningImportance,
    MagnitudePruningImportance,
)

# Hessian-based
from .hessian_based import (
    HessianDiagonalImportance,
    OptimalBrainDamageImportance,
    EmpiricalFisherImportance,
    BlockwiseHessianImportance,
)

# Ablation-based
from .ablation_based import (
    LayerKnockoutImportance,
    WeightPerturbationImportance,
    ChannelAblationImportance,
    DropoutBasedImportance,
)

# Information-theoretic
from .information_theoretic import (
    MutualInformationImportance,
    EntropyBasedImportance,
    RepresentationSimilarityImportance,
    ActivationSparsityImportance,
)

# Layer geometry
from .layer_geometry import (
    EffectiveRankImportance,
    ConditionNumberImportance,
    SpectralNormImportance,
    NuclearNormImportance,
    StableRankImportance,
    LipschitzConstantImportance,
)

__all__ = [
    # Core
    "compute_importance_scores",
    "select_top_k_layers",
    "print_importance_scores",

    # Gradient-based
    "GradientImportance",
    "TaylorImportance",

    # Activation-based
    "ActivationImportance",
    "SensitivityImportance",

    # Fisher
    "FisherImportance",
    "EWCImportance",

    # Pruning-based
    "SNIPImportance",
    "GraSPImportance",
    "SynFlowImportance",
    "MovementPruningImportance",
    "MagnitudePruningImportance",

    # Hessian-based
    "HessianDiagonalImportance",
    "OptimalBrainDamageImportance",
    "EmpiricalFisherImportance",
    "BlockwiseHessianImportance",

    # Ablation-based
    "LayerKnockoutImportance",
    "WeightPerturbationImportance",
    "ChannelAblationImportance",
    "DropoutBasedImportance",

    # Information-theoretic
    "MutualInformationImportance",
    "EntropyBasedImportance",
    "RepresentationSimilarityImportance",
    "ActivationSparsityImportance",

    # Layer geometry
    "EffectiveRankImportance",
    "ConditionNumberImportance",
    "SpectralNormImportance",
    "NuclearNormImportance",
    "StableRankImportance",
    "LipschitzConstantImportance",
]
