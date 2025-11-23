"""
Unified interface for computing neuron/layer importance scores.

This module provides a simple API for computing importance scores using different methods.
"""

from typing import Dict, Union, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_importance_scores(
    model: nn.Module,
    data_loader: Optional[DataLoader] = None,
    method: str = "gradient",
    criterion: Optional[Callable] = None,
    device: str = "cpu",
    input_shape: Optional[tuple] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Unified interface for computing importance scores using 30+ methods.

    Args:
        model: PyTorch model to analyze
        data_loader: DataLoader (required for most methods, not all)
        method: Importance scoring method (see categories below)
        criterion: Loss function (required for some methods)
        device: Device to run computation on
        input_shape: Input shape (required for data-free methods like SynFlow)
        **kwargs: Additional method-specific arguments

    Methods Categories:
        Gradient-based:
            - 'gradient': Gradient magnitude (fast, general purpose)
            - 'taylor': Taylor expansion (RECOMMENDED for production)

        Pruning-based:
            - 'snip': Connection sensitivity
            - 'grasp': Gradient signal preservation
            - 'synflow': Synaptic flow (NO DATA NEEDED)
            - 'movement': Parameter movement
            - 'magnitude': Weight magnitude (NO DATA NEEDED)

        Hessian-based:
            - 'hessian': Hessian diagonal
            - 'obd': Optimal Brain Damage
            - 'fisher': Fisher Information
            - 'fisher_empirical': Empirical Fisher

        Ablation-based:
            - 'knockout': Layer knockout (MOST ACCURATE, SLOWEST)
            - 'perturbation': Weight perturbation
            - 'channel_ablation': Channel ablation (CNNs)
            - 'dropout': Dropout-based

        Information-theoretic:
            - 'mutual_info': Mutual information
            - 'entropy': Activation entropy
            - 'rsa': Representation similarity
            - 'sparsity': Activation sparsity (APoZ)

        Layer Geometry (NO DATA NEEDED):
            - 'effective_rank': Effective rank
            - 'condition_number': Condition number
            - 'spectral_norm': Spectral norm (FAST)
            - 'nuclear_norm': Nuclear norm
            - 'stable_rank': Stable rank
            - 'lipschitz': Lipschitz constant

    Returns:
        Dictionary mapping layer names to importance scores

    Examples:
        >>> # Fast general purpose
        >>> scores = compute_importance_scores(model, val_loader, method='gradient')

        >>> # Best for production/ads models
        >>> scores = compute_importance_scores(model, val_loader, method='taylor', criterion=loss_fn)

        >>> # No data available
        >>> scores = compute_importance_scores(model, method='synflow', input_shape=(3, 224, 224))
    """
    model = model.to(device)

    # Check if method needs data
    data_free_methods = ['synflow', 'effective_rank', 'condition_number', 'spectral_norm',
                         'nuclear_norm', 'stable_rank', 'lipschitz', 'magnitude', 'weight', 'movement']

    if method not in data_free_methods and data_loader is None:
        raise ValueError(f"Method '{method}' requires data_loader")

    model.eval()

    # Gradient-based methods
    if method == "gradient":
        from .gradient_based import GradientImportance
        scorer = GradientImportance()
        return scorer.compute_scores(model, data_loader, criterion, device, **kwargs)

    elif method == "taylor":
        from .gradient_based import TaylorImportance
        scorer = TaylorImportance()
        return scorer.compute_scores(model, data_loader, criterion, device, **kwargs)

    # Pruning-based methods
    elif method == "snip":
        from .pruning_based import SNIPImportance
        scorer = SNIPImportance()
        return scorer.compute_scores(model, data_loader, criterion, device, **kwargs)

    elif method == "grasp":
        from .pruning_based import GraSPImportance
        scorer = GraSPImportance()
        return scorer.compute_scores(model, data_loader, criterion, device, **kwargs)

    elif method == "synflow":
        from .pruning_based import SynFlowImportance
        if input_shape is None:
            raise ValueError("SynFlow requires input_shape parameter")
        scorer = SynFlowImportance()
        return scorer.compute_scores(model, input_shape, device, **kwargs)

    elif method == "movement":
        from .pruning_based import MovementPruningImportance
        scorer = MovementPruningImportance()
        return scorer.compute_scores(model, **kwargs)

    elif method in ["magnitude", "weight"]:
        from .pruning_based import MagnitudePruningImportance
        scorer = MagnitudePruningImportance()
        return scorer.compute_scores(model, **kwargs)

    # Hessian-based methods
    elif method == "hessian":
        from .hessian_based import HessianDiagonalImportance
        scorer = HessianDiagonalImportance()
        return scorer.compute_scores(model, data_loader, criterion, device, **kwargs)

    elif method == "obd":
        from .hessian_based import OptimalBrainDamageImportance
        scorer = OptimalBrainDamageImportance()
        return scorer.compute_scores(model, data_loader, criterion, device, **kwargs)

    elif method in ["fisher", "fisher_diagonal"]:
        from .fisher_information import FisherImportance
        scorer = FisherImportance()
        return scorer.compute_scores(model, data_loader, criterion, device, **kwargs)

    elif method == "fisher_empirical":
        from .hessian_based import EmpiricalFisherImportance
        scorer = EmpiricalFisherImportance()
        return scorer.compute_scores(model, data_loader, criterion, device, **kwargs)

    # Ablation-based methods
    elif method == "knockout":
        from .ablation_based import LayerKnockoutImportance
        scorer = LayerKnockoutImportance()
        return scorer.compute_scores(model, data_loader, criterion, device, **kwargs)

    elif method == "perturbation":
        from .ablation_based import WeightPerturbationImportance
        scorer = WeightPerturbationImportance()
        return scorer.compute_scores(model, data_loader, criterion, device, **kwargs)

    elif method == "channel_ablation":
        from .ablation_based import ChannelAblationImportance
        scorer = ChannelAblationImportance()
        return scorer.compute_scores(model, data_loader, criterion, device, **kwargs)

    elif method == "dropout":
        from .ablation_based import DropoutBasedImportance
        scorer = DropoutBasedImportance()
        return scorer.compute_scores(model, data_loader, device, **kwargs)

    # Information-theoretic methods
    elif method == "mutual_info":
        from .information_theoretic import MutualInformationImportance
        scorer = MutualInformationImportance()
        return scorer.compute_scores(model, data_loader, device, **kwargs)

    elif method == "entropy":
        from .information_theoretic import EntropyBasedImportance
        scorer = EntropyBasedImportance()
        return scorer.compute_scores(model, data_loader, device, **kwargs)

    elif method == "rsa":
        from .information_theoretic import RepresentationSimilarityImportance
        scorer = RepresentationSimilarityImportance()
        return scorer.compute_scores(model, data_loader, device, **kwargs)

    elif method == "sparsity":
        from .information_theoretic import ActivationSparsityImportance
        scorer = ActivationSparsityImportance()
        return scorer.compute_scores(model, data_loader, device, **kwargs)

    # Layer geometry methods
    elif method == "effective_rank":
        from .layer_geometry import EffectiveRankImportance
        scorer = EffectiveRankImportance()
        return scorer.compute_scores(model, **kwargs)

    elif method == "condition_number":
        from .layer_geometry import ConditionNumberImportance
        scorer = ConditionNumberImportance()
        return scorer.compute_scores(model, **kwargs)

    elif method == "spectral_norm":
        from .layer_geometry import SpectralNormImportance
        scorer = SpectralNormImportance()
        return scorer.compute_scores(model, **kwargs)

    elif method == "nuclear_norm":
        from .layer_geometry import NuclearNormImportance
        scorer = NuclearNormImportance()
        return scorer.compute_scores(model, **kwargs)

    elif method == "stable_rank":
        from .layer_geometry import StableRankImportance
        scorer = StableRankImportance()
        return scorer.compute_scores(model, **kwargs)

    elif method == "lipschitz":
        from .layer_geometry import LipschitzConstantImportance
        scorer = LipschitzConstantImportance()
        return scorer.compute_scores(model, **kwargs)

    # Legacy methods
    elif method == "activation":
        from .activation_based import ActivationImportance
        scorer = ActivationImportance()
        return scorer.compute_scores(model, data_loader, device, **kwargs)

    else:
        raise ValueError(
            f"Unknown importance method: '{method}'. "
            f"See docstring for list of available methods (30+ options)."
        )


def _compute_weight_importance(
    model: nn.Module,
    norm: str = "l2",
    **kwargs
) -> Dict[str, float]:
    """
    Compute importance based on weight magnitudes.

    This is a simple baseline that doesn't require any data.

    Args:
        model: PyTorch model
        norm: Norm to use ('l1', 'l2', 'fro')

    Returns:
        Dictionary mapping layer names to importance scores
    """
    importance_scores = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            weight = module.weight.data

            if norm == "l1":
                score = torch.abs(weight).sum().item()
            elif norm == "l2":
                score = torch.norm(weight, p=2).item()
            elif norm == "fro":
                score = torch.norm(weight, p='fro').item()
            else:
                raise ValueError(f"Unknown norm: {norm}")

            importance_scores[name] = score

    # Normalize scores
    max_score = max(importance_scores.values()) if importance_scores else 1.0
    importance_scores = {k: v / max_score for k, v in importance_scores.items()}

    return importance_scores


def select_top_k_layers(
    importance_scores: Dict[str, float],
    k: int,
    exclude_layers: Optional[list] = None
) -> list:
    """
    Select top-k most important layers.

    Args:
        importance_scores: Dictionary of importance scores
        k: Number of layers to select
        exclude_layers: List of layer names to exclude

    Returns:
        List of (layer_name, score) tuples for top-k layers
    """
    if exclude_layers is None:
        exclude_layers = []

    # Filter out excluded layers
    filtered_scores = {
        name: score for name, score in importance_scores.items()
        if name not in exclude_layers
    }

    # Sort by score descending
    sorted_layers = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)

    # Return top-k
    return sorted_layers[:k]


def print_importance_scores(
    importance_scores: Dict[str, float],
    top_k: int = 10
):
    """
    Print importance scores in a readable format.

    Args:
        importance_scores: Dictionary of importance scores
        top_k: Number of top layers to print
    """
    sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_k} Most Important Layers:")
    print("-" * 60)
    print(f"{'Rank':<6} {'Layer Name':<40} {'Score':<10}")
    print("-" * 60)

    for i, (name, score) in enumerate(sorted_layers[:top_k], 1):
        print(f"{i:<6} {name:<40} {score:<10.4f}")

    print("-" * 60)
