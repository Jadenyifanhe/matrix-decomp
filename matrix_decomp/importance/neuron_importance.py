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
    data_loader: DataLoader,
    method: str = "gradient",
    criterion: Optional[Callable] = None,
    device: str = "cpu",
    **kwargs
) -> Dict[str, float]:
    """
    Compute importance scores for all layers in a model.

    Args:
        model: PyTorch model to analyze
        data_loader: DataLoader for validation/calibration data
        method: Importance scoring method:
            - 'gradient': Gradient-based importance (requires labels)
            - 'activation': Activation magnitude-based
            - 'weight': Weight magnitude-based (no data needed)
            - 'fisher': Fisher information-based
        criterion: Loss function (required for gradient and fisher methods)
        device: Device to run computation on
        **kwargs: Additional method-specific arguments

    Returns:
        Dictionary mapping layer names to importance scores
    """
    model = model.to(device)
    model.eval()

    if method == "gradient":
        from .gradient_based import GradientImportance
        scorer = GradientImportance()
        return scorer.compute_scores(model, data_loader, criterion, device, **kwargs)

    elif method == "activation":
        from .activation_based import ActivationImportance
        scorer = ActivationImportance()
        return scorer.compute_scores(model, data_loader, device, **kwargs)

    elif method == "weight":
        return _compute_weight_importance(model, **kwargs)

    elif method == "fisher":
        from .fisher_information import FisherImportance
        scorer = FisherImportance()
        return scorer.compute_scores(model, data_loader, criterion, device, **kwargs)

    else:
        raise ValueError(f"Unknown importance method: {method}")


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
