"""
Gradient-based importance scoring.

Importance is measured by the magnitude of gradients flowing through each layer.
Layers with larger gradients have more impact on the loss and are considered more important.
"""

from typing import Dict, Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class GradientImportance:
    """
    Compute layer importance based on gradient magnitudes.

    Higher gradients indicate the layer has more influence on the loss.
    """

    def __init__(self, aggregate: str = "mean"):
        """
        Initialize gradient importance scorer.

        Args:
            aggregate: How to aggregate gradients across batches
                - 'mean': Average gradient magnitude
                - 'max': Maximum gradient magnitude
                - 'sum': Sum of gradient magnitudes
        """
        self.aggregate = aggregate

    def compute_scores(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: Callable,
        device: str = "cpu",
        max_batches: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute gradient-based importance scores.

        Args:
            model: PyTorch model
            data_loader: DataLoader with (input, target) pairs
            criterion: Loss function
            device: Device to run on
            max_batches: Maximum number of batches to process (None for all)
            verbose: Show progress bar

        Returns:
            Dictionary mapping layer names to importance scores
        """
        model = model.to(device)
        model.train()  # Enable gradients

        # Storage for gradient statistics
        gradient_stats = {}

        # Hook to capture gradients
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                gradient_stats[name] = []

                def hook_fn(grad, layer_name=name):
                    """Hook function to capture gradients."""
                    if grad is not None:
                        gradient_stats[layer_name].append(grad.detach().abs().mean().item())
                    return grad

                # Register hook on weight parameter
                if hasattr(module, 'weight') and module.weight is not None:
                    handle = module.weight.register_hook(lambda grad, n=name: hook_fn(grad, n))
                    handles.append(handle)

        # Process batches
        iterator = enumerate(data_loader)
        if verbose:
            iterator = tqdm(iterator, desc="Computing gradient importance", total=len(data_loader))

        for batch_idx, batch in iterator:
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Handle different batch formats
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, targets = batch
            else:
                # If only inputs (no targets), skip
                continue

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradients are captured by hooks

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Aggregate gradient statistics
        importance_scores = {}
        for name, grads in gradient_stats.items():
            if len(grads) > 0:
                if self.aggregate == "mean":
                    score = sum(grads) / len(grads)
                elif self.aggregate == "max":
                    score = max(grads)
                elif self.aggregate == "sum":
                    score = sum(grads)
                else:
                    raise ValueError(f"Unknown aggregate method: {self.aggregate}")

                importance_scores[name] = score
            else:
                importance_scores[name] = 0.0

        # Normalize scores
        max_score = max(importance_scores.values()) if importance_scores else 1.0
        if max_score > 0:
            importance_scores = {k: v / max_score for k, v in importance_scores.items()}

        return importance_scores


class TaylorImportance:
    """
    Compute importance using Taylor expansion approximation.

    Importance = |weight * gradient|

    This estimates the change in loss if the weight were removed.
    """

    def compute_scores(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: Callable,
        device: str = "cpu",
        max_batches: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute Taylor importance scores.

        Args:
            model: PyTorch model
            data_loader: DataLoader
            criterion: Loss function
            device: Device
            max_batches: Max batches to process
            verbose: Show progress

        Returns:
            Importance scores
        """
        model = model.to(device)
        model.train()

        # Storage for weight * gradient
        taylor_scores = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                taylor_scores[name] = []

        # Process batches
        iterator = enumerate(data_loader)
        if verbose:
            iterator = tqdm(iterator, desc="Computing Taylor importance", total=len(data_loader))

        for batch_idx, batch in iterator:
            if max_batches is not None and batch_idx >= max_batches:
                break

            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, targets = batch
            else:
                continue

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward and backward
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Compute weight * gradient
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                    if hasattr(module, 'weight') and module.weight.grad is not None:
                        taylor = (module.weight.data * module.weight.grad).abs().sum().item()
                        taylor_scores[name].append(taylor)

        # Aggregate
        importance_scores = {}
        for name, scores in taylor_scores.items():
            if len(scores) > 0:
                importance_scores[name] = sum(scores) / len(scores)
            else:
                importance_scores[name] = 0.0

        # Normalize
        max_score = max(importance_scores.values()) if importance_scores else 1.0
        if max_score > 0:
            importance_scores = {k: v / max_score for k, v in importance_scores.items()}

        return importance_scores
