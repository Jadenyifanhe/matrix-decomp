"""
Activation-based importance scoring.

Importance is measured by the magnitude and variance of activations.
Layers with larger or more varied activations are considered more important.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class ActivationImportance:
    """
    Compute layer importance based on activation statistics.

    Higher activation magnitudes and variances indicate more active layers.
    """

    def __init__(self, metric: str = "mean"):
        """
        Initialize activation importance scorer.

        Args:
            metric: Metric to use for scoring
                - 'mean': Mean activation magnitude
                - 'std': Standard deviation of activations
                - 'combined': Mean + variance
        """
        self.metric = metric

    def compute_scores(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: str = "cpu",
        max_batches: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute activation-based importance scores.

        Args:
            model: PyTorch model
            data_loader: DataLoader with inputs
            device: Device to run on
            max_batches: Maximum number of batches
            verbose: Show progress bar

        Returns:
            Dictionary mapping layer names to importance scores
        """
        model = model.to(device)
        model.eval()

        # Storage for activation statistics
        activation_stats = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                activation_stats[name] = {
                    'sum': 0.0,
                    'sum_sq': 0.0,
                    'count': 0
                }

        # Hook to capture activations
        handles = []
        activations = {}

        def get_activation_hook(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                handle = module.register_forward_hook(get_activation_hook(name))
                handles.append(handle)

        # Process batches
        iterator = enumerate(data_loader)
        if verbose:
            iterator = tqdm(iterator, desc="Computing activation importance", total=len(data_loader))

        with torch.no_grad():
            for batch_idx, batch in iterator:
                if max_batches is not None and batch_idx >= max_batches:
                    break

                # Handle different batch formats
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch

                inputs = inputs.to(device)

                # Forward pass
                _ = model(inputs)

                # Collect activation statistics
                for name, act in activations.items():
                    act_abs = act.abs()
                    activation_stats[name]['sum'] += act_abs.sum().item()
                    activation_stats[name]['sum_sq'] += (act_abs ** 2).sum().item()
                    activation_stats[name]['count'] += act.numel()

                activations.clear()

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Compute importance scores
        importance_scores = {}
        for name, stats in activation_stats.items():
            if stats['count'] > 0:
                mean = stats['sum'] / stats['count']
                mean_sq = stats['sum_sq'] / stats['count']
                var = mean_sq - mean ** 2
                std = var ** 0.5 if var > 0 else 0.0

                if self.metric == "mean":
                    score = mean
                elif self.metric == "std":
                    score = std
                elif self.metric == "combined":
                    score = mean + std
                else:
                    raise ValueError(f"Unknown metric: {self.metric}")

                importance_scores[name] = score
            else:
                importance_scores[name] = 0.0

        # Normalize scores
        max_score = max(importance_scores.values()) if importance_scores else 1.0
        if max_score > 0:
            importance_scores = {k: v / max_score for k, v in importance_scores.items()}

        return importance_scores


class SensitivityImportance:
    """
    Compute importance based on sensitivity to input perturbations.

    Measures how much the output changes when the layer's activations are perturbed.
    """

    def __init__(self, perturbation_scale: float = 0.1):
        """
        Initialize sensitivity importance scorer.

        Args:
            perturbation_scale: Scale of random perturbations to apply
        """
        self.perturbation_scale = perturbation_scale

    def compute_scores(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: str = "cpu",
        max_batches: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute sensitivity-based importance scores.

        Args:
            model: PyTorch model
            data_loader: DataLoader
            device: Device
            max_batches: Max batches
            verbose: Show progress

        Returns:
            Importance scores
        """
        model = model.to(device)
        model.eval()

        sensitivity_scores = {}
        for name in model.named_modules():
            if isinstance(name[1], (nn.Linear, nn.Conv2d, nn.Conv1d)):
                sensitivity_scores[name[0]] = []

        # Hook to perturb activations
        def get_perturbation_hook(name, perturb):
            def hook(module, input, output):
                if perturb:
                    noise = torch.randn_like(output) * self.perturbation_scale
                    return output + noise
                return output
            return hook

        iterator = enumerate(data_loader)
        if verbose:
            iterator = tqdm(iterator, desc="Computing sensitivity", total=len(data_loader))

        with torch.no_grad():
            for batch_idx, batch in iterator:
                if max_batches is not None and batch_idx >= max_batches:
                    break

                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch

                inputs = inputs.to(device)

                # Get baseline output
                baseline_output = model(inputs)

                # Test each layer
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                        # Register perturbation hook
                        handle = module.register_forward_hook(get_perturbation_hook(name, True))

                        # Get perturbed output
                        perturbed_output = model(inputs)

                        # Compute difference
                        diff = (baseline_output - perturbed_output).abs().mean().item()
                        sensitivity_scores[name].append(diff)

                        # Remove hook
                        handle.remove()

        # Aggregate scores
        importance_scores = {}
        for name, scores in sensitivity_scores.items():
            if len(scores) > 0:
                importance_scores[name] = sum(scores) / len(scores)
            else:
                importance_scores[name] = 0.0

        # Normalize
        max_score = max(importance_scores.values()) if importance_scores else 1.0
        if max_score > 0:
            importance_scores = {k: v / max_score for k, v in importance_scores.items()}

        return importance_scores
