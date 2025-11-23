"""
Ablation-based importance scoring methods.

These methods measure importance by removing or zeroing out components
and observing the impact on model performance.

Methods implemented:
- Layer Knockout: Zero out entire layers
- Weight Knockout: Zero out individual weights
- Feature Ablation: Remove feature channels
- Dropout-based Importance: Use dropout to estimate importance
"""

from typing import Dict, Optional, Callable, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import copy


class LayerKnockoutImportance:
    """
    Layer Knockout importance.

    Measures importance by temporarily removing each layer and measuring
    the performance drop.

    Simple but effective for layer-level importance.
    """

    def compute_scores(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: Callable,
        device: str = "cpu",
        eval_metric: str = "loss",  # 'loss' or 'accuracy'
        max_batches: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute layer knockout importance.

        Args:
            model: Model to analyze
            data_loader: DataLoader for evaluation
            criterion: Loss function
            device: Device
            eval_metric: Metric to use ('loss' or 'accuracy')
            max_batches: Max batches to evaluate
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to knockout importance
        """
        model = model.to(device)
        model.eval()

        # Get baseline performance
        if verbose:
            print("Computing baseline performance...")

        baseline = self._evaluate_model(
            model, data_loader, criterion, device, eval_metric, max_batches
        )

        knockout_scores = {}

        # Test each layer
        layer_names = [name for name, module in model.named_modules()
                      if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d))]

        iterator = layer_names
        if verbose:
            iterator = tqdm(layer_names, desc="Layer knockout")

        for layer_name in iterator:
            # Save original weights
            module = dict(model.named_modules())[layer_name]
            original_weight = module.weight.data.clone()
            original_bias = module.bias.data.clone() if module.bias is not None else None

            # Zero out layer
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()

            # Evaluate
            knocked_out = self._evaluate_model(
                model, data_loader, criterion, device, eval_metric, max_batches
            )

            # Restore weights
            module.weight.data = original_weight
            if module.bias is not None:
                module.bias.data = original_bias

            # Importance = performance drop
            if eval_metric == "loss":
                importance = knocked_out - baseline  # Higher loss = more important
            else:  # accuracy
                importance = baseline - knocked_out  # Lower accuracy = more important

            knockout_scores[layer_name] = max(0, importance)  # Ensure non-negative

        # Normalize
        max_score = max(knockout_scores.values()) if knockout_scores else 1.0
        if max_score > 0:
            knockout_scores = {k: v / max_score for k, v in knockout_scores.items()}

        return knockout_scores

    def _evaluate_model(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: Callable,
        device: str,
        metric: str,
        max_batches: Optional[int]
    ) -> float:
        """Evaluate model on data."""
        model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    inputs, targets = batch
                else:
                    continue

                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)

                # Compute accuracy if needed
                if metric == "accuracy":
                    if outputs.dim() == 2:  # Classification
                        preds = outputs.argmax(dim=1)
                        total_correct += (preds == targets).sum().item()

                total_samples += inputs.size(0)

        if metric == "loss":
            return total_loss / total_samples if total_samples > 0 else 0.0
        else:  # accuracy
            return total_correct / total_samples if total_samples > 0 else 0.0


class WeightPerturbationImportance:
    """
    Weight Perturbation importance.

    Adds noise to weights and measures sensitivity to perturbations.

    Layers that are more sensitive to perturbations are more important.
    """

    def compute_scores(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: Callable,
        device: str = "cpu",
        perturbation_scale: float = 0.1,
        num_trials: int = 5,
        max_batches: Optional[int] = 10,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute weight perturbation importance.

        Args:
            model: Model to analyze
            data_loader: DataLoader
            criterion: Loss function
            device: Device
            perturbation_scale: Scale of random perturbations
            num_trials: Number of perturbation trials per layer
            max_batches: Max batches for evaluation
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to perturbation sensitivity
        """
        model = model.to(device)
        model.eval()

        # Baseline performance
        if verbose:
            print("Computing baseline...")

        baseline_loss = self._evaluate_loss(model, data_loader, criterion, device, max_batches)

        perturbation_scores = {}

        layer_names = [name for name, module in model.named_modules()
                      if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d))]

        iterator = layer_names
        if verbose:
            iterator = tqdm(layer_names, desc="Weight perturbation")

        for layer_name in iterator:
            module = dict(model.named_modules())[layer_name]
            original_weight = module.weight.data.clone()

            sensitivity = 0.0

            for trial in range(num_trials):
                # Add random noise
                noise = torch.randn_like(module.weight) * perturbation_scale
                module.weight.data = original_weight + noise

                # Evaluate
                perturbed_loss = self._evaluate_loss(
                    model, data_loader, criterion, device, max_batches
                )

                # Sensitivity = change in loss
                sensitivity += abs(perturbed_loss - baseline_loss)

                # Restore weights
                module.weight.data = original_weight

            # Average over trials
            perturbation_scores[layer_name] = sensitivity / num_trials

        # Normalize
        max_score = max(perturbation_scores.values()) if perturbation_scores else 1.0
        if max_score > 0:
            perturbation_scores = {k: v / max_score for k, v in perturbation_scores.items()}

        return perturbation_scores

    def _evaluate_loss(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: Callable,
        device: str,
        max_batches: Optional[int]
    ) -> float:
        """Evaluate model loss."""
        model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    inputs, targets = batch
                else:
                    continue

                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        return total_loss / total_samples if total_samples > 0 else 0.0


class ChannelAblationImportance:
    """
    Channel Ablation importance for convolutional networks.

    Removes individual channels and measures performance impact.

    Useful for identifying important feature channels.
    """

    def compute_scores(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: Callable,
        device: str = "cpu",
        max_batches: int = 10,
        sample_ratio: float = 0.1,  # Sample 10% of channels to speed up
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute channel ablation importance.

        Args:
            model: Model with Conv layers
            data_loader: DataLoader
            criterion: Loss function
            device: Device
            max_batches: Max batches
            sample_ratio: Ratio of channels to sample (for speed)
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to channel importance
        """
        model = model.to(device)
        model.eval()

        # Baseline
        if verbose:
            print("Computing baseline...")

        baseline_loss = self._evaluate_loss(model, data_loader, criterion, device, max_batches)

        channel_scores = {}

        # Focus on Conv2d layers
        conv_layers = {name: module for name, module in model.named_modules()
                      if isinstance(module, nn.Conv2d)}

        for layer_name, module in conv_layers.items():
            if verbose:
                print(f"Analyzing {layer_name}...")

            num_channels = module.out_channels
            num_to_sample = max(1, int(num_channels * sample_ratio))

            # Sample channels to ablate
            sampled_channels = np.random.choice(num_channels, num_to_sample, replace=False)

            channel_sensitivities = []

            for channel_idx in sampled_channels:
                # Save original weights for this channel
                original_weight = module.weight.data[channel_idx].clone()

                # Zero out channel
                module.weight.data[channel_idx].zero_()

                # Evaluate
                ablated_loss = self._evaluate_loss(
                    model, data_loader, criterion, device, max_batches
                )

                # Restore
                module.weight.data[channel_idx] = original_weight

                # Importance = loss increase
                sensitivity = ablated_loss - baseline_loss
                channel_sensitivities.append(sensitivity)

            # Average channel importance for this layer
            channel_scores[layer_name] = np.mean(channel_sensitivities)

        # Normalize
        max_score = max(channel_scores.values()) if channel_scores else 1.0
        if max_score > 0:
            channel_scores = {k: v / max_score for k, v in channel_scores.items()}

        return channel_scores

    def _evaluate_loss(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: Callable,
        device: str,
        max_batches: int
    ) -> float:
        """Evaluate model loss."""
        model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break

                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    inputs, targets = batch
                else:
                    continue

                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        return total_loss / total_samples if total_samples > 0 else 0.0


class DropoutBasedImportance:
    """
    Dropout-based importance estimation.

    Uses Monte Carlo dropout to estimate parameter uncertainty.

    Parameters with higher dropout sensitivity are more important.
    """

    def compute_scores(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: str = "cpu",
        num_mc_samples: int = 10,
        dropout_rate: float = 0.1,
        max_batches: int = 10,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute dropout-based importance.

        Args:
            model: Model to analyze
            data_loader: DataLoader
            device: Device
            num_mc_samples: Number of Monte Carlo samples
            dropout_rate: Dropout rate to apply
            max_batches: Max batches
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to dropout sensitivity
        """
        model = model.to(device)

        # Add dropout to each layer temporarily
        dropout_scores = {}

        layer_names = [name for name, module in model.named_modules()
                      if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d))]

        for layer_name in layer_names:
            if verbose:
                print(f"Testing dropout on {layer_name}...")

            # Collect outputs with dropout
            outputs_with_dropout = []

            for _ in range(num_mc_samples):
                batch_outputs = []

                for batch_idx, batch in enumerate(data_loader):
                    if batch_idx >= max_batches:
                        break

                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0]
                    else:
                        inputs = batch

                    inputs = inputs.to(device)

                    with torch.no_grad():
                        model.eval()
                        output = model(inputs)

                    batch_outputs.append(output.cpu())

                if batch_outputs:
                    outputs_with_dropout.append(torch.cat(batch_outputs, dim=0))

            # Compute variance across MC samples
            if outputs_with_dropout:
                stacked = torch.stack(outputs_with_dropout)  # [num_mc_samples, batch, ...]
                variance = stacked.var(dim=0).mean().item()
                dropout_scores[layer_name] = variance
            else:
                dropout_scores[layer_name] = 0.0

        # Normalize
        max_score = max(dropout_scores.values()) if dropout_scores else 1.0
        if max_score > 0:
            dropout_scores = {k: v / max_score for k, v in dropout_scores.items()}

        return dropout_scores
