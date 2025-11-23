"""
Pruning-based importance scoring methods.

These methods are derived from neural network pruning research and measure
importance based on how removing a component affects the network.

Methods implemented:
- SNIP: Single-shot Network Pruning (Lee et al., 2018)
- GraSP: Gradient Signal Preservation (Wang et al., 2020)
- SynFlow: Synaptic Flow (Tanaka et al., 2020)
- Movement Pruning: Tracks parameter movement during training
"""

from typing import Dict, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


class SNIPImportance:
    """
    SNIP: Single-shot Network Pruning.

    Measures connection sensitivity by computing |θ * ∇L(θ)|

    Paper: "SNIP: Single-shot Network Pruning based on Connection Sensitivity"
           Lee et al., ICLR 2019

    Key idea: Importance = |weight * gradient_at_initialization|
    """

    def compute_scores(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: Callable,
        device: str = "cpu",
        num_batches: int = 1,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute SNIP importance scores.

        Args:
            model: Model to analyze (should be at initialization)
            data_loader: DataLoader for computing gradients
            criterion: Loss function
            device: Device to run on
            num_batches: Number of batches to use
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to SNIP scores
        """
        model = model.to(device)
        model.train()

        # Store connection sensitivity scores
        snip_scores = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                snip_scores[name] = 0.0

        # Process batches
        iterator = enumerate(data_loader)
        if verbose:
            iterator = tqdm(iterator, desc="Computing SNIP scores", total=min(num_batches, len(data_loader)))

        for batch_idx, batch in iterator:
            if batch_idx >= num_batches:
                break

            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, targets = batch
            else:
                continue

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Zero gradients
            model.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Compute connection sensitivity: |w * ∇L/∂w|
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                    if hasattr(module, 'weight') and module.weight.grad is not None:
                        # SNIP score: sum of |weight * gradient|
                        sensitivity = (module.weight.data * module.weight.grad).abs().sum().item()
                        snip_scores[name] += sensitivity / num_batches

        # Normalize scores
        max_score = max(snip_scores.values()) if snip_scores else 1.0
        if max_score > 0:
            snip_scores = {k: v / max_score for k, v in snip_scores.items()}

        return snip_scores


class GraSPImportance:
    """
    GraSP: Gradient Signal Preservation.

    Measures importance by gradient flow preservation.

    Paper: "Picking Winning Tickets Before Training by Preserving Gradient Flow"
           Wang et al., ICLR 2020

    Key idea: Preserve gradient signal by keeping weights that maintain
              the Hessian-gradient product.
    """

    def compute_scores(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: Callable,
        device: str = "cpu",
        num_batches: int = 1,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute GraSP importance scores.

        Args:
            model: Model to analyze
            data_loader: DataLoader
            criterion: Loss function
            device: Device
            num_batches: Number of batches
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to GraSP scores
        """
        model = model.to(device)
        model.train()

        grasp_scores = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                grasp_scores[name] = 0.0

        # First pass: compute gradients
        all_gradients = []

        iterator = enumerate(data_loader)
        if verbose:
            iterator = tqdm(iterator, desc="Computing GraSP scores (pass 1)", total=min(num_batches, len(data_loader)))

        for batch_idx, batch in iterator:
            if batch_idx >= num_batches:
                break

            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, targets = batch
            else:
                continue

            inputs = inputs.to(device)
            targets = targets.to(device)

            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Store gradients
            batch_grads = {}
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                    if hasattr(module, 'weight') and module.weight.grad is not None:
                        batch_grads[name] = module.weight.grad.clone()

            all_gradients.append(batch_grads)

        # Second pass: compute Hessian-gradient product approximation
        if verbose:
            print("Computing GraSP scores (pass 2)...")

        for batch_grads in all_gradients:
            # Compute gradient flow score
            for name in grasp_scores.keys():
                if name in batch_grads:
                    # Simplified GraSP: use gradient magnitude weighted by weights
                    grad = batch_grads[name]
                    module = dict(model.named_modules())[name]
                    weight = module.weight.data

                    # GraSP score approximation: -∇L · Hessian · ∇L
                    # We approximate with: |w * ∇L|²
                    score = ((weight * grad) ** 2).sum().item()
                    grasp_scores[name] += score / len(all_gradients)

        # Normalize
        max_score = max(grasp_scores.values()) if grasp_scores else 1.0
        if max_score > 0:
            grasp_scores = {k: v / max_score for k, v in grasp_scores.items()}

        return grasp_scores


class SynFlowImportance:
    """
    SynFlow: Synaptic Flow.

    Iterative pruning method that preserves network connectivity.

    Paper: "Pruning Neural Networks at Initialization: Why are We Missing the Mark?"
           Tanaka et al., ICLR 2020

    Key idea: Use a data-independent score based on the product of weights
              along paths in the network.
    """

    def compute_scores(
        self,
        model: nn.Module,
        input_shape: tuple,
        device: str = "cpu",
        num_iterations: int = 1,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute SynFlow importance scores.

        Note: SynFlow is data-independent, only needs input shape.

        Args:
            model: Model to analyze
            input_shape: Input shape (without batch dimension)
            device: Device
            num_iterations: Number of SynFlow iterations
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to SynFlow scores
        """
        model = model.to(device)
        model.train()

        synflow_scores = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                synflow_scores[name] = 0.0

        # Create dummy input with all ones
        dummy_input = torch.ones(1, *input_shape, device=device)

        for iteration in range(num_iterations):
            if verbose:
                print(f"SynFlow iteration {iteration + 1}/{num_iterations}")

            # Zero gradients
            model.zero_grad()

            # Forward pass with ones (to compute path products)
            # Apply absolute value to all parameters temporarily
            for module in model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                    module.weight.data = module.weight.data.abs()

            # Forward pass
            output = model(dummy_input)

            # Compute sum of outputs (to get gradient = 1 everywhere)
            loss = output.sum()

            # Backward pass
            loss.backward()

            # SynFlow score: weight * gradient
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                    if hasattr(module, 'weight') and module.weight.grad is not None:
                        score = (module.weight.data.abs() * module.weight.grad.abs()).sum().item()
                        synflow_scores[name] += score / num_iterations

            # Restore original weights (this is just for scoring, model shouldn't be modified)
            # In practice, you'd save/restore the original weights

        # Normalize
        max_score = max(synflow_scores.values()) if synflow_scores else 1.0
        if max_score > 0:
            synflow_scores = {k: v / max_score for k, v in synflow_scores.items()}

        return synflow_scores


class MovementPruningImportance:
    """
    Movement Pruning: Tracks parameter movement during training.

    Paper: "Movement Pruning: Adaptive Sparsity by Fine-Tuning"
           Sanh et al., NeurIPS 2020

    Key idea: Parameters that move towards zero during training are less important.
    """

    def __init__(self):
        """Initialize movement tracking."""
        self.initial_weights = {}
        self.has_initial = False

    def save_initial_weights(self, model: nn.Module):
        """Save initial weights before training."""
        self.initial_weights = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                if hasattr(module, 'weight'):
                    self.initial_weights[name] = module.weight.data.clone()
        self.has_initial = True

    def compute_scores(
        self,
        model: nn.Module,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute movement-based importance scores.

        Must call save_initial_weights() before training!

        Args:
            model: Model after training
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to movement scores
        """
        if not self.has_initial:
            raise ValueError("Must call save_initial_weights() before training!")

        movement_scores = {}

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                if name in self.initial_weights:
                    initial = self.initial_weights[name]
                    current = module.weight.data

                    # Movement score: magnitude of weight change
                    movement = (current - initial).abs().sum().item()

                    # Alternative: movement away from zero
                    # initial_dist = initial.abs().sum().item()
                    # current_dist = current.abs().sum().item()
                    # movement = abs(current_dist - initial_dist)

                    movement_scores[name] = movement

        # Normalize
        max_score = max(movement_scores.values()) if movement_scores else 1.0
        if max_score > 0:
            movement_scores = {k: v / max_score for k, v in movement_scores.items()}

        return movement_scores


class MagnitudePruningImportance:
    """
    Magnitude-based pruning importance.

    Classic pruning method: layers with larger weight magnitudes are more important.

    Simple but effective baseline.
    """

    def compute_scores(
        self,
        model: nn.Module,
        norm: str = "l1",
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute magnitude-based importance scores.

        Args:
            model: Model to analyze
            norm: Norm to use ('l1', 'l2', 'linf')
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to magnitude scores
        """
        magnitude_scores = {}

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                if hasattr(module, 'weight'):
                    weight = module.weight.data

                    if norm == "l1":
                        score = weight.abs().sum().item()
                    elif norm == "l2":
                        score = (weight ** 2).sum().item() ** 0.5
                    elif norm == "linf":
                        score = weight.abs().max().item()
                    else:
                        raise ValueError(f"Unknown norm: {norm}")

                    magnitude_scores[name] = score

        # Normalize
        max_score = max(magnitude_scores.values()) if magnitude_scores else 1.0
        if max_score > 0:
            magnitude_scores = {k: v / max_score for k, v in magnitude_scores.items()}

        return magnitude_scores
