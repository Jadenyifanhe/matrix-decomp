"""
Hessian-based importance scoring methods.

These methods use second-order information (Hessian matrix) to measure importance.

Methods implemented:
- Diagonal Fisher Approximation
- Empirical Fisher Information
- Hessian Diagonal Approximation
- Optimal Brain Damage style (LeCun et al., 1990)
"""

from typing import Dict, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


class HessianDiagonalImportance:
    """
    Hessian diagonal approximation for importance.

    Approximates the diagonal of the Hessian matrix using gradient information.

    The Hessian diagonal tells us the curvature of the loss landscape
    with respect to each parameter.
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
        Compute Hessian diagonal approximation.

        Args:
            model: Model to analyze
            data_loader: DataLoader
            criterion: Loss function
            device: Device
            max_batches: Max batches to process
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to Hessian diagonal scores
        """
        model = model.to(device)
        model.train()

        hessian_scores = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                hessian_scores[name] = 0.0

        # Approximate Hessian diagonal using gradient outer product
        iterator = enumerate(data_loader)
        if verbose:
            iterator = tqdm(iterator, desc="Computing Hessian diagonal", total=len(data_loader))

        num_processed = 0

        for batch_idx, batch in iterator:
            if max_batches is not None and batch_idx >= max_batches:
                break

            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, targets = batch
            else:
                continue

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Process each sample individually for better Hessian approximation
            for i in range(inputs.size(0)):
                model.zero_grad()

                # Forward pass on single sample
                output = model(inputs[i:i+1])
                loss = criterion(output, targets[i:i+1])

                # Backward pass
                loss.backward()

                # Approximate Hessian diagonal: E[∇²L] ≈ E[(∇L)²]
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                        if hasattr(module, 'weight') and module.weight.grad is not None:
                            hessian_approx = (module.weight.grad ** 2).sum().item()
                            hessian_scores[name] += hessian_approx

                num_processed += 1

        # Average
        if num_processed > 0:
            hessian_scores = {k: v / num_processed for k, v in hessian_scores.items()}

        # Normalize
        max_score = max(hessian_scores.values()) if hessian_scores else 1.0
        if max_score > 0:
            hessian_scores = {k: v / max_score for k, v in hessian_scores.items()}

        return hessian_scores


class OptimalBrainDamageImportance:
    """
    Optimal Brain Damage (OBD) style importance.

    Paper: "Optimal Brain Damage" (LeCun et al., 1990)

    Computes importance as: I = (weight²) * (Hessian_diagonal) / 2

    This estimates the increase in loss if a parameter is removed.
    """

    def compute_scores(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: Callable,
        device: str = "cpu",
        max_batches: Optional[int] = 10,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute OBD-style importance scores.

        Args:
            model: Model to analyze
            data_loader: DataLoader
            criterion: Loss function
            device: Device
            max_batches: Max batches (OBD is expensive, use fewer)
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to OBD scores
        """
        model = model.to(device)
        model.train()

        # First, compute Hessian diagonal approximation
        hessian_diag = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                if hasattr(module, 'weight'):
                    hessian_diag[name] = torch.zeros_like(module.weight)

        iterator = enumerate(data_loader)
        if verbose:
            iterator = tqdm(iterator, desc="Computing OBD scores", total=min(max_batches, len(data_loader)))

        num_samples = 0

        for batch_idx, batch in iterator:
            if batch_idx >= max_batches:
                break

            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, targets = batch
            else:
                continue

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Process each sample
            for i in range(inputs.size(0)):
                model.zero_grad()

                output = model(inputs[i:i+1])
                loss = criterion(output, targets[i:i+1])
                loss.backward()

                # Accumulate squared gradients (Hessian diagonal approximation)
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                        if hasattr(module, 'weight') and module.weight.grad is not None:
                            hessian_diag[name] += module.weight.grad ** 2

                num_samples += 1

        # Average Hessian diagonal
        for name in hessian_diag:
            hessian_diag[name] /= num_samples

        # Compute OBD importance: I = 0.5 * w² * H_diag
        obd_scores = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                if name in hessian_diag:
                    weight = module.weight.data
                    h_diag = hessian_diag[name]

                    # OBD score per layer (sum over all weights)
                    score = (0.5 * weight ** 2 * h_diag).sum().item()
                    obd_scores[name] = score

        # Normalize
        max_score = max(obd_scores.values()) if obd_scores else 1.0
        if max_score > 0:
            obd_scores = {k: v / max_score for k, v in obd_scores.items()}

        return obd_scores


class EmpiricalFisherImportance:
    """
    Empirical Fisher Information Matrix (diagonal).

    More accurate Fisher estimation using the actual data distribution.

    Fisher Information: I_F = E_x,y[(∇log p(y|x))²]
    """

    def compute_scores(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: Optional[Callable] = None,
        device: str = "cpu",
        max_batches: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute empirical Fisher information.

        Args:
            model: Model to analyze
            data_loader: DataLoader with true labels
            criterion: Loss function (optional)
            device: Device
            max_batches: Max batches
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to Fisher scores
        """
        model = model.to(device)
        model.eval()  # Use eval mode for better estimates

        fisher_scores = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                fisher_scores[name] = 0.0

        iterator = enumerate(data_loader)
        if verbose:
            iterator = tqdm(iterator, desc="Computing Empirical Fisher", total=len(data_loader))

        num_samples = 0

        for batch_idx, batch in iterator:
            if max_batches is not None and batch_idx >= max_batches:
                break

            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, targets = batch
            else:
                continue

            inputs = inputs.to(device)
            targets = targets.to(device)

            model.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute log likelihood gradient
            if criterion is not None:
                loss = criterion(outputs, targets)
            else:
                # Use cross-entropy for classification
                if outputs.dim() == 2:
                    loss = nn.functional.cross_entropy(outputs, targets)
                else:
                    loss = (outputs - targets).pow(2).mean()

            # Backward
            loss.backward()

            # Accumulate squared gradients
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                    if hasattr(module, 'weight') and module.weight.grad is not None:
                        fisher = (module.weight.grad ** 2).sum().item()
                        fisher_scores[name] += fisher

            num_samples += inputs.size(0)

        # Average
        if num_samples > 0:
            fisher_scores = {k: v / num_samples for k, v in fisher_scores.items()}

        # Normalize
        max_score = max(fisher_scores.values()) if fisher_scores else 1.0
        if max_score > 0:
            fisher_scores = {k: v / max_score for k, v in fisher_scores.items()}

        return fisher_scores


class BlockwiseHessianImportance:
    """
    Block-wise Hessian importance for large models.

    Computes Hessian blocks for different layers separately to reduce memory.
    """

    def compute_scores(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: Callable,
        device: str = "cpu",
        max_batches: int = 5,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute block-wise Hessian importance.

        Args:
            model: Model to analyze
            data_loader: DataLoader
            criterion: Loss function
            device: Device
            max_batches: Max batches (Hessian is expensive!)
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to Hessian trace scores
        """
        model = model.to(device)
        model.train()

        hessian_traces = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                hessian_traces[name] = 0.0

        iterator = enumerate(data_loader)
        if verbose:
            iterator = tqdm(iterator, desc="Computing Blockwise Hessian", total=min(max_batches, len(data_loader)))

        num_batches = 0

        for batch_idx, batch in iterator:
            if batch_idx >= max_batches:
                break

            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, targets = batch
            else:
                continue

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Compute Hessian trace for each layer using Hutchinson's estimator
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                    if not hasattr(module, 'weight'):
                        continue

                    # Random vector for Hutchinson's trace estimator
                    v = torch.randn_like(module.weight)

                    # Zero gradients
                    model.zero_grad()

                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    # First gradient
                    grads = torch.autograd.grad(loss, module.weight, create_graph=True)[0]

                    # Gradient-vector product
                    gv = (grads * v).sum()

                    # Second derivative (Hessian-vector product)
                    if module.weight.grad is not None:
                        module.weight.grad.zero_()

                    hvp = torch.autograd.grad(gv, module.weight, retain_graph=False)[0]

                    # Trace estimate: v^T H v
                    trace = (v * hvp).sum().item()
                    hessian_traces[name] += abs(trace)

            num_batches += 1

        # Average
        if num_batches > 0:
            hessian_traces = {k: v / num_batches for k, v in hessian_traces.items()}

        # Normalize
        max_score = max(hessian_traces.values()) if hessian_traces else 1.0
        if max_score > 0:
            hessian_traces = {k: v / max_score for k, v in hessian_traces.items()}

        return hessian_traces
