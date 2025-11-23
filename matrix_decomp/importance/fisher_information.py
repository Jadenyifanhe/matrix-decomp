"""
Fisher Information-based importance scoring.

Fisher Information measures the sensitivity of the model's output distribution
to changes in parameters. Higher Fisher Information indicates more important parameters.
"""

from typing import Dict, Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class FisherImportance:
    """
    Compute layer importance based on Fisher Information.

    Fisher Information = E[(∇log p(y|x))^2]

    This measures how much information the parameters carry about the predictions.
    """

    def __init__(self, diagonal: bool = True):
        """
        Initialize Fisher importance scorer.

        Args:
            diagonal: If True, use diagonal Fisher (empirical Fisher)
                     If False, use full Fisher (computationally expensive)
        """
        self.diagonal = diagonal

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
        Compute Fisher Information-based importance scores.

        Args:
            model: PyTorch model
            data_loader: DataLoader
            criterion: Loss function (optional, uses log-likelihood by default)
            device: Device
            max_batches: Max batches to process
            verbose: Show progress

        Returns:
            Importance scores
        """
        model = model.to(device)
        model.eval()

        # Storage for Fisher Information
        fisher_scores = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                fisher_scores[name] = []

        # Process batches
        iterator = enumerate(data_loader)
        if verbose:
            iterator = tqdm(iterator, desc="Computing Fisher Information", total=len(data_loader))

        for batch_idx, batch in iterator:
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Handle different batch formats
            if isinstance(batch, (tuple, list)):
                if len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs = batch[0]
                    targets = None
            else:
                inputs = batch
                targets = None

            inputs = inputs.to(device)
            if targets is not None:
                targets = targets.to(device)

            # Forward pass
            model.zero_grad()
            outputs = model(inputs)

            # Compute loss
            if criterion is not None and targets is not None:
                loss = criterion(outputs, targets)
            else:
                # Use negative log-likelihood for classification
                if outputs.dim() == 2:
                    # Classification: use predicted class probabilities
                    probs = torch.softmax(outputs, dim=1)
                    # Sample from predicted distribution
                    sampled_y = torch.multinomial(probs, 1).squeeze()
                    loss = torch.nn.functional.cross_entropy(outputs, sampled_y)
                else:
                    # Regression: use MSE with outputs
                    loss = (outputs ** 2).mean()

            # Backward pass
            loss.backward()

            # Collect squared gradients (empirical Fisher)
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                    if hasattr(module, 'weight') and module.weight.grad is not None:
                        fisher = (module.weight.grad ** 2).sum().item()
                        fisher_scores[name].append(fisher)

        # Aggregate scores
        importance_scores = {}
        for name, scores in fisher_scores.items():
            if len(scores) > 0:
                # Average Fisher Information
                importance_scores[name] = sum(scores) / len(scores)
            else:
                importance_scores[name] = 0.0

        # Normalize scores
        max_score = max(importance_scores.values()) if importance_scores else 1.0
        if max_score > 0:
            importance_scores = {k: v / max_score for k, v in importance_scores.items()}

        return importance_scores


class EWCImportance(FisherImportance):
    """
    Elastic Weight Consolidation (EWC) style Fisher Information.

    This is similar to Fisher Information but specifically designed for
    identifying parameters important for a task (useful for continual learning).
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
        Compute EWC-style importance scores.

        Uses the actual task loss (not sampled) to compute Fisher Information.
        """
        if criterion is None:
            raise ValueError("EWC requires a criterion (loss function)")

        # EWC uses the actual loss with real targets
        return super().compute_scores(
            model=model,
            data_loader=data_loader,
            criterion=criterion,
            device=device,
            max_batches=max_batches,
            verbose=verbose
        )
