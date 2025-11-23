"""
Layer geometry-based importance scoring methods.

These methods analyze the geometric properties of weight matrices.

Methods implemented:
- Effective Rank: Measures the dimensionality of weight matrices
- Condition Number: Measures numerical stability
- Spectral Norm: Largest singular value
- Nuclear Norm: Sum of singular values
- Lipschitz Constant: Bounds on gradient flow
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


class EffectiveRankImportance:
    """
    Effective Rank of weight matrices.

    Effective rank measures the "true" dimensionality of the weight matrix
    using the entropy of singular values.

    rank_eff = exp(H(σ)) where H is entropy of normalized singular values.

    Higher effective rank = more complex representations = more important.
    """

    def compute_scores(
        self,
        model: nn.Module,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute effective rank scores.

        Args:
            model: Model to analyze
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to effective rank scores
        """
        rank_scores = {}

        layers = [(name, module) for name, module in model.named_modules()
                 if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d))]

        iterator = layers
        if verbose:
            iterator = tqdm(layers, desc="Computing effective rank")

        for name, module in iterator:
            if not hasattr(module, 'weight'):
                continue

            weight = module.weight.data

            # Reshape Conv weights to 2D
            if weight.dim() > 2:
                weight = weight.flatten(1)

            # Compute SVD
            try:
                _, S, _ = torch.svd(weight)

                # Normalize singular values to get probability distribution
                S_norm = S / S.sum()

                # Compute entropy
                entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum().item()

                # Effective rank
                eff_rank = np.exp(entropy)

                rank_scores[name] = eff_rank
            except:
                rank_scores[name] = 0.0

        # Normalize
        max_score = max(rank_scores.values()) if rank_scores else 1.0
        if max_score > 0:
            rank_scores = {k: v / max_score for k, v in rank_scores.items()}

        return rank_scores


class ConditionNumberImportance:
    """
    Condition Number of weight matrices.

    Condition number = σ_max / σ_min

    Lower condition number = more stable = potentially more important for training.
    However, for importance, we might want to use the inverse or consider
    that high condition number indicates sensitivity.
    """

    def compute_scores(
        self,
        model: nn.Module,
        inverse: bool = True,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute condition number scores.

        Args:
            model: Model to analyze
            inverse: If True, use 1/cond (lower cond = higher score)
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to condition number scores
        """
        cond_scores = {}

        layers = [(name, module) for name, module in model.named_modules()
                 if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d))]

        iterator = layers
        if verbose:
            iterator = tqdm(layers, desc="Computing condition number")

        for name, module in iterator:
            if not hasattr(module, 'weight'):
                continue

            weight = module.weight.data

            # Reshape Conv weights to 2D
            if weight.dim() > 2:
                weight = weight.flatten(1)

            # Compute SVD
            try:
                _, S, _ = torch.svd(weight)

                # Condition number
                sigma_max = S[0].item()
                sigma_min = S[-1].item()

                if sigma_min > 1e-10:
                    cond = sigma_max / sigma_min

                    if inverse:
                        score = 1.0 / (1.0 + cond)  # Normalize to [0, 1]
                    else:
                        score = cond

                    cond_scores[name] = score
                else:
                    cond_scores[name] = 0.0
            except:
                cond_scores[name] = 0.0

        # Normalize
        max_score = max(cond_scores.values()) if cond_scores else 1.0
        if max_score > 0:
            cond_scores = {k: v / max_score for k, v in cond_scores.items()}

        return cond_scores


class SpectralNormImportance:
    """
    Spectral Norm (largest singular value).

    Spectral norm = σ_max = largest singular value

    Used in spectral normalization for GANs and indicates the maximum
    amplification of the layer.

    Higher spectral norm = stronger layer effect.
    """

    def compute_scores(
        self,
        model: nn.Module,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute spectral norm scores.

        Args:
            model: Model to analyze
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to spectral norm scores
        """
        spectral_scores = {}

        layers = [(name, module) for name, module in model.named_modules()
                 if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d))]

        iterator = layers
        if verbose:
            iterator = tqdm(layers, desc="Computing spectral norm")

        for name, module in iterator:
            if not hasattr(module, 'weight'):
                continue

            weight = module.weight.data

            # Reshape Conv weights to 2D
            if weight.dim() > 2:
                weight = weight.flatten(1)

            # Compute largest singular value using power iteration
            # This is faster than full SVD
            try:
                spectral_norm = self._power_iteration(weight)
                spectral_scores[name] = spectral_norm
            except:
                spectral_scores[name] = 0.0

        # Normalize
        max_score = max(spectral_scores.values()) if spectral_scores else 1.0
        if max_score > 0:
            spectral_scores = {k: v / max_score for k, v in spectral_scores.items()}

        return spectral_scores

    def _power_iteration(self, W: torch.Tensor, num_iters: int = 10) -> float:
        """
        Compute spectral norm using power iteration.

        Faster than full SVD for large matrices.
        """
        # Random initialization
        v = torch.randn(W.size(1), device=W.device)
        v = v / torch.norm(v)

        for _ in range(num_iters):
            # Power iteration
            u = W @ v
            u = u / (torch.norm(u) + 1e-10)

            v = W.t() @ u
            v = v / (torch.norm(v) + 1e-10)

        # Spectral norm = ||W @ v||
        spectral_norm = torch.norm(W @ v).item()

        return spectral_norm


class NuclearNormImportance:
    """
    Nuclear Norm (trace norm).

    Nuclear norm = Σ σ_i = sum of all singular values

    Encourages low-rank solutions. Higher nuclear norm = more capacity.
    """

    def compute_scores(
        self,
        model: nn.Module,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute nuclear norm scores.

        Args:
            model: Model to analyze
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to nuclear norm scores
        """
        nuclear_scores = {}

        layers = [(name, module) for name, module in model.named_modules()
                 if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d))]

        iterator = layers
        if verbose:
            iterator = tqdm(layers, desc="Computing nuclear norm")

        for name, module in iterator:
            if not hasattr(module, 'weight'):
                continue

            weight = module.weight.data

            # Reshape Conv weights to 2D
            if weight.dim() > 2:
                weight = weight.flatten(1)

            # Compute SVD
            try:
                _, S, _ = torch.svd(weight)

                # Nuclear norm = sum of singular values
                nuclear_norm = S.sum().item()

                nuclear_scores[name] = nuclear_norm
            except:
                nuclear_scores[name] = 0.0

        # Normalize
        max_score = max(nuclear_scores.values()) if nuclear_scores else 1.0
        if max_score > 0:
            nuclear_scores = {k: v / max_score for k, v in nuclear_scores.items()}

        return nuclear_scores


class StableRankImportance:
    """
    Stable Rank: ||W||²_F / ||W||²_2

    Ratio of Frobenius norm to spectral norm squared.

    Approximates the effective number of non-zero singular values.
    """

    def compute_scores(
        self,
        model: nn.Module,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute stable rank scores.

        Args:
            model: Model to analyze
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to stable rank scores
        """
        stable_rank_scores = {}

        layers = [(name, module) for name, module in model.named_modules()
                 if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d))]

        iterator = layers
        if verbose:
            iterator = tqdm(layers, desc="Computing stable rank")

        for name, module in iterator:
            if not hasattr(module, 'weight'):
                continue

            weight = module.weight.data

            # Reshape Conv weights to 2D
            if weight.dim() > 2:
                weight = weight.flatten(1)

            # Frobenius norm
            frob_norm = torch.norm(weight, p='fro').item()

            # Spectral norm (largest singular value)
            try:
                spectral_norm = self._power_iteration(weight)

                if spectral_norm > 1e-10:
                    stable_rank = (frob_norm ** 2) / (spectral_norm ** 2)
                    stable_rank_scores[name] = stable_rank
                else:
                    stable_rank_scores[name] = 0.0
            except:
                stable_rank_scores[name] = 0.0

        # Normalize
        max_score = max(stable_rank_scores.values()) if stable_rank_scores else 1.0
        if max_score > 0:
            stable_rank_scores = {k: v / max_score for k, v in stable_rank_scores.items()}

        return stable_rank_scores

    def _power_iteration(self, W: torch.Tensor, num_iters: int = 10) -> float:
        """Compute spectral norm using power iteration."""
        v = torch.randn(W.size(1), device=W.device)
        v = v / torch.norm(v)

        for _ in range(num_iters):
            u = W @ v
            u = u / (torch.norm(u) + 1e-10)
            v = W.t() @ u
            v = v / (torch.norm(v) + 1e-10)

        return torch.norm(W @ v).item()


class LipschitzConstantImportance:
    """
    Lipschitz Constant estimation.

    Measures how much the layer can amplify perturbations.

    Important for understanding gradient flow and stability.
    """

    def compute_scores(
        self,
        model: nn.Module,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute Lipschitz constant approximation.

        For linear layers, Lipschitz constant = spectral norm.

        Args:
            model: Model to analyze
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to Lipschitz constant scores
        """
        lipschitz_scores = {}

        layers = [(name, module) for name, module in model.named_modules()
                 if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d))]

        iterator = layers
        if verbose:
            iterator = tqdm(layers, desc="Computing Lipschitz constant")

        for name, module in iterator:
            if not hasattr(module, 'weight'):
                continue

            weight = module.weight.data

            # Reshape Conv weights to 2D
            if weight.dim() > 2:
                weight = weight.flatten(1)

            # For linear layers, Lipschitz constant = spectral norm
            try:
                spectral_norm = self._power_iteration(weight)
                lipschitz_scores[name] = spectral_norm
            except:
                lipschitz_scores[name] = 0.0

        # Normalize
        max_score = max(lipschitz_scores.values()) if lipschitz_scores else 1.0
        if max_score > 0:
            lipschitz_scores = {k: v / max_score for k, v in lipschitz_scores.items()}

        return lipschitz_scores

    def _power_iteration(self, W: torch.Tensor, num_iters: int = 10) -> float:
        """Compute spectral norm using power iteration."""
        v = torch.randn(W.size(1), device=W.device)
        v = v / torch.norm(v)

        for _ in range(num_iters):
            u = W @ v
            u = u / (torch.norm(u) + 1e-10)
            v = W.t() @ u
            v = v / (torch.norm(v) + 1e-10)

        return torch.norm(W @ v).item()
