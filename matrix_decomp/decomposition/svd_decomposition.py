"""
SVD (Singular Value Decomposition) based matrix decomposition.

SVD decomposes a matrix W into: W ≈ U @ S @ V^T
where U and V are orthogonal matrices and S is diagonal.

PERFORMANCE NOTE:
While SVD provides optimal low-rank approximation, it can hurt QPS because:
1. Two sequential matrix multiplications (U @ S @ V^T @ x) instead of one (W @ x)
2. Sequential operations reduce parallelization opportunities
3. Memory access patterns may be less cache-friendly
4. GPU kernel launch overhead for multiple operations

For better QPS, consider using LoRA adapters or Fused SVD instead.
"""

from typing import Tuple, Union, Optional
import torch
import numpy as np
from .base import BaseDecomposer


class SVDDecomposer(BaseDecomposer):
    """
    SVD-based matrix decomposition.

    Decomposes weight matrices using Singular Value Decomposition.
    """

    def __init__(
        self,
        rank: int,
        use_full_matrices: bool = False,
        rank_selection: str = "fixed",  # 'fixed', 'energy', 'adaptive'
        energy_threshold: float = 0.9,  # For energy-based rank selection
    ):
        """
        Initialize SVD decomposer.

        Args:
            rank: Target rank for decomposition
            use_full_matrices: If True, compute full U and V matrices
            rank_selection: Method for selecting rank
                - 'fixed': Use the specified rank
                - 'energy': Select rank to preserve energy_threshold of singular values
                - 'adaptive': Adaptive rank based on matrix shape
            energy_threshold: Threshold for energy-based rank selection (0-1)
        """
        super().__init__(rank)
        self.use_full_matrices = use_full_matrices
        self.rank_selection = rank_selection
        self.energy_threshold = energy_threshold

    def decompose(
        self,
        weight: Union[torch.Tensor, np.ndarray],
        return_singular_values: bool = True
    ) -> Tuple[Union[torch.Tensor, np.ndarray], ...]:
        """
        Decompose weight matrix using SVD.

        Args:
            weight: Weight matrix to decompose [m, n]
            return_singular_values: If True, return S separately; if False, merge into U

        Returns:
            If return_singular_values=True: (U, S, Vt)
                - U: Left singular vectors [m, rank]
                - S: Singular values [rank]
                - Vt: Right singular vectors transposed [rank, n]
            If return_singular_values=False: (U_scaled, Vt)
                - U_scaled: U @ sqrt(S) [m, rank]
                - Vt: sqrt(S) @ Vt [rank, n]
        """
        is_torch = isinstance(weight, torch.Tensor)
        device = weight.device if is_torch else None

        # Convert to numpy for SVD computation
        W = self._to_numpy(weight)

        # Perform SVD
        # W = U @ diag(S) @ Vt
        U, S, Vt = np.linalg.svd(W, full_matrices=self.use_full_matrices)

        # Select rank
        actual_rank = self._select_rank(S)

        # Truncate to rank
        U_r = U[:, :actual_rank]  # [m, rank]
        S_r = S[:actual_rank]      # [rank]
        Vt_r = Vt[:actual_rank, :] # [rank, n]

        # Convert back to torch if needed
        if is_torch:
            U_r = self._to_tensor(U_r, device)
            S_r = self._to_tensor(S_r, device)
            Vt_r = self._to_tensor(Vt_r, device)

        if return_singular_values:
            return U_r, S_r, Vt_r
        else:
            # Merge singular values into U and Vt for single matrix multiplication
            # Split singular values equally: U @ sqrt(S) and sqrt(S) @ Vt
            sqrt_S = np.sqrt(S_r) if not is_torch else torch.sqrt(S_r)
            if is_torch:
                U_scaled = U_r * sqrt_S.unsqueeze(0)
                Vt_scaled = sqrt_S.unsqueeze(1) * Vt_r
            else:
                U_scaled = U_r * sqrt_S[np.newaxis, :]
                Vt_scaled = sqrt_S[:, np.newaxis] * Vt_r
            return U_scaled, Vt_scaled

    def reconstruct(
        self,
        *components,
        as_tensor: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Reconstruct weight matrix from SVD components.

        Args:
            *components: Either (U, S, Vt) or (U_scaled, Vt_scaled)
            as_tensor: If True, return torch.Tensor; otherwise numpy array

        Returns:
            Reconstructed weight matrix
        """
        if len(components) == 3:
            U, S, Vt = components
            is_torch = isinstance(U, torch.Tensor)
            if is_torch:
                # W = U @ diag(S) @ Vt
                W = U @ torch.diag(S) @ Vt
            else:
                W = U @ np.diag(S) @ Vt
        elif len(components) == 2:
            U_scaled, Vt_scaled = components
            is_torch = isinstance(U_scaled, torch.Tensor)
            if is_torch:
                W = U_scaled @ Vt_scaled
            else:
                W = U_scaled @ Vt_scaled
        else:
            raise ValueError(f"Expected 2 or 3 components, got {len(components)}")

        if as_tensor and not isinstance(W, torch.Tensor):
            W = self._to_tensor(W)
        elif not as_tensor and isinstance(W, torch.Tensor):
            W = self._to_numpy(W)

        return W

    def _select_rank(self, singular_values: np.ndarray) -> int:
        """
        Select the actual rank to use based on rank_selection method.

        Args:
            singular_values: Array of singular values in descending order

        Returns:
            Selected rank
        """
        max_rank = len(singular_values)

        if self.rank_selection == "fixed":
            return min(self.rank, max_rank)

        elif self.rank_selection == "energy":
            # Select rank to preserve certain percentage of energy
            total_energy = np.sum(singular_values ** 2)
            cumulative_energy = np.cumsum(singular_values ** 2)
            preserved_ratio = cumulative_energy / total_energy
            rank = np.searchsorted(preserved_ratio, self.energy_threshold) + 1
            return min(rank, max_rank)

        elif self.rank_selection == "adaptive":
            # Adaptive: use rank proportional to matrix dimensions
            # but not exceeding the specified rank
            adaptive_rank = int(np.sqrt(max_rank))
            return min(adaptive_rank, self.rank, max_rank)

        else:
            raise ValueError(f"Unknown rank_selection method: {self.rank_selection}")

    def analyze_singular_values(
        self,
        weight: Union[torch.Tensor, np.ndarray],
        plot: bool = False
    ) -> dict:
        """
        Analyze the singular value distribution of a weight matrix.

        This is useful for understanding how much information is captured
        at different ranks.

        Args:
            weight: Weight matrix to analyze
            plot: If True, plot the singular value distribution

        Returns:
            Dictionary with analysis results
        """
        W = self._to_numpy(weight)
        _, S, _ = np.linalg.svd(W, full_matrices=False)

        # Compute energy preserved at different ranks
        total_energy = np.sum(S ** 2)
        cumulative_energy = np.cumsum(S ** 2)
        energy_ratio = cumulative_energy / total_energy

        # Find rank for different energy thresholds
        thresholds = [0.5, 0.75, 0.9, 0.95, 0.99]
        ranks_for_thresholds = {}
        for threshold in thresholds:
            rank = np.searchsorted(energy_ratio, threshold) + 1
            ranks_for_thresholds[f"{threshold:.0%}"] = rank

        analysis = {
            "num_singular_values": len(S),
            "largest_singular_value": float(S[0]),
            "smallest_singular_value": float(S[-1]),
            "condition_number": float(S[0] / S[-1]) if S[-1] > 1e-10 else float('inf'),
            "singular_values": S,
            "energy_ratio": energy_ratio,
            "ranks_for_energy": ranks_for_thresholds,
        }

        if plot:
            try:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                # Plot singular values
                axes[0].semilogy(S)
                axes[0].set_xlabel("Index")
                axes[0].set_ylabel("Singular Value")
                axes[0].set_title("Singular Value Distribution")
                axes[0].grid(True)

                # Plot cumulative energy
                axes[1].plot(energy_ratio)
                axes[1].axhline(y=0.9, color='r', linestyle='--', label='90% energy')
                axes[1].axhline(y=0.95, color='g', linestyle='--', label='95% energy')
                axes[1].set_xlabel("Rank")
                axes[1].set_ylabel("Energy Preserved")
                axes[1].set_title("Cumulative Energy")
                axes[1].legend()
                axes[1].grid(True)

                plt.tight_layout()
                plt.show()
            except ImportError:
                print("matplotlib not available for plotting")

        return analysis


class TruncatedSVD(SVDDecomposer):
    """
    Truncated SVD with automatic rank selection based on explained variance.

    This is useful when you don't know the optimal rank in advance.
    """

    def __init__(self, explained_variance: float = 0.9):
        """
        Initialize Truncated SVD.

        Args:
            explained_variance: Target explained variance ratio (0-1)
        """
        # Use a large rank initially, will be truncated based on variance
        super().__init__(
            rank=1000000,  # Will be overridden
            rank_selection="energy",
            energy_threshold=explained_variance
        )
        self.explained_variance = explained_variance
