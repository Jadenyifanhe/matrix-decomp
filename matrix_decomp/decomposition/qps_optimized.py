"""
QPS-Optimized Decomposition Methods

This module contains decomposition methods specifically designed to maintain
or improve QPS (Queries Per Second) during inference while still achieving
knowledge transfer from larger models.

WHY VANILLA SVD HURTS QPS:
--------------------------
Even though SVD reduces total FLOPs, it often hurts QPS because:
1. Two sequential matrix operations instead of one
2. Poor memory access patterns (cache misses)
3. GPU kernel launch overhead
4. Smaller matrices are less efficient on modern hardware

METHODS IN THIS MODULE:
-----------------------
1. DoRA (Weight-Decomposed LoRA): Better accuracy, similar QPS to LoRA
2. MergedLoRA: Merge adapter into base weight for deployment (best QPS)
3. RandomizedSVD: Faster SVD computation for large matrices
4. QRDecomposition: Alternative to SVD, sometimes better stability
5. KroneckerFactorization: Structured compression using Kronecker products
6. SemiStructuredDecomposition: For hardware-friendly sparsity patterns
"""

from typing import Tuple, Union, Optional, List
import torch
import torch.nn as nn
import numpy as np
from .base import BaseDecomposer


class DoRAAdapter(BaseDecomposer):
    """
    DoRA: Weight-Decomposed Low-Rank Adaptation

    DoRA decomposes weight updates into magnitude and direction components:
    W' = m * (W + BA) / ||W + BA||

    This often provides better accuracy than standard LoRA while maintaining
    similar computational efficiency.

    Reference: https://arxiv.org/abs/2402.09353

    Advantages over LoRA:
    - Better preservation of pretrained knowledge
    - More stable training
    - Often achieves higher accuracy

    Performance: Similar to LoRA when merged for inference
    """

    def __init__(
        self,
        rank: int,
        alpha: float = 1.0,
        init_method: str = "svd",
    ):
        """
        Initialize DoRA adapter.

        Args:
            rank: Low-rank dimension
            alpha: Scaling factor
            init_method: Initialization method ('svd', 'random')
        """
        super().__init__(rank)
        self.alpha = alpha
        self.init_method = init_method

    def decompose(
        self,
        weight: Union[torch.Tensor, np.ndarray],
        reference_weight: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Tuple:
        """
        Decompose weight into DoRA components.

        Args:
            weight: Target weight (from large model)
            reference_weight: Reference weight (from small model)

        Returns:
            (magnitude, A, B) where:
            - magnitude: Column-wise magnitude [out_features]
            - A: Low-rank matrix [rank, in_features]
            - B: Low-rank matrix [out_features, rank]
        """
        is_torch = isinstance(weight, torch.Tensor)
        device = weight.device if is_torch else None

        W = self._to_numpy(weight)
        W_ref = self._to_numpy(reference_weight) if reference_weight is not None else np.zeros_like(W)

        # Compute residual
        residual = W - W_ref
        out_features, in_features = residual.shape

        # Compute magnitude (column-wise L2 norm)
        magnitude = np.linalg.norm(W, axis=1)

        # Decompose residual using SVD
        if self.init_method == "svd":
            U, S, Vt = np.linalg.svd(residual, full_matrices=False)
            actual_rank = min(self.rank, len(S))

            U_r = U[:, :actual_rank]
            S_r = S[:actual_rank]
            Vt_r = Vt[:actual_rank, :]

            # Split singular values
            sqrt_S = np.sqrt(S_r)
            B = U_r * sqrt_S[np.newaxis, :]  # [out_features, rank]
            A = sqrt_S[:, np.newaxis] * Vt_r  # [rank, in_features]
        else:
            # Random initialization
            B = np.random.randn(out_features, self.rank) * 0.01
            A = np.zeros((self.rank, in_features))

        if is_torch:
            magnitude = self._to_tensor(magnitude, device)
            A = self._to_tensor(A, device)
            B = self._to_tensor(B, device)

        return magnitude, A, B

    def reconstruct(
        self,
        magnitude: Union[torch.Tensor, np.ndarray],
        A: Union[torch.Tensor, np.ndarray],
        B: Union[torch.Tensor, np.ndarray],
        base_weight: Optional[Union[torch.Tensor, np.ndarray]] = None,
        as_tensor: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Reconstruct weight with DoRA decomposition.

        Args:
            magnitude: Column magnitude
            A: Low-rank matrix A
            B: Low-rank matrix B
            base_weight: Base weight to add to
            as_tensor: Return as tensor

        Returns:
            Reconstructed weight
        """
        is_torch = isinstance(A, torch.Tensor)

        if is_torch:
            delta_W = B @ A
            if base_weight is not None:
                W = base_weight + self.alpha * delta_W
            else:
                W = delta_W

            # Apply magnitude normalization
            col_norm = torch.norm(W, dim=1, keepdim=True) + 1e-8
            W = magnitude.unsqueeze(1) * W / col_norm
        else:
            delta_W = B @ A
            if base_weight is not None:
                W = base_weight + self.alpha * delta_W
            else:
                W = delta_W

            col_norm = np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
            W = magnitude[:, np.newaxis] * W / col_norm

        if as_tensor and not isinstance(W, torch.Tensor):
            W = self._to_tensor(W)
        elif not as_tensor and isinstance(W, torch.Tensor):
            W = self._to_numpy(W)

        return W


class DoRALayer(nn.Module):
    """
    PyTorch module for DoRA layer.

    Can be merged for deployment to achieve same QPS as standard linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 1.0,
        base_weight: Optional[torch.Tensor] = None,
        bias: bool = False,
        device: str = "cpu"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # Base weight
        if base_weight is not None:
            self.weight = nn.Parameter(base_weight.to(device))
        else:
            self.weight = nn.Parameter(torch.randn(out_features, in_features, device=device) * 0.01)

        # Magnitude vector (column-wise)
        self.magnitude = nn.Parameter(torch.ones(out_features, device=device))

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features, device=device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device))
        else:
            self.register_parameter('bias', None)

        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return torch.nn.functional.linear(x, self.weight, self.bias)

        # Compute W + BA
        adapted_weight = self.weight + self.alpha * (self.lora_B @ self.lora_A)

        # Apply magnitude normalization
        col_norm = torch.norm(adapted_weight, dim=1, keepdim=True) + 1e-8
        normalized_weight = self.magnitude.unsqueeze(1) * adapted_weight / col_norm

        return torch.nn.functional.linear(x, normalized_weight, self.bias)

    def merge_weights(self):
        """Merge DoRA weights for deployment (better QPS)."""
        if not self.merged:
            with torch.no_grad():
                adapted_weight = self.weight + self.alpha * (self.lora_B @ self.lora_A)
                col_norm = torch.norm(adapted_weight, dim=1, keepdim=True) + 1e-8
                self.weight.data = self.magnitude.unsqueeze(1) * adapted_weight / col_norm

            # Zero out adapter
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()
            self.magnitude.data.fill_(1.0)
            self.merged = True


class MergedLoRADecomposer(BaseDecomposer):
    """
    LoRA decomposer that produces weights ready for merging.

    This is optimized for deployment: after decomposition, the adapter
    can be merged into the base weight for zero QPS overhead.

    Use case: When you want LoRA's knowledge transfer benefits but
    need maximum inference speed.
    """

    def __init__(
        self,
        rank: int,
        alpha: float = 1.0,
        merge_immediately: bool = True,
    ):
        """
        Initialize MergedLoRA decomposer.

        Args:
            rank: Low-rank dimension
            alpha: Scaling factor
            merge_immediately: If True, return merged weight directly
        """
        super().__init__(rank)
        self.alpha = alpha
        self.merge_immediately = merge_immediately

    def decompose(
        self,
        weight: Union[torch.Tensor, np.ndarray],
        reference_weight: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Union[torch.Tensor, Tuple]:
        """
        Decompose and optionally merge immediately.

        Args:
            weight: Target weight
            reference_weight: Base weight

        Returns:
            If merge_immediately: merged weight
            Otherwise: (A, B, merged_weight) tuple
        """
        is_torch = isinstance(weight, torch.Tensor)
        device = weight.device if is_torch else None

        W = self._to_numpy(weight)
        W_ref = self._to_numpy(reference_weight) if reference_weight is not None else np.zeros_like(W)

        # Compute residual
        residual = W - W_ref

        # SVD decomposition
        U, S, Vt = np.linalg.svd(residual, full_matrices=False)
        actual_rank = min(self.rank, len(S))

        U_r = U[:, :actual_rank]
        S_r = S[:actual_rank]
        Vt_r = Vt[:actual_rank, :]

        # Split singular values
        sqrt_S = np.sqrt(S_r)
        B = U_r * sqrt_S[np.newaxis, :]
        A = sqrt_S[:, np.newaxis] * Vt_r

        # Compute merged weight
        delta_W = B @ A
        merged = W_ref + self.alpha * delta_W

        if is_torch:
            merged = self._to_tensor(merged, device)
            if not self.merge_immediately:
                A = self._to_tensor(A, device)
                B = self._to_tensor(B, device)

        if self.merge_immediately:
            return merged
        else:
            return A, B, merged

    def reconstruct(self, *args, **kwargs):
        """For merged LoRA, the weight is already complete."""
        if len(args) == 1:
            return args[0]
        elif len(args) == 3:
            return args[2]  # Return merged weight
        else:
            raise ValueError("Unexpected number of arguments")


class RandomizedSVDDecomposer(BaseDecomposer):
    """
    Randomized SVD for faster decomposition of large matrices.

    Uses randomized algorithms to compute approximate SVD much faster
    than full SVD, especially for large matrices.

    This is particularly useful for:
    - Large embedding matrices
    - Wide/tall matrices where full SVD is expensive
    - When approximate decomposition is acceptable

    Reference: Halko et al., "Finding structure with randomness"
    """

    def __init__(
        self,
        rank: int,
        n_oversamples: int = 10,
        n_power_iterations: int = 2,
        random_state: Optional[int] = None,
    ):
        """
        Initialize Randomized SVD.

        Args:
            rank: Target rank
            n_oversamples: Extra samples for accuracy (higher = more accurate)
            n_power_iterations: Power iterations for accuracy
            random_state: Random seed for reproducibility
        """
        super().__init__(rank)
        self.n_oversamples = n_oversamples
        self.n_power_iterations = n_power_iterations
        self.random_state = random_state

    def decompose(
        self,
        weight: Union[torch.Tensor, np.ndarray],
        return_singular_values: bool = True
    ) -> Tuple:
        """
        Decompose using randomized SVD.

        Args:
            weight: Weight matrix
            return_singular_values: If True, return U, S, Vt

        Returns:
            (U, S, Vt) or (U_scaled, Vt_scaled)
        """
        is_torch = isinstance(weight, torch.Tensor)
        device = weight.device if is_torch else None

        W = self._to_numpy(weight)
        m, n = W.shape

        # Target rank with oversampling
        k = min(self.rank + self.n_oversamples, min(m, n))

        # Set random state
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Random projection
        Omega = np.random.randn(n, k)

        # Form Y = W @ Omega
        Y = W @ Omega

        # Power iterations for accuracy
        for _ in range(self.n_power_iterations):
            Y = W @ (W.T @ Y)

        # QR decomposition
        Q, _ = np.linalg.qr(Y)

        # Project and compute SVD of smaller matrix
        B = Q.T @ W
        U_B, S, Vt = np.linalg.svd(B, full_matrices=False)

        # Recover U
        U = Q @ U_B

        # Truncate to target rank
        actual_rank = min(self.rank, len(S))
        U_r = U[:, :actual_rank]
        S_r = S[:actual_rank]
        Vt_r = Vt[:actual_rank, :]

        if is_torch:
            U_r = self._to_tensor(U_r, device)
            S_r = self._to_tensor(S_r, device)
            Vt_r = self._to_tensor(Vt_r, device)

        if return_singular_values:
            return U_r, S_r, Vt_r
        else:
            sqrt_S = np.sqrt(S_r) if not is_torch else torch.sqrt(S_r)
            if is_torch:
                U_scaled = U_r * sqrt_S.unsqueeze(0)
                Vt_scaled = sqrt_S.unsqueeze(1) * Vt_r
            else:
                U_scaled = U_r * sqrt_S[np.newaxis, :]
                Vt_scaled = sqrt_S[:, np.newaxis] * Vt_r
            return U_scaled, Vt_scaled

    def reconstruct(self, *components, as_tensor: bool = True):
        """Reconstruct from SVD components."""
        if len(components) == 3:
            U, S, Vt = components
            is_torch = isinstance(U, torch.Tensor)
            if is_torch:
                W = U @ torch.diag(S) @ Vt
            else:
                W = U @ np.diag(S) @ Vt
        elif len(components) == 2:
            U_scaled, Vt_scaled = components
            is_torch = isinstance(U_scaled, torch.Tensor)
            W = U_scaled @ Vt_scaled
        else:
            raise ValueError("Expected 2 or 3 components")

        if as_tensor and not isinstance(W, torch.Tensor):
            W = self._to_tensor(W)
        elif not as_tensor and isinstance(W, torch.Tensor):
            W = self._to_numpy(W)

        return W


class QRDecomposer(BaseDecomposer):
    """
    QR-based decomposition as an alternative to SVD.

    QR decomposition: W = Q @ R where Q is orthogonal and R is upper triangular.

    Advantages over SVD:
    - Faster computation (O(mn^2) vs O(mn*min(m,n)))
    - More numerically stable in some cases
    - Can be updated incrementally

    For low-rank approximation, we use truncated QR with column pivoting.
    """

    def __init__(
        self,
        rank: int,
        use_pivoting: bool = True,
    ):
        """
        Initialize QR decomposer.

        Args:
            rank: Target rank
            use_pivoting: Use column pivoting for rank-revealing QR
        """
        super().__init__(rank)
        self.use_pivoting = use_pivoting

    def decompose(
        self,
        weight: Union[torch.Tensor, np.ndarray],
    ) -> Tuple:
        """
        Decompose using truncated QR.

        Args:
            weight: Weight matrix [m, n]

        Returns:
            (Q, R) where Q is [m, rank] and R is [rank, n]
        """
        is_torch = isinstance(weight, torch.Tensor)
        device = weight.device if is_torch else None

        W = self._to_numpy(weight)
        m, n = W.shape

        if self.use_pivoting:
            # QR with column pivoting (rank-revealing)
            from scipy.linalg import qr
            Q, R, P = qr(W, mode='economic', pivoting=True)

            actual_rank = min(self.rank, min(m, n))

            Q_r = Q[:, :actual_rank]
            R_r = R[:actual_rank, :]

            # Apply permutation to R
            P_inv = np.argsort(P)
            R_r = R_r[:, P_inv]
        else:
            Q, R = np.linalg.qr(W, mode='reduced')
            actual_rank = min(self.rank, min(m, n))
            Q_r = Q[:, :actual_rank]
            R_r = R[:actual_rank, :]

        if is_torch:
            Q_r = self._to_tensor(Q_r, device)
            R_r = self._to_tensor(R_r, device)

        return Q_r, R_r

    def reconstruct(
        self,
        Q: Union[torch.Tensor, np.ndarray],
        R: Union[torch.Tensor, np.ndarray],
        as_tensor: bool = True
    ):
        """Reconstruct from QR components."""
        is_torch = isinstance(Q, torch.Tensor)

        W = Q @ R

        if as_tensor and not isinstance(W, torch.Tensor):
            W = self._to_tensor(W)
        elif not as_tensor and isinstance(W, torch.Tensor):
            W = self._to_numpy(W)

        return W


class KroneckerFactorization(BaseDecomposer):
    """
    Kronecker Product Factorization

    Approximates W as a sum of Kronecker products:
    W ≈ Σ A_i ⊗ B_i

    This can provide significant compression for matrices that have
    inherent block structure (common in embedding layers).

    Advantages:
    - Very high compression ratios possible
    - Efficient computation using properties of Kronecker products
    - Works well for structured weight matrices

    Note: Best suited for square or near-square matrices where
    dimensions can be factored nicely.
    """

    def __init__(
        self,
        rank: int = 1,
        block_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize Kronecker factorization.

        Args:
            rank: Number of Kronecker terms (higher = more accurate)
            block_size: Size of B matrices (auto-determined if None)
        """
        super().__init__(rank)
        self.block_size = block_size

    def _find_factors(self, n: int) -> Tuple[int, int]:
        """Find two factors of n that are closest to sqrt(n)."""
        sqrt_n = int(np.sqrt(n))
        for i in range(sqrt_n, 0, -1):
            if n % i == 0:
                return i, n // i
        return 1, n

    def decompose(
        self,
        weight: Union[torch.Tensor, np.ndarray],
    ) -> Tuple:
        """
        Decompose using Kronecker factorization.

        Args:
            weight: Weight matrix [m, n]

        Returns:
            List of (A_i, B_i) pairs
        """
        is_torch = isinstance(weight, torch.Tensor)
        device = weight.device if is_torch else None

        W = self._to_numpy(weight)
        m, n = W.shape

        # Determine block sizes
        if self.block_size is not None:
            b_m, b_n = self.block_size
        else:
            b_m, _ = self._find_factors(m)
            b_n, _ = self._find_factors(n)

        a_m, a_n = m // b_m, n // b_n

        # Reshape to 4D tensor
        W_4d = W.reshape(a_m, b_m, a_n, b_n).transpose(0, 2, 1, 3).reshape(a_m * a_n, b_m * b_n)

        # Use SVD to find best rank-k Kronecker approximation
        U, S, Vt = np.linalg.svd(W_4d, full_matrices=False)

        components = []
        for i in range(min(self.rank, len(S))):
            # Extract rank-1 component
            u = U[:, i].reshape(a_m, a_n)  # A_i
            v = S[i] * Vt[i, :].reshape(b_m, b_n)  # B_i (includes singular value)

            if is_torch:
                u = self._to_tensor(u, device)
                v = self._to_tensor(v, device)

            components.append((u, v))

        return tuple(components)

    def reconstruct(
        self,
        *components,
        as_tensor: bool = True
    ):
        """Reconstruct from Kronecker components."""
        if len(components) == 1 and isinstance(components[0], (list, tuple)):
            components = components[0]

        # Get first component to determine dimensions and type
        A0, B0 = components[0]
        is_torch = isinstance(A0, torch.Tensor)

        if is_torch:
            # Use torch.kron
            W = torch.zeros(A0.shape[0] * B0.shape[0], A0.shape[1] * B0.shape[1],
                          device=A0.device, dtype=A0.dtype)
            for A, B in components:
                W += torch.kron(A, B)
        else:
            W = np.zeros((A0.shape[0] * B0.shape[0], A0.shape[1] * B0.shape[1]))
            for A, B in components:
                W += np.kron(A, B)

        if as_tensor and not isinstance(W, torch.Tensor):
            W = self._to_tensor(W)
        elif not as_tensor and isinstance(W, torch.Tensor):
            W = self._to_numpy(W)

        return W


class SemiStructuredDecomposition(BaseDecomposer):
    """
    Semi-Structured Sparsity Decomposition

    Combines low-rank decomposition with structured sparsity patterns
    that are hardware-friendly (e.g., 2:4 sparsity on NVIDIA Ampere GPUs).

    This can provide:
    - Better QPS than vanilla SVD (hardware acceleration)
    - More compression than pure low-rank
    - Maintained accuracy with careful selection

    Best for: Deployment on modern GPUs with sparsity support.
    """

    def __init__(
        self,
        rank: int,
        sparsity_ratio: float = 0.5,
        sparsity_pattern: str = "2:4",  # "2:4", "1:4", "unstructured"
    ):
        """
        Initialize semi-structured decomposition.

        Args:
            rank: Low-rank dimension
            sparsity_ratio: Fraction of weights to keep
            sparsity_pattern: Sparsity pattern type
        """
        super().__init__(rank)
        self.sparsity_ratio = sparsity_ratio
        self.sparsity_pattern = sparsity_pattern

    def _apply_24_sparsity(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply 2:4 sparsity pattern (keep 2 largest in each group of 4).

        Returns:
            (sparse_matrix, mask)
        """
        shape = matrix.shape
        flat = matrix.flatten()

        # Pad to multiple of 4
        pad_len = (4 - len(flat) % 4) % 4
        if pad_len > 0:
            flat = np.pad(flat, (0, pad_len))

        # Reshape to groups of 4
        groups = flat.reshape(-1, 4)

        # Find top 2 in each group
        mask = np.zeros_like(groups, dtype=bool)
        for i, group in enumerate(groups):
            top2_idx = np.argsort(np.abs(group))[-2:]
            mask[i, top2_idx] = True

        # Apply mask
        sparse = groups * mask

        # Reshape back
        sparse = sparse.flatten()[:np.prod(shape)].reshape(shape)
        mask = mask.flatten()[:np.prod(shape)].reshape(shape)

        return sparse, mask

    def decompose(
        self,
        weight: Union[torch.Tensor, np.ndarray],
    ) -> Tuple:
        """
        Decompose with semi-structured sparsity.

        Args:
            weight: Weight matrix

        Returns:
            (U_sparse, S, Vt_sparse, mask_U, mask_Vt)
        """
        is_torch = isinstance(weight, torch.Tensor)
        device = weight.device if is_torch else None

        W = self._to_numpy(weight)

        # First do SVD
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        actual_rank = min(self.rank, len(S))

        U_r = U[:, :actual_rank]
        S_r = S[:actual_rank]
        Vt_r = Vt[:actual_rank, :]

        # Apply sparsity to U and Vt
        if self.sparsity_pattern == "2:4":
            U_sparse, mask_U = self._apply_24_sparsity(U_r)
            Vt_sparse, mask_Vt = self._apply_24_sparsity(Vt_r)
        else:
            # Unstructured magnitude-based sparsity
            threshold_U = np.percentile(np.abs(U_r), (1 - self.sparsity_ratio) * 100)
            threshold_Vt = np.percentile(np.abs(Vt_r), (1 - self.sparsity_ratio) * 100)

            mask_U = np.abs(U_r) >= threshold_U
            mask_Vt = np.abs(Vt_r) >= threshold_Vt

            U_sparse = U_r * mask_U
            Vt_sparse = Vt_r * mask_Vt

        if is_torch:
            U_sparse = self._to_tensor(U_sparse, device)
            S_r = self._to_tensor(S_r, device)
            Vt_sparse = self._to_tensor(Vt_sparse, device)
            mask_U = self._to_tensor(mask_U.astype(np.float32), device)
            mask_Vt = self._to_tensor(mask_Vt.astype(np.float32), device)

        return U_sparse, S_r, Vt_sparse, mask_U, mask_Vt

    def reconstruct(
        self,
        U_sparse, S, Vt_sparse, mask_U=None, mask_Vt=None,
        as_tensor: bool = True
    ):
        """Reconstruct from sparse SVD components."""
        is_torch = isinstance(U_sparse, torch.Tensor)

        if is_torch:
            W = U_sparse @ torch.diag(S) @ Vt_sparse
        else:
            W = U_sparse @ np.diag(S) @ Vt_sparse

        if as_tensor and not isinstance(W, torch.Tensor):
            W = self._to_tensor(W)
        elif not as_tensor and isinstance(W, torch.Tensor):
            W = self._to_numpy(W)

        return W


# Convenience layers for PyTorch deployment

class MergedLoRALayer(nn.Module):
    """
    LoRA layer that can be fully merged for deployment.

    During training: W' = W + alpha * B @ A
    During inference (merged): W' is a single matrix

    This gives LoRA's training benefits with zero inference overhead.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 1.0,
        base_weight: Optional[torch.Tensor] = None,
        bias: bool = False,
        device: str = "cpu"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        if base_weight is not None:
            self.weight = nn.Parameter(base_weight.to(device))
        else:
            self.weight = nn.Parameter(torch.randn(out_features, in_features, device=device) * 0.01)

        self.lora_A = nn.Parameter(torch.randn(rank, in_features, device=device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device))
        else:
            self.register_parameter('bias', None)

        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return nn.functional.linear(x, self.weight, self.bias)
        else:
            base_out = nn.functional.linear(x, self.weight, self.bias)
            lora_out = nn.functional.linear(nn.functional.linear(x, self.lora_A), self.lora_B)
            return base_out + self.alpha * lora_out

    def merge(self):
        """Merge adapter into base weight for deployment."""
        if not self.merged:
            with torch.no_grad():
                self.weight.data += self.alpha * self.lora_B @ self.lora_A
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()
            self.merged = True

    def unmerge(self):
        """Unmerge is not possible after zeroing - use with caution."""
        raise NotImplementedError("Cannot unmerge after weights are zeroed")


class RandomizedSVDLayer(nn.Module):
    """
    Layer using randomized SVD decomposition.

    More efficient than full SVD for very large matrices.
    """

    def __init__(
        self,
        U: torch.Tensor,
        S: torch.Tensor,
        Vt: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        fused: bool = True
    ):
        super().__init__()

        if fused:
            # Fuse U and S for single multiplication
            self.U_fused = nn.Parameter(U * S.unsqueeze(0))
            self.Vt = nn.Parameter(Vt)
            self.register_parameter('S', None)
        else:
            self.U = nn.Parameter(U)
            self.S = nn.Parameter(S)
            self.Vt = nn.Parameter(Vt)
            self.U_fused = None

        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)

        self.fused = fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            intermediate = nn.functional.linear(x, self.Vt)
            output = nn.functional.linear(intermediate, self.U_fused, self.bias)
        else:
            # Three operations (worse QPS)
            intermediate = nn.functional.linear(x, self.Vt)
            intermediate = intermediate * self.S
            output = nn.functional.linear(intermediate, self.U, self.bias)
        return output
