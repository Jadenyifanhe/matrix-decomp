"""
CP (CANDECOMP/PARAFAC) Decomposition for tensor compression.

CP decomposition factorizes a tensor as a sum of rank-1 tensors:
W ≈ Σ(r=1 to R) λ_r * (a_r ⊗ b_r ⊗ c_r ⊗ ...)

where:
- λ_r are weights
- a_r, b_r, c_r are factor vectors
- ⊗ denotes outer product

ADVANTAGES:
1. Maximum compression - separates all dimensions
2. Simpler structure than Tucker (no core tensor)
3. Fewer parameters than Tucker for same rank

DISADVANTAGES:
1. Less stable than Tucker decomposition
2. May require higher rank to maintain accuracy
3. Harder to optimize

PERFORMANCE:
- Can be faster than Tucker (fewer operations)
- Best for extreme compression scenarios
"""

from typing import Tuple, Union, List, Optional
import torch
import numpy as np

try:
    import tensorly as tl
    from tensorly.decomposition import parafac
    TENSORLY_AVAILABLE = True
except ImportError:
    TENSORLY_AVAILABLE = False

from .base import BaseDecomposer


class CPDecomposer(BaseDecomposer):
    """
    CP (CANDECOMP/PARAFAC) decomposition for tensor compression.

    Decomposes a tensor into a sum of rank-1 components.
    """

    def __init__(
        self,
        rank: int,
        init: str = "svd",
        n_iter_max: int = 100,
        tol: float = 1e-6,
    ):
        """
        Initialize CP decomposer.

        Args:
            rank: Number of components (CP rank)
            init: Initialization method ('svd', 'random')
            n_iter_max: Maximum number of iterations
            tol: Convergence tolerance
        """
        if not TENSORLY_AVAILABLE:
            raise ImportError(
                "TensorLy is required for CP decomposition. "
                "Install it with: pip install tensorly"
            )

        super().__init__(rank)
        self.init = init
        self.n_iter_max = n_iter_max
        self.tol = tol

    def decompose(
        self,
        weight: Union[torch.Tensor, np.ndarray],
        return_weights: bool = True
    ) -> Union[
        Tuple[Union[torch.Tensor, np.ndarray], List[Union[torch.Tensor, np.ndarray]]],
        List[Union[torch.Tensor, np.ndarray]]
    ]:
        """
        Decompose weight tensor using CP decomposition.

        Args:
            weight: Weight tensor to decompose
            return_weights: If True, return (weights, factors); else just factors

        Returns:
            If return_weights=True: (weights, factors)
                - weights: Component weights [rank]
                - factors: List of factor matrices [dim_i, rank]
            If return_weights=False: factors only
                Weights are absorbed into the first factor
        """
        is_torch = isinstance(weight, torch.Tensor)
        device = weight.device if is_torch else None

        # Convert to numpy for tensorly
        W = self._to_numpy(weight)

        # Perform CP decomposition
        # Returns (weights, factors)
        tl.set_backend('numpy')
        cp_tensor = parafac(
            W,
            rank=self.rank,
            init=self.init,
            n_iter_max=self.n_iter_max,
            tol=self.tol
        )

        weights, factors = cp_tensor

        # Convert back to torch if needed
        if is_torch:
            weights = self._to_tensor(weights, device)
            factors = [self._to_tensor(f, device) for f in factors]

        if return_weights:
            return weights, factors
        else:
            # Absorb weights into first factor
            if is_torch:
                factors[0] = factors[0] * weights.unsqueeze(0)
            else:
                factors[0] = factors[0] * weights[np.newaxis, :]
            return factors

    def reconstruct(
        self,
        *args,
        as_tensor: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Reconstruct tensor from CP components.

        Args:
            *args: Either (weights, factors) or (factors,)
            as_tensor: Return as torch tensor

        Returns:
            Reconstructed weight tensor
        """
        if len(args) == 2:
            weights, factors = args
        elif len(args) == 1:
            factors = args[0]
            # Create uniform weights
            if isinstance(factors[0], torch.Tensor):
                weights = torch.ones(factors[0].shape[1])
            else:
                weights = np.ones(factors[0].shape[1])
        else:
            raise ValueError(f"Expected 1 or 2 arguments, got {len(args)}")

        is_torch = isinstance(factors[0], torch.Tensor)

        if is_torch:
            W = self._reconstruct_torch(weights, factors)
        else:
            # Use tensorly for numpy
            weights_np = self._to_numpy(weights)
            factors_np = [self._to_numpy(f) for f in factors]
            tl.set_backend('numpy')
            W = tl.cp_to_tensor((weights_np, factors_np))

        if as_tensor and not isinstance(W, torch.Tensor):
            W = self._to_tensor(W)
        elif not as_tensor and isinstance(W, torch.Tensor):
            W = self._to_numpy(W)

        return W

    def _reconstruct_torch(
        self,
        weights: torch.Tensor,
        factors: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Reconstruct tensor from CP components using PyTorch.

        Args:
            weights: Component weights [rank]
            factors: List of factor matrices [dim_i, rank]

        Returns:
            Reconstructed tensor
        """
        rank = weights.shape[0]
        shape = [f.shape[0] for f in factors]

        # Initialize output tensor
        W = torch.zeros(shape, device=weights.device, dtype=weights.dtype)

        # Sum over rank-1 components
        for r in range(rank):
            # Compute rank-1 tensor: λ_r * (a_r ⊗ b_r ⊗ c_r ⊗ ...)
            component = weights[r]
            for factor in factors:
                component = torch.outer(component.flatten(), factor[:, r]).reshape(-1)

            # Reshape to original shape
            component = component.reshape(shape)
            W += component

        return W

    def create_cp_layer(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        device: str = "cpu"
    ) -> 'CPLayer':
        """
        Create a CP decomposed layer.

        Args:
            weight: Original weight to decompose
            bias: Optional bias
            device: Device to create layer on

        Returns:
            CPLayer module
        """
        weights, factors = self.decompose(weight, return_weights=True)

        return CPLayer(
            weights=weights.to(device),
            factors=[f.to(device) for f in factors],
            bias=bias.to(device) if bias is not None else None,
            original_shape=weight.shape
        )


class CPLayer(torch.nn.Module):
    """
    PyTorch module implementing CP decomposed layer.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        factors: List[torch.Tensor],
        bias: Optional[torch.Tensor] = None,
        original_shape: Tuple[int, ...] = None
    ):
        """
        Initialize CP layer.

        Args:
            weights: Component weights [rank]
            factors: List of factor matrices [dim_i, rank]
            bias: Optional bias
            original_shape: Original weight shape
        """
        super().__init__()
        self.weights = torch.nn.Parameter(weights)
        self.factors = torch.nn.ParameterList([torch.nn.Parameter(f) for f in factors])

        if bias is not None:
            self.bias = torch.nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)

        self.original_shape = original_shape
        self.ndim = len(factors)
        self.rank = weights.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        For 2D case, we can compute efficiently without full reconstruction.
        """
        if self.ndim == 2:
            # For matrix: W = Σ λ_r * (a_r ⊗ b_r) = Σ λ_r * a_r * b_r^T
            # W @ x = Σ λ_r * a_r * (b_r^T @ x)
            # This can be computed efficiently
            output = torch.zeros(x.shape[0], self.factors[0].shape[0], device=x.device)

            for r in range(self.rank):
                # Compute b_r^T @ x
                proj = torch.nn.functional.linear(x, self.factors[1][:, r].unsqueeze(0))
                # Scale by weight and outer product with a_r
                output += self.weights[r] * proj * self.factors[0][:, r].unsqueeze(0)

            if self.bias is not None:
                output += self.bias

            return output
        else:
            # For higher dimensions, reconstruct and apply
            weight = self._reconstruct_weight()
            if self.ndim == 4:
                return torch.nn.functional.conv2d(x, weight, self.bias)
            else:
                raise NotImplementedError(f"Forward pass not implemented for {self.ndim}D tensors")

    def _reconstruct_weight(self) -> torch.Tensor:
        """Reconstruct full weight tensor."""
        rank = self.weights.shape[0]
        shape = [f.shape[0] for f in self.factors]

        W = torch.zeros(shape, device=self.weights.device, dtype=self.weights.dtype)

        for r in range(rank):
            component = self.weights[r]
            for factor in self.factors:
                component = torch.outer(component.flatten(), factor[:, r]).reshape(-1)
            component = component.reshape(shape)
            W += component

        return W

    def extra_repr(self) -> str:
        """String representation."""
        return (f'original_shape={self.original_shape}, '
                f'rank={self.rank}, '
                f'factor_shapes={[tuple(f.shape) for f in self.factors]}')
