"""
Tucker Decomposition for tensor compression.

Tucker decomposition factorizes a tensor into:
W ≈ G ×1 U1 ×2 U2 ×3 U3 ... ×N UN

where:
- G is a smaller core tensor
- U1, U2, ..., UN are factor matrices for each dimension

This is particularly useful for:
1. Convolutional layers with 4D weights [out_channels, in_channels, height, width]
2. Higher compression ratios for multi-dimensional tensors
3. Flexible rank selection per dimension

PERFORMANCE NOTES:
- More complex than SVD, requires multiple matrix multiplications
- Can achieve higher compression ratios
- Best for convolutional layers where 4D structure is natural
"""

from typing import Tuple, Union, List, Optional
import torch
import numpy as np

try:
    import tensorly as tl
    from tensorly.decomposition import tucker
    TENSORLY_AVAILABLE = True
except ImportError:
    TENSORLY_AVAILABLE = False

from .base import BaseDecomposer


class TuckerDecomposer(BaseDecomposer):
    """
    Tucker decomposition for multi-dimensional weight tensors.

    Decomposes a tensor into a core tensor and factor matrices.
    """

    def __init__(
        self,
        rank: Union[int, List[int], Tuple[int, ...]],
        rank_selection: str = "fixed",  # 'fixed', 'percentage'
        rank_percentage: float = 0.5,
    ):
        """
        Initialize Tucker decomposer.

        Args:
            rank: Target rank(s) for decomposition
                - If int: use same rank for all dimensions
                - If list/tuple: specify rank for each dimension
            rank_selection: Method for selecting ranks
                - 'fixed': Use specified ranks
                - 'percentage': Use percentage of original dimension
            rank_percentage: Percentage of original dimension to use (0-1)
        """
        if not TENSORLY_AVAILABLE:
            raise ImportError(
                "TensorLy is required for Tucker decomposition. "
                "Install it with: pip install tensorly"
            )

        super().__init__(rank if isinstance(rank, int) else rank[0])
        self.rank_list = rank if isinstance(rank, (list, tuple)) else None
        self.rank_selection = rank_selection
        self.rank_percentage = rank_percentage

    def decompose(
        self,
        weight: Union[torch.Tensor, np.ndarray],
        init: str = "svd"
    ) -> Tuple[Union[torch.Tensor, np.ndarray], List[Union[torch.Tensor, np.ndarray]]]:
        """
        Decompose weight tensor using Tucker decomposition.

        Args:
            weight: Weight tensor to decompose (can be 2D, 3D, 4D, etc.)
            init: Initialization method for Tucker decomposition

        Returns:
            (core, factors) where:
            - core: Core tensor with reduced dimensions
            - factors: List of factor matrices, one per dimension
        """
        is_torch = isinstance(weight, torch.Tensor)
        device = weight.device if is_torch else None

        # Convert to numpy for tensorly
        W = self._to_numpy(weight)

        # Determine ranks for each dimension
        ranks = self._get_ranks_for_shape(W.shape)

        # Perform Tucker decomposition
        # Returns (core, factors) where factors is a list of matrices
        core, factors = tucker(W, rank=ranks, init=init)

        # Convert back to torch if needed
        if is_torch:
            core = self._to_tensor(core, device)
            factors = [self._to_tensor(f, device) for f in factors]

        return core, factors

    def reconstruct(
        self,
        core: Union[torch.Tensor, np.ndarray],
        factors: List[Union[torch.Tensor, np.ndarray]],
        as_tensor: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Reconstruct tensor from Tucker components.

        Args:
            core: Core tensor
            factors: List of factor matrices
            as_tensor: Return as torch tensor

        Returns:
            Reconstructed weight tensor
        """
        is_torch = isinstance(core, torch.Tensor)

        if is_torch:
            # Perform tensor contraction using einsum
            W = core
            for i, factor in enumerate(factors):
                # Contract along dimension i
                W = self._mode_product_torch(W, factor, mode=i)
        else:
            # Use tensorly for numpy
            core_np = self._to_numpy(core)
            factors_np = [self._to_numpy(f) for f in factors]
            tl.set_backend('numpy')
            W = tl.tucker_to_tensor((core_np, factors_np))

        if as_tensor and not isinstance(W, torch.Tensor):
            W = self._to_tensor(W)
        elif not as_tensor and isinstance(W, torch.Tensor):
            W = self._to_numpy(W)

        return W

    def _mode_product_torch(
        self,
        tensor: torch.Tensor,
        matrix: torch.Tensor,
        mode: int
    ) -> torch.Tensor:
        """
        Compute mode-n product of tensor with matrix.

        Args:
            tensor: Input tensor
            matrix: Matrix to multiply
            mode: Mode (dimension) along which to multiply

        Returns:
            Result of mode-n product
        """
        # Move the target mode to the front
        dims = list(range(tensor.ndim))
        dims[0], dims[mode] = dims[mode], dims[0]
        tensor_permuted = tensor.permute(dims)

        # Reshape to matrix
        original_shape = tensor_permuted.shape
        tensor_matrix = tensor_permuted.reshape(original_shape[0], -1)

        # Multiply
        result = matrix @ tensor_matrix

        # Reshape back
        new_shape = (matrix.shape[0],) + original_shape[1:]
        result = result.reshape(new_shape)

        # Permute back
        result = result.permute(dims)

        return result

    def _get_ranks_for_shape(self, shape: Tuple[int, ...]) -> List[int]:
        """
        Get ranks for each dimension based on tensor shape.

        Args:
            shape: Shape of the tensor

        Returns:
            List of ranks for each dimension
        """
        if self.rank_list is not None:
            # Use provided ranks
            if len(self.rank_list) != len(shape):
                raise ValueError(
                    f"Number of ranks ({len(self.rank_list)}) must match "
                    f"number of dimensions ({len(shape)})"
                )
            return list(self.rank_list)

        elif self.rank_selection == "fixed":
            # Use same rank for all dimensions
            return [min(self.rank, dim) for dim in shape]

        elif self.rank_selection == "percentage":
            # Use percentage of each dimension
            return [max(1, int(dim * self.rank_percentage)) for dim in shape]

        else:
            raise ValueError(f"Unknown rank_selection: {self.rank_selection}")

    def create_tucker_layer(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        device: str = "cpu"
    ) -> 'TuckerLayer':
        """
        Create a Tucker decomposed layer.

        Args:
            weight: Original weight to decompose
            bias: Optional bias
            device: Device to create layer on

        Returns:
            TuckerLayer module
        """
        core, factors = self.decompose(weight)

        return TuckerLayer(
            core=core.to(device),
            factors=[f.to(device) for f in factors],
            bias=bias.to(device) if bias is not None else None,
            original_shape=weight.shape
        )


class TuckerLayer(torch.nn.Module):
    """
    PyTorch module implementing Tucker decomposed layer.

    Useful for convolutional layers with 4D weights.
    """

    def __init__(
        self,
        core: torch.Tensor,
        factors: List[torch.Tensor],
        bias: Optional[torch.Tensor] = None,
        original_shape: Tuple[int, ...] = None
    ):
        """
        Initialize Tucker layer.

        Args:
            core: Core tensor
            factors: List of factor matrices
            bias: Optional bias
            original_shape: Original weight shape (for reference)
        """
        super().__init__()
        self.core = torch.nn.Parameter(core)
        self.factors = torch.nn.ParameterList([torch.nn.Parameter(f) for f in factors])

        if bias is not None:
            self.bias = torch.nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)

        self.original_shape = original_shape
        self.ndim = len(factors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - should be implemented based on specific layer type.

        This is a base implementation for 2D (matrix) case.
        Override for Conv2D or other layer types.
        """
        if self.ndim == 2:
            # For 2D matrix case: W = U1 @ core @ U2^T
            weight = self.factors[0] @ self.core @ self.factors[1].T
            return torch.nn.functional.linear(x, weight, self.bias)
        else:
            # For higher dimensions, reconstruct full weight
            # (In practice, you'd want to optimize this for specific layer types)
            weight = self._reconstruct_weight()
            if self.ndim == 4:
                # Assume Conv2D
                return torch.nn.functional.conv2d(x, weight, self.bias)
            else:
                raise NotImplementedError(f"Forward pass not implemented for {self.ndim}D tensors")

    def _reconstruct_weight(self) -> torch.Tensor:
        """Reconstruct the full weight tensor."""
        W = self.core
        for i, factor in enumerate(self.factors):
            W = self._mode_product(W, factor, i)
        return W

    def _mode_product(self, tensor: torch.Tensor, matrix: torch.Tensor, mode: int) -> torch.Tensor:
        """Mode-n product."""
        dims = list(range(tensor.ndim))
        dims[0], dims[mode] = dims[mode], dims[0]
        tensor_permuted = tensor.permute(dims)

        original_shape = tensor_permuted.shape
        tensor_matrix = tensor_permuted.reshape(original_shape[0], -1)
        result = matrix @ tensor_matrix

        new_shape = (matrix.shape[0],) + original_shape[1:]
        result = result.reshape(new_shape)
        result = result.permute(dims)

        return result

    def extra_repr(self) -> str:
        """String representation."""
        return (f'original_shape={self.original_shape}, '
                f'core_shape={tuple(self.core.shape)}, '
                f'ranks={[f.shape[0] for f in self.factors]}')
