"""
Dimension adaptation utilities for handling size mismatches between models.

When transferring decomposed weights from a large model to a small model,
the dimensions may not match. This module provides utilities to adapt dimensions.
"""

from typing import Union, Tuple, Optional
import torch
import numpy as np


class DimensionAdapter:
    """
    Adapter for handling dimension mismatches between source and target models.
    """

    def __init__(self, method: str = "truncate"):
        """
        Initialize dimension adapter.

        Args:
            method: Adaptation method
                - 'truncate': Simply truncate larger dimensions
                - 'pad': Pad smaller dimensions with zeros
                - 'interpolate': Interpolate to match dimensions
                - 'project': Use learned projection
        """
        self.method = method

    def adapt_matrix(
        self,
        matrix: Union[torch.Tensor, np.ndarray],
        target_shape: Tuple[int, ...],
        dim: Optional[int] = None
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Adapt a matrix to match target shape.

        Args:
            matrix: Input matrix
            target_shape: Desired output shape
            dim: Specific dimension to adapt (None for all)

        Returns:
            Adapted matrix
        """
        is_torch = isinstance(matrix, torch.Tensor)

        if self.method == "truncate":
            return self._truncate(matrix, target_shape, is_torch)
        elif self.method == "pad":
            return self._pad(matrix, target_shape, is_torch)
        elif self.method == "interpolate":
            return self._interpolate(matrix, target_shape, is_torch)
        else:
            raise ValueError(f"Unknown adaptation method: {self.method}")

    def _truncate(
        self,
        matrix: Union[torch.Tensor, np.ndarray],
        target_shape: Tuple[int, ...],
        is_torch: bool
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Truncate matrix to target shape.

        If target is larger, this will fail. Use 'pad' method instead.
        """
        if is_torch:
            slices = tuple(slice(0, min(s, t)) for s, t in zip(matrix.shape, target_shape))
            result = matrix[slices]

            # If target is larger, need to pad
            if any(t > s for s, t in zip(result.shape, target_shape)):
                return self._pad(matrix, target_shape, is_torch)

            return result
        else:
            slices = tuple(slice(0, min(s, t)) for s, t in zip(matrix.shape, target_shape))
            result = matrix[slices]

            if any(t > s for s, t in zip(result.shape, target_shape)):
                return self._pad(matrix, target_shape, is_torch)

            return result

    def _pad(
        self,
        matrix: Union[torch.Tensor, np.ndarray],
        target_shape: Tuple[int, ...],
        is_torch: bool
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Pad matrix to target shape with zeros.
        """
        if is_torch:
            # Create zero tensor with target shape
            result = torch.zeros(target_shape, dtype=matrix.dtype, device=matrix.device)
            # Copy over existing values
            slices = tuple(slice(0, min(s, t)) for s, t in zip(matrix.shape, target_shape))
            result[slices] = matrix[slices]
            return result
        else:
            result = np.zeros(target_shape, dtype=matrix.dtype)
            slices = tuple(slice(0, min(s, t)) for s, t in zip(matrix.shape, target_shape))
            result[slices] = matrix[slices]
            return result

    def _interpolate(
        self,
        matrix: Union[torch.Tensor, np.ndarray],
        target_shape: Tuple[int, ...],
        is_torch: bool
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Interpolate matrix to target shape.

        Uses bilinear interpolation for 2D matrices.
        """
        if len(matrix.shape) != 2 or len(target_shape) != 2:
            # Fall back to truncate for non-2D
            return self._truncate(matrix, target_shape, is_torch)

        if is_torch:
            # Use PyTorch's interpolate function
            # Reshape to [1, 1, H, W] for interpolate
            matrix_4d = matrix.unsqueeze(0).unsqueeze(0)
            interpolated = torch.nn.functional.interpolate(
                matrix_4d,
                size=target_shape,
                mode='bilinear',
                align_corners=False
            )
            return interpolated.squeeze(0).squeeze(0)
        else:
            # Use scipy for numpy
            try:
                from scipy.ndimage import zoom
                zoom_factors = [t / s for s, t in zip(matrix.shape, target_shape)]
                return zoom(matrix, zoom_factors, order=1)
            except ImportError:
                # Fall back to truncate if scipy not available
                return self._truncate(matrix, target_shape, is_torch)


def adapt_dimensions(
    matrix: Union[torch.Tensor, np.ndarray],
    target_rows: Optional[int] = None,
    target_cols: Optional[int] = None,
    method: str = "truncate"
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convenience function to adapt matrix dimensions.

    Args:
        matrix: Input matrix
        target_rows: Target number of rows (None to keep original)
        target_cols: Target number of columns (None to keep original)
        method: Adaptation method ('truncate', 'pad', 'interpolate')

    Returns:
        Adapted matrix
    """
    current_shape = matrix.shape
    target_shape = (
        target_rows if target_rows is not None else current_shape[0],
        target_cols if target_cols is not None else current_shape[1]
    )

    adapter = DimensionAdapter(method=method)
    return adapter.adapt_matrix(matrix, target_shape)


def smart_dimension_adaptation(
    source_components: Tuple,
    target_shape: Tuple[int, int],
    decomposition_type: str = "svd"
) -> Tuple:
    """
    Smart dimension adaptation for decomposed components.

    This handles the common case where a large model's decomposed weights
    need to be adapted to fit a smaller model.

    Args:
        source_components: Decomposed components from large model
            - For SVD: (U, S, Vt) or (U_fused, Vt_fused)
            - For LoRA: (A, B)
        target_shape: Target weight shape [out_dim, in_dim]
        decomposition_type: Type of decomposition ('svd', 'lora', 'fused_svd')

    Returns:
        Adapted components that match target shape
    """
    target_out, target_in = target_shape

    if decomposition_type == "svd":
        if len(source_components) == 3:
            U, S, Vt = source_components
            # Adapt U to match output dimension
            U_adapted = adapt_dimensions(U, target_rows=target_out, method="truncate")
            # Adapt Vt to match input dimension
            Vt_adapted = adapt_dimensions(Vt, target_cols=target_in, method="truncate")
            # S stays the same (might need to truncate rank if U or Vt got smaller)
            new_rank = min(U_adapted.shape[1], Vt_adapted.shape[0], len(S))
            S_adapted = S[:new_rank]
            U_adapted = U_adapted[:, :new_rank]
            Vt_adapted = Vt_adapted[:new_rank, :]
            return U_adapted, S_adapted, Vt_adapted
        else:
            raise ValueError("SVD decomposition should have 3 components")

    elif decomposition_type == "fused_svd":
        if len(source_components) == 2:
            U_fused, Vt_fused = source_components
            # Adapt dimensions
            U_adapted = adapt_dimensions(U_fused, target_rows=target_out, method="truncate")
            Vt_adapted = adapt_dimensions(Vt_fused, target_cols=target_in, method="truncate")
            # Ensure rank matches
            new_rank = min(U_adapted.shape[1], Vt_adapted.shape[0])
            U_adapted = U_adapted[:, :new_rank]
            Vt_adapted = Vt_adapted[:new_rank, :]
            return U_adapted, Vt_adapted
        else:
            raise ValueError("Fused SVD should have 2 components")

    elif decomposition_type == "lora":
        if len(source_components) == 2:
            A, B = source_components
            # A is [rank, in_dim], B is [out_dim, rank]
            A_adapted = adapt_dimensions(A, target_cols=target_in, method="truncate")
            B_adapted = adapt_dimensions(B, target_rows=target_out, method="truncate")
            # Ensure rank matches
            new_rank = min(A_adapted.shape[0], B_adapted.shape[1])
            A_adapted = A_adapted[:new_rank, :]
            B_adapted = B_adapted[:, :new_rank]
            return A_adapted, B_adapted
        else:
            raise ValueError("LoRA should have 2 components (A, B)")

    else:
        raise ValueError(f"Unknown decomposition_type: {decomposition_type}")


def compute_dimension_mismatch_loss(
    source_shape: Tuple[int, int],
    target_shape: Tuple[int, int]
) -> dict:
    """
    Compute statistics about dimension mismatch.

    Args:
        source_shape: Source matrix shape
        target_shape: Target matrix shape

    Returns:
        Dictionary with mismatch statistics
    """
    source_out, source_in = source_shape
    target_out, target_in = target_shape

    out_ratio = target_out / source_out
    in_ratio = target_in / source_in
    param_ratio = (target_out * target_in) / (source_out * source_in)

    return {
        "source_shape": source_shape,
        "target_shape": target_shape,
        "output_dim_ratio": out_ratio,
        "input_dim_ratio": in_ratio,
        "parameter_ratio": param_ratio,
        "output_truncated": target_out < source_out,
        "input_truncated": target_in < source_in,
        "requires_adaptation": source_shape != target_shape,
    }
