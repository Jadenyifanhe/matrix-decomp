"""
LoRA (Low-Rank Adaptation) style adapters for knowledge transfer.

LoRA represents weight updates as: W' = W + BA
where B and A are low-rank matrices.

PERFORMANCE ADVANTAGES over vanilla SVD:
1. Preserves original matrix W - no need to replace existing weights
2. Can be computed in parallel: W@x + B@A@x (parallel execution)
3. Better cache locality - original matrix stays intact
4. Can selectively enable/disable adapters without changing model structure
5. Can merge adapters into weights for deployment: W_merged = W + BA

This typically provides better QPS than vanilla SVD decomposition.
"""

from typing import Tuple, Union, Optional, Dict
import torch
import torch.nn as nn
import numpy as np
from .base import BaseDecomposer


class LoRAAdapter(BaseDecomposer):
    """
    LoRA-style adapter for low-rank weight updates.

    Instead of replacing W with U@S@V, we keep W and add a low-rank residual:
    W_new = W + alpha * (B @ A)

    where A is [rank, input_dim] and B is [output_dim, rank]
    """

    def __init__(
        self,
        rank: int,
        alpha: float = 1.0,
        scaling: str = "alpha",  # 'alpha', 'rank', 'none'
        init_method: str = "svd",  # 'svd', 'random', 'zeros'
    ):
        """
        Initialize LoRA adapter.

        Args:
            rank: Rank of the adapter
            alpha: Scaling factor for the adapter output
            scaling: How to scale the adapter:
                - 'alpha': Scale by alpha constant
                - 'rank': Scale by alpha/rank (LoRA paper recommendation)
                - 'none': No scaling
            init_method: How to initialize adapter matrices:
                - 'svd': Initialize from SVD of residual (knowledge transfer)
                - 'random': Random initialization (for training from scratch)
                - 'zeros': Zero initialization for A or B
        """
        super().__init__(rank)
        self.alpha = alpha
        self.scaling = scaling
        self.init_method = init_method

    def get_scaling_factor(self) -> float:
        """Get the scaling factor based on scaling method."""
        if self.scaling == "alpha":
            return self.alpha
        elif self.scaling == "rank":
            return self.alpha / self.rank
        elif self.scaling == "none":
            return 1.0
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling}")

    def decompose(
        self,
        weight: Union[torch.Tensor, np.ndarray],
        reference_weight: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        """
        Create LoRA adapter matrices from a weight matrix.

        If reference_weight is provided, the adapter represents the residual:
        adapter = weight - reference_weight

        Args:
            weight: Weight matrix to decompose (typically from large model)
            reference_weight: Reference weight (typically from small model).
                            If None, decompose the weight directly.

        Returns:
            (A, B) tuple where A is [rank, input_dim], B is [output_dim, rank]
            The adapter output is: B @ A
        """
        is_torch = isinstance(weight, torch.Tensor)
        device = weight.device if is_torch else None

        # Compute residual if reference provided
        if reference_weight is not None:
            W_ref = self._to_numpy(reference_weight)
            W = self._to_numpy(weight)
            residual = W - W_ref
        else:
            residual = self._to_numpy(weight)

        output_dim, input_dim = residual.shape

        if self.init_method == "svd":
            # Decompose residual using SVD
            U, S, Vt = np.linalg.svd(residual, full_matrices=False)

            # Truncate to rank
            actual_rank = min(self.rank, len(S))
            U_r = U[:, :actual_rank]  # [output_dim, rank]
            S_r = S[:actual_rank]      # [rank]
            Vt_r = Vt[:actual_rank, :] # [rank, input_dim]

            # Split singular values between B and A
            # B @ A = (U @ sqrt(S)) @ (sqrt(S) @ Vt)
            sqrt_S = np.sqrt(S_r)
            B = U_r * sqrt_S[np.newaxis, :]  # [output_dim, rank]
            A = sqrt_S[:, np.newaxis] * Vt_r  # [rank, input_dim]

        elif self.init_method == "random":
            # Random initialization (Gaussian)
            B = np.random.randn(output_dim, self.rank) * 0.01
            A = np.random.randn(self.rank, input_dim) * 0.01

        elif self.init_method == "zeros":
            # Zero initialization for A, random for B (LoRA paper style)
            B = np.random.randn(output_dim, self.rank) * 0.01
            A = np.zeros((self.rank, input_dim))

        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")

        # Convert back to torch if needed
        if is_torch:
            A = self._to_tensor(A, device)
            B = self._to_tensor(B, device)

        return A, B

    def reconstruct(
        self,
        A: Union[torch.Tensor, np.ndarray],
        B: Union[torch.Tensor, np.ndarray],
        as_tensor: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Reconstruct the weight update from adapter matrices.

        Args:
            A: Matrix A [rank, input_dim]
            B: Matrix B [output_dim, rank]
            as_tensor: Return as torch tensor

        Returns:
            Weight update: B @ A [output_dim, input_dim]
        """
        is_torch = isinstance(A, torch.Tensor)

        if is_torch:
            delta_W = B @ A
        else:
            delta_W = B @ A

        if as_tensor and not isinstance(delta_W, torch.Tensor):
            delta_W = self._to_tensor(delta_W)
        elif not as_tensor and isinstance(delta_W, torch.Tensor):
            delta_W = self._to_numpy(delta_W)

        return delta_W

    def apply_adapter(
        self,
        original_weight: Union[torch.Tensor, np.ndarray],
        A: Union[torch.Tensor, np.ndarray],
        B: Union[torch.Tensor, np.ndarray],
        merge: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Apply adapter to original weight.

        Args:
            original_weight: Original weight matrix W
            A: Adapter matrix A
            B: Adapter matrix B
            merge: If True, merge adapter into weight (W + scale * BA)
                  If False, return tuple (W, A, B) for runtime application

        Returns:
            If merge=True: Merged weight W + scale * BA
            If merge=False: Tuple (W, A, B)
        """
        scale = self.get_scaling_factor()

        if merge:
            delta_W = self.reconstruct(A, B, as_tensor=isinstance(original_weight, torch.Tensor))
            return original_weight + scale * delta_W
        else:
            return (original_weight, A, B)

    def create_lora_layer(
        self,
        in_features: int,
        out_features: int,
        base_weight: Optional[torch.Tensor] = None,
        bias: bool = False,
        device: str = "cpu"
    ) -> 'LoRALayer':
        """
        Create a LoRA layer module that can be used as a drop-in replacement.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            base_weight: Optional base weight matrix to initialize with
            bias: Whether to include bias
            device: Device to create layer on

        Returns:
            LoRALayer module
        """
        return LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=self.rank,
            alpha=self.alpha,
            scaling=self.scaling,
            base_weight=base_weight,
            bias=bias,
            device=device
        )


class LoRALayer(nn.Module):
    """
    PyTorch module implementing LoRA adapter layer.

    This can be used as a drop-in replacement for nn.Linear with LoRA adapter.
    Forward pass computes: output = W@x + scale * B@A@x
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 1.0,
        scaling: str = "alpha",
        base_weight: Optional[torch.Tensor] = None,
        bias: bool = False,
        device: str = "cpu"
    ):
        """
        Initialize LoRA layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: Rank of LoRA adapter
            alpha: Scaling factor
            scaling: Scaling method ('alpha', 'rank', 'none')
            base_weight: Optional base weight [out_features, in_features]
            bias: Whether to use bias
            device: Device to create tensors on
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = scaling

        # Base linear layer
        if base_weight is not None:
            self.weight = nn.Parameter(base_weight.to(device))
        else:
            self.weight = nn.Parameter(torch.randn(out_features, in_features, device=device) * 0.01)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device))
        else:
            self.register_parameter('bias', None)

        # LoRA adapter matrices
        # A: [rank, in_features], B: [out_features, rank]
        self.lora_A = nn.Parameter(torch.randn(rank, in_features, device=device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device))

        # Adapter enabled flag
        self.adapter_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adapter.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        # Base linear transformation
        output = torch.nn.functional.linear(x, self.weight, self.bias)

        # Add LoRA adapter if enabled
        if self.adapter_enabled:
            scale = self.get_scaling_factor()
            # Compute B @ A @ x efficiently
            adapter_out = torch.nn.functional.linear(
                torch.nn.functional.linear(x, self.lora_A),
                self.lora_B
            )
            output = output + scale * adapter_out

        return output

    def get_scaling_factor(self) -> float:
        """Get the scaling factor."""
        if self.scaling == "alpha":
            return self.alpha
        elif self.scaling == "rank":
            return self.alpha / self.rank
        else:
            return 1.0

    def merge_adapter(self):
        """Merge adapter into base weight for deployment."""
        if self.adapter_enabled:
            with torch.no_grad():
                scale = self.get_scaling_factor()
                delta_W = scale * (self.lora_B @ self.lora_A)
                self.weight.add_(delta_W)
            # Zero out adapter
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()
            self.adapter_enabled = False

    def enable_adapter(self):
        """Enable adapter."""
        self.adapter_enabled = True

    def disable_adapter(self):
        """Disable adapter."""
        self.adapter_enabled = False

    def extra_repr(self) -> str:
        """String representation."""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'rank={self.rank}, alpha={self.alpha}, '
                f'adapter_enabled={self.adapter_enabled}')
