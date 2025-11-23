"""
Base class for all matrix decomposition methods.

This module defines the interface that all decomposition methods should implement.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union, Dict, Any
import torch
import numpy as np


class BaseDecomposer(ABC):
    """
    Abstract base class for matrix decomposition methods.

    All decomposition methods should inherit from this class and implement
    the decompose() and reconstruct() methods.
    """

    def __init__(self, rank: int, **kwargs):
        """
        Initialize the decomposer.

        Args:
            rank: Target rank for the decomposition
            **kwargs: Additional method-specific parameters
        """
        self.rank = rank
        self.kwargs = kwargs

    @abstractmethod
    def decompose(self, weight: Union[torch.Tensor, np.ndarray]) -> Tuple:
        """
        Decompose a weight matrix into low-rank components.

        Args:
            weight: Weight matrix to decompose (can be 2D or higher dimensional)

        Returns:
            Tuple of decomposed components (specific to each method)
        """
        pass

    @abstractmethod
    def reconstruct(self, *components) -> Union[torch.Tensor, np.ndarray]:
        """
        Reconstruct the original matrix from decomposed components.

        Args:
            *components: Decomposed components from decompose()

        Returns:
            Reconstructed weight matrix
        """
        pass

    def compute_compression_ratio(self, original_shape: Tuple, *components) -> float:
        """
        Compute the compression ratio achieved by the decomposition.

        Args:
            original_shape: Shape of the original weight matrix
            *components: Decomposed components

        Returns:
            Compression ratio (original_params / compressed_params)
        """
        original_params = np.prod(original_shape)
        compressed_params = sum(np.prod(c.shape) for c in components)
        return original_params / compressed_params

    def compute_reconstruction_error(
        self,
        original: Union[torch.Tensor, np.ndarray],
        reconstructed: Union[torch.Tensor, np.ndarray],
        metric: str = "frobenius"
    ) -> float:
        """
        Compute the reconstruction error between original and reconstructed matrices.

        Args:
            original: Original weight matrix
            reconstructed: Reconstructed weight matrix
            metric: Error metric ('frobenius', 'mse', 'relative')

        Returns:
            Reconstruction error
        """
        if isinstance(original, torch.Tensor):
            original = original.detach().cpu().numpy()
        if isinstance(reconstructed, torch.Tensor):
            reconstructed = reconstructed.detach().cpu().numpy()

        diff = original - reconstructed

        if metric == "frobenius":
            return np.linalg.norm(diff, ord='fro')
        elif metric == "mse":
            return np.mean(diff ** 2)
        elif metric == "relative":
            return np.linalg.norm(diff, ord='fro') / np.linalg.norm(original, ord='fro')
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the decomposer configuration.

        Returns:
            Dictionary containing decomposer info
        """
        return {
            "method": self.__class__.__name__,
            "rank": self.rank,
            **self.kwargs
        }

    def _to_tensor(self, array: Union[torch.Tensor, np.ndarray], device: str = "cpu") -> torch.Tensor:
        """Helper method to convert numpy array to torch tensor."""
        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).to(device)
        return array.to(device)

    def _to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Helper method to convert torch tensor to numpy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
