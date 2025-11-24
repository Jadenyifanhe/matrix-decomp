"""
Hybrid Decomposition Approaches

This module provides hybrid and ensemble methods that combine multiple
decomposition techniques for better results.

Key approaches:
1. HybridDecomposer: Combines SVD and LoRA for different layers
2. AdaptiveRankDecomposer: Selects rank based on layer properties
3. LayerWiseDecomposer: Different methods for different layer types
4. ImportanceWeightedDecomposer: Weights decomposition by importance scores

These hybrid approaches often provide the best trade-off between
prediction accuracy and inference performance.
"""

from typing import Dict, List, Optional, Union, Tuple, Callable
import torch
import torch.nn as nn
import numpy as np

from .base import BaseDecomposer
from .svd_decomposition import SVDDecomposer
from .lora_adapter import LoRAAdapter
from .fused_svd import FusedSVDDecomposer
from .qps_optimized import (
    DoRAAdapter,
    MergedLoRADecomposer,
    RandomizedSVDDecomposer,
    QRDecomposer,
)


class HybridDecomposer(BaseDecomposer):
    """
    Hybrid decomposition that combines SVD and LoRA.

    For important layers: Uses LoRA (preserves original + adds adapter)
    For less important layers: Uses Fused SVD (more compression)

    This approach balances accuracy and efficiency by using the right
    method for each layer based on its importance.

    Example:
        >>> hybrid = HybridDecomposer(rank=64, lora_threshold=0.7)
        >>> # For highly important layers (score >= 0.7), use LoRA
        >>> # For less important layers, use Fused SVD
    """

    def __init__(
        self,
        rank: int,
        lora_threshold: float = 0.5,
        svd_method: str = "fused",  # "fused", "vanilla", "randomized"
        lora_method: str = "standard",  # "standard", "dora", "merged"
    ):
        """
        Initialize hybrid decomposer.

        Args:
            rank: Base rank for decomposition
            lora_threshold: Importance threshold above which to use LoRA
            svd_method: SVD variant to use for less important layers
            lora_method: LoRA variant to use for important layers
        """
        super().__init__(rank)
        self.lora_threshold = lora_threshold

        # Initialize sub-decomposers
        if svd_method == "fused":
            self.svd_decomposer = FusedSVDDecomposer(rank=rank)
        elif svd_method == "randomized":
            self.svd_decomposer = RandomizedSVDDecomposer(rank=rank)
        else:
            self.svd_decomposer = SVDDecomposer(rank=rank)

        if lora_method == "dora":
            self.lora_decomposer = DoRAAdapter(rank=rank)
        elif lora_method == "merged":
            self.lora_decomposer = MergedLoRADecomposer(rank=rank)
        else:
            self.lora_decomposer = LoRAAdapter(rank=rank)

        self.svd_method = svd_method
        self.lora_method = lora_method

    def decompose(
        self,
        weight: Union[torch.Tensor, np.ndarray],
        importance_score: float = 0.5,
        reference_weight: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Tuple:
        """
        Decompose using hybrid approach based on importance.

        Args:
            weight: Weight matrix to decompose
            importance_score: Layer importance score (0-1)
            reference_weight: Reference weight for LoRA

        Returns:
            Tuple of (method_used, *components)
        """
        if importance_score >= self.lora_threshold:
            # Use LoRA for important layers
            method = "lora"
            if self.lora_method == "dora":
                components = self.lora_decomposer.decompose(weight, reference_weight)
            elif self.lora_method == "merged":
                components = (self.lora_decomposer.decompose(weight, reference_weight),)
            else:
                components = self.lora_decomposer.decompose(weight, reference_weight)
        else:
            # Use SVD for less important layers
            method = "svd"
            components = self.svd_decomposer.decompose(weight)

        return (method,) + (components if isinstance(components, tuple) else (components,))

    def reconstruct(self, method: str, *components, **kwargs):
        """Reconstruct based on method used."""
        if method == "lora":
            return self.lora_decomposer.reconstruct(*components, **kwargs)
        else:
            return self.svd_decomposer.reconstruct(*components, **kwargs)


class AdaptiveRankDecomposer(BaseDecomposer):
    """
    Adaptive rank selection based on layer properties.

    Automatically selects the appropriate rank for each layer based on:
    - Matrix dimensions
    - Singular value distribution
    - Target reconstruction error
    - QPS budget

    This prevents over-compression of critical layers and under-compression
    of less important ones.
    """

    def __init__(
        self,
        min_rank: int = 8,
        max_rank: int = 256,
        target_error: float = 0.1,
        energy_threshold: float = 0.95,
        rank_budget: Optional[int] = None,
        decomposer_type: str = "fused_svd",
    ):
        """
        Initialize adaptive rank decomposer.

        Args:
            min_rank: Minimum rank to use
            max_rank: Maximum rank to use
            target_error: Target relative reconstruction error
            energy_threshold: Fraction of singular value energy to preserve
            rank_budget: Total rank budget across all layers (None = unlimited)
            decomposer_type: Type of decomposer to use
        """
        super().__init__(rank=max_rank)
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.target_error = target_error
        self.energy_threshold = energy_threshold
        self.rank_budget = rank_budget
        self.decomposer_type = decomposer_type

        self._rank_history = {}  # Track ranks used per layer

    def _estimate_optimal_rank(
        self,
        weight: np.ndarray,
        layer_name: Optional[str] = None
    ) -> int:
        """
        Estimate optimal rank for a weight matrix.

        Uses singular value analysis to find the rank that preserves
        the target energy/error threshold.
        """
        # Compute full SVD (just singular values)
        _, S, _ = np.linalg.svd(weight, full_matrices=False)

        # Compute cumulative energy
        total_energy = np.sum(S ** 2)
        cumulative_energy = np.cumsum(S ** 2)
        energy_ratio = cumulative_energy / total_energy

        # Find rank that achieves energy threshold
        rank_for_energy = np.searchsorted(energy_ratio, self.energy_threshold) + 1

        # Compute reconstruction errors at different ranks
        for r in range(1, len(S) + 1):
            # Error is sum of squared singular values we're discarding
            error = np.sqrt(np.sum(S[r:] ** 2)) / np.sqrt(total_energy)
            if error <= self.target_error:
                rank_for_error = r
                break
        else:
            rank_for_error = len(S)

        # Take minimum of both criteria, constrained by min/max
        optimal_rank = max(self.min_rank, min(self.max_rank, min(rank_for_energy, rank_for_error)))

        # Store for analysis
        if layer_name:
            self._rank_history[layer_name] = {
                'rank': optimal_rank,
                'rank_for_energy': rank_for_energy,
                'rank_for_error': rank_for_error,
                'condition_number': S[0] / S[-1] if S[-1] > 1e-10 else float('inf'),
            }

        return optimal_rank

    def decompose(
        self,
        weight: Union[torch.Tensor, np.ndarray],
        layer_name: Optional[str] = None
    ) -> Tuple:
        """
        Decompose with adaptively selected rank.

        Args:
            weight: Weight matrix
            layer_name: Optional layer name for tracking

        Returns:
            (optimal_rank, *decomposed_components)
        """
        is_torch = isinstance(weight, torch.Tensor)
        device = weight.device if is_torch else None

        W = self._to_numpy(weight)

        # Estimate optimal rank
        optimal_rank = self._estimate_optimal_rank(W, layer_name)

        # Create decomposer with optimal rank
        if self.decomposer_type == "fused_svd":
            decomposer = FusedSVDDecomposer(rank=optimal_rank)
        elif self.decomposer_type == "lora":
            decomposer = LoRAAdapter(rank=optimal_rank)
        elif self.decomposer_type == "randomized":
            decomposer = RandomizedSVDDecomposer(rank=optimal_rank)
        else:
            decomposer = SVDDecomposer(rank=optimal_rank)

        # Decompose
        components = decomposer.decompose(weight)

        return (optimal_rank,) + (components if isinstance(components, tuple) else (components,))

    def reconstruct(self, optimal_rank: int, *components, **kwargs):
        """Reconstruct from components."""
        if self.decomposer_type == "fused_svd":
            decomposer = FusedSVDDecomposer(rank=optimal_rank)
        elif self.decomposer_type == "lora":
            decomposer = LoRAAdapter(rank=optimal_rank)
        else:
            decomposer = SVDDecomposer(rank=optimal_rank)

        return decomposer.reconstruct(*components, **kwargs)

    def get_rank_summary(self) -> Dict:
        """Get summary of ranks used across layers."""
        if not self._rank_history:
            return {}

        ranks = [info['rank'] for info in self._rank_history.values()]
        return {
            'num_layers': len(ranks),
            'mean_rank': np.mean(ranks),
            'min_rank': np.min(ranks),
            'max_rank': np.max(ranks),
            'total_rank_budget_used': np.sum(ranks),
            'per_layer': self._rank_history,
        }


class LayerWiseDecomposer(BaseDecomposer):
    """
    Layer-wise decomposition with different methods for different layer types.

    Uses the most appropriate decomposition method based on layer characteristics:
    - Linear layers: LoRA or Fused SVD
    - Attention layers: Higher rank LoRA
    - FFN layers: Lower rank Fused SVD
    - Embedding layers: Randomized SVD

    This targeted approach often yields better results than one-size-fits-all.
    """

    def __init__(
        self,
        default_rank: int = 64,
        layer_configs: Optional[Dict[str, Dict]] = None,
    ):
        """
        Initialize layer-wise decomposer.

        Args:
            default_rank: Default rank for unspecified layers
            layer_configs: Configuration per layer type/name pattern
                Example: {
                    'attention': {'method': 'lora', 'rank': 128},
                    'ffn': {'method': 'fused_svd', 'rank': 32},
                    'embedding': {'method': 'randomized_svd', 'rank': 256},
                }
        """
        super().__init__(default_rank)
        self.default_rank = default_rank
        self.layer_configs = layer_configs or self._get_default_configs()

        # Cache decomposers
        self._decomposer_cache = {}

    def _get_default_configs(self) -> Dict:
        """Get default configurations for common layer patterns."""
        return {
            # Attention layers - high importance, use LoRA
            'attention': {'method': 'lora', 'rank': 128, 'alpha': 32},
            'query': {'method': 'lora', 'rank': 128, 'alpha': 32},
            'key': {'method': 'lora', 'rank': 128, 'alpha': 32},
            'value': {'method': 'lora', 'rank': 128, 'alpha': 32},

            # FFN layers - can compress more
            'ffn': {'method': 'fused_svd', 'rank': 64},
            'dense': {'method': 'fused_svd', 'rank': 64},
            'mlp': {'method': 'fused_svd', 'rank': 64},

            # Embedding - use randomized for speed
            'embedding': {'method': 'randomized_svd', 'rank': 256},
            'embed': {'method': 'randomized_svd', 'rank': 256},

            # Output layers - be careful, use LoRA
            'output': {'method': 'lora', 'rank': 64, 'alpha': 16},
            'classifier': {'method': 'lora', 'rank': 64, 'alpha': 16},
        }

    def _get_config_for_layer(self, layer_name: str) -> Dict:
        """Get configuration for a specific layer based on name patterns."""
        layer_name_lower = layer_name.lower()

        for pattern, config in self.layer_configs.items():
            if pattern.lower() in layer_name_lower:
                return config

        # Default configuration
        return {'method': 'fused_svd', 'rank': self.default_rank}

    def _get_decomposer(self, config: Dict) -> BaseDecomposer:
        """Get or create decomposer for a configuration."""
        cache_key = (config['method'], config.get('rank', self.default_rank))

        if cache_key not in self._decomposer_cache:
            method = config['method']
            rank = config.get('rank', self.default_rank)

            if method == 'lora':
                alpha = config.get('alpha', 1.0)
                self._decomposer_cache[cache_key] = LoRAAdapter(rank=rank, alpha=alpha)
            elif method == 'dora':
                self._decomposer_cache[cache_key] = DoRAAdapter(rank=rank)
            elif method == 'fused_svd':
                self._decomposer_cache[cache_key] = FusedSVDDecomposer(rank=rank)
            elif method == 'randomized_svd':
                self._decomposer_cache[cache_key] = RandomizedSVDDecomposer(rank=rank)
            elif method == 'qr':
                self._decomposer_cache[cache_key] = QRDecomposer(rank=rank)
            else:
                self._decomposer_cache[cache_key] = SVDDecomposer(rank=rank)

        return self._decomposer_cache[cache_key]

    def decompose(
        self,
        weight: Union[torch.Tensor, np.ndarray],
        layer_name: str = "",
        reference_weight: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Tuple:
        """
        Decompose using layer-appropriate method.

        Args:
            weight: Weight matrix
            layer_name: Name of the layer (for method selection)
            reference_weight: Reference weight for LoRA methods

        Returns:
            (method_used, config, *components)
        """
        config = self._get_config_for_layer(layer_name)
        decomposer = self._get_decomposer(config)

        # Call decompose with appropriate arguments
        if config['method'] in ['lora', 'dora', 'merged_lora']:
            components = decomposer.decompose(weight, reference_weight=reference_weight)
        else:
            components = decomposer.decompose(weight)

        return (config['method'], config) + (components if isinstance(components, tuple) else (components,))

    def reconstruct(self, method: str, config: Dict, *components, **kwargs):
        """Reconstruct using the appropriate method."""
        decomposer = self._get_decomposer(config)
        return decomposer.reconstruct(*components, **kwargs)


class ImportanceWeightedDecomposer(BaseDecomposer):
    """
    Importance-weighted decomposition.

    Adjusts the decomposition rank based on layer importance scores.
    More important layers get higher ranks (less compression).
    Less important layers get lower ranks (more compression).

    This optimizes the rank budget allocation across layers.
    """

    def __init__(
        self,
        base_rank: int = 64,
        rank_multiplier_range: Tuple[float, float] = (0.25, 2.0),
        total_rank_budget: Optional[int] = None,
        decomposer_type: str = "fused_svd",
    ):
        """
        Initialize importance-weighted decomposer.

        Args:
            base_rank: Base rank before importance scaling
            rank_multiplier_range: (min_mult, max_mult) for rank scaling
            total_rank_budget: Optional total rank budget constraint
            decomposer_type: Type of decomposer to use
        """
        super().__init__(base_rank)
        self.base_rank = base_rank
        self.min_multiplier, self.max_multiplier = rank_multiplier_range
        self.total_rank_budget = total_rank_budget
        self.decomposer_type = decomposer_type

        self._allocated_ranks = {}

    def compute_ranks(
        self,
        importance_scores: Dict[str, float],
        layer_shapes: Optional[Dict[str, Tuple]] = None
    ) -> Dict[str, int]:
        """
        Compute rank allocation based on importance scores.

        Args:
            importance_scores: Layer name -> importance score
            layer_shapes: Optional layer shapes for constraints

        Returns:
            Layer name -> allocated rank
        """
        if not importance_scores:
            return {}

        # Normalize importance scores to [0, 1]
        scores = list(importance_scores.values())
        min_score, max_score = min(scores), max(scores)
        score_range = max_score - min_score if max_score > min_score else 1.0

        # Compute initial ranks
        initial_ranks = {}
        for name, score in importance_scores.items():
            # Map score to multiplier range
            normalized = (score - min_score) / score_range
            multiplier = self.min_multiplier + normalized * (self.max_multiplier - self.min_multiplier)
            rank = int(self.base_rank * multiplier)

            # Constrain by layer shape if provided
            if layer_shapes and name in layer_shapes:
                max_possible = min(layer_shapes[name])
                rank = min(rank, max_possible)

            initial_ranks[name] = max(4, rank)  # Minimum rank of 4

        # Apply budget constraint if specified
        if self.total_rank_budget is not None:
            total = sum(initial_ranks.values())
            if total > self.total_rank_budget:
                # Scale down proportionally
                scale = self.total_rank_budget / total
                initial_ranks = {
                    name: max(4, int(rank * scale))
                    for name, rank in initial_ranks.items()
                }

        self._allocated_ranks = initial_ranks
        return initial_ranks

    def decompose(
        self,
        weight: Union[torch.Tensor, np.ndarray],
        layer_name: str,
        importance_score: float,
        reference_weight: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Tuple:
        """
        Decompose with importance-scaled rank.

        Args:
            weight: Weight matrix
            layer_name: Layer name
            importance_score: Importance score for this layer
            reference_weight: Reference weight for LoRA

        Returns:
            (allocated_rank, *components)
        """
        # Compute rank for this layer
        normalized = min(1.0, max(0.0, importance_score))
        multiplier = self.min_multiplier + normalized * (self.max_multiplier - self.min_multiplier)
        rank = max(4, int(self.base_rank * multiplier))

        # Constrain by matrix size
        is_torch = isinstance(weight, torch.Tensor)
        shape = weight.shape
        rank = min(rank, min(shape[0], shape[1]))

        self._allocated_ranks[layer_name] = rank

        # Create decomposer with computed rank
        if self.decomposer_type == "lora":
            decomposer = LoRAAdapter(rank=rank)
            components = decomposer.decompose(weight, reference_weight=reference_weight)
        elif self.decomposer_type == "dora":
            decomposer = DoRAAdapter(rank=rank)
            components = decomposer.decompose(weight, reference_weight=reference_weight)
        elif self.decomposer_type == "fused_svd":
            decomposer = FusedSVDDecomposer(rank=rank)
            components = decomposer.decompose(weight)
        else:
            decomposer = SVDDecomposer(rank=rank)
            components = decomposer.decompose(weight)

        return (rank,) + (components if isinstance(components, tuple) else (components,))

    def reconstruct(self, rank: int, *components, **kwargs):
        """Reconstruct with specified rank."""
        if self.decomposer_type == "lora":
            decomposer = LoRAAdapter(rank=rank)
        elif self.decomposer_type == "fused_svd":
            decomposer = FusedSVDDecomposer(rank=rank)
        else:
            decomposer = SVDDecomposer(rank=rank)

        return decomposer.reconstruct(*components, **kwargs)

    def get_allocation_summary(self) -> Dict:
        """Get summary of rank allocations."""
        if not self._allocated_ranks:
            return {}

        ranks = list(self._allocated_ranks.values())
        return {
            'num_layers': len(ranks),
            'total_rank_budget_used': sum(ranks),
            'mean_rank': np.mean(ranks),
            'min_rank': min(ranks),
            'max_rank': max(ranks),
            'per_layer': self._allocated_ranks.copy(),
        }


class EnsembleDecomposer(BaseDecomposer):
    """
    Ensemble of multiple decomposition methods.

    Combines multiple decomposition approaches and selects the best
    one based on reconstruction error or other metrics.

    Useful for:
    - Finding the best method for each layer
    - Research and experimentation
    - When you're unsure which method works best
    """

    def __init__(
        self,
        rank: int,
        methods: Optional[List[str]] = None,
        selection_criterion: str = "error",  # "error", "compression", "qps_estimate"
    ):
        """
        Initialize ensemble decomposer.

        Args:
            rank: Rank for all methods
            methods: List of methods to try
            selection_criterion: How to select the best method
        """
        super().__init__(rank)
        self.methods = methods or ["lora", "fused_svd", "svd", "qr"]
        self.selection_criterion = selection_criterion

        # Initialize all decomposers
        self.decomposers = {}
        for method in self.methods:
            if method == "lora":
                self.decomposers[method] = LoRAAdapter(rank=rank)
            elif method == "fused_svd":
                self.decomposers[method] = FusedSVDDecomposer(rank=rank)
            elif method == "svd":
                self.decomposers[method] = SVDDecomposer(rank=rank)
            elif method == "qr":
                self.decomposers[method] = QRDecomposer(rank=rank)
            elif method == "dora":
                self.decomposers[method] = DoRAAdapter(rank=rank)
            elif method == "randomized":
                self.decomposers[method] = RandomizedSVDDecomposer(rank=rank)

    def decompose(
        self,
        weight: Union[torch.Tensor, np.ndarray],
        reference_weight: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Tuple:
        """
        Try all methods and return the best one.

        Returns:
            (best_method, components, all_results)
        """
        W = self._to_numpy(weight)
        original_norm = np.linalg.norm(W, 'fro')

        results = {}

        for method, decomposer in self.decomposers.items():
            try:
                # Decompose
                if method in ["lora", "dora"]:
                    components = decomposer.decompose(weight, reference_weight=reference_weight)
                else:
                    components = decomposer.decompose(weight)

                # Reconstruct
                if method in ["lora", "dora"]:
                    reconstructed = decomposer.reconstruct(*components)
                    if reference_weight is not None:
                        W_ref = self._to_numpy(reference_weight)
                        reconstructed = self._to_numpy(reconstructed)
                        reconstructed = W_ref + reconstructed
                else:
                    reconstructed = decomposer.reconstruct(*components)

                # Compute error
                reconstructed_np = self._to_numpy(reconstructed)
                error = np.linalg.norm(W - reconstructed_np, 'fro') / original_norm

                # Estimate compression
                if isinstance(components, tuple):
                    compressed_params = sum(
                        np.prod(self._to_numpy(c).shape)
                        for c in components if hasattr(c, 'shape')
                    )
                else:
                    compressed_params = np.prod(self._to_numpy(components).shape)

                compression = W.size / compressed_params if compressed_params > 0 else 1.0

                results[method] = {
                    'components': components,
                    'error': error,
                    'compression': compression,
                }
            except Exception as e:
                results[method] = {'error': float('inf'), 'exception': str(e)}

        # Select best method
        if self.selection_criterion == "error":
            best_method = min(results.keys(), key=lambda m: results[m].get('error', float('inf')))
        elif self.selection_criterion == "compression":
            best_method = max(results.keys(), key=lambda m: results[m].get('compression', 0))
        else:
            best_method = self.methods[0]

        best_result = results[best_method]
        return (best_method, best_result.get('components'), results)

    def reconstruct(self, method: str, components, *args, **kwargs):
        """Reconstruct using the specified method."""
        if method in self.decomposers:
            return self.decomposers[method].reconstruct(*components, **kwargs)
        raise ValueError(f"Unknown method: {method}")
