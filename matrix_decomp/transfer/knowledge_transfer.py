"""
Main knowledge transfer module for moving decomposed weights from large to small models.

This module orchestrates the complete process of:
1. Decomposing weights from a large model
2. Adapting dimensions to match small model
3. Injecting decomposed weights into small model
"""

from typing import Dict, Union, Tuple, List, Optional
import torch
import torch.nn as nn

from ..decomposition import (
    SVDDecomposer,
    LoRAAdapter,
    FusedSVDDecomposer,
    TuckerDecomposer,
    CPDecomposer,
)
from .dimension_adapter import smart_dimension_adaptation, compute_dimension_mismatch_loss


class KnowledgeTransfer:
    """
    Main class for transferring knowledge from large model to small model.
    """

    def __init__(
        self,
        method: str = "lora",
        rank: int = 64,
        **method_kwargs
    ):
        """
        Initialize knowledge transfer.

        Args:
            method: Decomposition method to use
                - 'svd': Vanilla SVD
                - 'fused_svd': Fused SVD (better QPS)
                - 'lora': LoRA adapters (recommended)
                - 'tucker': Tucker decomposition
                - 'cp': CP decomposition
            rank: Rank for decomposition
            **method_kwargs: Additional arguments for decomposition method
        """
        self.method = method
        self.rank = rank
        self.method_kwargs = method_kwargs

        # Initialize decomposer
        if method == "svd":
            self.decomposer = SVDDecomposer(rank=rank, **method_kwargs)
        elif method == "fused_svd":
            self.decomposer = FusedSVDDecomposer(rank=rank, **method_kwargs)
        elif method == "lora":
            self.decomposer = LoRAAdapter(rank=rank, **method_kwargs)
        elif method == "tucker":
            self.decomposer = TuckerDecomposer(rank=rank, **method_kwargs)
        elif method == "cp":
            self.decomposer = CPDecomposer(rank=rank, **method_kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def transfer_layer(
        self,
        source_weight: torch.Tensor,
        target_module: nn.Module,
        layer_name: str,
        adapt_dims: bool = True
    ) -> Dict:
        """
        Transfer a single layer from source to target.

        Args:
            source_weight: Weight from large model
            target_module: Target module in small model
            layer_name: Name of the layer
            adapt_dims: Whether to adapt dimensions

        Returns:
            Dictionary with transfer statistics
        """
        # Get target weight
        if hasattr(target_module, 'weight'):
            target_weight = target_module.weight.data
        else:
            raise ValueError(f"Target module has no weight parameter")

        target_shape = target_weight.shape

        # Check dimension mismatch
        mismatch_stats = compute_dimension_mismatch_loss(
            source_weight.shape,
            target_shape
        )

        # Decompose source weight
        if self.method == "lora":
            # For LoRA, we can use target weight as reference
            components = self.decomposer.decompose(source_weight, reference_weight=target_weight)
        else:
            components = self.decomposer.decompose(source_weight)

        # Adapt dimensions if needed
        if adapt_dims and mismatch_stats['requires_adaptation']:
            components = smart_dimension_adaptation(
                components,
                target_shape,
                decomposition_type=self.method
            )

        # Apply to target based on method
        if self.method in ["svd", "fused_svd"]:
            # Replace target weight with reconstructed low-rank version
            reconstructed = self.decomposer.reconstruct(*components, as_tensor=True)
            target_module.weight.data = reconstructed

        elif self.method == "lora":
            # Attach LoRA adapter
            A, B = components
            # Convert module to LoRA layer or attach adapter
            # For simplicity, we'll add the adapter weights as a residual
            delta_W = self.decomposer.reconstruct(A, B, as_tensor=True)
            scale = self.decomposer.get_scaling_factor()
            target_module.weight.data = target_weight + scale * delta_W

        elif self.method in ["tucker", "cp"]:
            # Reconstruct and replace
            reconstructed = self.decomposer.reconstruct(*components, as_tensor=True)
            target_module.weight.data = reconstructed

        # Compute reconstruction error (on adapted components)
        reconstructed_check = self.decomposer.reconstruct(*components, as_tensor=True)
        if adapt_dims:
            # Compare with truncated source
            source_truncated = source_weight[:target_shape[0], :target_shape[1]]
        else:
            source_truncated = source_weight

        error = torch.norm(source_truncated - reconstructed_check, p='fro').item()
        relative_error = error / torch.norm(source_truncated, p='fro').item()

        return {
            'layer_name': layer_name,
            'method': self.method,
            'rank': self.rank,
            'source_shape': source_weight.shape,
            'target_shape': target_shape,
            'reconstruction_error': error,
            'relative_error': relative_error,
            **mismatch_stats
        }

    def transfer_model(
        self,
        source_model: nn.Module,
        target_model: nn.Module,
        layer_mapping: Optional[Dict[str, str]] = None,
        layer_names: Optional[List[str]] = None,
        adapt_dims: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Transfer multiple layers from source to target model.

        Args:
            source_model: Large source model
            target_model: Small target model
            layer_mapping: Mapping from source layer names to target layer names
                          If None, assumes same names
            layer_names: List of layer names to transfer (None for all compatible)
            adapt_dims: Whether to adapt dimensions
            verbose: Print progress

        Returns:
            Dictionary with transfer statistics for all layers
        """
        if layer_mapping is None:
            layer_mapping = {}

        # Get all transferable layers
        source_layers = {name: module for name, module in source_model.named_modules()
                        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d))}
        target_layers = {name: module for name, module in target_model.named_modules()
                        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d))}

        # Determine which layers to transfer
        if layer_names is None:
            # Transfer all common layers
            layer_names = list(set(source_layers.keys()) & set(target_layers.keys()))

        transfer_stats = {}

        for source_name in layer_names:
            # Get target name
            target_name = layer_mapping.get(source_name, source_name)

            if source_name not in source_layers:
                if verbose:
                    print(f"Warning: Source layer '{source_name}' not found")
                continue

            if target_name not in target_layers:
                if verbose:
                    print(f"Warning: Target layer '{target_name}' not found")
                continue

            source_module = source_layers[source_name]
            target_module = target_layers[target_name]

            if verbose:
                print(f"Transferring {source_name} -> {target_name}...")

            # Transfer layer
            stats = self.transfer_layer(
                source_module.weight.data,
                target_module,
                source_name,
                adapt_dims=adapt_dims
            )

            transfer_stats[source_name] = stats

        # Compute aggregate statistics
        aggregate_stats = self._compute_aggregate_stats(transfer_stats)

        return {
            'per_layer': transfer_stats,
            'aggregate': aggregate_stats
        }

    def _compute_aggregate_stats(self, transfer_stats: Dict) -> Dict:
        """Compute aggregate statistics across all transferred layers."""
        if not transfer_stats:
            return {}

        errors = [stats['reconstruction_error'] for stats in transfer_stats.values()]
        relative_errors = [stats['relative_error'] for stats in transfer_stats.values()]

        return {
            'num_layers': len(transfer_stats),
            'mean_error': sum(errors) / len(errors),
            'max_error': max(errors),
            'mean_relative_error': sum(relative_errors) / len(relative_errors),
            'max_relative_error': max(relative_errors),
        }

    def inject_lora_adapters(
        self,
        source_model: nn.Module,
        target_model: nn.Module,
        layer_names: List[str],
        merge: bool = False
    ) -> nn.Module:
        """
        Inject LoRA adapters into target model.

        Args:
            source_model: Source model with better weights
            target_model: Target model to enhance
            layer_names: Layers to add adapters to
            merge: If True, merge adapters into weights; if False, keep separate

        Returns:
            Enhanced target model
        """
        if self.method != "lora":
            raise ValueError("This method only works with LoRA decomposition")

        source_layers = dict(source_model.named_modules())
        target_layers = dict(target_model.named_modules())

        for layer_name in layer_names:
            if layer_name not in source_layers or layer_name not in target_layers:
                continue

            source_module = source_layers[layer_name]
            target_module = target_layers[layer_name]

            if not isinstance(target_module, (nn.Linear, nn.Conv2d)):
                continue

            # Decompose difference
            source_weight = source_module.weight.data
            target_weight = target_module.weight.data

            # Adapt source to target dimensions
            min_out = min(source_weight.shape[0], target_weight.shape[0])
            min_in = min(source_weight.shape[1], target_weight.shape[1])

            source_truncated = source_weight[:min_out, :min_in]
            target_truncated = target_weight[:min_out, :min_in]

            # Get LoRA components
            A, B = self.decomposer.decompose(source_truncated, reference_weight=target_truncated)

            if merge:
                # Merge into weight
                delta_W = self.decomposer.reconstruct(A, B, as_tensor=True)
                scale = self.decomposer.get_scaling_factor()
                target_weight[:min_out, :min_in] += scale * delta_W
            else:
                # Store as separate parameters (would need custom layer implementation)
                # For now, just merge
                delta_W = self.decomposer.reconstruct(A, B, as_tensor=True)
                scale = self.decomposer.get_scaling_factor()
                target_weight[:min_out, :min_in] += scale * delta_W

        return target_model
