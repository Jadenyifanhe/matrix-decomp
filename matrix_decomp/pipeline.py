"""
End-to-end Model Distillation Pipeline

This module provides a high-level interface for the complete knowledge transfer workflow:
1. Train/load a large scale-up model
2. Identify important layers using various importance scoring methods
3. Decompose important layers using matrix decomposition
4. Transfer decomposed knowledge to a smaller model
5. Fine-tune if needed
6. Benchmark the resulting model

This is the recommended entry point for most users.
"""

from typing import Dict, List, Optional, Union, Callable, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .decomposition import (
    SVDDecomposer,
    LoRAAdapter,
    FusedSVDDecomposer,
    TuckerDecomposer,
    CPDecomposer,
)
from .importance import compute_importance_scores, select_top_k_layers
from .transfer import KnowledgeTransfer
from .benchmarks import measure_qps, count_flops, compute_compression_ratio


class ModelDistillationPipeline:
    """
    End-to-end pipeline for knowledge transfer from large to small models.

    This class orchestrates the complete workflow of:
    1. Computing importance scores to identify critical layers
    2. Decomposing weights from the large model
    3. Transferring knowledge to the small model
    4. Optionally fine-tuning the result
    5. Benchmarking performance (QPS, FLOPs, accuracy)

    Example:
        >>> pipeline = ModelDistillationPipeline(
        ...     large_model=large_model,
        ...     small_model=small_model,
        ...     decomposition_method='lora',
        ...     rank=64,
        ...     importance_method='gradient'
        ... )
        >>> enhanced_model = pipeline.run(
        ...     validation_data=val_loader,
        ...     top_k_layers=10,
        ...     fine_tune_epochs=5,
        ...     measure_qps=True
        ... )
        >>> pipeline.print_report()
    """

    def __init__(
        self,
        large_model: nn.Module,
        small_model: nn.Module,
        decomposition_method: str = "lora",
        rank: int = 64,
        importance_method: str = "gradient",
        device: str = "cpu",
        **method_kwargs
    ):
        """
        Initialize the distillation pipeline.

        Args:
            large_model: The larger, well-trained source model (e.g., 5x scale-up)
            small_model: The smaller target model for deployment
            decomposition_method: Method for decomposing weights
                - 'svd': Vanilla SVD (WARNING: may hurt QPS)
                - 'fused_svd': Fused SVD (better QPS than vanilla)
                - 'lora': LoRA adapters (RECOMMENDED for best QPS)
                - 'tucker': Tucker decomposition (for conv layers)
                - 'cp': CP decomposition (maximum compression)
            rank: Target rank for decomposition
            importance_method: Method for scoring layer importance
                - 'gradient': Gradient magnitude (fast, general)
                - 'taylor': Taylor expansion (recommended for production)
                - 'synflow': Data-free method
                - 'knockout': Most accurate but slowest
                - See compute_importance_scores for all 30+ options
            device: Device to run computations on
            **method_kwargs: Additional arguments for decomposition method
        """
        self.large_model = large_model.to(device)
        self.small_model = small_model.to(device)
        self.decomposition_method = decomposition_method
        self.rank = rank
        self.importance_method = importance_method
        self.device = device
        self.method_kwargs = method_kwargs

        # Initialize knowledge transfer
        self.transfer = KnowledgeTransfer(
            method=decomposition_method,
            rank=rank,
            **method_kwargs
        )

        # Results storage
        self.importance_scores = None
        self.selected_layers = None
        self.transfer_stats = None
        self.benchmark_results = {}

    def compute_importance(
        self,
        data_loader: DataLoader,
        criterion: Optional[Callable] = None,
        input_shape: Optional[Tuple] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute importance scores for all layers.

        Args:
            data_loader: DataLoader for computing importance (not needed for data-free methods)
            criterion: Loss function (needed for gradient-based methods)
            input_shape: Input shape (needed for data-free methods like synflow)
            **kwargs: Additional arguments for importance scorer

        Returns:
            Dictionary mapping layer names to importance scores
        """
        print(f"Computing importance scores using '{self.importance_method}' method...")

        self.importance_scores = compute_importance_scores(
            self.large_model,
            data_loader=data_loader,
            method=self.importance_method,
            criterion=criterion,
            device=self.device,
            input_shape=input_shape,
            **kwargs
        )

        return self.importance_scores

    def select_layers(
        self,
        top_k: int = 10,
        exclude_layers: Optional[List[str]] = None,
        min_importance: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Select the most important layers for decomposition.

        Args:
            top_k: Number of top layers to select
            exclude_layers: Layer names to exclude (e.g., first/last layers)
            min_importance: Minimum importance score threshold

        Returns:
            List of (layer_name, score) tuples for selected layers
        """
        if self.importance_scores is None:
            raise ValueError("Must call compute_importance() first")

        # Filter by minimum importance
        filtered_scores = {
            name: score for name, score in self.importance_scores.items()
            if score >= min_importance
        }

        self.selected_layers = select_top_k_layers(
            filtered_scores,
            k=top_k,
            exclude_layers=exclude_layers
        )

        print(f"\nSelected top {len(self.selected_layers)} layers for decomposition:")
        for i, (name, score) in enumerate(self.selected_layers, 1):
            print(f"  {i}. {name}: {score:.4f}")

        return self.selected_layers

    def transfer_knowledge(
        self,
        layer_names: Optional[List[str]] = None,
        adapt_dims: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Transfer knowledge from large model to small model.

        Args:
            layer_names: Specific layers to transfer (uses selected_layers if None)
            adapt_dims: Whether to adapt dimensions for mismatched sizes
            verbose: Print progress

        Returns:
            Dictionary with transfer statistics
        """
        if layer_names is None:
            if self.selected_layers is None:
                raise ValueError("Must call select_layers() first or provide layer_names")
            layer_names = [name for name, _ in self.selected_layers]

        print(f"\nTransferring knowledge using {self.decomposition_method.upper()}...")

        self.transfer_stats = self.transfer.transfer_model(
            self.large_model,
            self.small_model,
            layer_names=layer_names,
            adapt_dims=adapt_dims,
            verbose=verbose
        )

        return self.transfer_stats

    def fine_tune(
        self,
        train_loader: DataLoader,
        criterion: Callable,
        epochs: int = 5,
        learning_rate: float = 1e-4,
        optimizer_class: type = None,
        verbose: bool = True
    ):
        """
        Fine-tune the small model after knowledge transfer.

        Args:
            train_loader: Training data
            criterion: Loss function
            epochs: Number of epochs
            learning_rate: Learning rate
            optimizer_class: Optimizer class (default: AdamW)
            verbose: Print progress
        """
        if optimizer_class is None:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            self.small_model.parameters(),
            lr=learning_rate
        )

        self.small_model.train()

        print(f"\nFine-tuning for {epochs} epochs...")

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for batch in tqdm(train_loader, disable=not verbose, desc=f"Epoch {epoch+1}"):
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = None

                optimizer.zero_grad()
                outputs = self.small_model(inputs)

                if targets is not None:
                    loss = criterion(outputs, targets)
                else:
                    loss = criterion(outputs)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            if verbose:
                avg_loss = total_loss / num_batches
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        self.small_model.eval()

    def benchmark(
        self,
        input_shape: Tuple,
        batch_size: int = 1,
        num_iterations: int = 100,
        benchmark_qps: bool = True,
        benchmark_flops: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Benchmark the models.

        Args:
            input_shape: Input shape for benchmarking
            batch_size: Batch size
            num_iterations: Number of iterations for QPS measurement
            benchmark_qps: Whether to measure QPS
            benchmark_flops: Whether to count FLOPs
            verbose: Print results

        Returns:
            Dictionary with benchmark results
        """
        print("\n" + "="*60)
        print("Benchmarking")
        print("="*60)

        models = {
            "Large Model": self.large_model,
            "Small Model (Enhanced)": self.small_model,
        }

        if benchmark_qps:
            print("\nMeasuring QPS...")
            for name, model in models.items():
                qps_result = measure_qps(
                    model,
                    input_shape,
                    batch_size=batch_size,
                    num_iterations=num_iterations,
                    device=self.device,
                    verbose=verbose
                )
                self.benchmark_results[f"{name}_qps"] = qps_result

        if benchmark_flops:
            print("\nCounting FLOPs...")
            for name, model in models.items():
                flops_result = count_flops(model, input_shape, verbose=verbose)
                self.benchmark_results[f"{name}_flops"] = flops_result

        # Compute compression ratio
        compression = compute_compression_ratio(self.large_model, self.small_model)
        self.benchmark_results["compression"] = compression

        return self.benchmark_results

    def run(
        self,
        validation_data: DataLoader,
        top_k_layers: int = 10,
        criterion: Optional[Callable] = None,
        fine_tune_epochs: int = 0,
        train_data: Optional[DataLoader] = None,
        input_shape: Optional[Tuple] = None,
        measure_qps: bool = True,
        verbose: bool = True,
        **importance_kwargs
    ) -> nn.Module:
        """
        Run the complete pipeline.

        This is the main entry point that runs all steps:
        1. Compute importance scores
        2. Select top-k layers
        3. Transfer knowledge
        4. Fine-tune (optional)
        5. Benchmark (optional)

        Args:
            validation_data: DataLoader for importance computation
            top_k_layers: Number of layers to transfer
            criterion: Loss function
            fine_tune_epochs: Number of fine-tuning epochs (0 to skip)
            train_data: Training data for fine-tuning
            input_shape: Input shape for benchmarking
            measure_qps: Whether to benchmark QPS
            verbose: Print progress
            **importance_kwargs: Additional args for importance computation

        Returns:
            Enhanced small model
        """
        # Step 1: Compute importance
        self.compute_importance(
            validation_data,
            criterion=criterion,
            **importance_kwargs
        )

        # Step 2: Select layers
        self.select_layers(top_k=top_k_layers)

        # Step 3: Transfer knowledge
        self.transfer_knowledge(verbose=verbose)

        # Step 4: Fine-tune (optional)
        if fine_tune_epochs > 0:
            if train_data is None:
                train_data = validation_data
            if criterion is None:
                criterion = nn.CrossEntropyLoss()
            self.fine_tune(train_data, criterion, epochs=fine_tune_epochs)

        # Step 5: Benchmark (optional)
        if measure_qps and input_shape is not None:
            self.benchmark(input_shape, verbose=verbose)

        return self.small_model

    def print_report(self):
        """Print a summary report of the pipeline results."""
        print("\n" + "="*70)
        print("DISTILLATION PIPELINE REPORT")
        print("="*70)

        # Configuration
        print("\n[Configuration]")
        print(f"  Decomposition Method: {self.decomposition_method.upper()}")
        print(f"  Rank: {self.rank}")
        print(f"  Importance Method: {self.importance_method}")

        # Transfer statistics
        if self.transfer_stats:
            print("\n[Transfer Statistics]")
            agg = self.transfer_stats.get('aggregate', {})
            print(f"  Layers Transferred: {agg.get('num_layers', 'N/A')}")
            print(f"  Mean Relative Error: {agg.get('mean_relative_error', 'N/A'):.6f}")
            print(f"  Max Relative Error: {agg.get('max_relative_error', 'N/A'):.6f}")

        # Benchmark results
        if self.benchmark_results:
            print("\n[Benchmark Results]")

            # QPS comparison
            large_qps = self.benchmark_results.get("Large Model_qps", {}).get("mean_qps", 0)
            small_qps = self.benchmark_results.get("Small Model (Enhanced)_qps", {}).get("mean_qps", 0)

            if large_qps > 0 and small_qps > 0:
                qps_improvement = ((small_qps - large_qps) / large_qps) * 100
                print(f"  Large Model QPS: {large_qps:.2f}")
                print(f"  Small Model QPS: {small_qps:.2f}")
                print(f"  QPS Improvement: {qps_improvement:+.1f}%")

            # Compression
            compression = self.benchmark_results.get("compression", {})
            if compression:
                print(f"  Compression Ratio: {compression.get('compression_ratio', 'N/A'):.2f}x")
                print(f"  Size Reduction: {compression.get('size_reduction_percentage', 'N/A'):.1f}%")

        # Recommendations
        print("\n[Recommendations]")
        if self.decomposition_method == "svd":
            print("  WARNING: Vanilla SVD may hurt QPS. Consider using 'lora' or 'fused_svd'.")
        elif self.decomposition_method == "lora":
            print("  GOOD: LoRA typically maintains best QPS while adding knowledge.")
        elif self.decomposition_method == "fused_svd":
            print("  GOOD: Fused SVD balances compression and QPS well.")

        print("="*70)

    def get_enhanced_model(self) -> nn.Module:
        """Get the enhanced small model."""
        return self.small_model

    def save_checkpoint(self, path: str):
        """Save the enhanced model and pipeline state."""
        checkpoint = {
            'model_state_dict': self.small_model.state_dict(),
            'importance_scores': self.importance_scores,
            'selected_layers': self.selected_layers,
            'transfer_stats': self.transfer_stats,
            'benchmark_results': self.benchmark_results,
            'config': {
                'decomposition_method': self.decomposition_method,
                'rank': self.rank,
                'importance_method': self.importance_method,
            }
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")


class AutoPipeline(ModelDistillationPipeline):
    """
    Automatic pipeline that selects the best decomposition method.

    This pipeline automatically chooses between methods based on:
    - QPS requirements
    - Accuracy requirements
    - Layer types (Linear vs Conv)
    - Available memory
    """

    def __init__(
        self,
        large_model: nn.Module,
        small_model: nn.Module,
        qps_priority: float = 0.5,  # 0 = accuracy only, 1 = QPS only
        device: str = "cpu",
        **kwargs
    ):
        """
        Initialize auto pipeline.

        Args:
            large_model: Large source model
            small_model: Small target model
            qps_priority: How much to prioritize QPS over accuracy (0-1)
            device: Device to run on
        """
        # Automatically select method based on priority
        if qps_priority > 0.7:
            method = "lora"  # Best for QPS
            rank = 32  # Lower rank for speed
        elif qps_priority > 0.3:
            method = "fused_svd"  # Balanced
            rank = 64
        else:
            method = "svd"  # Best accuracy but may hurt QPS
            rank = 128  # Higher rank for accuracy

        super().__init__(
            large_model=large_model,
            small_model=small_model,
            decomposition_method=method,
            rank=rank,
            device=device,
            **kwargs
        )

        self.qps_priority = qps_priority

        print(f"AutoPipeline selected: {method.upper()} with rank={rank}")
        print(f"  (QPS priority: {qps_priority:.1f})")
