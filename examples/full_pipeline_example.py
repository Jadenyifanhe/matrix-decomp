"""
Full Pipeline Example

This example demonstrates the complete workflow:
1. Train a large model (simulated)
2. Identify important layers
3. Decompose important layers
4. Transfer to small model
5. Benchmark performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from matrix_decomp.importance import compute_importance_scores, select_top_k_layers
from matrix_decomp.transfer import KnowledgeTransfer
from matrix_decomp.benchmarks import compare_qps, compare_flops, compute_compression_ratio


class SimpleClassifier(nn.Module):
    """Simple classifier for demonstration."""

    def __init__(self, input_dim=128, hidden_dims=[256, 512, 256], num_classes=10):
        super().__init__()
        layers = []

        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def create_dummy_data(num_samples=1000, input_dim=128, num_classes=10):
    """Create dummy dataset for demonstration."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


def train_model(model, train_loader, epochs=5):
    """Quick training (simulated improvement)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


def main():
    print("="*80)
    print("Full Pipeline: Large Model -> Small Model Knowledge Transfer")
    print("="*80)

    # Step 1: Create models
    print("\nStep 1: Creating Models")
    print("-"*80)

    # Large model (5x scale-up)
    large_model = SimpleClassifier(
        input_dim=128,
        hidden_dims=[1280, 2560, 1280],  # 5x larger
        num_classes=10
    )

    # Small model (original)
    small_model = SimpleClassifier(
        input_dim=128,
        hidden_dims=[256, 512, 256],
        num_classes=10
    )

    print(f"Large model: {sum(p.numel() for p in large_model.parameters()):,} parameters")
    print(f"Small model: {sum(p.numel() for p in small_model.parameters()):,} parameters")

    # Step 2: Train large model
    print("\nStep 2: Training Large Model")
    print("-"*80)

    train_dataset = create_dummy_data(num_samples=1000)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print("Training large model...")
    train_model(large_model, train_loader, epochs=5)

    # Step 3: Identify important layers
    print("\nStep 3: Identifying Important Layers")
    print("-"*80)

    criterion = nn.CrossEntropyLoss()

    print("Computing importance scores...")
    importance_scores = compute_importance_scores(
        large_model,
        train_loader,
        method='gradient',  # Use gradient-based importance
        criterion=criterion,
        verbose=False
    )

    # Select top-k layers
    top_k = 3
    top_layers = select_top_k_layers(importance_scores, k=top_k)

    print(f"\nTop {top_k} most important layers:")
    for i, (layer_name, score) in enumerate(top_layers, 1):
        print(f"  {i}. {layer_name}: {score:.4f}")

    # Step 4: Transfer knowledge using different methods
    print("\nStep 4: Knowledge Transfer")
    print("-"*80)

    methods = ['lora', 'fused_svd']
    transferred_models = {}

    for method in methods:
        print(f"\nUsing {method.upper()} method...")

        # Create fresh small model
        target_model = SimpleClassifier(
            input_dim=128,
            hidden_dims=[256, 512, 256],
            num_classes=10
        )

        # Initialize transfer
        transfer = KnowledgeTransfer(method=method, rank=64)

        # Get layer names to transfer (only Linear layers)
        layer_names = [name for name, _ in top_layers if 'network.' in name]

        # Transfer
        stats = transfer.transfer_model(
            large_model,
            target_model,
            layer_names=layer_names,
            adapt_dims=True,
            verbose=False
        )

        print(f"  Transferred {stats['aggregate']['num_layers']} layers")
        print(f"  Mean relative error: {stats['aggregate']['mean_relative_error']:.6f}")

        transferred_models[f"Small + {method.upper()}"] = target_model

    # Step 5: Benchmark QPS
    print("\nStep 5: QPS Benchmarking")
    print("-"*80)

    all_models = {
        "Small (Baseline)": small_model,
        **transferred_models
    }

    input_shape = (128,)

    qps_results = compare_qps(
        all_models,
        input_shape,
        batch_size=1,
        num_iterations=100,
        verbose=True
    )

    # Step 6: Benchmark FLOPs
    print("\nStep 6: FLOPs Analysis")
    print("-"*80)

    flops_results = compare_flops(
        all_models,
        input_shape,
        verbose=True
    )

    # Step 7: Model compression analysis
    print("\nStep 7: Compression Analysis")
    print("-"*80)

    baseline_params = sum(p.numel() for p in small_model.parameters())

    print(f"{'Model':<30} {'Parameters':<15} {'Relative Size':<15}")
    print("-" * 60)

    for name, model in all_models.items():
        params = sum(p.numel() for p in model.parameters())
        relative = params / baseline_params
        print(f"{name:<30} {params:<15,} {relative:<15.2f}x")

    # Step 8: Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    print("\nPerformance Comparison:")
    baseline_qps = qps_results["Small (Baseline)"]["mean_qps"]

    for name in transferred_models.keys():
        qps = qps_results[name]["mean_qps"]
        qps_change = ((qps - baseline_qps) / baseline_qps) * 100

        print(f"\n{name}:")
        print(f"  QPS change: {qps_change:+.1f}%")
        print(f"  Method benefits: Knowledge from 5x larger model")

    print("\n" + "="*80)
    print("Key Insights:")
    print("="*80)
    print("1. LoRA typically maintains better QPS than vanilla SVD")
    print("2. Gradient-based importance helps select the right layers")
    print("3. Knowledge transfer improves small model without size increase")
    print("4. Fused SVD is a good middle ground between SVD and LoRA")
    print("="*80)


if __name__ == "__main__":
    main()
