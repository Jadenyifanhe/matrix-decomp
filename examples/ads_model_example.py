"""
Ads Model Knowledge Transfer Example

This example demonstrates the complete workflow for ads models:
1. Simulate large-scale ads model (5x scale-up)
2. Train large model
3. Identify important components
4. Transfer knowledge to production model
5. Ensure QPS requirements are met
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from matrix_decomp.importance import compute_importance_scores
from matrix_decomp.transfer import KnowledgeTransfer
from matrix_decomp.benchmarks import measure_qps, count_flops


class AdsModel(nn.Module):
    """
    Simplified Ads Prediction Model.

    In production, this would be more complex with:
    - Embedding layers for categorical features
    - Cross-product layers for feature interactions
    - Deep & Wide architecture
    - etc.
    """

    def __init__(self, feature_dim=512, hidden_dims=[256, 128, 64]):
        super().__init__()

        # Feature processing
        layers = []
        prev_dim = feature_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        # Prediction head (CTR prediction)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def create_ads_dataset(num_samples=10000, feature_dim=512):
    """Create synthetic ads dataset."""
    # Features: user features + ad features + context features
    X = torch.randn(num_samples, feature_dim)

    # Labels: click probability (0 or 1)
    y = (torch.randn(num_samples, 1) > 0).float()

    return TensorDataset(X, y)


def main():
    print("="*80)
    print("Ads Model Knowledge Transfer Pipeline")
    print("="*80)

    # Configuration
    FEATURE_DIM = 512
    PRODUCTION_HIDDEN = [256, 128, 64]  # Production model
    LARGE_HIDDEN = [1280, 640, 320]     # 5x scale-up
    RANK = 64
    TOP_K_LAYERS = 3

    # QPS requirements
    MIN_QPS_REQUIREMENT = 100  # Minimum QPS for production

    # Step 1: Create models
    print("\nStep 1: Model Setup")
    print("-"*80)

    production_model = AdsModel(FEATURE_DIM, PRODUCTION_HIDDEN)
    large_model = AdsModel(FEATURE_DIM, LARGE_HIDDEN)

    prod_params = sum(p.numel() for p in production_model.parameters())
    large_params = sum(p.numel() for p in large_model.parameters())

    print(f"Production model: {prod_params:,} parameters")
    print(f"Large model: {large_params:,} parameters ({large_params/prod_params:.1f}x)")

    # Step 2: Simulate training on large model
    print("\nStep 2: Training Large Model")
    print("-"*80)
    print("(In production, train this on your full dataset for multiple epochs)")

    dataset = create_ads_dataset(num_samples=10000, feature_dim=FEATURE_DIM)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Quick training simulation
    optimizer = torch.optim.Adam(large_model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    large_model.train()
    for epoch in range(3):
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = large_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

    # Step 3: Identify important layers
    print("\nStep 3: Identifying Important Components")
    print("-"*80)

    importance_scores = compute_importance_scores(
        large_model,
        train_loader,
        method='gradient',
        criterion=criterion,
        verbose=False
    )

    # Filter for Linear layers only
    linear_scores = {k: v for k, v in importance_scores.items() if 'Linear' in str(type(large_model))}

    # Sort and select top-k
    sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    top_layers = sorted_layers[:TOP_K_LAYERS]

    print(f"\nTop {TOP_K_LAYERS} most important layers:")
    for i, (name, score) in enumerate(top_layers, 1):
        print(f"  {i}. {name}: importance = {score:.4f}")

    # Step 4: Test different transfer methods
    print("\nStep 4: Testing Transfer Methods")
    print("-"*80)

    methods_to_test = [
        ('lora', 'LoRA (Recommended for Production)'),
        ('fused_svd', 'Fused SVD'),
    ]

    results = {}

    for method, description in methods_to_test:
        print(f"\nTesting: {description}")
        print("-" * 40)

        # Create fresh production model
        enhanced_model = AdsModel(FEATURE_DIM, PRODUCTION_HIDDEN)

        # Transfer knowledge
        transfer = KnowledgeTransfer(method=method, rank=RANK)

        layer_names = [name for name, _ in top_layers]

        transfer_stats = transfer.transfer_model(
            large_model,
            enhanced_model,
            layer_names=layer_names,
            adapt_dims=True,
            verbose=False
        )

        print(f"  Transferred {transfer_stats['aggregate']['num_layers']} layers")
        print(f"  Avg reconstruction error: {transfer_stats['aggregate']['mean_relative_error']:.6f}")

        # Measure QPS
        input_shape = (FEATURE_DIM,)
        qps_result = measure_qps(
            enhanced_model,
            input_shape,
            batch_size=1,
            num_iterations=100,
            verbose=False
        )

        # Count FLOPs
        flops_result = count_flops(enhanced_model, input_shape, verbose=False)

        results[method] = {
            'model': enhanced_model,
            'qps': qps_result['mean_qps'],
            'latency_ms': qps_result['mean_latency_ms'],
            'gflops': flops_result['gflops'],
            'transfer_error': transfer_stats['aggregate']['mean_relative_error']
        }

        print(f"  QPS: {qps_result['mean_qps']:.2f}")
        print(f"  Latency: {qps_result['mean_latency_ms']:.2f} ms")

    # Step 5: Compare with baseline
    print("\nStep 5: Performance Analysis")
    print("-"*80)

    # Baseline production model
    baseline_qps = measure_qps(
        production_model,
        (FEATURE_DIM,),
        batch_size=1,
        num_iterations=100,
        verbose=False
    )

    print(f"\n{'Model':<30} {'QPS':<15} {'Latency (ms)':<15} {'QPS Change':<15}")
    print("-" * 75)

    baseline_qps_val = baseline_qps['mean_qps']
    print(f"{'Production (Baseline)':<30} {baseline_qps_val:<15.2f} {baseline_qps['mean_latency_ms']:<15.2f} {'---':<15}")

    for method, description in methods_to_test:
        result = results[method]
        qps = result['qps']
        latency = result['latency_ms']
        qps_change = ((qps - baseline_qps_val) / baseline_qps_val) * 100

        print(f"{description:<30} {qps:<15.2f} {latency:<15.2f} {qps_change:+.1f}%")

    # Step 6: Make recommendation
    print("\nStep 6: Recommendation")
    print("-"*80)

    # Find method with best QPS that meets requirements
    best_method = None
    best_qps = 0

    for method in results:
        if results[method]['qps'] >= MIN_QPS_REQUIREMENT:
            if results[method]['qps'] > best_qps:
                best_qps = results[method]['qps']
                best_method = method

    if best_method:
        method_names = dict(methods_to_test)
        print(f"\n✓ RECOMMENDED: {method_names[best_method]}")
        print(f"  Reason: Best QPS ({results[best_method]['qps']:.2f}) while meeting requirements")
        print(f"  Transfer error: {results[best_method]['transfer_error']:.6f}")
        print(f"  This method successfully transferred knowledge from the 5x larger model")
        print(f"  while maintaining production QPS requirements.")
    else:
        print("\n✗ WARNING: No method meets minimum QPS requirement")
        print(f"  Consider: Lower rank, fewer layers, or different method")

    # Step 7: Deployment checklist
    print("\n" + "="*80)
    print("Deployment Checklist")
    print("="*80)
    print("""
    □ Verify QPS on production hardware (CPU/GPU)
    □ Test with production batch sizes
    □ Measure actual prediction performance (AUC/etc)
    □ A/B test against current production model
    □ Monitor latency percentiles (P95, P99)
    □ Set up rollback plan if QPS degrades
    □ Document which layers were enhanced
    □ Set up monitoring for model drift
    """)

    print("="*80)
    print("Next Steps")
    print("="*80)
    print(f"""
    1. Use the enhanced model: results['{best_method}']['model']
    2. Fine-tune on your production dataset if needed
    3. Run offline evaluation to measure prediction improvement
    4. Deploy to shadow traffic first
    5. Gradually ramp up to production traffic

    Expected outcome:
    - Similar or better QPS than baseline
    - Improved prediction performance from large model knowledge
    - Same model size as production baseline
    """)


if __name__ == "__main__":
    main()
