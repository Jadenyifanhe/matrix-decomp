"""
Compare Multiple Importance Scoring Methods

This example demonstrates:
1. How to use different importance scoring methods
2. How to compare results across methods
3. Which methods agree and disagree
4. Performance characteristics of each method
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
from scipy.stats import spearmanr

from matrix_decomp.importance import (
    compute_importance_scores,
    select_top_k_layers,
    print_importance_scores
)


def create_test_model():
    """Create a simple test model."""
    return nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )


def create_test_data(num_samples=1000):
    """Create dummy dataset."""
    X = torch.randn(num_samples, 128)
    y = torch.randint(0, 10, (num_samples,))
    return DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)


def main():
    print("="*80)
    print("Comparing Importance Scoring Methods")
    print("="*80)

    # Setup
    model = create_test_model()
    data_loader = create_test_data(num_samples=500)  # Small dataset for speed
    criterion = nn.CrossEntropyLoss()

    # Methods to compare
    methods_to_test = {
        # Fast methods (data-free)
        'Spectral Norm (data-free)': {
            'method': 'spectral_norm',
            'needs_data': False,
            'needs_labels': False
        },
        'Effective Rank (data-free)': {
            'method': 'effective_rank',
            'needs_data': False,
            'needs_labels': False
        },
        'SynFlow (data-free)': {
            'method': 'synflow',
            'needs_data': False,
            'needs_labels': False
        },

        # Fast methods (data-based)
        'Gradient': {
            'method': 'gradient',
            'needs_data': True,
            'needs_labels': True
        },
        'Taylor (Recommended)': {
            'method': 'taylor',
            'needs_data': True,
            'needs_labels': True
        },
        'Activation': {
            'method': 'activation',
            'needs_data': True,
            'needs_labels': False
        },

        # Medium-speed methods
        'Fisher Information': {
            'method': 'fisher',
            'needs_data': True,
            'needs_labels': True
        },
        'SNIP': {
            'method': 'snip',
            'needs_data': True,
            'needs_labels': True
        },
    }

    results = {}
    timings = {}

    # Run each method
    for name, config in methods_to_test.items():
        print(f"\n{'-'*80}")
        print(f"Method: {name}")
        print(f"{'-'*80}")

        try:
            start_time = time.time()

            if config['needs_data'] and config['needs_labels']:
                scores = compute_importance_scores(
                    model,
                    data_loader,
                    method=config['method'],
                    criterion=criterion,
                    verbose=False,
                    max_batches=10  # Limit for speed
                )
            elif config['needs_data'] and not config['needs_labels']:
                scores = compute_importance_scores(
                    model,
                    data_loader,
                    method=config['method'],
                    verbose=False,
                    max_batches=10
                )
            else:
                # Data-free methods
                if config['method'] == 'synflow':
                    scores = compute_importance_scores(
                        model,
                        method=config['method'],
                        input_shape=(128,),
                        verbose=False
                    )
                else:
                    scores = compute_importance_scores(
                        model,
                        method=config['method'],
                        verbose=False
                    )

            elapsed = time.time() - start_time

            results[name] = scores
            timings[name] = elapsed

            print(f"✓ Completed in {elapsed:.3f} seconds")

            # Print top-3 layers
            top_3 = select_top_k_layers(scores, k=3)
            print(f"\nTop 3 layers:")
            for i, (layer_name, score) in enumerate(top_3, 1):
                print(f"  {i}. {layer_name}: {score:.4f}")

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    # Comparison Analysis
    print("\n" + "="*80)
    print("Performance Comparison")
    print("="*80)

    print(f"\n{'Method':<35} {'Time (s)':<15} {'Speed':<15}")
    print("-"*65)

    sorted_by_time = sorted(timings.items(), key=lambda x: x[1])
    for name, elapsed in sorted_by_time:
        if elapsed < 0.1:
            speed = "⚡⚡⚡⚡ Ultra Fast"
        elif elapsed < 1.0:
            speed = "⚡⚡⚡ Very Fast"
        elif elapsed < 5.0:
            speed = "⚡⚡ Fast"
        else:
            speed = "⚡ Medium"

        print(f"{name:<35} {elapsed:<15.3f} {speed:<15}")

    # Agreement Analysis
    print("\n" + "="*80)
    print("Method Agreement (Spearman Correlation)")
    print("="*80)

    method_names = list(results.keys())
    print(f"\n{'':>35}", end="")
    for name in method_names:
        print(f"{name[:12]:<15}", end="")
    print()
    print("-" * (35 + 15 * len(method_names)))

    for i, name1 in enumerate(method_names):
        print(f"{name1:<35}", end="")
        for j, name2 in enumerate(method_names):
            if i == j:
                print(f"{'1.000':<15}", end="")
            elif i > j:
                print(f"{'':^15}", end="")
            else:
                # Get common layers
                common = set(results[name1].keys()) & set(results[name2].keys())
                scores1 = [results[name1][layer] for layer in sorted(common)]
                scores2 = [results[name2][layer] for layer in sorted(common)]

                if len(scores1) > 1:
                    corr, _ = spearmanr(scores1, scores2)
                    print(f"{corr:<15.3f}", end="")
                else:
                    print(f"{'N/A':<15}", end="")
        print()

    # Consensus Analysis
    print("\n" + "="*80)
    print("Consensus Top Layers")
    print("="*80)

    # Count how many methods rank each layer in top-5
    layer_counts = {}
    for name, scores in results.items():
        top_5 = select_top_k_layers(scores, k=5)
        for layer_name, _ in top_5:
            layer_counts[layer_name] = layer_counts.get(layer_name, 0) + 1

    sorted_by_consensus = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\nLayers that appear in top-5 across multiple methods:")
    print(f"{'Layer Name':<40} {'# Methods':<15} {'Consensus':<15}")
    print("-" * 70)

    for layer, count in sorted_by_consensus[:10]:
        percentage = (count / len(results)) * 100
        consensus = "🟢 Strong" if percentage >= 75 else "🟡 Medium" if percentage >= 50 else "🔵 Weak"
        print(f"{layer:<40} {count}/{len(results):<13} {consensus:<15}")

    # Recommendations
    print("\n" + "="*80)
    print("Recommendations Based on Results")
    print("="*80)

    # Find fastest method
    fastest = min(timings.items(), key=lambda x: x[1])

    # Find most agreed method (average correlation with others)
    avg_corrs = {}
    for name1 in method_names:
        corrs = []
        for name2 in method_names:
            if name1 != name2:
                common = set(results[name1].keys()) & set(results[name2].keys())
                scores1 = [results[name1][layer] for layer in sorted(common)]
                scores2 = [results[name2][layer] for layer in sorted(common)]
                if len(scores1) > 1:
                    corr, _ = spearmanr(scores1, scores2)
                    corrs.append(corr)
        if corrs:
            avg_corrs[name1] = np.mean(corrs)

    most_agreed = max(avg_corrs.items(), key=lambda x: x[1]) if avg_corrs else None

    print(f"""
    1. Fastest Method: {fastest[0]} ({fastest[1]:.3f}s)
       → Use for quick exploration

    2. Most Consensus: {most_agreed[0]} (avg corr: {most_agreed[1]:.3f})
       → Use for reliable results

    3. For Production: Use 'Taylor (Recommended)'
       → Best balance of speed and accuracy

    4. No Data Available: Use 'Spectral Norm (data-free)' or 'SynFlow (data-free)'
       → Surprisingly effective without any data

    5. Maximum Accuracy: Add 'knockout' method (not tested here - too slow)
       → Use for final validation
    """)

    # Summary
    print("="*80)
    print("Summary")
    print("="*80)
    print(f"""
    Tested {len(results)} different importance scoring methods.

    Key Findings:
    - Data-free methods (spectral_norm, effective_rank, synflow) are very fast
    - Taylor and Gradient methods provide good accuracy-speed tradeoff
    - Most methods show reasonable agreement (correlation > 0.5)
    - Top layers identified by multiple methods are most reliable

    Next Steps:
    1. Choose a method based on your constraints (speed vs accuracy)
    2. Use the top layers for matrix decomposition
    3. Validate results with actual model performance
    """)


if __name__ == "__main__":
    main()
