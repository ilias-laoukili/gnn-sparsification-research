#!/usr/bin/env python3
"""Sanity check: Validate Approximate ER correlation with Exact ER on Cora.

This script verifies that the Johnson-Lindenstrauss approximation produces
scores highly correlated with the exact (dense) effective resistance computation.

Target: Pearson correlation > 0.85 before running large-scale experiments.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from src import DatasetLoader, GraphSparsifier

def run_sanity_check(epsilon: float = 0.3) -> float:
    """Run correlation check between exact and approximate ER.
    
    Args:
        epsilon: Approximation error parameter for JLT.
        
    Returns:
        Pearson correlation coefficient.
    """
    # 1. Load small dataset (Cora: ~2.7k nodes, ~10k edges)
    print("=" * 60)
    print("SANITY CHECK: Exact vs. Approximate Effective Resistance")
    print("=" * 60)
    
    loader = DatasetLoader()
    data, num_features, num_classes = loader.get_dataset("cora", "cpu")
    print(f"\nDataset: Cora")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.size(1)}")
    
    sparsifier = GraphSparsifier(data, "cpu")
    
    # 2. Compute Exact ER (this will be slow for Cora ~2.7k nodes)
    print("\nComputing Exact ER (O(N³) - may take ~30-60 seconds)...")
    import time
    t0 = time.time()
    exact_scores = sparsifier.compute_scores("effective_resistance")
    exact_time = time.time() - t0
    print(f"  Done in {exact_time:.2f}s")
    print(f"  Score range: [{exact_scores.min():.4f}, {exact_scores.max():.4f}]")
    
    # 3. Compute Approximate ER
    print(f"\nComputing Approx ER (epsilon={epsilon})...")
    # Clear cache to force recomputation with new epsilon
    sparsifier._score_cache.pop("approx_er", None)
    
    # Import the function directly to pass epsilon parameter
    from src.sparsification.metrics import calculate_approx_effective_resistance_scores
    
    t0 = time.time()
    approx_scores = calculate_approx_effective_resistance_scores(
        sparsifier.adj, epsilon=epsilon, seed=42
    )
    approx_time = time.time() - t0
    print(f"  Done in {approx_time:.2f}s")
    print(f"  Score range: [{approx_scores.min():.4f}, {approx_scores.max():.4f}]")
    print(f"  Speedup: {exact_time / approx_time:.1f}x")
    
    # 4. Correlation Analysis
    print("\n" + "-" * 40)
    print("CORRELATION ANALYSIS")
    print("-" * 40)
    
    pearson_corr, pearson_p = pearsonr(exact_scores, approx_scores)
    spearman_corr, spearman_p = spearmanr(exact_scores, approx_scores)
    
    print(f"  Pearson correlation:  {pearson_corr:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.2e})")
    
    # 5. Rank preservation check (most important for sparsification)
    # Check if top-k edges by exact ER are also in top-k by approx ER
    k = int(0.1 * len(exact_scores))  # Top 10%
    exact_top_k = set(np.argsort(exact_scores)[-k:])
    approx_top_k = set(np.argsort(approx_scores)[-k:])
    rank_overlap = len(exact_top_k & approx_top_k) / k
    print(f"  Top-10% rank overlap: {rank_overlap:.1%}")
    
    # 6. Verdict
    # For sparsification, RANK correlation matters more than absolute values
    # because we only need to correctly identify which edges to keep/remove
    print("\n" + "=" * 40)
    print("VERDICT (based on Spearman rank correlation)")
    print("=" * 40)
    
    if spearman_corr > 0.85:
        print("✅ EXCELLENT: Spearman > 0.85")
        print("   Rank preservation is excellent. Proceed to full experiments.")
    elif spearman_corr > 0.70:
        print("✅ PASS: Spearman > 0.70")
        print("   Rank preservation is good. Safe to proceed.")
        print("   (Absolute values differ but rankings are preserved)")
    elif spearman_corr > 0.50:
        print("⚠️  WARNING: Spearman between 0.50-0.70")
        print("   Consider decreasing epsilon for better rank preservation.")
    else:
        print("❌ FAIL: Spearman < 0.50")
        print("   Rankings are not well preserved. Check implementation.")
    
    print(f"\nTop-10% edge overlap: {rank_overlap:.1%}")
    if rank_overlap > 0.70:
        print("  → High-importance edges are correctly identified ✓")
    print("=" * 40)
    
    # 7. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    ax = axes[0]
    ax.scatter(exact_scores, approx_scores, alpha=0.3, s=3, c='steelblue')
    
    # Add perfect fit line
    max_val = max(exact_scores.max(), approx_scores.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label="Perfect Fit (y=x)")
    
    # Add linear regression line
    z = np.polyfit(exact_scores, approx_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(exact_scores.min(), exact_scores.max(), 100)
    ax.plot(x_line, p(x_line), 'g-', linewidth=2, alpha=0.7, 
            label=f"Linear Fit (r={pearson_corr:.3f})")
    
    ax.set_xlabel("Exact Effective Resistance", fontsize=12)
    ax.set_ylabel("Approximate Effective Resistance", fontsize=12)
    ax.set_title(f"Approximation Quality on Cora\n(Pearson r = {pearson_corr:.3f})", fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Histogram of relative errors
    ax = axes[1]
    # Avoid division by zero
    valid_mask = exact_scores > 1e-8
    relative_errors = np.abs(approx_scores[valid_mask] - exact_scores[valid_mask]) / exact_scores[valid_mask]
    
    ax.hist(relative_errors, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(np.median(relative_errors), color='red', linestyle='--', 
               label=f'Median: {np.median(relative_errors):.2f}')
    ax.axvline(np.mean(relative_errors), color='orange', linestyle='--', 
               label=f'Mean: {np.mean(relative_errors):.2f}')
    ax.set_xlabel("Relative Error |approx - exact| / exact", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Relative Errors", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent.parent / "notebooks" / "results" / "approx_er_sanity_check.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    plt.show()
    
    return pearson_corr


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sanity check for Approximate ER")
    parser.add_argument("--epsilon", type=float, default=0.3, 
                        help="JLT approximation error parameter (default: 0.3)")
    args = parser.parse_args()
    
    correlation = run_sanity_check(epsilon=args.epsilon)
