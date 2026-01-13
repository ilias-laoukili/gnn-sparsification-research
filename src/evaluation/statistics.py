"""Statistical analysis tools for experimental results."""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Any
import pandas as pd


def compute_confidence_intervals(
    results: List[float], confidence: float = 0.95
) -> Dict[str, float]:
    """Compute confidence intervals for experimental results.

    Args:
        results: List of accuracy values from multiple runs
        confidence: Confidence level (default 95%)

    Returns:
        Dictionary with mean, std, ci_lower, ci_upper, n_runs
    """
    if not results or len(results) == 0:
        raise ValueError("Results list cannot be empty")

    results = np.array(results)
    mean = np.mean(results)
    std = np.std(results, ddof=1)
    n = len(results)

    # t-distribution for small sample sizes
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_val * std / np.sqrt(n)

    return {
        "mean": float(mean),
        "std": float(std),
        "ci_lower": float(mean - margin),
        "ci_upper": float(mean + margin),
        "margin": float(margin),
        "n_runs": n,
        "confidence": confidence,
    }


def statistical_significance_test(
    results_a: List[float],
    results_b: List[float],
    paired: bool = True,
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """Test statistical significance between two methods.

    Args:
        results_a: Results from method A (e.g., accuracy values)
        results_b: Results from method B
        paired: Whether samples are paired (same seeds/splits)
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Dictionary with test statistics and interpretation
    """
    if len(results_a) != len(results_b):
        raise ValueError("Result lists must have same length for paired test")

    results_a = np.array(results_a)
    results_b = np.array(results_b)

    if paired:
        # Paired t-test (same random seeds, splits, etc.)
        t_stat, p_value = stats.ttest_rel(results_a, results_b, alternative=alternative)
        test_type = "paired_t_test"
    else:
        # Independent t-test
        t_stat, p_value = stats.ttest_ind(results_a, results_b, alternative=alternative)
        test_type = "independent_t_test"

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(results_a, ddof=1) + np.var(results_b, ddof=1)) / 2)
    cohens_d = (np.mean(results_a) - np.mean(results_b)) / pooled_std if pooled_std > 0 else 0.0

    # Interpretation
    significant = p_value < 0.05
    interpretation = {
        "significant_at_0.05": significant,
        "significant_at_0.01": p_value < 0.01,
        "significant_at_0.001": p_value < 0.001,
    }

    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_size_interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size_interpretation = "small"
    elif abs(cohens_d) < 0.8:
        effect_size_interpretation = "medium"
    else:
        effect_size_interpretation = "large"

    return {
        "test_type": test_type,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": significant,
        "cohens_d": float(cohens_d),
        "effect_size": effect_size_interpretation,
        "mean_a": float(np.mean(results_a)),
        "mean_b": float(np.mean(results_b)),
        "mean_difference": float(np.mean(results_a) - np.mean(results_b)),
        "interpretation": interpretation,
    }


def multiple_comparison_correction(
    p_values: List[float], method: str = "bonferroni", alpha: float = 0.05
) -> Dict[str, Any]:
    """Apply multiple comparison correction to p-values.

    Args:
        p_values: List of p-values from multiple tests
        method: Correction method ('bonferroni', 'holm', 'fdr_bh')
        alpha: Family-wise error rate

    Returns:
        Dictionary with corrected p-values and reject decisions
    """
    p_values = np.array(p_values)
    n = len(p_values)

    if method == "bonferroni":
        # Bonferroni correction (most conservative)
        corrected_alpha = alpha / n
        reject = p_values < corrected_alpha
        corrected_p = np.minimum(p_values * n, 1.0)

    elif method == "holm":
        # Holm-Bonferroni (less conservative)
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        reject = np.zeros(n, dtype=bool)
        corrected_p = np.ones(n)

        for i in range(n):
            corrected_alpha = alpha / (n - i)
            if sorted_p[i] < corrected_alpha:
                reject[sorted_indices[i]] = True
                corrected_p[sorted_indices[i]] = min(sorted_p[i] * (n - i), 1.0)
            else:
                break

    elif method == "fdr_bh":
        # Benjamini-Hochberg (FDR control)
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        reject = np.zeros(n, dtype=bool)
        corrected_p = np.ones(n)

        for i in range(n - 1, -1, -1):
            threshold = (i + 1) / n * alpha
            if sorted_p[i] <= threshold:
                reject[sorted_indices[: i + 1]] = True
                corrected_p[sorted_indices[: i + 1]] = (
                    sorted_p[: i + 1] * n / (np.arange(i + 1) + 1)
                )
                break

    else:
        raise ValueError(f"Unknown correction method: {method}")

    return {
        "method": method,
        "alpha": alpha,
        "n_tests": n,
        "corrected_p_values": corrected_p.tolist(),
        "reject_null": reject.tolist(),
        "n_significant": int(np.sum(reject)),
    }


def bootstrap_confidence_interval(
    data: List[float],
    statistic: callable = np.mean,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute bootstrap confidence interval for any statistic.

    Args:
        data: Original data samples
        statistic: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        seed: Random seed

    Returns:
        Dictionary with statistic value and confidence interval
    """
    rng = np.random.default_rng(seed)
    data = np.array(data)
    n = len(data)

    # Compute observed statistic
    observed = statistic(data)

    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Percentile method
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return {
        "statistic": float(observed),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "bootstrap_mean": float(np.mean(bootstrap_stats)),
        "bootstrap_std": float(np.std(bootstrap_stats)),
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
    }


def summarize_experimental_results(
    results_df: pd.DataFrame, group_by: List[str], metric_columns: List[str]
) -> pd.DataFrame:
    """Summarize experimental results with statistics.

    Args:
        results_df: DataFrame with experimental results
        group_by: Columns to group by (e.g., ['dataset', 'method'])
        metric_columns: Metric columns to summarize (e.g., ['accuracy', 'f1'])

    Returns:
        DataFrame with summary statistics
    """
    summary_rows = []

    for group_keys, group_df in results_df.groupby(group_by):
        row = {}

        # Add group identifiers
        if isinstance(group_keys, tuple):
            for key, value in zip(group_by, group_keys):
                row[key] = value
        else:
            row[group_by[0]] = group_keys

        # Compute statistics for each metric
        for metric in metric_columns:
            values = group_df[metric].dropna().values

            if len(values) > 0:
                ci = compute_confidence_intervals(values.tolist())

                row[f"{metric}_mean"] = ci["mean"]
                row[f"{metric}_std"] = ci["std"]
                row[f"{metric}_ci_lower"] = ci["ci_lower"]
                row[f"{metric}_ci_upper"] = ci["ci_upper"]
                row[f"{metric}_n"] = ci["n_runs"]
            else:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_ci_lower"] = np.nan
                row[f"{metric}_ci_upper"] = np.nan
                row[f"{metric}_n"] = 0

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def compare_methods_pairwise(
    results_df: pd.DataFrame, method_column: str, metric_column: str, group_by: List[str] = None
) -> pd.DataFrame:
    """Perform pairwise statistical comparisons between methods.

    Args:
        results_df: DataFrame with experimental results
        method_column: Column name containing method names
        metric_column: Column name containing metric values
        group_by: Optional columns to group by (e.g., ['dataset'])

    Returns:
        DataFrame with pairwise comparison results
    """
    if group_by is None:
        group_by = []

    comparison_rows = []

    # Get unique methods
    methods = results_df[method_column].unique()

    # Iterate over groups
    if group_by:
        groups = results_df.groupby(group_by)
    else:
        groups = [(None, results_df)]

    for group_keys, group_df in groups:
        # Pairwise comparisons
        for i, method_a in enumerate(methods):
            for method_b in methods[i + 1 :]:
                results_a = group_df[group_df[method_column] == method_a][metric_column].values
                results_b = group_df[group_df[method_column] == method_b][metric_column].values

                if len(results_a) > 0 and len(results_b) > 0:
                    test_result = statistical_significance_test(
                        results_a.tolist(), results_b.tolist(), paired=True
                    )

                    row = {
                        "method_a": method_a,
                        "method_b": method_b,
                        "metric": metric_column,
                        **test_result,
                    }

                    if group_by and group_keys is not None:
                        if isinstance(group_keys, tuple):
                            for key, value in zip(group_by, group_keys):
                                row[key] = value
                        else:
                            row[group_by[0]] = group_keys

                    comparison_rows.append(row)

    return pd.DataFrame(comparison_rows)
