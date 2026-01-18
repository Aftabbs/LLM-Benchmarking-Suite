"""
Utility functions for LLM Benchmarking Suite.
"""

import json
import hashlib
import statistics
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats


@dataclass
class StatisticalResult:
    """Statistical analysis result."""

    mean: float
    median: float
    std_dev: float
    min_val: float
    max_val: float
    confidence_interval_95: tuple[float, float]
    sample_size: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


def calculate_statistics(values: list[float]) -> StatisticalResult:
    """
    Calculate comprehensive statistics for a list of values.

    Args:
        values: List of numeric values

    Returns:
        StatisticalResult with all statistics
    """
    if not values:
        return StatisticalResult(
            mean=0.0,
            median=0.0,
            std_dev=0.0,
            min_val=0.0,
            max_val=0.0,
            confidence_interval_95=(0.0, 0.0),
            sample_size=0,
        )

    n = len(values)
    mean = statistics.mean(values)
    median = statistics.median(values)
    std_dev = statistics.stdev(values) if n > 1 else 0.0
    min_val = min(values)
    max_val = max(values)

    # Calculate 95% confidence interval
    if n > 1:
        se = std_dev / np.sqrt(n)
        ci = stats.t.interval(0.95, n - 1, loc=mean, scale=se)
        confidence_interval_95 = (ci[0], ci[1])
    else:
        confidence_interval_95 = (mean, mean)

    return StatisticalResult(
        mean=mean,
        median=median,
        std_dev=std_dev,
        min_val=min_val,
        max_val=max_val,
        confidence_interval_95=confidence_interval_95,
        sample_size=n,
    )


def calculate_p_value(
    group1: list[float],
    group2: list[float],
) -> float:
    """
    Calculate p-value for difference between two groups.

    Uses Welch's t-test for unequal variances.

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        P-value for the difference
    """
    if len(group1) < 2 or len(group2) < 2:
        return 1.0

    _, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    return p_value


def normalize_score(
    value: float,
    min_val: float,
    max_val: float,
    higher_is_better: bool = True,
) -> float:
    """
    Normalize a score to 0-1 range.

    Args:
        value: Value to normalize
        min_val: Minimum expected value
        max_val: Maximum expected value
        higher_is_better: If True, higher values get higher scores

    Returns:
        Normalized score between 0 and 1
    """
    if max_val == min_val:
        return 0.5

    normalized = (value - min_val) / (max_val - min_val)
    normalized = max(0.0, min(1.0, normalized))

    if not higher_is_better:
        normalized = 1.0 - normalized

    return normalized


def generate_run_id() -> str:
    """Generate a unique run ID."""
    timestamp = datetime.now().isoformat()
    hash_input = f"{timestamp}".encode()
    return hashlib.sha256(hash_input).hexdigest()[:12]


def save_results(
    results: dict[str, Any],
    output_path: Path,
    format: str = "json",
) -> Path:
    """
    Save benchmark results to file.

    Args:
        results: Results dictionary
        output_path: Path to save to
        format: Output format (json, csv)

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
    else:
        raise ValueError(f"Unsupported format: {format}")

    return output_path


def load_results(input_path: Path) -> dict[str, Any]:
    """
    Load benchmark results from file.

    Args:
        input_path: Path to load from

    Returns:
        Results dictionary
    """
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    """
    Load a benchmark dataset.

    Args:
        dataset_path: Path to dataset file

    Returns:
        List of dataset entries
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "data" in data:
        return data["data"]
    else:
        raise ValueError("Invalid dataset format")


def format_duration(ms: float) -> str:
    """
    Format duration in milliseconds to human-readable string.

    Args:
        ms: Duration in milliseconds

    Returns:
        Formatted string
    """
    if ms < 1000:
        return f"{ms:.2f}ms"
    elif ms < 60000:
        return f"{ms / 1000:.2f}s"
    else:
        minutes = int(ms // 60000)
        seconds = (ms % 60000) / 1000
        return f"{minutes}m {seconds:.1f}s"


def format_cost(cost: float) -> str:
    """
    Format cost to human-readable string.

    Args:
        cost: Cost in dollars

    Returns:
        Formatted string
    """
    if cost < 0.01:
        return f"${cost:.6f}"
    elif cost < 1.0:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def create_comparison_table(
    results: dict[str, dict[str, Any]],
    metrics: list[str],
) -> list[dict[str, Any]]:
    """
    Create a comparison table from benchmark results.

    Args:
        results: Dictionary of model results
        metrics: List of metrics to include

    Returns:
        List of rows for the comparison table
    """
    rows = []

    for model_name, model_results in results.items():
        row = {"model": model_name}
        for metric in metrics:
            if metric in model_results:
                value = model_results[metric]
                if isinstance(value, dict) and "mean" in value:
                    row[metric] = value["mean"]
                else:
                    row[metric] = value
            else:
                row[metric] = None
        rows.append(row)

    return rows
