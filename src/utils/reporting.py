import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

def print_text_table(
    data: Dict[str, Dict[str, Any]],
    title: str = "Statistics",
    float_fmt: str = "{:.4f}",
    col_width: int = 15,
    index_width: int = 25
) -> None:
    """
    Prints a formatted text table from a dictionary of statistics.
    
    Args:
        data: Dictionary where keys are column headers (e.g. Dataset names)
              and values are dictionaries of {metric_name: value}.
        title: Title to display above the table.
        float_fmt: Format string for floating point numbers.
        col_width: Width of data columns.
        index_width: Width of the index (metric name) column.
    """
    if not data:
        print("No data to display.")
        return

    columns = list(data.keys())
    
    # Collect all unique metrics to ensure we cover everything
    metrics = []
    seen = set()
    for col in columns:
        for metric in data[col].keys():
            if metric not in seen:
                metrics.append(metric)
                seen.add(metric)

    # Calculate total width
    total_width = index_width + (col_width * len(columns))
    
    separator = "*" * total_width
    
    if title:
        print(f" {title} ".center(total_width, "*"))
    else:
        print(separator)
    
    # Print Headers
    header = f"{'Metric':<{index_width}}" + "".join([f"{c:>{col_width}}" for c in columns])
    print(header)
    print(separator)

    # Print Rows
    for metric in metrics:
        row = f"{metric:<{index_width}}"
        for col in columns:
            val = data[col].get(metric, "-")
            if isinstance(val, (float, np.floating)):
                val_str = float_fmt.format(val)
            elif isinstance(val, (int, np.integer)):
                val_str = f"{val:,}"
            else:
                val_str = str(val)
            row += f"{val_str:>{col_width}}"
        print(row)