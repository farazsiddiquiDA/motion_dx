"""
Bilateral symmetry calculation algorithms for clinical movement comparison.
Does not label values as normal/abnormal; reports objective differential metrics.
"""

from typing import Dict, Any


def calculate_symmetry_index(left_value: float, right_value: float) -> Dict[str, Any]:
    """
    Calculate bilateral symmetry metrics between left and right limb or facial movements.
    
    Formula:
        Absolute Difference = |Left - Right|
        Symmetry Index (SI %) = max(0.0, 100.0 - (|Left - Right| / Mean(Left, Right)) * 100)
    
    Returns:
        Dict with left, right, difference, symmetry_index_pct, and descriptive observation.
    """
    left = float(left_value)
    right = float(right_value)
    diff = round(left - right, 1)
    abs_diff = round(abs(left - right), 1)
    mean_val = (left + right) / 2.0

    if mean_val > 1e-4:
        asymmetry_pct = round((abs_diff / mean_val) * 100.0, 1)
        symmetry_idx = round(max(0.0, min(100.0, 100.0 - asymmetry_pct)), 1)
    else:
        asymmetry_pct = 0.0
        symmetry_idx = 100.0

    observation = f"Observed difference: {abs_diff} units (L: {left}, R: {right})"

    return {
        "left_value": left,
        "right_value": right,
        "difference": diff,
        "absolute_difference": abs_diff,
        "symmetry_index_pct": symmetry_idx,
        "asymmetry_pct": asymmetry_pct,
        "observation": observation
    }
