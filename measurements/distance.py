"""
Distance and geometric deviation calculations for MOTION DX.
Provides 2D pixel/normalized distance, 3D Euclidean metric distance, and perpendicular line deviation.
"""

from typing import Sequence, Tuple
import numpy as np


def calculate_distance_2d(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Calculate Euclidean distance between two 2D points."""
    if p1 is None or p2 is None:
        return 0.0
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return float(np.sqrt(dx * dx + dy * dy))


def calculate_distance_3d(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Calculate Euclidean distance between two 3D points in the point's units (e.g. mm or meters)."""
    if p1 is None or p2 is None:
        return 0.0
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def point_to_line_distance_2d(
    point: Sequence[float],
    line_p1: Sequence[float],
    line_p2: Sequence[float]
) -> Tuple[float, float]:
    """
    Calculate the perpendicular distance and signed lateral deviation of `point` from the line passing through `line_p1` and `line_p2`.
    
    Useful for measuring lateral jaw excursion relative to facial midline (e.g. Glabella -> Philtrum line).
    
    Returns:
        (abs_distance, signed_deviation):
        signed_deviation > 0 means to the right of vector line_p1 -> line_p2,
        signed_deviation < 0 means to the left.
    """
    if point is None or line_p1 is None or line_p2 is None:
        return 0.0, 0.0

    x0, y0 = point[0], point[1]
    x1, y1 = line_p1[0], line_p1[1]
    x2, y2 = line_p2[0], line_p2[1]

    dx = x2 - x1
    dy = y2 - y1
    line_len = np.sqrt(dx * dx + dy * dy)
    if line_len < 1e-6:
        return 0.0, 0.0

    # Cross product (line_p1->line_p2) x (line_p1->point) gives signed distance * line_len
    # 2D cross product: dx*(y0 - y1) - dy*(x0 - x1)
    cross = dx * (y0 - y1) - dy * (x0 - x1)
    signed_dist = float(cross / line_len)
    return abs(signed_dist), signed_dist
