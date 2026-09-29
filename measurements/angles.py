"""
Angle calculation algorithms for 2D and 3D human kinematic analysis.
Separated strictly from UI and diagnostic logic.
"""

from typing import Optional, Sequence, Tuple, Union
import numpy as np


def calculate_angle_2d(
    p1: Sequence[float],
    vertex: Sequence[float],
    p3: Sequence[float]
) -> float:
    """
    Calculate the interior angle at `vertex` formed by points (p1, vertex, p3) in 2D space.
    
    Args:
        p1: (x, y) coordinates of first landmark.
        vertex: (x, y) coordinates of the joint center landmark.
        p3: (x, y) coordinates of third landmark.
        
    Returns:
        Angle in degrees in the range [0.0, 180.0]. Returns 0.0 on degenerate vectors.
    """
    if p1 is None or vertex is None or p3 is None:
        return 0.0

    v1 = np.array([p1[0] - vertex[0], p1[1] - vertex[1]], dtype=np.float64)
    v2 = np.array([p3[0] - vertex[0], p3[1] - vertex[1]], dtype=np.float64)

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 < 1e-6 or norm_v2 < 1e-6:
        return 0.0

    cosine = np.dot(v1, v2) / (norm_v1 * norm_v2)
    # Clip to prevent floating-point inaccuracies outside [-1.0, 1.0]
    cosine = np.clip(cosine, -1.0, 1.0)
    angle_rad = np.arccos(cosine)
    return float(np.degrees(angle_rad))


def calculate_angle_3d(
    p1: Sequence[float],
    vertex: Sequence[float],
    p3: Sequence[float]
) -> float:
    """
    Calculate the interior angle at `vertex` formed by points (p1, vertex, p3) in 3D Euclidean space.
    
    Args:
        p1: (x, y, z) coordinates of first landmark.
        vertex: (x, y, z) coordinates of joint center.
        p3: (x, y, z) coordinates of third landmark.
        
    Returns:
        Angle in degrees in the range [0.0, 180.0].
    """
    if p1 is None or vertex is None or p3 is None:
        return 0.0

    v1 = np.array([p1[0] - vertex[0], p1[1] - vertex[1], p1[2] - vertex[2]], dtype=np.float64)
    v2 = np.array([p3[0] - vertex[0], p3[1] - vertex[1], p3[2] - vertex[2]], dtype=np.float64)

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 < 1e-6 or norm_v2 < 1e-6:
        return 0.0

    cosine = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle_rad = np.arccos(cosine)
    return float(np.degrees(angle_rad))


def calculate_joint_angle(
    p1_2d: Sequence[float],
    vertex_2d: Sequence[float],
    p3_2d: Sequence[float],
    p1_3d: Optional[Sequence[float]] = None,
    vertex_3d: Optional[Sequence[float]] = None,
    p3_3d: Optional[Sequence[float]] = None,
    prefer_3d: bool = True
) -> Tuple[float, str]:
    """
    Calculate joint angle with automatic fallback to 2D if 3D coordinates are invalid or missing.
    
    Returns:
        (angle_deg, mode_used: '3D' | '2D')
    """
    if prefer_3d and p1_3d is not None and vertex_3d is not None and p3_3d is not None:
        # Check validity of depth coordinates (z > 0 and finite)
        if (p1_3d[2] > 0 and np.isfinite(p1_3d[2]) and
            vertex_3d[2] > 0 and np.isfinite(vertex_3d[2]) and
            p3_3d[2] > 0 and np.isfinite(p3_3d[2])):
            angle = calculate_angle_3d(p1_3d, vertex_3d, p3_3d)
            if angle > 0.0:
                return round(angle, 1), "3D"

    # Fallback to 2D projection
    angle = calculate_angle_2d(p1_2d, vertex_2d, p3_2d)
    return round(angle, 1), "2D"
