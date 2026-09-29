"""
Base camera abstractions and data models for MOTION DX.

Provides:
- RGB frame representation
- Depth representation
- Depth sampling
- 3D pixel deprojection
"""

from abc import ABC, abstractmethod

from dataclasses import dataclass

from typing import (
    Optional,
    Tuple,
    Dict,
    Any
)

import numpy as np


@dataclass
class CameraFrame:
    """
    Represents one synchronized RGB-D frame.
    """

    # BGR image from OpenCV / RealSense
    color: np.ndarray

    # Depth image in meters
    depth: Optional[np.ndarray] = None

    # Depth conversion scale
    depth_scale: float = 0.001

    # Whether depth data exists
    is_depth_available: bool = False

    # Whether depth is aligned with color
    is_aligned: bool = False

    # Frame timestamp
    timestamp: float = 0.0

    # Camera name
    camera_name: str = "Generic"

    # True for synthetic demo frames
    is_demo: bool = False

    # Camera intrinsics
    intrinsics: Optional[
        Dict[str, Any]
    ] = None


class BaseCamera(ABC):
    """
    Abstract camera interface.
    """

    def __init__(self):

        self.is_running = False

    # ============================================================
    # ABSTRACT METHODS
    # ============================================================

    @abstractmethod
    def connect(self) -> bool:
        """
        Initialize and connect to camera.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Safely release camera resources.
        """
        pass

    @abstractmethod
    def read_frame(
        self
    ) -> Optional[CameraFrame]:
        """
        Capture one frame.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Return current connection state.
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Return diagnostic information.
        """
        pass

    # ============================================================
    # DEPTH MEDIAN
    # ============================================================

    def sample_depth_median(
        self,
        depth_map: Optional[np.ndarray],
        u: int,
        v: int,
        patch_size: int = 5
    ) -> float:

        if depth_map is None:
            return 0.0

        if depth_map.ndim != 2:
            return 0.0

        height, width = depth_map.shape

        if u < 0 or u >= width:
            return 0.0

        if v < 0 or v >= height:
            return 0.0

        radius = patch_size // 2

        u_min = max(
            0,
            u - radius
        )

        u_max = min(
            width,
            u + radius + 1
        )

        v_min = max(
            0,
            v - radius
        )

        v_max = min(
            height,
            v + radius + 1
        )

        patch = depth_map[
            v_min:v_max,
            u_min:u_max
        ]

        valid_depths = patch[
            (patch > 0.05)
            &
            (patch < 5.0)
            &
            np.isfinite(patch)
        ]

        if len(valid_depths) == 0:
            return 0.0

        return float(
            np.median(valid_depths)
        )

    # ============================================================
    # 3D DEPROJECTION
    # ============================================================

    def deproject_pixel_to_3d(
        self,
        u: int,
        v: int,
        depth_m: float,
        intrinsics: Optional[
            Dict[str, Any]
        ] = None
    ) -> Optional[
        Tuple[float, float, float]
    ]:

        if depth_m <= 0.0:
            return None

        # --------------------------------------------------------
        # Use real camera intrinsics when available
        # --------------------------------------------------------

        if intrinsics:

            required = [
                "fx",
                "fy",
                "ppx",
                "ppy"
            ]

            if all(
                key in intrinsics
                for key in required
            ):

                fx = intrinsics["fx"]
                fy = intrinsics["fy"]
                ppx = intrinsics["ppx"]
                ppy = intrinsics["ppy"]

                if fx == 0 or fy == 0:
                    return None

                x = (
                    (u - ppx)
                    * depth_m
                    / fx
                )

                y = (
                    (v - ppy)
                    * depth_m
                    / fy
                )

                z = depth_m

                return (
                    float(x),
                    float(y),
                    float(z)
                )

        # --------------------------------------------------------
        # Fallback pinhole model
        # --------------------------------------------------------

        fov_rad = np.radians(60.0)

        image_width = 640.0
        image_height = 480.0

        fx = (
            image_width
            /
            (
                2.0
                *
                np.tan(
                    fov_rad / 2.0
                )
            )
        )

        fy = fx

        ppx = image_width / 2.0
        ppy = image_height / 2.0

        x = (
            (u - ppx)
            * depth_m
            / fx
        )

        y = (
            (v - ppy)
            * depth_m
            / fy
        )

        z = depth_m

        return (
            float(x),
            float(y),
            float(z)
        )