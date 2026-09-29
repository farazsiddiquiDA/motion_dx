"""
MediaPipe Pose Landmarker wrapper using the Tasks API.
Processes RGB frames and samples aligned depth to produce 2D and 3D joint landmarks.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from camera.base import CameraFrame

# Standard landmark indices for human biomechanics
LANDMARK_NAMES = {
    11: "LEFT_SHOULDER", 12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",    16: "RIGHT_WRIST",
    23: "LEFT_HIP",      24: "RIGHT_HIP",
    25: "LEFT_KNEE",     26: "RIGHT_KNEE",
    27: "LEFT_ANKLE",    28: "RIGHT_ANKLE"
}

# Anatomical kinetic segments for skeleton rendering
SKELETON_CONNECTIONS = [
    # Trunk
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left Arm
    (11, 13), (13, 15),
    # Right Arm
    (12, 14), (14, 16),
    # Left Leg
    (23, 25), (25, 27),
    # Right Leg
    (24, 26), (26, 28)
]


@dataclass
class JointLandmark:
    index: int
    name: str
    pixel_x: int
    pixel_y: int
    norm_x: float
    norm_y: float
    norm_z: float
    visibility: float
    presence: float
    depth_m: float = 0.0
    point_3d: Optional[Tuple[float, float, float]] = None


@dataclass
class PoseResult:
    detected: bool
    landmarks: Dict[int, JointLandmark] = field(default_factory=dict)
    tracking_quality: str = "POOR"  # 'GOOD', 'FAIR', 'POOR'
    guidance_message: str = ""
    timestamp: float = 0.0


class PoseTracker:
    """Encapsulates MediaPipe Tasks API PoseLandmarker."""

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "pose_landmarker.task")

        self.model_path = model_path
        self.detector: Optional[vision.PoseLandmarker] = None
        self._init_detector()

    def _init_detector(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def process_frame(self, frame: CameraFrame, camera_backend: Optional[Any] = None) -> PoseResult:
        """
        Process a single CameraFrame through PoseLandmarker and enrich with depth sampling.
        """
        if not self.detector or frame.color is None:
            return PoseResult(detected=False, tracking_quality="POOR", guidance_message="Camera frame unavailable.")

        h, w, _ = frame.color.shape
        # MediaPipe expects RGB format
        rgb_color = np.ascontiguousarray(frame.color[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_color)

        try:
            detection_result = self.detector.detect(mp_image)
        except Exception as e:
            return PoseResult(detected=False, tracking_quality="POOR", guidance_message=f"Detection error: {e}")

        if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
            return PoseResult(
                detected=False,
                tracking_quality="POOR",
                guidance_message="No person detected. Move into camera view."
            )

        pose_landmarks_list = detection_result.pose_landmarks[0]
        landmarks_dict: Dict[int, JointLandmark] = {}
        visibilities: List[float] = []

        for idx, lm in enumerate(pose_landmarks_list):
            px = int(np.clip(lm.x * w, 0, w - 1))
            py = int(np.clip(lm.y * h, 0, h - 1))
            name = LANDMARK_NAMES.get(idx, f"LM_{idx}")

            depth_val = 0.0
            point_3d = None

            # Robust depth sampling if depth map is available
            if frame.depth is not None and camera_backend is not None:
                depth_val = camera_backend.sample_depth_median(frame.depth, px, py, patch_size=7)
                if depth_val > 0.0:
                    point_3d = camera_backend.deproject_pixel_to_3d(px, py, depth_val, frame.intrinsics)

            visibility = float(getattr(lm, "visibility", 1.0))
            presence = float(getattr(lm, "presence", 1.0))
            visibilities.append(visibility)

            landmarks_dict[idx] = JointLandmark(
                index=idx,
                name=name,
                pixel_x=px,
                pixel_y=py,
                norm_x=float(lm.x),
                norm_y=float(lm.y),
                norm_z=float(lm.z),
                visibility=visibility,
                presence=presence,
                depth_m=depth_val,
                point_3d=point_3d
            )

        # Evaluate tracking quality
        mean_vis = float(np.mean(visibilities)) if visibilities else 0.0
        if mean_vis >= 0.75:
            quality = "GOOD"
            guidance = "Subject tracking optimal."
        elif mean_vis >= 0.50:
            quality = "FAIR"
            guidance = "Moderate tracking confidence. Ensure good lighting."
        else:
            quality = "POOR"
            guidance = "Please reposition the subject or step back into full camera view."

        return PoseResult(
            detected=True,
            landmarks=landmarks_dict,
            tracking_quality=quality,
            guidance_message=guidance,
            timestamp=time.time()
        )
