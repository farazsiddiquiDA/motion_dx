"""
MediaPipe Face Landmarker Tasks API wrapper for TMD and jaw kinematics.
Computes vertical opening aperture, midline lateral deviation, and bilateral commissure symmetry.
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
from measurements.distance import calculate_distance_2d, calculate_distance_3d, point_to_line_distance_2d

# Key anatomical landmarks for temporomandibular kinematics
FACE_LANDMARK_MAP = {
    "GLABELLA": 10,       # Superior midline anchor
    "NASION": 168,        # Midline nasal bridge
    "SUBNASALE": 1,       # Inferior nose / philtrum top
    "UPPER_LIP": 13,      # Superior stomion / inner upper lip
    "LOWER_LIP": 14,      # Inferior stomion / inner lower lip
    "COMMISSURE_L": 291,  # Left mouth corner
    "COMMISSURE_R": 61,   # Right mouth corner
    "MENTON": 152         # Chin vertex / mandibular base
}


@dataclass
class TMDMetrics:
    opening_distance: float  # mm if 3D depth valid, or normalized pixels
    opening_unit: str        # 'mm (3D Depth)' or 'px (2D Projected)'
    lateral_deviation: float # signed deviation from midline (+ right, - left)
    abs_lateral_deviation: float
    commissure_left_dist: float
    commissure_right_dist: float
    commissure_symmetry_index: float
    chin_point_2d: Tuple[int, int]
    upper_lip_2d: Tuple[int, int]
    lower_lip_2d: Tuple[int, int]
    midline_p1: Tuple[int, int]
    midline_p2: Tuple[int, int]


@dataclass
class FaceResult:
    detected: bool
    metrics: Optional[TMDMetrics] = None
    tracking_quality: str = "POOR"
    guidance_message: str = ""
    timestamp: float = 0.0


class FaceTracker:
    """Encapsulates MediaPipe Tasks API FaceLandmarker for jaw kinematics."""

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "face_landmarker.task")

        self.model_path = model_path
        self.detector: Optional[vision.FaceLandmarker] = None
        self._init_detector()

    def _init_detector(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame: CameraFrame, camera_backend: Optional[Any] = None) -> FaceResult:
        if not self.detector or frame.color is None:
            return FaceResult(detected=False, tracking_quality="POOR", guidance_message="Camera frame unavailable.")

        h, w, _ = frame.color.shape
        rgb_color = np.ascontiguousarray(frame.color[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_color)

        try:
            result = self.detector.detect(mp_image)
        except Exception as e:
            return FaceResult(detected=False, tracking_quality="POOR", guidance_message=f"Detection error: {e}")

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return FaceResult(
                detected=False,
                tracking_quality="POOR",
                guidance_message="No face detected. Face the camera directly."
            )

        landmarks = result.face_landmarks[0]

        def get_pixel(idx: int) -> Tuple[int, int]:
            lm = landmarks[idx]
            return (int(np.clip(lm.x * w, 0, w - 1)), int(np.clip(lm.y * h, 0, h - 1)))

        def get_3d(idx: int) -> Optional[Tuple[float, float, float]]:
            px, py = get_pixel(idx)
            if frame.depth is not None and camera_backend is not None:
                d = camera_backend.sample_depth_median(frame.depth, px, py, patch_size=5)
                if d > 0.0:
                    return camera_backend.deproject_pixel_to_3d(px, py, d, frame.intrinsics)
            return None

        # Coordinates
        p_glabella = get_pixel(FACE_LANDMARK_MAP["GLABELLA"])
        p_subnasale = get_pixel(FACE_LANDMARK_MAP["SUBNASALE"])
        p_upper_lip = get_pixel(FACE_LANDMARK_MAP["UPPER_LIP"])
        p_lower_lip = get_pixel(FACE_LANDMARK_MAP["LOWER_LIP"])
        p_comm_l = get_pixel(FACE_LANDMARK_MAP["COMMISSURE_L"])
        p_comm_r = get_pixel(FACE_LANDMARK_MAP["COMMISSURE_R"])
        p_menton = get_pixel(FACE_LANDMARK_MAP["MENTON"])

        # Try 3D metric measurement for opening aperture
        pt_upper_3d = get_3d(FACE_LANDMARK_MAP["UPPER_LIP"])
        pt_lower_3d = get_3d(FACE_LANDMARK_MAP["LOWER_LIP"])

        if pt_upper_3d and pt_lower_3d:
            # 3D distance in meters * 1000 = millimeters
            opening_dist = calculate_distance_3d(pt_upper_3d, pt_lower_3d) * 1000.0
            opening_unit = "mm (3D Depth)"
        else:
            # 2D Euclidean distance in pixels
            opening_dist = calculate_distance_2d(p_upper_lip, p_lower_lip)
            opening_unit = "px (2D Projected)"

        # Facial midline defined by vector Glabella -> Subnasale
        _, signed_dev = point_to_line_distance_2d(p_menton, p_glabella, p_subnasale)
        abs_dev = abs(signed_dev)

        # Bilateral commissure symmetry to midline
        dist_l, _ = point_to_line_distance_2d(p_comm_l, p_glabella, p_subnasale)
        dist_r, _ = point_to_line_distance_2d(p_comm_r, p_glabella, p_subnasale)
        mean_comm = (dist_l + dist_r) / 2.0
        if mean_comm > 1e-4:
            sym_idx = max(0.0, min(100.0, 100.0 - (abs(dist_l - dist_r) / mean_comm) * 100.0))
        else:
            sym_idx = 100.0

        metrics = TMDMetrics(
            opening_distance=round(opening_dist, 1),
            opening_unit=opening_unit,
            lateral_deviation=round(signed_dev, 1),
            abs_lateral_deviation=round(abs_dev, 1),
            commissure_left_dist=round(dist_l, 1),
            commissure_right_dist=round(dist_r, 1),
            commissure_symmetry_index=round(sym_idx, 1),
            chin_point_2d=p_menton,
            upper_lip_2d=p_upper_lip,
            lower_lip_2d=p_lower_lip,
            midline_p1=p_glabella,
            midline_p2=p_subnasale
        )

        return FaceResult(
            detected=True,
            metrics=metrics,
            tracking_quality="GOOD",
            guidance_message="Facial tracking active.",
            timestamp=time.time()
        )
