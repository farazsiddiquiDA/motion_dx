"""
Synthetic Demo Camera generator for zero-hardware evaluation.
Generates biomechanically patterned kinematic animations with synthetic depth.
Strictly stamped as DEMO MODE to prevent misinterpretation as real patient data.
"""

import time
import math
from typing import Optional, Dict, Any
import cv2
import numpy as np
from .base import BaseCamera, CameraFrame


class DemoCamera(BaseCamera):
    """
    Simulated kinematic camera generator for presentation and offline testing.
    Outputs synthetic RGB and aligned Depth frames with a visible watermark.
    """

    def __init__(self, width: int = 640, height: int = 480, scenario: str = "knee"):
        super().__init__()
        self.width = width
        self.height = height
        self.scenario = scenario  # 'knee', 'tmd', 'gait', or 'general'
        self.start_t = time.time()
        self.device_name = "Synthetic Demo Generator"

    def connect(self) -> bool:
        self.start_t = time.time()
        self.is_running = True
        return True

    def disconnect(self) -> None:
        self.is_running = False

    def read_frame(self) -> Optional[CameraFrame]:
        if not self.is_running:
            return None

        t = time.time() - self.start_t
        # Create clinical dark background canvas
        frame = np.full((self.height, self.width, 3), 28, dtype=np.uint8)

        # Draw medical coordinate grid
        for x in range(0, self.width, 60):
            cv2.line(frame, (x, 0), (x, self.height), (38, 42, 48), 1)
        for y in range(0, self.height, 60):
            cv2.line(frame, (0, y), (self.width, y), (38, 42, 48), 1)

        # Generate realistic kinematic motion silhouette based on scenario
        if self.scenario == "tmd":
            self._draw_synthetic_face(frame, t)
        elif self.scenario == "gait":
            self._draw_synthetic_gait(frame, t)
        else:
            self._draw_synthetic_leg(frame, t)

        # Prominent non-fake medical disclaimer banner
        cv2.rectangle(frame, (10, 10), (self.width - 10, 48), (20, 20, 30), -1)
        cv2.rectangle(frame, (10, 10), (self.width - 10, 48), (0, 165, 255), 2)
        cv2.putText(
            frame,
            "DEMO MODE - SYNTHETIC DATA (NO PHYSICAL SENSOR REQUIRED)",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
            cv2.LINE_AA
        )

        # Synthesize realistic depth map in meters (subject at ~0.8m, background at ~2.0m)
        depth_map = np.full((self.height, self.width), 2.2, dtype=np.float32)
        # Find non-background pixels and set depth to 0.75m + slight sinusoidal variation
        mask = (frame[:, :, 0] > 40) | (frame[:, :, 1] > 40) | (frame[:, :, 2] > 40)
        depth_map[mask] = 0.75 + 0.05 * math.sin(t)

        return CameraFrame(
            color=frame,
            depth=depth_map,
            depth_scale=0.001,
            is_depth_available=True,
            is_aligned=True,
            timestamp=time.time(),
            camera_name=self.device_name,
            is_demo=True,
            intrinsics={"fx": 580.0, "fy": 580.0, "ppx": 320.0, "ppy": 240.0, "width": 640, "height": 480}
        )

    def _draw_synthetic_leg(self, frame: np.ndarray, t: float) -> None:
        """Draw an anatomically proportioned leg undergoing cyclic flexion/extension."""
        # Hip anchor
        hip_x, hip_y = 260, 120
        # Knee angle oscillates smoothly between ~35 deg and ~150 deg
        cycle = 0.5 * (math.sin(t * 1.8) + 1.0)  # [0, 1]
        knee_angle_deg = 40.0 + cycle * 105.0  # 40 deg to 145 deg
        knee_angle_rad = math.radians(knee_angle_deg)

        # Thigh vector (pointing downward-right)
        thigh_len = 130
        thigh_angle = math.radians(75.0)
        knee_x = int(hip_x + thigh_len * math.cos(thigh_angle))
        knee_y = int(hip_y + thigh_len * math.sin(thigh_angle))

        # Shank vector based on knee flexion
        shank_len = 130
        shank_angle = thigh_angle + math.pi - knee_angle_rad
        ankle_x = int(knee_x + shank_len * math.cos(shank_angle))
        ankle_y = int(knee_y + shank_len * math.sin(shank_angle))

        # Render limbs with soft anatomical contours
        cv2.line(frame, (hip_x, hip_y), (knee_x, knee_y), (140, 140, 150), 28)
        cv2.line(frame, (knee_x, knee_y), (ankle_x, ankle_y), (140, 140, 150), 22)

        # Trunk indicator
        cv2.line(frame, (hip_x, hip_y), (hip_x - 20, hip_y - 80), (120, 120, 130), 32)
        cv2.circle(frame, (hip_x - 30, hip_y - 110), 22, (130, 130, 140), -1)

    def _draw_synthetic_face(self, frame: np.ndarray, t: float) -> None:
        """Draw a synthetic facial profile with jaw opening and lateral excursion."""
        cx, cy = self.width // 2, 220
        # Head contour
        cv2.ellipse(frame, (cx, cy), (90, 120), 0, 0, 360, (140, 140, 150), -1)
        # Eyes
        cv2.circle(frame, (cx - 35, cy - 30), 10, (50, 50, 60), -1)
        cv2.circle(frame, (cx + 35, cy - 30), 10, (50, 50, 60), -1)
        # Nose
        cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (100, 100, 110), 3)

        # Jaw aperture cycle: 0 to 30px opening, with slight lateral translation
        open_amt = int(18 * (0.5 * math.sin(t * 2.0) + 0.5))
        lat_amt = int(12 * math.sin(t * 1.2))

        # Upper lip / philtrum
        cv2.ellipse(frame, (cx, cy + 40), (28, 6), 0, 0, 180, (80, 80, 90), -1)
        # Lower lip (moving with jaw)
        cv2.ellipse(frame, (cx + lat_amt, cy + 48 + open_amt), (26, 7), 0, 0, 180, (70, 70, 80), -1)
        # Chin
        cv2.circle(frame, (cx + lat_amt, cy + 85 + open_amt), 18, (130, 130, 140), -1)

    def _draw_synthetic_gait(self, frame: np.ndarray, t: float) -> None:
        """Draw synthetic bilateral lower limb walking motion."""
        mid_x, hip_y = self.width // 2, 140
        left_t = t * 2.2
        right_t = left_t + math.pi

        for side, phase_t, color in [("left", left_t, (160, 160, 170)), ("right", right_t, (120, 120, 130))]:
            hip_x = mid_x - 30 if side == "left" else mid_x + 30
            thigh_ang = math.radians(90.0 + 25.0 * math.sin(phase_t))
            knee_x = int(hip_x + 120 * math.cos(thigh_ang))
            knee_y = int(hip_y + 120 * math.sin(thigh_ang))

            shank_flex = max(0.0, math.sin(phase_t - 0.5)) * 40.0
            shank_ang = thigh_ang + math.radians(shank_flex)
            ankle_x = int(knee_x + 110 * math.cos(shank_ang))
            ankle_y = int(knee_y + 110 * math.sin(shank_ang))

            cv2.line(frame, (hip_x, hip_y), (knee_x, knee_y), color, 16)
            cv2.line(frame, (knee_x, knee_y), (ankle_x, ankle_y), color, 14)

    def is_connected(self) -> bool:
        return self.is_running

    def get_status(self) -> Dict[str, Any]:
        return {
            "type": "Synthetic Kinematic Generator (Demo Mode)",
            "device_name": self.device_name,
            "serial_number": "DEMO-SYNTH-001",
            "connected": self.is_running,
            "depth_available": True,
            "rgb_available": True,
            "aligned": True,
            "tracking_ready": True
        }
