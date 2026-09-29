"""
Basic Lower-Limb Gait Observation Module for MOTION DX.
Tracks bilateral Hip, Knee, and Ankle kinematics and sagittal swing trajectory.
Strictly non-diagnostic: provides objective kinematic curves for clinical review.
"""

import time
from typing import Dict, List, Optional, Any
import numpy as np

from measurements.angles import calculate_joint_angle
from measurements.symmetry import calculate_symmetry_index
from vision.pose import PoseResult
from .base import AssessmentSession, AssessmentState

LOWER_LIMB_LANDMARKS = [23, 24, 25, 26, 27, 28]


class BasicGaitAssessment:
    """Manages continuous bilateral lower-limb gait kinematic observations."""

    def __init__(self):
        self.active_session: Optional[AssessmentSession] = None
        self.left_knee_angles: List[float] = []
        self.right_knee_angles: List[float] = []
        self.trajectory_samples: List[Dict[str, Any]] = []
        self.start_time: float = 0.0
        self.step_event_count: int = 0
        self._last_extrema_time: float = 0.0

    def start_session(self, session: AssessmentSession) -> None:
        self.active_session = session
        self.active_session.state = AssessmentState.ACTIVE
        self.left_knee_angles.clear()
        self.right_knee_angles.clear()
        self.trajectory_samples.clear()
        self.start_time = time.time()
        self.step_event_count = 0
        self._last_extrema_time = self.start_time

    def update(self, pose_result: PoseResult) -> Dict[str, Any]:
        if not pose_result.detected:
            return {
                "detected": False,
                "tracking_quality": pose_result.tracking_quality,
                "guidance_message": pose_result.guidance_message,
                "left_knee_angle": 0.0,
                "right_knee_angle": 0.0,
                "session_duration": round(time.time() - self.start_time, 1) if self.start_time else 0.0
            }

        lm = pose_result.landmarks
        now = time.time()
        elapsed = round(now - self.start_time, 2)

        # 1. Left knee angle (Hip 23, Knee 25, Ankle 27)
        left_angle = 0.0
        if 23 in lm and 25 in lm and 27 in lm:
            left_angle, _ = calculate_joint_angle(
                (lm[23].pixel_x, lm[23].pixel_y),
                (lm[25].pixel_x, lm[25].pixel_y),
                (lm[27].pixel_x, lm[27].pixel_y),
                lm[23].point_3d, lm[25].point_3d, lm[27].point_3d
            )
            self.left_knee_angles.append(left_angle)

        # 2. Right knee angle (Hip 24, Knee 26, Ankle 28)
        right_angle = 0.0
        if 24 in lm and 26 in lm and 28 in lm:
            right_angle, _ = calculate_joint_angle(
                (lm[24].pixel_x, lm[24].pixel_y),
                (lm[26].pixel_x, lm[26].pixel_y),
                (lm[28].pixel_x, lm[28].pixel_y),
                lm[24].point_3d, lm[26].point_3d, lm[28].point_3d
            )
            self.right_knee_angles.append(right_angle)

        # Basic movement cycle observation: detect flexion oscillations
        if len(self.left_knee_angles) > 5 and (now - self._last_extrema_time > 0.4):
            # If left knee flexes below 120 and then extends above 150, estimate a step cycle
            recent = self.left_knee_angles[-5:]
            if max(recent) - min(recent) > 25.0:
                self.step_event_count += 1
                self._last_extrema_time = now

        # Record trajectory
        sample = {
            "elapsed_sec": elapsed,
            "left_knee_angle": left_angle,
            "right_knee_angle": right_angle,
            "left_ankle_y": lm[27].pixel_y if 27 in lm else 0,
            "right_ankle_y": lm[28].pixel_y if 28 in lm else 0,
            "tracking_quality": pose_result.tracking_quality
        }
        self.trajectory_samples.append(sample)

        # Real-time symmetry index between mean/latest knee angles
        sym_info = calculate_symmetry_index(left_angle, right_angle) if (left_angle > 0 and right_angle > 0) else None

        return {
            "detected": True,
            "left_knee_angle": left_angle,
            "right_knee_angle": right_angle,
            "step_count_estimate": self.step_event_count,
            "session_duration": elapsed,
            "symmetry_info": sym_info,
            "tracking_quality": pose_result.tracking_quality,
            "guidance_message": pose_result.guidance_message,
            "highlight_joints": LOWER_LIMB_LANDMARKS
        }

    def stop_session(self) -> AssessmentSession:
        if not self.active_session:
            raise RuntimeError("No active session.")

        self.active_session.state = AssessmentState.COMPLETED
        self.active_session.end_time = time.time()

        l_valid = [a for a in self.left_knee_angles if a > 10.0]
        r_valid = [a for a in self.right_knee_angles if a > 10.0]

        l_rom = round(max(l_valid) - min(l_valid), 1) if l_valid else 0.0
        r_rom = round(max(r_valid) - min(r_valid), 1) if r_valid else 0.0

        sym = calculate_symmetry_index(l_rom, r_rom)

        summary = {
            "assessment_name": "Basic Lower-Limb Gait Observation",
            "session_duration_sec": self.active_session.duration_seconds,
            "estimated_movement_cycles": self.step_event_count,
            "left_knee_observed_rom": l_rom,
            "right_knee_observed_rom": r_rom,
            "bilateral_knee_symmetry_pct": sym["symmetry_index_pct"],
            "bilateral_rom_difference": sym["absolute_difference"],
            "clinical_interpretation_note": "Kinematic observations for clinician review. Does not diagnose neurological or musculoskeletal gait disorders."
        }

        self.active_session.metrics_summary = summary
        self.active_session.time_series = list(self.trajectory_samples)
        self.active_session.tracking_quality = "GOOD" if len(self.trajectory_samples) > 20 else "FAIR"

        return self.active_session
