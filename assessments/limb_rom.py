"""
Limb Range of Motion (ROM) assessment module for MOTION DX.
Tracks Knee, Elbow, and Shoulder kinematics with bilateral comparison.
"""

from typing import Dict, List, Optional, Tuple, Any
from measurements.angles import calculate_joint_angle
from measurements.rom import ROMTracker
from measurements.symmetry import calculate_symmetry_index
from vision.pose import PoseResult
from .base import AssessmentSession, AssessmentState

JOINT_LANDMARK_MAP = {
    "KNEE": {
        "LEFT": {"proximal": 23, "vertex": 25, "distal": 27, "name": "Left Knee (Hip-Knee-Ankle)"},
        "RIGHT": {"proximal": 24, "vertex": 26, "distal": 28, "name": "Right Knee (Hip-Knee-Ankle)"}
    },
    "ELBOW": {
        "LEFT": {"proximal": 11, "vertex": 13, "distal": 15, "name": "Left Elbow (Shoulder-Elbow-Wrist)"},
        "RIGHT": {"proximal": 12, "vertex": 14, "distal": 16, "name": "Right Elbow (Shoulder-Elbow-Wrist)"}
    },
    "SHOULDER": {
        "LEFT": {"proximal": 23, "vertex": 11, "distal": 13, "name": "Left Shoulder (Hip-Shoulder-Elbow)"},
        "RIGHT": {"proximal": 24, "vertex": 12, "distal": 14, "name": "Right Shoulder (Hip-Shoulder-Elbow)"}
    }
}


class LimbROMAssessment:
    """Orchestrates live Limb ROM tracking, real-time metrics, and bilateral analysis."""

    def __init__(self, joint: str = "KNEE", primary_side: str = "RIGHT"):
        self.joint = joint.upper()
        self.primary_side = primary_side.upper()
        self.primary_tracker = ROMTracker()
        self.contralateral_tracker = ROMTracker()
        self.active_session: Optional[AssessmentSession] = None

    def start_session(self, session: AssessmentSession) -> None:
        self.active_session = session
        self.active_session.state = AssessmentState.ACTIVE
        self.primary_tracker.start()
        self.contralateral_tracker.start()

    def stop_session(self) -> AssessmentSession:
        if not self.active_session:
            raise RuntimeError("No active assessment session to stop.")

        self.primary_tracker.stop()
        self.contralateral_tracker.stop()
        self.active_session.state = AssessmentState.COMPLETED
        self.active_session.end_time = self.primary_tracker.end_time

        # Compile final summary
        summary = self.primary_tracker.get_summary()
        contra_summary = self.contralateral_tracker.get_summary()

        # Compute bilateral symmetry if contralateral data was recorded
        symmetry_info = None
        if contra_summary["sample_count"] > 10 and summary["sample_count"] > 10:
            symmetry_info = calculate_symmetry_index(
                summary["observed_rom"], contra_summary["observed_rom"]
            )

        summary["contralateral"] = contra_summary
        summary["bilateral_symmetry"] = symmetry_info
        summary["primary_side"] = self.primary_side
        summary["joint"] = self.joint

        self.active_session.metrics_summary = summary
        self.active_session.time_series = list(self.primary_tracker.samples)
        self.active_session.tracking_quality = summary["tracking_quality"]

        return self.active_session

    def reset(self) -> None:
        self.primary_tracker.reset()
        self.contralateral_tracker.reset()
        if self.active_session:
            self.active_session.state = AssessmentState.READY

    def update(self, pose_result: PoseResult) -> Dict[str, Any]:
        """
        Process incoming pose result, calculate active angles, and update tracking state.
        
        Returns real-time telemetry dictionary for UI gauge rendering.
        """
        if not pose_result.detected:
            return {
                "detected": False,
                "current_angle": self.primary_tracker.current_angle,
                "min_angle": self.primary_tracker.min_angle if self.primary_tracker.min_angle < 900 else 0.0,
                "max_angle": self.primary_tracker.max_angle if self.primary_tracker.max_angle > -900 else 0.0,
                "observed_rom": self.primary_tracker.observed_rom,
                "tracking_quality": pose_result.tracking_quality,
                "guidance_message": pose_result.guidance_message,
                "mode": "2D",
                "highlight_joints": []
            }

        joint_cfg = JOINT_LANDMARK_MAP.get(self.joint, JOINT_LANDMARK_MAP["KNEE"])
        p_cfg = joint_cfg.get(self.primary_side, joint_cfg["RIGHT"])
        c_side = "LEFT" if self.primary_side == "RIGHT" else "RIGHT"
        c_cfg = joint_cfg.get(c_side, joint_cfg["LEFT"])

        lm = pose_result.landmarks
        highlight_joints = [p_cfg["proximal"], p_cfg["vertex"], p_cfg["distal"]]

        # Calculate primary joint angle
        primary_angle, p_mode = 0.0, "2D"
        if (p_cfg["proximal"] in lm and p_cfg["vertex"] in lm and p_cfg["distal"] in lm):
            p1 = lm[p_cfg["proximal"]]
            p2 = lm[p_cfg["vertex"]]
            p3 = lm[p_cfg["distal"]]

            p1_2d = (p1.pixel_x, p1.pixel_y)
            p2_2d = (p2.pixel_x, p2.pixel_y)
            p3_2d = (p3.pixel_x, p3.pixel_y)

            primary_angle, p_mode = calculate_joint_angle(
                p1_2d, p2_2d, p3_2d,
                p1.point_3d, p2.point_3d, p3.point_3d,
                prefer_3d=True
            )

            self.primary_tracker.add_sample(
                primary_angle,
                tracking_quality=pose_result.tracking_quality,
                mode=p_mode,
                metadata={"side": self.primary_side, "joint": self.joint}
            )

        # Calculate contralateral joint angle if visible
        contra_angle = 0.0
        if (c_cfg["proximal"] in lm and c_cfg["vertex"] in lm and c_cfg["distal"] in lm):
            cp1 = lm[c_cfg["proximal"]]
            cp2 = lm[c_cfg["vertex"]]
            cp3 = lm[c_cfg["distal"]]
            contra_angle, c_mode = calculate_joint_angle(
                (cp1.pixel_x, cp1.pixel_y), (cp2.pixel_x, cp2.pixel_y), (cp3.pixel_x, cp3.pixel_y),
                cp1.point_3d, cp2.point_3d, cp3.point_3d,
                prefer_3d=True
            )
            self.contralateral_tracker.add_sample(
                contra_angle,
                tracking_quality=pose_result.tracking_quality,
                mode=c_mode,
                metadata={"side": c_side, "joint": self.joint}
            )

        min_val = self.primary_tracker.min_angle if self.primary_tracker.min_angle < 900.0 else 0.0
        max_val = self.primary_tracker.max_angle if self.primary_tracker.max_angle > -900.0 else 0.0

        return {
            "detected": True,
            "joint": self.joint,
            "side": self.primary_side,
            "current_angle": primary_angle,
            "contralateral_angle": contra_angle,
            "min_angle": min_val,
            "max_angle": max_val,
            "observed_rom": self.primary_tracker.observed_rom,
            "duration": self.primary_tracker.duration,
            "tracking_quality": pose_result.tracking_quality,
            "guidance_message": pose_result.guidance_message,
            "mode": p_mode,
            "highlight_joints": highlight_joints,
            "is_active": self.primary_tracker.is_active
        }
