"""
Temporomandibular Joint (TMD) and Jaw Kinematics Assessment Module.

Features:
- 5-step guided TMD protocol
- Continuous jaw opening measurement
- Automatic peak opening detection
- Left/right lateral excursion tracking
- Midline trajectory tracking
- Works with 2D pixels or 3D depth measurements
- Strictly non-diagnostic
"""

from typing import Dict, List, Optional, Any

from vision.face import FaceResult
from .base import AssessmentSession, AssessmentState


TMD_STEPS = [
    {
        "step_id": 1,
        "name": "Neutral Baseline",
        "instruction": "Subject maintains relaxed neutral head posture and closed resting lips."
    },
    {
        "step_id": 2,
        "name": "Active Opening",
        "instruction": "Subject slowly opens mouth to comfortable maximum active range."
    },
    {
        "step_id": 3,
        "name": "Closing Return",
        "instruction": "Subject returns jaw smoothly to closed resting position."
    },
    {
        "step_id": 4,
        "name": "Left Lateral Excursion",
        "instruction": "Subject shifts lower jaw gently towards the left side."
    },
    {
        "step_id": 5,
        "name": "Right Lateral Excursion",
        "instruction": "Subject shifts lower jaw gently towards the right side."
    }
]


class TMDAssessment:
    """Manages the 5-step TMD/Jaw kinematic assessment session."""

    def __init__(self):
        self.current_step_idx: int = 0
        self.active_session: Optional[AssessmentSession] = None

        self.baseline_aperture: float = 0.0
        self.peak_opening: float = 0.0
        self.closing_residual: float = 0.0

        self.max_left_excursion: float = 0.0
        self.max_right_excursion: float = 0.0

        self.trajectory_samples: List[Dict[str, Any]] = []

        self.unit: str = "px"

    # ---------------------------------------------------------
    # SESSION
    # ---------------------------------------------------------

    def start_session(self, session: AssessmentSession) -> None:
        self.active_session = session
        self.active_session.state = AssessmentState.ACTIVE

        self.current_step_idx = 0

        self.baseline_aperture = 0.0
        self.peak_opening = 0.0
        self.closing_residual = 0.0

        self.max_left_excursion = 0.0
        self.max_right_excursion = 0.0

        self.trajectory_samples.clear()

        self.unit = "px"

    # ---------------------------------------------------------
    # STEP CONTROL
    # ---------------------------------------------------------

    def next_step(self) -> bool:
        """Move to the next protocol step."""

        if self.current_step_idx < len(TMD_STEPS) - 1:
            self.current_step_idx += 1
            return True

        return False

    def previous_step(self) -> bool:
        """Move to the previous protocol step."""

        if self.current_step_idx > 0:
            self.current_step_idx -= 1
            return True

        return False

    def get_current_step(self) -> Dict[str, Any]:
        return TMD_STEPS[self.current_step_idx]

    # ---------------------------------------------------------
    # FRAME UPDATE
    # ---------------------------------------------------------

    def update(self, face_result: FaceResult) -> Dict[str, Any]:

        if not face_result.detected or not face_result.metrics:

            return {
                "detected": False,
                "step": self.get_current_step(),
                "step_index": self.current_step_idx + 1,
                "total_steps": len(TMD_STEPS),
                "tracking_quality": face_result.tracking_quality,
                "guidance_message": face_result.guidance_message,
                "current_opening": 0.0,
                "peak_opening": self.peak_opening,
                "lateral_deviation": 0.0,
                "max_left_excursion": self.max_left_excursion,
                "max_right_excursion": self.max_right_excursion,
                "lateral_symmetry_index": self._calculate_lateral_symmetry(),
                "unit": self.unit
            }

        m = face_result.metrics

        self.unit = m.opening_unit

        step_id = TMD_STEPS[self.current_step_idx]["step_id"]

        # -----------------------------------------------------
        # BASELINE
        # -----------------------------------------------------

        if step_id == 1:

            if self.baseline_aperture == 0.0:
                self.baseline_aperture = m.opening_distance
            else:
                self.baseline_aperture = round(
                    0.9 * self.baseline_aperture
                    + 0.1 * m.opening_distance,
                    1
                )

        # -----------------------------------------------------
        # PEAK OPENING
        # -----------------------------------------------------
        #
        # IMPORTANT:
        # We now track peak opening continuously.
        #
        # Previously this only happened during Step 2.
        # That caused the UI to show 0 if the user recorded
        # movement without being exactly on Step 2.
        #

        if m.opening_distance > self.peak_opening:
            self.peak_opening = m.opening_distance

        # -----------------------------------------------------
        # CLOSING
        # -----------------------------------------------------

        if step_id == 3:

            self.closing_residual = m.opening_distance

        # -----------------------------------------------------
        # LEFT EXCURSION
        # -----------------------------------------------------

        if step_id == 4:

            if m.lateral_deviation < 0:

                excursion = abs(m.lateral_deviation)

                if excursion > self.max_left_excursion:
                    self.max_left_excursion = excursion

        # -----------------------------------------------------
        # RIGHT EXCURSION
        # -----------------------------------------------------

        if step_id == 5:

            if m.lateral_deviation > 0:

                excursion = m.lateral_deviation

                if excursion > self.max_right_excursion:
                    self.max_right_excursion = excursion

        # -----------------------------------------------------
        # TRAJECTORY
        # -----------------------------------------------------

        sample = {
            "step_id": step_id,
            "step_name": TMD_STEPS[self.current_step_idx]["name"],
            "opening": round(m.opening_distance, 2),
            "lateral_dev": round(m.lateral_deviation, 2),
            "chin_x": m.chin_point_2d[0],
            "chin_y": m.chin_point_2d[1],
            "symmetry_idx": m.commissure_symmetry_index,
            "timestamp": face_result.timestamp
        }

        self.trajectory_samples.append(sample)

        # -----------------------------------------------------
        # LATERAL SYMMETRY
        # -----------------------------------------------------

        lat_sym_idx = self._calculate_lateral_symmetry()

        # -----------------------------------------------------
        # RETURN TELEMETRY
        # -----------------------------------------------------

        return {
            "detected": True,

            "step": self.get_current_step(),
            "step_index": self.current_step_idx + 1,
            "total_steps": len(TMD_STEPS),

            "current_opening": round(m.opening_distance, 1),

            "peak_opening": round(self.peak_opening, 1),

            "baseline_aperture": round(self.baseline_aperture, 1),

            "closing_residual": round(self.closing_residual, 1),

            "lateral_deviation": round(m.lateral_deviation, 1),

            "max_left_excursion": round(self.max_left_excursion, 1),

            "max_right_excursion": round(self.max_right_excursion, 1),

            "lateral_symmetry_index": lat_sym_idx,

            "unit": self.unit,

            "commissure_symmetry": round(
                m.commissure_symmetry_index,
                1
            ),

            "tracking_quality": face_result.tracking_quality,

            "guidance_message": face_result.guidance_message
        }

    # ---------------------------------------------------------
    # LATERAL SYMMETRY
    # ---------------------------------------------------------

    def _calculate_lateral_symmetry(self) -> float:

        mean_excursion = (
            self.max_left_excursion
            + self.max_right_excursion
        ) / 2.0

        if mean_excursion <= 1e-4:
            return 100.0

        difference = abs(
            self.max_left_excursion
            - self.max_right_excursion
        )

        symmetry = 100.0 - (
            difference / mean_excursion
        ) * 100.0

        return round(
            max(0.0, min(100.0, symmetry)),
            1
        )

    # ---------------------------------------------------------
    # STOP SESSION
    # ---------------------------------------------------------

    def stop_session(self) -> AssessmentSession:

        if not self.active_session:
            raise RuntimeError("No active session.")

        self.active_session.state = AssessmentState.COMPLETED

        # Use actual current time instead of None
        import time
        self.active_session.end_time = time.time()

        lat_sym_idx = self._calculate_lateral_symmetry()

        summary = {

            "assessment_name":
                "TMD / Jaw Movement Kinematics",

            "unit":
                self.unit,

            "baseline_aperture":
                round(self.baseline_aperture, 1),

            "observed_peak_opening":
                round(self.peak_opening, 1),

            "closing_residual_aperture":
                round(self.closing_residual, 1),

            "max_left_lateral_excursion":
                round(self.max_left_excursion, 1),

            "max_right_lateral_excursion":
                round(self.max_right_excursion, 1),

            "observed_lateral_difference":
                round(
                    abs(
                        self.max_left_excursion
                        - self.max_right_excursion
                    ),
                    1
                ),

            "lateral_symmetry_index_pct":
                lat_sym_idx,

            "trajectory_points_count":
                len(self.trajectory_samples),

            "clinical_interpretation_note":
                "Observed movement parameters available for clinician review. "
                "Does not diagnose TMD."
        }

        self.active_session.metrics_summary = summary

        self.active_session.time_series = list(
            self.trajectory_samples
        )

        self.active_session.tracking_quality = (
            "GOOD"
            if len(self.trajectory_samples) > 20
            else "FAIR"
        )

        return self.active_session