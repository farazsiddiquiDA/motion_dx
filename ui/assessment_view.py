"""
MOTION DX - Live Clinical Assessment View

Stable version:
- Continuous camera feed
- No full-page blinking
- No raw HTML appearing on screen
- TMD face tracking
- Limb ROM tracking
- Basic gait tracking
- Start / Stop / Reset
- TMD step navigation
"""

from typing import Dict, Any, Optional

import streamlit as st
import cv2

from camera.base import BaseCamera

from vision.pose import PoseTracker
from vision.face import FaceTracker

from vision.renderer import (
    render_pose_skeleton,
    render_tmd_overlay,
)

from assessments.limb_rom import LimbROMAssessment
from assessments.tmd import TMDAssessment, TMD_STEPS
from assessments.gait import BasicGaitAssessment

from assessments.base import (
    AssessmentSession,
    AssessmentState,
)


# ================================================================
# SESSION CONTROLLERS
# ================================================================

def _initialize_controllers(session):

    if "limb_controller" not in st.session_state:

        st.session_state.limb_controller = (
            LimbROMAssessment(
                joint=session.target_joint,
                primary_side=session.target_side,
            )
        )

    if "tmd_controller" not in st.session_state:

        st.session_state.tmd_controller = (
            TMDAssessment()
        )

    if "gait_controller" not in st.session_state:

        st.session_state.gait_controller = (
            BasicGaitAssessment()
        )


# ================================================================
# LIVE CAMERA PROCESSING
# ================================================================

@st.fragment(run_every=0.15)
def _live_camera(
    camera: BaseCamera,
    pose_tracker: PoseTracker,
    face_tracker: FaceTracker,
    session: AssessmentSession,
    video_placeholder,
    telemetry_placeholder,
):
    """
    Only this part refreshes continuously.

    IMPORTANT:
    No Start/Stop/Reset buttons are inside this function.
    No Prev/Next buttons are inside this function.

    This prevents the entire UI from blinking.
    """

    # ------------------------------------------------------------
    # GET NEW FRAME
    # ------------------------------------------------------------

    frame = camera.read_frame()

    if frame is None or frame.color is None:

        video_placeholder.warning(
            "Waiting for camera frame..."
        )

        return

    # ============================================================
    # TMD
    # ============================================================

    if session.assessment_type == "TMD":

        tmd_ctrl = (
            st.session_state.tmd_controller
        )

        # --------------------------------------------------------
        # FACE TRACKING
        # --------------------------------------------------------

        face_res = face_tracker.process_frame(
            frame,
            camera_backend=camera,
        )

        # --------------------------------------------------------
        # UPDATE ASSESSMENT
        # --------------------------------------------------------

        telemetry = tmd_ctrl.update(
            face_res
        )

        # --------------------------------------------------------
        # CURRENT STEP
        # --------------------------------------------------------

        current_step = (
            tmd_ctrl.get_current_step()
        )

        # --------------------------------------------------------
        # RENDER FACE OVERLAY
        # --------------------------------------------------------

        rendered_img = render_tmd_overlay(
            frame.color,
            face_res,
            current_step_name=current_step["name"],
        )

        # --------------------------------------------------------
        # CONVERT BGR -> RGB
        # --------------------------------------------------------

        rgb_img = cv2.cvtColor(
            rendered_img,
            cv2.COLOR_BGR2RGB,
        )

        # --------------------------------------------------------
        # CAMERA STATUS
        # --------------------------------------------------------

        try:

            status = camera.get_status()

            device_name = status.get(
                "device_name",
                "Camera",
            )

        except Exception:

            device_name = "Camera"

        # --------------------------------------------------------
        # UPDATE ONLY IMAGE PLACEHOLDER
        # --------------------------------------------------------

        video_placeholder.image(
            rgb_img,
            caption=(
                f"MOTION DX — TMD / Jaw Kinematics "
                f"({device_name})"
            ),
            use_container_width=True,
        )

        # --------------------------------------------------------
        # TELEMETRY
        # --------------------------------------------------------

        with telemetry_placeholder.container():

            _render_tmd_telemetry_native(
                telemetry,
                current_step,
            )

    # ============================================================
    # GAIT
    # ============================================================

    elif session.assessment_type == "GAIT":

        gait_ctrl = (
            st.session_state.gait_controller
        )

        # --------------------------------------------------------
        # POSE
        # --------------------------------------------------------

        pose_res = pose_tracker.process_frame(
            frame,
            camera_backend=camera,
        )

        # --------------------------------------------------------
        # UPDATE GAIT
        # --------------------------------------------------------

        telemetry = gait_ctrl.update(
            pose_res
        )

        # --------------------------------------------------------
        # RENDER SKELETON
        # --------------------------------------------------------

        rendered_img = render_pose_skeleton(
            frame.color,
            pose_res,
            highlight_joints=telemetry.get(
                "highlight_joints"
            ),
        )

        rgb_img = cv2.cvtColor(
            rendered_img,
            cv2.COLOR_BGR2RGB,
        )

        # --------------------------------------------------------
        # DISPLAY
        # --------------------------------------------------------

        video_placeholder.image(
            rgb_img,
            caption="MOTION DX — Lower-Limb Gait Observation",
            use_container_width=True,
        )

        # --------------------------------------------------------
        # TELEMETRY
        # --------------------------------------------------------

        with telemetry_placeholder.container():

            _render_gait_telemetry_native(
                telemetry
            )

    # ============================================================
    # LIMB ROM
    # ============================================================

    else:

        limb_ctrl = (
            st.session_state.limb_controller
        )

        # --------------------------------------------------------
        # POSE
        # --------------------------------------------------------

        pose_res = pose_tracker.process_frame(
            frame,
            camera_backend=camera,
        )

        # --------------------------------------------------------
        # UPDATE ROM
        # --------------------------------------------------------

        telemetry = limb_ctrl.update(
            pose_res
        )

        # --------------------------------------------------------
        # RENDER SKELETON
        # --------------------------------------------------------

        rendered_img = render_pose_skeleton(
            frame.color,
            pose_res,
            highlight_joints=telemetry.get(
                "highlight_joints"
            ),
        )

        rgb_img = cv2.cvtColor(
            rendered_img,
            cv2.COLOR_BGR2RGB,
        )

        # --------------------------------------------------------
        # DISPLAY
        # --------------------------------------------------------

        video_placeholder.image(
            rgb_img,
            caption="MOTION DX — Live Kinematics",
            use_container_width=True,
        )

        # --------------------------------------------------------
        # TELEMETRY
        # --------------------------------------------------------

        with telemetry_placeholder.container():

            _render_limb_telemetry_native(
                telemetry,
                session,
            )


# ================================================================
# MAIN ASSESSMENT VIEW
# ================================================================

def render_live_assessment_view(
    camera: BaseCamera,
    pose_tracker: PoseTracker,
    face_tracker: FaceTracker,
    session: AssessmentSession,
) -> Optional[AssessmentSession]:

    # ============================================================
    # INITIALIZE
    # ============================================================

    _initialize_controllers(session)

    # ============================================================
    # HEADER
    # ============================================================

    st.subheader("Clinical Assessment")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.write("**Patient ID**")

        st.write(
            session.patient_id
        )

    with col2:

        st.write("**Assessment**")

        st.write(
            session.assessment_type.replace(
                "_",
                " ",
            )
        )

    with col3:

        st.write("**Target**")

        st.write(
            session.target_joint
        )

    with col4:

        st.write("**Status**")

        if session.state == AssessmentState.ACTIVE:

            st.success("ACTIVE")

        else:

            st.info(
                session.state.value
            )

    st.divider()

    # ============================================================
    # MAIN CONTROLS
    # ============================================================

    col_start, col_stop, col_reset = st.columns(3)

    with col_start:

        start_btn = st.button(
            "▶ START",
            type="primary",
            use_container_width=True,
            disabled=(
                session.state
                == AssessmentState.ACTIVE
            ),
            key="assessment_start",
        )

    with col_stop:

        stop_btn = st.button(
            "■ STOP",
            use_container_width=True,
            disabled=(
                session.state
                != AssessmentState.ACTIVE
            ),
            key="assessment_stop",
        )

    with col_reset:

        reset_btn = st.button(
            "↻ RESET",
            use_container_width=True,
            key="assessment_reset",
        )

    # ============================================================
    # START
    # ============================================================

    if start_btn:

        session.state = (
            AssessmentState.ACTIVE
        )

        if session.assessment_type == "TMD":

            st.session_state.tmd_controller.start_session(
                session
            )

        elif session.assessment_type == "GAIT":

            st.session_state.gait_controller.start_session(
                session
            )

        else:

            st.session_state.limb_controller.start_session(
                session
            )

        st.rerun()

    # ============================================================
    # RESET
    # ============================================================

    if reset_btn:

        if session.assessment_type == "TMD":

            st.session_state.tmd_controller = (
                TMDAssessment()
            )

        elif session.assessment_type == "GAIT":

            st.session_state.gait_controller = (
                BasicGaitAssessment()
            )

        else:

            st.session_state.limb_controller = (
                LimbROMAssessment(
                    joint=session.target_joint,
                    primary_side=session.target_side,
                )
            )

        session.state = (
            AssessmentState.READY
        )

        st.rerun()

    # ============================================================
    # STOP
    # ============================================================

    if stop_btn:

        completed_session = None

        if session.assessment_type == "TMD":

            completed_session = (
                st.session_state.tmd_controller.stop_session()
            )

        elif session.assessment_type == "GAIT":

            completed_session = (
                st.session_state.gait_controller.stop_session()
            )

        else:

            completed_session = (
                st.session_state.limb_controller.stop_session()
            )

        session.state = (
            AssessmentState.COMPLETED
        )

        return completed_session

    # ============================================================
    # VIDEO AREA
    # ============================================================

    if session.state != AssessmentState.ACTIVE:

        st.info(
            "Press ▶ START to begin the live assessment."
        )

        return None

    # ------------------------------------------------------------
    # CREATE PLACEHOLDERS OUTSIDE FRAGMENT
    # ------------------------------------------------------------

    video_col, telemetry_col = st.columns(
        [1.55, 1.0]
    )

    with video_col:

        video_placeholder = st.empty()

    with telemetry_col:

        telemetry_placeholder = st.empty()

    # ============================================================
    # TMD STEP CONTROLS
    #
    # IMPORTANT:
    # These are OUTSIDE the live fragment.
    # ============================================================

    if session.assessment_type == "TMD":

        st.divider()

        tmd_ctrl = (
            st.session_state.tmd_controller
        )

        current_step = (
            tmd_ctrl.get_current_step()
        )

        st.write(
            f"### Step {current_step['step_id']}: "
            f"{current_step['name']}"
        )

        st.caption(
            current_step["instruction"]
        )

        step_prev, step_next = st.columns(2)

        with step_prev:

            if st.button(
                "◀ Previous Step",
                disabled=(
                    tmd_ctrl.current_step_idx <= 0
                ),
                use_container_width=True,
                key="previous_tmd_step",
            ):

                tmd_ctrl.previous_step()

                st.rerun()

        with step_next:

            if st.button(
                "Next Step ▶",
                disabled=(
                    tmd_ctrl.current_step_idx
                    >= len(TMD_STEPS) - 1
                ),
                use_container_width=True,
                key="next_tmd_step",
            ):

                tmd_ctrl.next_step()

                st.rerun()

    # ============================================================
    # START LIVE FRAGMENT
    # ============================================================

    _live_camera(
        camera,
        pose_tracker,
        face_tracker,
        session,
        video_placeholder,
        telemetry_placeholder,
    )

    return None


# ================================================================
# TMD TELEMETRY
# ================================================================

def _render_tmd_telemetry_native(
    telemetry: Dict[str, Any],
    current_step: Dict[str, Any],
):

    quality = telemetry.get(
        "tracking_quality",
        "POOR",
    )

    opening = telemetry.get(
        "current_opening",
        0.0,
    )

    peak = telemetry.get(
        "peak_opening",
        0.0,
    )

    lateral = telemetry.get(
        "lateral_deviation",
        0.0,
    )

    max_left = telemetry.get(
        "max_left_excursion",
        0.0,
    )

    max_right = telemetry.get(
        "max_right_excursion",
        0.0,
    )

    symmetry = telemetry.get(
        "lateral_symmetry_index",
        100.0,
    )

    unit = telemetry.get(
        "unit",
        "px",
    )

    # ------------------------------------------------------------
    # HEADER
    # ------------------------------------------------------------

    st.subheader(
        "Jaw / TMD Kinematics"
    )

    st.caption(
        f"Step {current_step['step_id']} — "
        f"{current_step['name']}"
    )

    if quality.upper() == "GOOD":

        st.success(
            f"Tracking: {quality}"
        )

    elif quality.upper() == "FAIR":

        st.warning(
            f"Tracking: {quality}"
        )

    else:

        st.error(
            f"Tracking: {quality}"
        )

    # ------------------------------------------------------------
    # MAIN METRICS
    # ------------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Current Aperture",
            f"{opening} {unit}",
        )

    with c2:

        st.metric(
            "Peak Opening",
            f"{peak} {unit}",
        )

    # ------------------------------------------------------------
    # LATERAL
    # ------------------------------------------------------------

    st.markdown(
        "### Lateral Excursion"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Maximum Left",
            f"{max_left} {unit}",
        )

    with c2:

        st.metric(
            "Maximum Right",
            f"{max_right} {unit}",
        )

    st.metric(
        "Lateral Symmetry",
        f"{symmetry}%",
    )


# ================================================================
# LIMB TELEMETRY
# ================================================================

def _render_limb_telemetry_native(
    telemetry: Dict[str, Any],
    session: AssessmentSession,
):

    quality = telemetry.get(
        "tracking_quality",
        "POOR",
    )

    current = telemetry.get(
        "current_angle",
        0.0,
    )

    minimum = telemetry.get(
        "min_angle",
        0.0,
    )

    maximum = telemetry.get(
        "max_angle",
        0.0,
    )

    rom = telemetry.get(
        "observed_rom",
        0.0,
    )

    contra = telemetry.get(
        "contralateral_angle",
        0.0,
    )

    mode = telemetry.get(
        "mode",
        "2D",
    )

    st.subheader(
        f"{session.target_joint} Range of Motion"
    )

    st.caption(
        f"Mode: {mode} | "
        f"Side: {session.target_side}"
    )

    if quality.upper() == "GOOD":

        st.success(
            f"Tracking: {quality}"
        )

    elif quality.upper() == "FAIR":

        st.warning(
            f"Tracking: {quality}"
        )

    else:

        st.error(
            f"Tracking: {quality}"
        )

    # ------------------------------------------------------------
    # CURRENT
    # ------------------------------------------------------------

    st.metric(
        "Current Joint Angle",
        f"{current}°",
    )

    # ------------------------------------------------------------
    # MIN / MAX
    # ------------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Minimum",
            f"{minimum}°",
        )

    with c2:

        st.metric(
            "Maximum",
            f"{maximum}°",
        )

    # ------------------------------------------------------------
    # ROM
    # ------------------------------------------------------------

    st.metric(
        "Observed Active ROM",
        f"{rom}°",
    )

    # ------------------------------------------------------------
    # CONTRALATERAL
    # ------------------------------------------------------------

    if contra > 0:

        st.metric(
            "Contralateral Angle",
            f"{contra}°",
        )

    else:

        st.caption(
            "Contralateral: Standby"
        )


# ================================================================
# GAIT TELEMETRY
# ================================================================

def _render_gait_telemetry_native(
    telemetry: Dict[str, Any],
):

    quality = telemetry.get(
        "tracking_quality",
        "POOR",
    )

    left_knee = telemetry.get(
        "left_knee_angle",
        0.0,
    )

    right_knee = telemetry.get(
        "right_knee_angle",
        0.0,
    )

    steps = telemetry.get(
        "step_count_estimate",
        0,
    )

    duration = telemetry.get(
        "session_duration",
        0.0,
    )

    st.subheader(
        "Basic Gait Observation"
    )

    st.caption(
        "Descriptive lower-limb kinematics"
    )

    if quality.upper() == "GOOD":

        st.success(
            f"Tracking: {quality}"
        )

    elif quality.upper() == "FAIR":

        st.warning(
            f"Tracking: {quality}"
        )

    else:

        st.error(
            f"Tracking: {quality}"
        )

    # ------------------------------------------------------------
    # KNEE
    # ------------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Left Knee",
            f"{left_knee}°",
        )

    with c2:

        st.metric(
            "Right Knee",
            f"{right_knee}°",
        )

    # ------------------------------------------------------------
    # STEPS
    # ------------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Step Cycles",
            steps,
        )

    with c2:

        st.metric(
            "Duration",
            f"{duration:.1f}s",
        )