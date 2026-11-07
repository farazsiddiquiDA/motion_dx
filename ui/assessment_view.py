
"""
MOTION DX - Live Clinical Assessment View
Browser Webcam + WebRTC Edition

Features:
- Browser webcam through WebRTC
- STUN server configuration
- Limb ROM assessment
- TMD face tracking
- Basic gait tracking
- Start / Stop / Reset
- TMD step navigation
- Existing MediaPipe processing pipeline
"""

import time
import queue
from typing import Dict, Any, Optional

import streamlit as st
import cv2

from streamlit_webrtc import webrtc_streamer, WebRtcMode

from camera.base import BaseCamera, CameraFrame

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
        st.session_state.limb_controller = LimbROMAssessment(
            joint=session.target_joint,
            primary_side=session.target_side,
        )

    if "tmd_controller" not in st.session_state:
        st.session_state.tmd_controller = TMDAssessment()

    if "gait_controller" not in st.session_state:
        st.session_state.gait_controller = BasicGaitAssessment()


# ================================================================
# WEBRTC CONFIGURATION
# ================================================================

RTC_CONFIGURATION = {
    "iceServers": [
        {
            "urls": [
                "stun:stun.l.google.com:19302"
            ]
        }
    ]
}


# ================================================================
# BROWSER CAMERA
# ================================================================

def _create_browser_camera():

    return webrtc_streamer(
        key="motion-dx-browser-camera",

        mode=WebRtcMode.SENDONLY,

        rtc_configuration=RTC_CONFIGURATION,

        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 30},
            },
            "audio": False,
        },

        async_processing=True,
    )


# ================================================================
# FRAME PROCESSING
# ================================================================

def _process_frame(
    frame_bgr,
    pose_tracker,
    face_tracker,
    session,
):
    """
    Process one browser webcam frame.

    Browser frames are BGR and are adapted to the
    existing CameraFrame interface.

    No depth information is available.
    """

    camera_frame = CameraFrame(
        color=frame_bgr,
        depth=None,
        depth_scale=0.001,
        is_depth_available=False,
        is_aligned=False,
        timestamp=time.time(),
        camera_name="Browser Webcam",
        is_demo=False,
        intrinsics=None,
    )

    # ------------------------------------------------------------
    # TMD
    # ------------------------------------------------------------

    if session.assessment_type == "TMD":

        controller = st.session_state.tmd_controller

        face_result = face_tracker.process_frame(
            camera_frame,
            camera_backend=None,
        )

        telemetry = controller.update(face_result)

        current_step = controller.get_current_step()

        rendered = render_tmd_overlay(
            camera_frame.color,
            face_result,
            current_step_name=current_step["name"],
        )

        rgb = cv2.cvtColor(
            rendered,
            cv2.COLOR_BGR2RGB,
        )

        return rgb, telemetry, current_step

    # ------------------------------------------------------------
    # GAIT
    # ------------------------------------------------------------

    if session.assessment_type == "GAIT":

        controller = st.session_state.gait_controller

        pose_result = pose_tracker.process_frame(
            camera_frame,
            camera_backend=None,
        )

        telemetry = controller.update(pose_result)

        rendered = render_pose_skeleton(
            camera_frame.color,
            pose_result,
            highlight_joints=telemetry.get(
                "highlight_joints"
            ),
        )

        rgb = cv2.cvtColor(
            rendered,
            cv2.COLOR_BGR2RGB,
        )

        return rgb, telemetry, None

    # ------------------------------------------------------------
    # LIMB ROM
    # ------------------------------------------------------------

    controller = st.session_state.limb_controller

    pose_result = pose_tracker.process_frame(
        camera_frame,
        camera_backend=None,
    )

    telemetry = controller.update(pose_result)

    rendered = render_pose_skeleton(
        camera_frame.color,
        pose_result,
        highlight_joints=telemetry.get(
            "highlight_joints"
        ),
    )

    rgb = cv2.cvtColor(
        rendered,
        cv2.COLOR_BGR2RGB,
    )

    return rgb, telemetry, None


# ================================================================
# TELEMETRY RENDERING
# ================================================================

def _render_tmd_telemetry_native(
    telemetry: Dict[str, Any],
    current_step: Dict[str, Any],
):

    quality = telemetry.get("tracking_quality", "POOR")

    opening = telemetry.get("current_opening", 0.0)
    peak = telemetry.get("peak_opening", 0.0)

    max_left = telemetry.get("max_left_excursion", 0.0)
    max_right = telemetry.get("max_right_excursion", 0.0)

    symmetry = telemetry.get("lateral_symmetry_index", 100.0)

    unit = telemetry.get("unit", "px")

    st.subheader("Jaw / TMD Kinematics")

    st.caption(
        f"Step {current_step['step_id']} — "
        f"{current_step['name']}"
    )

    _render_tracking_quality(quality)

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Current Aperture", f"{opening} {unit}")

    with c2:
        st.metric("Peak Opening", f"{peak} {unit}")

    st.markdown("### Lateral Excursion")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Maximum Left", f"{max_left} {unit}")

    with c2:
        st.metric("Maximum Right", f"{max_right} {unit}")

    st.metric("Lateral Symmetry", f"{symmetry}%")


def _render_limb_telemetry_native(
    telemetry: Dict[str, Any],
    session: AssessmentSession,
):

    quality = telemetry.get("tracking_quality", "POOR")

    current = telemetry.get("current_angle", 0.0)
    minimum = telemetry.get("min_angle", 0.0)
    maximum = telemetry.get("max_angle", 0.0)
    rom = telemetry.get("observed_rom", 0.0)

    contra = telemetry.get("contralateral_angle", 0.0)
    mode = telemetry.get("mode", "2D")

    st.subheader(
        f"{session.target_joint} Range of Motion"
    )

    st.caption(
        f"Mode: {mode} | Side: {session.target_side}"
    )

    _render_tracking_quality(quality)

    st.metric(
        "Current Joint Angle",
        f"{current}°",
    )

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Minimum", f"{minimum}°")

    with c2:
        st.metric("Maximum", f"{maximum}°")

    st.metric(
        "Observed Active ROM",
        f"{rom}°",
    )

    if contra > 0:
        st.metric("Contralateral Angle", f"{contra}°")
    else:
        st.caption("Contralateral: Standby")


def _render_gait_telemetry_native(
    telemetry: Dict[str, Any],
):

    quality = telemetry.get("tracking_quality", "POOR")

    left_knee = telemetry.get("left_knee_angle", 0.0)
    right_knee = telemetry.get("right_knee_angle", 0.0)

    steps = telemetry.get("step_count_estimate", 0)
    duration = telemetry.get("session_duration", 0.0)

    st.subheader("Basic Gait Observation")

    st.caption("Descriptive lower-limb kinematics")

    _render_tracking_quality(quality)

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Left Knee", f"{left_knee}°")

    with c2:
        st.metric("Right Knee", f"{right_knee}°")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Step Cycles", steps)

    with c2:
        st.metric("Duration", f"{duration:.1f}s")


def _render_tracking_quality(quality):

    quality = str(quality).upper()

    if quality == "GOOD":
        st.success(f"Tracking: {quality}")

    elif quality == "FAIR":
        st.warning(f"Tracking: {quality}")

    else:
        st.error(f"Tracking: {quality}")


# ================================================================
# BROWSER FRAME CONSUMER
# ================================================================

@st.fragment(run_every=0.15)
def _live_browser_assessment(
    ctx,
    pose_tracker,
    face_tracker,
    session,
    video_placeholder,
    telemetry_placeholder,
):

    if not ctx.state.playing:

        video_placeholder.info(
            "Browser camera is not connected. "
            "Start the webcam and allow camera permission."
        )

        return

    if ctx.video_receiver is None:

        video_placeholder.info(
            "Waiting for incoming browser video frames..."
        )

        return

    try:

        frames = ctx.video_receiver.get_frames(
            timeout=0.1
        )

    except queue.Empty:
        return

    except Exception as exc:

        video_placeholder.warning(
            f"Video receiver error: {exc}"
        )

        return

    if not frames:
        return

    # Use the latest available frame.
    latest_frame = frames[-1]

    frame_bgr = latest_frame.to_ndarray(
        format="bgr24"
    )

    if frame_bgr is None:
        return

    try:

        rgb, telemetry, current_step = _process_frame(
            frame_bgr,
            pose_tracker,
            face_tracker,
            session,
        )

    except Exception as exc:

        video_placeholder.error(
            f"Frame processing error: {exc}"
        )

        return

    video_placeholder.image(
        rgb,
        caption="MOTION DX — Browser Webcam / Live Assessment",
        use_container_width=True,
    )

    with telemetry_placeholder.container():

        if session.assessment_type == "TMD":

            _render_tmd_telemetry_native(
                telemetry,
                current_step,
            )

        elif session.assessment_type == "GAIT":

            _render_gait_telemetry_native(
                telemetry
            )

        else:

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

    _initialize_controllers(session)

    # ============================================================
    # HEADER
    # ============================================================

    st.subheader("Clinical Assessment")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write("**Patient ID**")
        st.write(session.patient_id)

    with col2:
        st.write("**Assessment**")
        st.write(
            session.assessment_type.replace("_", " ")
        )

    with col3:
        st.write("**Target**")
        st.write(session.target_joint)

    with col4:
        st.write("**Status**")

        if session.state == AssessmentState.ACTIVE:
            st.success("ACTIVE")
        else:
            st.info(session.state.value)

    st.divider()

    # ============================================================
    # CONTROLS
    # ============================================================

    col_start, col_stop, col_reset = st.columns(3)

    with col_start:

        start_btn = st.button(
            "▶ START",
            type="primary",
            use_container_width=True,
            disabled=session.state == AssessmentState.ACTIVE,
            key="assessment_start",
        )

    with col_stop:

        stop_btn = st.button(
            "■ STOP",
            use_container_width=True,
            disabled=session.state != AssessmentState.ACTIVE,
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

        session.state = AssessmentState.ACTIVE

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

            st.session_state.tmd_controller = TMDAssessment()

        elif session.assessment_type == "GAIT":

            st.session_state.gait_controller = BasicGaitAssessment()

        else:

            st.session_state.limb_controller = LimbROMAssessment(
                joint=session.target_joint,
                primary_side=session.target_side,
            )

        session.state = AssessmentState.READY

        st.rerun()

    # ============================================================
    # STOP
    # ============================================================

    if stop_btn:

        if session.assessment_type == "TMD":

            completed = (
                st.session_state.tmd_controller.stop_session()
            )

        elif session.assessment_type == "GAIT":

            completed = (
                st.session_state.gait_controller.stop_session()
            )

        else:

            completed = (
                st.session_state.limb_controller.stop_session()
            )

        session.state = AssessmentState.COMPLETED

        return completed

    # ============================================================
    # WAITING STATE
    # ============================================================

    if session.state != AssessmentState.ACTIVE:

        st.info(
            "Press START to begin the live assessment."
        )

        return None

    # ============================================================
    # BROWSER CAMERA
    # ============================================================

    st.markdown("### Browser Webcam")

    st.caption(
        "Allow webcam permission when your browser requests it. "
        "Keep the required body region visible."
    )

    ctx = _create_browser_camera()

    video_col, telemetry_col = st.columns([1.55, 1.0])

    with video_col:
        video_placeholder = st.empty()

    with telemetry_col:
        telemetry_placeholder = st.empty()

    # ============================================================
    # TMD STEP NAVIGATION
    # ============================================================

    if session.assessment_type == "TMD":

        st.divider()

        controller = st.session_state.tmd_controller

        current_step = controller.get_current_step()

        st.markdown(
            f"### Step {current_step['step_id']}: "
            f"{current_step['name']}"
        )

        st.caption(current_step["instruction"])

        prev_col, next_col = st.columns(2)

        with prev_col:

            if st.button(
                "◀ Previous Step",
                disabled=controller.current_step_idx <= 0,
                use_container_width=True,
                key="previous_tmd_step",
            ):

                controller.previous_step()

                st.rerun()

        with next_col:

            if st.button(
                "Next Step ▶",
                disabled=controller.current_step_idx >= len(TMD_STEPS) - 1,
                use_container_width=True,
                key="next_tmd_step",
            ):

                controller.next_step()

                st.rerun()

    # ============================================================
    # LIVE PROCESSING
    # ============================================================

    _live_browser_assessment(
        ctx,
        pose_tracker,
        face_tracker,
        session,
        video_placeholder,
        telemetry_placeholder,
    )

    return None
