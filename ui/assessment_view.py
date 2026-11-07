
"""
MOTION DX - Live Clinical Assessment View
Browser Webcam Edition
Uses Streamlit WebRTC + existing MediaPipe trackers.
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
# BROWSER VIDEO PROCESSING
# ================================================================

@st.fragment(run_every=0.15)
def _process_browser_frames(
    ctx,
    pose_tracker,
    face_tracker,
    session,
    video_placeholder,
    telemetry_placeholder,
):
    """
    Reads browser webcam frames through WebRTC.

    Processing happens in the Streamlit script thread,
    not inside the WebRTC video callback.
    """

    if not ctx.state.playing:

        video_placeholder.info(
            "Waiting for browser webcam connection. "
            "Allow camera permission in your browser."
        )
        return

    if ctx.video_receiver is None:

        video_placeholder.info(
            "Connecting to browser webcam..."
        )
        return

    try:
        frames = ctx.video_receiver.get_frames(timeout=0.1)

    except queue.Empty:
        return

    except Exception as exc:
        video_placeholder.warning(
            f"Waiting for video frames: {exc}"
        )
        return

    if not frames:
        return

    # Process the newest available frame.
    video_frame = frames[-1]

    frame_bgr = video_frame.to_ndarray(
        format="bgr24"
    )

    if frame_bgr is None:
        return

    # Convert browser frame to existing MOTION DX format.
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

    # ============================================================
    # TMD
    # ============================================================

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

        video_placeholder.image(
            rgb,
            caption="MOTION DX — Browser Webcam / TMD",
            use_container_width=True,
        )

        with telemetry_placeholder.container():
            _render_tmd_telemetry_native(
                telemetry,
                current_step,
            )

    # ============================================================
    # GAIT
    # ============================================================

    elif session.assessment_type == "GAIT":

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

        video_placeholder.image(
            rgb,
            caption="MOTION DX — Browser Webcam / Gait",
            use_container_width=True,
        )

        with telemetry_placeholder.container():
            _render_gait_telemetry_native(telemetry)

    # ============================================================
    # LIMB ROM
    # ============================================================

    else:

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

        video_placeholder.image(
            rgb,
            caption="MOTION DX — Browser Webcam / Live Kinematics",
            use_container_width=True,
        )

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

    _initialize_controllers(session)

    st.subheader("Clinical Assessment")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write("**Patient ID**")
        st.write(session.patient_id)

    with col2:
        st.write("**Assessment**")
        st.write(session.assessment_type.replace("_", " "))

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
            st.session_state.tmd_controller.start_session(session)

        elif session.assessment_type == "GAIT":
            st.session_state.gait_controller.start_session(session)

        else:
            st.session_state.limb_controller.start_session(session)

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
            completed = st.session_state.tmd_controller.stop_session()

        elif session.assessment_type == "GAIT":
            completed = st.session_state.gait_controller.stop_session()

        else:
            completed = st.session_state.limb_controller.stop_session()

        session.state = AssessmentState.COMPLETED

        return completed

    # ============================================================
    # CAMERA INTERFACE
    # ============================================================

    if session.state != AssessmentState.ACTIVE:

        st.info(
            "Press START to begin the assessment."
        )

        return None

    st.markdown("### Browser Camera")

    st.caption(
        "Allow camera access when your browser requests permission. "
        "Position your body clearly inside the camera frame."
    )

    ctx = webrtc_streamer(
        key="motion-dx-browser-camera",
        mode=WebRtcMode.SENDONLY,
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
    # PROCESS VIDEO
    # ============================================================

    _process_browser_frames(
        ctx,
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

    quality = telemetry.get("tracking_quality", "POOR")

    opening = telemetry.get("current_opening", 0.0)
    peak = telemetry.get("peak_opening", 0.0)
    max_left = telemetry.get("max_left_excursion", 0.0)
    max_right = telemetry.get("max_right_excursion", 0.0)
    symmetry = telemetry.get("lateral_symmetry_index", 100.0)
    unit = telemetry.get("unit", "px")

    st.subheader("Jaw / TMD Kinematics")

    st.caption(
        f"Step {current_step['step_id']} — {current_step['name']}"
    )

    if quality.upper() == "GOOD":
        st.success(f"Tracking: {quality}")
    elif quality.upper() == "FAIR":
        st.warning(f"Tracking: {quality}")
    else:
        st.error(f"Tracking: {quality}")

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


# ================================================================
# LIMB TELEMETRY
# ================================================================

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

    st.subheader(f"{session.target_joint} Range of Motion")

    st.caption(
        f"Mode: {mode} | Side: {session.target_side}"
    )

    if quality.upper() == "GOOD":
        st.success(f"Tracking: {quality}")
    elif quality.upper() == "FAIR":
        st.warning(f"Tracking: {quality}")
    else:
        st.error(f"Tracking: {quality}")

    st.metric("Current Joint Angle", f"{current}°")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Minimum", f"{minimum}°")

    with c2:
        st.metric("Maximum", f"{maximum}°")

    st.metric("Observed Active ROM", f"{rom}°")

    if contra > 0:
        st.metric("Contralateral Angle", f"{contra}°")
    else:
        st.caption("Contralateral: Standby")


# ================================================================
# GAIT TELEMETRY
# ================================================================

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

    if quality.upper() == "GOOD":
        st.success(f"Tracking: {quality}")
    elif quality.upper() == "FAIR":
        st.warning(f"Tracking: {quality}")
    else:
        st.error(f"Tracking: {quality}")

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
