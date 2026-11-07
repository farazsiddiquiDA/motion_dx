"""
MOTION DX - Live Clinical Assessment View
Browser Webcam + WebRTC Edition (Cloud-optimized)

Optimizations:
- Requests 480x360 at 15 FPS instead of 640x480 at 30 FPS
- Processes the newest available frame only
- Limits Streamlit-side processing to roughly 6-7 updates/second
- Uses a short frame-receiver timeout
- Resizes unexpectedly large frames before MediaPipe processing
- Keeps the existing limb ROM, TMD, gait, and telemetry flow

Note:
This remains the browser-WebRTC edition. The `camera` argument is retained
for compatibility with the existing app.py signature, but this view does
not read from the local RealSense/Webcam camera object.
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
# PERFORMANCE SETTINGS
# ================================================================

# Lower resolution reduces the number of pixels MediaPipe must process.
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 360
CAMERA_FPS = 15

# Streamlit fragment refresh interval. This gives approximately 6-7
# processing opportunities per second, depending on server performance.
FRAGMENT_INTERVAL_SECONDS = 0.15

# Keep the receiver wait short so the UI does not block waiting for a frame.
FRAME_RECEIVER_TIMEOUT = 0.01


# ================================================================
# SESSION CONTROLLERS
# ================================================================

def _initialize_controllers(session: AssessmentSession) -> None:
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
        {"urls": ["stun:stun.l.google.com:19302"]}
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
                "width": {"ideal": CAMERA_WIDTH},
                "height": {"ideal": CAMERA_HEIGHT},
                "frameRate": {"ideal": CAMERA_FPS},
            },
            "audio": False,
        },
        async_processing=True,
    )


# ================================================================
# FRAME PROCESSING
# ================================================================

def _prepare_frame(frame_bgr):
    """Ensure frames are small enough for responsive cloud processing."""
    if frame_bgr is None:
        return None

    height, width = frame_bgr.shape[:2]

    if width > CAMERA_WIDTH or height > CAMERA_HEIGHT:
        frame_bgr = cv2.resize(
            frame_bgr,
            (CAMERA_WIDTH, CAMERA_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

    return frame_bgr


def _process_frame(
    frame_bgr,
    pose_tracker: PoseTracker,
    face_tracker: FaceTracker,
    session: AssessmentSession,
):
    """
    Process one browser webcam frame using the existing MOTION DX pipeline.

    Browser frames are BGR and are adapted to CameraFrame.
    Browser webcam mode does not provide RealSense depth data.
    """
    frame_bgr = _prepare_frame(frame_bgr)
    if frame_bgr is None:
        return None, {}, None

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

    # TMD: Face tracking only; do not run pose tracking unnecessarily.
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

        return (
            cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB),
            telemetry,
            current_step,
        )

    # Gait: pose tracking.
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
            highlight_joints=telemetry.get("highlight_joints"),
        )

        return (
            cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB),
            telemetry,
            None,
        )

    # Limb ROM: pose tracking.
    controller = st.session_state.limb_controller

    pose_result = pose_tracker.process_frame(
        camera_frame,
        camera_backend=None,
    )

    telemetry = controller.update(pose_result)

    rendered = render_pose_skeleton(
        camera_frame.color,
        pose_result,
        highlight_joints=telemetry.get("highlight_joints"),
    )

    return (
        cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB),
        telemetry,
        None,
    )


# ================================================================
# TELEMETRY RENDERING
# ================================================================

def _render_tracking_quality(quality):
    quality = str(quality).upper()

    if quality == "GOOD":
        st.success(f"Tracking: {quality}")
    elif quality == "FAIR":
        st.warning(f"Tracking: {quality}")
    else:
        st.error(f"Tracking: {quality}")


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

    st.subheader(f"{session.target_joint} Range of Motion")
    st.caption(f"Mode: {mode} | Side: {session.target_side}")
    _render_tracking_quality(quality)

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


def _render_gait_telemetry_native(telemetry: Dict[str, Any]):
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


# ================================================================
# BROWSER FRAME CONSUMER
# ================================================================

@st.fragment(run_every=FRAGMENT_INTERVAL_SECONDS)
def _live_browser_assessment(
    ctx,
    pose_tracker: PoseTracker,
    face_tracker: FaceTracker,
    session: AssessmentSession,
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
        # Fetch available frames quickly. The newest frame is preferred
        # so processing does not build up a backlog of old frames.
        frames = ctx.video_receiver.get_frames(
            timeout=FRAME_RECEIVER_TIMEOUT
        )
    except queue.Empty:
        return
    except Exception as exc:
        video_placeholder.warning(f"Video receiver error: {exc}")
        return

    if not frames:
        return

    latest_frame = frames[-1]
    frame_bgr = latest_frame.to_ndarray(format="bgr24")

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
        video_placeholder.error(f"Frame processing error: {exc}")
        return

    if rgb is None:
        return

    video_placeholder.image(
        rgb,
        caption="MOTION DX — Browser Webcam / Live Assessment",
        use_container_width=True,
    )

    with telemetry_placeholder.container():
        if session.assessment_type == "TMD":
            if current_step is not None:
                _render_tmd_telemetry_native(
                    telemetry,
                    current_step,
                )
        elif session.assessment_type == "GAIT":
            _render_gait_telemetry_native(telemetry)
        else:
            _render_limb_telemetry_native(telemetry, session)


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

    if start_btn:
        session.state = AssessmentState.ACTIVE

        if session.assessment_type == "TMD":
            st.session_state.tmd_controller.start_session(session)
        elif session.assessment_type == "GAIT":
            st.session_state.gait_controller.start_session(session)
        else:
            st.session_state.limb_controller.start_session(session)

        st.rerun()

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

    if stop_btn:
        if session.assessment_type == "TMD":
            completed = st.session_state.tmd_controller.stop_session()
        elif session.assessment_type == "GAIT":
            completed = st.session_state.gait_controller.stop_session()
        else:
            completed = st.session_state.limb_controller.stop_session()

        session.state = AssessmentState.COMPLETED
        return completed

    if session.state != AssessmentState.ACTIVE:
        st.info("Press START to begin the live assessment.")
        return None

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

    _live_browser_assessment(
        ctx,
        pose_tracker,
        face_tracker,
        session,
        video_placeholder,
        telemetry_placeholder,
    )

    return None
