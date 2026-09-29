"""
MOTION DX — Main Clinical Application Entrypoint.
"Objective movement measurement using computer vision."

Clinical prototype orchestrator integrating Intel RealSense D405 RGB-D pipeline,
MediaPipe Tasks API, real-time biomechanics calculation, and local clinical reporting.
"""

import sys
import os
import streamlit as st

# Configure modern medical page settings
st.set_page_config(
    page_title="MOTION DX — Objective Movement Measurement",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from camera import get_camera, BaseCamera
from camera.realsense import RealSenseCamera
from camera.webcam import WebcamCamera
from vision.pose import PoseTracker
from vision.face import FaceTracker
from storage.session_store import SessionStore
from assessments.base import AssessmentSession, AssessmentState
from ui.styles import CLINICAL_CSS
from ui.home import render_home_view
from ui.camera_check import render_camera_check_view
from ui.assessment_view import render_live_assessment_view
from ui.results_view import render_results_view
from ui.validation_view import render_validation_view

# Inject clinical CSS
st.markdown(CLINICAL_CSS, unsafe_allow_html=True)


# --- Persistent Resource Caching ---
@st.cache_resource
def load_vision_trackers():
    """Load MediaPipe Tasks API models once into memory."""
    pose_tracker = PoseTracker()
    face_tracker = FaceTracker()
    return pose_tracker, face_tracker


@st.cache_resource
def get_session_store():
    return SessionStore()


# --- Session State Initialization ---
if "current_page" not in st.session_state:
    st.session_state.current_page = "HOME"

if "camera_preference" not in st.session_state:
    st.session_state.camera_preference = "auto"

if "active_session" not in st.session_state:
    st.session_state.active_session = None

if "completed_session" not in st.session_state:
    st.session_state.completed_session = None

if "active_assessment_type" not in st.session_state:
    st.session_state.active_assessment_type = "LIMB_ROM"

if "active_target_joint" not in st.session_state:
    st.session_state.active_target_joint = "KNEE"

if "active_target_side" not in st.session_state:
    st.session_state.active_target_side = "RIGHT"

# Initialize cached singletons
pose_tracker, face_tracker = load_vision_trackers()
session_store = get_session_store()


# ============================================================
# CAMERA HARDWARE MANAGEMENT
# ============================================================

scenario = "knee"

if st.session_state.active_assessment_type == "TMD":
    scenario = "tmd"

elif st.session_state.active_assessment_type == "GAIT":
    scenario = "gait"


# Camera must be recreated when either:
# 1. Camera preference changes
# 2. Assessment scenario changes

camera_key = (
    f"{st.session_state.camera_preference}"
    f"_{scenario}"
)


if (
    "camera_instance" not in st.session_state
    or st.session_state.get("last_camera_key")
    != camera_key
):

    # Disconnect old camera
    if (
        "camera_instance" in st.session_state
        and st.session_state.camera_instance
        is not None
    ):

        try:
            st.session_state.camera_instance.disconnect()

        except Exception:
            pass

    # Create new camera
    cam = get_camera(
        preferred=st.session_state.camera_preference,
        scenario=scenario
    )

    st.session_state.camera_instance = cam

    st.session_state.last_camera_key = camera_key


camera: BaseCamera = (
    st.session_state.camera_instance
)

# --- Sidebar Navigation & Clinical Controls ---
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 16px;">
            <div style="font-size: 20px; font-weight: 800; color: #0f172a; letter-spacing: -0.5px;">MOTION DX</div>
            <div style="font-size: 11px; color: #64748b; font-weight: 500;">Clinical Computer Vision Platform</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Navigation menu
    nav_options = [
        "Home Dashboard",
        "New Assessment",
        "Live Assessment",
        "Results & Review",
        "Camera Diagnostics",
        "Validation Mode",
        "Session Archives"
    ]
    page_map = {
        "Home Dashboard": "HOME",
        "New Assessment": "NEW_ASSESSMENT",
        "Live Assessment": "LIVE_ASSESSMENT",
        "Results & Review": "RESULTS",
        "Camera Diagnostics": "CAMERA_CHECK",
        "Validation Mode": "VALIDATION",
        "Session Archives": "ARCHIVES"
    }
    rev_page_map = {v: k for k, v in page_map.items()}

    current_nav_label = rev_page_map.get(st.session_state.current_page, "Home Dashboard")
    selected_nav = st.radio("Navigation", nav_options, index=nav_options.index(current_nav_label))

    if page_map[selected_nav] != st.session_state.current_page:
        st.session_state.current_page = page_map[selected_nav]
        st.rerun()

    st.markdown("---")

    # Camera hardware selector
    st.markdown("##### Hardware Configuration")
    cam_pref = st.selectbox(
        "Camera Input Driver",
        options=["auto", "realsense", "webcam", "demo"],
        format_func=lambda x: {
            "auto": "⚡ Auto-Detect (D405 > Webcam > Demo)",
            "realsense": "📷 Intel RealSense D405 (RGB-D)",
            "webcam": "💻 Fallback Webcam (2D)",
            "demo": "🎮 Demo Mode (Synthetic Feed)"
        }.get(x, x),
        index=["auto", "realsense", "webcam", "demo"].index(st.session_state.camera_preference)
    )
    if cam_pref != st.session_state.camera_preference:
        st.session_state.camera_preference = cam_pref
        st.rerun()

    cam_stat = camera.get_status()
    st.caption(f"Active: **{cam_stat.get('device_name')}**")
    if cam_stat.get("depth_available"):
        st.markdown("<span style='font-size:11px; color:#16a34a; font-weight:600;'>● 3D Spatial Depth Enabled</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='font-size:11px; color:#f59e0b; font-weight:600;'>▲ 2D Fallback Mode</span>", unsafe_allow_html=True)

    st.markdown("---")

    # Mandatory Clinical Safety Statement in Sidebar
    st.markdown(
        """
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:10px; font-size:11px; color:#64748b; line-height:1.4;">
            <b>CLINICAL SAFETY STATEMENT:</b><br/>
            MOTION DX converts visible human movement into objective kinematic measurements.
            The system does NOT diagnose diseases, prescribe therapies, or make diagnostic classifications.
        </div>
        """,
        unsafe_allow_html=True
    )


# --- Screen Routing ---

# 1. HOME SCREEN
if st.session_state.current_page == "HOME":
    render_home_view(camera=camera, session_store=session_store)


# 2. NEW ASSESSMENT SETUP
elif st.session_state.current_page == "NEW_ASSESSMENT":
    st.markdown("### Configure New Clinical Assessment")
    st.markdown(
        "<p style='color: #64748b; font-size: 13.5px; margin-top: -8px;'>"
        "Enter patient identification and designate the anatomical joint or kinematic protocol."
        "</p>",
        unsafe_allow_html=True
    )

    col_form, col_info = st.columns([1.2, 1])

    with col_form:
        patient_id = st.text_input("Patient / Subject ID", value="PT-2026-042", help="Local identifier; no data is uploaded.")

        ass_type = st.selectbox(
            "Assessment Type",
            options=["LIMB_ROM", "TMD", "GAIT"],
            format_func=lambda x: {
                "LIMB_ROM": "Limb Range of Motion (Knee, Elbow, Shoulder)",
                "TMD": "TMD / Jaw Movement Kinematics (5-Step Protocol)",
                "GAIT": "Basic Gait Observation (Lower-Limb Kinematics)"
            }.get(x, x),
            index=["LIMB_ROM", "TMD", "GAIT"].index(st.session_state.active_assessment_type)
        )
        st.session_state.active_assessment_type = ass_type

        target_joint = "KNEE"
        target_side = "RIGHT"

        if ass_type == "LIMB_ROM":
            target_joint = st.selectbox("Target Joint", ["KNEE", "ELBOW", "SHOULDER"])
            target_side = st.radio("Primary Side", ["RIGHT", "LEFT"], horizontal=True)
            st.session_state.active_target_joint = target_joint
            st.session_state.active_target_side = target_side
        elif ass_type == "TMD":
            target_joint = "JAW"
            target_side = "MIDLINE"
        elif ass_type == "GAIT":
            target_joint = "LOWER_LIMBS"
            target_side = "BILATERAL"

        st.markdown("<br/>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Check Camera Setup", use_container_width=True):
                st.session_state.current_page = "CAMERA_CHECK"
                st.rerun()
        with col_btn2:
            if st.button("Proceed to Live Assessment ▶", type="primary", use_container_width=True):
                new_session = AssessmentSession(
                    patient_id=patient_id,
                    assessment_type=ass_type,
                    target_joint=target_joint,
                    target_side=target_side,
                    camera_type=camera.get_status().get("device_name", "Generic"),
                    is_depth_enabled=camera.get_status().get("depth_available", False),
                    state=AssessmentState.READY
                )
                st.session_state.active_session = new_session
                st.session_state.current_page = "LIVE_ASSESSMENT"
                st.rerun()

    with col_info:
        st.markdown("#### Clinical Protocol Guidelines")
        if ass_type == "LIMB_ROM":
            st.markdown(
                """
                - **Knee ROM:** Evaluates active flexion and extension along the Hip-Knee-Ankle sagittal plane.
                - **Elbow ROM:** Evaluates flexion/extension along the Shoulder-Elbow-Wrist kinetic chain.
                - **Shoulder:** Quantifies abduction / elevation relative to the lateral trunk axis.
                - **Contralateral Feed:** Bilateral angles are recorded concurrently when both limbs remain in frame.
                """
            )
        elif ass_type == "TMD":
            st.markdown(
                """
                - **Guided 5-Step Protocol:**
                  1. Neutral Baseline calibration
                  2. Active maximum incisal opening
                  3. Closing return to rest
                  4. Left lateral excursion
                  5. Right lateral excursion
                - Measures vertical aperture and millimeter/pixel deviation from facial midline.
                """
            )
        else:
            st.markdown(
                """
                - **Basic Gait Observation:**
                  - Observational sagittal kinematics of bilateral lower extremities.
                  - Records knee flexion arcs and step cycle frequency.
                  - Strictly descriptive. Does not diagnose movement disorders.
                """
            )


# 3. CAMERA CHECK
elif st.session_state.current_page == "CAMERA_CHECK":
    render_camera_check_view(camera=camera, pose_tracker=pose_tracker)
    st.markdown("<br/>", unsafe_allow_html=True)
    if st.button("Return to Setup"):
        st.session_state.current_page = "NEW_ASSESSMENT"
        st.rerun()


# 4. LIVE ASSESSMENT
elif st.session_state.current_page == "LIVE_ASSESSMENT":
    if not st.session_state.active_session:
        st.session_state.active_session = AssessmentSession(
            patient_id="PT-2026-042",
            assessment_type=st.session_state.active_assessment_type,
            target_joint=st.session_state.active_target_joint,
            target_side=st.session_state.active_target_side,
            camera_type=camera.get_status().get("device_name", "Generic"),
            is_depth_enabled=camera.get_status().get("depth_available", False),
            state=AssessmentState.READY
        )

    completed = render_live_assessment_view(
        camera=camera,
        pose_tracker=pose_tracker,
        face_tracker=face_tracker,
        session=st.session_state.active_session
    )

    if completed:
        st.session_state.completed_session = completed
        st.session_state.current_page = "RESULTS"
        st.rerun()


# 5. RESULTS & REVIEW
elif st.session_state.current_page == "RESULTS":
    if st.session_state.completed_session:
        render_results_view(session=st.session_state.completed_session, session_store=session_store)
        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("← Start Another Assessment"):
            st.session_state.current_page = "NEW_ASSESSMENT"
            st.rerun()
    else:
        st.info("No completed assessment in active buffer. Select an assessment or load from archives.")
        if st.button("Configure Assessment"):
            st.session_state.current_page = "NEW_ASSESSMENT"
            st.rerun()


# 6. VALIDATION MODE
elif st.session_state.current_page == "VALIDATION":
    render_validation_view()


# 7. SESSION ARCHIVES
elif st.session_state.current_page == "ARCHIVES":
    st.markdown("### Local Session Archives")
    st.markdown(
        "<p style='color: #64748b; font-size: 13.5px; margin-top: -8px;'>"
        "Locally stored patient kinematic recordings. 100% offline JSON persistence."
        "</p>",
        unsafe_allow_html=True
    )

    sessions = session_store.list_sessions()
    if sessions:
        for s in sessions:
            s_id = s.get("session_id")
            with st.expander(f"📁 {s.get('patient_id')} — {s.get('target_joint')} {s.get('assessment_type')} ({s_id})"):
                full_s = session_store.get_session(s_id)
                if full_s:
                    st.json(full_s.get("metrics_summary", {}))
                    c_del, c_rev = st.columns([1, 1])
                    with c_rev:
                        if st.button(f"Load into Review Screen", key=f"load_{s_id}"):
                            # Rehydrate session object
                            rehydrated = AssessmentSession(
                                session_id=full_s["session_id"],
                                patient_id=full_s["patient_id"],
                                assessment_type=full_s["assessment_type"],
                                target_joint=full_s["target_joint"],
                                target_side=full_s["target_side"],
                                camera_type=full_s["camera_type"],
                                is_depth_enabled=full_s["is_depth_enabled"],
                                start_time=full_s["start_time"],
                                end_time=full_s.get("end_time"),
                                state=AssessmentState.COMPLETED,
                                tracking_quality=full_s.get("tracking_quality", "GOOD"),
                                metrics_summary=full_s.get("metrics_summary", {}),
                                time_series=full_s.get("time_series", []),
                                clinical_notes=full_s.get("clinical_notes", "")
                            )
                            st.session_state.completed_session = rehydrated
                            st.session_state.current_page = "RESULTS"
                            st.rerun()
                    with c_del:
                        if st.button(f"Delete Record", key=f"del_{s_id}"):
                            session_store.delete_session(s_id)
                            st.rerun()
    else:
        st.info("No archived sessions found in local data store.")
