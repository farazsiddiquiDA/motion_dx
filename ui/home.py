"""
Modern clinical home dashboard for MOTION DX.
Displays system overview, module selection cards, camera status, previous sessions, and privacy notice.
"""

import streamlit as st
import time
from typing import Dict, Any, List
from camera.base import BaseCamera
from storage.session_store import SessionStore


def render_home_view(camera: BaseCamera, session_store: SessionStore) -> None:
    # Medical branding header
    st.markdown(
        """
        <div class="clinical-brand-header">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h1 class="clinical-brand-title">MOTION DX</h1>
                    <div class="clinical-brand-subtitle">Objective movement measurement using computer vision.</div>
                </div>
                <div style="text-align:right;">
                    <span style="background:rgba(255,255,255,0.15); color:#ffffff; padding:4px 10px; border-radius:6px; font-size:12px; font-weight:600;">
                        CLINICAL PROTOTYPE v1.0
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Privacy & Safety Banner
    st.markdown(
        """
        <div class="clinical-safety-alert">
            🔒 <b>LOCAL PRIVACY GUARANTEE:</b> Video and depth data are processed locally on this device.
            No frames, landmark geometries, or biometric recordings are transmitted to external cloud servers.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Module Selection Section
    st.markdown("### Clinical Assessment Modules")
    col_mod1, col_mod2, col_mod3 = st.columns(3)

    with col_mod1:
        st.markdown(
            """
            <div class="module-card">
                <div class="module-icon">🦵</div>
                <div class="module-title">Limb Range of Motion</div>
                <div class="module-desc">
                    Real-time angular excursion and dynamic ROM tracking for Knee, Elbow, and Shoulder with bilateral comparison.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Start Limb ROM", key="btn_start_limb", type="primary", use_container_width=True):
            st.session_state.active_assessment_type = "LIMB_ROM"
            st.session_state.active_target_joint = "KNEE"
            st.session_state.current_page = "NEW_ASSESSMENT"
            st.rerun()

    with col_mod2:
        st.markdown(
            """
            <div class="module-card">
                <div class="module-icon">🗣️</div>
                <div class="module-title">TMD / Jaw Movement</div>
                <div class="module-desc">
                    5-step guided protocol tracking maximum opening aperture, closing return, and midline lateral excursion.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Start TMD / Jaw", key="btn_start_tmd", use_container_width=True):
            st.session_state.active_assessment_type = "TMD"
            st.session_state.active_target_joint = "JAW"
            st.session_state.current_page = "NEW_ASSESSMENT"
            st.rerun()

    with col_mod3:
        st.markdown(
            """
            <div class="module-card">
                <div class="module-icon">🚶</div>
                <div class="module-title">Basic Gait Observation</div>
                <div class="module-desc">
                    Descriptive kinematic tracking of bilateral hip, knee, and ankle movement cycles and sagittal swing trajectory.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Start Gait Observation", key="btn_start_gait", use_container_width=True):
            st.session_state.active_assessment_type = "GAIT"
            st.session_state.active_target_joint = "LOWER_LIMBS"
            st.session_state.current_page = "NEW_ASSESSMENT"
            st.rerun()

    st.markdown("---")

    # Lower Split: Camera Health & Previous Sessions
    col_cam, col_sessions = st.columns([1, 1.4])

    with col_cam:
        st.markdown("#### Camera System Status")
        status = camera.get_status()
        is_conn = status.get("connected", False)
        depth_ok = status.get("depth_available", False)

        st.markdown(
            f"""
            <div class="metric-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="font-weight:600; color:#1e293b;">{status.get('device_name', 'Generic')}</span>
                    <span style="background:{'#ecfdf5' if is_conn else '#fef2f2'}; color:{'#065f46' if is_conn else '#991b1b'}; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:700;">
                        {'CONNECTED' if is_conn else 'DISCONNECTED'}
                    </span>
                </div>
                <div style="font-size:12.5px; color:#64748b; line-height:1.6;">
                    • Optical Depth: <b>{'Available (RGB-D)' if depth_ok else 'Unavailable (2D Fallback)'}</b><br/>
                    • RGB Video Stream: <b>{'Active' if status.get('rgb_available') else 'Standby'}</b><br/>
                    • MediaPipe Tasks Pipeline: <b>Ready</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("Run Full Camera Check"):
            st.session_state.current_page = "CAMERA_CHECK"
            st.rerun()

    with col_sessions:
        st.markdown("#### Previous Clinical Sessions")
        recent_sessions = session_store.list_sessions()[:5]

        if recent_sessions:
            for s in recent_sessions:
                s_id = s.get("session_id", "Unknown")
                p_id = s.get("patient_id", "Unknown")
                a_type = s.get("assessment_type", "").replace("_", " ")
                j_type = s.get("target_joint", "")
                dur = s.get("duration_seconds", 0)
                date_str = time.strftime("%b %d, %H:%M", time.localtime(s.get("start_time", time.time())))

                st.markdown(
                    f"""
                    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:8px 12px; margin-bottom:6px; font-size:13px; display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <b>{p_id}</b> ({j_type} {a_type})
                            <span style="color:#94a3b8; margin-left:8px;">{date_str}</span>
                        </div>
                        <div style="color:#64748b; font-size:12px;">
                            {dur}s | <b>{s.get('tracking_quality', 'GOOD')}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No prior sessions recorded yet in local storage.")
