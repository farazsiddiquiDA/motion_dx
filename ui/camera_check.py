"""
Camera check and subject positioning guide screen for MOTION DX.
Verifies hardware connectivity, depth stream alignment, and subject framing before assessment.
"""

import streamlit as st
import cv2
from camera.base import BaseCamera
from camera.realsense import RealSenseCamera
from camera.webcam import WebcamCamera
from vision.pose import PoseTracker
from vision.renderer import render_pose_skeleton


def render_camera_check_view(camera: BaseCamera, pose_tracker: PoseTracker) -> None:
    st.markdown("### Camera Diagnostics & Positioning Guide")
    st.markdown(
        "<p style='color: #64748b; font-size: 13.5px; margin-top: -8px;'>"
        "Verify camera alignment, optical depth availability, and subject positioning before initiating kinematic capture."
        "</p>",
        unsafe_allow_html=True
    )

    col_diag, col_guide = st.columns([1.1, 1.2])

    with col_diag:
        st.markdown("#### Hardware Diagnostics")
        status = camera.get_status()

        cam_conn = status.get("connected", False)
        depth_avail = status.get("depth_available", False)
        rgb_avail = status.get("rgb_available", False)
        tracking_ready = status.get("tracking_ready", False)

        def badge(ok: bool, yes_txt: str = "YES", no_txt: str = "NO"):
            color = "#065f46" if ok else "#991b1b"
            bg = "#ecfdf5" if ok else "#fef2f2"
            border = "#a7f3d0" if ok else "#fecaca"
            txt = yes_txt if ok else no_txt
            return f"<span style='background:{bg}; color:{color}; border:1px solid {border}; padding:3px 10px; border-radius:12px; font-weight:600; font-size:12px;'>{txt}</span>"

        diag_html = f"""
        <table style="width:100%; border-collapse: collapse; margin-top: 10px; font-size: 13.5px;">
            <tr style="border-bottom: 1px solid #e2e8f0; height: 38px;">
                <td style="color:#475569; font-weight:500;">Camera Connected</td>
                <td style="text-align:right;">{badge(cam_conn)}</td>
            </tr>
            <tr style="border-bottom: 1px solid #e2e8f0; height: 38px;">
                <td style="color:#475569; font-weight:500;">Depth Stream Available</td>
                <td style="text-align:right;">{badge(depth_avail)}</td>
            </tr>
            <tr style="border-bottom: 1px solid #e2e8f0; height: 38px;">
                <td style="color:#475569; font-weight:500;">RGB Stream Available</td>
                <td style="text-align:right;">{badge(rgb_avail)}</td>
            </tr>
            <tr style="border-bottom: 1px solid #e2e8f0; height: 38px;">
                <td style="color:#475569; font-weight:500;">Tracking Pipeline Status</td>
                <td style="text-align:right;">{badge(tracking_ready, 'READY', 'NOT READY')}</td>
            </tr>
            <tr style="height: 38px;">
                <td style="color:#475569; font-weight:500;">Active Hardware Driver</td>
                <td style="text-align:right; font-weight:600; color:#1e293b;">{status.get('device_name', 'Generic')}</td>
            </tr>
        </table>
        """
        st.markdown(diag_html, unsafe_allow_html=True)

        if not depth_avail:
            st.info(
                "Notice: 2D Projected mode active. Connect Intel RealSense D405 for aligned millimeter depth quantification."
            )

    with col_guide:
        st.markdown("#### Subject Positioning Checklist")
        st.markdown(
            """
            - **Lighting:** Ensure uniform frontal illumination without strong glare or backlighting.
            - **Framing:**
                - *Limb ROM:* Position subject sideways or front-facing ~1.2m away. Joint landmarks must remain fully in view throughout entire arc.
                - *TMD / Jaw:* Position subject facing camera ~0.5m-0.8m away. Lips and chin unobstructed.
                - *Gait:* Position subject ~2.0m-3.0m away for unobstructed view of both lower limbs.
            - **Clothing:** Form-fitting clothing improves anatomical landmark localization.
            - **Depth Safety:** For RealSense D405, maintain ideal distance (0.1m - 1.5m).
            """
        )

    st.markdown("---")
    st.markdown("#### Real-Time Hardware Sensor Preview")
    preview_box = st.empty()

    frame = camera.read_frame()
    if frame is not None and frame.color is not None:
        pose_res = pose_tracker.process_frame(frame, camera_backend=camera)
        rendered = render_pose_skeleton(frame.color, pose_res)
        # Convert BGR to RGB for Streamlit
        rgb_disp = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        preview_box.image(rgb_disp, caption=f"Live Feed — {camera.get_status().get('device_name')}", use_container_width=True)
    else:
        preview_box.warning("No camera frame received. Check sensor connection or switch to Demo Mode.")
