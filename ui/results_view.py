"""
Clinical post-assessment results and clinician review screen for MOTION DX.
Displays observed kinematics, movement graphs, bilateral comparison, notes editor, and report exports.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any
from assessments.base import AssessmentSession
from reports.csv_report import generate_csv_telemetry, generate_summary_csv
from reports.pdf_report import generate_pdf_report
from storage.session_store import SessionStore
from ui.styles import get_quality_badge_html


def render_results_view(session: AssessmentSession, session_store: SessionStore) -> None:
    st.markdown("### Assessment Complete — Clinician Review")
    st.markdown(
        "<p style='color: #64748b; font-size: 13.5px; margin-top: -8px;'>"
        "Review observed kinematic metrics, inspect trajectory curves, append clinical notes, and export session reports."
        "</p>",
        unsafe_allow_html=True
    )

    # Prominent Non-diagnostic notice
    st.markdown(
        """
        <div class="clinical-safety-alert">
            <b>CLINICAL REVIEW NOTICE:</b> All values displayed below represent <b>OBSERVED KINEMATIC MEASUREMENTS</b>.
            Parameters are not evaluated as normal or abnormal unless a clinically validated reference has explicitly been configured.
        </div>
        """,
        unsafe_allow_html=True
    )

    summary = session.metrics_summary

    # Summary Cards Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">ASSESSED TARGET</div>
                <div style="font-size:20px; font-weight:700; color:#0f172a;">{session.target_joint}</div>
                <div class="metric-subtext">Side: {session.target_side}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        if "observed_rom" in summary:
            main_val = f"{summary.get('observed_rom', 0.0)}°"
            main_lbl = "OBSERVED ROM"
        elif "observed_peak_opening" in summary:
            main_val = f"{summary.get('observed_peak_opening', 0.0)} {summary.get('unit', '')}"
            main_lbl = "PEAK OPENING"
        else:
            main_val = f"{summary.get('left_knee_observed_rom', 0.0)}°"
            main_lbl = "LEFT KNEE ROM"

        st.markdown(
            f"""
            <div class="metric-card" style="background:#f0fdf4; border-color:#86efac;">
                <div class="metric-label" style="color:#166534;">{main_lbl}</div>
                <div class="metric-value-huge" style="color:#15803d; font-size:32px;">{main_val}</div>
                <div class="metric-subtext" style="color:#166534;">Observed Excursion</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        if "minimum_angle" in summary:
            ext_text = f"Min: {summary.get('minimum_angle')}° | Max: {summary.get('maximum_angle')}°"
        elif "max_left_lateral_excursion" in summary:
            ext_text = f"L: {summary.get('max_left_lateral_excursion')} | R: {summary.get('max_right_lateral_excursion')}"
        else:
            ext_text = f"Right Knee: {summary.get('right_knee_observed_rom', 0.0)}°"

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">OBSERVED EXTREMA</div>
                <div style="font-size:16px; font-weight:600; color:#1e293b; margin-top:6px;">{ext_text}</div>
                <div class="metric-subtext">Peak kinematic limits</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">TRACKING INTEGRITY</div>
                <div style="margin-top:6px;">{get_quality_badge_html(session.tracking_quality)}</div>
                <div class="metric-subtext">Duration: {session.duration_seconds}s</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Kinematic Movement Graph (Angle / Movement vs. Time)
    st.markdown("#### Kinematic Movement Trajectory")
    if session.time_series and len(session.time_series) > 1:
        df = pd.DataFrame(session.time_series)
        if "angle" in df.columns:
            chart_df = df[["elapsed_sec", "angle"]].set_index("elapsed_sec")
            st.line_chart(chart_df, height=280)
        elif "opening" in df.columns:
            chart_df = df[["timestamp", "opening"]].reset_index()
            st.line_chart(chart_df["opening"], height=280)
        elif "left_knee_angle" in df.columns:
            chart_df = df[["elapsed_sec", "left_knee_angle", "right_knee_angle"]].set_index("elapsed_sec")
            st.line_chart(chart_df, height=280)
    else:
        st.info("No time-series samples recorded during session.")

    # Bilateral Symmetry Comparison Section
    if "bilateral_symmetry" in summary and summary["bilateral_symmetry"]:
        sym = summary["bilateral_symmetry"]
        st.markdown("#### Bilateral Symmetry Analysis")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Primary ROM", f"{sym.get('left_value')}°")
        with col_s2:
            st.metric("Contralateral ROM", f"{sym.get('right_value')}°")
        with col_s3:
            st.metric("Symmetry Index", f"{sym.get('symmetry_index_pct')}%", delta=f"{sym.get('difference')}° difference")

    # Clinician Review & Notes Section
    st.markdown("#### Clinician Notes & Observations")
    notes = st.text_area(
        "Enter clinical remarks or patient presentation details:",
        value=session.clinical_notes,
        placeholder="e.g., Patient presented with mild left knee hesitation during terminal extension. Movement was pain-free.",
        height=100
    )
    if notes != session.clinical_notes:
        session.clinical_notes = notes

    col_save, col_csv, col_pdf = st.columns([1.2, 1, 1])

    with col_save:
        if st.button("💾 Save Session Locally", type="primary", use_container_width=True):
            saved_path = session_store.save_session(session)
            st.success(f"Session stored locally at: {saved_path}")

    # Export Buttons
    session_dict = {
        "session_id": session.session_id,
        "patient_id": session.patient_id,
        "assessment_type": session.assessment_type,
        "target_joint": session.target_joint,
        "camera_type": session.camera_type,
        "is_depth_enabled": session.is_depth_enabled,
        "start_time": session.start_time,
        "duration_seconds": session.duration_seconds,
        "tracking_quality": session.tracking_quality,
        "metrics_summary": session.metrics_summary,
        "time_series": session.time_series,
        "clinical_notes": session.clinical_notes,
        "disclaimer": session.disclaimer
    }

    with col_csv:
        csv_data = generate_csv_telemetry(session_dict)
        st.download_button(
            label="📥 Export CSV Telemetry",
            data=csv_data,
            file_name=f"{session.session_id}_telemetry.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_pdf:
        try:
            pdf_bytes = generate_pdf_report(session_dict)
            st.download_button(
                label="📄 Generate PDF Report",
                data=pdf_bytes,
                file_name=f"{session.session_id}_clinical_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error compiling PDF: {e}")
