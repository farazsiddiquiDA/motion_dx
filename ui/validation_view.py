"""
Validation Mode UI for MOTION DX.
Enables clinicians and researchers to calibrate and benchmark automated system measurements
against manual goniometer or ground-truth reference measurements.
Computes empirical Mean Absolute Error (MAE) and displays trial comparison tables.
"""

import streamlit as st
import pandas as pd
from validation.validation import ValidationSession


def render_validation_view() -> None:
    st.markdown("### Clinical Measurement Validation & Benchmarking")
    st.markdown(
        "<p style='color: #64748b; font-size: 13.5px; margin-top: -8px;'>"
        "Compare MOTION DX automated computer-vision measurements directly against manual clinical goniometer or motion-capture references."
        "</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="clinical-safety-alert">
            <b>VALIDATION PROTOCOL NOTICE:</b> This screen records empirical validation trials.
            Accuracy metrics such as <b>Mean Absolute Error (MAE)</b> and <b>RMSE</b> are calculated
            <b>strictly from entered experimental trials</b>. The system does not claim unvalidated accuracy percentages.
        </div>
        """,
        unsafe_allow_html=True
    )

    if "validation_session" not in st.session_state:
        st.session_state.validation_session = ValidationSession(assessment_type="KNEE_ROM")

    val_session: ValidationSession = st.session_state.validation_session

    # Input Form for New Trial
    with st.expander("➕ Record New Validation Trial", expanded=True):
        col_sys, col_ref, col_unit, col_add = st.columns([1.2, 1.2, 0.8, 1])

        with col_sys:
            sys_val = st.number_input("System Observed Value", value=84.2, step=0.1, format="%.1f")
        with col_ref:
            ref_val = st.number_input("Reference Value (e.g. Goniometer)", value=82.0, step=0.1, format="%.1f")
        with col_unit:
            unit = st.selectbox("Unit", ["° (Degrees)", "mm (Millimeters)", "px (Pixels)"])
        with col_add:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Add Trial", type="primary", use_container_width=True):
                val_session.add_trial(sys_val, ref_val, unit=unit)
                st.rerun()

    # Pre-populate sample benchmark trials if empty to illustrate
    if len(val_session.trials) == 0:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.info("No validation trials recorded yet. Enter a trial above or load sample benchmark data.")
        with c2:
            if st.button("Load Sample Benchmarks"):
                val_session.add_trial(84.2, 82.0, "°", "Trial 1 - Knee Flexion")
                val_session.add_trial(91.5, 90.0, "°", "Trial 2 - Knee Flexion")
                val_session.add_trial(135.0, 134.5, "°", "Trial 3 - Knee Extension")
                val_session.add_trial(45.2, 47.0, "°", "Trial 4 - Deep Flexion")
                st.rerun()

    # Statistics & MAE Cards
    stats = val_session.get_statistics()
    if stats["trial_count"] > 0:
        col_mae, col_rmse, col_max, col_trials = st.columns(4)

        with col_mae:
            st.markdown(
                f"""
                <div class="metric-card" style="background:#eff6ff; border-color:#93c5fd;">
                    <div class="metric-label" style="color:#1e40af;">MEAN ABSOLUTE ERROR (MAE)</div>
                    <div class="metric-value-huge" style="color:#1d4ed8; font-size:32px;">{stats['mean_absolute_error']} {stats.get('unit', '°')}</div>
                    <div class="metric-subtext" style="color:#1e40af;">Calculated from {stats['trial_count']} trial(s)</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_rmse:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">ROOT MEAN SQUARE ERROR (RMSE)</div>
                    <div class="metric-value-large">{stats['rmse']} {stats.get('unit', '°')}</div>
                    <div class="metric-subtext">Quadratic scoring metric</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_max:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">MAX ABSOLUTE DIFFERENCE</div>
                    <div class="metric-value-large">{stats['max_absolute_error']} {stats.get('unit', '°')}</div>
                    <div class="metric-subtext">Worst-case deviation</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_trials:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">TOTAL EMPIRICAL TRIALS</div>
                    <div class="metric-value-large">{stats['trial_count']}</div>
                    <div class="metric-subtext">Active dataset</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Trials Table
        st.markdown("#### Validation Trials Comparison Table")
        table_data = []
        for t in val_session.trials:
            table_data.append({
                "Trial #": t.trial_number,
                "System Observed Value": f"{t.system_value} {t.unit}",
                "Reference Standard Value": f"{t.reference_value} {t.unit}",
                "Absolute Difference": f"{t.absolute_difference} {t.unit}",
                "Signed Error": f"{t.error:+0.2f} {t.unit}",
                "Notes": t.notes
            })

        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        if st.button("🗑 Clear All Validation Trials"):
            val_session.clear()
            st.rerun()
