"""
Professional Clinical PDF Report Generator for MOTION DX.
Built using ReportLab and Matplotlib for clean, high-resolution medical documentation.
"""

import io
import time
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_kinematics_plot_image(session_data: Dict[str, Any]) -> Optional[io.BytesIO]:
    """Render an aesthetic medical time-series plot as an in-memory PNG buffer."""
    time_series = session_data.get("time_series", [])
    if not time_series or len(time_series) < 2:
        return None

    times = [s.get("elapsed_sec", idx * 0.1) for idx, s in enumerate(time_series)]
    # Look for angle or opening
    if "angle" in time_series[0]:
        vals = [s.get("angle", 0.0) for s in time_series]
        y_label = "Joint Angle (degrees)"
        title = f"Observed Kinematic Trajectory: {session_data.get('target_joint', 'Joint')} Angle vs. Time"
    elif "opening" in time_series[0]:
        vals = [s.get("opening", 0.0) for s in time_series]
        y_label = f"Aperture ({session_data.get('metrics_summary', {}).get('unit', 'units')})"
        title = "TMD / Jaw Movement Kinematic Curve"
    elif "left_knee_angle" in time_series[0]:
        vals = [s.get("left_knee_angle", 0.0) for s in time_series]
        y_label = "Knee Angle (degrees)"
        title = "Gait Observation: Bilateral Kinematic Cycles"
    else:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 2.6), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8F9FA")

    # Plot line
    ax.plot(times, vals, color="#1E88E5", linewidth=2.0, label="Observed Kinematics")

    # If contralateral is present
    if "right_knee_angle" in time_series[0]:
        r_vals = [s.get("right_knee_angle", 0.0) for s in time_series]
        ax.plot(times, r_vals, color="#43A047", linewidth=2.0, linestyle="--", label="Right Knee")

    ax.set_title(title, fontsize=10, fontweight="bold", pad=8, color="#212529")
    ax.set_xlabel("Elapsed Time (seconds)", fontsize=8, color="#495057")
    ax.set_ylabel(y_label, fontsize=8, color="#495057")
    ax.grid(True, linestyle=":", alpha=0.6, color="#CED4DA")
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf_report(session_data: Dict[str, Any]) -> bytes:
    """Generate a publication-grade clinical assessment report PDF."""
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0D47A1")
    )
    subtitle_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#546E7A")
    )
    section_h1 = ParagraphStyle(
        "SecH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#263238"),
        spaceBefore=10,
        spaceAfter=4
    )
    disclaimer_style = ParagraphStyle(
        "Discl",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#C62828")
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#37474F")
    )

    story = []

    # 1. Header & Branding
    story.append(Paragraph("MOTION DX — CLINICAL MOVEMENT MEASUREMENT", title_style))
    story.append(Paragraph("Objective Kinematics Assessment Report | Computer-Vision Biomechanics", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0D47A1"), spaceAfter=10))

    # 2. Safety Statement Box
    disclaimer_text = (
        "<b>REGULATORY & CLINICAL SAFETY NOTICE:</b> "
        "This system provides objective movement measurements for clinician review and biomechanical quantification. "
        "It does NOT diagnose medical conditions, diseases, or provide automated clinical treatment prescriptions."
    )
    safety_table = Table(
        [[Paragraph(disclaimer_text, disclaimer_style)]],
        colWidths=[530]
    )
    safety_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFEBEE")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#EF9A9A")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(safety_table)
    story.append(Spacer(1, 10))

    # 3. Patient & Hardware Metadata Table
    patient_id = session_data.get("patient_id", "N/A")
    session_id = session_data.get("session_id", "N/A")
    date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session_data.get("start_time", time.time())))
    camera_name = session_data.get("camera_type", "Generic")
    depth_status = "Enabled (RGB-D Aligned)" if session_data.get("is_depth_enabled") else "Disabled (2D Projection)"
    duration = f"{session_data.get('duration_seconds', 0)} seconds"

    meta_data = [
        [Paragraph("<b>Patient ID:</b>", body_style), Paragraph(patient_id, body_style),
         Paragraph("<b>Date / Time:</b>", body_style), Paragraph(date_str, body_style)],
        [Paragraph("<b>Session ID:</b>", body_style), Paragraph(session_id, body_style),
         Paragraph("<b>Duration:</b>", body_style), Paragraph(duration, body_style)],
        [Paragraph("<b>Camera System:</b>", body_style), Paragraph(camera_name, body_style),
         Paragraph("<b>Depth Pipeline:</b>", body_style), Paragraph(depth_status, body_style)],
        [Paragraph("<b>Assessment:</b>", body_style), Paragraph(session_data.get("assessment_type", "Limb ROM"), body_style),
         Paragraph("<b>Tracking Quality:</b>", body_style), Paragraph(session_data.get("tracking_quality", "GOOD"), body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[100, 165, 100, 165])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8F9FA")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CFD8DC")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ECEFF1")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # 4. Observed Kinematics Results Table
    story.append(Paragraph("Observed Kinematic Measurements", section_h1))
    summary = session_data.get("metrics_summary", {})

    results_rows = [
        [Paragraph("<b>Parameter</b>", body_style), Paragraph("<b>Observed Value</b>", body_style), Paragraph("<b>Clinical Description</b>", body_style)]
    ]

    if "observed_rom" in summary:
        results_rows.append([
            Paragraph("Observed ROM", body_style),
            Paragraph(f"<b>{summary.get('observed_rom', 0.0)}°</b>", body_style),
            Paragraph("Active dynamic excursion across recorded session.", body_style)
        ])
        results_rows.append([
            Paragraph("Peak Flexion / Min Angle", body_style),
            Paragraph(f"{summary.get('minimum_angle', 0.0)}°", body_style),
            Paragraph("Maximum flexion reached during assessment.", body_style)
        ])
        results_rows.append([
            Paragraph("Peak Extension / Max Angle", body_style),
            Paragraph(f"{summary.get('maximum_angle', 0.0)}°", body_style),
            Paragraph("Maximum extension reached during assessment.", body_style)
        ])
        results_rows.append([
            Paragraph("Mean Joint Angle", body_style),
            Paragraph(f"{summary.get('mean_angle', 0.0)}° (±{summary.get('std_angle', 0.0)}°)", body_style),
            Paragraph("Mean kinematic position across valid samples.", body_style)
        ])

    elif "observed_peak_opening" in summary:
        u = summary.get("unit", "units")
        results_rows.append([
            Paragraph("Peak Jaw Opening", body_style),
            Paragraph(f"<b>{summary.get('observed_peak_opening', 0.0)} {u}</b>", body_style),
            Paragraph("Maximum inter-incisal/oral aperture recorded.", body_style)
        ])
        results_rows.append([
            Paragraph("Left Lateral Excursion", body_style),
            Paragraph(f"{summary.get('max_left_lateral_excursion', 0.0)} {u}", body_style),
            Paragraph("Maximum deviation towards left side.", body_style)
        ])
        results_rows.append([
            Paragraph("Right Lateral Excursion", body_style),
            Paragraph(f"{summary.get('max_right_lateral_excursion', 0.0)} {u}", body_style),
            Paragraph("Maximum deviation towards right side.", body_style)
        ])
        results_rows.append([
            Paragraph("Lateral Excursion Symmetry", body_style),
            Paragraph(f"{summary.get('lateral_symmetry_index_pct', 100.0)}%", body_style),
            Paragraph(f"Observed asymmetry: {summary.get('observed_lateral_difference', 0.0)} {u}", body_style)
        ])

    elif "left_knee_observed_rom" in summary:
        results_rows.append([
            Paragraph("Left Knee ROM", body_style),
            Paragraph(f"{summary.get('left_knee_observed_rom', 0.0)}°", body_style),
            Paragraph("Observed swing-phase sagittal knee excursion.", body_style)
        ])
        results_rows.append([
            Paragraph("Right Knee ROM", body_style),
            Paragraph(f"{summary.get('right_knee_observed_rom', 0.0)}°", body_style),
            Paragraph("Observed swing-phase sagittal knee excursion.", body_style)
        ])
        results_rows.append([
            Paragraph("Bilateral Knee Symmetry", body_style),
            Paragraph(f"{summary.get('bilateral_knee_symmetry_pct', 100.0)}%", body_style),
            Paragraph(f"Observed difference: {summary.get('bilateral_rom_difference', 0.0)}°", body_style)
        ])

    results_table = Table(results_rows, colWidths=[160, 130, 240])
    results_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E0E0E0")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#B0BEC5")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CFD8DC")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(results_table)
    story.append(Spacer(1, 10))

    # 5. Kinematic Graph
    plot_buf = generate_kinematics_plot_image(session_data)
    if plot_buf:
        story.append(Paragraph("Kinematic Movement Profile", section_h1))
        img = Image(plot_buf, width=7.0 * inch, height=2.6 * inch)
        story.append(img)
        story.append(Spacer(1, 10))

    # 6. Clinical Notes
    story.append(Paragraph("Clinician Notes & Observations", section_h1))
    notes = session_data.get("clinical_notes", "").strip() or "No clinician observations recorded."
    notes_table = Table([[Paragraph(notes, body_style)]], colWidths=[530])
    notes_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F9FBE7")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE775")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(notes_table)
    story.append(Spacer(1, 10))

    # 7. Limitations & Technical Notes
    story.append(Paragraph("Technical Limitations", section_h1))
    limitation_text = (
        "Measurements rely on computer-vision landmark localization and optical depth sampling. "
        "Loose clothing, acute camera oblique angles, rapid ballistic motion, and severe occlusions "
        "can influence landmark precision. Bilateral symmetry metrics are descriptive comparisons."
    )
    story.append(Paragraph(limitation_text, subtitle_style))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
