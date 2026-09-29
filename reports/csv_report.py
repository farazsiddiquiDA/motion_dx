"""
CSV telemetry and session summary exporter for MOTION DX.
"""

import io
import csv
from typing import Dict, Any, List


def generate_csv_telemetry(session_data: Dict[str, Any]) -> str:
    """Generate detailed timestamped time-series CSV export."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Header metadata comments
    writer.writerow(["# MOTION DX KINEMATIC TELEMETRY EXPORT"])
    writer.writerow(["# Session ID", session_data.get("session_id", "N/A")])
    writer.writerow(["# Patient ID", session_data.get("patient_id", "N/A")])
    writer.writerow(["# Assessment Type", session_data.get("assessment_type", "N/A")])
    writer.writerow(["# Target Joint", session_data.get("target_joint", "N/A")])
    writer.writerow(["# Camera", session_data.get("camera_type", "N/A")])
    writer.writerow(["# Safety Disclaimer", session_data.get("disclaimer", "")])
    writer.writerow([])

    # Table columns
    time_series: List[Dict[str, Any]] = session_data.get("time_series", [])
    if time_series:
        keys = list(time_series[0].keys())
        writer.writerow(keys)
        for row in time_series:
            writer.writerow([row.get(k, "") for k in keys])
    else:
        writer.writerow(["elapsed_sec", "angle", "quality", "mode"])

    return output.getvalue()


def generate_summary_csv(session_data: Dict[str, Any]) -> str:
    """Generate high-level clinical summary CSV."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Metric", "Value"])
    writer.writerow(["Session ID", session_data.get("session_id")])
    writer.writerow(["Patient ID", session_data.get("patient_id")])
    writer.writerow(["Assessment Type", session_data.get("assessment_type")])
    writer.writerow(["Target Joint", session_data.get("target_joint")])
    writer.writerow(["Duration (seconds)", session_data.get("duration_seconds")])
    writer.writerow(["Tracking Quality", session_data.get("tracking_quality")])

    summary = session_data.get("metrics_summary", {})
    for k, v in summary.items():
        if isinstance(v, (int, float, str)):
            writer.writerow([k.replace("_", " ").title(), v])

    return output.getvalue()
