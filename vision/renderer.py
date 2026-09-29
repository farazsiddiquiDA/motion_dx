"""
Clinical visual overlay renderer for MOTION DX.
Renders high-contrast green skeleton, joint nodes, measurement labels, and clinical quality indicators.
"""

from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from vision.pose import PoseResult, SKELETON_CONNECTIONS, LANDMARK_NAMES
from vision.face import FaceResult

# Medical Color Palette (BGR)
COLOR_CLINICAL_GREEN = (46, 204, 113)     # Vibrant clinical green for skeleton
COLOR_DARK_GREEN     = (39, 174, 96)      # Outline
COLOR_WHITE          = (255, 255, 255)
COLOR_DARK_BG        = (24, 28, 36)
COLOR_TEXT_DIM       = (180, 190, 205)
COLOR_WARNING_AMBER  = (40, 160, 240)     # Fair / warning
COLOR_ALERT_RED      = (60, 60, 230)      # Poor / reposition
COLOR_CYAN_ACCENT    = (230, 216, 80)     # RealSense Depth indicator
COLOR_MIDLINE_BLUE   = (235, 150, 50)


def render_pose_skeleton(
    image: np.ndarray,
    pose_result: PoseResult,
    highlight_joints: Optional[List[int]] = None,
    joint_angles: Optional[Dict[str, float]] = None,
    show_all_bones: bool = True
) -> np.ndarray:
    """
    Render high-visibility clinical green skeleton overlay on the camera frame.
    """
    output = image.copy()
    h, w, _ = output.shape

    if not pose_result.detected:
        _render_guidance_banner(output, pose_result.guidance_message, pose_result.tracking_quality)
        return output

    landmarks = pose_result.landmarks

    # 1. Draw connecting bones with thick anti-aliased green lines
    if show_all_bones:
        for p1_idx, p2_idx in SKELETON_CONNECTIONS:
            if p1_idx in landmarks and p2_idx in landmarks:
                lm1 = landmarks[p1_idx]
                lm2 = landmarks[p2_idx]
                if lm1.visibility > 0.4 and lm2.visibility > 0.4:
                    pt1 = (lm1.pixel_x, lm1.pixel_y)
                    pt2 = (lm2.pixel_x, lm2.pixel_y)
                    # Outer dark outline for contrast against any background
                    cv2.line(output, pt1, pt2, (10, 30, 15), 6, cv2.LINE_AA)
                    # Inner vibrant green bone
                    cv2.line(output, pt1, pt2, COLOR_CLINICAL_GREEN, 3, cv2.LINE_AA)

    # 2. Draw joint landmark nodes
    for idx, lm in landmarks.items():
        if lm.visibility > 0.4:
            pt = (lm.pixel_x, lm.pixel_y)
            is_highlighted = highlight_joints and (idx in highlight_joints)

            if is_highlighted:
                # Large pulsating / dual-ring node for active measurement joint
                cv2.circle(output, pt, 11, (10, 30, 15), -1, cv2.LINE_AA)
                cv2.circle(output, pt, 9, COLOR_WHITE, -1, cv2.LINE_AA)
                cv2.circle(output, pt, 6, COLOR_CLINICAL_GREEN, -1, cv2.LINE_AA)
            else:
                # Standard green dot
                cv2.circle(output, pt, 7, (10, 30, 15), -1, cv2.LINE_AA)
                cv2.circle(output, pt, 5, COLOR_CLINICAL_GREEN, -1, cv2.LINE_AA)

    # 3. Draw joint angle callout tags
    if joint_angles:
        for label, angle in joint_angles.items():
            # If the label has associated landmark coords, render clinical callout box
            # Default to top-left overlay corner if no specific coord
            pass

    # 4. Render tracking quality badge and guidance
    _render_quality_badge(output, pose_result.tracking_quality, pose_result.guidance_message)

    return output


def render_tmd_overlay(
    image: np.ndarray,
    face_result: FaceResult,
    current_step_name: str = ""
) -> np.ndarray:
    """
    Render clinical facial midline, jaw aperture vector, and deviation indicators.
    """
    output = image.copy()
    h, w, _ = output.shape

    if not face_result.detected or not face_result.metrics:
        _render_guidance_banner(output, face_result.guidance_message, face_result.tracking_quality)
        return output

    m = face_result.metrics

    # 1. Facial Midline (Vertical reference from Glabella to Subnasale and extended)
    p1 = m.midline_p1
    p2 = m.midline_p2
    # Extend line downward past chin
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    ext_bottom = (p2[0] + dx * 2, p2[1] + dy * 2)
    cv2.line(output, p1, ext_bottom, (40, 40, 50), 3, cv2.LINE_AA)
    cv2.line(output, p1, ext_bottom, COLOR_MIDLINE_BLUE, 2, cv2.LINE_AA)

    # 2. Oral Aperture (Upper lip -> Lower lip)
    cv2.line(output, m.upper_lip_2d, m.lower_lip_2d, COLOR_CLINICAL_GREEN, 3, cv2.LINE_AA)
    cv2.circle(output, m.upper_lip_2d, 5, COLOR_WHITE, -1, cv2.LINE_AA)
    cv2.circle(output, m.lower_lip_2d, 5, COLOR_CLINICAL_GREEN, -1, cv2.LINE_AA)

    # 3. Chin Landmark and Lateral Deviation
    cv2.circle(output, m.chin_point_2d, 7, COLOR_CLINICAL_GREEN, -1, cv2.LINE_AA)
    # Perpendicular line from chin to midline
    # Find projected point on midline
    # Draw horizontal deviation line
    cv2.line(
        output,
        m.chin_point_2d,
        (int(m.chin_point_2d[0] - m.lateral_deviation), m.chin_point_2d[1]),
        COLOR_WARNING_AMBER,
        2,
        cv2.LINE_AA
    )

    # 4. Step banner and telemetry
    if current_step_name:
        cv2.rectangle(output, (15, 15), (320, 55), COLOR_DARK_BG, -1)
        cv2.rectangle(output, (15, 15), (320, 55), COLOR_CLINICAL_GREEN, 1)
        cv2.putText(
            output,
            f"GUIDED STEP: {current_step_name.upper()}",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_WHITE,
            1,
            cv2.LINE_AA
        )

    _render_quality_badge(output, face_result.tracking_quality, face_result.guidance_message)
    return output


def _render_quality_badge(frame: np.ndarray, quality: str, guidance: str) -> None:
    """Render subtle medical quality indicator badge on top right corner."""
    h, w, _ = frame.shape
    badge_w, badge_h = 160, 40
    x1, y1 = w - badge_w - 15, 15
    x2, y2 = w - 15, 15 + badge_h

    # Select color
    if quality == "GOOD":
        dot_color = COLOR_CLINICAL_GREEN
    elif quality == "FAIR":
        dot_color = COLOR_WARNING_AMBER
    else:
        dot_color = COLOR_ALERT_RED

    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_DARK_BG, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), dot_color, 1)
    cv2.circle(frame, (x1 + 18, y1 + 20), 6, dot_color, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"TRACKING: {quality}",
        (x1 + 32, y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        COLOR_WHITE,
        1,
        cv2.LINE_AA
    )

    # If quality is POOR, display high-contrast guidance overlay
    if quality == "POOR" and guidance:
        bar_h = 36
        cv2.rectangle(frame, (0, h - bar_h), (w, h), (15, 15, 20), -1)
        cv2.putText(
            frame,
            f"CLINICAL NOTICE: {guidance}",
            (20, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_WARNING_AMBER,
            1,
            cv2.LINE_AA
        )


def _render_guidance_banner(frame: np.ndarray, guidance: str, quality: str) -> None:
    """Render full-frame repositioning advice when tracking is lost."""
    h, w, _ = frame.shape
    cv2.rectangle(frame, (40, h // 2 - 40), (w - 40, h // 2 + 40), COLOR_DARK_BG, -1)
    cv2.rectangle(frame, (40, h // 2 - 40), (w - 40, h // 2 + 40), COLOR_ALERT_RED, 2)
    cv2.putText(
        frame,
        "TRACKING INTERRUPTED",
        (w // 2 - 130, h // 2 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        COLOR_WHITE,
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        guidance or "Please reposition the subject in camera view.",
        (w // 2 - 170, h // 2 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLOR_TEXT_DIM,
        1,
        cv2.LINE_AA
    )
