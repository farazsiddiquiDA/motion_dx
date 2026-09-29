"""
Camera subsystem factory for MOTION DX.

Manages:
- Intel RealSense D405
- Standard webcam
- Synthetic demo camera
"""

from .base import BaseCamera, CameraFrame
from .realsense import RealSenseCamera
from .webcam import WebcamCamera
from .demo_camera import DemoCamera


def get_camera(
    preferred: str = "auto",
    scenario: str = "knee"
) -> BaseCamera:
    """
    Create and connect the requested camera.

    preferred:
        auto
        realsense
        webcam
        demo

    scenario:
        knee
        tmd
        gait
    """

    # ============================================================
    # MANUAL REALSENSE
    # ============================================================

    if preferred == "realsense":

        print("[Camera] Requested: Intel RealSense D405")

        camera = RealSenseCamera()

        if camera.connect():
            print("[Camera] RealSense connected successfully.")
            return camera

        print("[Camera] ERROR: RealSense connection failed.")

        # Return the object so the diagnostic page can show
        # the actual failed state.
        return camera

    # ============================================================
    # MANUAL WEBCAM
    # ============================================================

    if preferred == "webcam":

        print("[Camera] Requested: Webcam")

        camera = WebcamCamera()

        if camera.connect():
            print("[Camera] Webcam connected successfully.")
            return camera

        print("[Camera] ERROR: Webcam connection failed.")

        return camera

    # ============================================================
    # MANUAL DEMO
    # ============================================================

    if preferred == "demo":

        print(f"[Camera] Requested: Demo Mode ({scenario})")

        camera = DemoCamera(
            scenario=scenario
        )

        camera.connect()

        return camera

    # ============================================================
    # AUTO MODE
    # ============================================================

    print("[Camera] Automatic camera detection started.")

    # ------------------------------------------------------------
    # Try RealSense first
    # ------------------------------------------------------------

    if RealSenseCamera.is_hardware_available():

        print("[Camera] RealSense hardware detected.")

        rs_camera = RealSenseCamera()

        if rs_camera.connect():

            print("[Camera] RealSense selected.")

            return rs_camera

        print(
            "[Camera] RealSense detected "
            "but connection failed."
        )

    else:

        print("[Camera] No RealSense device detected.")

    # ------------------------------------------------------------
    # Try webcam
    # ------------------------------------------------------------

    if WebcamCamera.is_hardware_available(0):

        print("[Camera] Webcam detected.")

        webcam = WebcamCamera(0)

        if webcam.connect():

            print("[Camera] Webcam selected.")

            return webcam

        print("[Camera] Webcam connection failed.")

    else:

        print("[Camera] No webcam detected.")

    # ------------------------------------------------------------
    # Final fallback
    # ------------------------------------------------------------

    print(
        "[Camera] No working physical camera found."
    )

    print(
        "[Camera] Starting synthetic Demo Mode."
    )

    demo = DemoCamera(
        scenario=scenario
    )

    demo.connect()

    return demo


__all__ = [
    "BaseCamera",
    "CameraFrame",
    "RealSenseCamera",
    "WebcamCamera",
    "DemoCamera",
    "get_camera",
]