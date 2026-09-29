"""
Standard OpenCV webcam fallback driver for MOTION DX.
"""

import time

from typing import (
    Optional,
    Dict,
    Any
)

import cv2

import numpy as np

from .base import (
    BaseCamera,
    CameraFrame
)


class WebcamCamera(BaseCamera):
    """
    Standard RGB webcam.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480
    ):

        super().__init__()

        self.camera_index = camera_index

        self.width = width

        self.height = height

        self.cap = None

        self.device_name = (
            f"Webcam (Index {camera_index})"
        )

        self.last_error = ""

    # ============================================================
    # HARDWARE DETECTION
    # ============================================================

    @staticmethod
    def is_hardware_available(
        index: int = 0
    ) -> bool:

        cap = None

        try:

            cap = cv2.VideoCapture(
                index,
                cv2.CAP_DSHOW
            )

            if not cap.isOpened():

                cap.release()

                cap = cv2.VideoCapture(
                    index
                )

            if not cap.isOpened():

                return False

            ret, frame = cap.read()

            return (
                ret
                and frame is not None
            )

        except Exception:

            return False

        finally:

            if cap is not None:

                cap.release()

    # ============================================================
    # CONNECT
    # ============================================================

    def connect(self) -> bool:

        self.last_error = ""

        try:

            # Try DirectShow first on Windows
            self.cap = cv2.VideoCapture(
                self.camera_index,
                cv2.CAP_DSHOW
            )

            # Fallback
            if not self.cap.isOpened():

                self.cap.release()

                self.cap = cv2.VideoCapture(
                    self.camera_index
                )

            if not self.cap.isOpened():

                self.last_error = (
                    "Could not open webcam."
                )

                self.disconnect()

                return False

            # Resolution
            self.cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                self.width
            )

            self.cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                self.height
            )

            # Test frame
            ret, frame = self.cap.read()

            if (
                not ret
                or frame is None
            ):

                self.last_error = (
                    "Webcam opened but "
                    "no frame was received."
                )

                self.disconnect()

                return False

            self.is_running = True

            print(
                "[WebcamCamera] "
                "Webcam connected."
            )

            return True

        except Exception as e:

            self.last_error = str(e)

            print(
                "[WebcamCamera] "
                f"Connection error: {e}"
            )

            self.disconnect()

            return False

    # ============================================================
    # DISCONNECT
    # ============================================================

    def disconnect(self) -> None:

        if self.cap is not None:

            try:

                self.cap.release()

            except Exception:
                pass

        self.cap = None

        self.is_running = False

    # ============================================================
    # READ FRAME
    # ============================================================

    def read_frame(
        self
    ) -> Optional[CameraFrame]:

        if (
            not self.is_running
            or self.cap is None
        ):

            return None

        try:

            ret, frame = self.cap.read()

            if (
                not ret
                or frame is None
            ):

                return None

            # Ensure resolution
            if (
                frame.shape[1] != self.width
                or frame.shape[0] != self.height
            ):

                frame = cv2.resize(
                    frame,
                    (
                        self.width,
                        self.height
                    )
                )

            return CameraFrame(

                color=frame,

                depth=None,

                depth_scale=0.001,

                is_depth_available=False,

                is_aligned=False,

                timestamp=time.time(),

                camera_name=self.device_name,

                is_demo=False,

                intrinsics=None
            )

        except Exception as e:

            self.last_error = str(e)

            return None

    # ============================================================
    # STATUS
    # ============================================================

    def is_connected(self) -> bool:

        return (
            self.is_running
            and self.cap is not None
            and self.cap.isOpened()
        )

    def get_status(
        self
    ) -> Dict[str, Any]:

        connected = self.is_connected()

        return {

            "type": "Standard Webcam",

            "device_name": self.device_name,

            "serial_number": "N/A",

            "connected": connected,

            "depth_available": False,

            "rgb_available": connected,

            "aligned": False,

            "tracking_ready": connected,

            "error": self.last_error
        }