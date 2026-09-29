"""
Intel RealSense RGB-D camera driver.

Designed for:
- Intel RealSense D405
- RGB streaming
- Depth streaming
- Depth-to-color alignment
- 3D deprojection
"""

import time

from typing import (
    Optional,
    Dict,
    Any
)

import numpy as np


# ================================================================
# IMPORT REALSENSE
# ================================================================

try:

    import pyrealsense2 as rs

    PYREALSENSE_AVAILABLE = True

except ImportError:

    rs = None

    PYREALSENSE_AVAILABLE = False

    print(
        "[RealSenseCamera] "
        "pyrealsense2 is not installed."
    )


from .base import (
    BaseCamera,
    CameraFrame
)


class RealSenseCamera(BaseCamera):
    """
    Intel RealSense camera driver.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30
    ):

        super().__init__()

        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = None
        self.config = None
        self.align = None

        self.depth_scale = 0.001

        self.intrinsics = None

        self.device_name = (
            "Intel RealSense D405"
        )

        self.serial_number = "Unknown"

        self.is_d405 = False

        self.last_error = ""

    # ============================================================
    # HARDWARE DETECTION
    # ============================================================

    @staticmethod
    def is_hardware_available() -> bool:

        if not PYREALSENSE_AVAILABLE:
            return False

        try:

            context = rs.context()

            devices = context.query_devices()

            return len(devices) > 0

        except Exception as e:

            print(
                "[RealSenseCamera] "
                f"Hardware detection error: {e}"
            )

            return False

    # ============================================================
    # CONNECT
    # ============================================================

    def connect(self) -> bool:

        # Reset previous state
        self.is_running = False
        self.last_error = ""

        if not PYREALSENSE_AVAILABLE:

            self.last_error = (
                "pyrealsense2 is not installed."
            )

            print(
                "[RealSenseCamera] "
                + self.last_error
            )

            return False

        try:

            # ----------------------------------------------------
            # Find devices
            # ----------------------------------------------------

            context = rs.context()

            devices = context.query_devices()

            if len(devices) == 0:

                self.last_error = (
                    "No RealSense device detected."
                )

                print(
                    "[RealSenseCamera] "
                    + self.last_error
                )

                return False

            # ----------------------------------------------------
            # Find D405 specifically
            # ----------------------------------------------------

            selected_device = None

            for device in devices:

                try:

                    name = device.get_info(
                        rs.camera_info.name
                    )

                    print(
                        "[RealSenseCamera] "
                        f"Found device: {name}"
                    )

                    if "D405" in name.upper():

                        selected_device = device

                        break

                except Exception:
                    continue

            # If D405 wasn't found, use first device
            if selected_device is None:

                selected_device = devices[0]

            self.device_name = (
                selected_device.get_info(
                    rs.camera_info.name
                )
            )

            self.serial_number = (
                selected_device.get_info(
                    rs.camera_info.serial_number
                )
            )

            self.is_d405 = (
                "D405"
                in self.device_name.upper()
            )

            print(
                "[RealSenseCamera] "
                f"Using: {self.device_name}"
            )

            print(
                "[RealSenseCamera] "
                f"Serial: {self.serial_number}"
            )

            # ----------------------------------------------------
            # Create pipeline
            # ----------------------------------------------------

            self.pipeline = rs.pipeline()

            self.config = rs.config()

            # Explicitly select this device
            self.config.enable_device(
                self.serial_number
            )

            # ----------------------------------------------------
            # Enable streams
            # ----------------------------------------------------

            self.config.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.fps
            )

            self.config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.bgr8,
                self.fps
            )

            print(
                "[RealSenseCamera] "
                "Starting pipeline..."
            )

            # ----------------------------------------------------
            # Start
            # ----------------------------------------------------

            profile = self.pipeline.start(
                self.config
            )

            print(
                "[RealSenseCamera] "
                "Pipeline started."
            )

            # ----------------------------------------------------
            # Depth scale
            # ----------------------------------------------------

            depth_sensor = (
                profile
                .get_device()
                .first_depth_sensor()
            )

            self.depth_scale = (
                depth_sensor.get_depth_scale()
            )

            print(
                "[RealSenseCamera] "
                f"Depth scale: {self.depth_scale}"
            )

            # ----------------------------------------------------
            # Alignment
            # ----------------------------------------------------

            self.align = rs.align(
                rs.stream.color
            )

            # ----------------------------------------------------
            # Color intrinsics
            # ----------------------------------------------------

            color_stream = (
                profile
                .get_stream(
                    rs.stream.color
                )
                .as_video_stream_profile()
            )

            intrinsics = (
                color_stream.get_intrinsics()
            )

            self.intrinsics = {

                "fx": float(intrinsics.fx),

                "fy": float(intrinsics.fy),

                "ppx": float(intrinsics.ppx),

                "ppy": float(intrinsics.ppy),

                "width": int(intrinsics.width),

                "height": int(intrinsics.height)
            }

            # ----------------------------------------------------
            # Mark running
            # ----------------------------------------------------

            self.is_running = True

            print(
                "[RealSenseCamera] "
                "CONNECTED SUCCESSFULLY."
            )

            # ----------------------------------------------------
            # Warm-up frames
            # ----------------------------------------------------

            print(
                "[RealSenseCamera] "
                "Warming up camera..."
            )

            for _ in range(10):

                try:

                    self.pipeline.wait_for_frames(
                        timeout_ms=2000
                    )

                except Exception:
                    pass

            print(
                "[RealSenseCamera] "
                "Camera ready."
            )

            return True

        except Exception as e:

            self.last_error = str(e)

            print(
                "[RealSenseCamera] "
                f"Connection error: {e}"
            )

            self.disconnect()

            return False

    # ============================================================
    # DISCONNECT
    # ============================================================

    def disconnect(self) -> None:

        if self.pipeline is not None:

            try:

                self.pipeline.stop()

            except Exception:
                pass

        self.pipeline = None

        self.config = None

        self.align = None

        self.is_running = False

    # ============================================================
    # READ FRAME
    # ============================================================

    def read_frame(
        self
    ) -> Optional[CameraFrame]:

        if (
            not self.is_running
            or self.pipeline is None
        ):

            return None

        try:

            # ----------------------------------------------------
            # Get synchronized frames
            # ----------------------------------------------------

            frames = (
                self.pipeline.wait_for_frames(
                    timeout_ms=2000
                )
            )

            if not frames:

                return None

            # ----------------------------------------------------
            # Align depth to color
            # ----------------------------------------------------

            if self.align is not None:

                aligned_frames = (
                    self.align.process(frames)
                )

            else:

                aligned_frames = frames

            depth_frame = (
                aligned_frames.get_depth_frame()
            )

            color_frame = (
                aligned_frames.get_color_frame()
            )

            if (
                not depth_frame
                or not color_frame
            ):

                return None

            # ----------------------------------------------------
            # Convert color
            # ----------------------------------------------------

            color_image = np.asanyarray(
                color_frame.get_data()
            )

            # ----------------------------------------------------
            # Convert depth
            # ----------------------------------------------------

            depth_raw = np.asanyarray(
                depth_frame.get_data()
            )

            depth_meters = (
                depth_raw.astype(
                    np.float32
                )
                *
                self.depth_scale
            )

            # ----------------------------------------------------
            # Return frame
            # ----------------------------------------------------

            return CameraFrame(

                color=color_image,

                depth=depth_meters,

                depth_scale=self.depth_scale,

                is_depth_available=True,

                is_aligned=True,

                timestamp=time.time(),

                camera_name=self.device_name,

                is_demo=False,

                intrinsics=self.intrinsics
            )

        except Exception as e:

            self.last_error = str(e)

            print(
                "[RealSenseCamera] "
                f"Frame error: {e}"
            )

            return None

    # ============================================================
    # CONNECTION STATUS
    # ============================================================

    def is_connected(self) -> bool:

        if not self.is_running:
            return False

        if self.pipeline is None:
            return False

        return True

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self
    ) -> Dict[str, Any]:

        connected = self.is_connected()

        return {

            "type": (
                "RealSense D405"
                if self.is_d405
                else "RealSense Generic"
            ),

            "device_name": self.device_name,

            "serial_number": self.serial_number,

            "connected": connected,

            "depth_available": (
                connected
            ),

            "rgb_available": (
                connected
            ),

            "aligned": (
                connected
                and self.align is not None
            ),

            "tracking_ready": (
                connected
            ),

            "error": self.last_error
        }