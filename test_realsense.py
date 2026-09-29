import pyrealsense2 as rs
import cv2

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

print("Starting RealSense...")

pipeline.start(config)

print("Camera started successfully!")

try:
    while True:
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("Missing frame")
            continue

        color = cv2.cvtColor(
            __import__("numpy").asanyarray(color_frame.get_data()),
            cv2.COLOR_RGB2BGR
        )

        cv2.imshow("RealSense D405", color)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()