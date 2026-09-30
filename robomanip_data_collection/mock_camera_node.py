import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image

import numpy as np


class MockCameraNode(Node):

    def __init__(self):
        super().__init__("mock_camera_node")

        # ============================================================
        # Parameters
        # ============================================================

        self.declare_parameter(
            "publish_rate",
            15.0,
        )

        rate = float(
            self.get_parameter("publish_rate").value
        )

        # ============================================================
        # Camera configuration
        # ============================================================

        self.width = 1280
        self.height = 720

        # ============================================================
        # Publishers
        # ============================================================

        self.rgb_pub = self.create_publisher(
            Image,
            "color/image_rect_raw",
            10,
        )

        self.depth_pub = self.create_publisher(
            Image,
            "aligned_depth_to_color/image_raw",
            10,
        )

        # ============================================================
        # PRE-CREATE MOCK IMAGES
        #
        # Important:
        # Do NOT generate 1280x720 NumPy arrays every callback.
        # ============================================================

        # ------------------------------------------------------------
        # RGB image
        # ------------------------------------------------------------

        rgb = np.zeros(
            (
                self.height,
                self.width,
                3,
            ),
            dtype=np.uint8,
        )

        # Simple static gradient.
        rgb[:, :, 0] = np.linspace(
            0,
            255,
            self.width,
            dtype=np.uint8,
        )[None, :]

        rgb[:, :, 1] = np.linspace(
            0,
            255,
            self.height,
            dtype=np.uint8,
        )[:, None]

        rgb[:, :, 2] = 100

        # Convert only ONCE.
        self.rgb_bytes = rgb.tobytes()

        # ------------------------------------------------------------
        # Depth image
        #
        # D405 16UC1:
        # values represent depth in millimeters.
        # ------------------------------------------------------------

        depth = np.full(
            (
                self.height,
                self.width,
            ),
            1000,
            dtype=np.uint16,
        )

        # Fake object 500 mm away.
        depth[
            220:500,
            500:780
        ] = 500

        # Convert only ONCE.
        self.depth_bytes = depth.tobytes()

        # ============================================================
        # Pre-create message structure
        # ============================================================

        self.rgb_msg = Image()

        self.rgb_msg.height = self.height
        self.rgb_msg.width = self.width

        self.rgb_msg.encoding = "rgb8"

        self.rgb_msg.is_bigendian = False

        self.rgb_msg.step = (
            self.width * 3
        )

        self.rgb_msg.data = (
            self.rgb_bytes
        )

        # ------------------------------------------------------------

        self.depth_msg = Image()

        self.depth_msg.height = self.height
        self.depth_msg.width = self.width

        self.depth_msg.encoding = "16UC1"

        self.depth_msg.is_bigendian = False

        self.depth_msg.step = (
            self.width * 2
        )

        self.depth_msg.data = (
            self.depth_bytes
        )

        # ============================================================
        # Frame ID
        # ============================================================

        namespace = (
            self.get_namespace()
            .strip("/")
        )

        self.frame_id = (
            namespace +
            "_color_optical_frame"
        )

        self.rgb_msg.header.frame_id = (
            self.frame_id
        )

        self.depth_msg.header.frame_id = (
            self.frame_id
        )

        # ============================================================
        # Timer
        # ============================================================

        self.timer = self.create_timer(
            1.0 / rate,
            self.publish_images,
        )

        self.frame_number = 0

        self.get_logger().info(
            f"Mock D405 started: "
            f"{self.width}x{self.height} "
            f"@ {rate:.1f} Hz"
        )

    # ================================================================
    # Publish
    # ================================================================

    def publish_images(self):

        # ------------------------------------------------------------
        # ONE timestamp for RGB + aligned depth
        # ------------------------------------------------------------

        now = (
            self.get_clock()
            .now()
            .to_msg()
        )

        self.rgb_msg.header.stamp = now
        self.depth_msg.header.stamp = now

        # ------------------------------------------------------------
        # Publish
        # ------------------------------------------------------------

        self.rgb_pub.publish(
            self.rgb_msg
        )

        self.depth_pub.publish(
            self.depth_msg
        )

        self.frame_number += 1


def main(args=None):

    rclpy.init(args=args)

    node = MockCameraNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()