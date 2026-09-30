import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped


class MockFairinoNode(Node):

    def __init__(self):
        super().__init__("mock_fairino_node")

        self.declare_parameter("robot_name", "fairino_1")
        self.declare_parameter("publish_rate", 15.0)

        self.robot_name = (
            self.get_parameter("robot_name")
            .get_parameter_value()
            .string_value
        )

        rate = (
            self.get_parameter("publish_rate")
            .get_parameter_value()
            .double_value
        )

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.joint_pub = self.create_publisher(
            JointState,
            "joint_states",
            qos,
        )

        self.eef_pub = self.create_publisher(
            PoseStamped,
            "eef_pose",
            qos,
        )

        self.start_time = self.get_clock().now()

        self.timer = self.create_timer(
            1.0 / rate,
            self.publish_state,
        )

        self.get_logger().info(
            f"Mock {self.robot_name} publishing at {rate} Hz"
        )

    def publish_state(self):

        now = self.get_clock().now()

        t = (
            now.nanoseconds -
            self.start_time.nanoseconds
        ) / 1e9

        # -------------------------
        # Joint state
        # -------------------------

        joint_msg = JointState()

        # Same timestamp for joint + EE state
        joint_msg.header.stamp = now.to_msg()
        joint_msg.header.frame_id = ""

        joint_msg.name = [
            f"joint_{i}"
            for i in range(1, 7)
        ]

        joint_msg.position = [
            0.5 * math.sin(t + i * 0.2)
            for i in range(6)
        ]

        joint_msg.velocity = [
            0.5 * math.cos(t + i * 0.2)
            for i in range(6)
        ]

        # Match real robot: effort empty
        joint_msg.effort = []

        self.joint_pub.publish(joint_msg)

        # -------------------------
        # End-effector pose
        # -------------------------

        pose_msg = PoseStamped()

        pose_msg.header.stamp = now.to_msg()
        pose_msg.header.frame_id = ""

        # Fake smooth EE trajectory
        pose_msg.pose.position.x = (
            0.4 + 0.05 * math.sin(t)
        )

        pose_msg.pose.position.y = (
            0.05 * math.cos(t)
        )

        pose_msg.pose.position.z = (
            0.3 + 0.02 * math.sin(t * 0.5)
        )

        # Valid normalized quaternion.
        yaw = 0.2 * math.sin(t)

        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = math.sin(yaw / 2.0)
        pose_msg.pose.orientation.w = math.cos(yaw / 2.0)

        self.eef_pub.publish(pose_msg)


def main(args=None):

    rclpy.init(args=args)

    node = MockFairinoNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()