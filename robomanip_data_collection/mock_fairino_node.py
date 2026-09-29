import math

import rclpy

from rclpy.node import Node
from sensor_msgs.msg import JointState


class MockFairinoNode(Node):

    def __init__(self):

        super().__init__("mock_fairino_node")

        self.declare_parameter("robot_name", "left_arm")
        self.declare_parameter("publish_rate", 100.0)

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

        self.publisher = self.create_publisher(
            JointState,
            "joint_states",
            10,
        )

        self.start_time = self.get_clock().now()

        self.timer = self.create_timer(
            1.0 / rate,
            self.publish_state,
        )

    def publish_state(self):

        now = self.get_clock().now()

        t = (
            now.nanoseconds -
            self.start_time.nanoseconds
        ) / 1e9

        msg = JointState()
        msg.header.stamp = now.to_msg()

        msg.name = [
            f"{self.robot_name}_joint{i}"
            for i in range(1, 7)
        ]

        msg.position = [
            0.5 * math.sin(t + i * 0.2)
            for i in range(6)
        ]

        self.publisher.publish(msg)


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