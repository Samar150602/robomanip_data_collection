import rclpy

from rclpy.node import Node
from sensor_msgs.msg import JointState

from robomanip_data_collection.fairino_interface import FairinoInterface


class FairinoStateNode(Node):

    def __init__(self):
        super().__init__("fairino_state_node")

        self.declare_parameter("robot_ip", "192.168.58.2")
        self.declare_parameter("robot_name", "left_arm")
        self.declare_parameter("publish_rate", 100.0)

        robot_ip = (
            self.get_parameter("robot_ip")
            .get_parameter_value()
            .string_value
        )

        self.robot_name = (
            self.get_parameter("robot_name")
            .get_parameter_value()
            .string_value
        )

        publish_rate = (
            self.get_parameter("publish_rate")
            .get_parameter_value()
            .double_value
        )

        self.publisher = self.create_publisher(
            JointState,
            "joint_states",
            10,
        )

        self.robot = FairinoInterface(robot_ip)

        self.get_logger().info(
            f"Connecting to {self.robot_name} at {robot_ip}"
        )

        self.robot.connect()

        self.get_logger().info("FAIRINO connected")

        self.timer = self.create_timer(
            1.0 / publish_rate,
            self.publish_state,
        )

    def publish_state(self):

        try:
            joints = self.robot.get_joint_positions()

            msg = JointState()

            # Timestamp immediately after receiving state.
            msg.header.stamp = self.get_clock().now().to_msg()

            msg.name = [
                f"{self.robot_name}_joint1",
                f"{self.robot_name}_joint2",
                f"{self.robot_name}_joint3",
                f"{self.robot_name}_joint4",
                f"{self.robot_name}_joint5",
                f"{self.robot_name}_joint6",
            ]

            msg.position = joints

            self.publisher.publish(msg)

        except Exception as exc:
            self.get_logger().error(
                f"Failed to read robot state: {exc}"
            )


def main(args=None):

    rclpy.init(args=args)

    node = FairinoStateNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()