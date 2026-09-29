import os
import signal
import subprocess
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class EpisodeRecorder(Node):
    """Start and stop individual rosbag episodes through ROS services."""

    def __init__(self):
        super().__init__("episode_recorder")

        self.declare_parameter(
            "output_directory",
            str(Path.home() / "robomanip_datasets"),
        )

        self.declare_parameter(
            "topics",
            [
                "/left_arm/joint_states",
                "/right_arm/joint_states",
            ],
        )

        self.output_directory = Path(
            self.get_parameter("output_directory")
            .get_parameter_value()
            .string_value
        )

        self.topics = (
            self.get_parameter("topics")
            .get_parameter_value()
            .string_array_value
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.recording_process = None
        self.current_episode = None

        self.start_service = self.create_service(
            Trigger,
            "start_episode",
            self.start_episode_callback,
        )

        self.stop_service = self.create_service(
            Trigger,
            "stop_episode",
            self.stop_episode_callback,
        )

        self.get_logger().info(
            f"Dataset directory: {self.output_directory}"
        )

        self.get_logger().info(
            "Episode recorder ready"
        )

    def get_next_episode_name(self):
        existing = []

        for path in self.output_directory.glob("episode_*"):
            if not path.is_dir():
                continue

            try:
                number = int(path.name.split("_")[1])
                existing.append(number)
            except (IndexError, ValueError):
                continue

        next_number = max(existing, default=0) + 1

        return f"episode_{next_number:06d}"

    def start_episode_callback(self, request, response):

        if self.recording_process is not None:
            response.success = False
            response.message = (
                f"Already recording {self.current_episode}"
            )
            return response

        episode_name = self.get_next_episode_name()

        episode_path = (
            self.output_directory / episode_name
        )

        command = [
            "ros2",
            "bag",
            "record",
            "-o",
            str(episode_path),
            *self.topics,
        ]

        self.get_logger().info(
            f"Starting {episode_name}"
        )

        try:
            self.recording_process = subprocess.Popen(
                command,
                preexec_fn=os.setsid,
            )

            self.current_episode = episode_name

            response.success = True
            response.message = (
                f"Started {episode_name}"
            )

        except Exception as exc:
            self.recording_process = None
            self.current_episode = None

            response.success = False
            response.message = (
                f"Failed to start recording: {exc}"
            )

        return response

    def stop_episode_callback(self, request, response):

        if self.recording_process is None:
            response.success = False
            response.message = "No episode is being recorded"
            return response

        episode_name = self.current_episode

        self.get_logger().info(
            f"Stopping {episode_name}"
        )

        try:
            os.killpg(
                os.getpgid(self.recording_process.pid),
                signal.SIGINT,
            )

            self.recording_process.wait(timeout=10)

            self.recording_process = None
            self.current_episode = None

            response.success = True
            response.message = (
                f"Saved {episode_name}"
            )

        except Exception as exc:
            response.success = False
            response.message = (
                f"Failed to stop recording: {exc}"
            )

        return response

    def destroy_node(self):

        if self.recording_process is not None:

            self.get_logger().warning(
                "Recorder shutting down while an episode "
                "is active. Stopping rosbag."
            )

            try:
                os.killpg(
                    os.getpgid(self.recording_process.pid),
                    signal.SIGINT,
                )

                self.recording_process.wait(timeout=10)

            except Exception as exc:
                self.get_logger().error(
                    f"Failed to stop rosbag: {exc}"
                )

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = EpisodeRecorder()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()