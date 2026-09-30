from pathlib import Path
import shutil

import h5py
import numpy as np
import videoio

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger

from message_filters import (
    Subscriber,
    ApproximateTimeSynchronizer,
)


class EpisodeRecorder(Node):

    def __init__(self):
        super().__init__("episode_recorder")

        # ============================================================
        # Parameters
        # ============================================================

        self.declare_parameter(
            "output_directory",
            str(Path.home() / "robomanip_datasets"),
        )

        self.declare_parameter(
            "sync_slop",
            0.04,  # 40 ms
        )

        self.declare_parameter(
            "target_rate",
            15.0,
        )

        self.output_directory = Path(
            self.get_parameter("output_directory").value
        )

        self.sync_slop = float(
            self.get_parameter("sync_slop").value
        )

        self.target_rate = float(
            self.get_parameter("target_rate").value
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ============================================================
        # Recording state
        # ============================================================

        self.recording = False
        self.current_episode = None

        self.first_timestamp = None
        self.sample_count = 0

        self.clear_buffers()

        # ============================================================
        # Subscribers
        # ============================================================

        # FR5 #1

        self.joint_1_sub = Subscriber(
            self,
            JointState,
            "/fairino_1/joint_states",
        )

        self.eef_1_sub = Subscriber(
            self,
            PoseStamped,
            "/fairino_1/eef_pose",
        )

        # FR5 #2

        self.joint_2_sub = Subscriber(
            self,
            JointState,
            "/fairino_2/joint_states",
        )

        self.eef_2_sub = Subscriber(
            self,
            PoseStamped,
            "/fairino_2/eef_pose",
        )

        # D405 #1

        self.rgb_1_sub = Subscriber(
            self,
            Image,
            "/camera_d405_1/color/image_rect_raw",
        )

        self.depth_1_sub = Subscriber(
            self,
            Image,
            "/camera_d405_1/aligned_depth_to_color/image_raw",
        )

        # D405 #2

        self.rgb_2_sub = Subscriber(
            self,
            Image,
            "/camera_d405_2/color/image_rect_raw",
        )

        self.depth_2_sub = Subscriber(
            self,
            Image,
            "/camera_d405_2/aligned_depth_to_color/image_raw",
        )

        # ============================================================
        # Synchronizer
        # ============================================================

        self.synchronizer = ApproximateTimeSynchronizer(
            [
                self.joint_1_sub,
                self.eef_1_sub,
                self.joint_2_sub,
                self.eef_2_sub,
                self.rgb_1_sub,
                self.depth_1_sub,
                self.rgb_2_sub,
                self.depth_2_sub,
            ],
            queue_size=30,
            slop=self.sync_slop,
        )

        self.synchronizer.registerCallback(
            self.synchronized_callback
        )

        # ============================================================
        # Services
        # ============================================================

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

        # ============================================================
        # Startup
        # ============================================================

        self.get_logger().info(
            "RMB Compact episode recorder ready"
        )

        self.get_logger().info(
            f"Output directory: {self.output_directory}"
        )

        self.get_logger().info(
            f"Target rate: {self.target_rate:.1f} Hz"
        )

        self.get_logger().info(
            f"Synchronization tolerance: "
            f"{self.sync_slop * 1000:.1f} ms"
        )

        self.get_logger().info(
            "Waiting for /start_episode"
        )

    # ================================================================
    # Buffers
    # ================================================================

    def clear_buffers(self):

        # RMB data

        self.time_data = []

        self.joint_pos_data = []
        self.joint_vel_data = []
        self.eef_pose_data = []

        self.rgb1_data = []
        self.depth1_data = []

        self.rgb2_data = []
        self.depth2_data = []

        # Synchronization diagnostics

        self.timestamp_spread_data = []

    # ================================================================
    # Episode numbering
    # ================================================================

    def get_next_episode_name(self):

        numbers = []

        for path in self.output_directory.glob(
            "episode_*.rmb"
        ):

            if not path.is_dir():
                continue

            try:
                number = int(
                    path.name
                    .replace(".rmb", "")
                    .split("_")[1]
                )

                numbers.append(number)

            except (IndexError, ValueError):
                continue

        next_number = max(
            numbers,
            default=0,
        ) + 1

        return f"episode_{next_number:06d}"

    # ================================================================
    # Timestamp conversion
    # ================================================================

    @staticmethod
    def stamp_to_sec(stamp):

        return (
            float(stamp.sec)
            +
            float(stamp.nanosec) * 1e-9
        )

    # ================================================================
    # RGB Image conversion
    # ================================================================

    @staticmethod
    def rgb_msg_to_numpy(msg):

        if msg.encoding != "rgb8":

            raise ValueError(
                f"Expected rgb8, got {msg.encoding}"
            )

        image = np.frombuffer(
            msg.data,
            dtype=np.uint8,
        )

        expected_size = (
            msg.height *
            msg.width *
            3
        )

        if image.size != expected_size:

            raise ValueError(
                f"RGB image size mismatch: "
                f"{image.size} != {expected_size}"
            )

        return image.reshape(
            msg.height,
            msg.width,
            3,
        ).copy()

    # ================================================================
    # Depth Image conversion
    # ================================================================

    @staticmethod
    def depth_msg_to_numpy(msg):

        if msg.encoding != "16UC1":

            raise ValueError(
                f"Expected 16UC1, got {msg.encoding}"
            )

        image = np.frombuffer(
            msg.data,
            dtype=np.uint16,
        )

        expected_size = (
            msg.height *
            msg.width
        )

        if image.size != expected_size:

            raise ValueError(
                f"Depth image size mismatch: "
                f"{image.size} != {expected_size}"
            )

        return image.reshape(
            msg.height,
            msg.width,
        ).copy()

    # ================================================================
    # Start episode
    # ================================================================

    def start_episode_callback(
        self,
        request,
        response,
    ):

        if self.recording:

            response.success = False

            response.message = (
                f"Already recording "
                f"{self.current_episode}"
            )

            return response

        self.current_episode = (
            self.get_next_episode_name()
        )

        self.clear_buffers()

        self.first_timestamp = None
        self.sample_count = 0

        self.recording = True

        self.get_logger().info(
            "===================================="
        )

        self.get_logger().info(
            f"Started {self.current_episode}"
        )

        self.get_logger().info(
            "Waiting for synchronized samples..."
        )

        response.success = True

        response.message = (
            f"Started {self.current_episode}"
        )

        return response

    # ================================================================
    # Synchronized callback
    # ================================================================

    def synchronized_callback(
        self,
        joint_1,
        eef_1,
        joint_2,
        eef_2,
        rgb_1,
        depth_1,
        rgb_2,
        depth_2,
    ):

        if not self.recording:
            return

        # ============================================================
        # Collect ALL timestamps
        # ============================================================

        timestamps = np.array(
            [
                self.stamp_to_sec(
                    joint_1.header.stamp
                ),

                self.stamp_to_sec(
                    eef_1.header.stamp
                ),

                self.stamp_to_sec(
                    joint_2.header.stamp
                ),

                self.stamp_to_sec(
                    eef_2.header.stamp
                ),

                self.stamp_to_sec(
                    rgb_1.header.stamp
                ),

                self.stamp_to_sec(
                    depth_1.header.stamp
                ),

                self.stamp_to_sec(
                    rgb_2.header.stamp
                ),

                self.stamp_to_sec(
                    depth_2.header.stamp
                ),
            ],
            dtype=np.float64,
        )

        # Difference between oldest and newest message
        # in this synchronized sample.

        timestamp_spread = (
            np.max(timestamps)
            -
            np.min(timestamps)
        )

        # ============================================================
        # Reference time
        #
        # Camera 1 RGB is the dataset reference clock.
        # ============================================================

        timestamp = self.stamp_to_sec(
            rgb_1.header.stamp
        )

        if self.first_timestamp is None:
            self.first_timestamp = timestamp

        episode_time = (
            timestamp -
            self.first_timestamp
        )

        # ============================================================
        # Joint positions
        # ============================================================

        joint_pos_1 = list(
            joint_1.position
        )

        joint_pos_2 = list(
            joint_2.position
        )

        if len(joint_pos_1) != 6:

            self.get_logger().warning(
                f"FR5 #1 has "
                f"{len(joint_pos_1)} joints; "
                f"expected 6"
            )

            return

        if len(joint_pos_2) != 6:

            self.get_logger().warning(
                f"FR5 #2 has "
                f"{len(joint_pos_2)} joints; "
                f"expected 6"
            )

            return

        joint_pos = (
            joint_pos_1 +
            joint_pos_2
        )

        # ============================================================
        # Joint velocities
        # ============================================================

        joint_vel_1 = list(
            joint_1.velocity
        )

        joint_vel_2 = list(
            joint_2.velocity
        )

        if len(joint_vel_1) != 6:

            joint_vel_1 = [
                np.nan
            ] * 6

        if len(joint_vel_2) != 6:

            joint_vel_2 = [
                np.nan
            ] * 6

        joint_vel = (
            joint_vel_1 +
            joint_vel_2
        )

        # ============================================================
        # EEF poses
        #
        # RMB:
        #
        # tx ty tz qw qx qy qz
        #
        # FR5-1 followed by FR5-2
        # ============================================================

        eef_pose = [

            # FR5 #1

            eef_1.pose.position.x,
            eef_1.pose.position.y,
            eef_1.pose.position.z,

            eef_1.pose.orientation.w,
            eef_1.pose.orientation.x,
            eef_1.pose.orientation.y,
            eef_1.pose.orientation.z,

            # FR5 #2

            eef_2.pose.position.x,
            eef_2.pose.position.y,
            eef_2.pose.position.z,

            eef_2.pose.orientation.w,
            eef_2.pose.orientation.x,
            eef_2.pose.orientation.y,
            eef_2.pose.orientation.z,
        ]

        # ============================================================
        # Images
        # ============================================================

        try:

            rgb1 = self.rgb_msg_to_numpy(
                rgb_1
            )

            depth1 = self.depth_msg_to_numpy(
                depth_1
            )

            rgb2 = self.rgb_msg_to_numpy(
                rgb_2
            )

            depth2 = self.depth_msg_to_numpy(
                depth_2
            )

        except ValueError as exc:

            self.get_logger().error(
                f"Image conversion failed: {exc}"
            )

            return

        # ============================================================
        # Store sample
        # ============================================================

        self.time_data.append(
            episode_time
        )

        self.joint_pos_data.append(
            joint_pos
        )

        self.joint_vel_data.append(
            joint_vel
        )

        self.eef_pose_data.append(
            eef_pose
        )

        self.rgb1_data.append(
            rgb1
        )

        self.depth1_data.append(
            depth1
        )

        self.rgb2_data.append(
            rgb2
        )

        self.depth2_data.append(
            depth2
        )

        self.timestamp_spread_data.append(
            timestamp_spread
        )

        # ============================================================
        # Statistics
        # ============================================================

        self.sample_count += 1

        if self.sample_count % 15 == 0:

            avg_spread = (
                np.mean(
                    self.timestamp_spread_data[-15:]
                )
                * 1000.0
            )

            max_spread = (
                np.max(
                    self.timestamp_spread_data[-15:]
                )
                * 1000.0
            )

            self.get_logger().info(
                f"Recorded {self.sample_count} samples "
                f"({episode_time:.2f} s) | "
                f"sync avg={avg_spread:.1f} ms, "
                f"max={max_spread:.1f} ms"
            )

            # Warn when accepted samples are getting
            # close to the synchronizer limit.

            if max_spread > (
                self.sync_slop * 0.8 * 1000.0
            ):

                self.get_logger().warning(
                    "Timestamp spread is close to "
                    "the synchronization tolerance."
                )

    # ================================================================
    # Stop episode
    # ================================================================

    def stop_episode_callback(
        self,
        request,
        response,
    ):

        if not self.recording:

            response.success = False
            response.message = (
                "No episode is currently recording"
            )

            return response

        self.recording = False

        episode_name = (
            self.current_episode
        )

        sample_count = len(
            self.time_data
        )

        if sample_count == 0:

            self.get_logger().error(
                "No synchronized samples received."
            )

            response.success = False

            response.message = (
                "No synchronized samples received"
            )

            self.current_episode = None

            return response

        # ============================================================
        # Episode directory
        # ============================================================

        episode_directory = (
            self.output_directory /
            f"{episode_name}.rmb"
        )

        # Safety check

        if episode_directory.exists():

            shutil.rmtree(
                episode_directory
            )

        episode_directory.mkdir(
            parents=True
        )

        # ============================================================
        # Save
        # ============================================================

        try:

            self.write_rmb(
                episode_directory
            )

        except Exception as exc:

            self.get_logger().error(
                f"Failed to save RMB episode: {exc}"
            )

            # Remove incomplete RMB directory.

            if episode_directory.exists():

                shutil.rmtree(
                    episode_directory
                )

            response.success = False

            response.message = (
                f"Failed to save episode: {exc}"
            )

            return response

        # ============================================================
        # Statistics
        # ============================================================

        duration = float(
            self.time_data[-1]
        )

        effective_rate = 0.0

        if duration > 0:

            effective_rate = (
                (sample_count - 1)
                /
                duration
            )

        spreads = np.asarray(
            self.timestamp_spread_data
        )

        mean_spread_ms = (
            float(np.mean(spreads))
            * 1000.0
        )

        max_spread_ms = (
            float(np.max(spreads))
            * 1000.0
        )

        # ============================================================
        # Final report
        # ============================================================

        self.get_logger().info(
            "===================================="
        )

        self.get_logger().info(
            f"Saved {episode_name}.rmb"
        )

        self.get_logger().info(
            f"Samples: {sample_count}"
        )

        self.get_logger().info(
            f"Duration: {duration:.3f} s"
        )

        self.get_logger().info(
            f"Effective rate: "
            f"{effective_rate:.2f} Hz"
        )

        self.get_logger().info(
            f"Mean timestamp spread: "
            f"{mean_spread_ms:.2f} ms"
        )

        self.get_logger().info(
            f"Maximum timestamp spread: "
            f"{max_spread_ms:.2f} ms"
        )

        self.get_logger().info(
            f"RMB directory: "
            f"{episode_directory}"
        )

        # ============================================================
        # Synchronization health
        # ============================================================

        if effective_rate < (
            self.target_rate * 0.8
        ):

            self.get_logger().warning(
                "Effective synchronized rate is "
                "significantly below target rate."
            )

            self.get_logger().warning(
                "This may indicate timestamp "
                "misalignment or dropped messages."
            )

        if max_spread_ms > (
            self.sync_slop * 0.8 * 1000.0
        ):

            self.get_logger().warning(
                "Maximum timestamp spread is close "
                "to sync_slop."
            )

        response.success = True

        response.message = (
            f"Saved {episode_name}.rmb: "
            f"{sample_count} samples, "
            f"{effective_rate:.2f} Hz, "
            f"mean sync {mean_spread_ms:.1f} ms, "
            f"max sync {max_spread_ms:.1f} ms"
        )

        self.current_episode = None

        return response

    # ================================================================
    # RMB Compact writer
    # ================================================================

    def write_rmb(
        self,
        episode_directory,
    ):

        self.get_logger().info(
            "Preparing RMB Compact data..."
        )

        # ============================================================
        # Robot arrays
        # ============================================================

        time_array = np.asarray(
            self.time_data,
            dtype=np.float64,
        )

        joint_pos_array = np.asarray(
            self.joint_pos_data,
            dtype=np.float64,
        )

        joint_vel_array = np.asarray(
            self.joint_vel_data,
            dtype=np.float64,
        )

        eef_pose_array = np.asarray(
            self.eef_pose_data,
            dtype=np.float64,
        )

        # ============================================================
        # Images
        # ============================================================

        rgb1_array = np.asarray(
            self.rgb1_data,
            dtype=np.uint8,
        )

        depth1_array = np.asarray(
            self.depth1_data,
            dtype=np.uint16,
        )

        rgb2_array = np.asarray(
            self.rgb2_data,
            dtype=np.uint8,
        )

        depth2_array = np.asarray(
            self.depth2_data,
            dtype=np.uint16,
        )

        # ============================================================
        # main.rmb.hdf5
        # ============================================================

        hdf5_path = (
            episode_directory /
            "main.rmb.hdf5"
        )

        self.get_logger().info(
            "Writing main.rmb.hdf5..."
        )

        with h5py.File(
            hdf5_path,
            "w",
        ) as h5file:

            h5file.create_dataset(
                "time",
                data=time_array,
            )

            h5file.create_dataset(
                "measured_joint_pos",
                data=joint_pos_array,
            )

            h5file.create_dataset(
                "measured_joint_vel",
                data=joint_vel_array,
            )

            h5file.create_dataset(
                "measured_eef_pose",
                data=eef_pose_array,
            )

            # --------------------------------------------------------
            # RMB metadata
            # --------------------------------------------------------

            h5file.attrs[
                "format"
            ] = "RmbData-Compact"

            h5file.attrs[
                "camera_names"
            ] = np.array(
                [
                    "camera_d405_1",
                    "camera_d405_2",
                ],
                dtype=h5py.string_dtype(
                    encoding="utf-8"
                ),
            )

            # Project-specific useful metadata

            h5file.attrs[
                "target_rate_hz"
            ] = self.target_rate

            h5file.attrs[
                "num_arms"
            ] = 2

            h5file.attrs[
                "num_cameras"
            ] = 2

            h5file.attrs[
                "joint_dim"
            ] = 12

            h5file.attrs[
                "eef_pose_dim"
            ] = 14

        # ============================================================
        # RGB camera 1
        # ============================================================

        rgb1_path = (
            episode_directory /
            "camera_d405_1_rgb_image.rmb.mp4"
        )

        self.get_logger().info(
            "Writing camera 1 RGB video..."
        )

        videoio.videosave(
            str(rgb1_path),
            rgb1_array,
        )

        # ============================================================
        # Depth camera 1
        #
        # D405 16UC1 is already millimetres.
        #
        # RMB videoio depth representation stores uint16 values
        # quantized in millimetres, so DO NOT multiply by 1000 here.
        # ============================================================

        depth1_path = (
            episode_directory /
            "camera_d405_1_depth_image.rmb.mp4"
        )

        self.get_logger().info(
            "Writing camera 1 depth video..."
        )

        videoio.uint16save(
            str(depth1_path),
            depth1_array,
        )

        # ============================================================
        # RGB camera 2
        # ============================================================

        rgb2_path = (
            episode_directory /
            "camera_d405_2_rgb_image.rmb.mp4"
        )

        self.get_logger().info(
            "Writing camera 2 RGB video..."
        )

        videoio.videosave(
            str(rgb2_path),
            rgb2_array,
        )

        # ============================================================
        # Depth camera 2
        # ============================================================

        depth2_path = (
            episode_directory /
            "camera_d405_2_depth_image.rmb.mp4"
        )

        self.get_logger().info(
            "Writing camera 2 depth video..."
        )

        videoio.uint16save(
            str(depth2_path),
            depth2_array,
        )

        self.get_logger().info(
            "RMB Compact write complete"
        )


# ====================================================================
# Main
# ====================================================================

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