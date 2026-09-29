import math


class FairinoInterface:
    """Hardware interface for a FAIRINO robot."""

    def __init__(self, ip):
        self.ip = ip
        self.robot = None

    def connect(self):
        from fairino import Robot

        self.robot = Robot.RPC(self.ip)

    def get_joint_positions(self):
        """Return joint positions in radians."""

        if self.robot is None:
            raise RuntimeError("Robot is not connected")

        error, joints_deg = self.robot.GetActualJointPosDegree()

        if error != 0:
            raise RuntimeError(
                f"GetActualJointPosDegree failed with error {error}"
            )

        return [math.radians(q) for q in joints_deg]