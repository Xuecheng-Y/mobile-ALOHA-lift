import time
import numpy as np
import collections
import dm_env

from constants import DT, LIFT_DIM
from robot_utils import Recorder, ImageRecorder

try:
    import IPython
    e = IPython.embed
except ImportError:
    pass


class RealEnv:
    """
    Environment for a real lift-axis robot.
    Action space:      [joint_0, ..., joint_28]           # 29-dim absolute joint positions

    Observation space: {"qpos":  [joint_0, ..., joint_28]  # 29-dim joint positions
                        "qvel":  [joint_0, ..., joint_28]  # 29-dim joint velocities
                        "effort":[joint_0, ..., joint_28]  # 29-dim joint efforts
                        "images": {"cam_high": (480x640x3),
                                   "cam_left_wrist": (480x640x3),
                                   "cam_right_wrist": (480x640x3)}}
    """

    def __init__(self, init_node, robot_namespace='puppet'):
        """
        Args:
            init_node: whether to initialize a ROS node
            robot_namespace: ROS namespace for the robot (e.g. 'puppet')
        """
        self.robot_namespace = robot_namespace
        self.recorder = Recorder(robot_namespace, init_node=init_node)
        self.image_recorder = ImageRecorder(init_node=False)

        # Robot command publishers (customize to your robot)
        self._init_command_publishers(init_node)

    def _init_command_publishers(self, init_node):
        """Initialize ROS publishers for sending joint commands.
        Override this for your specific robot hardware."""
        import rospy
        from interbotix_xs_msgs.msg import JointGroupCommand

        if init_node:
            rospy.init_node('real_env', anonymous=True)
        self.arm_pub = rospy.Publisher(
            f"/{self.robot_namespace}/commands/joint_group",
            JointGroupCommand, queue_size=10
        )
        time.sleep(0.1)

    def get_qpos(self):
        return np.array(self.recorder.qpos)

    def get_qvel(self):
        return np.array(self.recorder.qvel)

    def get_effort(self):
        return np.array(self.recorder.effort)

    def get_images(self):
        return self.image_recorder.get_images()

    def get_observation(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos()
        obs['qvel'] = self.get_qvel()
        obs['effort'] = self.get_effort()
        obs['images'] = self.get_images()
        return obs

    def get_reward(self):
        return 0

    def reset(self, fake=False):
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

    def step(self, action):
        """
        Execute a 29-dim action on the robot.

        Args:
            action: numpy array of shape (29,), absolute joint positions
        """
        from interbotix_xs_msgs.msg import JointGroupCommand

        cmd = JointGroupCommand()
        cmd.name = "arm"
        cmd.cmd = list(action)
        self.arm_pub.publish(cmd)
        time.sleep(DT)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())


def make_real_env(init_node, robot_namespace='puppet'):
    env = RealEnv(init_node, robot_namespace)
    return env
