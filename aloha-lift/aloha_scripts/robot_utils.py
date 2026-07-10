import numpy as np
import time
from constants import DT, LIFT_DIM

try:
    import IPython
    e = IPython.embed
except ImportError:
    pass


class ImageRecorder:
    """Subscribes to ROS image topics for 3 cameras."""

    def __init__(self, init_node=True, is_debug=False):
        from collections import deque
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        self.is_debug = is_debug
        self.bridge = CvBridge()
        self.camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
        if init_node:
            rospy.init_node('image_recorder', anonymous=True)
        for cam_name in self.camera_names:
            setattr(self, f'{cam_name}_image', None)
            setattr(self, f'{cam_name}_secs', None)
            setattr(self, f'{cam_name}_nsecs', None)
            if cam_name == 'cam_high':
                callback_func = self.image_cb_cam_high
            elif cam_name == 'cam_left_wrist':
                callback_func = self.image_cb_cam_left_wrist
            elif cam_name == 'cam_right_wrist':
                callback_func = self.image_cb_cam_right_wrist
            else:
                raise NotImplementedError
            rospy.Subscriber(f"/usb_{cam_name}/image_raw", Image, callback_func)
            if self.is_debug:
                setattr(self, f'{cam_name}_timestamps', deque(maxlen=50))
        time.sleep(0.5)

    def image_cb(self, cam_name, data):
        setattr(self, f'{cam_name}_image', self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough'))
        setattr(self, f'{cam_name}_secs', data.header.stamp.secs)
        setattr(self, f'{cam_name}_nsecs', data.header.stamp.nsecs)
        if self.is_debug:
            getattr(self, f'{cam_name}_timestamps').append(data.header.stamp.secs + data.header.stamp.nsecs * 1e-9)

    def image_cb_cam_high(self, data):
        return self.image_cb('cam_high', data)

    def image_cb_cam_left_wrist(self, data):
        return self.image_cb('cam_left_wrist', data)

    def image_cb_cam_right_wrist(self, data):
        return self.image_cb('cam_right_wrist', data)

    def get_images(self):
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = getattr(self, f'{cam_name}_image')
        return image_dict

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)
        for cam_name in self.camera_names:
            image_freq = 1 / dt_helper(getattr(self, f'{cam_name}_timestamps'))
            print(f'{cam_name} {image_freq=:.2f}')
        print()


class Recorder:
    """
    Records joint states and commands from ROS topics.
    Uses a single robot_namespace (e.g. 'puppet') for the custom robot.
    """

    def __init__(self, robot_namespace='puppet', init_node=True, is_debug=False):
        from collections import deque
        import rospy
        from sensor_msgs.msg import JointState
        from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand

        self.secs = None
        self.nsecs = None
        self.qpos = None
        self.qvel = None
        self.effort = None
        self.arm_command = None
        self.gripper_command = None
        self.is_debug = is_debug

        if init_node:
            rospy.init_node('recorder', anonymous=True)
        rospy.Subscriber(f"/{robot_namespace}/joint_states", JointState, self.joint_state_cb)
        rospy.Subscriber(f"/{robot_namespace}/commands/joint_group", JointGroupCommand, self.arm_commands_cb)
        rospy.Subscriber(f"/{robot_namespace}/commands/joint_single", JointSingleCommand, self.gripper_commands_cb)
        if self.is_debug:
            self.joint_timestamps = deque(maxlen=50)
            self.arm_command_timestamps = deque(maxlen=50)
            self.gripper_command_timestamps = deque(maxlen=50)
        time.sleep(0.1)

    def joint_state_cb(self, data):
        self.qpos = data.position
        self.qvel = data.velocity
        self.effort = data.effort
        if self.is_debug:
            self.joint_timestamps.append(time.time())

    def arm_commands_cb(self, data):
        self.arm_command = data.cmd
        if self.is_debug:
            self.arm_command_timestamps.append(time.time())

    def gripper_commands_cb(self, data):
        self.gripper_command = data.cmd
        if self.is_debug:
            self.gripper_command_timestamps.append(time.time())

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        joint_freq = 1 / dt_helper(self.joint_timestamps)
        arm_command_freq = 1 / dt_helper(self.arm_command_timestamps)
        gripper_command_freq = 1 / dt_helper(self.gripper_command_timestamps)
        print(f'{joint_freq=:.2f}\n{arm_command_freq=:.2f}\n{gripper_command_freq=:.2f}\n')


# ---------- Robot movement helpers ----------

def get_joint_positions(bot):
    """Get current joint positions from an InterbotixManipulatorXS instance."""
    return list(bot.arm.core.joint_states.position[:6])


def get_gripper_position(bot):
    """Get current gripper position from an InterbotixManipulatorXS instance."""
    return bot.gripper.core.joint_states.position[6]


def move_arms(bot_list, target_pose_list, move_time=1):
    """Move robot arms to target positions over a specified duration."""
    num_steps = int(move_time / DT)
    curr_pose_list = [get_joint_positions(bot) for bot in bot_list]
    traj_list = [np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            bot.arm.set_joint_positions(traj_list[bot_id][t], blocking=False)
        time.sleep(DT)


def move_grippers(bot_list, target_pose_list, move_time):
    """Move robot grippers to target positions."""
    from interbotix_xs_msgs.msg import JointSingleCommand
    gripper_command = JointSingleCommand(name="gripper")
    num_steps = int(move_time / DT)
    curr_pose_list = [get_gripper_position(bot) for bot in bot_list]
    traj_list = [np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            gripper_command.cmd = traj_list[bot_id][t]
            bot.gripper.core.pub_single.publish(gripper_command)
        time.sleep(DT)


def torque_on(bot):
    bot.dxl.robot_torque_enable("group", "arm", True)
    bot.dxl.robot_torque_enable("single", "gripper", True)


def torque_off(bot):
    bot.dxl.robot_torque_enable("group", "arm", False)
    bot.dxl.robot_torque_enable("single", "gripper", False)


def setup_puppet_bot(bot):
    bot.dxl.robot_reboot_motors("single", "gripper", True)
    bot.dxl.robot_set_operating_modes("group", "arm", "position")
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_on(bot)


def setup_master_bot(bot):
    bot.dxl.robot_set_operating_modes("group", "arm", "pwm")
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_off(bot)
