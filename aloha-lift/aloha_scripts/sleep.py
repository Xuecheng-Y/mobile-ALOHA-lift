"""Move robot to sleep/homing position."""

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from robot_utils import move_arms, torque_on

def main():
    puppet_bot = InterbotixManipulatorXS(
        robot_model="vx300s", group_name="arm", gripper_name="gripper",
        robot_name='puppet', init_node=True
    )
    torque_on(puppet_bot)

    sleep_position = (0, -1.7, 1.55, 0.12, 0.65, 0)  # adjust for your robot
    move_arms([puppet_bot], [sleep_position], move_time=2)
    print('Robot moved to sleep position.')

if __name__ == '__main__':
    main()
