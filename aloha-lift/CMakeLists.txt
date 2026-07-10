cmake_minimum_required(VERSION 3.0.2)
project(aloha_lift)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  interbotix_xs_msgs
  cv_bridge
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
