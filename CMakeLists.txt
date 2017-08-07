cmake_minimum_required(VERSION 2.8)
project( stereo_visual_odometry )
find_package( OpenCV REQUIRED )
#find_package(catkin REQUIRED COMPONENTS
#  cv_bridge
#  tf
#  nav_msgs
#  sensor_msgs
#  image_transport
#  roscpp
#  rospy
#  std_msgs
#)

include_directories(  #${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS} 
  /home/anoop/eigen/ )

#add_subdirectory(five-point-nister)

file(GLOB viso
	"src/poseEstimationRansac.cpp"
	"src/main.cpp"
	"src/SVO.cpp"
    "src/SVO.h"
    "src/SVO2.cpp"
    "fast-cpp-csv-parser/csv.h"
)
add_executable( svo ${viso} )
set_target_properties(svo PROPERTIES COMPILE_FLAGS "-std=c++11 -w -g -O3") 
target_link_libraries( svo ${OpenCV_LIBS} )  #five-point-nister 
#${catkin_LIBRARIES} 
