cmake_minimum_required(VERSION 3.8)
project(mid360_map_localizer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters io registration)
find_package(pcl_conversions REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_definitions(${PCL_DEFINITIONS})
include_directories(${PCL_INCLUDE_DIRS})

add_executable(mid360_scan_matcher_node src/mid360_scan_matcher_node.cpp)
ament_target_dependencies(
  mid360_scan_matcher_node
  geometry_msgs
  nav_msgs
  pcl_conversions
  rclcpp
  sensor_msgs
  std_msgs
)
target_link_libraries(mid360_scan_matcher_node ${PCL_LIBRARIES})

install(TARGETS mid360_scan_matcher_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch docs DESTINATION share/${PROJECT_NAME})
install(FILES README.md DESIGN_NOTES.md DESTINATION share/${PROJECT_NAME})

ament_package()
