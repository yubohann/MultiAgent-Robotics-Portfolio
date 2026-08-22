cmake_minimum_required(VERSION 3.10)
project(robocon_mid360_simulation)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake_auto REQUIRED)
find_package(Boost REQUIRED COMPONENTS chrono)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)

include_directories(include ${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})

ament_auto_add_library(robocon_mid360_livox SHARED
  src/livox_points_plugin.cpp
  src/livox_ode_multiray_shape.cpp
)
ament_target_dependencies(robocon_mid360_livox
  gazebo_dev gazebo_msgs gazebo_ros livox_ros_driver2 rclcpp sensor_msgs
)
target_link_libraries(robocon_mid360_livox
  ${GAZEBO_LIBRARIES} RayPlugin GpuRayPlugin ${Boost_LIBRARIES}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(simulation_assets test/test_simulation_assets.py)
endif()

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch models scan_mode urdf worlds config DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/controlled_lio_run.py scripts/depth_visualizer.py scripts/fixed_map_localization_smoke.py scripts/fixed_map_interlock_integration.py scripts/localization_recovery_interlock_smoke.py scripts/probe_simulation_streams.py scripts/basketball_demo_controller.py scripts/run_controlled_lio_smoke.sh scripts/run_fixed_map_localization_smoke.sh scripts/run_fixed_map_interlock_integration.sh scripts/run_localization_recovery_interlock_smoke.sh scripts/run_simulation_mapping.sh DESTINATION lib/${PROJECT_NAME})
install(FILES scripts/run_contract.py DESTINATION lib/${PROJECT_NAME})
install(FILES LICENSE_UPSTREAM_PB_RM_SIMULATION.md LICENSE_UPSTREAM_FIELD_ASSET.md
  DESTINATION share/${PROJECT_NAME})

ament_auto_package()
