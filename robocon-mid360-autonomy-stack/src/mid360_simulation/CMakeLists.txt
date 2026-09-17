cmake_minimum_required(VERSION 3.10)
project(mid360_simulation)

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
install(PROGRAMS scripts/python/controlled_lio_run.py scripts/python/depth_visualizer.py scripts/python/fixed_map_localization_smoke.py scripts/python/fixed_map_interlock_integration.py scripts/python/localization_recovery_interlock_smoke.py scripts/python/probe_simulation_streams.py scripts/python/basketball_demo_controller.py scripts/shell/run_controlled_lio_smoke.sh scripts/shell/run_fixed_map_localization_smoke.sh scripts/shell/run_fixed_map_interlock_integration.sh scripts/shell/run_localization_recovery_interlock_smoke.sh scripts/shell/run_simulation_mapping.sh DESTINATION lib/${PROJECT_NAME})
install(FILES scripts/python/run_contract.py DESTINATION lib/${PROJECT_NAME})
install(FILES LICENSE_UPSTREAM_PB_RM_SIMULATION.md LICENSE_UPSTREAM_FIELD_ASSET.md
  DESTINATION share/${PROJECT_NAME})

ament_auto_package()
