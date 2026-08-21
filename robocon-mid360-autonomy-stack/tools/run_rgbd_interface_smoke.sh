#!/usr/bin/env bash
# Verify the optional Gazebo RGB-D topics without running a quality experiment.
set -o pipefail

workspace="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
run_dir="${1:-$workspace/runs/rgbd_interface_smoke_$(date +%Y%m%d_%H%M%S)}"
world="${WORLD:-$workspace/install/robocon_mid360_simulation/share/robocon_mid360_simulation/worlds/indoor_competition_candidate.world}"
mkdir -p "$run_dir"

source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
set -euo pipefail
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-203}"

setsid ros2 launch robocon_mid360_simulation gazebo_mid360_candidate.launch.py \
  use_gui:=false enable_rgbd:=true lidar_samples:=2000 lidar_downsample:=1 \
  world:="$world" >"$run_dir/gazebo.log" 2>&1 &
launch_pid=$!

cleanup() {
  if kill -0 "$launch_pid" 2>/dev/null; then
    kill -INT -- "-$launch_pid" 2>/dev/null || true
    sleep 2
    kill -TERM -- "-$launch_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ready=0
for _ in $(seq 1 45); do
  if ros2 topic type /simulated_rgbd_camera/image_raw 2>/dev/null | grep -q 'sensor_msgs/msg/Image'; then
    ready=1
    break
  fi
  sleep 1
done
if (( ready == 0 )); then
  printf 'RGBD_READINESS_TIMEOUT\n' >"$run_dir/failure_reason.txt"
  exit 1
fi

for topic in /simulated_rgbd_camera/image_raw \
            /simulated_rgbd_camera/depth/image_raw \
            /simulated_rgbd_camera/camera_info; do
  safe_name="$(printf '%s' "$topic" | tr '/' '_')"
  ros2 topic type "$topic" >"$run_dir/${safe_name}.type"
  timeout --signal=INT --kill-after=3s 6s ros2 topic hz "$topic" \
    >"$run_dir/${safe_name}.hz" 2>&1 || true
done
ros2 topic echo --once /simulated_rgbd_camera/image_raw --field header \
  >"$run_dir/color_header.txt" 2>"$run_dir/color_header.err" || true
printf 'evidence=gazebo_simulation\nray_count=2000\nenable_rgbd=true\n' \
  >"$run_dir/manifest.txt"
printf 'RGB-D topic smoke passed; no detector-quality claim.\n' \
  >"$run_dir/summary.txt"
printf 'rgbd smoke passed: %s\n' "$run_dir"
