#!/usr/bin/env bash
# Run a bounded RGB-D content/timing measurement against the Gazebo candidate scene.
set -o pipefail

workspace="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
run_dir="${1:-$workspace/runs/rgbd_quality_smoke_$(date +%Y%m%d_%H%M%S)}"
world="${WORLD:-$workspace/install/robocon_mid360_simulation/share/robocon_mid360_simulation/worlds/indoor_competition_candidate.world}"
mkdir -p "$run_dir"
source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
set -euo pipefail
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-204}"

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
for _ in $(seq 1 90); do
  if ros2 topic type /simulated_rgbd_camera/image_raw 2>/dev/null | grep -q 'sensor_msgs/msg/Image'; then
    ready=1
    break
  fi
  sleep 1
done
if (( ready == 0 )); then
  printf 'RGBD_QUALITY_READINESS_TIMEOUT\n' >"$run_dir/failure_reason.txt"
  exit 1
fi

python3 "$workspace/tools/rgbd_quality_probe.py" --run-dir "$run_dir" --duration-sec "${RGBD_DURATION_SEC:-12}" --max-wall-sec "${RGBD_MAX_WALL_SEC:-60}"
sha256sum "$run_dir/rgbd_quality_summary.json" >"$run_dir/manifest.sha256"
cat "$run_dir/rgbd_quality_summary.json"
