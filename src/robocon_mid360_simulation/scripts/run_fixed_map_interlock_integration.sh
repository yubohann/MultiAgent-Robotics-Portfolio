#!/usr/bin/env bash
# Run the real Gazebo -> FAST-LIO2 -> fixed-map -> supervisor integration.
set -eo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 OUTPUT_DIR MAP_FILE [ROS_DOMAIN_ID]" >&2
  exit 64
fi
run_dir="$1"
map_file="$2"
ros_domain_id="${3:-221}"
workspace="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
mkdir -p "$run_dir"
if [[ ! -f "$map_file" ]]; then
  echo "integration_failed: map file does not exist: $map_file" > "$run_dir/failure_reason.txt"
  exit 64
fi
source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
set -u
export ROS_DOMAIN_ID="$ros_domain_id"
world="$workspace/install/robocon_mid360_simulation/share/robocon_mid360_simulation/worlds/indoor_competition_candidate.world"
pcd="$run_dir/online_map.pcd"
metadata="$run_dir/online_map.yaml"

setsid ros2 launch robocon_mid360_simulation gazebo_fixed_map_competition.launch.py \
  use_gui:=false lidar_samples:=30000 lidar_downsample:=1 \
  world:="$world" map_file:="$map_file" \
  map_output_file:="$pcd" metadata_output_file:="$metadata" \
  task_id:=gazebo-fixed-map-interlock > "$run_dir/launch.log" 2>&1 &
launch_pid=$!

cleanup_group() {
  if kill -0 "$launch_pid" 2>/dev/null; then
    kill -INT -- "-$launch_pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$launch_pid" 2>/dev/null || break
      sleep 1
    done
    kill -TERM -- "-$launch_pid" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-$launch_pid" 2>/dev/null || true
  fi
}
cleanup() {
  cleanup_group
}
trap cleanup EXIT INT TERM

set +e
timeout 300s python3 "$workspace/src/robocon_mid360_simulation/scripts/fixed_map_interlock_integration.py" \
  --run-dir "$run_dir" --map-file "$map_file" --timeout-sec 270 \
  > "$run_dir/driver.log" 2>&1
driver_status=$?
set -e
cleanup
wait "$launch_pid" 2>/dev/null || true
if [[ "$driver_status" -ne 0 ]]; then
  echo "integration_failed: driver exited $driver_status; inspect integration_summary.json" > "$run_dir/failure_reason.txt"
  exit 1
fi
