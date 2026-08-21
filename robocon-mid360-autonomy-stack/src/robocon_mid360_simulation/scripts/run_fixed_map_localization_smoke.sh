#!/usr/bin/env bash
# Validate /initialpose -> ICP correction -> map lock against a frozen Gazebo map.
set -eo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
  echo "usage: $0 RUN_DIRECTORY FROZEN_MAP_PCD [DURATION_SEC] [READINESS_TIMEOUT_SEC] [ENABLE_GROUND_TRUTH]" >&2
  exit 64
fi

run_dir="$1"
map_file="$2"
duration_sec="${3:-25}"
readiness_timeout_sec="${4:-180}"
enable_ground_truth="${5:-true}"
workspace="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
mapping_params="$workspace/install/robocon_mid360_simulation/share/robocon_mid360_simulation/config/fast_lio_mapping_simulation.yaml"
world_file="$workspace/install/robocon_mid360_simulation/share/robocon_mid360_simulation/worlds/indoor_competition_candidate.world"
mkdir -p "$run_dir"

source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
set -u
ros_domain_id="${ROS_DOMAIN_ID:-42}"
if ! [[ "$ros_domain_id" =~ ^[0-9]+$ ]] || (( ros_domain_id < 0 || ros_domain_id > 232 )); then
  echo "fixed_map_smoke_failed: ROS_DOMAIN_ID must be an integer in [0, 232]" > "$run_dir/failure_reason.txt"
  exit 64
fi
if [[ ! -s "$map_file" ]]; then
  echo "fixed_map_smoke_failed: missing frozen map $map_file" > "$run_dir/failure_reason.txt"
  exit 64
fi
if pgrep -f 'gzserver .*robocon_mid360_simulation' >/dev/null 2>&1; then
  echo "fixed_map_smoke_failed: an existing Gazebo server would contaminate this run" > "$run_dir/failure_reason.txt"
  exit 1
fi
export ROS_DOMAIN_ID="$ros_domain_id"
sha256sum "$map_file" > "$run_dir/map_input.sha256"

setsid ros2 launch robocon_mid360_simulation gazebo_mid360_lio.launch.py \
  use_gui:=false lidar_samples:=30000 lidar_downsample:=1 \
  fast_lio_parameters:="$mapping_params" enable_ground_truth:="$enable_ground_truth" \
  world:="$world_file" > "$run_dir/gazebo_lio.log" 2>&1 &
lio_pid=$!
setsid ros2 launch mid360_map_localizer fixed_map_localization.launch.py \
  map_file:="$map_file" scan_topic:=/cloud_registered \
  odom_topic:=/mid360/local_odometry > "$run_dir/localizer.log" 2>&1 &
localizer_pid=$!

cleanup_group() {
  local pid="$1"
  if kill -0 "$pid" 2>/dev/null; then
    kill -INT -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 12); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -TERM -- "-$pid" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
}
cleanup() {
  cleanup_group "$localizer_pid"
  cleanup_group "$lio_pid"
}
trap cleanup EXIT INT TERM

set +e
timeout 240s python3 "$workspace/src/robocon_mid360_simulation/scripts/fixed_map_localization_smoke.py" \
  --run-dir "$run_dir" --map-file "$map_file" --duration-sec "$duration_sec" \
  --readiness-timeout-sec "$readiness_timeout_sec" \
  --truth-topic /simulation/ground_truth/odom > "$run_dir/runner.log" 2>&1
runner_status=$?
set -e
cleanup
wait "$localizer_pid" 2>/dev/null || true
wait "$lio_pid" 2>/dev/null || true

if [[ "$runner_status" -ne 0 ]]; then
  echo "fixed_map_smoke_failed: runner exited $runner_status" > "$run_dir/failure_reason.txt"
  exit 1
fi
