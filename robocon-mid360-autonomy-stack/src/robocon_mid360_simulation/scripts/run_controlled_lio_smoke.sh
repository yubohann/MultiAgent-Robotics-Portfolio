#!/usr/bin/env bash
# Start one bounded Gazebo LIO run and preserve all evidence in a unique directory.

set -eo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo "usage: $0 RUN_DIRECTORY [DURATION_SEC] [LIDAR_DOWNSAMPLE] [WORLD_FILE]" >&2
  exit 64
fi

run_dir="$1"
duration_sec="${2:-60}"
lidar_downsample="${3:-1}"
world_file="${4:-}"
scene_name="public-candidate-not-official"
profile_name="30000-ray-controlled-motion"
if [[ "$world_file" == *"open_field_degraded.world" ]]; then
  scene_name="open_field_degraded"
  profile_name="open-field-degraded"
elif [[ "$world_file" == *"indoor_competition_candidate.world" ]]; then
  scene_name="indoor_competition_candidate"
  profile_name="indoor-competition-candidate"
fi
workspace="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
mkdir -p "$run_dir"

source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
set -u
ros_domain_id="${ROS_DOMAIN_ID:-42}"
if ! [[ "$ros_domain_id" =~ ^[0-9]+$ ]] || (( ros_domain_id < 0 || ros_domain_id > 232 )); then
  echo "ROS_DOMAIN_ID must be an integer in [0, 232]" > "$run_dir/failure_reason.txt"
  exit 64
fi
export ROS_DOMAIN_ID="$ros_domain_id"

world_arg=""
if [[ -n "$world_file" ]]; then
  world_arg="world:=${world_file}"
fi

cat > "$run_dir/launch_command.txt" <<EOF
ros2 launch robocon_mid360_simulation gazebo_mid360_lio.launch.py use_gui:=false lidar_samples:=30000 lidar_downsample:=${lidar_downsample} ${world_arg}
EOF

setsid ros2 launch robocon_mid360_simulation gazebo_mid360_lio.launch.py \
  use_gui:=false lidar_samples:=30000 lidar_downsample:="$lidar_downsample" ${world_arg} > "$run_dir/launch.stdout_stderr.log" 2>&1 &
launch_pid=$!

cleanup() {
  if kill -0 "$launch_pid" 2>/dev/null; then
    kill -INT -- "-$launch_pid" 2>/dev/null || true
    for attempt in $(seq 1 10); do
      kill -0 "$launch_pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$launch_pid" 2>/dev/null; then
      kill -TERM -- "-$launch_pid" 2>/dev/null || true
      sleep 2
    fi
    if kill -0 "$launch_pid" 2>/dev/null; then
      kill -KILL -- "-$launch_pid" 2>/dev/null || true
    fi
    wait "$launch_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# The recorder waits for multiple valid packets itself. Do not use a separate
# wall-clock probe here: Gazebo model spawn and DDS discovery can take longer
# than the sensor-free launch phase on a mounted Windows workspace.
python3 "$workspace/src/robocon_mid360_simulation/scripts/controlled_lio_run.py" \
  --run-dir "$run_dir" --duration-sec "$duration_sec" --lidar-samples 30000 \
  --lidar-downsample "$lidar_downsample" --scene "$scene_name" --profile "$profile_name" \
  --ready-timeout-sec 180 --required-lidar-packets 3 --required-imu-packets 10 \
  --max-wall-sec 480 > "$run_dir/recorder.stdout_stderr.log" 2>&1 &
recorder_pid=$!

while kill -0 "$recorder_pid" 2>/dev/null; do
  {
    date --iso-8601=seconds
    ps -o pid,ppid,pcpu,pmem,rss,stat,comm -p "$launch_pid" --forest
    ps -C gzserver -C fastlio_mapping -C python3 -o pid,ppid,pcpu,pmem,rss,stat,comm,args || true
    free -h
    df -h "$run_dir"
  } >> "$run_dir/resource_snapshot.log"
  sleep 1
done

wait "$recorder_pid"
grep -E 'Too few input point cloud|No point|No Effective Points|Initialize the map kdtree' \
  "$run_dir/launch.stdout_stderr.log" > "$run_dir/fastlio_events.log" || true
