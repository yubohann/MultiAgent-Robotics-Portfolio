#!/usr/bin/env bash
# Inspect synthetic input and target-gate topics with bounded cleanup.

set -eo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
log_file="${TMPDIR:-/tmp}/robocon_synthetic_topic_probe.log"

source /opt/ros/humble/setup.bash
source "${workspace_root}/install/setup.bash"

ros2 launch robocon_game_supervisor synthetic_competition.launch.py >"${log_file}" 2>&1 &
launch_pid=$!
cleanup() {
  kill "${launch_pid}" 2>/dev/null || true
  wait "${launch_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ready_topic_found=false
for attempt in $(seq 1 100); do
  if ros2 topic list | grep -Fxq "/mid360/preflight_ready"; then
    ready_topic_found=true
    break
  fi
  sleep 0.1
done
if [[ "${ready_topic_found}" != true ]]; then
  cat "${log_file}"
  exit 1
fi
ros2 topic info /mid360/preflight_ready --verbose
ros2 topic echo --once /mid360/preflight_ready
ros2 topic info /robocon/perception/target_valid --verbose
ros2 topic echo --once /robocon/perception/target_valid
