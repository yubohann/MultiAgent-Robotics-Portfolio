#!/usr/bin/env bash
# Verify that a misconfigured camera adapter stays diagnostic-only.

set -eo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source /opt/ros/humble/setup.bash
source "${workspace_root}/install/setup.bash"

ros2 run robocon_camera_yolo_adapter camera_yolo_adapter \
  --ros-args -p enabled:=true -p legacy_module_path:=TBD -p weights_path:=TBD \
  >/tmp/robocon_camera_yolo_failure.log 2>&1 &
node_pid=$!
cleanup() {
  kill "${node_pid}" 2>/dev/null || true
  wait "${node_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

status_seen=false
for attempt in $(seq 1 100); do
  if ros2 topic list | grep -Fxq "/camera/target_status"; then
    status_seen=true
    break
  fi
  sleep 0.1
done
if [[ "${status_seen}" != true ]]; then
  cat /tmp/robocon_camera_yolo_failure.log
  exit 1
fi

ros2 topic echo --once /camera/target_status
if timeout 1s ros2 topic echo --once /camera/target_observation; then
  echo "unexpected target observation in failed camera mode" >&2
  exit 1
fi
