#!/usr/bin/env bash
# Run the synthetic runtime check and always stop its launch process.

set -eo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
log_file="${TMPDIR:-/tmp}/robocon_synthetic_runtime_smoke.log"

source /opt/ros/humble/setup.bash
source "${workspace_root}/install/setup.bash"

setsid ros2 launch robocon_game_supervisor synthetic_competition.launch.py >"${log_file}" 2>&1 &
launch_pid=$!

cleanup() {
  if kill -0 "${launch_pid}" 2>/dev/null; then
    kill -INT -- "-${launch_pid}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "${launch_pid}" 2>/dev/null || break
      sleep 0.1
    done
    kill -TERM -- "-${launch_pid}" 2>/dev/null || true
    sleep 0.5
    kill -KILL -- "-${launch_pid}" 2>/dev/null || true
  fi
  # The launch process owns child ROS nodes and may not reap promptly after a
  # process-group signal. The group has already been bounded above; do not
  # block the finite smoke result on launch's shutdown handshake.
}
trap cleanup EXIT INT TERM

sleep 1
set +e
python3 "${workspace_root}/src/robocon_game_supervisor/tools/synthetic_runtime_smoke.py"
result=$?
set -e

if [[ ${result} -ne 0 ]]; then
  tail -n 160 "${log_file}"
fi

exit "${result}"
