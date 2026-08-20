#!/usr/bin/env bash
# Run the ROS-level localization-loss and competition-interlock contract smoke.
set -eo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_DIRECTORY [AUTO_RECOVERY=true|false]" >&2
  exit 64
fi

run_dir="$1"
auto_recovery="${2:-false}"
workspace="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
mkdir -p "$run_dir"

source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
set -u
if [[ "$auto_recovery" != "true" && "$auto_recovery" != "false" ]]; then
  echo "interlock_smoke_failed: AUTO_RECOVERY must be true or false" > "$run_dir/failure_reason.txt"
  exit 64
fi
if pgrep -f 'mid360_map_odom_anchor|robocon_game_supervisor' >/dev/null 2>&1; then
  echo "interlock_smoke_failed: an existing localization or supervisor node would contaminate this run" > "$run_dir/failure_reason.txt"
  exit 1
fi

setsid ros2 run mid360_localization_contract mid360_map_odom_anchor > "$run_dir/map_anchor.log" 2>&1 &
anchor_pid=$!
setsid ros2 run robocon_game_supervisor robocon_game_supervisor --ros-args \
  -p task_id:=localization-interlock-smoke \
  -p require_teammate_heartbeat:=false \
  -p auto_recovery_on_signal_loss:="$auto_recovery" \
  -p action_ttl_sec:=10.0 > "$run_dir/supervisor.log" 2>&1 &
supervisor_pid=$!

cleanup_group() {
  local pid="$1"
  if kill -0 "$pid" 2>/dev/null; then
    kill -INT -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 8); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -TERM -- "-$pid" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
}
cleanup() {
  cleanup_group "$supervisor_pid"
  cleanup_group "$anchor_pid"
}
trap cleanup EXIT INT TERM

set +e
timeout 85s python3 "$workspace/src/robocon_mid360_simulation/scripts/localization_recovery_interlock_smoke.py" \
  --run-dir "$run_dir" --auto-recovery "$auto_recovery" > "$run_dir/runner.log" 2>&1
runner_status=$?
set -e
cleanup
wait "$supervisor_pid" 2>/dev/null || true
wait "$anchor_pid" 2>/dev/null || true

if [[ "$runner_status" -ne 0 ]]; then
  echo "interlock_smoke_failed: runner exited $runner_status" > "$run_dir/failure_reason.txt"
  exit 1
fi
