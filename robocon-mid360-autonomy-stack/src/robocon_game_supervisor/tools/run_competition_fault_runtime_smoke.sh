#!/usr/bin/env bash
# Run one bounded ROS-level competition fault mode with retained evidence.
set -eo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_DIRECTORY protocol|mechanism" >&2
  exit 64
fi
run_dir="$1"
mode="$2"
workspace="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
mkdir -p "$run_dir"
source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
set -u
if [[ "$mode" != "protocol" && "$mode" != "mechanism" ]]; then
  echo "fault_runtime_failed: unsupported mode $mode" > "$run_dir/failure_reason.txt"
  exit 64
fi

setsid ros2 run robocon_game_supervisor robocon_game_supervisor --ros-args \
  -p task_id:=competition-fault-smoke \
  -p require_teammate_heartbeat:=$( [[ "$mode" == protocol ]] && echo true || echo false ) \
  -p auto_recovery_on_signal_loss:=true \
  -p action_ttl_sec:=2.0 > "$run_dir/supervisor.log" 2>&1 &
supervisor_pid=$!
action_pid=""
target_pid=""
if [[ "$mode" == "mechanism" ]]; then
  setsid ros2 run robocon_game_supervisor robocon_action_simulator --ros-args \
    -p task_id:=competition-fault-smoke -p failure_action:=ExecutePass > "$run_dir/action_simulator.log" 2>&1 &
  action_pid=$!
  setsid ros2 run robocon_perception_adapter target_gate > "$run_dir/target_gate.log" 2>&1 &
  target_pid=$!
fi

cleanup_group() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
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
  cleanup_group "$target_pid"
  cleanup_group "$action_pid"
  cleanup_group "$supervisor_pid"
}
trap cleanup EXIT INT TERM

set +e
timeout 35s python3 "$workspace/src/robocon_game_supervisor/tools/competition_fault_runtime_smoke.py" \
  --run-dir "$run_dir" --mode "$mode" > "$run_dir/runner.log" 2>&1
runner_status=$?
set -e
cleanup
wait "$target_pid" 2>/dev/null || true
wait "$action_pid" 2>/dev/null || true
wait "$supervisor_pid" 2>/dev/null || true
if [[ "$runner_status" -ne 0 ]]; then
  echo "fault_runtime_failed: runner exited $runner_status" > "$run_dir/failure_reason.txt"
  exit 1
fi
