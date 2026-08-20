#!/usr/bin/env bash
# Exercise systemd lifecycle semantics with a local simulation-only ROS node.
set -eo pipefail

workspace="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
run_dir="${1:-$workspace/runs/systemd_simulation_test_$(date +%Y%m%d_%H%M%S)}"
map_file="${MAP_FILE:-$workspace/deliverables/MID360_MAIN/dense_mapping_30000_240s_slow025_20260820/frozen_map.pcd}"
mkdir -p "$run_dir"
source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
set -u

systemctl_cmd=(systemctl --user)
systemd_run_cmd=(systemd-run --user)
systemd_state="$("${systemctl_cmd[@]}" is-system-running 2>/dev/null || true)"
if [[ "$systemd_state" != "running" && "$systemd_state" != "degraded" ]]; then
  echo "systemd is not running" >&2
  exit 64
fi
if [[ ! -s "$map_file" ]]; then
  echo "missing simulation map: $map_file" >&2
  exit 64
fi

domain="${ROS_DOMAIN_ID:-231}"
unit="robocon-mid360-sim-lifecycle"
restart_unit="robocon-mid360-sim-restart"
timeout_unit="robocon-mid360-sim-stop-timeout"
units=("$unit" "$restart_unit" "$timeout_unit")
cleanup() {
  for item in "${units[@]}"; do
    "${systemctl_cmd[@]}" stop "$item" >/dev/null 2>&1 || true
    "${systemctl_cmd[@]}" reset-failed "$item" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT INT TERM
cleanup

start_wall="$(date +%s%3N)"
"${systemd_run_cmd[@]}" --unit="$unit" --property=Type=simple --property=Restart=no \
  --property=TimeoutStopSec=5s --property=Environment="ROS_DOMAIN_ID=$domain" \
  /bin/bash -lc "source /opt/ros/humble/setup.bash; source '$workspace/install/setup.bash'; exec ros2 run mid360_map_localizer mid360_scan_matcher_node --ros-args -p map_file:='$map_file'" \
  >"$run_dir/start.stdout" 2>"$run_dir/start.stderr"
sleep 3
start_state="$("${systemctl_cmd[@]}" is-active "$unit" || true)"
main_pid="$("${systemctl_cmd[@]}" show "$unit" -p MainPID --value || true)"
"${systemctl_cmd[@]}" stop "$unit"
stop_state="$("${systemctl_cmd[@]}" is-active "$unit" || true)"

"${systemd_run_cmd[@]}" --unit="$restart_unit" --property=Restart=on-failure --property=RestartSec=0.5s \
  --property=StartLimitIntervalSec=10s --property=StartLimitBurst=5 \
  /bin/bash -lc 'exit 42' >"$run_dir/restart.stdout" 2>"$run_dir/restart.stderr"
sleep 3
restart_count="$("${systemctl_cmd[@]}" show "$restart_unit" -p NRestarts --value || true)"
restart_result="$("${systemctl_cmd[@]}" show "$restart_unit" -p Result --value || true)"

"${systemd_run_cmd[@]}" --unit="$timeout_unit" --property=TimeoutStopSec=2s \
  /bin/bash -lc "trap 'while true; do sleep 1; done' TERM; while true; do sleep 1; done" \
  >"$run_dir/timeout.stdout" 2>"$run_dir/timeout.stderr"
sleep 1
timeout_start="$(date +%s)"
"${systemctl_cmd[@]}" stop "$timeout_unit" || true
timeout_elapsed="$(( $(date +%s) - timeout_start ))"
timeout_result="$("${systemctl_cmd[@]}" show "$timeout_unit" -p Result --value || true)"

cat >"$run_dir/systemd_lifecycle_summary.json" <<EOF
{
  "evidence_level": "gazebo_simulation",
  "scope": "local systemd lifecycle harness; no physical transport or actuator claim",
  "systemd_state": "$systemd_state",
  "map_file": "$map_file",
  "ros_domain_id": "$domain",
  "start_state": "$start_state",
  "main_pid_seen": "$main_pid",
  "stop_state": "$stop_state",
  "restart_count": "$restart_count",
  "restart_result": "$restart_result",
  "stop_timeout_result": "$timeout_result",
  "stop_timeout_elapsed_sec": $timeout_elapsed,
  "start_pass": $( [[ "$start_state" == active ]] && echo true || echo false ),
  "restart_pass": $( [[ "$restart_count" =~ ^[1-9][0-9]*$ ]] && echo true || echo false ),
  "stop_timeout_pass": $( [[ "$timeout_elapsed" -ge 2 ]] && echo true || echo false )
}
EOF
cat "$run_dir/systemd_lifecycle_summary.json"
if ! grep -q '"start_pass": true' "$run_dir/systemd_lifecycle_summary.json" || \
   ! grep -q '"restart_pass": true' "$run_dir/systemd_lifecycle_summary.json" || \
   ! grep -q '"stop_timeout_pass": true' "$run_dir/systemd_lifecycle_summary.json"; then
  exit 1
fi
