#!/usr/bin/env bash
# Render Gazebo UI in an isolated Xvfb display; never opens a desktop window.
set -eo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIRECTORY" >&2
  exit 64
fi

workspace="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
run_dir=$1
display_num=94
display_name=localhost:${display_num}
deps="${HEADLESS_VIDEO_DEPS:-}"
if [[ -z "$deps" || ! -x "$deps/usr/bin/Xvfb_headless" ]]; then
  echo "set HEADLESS_VIDEO_DEPS to the prepared headless-video dependency directory" >&2
  exit 2
fi
mkdir -p "$run_dir/raw_gazebo_frames"

source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-174}"
export LD_LIBRARY_PATH="$deps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$deps/usr/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
export LIBGL_ALWAYS_SOFTWARE=1
# Xvfb's private copy is patched to invoke this short path for xkbcomp.
mkdir -p /tmp/xk
cp "$deps/usr/bin/xkbcomp" /tmp/xk/xkbcomp
chmod 755 /tmp/xk/xkbcomp

cleanup_group() {
  local pid=$1
  if kill -0 "$pid" 2>/dev/null; then
    kill -INT -- "-$pid" 2>/dev/null || true
    sleep 2
    kill -TERM -- "-$pid" 2>/dev/null || true
  fi
}

setsid "$deps/usr/bin/Xvfb_headless" ":$display_num" -screen 0 1920x1080x24 -ac -nolisten unix -listen tcp > "$run_dir/xvfb.log" 2>&1 &
xvfb_pid=$!
setsid ros2 launch robocon_mid360_simulation gazebo_mid360_lio.launch.py \
  use_gui:=false lidar_samples:=2000 lidar_downsample:=8 \
  world:="$workspace/install/robocon_mid360_simulation/share/robocon_mid360_simulation/worlds/indoor_competition_candidate.world" > "$run_dir/gazebo_lio.log" 2>&1 &
lio_pid=$!
gzclient_pid=""
motion_pid=""
cleanup() {
  [[ -n "$motion_pid" ]] && cleanup_group "$motion_pid"
  [[ -n "$gzclient_pid" ]] && cleanup_group "$gzclient_pid"
  cleanup_group "$lio_pid"
  cleanup_group "$xvfb_pid"
}
trap cleanup EXIT INT TERM

# Gazebo startup is slow on this workspace; wait for the server before opening the virtual GUI.
for _ in $(seq 1 90); do
  if pgrep -P "$lio_pid" -f gzserver >/dev/null 2>&1 || pgrep -f "gzserver .*indoor_competition_candidate" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! pgrep -f "gzserver .*indoor_competition_candidate" >/dev/null 2>&1; then
  echo "gzserver did not become ready" > "$run_dir/failure_reason.txt"
  exit 1
fi

setsid env DISPLAY="$display_name" LIBGL_ALWAYS_SOFTWARE=1 gzclient --verbose > "$run_dir/gzclient.log" 2>&1 &
gzclient_pid=$!

# A real motion sequence makes the Gazebo GUI evidence visibly dynamic.
setsid bash -lc 'sleep 8; timeout 22s ros2 topic pub --rate 10 /cmd_vel_chassis geometry_msgs/msg/Twist "{linear: {x: 0.20, y: 0.04, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.24}}"' > "$run_dir/motion.log" 2>&1 &
motion_pid=$!

timeout 90s python3 "$workspace/tools/capture_xvfb_frames.py" \
  --display "$display_name" --output-dir "$run_dir/raw_gazebo_frames" \
  --duration-sec 20 --fps 10 --wait-sec 70 > "$run_dir/capture.log" 2>&1
capture_status=$?
cleanup
wait "$lio_pid" 2>/dev/null || true
wait "$gzclient_pid" 2>/dev/null || true
wait "$xvfb_pid" 2>/dev/null || true
if [[ "$capture_status" -ne 0 ]]; then
  echo "capture failed with status $capture_status" > "$run_dir/failure_reason.txt"
  exit "$capture_status"
fi
