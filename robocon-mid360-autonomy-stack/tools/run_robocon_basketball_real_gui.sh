#!/usr/bin/env bash
# Record real Gazebo/RViz windows fed by live ROS 2 topics.
set -Ee -o pipefail

WORKSPACE="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DURATION=${1:-180}
LIDAR_SAMPLES=${2:-50000}
PLAYBACK_SPEED=${3:-2.0}
RUN_DIR=${4:-"$WORKSPACE/runs/robocon_basketball_real_gui_$(date +%Y%m%d_%H%M%S)"}
# Quality-first recording keeps the full 50k-ray packet. Use 30000/15 only
# when interactive responsiveness is more important than point density.
LIDAR_DOWNSAMPLE=${5:-1}
DOMAIN_ID=${ROS_DOMAIN_ID:-192}
DEPS=${HEADLESS_VIDEO_DEPS:-$HOME/.local/headless-video-deps}
FFMPEG=${FFMPEG:-$(command -v ffmpeg || true)}
FFPROBE=${FFPROBE:-$(command -v ffprobe || true)}
GUI_WARMUP_SEC=${ROBOCON_GUI_WARMUP_SEC:-45}
DISPLAY_BASE=${ROBOCON_GUI_BASE_DISPLAY:-$((140 + $$ % 40))}
declare -a XVFB_PIDS=()
declare -a GUI_PIDS=()
declare -a CAPTURE_PIDS=()
LAUNCH_PID=""
CONTROLLER_PID=""

if (( DURATION < 60 )); then echo "duration must be >= 60 seconds" >&2; exit 2; fi
if (( LIDAR_SAMPLES < 128 )); then echo "lidar samples must be >= 128" >&2; exit 2; fi
if (( LIDAR_DOWNSAMPLE < 1 )); then echo "lidar downsample must be >= 1" >&2; exit 2; fi
if (( LIDAR_SAMPLES >= 200000 && LIDAR_DOWNSAMPLE < 4 )); then
  echo "dual 200000+ ray capture requires lidar downsample >= 4 on the default WSL memory profile" >&2
  exit 2
fi
if ! awk "BEGIN { exit !($PLAYBACK_SPEED > 0) }"; then
  echo "playback speed must be a positive number" >&2; exit 2
fi
if [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing to overwrite non-empty run directory: $RUN_DIR" >&2; exit 2
fi
for required in "$DEPS/usr/bin/Xvfb_headless" "$FFMPEG" "$FFPROBE"; do
  [[ -x "$required" ]] || { echo "missing executable: $required" >&2; exit 2; }
done
mkdir -p "$RUN_DIR" "$RUN_DIR/gui_gazebo" "$RUN_DIR/gui_rviz_pointcloud" "$RUN_DIR/gui_rviz_map2d" "$RUN_DIR/gui_rviz_rgbd"

cleanup_group() {
  local pid=${1:-}
  [[ -z "$pid" ]] && return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
    sleep 1
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  fi
}
cleanup() {
  local rc=$?
  for pid in "${CAPTURE_PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  cleanup_group "$CONTROLLER_PID"
  cleanup_group "$LAUNCH_PID"
  for pid in "${GUI_PIDS[@]:-}"; do cleanup_group "$pid"; done
  for pid in "${XVFB_PIDS[@]:-}"; do cleanup_group "$pid"; done
  printf 'exit_code=%s\n' "$rc" >> "$RUN_DIR/process_status.txt"
  exit "$rc"
}
trap cleanup EXIT INT TERM

source /opt/ros/humble/setup.bash
source "$WORKSPACE/install/setup.bash"
set -u
export ROS_DOMAIN_ID="$DOMAIN_ID"
export LIBGL_ALWAYS_SOFTWARE=1
export QT_X11_NO_MITSHM=1
export LD_LIBRARY_PATH="$DEPS/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GAZEBO_MODEL_PATH="$WORKSPACE/install/robocon_mid360_simulation/share/robocon_mid360_simulation/models${GAZEBO_MODEL_PATH:+:$GAZEBO_MODEL_PATH}"
mkdir -p /tmp/xk
cp "$DEPS/usr/bin/xkbcomp" /tmp/xk/xkbcomp
chmod 755 /tmp/xk/xkbcomp

python3 - "$RUN_DIR/manifest.json" <<PY
import json, pathlib
path = pathlib.Path("$RUN_DIR/manifest.json")
path.write_text(json.dumps({
  "profile": "gazebo_rviz_real_gui_two_robot_basketball",
  "scene_source": "Gazebo Classic + live ROS 2 topics",
  "duration_requested_sec": float("$DURATION"),
  "lidar_samples": int("$LIDAR_SAMPLES"),
  "lidar_downsample": int("$LIDAR_DOWNSAMPLE"),
  "playback_speed": float("$PLAYBACK_SPEED"),
  "ros_domain_id": int("$DOMAIN_ID"),
  "synthetic_view_panels": False,
  "gui_windows": ["Gazebo", "RViz point cloud", "RViz 2D map", "RViz RGB-D image"]
}, indent=2) + "\n", encoding="utf-8")
PY

start_xvfb() {
  local display_num=$1 log=$2
  setsid "$DEPS/usr/bin/Xvfb_headless" ":$display_num" -screen 0 1280x720x24 -ac -nolisten unix -listen tcp >"$log" 2>&1 &
  XVFB_PIDS+=("$!")
}
start_gui() {
  local display_num=$1 log=$2; shift 2
  setsid env DISPLAY="localhost:$display_num" LIBGL_ALWAYS_SOFTWARE=1 QT_X11_NO_MITSHM=1 "$@" >"$log" 2>&1 &
  GUI_PIDS+=("$!")
}

# Each GUI has its own X display, so focus changes in one application cannot
# move or resize another recording.
start_xvfb "$DISPLAY_BASE" "$RUN_DIR/xvfb_gazebo.log"
start_xvfb "$((DISPLAY_BASE + 1))" "$RUN_DIR/xvfb_rviz_pointcloud.log"
start_xvfb "$((DISPLAY_BASE + 2))" "$RUN_DIR/xvfb_rviz_map2d.log"
start_xvfb "$((DISPLAY_BASE + 3))" "$RUN_DIR/xvfb_rviz_rgbd.log"

setsid ros2 launch robocon_mid360_simulation gazebo_mid360_dual_mapping.launch.py \
  use_gui:=false enable_rgbd:=true lidar_samples:="$LIDAR_SAMPLES" lidar_downsample:="$LIDAR_DOWNSAMPLE" \
  >"$RUN_DIR/ros_launch.log" 2>&1 &
LAUNCH_PID=$!
for _ in $(seq 1 120); do
  pgrep -f "gzserver .*robocon25_candidate" >/dev/null 2>&1 && break
  sleep 1
done
pgrep -f "gzserver .*robocon25_candidate" >/dev/null 2>&1 || { echo "gzserver did not become ready" >"$RUN_DIR/failure_reason.txt"; exit 1; }

start_gui "$DISPLAY_BASE" "$RUN_DIR/gazebo_gui.log" gzclient --verbose
start_gui "$((DISPLAY_BASE + 1))" "$RUN_DIR/rviz_pointcloud.log" rviz2 -d "$WORKSPACE/src/robocon_mid360_simulation/config/gazebo_mapping.rviz"
start_gui "$((DISPLAY_BASE + 2))" "$RUN_DIR/rviz_map2d.log" rviz2 -d "$WORKSPACE/src/robocon_mid360_simulation/config/gazebo_2d_mapping.rviz"
start_gui "$((DISPLAY_BASE + 3))" "$RUN_DIR/rviz_rgbd.log" rviz2 -d "$WORKSPACE/src/robocon_mid360_simulation/config/rviz_basketball_rgbd.rviz"

capture_window() {
  local display_num=$1 title=$2 output_dir=$3 prefix=$4
  setsid timeout --signal=INT "$((DURATION + 90))" python3 "$WORKSPACE/tools/capture_xvfb_frames.py" \
    --display "localhost:$display_num" --window-title "$title" --output-dir "$output_dir" \
    --duration-sec "$((DURATION + 5))" --fps 30 --wait-sec 90 \
    --window-only --output-width 1280 --output-height 720 \
    --file-prefix "$prefix" --video-output "$output_dir/$prefix.avi" \
    >"$output_dir/capture.log" 2>&1 &
  CAPTURE_PIDS+=("$!")
}
capture_window "$DISPLAY_BASE" gazebo "$RUN_DIR/gui_gazebo" gazebo_real
capture_window "$((DISPLAY_BASE + 1))" rviz2 "$RUN_DIR/gui_rviz_pointcloud" rviz_pointcloud_real
capture_window "$((DISPLAY_BASE + 2))" rviz2 "$RUN_DIR/gui_rviz_map2d" rviz_map2d_real
capture_window "$((DISPLAY_BASE + 3))" rviz2 "$RUN_DIR/gui_rviz_rgbd" rviz_rgbd_real

# Let all real windows finish their first render before motion starts.
sleep "$GUI_WARMUP_SEC"
timeout --signal=INT "$((DURATION + 120))" python3 \
  "$WORKSPACE/install/robocon_mid360_simulation/lib/robocon_mid360_simulation/basketball_demo_controller.py" \
  --duration "$DURATION" --output-dir "$RUN_DIR" >"$RUN_DIR/controller.log" 2>&1 &
CONTROLLER_PID=$!
wait "$CONTROLLER_PID"
CONTROLLER_RC=$?
for pid in "${CAPTURE_PIDS[@]}"; do wait "$pid" || true; done

encode_gui() {
  local source=$1 target=$2
  local src_win target_win
  src_win=$(wslpath -w "$source")
  target_win=$(wslpath -w "$target")
  "$FFMPEG" -y -hide_banner -loglevel error -i "$src_win" \
    -vf "hqdn3d=1.1:1.1:4:4,minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir,setpts=PTS/$PLAYBACK_SPEED,scale=1280:720:flags=lanczos,format=yuv420p" \
    -an -c:v libx264 -preset slow -crf 18 "$target_win"
}
encode_gui "$RUN_DIR/gui_gazebo/gazebo_real.avi" "$RUN_DIR/gazebo_real_gui.mp4"
encode_gui "$RUN_DIR/gui_rviz_pointcloud/rviz_pointcloud_real.avi" "$RUN_DIR/rviz_pointcloud_real_gui.mp4"
encode_gui "$RUN_DIR/gui_rviz_map2d/rviz_map2d_real.avi" "$RUN_DIR/rviz_map2d_real_gui.mp4"
encode_gui "$RUN_DIR/gui_rviz_rgbd/rviz_rgbd_real.avi" "$RUN_DIR/rviz_rgbd_real_gui.mp4"
printf 'controller_rc=%s\n' "$CONTROLLER_RC" >"$RUN_DIR/process_status.txt"

python3 - "$RUN_DIR" "$FFPROBE" <<'PY'
import json, pathlib, subprocess, sys
run, ffprobe = pathlib.Path(sys.argv[1]), sys.argv[2]
summary = json.loads((run / "success_summary.json").read_text(encoding="utf-8"))
events = (run / "demo_events.jsonl").read_text(encoding="utf-8")
failures = []
if summary.get("status") != "PASS" or not summary.get("shot_success"):
    failures.append("basketball controller did not report PASS")
for token in ('"state": "PASS_SUCCESS"', '"event": "shot_success"', '"event": "demo_complete"'):
    if token not in events: failures.append(f"missing {token}")
videos = [
  "gazebo_real_gui.mp4", "rviz_pointcloud_real_gui.mp4",
  "rviz_map2d_real_gui.mp4", "rviz_rgbd_real_gui.mp4"
]
video_info = {}
for name in videos:
    path = run / name
    if not path.is_file() or path.stat().st_size < 50000:
        failures.append(f"missing or undersized {name}")
        continue
    win_path = subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()
    probe = subprocess.check_output([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", win_path], text=True)
    duration = float(json.loads(probe)["format"]["duration"])
    video_info[name] = {"bytes": path.stat().st_size, "duration_sec": round(duration, 3), "source": "real GUI window"}
    if duration <= 1.0: failures.append(f"invalid duration {name}")
for directory in ("gui_gazebo", "gui_rviz_pointcloud", "gui_rviz_map2d", "gui_rviz_rgbd"):
    metadata_path = run / directory / "capture_metadata.json"
    if not metadata_path.is_file(): failures.append(f"missing {directory}/capture_metadata.json")
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("unique_frame_hashes", 0) < 20: failures.append(f"unstable/empty {directory}: {metadata}")
report = {"status": "PASS" if not failures else "FAIL", "summary": summary, "videos": video_info, "failures": failures, "synthetic_view_panels": False}
(run / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
if failures: raise SystemExit("REAL GUI VALIDATION FAILED: " + "; ".join(failures))
print(json.dumps(report, indent=2))
PY
echo "PASS: $RUN_DIR"
