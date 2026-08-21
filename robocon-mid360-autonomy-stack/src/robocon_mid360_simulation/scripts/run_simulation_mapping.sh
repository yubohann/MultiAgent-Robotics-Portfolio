#!/usr/bin/env bash
# Build a frozen map from a bounded Gazebo FAST-LIO2 run.
set -eo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo "usage: $0 OUTPUT_DIR [DURATION_SEC] [WORLD_FILE] [LIDAR_DOWNSAMPLE]" >&2
  exit 64
fi

output_dir="$1"
duration_sec="${2:-45}"
workspace="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
world_file="${3:-$workspace/install/robocon_mid360_simulation/share/robocon_mid360_simulation/worlds/indoor_competition_candidate.world}"
lidar_downsample="${4:-1}"
min_frozen_map_points="${MIN_FROZEN_MAP_POINTS:-500}"
lidar_samples="${LIDAR_SAMPLES:-30000}"
motion_scale="${MOTION_SCALE:-1.0}"
sparse_coverage="${SPARSE_COVERAGE:-false}"
max_wall_sec="${MAPPING_MAX_WALL_SEC:-720}"
mkdir -p "$output_dir"
pcd="$output_dir/frozen_map.pcd"
metadata="$output_dir/frozen_map.yaml"

source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
set -u
ros_domain_id="${ROS_DOMAIN_ID:-42}"
if ! [[ "$ros_domain_id" =~ ^[0-9]+$ ]] || (( ros_domain_id < 0 || ros_domain_id > 232 )); then
  echo "map_generation_failed: ROS_DOMAIN_ID must be an integer in [0, 232]" > "$output_dir/failure_reason.txt"
  exit 64
fi
if [[ "$sparse_coverage" != "true" && "$sparse_coverage" != "false" ]]; then
  echo "map_generation_failed: SPARSE_COVERAGE must be true or false" > "$output_dir/failure_reason.txt"
  exit 64
fi
export ROS_DOMAIN_ID="$ros_domain_id"

if pgrep -f 'gzserver .*robocon_mid360_simulation' >/dev/null 2>&1; then
  echo "map_generation_failed: an existing Gazebo server would contaminate this run" > "$output_dir/failure_reason.txt"
  exit 1
fi
# The dedicated session makes the launch PID a process-group leader so cleanup
# cannot strand same-named ROS nodes into a later diagnostic run.
setsid ros2 launch robocon_mid360_simulation gazebo_mid360_mapping.launch.py \
  use_gui:=false lidar_samples:="$lidar_samples" lidar_downsample:="$lidar_downsample" \
  sparse_coverage:="$sparse_coverage" \
  world:="$world_file" map_output_file:="$pcd" \
  metadata_output_file:="$metadata" > "$output_dir/launch.log" 2>&1 &
launch_pid=$!
cleanup() {
  if kill -0 "$launch_pid" 2>/dev/null; then
    kill -INT -- "-$launch_pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$launch_pid" 2>/dev/null || break
      sleep 1
    done
    kill -TERM -- "-$launch_pid" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-$launch_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# The recorder owns readiness. Starting a separate probe first can race a slow
# Gazebo model spawn and discard a run before the simulator has created its
# sensor topics. Motion remains disabled until the recorder sees valid input.
set +e
python3 "$workspace/src/robocon_mid360_simulation/scripts/controlled_lio_run.py" \
  --run-dir "$output_dir" --duration-sec "$duration_sec" --lidar-samples "$lidar_samples" \
  --lidar-downsample "$lidar_downsample" \
  --motion-scale "$motion_scale" \
  --scene indoor_competition_candidate \
  --profile "$(if [[ "$sparse_coverage" == "true" ]]; then echo mapping-sparse-coverage-diagnostic; else echo mapping-controlled; fi)" \
  --ready-timeout-sec 180 --required-lidar-packets 3 --required-imu-packets 10 \
  --max-wall-sec "$max_wall_sec" > "$output_dir/recorder.log" 2>&1
recorder_status=$?
set -e
cleanup
wait "$launch_pid" 2>/dev/null || true

if [[ "$recorder_status" -ne 0 ]]; then
  echo "map_generation_failed: controlled recorder exited ${recorder_status}; inspect summary.json" > "$output_dir/failure_reason.txt"
  exit 1
fi

if [[ ! -s "$pcd" || ! -s "$metadata" ]]; then
  echo "map_generation_failed: no non-empty PCD was written" > "$output_dir/failure_reason.txt"
  exit 1
fi
python3 - "$pcd" "$metadata" "$min_frozen_map_points" "$sparse_coverage" \
  "$lidar_samples" "$lidar_downsample" "$world_file" <<'PY'
import hashlib, json, sys
from pathlib import Path
pcd, metadata = map(Path, sys.argv[1:3])
minimum_points = int(sys.argv[3])
sparse_coverage = sys.argv[4] == "true"
requested_rays = int(sys.argv[5])
downsample = int(sys.argv[6])
world_file = sys.argv[7]
scene = "indoor_competition_candidate" if world_file.endswith("indoor_competition_candidate.world") else "other"
header = pcd.read_text(encoding="utf-8", errors="strict").split("DATA ascii", 1)[0]
point_line = next((line for line in header.splitlines() if line.startswith("POINTS ")), "POINTS 0")
point_count = int(point_line.split()[1])
eligible = (
    point_count >= minimum_points
    and not sparse_coverage
    and requested_rays >= 30000
    and downsample == 1
    and scene == "indoor_competition_candidate"
)
diagnostic_only = sparse_coverage or requested_rays < 30000 or downsample != 1 or scene != "indoor_competition_candidate"
payload = {
    "pcd_sha256": hashlib.sha256(pcd.read_bytes()).hexdigest(),
    "pcd_bytes": pcd.stat().st_size,
    "point_count": point_count,
    "minimum_points_for_fixed_map": minimum_points,
    "eligible_for_fixed_map": eligible,
    "promotion_status": "diagnostic_only" if diagnostic_only else ("eligible" if eligible else "rejected"),
    "lidar_samples_requested": requested_rays,
    "lidar_downsample": downsample,
    "scene": scene,
    "metadata_file": str(metadata),
    "evidence_level": "gazebo_simulation",
}
(pcd.parent / "map_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
if point_count < minimum_points:
    (pcd.parent / "failure_reason.txt").write_text(
        f"map_generation_failed: PCD has {point_count} points; fixed-map minimum is {minimum_points}\n",
        encoding="utf-8")
    raise SystemExit(1)
PY
