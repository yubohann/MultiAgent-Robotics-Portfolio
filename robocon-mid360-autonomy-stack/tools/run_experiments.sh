#!/usr/bin/env bash
# One-command dispatcher for the local MID-360 simulation experiments.
# Each step owns a unique ROS domain and run directory; steps are sequential.

set -o pipefail

workspace="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
delivery="${DELIVERY:-$workspace/deliverables/MID360_MAIN}"
run_root="${RUN_ROOT:-$delivery/one_command_runs/$(date +%Y%m%d_%H%M%S)}"
world="${WORLD:-$workspace/install/robocon_mid360_simulation/share/robocon_mid360_simulation/worlds/indoor_competition_candidate.world}"
map_file="${MAP_FILE:-$delivery/01_frozen_map_dense.pcd}"
map_file_explicit=false
if [[ -n "${MAP_FILE:-}" ]]; then
  map_file_explicit=true
fi
lio_duration="${LIO_DURATION_SEC:-60}"
mapping_duration="${MAPPING_DURATION_SEC:-45}"
localization_duration="${LOCALIZATION_DURATION_SEC:-25}"
min_free_space_gb="${MIN_FREE_SPACE_GB:-20}"

lio_script="$workspace/src/robocon_mid360_simulation/scripts/run_controlled_lio_smoke.sh"
mapping_script="$workspace/src/robocon_mid360_simulation/scripts/run_simulation_mapping.sh"
localization_script="$workspace/src/robocon_mid360_simulation/scripts/run_fixed_map_localization_smoke.sh"
interlock_script="$workspace/src/robocon_mid360_simulation/scripts/run_localization_recovery_interlock_smoke.sh"
synthetic_script="$workspace/src/robocon_game_supervisor/tools/run_synthetic_runtime_smoke.sh"
fault_script="$workspace/src/robocon_game_supervisor/tools/run_competition_fault_runtime_smoke.sh"
ab_script="$workspace/tools/run_pointlio_ab_smoke.sh"
rgbd_script="$workspace/tools/run_rgbd_quality_smoke.sh"
rviz_config="$workspace/src/robocon_mid360_simulation/config/gazebo_mapping.rviz"

usage() {
  cat <<'EOF'
用法：
  bash tools/run_experiments.sh all          顺序运行全部有界自动实验
  bash tools/run_experiments.sh quality      LIO + 建图 + 固定地图定位
  bash tools/run_experiments.sh faults       失锁互锁 + 比赛总控 + 双车/机构故障
  bash tools/run_experiments.sh lio          只运行 30000-ray LIO 受控实验
  bash tools/run_experiments.sh mapping      只运行建图并生成 PCD
  bash tools/run_experiments.sh localization 只运行固定地图定位
  bash tools/run_experiments.sh interlock    只运行失锁安全互锁
  bash tools/run_experiments.sh competition 只运行总控和两类故障
  bash tools/run_experiments.sh ab          只运行 FAST-LIO2 / Point-LIO 同包回放
  bash tools/run_experiments.sh rgbd        只运行 Gazebo RGB-D 质量探针
  bash tools/run_experiments.sh gui         启动可见 Gazebo + RViz（按 Ctrl+C 停止）
  bash tools/run_experiments.sh --dry-run all 只打印 all 将执行的命令

可选环境变量：
  LIO_DURATION_SEC=60 MAPPING_DURATION_SEC=45 LOCALIZATION_DURATION_SEC=25
  MIN_FREE_SPACE_GB=20（长时间 Gazebo 步骤的最低可用磁盘空间）
  RUN_ROOT=<output-directory> MAP_FILE=<frozen-map.pcd>
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi
mode="${1:-all}"

case "$mode" in
  all|quality|faults|lio|mapping|localization|interlock|competition|ab|rgbd|gui) ;;
  *) usage >&2; exit 64 ;;
esac

if [[ "$dry_run" == false ]]; then
  # ROS setup scripts can read unset variables, so strict unset checking starts
  # only after both setup files have been sourced.
  source /opt/ros/humble/setup.bash
  source "$workspace/install/setup.bash"
fi

mkdir -p "$run_root"
summary="$run_root/summary.tsv"
printf "step\tstatus\tros_domain\trun_directory\n" > "$summary"

require_free_space() {
  local path="$1"
  local available_kb
  available_kb="$(df -Pk "$path" | awk 'NR == 2 {print $4}')"
  if [[ ! "$available_kb" =~ ^[0-9]+$ ]]; then
    printf '无法读取磁盘可用空间：%s\n' "$path" >&2
    return 1
  fi
  local required_kb=$((min_free_space_gb * 1024 * 1024))
  if (( available_kb < required_kb )); then
    printf '磁盘空间不足：%s 仅剩 %s GiB，长时间 Gazebo 步骤至少需要 %s GiB。\n' \
      "$path" "$((available_kb / 1024 / 1024))" "$min_free_space_gb" >&2
    return 1
  fi
}

log_command() {
  local dir="$1"
  shift
  printf '%q ' "$@" > "$dir/command.txt"
  printf '\n' >> "$dir/command.txt"
}

run_step() {
  local name="$1"
  local domain="$2"
  shift 2
  local dir="$run_root/$name"
  mkdir -p "$dir"
  log_command "$dir" "$@"
  printf '[开始] %-18s domain=%s\n' "$name" "$domain"
  if [[ "$dry_run" == true ]]; then
    printf '[试运行] '; printf '%q ' env ROS_DOMAIN_ID="$domain" "$@"; printf '\n'
    printf "%s\tDRY_RUN\t%s\t%s\n" "$name" "$domain" "$dir" >> "$summary"
    return 0
  fi
  env ROS_DOMAIN_ID="$domain" "$@" > >(tee "$dir/stdout.log") 2> >(tee "$dir/stderr.log" >&2)
  local status=$?
  if (( status == 0 )); then
    printf "%s\tPASS\t%s\t%s\n" "$name" "$domain" "$dir" >> "$summary"
    printf '[完成] %-18s PASS\n' "$name"
    return 0
  fi
  printf "%s\tFAIL(%s)\t%s\t%s\n" "$name" "$status" "$domain" "$dir" >> "$summary"
  printf '[完成] %-18s FAIL(%s)，已保留日志\n' "$name" "$status" >&2
  return 1
}

run_lio() {
  [[ "$dry_run" == true ]] || require_free_space "$run_root" || return 1
  run_step "01_lio_30000ray" 181 bash "$lio_script" \
    "$run_root/01_lio_30000ray" "$lio_duration" 1 "$world"
}

run_mapping() {
  [[ "$dry_run" == true ]] || require_free_space "$run_root" || return 1
  run_step "02_mapping_dense" 182 bash "$mapping_script" \
    "$run_root/02_mapping_dense" "$mapping_duration" "$world" 1
}

map_is_eligible() {
  local candidate="$1"
  local points
  local manifest="${candidate%/frozen_map.pcd}/map_manifest.json"
  local run_manifest="${candidate%/frozen_map.pcd}/run_manifest.json"
  [[ -s "$candidate" ]] || return 1
  [[ -s "$manifest" ]] || return 1
  [[ -s "$run_manifest" ]] || return 1
  points="$(awk '/^POINTS / {print $2; exit}' "$candidate" 2>/dev/null || true)"
  [[ "$points" =~ ^[0-9]+$ ]] && (( points >= 500 )) || return 1
  python3 - "$manifest" "$run_manifest" "$candidate" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        map_manifest = json.load(handle)
    with open(sys.argv[2], encoding="utf-8") as handle:
        run_manifest = json.load(handle)
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)

eligible = map_manifest.get("eligible_for_fixed_map") is True
pcd_sha256 = map_manifest.get("pcd_sha256")
try:
    import hashlib
    actual_sha256 = hashlib.sha256(open(sys.argv[3], "rb").read()).hexdigest()
except OSError:
    raise SystemExit(1)
requested_rays = run_manifest.get("lidar_samples_requested")
try:
    requested_rays = int(requested_rays)
except (TypeError, ValueError):
    requested_rays = 0
profile = run_manifest.get("profile")
scene = run_manifest.get("scene")
downsample = run_manifest.get("lidar_downsample", 1)
try:
    downsample = int(downsample)
except (TypeError, ValueError):
    downsample = -1

# A large diagnostic PCD can exceed the point-count gate while still being
# explicitly unsuitable for competition localization. Require the controlled
# 30,000-ray indoor profile as well as the map manifest's promotion flag.
accepted_profile = profile in {"mapping-controlled", "30000-ray-controlled-motion"}
accepted = eligible and pcd_sha256 == actual_sha256 and requested_rays >= 30000 and downsample == 1 \
    and scene == "indoor_competition_candidate" and accepted_profile
raise SystemExit(0 if accepted else 1)
PY
}

latest_generated_map() {
  local candidate
  while IFS= read -r candidate; do
    if map_is_eligible "$candidate"; then
      printf '%s' "$candidate"
      return 0
    fi
  done < <(
    # Include retained dense mapping artifacts as well as maps produced by
    # this dispatcher.  A standalone `localization` run must work from a
    # fresh shell after the mapping run has already been archived.
    find "$delivery" -type f -name 'frozen_map.pcd' \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr | cut -d' ' -f2-
  )
  return 1
}

map_for_localization() {
  local candidate="$run_root/02_mapping_dense/frozen_map.pcd"
  if map_is_eligible "$candidate"; then
    printf '%s' "$candidate"
    return 0
  fi
  if [[ "$map_file_explicit" == false ]]; then
    candidate="$(latest_generated_map || true)"
    if [[ -n "$candidate" ]]; then
      printf '%s' "$candidate"
      return 0
    fi
  fi
  printf '%s' "$map_file"
}

run_localization() {
  local selected_map
  # In a combined run, localization must consume the map made by this exact
  # invocation. Falling back to an archived map would turn a failed mapping
  # step into misleadingly successful downstream evidence.
  if [[ "$mode" == "all" || "$mode" == "quality" ]]; then
    selected_map="$run_root/02_mapping_dense/frozen_map.pcd"
    if [[ "$dry_run" == false ]] && ! map_is_eligible "$selected_map"; then
      printf '本轮建图未产生可用固定地图，跳过定位：%s\n' "$selected_map" >&2
      return 1
    fi
  else
    selected_map="$(map_for_localization)"
  fi
  if [[ "$dry_run" == false && ! -s "$selected_map" ]]; then
    printf '缺少固定地图：%s\n' "$selected_map" >&2
    return 1
  fi
  run_step "03_fixed_map_localization" 183 bash "$localization_script" \
    "$run_root/03_fixed_map_localization" "$selected_map" "$localization_duration"
}

run_interlock() {
  run_step "04_localization_interlock" 184 bash "$interlock_script" \
    "$run_root/04_localization_interlock" true
}

run_competition() {
  local failed=0
  run_step "05_synthetic_competition" 185 bash -c "bash '$synthetic_script'" || failed=1
  run_step "06_protocol_fault" 186 bash "$fault_script" \
    "$run_root/06_protocol_fault" protocol || failed=1
  run_step "07_mechanism_fault" 187 bash "$fault_script" \
    "$run_root/07_mechanism_fault" mechanism || failed=1
  return "$failed"
}

run_ab() {
  # The helper owns its sensor, FAST-LIO2, and Point-LIO domains. Keep the
  # complete bag and estimator logs below this all-run directory.
  run_step "08_pointlio_ab" 194 env RUN_ROOT="$run_root/08_pointlio_ab" \
    bash "$ab_script" "pointlio_ab" 2000
}

run_rgbd() {
  run_step "09_rgbd_quality" 188 bash "$rgbd_script" "$run_root/09_rgbd_quality"
}

run_gui() {
  local dir="$run_root/00_gui"
  mkdir -p "$dir"
  local launch_pid
  export ROS_DOMAIN_ID=180
  printf 'GUI 运行目录：%s\n' "$dir"
  setsid ros2 launch robocon_mid360_simulation gazebo_mid360_mapping.launch.py \
    use_gui:=true lidar_samples:=30000 lidar_downsample:=1 world:="$world" \
    map_output_file:="$dir/frozen_map.pcd" \
    metadata_output_file:="$dir/frozen_map.yaml" > "$dir/gazebo.log" 2>&1 &
  launch_pid=$!
  cleanup_gui() {
    if kill -0 "$launch_pid" 2>/dev/null; then
      kill -INT -- "-$launch_pid" 2>/dev/null || true
      sleep 2
      kill -TERM -- "-$launch_pid" 2>/dev/null || true
    fi
  }
  trap cleanup_gui EXIT INT TERM
  rviz2 -d "$rviz_config"
}

run_mode() {
  local failed=0
  case "$1" in
    all)
      run_lio || failed=1
      run_mapping || failed=1
      run_localization || failed=1
      run_interlock || failed=1
      run_competition || failed=1
      run_ab || failed=1
      run_rgbd || failed=1
      ;;
    quality)
      run_lio || failed=1
      run_mapping || failed=1
      run_localization || failed=1
      ;;
    faults)
      run_interlock || failed=1
      run_competition || failed=1
      ;;
    lio) run_lio || failed=1 ;;
    mapping) run_mapping || failed=1 ;;
    localization) run_localization || failed=1 ;;
    interlock) run_interlock || failed=1 ;;
    competition) run_competition || failed=1 ;;
    ab) run_ab || failed=1 ;;
    rgbd) run_rgbd || failed=1 ;;
    gui) run_gui; return $? ;;
  esac
  return "$failed"
}

printf '模式：%s\n运行根目录：%s\n' "$mode" "$run_root"
if run_mode "$mode"; then
  printf '\n全部请求步骤已完成。汇总：%s\n' "$summary"
  exit 0
fi
printf '\n有步骤失败；失败日志已保留。汇总：%s\n' "$summary" >&2
exit 1
