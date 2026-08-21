#!/usr/bin/env bash
# Capture one Gazebo input bag and replay it through FAST-LIO2 and Point-LIO.
set -o pipefail

workspace="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
pointlio_install="${POINTLIO_INSTALL:-$workspace/../mid360_pointlio_ab/install}"
run_id="${1:-pointlio_ab_smoke_$(date +%Y%m%d_%H%M%S)}"
ray_count="${2:-2000}"
run_dir="${RUN_ROOT:-$workspace/runs/$run_id}"
bag_dir="$run_dir/input_bag"
world="${WORLD:-$workspace/install/robocon_mid360_simulation/share/robocon_mid360_simulation/worlds/indoor_competition_candidate.world}"
fastlio_params="${FASTLIO_PARAMS:-$workspace/install/robocon_mid360_simulation/share/robocon_mid360_simulation/config/fast_lio_simulation.yaml}"

# Accept both colcon isolated installs (install/point_lio/...) and merged
# installs (install/lib/point_lio/...); the persistent A/B build uses merged.
if [[ -x "$pointlio_install/point_lio/lib/point_lio/pointlio_mapping" ]]; then
  pointlio_prefix="$pointlio_install/point_lio"
elif [[ -x "$pointlio_install/lib/point_lio/pointlio_mapping" ]]; then
  pointlio_prefix="$pointlio_install"
else
  pointlio_prefix=""
fi
pointlio_params="$pointlio_prefix/share/point_lio/config/mid360.yaml"

source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
if [[ ! -f "$pointlio_install/setup.bash" ]]; then
  echo "Point-LIO setup file not found: $pointlio_install/setup.bash" >&2
  exit 2
fi
source "$pointlio_install/setup.bash"
set -euo pipefail

if [[ -z "$pointlio_prefix" ]]; then
  echo "缺少 Point-LIO 安装：$pointlio_install" >&2
  exit 2
fi
if [[ ! -f "$world" || ! -f "$fastlio_params" || ! -f "$pointlio_params" ]]; then
  echo "缺少仿真 world 或参数文件" >&2
  exit 2
fi

mkdir -p "$run_dir"
printf 'evidence=gazebo_simulation\nray_count=%s\nworld=%s\npointlio_install=%s\n' \
  "$ray_count" "$world" "$pointlio_install" > "$run_dir/manifest.txt"
printf 'pointlio_tf_bridge=base_link->livox_frame (adapter-only)\n' >> "$run_dir/manifest.txt"
printf 'ground_truth_topic=/simulation/ground_truth/odom\n' >> "$run_dir/manifest.txt"

sensor_pid=""
cleanup_sensor() {
  if [[ -n "$sensor_pid" ]] && kill -0 "$sensor_pid" 2>/dev/null; then
    kill -INT -- "-$sensor_pid" 2>/dev/null || true
    sleep 2
    kill -TERM -- "-$sensor_pid" 2>/dev/null || true
  fi
}
trap cleanup_sensor EXIT INT TERM

export ROS_DOMAIN_ID=194
setsid ros2 launch robocon_mid360_simulation gazebo_mid360_candidate.launch.py \
  use_gui:=false lidar_samples:="$ray_count" lidar_downsample:=1 world:="$world" \
  enable_ground_truth:=true \
  > "$run_dir/gazebo.log" 2>&1 &
sensor_pid=$!

wait_for_topic_type() {
  local topic="$1" expected="$2" deadline=$((SECONDS + 90))
  while (( SECONDS < deadline )); do
    if ros2 topic type "$topic" 2>/dev/null | grep -q "$expected"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

if ! wait_for_topic_type /livox/lidar livox_ros_driver2/msg/CustomMsg; then
  echo "LiDAR readiness timeout" > "$run_dir/failure_reason.txt"
  exit 1
fi
ros2 topic echo --once /livox/lidar > "$run_dir/readiness_lidar.txt" 2> "$run_dir/readiness_lidar.err" || true
if ! wait_for_topic_type /livox/imu sensor_msgs/msg/Imu; then
  echo "IMU readiness timeout" > "$run_dir/failure_reason.txt"
  exit 1
fi
ros2 topic echo --once /livox/imu > "$run_dir/readiness_imu.txt" 2> "$run_dir/readiness_imu.err" || true

timeout --signal=INT --kill-after=5s 22s ros2 bag record -o "$bag_dir" \
  /livox/lidar /livox/imu /clock /tf /tf_static /simulation/ground_truth/odom > "$run_dir/record.log" 2>&1 &
record_pid=$!
sleep 2
publish_segment() {
  local seconds="$1" linear_x="$2" linear_y="$3" angular_z="$4"
  timeout --signal=INT "${seconds}s" ros2 topic pub --rate 10 /cmd_vel_chassis \
    geometry_msgs/msg/Twist "{linear: {x: $linear_x, y: $linear_y, z: 0.0}, angular: {z: $angular_z}}" \
    >/dev/null 2>&1 || true
}
publish_segment 4 0.0 0.0 0.0
publish_segment 5 0.2 0.0 0.0
publish_segment 5 0.0 0.2 0.0
publish_segment 4 0.0 0.0 0.25
wait "$record_pid" 2>/dev/null || true
ros2 bag info "$bag_dir" > "$run_dir/bag_info.txt" 2>&1 || true
if [[ ! -f "$bag_dir/metadata.yaml" ]]; then
  echo "bag recording did not produce metadata.yaml" > "$run_dir/failure_reason.txt"
  exit 1
fi

replay_algorithm() {
  local name="$1" domain="$2" output_topic="$3"
  shift 3
  local dir="$run_dir/$name"
  mkdir -p "$dir"
  export ROS_DOMAIN_ID="$domain"
  setsid "$@" > "$dir/node.log" 2>&1 &
  local node_pid=$!
  setsid ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link livox_frame \
    > "$dir/tf_bridge.log" 2>&1 &
  local tf_pid=$!
  setsid python3 "$workspace/tools/evaluate_odom_truth.py" \
    --truth-topic /simulation/ground_truth/odom --estimate-topic "$output_topic" \
    --output "$dir/truth_metrics.json" > "$dir/truth_eval.log" 2>&1 &
  local eval_pid=$!
  sleep 2
  timeout --signal=INT --kill-after=5s 35s ros2 topic hz "$output_topic" > "$dir/topic_hz.log" 2>&1 &
  local hz_pid=$!
  # A crashed estimator must not leave bag playback and the evaluator orphaned.
  timeout --signal=INT --kill-after=5s 45s ros2 bag play "$bag_dir" \
    --clock --rate 1.0 > "$dir/bag_play.log" 2>&1 || true
  wait "$hz_pid" 2>/dev/null || true
  kill -INT -- "-$node_pid" 2>/dev/null || true
  kill -INT "$node_pid" 2>/dev/null || true
  kill -INT -- "-$tf_pid" 2>/dev/null || true
  kill -INT "$tf_pid" 2>/dev/null || true
  kill -INT -- "-$eval_pid" 2>/dev/null || true
  kill -INT "$eval_pid" 2>/dev/null || true
  # The evaluator writes its JSON summary from the SIGINT handler. Give it a
  # bounded window to flush before escalating; a fixed one-second sleep can
  # race this write and leave only an empty truth_eval.log.
  for _ in $(seq 1 10); do
    if [[ -f "$dir/truth_metrics.json" ]]; then
      break
    fi
    if ! kill -0 "$eval_pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  kill -TERM -- "-$node_pid" 2>/dev/null || true
  kill -TERM "$node_pid" 2>/dev/null || true
  kill -TERM -- "-$tf_pid" 2>/dev/null || true
  kill -TERM "$tf_pid" 2>/dev/null || true
  if [[ ! -f "$dir/truth_metrics.json" ]]; then
    kill -TERM -- "-$eval_pid" 2>/dev/null || true
    kill -TERM "$eval_pid" 2>/dev/null || true
    sleep 2
    kill -KILL "$eval_pid" 2>/dev/null || true
  fi
  wait "$eval_pid" 2>/dev/null || true
  if [[ ! -f "$dir/truth_metrics.json" ]]; then
    echo "truth evaluator did not produce truth_metrics.json" > "$dir/truth_eval_failure.txt"
  fi
  printf '%s\t%s\t%s\n' "$name" "$output_topic" "$dir" >> "$run_dir/results.tsv"
  grep -E "average rate|No point|Too few|Initialize|catch sig|error|ERROR" \
    "$dir/topic_hz.log" "$dir/node.log" > "$dir/key_events.log" || true
}

: > "$run_dir/results.tsv"
replay_algorithm fast_lio 195 /Odometry \
  ros2 run fast_lio fastlio_mapping --ros-args --params-file "$fastlio_params" -p use_sim_time:=true
replay_algorithm point_lio 196 /Odometry \
  ros2 run point_lio pointlio_mapping --ros-args --params-file "$pointlio_params" \
  -p use_sim_time:=true -p use_imu_as_input:=false -p prop_at_freq_of_imu:=true \
  -p check_satu:=true -p init_map_size:=10 -p point_filter_num:=3 \
  -p space_down_sample:=true -p filter_size_surf:=0.5 -p filter_size_map:=0.5 \
  -p ivox_nearby_type:=6 -p runtime_pos_log_enable:=false -p pcd_save.pcd_save_en:=false

printf 'A/B bag replay completed: %s\n' "$run_dir"
