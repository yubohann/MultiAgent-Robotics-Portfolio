#!/usr/bin/env bash
# Replay one recorded Livox + Gazebo-truth bag through both estimators.
set -o pipefail

workspace="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
pointlio_install="${POINTLIO_INSTALL:-$workspace/../mid360_pointlio_ab/install}"
run_dir="${1:?usage: replay_odom_truth.sh RUN_DIRECTORY BAG_DIRECTORY}"
bag_dir="${2:?usage: replay_odom_truth.sh RUN_DIRECTORY BAG_DIRECTORY}"
fastlio_params="${FASTLIO_PARAMS:-$workspace/install/robocon_mid360_simulation/share/robocon_mid360_simulation/config/fast_lio_simulation.yaml}"
pointlio_params="$pointlio_install/point_lio/share/point_lio/config/mid360.yaml"

source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
if [[ ! -f "$pointlio_install/setup.bash" ]]; then
  echo "Point-LIO setup file not found: $pointlio_install/setup.bash" >&2
  exit 2
fi
source "$pointlio_install/setup.bash"
set -euo pipefail

if [[ ! -f "$bag_dir/metadata.yaml" ]]; then
  echo "缺少 rosbag metadata.yaml: $bag_dir" >&2
  exit 2
fi
if [[ ! -x "$pointlio_install/point_lio/lib/point_lio/pointlio_mapping" ]]; then
  echo "缺少 Point-LIO 安装: $pointlio_install" >&2
  exit 2
fi
mkdir -p "$run_dir"
printf 'evidence=gazebo_simulation/bag_replay\ninput_bag=%s\n' "$bag_dir" > "$run_dir/manifest.txt"
printf 'ground_truth_topic=/simulation/ground_truth/odom\n' >> "$run_dir/manifest.txt"

stop_group() {
  local pid="$1"
  kill -INT -- "-$pid" 2>/dev/null || true
  kill -INT "$pid" 2>/dev/null || true
  sleep 1
  kill -TERM -- "-$pid" 2>/dev/null || true
  kill -TERM "$pid" 2>/dev/null || true
}

replay_one() {
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
  timeout --signal=INT --kill-after=5s 45s ros2 bag play "$bag_dir" --clock --rate 1.0 \
    > "$dir/bag_play.log" 2>&1 || true
  wait "$hz_pid" 2>/dev/null || true
  stop_group "$node_pid"
  stop_group "$tf_pid"
  stop_group "$eval_pid"
  sleep 2
  kill -KILL "$eval_pid" 2>/dev/null || true
  wait "$eval_pid" 2>/dev/null || true
  printf '%s\t%s\t%s\n' "$name" "$output_topic" "$dir" >> "$run_dir/results.tsv"
}

: > "$run_dir/results.tsv"
replay_one fast_lio 205 /Odometry \
  ros2 run fast_lio fastlio_mapping --ros-args --params-file "$fastlio_params" -p use_sim_time:=true
replay_one point_lio 206 /Odometry \
  ros2 run point_lio pointlio_mapping --ros-args --params-file "$pointlio_params" \
  -p use_sim_time:=true -p use_imu_as_input:=false -p prop_at_freq_of_imu:=true \
  -p check_satu:=true -p init_map_size:=10 -p point_filter_num:=3 \
  -p space_down_sample:=true -p filter_size_surf:=0.5 -p filter_size_map:=0.5 \
  -p ivox_nearby_type:=6 -p runtime_pos_log_enable:=false -p pcd_save.pcd_save_en:=false

printf 'A/B truth replay completed: %s\n' "$run_dir"
