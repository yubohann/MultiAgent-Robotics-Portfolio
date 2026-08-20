#!/usr/bin/env bash
set -eo pipefail

workspace="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
run_dir=${1:?run directory required}
map_file="${MAP_FILE:?set MAP_FILE to an eligible frozen PCD path}"
source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-171}"
exec "$workspace/src/robocon_mid360_simulation/scripts/run_fixed_map_localization_smoke.sh" "$run_dir" "$map_file" 25
