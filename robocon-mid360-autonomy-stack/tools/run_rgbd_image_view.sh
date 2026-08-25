#!/usr/bin/env bash
# Open one RGB-D stream in an image-only Qt window for clean recording.
set -Eeuo pipefail

workspace="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
robot="${1:-robot1}"
view="${2:-depth}"
case "$robot:$view" in
  robot1:rgb) topic=/robot1/simulated_rgbd_camera/image_raw ;;
  robot1:depth) topic=/robot1/simulated_rgbd_camera/depth/image_visualized ;;
  robot2:rgb) topic=/robot2/simulated_rgbd_camera/image_raw ;;
  robot2:depth) topic=/robot2/simulated_rgbd_camera/depth/image_visualized ;;
  *)
    echo "usage: $0 {robot1|robot2} {rgb|depth}" >&2
    exit 64
    ;;
esac

source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
exec ros2 run rqt_image_view rqt_image_view "$topic"
