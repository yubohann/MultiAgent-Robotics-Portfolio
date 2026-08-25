# FUEL External Build Boundary

This directory contains only the build recipe and provenance lock for the
upstream FUEL exploration planner. FUEL is GPL-3.0-only. Its source, compiled
binaries, ROS graph, and any future adapter must stay in a separate container
or process; none belong in the AeroCityBench BSD core.

The Docker build context must be a clean checkout of the commit in
`source-lock.json`. The benchmark repository is not a build context and is not
copied into the image. The recipe intentionally builds only FUEL's planning
packages and `quadrotor_msgs`; it does not build RViz, the upstream simulator,
or an AeroCityBench bridge.

On this host, first verify the existing locked checkout without invoking Docker:

```powershell
python tools/build_fuel_container.py --source E:\github_repos\FUEL-locked-662dd23c7b52 --verify-only
```

Then build through WSL Docker. The image must remain local until an independent
license and publication review approves any distribution:

```powershell
wsl.exe -d Ubuntu-22.04 -- docker build --file /mnt/c/Users/Administrator/IsaacLab/isaac_drone_racer/experiments/aerocity-bench/external/fuel/Dockerfile --build-arg FUEL_UPSTREAM_URL=https://github.com/HKUST-Aerial-Robotics/FUEL.git --build-arg FUEL_UPSTREAM_COMMIT=662dd23c7b52b258d3c4a0155ff6632118e8984f --tag aerocity-external-fuel:662dd23c7b52 /mnt/e/github_repos/FUEL-locked-662dd23c7b52
```

A successful build proves only that the locked upstream planner compiles in an
isolated environment. It does not prove a G2-I integration, an L0/L1 result,
or formal-score eligibility.

The repository also contains a single-UAV ROS graph smoke that starts only
FUEL's exploration node, provides synthetic public-style odometry, pose, and
local point-cloud inputs, and captures `/planning/bspline`. It runs with no
network and a read-only root filesystem; it never starts FUEL's trajectory
server or a position-command topic:

```powershell
python tools/run_fuel_ros_smoke.py --source E:\github_repos\FUEL-locked-662dd23c7b52 --duration-s 12 --output reason\benchmark-external-methodology-audit-20260802\fuel-ros-smoke-20260803.json
```

The current smoke establishes the ROS input graph but reports no route. That
negative result is preserved because FUEL requires a dense ray-built free-space
map, while G1-U currently exposes only coarse occupied voxels. Do not solve
this by providing private scene geometry or by changing FUEL thresholds; either
would make the method incomparable. FUEL is not a G2-I integration or C-gate
external method.
