# Build with the clean, locked FUEL checkout as the context, not this repository.
# The resulting GPL image remains an external container and must not be published
# as an AeroCityBench package artifact.
ARG ROS_BASE=ros@sha256:72b8bc59035dc0a5b8e07aae28c16caa84192971d72d207c72ed734fb1d5e97d
FROM ${ROS_BASE}

ARG FUEL_UPSTREAM_URL
ARG FUEL_UPSTREAM_COMMIT
LABEL org.opencontainers.image.source=${FUEL_UPSTREAM_URL}
LABEL org.opencontainers.image.revision=${FUEL_UPSTREAM_COMMIT}
LABEL org.opencontainers.image.licenses="GPL-3.0-only"
LABEL org.aerocity.bench.boundary="external-container-only"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# FUEL's upstream CMake hard-codes /usr/local for NLopt. The symlinks preserve
# that unmodified upstream contract while using the Focal distribution package.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libarmadillo-dev \
        libeigen3-dev \
        libnlopt-dev \
        libopencv-dev \
        libpcl-dev \
        ros-noetic-cv-bridge \
        ros-noetic-image-transport \
        ros-noetic-message-filters \
        ros-noetic-pcl-conversions \
        ros-noetic-pcl-ros \
        ros-noetic-tf \
        ros-noetic-tf-conversions \
    && ln -sf /usr/include/nlopt.h /usr/local/include/nlopt.h \
    && ln -sf /usr/lib/x86_64-linux-gnu/libnlopt.so /usr/local/lib/libnlopt.so \
    && rm -rf /var/lib/apt/lists/*

# FUEL includes nlopt.hpp while its unmodified CMake links the C library above.
# Keep this small header-only compatibility addition separate so rebuilding it
# does not duplicate the large ROS/PCL dependency layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libnlopt-cxx-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/fuel/ws
COPY . /opt/fuel/ws/src/fuel

# The locked checkout is verified clean on the host before Docker starts. A
# Windows checkout carries CRLF working-tree files into this Linux image, so
# retain its checkout convention before checking its status here.
RUN test -n "${FUEL_UPSTREAM_URL}" \
    && test -n "${FUEL_UPSTREAM_COMMIT}" \
    && test "$(git -C src/fuel remote get-url origin)" = "${FUEL_UPSTREAM_URL}" \
    && test "$(git -C src/fuel rev-parse HEAD)" = "${FUEL_UPSTREAM_COMMIT}" \
    && git -C src/fuel config core.autocrlf true \
    && test -z "$(git -C src/fuel status --porcelain)" \
    && source /opt/ros/noetic/setup.bash \
    && catkin_make -DCMAKE_BUILD_TYPE=Release \
        -DCATKIN_WHITELIST_PACKAGES=quadrotor_msgs\;plan_env\;path_searching\;poly_traj\;bspline\;active_perception\;bspline_opt\;traj_utils\;lkh_tsp_solver\;plan_manage\;exploration_manager

CMD ["bash", "-lc", "source /opt/ros/noetic/setup.bash && source /opt/fuel/ws/devel/setup.bash && rospack find exploration_manager"]
