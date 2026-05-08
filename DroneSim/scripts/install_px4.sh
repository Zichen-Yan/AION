#!/bin/bash
set -e

export MODULE_DIR=~/PX4-Autopilot

git clone -b v1.16.0 \
  https://github.com/Temasek-Dynamics/PX4-Autopilot.git \
  $MODULE_DIR \
  --recursive

bash $MODULE_DIR/Tools/setup/ubuntu.sh --no-sim-tools

make -C $MODULE_DIR DONT_RUN=1 px4_sitl_default none