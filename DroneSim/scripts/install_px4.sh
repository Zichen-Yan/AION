#!/bin/bash
set -e

#cd ~
#if [ ! -d "PX4-Autopilot" ]; then
#    git clone --branch v1.16.0 https://github.com/PX4/PX4-Autopilot.git --recursive
#fi
#cd PX4-Autopilot
#git submodule update --init --recursive
#bash ./Tools/setup/ubuntu.sh
#make px4_sitl

export MODULE_DIR=~/PX4-Autopilot

git clone -b v1.13.0_raynor \
  https://github.com/Temasek-Dynamics/PX4-Autopilot.git \
  $MODULE_DIR \
  --recursive

bash $MODULE_DIR/Tools/setup/ubuntu.sh --no-sim-tools

make -C $MODULE_DIR DONT_RUN=1 px4_sitl_default none