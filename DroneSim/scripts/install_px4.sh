#!/bin/bash
set -e

cd ~
if [ ! -d "PX4-Autopilot" ]; then
    git clone --branch v1.16.0 https://github.com/PX4/PX4-Autopilot.git --recursive
fi
cd PX4-Autopilot
git submodule update --init --recursive
bash ./Tools/setup/ubuntu.sh
make px4_sitl