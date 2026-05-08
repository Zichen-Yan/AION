#!/bin/bash
set -e

MODULE_NAME="PX4-Autopilot"

# Install yq if missing
if ! command -v yq &>/dev/null; then
    sudo apt update
    sudo snap install yq
fi

# Clone PX4
URL=$(yq e ".github_repos[] | select(.name == \"$MODULE_NAME\") | .url" $CONFIG_FILE)
BRANCH=$(yq e ".github_repos[] | select(.name == \"$MODULE_NAME\") | .branch" $CONFIG_FILE)
MODULE_DIR="$HOME/PX4-Autopilot"

if [ ! -d "$MODULE_DIR" ]; then
    git clone -b "$BRANCH" "$URL" "$MODULE_DIR" --recursive
else
    echo "[INFO] $MODULE_DIR already exists, skipping clone."
fi

bash "$MODULE_DIR/Tools/setup/ubuntu.sh" --no-sim-tools
make -C "$MODULE_DIR" DONT_RUN=1 px4_sitl_default none

# Download QGroundControl
QGC_NAME="QGroundControl.AppImage"
QGC_DIR="$SETUP_DIR/submodules/QGroundControl"
QGC_VERSION=$(yq e ".github_repos[] | select(.name == \"$MODULE_NAME\") | .QGC_VERSION" $CONFIG_FILE)
QGC_URL="https://github.com/mavlink/qgroundcontrol/releases/download/$QGC_VERSION/$QGC_NAME"

if [ ! -f "$QGC_DIR/$QGC_NAME" ]; then
    wget "$QGC_URL" -P "$QGC_DIR"
    chmod +x "$QGC_DIR/$QGC_NAME"
else
    echo "[INFO] $QGC_NAME already exists, skipping download."
fi