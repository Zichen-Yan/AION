#!/bin/bash
set -e

# Set configuration file
FILE_NAME="ubuntu22.04.yaml"
export SETUP_DIR=$(realpath "$(dirname "$0")")
export SCRIPT_DIR=$(realpath $SETUP_DIR/scripts)
export CONFIG_FILE=$(realpath $SETUP_DIR/scripts/$FILE_NAME)

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "\e[1;31m❌ [ERROR] $CONFIG_FILE does not exist!\e[0m" 1>&2
    exit 1
else
    echo -e "\e[1;32m✅ [INFO] Using config: $CONFIG_FILE\e[0m"
fi

# Run the installation scripts
bash $SCRIPT_DIR/install_common.sh
bash $SCRIPT_DIR/install_ros2.sh
bash $SCRIPT_DIR/install_px4.sh

