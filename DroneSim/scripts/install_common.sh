#!/bin/bash
set -e

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] CONFIG_FILE is not set or does not exist." 1>&2
    exit 1
fi

# Ensure yq is installed
if ! command -v yq &>/dev/null; then
    sudo snap install yq
fi

# Install system packages
sudo apt update -y
while read package; do
    if ! dpkg -l | grep -q "^ii  $package "; then
        sudo apt install -y "$package"
    fi
done < <(yq e '.system_packages[]' "$CONFIG_FILE")

# Install Python packages
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python3 is not installed." 1>&2
    exit 1
fi

if ! pip3 list &>/dev/null; then
    sudo apt install -y python3-pip
fi

python3 -m pip install --upgrade pip setuptools wheel

while read package; do
    PACKAGE_NAME=$(echo "$package" | cut -d= -f1)
    if ! python3 -m pip show "$PACKAGE_NAME" &>/dev/null; then
        python3 -m pip install "$package"
    fi
done < <(yq e '.python_packages[]' "$CONFIG_FILE")
