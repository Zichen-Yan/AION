[![arXiv](https://img.shields.io/badge/arXiv-2310.07473-b31b1b.svg)](https://arxiv.org/pdf/2601.15614)
[![YouTube](https://img.shields.io/badge/YouTube-video-FF0000?logo=youtube&logoColor=white)](https://youtu.be/TgsUm6bb7zg)

## AION implements end2end 3D ObjectNav from AI2THOR training to Isaac-sim deployment on a realistic drone.

# 1. Training and Evaluation in AI2THOR 
![img.png](figures/ai2thor.png)
## 1.1 Prerequisite
```
conda create -n aion python=3.10
conda activate aion

cd ~
git clone https://github.com/Zichen-Yan/AION.git
cd ~/AION
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```
## 1.2 Data Preparation
The [data/](https://drive.google.com/file/d/1TdiPQuChbyrh9JzIoBvRGpuknzoWLXFg/view?usp=drive_link) and [ckpt/](https://drive.google.com/file/d/1a_Dk09y2GhtZvlXo3c4TjSxf_WI-SdwS/view?usp=drive_link) 
dirs should be put in the root.

The [IsaacSimAssets/](https://drive.google.com/file/d/1pSPtM-zyJQVePGHsuWaqhnbXPOmrfEyr/view?usp=drive_link) and 
[Scenes/](https://drive.google.com/file/d/1zsqSrWUbcPsBzIngnzc_AkfnI-mn4vvF/view?usp=drive_link) dirs should be put in DroneSim/.

## 1.3 Train Goal-Reaching Policy
### AIONg 
#### split = [14/8, 18/4]
```bash
python main.py \
    --title AIONg\
    --model AIONg\
    --gpu_ids 0\
    --workers 25\
    --vis False\
    --save_model_dir trained_models\
    --max_episode_length 50\
    --snapToGrid False\
    --rollout_steps 128\
    --max_steps 1e7\
    --split 18/4\
    --add_clip_align True\
    --add_stats True\
    --add_depth True\
    --add_rgb True\
    --add_dis_reward True\
    --add_bbox_reward True\
    --add_parent_reward True\
    --add_collision_reward True
```
 
### Baselines
#### model = [ZSON, BaseModel, GCN, MJO]
```bash
python main.py \
    --title ZSON \
    --model ZSON \
    --gpu_ids 0 \
    --workers 30 \
    --vis False \
    --save_model_dir trained_models \
    --max_episode_length 50 \
    --snapToGrid False \
    --rollout_steps 128 \
    --max_steps 1e7 \
    --split 18/4 \
    --add_stats True
```

## 1.4 Evaluate Goal-Reaching Policy
### AIONg
#### get_seen_data = [False, True]
```bash
python main.py \
    --eval \
    --test_or_val test \
    --episode_type NavTestEpisode \
    --load_model ckpt/AION-g-18-4.dat \
    --model AIONg \
    --results_json 3D.json \
    --gpu_ids 0 \
    --vis False \
    --get_seen_data False \
    --snapToGrid False \
    --split 18/4 \
    --save_visuals True \
    --save_episode_data True
```

#### Baselines
```bash
python main.py \
    --eval \
    --test_or_val test \
    --episode_type NavTestEpisode \
    --load_model ckpt/ZSON.dat \
    --model ZSON \
    --results_json 3D.json \
    --gpu_ids 0 \
    --vis False \
    --get_seen_data False \
    --snapToGrid False \
    --split 18/4 \
    --save_visuals True \
    --save_episode_data True
```

## 1.5 Train Exploration Policy
### AIONe
```bash
python main.py \
    --title AIONe \
    --model AIONe \
    --episode_type ExplorationTrainEpisode \
    --gpu_ids 0 \
    --workers 40 \
    --vis False \
    --save_model_dir trained_models \
    --max_episode_length 50 \
    --snapToGrid False \
    --rollout_steps 128 \
    --scene procthor \
    --offline_data_dir ./data/procthor_offline_data/train \
    --action_space 5 \
    --max_steps 4e6
```

# 2. IsaacSim Env Installation
[![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-E95420)](https://ubuntu.com/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB)](https://www.python.org/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-22314E)](https://docs.ros.org/en/humble/)
[![IsaacSim](https://img.shields.io/badge/Isaac%20Sim-5.1.0-76B900)](https://developer.nvidia.com/isaac/sim)
[![Pegasus](https://img.shields.io/badge/Pegasus-enabled-blue)](https://github.com/PegasusSimulator/PegasusSimulator)

![img.png](figures/isaac.png)
## 2.1 Download [ISAAC-SIM](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/quick-install.html) to 
### ~/.local/share/ov/pkg/isaac-sim-5.1.0
## 2.2 Modify .bashrc
```bash
export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-5.1.0/"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
alias ISAACSIM_PYTHON="$ISAACSIM_PATH/python.sh"
alias ISAACSIM="$ISAACSIM_PATH/isaac-sim.sh"
```
## 2.3 ROS2 PX4 Prerequisite
#### Change the ROS2 name to your own version in DroneSim/scripts/install_ros2.sh (ROS2_VERSION, default is "humble") 
```bash
cd ~
git clone https://github.com/isaac-sim/IsaacSim-ros_workspaces.git

cd ~/AION/DroneSim/
bash install.sh
```
## 2.4 Install Micro-XRCE-DDS
```bash
cd ~
git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig /usr/local/lib/
```
## 2.5 Install Pegasus Simulator
```bash
cd ~
git clone https://github.com/PegasusSimulator/PegasusSimulator.git
cd ~/PegasusSimulator/extensions
ISAACSIM_PYTHON -m pip install -e pegasus.simulator/
```
## 2.6 Modify the PX4 path in
#### PegasusSimulator/extensions/pegasus.simulator/config/configs.yaml
```bash
px4_dir: ~/PX4-Autopilot
```
## 2.7 Compile px4msgs
```bash
sudo apt install python3-empy python3-colcon-common-extensions

mkdir -p ~/px4msg_ws/src
cd ~/px4msg_ws/src
git clone -b release/1.16 https://github.com/PX4/px4_msgs.git
cd ~/px4msg_ws
colcon build
echo "source ~/px4msg_ws/install/setup.bash" >> ~/.bashrc
```
# 3. Evaluation in ISAAC-SIM
## 3.1 Terminal 1 Start Simulator
```bash
# start pegasus simulator in IsaacSim with PX4 SITL
conda deactivate
source ~/IsaacSim-ros_workspaces/build_ws/humble/humble_ws/install/setup.bash

cd ~/AION
export MDL_USER_PATH=DroneSim/Scenes/MTL
ISAACSIM_PYTHON DroneSim/isaac_env.py
```
## 3.2 Terminal 2 Run DDS
```bash
MicroXRCEAgent udp4 -p 8888
```
## 3.3 Terminal 3 Run Algorithm
```bash
conda activate aion
cd ~/AION
python main_isaacsim.py
```
# 4. Citation
```
@article{yan2026aion,
  title={AION: Aerial Indoor Object-Goal Navigation Using Dual-Policy Reinforcement Learning},
  author={Yan, Zichen and Hou, Yuchen and Wang, Shenao and Gao, Yichao and Huang, Rui and Zhao, Lin},
  journal={arXiv preprint arXiv:2601.15614},
  year={2026}
}
```
