## Terminal 1 Start Simulator
### 1.source
```bash
# start pegasus simulator in IsaacSim with PX4 SITL
conda deactivate
source /home/zichen/IsaacSim-ros_workspaces/build_ws/humble/humble_ws/install/setup.bash
```
### 2.start IsaacSim 
```bash
# start pegasus simulator in IsaacSim with PX4 SITL
ISAACSIM_PYTHON [TBD]/AION/DroneSim/isaac_env.py
```
## Terminal 2 Run ObjNav
### 3. excute main_isaacsim.py 
```bash
MicroXRCEAgent udp4 -p 8888
python main_isaacsim.py
```

