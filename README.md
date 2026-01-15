## Prerequisite

1.
```
conda create -n aion python=3.10
conda activate aion
```
2.
```
pip install git+https://github.com/openai/CLIP.git
```
3. 
```
pip install -r requirements.txt
```
## Data
The data file can be found [here](https://drive.google.com/file/d/1TdiPQuChbyrh9JzIoBvRGpuknzoWLXFg/view?usp=drive_link).
The ckpt can be found [here](https://drive.google.com/file/d/1a_Dk09y2GhtZvlXo3c4TjSxf_WI-SdwS/view?usp=drive_link).

## Train 3D
```bash
python main.py \
    --title AIONg\
    --model AIONg\
    --gpu_ids 0\
    --workers 12\
    --vis False\
    --save_model_dir trained_models\
    --max_episode_length 50\
    --snapToGrid False\
    --train_thin 250\
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

## Evaluation

#### Evaluate Trained Model
```bash
python main.py \
    --eval \
    --test_or_val test \
    --episode_type NavTestEpisode \
    --load_model trained_models/baseline.dat \
    --model DinoAttDet \
    --results_json 3D.json \
    --gpu_ids 0 \
    --vis True \
    --save_model_dir trained_models \
    --snapToGrid False \
    --save_visuals True \
    --save_episode_data True
```

Train 3D with baseline model [ZSON, BaseModel, GCN, MJO]:
```bash
python main.py \
    --title ZSON \
    --model ZSON \
    --gpu_ids 0 \
    --workers 8 \
    --vis False \
    --save_model_dir trained_models \
    --max_episode_length 50 \
    --snapToGrid False \
    --train_thin 250 \
    --rollout_steps 128 \
    --split 18/4 \
    --add_stats True
```

#### Train Exploration Model
```bash
python main.py \
    --title AIONe \
    --model AIONe \
    --episode_type ExplorationTrainEpisode \
    --gpu_ids 0 \
    --workers 20 \
    --vis False \
    --save_model_dir trained_models \
    --max_episode_length 50 \
    --snapToGrid False \
    --train_thin 50 \
    --rollout_steps 128 \
    --scene procthor \
    --offline_data_dir ./data/procthor_offline_data/train \
    --action_space 5 \
    --max_steps 4e6
```
