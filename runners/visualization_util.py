# visualization_util.py

import os
import cv2
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
import torch

class ResizeTo224:
    def __init__(self, size=(224, 224), normalize=True):
        # size can be int or (H, W). For ResNet-18, (224,224) is typical.
        self.size = size
        self.normalize = normalize
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts:
          - (H, W, C) uint8 or float
          - (C, H, W) uint8 or float
          - (N, H, W, C)
          - (N, C, H, W)
        Returns:
          - (N, 3, H, W) float32, normalized if normalize=True
        """
        # Ensure batch dim and channel-last -> channel-first
        if x.ndim == 3:
            # Heuristic: treat as HWC if last dim looks like channels
            if x.shape[-1] in (1, 3, 4):
                x = x.permute(2, 0, 1)  # HWC -> CHW
            # else assume already CHW
            x = x.unsqueeze(0)  # -> NCHW
        elif x.ndim == 4:
            # Convert NHWC -> NCHW if needed
            if x.shape[-1] in (1, 3, 4) and x.shape[1] not in (1, 3, 4):
                x = x.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(f"Unsupported shape {tuple(x.shape)}")

        # To float in [0,1] if integer input
        if x.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            x = x.float() / 255.0
        else:
            x = x.float()

        # Resize to exact (H, W) — no crop needed for square output
        x = TF.resize(
            x, self.size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True
        )

        if self.normalize:
            # Move mean/std to same device
            mean = self.mean.to(x.device, x.dtype)
            std  = self.std.to(x.device, x.dtype)
            x = (x - mean) / std

        return x

def save_episode_data(player, count, rank):
    root_dir = "visualizations"
    scene_name = player.environment.scene_name
    target_object = player.episode.target_object
    
    episode_dir = os.path.join(root_dir, scene_name, f"{target_object}_{count}")
    os.makedirs(episode_dir, exist_ok=True)
    file_path = os.path.join(episode_dir, "trajectory_data.json")

    map_event = player.environment.controller.step(action="ToggleMapView")
    scene_bounds = map_event.metadata['sceneBounds']
    player.environment.controller.step(action="ToggleMapView")

    target_ground_truth = None
    if player.episode.task_data:
        target_ground_truth = player.episode.task_data[0]

    path_length = len(player.trajectory)
    trajectory = player.trajectory
    data_to_save = {
        "metadata": {
            "scene_name": scene_name,
            "target_object": target_object,
            "scene_bounds": scene_bounds,
            "target_ground_truth": target_ground_truth
        },
        "results": {
            "path_length": path_length
        },
        "trajectory": trajectory
    }

    with open(file_path, 'w') as f:
        json.dump(data_to_save, f, indent=4)

    print(f"[{rank}] Saved episode data to {file_path}")

def save_third_person_video(player, real_controller, episode_save_path, rank):
    """
    Generates and saves ONLY the third-person video for an episode.
    """
    print(f"[{rank}] Generating third-person video...")
    video_path = os.path.join(episode_save_path, "video.mp4")
    video_writer = None

    if player.trajectory:
        initial_state = player.trajectory[0]['state']
        agent_pos = initial_state['position']
        agent_yaw_rad = math.radians(initial_state['rotation']['y'])
        D, H = 0.8, 0.75
        
        cam_x = agent_pos['x'] - D * math.sin(agent_yaw_rad)
        cam_z = agent_pos['z'] - D * math.cos(agent_yaw_rad)
        cam_y = agent_pos['y'] + H
        cam_yaw = initial_state['rotation']['y']
        cam_pitch = math.degrees(math.atan2(H, D))

        real_controller.step(action='AddThirdPartyCamera', position={'x': cam_x, 'y': cam_y, 'z': cam_z}, rotation={'x': cam_pitch, 'y': cam_yaw, 'z': 0})

        for trajectory_item in player.trajectory:
            state = trajectory_item['state']
            real_controller.step(action='TeleportFull', position=state['position'], rotation=state['rotation'], horizon=state['horizon'], forceAction=True)

            agent_pos = state['position']
            agent_yaw_rad = math.radians(state['rotation']['y'])
            cam_x = agent_pos['x'] - D * math.sin(agent_yaw_rad)
            cam_z = agent_pos['z'] - D * math.cos(agent_yaw_rad)
            cam_y = agent_pos['y'] + H
            cam_yaw = state['rotation']['y']
            cam_pitch = math.degrees(math.atan2(H, D))

            event = real_controller.step(action='UpdateThirdPartyCamera', position={'x': cam_x, 'y': cam_y, 'z': cam_z}, rotation={'x': cam_pitch, 'y': cam_yaw, 'z': 0})
            
            if event.third_party_camera_frames:
                frame = event.third_party_camera_frames[0]
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if video_writer is None:
                    height, width, _ = frame.shape
                    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))
                video_writer.write(frame)

    if video_writer is not None:
        video_writer.release()
        print(f"[{rank}] Saved third-person video to {video_path}")
    else:
        print(f"[{rank}] Skipping video generation as no frames were produced.")

def generate_visualizations(player, count, rank):
    """
    Generates and saves all visualizations (top-down map and third-person video) for an episode.
    This is the main function to be called from the validation script.
    """
    print(f"[{rank}] Generating visualizations for episode {count}...")
    
    VIZ_ROOT_DIR = "visualizations"
    scene_name = player.environment.scene_name
    target_object = player.episode.target_object
    episode_save_path = os.path.join(VIZ_ROOT_DIR, scene_name, f"{target_object}_{count}")
    os.makedirs(episode_save_path, exist_ok=True)

    try:
        real_controller = player.environment.controller.controller
        save_third_person_video(player, real_controller, episode_save_path, rank)

    except Exception as e:
        print(f"[{rank}] A critical error occurred during visualization generation: {e}")