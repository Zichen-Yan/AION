import torch
import random

from datasets.environment import Environment
from obj_utils.action_util import get_actions
from obj_utils.net_util import gpuify
from .episode import Episode
from datasets.controller_3D import ExhaustiveBFSController
import time
import prior
import numpy as np
import math
from obj_utils.depth_transform import get_depth_ROI, depth_layer_scan_api

class ExplorationTrainEpisode(Episode):
    def __init__(self, args, gpu_id, strict_done):
        """
        Initializes the exploration episode.

        Args:
            args (Namespace): Program arguments, containing hyperparameters.
            gpu_id (int): The GPU ID to use for this episode.
            strict_done (bool): Unused for this episode type but kept for API consistency.
        """
        super(ExplorationTrainEpisode, self).__init__()

        # --- Framework and Environment Attributes ---
        self.args = args
        self.gpu_id = gpu_id
        self.strict_done = True
        self.done_count = 0
        self.forward_cnt = 0
        self.turn_cnt = 0
        self.updown_cnt = 0
        self._env = None
        self.actions = get_actions(args)

        self.model = None
        self.current_frame = None
        self.total_navigable_grids = 0
        self.bounds = None
        self.grid_size = args.grid_size

        self.procthor_dataset = prior.load_dataset("procthor-10k")
        self.target_object = None
        self.ROI = None
        self.next_ROI = None

    def set_model(self, model):
        self.model = model

    @property
    def environment(self):
        return self._env

    @property
    def actions_list(self):
        return [{"action": a} for a in self.actions]

    def state_for_agent(self):
        """Returns the current RGB frame from the environment."""
        return self.environment.current_frame

    def _preprocess_frame(self, frame):
        """Preprocesses a frame for model input."""
        # Convert to tensor and move to the correct GPU.
        state_tensor = torch.tensor(frame.copy(), dtype=torch.float32)
        state_tensor = gpuify(state_tensor, self.gpu_id)
        return self.model.resize(state_tensor.unsqueeze(0))    # Add batch dimension

    def new_episode(self, args, scenes, possible_targets=None, targets=None, rooms=None):
        """
        Resets the environment and all internal states for a new exploration run.
        """
        # 1. Clear internal state from the previous episode. This is critical.
        self.current_frame = None
        self.ROI = None
        self.next_ROI = None

        self.done_count = 0
        self.forward_cnt = 0
        self.turn_cnt = 0
        self.updown_cnt = 0
        self.move_steps = 0

        scene_index = random.randint(0, len(self.procthor_dataset["train"]) - 1)
        scene = scene_index

        if self._env is None:
            self._env = Environment(args=args, procthor_dataset=self.procthor_dataset)

        while True:
            try:
                self._env.reset(scene)
                self._env.randomize_agent_location()
                self._env.controller.move_relative(up=0.2)
                break
            except (TimeoutError, Exception):
                print("[WARN] Reset timeout")
                self._env.controller.controller.stop()
                del self._env.controller.controller
                while True:
                    try:
                        self._env.controller.controller = ExhaustiveBFSController(args=args)
                        break
                    except TimeoutError:
                        time.sleep(0.1)

        if self.model is None:
            raise ValueError("Model reference has not been set. Call set_model() before starting.")

    def step(self, action):
        action = self.actions_list[action]
        self.environment.step(action)
        reward = self.judge(action)

        # For pure exploration
        terminal = False
        success = False

        return reward, terminal, success

    def judge(self, action):
        action_name = action['action']
        # training log
        if action_name == "forward":
            self.forward_cnt += 1
        elif action_name in ["turn_left", "turn_right"]:
            self.turn_cnt += 1
        elif action_name in ["up", "down"]:
            self.updown_cnt += 1
        reward = -0.01
        ROI = self.ROI
        next_depth = self.environment.last_event.depth_frame
        next_depth = np.where(np.isnan(next_depth) | np.isinf(next_depth), 10.0, next_depth)
        height = self.environment.last_event.metadata['agent']['position']['y'] - self.environment.start_height

        angles, dists = depth_layer_scan_api(depth=next_depth, height=height)
        info = get_depth_ROI(next_depth, camera_pitch_degrees=30, height=height)

        self.next_ROI = [info[k] for k in ['center_x', 'center_y', 'found_flag', 'mean_depth', 'y_horizon']]
        self.next_ROI = torch.tensor(self.next_ROI, dtype=torch.float32).to(ROI.device)

        if ROI[2] > 0 and self.next_ROI[2]>0: # found_flag
            if action_name == 'forward':
                mean_depth = ROI[3]
                next_mean_depth = self.next_ROI[3]
                f_reward = torch.clip(mean_depth-next_mean_depth, -0.2, 0.2)
                reward += f_reward

            next_dis = math.hypot(self.next_ROI[0], 2*(self.next_ROI[1]-self.next_ROI[4]))
            if next_dis>0.3:
                c_reward = -0.75*next_dis
                reward += c_reward

        d_min = np.min(dists)
        safe_dist = 1.0/3.0
        if d_min <= safe_dist:
            reward += 1 - np.exp(2 * (safe_dist - d_min))

        return reward