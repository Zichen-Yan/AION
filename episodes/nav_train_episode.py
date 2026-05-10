import random
import torch

from datasets.constants import GOAL_SUCCESS_REWARD, STEP_PENALTY, DONE
from datasets.environment import Environment

from obj_utils.action_util import get_actions
from obj_utils.net_util import gpuify
from .episode import Episode
from datasets.controller_3D import ExhaustiveBFSController

import json
import numpy as np
import time

c2p_prob = json.load(open("./data/c2p_prob.json"))

class NavTrainEpisode(Episode):
    """ Episode for Navigation. """

    def __init__(self, args, gpu_id, strict_done=False):
        super(NavTrainEpisode, self).__init__()

        self.args = args
        self._env = None
        self.gpu_id = gpu_id
        self.strict_done = strict_done
        self.task_data = None
        self.actions = get_actions(args)
        self.done_count = 0
        self.target_object = None
        self.prev_frame = None
        self.current_frame = None

        self.seen_list = []
        if args.eval:
            random.seed(args.seed)
        self.room = None
        self.box_area_ratio = 0
        self.stats = None
        self.success_distance = args.success_distance
        self.move_steps = 0

    @property
    def environment(self):
        return self._env

    @property
    def actions_list(self):
        return [{"action": a} for a in self.actions]

    def state_for_agent(self):
        return self.environment.current_frame

    def current_agent_position(self):
        """ Get the current position of the agent in the scene. """
        return self.environment.current_agent_position

    def step(self, action):
        action = self.actions_list[action]
        if action["action"] != DONE:
            self.environment.step(action)
        else:
            self.done_count += 1

        movement_actions = ["forward", "up", "down"]
        if action["action"] in movement_actions:
            self.move_steps += 1

        reward, terminal, action_was_successful = self.judge(action)
        if self.environment.controller.env_collapse == True:
            terminal = True
            action_was_successful = False
        return reward, terminal, action_was_successful

    def judge(self, action):
        """ Judge the last event. """
        # 1. step reward
        reward = STEP_PENALTY
        # 2. dist reward
        if self.args.add_dis_reward:
            cur_state = self.environment.last_event.metadata['agent']
            pos = cur_state['position']

            min_dis = 100
            for obj in self.task_data:
                target_pos = [float(v) for v in obj.split('|')[1:]]
                dist = np.sqrt(
                    (pos['x'] - target_pos[0]) ** 2 + (pos['y'] - target_pos[1]) ** 2 + (pos['z'] - target_pos[2]) ** 2)
                min_dis = min(min_dis, dist)

            if self.pre_dis != 0:
                reward += self.pre_dis - min_dis
            self.pre_dis = min_dis
        # 3. parent reward
        if self.args.add_parent_reward:
            if action["action"] != DONE:
                reward += self.get_partial_reward()
        # 4. Obj Box Area
        if self.args.add_bbox_reward:
            reward += min(self.box_area_ratio, 0.1)
        # 5. collision penalty
        if self.environment.controller.collision and self.args.add_collision_reward:
            reward -= 0.1
        # 6. success reward
        self.min_dis = 100
        task_success = False
        if action["action"] == DONE:
            for obj in self.environment.last_event.metadata['objects']:
                name = obj['objectId'].split('|')[0]
                if name == self.target_object and obj['visible']:
                    cur_state = self.environment.last_event.metadata['agent']
                    pos = cur_state['position']

                    target_pos = [float(v) for v in obj['objectId'].split('|')[1:]]
                    dist = np.sqrt((pos['x'] - target_pos[0]) ** 2 + (pos['y'] - target_pos[1]) ** 2 + (
                            pos['z'] - target_pos[2]) ** 2)
                    self.min_dis = dist

                    if (dist <= self.success_distance and self.is_center_in_middle(self.stats[0], self.stats[1],
                                                                                   threshold=0.8)):
                        reward += GOAL_SUCCESS_REWARD
                        task_success = True
                        break
            self.seen_list = []

        reward =float(reward) if not isinstance(reward, torch.Tensor) else float(reward.item())
        return reward, action["action"] == DONE, task_success

    def is_center_in_middle(self, center_x, center_y, threshold):
        x_min = 0.5 - threshold / 2
        x_max = 0.5 + threshold / 2
        y_min = 0.5 - threshold / 2
        y_max = 0.5 + threshold / 2

        in_center = (center_x >= x_min) & (center_x <= x_max) & \
                    (center_y >= y_min) & (center_y <= y_max)
        return in_center

    # Set the target index.
    @property
    def target_object_index(self):
        """ Return the index which corresponds to the target object. """
        return self._target_object_index

    @target_object_index.setter
    def target_object_index(self, target_object_index):
        """ Set the target object by specifying the index. """
        self._target_object_index = gpuify(torch.LongTensor([target_object_index]), self.gpu_id)

    def get_partial_reward(self):
        """ get partial reward if parent object is seen for the first time"""
        reward = 0
        reward_dict = {}
        if self.target_parents is not None:
            for parent_type in self.target_parents:
                parent_ids = self.environment.find_id(parent_type)
                for parent_id in parent_ids:
                    if self.environment.object_is_visible(parent_id) and parent_id not in self.seen_list:
                        reward_dict[parent_id] = self.target_parents[parent_type]
        if len(reward_dict) != 0:
            v = list(reward_dict.values())
            k = list(reward_dict.keys())
            reward = max(v)           # pick one with greatest reward if multiple in scene
            self.seen_list.append(k[v.index(reward)])
        return reward

    def _new_episode(self, args, scenes, possible_targets, targets=None, room = None):
        """ New navigation episode. """
        if "FloorPlan212" in scenes: # camera issue in this scene
            scenes.remove("FloorPlan212")
        scene = random.choice(scenes)
        self.room = room
        if self._env is None:
            self._env = Environment(args=args)

        while True:
            try:
                self._env.reset(scene)
                self._env.randomize_agent_location()
                self._env.controller.move_relative(up=0.2)
                break
            except (TimeoutError, Exception):
                self._env.controller.controller.stop()
                del self._env.controller.controller
                while True:
                    try:
                        self._env.controller.controller = ExhaustiveBFSController(args=args)
                        break
                    except TimeoutError:
                        time.sleep(0.1)

        objects = self._env.all_objects()

        visible_objects = [obj.split("|")[0] for obj in objects]
        intersection = [obj for obj in visible_objects if obj in targets]

        self.task_data = []

        idx = random.randint(0, len(intersection) - 1)
        goal_object_type = intersection[idx]
        self.target_object = goal_object_type

        for id_ in objects:
            type_ = id_.split("|")[0]
            if goal_object_type == type_:
                self.task_data.append(id_)

        child_object = self.task_data[0].split("|")[0]
        try:
            self.target_parents = c2p_prob[self.room][child_object]
        except:
            self.target_parents = None

    def new_episode(
        self,
        args,
        scenes,
        possible_targets=None,
        targets=None,
        rooms=None,
    ):
        self.pre_dis = 0
        self.done_count = 0
        self.move_steps = 0
        self.prev_frame = None
        self.current_frame = None
        self._new_episode(args, scenes, possible_targets, targets, rooms)
