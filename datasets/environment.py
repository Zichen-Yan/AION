"""A wrapper for engaging with the THOR environment."""

import copy
import json
import os
import random
from .controller_3D import OfflineControllerWithSmallRotation as Three

class Environment:
    """ Abstraction of the ai2thor enviroment. """

    def __init__(self, args, procthor_dataset=None):
        self.offline_data_dir = args.offline_data_dir
        self.procthor_dataset = procthor_dataset
        self.controller = Three(args=args)

        self._reachable_points = None
        self.start_state = None
        self.last_action = None

    @property
    def scene_name(self):
        return self.controller.scene_name

    @property
    def current_frame(self):
        return self.controller.last_event.frame

    @property
    def current_objs(self):
        return self.controller.last_event.objbb

    @property
    def last_event(self):
        return self.controller.last_event

    @property
    def last_action_success(self):
        return self.controller.last_event.metadata["lastActionSuccess"]

    def object_is_visible(self, objId):
        objects = self.last_event.metadata["objects"]
        visible_objects = [o["objectId"] for o in objects if o["visible"]]
        return objId in visible_objects

    def find_id(self, parentType):
        return self.controller.find_id(parentType)

    def reset(self, scene_name):
        if self.controller.args.episode_type == 'ExplorationTrainEpisode':
            scene_object = self.procthor_dataset["train"][scene_name]
            scene_object['idx'] = str(scene_name)
            self.controller.reset(scene_object)
        else:
            self.controller.reset(scene_name)

    def all_objects(self):
        objects = self.last_event.metadata["objects"]
        return [o["objectId"] for o in objects]

    def step(self, action_dict):
        return self.controller.step(action_dict)

    def random_reachable_state(self, seed=None):
        """ Get a random reachable state. """
        if seed is not None:
            random.seed(seed)
        xyz = random.choice(self.reachable_points)
        rotation = random.choice([0, 45, 90, 135, 180, 225, 270, 315])
        xyz['y'] = 0.5
        horizon = random.choice([30])
        state = copy.copy(xyz)
        state["rotation"] = rotation
        state["horizon"] = horizon
        return state

    def randomize_agent_location(self, seed=None):
        state = self.random_reachable_state(seed=seed)
        self.controller.teleport_agent_to(**state)
        self.start_height = 0.2 # self.last_event.metadata['agent']['position']['y']
        return

    @property
    def reachable_points(self):
        """ Use the JSON file to get the reachable points. """
        if not self.controller.args.scene == 'procthor':
            if self._reachable_points is not None:
                return self._reachable_points

        if isinstance(self.scene_name, dict) and 'idx' in self.scene_name.keys():
            points_path = os.path.join(self.offline_data_dir, self.scene_name['idx'], "grid.json")
        else:
            points_path = os.path.join(self.offline_data_dir, self.scene_name, "grid.json")

        if os.path.exists(points_path):
            self._reachable_points = json.load(open(points_path))
        else:
            raise

        return self._reachable_points
