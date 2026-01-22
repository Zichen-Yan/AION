""" Exhaustive BFS and Offline Controller. """

from collections import deque
import json
import time
import math
import os
from ai2thor.platform import CloudRendering

from ai2thor.controller import Controller
from .base_controller import BaseController, ThorAgentState

class ExhaustiveBFSController(Controller):
    def __init__(self, args):
        super(ExhaustiveBFSController, self).__init__(
            agentMode="drone",
            grid_size=args.grid_size,
            snapToGrid=args.snapToGrid,

            visibilityDistance=5.0,
            width=300, # 640
            height=300, # 480
            fieldOfView=args.fov,

            renderDepthImage=True,
            renderClassImage=False,
            renderInstanceSegmentation=True,

            platform=None if args.vis else CloudRendering,
            server_timeout=10.0
        )
        self.allow_enqueue = True
        self.queue = deque()
        self.seen_points = []
        self.grid_points = []
        self.seen_states = []
        self.bad_seen_states = []
        self.visited_seen_states = []
        self.grid_states = []
        self.grid_size = args.grid_size
        self._check_visited = False
        self.scene_name = None
        self.y = None
        self.metadata = {}

class CostumeEvent:
    """ A stripped down version of an event. Only contains lastActionSuccess, sceneName,
        and optionally state and frame. Does not contain the rest of the metadata. """

    def __init__(self, last_action_success, scene_name, state=None, frame=None):
        self.metadata = {
            "lastActionSuccess": last_action_success,
            "sceneName": scene_name,
        }
        if state is not None:
            self.metadata["agent"] = {}
            self.metadata["agent"]["position"] = state.position()
            self.metadata["agent"]["rotation"] = {
                "x": 0.0,
                "y": state.rotation,
                "z": 0.0,
            }
            self.metadata["agent"]["cameraHorizon"] = state.horizon
        self.frame = frame


class OfflineControllerWithSmallRotation(BaseController):
    def __init__(
        self,
        args,
        grid_file_name="grid.json",
        metadata_file_name="visible_object_map.json"
    ):
        super(OfflineControllerWithSmallRotation, self).__init__()
        self.args = args
        self.grid_size = args.grid_size
        self.offline_data_dir = args.offline_data_dir
        self.grid_file_name = grid_file_name
        self.graph = None
        self.metadata_file_name = metadata_file_name
        self.grid = None
        self.metadata = None
        self.actions = None
        self.y = None
        self.last_event = None
        while True:
            try:
                self.controller = ExhaustiveBFSController(args=args)
                break
            except (TimeoutError, Exception):
                time.sleep(0.1)

        self.scene_name = None
        self.state = None
        self.last_action_success = True

    def teleport_agent_to(self, x, y, z, rotation, horizon):
        """ Teleport the agent to (x,y,z) with given rotation and horizon. """
        self.last_event = self.controller.step(
            action="Teleport",
            position=dict(x=x, y=y, z=z),
            rotation=dict(x=0, y=rotation, z=0),
            horizon=horizon)
        self.state = self.get_full_state(x, y, z, rotation, horizon)

    def get_full_state(self, x, y, z, rotation=0.0, horizon=0.0):
        return ThorAgentState(x, y, z, rotation, horizon)

    def reset(self, scene_name):
        if scene_name is None:
            scene_name = "FloorPlan28"

        if self.args.scene == 'procthor':
            self.scene_name = scene_name

            if self.args.episode_type == 'ExplorationTrainEpisode':
                with open(os.path.join(self.offline_data_dir, self.scene_name['idx'], self.grid_file_name), "r") as f:
                    self.grid = json.load(f)
            else:
                self.grid = None

            self.metadata = None
            self.last_event = self.controller.reset(scene_name)
        else:
            if scene_name != self.scene_name:
                self.scene_name = scene_name
                with open(os.path.join(self.offline_data_dir, self.scene_name, self.grid_file_name), "r") as f:
                    self.grid = json.load(f)
                with open(os.path.join(self.offline_data_dir, self.scene_name, self.metadata_file_name), "r") as f:
                    self.metadata = json.load(f)

            self.last_event = self.controller.reset(scene_name)

        self.env_collapse = False
        self.collision = False
        self.bounds = self.last_event.metadata["sceneBounds"]["size"]
        self.center = self.last_event.metadata["sceneBounds"]["center"]
        self.half_size = [b / 2.0 for b in self.bounds.values()]

    def floor_to_precision(self, x, precision):
        return math.floor(x / precision) * precision

    def step(self, action):
        self.collision = False
        old_pos = self.last_event.metadata['agent']["position"]
        if isinstance(action, dict):
            action = action["action"]
            if action == 'forward':
                self.move_relative(forward=0.15)
            elif action == 'up':
                self.move_relative(up=0.15)
            elif action == 'down':
                self.move_relative(up=-0.15)
            elif action == 'turn_left':
                if self.args.episode_type == 'ExplorationTrainEpisode':
                    self.turn(-15)
                else:
                    self.turn(-30)
            elif action == 'turn_right':
                if self.args.episode_type == 'ExplorationTrainEpisode':
                    self.turn(15)
                else:
                    self.turn(30)

            new_pos = self.last_event.metadata['agent']["position"]
            if action in ['forward', 'up', 'down'] and new_pos==old_pos:
                self.collision = True
        else:
            return self.controller.step(action=action)


    def move_relative(self, forward=0, up=0, right=0):
        rotation = self.last_event.metadata['agent']["rotation"]
        yaw = math.radians(rotation["y"])
        dx = forward * math.sin(yaw) + right * math.cos(yaw)
        dz = forward * math.cos(yaw) - right * math.sin(yaw)
        dy = up
        self.move(dx=dx, dy=dy, dz=dz)

        horizon = round(self.last_event.metadata['agent']['cameraHorizon'])
        if horizon != 30:
            self.set_horizon(30)

    def move(self, dx=0, dy=0, dz=0):
        pos = self.last_event.metadata['agent']["position"]
        rotation = self.last_event.metadata['agent']["rotation"]
        new_pos = {
            "x": pos["x"] + dx,
            "y": pos["y"] + dy,
            "z": pos["z"] + dz
        }
        try:
            self.last_event = self.controller.step(
                action="Teleport",
                position=new_pos,
                rotation=rotation,
                horizon=30
            )
            agent_meta = self.last_event.metadata['agent']
            self.state = self.get_full_state(
                agent_meta['position']['x'],
                agent_meta['position']['y'],
                agent_meta['position']['z'],
                agent_meta['rotation']['y'],
                agent_meta['cameraHorizon']
            )

        except TimeoutError:
            print(f"[WARN] Teleport timeout at pos: {new_pos}")
            self.controller.stop()
            del self.controller
            self.env_collapse = True
            while True:
                try:
                    self.controller = ExhaustiveBFSController(args=self.args)
                    break
                except TimeoutError:
                    time.sleep(0.1)

    def set_horizon(self, h=30):
        self.last_event = self.controller.step(
            action="Teleport",
            horizon=h
        )

    def turn(self, yaw_delta):
        pos = self.last_event.metadata['agent']["position"]
        rotation = self.last_event.metadata['agent']["rotation"]
        new_rot = dict(rotation)
        new_rot["y"] = (new_rot["y"] + yaw_delta) % 360

        try:
            self.last_event = self.controller.step(
                action="Teleport",
                position=pos,
                rotation=new_rot,
                horizon=30,
            )
            horizon = round(self.last_event.metadata['agent']['cameraHorizon'])
            if horizon != 30:
                self.set_horizon(30)

            agent_meta = self.last_event.metadata['agent']
            self.state = self.get_full_state(
                agent_meta['position']['x'],
                agent_meta['position']['y'],
                agent_meta['position']['z'],
                agent_meta['rotation']['y'],
                agent_meta['cameraHorizon']
            )

        except TimeoutError:
            print(f"[WARN] Teleport timeout at pos: {pos} and rot: {new_rot}")
            self.controller.stop()
            del self.controller
            self.env_collapse = True
            while True:
                try:
                    self.controller = ExhaustiveBFSController(args=self.args)
                    break
                except TimeoutError:
                    time.sleep(0.1)

    def find_id(self, parentType):
        parentId = []
        for key in self.metadata:
            if parentType in key:
                parentId.append(key)
        return parentId

    def all_objects(self):
        return self.metadata.keys()

    def get_state_from_str(self, x, z, rotation=0.0, horizon=0.0):
        y = self.y if self.y is not None else self.last_event.metadata['agent']['position']['y']
        return ThorAgentState(x, y, z, rotation, horizon)

    def shortest_path_to_target(self, source_state, target_state):
        return self._calculate_fallback_path_length(source_state, target_state)

    def _calculate_fallback_path_length(self, source_state, target_state):
        ''' 
        Fallback method to estimate path length when graph-based pathfinding fails. 
        '''

        distance_xz = math.sqrt(
            (source_state.x - target_state.x)**2 +
            (source_state.z - target_state.z)**2
        )

        distance_y = abs(source_state.y - target_state.y)

        total_distance = distance_xz + distance_y

        estimated_steps = math.ceil(total_distance / 0.15)
        
        return None, estimated_steps
