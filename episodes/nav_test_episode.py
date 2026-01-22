""" Contains the Episodes for Navigation. """
from datasets.environment import Environment
from .nav_train_episode import NavTrainEpisode
import pickle
from datasets.data import num_to_name
import json
from datasets.controller_3D import ExhaustiveBFSController
import time

c2p_prob = json.load(open("./data/c2p_prob.json"))
rooms = ['Kitchen', 'Living_Room', 'Bedroom', 'Bathroom']
metadata_dir = "./data/thor_v1_offline_data/"

class NavTestEpisode(NavTrainEpisode):
    """ Episode for Navigation. """

    def __init__(self, args, gpu_id, strict_done=False):
        super(NavTestEpisode, self).__init__(args, gpu_id, strict_done)
        self.file = None
        self.all_data = None
        self.all_data_enumerator = 0
        self.target_parents = None
        self.room = None

    def _new_episode(self, args, episode):
        """ New navigation episode. """
        scene = episode["scene"]
        if "physics" in scene:
            scene = scene[:-8]

        if self._env is None:
            self._env = Environment(args=args)
        height = 0.2
        while True:
            try:
                self._env.reset(scene)
                self._env.controller.teleport_agent_to(episode["state"].x, height, episode["state"].z,
                                                       episode["state"].rotation, episode["state"].horizon)
                self._env.start_height = 0.2
                self._env.controller.move_relative(up=0.2)
                break
            except TimeoutError:
                print("[WARN] Reset timeout at" + scene)
                print("Error Pos:", episode["state"].x, height, episode["state"].z)
                height = episode["state"].y
                self._env.controller.controller.stop()
                del self._env.controller.controller
                while True:
                    try:
                        self._env.controller.controller = ExhaustiveBFSController(args=args)
                        break
                    except TimeoutError:
                        time.sleep(0.1)

        self.task_data = episode["task_data"]
        self.target_object = episode["goal_object_type"]
        self.room = episode["room"]
        try:
            self.target_parents = c2p_prob[self.room][self.target_object]
        except KeyError:
            self.target_parents = None

    def new_episode(
        self,
        args,
        scenes,
        possible_targets=None,
        targets=None,
        room = None,
        keep_obj=False,
    ):
        self.pre_dis = 0
        self.done_count = 0
        self.move_steps = 0
        self.prev_frame = None
        self.current_frame = None
        self.current_objs = None
        self.room = None

        if self.file is None:
            sample_scene = scenes[0]
            scene_num = sample_scene[len("FloorPlan") :]
            scene_num = int(scene_num)
            scene_type = num_to_name(scene_num)
            task_type = args.test_or_val
            self.file = open("test_val_split/" + scene_type + "_" + task_type + ".pkl", "rb")
            self.all_data = pickle.load(self.file)
            self.file.close()
            self.all_data_enumerator = 0

        episode = self.all_data[self.all_data_enumerator]
        while True:
            if episode["goal_object_type"] == "CD": # CD is missing in the scene
                episode["goal_object_type"] = 'Book'

            if episode["goal_object_type"] not in targets:
                self.all_data_enumerator += 1
                episode = self.all_data[self.all_data_enumerator]
            else:
                self.all_data_enumerator += 1
                break

        self._new_episode(args, episode)

    def object_is_visible(self, objId):
        objects = self.environment.last_event.metadata["objects"]
        visible_objects = [o["objectId"].split('|')[0] for o in objects if o["visible"]]
        return objId.split('|')[0] in visible_objects
