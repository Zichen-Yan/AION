#!/usr/bin/env python

# For sim running on Isaac Sim 5.1 (ROS2)

import carb
from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
import omni.usd

from omni.isaac.core.world import World
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.numpy.rotations as rot_utils
from omni.isaac.sensor import Camera
import omni.graph.core as og
from omni.isaac.core.utils.stage import get_current_stage

import usdrt.Sdf
# Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# Auxiliary modules
from scipy.spatial.transform import Rotation
import numpy as np
import subprocess, math, time
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf, UsdLux, UsdShade, PhysxSchema, PhysicsSchemaTools

import omni.kit.viewport
import omni.kit.commands
from omni.physx.scripts import utils as physx_utils
from std_msgs.msg import Bool
from omni.physx import get_physx_scene_query_interface

CAMERA_PRIM_PATH = "/World/quadrotor/body/camera_fpv"

import isaacsim.core.utils.extensions as extensions
import rclpy
rclpy.init()

import os, json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # ObjectNav/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class PegasusApp:
    def __init__(self, cfg, obj_name=None):
        extensions.enable_extension("isaacsim.ros2.bridge")
        simulation_app.update()
        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        env_path = Path(cfg["path"]).expanduser().resolve()
        if not env_path.exists():
            raise FileNotFoundError(f"Environment file not found: {env_path}")
        self.pg.load_environment(env_path.as_uri())

        simulation_app.update()

        stage = get_current_stage()
        env_prim = stage.GetPrimAtPath("/World/layout")

        xform = UsdGeom.Xformable(env_prim)
        xform.AddScaleOp().Set((1.5, 1.5, 1.5))  # shrink scene by 50%
        simulation_app.update()

        # Configure and spawn the drone
        config_multirotor = MultirotorConfig()
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": "none_iris",
        })
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]
        Multirotor(
            "/World/quadrotor", ROBOTS['Iris'], 0, cfg["start_pose"]["position"],
            Rotation.from_euler("XYZ", cfg["start_pose"]["rotation"], degrees=True).as_quat(),
            config=config_multirotor
        )

        if isinstance(obj_name, list):
            for obj in obj_name:
                self._add_marker(cfg["obj"][obj], obj)
        else:
            self._add_marker(cfg["obj"][obj_name], obj_name)

        self._add_camera(
            prim_path="/World/quadrotor/body/camera_fpv",
            pos=np.array([0.2, 0.0, 0.0]),
            rot=np.array([0.0, 30.0, 0.0]),
            graph_path="/ROS_FPV_Graph",
            graph_prefix="fpv"
        )

        self._add_camera(
            prim_path="/World/quadrotor/body/camera_3rd",
            pos=np.array([-0.2, 0.0, 1.0]),
            rot=np.array([0.0, 40.0, 0.0]),
            graph_path="/ROS_3RD_Graph",
            graph_prefix="third"
        )

        self.create_collision_node()
        simulation_app.update()

    def _add_marker(self, cfg, obj_name):
        if "usd_path" in cfg:
            usd_url = cfg["usd_path"]
            prim_path = "/World/"+obj_name
            scale = cfg["scale"] if "scale" in cfg else np.array([1.0, 1.0, 1.0])
            prim_utils.create_prim(
                prim_path=prim_path, prim_type="Xform", usd_path=usd_url,
                position=cfg["position"], orientation=Rotation.from_euler("XYZ", cfg["rotation"], degrees=True).as_quat(),
                scale=scale
            )
            # Get prim object from path
            stage = get_current_stage()
            prim = stage.GetPrimAtPath(prim_path)
            physx_utils.setRigidBody(prim, "convexHull", True)

    def _add_camera(
        self,
        prim_path: str,
        pos: np.ndarray,
        rot: np.ndarray,
        graph_path: str,
        graph_prefix: str,
        CAMERA_RESOLUTION=(640, 480)
    ):
        desired_fov_degrees = 90.0
        horizontal_aperture = 24.0
        focal_length = horizontal_aperture / (2 * math.tan(math.radians(desired_fov_degrees) / 2.0))
        vertical_aperture = horizontal_aperture * (CAMERA_RESOLUTION[1] / CAMERA_RESOLUTION[0])

        # Create the camera prim with all its properties (intrinsics) defined at creation.
        # This is a more robust way to set these values than using the high-level Camera class alone.
        prim_utils.create_prim(
            prim_path=prim_path,
            prim_type="Camera",
            attributes={
                "focalLength": focal_length,
                "horizontalAperture": horizontal_aperture,
                "verticalAperture": vertical_aperture,
                "clippingRange": (0.1, 1000.0),  # Near and far clipping planes
            },
        )

        # Now, apply the high-level Isaac Sim Camera API to this prim for easy control
        self.camera = Camera(
            prim_path=prim_path,
            resolution=CAMERA_RESOLUTION,
            frequency=30
        )
        self.camera.initialize()

        cam_ori = rot_utils.euler_angles_to_quats(rot, degrees=True)
        self.camera.set_local_pose(translation=pos, orientation=cam_ori)
        self._build_camera_graph(prim_path, graph_path, graph_prefix)
        og.Controller.evaluate_sync(graph_path)

    def _build_camera_graph(self, prim_path, graph_path, graph_prefix):
        """
        Creates the OmniGraph for streaming the FPV camera's view via ROS2.
        This remains unchanged, as it correctly uses the camera prim defined above.
        """
        keys = og.Controller.Keys
        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "push"},
            {
                keys.CREATE_NODES: [
                    ("OnTick", "omni.graph.action.OnTick"),
                    ("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
                    ("createRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ("setCamera", "isaacsim.core.nodes.IsaacSetCameraOnRenderProduct"),
                    ("cameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ("cameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ],
                keys.CONNECT: [
                    # ---- init once ----
                    ("OnTick.outputs:tick", "RunOnce.inputs:execIn"),
                    ("RunOnce.outputs:step", "createRenderProduct.inputs:execIn"),
                    ("createRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),

                    # ---- publish every tick ----
                    ("OnTick.outputs:tick", "cameraHelperRgb.inputs:execIn"),
                    ("OnTick.outputs:tick", "cameraHelperDepth.inputs:execIn"),

                    # ---- renderProduct wiring ----
                    ("createRenderProduct.outputs:renderProductPath", "setCamera.inputs:renderProductPath"),
                    ("createRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
                    ("createRenderProduct.outputs:renderProductPath", "cameraHelperDepth.inputs:renderProductPath"),
                ],
                keys.SET_VALUES: [
                    ("createRenderProduct.inputs:width", 640),
                    ("createRenderProduct.inputs:height", 480),
                    ("createRenderProduct.inputs:cameraPrim", usdrt.Sdf.Path(prim_path)),
                    ("setCamera.inputs:cameraPrim", usdrt.Sdf.Path(prim_path)),

                    ("cameraHelperRgb.inputs:frameId", f"{graph_prefix}_camera"),
                    ("cameraHelperRgb.inputs:topicName", f"{graph_prefix}_rgb"),
                    ("cameraHelperRgb.inputs:type", "rgb"),

                    ("cameraHelperDepth.inputs:frameId", f"{graph_prefix}_camera"),
                    ("cameraHelperDepth.inputs:topicName", f"{graph_prefix}_depth"),
                    ("cameraHelperDepth.inputs:type", "depth"),
                ],
            },
        )

    def create_collision_node(self):
        node = rclpy.create_node("drone_collision_pub")
        stage = get_current_stage()
        self.robot_root = "/World/quadrotor"
        self.capsule_path = "/World/quadrotor/body/collision_capsule"
        collider = UsdGeom.Capsule.Define(stage, self.capsule_path)

        collider.CreateRadiusAttr(0.3)
        collider.CreateHeightAttr(0.15)

        xform = UsdGeom.XformCommonAPI(collider)
        xform.SetTranslate(Gf.Vec3d(0, 0, 0))
        xform.SetRotate(Gf.Vec3f(0, 90, 0))  # align horizontally

        # PhysxSchema.PhysxCollisionAPI.Apply(collider.GetPrim())
        UsdGeom.Imageable(collider.GetPrim()).MakeInvisible()

        self.node = node
        self.collision_pub = self.node.create_publisher(Bool, "/drone/collision", 10)

        self.scene_query = get_physx_scene_query_interface()
        encoded = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(self.capsule_path))
        self._shape_path0 = encoded[0]
        self._shape_path1 = encoded[1]
        self._collisions = set()

    def _report_overlap(self, hit):
        c = hit.collision

        # 1) Ignore the query capsule itself
        if c == self.capsule_path or c.startswith(self.capsule_path + "/"):
            return True

        # 2) Ignore any collider that belongs to the drone (self-collision)
        if c == self.robot_root or c.startswith(self.robot_root + "/"):
            return True

        # 3) Everything else is a "real" collision
        self._collisions.add(c)
        return True

    def update_collision(self):
        # Clear previous collisions
        self._collisions.clear()

        num_hits = self.scene_query.overlap_shape(
            self._shape_path0,
            self._shape_path1,
            self._report_overlap,
            False,  # anyHit=False: collect all; True: first hit then stop
        )

        collided = (num_hits > 0) and (len(self._collisions) > 0)

        self.collision_pub.publish(Bool(data=collided))

    def run(self):
        self.timeline.play()
        while simulation_app.is_running():
            self.world.step(render=True)
            self.update_collision()   # <-- HERE
            rclpy.spin_once(self.node, timeout_sec=0.0)
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():
    env_idx = 2
    env_name = [
                "school_chemistry",
                "Beechwood_0_int",
                "Ihlen_1_int",
               ][env_idx]

    # UnseenObj = ["Sofa", "Plant", "Laptop", "Microwave"]
    obj_name = "Sofa"

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Scene.json")
    with open(config_path, "r") as f:
        data = json.load(f)
    cfg = data[env_name]

    pg_app = PegasusApp(cfg, obj_name)
    pg_app.run()

if __name__ == "__main__":
    main()
