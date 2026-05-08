#!/usr/bin/env python
import os
import math
import json
import random
import argparse
from copy import deepcopy

import torch
import numpy as np
import torch.nn.functional as F
import rclpy
rclpy.init()
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.duration import Duration

from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from cv_bridge import CvBridge

from DroneSim.ROS_dual_agent import ROSDualAgent
from models import AIONe, AIONg

from obj_utils.action_util import get_actions
import time
from datetime import datetime

def enu_to_ned(x_enu: float, y_enu: float, z_enu: float):
    """
    ENU(x East, y North, z Up)  →  NED(x North, y East, z Down)
    """
    x_ned = y_enu
    y_ned = x_enu
    z_ned = -z_enu
    return x_ned, y_ned, z_ned

def forward_speed_to_enu(v_forward, yaw_rad):
    vx_enu = v_forward * math.cos(yaw_rad)
    vy_enu = v_forward * math.sin(yaw_rad)
    return vy_enu, vx_enu

def now_us(node: Node) -> int:
    return node.get_clock().now().nanoseconds // 1000

class IsaacSimEnv(Node):
    """
    ROS2 + PX4 (via microRTPS bridge) compatible environment.
    Reimplements reset()/step() for RL policy evaluation.
    """

    def __init__(self, objcfg):
        super().__init__("isaac_sim_nav_node")

        qos_sensor = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_ctrl = 10

        # --- parameters ---
        self.bridge = CvBridge()
        self.depth_max_dis = 10.0

        self.rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.depth_image = np.zeros((480, 640, 1), dtype=np.float32)
        self.rgb_3rd = np.zeros((480, 640, 3), dtype=np.uint8)

        self.target_obj_class = objcfg["clip_class"]
        self.yolo_class_id = objcfg["yolo_class_id"]

        # --- PX4 publishers ---
        self.pub_offb = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", qos_ctrl)
        self.pub_ts = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos_ctrl)
        self.pub_cmd = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos_ctrl)

        # ---------- Subscribers ----------
        self.sub_lpos = self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position", self._on_lpos, qos_sensor
        )
        self.sub_status = self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status_v1", self._on_status, qos_sensor
        )

        self.create_subscription(Image, "/fpv_rgb", self._on_rgb, qos_sensor)
        self.create_subscription(Image, "/fpv_depth", self._on_depth, qos_sensor)
        self.create_subscription(Image, "/third_rgb", self._on_rgb_3rd, qos_sensor)
        self.create_subscription(Bool, "/drone/collision", self._col, qos_sensor)

        self.keepalive_period = 1.0 / 30.0

        self.armed = False
        self.nav_state = 0
        self.x_ned = None
        self.y_ned = None
        self.z_ned = None  # VehicleLocalPosition.z (down is positive)
        self.collision = False
        self._heading_rad = math.nan

        self._last_sp = None
        self._use_velocity_mode = False

        self._timer = self.create_timer(self.keepalive_period, self._on_timer)
        self.get_logger().info("ROS2 IsaacSimEnv initialized with PX4 bridge topics.")

    # ---------- Callbacks ----------
    def _on_lpos(self, msg: VehicleLocalPosition):
        self.x_ned = float(msg.x)
        self.y_ned = float(msg.y)
        self.z_ned = float(msg.z)  # down is positive
        self._heading_rad = float(getattr(msg, "heading", math.nan))

    def _on_status(self, msg: VehicleStatus):
        self.armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self.nav_state = msg.nav_state

    def _decode_rgb(self, msg: Image, attr: str):
        try:
            setattr(self, attr, self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8"))
        except Exception:
            print(f"RGB decode error ({attr})!")

    def _on_rgb(self, msg: Image):
        self._decode_rgb(msg, "rgb_image")

    def _on_rgb_3rd(self, msg: Image):
        self._decode_rgb(msg, "rgb_3rd")

    def _on_depth(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            self.depth_image = np.nan_to_num(img, nan=self.depth_max_dis)[:, :, None]
        except Exception:
            print("Depth decode error!")

    def _col(self, msg: Bool):
        self.collision = msg.data

    def _on_timer(self):
        # OffboardControlMode: position or velocity
        offb = OffboardControlMode()
        offb.timestamp = now_us(self)
        offb.position = not self._use_velocity_mode
        offb.velocity = self._use_velocity_mode
        offb.acceleration = False
        offb.attitude = False
        offb.body_rate = False
        offb.thrust_and_torque = False
        offb.direct_actuator = False
        self.pub_offb.publish(offb)

        if self._last_sp is not None:
            self._last_sp.timestamp = offb.timestamp
            self.pub_ts.publish(self._last_sp)

    # ---------- PX4 Commands ----------
    def _send_vehicle_command(self, command: int, **params):
        """
        Send VehicleCommand
          - ARM/DISARM: command=VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1/0
          - SET_MODE:   command=VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1(custom), param2=6(OFFBOARD)
        """
        cmd = VehicleCommand()
        cmd.timestamp = now_us(self)
        cmd.command = int(command)
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True

        for i in range(1, 8):
            key = f"param{i}"
            if key in params:
                setattr(cmd, key, float(params[key]))

        self.pub_cmd.publish(cmd)

    def enter_offboard_and_arm(self):
        t_end = self.get_clock().now() + Duration(seconds=0.05)
        while self.get_clock().now() < t_end:
            offb = OffboardControlMode()
            offb.timestamp = now_us(self)
            offb.position = True
            offb.velocity = False
            self.pub_offb.publish(offb)

            sp = TrajectorySetpoint()
            sp.timestamp = offb.timestamp
            sp.position[:] = [0.0] * 3
            sp.velocity[:] = [math.nan] * 3
            sp.acceleration[:] = [math.nan] * 3
            if hasattr(sp, "jerk"): sp.jerk[:] = [math.nan] * 3
            sp.yaw = math.nan
            sp.yawspeed = 0.0
            self.pub_ts.publish(sp)
            rclpy.spin_once(self, timeout_sec=0.0)

        self._send_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self._send_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0, param2=0.0)

    def arm(self):
        self._use_velocity_mode = False
        self._last_sp = None
        while not self.armed or self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.enter_offboard_and_arm()

        self.get_logger().info("Offboard + arm complete.")

    def _set_position_target(self, x_ned, y_ned, z_ned, yaw = math.nan):
        self._use_velocity_mode = False
        sp = TrajectorySetpoint()
        sp.timestamp = now_us(self)
        sp.position[:] = [x_ned, y_ned, z_ned]
        sp.velocity[:] = [math.nan, math.nan, math.nan]
        sp.acceleration[:] = [math.nan, math.nan, math.nan]
        if hasattr(sp, "jerk"):
            sp.jerk[:] = [math.nan, math.nan, math.nan]
        sp.yaw = yaw
        sp.yawspeed = 0.0
        self._last_sp = sp

    def take_off(self,
              position_enu = None,
              yaw: float = math.nan):
        if position_enu is None:
            x_ned = self.x_ned
            y_ned = self.y_ned
            z_ned = self.z_ned - 0.5
        else:
            x_enu, y_enu, z_enu = map(float, position_enu)
            x_ned, y_ned, z_ned = enu_to_ned(x_enu, y_enu, z_enu)

        self._set_position_target(x_ned, y_ned, z_ned, yaw=yaw)
        return z_ned

    def turn(self, rad: float):
        self.vel_cnt((0, 0, 0), rad)

    def pos_cnt(self,
                position_ned: tuple,
                relative: bool = False,
                yaw: float = math.nan):

        x_ned, y_ned, z_ned = map(float, position_ned)
        if relative:
            x_ned += self.x_ned
            y_ned += self.y_ned
            z_ned += self.z_ned

        self._set_position_target(x_ned, y_ned, z_ned, yaw=yaw)

    def vel_cnt(self,
                velocity_enu: tuple,
                yaw_rate_rad: float = 0.0):
        vx_enu, vy_enu, vz_enu = map(float, velocity_enu)

        self._use_velocity_mode = True
        sp = TrajectorySetpoint()
        sp.timestamp = now_us(self)
        sp.position[:] = [math.nan, math.nan, math.nan]
        vx_ned, vy_ned, vz_ned = enu_to_ned(vx_enu, vy_enu, vz_enu)
        sp.velocity[:] = [vx_ned, vy_ned, vz_ned]
        sp.acceleration[:] = [math.nan, math.nan, math.nan]
        if hasattr(sp, "jerk"):
            sp.jerk[:] = [math.nan, math.nan, math.nan]
        sp.yaw = math.nan
        sp.yawspeed = -yaw_rate_rad
        self._last_sp = sp

    def _get_obs(self) -> dict:
        return {
            "rgb": self.rgb_image,
            "depth": self.depth_image,
            "rgb_3rd": self.rgb_3rd,
            "height": self.start_height - self.z_ned,
            "target_obj": self.target_obj_class,
            "yolo_class_id": self.yolo_class_id,
            "pos": [self.x_ned, self.y_ned, self.z_ned],
            "heading_rad": self._heading_rad,
            "collision": self.collision,
        }

    def reset(self):
        self.arm()
        self.start_height = self.z_ned
        z_target = self.take_off()
        while abs(self.z_ned - z_target) > 0.05:
            rclpy.spin_once(self, timeout_sec=0.05)
        self.vel_cnt((0.0, 0.0, 0.0))
        return self._get_obs()

    def step(self, action_str):
        ang_vel, lin_vel, done = self.normalize_action(action_str)
        if done:
            self.vel_cnt((0.0, 0.0, 0.0))
        else:
            self.vel_cnt(lin_vel, ang_vel)
        self.spin_sleep(0.5)
        return self._get_obs(), done

    def spin_sleep(self, duration_sec):
        start = time.time()
        while (time.time() - start) < duration_sec and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

    # ---------- Helper ----------
    def normalize_action(self, action):
        done = False
        vx = vy = vz = yaw_rate = 0.0
        if action == 'up':
            vz = 0.15
        elif action == 'down':
            vz = -0.15
        elif action == 'turn_left':
            yaw_rate = np.deg2rad(+25.0)
        elif action == 'turn_right':
            yaw_rate = np.deg2rad(-25.0)
        elif action == 'forward':
            vx = +0.8
            vx, vy = forward_speed_to_enu(vx, self._heading_rad)
        elif action == 'Done':
            done = True
        return yaw_rate, (vx, vy, vz), done


def isaac_val(args, max_count, objcfg, nav_model, exp_model, dual_mode):
    args.max_episode_length = max_count
    actions = get_actions(args)

    player = ROSDualAgent(args, args.gpu_id, nav_model, exp_model)
    env = IsaacSimEnv(objcfg)

    try:
        player.reset_hidden()
        obs = env.reset()
        last_t = time.time()
        start_time = last_t
        success_flag = False
        step_log = {}

        for i in range(args.max_episode_length):
            print(f"Step {i + 1}/{args.max_episode_length}")
            out = player.action(obs, dual_mode)

            player.hidden = out.hidden
            prob = F.softmax(out.logit, dim=1)
            action_idx = torch.multinomial(prob, num_samples=1)
            action_str = actions[action_idx.item()]

            if dual_mode == 0:
                mode = "Exploration" if not player.detect_obj else "Goal-Reaching"
            elif dual_mode == 1:
                mode = "Exploration"
            else:
                mode = "Goal-Reaching"

            step_log[i] = {
                "action": action_str,
                "Mode": mode,
                "pos": obs['pos'],
                "heading_rad": obs['heading_rad'],
                "collision": obs['collision']
            }

            if action_str == "Done" and dual_mode == 2:
                idx = out.logit[:, :-1].argmax(dim=1, keepdim=True)
                action_idx = idx
                action_str = actions[action_idx.item()]

            if player.success and dual_mode==0:
                action_str = "Done"

            player.last_action = action_idx
            print(f"  - Action: {action_str}")
            if mode != "Exploration":
                player.last_action_probs = prob.detach()

            obs, done = env.step(action_str)
            now = time.time()
            dt = now - last_t
            last_t = now
            freq = 1.0 / dt if dt > 0 else 0
            print(f"[Step {i}] dt={dt:.4f}s  freq={freq:.2f} Hz")

            if done:
                success_flag = True
                print("Success!!!!!!!!!")
                break

        save_file = os.path.join(player.save_dir, "step_log.json")
        with open(save_file, "w") as f:
            json.dump(step_log, f, indent=4)

        print(f"[INFO] Saved step log → {save_file}")
        total_time = time.time() - start_time
        time_file = os.path.join(player.save_dir, "episode_info.json")

        episode_info = {
            "total_time_sec": total_time,
            "success": success_flag,
            "object": obs["target_obj"]
        }
        with open(time_file, "w") as f:
            json.dump(episode_info, f, indent=4)

    finally:
        print("Shutting down ROS2 node.")
        env.destroy_node()
        rclpy.shutdown()

def main(args, nav_model, exp_model, dual_mode, env_name, env_step, obj_name):
    # Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.logdir = f"{env_name}_{obj_name}_{timestamp}"

    with open("DroneSim/Scene.json", "r") as f:
        data = json.load(f)
    cfg = data[env_name]

    isaac_val(
        args=args,
        max_count=env_step,
        objcfg=cfg["obj"][obj_name],
        nav_model=nav_model,
        exp_model=exp_model,
        dual_mode=dual_mode
    )


if __name__ == "__main__":
    with open("ckpt/args.json", "r") as f:
        args_dict = json.load(f)
    args = argparse.Namespace(**args_dict)

    exp_model_path = "ckpt/AION-e.dat"
    nav_model_path = "ckpt/AION-g-18-4.dat"

    args.gpu_id = 0
    nav_model = AIONg(deepcopy(args))
    saved_state = torch.load(nav_model_path, map_location=lambda storage, loc: storage)
    saved_state.pop("goal_text_emb", None)
    nav_model.load_state_dict(saved_state, strict=False)
    nav_model = nav_model.to(f"cuda:{args.gpu_id}")

    args.action_space = 5
    exp_model = AIONe(args)
    args.action_space = 6
    saved_state = torch.load(exp_model_path, map_location=lambda storage, loc: storage)
    exp_model.load_state_dict(saved_state, strict=False)
    exp_model = exp_model.to(f"cuda:{args.gpu_id}")

    dual_mode = 0  # 0: both, 1: explore only, 2: goal-reach only

    env_idx = 2
    env_name = [
                "school_chemistry",
                "Beechwood_0_int",
                "Ihlen_1_int",
               ][env_idx]
    if dual_mode == 0:
        env_step = 300
    else:
        env_step = 150

    # UnseenObj = ["Sofa", "Plant", "Laptop", "Microwave"]
    obj_name = "Sofa"
    main(args, nav_model, exp_model, dual_mode, env_name, env_step, obj_name)



