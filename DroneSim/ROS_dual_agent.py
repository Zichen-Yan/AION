import torch
from obj_utils.net_util import gpuify
from models.model_io import ModelInput
import numpy as np
from obj_utils.depth_transform import depth_layer_scan_api, get_depth_ROI
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

class ROSDualAgent:
    """ A navigation agent who learns with pretrained embeddings. """

    def __init__(self, args, gpu_id, nav_model, exp_model=None):
        self.action_space = args.action_space
        self.hidden_state_sz = args.hidden_state_sz
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        self.gpu_id = gpu_id
        self.args = args
        self.model = nav_model
        self.exp_model = exp_model

        self.hidden = None
        self.max_episode_length = args.max_episode_length

        torch.manual_seed(args.seed)
        if gpu_id >= 0:
            torch.cuda.manual_seed(args.seed)

        self.yolo_model = YOLO('yolov8x.pt')
        self.detect_obj_cnt = 0
        self.detect_obj = False

        self.save_dir = f"DroneSim/vis_results/{args.logdir}"
        os.makedirs(self.save_dir, exist_ok=True)

        # Define subfolders
        subfolders = ["rgb1", "depth_roi", "rgb3", "bbox", "depth_proj"]

        # Create dict to store full paths
        self.save_paths = {}
        for name in subfolders:
            folder_path = os.path.join(self.save_dir, name)
            os.makedirs(folder_path, exist_ok=True)
            self.save_paths[name] = folder_path

        self.fig1, self.ax1 = plt.subplots(figsize=(6.4, 4.8), dpi=100)
        self.fig5 = plt.figure(figsize=(3, 3), dpi=100)
        self.ax5 = self.fig5.add_subplot(111, projection="polar")

        self.success = False
        self.cnt = 0

    def action(self, obs, dual_mode):
        self.info = self.eval_at_state(obs)

        if dual_mode == 1:
            return self.exp_model.forward(self.model_input)
        elif dual_mode == 2:
            return self.model.forward(self.model_input)
        else:
            if not self.detect_obj:
                return self.exp_model.forward(self.model_input)
            else:
                return self.model.forward(self.model_input)

    def eval_at_state(self, obs):
        model_input = ModelInput()

        model_input.hidden = self.hidden
        model_input.target_class = obs["target_obj"]
        model_input.last_action = self.last_action

        # rgb================================
        model_input.raw_rgb = obs['rgb'].copy()
        rgb = torch.from_numpy(obs['rgb']).to(self.device) # 480 x 640 x 3
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0) # 1 x 480 x 640 x 3

        patches = self.split_into_patches(rgb)
        model_input.patch = torch.vstack(patches)

        model_input.rgb = self.model.resize(rgb)

        # depth & height======================
        height = torch.tensor([obs['height']]).to(self.device)

        depth = np.squeeze(obs['depth'])

        depth = np.where(np.isnan(depth) | np.isinf(depth), 10.0, depth)
        angles, dists = depth_layer_scan_api(depth=depth, height=height.cpu().numpy())
        model_input.angles = angles
        model_input.dists = dists
        model_input.depth = torch.from_numpy(dists).to(self.device)

        model_input.raw_depth = depth

        # obj detection box===================
        obj_attention = torch.zeros_like(rgb[0, :, :, :1]) # 480 x 640 x 1
        obj_area = 0
        img_area = rgb.shape[1] * rgb.shape[2]

        img_bgr = cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2BGR)
        results = self.yolo_model(img_bgr, device=0, imgsz=640, verbose=False)
        boxes = results[0].boxes

        if boxes is not None and boxes.cls.numel() > 0:
            labels = boxes.cls.cpu().numpy().astype(int)  # 类别索引
            conf = boxes.conf.cpu().numpy()  # 置信度
            xyxy = boxes.xyxy.cpu().numpy().astype(int)  # 坐标框

            class_id = obs["yolo_class_id"]
            mask = (labels == class_id) & (conf > 0.5)

            if mask.any():
                self.detect_obj_cnt += 1
                # index of the max-confidence box for this class
                best_idx = np.where(mask)[0][conf[mask].argmax()]

                if self.detect_obj_cnt >=2 and not self.detect_obj:
                    print("Found Target!")
                    self.detect_obj = True
                x1, y1, x2, y2 = xyxy[best_idx]

                # area
                obj_area += max(0, x2 - x1) * max(0, y2 - y1)

                # clamp and fill attention
                H, W, _ = obj_attention.shape
                x1 = max(0, min(x1, W - 1))
                x2 = max(0, min(x2, W - 1))
                y1 = max(0, min(y1, H - 1))
                y2 = max(0, min(y2, H - 1))
                if x2 > x1 and y2 > y1:
                    obj_attention[y1:y2, x1:x2, 0] = 1.0

        model_input.raw_obj_attention = obj_attention
        if obj_attention.ndim == 3:
            obj_attention = obj_attention.unsqueeze(0)
        obj_attention = self.model.resize(obj_attention) # 1x480x640x1 -> 1x1x224x224
        center_x, center_y, W, H = self.compute_box_center(obj_attention)

        # Compute area ratio
        box_area_ratio = torch.tensor([obj_area / img_area]).to(self.device)
        # if box_area_ratio.item()>0.15 and self.is_center_in_middle(center_x, center_y, threshold=0.8):
        #     print("Done by Metric!!!!")
        #     self.success = True

        stats = torch.cat([center_x, center_y, W, H, box_area_ratio, height])
        model_input.stats = stats

        if self.exp_model is not None:
            info = get_depth_ROI(model_input.raw_depth, fov=self.args.fov, camera_pitch_degrees=30,
                                  height=height.cpu().numpy())

            ROI_values = [info[k] for k in ['center_x', 'center_y', 'found_flag', 'mean_depth', 'y_horizon']]
            ROI_values = torch.tensor(ROI_values, dtype=torch.float32).to(self.device)
            tmp = ROI_values.float().flatten()
            center_x = tmp[0]
            center_y = tmp[1]
            found_flag = tmp[2]
            mean_depth = tmp[3]
            y_horizon = tmp[4]
            if found_flag:
                center_y -= y_horizon
                mean_depth = torch.clip(mean_depth, 0, 10) / 10.0
            else:
                center_y = -1
                center_x = -1
                mean_depth = -1
            tmp = torch.tensor([center_x, center_y, mean_depth], dtype=torch.float32).to(self.device)
            ROI = torch.cat([tmp, height])
            model_input.ROI = ROI
            # self.log(model_input, info, obs)

        self.cnt += 1
        self.model_input = model_input
        return info

    def log(self, model_input, info, obs):
        h, w = model_input.raw_depth.shape
        hy = int((info['y_horizon'] + 0.5) * h)
        # ---------- 准备ROI框 ----------
        if info["found_flag"]:
            cx = int((info["center_x"] + 0.5) * w)
            cy = int((info["center_y"] + 0.5) * h)
            W = int(info["W"] * w)
            H = int(info["H"] * h)

            pt1 = (cx - W // 2, cy - H // 2)
            pt2 = (cx + W // 2, cy + H // 2)
        else:
            pt1, pt2 = None, None

        # ---------- 深度图伪彩化 ----------
        depth_vis = np.clip(model_input.raw_depth, 0, 10.0)
        depth_vis = (depth_vis / depth_vis.max() * 255).astype(np.uint8)
        depth_vis = cv2.cvtColor(cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

        # ---------- 绘制ROI框 ----------
        if info["found_flag"]:
            cv2.rectangle(depth_vis, pt1, pt2, (255, 0, 0), 2)
            cv2.circle(depth_vis, (cx, cy), 4, (255, 0, 0), -1)

        # 1. Save RGB (raw)
        # ======================
        rgb = cv2.cvtColor(model_input.raw_rgb, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(self.save_paths["rgb1"], f"{self.cnt:03d}.png")
        cv2.imwrite(save_path, rgb)

        # ======================
        # 2. Save depth + horizon line (depth_roi)
        # ======================
        depth_img = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)
        # draw horizon
        cv2.line(depth_img, (0, hy), (depth_img.shape[1], hy), (0, 0, 0), 2)  # yellow BGR
        cv2.putText(depth_img, "Horizon Line", (10, hy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        save_path = os.path.join(self.save_paths["depth_roi"], f"{self.cnt:03d}.png")
        cv2.imwrite(save_path, depth_img)

        # ======================
        # 3. Save third-person rgb (rgb3)
        # ======================
        rgb3 = cv2.cvtColor(obs["rgb_3rd"], cv2.COLOR_RGB2BGR)
        save_path = os.path.join(self.save_paths["rgb3"], f"{self.cnt:03d}.png")
        cv2.imwrite(save_path, rgb3)

        # ======================
        # 4. Save bbox attention map
        # ======================
        att = (model_input.raw_obj_attention.cpu().numpy() * 255).astype(np.uint8)
        save_path = os.path.join(self.save_paths["bbox"], f"{self.cnt:03d}.png")
        cv2.imwrite(save_path, att)
        # --------------------
        gamma = 0.6
        r = np.power(model_input.dists, gamma)
        # angles → radians in [0, 2π)
        theta = np.deg2rad(model_input.angles)
        theta = (theta + 2 * np.pi) % (2 * np.pi)
        # keep only 0–π (0–180°)
        mask = theta <= np.pi
        theta = theta[mask]
        r = r[mask]
        c = model_input.dists[mask]

        # orientation and angular window
        self.ax5.cla()
        self.ax5.set_theta_zero_location("E")  # 0° at +x (use "N" if you prefer forward/up)
        self.ax5.set_theta_direction(1)  # CCW increasing
        self.ax5.set_thetamin(0)
        self.ax5.set_thetamax(180)
        self.ax5.set_thetagrids([0, 45, 90, 135, 180])  # <- requested ticks

        # points
        self.ax5.scatter(theta, r, c=c, cmap='viridis', s=18, vmin=0, vmax=1)

        # keep scan lines (rays to the origin)
        for t, rr in zip(theta, r):
            self.ax5.plot([t, t], [0, rr], alpha=0.25, linewidth=0.8)

        self.ax5.set_rlim(0, 1.0)
        self.ax5.grid(True)

        save_path = os.path.join(self.save_paths["depth_proj"], f"{self.cnt:03d}.png")
        self.fig5.savefig(save_path)

    def is_center_in_middle(self, center_x, center_y, threshold):
        x_min = 0.5 - threshold / 2
        x_max = 0.5 + threshold / 2
        y_min = 0.5 - threshold / 2
        y_max = 0.5 + threshold / 2

        in_center = (center_x >= x_min) & (center_x <= x_max) & \
                    (center_y >= y_min) & (center_y <= y_max)
        return in_center

    def split_into_patches(self, rgb, num_rows=3, num_cols=3):
        # Input: rgb shape (B, H, W, C)
        B, H, W, C = rgb.shape
        rgb = rgb.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)

        patch_H, patch_W = H // num_rows, W // num_cols
        patches = []
        for i in range(num_rows):
            for j in range(num_cols):
                patch = rgb[:, :, i * patch_H:(i + 1) * patch_H, j * patch_W:(j + 1) * patch_W]
                patch = patch.permute(0, 2, 3, 1)  # Convert back to (B, H, W, C)
                patches.append(patch)
        return patches  # list of (B, patch_H, patch_W, C)

    def reset_hidden(self):
        # LSTM
        self.hidden = (
            torch.zeros(1, self.hidden_state_sz).to(self.device),
            torch.zeros(1, self.hidden_state_sz).to(self.device),
        )
        action_num = self.model.args.action_space
        self.last_action = gpuify(torch.zeros((1, 1), dtype=torch.long), self.gpu_id)
        self.last_action_probs = torch.ones(1, action_num).to(self.device) / action_num

    def compute_box_center(self, mask):
        with torch.no_grad():
            B, _, H, W = mask.shape
            m = mask[:, 0]

            # Grid coordinates
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=mask.device),
                torch.arange(W, device=mask.device),
                indexing="ij"
            )

            m_flat = m.float()
            area = m_flat.sum(dim=(1, 2))
            eps = 1e-6 + (area == 0).float() * 1e-6

            # Center
            center_x = (m_flat * grid_x).sum(dim=(1, 2)) / (area + eps) / W
            center_y = (m_flat * grid_y).sum(dim=(1, 2)) / (area + eps) / H

            indices = torch.nonzero(m_flat[0], as_tuple=True)
            if len(indices[0]) == 0:
                width = torch.tensor([0]).to(mask.device)
                height = torch.tensor([0]).to(mask.device)
            else:
                ys, xs = indices
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                width = torch.tensor([x_max - x_min + 1]).to(mask.device) / W
                height = torch.tensor([y_max - y_min + 1]).to(mask.device) / H

        return center_x, center_y, width, height

