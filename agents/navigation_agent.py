import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from obj_utils.net_util import gpuify
from obj_utils.depth_transform import depth_layer_scan_api, get_depth_ROI

from models.model_io import ModelInput
from models.clip_utils import get_transform

from .agent import ThorAgent
from episodes.exp_train_episode import ExplorationTrainEpisode
import numpy as np
import h5py
from runners.visualization_util import ResizeTo224


class NavigationAgent(ThorAgent):
    """ A navigation agent who learns with pretrained embeddings. """

    def __init__(self, shared_model, args, rank, gpu_id):
        self.rank = rank
        max_episode_length = args.max_episode_length
        self.action_space = args.action_space
        from obj_utils.class_finder import episode_class

        episode_constructor = episode_class(args.episode_type)
        episode = episode_constructor(args, gpu_id, args.strict_done)
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        super(NavigationAgent, self).__init__(shared_model, args, rank, episode, max_episode_length, gpu_id)

        self.hidden_state_sz = args.hidden_state_sz
        self.img_list = []
        self.model_name = args.model

        if isinstance(self.episode, ExplorationTrainEpisode):
            self.episode.set_model(self.model)

        self.resize = get_transform(name="resize", size=224)

        if self.model_name in ['ZSON', 'BaseModel', 'GCN', 'MJO']:
            self.prep = ResizeTo224(size=(224, 224), normalize=True)
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(self.device)
            self.resnet18 = nn.Sequential(*list(resnet.children())[:-2])  # -> (N, 512, 7, 7)
            for p in self.resnet18.parameters():
                p.requires_grad = False

            self.obj_list = []
            with open("./data/gcn/objects.txt") as f:
                objects = f.readlines()
                for o in objects:
                    o = o.strip()
                    self.obj_list.append(o)
            n=len(self.obj_list)
            all_glove = torch.zeros(n, 300)
            glove = h5py.File("./data/gcn/glove.6B.300d.hdf5", "r")
            for i in range(n):
                all_glove[i, :] = torch.Tensor(glove[self.obj_list[i]][:])
            self.all_glove = nn.Parameter(all_glove)
            self.all_glove.requires_grad = False

    def _prepare_input_for_AION(self, model_options):
        model_input = ModelInput()
        # rgb================================
        if self.episode.current_frame is None:
            rgb = self.state()
        else:
            rgb = self.episode.current_frame

        model_input.raw_rgb = rgb.cpu().numpy() # 300x300x3
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0) # 1 x 300 x 300 x 3

        patches = self.split_into_patches(rgb)
        model_input.patch = torch.vstack(patches)
        model_input.rgb = self.model.resize(rgb)

        # depth=====================================
        height = torch.tensor([self.environment.last_event.metadata['agent']['position']['y']
                               - self.environment.start_height]).to(rgb.device)

        depth =self.environment.last_event.depth_frame # 300x300

        if self.args.add_depth:
            depth = np.where(np.isnan(depth) | np.isinf(depth), 10.0, depth)
            angles, dists = depth_layer_scan_api(depth=depth, height=height.cpu().numpy())
            model_input.depth = torch.from_numpy(dists).to(self.device)
            model_input.angles = torch.from_numpy(angles).to(self.device)

        model_input.raw_depth = depth
        # obj=======================================
        obj_attention = torch.zeros_like(rgb[0,:,:,:1]) # 300x300x1
        event = self.environment.last_event

        obj_area = 0
        img_area = rgb.shape[1] * rgb.shape[2]
        frame = event.frame.copy()
        obj_cnt = 0
        for obj in event.metadata['objects']:
            name = obj['objectId'].split('|')[0]
            if name == self.episode.target_object and obj['visible']:
                try:
                    (x1, y1, x2, y2) = event.instance_detections2D[obj['objectId']]
                except KeyError:
                    continue
                obj_cnt+=1
                obj_area += max(0, x2 - x1) * max(0, y2 - y1)
                # Draw bounding box on frame
                frame[y1, x1:x2] = (255, 0, 0)
                frame[y2, x1:x2] = (255, 0, 0)
                frame[y1:y2, x1] = (255, 0, 0)
                frame[y1:y2, x2] = (255, 0, 0)

                # Create attention mask
                H, W, _ = obj_attention.shape
                x1 = max(0, min(x1, W - 1))
                x2 = max(0, min(x2, W - 1))
                y1 = max(0, min(y1, H - 1))
                y2 = max(0, min(y2, H - 1))
                obj_attention[y1:y2, x1:x2, 0] = 1.0

        model_input.raw_obj_attention = obj_attention
        # Compute area ratio
        self.episode.box_area_ratio = torch.tensor([obj_area / img_area]).to(self.device)
        if obj_attention.ndim == 3:
            obj_attention = obj_attention.unsqueeze(0)
        obj_attention = self.model.resize(obj_attention) # 1x300x300x1 -> 1x1x224x224
        model_input.obj_attention = obj_attention
        center_x, center_y, W, H = self.compute_box_center(obj_attention)

        stats = torch.cat([center_x, center_y, W, H, self.episode.box_area_ratio, height])
        self.episode.stats = stats
        model_input.stats = stats

        model_input.hidden = self.hidden
        model_input.target_class = self.episode.target_object
        model_input.last_action = self.last_action
        model_input.timestep = model_options.timestep

        if self.args.episode_type == 'ExplorationTrainEpisode':
            info = get_depth_ROI(model_input.raw_depth, camera_pitch_degrees=30, height=height.cpu().numpy())

            if self.episode.next_ROI is None:
                ROI_values = [info[k] for k in ['center_x', 'center_y', 'found_flag', 'mean_depth', 'y_horizon']]
                ROI_values = torch.tensor(ROI_values, dtype=torch.float32).to(self.device)
            else:
                ROI_values = self.episode.next_ROI
            self.episode.ROI = ROI_values

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

        self.model_input = model_input
        return self.model.forward(model_input)

    def _prepare_input_for_baseline(self, model_options):
        event = self.environment.last_event
        model_input = ModelInput()

        objbb = {}
        for obj in event.metadata['objects']:
            if obj['visible']:
                name = obj['objectId'].split('|')[0]
                box = event.instance_detections2D.get(obj['objectId'])

                if box is not None:
                    if name in objbb:
                        objbb[name].extend(list(box))
                    else:
                        objbb[name] = list(box)

        model_input.objbb = objbb

        target_name = self.episode.target_object
        target_embedding = self.all_glove[self.obj_list.index(target_name)]
        model_input.target_class_embedding = gpuify(target_embedding, self.gpu_id)

        model_input.hidden = self.hidden
        model_input.action_probs = self.last_action_probs

        # obj=======================================
        if self.episode.current_frame is None:
            rgb = self.state()
        else:
            rgb = self.episode.current_frame

        x = self.prep.apply(rgb)  # 300x300x3 -> (1,3,224,224), float, normalized
        with torch.no_grad():
            feat = self.resnet18(x)
        model_input.state = feat

        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)

        obj_attention = torch.zeros_like(rgb[0,:,:,:1])
        obj_area = 0
        img_area = rgb.shape[1] * rgb.shape[2]

        frame = event.frame.copy()
        obj_cnt = 0
        for obj in event.metadata['objects']:
            name = obj['objectId'].split('|')[0]
            if name == self.episode.target_object and obj['visible']:
                try:
                    (x1, y1, x2, y2) = event.instance_detections2D[obj['objectId']]
                except KeyError:
                    continue
                obj_cnt+=1
                obj_area += max(0, x2 - x1) * max(0, y2 - y1)
                # Draw bounding box on frame
                frame[y1, x1:x2] = (255, 0, 0)
                frame[y2, x1:x2] = (255, 0, 0)
                frame[y1:y2, x1] = (255, 0, 0)
                frame[y1:y2, x2] = (255, 0, 0)

                # Create attention mask
                H, W, _ = obj_attention.shape
                x1 = max(0, min(x1, W - 1))
                x2 = max(0, min(x2, W - 1))
                y1 = max(0, min(y1, H - 1))
                y2 = max(0, min(y2, H - 1))
                obj_attention[y1:y2, x1:x2, 0] = 1.0

        # Compute area ratio
        self.episode.box_area_ratio = torch.tensor([obj_area / img_area]).to(self.device)
        if obj_attention.ndim == 3:
            obj_attention = obj_attention.unsqueeze(0)

        obj_attention = self.resize(obj_attention)
        center_x, center_y, W, H = self.compute_box_center(obj_attention)
        height = torch.tensor([event.metadata['agent']['position']['y'] - self.environment.start_height]).to(rgb.device)

        stats = torch.cat([center_x, center_y, W, H, self.episode.box_area_ratio, height])
        self.episode.stats = stats
        model_input.stats = stats

        self.episode.model_input = model_input
        return self.model.forward(model_input, model_options)

    def eval_at_state(self, model_options):
        if self.model_name in ['AIONg', 'AIONe']:
            return self._prepare_input_for_AION(model_options)
        elif self.model_name in ['ZSON', 'BaseModel', 'GCN', 'MJO']:
            return self._prepare_input_for_baseline(model_options)
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented in eval_at_state.")

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

    def preprocess_frame(self, frame):
        """ Preprocess the current frame for input into the model. """
        state = torch.tensor(frame.copy(), dtype=torch.uint8)
        state = gpuify(state, self.gpu_id)
        return state

    def reset_hidden(self):
        # LSTM
        self.hidden = (
            torch.zeros(1, self.hidden_state_sz).to(self.device),
            torch.zeros(1, self.hidden_state_sz).to(self.device),
        )
        self.last_action = gpuify(torch.zeros((1, 1), dtype=torch.long), self.gpu_id)
        num_actions = self.action_space
        self.last_action_probs = torch.ones(1, num_actions).to(self.device) / num_actions

    def repackage_hidden(self):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        self.last_action = self.last_action.detach()

    def state(self):
        return self.preprocess_frame(self.episode.state_for_agent())

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

