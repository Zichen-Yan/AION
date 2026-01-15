from __future__ import division

import torch
import torch.nn as nn
from obj_utils.net_util import weights_init
from models.clip_utils import get_transform

from .model_io import ModelOutput
import timm
from timm.data import create_transform

class AIONe(torch.nn.Module):
    def __init__(self, args):
        self.args = args
        action_space = args.action_space
        hidden_state_sz = args.hidden_state_sz
        super(AIONe, self).__init__()

        lstm_input_sz = 0
        # vision backbone --------------------
        self.resize = get_transform(name="resize", size=224)
        if args.add_rgb:
            self.layout_transform = create_transform(
                input_size=224,
                is_training=False
            )
            self.layout_encoder = timm.create_model("timm/vit_small_patch14_dinov2.lvd142m",
                                                    pretrained=True, num_classes=0, img_size=224)

            self.layout_encoder.eval()
            for param in self.layout_encoder.parameters():
                param.requires_grad = False

            self.layout_policy_mlp = nn.Linear(384, 128)
            lstm_input_sz += 128
        # -------------------------------------

        # state encoder------------------------
        self.mask_stats_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        lstm_input_sz += 32
        # -------------------------------------

        # action head--------------------------
        self.embed_action = nn.Embedding(action_space + 1, 32)
        self.actor_linear = nn.Linear(hidden_state_sz, action_space)
        lstm_input_sz += 32
        # -------------------------------------

        # depth -------------------------------
        self.depth_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> [B, 32, 1]
            nn.Flatten()  # -> [B, 32]
        )
        lstm_input_sz += 32
        # --------------------------------------

        self.lstm = nn.LSTMCell(lstm_input_sz, hidden_state_sz)
        self.critic_linear = nn.Linear(hidden_state_sz, 1)
        self.apply(weights_init)

    def a3clstm(self, embedding, prev_hidden):
        hx, cx = self.lstm(embedding, prev_hidden)
        x = hx
        actor_out = self.actor_linear(x)
        critic_out = self.critic_linear(x)
        return actor_out, critic_out, (hx, cx)

    def visual_encoding(self, obs):
        layout = self.layout_transform(obs)
        z_layout = self.layout_encoder(layout)
        z = torch.nn.functional.normalize(z_layout, dim=-1)  # TN * OV x d
        return z

    def forward(self, model_input):
        (hx, cx) = model_input.hidden
        x = []

        if self.args.add_rgb:
            rgb = model_input.rgb
            with torch.no_grad():
                z_layout = self.visual_encoding(rgb)
            visual_feature = self.layout_policy_mlp(z_layout)
            x.append(visual_feature)

        depth_vec = model_input.depth.float().view(1,-1)
        depth_feat = self.depth_net(depth_vec.unsqueeze(1))
        x.append(depth_feat)

        ROI_feature = self.mask_stats_mlp(model_input.ROI.float().view(1, -1))
        x.append(ROI_feature)

        prev_actions = model_input.last_action

        if isinstance(self.embed_action, nn.Linear):
            action_embed = self.embed_action((prev_actions.float()).squeeze(dim=-1))
        else:
            action_embed = self.embed_action((prev_actions.float() + 1).long().squeeze(dim=-1))
        x.append(action_embed)  # TN x 32

        lstm_input = torch.cat(x, dim=1)
        actor_out, critic_out, (hx, cx) = self.a3clstm(lstm_input, (hx, cx))

        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
        )