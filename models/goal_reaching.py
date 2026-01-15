from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from obj_utils.net_util import weights_init
from models.clip_utils import get_transform

from .model_io import ModelOutput
import timm
from timm.data import create_transform
import clip
import h5py

class AIONg(torch.nn.Module):
    def __init__(self, args):
        self.args = args
        action_space = args.action_space
        hidden_state_sz = args.hidden_state_sz
        super(AIONg, self).__init__()

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
        if args.add_stats:
            self.mask_stats_mlp = nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(),
                nn.Linear(64, 64)
            )
            lstm_input_sz += 64
        # -------------------------------------

        # action head--------------------------
        self.embed_action = nn.Embedding(action_space + 1, 32)
        self.actor_linear = nn.Linear(hidden_state_sz, action_space)
        lstm_input_sz += 32
        # -------------------------------------

        # Clip embedding------------------------
        with h5py.File("./data/clip_goal_text_embeddings.h5", "r") as f:
            embeddings = f["embeddings"][:]
            self.objects = [s.decode("utf-8") for s in f["texts"][:]]

        embeddings = torch.Tensor(embeddings)
        self.goal_text_emb = nn.Parameter(embeddings)
        self.goal_text_emb.requires_grad = False

        if args.add_clip_align:
            self.clip_transform = get_transform(name='clip', size=224)
            self.clip_encoder, _ = clip.load('RN50')
            for p in self.clip_encoder.parameters():
                p.requires_grad = False
            self.clip_encoder.eval()
            lstm_input_sz += 9
        else:
            self.goal_emb_mlp = nn.Linear(self.goal_text_emb.shape[1], 128)
            lstm_input_sz += 128
        # --------------------------------------

        # depth embedding------------------------
        if args.add_depth:
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
        z = F.normalize(z_layout, dim=-1)  # TN * OV x d
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

        if self.args.add_clip_align:
            idx = self.objects.index(model_input.target_class)
            text_emb = self.goal_text_emb[idx].view(1, -1)
            text_emb = F.normalize(text_emb, dim=-1)

            patches = model_input.patch
            patches = self.clip_transform(patches)  # (B x 3, 224, 224)

            with torch.no_grad():
                patch_embeddings = self.clip_encoder.encode_image(patches)  # (9, D)
                patch_embeddings = patch_embeddings.float()
                patch_embeddings = F.normalize(patch_embeddings, dim=-1)

            similarity = patch_embeddings @ text_emb.T
            x.append(similarity.view(1,-1))
        else:
            idx = self.objects.index(model_input.target_class)
            target_emb = self.goal_text_emb[idx].view(1, -1)
            t = torch.nn.functional.normalize(target_emb, dim=-1)
            target_emb = self.goal_emb_mlp(t)
            x.append(target_emb)

        if self.args.add_stats:
            stats = model_input.stats
            att_feature = self.mask_stats_mlp(stats.float().view(1, -1))
            x.append(att_feature)

        if self.args.add_depth:
            depth_vec = model_input.depth.float().view(1,-1)
            depth_feat = self.depth_net(depth_vec.unsqueeze(1))
            x.append(depth_feat)

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