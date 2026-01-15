from typing import Optional

import torch
import torchvision.transforms.functional as TF


class Transform:
    is_random: bool = False

    def apply(self, x: torch.Tensor):
        raise NotImplementedError

    def __call__(
        self,
        x: torch.Tensor,
        T: Optional[int] = None,
        N: Optional[int] = None,
        V: Optional[int] = None,
        skip_one_T: bool = True,
    ):
        if not self.is_random:
            return self.apply(x)

        if None in (T, N, V):
            return self.apply(x)

        if T == 1 and skip_one_T:
            return self.apply(x)

        # put environment (n) first
        _, A, B, C = x.shape
        x = torch.einsum("tnvabc->ntvabc", x.view(T, N, V, A, B, C)).flatten(1, 2)

        # apply the same transform within each environment
        x = torch.cat([self.apply(imgs) for imgs in x])

        # put timestep (t) first
        _, A, B, C = x.shape
        x = torch.einsum("ntvabc->tnvabc", x.view(N, T, V, A, B, C)).flatten(0, 2)

        return x

class CLIPTransform(Transform):
    def __init__(self, size):
        self.size = size
        self.mean = (0.48145466, 0.4578275, 0.40821073)
        self.std = (0.26862954, 0.26130258, 0.27577711)

    def apply(self, x):
        x = x.permute(0, 3, 1, 2)
        x = TF.resize(x, self.size, interpolation=TF.InterpolationMode.BICUBIC)
        x = TF.center_crop(x, output_size=self.size)
        x = x.float() / 255.0
        x = TF.normalize(x, self.mean, self.std)
        return x

class ResizeTransform(Transform):
    def __init__(self, size):
        self.size = size

    def apply(self, x):
        x = x.permute(0, 3, 1, 2)
        x = TF.resize(x, self.size)
        x = TF.center_crop(x, output_size=self.size)
        x = x.float() / 255.0
        return x

def get_transform(name, size):
    if name == "resize":
        return ResizeTransform(size)
    elif name == "clip":
        return CLIPTransform(size)