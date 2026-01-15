""" Borrowed from https://github.com/andrewliao11/pytorch-a3c-mujoco/blob/master/model.py."""


import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.init as init

def gpuify(tensor, gpu_id):
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            tensor = tensor.cuda()
    return tensor

def toFloatTensor(x, gpu_id):
    """ Convers x to a FloatTensor and puts on GPU. """
    return gpuify(torch.FloatTensor(x), gpu_id)

def resnet_input_transform(input_image, im_size):
    """Takes in numpy ndarray of size (H, W, 3) and transforms into tensor for
       resnet input.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # mean=[0, 0, 0], std=[1, 1, 1])
    all_transforms = transforms.Compose([
        transforms.ToPILImage(),
        ScaleBothSides(im_size),
        transforms.ToTensor(),
        normalize,
    ])
    transformed_image = all_transforms(input_image)
    return transformed_image


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x ** 2).sum(1, keepdim=True))
    return x


def weights_init(m):
    if isinstance(m, nn.Linear) and m.weight.requires_grad:
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.LSTMCell):
        if m.weight_ih.requires_grad:
            nn.init.orthogonal_(m.weight_ih)
        if m.weight_hh.requires_grad:
            nn.init.orthogonal_(m.weight_hh)
        if m.bias_ih is not None:
            nn.init.zeros_(m.bias_ih)
        if m.bias_hh is not None:
            nn.init.zeros_(m.bias_hh)

    elif isinstance(m, nn.Conv2d) and m.weight.requires_grad:
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class ScaleBothSides(object):
    """Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of both edges, and this can change aspect ratio.
    size: output size of both edges
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)


class ScalarMeanTracker(object):
    def __init__(self) -> None:
        self._sums = {}
        self._counts = {}

    def add_scalars(self, scalars):
        for k in scalars:
            if k not in self._sums:
                self._sums[k] = scalars[k]
                self._counts[k] = 1
            else:
                self._sums[k] += scalars[k]
                self._counts[k] += 1

    def pop_and_reset(self):
        means = {k: self._sums[k] / self._counts[k] for k in self._sums}
        if 'total_collision' in means:
            means['collision_rate'] = self._sums['total_collision']/self._sums['ep_length']
        self._sums = {}
        self._counts = {}
        return means
