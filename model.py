import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images):
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        return h3

#
# class CIR(nn.Module):
#     def __init__(self,  in_channels, out_channels, kernel_size=3, pad_size=1):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad_size)
#         self.instance_norm = nn.InstanceNorm2d(out_channels)
#
#     def forward(self, x):
#         h = self.conv(x)
#         h = self.instance_norm(h)
#         h = F.relu(h)
#         return h
#
#
# class Decoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.cir1 = CIR(256, 128, 3, 1)
#         self.cir2 = CIR(128, 128, 3, 1)
#         self.cir3 = CIR(128, 64, 3, 1)
#         self.cir4 = CIR(64, 64, 3, 1)
#         self.out_conv = nn.Conv2d(64, 3, 3, padding=1)
#
#     def forward(self, features):
#         h = self.cir1(features)
#         h = F.interpolate(h, scale_factor=2)
#         h = self.cir2(h)
#         h = self.cir3(h)
#         h = F.interpolate(h, scale_factor=2)
#         h = self.cir4(h)
#         h = self.out_conv(h)
#         return h


class RC(nn.Module):
    """A wrapper of ReflectionPad2d and Conv2d"""
    def __init__(self, in_channels, out_channels, kernel_size=3, pad_size=1, activated=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.activated = activated

    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        if self.activated:
            return F.relu(h)
        else:
            return h


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rc1 = RC(256, 128, 3, 1)
        self.rc2 = RC(128, 128, 3, 1)
        self.rc3 = RC(128, 64, 3, 1)
        self.rc4 = RC(64, 64, 3, 1)
        self.rc5 = RC(64, 3, 3, 1, False)

    def forward(self, features):
        h = self.rc1(features)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc2(h)
        h = self.rc3(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc4(h)
        h = self.rc5(h)
        return h
