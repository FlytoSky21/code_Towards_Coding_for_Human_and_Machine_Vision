from __future__ import print_function
import numpy as np
import random
import cv2
import math

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm2d, LeakyReLU, ConvTranspose2d, ReLU, Tanh, InstanceNorm2d
import torch.nn.functional as F
from torch.autograd import Variable


class myBlur(nn.Module):
    def __init__(self, kernel_size=121, channels=3):
        super(myBlur, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.GF = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=kernel_size, groups=channels, bias=False)
        # Set these to whatever you want for your gaussian filter
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(self.kernel_size + 0.)
        x_grid = x_cord.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
        y_grid = x_grid.t()
        self.xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        self.mean = (self.kernel_size - 1) / 2
        self.diff = -torch.sum((self.xy_grid - self.mean) ** 2., dim=-1)
        self.gaussian_filter = nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                                         kernel_size=self.kernel_size, groups=self.channels, bias=False)

        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x, sigma):
        sigma = sigma * 8. + 16.
        variance = sigma ** 2.
        gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(self.diff / (2 * variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)
        if torch.cuda.is_available():
            gaussian_kernel = gaussian_kernel.cuda()
        self.gaussian_filter.weight.data = gaussian_kernel
        return self.gaussian_filter(F.pad(x, (self.mean, self.mean, self.mean, self.mean), "replicate"))


class myDilate(nn.Module):
    def __init__(self, kernel_size=10, channels=3):
        super(myDilate, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        # Set these to whatever you want for your gaussian filter
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        gaussian_kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)
        if torch.cuda.is_available():
            gaussian_kernel = gaussian_kernel.cuda()
        self.mean = (self.kernel_size - 1) / 2
        self.gaussian_filter = nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                                         kernel_size=self.kernel_size, groups=self.channels, bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        x = F.pad((1 - x) * 0.5, (self.mean, self.mean, self.mean, self.mean), "replicate")
        y = self.gaussian_filter(x)
        return 1 - 2 * torch.clamp(y, min=0, max=1)


class myDilateBlur(nn.Module):
    def __init__(self, kernel_size=7, channels=3, sigma=0.8):
        super(myDilateBlur, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        # Set these to whatever you want for your gaussian filter
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(self.kernel_size + 0.)
        x_grid = x_cord.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
        y_grid = x_grid.t()
        self.xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        self.mean = (self.kernel_size - 1) / 2
        self.diff = -torch.sum((self.xy_grid - self.mean) ** 2., dim=-1)
        self.gaussian_filter = nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                                         kernel_size=self.kernel_size, groups=self.channels, bias=False)

        self.gaussian_filter.weight.requires_grad = False
        variance = sigma ** 2.
        gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(self.diff / (2 * variance))
        gaussian_kernel = 2 * gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)
        self.gaussian_filter.weight.data = gaussian_kernel

    def forward(self, x):
        tar = self.mean
        tar = int(tar)
        input_ = 1 - x

        y = self.gaussian_filter(F.pad(1 - x, (tar, tar, tar, tar), "replicate"))
        return 1 - 2 * torch.clamp(y, min=0, max=1)
