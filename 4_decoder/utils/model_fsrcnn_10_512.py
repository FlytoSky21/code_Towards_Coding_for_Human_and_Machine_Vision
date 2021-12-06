from __future__ import print_function
import os

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
from datetime import datetime
import time
import logging
import random
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn import Linear, Conv2d, BatchNorm2d, LayerNorm, LeakyReLU, ConvTranspose2d, ReLU, Tanh, InstanceNorm2d
from torch.nn import ReflectionPad2d, ReplicationPad2d
from torch.nn.utils import spectral_norm


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class UpsampleConLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, padding='reflect'):
        super(UpsampleConLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        if padding == 'repeat':
            self.reflection_pad = ReplicationPad2d(reflection_padding)
        else:
            self.reflection_pad = ReflectionPad2d(reflection_padding)
        self.conv2d = Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out    
    
class RCBBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=3, padding=1, useNorm='BN'):
        super(RCBBlock, self).__init__()
        
        self.relu = LeakyReLU(0.2)
        self.pad = ReflectionPad2d(padding=padding)
        self.conv = Conv2d(out_channels=out_channels, kernel_size=kernel_size, stride=1,
                              padding=0, in_channels=in_channels)
        if useNorm == 'IN':
            self.bn = InstanceNorm2d(num_features=out_channels, affine=True)
        elif useNorm == 'BN':
            self.bn = BatchNorm2d(num_features=out_channels)
        else:
            self.bn = Identity()
        
    def forward(self, x):
        return self.conv(self.pad(self.relu(x)))
    
class RDCBBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=4, padding=1, useNorm='BN', up=False):
        super(RDCBBlock, self).__init__()
        
        self.relu = ReLU()
        self.dconv = ConvTranspose2d(out_channels=out_channels, kernel_size=kernel_size,
                                     stride=2, padding=1, in_channels=in_channels)
        # if up == False:
        #     self.dconv = ConvTranspose2d(out_channels=out_channels, kernel_size=kernel_size,
        #                              stride=2, padding=1, in_channels=in_channels)
        #     print(up == False)
        # else:
        #     self.dconv = UpsampleConLayer(out_channels=out_channels, kernel_size=kernel_size,
        #                              stride=1, in_channels=in_channels, upsample=2)
        
        if useNorm == 'IN':
            self.bn = InstanceNorm2d(num_features=out_channels, affine=True)
        elif useNorm == 'BN':
            self.bn = BatchNorm2d(num_features=out_channels)
        else:
            self.bn = Identity()
            
    def forward(self, x):
        return self.dconv(self.relu(x))


class Pix2pix256(nn.Module):
    def __init__(self, nef=32, out_channels=3, in_channels = 3, useNorm='NN'):
        super(Pix2pix256, self).__init__()
                   
        # 256*256*3-->256*256*32
        self.pad1 = ReflectionPad2d(padding=1)
        self.conv1 = Conv2d(in_channels, nef, 3, 1, 0)
        # 256*256*32-->128*128*64
        self.rcb0 = RCBBlock(nef, nef*4, useNorm=useNorm)
        # 128*128*64-->64*64*128
        self.rcb1 = RCBBlock(nef*4, nef*8, useNorm=useNorm)
        # 64*64*128-->32*32*256
        self.rcb2 = RCBBlock(nef*8, nef*8, useNorm=useNorm)
        # 32*32*256-->16*16*512
        self.rcb3 = RCBBlock(nef*8, nef*8, useNorm=useNorm)
        # 16*16*512-->8*8*512
        self.rcb4 = RCBBlock(nef*8, nef*8, useNorm=useNorm)
        # 8*8*512-->4*4*512
        self.rcb5 = RCBBlock(nef*8, nef*8, useNorm=useNorm)
        # 4*4*512-->2*2*512
        self.rcb6 = RCBBlock(nef*8, nef*8, useNorm=useNorm)
        self.rcb7 = RCBBlock(nef * 8, nef * 4, useNorm=useNorm)
        # 2*2*512-->1*1*512
        self.relu = LeakyReLU(0.2)
        self.pad2 = ReflectionPad2d(padding=1)
        self.conv7 = Conv2d(nef*8, nef*8, 4, 2, 0)
        # 1*1*512-->2*2*512
        self.rdcb7 = RDCBBlock(nef*4, nef, useNorm=useNorm)  #, up=False, padding = 'repeat')
        # 2*2*1024-->4*4*512
        self.rdcb6 = RDCBBlock(nef*16, nef*8, useNorm=useNorm, up=True)        
        # 4*4*1024-->8*8*512
        self.rdcb5 = RDCBBlock(nef*16, nef*8, useNorm=useNorm, up=True)       
        # 8*8*1024-->16*16*512
        self.rdcb4 = RDCBBlock(nef*16, nef*8, useNorm=useNorm, up=True)
        # 16*16*512-->32*32*256
        self.rdcb3 = RDCBBlock(nef*16, nef*4, useNorm=useNorm, up=True)
        # 32*32*512-->64*64*128
        self.rdcb2 = RDCBBlock(nef*8, nef*2, useNorm=useNorm, up=True)
        # 64*64*256-->128*128*64
        self.rdcb1 = RDCBBlock(nef*4, nef*1, useNorm=useNorm, up=True)
        # 128*128*128-->256*256*32
        self.rdcb0 = RDCBBlock(nef*2, nef*2, useNorm=useNorm, up=True)
        # 256*256*32-->256*256*3
        self.pad3 = ReflectionPad2d(padding=1)
        self.dconv1 = Conv2d(nef, out_channels, 3, 1, 0)
        self.tanh = Tanh()
            
    def forward(self, x):
        x01 = self.conv1(self.pad1(x))
        x0 = self.rcb0(x01)
        x1 = self.rcb1(x0)
        x2 = self.rcb2(x1)
        x3 = self.rcb3(x2)
        x4 = self.rcb4(x3)
        x5 = self.rcb5(x4)
        x6 = self.rcb6(x5)
        x7 = self.rcb7(x6)
        x8 = self.rdcb7(x7)

        return self.dconv1(self.pad3(F.relu(x8)))


class DiscriminatorSN(nn.Module):
    def __init__(self, in_channels, ndf=64, n_layers=3, input_size=64):
        super(DiscriminatorSN, self).__init__()
        
        modelList = []       
        kernel_size = 4
        padding = int(np.ceil((kernel_size - 1)/2))
        modelList.append(ReflectionPad2d(padding=padding))
        modelList.append(spectral_norm(Conv2d(out_channels=ndf, kernel_size=kernel_size, stride=2,
                              padding=0, in_channels=in_channels)))
        modelList.append(LeakyReLU(0.2))
        
        # 32 --> 16 --> 8
        size = input_size/2
        nf_mult = 1
        for n in range(1, n_layers):
            size = size / 2
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            modelList.append(ReflectionPad2d(padding=padding))
            modelList.append(spectral_norm(Conv2d(out_channels=ndf * nf_mult, kernel_size=kernel_size, stride=2,
                                  padding=0, in_channels=ndf * nf_mult_prev)))
            modelList.append(LeakyReLU(0.2))
        # 7
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        modelList.append(ReflectionPad2d(padding=padding))
        modelList.append(spectral_norm(Conv2d(out_channels=ndf * nf_mult, kernel_size=kernel_size, stride=1,
                              padding=0, in_channels=ndf * nf_mult_prev)))
        # 6
        modelList.append(LeakyReLU(0.2))
        modelList.append(ReflectionPad2d(padding=padding))
        modelList.append(spectral_norm(Conv2d(out_channels=ndf, kernel_size=kernel_size, stride=1,
                              padding=0, in_channels=ndf * nf_mult)))

        self.model = nn.Sequential(*modelList)
        
    def forward(self, x):
        out = self.model(x).view(x.size(0), -1)
        return out.view(-1)    
