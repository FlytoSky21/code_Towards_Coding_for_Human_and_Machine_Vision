from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import argparse

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

from torch.utils.data import DataLoader
import torchvision.datasets as dset

from utils.utils import to_var, to_data, weights_init, visualize, load_image
from utils.blur_network import myBlur, myDilate, myDilateBlur
from utils.model import Pix2pix256, DiscriminatorSN

parser = argparse.ArgumentParser()
parser.add_argument('--roundt',type=int,default=0)
parser.add_argument('--start_epoch',type=int,default=0)
args = parser.parse_args()

# training settings
batch_size = 8 
lr = 0.0002
beta1 = 0.5

img_size = 256
epoches = 10
gan_type = 'wgan' 
hinge = 10

# dataset
dataroot = 'D:/VGGFace2/tt/256/'
workers = 0

# loss function
L1loss = nn.L1Loss()
L2loss = nn.MSELoss()
criterion = nn.BCEWithLogitsLoss()


dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

print(len(dataset))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)


def compute_loss(gan_type, d_out, target, hinge=0):
    targets = d_out.new_full(size=d_out.size(), fill_value=target)

    if gan_type == 'standard':
        loss = F.binary_cross_entropy_with_logits(d_out, targets)
    elif gan_type == 'wgan':
        if hinge == 0:
            loss = (1-2*target) * d_out.mean()
        else:
            loss = (F.relu(hinge+(1-2*target) * d_out)).mean()
    else:
        raise NotImplementedError

    return loss

vgg_mean = (torch.tensor([0.485, 0.456, 0.406]) + 1) * 0.5
vgg_std = torch.tensor([0.229, 0.224, 0.225]) * 0.5
vgg_mean = vgg_mean.view(-1, 1, 1)
vgg_std = vgg_std.view(-1, 1, 1)

class VGGFeature(nn.Module):
    def __init__(self, cnn):
        super(VGGFeature, self).__init__()
        
        self.model1 = cnn[:2]
        self.model2 = cnn[2:7]
        self.model3 = cnn[7:12]
        self.model4 = cnn[12:21]
        
    def forward(self, x):
        conv1_1 = self.model1(x)
        conv2_1 = self.model2(conv1_1)
        conv3_1 = self.model3(conv2_1)
        conv4_1 = self.model4(conv3_1)
        return [conv2_1, conv3_1]
    

VGGNet = models.vgg19(pretrained=True).features
VGGfeatures = VGGFeature(VGGNet)

for param in VGGfeatures.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    VGGfeatures.cuda()

vgg_mean = to_var(vgg_mean)
vgg_std = to_var(vgg_std)

content_weights = [1, 0.5]
loss_fns = nn.MSELoss()


from math import exp
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return (-1)*ssim_map.mean()
    else:
        return (-1)*ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


netG = Pix2pix256(nef=64, out_channels=3, in_channels = 3, useNorm='BN')
netD = DiscriminatorSN(6, n_layers = 5, input_size=256)
if torch.cuda.is_available():
    netG.cuda()
    netD.cuda()

netG.apply(weights_init)
netD.apply(weights_init)

log_dir = './1108data/MVmodel'
if os.path.exists(log_dir):
    log_dir_G = log_dir + '/' + 'round_{:d}_ICME-VGG-netG-MV_{:d}.ckpt'.format(args.roundt, args.start_epoch)
    checkpoint_G = torch.load(log_dir_G)
    log_dir_D = log_dir + '/' + 'round_{:d}_ICME-VGG-netD-MV_{:d}.ckpt'.format(args.roundt, args.start_epoch)
    checkpoint_D = torch.load(log_dir_D)
    netG.load_state_dict(checkpoint_G)
    netD.load_state_dict(checkpoint_D)
    epoch = args.start_epoch
    roundt = args.roundt
    print('round{}, epoch{}'.format(args.roundt, args.start_epoch))
else:
    epoch = 0
    roundt = 0
    print('no models')

trainernetG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
trainernetD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

netB2 = myDilateBlur() 
if torch.cuda.is_available():
    netB2.cuda()
epoch_s = epoch
epoches = 10
for iii in range(roundt,2):
    for epoch in range(epoch_s,epoches):
        for i, data in enumerate(dataloader, 0):
            with torch.no_grad():
                real_img = to_var(data[0][:,:,:,0:img_size])
                real_ske = to_var(data[0][:,:,:,img_size:img_size*2])
                real_ske = netB2(real_ske)
                real_input = real_ske
                
            netD.zero_grad()        
            real_img.requires_grad_()
            
            real_concat = torch.cat((real_img, real_ske), dim=1)
            real_output = netD(real_concat)
            errD_real = compute_loss(gan_type, real_output, 1, hinge = hinge)
            errD_real.backward()

            # train with fake
            with torch.no_grad():
                fake_img = netG(real_input)
                
            fake_img.requires_grad_()
            fake_concat = torch.cat((fake_img, real_ske), dim=1)
            fake_output = netD(fake_concat)
            errD_fake = compute_loss(gan_type, fake_output, 0, hinge = hinge)
            errD_fake.backward()           

            trainernetD.step()  
            errD = errD_real + errD_fake

            netG.zero_grad()
            fake_img = netG(real_input)   
            fake_concat = torch.cat((fake_img, real_ske), dim=1)
            fake_output = netD(fake_concat)
            errG = compute_loss(gan_type, fake_output, 1)
            
            with torch.no_grad():
                real_rgb_f = VGGfeatures((real_img - vgg_mean) / vgg_std)            
            fake_rgb_f = VGGfeatures((fake_img - vgg_mean) / vgg_std) 
            errF = sum([content_weights[a] * loss_fns(A, real_rgb_f[a]) for a,A in enumerate(fake_rgb_f)])           
            
            errR = L1loss(fake_img, real_img) * 100 
            errSSIM = ssim((fake_img+1)/2, (real_img+1)/2) * 1000
            
            errT = errG + errR + errF + errSSIM
            errT.backward()
            trainernetG.step()

            if i %(6400/batch_size) == 0:
                print('[%d/%d][%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f, Loss_R: %.4f, Loss_F: %.4f, Loss_S: %.4f'
                      % (iii+1, 2, epoch+1, epoches, i+1, len(dataloader), 
                         errD.data.mean(), errG.data.mean(), errR.data.mean(), errF.data.mean(), errSSIM.data.mean()))
            if i %(51200/batch_size) == 0:
                plt.figure(figsize=(10,10),dpi=80)
                visualize(to_data(torch.cat((real_ske[0], real_img[0], fake_img[0]), dim=2)))
                plt.show()

        torch.save(netG.state_dict(), './1108data/models/resize_img/round_{:d}_ICME-VGG-netG-MV_{:d}.ckpt'.format(iii, epoch))
        torch.save(netD.state_dict(), './1108data/models/resize_img/round_{:d}_ICME-VGG-netD-MV_{:d}.ckpt'.format(iii, epoch))


