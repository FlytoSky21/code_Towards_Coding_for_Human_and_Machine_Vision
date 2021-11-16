# coding:utf-8
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

from utils.utils import to_var, to_data, weights_init, visualize, load_image
from utils.blur_network import myBlur, myDilate, myDilateBlur
from utils.model import Pix2pix256, DiscriminatorSN

# testing settings
img_size = 256

# dataset
# filename = './data/1_ec.png'
# savename = './data/output/1_ec.png'


root_path= './data/dataHV1/128'
save_path='./data/HVoutput1103/128'

def get_dir_path(root_path, dir_list):
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path, dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)

netG = Pix2pix256(nef=64, out_channels=3, in_channels = 7, useNorm='BN')
if torch.cuda.is_available():
    netG.cuda()

netB2 = myDilateBlur() 
if torch.cuda.is_available():
    netB2.cuda()


netG.load_state_dict(torch.load('./model/ICME-VGG-netG-HV.ckpt'))

def save_image(img, filename):
    tmp = ((img.numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))

netG.eval()

file_list=[]
dir_list=[]
get_dir_path(root_path,dir_list)
print(dir_list)

for dir in dir_list:
    file_list = os.listdir(dir)
    start_time = time.time()
    for i in file_list:
        i = os.path.join(dir,i)
        full_name = i
        dir_name = os.path.split(i)[0]
        file_name = os.path.split(i)[1]
        new_folder = os.path.split(dir_name)[1]
        save_folder = os.path.join(save_path, new_folder)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_file = os.path.join(save_folder, file_name)
        targetsketch = to_var(load_image(full_name))
        masks = targetsketch[:, 0:1, :, img_size * 2:img_size * 3]
        real_img = targetsketch[:, :, :, 0:img_size]
        real_ske = netB2(targetsketch[:, :, :, img_size:img_size * 2])
        input_img = real_img * (1 + masks) / 2
        real_input = torch.cat((input_img, real_ske, masks), dim=1)
        with torch.no_grad():
            fake_output = netG(real_input)
        save_image(to_data(fake_output[0]), save_file)
    end_time = time.time()
    cost_time = start_time-end_time
    print(dir+':  '+str(cost_time))





# targetsketch = to_var(load_image(filename))
# masks = targetsketch[:,0:1,:,img_size*2:img_size*3]
# real_img = targetsketch[:,:,:,0:img_size]
# real_ske = netB2(targetsketch[:,:,:,img_size:img_size*2])
# input_img = real_img * (1+masks)/2
# real_input = torch.cat((input_img, real_ske, masks), dim=1)
# with torch.no_grad():
#     fake_output = netG(real_input)
# save_image(to_data(fake_output[0]), savename)

