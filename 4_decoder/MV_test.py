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
# filename = './data/1_e.png'
# savename = './data/output/1_e.png'

root_path = './1108data/test/resize_img'
save_path = './1108data/test/resize_output'


def get_dir_path(root_path, dir_list):
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path, dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)


netG = Pix2pix256(nef=64, out_channels=3, in_channels=3, useNorm='BN')
if torch.cuda.is_available():
    netG.cuda()

netB2 = myDilateBlur()
if torch.cuda.is_available():
    netB2.cuda()

netG.load_state_dict(torch.load('./1108data/models/resize_img/round_1_ICME-VGG-netG-MV_5.ckpt'))


def save_image(img, filename):
    tmp = ((img.numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))

netG.eval()

begin_time = time.time()
file_list = []
dir_list = []
get_dir_path(root_path,dir_list)
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
        real_ske = netB2(targetsketch[:, :, :, img_size:img_size * 2])
        with torch.no_grad():
            fake_output = netG(real_ske)
        save_image(to_data(fake_output[0]), save_file)
    end_time = time.time()
    cost_time = end_time-start_time
    print(dir+':  '+str(cost_time))

# for i in file_list:
#     full_name = i
#     dir_name = os.path.split(i)[0]
#     file_name = os.path.split(i)[1]
#     new_folder = os.path.split(dir_name)[1]
#     save_folder = os.path.join(save_path, new_folder)
#     if not os.path.exists(save_folder):
#         os.mkdir(save_folder)
#     save_file = os.path.join(save_folder, file_name)
#     targetsketch = to_var(load_image(full_name))
#     real_ske = netB2(targetsketch[:, :, :, img_size:img_size * 2])
#     with torch.no_grad():
#         fake_output = netG(real_ske)
#     save_image(to_data(fake_output[0]), save_file)
# end_time = time.time()
# cost_time = end_time-begin_time
# cost_mean_time = cost_time/file_num
# print(' file_num: %d, cost_time: %.4f, mean_cost_time: %.4f'
#       % (file_num,cost_time,cost_mean_time))
# targetsketch = to_var(load_image(filename))
# real_ske = netB2(targetsketch[:,:,:,img_size:img_size*2])
# with torch.no_grad():
#     fake_output = netG(real_ske)
# save_image(to_data(fake_output[0]), savename)
