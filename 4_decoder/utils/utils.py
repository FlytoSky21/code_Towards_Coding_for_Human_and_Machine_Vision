import torch

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import cv2
import scipy.ndimage as pyimg
import os
import random

def to_var(x):
    """Convert tensor to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_data(x):
    """Convert variable to tensor."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data

# custom weights initialization called on networks
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if not(m.bias is None):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# view images
def visualize(img_arr):
    plt.imshow(((img_arr.numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    # plt.axis('off')

# load one image in tensor format
def load_image(filename, load_type=0, wd=256, ht=256):
    #centerCrop = transforms.CenterCrop((wd, ht))
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])
    
    if load_type == 0:
        img = transform(Image.open(filename))
    else:
        img = transform(text_image_preprocessing(filename))
        
    return img.unsqueeze(dim=0)

def save_image(img, filename):
    tmp = ((img.numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))

# black and white text image to distance-based text image
def text_image_preprocessing(filename):
    I = np.array(Image.open(filename))
    BW = I[:,:,0] > 127
    G_channel = pyimg.distance_transform_edt(BW)
    G_channel[G_channel>255]=255
    B_channel = pyimg.distance_transform_edt(1-BW)
    B_channel[B_channel>255]=255
    I[:,:,1] = G_channel.astype('uint8')
    I[:,:,2] = B_channel.astype('uint8')
    return Image.fromarray(I)

# prepare batched filenames of all training data
def load_trainset_batchfnames(filepath, batch_size, usetrainnum=708, trainnum=100000):
    paths = os.listdir(filepath)
    stylenum = len(paths)
    trainnum = (trainnum / batch_size / 2) * batch_size * 2
    fnames = ['%s.png' % (i%usetrainnum+1) for i in range(trainnum)]
    pathid = [(i%stylenum) for i in range(trainnum/2)]
    random.shuffle(pathid)
    random.shuffle(fnames)
    trainbatches = [([]) for _ in range(trainnum/batch_size/2)]
    for i in range(trainnum/batch_size/2):
        traindatas = []
        for j in range(batch_size):
            ii = i * batch_size + j * 2
            traindatas += [[os.path.join(filepath, paths[pathid[ii/2]], 'train', fnames[ii]),
                          os.path.join(filepath, paths[pathid[(ii+1)/2]], 'train', fnames[ii+1])]]  
        trainbatches[i] += traindatas
    return trainbatches

# prepare batched filename of the style image for one shot learning
def load_oneshot_batchfnames(filename, batch_size, trainnum=100000):
    trainnum = (trainnum / batch_size / 2) * batch_size * 2
    trainbatches = [([]) for _ in range(trainnum/batch_size/2)]
    for i in range(trainnum/batch_size/2):
        traindatas = []
        for j in range(batch_size):
            traindatas += [[filename,filename]]  
        trainbatches[i] += traindatas
    return trainbatches