# Underwater Image Enhancement

# In[]:

import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from math import exp
from torch.utils.data import Dataset
# from torch.utils.data import ConcatDataset
from torchvision import transforms
from PIL import Image
import torchvision
from collections import OrderedDict
import glob
import cv2
import math
import sys
from skimage import io, color, filters
import matplotlib.pyplot as plt

# In[]:

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction='mean')
        
    def forward(self, input, target):
        return self.loss(input, target)

# In[]:

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='mean')
        
    def forward(self, input, target):
        return self.loss(input, target)

# In[]:

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
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
class SSIMLoss(torch.nn.Module):
    
    def __init__(self, window_size=11, channel=3, size_average=True):
        
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.size_average = size_average
        self.window = create_window(self.window_size, self.channel)

    def forward(self, img1, img2):
        
        self.window = self.window.to(img1.device)           
        loss = 1 - _ssim(img1, img2, self.window, self.window_size, self.channel, self.size_average)
        return loss

# In[]:

class UIFMLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction = 'mean')
        
    def forward(self, raw, output, tm1, tm2, b):
        
        raw_hat = torch.mul(output, tm1) + torch.mul(b, 1 - tm2)
        return self.loss(raw, raw_hat)

# In[]:

class Loss(nn.Module):
    
    def __init__(self, model_name='HUWIENet'):
        super().__init__()
        
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()
        self.uifm_loss = UIFMLoss()
        self.mse_loss = MSELoss()
                
        if model_name == 'HUWIENet':
            
            self.model_name = 'HUWIENet'
            
            self.loss_name = ['l1_loss', 'ssim_loss', 'mse_val', 'total_loss']
            self.num_loss_fn = len(self.loss_name)
            
            self.l1_loss_w = 1.0
            self.ssim_loss_w = 1.0
            
        elif model_name == 'RevisedHUWIENet':
            
            self.model_name = 'RevisedHUWIENet'

            self.loss_name = ['l1_loss', 'ssim_loss', 'uifm_loss', 'i2im_loss', 'pim_loss', 'mse_val', 'total_loss']
            self.num_loss_fn = len(self.loss_name)
            
            self.l1_loss_w = 1.0
            self.ssim_loss_w = 0.0
            self.uifm_loss_w = 0.0
            self.i2im_loss_w = 0.0
            self.pim_loss_w = 0.0
            
    def forward(self, raw, output, gt):
        
        if self.model_name == 'HUWIENet':
                        
            output, param = output
            
            # L1 loss
            l1_loss = self.l1_loss(output, gt)
            
            # SSIM loss
            ssim_loss = self.ssim_loss(output, gt)
                             
            # Total loss
            total_loss = (self.l1_loss_w * l1_loss + self.ssim_loss_w * ssim_loss)
            
            # MSE val
            mse_val = self.mse_loss(output, gt) * 255 * 255
            
            loss_val = [l1_loss, ssim_loss, mse_val, total_loss]
            
            log = ' '.join([self.loss_name[k] + ': ' + str(np.round(loss_val[k].item(), 5)) for k in range(self.num_loss_fn)])
            
        elif self.model_name == 'RevisedHUWIENet':
            
            output, param = output
            tm1, tm2, b, i2im_out, pim_out  = torch.split(param, [3, 3, 3, 3, 3], dim=1)
                        
            # L1 loss
            l1_loss = self.l1_loss(output, gt)
            
            # SSIM loss
            ssim_loss = self.ssim_loss(output, gt)
            
            # UIFM loss
            uifm_loss = self.uifm_loss(raw, output, tm1, tm2, b)
            
            # I2IM loss
            i2im_loss = self.l1_loss(i2im_out, gt)
            
            # PIM loss
            pim_loss = self.l1_loss(pim_out, gt)
            
            # Total loss
            total_loss = (self.l1_loss_w * l1_loss + self.ssim_loss_w * ssim_loss 
                          + self.uifm_loss_w * uifm_loss + self.i2im_loss_w * i2im_loss 
                          + self.pim_loss_w * pim_loss)
            
            # MSE val
            mse_val = self.mse_loss(output, gt) * 255 * 255
            
            loss_val = [self.l1_loss_w * l1_loss, self.ssim_loss_w * ssim_loss, 
                        self.uifm_loss_w * uifm_loss, self.i2im_loss_w * i2im_loss, 
                        self.pim_loss_w * pim_loss, mse_val, total_loss]
            
            log = ' '.join([self.loss_name[k] + ': ' + str(np.round(loss_val[k].item(), 5)) for k in range(self.num_loss_fn)])
            
        return loss_val, log

# In[]:

class HUWIENet(nn.Module):

    def __init__(self):
        
        super().__init__()
        
        self.init_layers()
        # self.init_weight()
        self.get_parameters()

    def init_layers(self):
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        # Image-to-Image Module
        
        self.i2im_conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.i2im_in1 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)       
        self.i2im_conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.i2im_in2 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.i2im_conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.i2im_in3 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.i2im_conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.i2im_in4 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.i2im_conv5 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.i2im_in5 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.i2im_conv6 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.i2im_in6 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.i2im_conv7 = nn.Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # Physics-Informed Module
        
        self.pim_conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pim_in1 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)       
        self.pim_conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pim_in2 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.pim_conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pim_in3 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.pim_conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pim_in4 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.pim_conv5 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pim_in5 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.pim_conv6 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pim_in6 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.pim_conv7 = nn.Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        
        # Fusion Module
               
        self.con1 = nn.Conv2d(9, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in1 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False) 
        self.con2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in2 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.con3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in3 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.con4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in4 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.con5 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in5 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.con6 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in6 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.con7 = nn.Conv2d(64, 6, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        
    def init_weight(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)

    def get_parameters(self):
        
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.net_parameters = {'Total': total_num, 'Trainable': trainable_num}
        
    def forward(self, x):
        
        h = self.relu(self.i2im_in1(self.i2im_conv1(x)))
        h = self.relu(self.i2im_in2(self.i2im_conv2(h)))
        h = self.relu(self.i2im_in3(self.i2im_conv3(h)))
        h = self.relu(self.i2im_in4(self.i2im_conv4(h)))
        h = self.relu(self.i2im_in5(self.i2im_conv5(h)))
        h = self.relu(self.i2im_in6(self.i2im_conv6(h)))
        h = self.i2im_conv7(h)
        h += x
        i2im_out = self.sigmoid(h)
        
        h2 = self.relu(self.pim_in1(self.pim_conv1(x)))
        h2 = self.relu(self.pim_in2(self.pim_conv2(h2)))
        h2 = self.relu(self.pim_in3(self.pim_conv3(h2)))
        h2 = self.relu(self.pim_in4(self.pim_conv4(h2)))
        h2 = self.relu(self.pim_in5(self.pim_conv5(h2)))
        h2 = self.relu(self.pim_in6(self.pim_conv6(h2)))
        t = self.pim_conv7(h2)
        t += x
        t = self.sigmoid(t)
        
        dark = self.dark_channel(x)
        b = self.atmospheric_light(x, dark)
        
        eps = 1e-05
        pim_out = torch.div((x - torch.mul(b, 1 - t)), (t + eps))
        pim_out = self.sigmoid(pim_out)
        
        att_in = torch.cat([x, i2im_out, pim_out], dim=1)
        h3 = self.relu(self.in1(self.con1(att_in)))
        h3 = self.relu(self.in2(self.con2(h3)))
        h3 = self.relu(self.in3(self.con3(h3)))
        h3 = self.relu(self.in4(self.con4(h3)))
        h3 = self.relu(self.in5(self.con5(h3)))
        h3 = self.relu(self.in6(self.con6(h3)))
        h3 = self.con7(h3)
        att_out = self.sigmoid(h3)
        
        m1, m2 = torch.split(att_out, 3, dim=1)
        
        output = 0.5 * torch.mul(m1, i2im_out) + 0.5 * torch.mul(m2, pim_out)
                
        return output, None
    
    def dark_channel(self, img):
            
        patch_size = 15
            
        # Step 1: Disable the red channel, use only blue and green channels
        no_red_img = img[:, 1:, :, :]  # Tensor of shape (batch_size, 2, H, W) (green and blue channels)
        
        # Step 2: Find the minimum values in each channel of the image
        min_img, _ = torch.min(no_red_img, dim=1, keepdim=True)
            
        # Step 3: Perform min pooling over the minimum values
        dark = -F.max_pool2d(-min_img, kernel_size=patch_size, stride=1, padding=patch_size//2)
            
        return dark
            
    def atmospheric_light(self, img, dark_channel):
            
        # Flatten the image and dark channel map
        flat_img = img.view(img.size(0), img.size(1), -1)  # (batch_size, 3, H*W)
        flat_dark = dark_channel.view(dark_channel.size(0), dark_channel.size(1), -1)  # (batch_size, 1, H*W)
        
        # Select the brightest 0.1% of pixels in the dark channel
        num_pixels = flat_dark.size(dim=2)
        num_top_pixels = int(0.001 * num_pixels)
        _, indices = torch.topk(flat_dark, k=num_top_pixels, dim=2, largest=True, sorted=False)
        
        # Retrieve the RGB values of the selected pixels and find the maximum value
        A = torch.gather(flat_img, 2, indices.expand(-1, img.size(1), -1)).max(dim=2)[0]
            
        A = A.unsqueeze(2).unsqueeze(3)
            
        return A

class RevisedHUWIENet(nn.Module):

    def __init__(self):
        
        super().__init__()
        
        self.init_layers()
        # self.init_weight()
        self.get_parameters()

    def init_layers(self):
        
        self.af = nn.ReLU(inplace=True)
        # self.af =  nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)
        
        # Image-to-Image Module
        
        self.i2im_conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.i2im_in1 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)       
        self.i2im_conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.i2im_in2 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.i2im_conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.i2im_in3 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.i2im_conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.i2im_in4 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.i2im_conv5 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.i2im_in5 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.i2im_conv6 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.i2im_in6 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.i2im_conv7 = nn.Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        
        # Physics-Informed Module
        
        self.pim_conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pim_in1 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)       
        self.pim_conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pim_in2 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.pim_conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pim_in3 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.pim_conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pim_in4 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.pim_conv5 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pim_in5 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.pim_conv6 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pim_in6 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.pim_conv7 = nn.Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        
        # BLE Module
                
        blem_kernel_size = 33
        self.blem_lpf = nn.Conv2d(1, 1, kernel_size=blem_kernel_size, stride=1, padding=blem_kernel_size // 2, bias=False, padding_mode='replicate')
        self.blem_lpf.weight.data = torch.ones(1, 1, blem_kernel_size, blem_kernel_size, dtype=torch.float32) / (blem_kernel_size * blem_kernel_size)
        self.blem_lpf.weight.requires_grad = False
        self.blem_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.blem_conv2 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)   
        
        # Fusion Module
               
        self.con1 = nn.Conv2d(9, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in1 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False) 
        self.con2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in2 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.con3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in3 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.con4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in4 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.con5 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in5 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.con6 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in6 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.con7 = nn.Conv2d(64, 6, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))    
        
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def get_parameters(self):
        
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.net_parameters = {'Total': total_num, 'Trainable': trainable_num}
        
    def forward(self, x):
        
        # regularization loss
        # rl1 = torch.max(self.relu(torch.abs(h)-4))
        
        # Image-to-Image Module
        
        h1 = self.i2im_conv1(x)
        h1_1 = self.i2im_in1(h1)
        h1_1_1 = self.af(h1_1)
        h2 = self.i2im_conv2(h1_1_1)
        h2_2 = self.i2im_in2(h2)
        h2_2_2 = self.af(h2_2)
        h3 = self.i2im_conv3(h2_2_2)
        h3_3 = self.i2im_in3(h3)
        h3_3_3 = self.af(h3_3)
        h4 = self.i2im_conv4(h3_3_3)
        h4_4 = self.i2im_in4(h4)
        h4_4_4 = self.af(h4_4)
        h5 = self.i2im_conv5(h4_4_4)
        h5_5 = self.i2im_in5(h5)
        h5_5_5 = self.af(h5_5)
        h6 = self.i2im_conv6(h5_5_5)
        h6_6 = self.i2im_in6(h6)
        h6_6_6 = self.af(h6_6)
        h7 = self.i2im_conv7(h6_6_6)
        h7_7 = self.sigmoid(h7)
        i2im_out = h7_7
                
        # Physics-Informed Module
        
        h1 = self.pim_conv1(x)
        h1_1 = self.pim_in1(h1)
        h1_1_1 = self.af(h1_1)
        h2 = self.pim_conv2(h1_1_1)
        h2_2 = self.pim_in2(h2)
        h2_2_2 = self.af(h2_2)
        h3 = self.pim_conv3(h2_2_2)
        h3_3 = self.pim_in3(h3)
        h3_3_3 = self.af(h3_3)
        h4 = self.pim_conv4(h3_3_3)
        h4_4 = self.pim_in4(h4)
        h4_4_4 = self.af(h4_4)
        h5 = self.pim_conv5(h4_4_4)
        h5_5 = self.pim_in5(h5)
        h5_5_5 = self.af(h5_5)
        h6 = self.pim_conv6(h5_5_5)
        h6_6 = self.pim_in6(h6)
        h6_6_6 = self.af(h6_6)
        h7 = self.pim_conv7(h6_6_6)
        h7_7 = self.sigmoid(h7)
        t = h7_7
        tm1 = t
        tm2 = t
                
        # BLE Module
        
        channels = torch.chunk(x, chunks=3, dim=1)
        filtered_channels = [self.blem_lpf(channel) for channel in channels]
        x2 = torch.cat(filtered_channels, dim=1)        
        b = self.blem_conv2(self.af(self.blem_conv1(x2)))
        b = self.sigmoid(b)
            
        eps = 1e-05
        pim_out = torch.div((x - b + torch.mul(b, tm2)), (tm1 + eps))
        pim_out = self.sigmoid(pim_out)
        
        # Fusion Module
        
        fm_in = torch.cat([x, i2im_out, pim_out], dim=1)
        h1 = self.con1(fm_in)
        h1_1 = self.in1(h1)
        h1_1_1 = self.af(h1_1)
        h2 = self.con2(h1_1_1)
        h2_2 = self.in2(h2)
        h2_2_2 = self.af(h2_2)
        h3 = self.con3(h2_2_2)
        h3_3 = self.in3(h3)
        h3_3_3 = self.af(h3_3)
        h4 = self.con4(h3_3_3)
        h4_4 = self.in4(h4)
        h4_4_4 = self.af(h4_4)
        h5 = self.con5(h4_4_4)
        h5_5 = self.in5(h5)
        h5_5_5 = self.af(h5_5)
        h6 = self.con6(h5_5_5)
        h6_6 = self.in6(h6)
        h6_6_6 = self.af(h6_6)
        h7 = self.con7(h6_6_6)
        h7_7 = self.sigmoid(h7)
        fm_out1, fm_out2 = torch.split(h7_7, [3,3], dim=1)
        
        param = torch.cat([tm1, tm2, b, i2im_out, pim_out], dim=1)
        output = 0.5 * torch.mul(fm_out1, i2im_out) + 0.5 * torch.mul(fm_out2, pim_out)
        
        return output, param

# In[]:

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, init_features=32):
        
        super().__init__()

        features = init_features
        
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
                
        self.get_parameters()

    def get_parameters(self):
        
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.net_parameters = {'Total': total_num, 'Trainable': trainable_num}
    
    def forward(self, x):
                
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        unet_out = torch.sigmoid(self.conv(dec1))
        
        return unet_out

    
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1", nn.Conv2d(in_channels=in_channels,out_channels=features,kernel_size=3,padding=1,bias=False)),
                    # (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "norm1", nn.InstanceNorm2d(num_features=features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "conv2", nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3,padding=1,bias=False)),
                    # (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "norm2", nn.InstanceNorm2d(num_features=features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)),
                    (name + "relu2", nn.ReLU(inplace=True))
                ]
            )
        )
    
# In[]:

class UIEBD(Dataset):
    def __init__(self, split='train'):
        
        self.split = split
        
        self.train_data_idx = [0, 800]
        self.test_data_idx = [800, 890]
        
        self.fp = '../../../Data/UIEBD/UIEBD_random_shuffle_3.txt'
        
        self.raw_root = '../../../Data/UIEBD/raw/'
        self.gt_root = '../../../Data/UIEBD/gt/'
                
        t = []
        with open(self.fp, 'r') as f:
            data_list = f.readlines()
            for data in data_list:
                data = data.split('\n')[0]
                t.append({'raw_paths': self.raw_root + data,
                          'gt_paths': self.gt_root + data})

        if self.split == 'train':
            self.data_infos = t[self.train_data_idx[0]:self.train_data_idx[1]]
        elif self.split == 'test':
            self.data_infos = t[self.test_data_idx[0]:self.test_data_idx[1]]
                        
    def __len__(self):

        return len(self.data_infos)

    def __getitem__(self, idx):
        
        imgs = copy.deepcopy(self.data_infos[idx])
        
        raw = Image.open(imgs['raw_paths']).convert('RGB')
        gt = Image.open(imgs['gt_paths']).convert('RGB')
        
        if self.split == 'train':

            f1 = transforms.ToTensor()           
            raw = f1(raw)
            gt = f1(gt)
            data = torch.cat((raw, gt), 0)
            
            f2 = transforms.RandomHorizontalFlip(p=0.5)
            data = f2(data)
            
            f3 = transforms.Resize([320, 320])
            data = f3(data) 
                        
            raw = data[0:3, :, :]
            gt = data[3:6, :, :]
            
            imgs['raw'] = raw
            imgs['gt'] = gt
            
        elif self.split == 'test':
            
            f1 = transforms.ToTensor()           
            raw = f1(raw)
            gt = f1(gt)
            data = torch.cat((raw, gt), 0)
            
            # f2 = transforms.RandomHorizontalFlip(p=0.5)
            # data = f2(data)
            
            f3 = transforms.Resize([320, 320])
            data = f3(data) 
                        
            raw = data[0:3, :, :]
            gt = data[3:6, :, :]
            
            imgs['raw'] = raw
            imgs['gt'] = gt
            
        return imgs
        
class CocoDataset(Dataset):
    def __init__(self, split='train'):

        np.random.seed(123)
        paths = glob.glob('../../../Data/literatur_datasetler/coco2017/train2017' + "/*.jpg")
        paths_subset = np.random.choice(paths, 10_000, replace=False)
        rand_idxs = np.random.permutation(10_000)
        train_idxs = rand_idxs[:8000]
        val_idxs = rand_idxs[8000:]
        self.train_paths = paths_subset[train_idxs]
        self.val_paths = paths_subset[val_idxs]
        
        self.size = 320
        
        if split == 'train':
            self.transforms = transforms.Compose([                  
                transforms.Resize((self.size, self.size),  Image.BICUBIC),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor()
            ])
        elif split == 'val':
            self.transforms = transforms.Compose([                  
                transforms.Resize((self.size, self.size),  Image.BICUBIC),
                transforms.ToTensor()
            ])
        
        self.split = split
        self.paths = paths
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        return img
    
    def __len__(self):
        return len(self.paths)

class LSUI(Dataset):
    def __init__(self, split='train'):
        
        self.split = split
        
        raw_paths = sorted(glob.glob('../../../Data/LSUI/input/*.jpg'))
        gt_paths = sorted(glob.glob('../../../Data/LSUI/GT/*.jpg'))

        np.random.seed(123)
        num_images = 4279
        rand_idxs = np.random.permutation(num_images)
        train_idxs = rand_idxs[:3879]
        test_idxs = rand_idxs[3879:]
        
        self.train_raw_paths = np.array(raw_paths)[train_idxs]
        self.train_gt_paths = np.array(gt_paths)[train_idxs]
        self.test_raw_paths = np.array(raw_paths)[test_idxs]
        self.test_gt_paths = np.array(gt_paths)[test_idxs]
        
    def __getitem__(self, idx):
        
        imgs = {}
        
        if self.split == 'train':
            
            raw = Image.open(self.train_raw_paths[idx]).convert("RGB")
            gt = Image.open(self.train_gt_paths[idx]).convert("RGB")
            
            # depth = self.da.infer_image(cv2.imread(self.train_raw_paths[idx]))
            depth = gt
            
            f1 = transforms.ToTensor()           
            raw = f1(raw)
            gt = f1(gt)
            depth = f1(depth)
            data = torch.cat((raw, gt, depth), 0)
            
            f2 = transforms.RandomHorizontalFlip(p=0.5)
            data = f2(data)
            
            f3 = transforms.Resize([320, 320])
            data = f3(data) 
                        
            raw = data[0:3, :, :]
            gt = data[3:6, :, :]
            depth = data[6:7, :, :]
            
            imgs['raw'] = raw
            imgs['gt'] = gt
            imgs['depth'] = depth
            imgs['raw_paths'] = self.train_raw_paths[idx]
            imgs['gt_paths'] = self.train_gt_paths[idx]
            
        elif self.split == 'test':
            
            raw = Image.open(self.test_raw_paths[idx]).convert("RGB")
            gt = Image.open(self.test_gt_paths[idx]).convert("RGB")
            
            # depth = self.da.infer_image(cv2.imread(self.test_raw_paths[idx]))
            depth = gt
            
            f1 = transforms.ToTensor()           
            raw = f1(raw)
            gt = f1(gt)
            depth = f1(depth)
            data = torch.cat((raw, gt, depth), 0)
            
            # f2 = transforms.RandomHorizontalFlip(p=0.5)
            # data = f2(data)
            
            f3 = transforms.Resize([320, 320])
            data = f3(data) 
                        
            raw = data[0:3, :, :]
            gt = data[3:6, :, :]
            depth = data[6:7, :, :]
            
            imgs['raw'] = raw
            imgs['gt'] = gt
            imgs['depth'] = depth
            imgs['raw_paths'] = self.test_raw_paths[idx]
            imgs['gt_paths'] = self.test_gt_paths[idx]
                        
        return imgs
    
    def __len__(self):
        
        if self.split == 'train':          
            return len(self.train_raw_paths)
        
        elif self.split == 'test':
            return len(self.test_raw_paths)

# In[]:
 
def setup_logger(work_dir, timestamp, level=logging.INFO):

    log_file = os.path.join(work_dir, f'{timestamp}.log')

    # Logger objesini yarat ve seviyesini ayarla
    logger = logging.getLogger(timestamp)
    logger.setLevel(level)

    # Log formatını belirle
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Konsol çıktısı için StreamHandler yarat ve ekle
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Dosya çıktısı için FileHandler yarat ve ekle
    file_handler = logging.FileHandler(log_file, 'w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# In[]:

def trainHUWIENet(model_name):
    
    model = model_name()
    name = 'Train258'    
    work_dir = '../../../Data/checkpoints/'
    num_epoch = 50
    lr = 1e-3
    step_size = 50
    gamma = 0.5
    train_batch_size = 8
    
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime()) 
    work_dir = work_dir + name + '_' + model.__class__.__name__ + '_' + timestamp + '/'    
    p = os.path.abspath(work_dir)
    if not os.path.exists(p): os.makedirs(p) 

    # create text log
    logger = setup_logger(work_dir, timestamp, level=logging.INFO)
    logger.info(name + ' model: ' + model.__class__.__name__ + 
                ' epoch: ' + str(num_epoch) + ' lr: ' + str(lr) + 
                ' step_size: ' + str(step_size) + ' gamma: ' + str(gamma) + 
                ' train batch size: ' + str(train_batch_size))
    
    # dataset
    train_dataset_uiebd = UIEBD(split='train')
    # train_dataset_lsui = LSUI(split='train')
    # combined_dataset = ConcatDataset([train_dataset_uiebd, train_dataset_lsui])
    logger.info('dataset: ' + train_dataset_uiebd.__class__.__name__)
    train_dataloader = DataLoader(train_dataset_uiebd, batch_size=train_batch_size, shuffle=True)   
        
    # models
    logger.info('model parameters: ' + str(model.net_parameters))
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logger.info('device: ' + str(device))
    model.to(device)

    # model's state_dict
    logger.info("model's state_dict:")
    for param_tensor in model.state_dict():
        logger.info(param_tensor + " - " + str(model.state_dict()[param_tensor].size()))

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # print optimizer's state_dictsss
    logger.info("optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        logger.info(var_name + " - " + str(optimizer.state_dict()[var_name]))

    # loss
    criterion = Loss(model_name=model.__class__.__name__)
        
    logger.info('start train')
    t = time.time()
    
    for epoch in range(1, num_epoch + 1):
        
        logger.info('epoch %d', epoch)        
        train_loss_avg = np.zeros(criterion.num_loss_fn, dtype='float32')
        
        model.train()
        
        for i, tdata in enumerate(train_dataloader):
            
            data_time = time.time() - t
            
            batch_size, _, _, _ = tdata['raw'].size()
            
            raw = tdata['raw'].to(device)
            gt = tdata['gt'].to(device)
            
            optimizer.zero_grad()
            output = model(raw)
            loss_val, log = criterion(raw, output, gt)
            total_loss = loss_val[-1]
            total_loss.backward()
            optimizer.step()
            
            logger.info('train epoch: [%d/%d][%d/%d] time: %.3f lr: %f ', 
                        epoch, num_epoch, i+1, len(train_dataloader), data_time, optimizer.param_groups[0]['lr'])
            logger.info(log)
              
            for k in range(criterion.num_loss_fn):
                train_loss_avg[k] += loss_val[k].item() * batch_size
                            
        train_loss_avg = train_loss_avg / len(train_dataloader.dataset)
        
        logger.info('train epoch (average): [%d] ' + ' '.join([criterion.loss_name[k] + ': ' + 
                    str(np.round(train_loss_avg[k], 5)) for k in range(criterion.num_loss_fn)]), epoch)
                
        if epoch == num_epoch:
            pth_dir = work_dir + 'epoch{}.pth'.format(epoch)
            torch.save(model.state_dict(), pth_dir)
        
        scheduler.step
        
    logger.info('finish')
    
    return work_dir, pth_dir

# In[]:

class MSEMetric(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
            
    def forward(self, toutputs, tlabels):
        data_range = 255
        mse = self.mse(toutputs, tlabels) * data_range**2
        return mse.item()

# In[]:

class PSNRMetric(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, toutputs, tlabels):
        data_range = 255
        mse = self.mse(toutputs, tlabels) * data_range**2
        psnr = 10 * torch.log10(data_range**2 / mse)
        return psnr.item()

# In[]:

class SSIMMetric(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIMMetric, self).__init__()
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
        
        val = _ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return val.item()

# In[]:

class UIQMMetric(nn.Module):
    
    def __init__(self):
        super().__init__()
            
    def forward(self, x):

        return 0

# In[]:
    
class UCIQEMetric(nn.Module):
    
    def __init__(self):
        super().__init__()
            
    def forward(self, x):

        return 0

# In[]:

def test(test_work_dir, test_pth_dir, test_data, model_name):
    
    model = model_name()
    name = 'Test'
    test_batch_size = 1
    
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime()) 
    test_work_dir = test_work_dir + 'Test' + '_' + timestamp + '/'    
    p = os.path.abspath(test_work_dir)
    if not os.path.exists(p): os.makedirs(p) 
    test_work_dir_img_output = test_work_dir + 'img_output' '/'
    p = os.path.abspath(test_work_dir_img_output)
    if not os.path.exists(test_work_dir_img_output): os.makedirs(test_work_dir_img_output) 
    
    # create text log
    logger = setup_logger(test_work_dir, timestamp, level=logging.INFO)
    logger.info(name + ' model: ' + model.__class__.__name__ + ' test batch size: ' + str(test_batch_size))
        
    # dataset
    if test_data =='uiebd':
        test_dataset = UIEBD(split='test')
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    elif test_data =='lsui':
        test_dataset = LSUI(split='test')
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    logger.info('dataset: ' + test_data)
               
    # model
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logger.info('device: ' + str(device))
    model.load_state_dict(torch.load(test_pth_dir, map_location=torch.device(device)))
    model.to(device)
                
    # metric
    metrics_MSE = MSEMetric()
    metrics_PSNRMetric = PSNRMetric()
    metrics_SSIMMetric = SSIMMetric()
    metrics_UIQMMetric = UIQMMetric()
    metrics_UCIQEMetric = UCIQEMetric()
    
    logger.info('start test')
    t = time.time()
    
    avg_score_MSE_raw_gt = 0.0
    avg_score_MSE_output_gt = 0.0
    
    avg_score_PSNR_raw_gt = 0.0
    avg_score_PSNR_output_gt = 0.0
    
    avg_score_SSIM_raw_gt = 0.0
    avg_score_SSIM_output_gt = 0.0
    
    avg_score_UIQM_raw = 0.0
    avg_score_UIQM_gt = 0.0
    avg_score_UIQM_output = 0.0
    
    avg_score_UCIQE_raw = 0.0
    avg_score_UCIQE_gt = 0.0
    avg_score_UCIQE_output = 0.0
    
    model.eval()
    with torch.no_grad():
        for k, testdata in enumerate(test_dataloader):
            
            data_time = time.time() - t
            
            batch_size, _, _, _ = testdata['raw'].size()
            
            tinputs = testdata['raw'].to(device)
            tlabels = testdata['gt'].to(device)
                        
            toutputs = model(tinputs)[0]
            
            score_MSE_raw_gt = metrics_MSE(tinputs, tlabels)
            avg_score_MSE_raw_gt += score_MSE_raw_gt * batch_size
            
            score_MSE_output_gt = metrics_MSE(toutputs, tlabels)
            avg_score_MSE_output_gt += score_MSE_output_gt * batch_size
            
            score_PSNR_raw_gt = metrics_PSNRMetric(tinputs, tlabels)
            avg_score_PSNR_raw_gt += score_PSNR_raw_gt * batch_size
            
            score_PSNR_output_gt = metrics_PSNRMetric(toutputs, tlabels)
            avg_score_PSNR_output_gt += score_PSNR_output_gt * batch_size
            
            score_SSIM_raw_gt = metrics_SSIMMetric(tinputs, tlabels)
            avg_score_SSIM_raw_gt += score_SSIM_raw_gt * batch_size
            
            score_SSIM_output_gt = metrics_SSIMMetric(toutputs, tlabels)
            avg_score_SSIM_output_gt += score_SSIM_output_gt * batch_size
                       
            score_UIQM_raw = metrics_UIQMMetric(tinputs)
            avg_score_UIQM_raw += score_UIQM_raw * batch_size
            
            score_UIQM_gt = metrics_UIQMMetric(tlabels)
            avg_score_UIQM_gt += score_UIQM_gt * batch_size
            
            score_UIQM_output = metrics_UIQMMetric(toutputs)
            avg_score_UIQM_output += score_UIQM_output * batch_size
            
            score_UCIQE_raw = metrics_UCIQEMetric(tinputs)
            avg_score_UCIQE_raw += score_UCIQE_raw * batch_size
            
            score_UCIQE_gt = metrics_UCIQEMetric(tlabels)
            avg_score_UCIQE_gt += score_UCIQE_gt * batch_size
            
            score_UCIQE_output = metrics_UCIQEMetric(toutputs)
            avg_score_UCIQE_output += score_UCIQE_output * batch_size
            
            file = testdata['raw_paths'][0].split('/')[-1]
            fp = os.path.join(test_work_dir_img_output, model.__class__.__name__ + '_' + file)
            torchvision.utils.save_image(toutputs, fp)
            
            # x_val = tlabels[0,0,:,:].cpu().numpy()
            # y_val = (tlabels[0,0,:,:] - toutputs[0,0,:,:]).cpu().numpy()
            # plt.scatter(x_val, y_val, alpha=0.7)
            # plt.xlabel('gt')
            # plt.ylabel('gt_enhanced_diff')
            # plt.grid(True)
            # fp2 = os.path.join(test_work_dir_img_output, model.__class__.__name__ + '_' + file.split('.')[0] + '_scatter_plot_R.png')
            # plt.savefig(fp2)
            # plt.clf()
            
            # x_val = tlabels[0,1,:,:].cpu().numpy()
            # y_val = (tlabels[0,1,:,:] - toutputs[0,1,:,:]).cpu().numpy()
            # plt.scatter(x_val, y_val, alpha=0.7)
            # plt.xlabel('gt')
            # plt.ylabel('gt_enhanced_diff')
            # plt.grid(True)
            # fp2 = os.path.join(test_work_dir_img_output, model.__class__.__name__ + '_' + file.split('.')[0] + '_scatter_plot_G.png')
            # plt.savefig(fp2)
            # plt.clf()
            
            # x_val = tlabels[0,2,:,:].cpu().numpy()
            # y_val = (tlabels[0,2,:,:] - toutputs[0,2,:,:]).cpu().numpy()
            # plt.scatter(x_val, y_val, alpha=0.7)
            # plt.xlabel('gt')
            # plt.ylabel('gt_enhanced_diff')
            # plt.grid(True)
            # fp2 = os.path.join(test_work_dir_img_output, model.__class__.__name__ + '_' + file.split('.')[0] + '_scatter_plot_B.png')
            # plt.savefig(fp2)
            # plt.clf()
            
            # img_plot = plt.imshow((tlabels[0,0,:,:] - toutputs[0,0,:,:]).cpu().numpy(), cmap='jet')
            # plt.colorbar(img_plot)
            # fp2 = os.path.join(test_work_dir_img_output, model.__class__.__name__ + '_' + file.split('.')[0] + '_scatter_plot_R2.png')
            # plt.axis('off')
            # plt.savefig(fp2)
            # plt.clf()
            
            # img_plot = plt.imshow((tlabels[0,1,:,:] - toutputs[0,1,:,:]).cpu().numpy(), cmap='jet')
            # plt.colorbar(img_plot)
            # fp2 = os.path.join(test_work_dir_img_output, model.__class__.__name__ + '_' + file.split('.')[0] + '_scatter_plot_G2.png')
            # plt.axis('off')
            # plt.savefig(fp2)
            # plt.clf()
            
            # img_plot = plt.imshow((tlabels[0,2,:,:] - toutputs[0,2,:,:]).cpu().numpy(), cmap='jet')
            # plt.colorbar(img_plot)
            # fp2 = os.path.join(test_work_dir_img_output, model.__class__.__name__ + '_' + file.split('.')[0] + '_scatter_plot_B2.png')
            # plt.axis('off')
            # plt.savefig(fp2)
            # plt.clf()
            
            logger.info('-------------------------------------------------')
            logger.info('test => [%d/%d] time: %.3f image: %s', k+1, len(test_dataloader), data_time, file)
            logger.info('mse_raw-gt: %.4f mse_out-gt: %.4f ', score_MSE_raw_gt, score_MSE_output_gt)
            logger.info('psnr_raw-gt: %.4f psnr_out-gt: %.4f ', score_PSNR_raw_gt, score_PSNR_output_gt)
            logger.info('ssim_raw-gt: %.4f ssim_out-gt: %.4f ', score_SSIM_raw_gt, score_SSIM_output_gt)
            logger.info('uiqm_raw: %.4f uiqm_gt: %.4f uiqm_output: %.4f ', score_UIQM_raw, score_UIQM_gt, score_UIQM_output)
            logger.info('uciqe_raw: %.4f uciqe_gt: %.4f uciqe_output: %.4f ', score_UCIQE_raw, score_UCIQE_gt, score_UCIQE_output)
            
    avg_score_MSE_raw_gt /= len(test_dataloader.dataset)
    avg_score_PSNR_raw_gt /= len(test_dataloader.dataset)
    avg_score_SSIM_raw_gt /= len(test_dataloader.dataset)
    avg_score_MSE_output_gt /= len(test_dataloader.dataset)
    avg_score_PSNR_output_gt /= len(test_dataloader.dataset)
    avg_score_SSIM_output_gt /= len(test_dataloader.dataset)
    
    avg_score_UIQM_raw /= len(test_dataloader.dataset)
    avg_score_UIQM_gt /= len(test_dataloader.dataset)
    avg_score_UIQM_output /= len(test_dataloader.dataset)
    avg_score_UCIQE_raw /= len(test_dataloader.dataset)
    avg_score_UCIQE_gt /= len(test_dataloader.dataset)
    avg_score_UCIQE_output /= len(test_dataloader.dataset)
    
    logger.info('-------------------------------------------------')
    logger.info('epoch test results (average) =>')
    logger.info('mse_raw-gt: %.4f mse_out-gt: %.4f', avg_score_MSE_raw_gt, avg_score_MSE_output_gt)
    logger.info('psnr_raw-gt: %.4f psnr_out-gt: %.4f', avg_score_PSNR_raw_gt, avg_score_PSNR_output_gt)
    logger.info('ssim_raw-gt: %.4f ssim_out-gt: %.4f', avg_score_SSIM_raw_gt, avg_score_SSIM_output_gt)
    logger.info('uiqm_raw: %.4f uiqm_gt: %.4f uiqm_output: %.4f ', avg_score_UIQM_raw, avg_score_UIQM_gt, avg_score_UIQM_output)
    logger.info('uciqe_raw: %.4f uciqe_gt: %.4f uciqe_output: %.4f ', avg_score_UCIQE_raw, avg_score_UCIQE_gt, avg_score_UCIQE_output)
                 
    logger.info('finish')

# In[]:
    
# test_work_dir = '/home/itu/OzanDemir_504192220/Data/checkpoints/Train232_HUWIENet_20250329_163120/'
# test_pth_dir = test_work_dir + 'epoch50.pth'
# test(test_work_dir, test_pth_dir, 'lsui', HUWIENet)
# test_work_dir, test_pth_dir = trainHUWIENet(HUWIENet)
# test(test_work_dir, test_pth_dir, 'uiebd', HUWIENet)

# test_work_dir, test_pth_dir = trainHUWIENet(HUWIENet)
# test(test_work_dir, test_pth_dir, 'uiebd', HUWIENet)

# test_work_dir, test_pth_dir = trainHUWIENet(HUWIENet)
# test(test_work_dir, test_pth_dir, 'uiebd', HUWIENet)

# test_work_dir = '/home/itu/OzanDemir_504192220/Data/checkpoints/kayitlar_konferans/kayit/2/Train257_RevisedHUWIENet_20250405_035016/'
# test_work_dir = '/home/itu/OzanDemir_504192220/Data/checkpoints/kayitlar_konferans/Train231_RevisedHUWIENet_20250329_162853/'
# test_work_dir = '/home/itu/OzanDemir_504192220/Data/checkpoints/Train248_RevisedHUWIENet_20250403_011459/'
# test_pth_dir = test_work_dir + 'epoch50.pth'
# test(test_work_dir, test_pth_dir, 'lsui', RevisedHUWIENet)
test_work_dir, test_pth_dir = trainHUWIENet(RevisedHUWIENet)
test(test_work_dir, test_pth_dir, 'uiebd', RevisedHUWIENet)

test_work_dir, test_pth_dir = trainHUWIENet(RevisedHUWIENet)
test(test_work_dir, test_pth_dir, 'uiebd', RevisedHUWIENet)

test_work_dir, test_pth_dir = trainHUWIENet(RevisedHUWIENet)
test(test_work_dir, test_pth_dir, 'uiebd', RevisedHUWIENet)







