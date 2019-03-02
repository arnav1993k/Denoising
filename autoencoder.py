#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 09:35:45 2019

@author: arnavk
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import librosa
import numpy as np
from utils import *
import python_speech_features as psf
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    """5x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, transfer='tanh'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(inplanes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv4 = conv1x1(inplanes, planes)
        self.bn4 = nn.BatchNorm2d(planes)
        self.htanh = nn.Hardtanh(min_val=-10.0, max_val=5.0)
        self.stride = stride
        self.transfer=transfer

    def forward(self, x):
        identity = self.conv4(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out_3x3 = self.conv3(x)
        # out_3x3 = self.bn3(out_3x3)

        out = out+identity
        # out = nn.Tanh()(out)
        # out = out+identity
        if self.transfer=='tanh':
            out = nn.Tanh()(out)
        else:
            out = nn.ReLU(inplace=True)(out)
        return out

class AutoEncoder_Lib(nn.Module):
    def __init__(self, channels):
        super(AutoEncoder_Lib, self).__init__()
        layers= []
        in_channels = channels[0]
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=channels[1], kernel_size=1, padding=0,bias=False))
        layers.append(nn.BatchNorm2d(channels[1]))
        for i in range(1,len(channels)-2):
            block = BasicBlock(channels[i],channels[i+1],transfer='tanh')
            layers.append(block)
        layers.append(nn.Conv2d(in_channels=channels[-2], out_channels=channels[-1], kernel_size=3, padding=1,bias=False))
        # layers.append(nn.Hardtanh(min_val=-5.0, max_val=4.0))
        self.dncnn = nn.Sequential(*layers)
        self.minloss= 1000
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
    def forward(self,x):
        out = self.dncnn(x)
        return out

class AutoEncoder(nn.Module):
    def __init__(self, channels):
        super(AutoEncoder, self).__init__()
        layers= []
        in_channels = channels[0]
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=channels[1], kernel_size=1, padding=0,bias=False))
        layers.append(nn.BatchNorm2d(channels[1]))
        for i in range(1,len(channels)-2):
            block = BasicBlock(channels[i],channels[i+1],transfer='relu')
            layers.append(block)
        layers.append(nn.Conv2d(in_channels=channels[-2], out_channels=channels[-1], kernel_size=3, padding=1,bias=False))
        layers.append(nn.Hardtanh(min_val=-5.0, max_val=4.0))
        # layers.append(nn.LayerNorm(64))
        self.dncnn = nn.Sequential(*layers)
        self.minloss= 1000
#        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        self.loss_func = nn.MSELoss()
    def forward(self,x):
        out = self.dncnn(x)
        return out

class AE_custom(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(AE_custom, self).__init__()
        kernel_size = 3
        padding = 1
        features = 24
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            # layers.append(nn.Hardtanh(min_val=-10.0, max_val=10.0))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.Hardtanh(min_val=-10.0, max_val=10.0))
        self.dncnn = nn.Sequential(*layers)
        self.minloss = 1000
#        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
    def forward(self, x):
        out = self.dncnn(x)
        return out
class Block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, name = "basic_block"):
        super(Block, self).__init__()
        self.conv1 = conv5x5(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(inplanes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.htanh = nn.Hardtanh(min_val=-10.0, max_val=10.0)
        self.conv4 = conv1x1(2*planes+inplanes,planes)
        self.bn4 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.name=name

    def forward(self, x):
#         print("*********************\n*********{}***********\n*********************".format(self.name))
#         print("Input shape {}".format(x.shape))
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        
        out_3x3 = self.conv3(x)
        out_3x3 = self.bn3(out_3x3)
#         print(out.shape,out_3x3.shape,identity.shape)
        out = torch.cat((identity,out,out_3x3),dim=1)
        
#         out = self.htanh(out)
#         print("After stacking",out.shape)
        out = self.conv4(out)
        out = self.bn4(out)
        out = nn.Tanh(out)
#         print("Output shape",out.shape)
#         print("\n\n")
        return out


class DualOutput(nn.Module):
    def __init__(self, channels=[1,16,32,64,32,16,1]):
        super(DualOutput, self).__init__()
        layers= []
        in_channels = channels[0]
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=channels[1], kernel_size=1, padding=0,bias=False))
        for i in range(1,len(channels)-4):
            block = Block(channels[i],channels[i+1],name="basic_block_{}".format(i))
            layers.append(block)
        self.dncnn = nn.Sequential(*layers)
        denoiser_layers = []
        noise_layers=[]
        for i in range(len(channels)-4,len(channels)-1):
            block = nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=3, padding=1,bias=False)
            denoiser_layers.append(block)
            bn = nn.BatchNorm2d(channels[i+1])
            denoiser_layers.append(bn)
        denoiser_layers.append(nn.Hardtanh(min_val=-10.0, max_val=10.0))
        self.denoiser = nn.Sequential(*denoiser_layers)
        for i in range(len(channels)-4,len(channels)-1):
            block = nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=3, padding=1,bias=False)
            bn = nn.BatchNorm2d(channels[i+1])
            noise_layers.append(block)
            noise_layers.append(bn)
        noise_layers.append(nn.Hardtanh(min_val=-10.0, max_val=10.0))
        self.noiser = nn.Sequential(*noise_layers)
        self.minloss = 1000
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    def forward(self,x):
        out = self.dncnn(x)
        denoised = self.denoiser(out)
        noise = self.noiser(out)
        return denoised,noise