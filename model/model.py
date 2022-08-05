#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 17:08:54 2022

@author: nibaran
"""
import numpy as np
import torch
import os
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size =2, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(8)
    self.pool1 = nn.MaxPool2d(2,2)
    
    self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size =2, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(16)
    self.dropout1=nn.Dropout(0.40)
    self.pool2 = nn.MaxPool2d(2,2)

    self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size =2, stride=1, padding=1)
    self.bn3 = nn.BatchNorm2d(32)
    self.dropout2 = nn.Dropout(0.40)
    self.pool3 = nn.MaxPool2d(2,2)

    self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size =2, stride=1, padding=1)
    self.bn4= nn.BatchNorm2d(64)
    self.dropout3 =nn.Dropout(0.40)

    self.conv5= nn.Conv2d(in_channels=64, out_channels=128, kernel_size =2, stride=1, padding=1)
    self.bn5= nn.BatchNorm2d(128)
    self.dropout4=nn.Dropout(0.40)
    

    self.fc1=nn.Linear(128*30*30, 120)
    self.fc2=nn.Linear(120, 84)
    self.fc3=nn.Linear(84, 2)


  def forward(self, input):
    #input = torch.cat((input,input_canny),dim=1)  
    output =F.relu(self.bn1(self.conv1(input)))
    #print(output.shape)
    output = self.pool1(output)
    #print(output.shape)
    output =F.relu(self.bn2(self.conv2(output)))
    output =self.dropout1(output)
    #print(output.shape)
    output = self.pool2(output)
    #print(output.shape)
    output =F.relu(self.bn3(self.conv3(output)))
    #print(output.shape)
    output =self.dropout2(output)
    #print(output.shape)
    output = self.pool3(output)
    #print(output.shape)
    output =F.relu(self.bn4(self.conv4(output)))
    output =self.dropout3(output)
    #print(output.shape)
    output =F.relu(self.bn5(self.conv5(output)))
    output =self.dropout4(output)
    #print(output.shape)
    #output =output.view(-1, 128*60*60)
    output = torch.flatten(output, 1)# flatten all dimensions except batch
    ##print(output.shape)
    
    output =F.relu(self.fc1(output))
    #rint(output.shape)
    output =F.relu(self.fc2(output))
    #print(output.shape)
    output =self.fc3(output)
    #print(output.shape)
    return output

model = Net()    
