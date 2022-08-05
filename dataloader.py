#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 17:03:10 2022

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

#%%

if torch.cuda.is_available():
    print('CUDA is available. Working on GPU')
    DEVICE = torch.device('cuda')
else:
    print('CUDA is not available. Working on CPU')
    DEVICE = torch.device('cpu')



#%%

class getDataset(Dataset):
    def __init__(self, data_root, transform_img, transform_lb):
        
        
        self.data_root = data_root
        self.transform_img = transform_img
        self.transform_lb = transform_lb
        
        healthy = []
        healthy_c = []
        sick = []
        sick_c = []
        label_h =[]
        label_s = []
        
        
        folder_name=sorted(os.listdir(self.data_root))
            
        h = folder_name[0]
        h_c = folder_name[1]
        s = folder_name[2]
        s_c = folder_name[3]
        
        hh = sorted(os.listdir(self.data_root + '/' + h + '/'))
        hh_c = sorted(os.listdir(self.data_root + '/' + h_c + '/'))
           
        for he, he_canny in zip(hh,hh_c):
            hhh = self.data_root + '/' + h + '/' + he
            hhh_c = self.data_root + '/' + h_c + '/' + he_canny
            
            healthy.append(hhh)
            healthy_c.append(hhh_c)
            label_h.append([0])
            
        ss = sorted(os.listdir(self.data_root + '/' + s + '/'))
        
        ss_c = sorted(os.listdir(self.data_root + '/' + s_c + '/'))
        for si, si_canny in zip(ss,ss_c):    
            sss = self.data_root + '/' + s + '/' + si
            sss_c = self.data_root + '/' + s_c + '/' + si_canny
            
            sick.append(sss)
            sick_c.append(sss_c)
            label_s.append([1])
   
        self.images = healthy + sick
        self.images_canny = healthy_c + sick_c
        self.labels = label_h + label_s
        
    def __len__(self):
        return len(self.images)    
        
    def __getitem__(self, index):
       
        #print(self.images[index])
        img = Image.open(self.images[index])
        
       
        img = self.transform_img(img)
        
        if img.size()[0]==4:
            #print("1D image")
            img = img[0:3,:,:]
       
        
        if img.size()[0]==1:
            #print("1D image")
            img = torch.cat([img]*3)
        
        
        img_canny = self.transform_lb(Image.open(self.images_canny[index]))
       
        if img_canny.size()[0]==1:
            #print("1D image")
            img_canny = torch.cat([img_canny]*3)  
        if img_canny.size()[0]==4:
            #print("1D image")
            img_canny = img_canny[0:3,:,:]    
         
         
        lb = self.labels[index]
        lb = lb[0]
        #print(type(img))
       
        
        
        return (img, img_canny, lb)
    
  
#%%
    




path = '/home/nibaran/Downloads/DMR/final Project/data/'

transform = transforms.Compose(
    [transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])

transform_b = transforms.Compose(
    [transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])

batch = 4
nw = 0



train_dataset=getDataset(path + 'train/', transform, transform_b)
trainloader = DataLoader(train_dataset, batch_size=batch,shuffle=True, num_workers=nw)

val_dataset=getDataset(path + 'val/', transform, transform_b)
valloader = DataLoader(val_dataset, batch_size=1,shuffle=False, num_workers=nw)

test_dataset=getDataset(path + 'test/', transform, transform_b)
testloader = DataLoader(test_dataset, batch_size=1,shuffle=False, num_workers=nw)


classes = ('Healthy', 'Sick')
