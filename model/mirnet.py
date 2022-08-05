#!/usr/bin/env python3

import numpy as np
import torch
import os
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image

class getDataset(Dataset):
    def __init__(self, data_root):
        
        
        self.data_root = data_root
        self.images = []
        self.images_canny = []
        self.healthy = []
        self.healthy_c = []
        self.sick = []
        self.sick_c = []
        self.label = []
        self.label_h =[]
        self.label_s = []
        for filename in os.listdir(self.data_root):
            data = np.load(data_root+'/'+filename)
            h = data["healthy"]
            h_c = data["healthy_canny"]
            s = data["sick"]
            s_c = data["sick_canny"]
            
            for healthy, healthy_canny in zip(h,h_c):
                hh = Image.fromarray(healthy)
                self.healthy.append(hh)
                hh_c = Image.fromarray(healthy_canny)
                self.healthy_c.append(hh_c)
                self.label_h.append([0])
            for sick, sick_canny in zip(s,s_c):    
                ss = Image.fromarray(sick)
                self.sick.append(ss)
                ss_c = Image.fromarray(sick_canny)
                self.sick_c.append(ss_c)
                self.label_s.append([1])
       
        self.images = self.healthy + self.sick
        self.images_canny = self.healthy_c + self.sick_c
        self.labels = self.labels_h + self.labels_s
         
      
            
        def __getitem__(self, index):
            image = self.image_transform(Image.open(self.images[index]))
            if image.size()[0]==1:
                #print("1D image")
                image = torch.cat([image]*3)
            image_canny = self.image_transform(Image.open(self.images_canny[index]))
            if image_canny.size()[0]==1:
                #print("1D image")
                image_canny = torch.cat([image_canny]*3)    
                
            label = self.labels[index]
            
            
            return (image, image_canny, label)
        
        def __len__(self):
            return (len(self.images))
    






dir = '/home/nibaran/Downloads/DMR/final Project/FINAL/'
train_dataset=getDataset(dir)
dataloader = DataLoader(train_dataset, batch_size=1,
                            shuffle=True, num_workers=2)





