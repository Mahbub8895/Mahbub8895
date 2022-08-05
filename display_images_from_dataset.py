#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 17:11:26 2022

@author: nibaran
"""

'''
import matplotlib.pyplot as plt
import numpy as np
import torchvision

# functions to show an image
batch = 4
train_dataset=getDataset(path + 'train/', transform, transform_b)
trainloader = DataLoader(train_dataset, batch_size=batch,shuffle=True, num_workers=nw)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)

for data in trainloader:
    images, images_canny, labels = data
    #print(type(images))
    #print(images.shape)
    #print(type(images_canny))
    #print(images_canny.shape)
    #print(len(labels))
    # show images
    imshow(torchvision.utils.make_grid(images))
    imshow(torchvision.utils.make_grid(images_canny))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch)))
    break

'''
