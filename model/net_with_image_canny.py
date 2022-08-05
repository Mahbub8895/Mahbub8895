#!/usr/bin/env python3

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



#%%

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








#%%


class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.conv1 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size =2, stride=1, padding=1)
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


  def forward(self, input, input_canny):
    input = torch.cat((input,input_canny),dim=1)  
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


#%%

import tqdm
import copy
epoch =10
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.33)
#inputs, labels = next(iter(trainloader))
train_loss_array = []
train_acc_array = []
val_loss_array = []
val_acc_array = []
lowest_val_loss = 9999999999999999
best_model = None
train_loss_line=[]
val_loss_line=[]
x=[]
loss_set=[]
for epoch in range(epoch):

    print('Epoch: {} | Learning rate: {}'.format(epoch + 1, scheduler.get_lr()))
        

    for phase in ['train', 'val']:

        train_epoch_loss = 0
        train_epoch_correct_items = 0
        train_epoch_items = 0
        val_epoch_loss = 0
        val_epoch_correct_items = 0
        val_epoch_items = 0

        if phase == 'train':
            model.train()
            with torch.enable_grad():
                for data in trainloader:
                    inputs, inputs_canny, labels = data 
                    inputs = inputs.to(DEVICE)
                    inputs_canny =  inputs_canny.to(DEVICE)
                    labels = labels.to(DEVICE)

                    optimizer.zero_grad()
                    outputs = model(inputs,inputs_canny)
                    
                    loss = loss_function(outputs, labels)
                    preds = outputs.argmax(dim=1)
                    correct_items = (preds == labels).float().sum()
                        
                    loss.backward()
                    optimizer.step()

                    train_epoch_loss +=loss.item()
                    train_epoch_correct_items += correct_items.item()
                    train_epoch_items += len(labels)
            train_loss =train_epoch_loss/4.0
            train_loss_array.append(train_epoch_loss/train_epoch_items)
            train_acc_array.append(train_epoch_correct_items/train_epoch_items)
            print('train loss:{}   train_accuracy:{}'.format((train_epoch_loss/train_epoch_items),(train_epoch_correct_items/train_epoch_items)))
            scheduler.step()

        elif phase == 'val':
            model.eval()
            with torch.no_grad():
                for data in valloader:
                    inputs, inputs_canny, labels = data 
                    inputs = inputs.to(DEVICE)
                    inputs_canny =  inputs_canny.to(DEVICE)
                    labels = labels.to(DEVICE)


                    outputs = model(inputs,inputs_canny)
                    
                    val_loss = loss_function(outputs, labels)
                    
                    preds = outputs.argmax(dim=1)
                    correct_items = (preds == labels).float().sum()

                    val_epoch_loss += val_loss.item()
                    val_epoch_correct_items += correct_items.item()
                    val_epoch_items += len(labels)

            val_loss_array.append(val_epoch_loss / val_epoch_items)
            val_acc_array.append(val_epoch_correct_items / val_epoch_items)
            print( ' val_loss :{}  val_accuracy:{}'.format((val_epoch_loss/val_epoch_items),(val_epoch_correct_items / val_epoch_items)))
            if (val_epoch_loss / val_epoch_items) < lowest_val_loss:
                lowest_val_loss = val_epoch_loss / val_epoch_items
                PATH = './net_with_image_canny.pth'
                torch.save(model.state_dict(), PATH)
                #torch.save(model.state_dict(), '{}_weights.pth'.format(model))
                #best_model = copy.deepcopy(model)
                print("\t| New lowest val loss for: {}".format(lowest_val_loss))
            #if(val_loss<lowest_val_loss):
                #net_min=model.state_dict()
                #otp_min=optimizer.state_dict()
                #PATH = './net_with_image_canny.pth'
                #torch.save(model.state_dict(), PATH)
                #print('model saved--------')
                #lowest_val_loss = val_loss

            loss_set.append([train_loss_array,val_loss_array])




import matplotlib.pyplot as plt




for i in range(len(loss_set)):
  train_loss_line.append(loss_set[i][0])
  val_loss_line.append(loss_set[i][1])
  x.append(i)
plt.plot(x, train_loss_line, label = "training loss")
plt.plot(x, val_loss_line , label = "validation loss")
plt.legend()
plt.savefig("img_canny.pdf")
plt.show()



#%%


#TESTING
model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()
# look at how the network performs on the whole dataset.
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
model.eval()
with torch.no_grad():
    for data in testloader:
        inputs, inputs_canny, labels = data 
        inputs = inputs.to(DEVICE)
        inputs_canny =  inputs_canny.to(DEVICE)
        labels = labels.to(DEVICE)
        # calculate outputs by running images through the network
        output = model(inputs,inputs_canny)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 503 test images: %d %%' % (
    100 * correct / total))







   



