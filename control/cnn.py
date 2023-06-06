#!/usr/bin/env python
# coding: utf-8

"""
This program is used as a classifier, with paper-based inputs as the training set and electronic signature inputs (currently only from tablets) as the validation set for calculations. 
Please note that the classification numbering should start from 0, which means subtracting one from the subject's identifier.
"""

# In[1]:


import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np
import time

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from torch import optim


# In[2]:


def isIpy():
    try:
        __IPYTHON__
        return True
    except NameError: return False

choice = torch.cuda.is_available() and not(isIpy())  # The model can be trained using CPU or CUDA for computation. If using CUDA in Jupyter Notebook causes a blue screen error, it is recommended to export the notebook and modify the code to utilize CUDA outside of the notebook.
print(choice)
if choice:
    dev = torch.device("cuda")
else: 
    dev = torch.device("cpu")


# In[3]:


class Sign_CNN(nn.Module):
    def __init__(self, person_num):
        super().__init__()

        self.resnet = torchvision.models.resnet18(weights=None)
        # as resnet18 reads (3,x,x) where 3 is RGB channels
        # whereas dataset has (1,x,x) where 1 is a gray-scale channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.fc_in_features = self.resnet.fc.in_features
        
        # add linear layers to compare images
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, person_num, bias = True)
        # Output a vector of length person_num as the classification probabilities. The subsequent loss function will involve applying softmax calculation.

        self.resnet.apply(self.init_weights)    # Initialize weights
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, val=0) 

    def forward(self, input1):
        output = self.resnet(input1)
        return output


# In[18]:


class SignDataset(Dataset):
    def __init__(self, person_num, type):
        self.item_index = list(range(1, person_num + 1))
        random.shuffle(self.item_index)     # Randomizing the order of input people.
        self.type = type
        self.person_num = person_num
    
    def __getitem__(self, item):
        imgs = []
        person = self.item_index[item]
        for i in range(1, 11):   # Change skel to bin for binary only images
            img = cv2.imread(f'skel/{person}/{10*self.type + i}.png', 0) / 255   # Read the binary image and convert it into a 0-1 matrix.
            img = img.astype(np.float32)
            imgs.append((img, person))  # The image and the corresponding classification label.
        return imgs
    
    def __len__(self):
        return self.person_num      # The size of the dataset.


# In[20]:


def get_model(person_num):
    model = Sign_CNN(person_num).to(dev)
    #opt = optim.RMSprop(model.parameters(), lr=0.1, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
    opt = optim.Adam(model.parameters(),lr=0.0005)
    #optim.lr_scheduler.StepLR(opt, 4, gamma=0.95, last_epoch=-1)
    return model, opt


# In[21]:


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()     # If the highest probability classification matches the target class, assign a value of 1; otherwise, assign a value of 0. Then calculate the average, which represents the accuracy.


# In[22]:


bs = 11     # batch size
epochs = 65536    # max epoch
person_num = 11 # The number of participants (subjects) in the study
sign_num = 10   # The number of signatures
train_logfile = './cnn.log'

train_ds = SignDataset(person_num, 0)
train_dl = DataLoader(train_ds, batch_size = bs)

valid1_ds = SignDataset(person_num, 1)
valid1_dl = DataLoader(valid1_ds, batch_size = bs)

model, opt = get_model(person_num)

with open(train_logfile, 'w') as f:
    f.write(time.asctime(time.localtime(time.time()))+'\n')

for epoch in range(epochs):
    model.train()   # Training mode
    for imgs in train_dl:
        for sign in range(sign_num - 1):
            pred = model(imgs[sign][0].view(-1, 1, 224, 224).to(dev))
            loss = F.cross_entropy(pred, (imgs[sign][1] - 1).to(dev))   # Cross Entropy
            loss.backward(retain_graph=True)    # Preserve the gradient map and accumulate gradients.
        pred = model(imgs[sign_num - 1][0].view(-1, 1, 224, 224).to(dev))
        loss = F.cross_entropy(pred, (imgs[9][1] - 1).to(dev))
        loss.backward() # The last person does not need to preserve.
            
        opt.step()
        opt.zero_grad()     # Clear gradient.
    
    model.eval()    # Computational mode
    sum_loss = 0
    train_acc = 0
    sum_valid1_loss = 0
    valid1_acc = 0
    with torch.no_grad():
        for imgs in train_dl:
            for sign in range(sign_num):
                pred = model(imgs[sign][0].view(-1, 1, 224, 224).to(dev))
                sum_loss += F.cross_entropy(pred, (imgs[sign][1] - 1).to(dev))
                train_acc += accuracy(pred, (imgs[sign][1] - 1).to(dev))
        
        for imgs in valid1_dl:
            for sign in range(sign_num):
                pred = model(imgs[sign][0].view(-1, 1, 224, 224).to(dev))
                sum_valid1_loss += F.cross_entropy(pred, (imgs[sign][1] - 1).to(dev))
                valid1_acc += accuracy(pred, (imgs[sign][1] - 1).to(dev))
                
    print(f'Epoch {epoch}')
    print(f"Learn {opt.state_dict()['param_groups'][0]['lr']}")
    print(f'Train: Loss {sum_loss/(len(train_dl) * sign_num)}, Acc {train_acc/(len(train_dl) * sign_num)}')
    print(f'Valid1: Loss {sum_valid1_loss/(len(valid1_dl) * sign_num)}, Acc {valid1_acc/(len(valid1_dl) * sign_num)}')
    with open(train_logfile, 'a') as f:
        f.write(f"{sum_loss/(len(train_dl) * sign_num)}, {train_acc/(len(train_dl) * sign_num)}, {sum_valid1_loss/(len(valid1_dl) * sign_num)},{valid1_acc/(len(valid1_dl) * sign_num)}\n")
    torch.save(model, f'./models/cnn_{epoch}.mod')