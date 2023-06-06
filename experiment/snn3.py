#!/usr/bin/env python
# coding: utf-8

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


# In[3]:


def isIpy():
    '''
    Judge whether it is using jupyter
    '''
    try:
        __IPYTHON__
        return True
    except NameError: return False

choice = torch.cuda.is_available() and not(isIpy())
if choice:
    print('Run on CUDA!')
    dev = torch.device("cuda")
else: 
    print('Run on CPUs')
    dev = torch.device("cpu")


# In[4]:


class Sign_SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights=None)
        
        # resnet18 reads (3,x,x) where 3 is RGB channels
        # (1,x,x) where 1 is a gray-scale channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        
        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )
        
        self.sigmoid = nn.Sigmoid()
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        return output1, output2

# In[5]:


def get_train_valid(person_num, sep):
    train_sig = [[]]
    valid_sig = [[]]
    
    shuf_sig = list(range(1, 11))
    for i in range(1, person_num + 1):
        random.shuffle(shuf_sig)
        shuf_tmp = shuf_sig.copy()
        train_sig.append(shuf_tmp[0:sep])
        valid_sig.append(shuf_tmp[sep:])
    return train_sig, valid_sig


# In[5]:


person_num = 93
sep = 7

train_index, valid_index = get_train_valid(person_num, sep)


# In[6]:


class SignDataset(Dataset):
    def __init__(self, person_num, dtype):
        self.type = dtype
        self.person_num = person_num
        
        self.person_pair = []
        for i in range(person_num):
            self.person_pair.append((i + 1, i + 1))

        diff_person = []
        for i in range(person_num):
            for j in range(i+1, person_num):
                diff_person.append((i + 1, j + 1))
        random.shuffle(diff_person)
        for i in range(person_num):
            self.person_pair.append(diff_person[i])
        random.shuffle(self.person_pair)
    
    def __getitem__(self, item):
        imgs = []
        persons = self.person_pair[item]
        
        if persons[0] == persons[1]:
            target = torch.tensor(0, dtype=torch.float)
        else:
            target = torch.tensor(1, dtype=torch.float)
        
        
        if self.type == 0:
            index_set = train_index
        else:
            index_set = valid_index
        for i in index_set[persons[0]]:    # Change skel to bin for binary only images
            img1 = cv2.imread(f'skel/{persons[0]}/{i}.png', 0) / 255
            img1 = img1.astype(np.float32)
            img2 = cv2.imread(f'skel/{persons[1]}/{20 + i}.png', 0) / 255
            img2 = img2.astype(np.float32)
            imgs.append((img1, img2, target))
        return imgs
    
    def __len__(self):
        return self.person_num


# In[69]:


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin=margin
  
    def forward(self, output1, output2, label):
        euc_dist = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label)*torch.pow(euc_dist,2) + 
                           (label)*torch.pow(torch.clamp(self.margin - euc_dist, min=0.0), 2))
        return loss_contrastive


# In[70]:


def get_model():
    model = Sign_SNN().to(dev)
    opt = optim.Adam(model.parameters(),lr=0.0005)
    #optim.lr_scheduler.StepLR(opt, 4, gamma=0.95, last_epoch=-1)
    return model, opt


# In[1]:


def accuracy(out1, out2, yb):
    out = F.pairwise_distance(out1, out2)
    zero = torch.zeros_like(out).to(dev)
    one = torch.ones_like(out).to(dev)
    out = torch.where(out > 0.9, one, out)     # dis > 0.9 fake
    out = torch.where(out <= 0.9, zero, out)   # dis <= 0.9 true
    ans = 1 - torch.abs(out - yb)
    return (ans.float().mean())


# In[7]:


bs = 31

model, opt = get_model()

train_ds = SignDataset(person_num, 0)
train_dl = DataLoader(train_ds, batch_size = bs)

valid_ds = SignDataset(person_num, 1)
valid_dl = DataLoader(valid_ds, batch_size = bs * 2)


# In[ ]:


epochs = 65536

train_logfile = './snn3.log'

#loss_func = nn.BCELoss()
loss_func = ContrastiveLoss()

with open(train_logfile, 'w') as f:
    f.write(time.asctime(time.localtime(time.time()))+'\n')

for epoch in range(epochs):
    model.train()
    sign_num = 7
    for imgs in train_dl:
        for sign in range(sign_num - 1):
            pred1, pred2 = model(imgs[sign][0].view(-1, 1, 224, 224).to(dev),
                         imgs[sign][1].view(-1, 1, 224, 224).to(dev))
            loss = loss_func(pred1, pred2, imgs[sign][2].to(dev))
            loss.backward(retain_graph=True)
        pred1, pred2 = model(imgs[sign_num - 1][0].view(-1, 1, 224, 224).to(dev),
                     imgs[sign_num - 1][1].view(-1, 1, 224, 224).to(dev))
        loss = loss_func(pred1, pred2, imgs[sign_num - 1][2].to(dev))
        loss.backward()
        
        opt.step()
        opt.zero_grad()
        
    model.eval()
    sum_loss = 0
    train_acc = 0
    sum_valid_loss = 0
    valid_acc = 0
    with torch.no_grad():
        print(f'Epoch {epoch}')

        sign_num = 7
        for imgs in train_dl:
            for sign in range(sign_num):
                pred1, pred2 = model(imgs[sign][0].view(-1, 1, 224, 224).to(dev),
                             imgs[sign][1].view(-1, 1, 224, 224).to(dev))
                sum_loss += loss_func(pred1, pred2, imgs[sign][2].to(dev))
                train_acc += accuracy(pred1, pred2, imgs[sign][2].to(dev))
        print(f'Train: Loss {sum_loss / (len(train_dl) * sign_num)}, Acc {train_acc / (len(train_dl) * sign_num)}')
        with open(train_logfile, 'a') as f:
            f.write(f"{sum_loss / (len(train_dl) * sign_num)}, {train_acc / (len(train_dl) * sign_num)}, ")
        
        sign_num = 3
        for imgs in valid_dl:
            for sign in range(sign_num):
                pred1, pred2 = model(imgs[sign][0].view(-1, 1, 224, 224).to(dev),
                             imgs[sign][1].view(-1, 1, 224, 224).to(dev))
                sum_valid_loss += loss_func(pred1, pred2, imgs[sign][2].to(dev))
                valid_acc += accuracy(pred1, pred2, imgs[sign][2].to(dev))
    print(f'Valid: Loss {sum_valid_loss / (len(valid_dl) * sign_num)}, Acc {valid_acc / (len(valid_dl) * sign_num)}')
    with open(train_logfile, 'a') as f:
        f.write(f"{sum_valid_loss / (len(valid_dl) * sign_num)}, {valid_acc / (len(valid_dl) * sign_num)}\n")
    torch.save(model, f'./models/snn3_{epoch}.mod')

    


# In[ ]:




