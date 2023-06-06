#!/usr/bin/env python
# coding: utf-8

# 本程序用作分类器，将纸质输入作为训练集，电子签名输入（暂时只有平板）作为验证集进行计算。注意分类编号需要从0开始，也就是受试者编号减一

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

choice = torch.cuda.is_available() and not(isIpy())  # 模型使用cpu或者cuda计算，jupyter用cuda可能会蓝屏，建议导出然后改成cuda
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
        # 输出长度person_num的向量作为分类概率，后续loss函数中还会经过softmax计算

        self.resnet.apply(self.init_weights)    # 初始化权重
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, val=0) 

    def forward(self, input1):       # 计算
        output = self.resnet(input1)
        return output


# In[18]:


class SignDataset(Dataset):
    def __init__(self, person_num, type):
        self.item_index = list(range(1, person_num + 1))
        random.shuffle(self.item_index)     # 随机化输入的人顺序
        self.type = type
        self.person_num = person_num
    
    def __getitem__(self, item):
        imgs = []
        person = self.item_index[item]
        for i in range(1, 11):
            img = cv2.imread(f'prenn/{person}/{10*self.type + i}.png', 0) / 255   # 读取二值图像并转为0-1矩阵
            img = img.astype(np.float32)
            imgs.append((img, person))  # 图像和分类标签
        return imgs
    
    def __len__(self):
        return self.person_num      # 数据集大小


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
    return (preds == yb).float().mean()     # 如果概率最高的分类正好为目标分类，则为1，否则为0，然后计算平均值就是正确率


# In[22]:


bs = 31     # batch size每批训练量大小
epochs = 65536    # 最大训练轮数
person_num = 93 # 受试者人数
sign_num = 10   # 签名数量
train_logfile = './cnn.log'

train_ds = SignDataset(person_num, 0)
train_dl = DataLoader(train_ds, batch_size = bs)

valid1_ds = SignDataset(person_num, 1)
valid1_dl = DataLoader(valid1_ds, batch_size = bs * 2)

valid2_ds = SignDataset(person_num, 2)
valid2_dl = DataLoader(valid2_ds, batch_size = bs * 2)

valid3_ds = SignDataset(person_num, 3)
valid3_dl = DataLoader(valid3_ds, batch_size = bs * 2)

model, opt = get_model(person_num)

with open(train_logfile, 'w') as f:
    f.write(time.asctime(time.localtime(time.time()))+'\n')

for epoch in range(epochs):
    model.train()   # 训练模式
    for imgs in train_dl:
        for sign in range(sign_num - 1):
            pred = model(imgs[sign][0].view(-1, 1, 224, 224).to(dev))
            loss = F.cross_entropy(pred, (imgs[sign][1] - 1).to(dev))   # 计算交叉熵
            loss.backward(retain_graph=True)    # 保持梯度图并累计梯度
        pred = model(imgs[sign_num - 1][0].view(-1, 1, 224, 224).to(dev))
        loss = F.cross_entropy(pred, (imgs[9][1] - 1).to(dev))
        loss.backward() # 最后一个人不需要保持
            
        opt.step()  # 梯度反向传播
        opt.zero_grad()     # 清空梯度
    
    model.eval()    # 计算模式
    sum_loss = 0
    train_acc = 0
    sum_valid1_loss = 0
    valid1_acc = 0
    sum_valid2_loss = 0
    valid2_acc = 0
    sum_valid3_loss = 0
    valid3_acc = 0
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
        
        for imgs in valid2_dl:
            for sign in range(sign_num):
                pred = model(imgs[sign][0].view(-1, 1, 224, 224).to(dev))
                sum_valid2_loss += F.cross_entropy(pred, (imgs[sign][1] - 1).to(dev))
                valid2_acc += accuracy(pred, (imgs[sign][1] - 1).to(dev))
    
        for imgs in valid3_dl:
            for sign in range(sign_num):
                pred = model(imgs[sign][0].view(-1, 1, 224, 224).to(dev))
                sum_valid3_loss += F.cross_entropy(pred, (imgs[sign][1] - 1).to(dev))
                valid3_acc += accuracy(pred, (imgs[sign][1] - 1).to(dev))
                
    print(f'Epoch {epoch}')
    print(f"Learn {opt.state_dict()['param_groups'][0]['lr']}")
    print(f'Train: Loss {sum_loss/(len(train_dl) * sign_num)}, Acc {train_acc/(len(train_dl) * sign_num)}')
    print(f'Valid1: Loss {sum_valid1_loss/(len(valid1_dl) * sign_num)}, Acc {valid1_acc/(len(valid1_dl) * sign_num)}')
    print(f'Valid2: Loss {sum_valid2_loss/(len(valid2_dl) * sign_num)}, Acc {valid2_acc/(len(valid2_dl) * sign_num)}')
    print(f'Valid3: Loss {sum_valid3_loss/(len(valid3_dl) * sign_num)}, Acc {valid3_acc/(len(valid3_dl) * sign_num)}')
    with open(train_logfile, 'a') as f:
        f.write(f"{sum_loss/(len(train_dl) * sign_num)}, {train_acc/(len(train_dl) * sign_num)}, {sum_valid1_loss/(len(valid1_dl) * sign_num)},{valid1_acc/(len(valid1_dl) * sign_num)}, {sum_valid2_loss/(len(valid2_dl) * sign_num)}, {valid2_acc/(len(valid2_dl) * sign_num)}, {sum_valid3_loss/(len(valid3_dl) * sign_num)}, {valid3_acc/(len(valid3_dl) * sign_num)}\n")
    torch.save(model, f'./models/cnn_{epoch}.mod')


# In[ ]:





# In[ ]:




