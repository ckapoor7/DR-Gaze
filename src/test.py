#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import datetime
from tqdm import tqdm
from torch.nn.functional import normalize
from torchsummary import summary


loss_fn = nn.L1Loss()


def validate(model, train_loader, val_loader, test_loader):
    for name, loader in [("train", train_loader), ("val", val_loader),('test',test_loader)]:


        loss_n = 0
        with torch.no_grad():  # <1>
            for imgs, labels in loader:
                imgs[0] = imgs[0].to(device=device)
                imgs[1] = imgs[1].to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs[0], imgs[1])

                loss = loss_fn(outputs, labels)

                loss_n += loss.item()


        print("Loss {}: {:.5f}".format(name , loss_n / len(loader)))

validate(model, train_loader, val_loader, test_loader)
