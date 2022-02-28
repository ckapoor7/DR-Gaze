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


optimizer = optim.Adam(model.parameters(), lr=lr, eps=eps)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [40,55], gamma=gamma)
loss_fn = nn.L1Loss()


def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (image_list, labels) in loop:
            image_list[0]= image_list[0].to(device=device)
            image_list[1]= image_list[1].to(device=device)

            outputs = model(image_list[0], image_list[1])
            labels = labels.to(device=device)

            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # progress bar updates
            loop.set_description(f"Epoch [{epoch}/{n_epochs}]")
            loop.set_postfix(loss = loss.item())
            loss_train += loss.item()
        scheduler.step()
        torch.save(model.state_dict(), '/content/drive/MyDrive/DGAZE/saved_weights/final_network.pth')



train(n_epochs=n_epochs,
      optimizer=optimizer,
      model=model,
      loss_fn=loss_fn,
      train_loader=train_loader,
      scheduler=scheduler)
