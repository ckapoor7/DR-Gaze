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
import argparse
from dataloader import Dgaze_DataSet
from tqdm import tqdm
from torch.nn.functional import normalize
from torchsummary import summary
from network import *


# command line arguments
def get_args():
   parser = argparse.ArgumentParser(description='PyTorch Gaze prediction')
   parser.add_argument('--n_epochs', type=int, default=60,
                       help='number of training epochs')
   parser.add_argument('--lr', type=int, default=1e-5,
                       help='learning rate')
   parser.add_argument('--eps', type=int, default=1e-5)
   parser.add_argument('--gamma', type=int, default=0.1, help='multi-step LR weight decay')
   parser.add_argument('--dest_dir', type=str, help='destination directory to save weights')
   parser.add_argument('--num_channels', type=int, default=3,
                       help='number of input channels')
   parser.add_argument('--num_features', type=int, default=32,
                       help='channel size')
   parser.add_argument('--growth_rate', type=int, default=32,
                       help='growth rate')
   parser.add_argument('--num_blocks', type=int, default=4,
                       help='number of residual dense blocks')
   parser.add_argument('--num_layers', type=int, default=4,
                       help='number of residual layers')

   args = parser.parse_args()
   return args


# training loop
def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, dest_dir):
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
        torch.save(model.state_dict(), dest_dir)


def main():
    args = get_args()

    model = DRGaze(num_channels=args.num_channels, num_features=args.num_features, growth_rate=args.growth_rate,
                   num_blocks=args.num_blocks, num_layers=args.num_layers)
    optimizer = optim.Adam(model.paramters(), lr=args.lr, eps=args.eps)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [40, 55], gamma=args.gamma)
    loss_fn = nn.L1Loss()
    train_loader = Dgaze_DataSet()

    train(n_epochs=args.n_epochs,
          optimizer=optimizer,
          model=model,
          loss_fn=loss_fn,
          train_loader=train_loader,
          scheduler=scheduler,
          dest_dir=args.dest_dir)


if __name__ == '__main__':
    main()
