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
from network import *
from dataloader import *
from tqdm import tqdm
from torch.nn.functional import normalize
from torchsummary import summary


loss_fn = nn.L1Loss()

def get_args():
   parser = argparse.ArgumentParser(description='PyTorch Gaze prediction')
   parser.add_argument('--weight_dir', type=str, help='directory with saved weights')
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
   parser.add_argument('--device', type=str, default="cuda")

   args = parser.parse_args()
   return args


def validate(model, train_loader, val_loader, test_loader):
    for name, loader in [("train", train_loader), ("val", val_loader),('test',test_loader)]:

        loss_n = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs[0] = imgs[0].to(device=args.device)
                imgs[1] = imgs[1].to(device=args.device)
                labels = labels.to(device=args.device)
                outputs = model(imgs[0], imgs[1])

                loss = loss_fn(outputs, labels)

                loss_n += loss.item()


        print("Loss {}: {:.5f}".format(name , loss_n / len(loader)))


def main():
    args = get_args()

    train_loader = Dgaze_DataSet()
    val_loader = Dgaze_ValSet()
    test_loader = Dgaze_TestSet()
    model = DRGaze(num_channels=args.num_channels, num_features=args.num_features, growth_rate=args.growth_rate,
                   num_blocks=args.num_blocks, num_layers=args.num_layers)
    model.load_state_dict(torch.load(args.weight_dir))

    validate(model=model,
             train_loader=train_loader,
             val_loader=val_loader,
             test_loader=test_loader)


if __name__ == '__main__':
    main()
