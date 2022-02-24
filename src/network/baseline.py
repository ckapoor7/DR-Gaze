#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
I-DGAZE
"""

class IDGAZE(nn.Module):
  def __init__(self):
    super(IDGAZE, self).__init__()

    # left eye branch
    self.left_eye = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=20, kernel_size=(3, 3), stride=1, padding=0),
        nn.BatchNorm2d(num_features=20),
        nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
        nn.Dropout(p=0.2),
        nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(3, 3), stride=1, padding=0),
        nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
    )

    # feature branch
    self.dense1_1 = Linear(in_features=13, out_features=16)

    # fused branch
    self.fusion = nn.Sequential(
            nn.Linear(in_features=4566, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=2)
        )

  def forward(self, x_eye, x_face):

    # forward pass on left eye branch
    x_left_fwd = self.left_eye(x_eye)

    x_left_flat = torch.flatten(x_left_fwd) # flatten left eye to 4550

    # fuse facial features and left eye image
    x_facial_flat = self.dense1_1(x_face) # transform to a vector of length 16
    x_facial_flat = torch.flatten(x_facial_flat)

    x_stack = torch.cat((x_left_flat, x_facial_flat), dim=0)
    x_stack = x_stack[0:4566] # prevent some weird double stacking

    # fusion branch
    x_final = self.fusion(x_stack)

    return x_final
