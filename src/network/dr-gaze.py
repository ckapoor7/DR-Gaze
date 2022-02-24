#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.funcitonal as F


"""
Dense block
"""

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=3, padding=3//2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


"""
Residual dense block
"""

class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseBlock(in_channels + growth_rate * i,
                                                 growth_rate) for i in range(num_layers)])

        # local fusion
        self.local_fusion = nn.Conv2d(in_channels=in_channels + growth_rate * num_layers,
                                      out_channels=growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.local_fusion(self.layers(x))


"""
RDB + Dense block
"""

class FinalNetwork(nn.Module):
    def __init__(self, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(FinalNetwork, self).__init__()
        self.feat = num_features
        self.growth = growth_rate
        self.blocks = num_blocks
        self.layers = num_layers

        # initial feature extraction
        self.feat1 = nn.Conv2d(in_channels=num_channels, out_channels=num_features,
                               kernel_size=3, padding=3//2)
        self.feat2 = nn.Conv2d(in_channels=num_features, out_channels=num_features,
                               kernel_size=3, padding=3//2)

        # residual dense blocks
        self.RDBs = nn.ModuleList([RDB(in_channels=self.feat, growth_rate=self.growth,
                                       num_layers=self.layers)])
        for _ in range(self.blocks - 1):
            self.RDBs.append(RDB(in_channels=self.growth, growth_rate=self.growth,
                                 num_layers=self.layers))

        # global fusion
        self.global_fusion = nn.Sequential(
            nn.Conv2d(in_channels=self.growth * self.blocks, out_channels=self.feat, kernel_size=1),
            nn.Conv2d(in_channels=self.feat, out_channels=self.feat, kernel_size=3, padding=3//2)
        )

        # final step to ensure that the final number of channels are 3
        self.d1 = nn.Conv2d(in_channels=self.feat, out_channels=self.feat//2,
                            kernel_size=3, padding=3//2)
        self.d2 = nn.Conv2d(in_channels=self.feat//2, out_channels=self.feat//4,
                            kernel_size=3, padding=3//2)
        self.d3 = nn.Conv2d(in_channels=self.feat//4, out_channels=3,
                            kernel_size=3, padding=3//2)

    def forward(self, x):
        num_samples = x.shape[0]
        feat1 = self.feat1(x)
        feat2 = self.feat2(feat1)

        x = feat2
        local_feature_vector = []
        for i in range(self.blocks):
            x = self.RDBs[i](x)
            local_feature_vector.append(x)

        # global residual connection
        x = self.global_fusion(torch.cat(local_feature_vector, 1)) + feat1

        x = self.d3(self.d2(self.d1(x)))

        return torch.reshape(x, (num_samples, -1))


"""
DR-Gaze
"""
class DRGaze(nn.Module):
    def __init__(self, num_channels=3, num_features=64, growth_rate=64, num_blocks=8, num_layers=8):
        super(DRGaze, self).__init__()

        self.c = num_channels
        self.f = num_features
        self.k = growth_rate
        self.b = num_blocks
        self.l = num_layers

        # left-eye branch
        self.left_eye = FinalNetwork(
            self.c,
            self.f,
            self.k,
            self.b,
            self.l
        )

        # feature branch
        self.dense1_1 = Linear(in_features=13, out_features=16)

        # fused branch
        self.fusion = nn.Sequential(
            nn.Linear(in_features=6496, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=2)
        )


    def forward(self, x_eye, x_face):
        # forward pass on left eye branch
        x_left_fwd = self.left_eye(x_eye)

        # fuse facial features and left eye image
        x_face = self.dense1_1(x_face) # transform to a vector of length 16
        x_face = x_face.reshape(-1, 16)
        x_stack = torch.cat((x_left_fwd, x_face), dim=1).to(device="cuda")

        # fusion branch
        x_final = self.fusion(x_stack)

        return x_final
