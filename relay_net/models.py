#!/usr/bin/env python3

import torch
import torch.nn as nn
from tqdm import tqdm


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(
                  in_channels, out_channels, kernel_size=(7, 3),
                  padding=(3, 1),
                  bias=False
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(2, return_indices=True)

    def forward(self, x):
        x = self.conv(x)
        skip_connection = x.clone()
        x = self.batchnorm(x)
        x = self.activation(x)
        x, idx = self.maxpool(x)
        return x, idx, skip_connection


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Conv2d(
                  2 * in_channels, out_channels, kernel_size=(7, 3),
                  padding=(3, 1),
                  bias=False
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.unmaxpool = nn.MaxUnpool2d(2)
        self.activation = nn.ReLU(True)

    def forward(self, x, idx, skip_connection):
        x = self.unmaxpool(x, idx)
        x = torch.cat((skip_connection, x), dim=1)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        return x


class RelayNet(nn.Module):
    def __init__(self, num_classes=9):
        super(RelayNet, self).__init__()
        self.encoder = nn.ModuleList(
                  [
                      EncoderBlock(1, 64),
                      EncoderBlock(64, 64),
                      EncoderBlock(64, 64)
                  ]
        )
        self.bottleneck = nn.Sequential(
                  nn.Conv2d(
                            64, 64, kernel_size=(7, 3), padding=(3, 1),
                            bias=False
                  ),
                  nn.BatchNorm2d(64),
                  nn.ReLU(True)
        )
        self.decoder = nn.ModuleList(
                  [
                      DecoderBlock(64, 64),
                      DecoderBlock(64, 64),
                      DecoderBlock(64, 64),
                  ]
        )
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        maxpool_idxs = []
        for module in self.encoder:
            x, maxpool_idx, skip_connection = module(x)
            skip_connections.append(skip_connection)
            maxpool_idxs.append(maxpool_idx)
        x = self.bottleneck(x)
        maxpool_idxs = maxpool_idxs[::-1]
        skip_connections = skip_connections[::-1]
        for module, maxpool_idx, skip_connection in zip(
                  self.decoder, maxpool_idxs, skip_connections
        ):
            x = module(x, maxpool_idx, skip_connection)

        x = self.classifier(x)
        return x