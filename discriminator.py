"""
    Patch based discriminator as in https://arxiv.org/pdf/1611.07004
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        layers = [nn.Conv2d(args.image_channels, args.discriminator_layers[0], 4, 2, 1), nn.LeakyReLU(0.2)]

        for i in range(1, len(args.discriminator_layers)):
            layers.append(nn.Conv2d(args.discriminator_layers[i-1], args.discriminator_layers[i], 4, 2, 1))
            layers.append(nn.BatchNorm2d(args.discriminator_layers[i]))
            layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.Conv2d(args.discriminator_layers[-1], 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)