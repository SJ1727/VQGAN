import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import DownSampleBlock, ResidualBlock, NonLocalBlock, Swish

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layers = [nn.Conv2d(args.image_channels, args.encoder_channels[0], 3, 1, 1)]

        for i in range(len(args.encoder_channels) - 1):
            layers.append(ResidualBlock(args.encoder_channels[i], args.encoder_channels[i+1]))
            layers.append(DownSampleBlock(args.encoder_channels[i+1]))

        layers.append(ResidualBlock(args.encoder_channels[-1], args.encoder_channels[-1]))
        layers.append(NonLocalBlock(args.encoder_channels[-1]))
        layers.append(ResidualBlock(args.encoder_channels[-1], args.encoder_channels[-1]))
        layers.append(nn.GroupNorm(32, args.encoder_channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(args.encoder_channels[-1], args.latent_dim, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)