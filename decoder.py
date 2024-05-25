import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import UpSampleBlock, ResidualBlock, NonLocalBlock, Swish

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        layers = [nn.Conv2d(args.latent_dim, args.encoder_channels[-1], 3, 1, 1)]

        layers.append(ResidualBlock(args.encoder_channels[-1], args.encoder_channels[-1]))
        layers.append(NonLocalBlock(args.encoder_channels[-1]))
        layers.append(ResidualBlock(args.encoder_channels[-1], args.encoder_channels[-1]))

        for i in range(len(args.encoder_channels) -1, 0, -1):
            layers.append(ResidualBlock(args.encoder_channels[i], args.encoder_channels[i-1]))
            layers.append(UpSampleBlock(args.encoder_channels[i-1]))

        layers.append(nn.GroupNorm(32, args.encoder_channels[0]))
        layers.append(Swish())
        layers.append(nn.Conv2d(args.encoder_channels[0], args.image_channels, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)