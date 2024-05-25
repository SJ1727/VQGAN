import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from encoder import Encoder
from decoder import Decoder
from codebook import Codebook

class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(args).to(args.device)
        self.decoder = Decoder(args).to(args.device)
        self.codebook = Codebook(args).to(args.device)

    def forward(self, x):
        z = self.encoder(x)
        z_q, codebook_indices, loss = self.codebook(z)
        x_hat = self.decoder(z_q)
        return x_hat, codebook_indices, loss

    def encode(self, x):
        z = self.encoder(x)
        z_q, codebook_indices, loss = self.codebook(z)
        return z_q, codebook_indices, loss

    def decode(self, z_q):
        x_hat = self.decoder(z_q)
        return x_hat

    def calculate_lambda(self, perceptual_loss, gan_loss):
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, self.decoder.model[-1].weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, self.decoder.model[-1].weight, retain_graph=True)[0]

        model_lambda = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-6)
        model_lambda = torch.clamp(model_lambda, 0, 1e4).detach()
        return model_lambda * 0.8