import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange

class Codebook(nn.Module):
    def __init__(self, args):
        super(Codebook, self).__init__()
        self.embedding = nn.Embedding(args.codebook_size, args.latent_dim)
        self.beta = args.beta

    def __getitem__(self, idx):
        return self.embedding(idx)

    def forward(self, z):
        B, C, H, W = z.shape
        flattened_z = rearrange(z, "b c h w-> b (h w) c")
        
        d = torch.cdist(flattened_z, self.embedding.weight[None, :].repeat((B, 1, 1)))
        
        min_d = torch.argmin(d, dim=-1)
        
        z_q = self.embedding(min_d).view((B, C, H, W))

        comitement_loss = torch.mean((z_q.detach() - z) ** 2)
        codebook_loss = torch.mean((z.detach() - z_q) ** 2)
        loss = codebook_loss + self.beta * comitement_loss
        
        z_q = z - (z_q - z).detach()
        
        return z_q, min_d, loss