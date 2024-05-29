"""
    Modified from https://github.com/SJ1727/transformer-from-scratch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from einops import rearrange
from typing import Optional
from vqgan import VQGAN

class TransformerConfig:
    def __init__(self, embed_dim, num_heads=8, num_layers=2, feed_forward_dim=256, feed_forward_dropout=0.1, attention_dropout=0.1, out_dropout=0.1, device="cpu"):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.feed_forward_dim = feed_forward_dim
        self.feed_forward_dropout = feed_forward_dropout
        self.attention_dropout = attention_dropout
        self.out_dropout = out_dropout
        self.device = device

class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super(VQGANTransformer, self).__init__()
        config = TransformerConfig(
            embed_dim=args.latent_dim,
            num_heads=args.num_heads,
            num_layers=args.transformer_layers,
            feed_forward_dim=args.feed_forward_dim,
            feed_forward_dropout=args.feed_forward_dropout,
            attention_dropout=args.attention_dropout,
            out_dropout=args.out_dropout,
            device=args.device
        )
        
        self.top_k = args.top_k

        self.transformer = Transformer(config)
        self.fc_out = nn.Linear(args.latent_dim, args.codebook_size)
        
        self.positional_encoding = PositionalEncoding(args.latent_dim, 1025, device=args.device) # Change the number if making much larger images

        self.vqgan = VQGAN(args)
        self.vqgan.load_state_dict(torch.load(args.vqgan_path))
        self.vqgan = self.vqgan.eval()
        
        self.one_hot = lambda x: F.one_hot(x, args.codebook_size).to(torch.float32).to(args.device)

        self.sos_token = torch.ones(args.latent_dim).to(args.device) * args.sos_token

    def generate(self, size):
        seq = torch.zeros(1, size*size + 1, self.sos_token.shape[-1])
        seq[:, 0, :] = self.sos_token

        for i in range(size*size):
            seq = seq.to(self.sos_token.device)
            in_seq = self.positional_encoding(seq)
            out_seq = self.transformer(in_seq)
            distribution = F.softmax(out_seq[:, i, :])

            # Randomly selecting index from top k
            top_k_values, top_k_indices = torch.topk(distribution, self.top_k, dim=-1)
            top_k_probs = top_k_values / top_k_values.sum(dim=-1, keepdim=True)
            sampled_index = torch.multinomial(top_k_probs, 1)
            random_index = top_k_indices.gather(-1, sampled_index).squeeze()

            seq[:, i+1, :] = self.vqgan.codebook[random_index].to(self.sos_token.device)

        decoder_input = rearrange(seq[:, 1:, :], "b (h w) l-> b l h w", h=size, w=size)
        generated_image = self.vqgan.decode(decoder_input)
        
        return generated_image

    def forward(self, x):
        B, C, H, W = x.shape
        z = self.vqgan.encoder(x)
        z_q, codebook_indices, _ = self.vqgan.codebook(z)
        z_q = rearrange(z_q, "b l h w -> b (h w) l")

        in_seq = torch.cat((self.sos_token.repeat((B, 1, 1)), z_q), dim=1)
        in_seq = self.positional_encoding(in_seq)
        out_seq = self.transformer(in_seq)

        return self.fc_out(out_seq), self.one_hot(codebook_indices)

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        self.encoder = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.num_layers)
        ]).to(config.device)

    def forward(self, x: torch.tensor) -> torch.tensor:
        for block in self.encoder:
            x = block(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.self_attention = AddAndNorm(MultiHeadAttention(
            config.embed_dim,
            num_heads=config.num_heads
        ), config.embed_dim)

        self.feed_forward = AddAndNorm(FeedForwardLayer(
            config.embed_dim,
            config.feed_forward_dim,
            dropout=config.feed_forward_dropout
        ), config.embed_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.self_attention(x, masked=True)
        x = self.feed_forward(x)
        return x

class MultiHeadAttention(nn.Module):    
    def __init__(self, embed_dim: int, num_heads: int=1, attention_dropout: float=0.1, out_dropout: float=0.1):
        super(MultiHeadAttention, self).__init__()
        if embed_dim % num_heads != 0:
            raise Exception("Embed dimension must be divisable by the number of heads")
        
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        self.key = nn.Linear(self.head_dim, self.head_dim)
        self.query = nn.Linear(self.head_dim, self.head_dim)
        self.value = nn.Linear(self.head_dim, self.head_dim)
        self.out = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out_dropout = nn.Dropout(out_dropout)

    def _mask_logits(self, logits: torch.tensor) -> torch.tensor: 
        # TODO: Precompute mask       
        mask = torch.ones(logits.size(2), logits.size(3))
        mask = torch.tril(mask, diagonal=0)
        mask = mask.to(logits.device)

        masked_logits = logits.masked_fill(mask == 0, float("-inf"))
        return masked_logits

    def _scaled_dot_product(self, q: torch.tensor, k: torch.tensor, v: torch.tensor, masked: Optional[bool]=False) -> torch.tensor:
        attention_logits = torch.matmul(q, torch.transpose(k, -2, -1))
        attention_logits *= 1 / np.sqrt(self.head_dim)

        if masked:
            attention_logits = self._mask_logits(attention_logits)

        attention = self.attention_dropout(torch.softmax(attention_logits, dim=-1))

        values = self.out_dropout(torch.matmul(attention, v))

        return values

    def _self_attention_projection(self, x: torch.tensor) -> tuple[torch.tensor]:
        q = rearrange(x, "b d (w n)->b d n w", n=self.num_heads)
        k = rearrange(x, "b d (w n)->b d n w", n=self.num_heads)
        v = rearrange(x, "b d (w n)->b d n w", n=self.num_heads)
        
        return q, k, v

    def forward(self, x: torch.tensor, masked: Optional[bool]=False) -> torch.tensor:
        # Shape of input (x): Batch size x Sequence length x Embedding dimension
        q, k, v = self._self_attention_projection(x)
        
        # Pass through linear layers
        # Batch size x Sequence length x Number of heads x Head dim
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        
        # Apply scaled dot product on all the heads
        output = self._scaled_dot_product(q, k, v, masked=masked)
        
        # Concatonating the output
        # Batch size x Sequence length x embedding dim
        output = rearrange(output, "b d n w->b d (n w)")
        
        # Pass through output layer
        output = self.out(output)
        return output

class Residual(nn.Module):
    def __init__(self, func):
        super(Residual, self).__init__()
        self.func = func

    def forward(self, x: torch.tensor, **kwargs) -> torch.tensor:
        x = x + self.func(x, **kwargs)
        return x

class AddAndNorm(nn.Module):
    def __init__(self, func, embed_dim: int):
        super(AddAndNorm, self).__init__()
        self.func = func
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.tensor, **kwargs) -> torch.tensor:
        x = Residual(self.func)(x, **kwargs)
        x = self.norm(x)
        return x

class FeedForwardLayer(nn.Module):
    def __init__(self, embed_dim: int, feed_forward_dim: int, dropout: float=0.1):
        super(FeedForwardLayer, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.layers(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, sequence_length: int, dropout: Optional[int]=0.1, device="cpu"):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout).to(device)

        positions = torch.arange(sequence_length).unsqueeze(1)
        div_term = torch.exp(-(torch.arange(0, embed_dim, 2) * np.log(10000.0) / embed_dim))
        terms = positions * div_term
        self.positional_encodings = torch.zeros(1, sequence_length, embed_dim)
        self.positional_encodings[0, :, 0::2] = torch.sin(terms)
        self.positional_encodings[0, :, 1::2] = torch.cos(terms)
        self.positional_encodings = self.positional_encodings.to(device)

    def forward(self, x):
        x = x + self.positional_encodings[:, :x.size(1)]
        x = self.dropout(x)
        return x