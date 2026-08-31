"""
A single spiking transformer block: spiking self-attention + spiking MLP.

Mirrors the structure of a standard transformer block, but every nonlinearity is a spiking LIF neuron instead of e.g. GELU, and attention is
computed via SpikingSelfAttention instead of softmax attention.
"""

import torch
import torch.nn as nn

from .attention import SpikingSelfAttention
from .neurons import LIFNeuron

class SpikingMLP(nn.Module):
    """
    Two-layer feed-forward block with a spiking nonlinearity in between.

    The structure is as follows: Linear -> LIF -> Linear -> LIF, applied identically at every time-step of the (T, B, N, embed_dim) input.

    embed_dim is the input/output feature dimension. hidden_dim is the hidden layer dimension (typically a multiple of embed_dim). threshold is the 
    firing threshold for both LIF layers. beta is the leak factor for both LIF layers.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, threshold: float = 1.0, beta: float = 0.9) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

        self.lif1 = LIFNeuron(threshold = threshold, beta = beta)
        self.lif2 = LIFNeuron(threshold = threshold, beta = beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the spiking feed-forward transform. x is a tensor of shape (T, B, N, embed_dim). This returns a tensor of shape (T, B, N, embed_dim).
        """
        h = self.fc1(x)
        h = self.lif1(h)
        h = self.fc2(h)
        h = self.lif2(h)

        return h

class SpikingTransformerBlock(nn.Module):
    """
    One spiking transformer block: residual spiking self-attention + residual spiking MLP.

    embed_dim is the token embedding dimension. num_heads is the number of attention heads for SpikingSelfAttention. mlp_ratio is the multiplier 
    applied to embed_dim to get the SpikingMLP hidden dim. threshold is the firing threshold shared across the block's LIF neurons. beta is the leak 
    factor shared across the block's LIF neurons.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        threshold: float = 1.0,
        beta: float = 0.9,
    ) -> None:
        super().__init__()
        self.attention = SpikingSelfAttention(embed_dim, num_heads, threshold = threshold, beta = beta)
        self.hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = SpikingMLP(embed_dim, self.hidden_dim, threshold = threshold, beta = beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spiking self-attention and spiking MLP, each with a residual connection. x is a tensor of shape (T, B, N, embed_dim). Returns a tensor 
        of shape (T, B, N, embed_dim).
        """
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return x
