"""
Full spiking transformer model for MNIST digit classification.

The pipeline is as follows: image (B, in_channels, image_size, image_size) -> patch embedding (conv) into token currents (B, N, embed_dim) -> encoded into
a spike/current train over T time-steps (T, B, N, embed_dim) -> a stack of SpikingTransformerBlock layers -> a classification head that reads out logits 
from the accumulated spike counts (firing rates) over the T time-steps
"""

import torch
import torch.nn as nn

from .block import SpikingTransformerBlock

class PatchEmbedding(nn.Module):
    """
    Convert an MNIST image into a sequence of patch embedding currents. image_size is the height/width of the (square) input image, e.g. 28. patch_size
    is the height/width of each square patch, e.g. 7 or 4, it must evenly divide image_size. in_channels is the number of input image channels
    (1 for MNIST). embed_dim is the output embedding dimension per patch/token.
    """
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        assert image_size % patch_size == 0, f"patch_size must be a factor of image_size. We got patch_size: {patch_size} and image_size: {image_size} instead."

        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size = patch_size, stride = patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed an image into a sequence of patch tokens. x is an image tensor of shape (B, in_channels, image_size, image_size). Returns a tensor of shape
        (B, N, embed_dim), where N is the number of patches.
        """
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class SpikingTransformer(nn.Module):
    """
    A minimal spiking vision transformer for MNIST digit classification. image_size is the input image height/width. patch_size is the patch height/width
    used by the patch embedding. in_channels is the number of input image channels. num_classes is the number of output classes (10 for MNIST). 
    embed_dim is the token embedding dimension. depth is the number of stacked SpikingTransformerBlock layers. num_heads is the number of attention 
    heads per block. mlp_ratio is the hidden-dim multiplier for each block's SpikingMLP. num_timesteps is the number of simulation time-steps T used to 
    encode each static image into a spike/current train. threshold is the firing threshold shared across all LIF neurons in the model. beta is the leak 
    factor shared across all LIF neurons in the model.
    """
    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        num_classes: int = 10,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 2,
        mlp_ratio: float = 4.0,
        num_timesteps: int = 8,
        threshold: float = 1.0,
        beta: float = 0.9,
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.blocks = nn.ModuleList([SpikingTransformerBlock(embed_dim, num_heads, mlp_ratio, threshold, beta) for _ in range(depth)])
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify a batch of MNIST images. x is an image tensor of shape (B, in_channels, image_size, image_size). Returns a logits tensor of shape 
        (B, num_classes), obtained by reading out the accumulated firing activity over the model's T time-steps.
        """
        tokens = self.patch_embedding(x)
        tokens_sequence = tokens.unsqueeze(0).repeat(self.num_timesteps, 1, 1, 1)
        
        for block in self.blocks:
            tokens_sequence = block(tokens_sequence)
        
        pooled = tokens_sequence.mean(dim = 2)
        logits_per_timestep = self.head(pooled)
        logits = logits_per_timestep.mean(dim = 0)

        return logits
