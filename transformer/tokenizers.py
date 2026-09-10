"""
Tokenizers: convert raw, modality-specific input into the (T, B, N, embed_dim) token/current sequence that SpikingTransformerBlock consumes, or
into the (B, N, embed_dim) static token sequence the dense ANN baseline (transformer/baseline.py) consumes.

This is what lets SpikingTransformer's backbone (block.py:SpikingTransformerBlock) stay entirely modality-agnostic: image datasets
(MNIST, CIFAR-10) and SHD (a genuinely event-based neuromorphic audio dataset) each get their own Tokenizer subclass, and SpikingTransformer just
calls whichever one it was given.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .encoding import rate_code

class Tokenizer(nn.Module, ABC):
    """
    Abstract base for a modality-specific tokenizer. num_tokens is N, the number of tokens/patches this tokenizer produces per sample. embed_dim
    is the token embedding dimension. Concrete subclasses set both in __init__.
    """
    num_tokens: int
    embed_dim: int

    @abstractmethod
    def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
        """
        Convert raw_input into a (T, B, N, embed_dim) token/current sequence, ready to feed into a stack of SpikingTransformerBlock layers.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_static(self, raw_input: torch.Tensor) -> torch.Tensor:
        """
        Convert raw_input into a (B, N, embed_dim) static token sequence (no time dimension), for the dense ANN baseline.
        """
        raise NotImplementedError

class PatchEmbedding(nn.Module):
    """
    Convert an image into a sequence of patch embedding currents. image_size is the height/width of the (square) input image, e.g. 28. patch_size
    is the height/width of each square patch, e.g. 7 or 4, it must evenly divide image_size. in_channels is the number of input image channels.
    embed_dim is the output embedding dimension per patch/token.
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

class ImagePatchTokenizer(Tokenizer):
    """
    Tokenizer for image datasets (MNIST, CIFAR-10): patchifies the image via a Conv2d (PatchEmbedding) and encodes it into a spike/current train
    over num_timesteps time-steps, using either "repeat" or "rate" coding.

    image_size, patch_size, in_channels, embed_dim configure the underlying PatchEmbedding. num_timesteps is T. coding is either "repeat" (embed
    the image once, then repeat the resulting currents T times: cheap, lets the first LIF layer do the spiking) or "rate" (rate-code the raw
    pixels into T Bernoulli spike frames *before* the conv, since Bernoulli sampling needs [0, 1]-valued probabilities, which post-conv currents
    are not guaranteed to be).
    """
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        num_timesteps: int,
        coding: str = "repeat",
    ) -> None:
        if coding not in {"repeat", "rate"}:
            raise ValueError(f"coding must be either 'repeat' or 'rate'; we got {coding!r} instead.")

        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.num_timesteps = num_timesteps
        self.coding = coding
        self.num_tokens = self.patch_embedding.num_patches
        self.embed_dim = embed_dim

    def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
        """
        raw_input is a batch of images, shape (B, in_channels, image_size, image_size), with pixel values in [0, 1]. Returns a (T, B, N,
        embed_dim) token/current sequence.
        """
        if self.coding == "repeat":
            tokens = self.patch_embedding(raw_input)
            return tokens.unsqueeze(0).repeat(self.num_timesteps, 1, 1, 1)

        # "rate": rate-code the raw pixels first (they must stay in [0, 1] to serve as Bernoulli probabilities), then patch-embed every frame.
        spike_frames = rate_code(raw_input, self.num_timesteps) # (T, B, C, H, W)
        T, B, C, H, W = spike_frames.shape
        embedded = self.patch_embedding(spike_frames.reshape(T * B, C, H, W)) # (T * B, N, embed_dim)
        return embedded.reshape(T, B, self.num_tokens, self.embed_dim)

    def embed_static(self, raw_input: torch.Tensor) -> torch.Tensor:
        """
        raw_input is a batch of images, shape (B, in_channels, image_size, image_size). Returns a (B, N, embed_dim) static token sequence.
        """
        return self.patch_embedding(raw_input)

class SHDTokenizer(Tokenizer):
    """
    Tokenizer for the Spiking Heidelberg Digits (SHD) dataset: a genuinely event-based neuromorphic audio dataset, so no artificial rate-coding
    is needed or well-defined here (see datasets.py::get_shd_dataloaders, which bins each sample's raw spike-time events into num_tokens fixed
    temporal chunks of num_channels spike counts each via tonic.transforms.ToFrame: that binning is what defines N, kept independent of this
    tokenizer's num_timesteps).

    num_channels is SHD's sensor channel count (700). num_tokens is N, the number of temporal chunks the dataset loader already binned each
    sample into. embed_dim is the token embedding dimension. num_timesteps is T, the number of LIF simulation time-steps: an independent axis
    from N, exactly as it is for the image tokenizer. Since each chunk's spike counts are already a genuine (if coarse) spike representation,
    this tokenizer only supports "repeat" coding: the per-chunk currents are computed once and then repeated across T.
    """
    def __init__(self, num_channels: int, num_tokens: int, embed_dim: int, num_timesteps: int) -> None:
        super().__init__()
        self.linear = nn.Linear(num_channels, embed_dim)
        self.num_channels = num_channels
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.num_timesteps = num_timesteps

    def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
        """
        raw_input is a batch of already-binned SHD spike counts, shape (B, num_tokens, num_channels). Returns a (T, B, num_tokens, embed_dim)
        token/current sequence (repeat-coded across T).
        """
        currents = self.linear(raw_input) # (B, N, embed_dim)
        return currents.unsqueeze(0).repeat(self.num_timesteps, 1, 1, 1)

    def embed_static(self, raw_input: torch.Tensor) -> torch.Tensor:
        """
        raw_input is a batch of already-binned SHD spike counts, shape (B, num_tokens, num_channels). Returns a (B, num_tokens, embed_dim) static
        token sequence.
        """
        return self.linear(raw_input)