"""
Full spiking transformer model, modality-agnostic over its tokenizer.

The pipeline is as follows: raw modality-specific input -> a Tokenizer (see transformer/tokenizers.py) encodes it into a spike/current train over
T time-steps (T, B, N, embed_dim) -> a stack of SpikingTransformerBlock layers -> a classification head that reads out logits from the
accumulated spike counts (firing rates) over the T time-steps.
"""

import torch
import torch.nn as nn

from .block import SpikingTransformerBlock
from .tokenizers import Tokenizer

class SpikingTransformer(nn.Module):
    """
    A minimal spiking transformer for classification, parameterized by a Tokenizer so the same backbone works across modalities (image patches,
    SHD temporal chunks, ...). tokenizer converts raw input into a (T, B, N, embed_dim) token/current sequence (see transformer/tokenizers.py).
    num_classes is the number of output classes. embed_dim is the token embedding dimension (must match tokenizer.embed_dim). depth is the number
    of stacked SpikingTransformerBlock layers. num_heads is the number of attention heads per block. mlp_ratio is the hidden-dim multiplier for
    each block's SpikingMLP. threshold is the firing threshold shared across all LIF neurons in the model. beta is the leak factor shared across
    all LIF neurons in the model. attention_mode is passed through to every block's SpikingSelfAttention (either "linear", the O(N) spike-driven
    form, or "quadratic", kept for comparison/ablation). neuron_type selects the spiking neuron implementation shared across the model (see
    transformer/neurons.py::build_neuron). track_firing_rate, if True, makes every LIF neuron in the model record its mean firing rate for energy
    accounting (see transformer/energy.py::estimate_model_energy).
    """
    def __init__(
        self,
        tokenizer: Tokenizer,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        threshold: float = 1.0,
        beta: float = 0.9,
        attention_mode: str = "linear",
        neuron_type: str = "lif",
        track_firing_rate: bool = False,
    ) -> None:
        assert tokenizer.embed_dim == embed_dim, (
            f"tokenizer.embed_dim ({tokenizer.embed_dim}) must match embed_dim ({embed_dim})."
        )

        super().__init__()
        self.tokenizer = tokenizer
        self.blocks = nn.ModuleList([
            SpikingTransformerBlock(
                embed_dim,
                num_heads,
                mlp_ratio,
                threshold,
                beta,
                attention_mode = attention_mode,
                neuron_type = neuron_type,
                track_firing_rate = track_firing_rate,
            )
            for _ in range(depth)
        ])
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
        """
        Classify a batch of raw, modality-specific input. raw_input's shape depends on self.tokenizer (e.g. (B, in_channels, image_size,
        image_size) for image tokenizers, (B, num_tokens, num_channels) for SHDTokenizer). Returns a logits tensor of shape (B, num_classes),
        obtained by reading out the accumulated firing activity over the model's T time-steps.
        """
        tokens_sequence = self.tokenizer(raw_input)

        for block in self.blocks:
            tokens_sequence = block(tokens_sequence)

        pooled = tokens_sequence.mean(dim = 2)
        logits_per_timestep = self.head(pooled)
        logits = logits_per_timestep.mean(dim = 0)

        return logits
