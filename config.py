"""
Hyperparameter configuration for the spiking transformer MNIST proof-of-concept.
"""

from dataclasses import dataclass

@dataclass
class SpikingTransformerConfig:
    """
    All hyperparameters needed to build, train, and evaluate the model.
    """

    # Model architecture:
    image_size: int # The input image height/width, e.g. 28 for MNIST.
    patch_size: int # The patch height/width; must evenly divide image_size.
    in_channels: int # The number of input image channels (1 for MNIST).
    num_classes: int # The number of output classes (10 for MNIST digits).
    embed_dim: int # The token embedding dimension.
    depth: int # The number of stacked SpikingTransformerBlock layers.
    num_heads: int # The attention heads per block; must evenly divide embed_dim.
    mlp_ratio: float # SpikingMLP hidden_dim = embed_dim * mlp_ratio.

    # Spiking neuron dynamics:
    num_timesteps: int # T, number of simulation time-steps per forward pass.
    threshold: float # The LIF firing threshold.
    beta: float # The LIF leak/decay factor, in (0, 1).
    surrogate_alpha: float # The surrogate gradient steepness (see spiking_transformer/surrogate.py).

    # Training:
    batch_size: int
    learning_rate: float
    num_epochs: int
    device: str # "cpu" or "mps" (Apple Silicon GPU).
