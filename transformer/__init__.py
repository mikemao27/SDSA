"""
Spiking transformer proof-of-concept package.

Exposes the core building blocks used to assemble a spiking vision transformer: the surrogate-gradient spike function, the LIF neuron, spiking 
self-attention, the spiking transformer block, and the full model.
"""

from .model import SpikingTransformer

__all__ = ["SpikingTransformer"]