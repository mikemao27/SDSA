"""
Input encoding utilities: convert static images into spike trains.

Spiking neurons operate on sequences of binary events over time, not on single static real-valued inputs. Before an MNIST image can be fed into the
spiking transformer, it must be converted into a tensor with an explicit time dimension of length T.
"""

import torch

def rate_code(x: torch.Tensor, num_steps: int) -> torch.Tensor:
    """
    Convert a static input into a spike train via rate (Bernoulli) coding.

    At each time-step, each element fires (1.0) with probability proportional to its (normalized) intensity in x, and stays silent (0.0) otherwise. x is 
    the static input tensor, e.g. shape (B, C, H, W), with values expected to lie in [0, 1] (normalized pixel intensities). num_steps is the number of 
    time-steps T to generate. Returns a spike train tensor of shape (T, B, C, H, W) containing only 0.0/1.0.
    """
    x = x.clamp(0.0, 1.0)
    x_rep = x.unsqueeze(0).repeat(num_steps, *([1] * x.dim()))
    spikes = torch.bernoulli(x_rep)
    return spikes

def repeat_code(x: torch.Tensor, num_steps: int) -> torch.Tensor:
    """
    Convert a static input into a time series by simply repeating it.

    This is a cheaper alternative to rate_code (sometimes called "direct" or "constant current" input coding): rather than stochastically sampling
    spikes, the same real-valued input is presented at every time-step, and it's left to the first LIF layer in the model to convert it into spikes.
    x is the static input tensor of any shape, e.g. (B, C, H, W) for images or (B, N, num_channels) for SHD's temporal-chunk currents. num_steps is
    the number of time-steps T to generate. Returns a tensor of shape (T, *x.shape), equal to `x` repeated T times along a new leading time
    dimension.
    """
    return x.unsqueeze(0).repeat(num_steps, *([1] * x.dim()))
